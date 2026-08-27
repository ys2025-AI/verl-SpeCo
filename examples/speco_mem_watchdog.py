"""Host memory / disk watchdog for SPECO training runs.

Runs in the background alongside a Ray training job.  When host available
memory (or a watched filesystem) drops below a configured threshold, it kills
the training process tree and runs ``ray stop --force`` so the server does not
freeze to death from Ray object-store / spilling pressure.

Usage (invoked from the launch script)::

    python3 examples/speco_mem_watchdog.py \
        --pid <training_pid> \
        --min-free-gb 200 \
        --watch-fs /dev/shm,/ \
        --fs-min-free-gb 50 \
        --poll-interval 10

Exits automatically when the watched PID disappears.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time


def _meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    out[parts[0].rstrip(":")] = int(parts[1]) * 1024  # -> bytes
    except OSError:
        pass
    return out


def _available_gb() -> float:
    info = _meminfo()
    avail = info.get("MemAvailable")
    if avail is None:
        free = info.get("MemFree", 0)
        cached = info.get("Cached", 0)
        buffers = info.get("Buffers", 0)
        avail = free + cached + buffers
    return float(avail) / (1024**3)


def _fs_free_gb(path: str) -> float:
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except OSError:
        return float("inf")


def _kill_tree(pid: int) -> None:
    """Kill a process and all its descendants."""
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            _ = f.read()
    except OSError:
        return
    children: list[int] = []
    try:
        cp = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        children = [int(x) for x in cp.stdout.split() if x.strip().isdigit()]
    except Exception:
        children = []
    for child in children:
        _kill_tree(child)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True, help="training process PID to watch")
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=200.0,
        help="kill when host MemAvailable drops below this (GB)",
    )
    ap.add_argument(
        "--watch-fs",
        default="/dev/shm,/",
        help="comma-separated filesystem paths to monitor",
    )
    ap.add_argument(
        "--fs-min-free-gb",
        type=float,
        default=50.0,
        help="kill when any watched filesystem free space drops below this (GB)",
    )
    ap.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="seconds between polls",
    )
    ap.add_argument(
        "--cooldown-s",
        type=float,
        default=30.0,
        help="seconds after a kill during which no further kills are attempted",
    )
    ap.add_argument(
        "--soft-warn-pct",
        type=float,
        default=0.8,
        help="warn (not kill) when /dev/shm usage exceeds this fraction of total",
    )
    args = ap.parse_args()

    fs_paths = [p.strip() for p in args.watch_fs.split(",") if p.strip()]
    killed = False

    while True:
        try:
            os.kill(args.pid, 0)
        except OSError:
            # Training process is gone — exit cleanly.
            return 0

        avail = _available_gb()
        if avail < args.min_free_gb:
            print(
                f"[speco-watchdog] KILL: MemAvailable={avail:.1f}GB < "
                f"threshold={args.min_free_gb:.1f}GB; killing pid={args.pid}",
                flush=True,
            )
            _kill_tree(args.pid)
            subprocess.run(["ray", "stop", "--force"], timeout=60)
            killed = True
            break

        for path in fs_paths:
            free = _fs_free_gb(path)
            if free < args.fs_min_free_gb:
                print(
                    f"[speco-watchdog] KILL: fs={path} free={free:.1f}GB < "
                    f"threshold={args.fs_min_free_gb:.1f}GB; killing pid={args.pid}",
                    flush=True,
                )
                _kill_tree(args.pid)
                subprocess.run(["ray", "stop", "--force"], timeout=60)
                killed = True
                break
        if killed:
            break

        # Soft warning for /dev/shm (RAM-backed object store + spills).
        try:
            shm_total = shutil.disk_usage("/dev/shm").total / (1024**3)
            shm_used = shm_total - _fs_free_gb("/dev/shm")
            if shm_total > 0 and (shm_used / shm_total) >= args.soft_warn_pct:
                print(
                    f"[speco-watchdog] WARN: /dev/shm usage "
                    f"{shm_used / shm_total * 100:.0f}% "
                    f"({shm_used:.0f}GB / {shm_total:.0f}GB)",
                    flush=True,
                )
        except OSError:
            pass

        time.sleep(args.poll_interval)

    # Brief cooldown to let the kill propagate, then ensure Ray is down.
    time.sleep(args.cooldown_s)
    subprocess.run(["ray", "stop", "--force"], timeout=60)
    return 0 if killed else 0


if __name__ == "__main__":
    sys.exit(main())
