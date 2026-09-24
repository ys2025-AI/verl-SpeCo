# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Numerical/throughput/memory A/B for the DSpark fused losses.

Compares the torch CE/L1 paths against the Triton ``fused_label_ce_loss`` and
``fused_tv_loss`` (L1 = 2 * TV) on identical random inputs: value and gradient
agreement, wall time, and peak accelerator memory for a forward + backward.
Run on an accelerator, e.g.::

    python tools/benchmark_fused_losses.py --num-tokens 512 2048
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from verl.utils.device import get_device_name


def _device() -> torch.device:
    name = str(get_device_name()).lower()
    if name not in {"npu", "cuda"}:
        raise SystemExit(
            f"fused Triton losses need an NPU or CUDA device; detected device={name!r}"
        )
    return torch.device(name)


def _sync(device: torch.device) -> None:
    getattr(torch, device.type).synchronize()


def _peak_memory(device: torch.device) -> float:
    return getattr(torch, device.type).max_memory_allocated() / 1024**3


def _reset_peak_memory(device: torch.device) -> None:
    getattr(torch, device.type).reset_peak_memory_stats()


def _bench(device, fn, *, warmup: int, iterations: int):
    for _ in range(warmup):
        fn()
    _sync(device)
    _reset_peak_memory(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    _sync(device)
    elapsed = (time.perf_counter() - start) / iterations
    return elapsed, _peak_memory(device)


def _run(
    *,
    device: torch.device,
    num_tokens: int,
    vocab_size: int,
    warmup: int,
    iterations: int,
) -> None:
    from verl_speco.backends.fused_losses import (
        fused_label_ce_loss,
        fused_tv_loss,
    )

    torch.manual_seed(0)
    logits = torch.randn(num_tokens, vocab_size, device=device, dtype=torch.bfloat16)
    targets = torch.randn(num_tokens, vocab_size, device=device, dtype=torch.bfloat16)
    labels = torch.randint(0, vocab_size, (num_tokens,), device=device)

    def ce_manual():
        lo = logits.detach().clone().requires_grad_(True)
        loss = F.nll_loss(F.log_softmax(lo.float(), dim=-1), labels, reduction="none")
        loss.sum().backward()
        return loss

    def ce_native():
        lo = logits.detach().clone().requires_grad_(True)
        loss = F.cross_entropy(lo.float(), labels, reduction="none")
        loss.sum().backward()
        return loss

    def ce_fused():
        lo = logits.detach().clone().requires_grad_(True)
        loss = fused_label_ce_loss(lo, labels)
        loss.sum().backward()
        return loss

    def l1_torch():
        lo = logits.detach().clone().requires_grad_(True)
        draft = torch.softmax(lo.float(), dim=-1)
        target = torch.softmax(targets.float(), dim=-1)
        loss = (draft - target).abs().sum(dim=-1)
        loss.sum().backward()
        return loss

    def l1_fused():
        lo = logits.detach().clone().requires_grad_(True)
        loss = 2.0 * fused_tv_loss(lo, targets)
        loss.sum().backward()
        return loss

    print(f"\n===== num_tokens={num_tokens} vocab_size={vocab_size} =====")

    def _grad(fn):
        from torch.autograd import grad

        lo = logits.detach().clone().requires_grad_(True)
        loss = fn(lo)
        (g,) = grad(loss.sum(), lo)
        return loss.detach(), g.detach()

    # Numerics (recompute directly so we keep the gradients).
    _, g_ce_manual = _grad(
        lambda lo: F.nll_loss(
            F.log_softmax(lo.float(), dim=-1), labels, reduction="none"
        )
    )
    loss_ce_fused, g_ce_fused = _grad(lambda lo: fused_label_ce_loss(lo, labels))
    _, g_l1_torch = _grad(
        lambda lo: (
            (torch.softmax(lo.float(), dim=-1) - torch.softmax(targets.float(), dim=-1))
            .abs()
            .sum(dim=-1)
        )
    )
    loss_l1_fused, g_l1_fused = _grad(lambda lo: 2.0 * fused_tv_loss(lo, targets))
    loss_ce_manual = F.nll_loss(
        F.log_softmax(logits.float(), dim=-1), labels, reduction="none"
    ).detach()
    loss_l1_torch = (
        (torch.softmax(logits.float(), dim=-1) - torch.softmax(targets.float(), dim=-1))
        .abs()
        .sum(dim=-1)
        .detach()
    )
    print(
        "  CE   value max|diff| = "
        f"{(loss_ce_manual - loss_ce_fused).abs().max().item():.3e}   "
        "grad max|diff| = "
        f"{(g_ce_manual - g_ce_fused).abs().max().item():.3e}"
    )
    print(
        "  L1   value max|diff| = "
        f"{(loss_l1_torch - loss_l1_fused).abs().max().item():.3e}   "
        "grad max|diff| = "
        f"{(g_l1_torch - g_l1_fused).abs().max().item():.3e}"
    )

    rows = [
        ("CE manual log_softmax+nll", ce_manual),
        ("CE native cross_entropy", ce_native),
        ("CE fused (Triton)", ce_fused),
        ("L1 torch softmax", l1_torch),
        ("L1 fused (2*TV Triton)", l1_fused),
    ]
    print(f"  {'implementation':>26} {'ms':>9} {'peak GiB':>10}")
    for name, fn in rows:
        elapsed, peak = _bench(device, fn, warmup=warmup, iterations=iterations)
        print(f"  {name:>26} {elapsed * 1000:>9.1f} {peak:>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[512, 2048],
        help="Active draft positions per measurement.",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=248320, help="Full draft/target vocab."
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    device = _device()
    print(f"device={device}")
    for num_tokens in args.num_tokens:
        _run(
            device=device,
            num_tokens=num_tokens,
            vocab_size=args.vocab_size,
            warmup=args.warmup,
            iterations=args.iterations,
        )


if __name__ == "__main__":
    main()
