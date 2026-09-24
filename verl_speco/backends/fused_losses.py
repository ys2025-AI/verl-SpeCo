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
"""Fused Triton losses for DSpark draft training.

``fused_tv_loss`` ports the total-variation kernel from
``speculators.losses.fused`` (Apache-2.0, derived from SpecForge/Liger); one
program streams a row without materializing the ``[T, V]`` fp32 softmax.
``fused_label_ce_loss`` is a hard-label cross-entropy variant: it takes the
ground-truth token ids (not ``argmax(target)``) so the CE term keeps verl's
label convention.

Both return per-position ``[..., T]`` losses whose backward only produces a
gradient for the draft logits (targets are treated as constants).
"""

import torch
import triton
import triton.language as tl

# Ascend NPU Unified Buffer bound; triton-ascend raises "ub overflow" beyond this.
MAX_FUSED_SIZE_NPU = 4096
_BLOCK_SIZES = (512, 1024, 2048, 4096, 8192, 16384, 32768)
_THREADS_PER_PROGRAM = (128, 256, 512, 1024)
_MAX_ELEMS_PER_THREAD = 128
_MIN_ELEMS_PER_THREAD = 4

# stats layout: [row max, row sum-exp, label] for CE; [draft max, draft sum-exp,
# target max, target sum-exp, extra] for TV.
_TV_STATS = 5
_CE_STATS = 3


def _is_npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except Exception:  # noqa: BLE001
        return False


def _tile_configs():
    if _is_npu_available():
        # triton-ascend is Unified-Buffer bound: exactly one tile fits.
        return [triton.Config({"BLOCK_SIZE": MAX_FUSED_SIZE_NPU}, num_warps=4)]
    warp_size = 64 if getattr(torch.version, "hip", None) is not None else 32
    return [
        triton.Config({"BLOCK_SIZE": block}, num_warps=threads // warp_size)
        for threads in _THREADS_PER_PROGRAM
        for block in _BLOCK_SIZES
        if _MIN_ELEMS_PER_THREAD <= block // threads <= _MAX_ELEMS_PER_THREAD
    ]


def _prune_oversized_tiles(configs, nargs, **_):
    cap = triton.next_power_of_2(nargs["n_cols"])
    keep = [config for config in configs if config.kwargs["BLOCK_SIZE"] <= cap]
    if keep:
        return keep
    floor = min(config.kwargs["BLOCK_SIZE"] for config in configs)
    return [config for config in configs if config.kwargs["BLOCK_SIZE"] == floor]


_autotune_tv = triton.autotune(
    configs=_tile_configs(),
    key=["n_cols"],
    prune_configs_by={"early_config_prune": _prune_oversized_tiles},
)
_autotune_ce = triton.autotune(
    configs=_tile_configs(),
    key=["n_cols"],
    prune_configs_by={"early_config_prune": _prune_oversized_tiles},
)


@triton.jit
def _online_stats(row_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    """Online max (m) and sum-exp (d) over one row."""
    m = float("-inf")
    d = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(row_ptr + offsets, mask=mask, other=float("-inf")).cast(tl.float32)
        block_max = tl.max(tl.where(mask, x, float("-inf")))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.where(mask, tl.exp(x - m_new), 0.0))
        m = m_new
    return m, d


@_autotune_tv
@triton.jit
def _tv_forward_kernel(
    logits_ptr,
    targets_ptr,
    loss_ptr,
    stats_ptr,
    stats_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols
    targets_ptr += pid * n_cols

    m_d, z_d = _online_stats(logits_ptr, n_cols, BLOCK_SIZE)
    lse_d = m_d + tl.log(z_d)
    m_t, z_t = _online_stats(targets_ptr, n_cols, BLOCK_SIZE)
    lse_t = m_t + tl.log(z_t)

    acc = 0.0
    extra = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        dp = tl.exp(x - lse_d)
        tp = tl.exp(y - lse_t)
        acc += tl.sum(tl.where(mask, tl.minimum(dp, tp), 0.0))
        extra += tl.sum(tl.where(mask & (dp <= tp), dp, 0.0))

    tl.store(loss_ptr + pid, 1.0 - acc)
    tl.store(stats_ptr + 0 * stats_row + pid, m_d)
    tl.store(stats_ptr + 1 * stats_row + pid, z_d)
    tl.store(stats_ptr + 2 * stats_row + pid, m_t)
    tl.store(stats_ptr + 3 * stats_row + pid, z_t)
    tl.store(stats_ptr + 4 * stats_row + pid, extra)


@_autotune_tv
@triton.jit
def _tv_backward_kernel(
    logits_ptr,
    targets_ptr,
    grad_in_ptr,
    grad_out_ptr,
    stats_ptr,
    stats_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols
    targets_ptr += pid * n_cols
    grad_in_ptr += pid * n_cols

    go = tl.load(grad_out_ptr + pid).cast(tl.float32)
    m_d = tl.load(stats_ptr + 0 * stats_row + pid)
    z_d = tl.load(stats_ptr + 1 * stats_row + pid)
    lse_d = m_d + tl.log(z_d)
    m_t = tl.load(stats_ptr + 2 * stats_row + pid)
    z_t = tl.load(stats_ptr + 3 * stats_row + pid)
    lse_t = m_t + tl.log(z_t)
    extra = tl.load(stats_ptr + 4 * stats_row + pid)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        dp = tl.exp(x - lse_d)
        tp = tl.exp(y - lse_t)
        grad = -dp * ((dp <= tp).to(tl.float32) - extra)
        tl.store(grad_in_ptr + offsets, go * grad, mask=mask)


@_autotune_ce
@triton.jit
def _label_ce_forward_kernel(
    logits_ptr,
    labels_ptr,
    loss_ptr,
    stats_ptr,
    stats_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols

    m_d, z_d = _online_stats(logits_ptr, n_cols, BLOCK_SIZE)
    lse_d = m_d + tl.log(z_d)
    label = tl.load(labels_ptr + pid).to(tl.int64)
    logit_at_label = tl.load(logits_ptr + label).cast(tl.float32)

    tl.store(loss_ptr + pid, lse_d - logit_at_label)
    tl.store(stats_ptr + 0 * stats_row + pid, m_d)
    tl.store(stats_ptr + 1 * stats_row + pid, z_d)
    tl.store(stats_ptr + 2 * stats_row + pid, label.to(tl.float32))


@_autotune_ce
@triton.jit
def _label_ce_backward_kernel(
    logits_ptr,
    labels_ptr,
    grad_in_ptr,
    grad_out_ptr,
    stats_ptr,
    stats_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols
    grad_in_ptr += pid * n_cols

    go = tl.load(grad_out_ptr + pid).cast(tl.float32)
    m_d = tl.load(stats_ptr + 0 * stats_row + pid)
    z_d = tl.load(stats_ptr + 1 * stats_row + pid)
    lse_d = m_d + tl.log(z_d)
    label = tl.load(stats_ptr + 2 * stats_row + pid).to(tl.int64)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        dp = tl.exp(x - lse_d)
        grad = dp - (offsets == label).to(tl.float32)
        tl.store(grad_in_ptr + offsets, go * grad, mask=mask)


class _FusedTV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        vocab = int(logits.shape[-1])
        lead_shape = tuple(logits.shape[:-1])
        logits_flat = logits.contiguous().view(-1, vocab)
        targets_flat = targets.contiguous().view(-1, vocab)
        rows = logits_flat.size(0)
        loss = torch.empty(rows, device=logits.device, dtype=torch.float32)
        stats = torch.empty(_TV_STATS, rows, device=logits.device, dtype=torch.float32)
        _tv_forward_kernel[(rows,)](
            logits_flat, targets_flat, loss, stats, stats.stride(0), vocab
        )
        ctx.save_for_backward(logits_flat, targets_flat, stats)
        ctx.shape = (lead_shape, vocab)
        return loss.view(lead_shape)

    @staticmethod
    def backward(ctx, grad_output):
        logits_flat, targets_flat, stats = ctx.saved_tensors
        lead_shape, vocab = ctx.shape
        grad_in = torch.empty_like(logits_flat)
        _tv_backward_kernel[(logits_flat.size(0),)](
            logits_flat,
            targets_flat,
            grad_in,
            grad_output.contiguous().view(-1),
            stats,
            stats.stride(0),
            vocab,
        )
        return grad_in.view(*lead_shape, vocab), None


class _FusedLabelCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        vocab = int(logits.shape[-1])
        lead_shape = tuple(logits.shape[:-1])
        logits_flat = logits.contiguous().view(-1, vocab)
        labels_flat = labels.contiguous().view(-1)
        rows = logits_flat.size(0)
        loss = torch.empty(rows, device=logits.device, dtype=torch.float32)
        stats = torch.empty(_CE_STATS, rows, device=logits.device, dtype=torch.float32)
        _label_ce_forward_kernel[(rows,)](
            logits_flat, labels_flat, loss, stats, stats.stride(0), vocab
        )
        ctx.save_for_backward(logits_flat, labels_flat, stats)
        ctx.shape = (lead_shape, vocab)
        return loss.view(lead_shape)

    @staticmethod
    def backward(ctx, grad_output):
        logits_flat, labels_flat, stats = ctx.saved_tensors
        lead_shape, vocab = ctx.shape
        grad_in = torch.empty_like(logits_flat)
        _label_ce_backward_kernel[(logits_flat.size(0),)](
            logits_flat,
            labels_flat,
            grad_in,
            grad_output.contiguous().view(-1),
            stats,
            stats.stride(0),
            vocab,
        )
        return grad_in.view(*lead_shape, vocab), None


def fused_tv_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-position TV distance ``1 - sum_v min(p, q)`` in ``[..., T]``."""

    return _FusedTV.apply(logits, targets)


def fused_label_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-position hard-label CE ``-log softmax(logits)[label]`` in ``[..., T]``."""

    return _FusedLabelCE.apply(logits, labels)


__all__ = ["fused_label_ce_loss", "fused_tv_loss"]
