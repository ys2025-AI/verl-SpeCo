# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Where per-expert FSDP shards every expert across all ranks, EP instead gives each"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .kernels import register_kernel
from .moe import _MOE_OP, GroupedExperts, swiglu_grouped
from .moe_grouped_gemm import _fused_permute_dispatch_npu, _grouped_matmul


@dataclass
class _EPCtx:
    group: object
    rank: int
    size: int
    experts_per_rank: int


_EP: _EPCtx | None = None


def configure(group, rank: int, size: int, experts_per_rank: int) -> None:
    global _EP
    _EP = _EPCtx(group=group, rank=rank, size=size, experts_per_rank=experts_per_rank)


class _AllToAll(torch.autograd.Function):
    """Variable-split all-to-all with an autograd backward (reverse all-to-all).

    forward:  send ``in_splits`` rows to each rank, receive ``out_splits`` -> [sum(out), *].
    backward: the transpose -- send grad with ``out_splits``, receive ``in_splits``.
    """

    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group):
        ctx.out_splits, ctx.in_splits, ctx.group = out_splits, in_splits, group
        y = x.new_empty([sum(out_splits), *x.shape[1:]])
        dist.all_to_all_single(y, x.contiguous(), out_splits, in_splits, group=group)
        return y

    @staticmethod
    def backward(ctx, g):
        if not torch.isfinite(g).all():
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        gx = g.new_empty([sum(ctx.in_splits), *g.shape[1:]])
        dist.all_to_all_single(gx, g.contiguous(), ctx.in_splits, ctx.out_splits, group=ctx.group)
        return gx, None, None, None


def _a2a_ints(x: torch.Tensor, out_splits, in_splits, group) -> torch.Tensor:
    y = x.new_empty([sum(out_splits), *x.shape[1:]])
    dist.all_to_all_single(y, x.contiguous(), out_splits, in_splits, group=group)
    return y


def _local_grouped_ffn(x: torch.Tensor, local_eid: torch.Tensor, w: torch.Tensor,
                        experts: GroupedExperts, n_local: int) -> torch.Tensor:
    # Handle empty input (expert collapse — all tokens routed to other ranks)
    if x.shape[0] == 0:
        return x
    # npu_moe_token_permute/unpermute don't support backward — use pure-torch path for training.
    if x.device.type == "npu" and not x.requires_grad:
        return _fused_permute_dispatch_npu(x, local_eid.reshape(-1, 1), w, experts, n_local)
    w1, w3, w2 = experts.local_weights()
    order = torch.argsort(local_eid, stable=True)
    inv = torch.argsort(order, stable=True)
    xs, ws = x[order].float(), w[order].float()
    counts = torch.bincount(local_eid, minlength=n_local)
    out = swiglu_grouped(xs, w1.float(), w3.float(), w2.float(), counts, ws,
                         experts.swiglu_limit, _grouped_matmul)
    return out[inv]


def _flatten_route(x, weights, indices):
    T = x.shape[0]
    topk = indices.shape[1]
    tok = torch.arange(T, device=x.device).repeat_interleave(topk)
    eid = indices.reshape(-1)
    w = weights.reshape(-1).float()
    return tok, eid, w


def moe_dispatch_ep(x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
                    experts: GroupedExperts, n_routed_experts: int) -> torch.Tensor:
    """"""
    ep = _EP
    T, dim = x.shape
    device = x.device
    tok, eid, w = _flatten_route(x, weights, indices)

    if ep is None or ep.size == 1 or not dist.is_initialized():
        n_local = ep.experts_per_rank if ep is not None else n_routed_experts
        out = _local_grouped_ffn(x[tok], eid, w, experts, n_local)
        y = torch.zeros(T, dim, dtype=torch.float32, device=device)
        y = y.index_add(0, tok, out.to(y.dtype))
        return y

    owner = torch.div(eid, ep.experts_per_rank, rounding_mode="floor")
    leid = eid - owner * ep.experts_per_rank
    order = torch.argsort(owner, stable=True)
    tok, owner, leid, w = tok[order], owner[order], leid[order], w[order]
    xf = x[tok].float()

    input_splits = torch.bincount(owner, minlength=ep.size)
    output_splits = torch.empty_like(input_splits)
    dist.all_to_all_single(output_splits, input_splits, group=ep.group)
    in_s, out_s = input_splits.tolist(), output_splits.tolist()

    payload = torch.cat([xf, w[:, None]], dim=1)
    recv = _AllToAll.apply(payload, out_s, in_s, ep.group)
    x_recv, w_recv = recv[:, :dim], recv[:, dim]
    leid_recv = _a2a_ints(leid, out_s, in_s, ep.group)

    out_recv = _local_grouped_ffn(x_recv, leid_recv, w_recv, experts, ep.experts_per_rank)

    out_back = _AllToAll.apply(out_recv, in_s, out_s, ep.group)
    y = torch.zeros(T, dim, dtype=torch.float32, device=device)
    y = y.index_add(0, tok, out_back.to(y.dtype))
    return y


register_kernel(_MOE_OP, "npu", moe_dispatch_ep)


def enable() -> None:
    from .kernels import set_active_backend
    set_active_backend("npu")
