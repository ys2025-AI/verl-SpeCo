# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
# SPDX-License-Identifier: Apache-2.0
"""DSV4 DSpark Expert Parallelism (EP) and FSDP2 setup utilities.

Extracted from `base_trainer.py` to keep the shared trainer file clean.
All functions are called only when `_is_ep_enabled()` returns True
(i.e., `speculative_algorithm == "DSV4_DSPARK"` and `dsv4_dspark_enable_ep == True`).
"""
from __future__ import annotations

import logging
import os
import tempfile
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

logger = logging.getLogger(__name__)


def _init_router_bias(model, world_size, experts_per_rank, rank):
    """Initialize MoE router bias so each rank's experts get roughly equal load.

    With bias=0, random router weights cause tokens to cluster on a few experts,
    producing 0-token ranks and deadlocking the all_to_all in moe_ep.
    We set bias[i] = +boost for local experts and bias[j] = -boost for remote,
    scaled to be large enough to influence routing even with large score variance.
    """
    n_experts = world_size * experts_per_rank
    boost = 0.01  # small perturbation to break symmetry, not force all-local routing
    for module in model.modules():
        router = getattr(module, "router", None)
        if router is None:
            continue
        if not hasattr(router, "bias") or router.bias.shape[0] != n_experts:
            continue
        with torch.no_grad():
            local_start = rank * experts_per_rank
            local_end = local_start + experts_per_rank
            bias = torch.full_like(router.bias, -boost)
            bias[local_start:local_end] = boost
            # Add small random noise to break exact symmetry within local experts
            bias[local_start:local_end] += torch.randn(experts_per_rank, device=bias.device) * 0.1
            router.bias.copy_(bias)
            if hasattr(router, "_bias_initialized"):
                router._bias_initialized = True
    if rank == 0:
        logger.info(f"[EP] Router bias initialized: boost=±{boost} for {experts_per_rank} local experts/rank")


def is_ep_enabled(config) -> bool:
    algorithm = str(
        config.rollout.drafter.get("speculative_algorithm", "")
    ).strip().upper()
    if algorithm != "DSV4_DSPARK":
        return False
    return bool(
        config.rollout.drafter.training.get("dsv4_dspark_enable_ep", False)
    )


def apply_ep_only(trainer, raw_model: torch.nn.Module, use_skip_loading: bool):
    """Apply EP only (no FSDP2). Each rank holds 32 experts locally, no param sharding.

    If moe_ep is already configured (pre-build), the model already has 32 experts
    per rank — skip the file-share and EP slice steps.
    """
    from verl_speco.models.dsv4_dspark.backbone import moe_ep

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    n_experts = int(
        trainer.config.rollout.drafter.training.get(
            "dsv4_dspark_backbone_n_routed_experts", 256
        )
    )
    experts_per_rank = n_experts // world_size
    pre_configured = moe_ep._EP is not None and moe_ep._EP.size > 1
    if pre_configured:
        # EP was configured before build_model — model already has 32 experts/rank.
        # Skip file-share and EP slice; just move to device and set up grads.
        logger.info(f"[Rank {rank}] EP pre-configured: model already has {experts_per_rank} experts, skipping file-share + slice")
    else:
        # Step 1: File-share weights from rank0
        shard_dir = tempfile.gettempdir()
        state_file = os.path.join(shard_dir, "dsv4_dspark_rank0_state.pt")
        ready_file = os.path.join(shard_dir, "dsv4_dspark_rank0_ready.flag")

        for f in (state_file, ready_file):
            try:
                os.remove(f)
            except OSError:
                pass
        dist.barrier()

        if rank == 0:
            state_dict = raw_model.state_dict()
            torch.save(state_dict, state_file)
            del state_dict
            with open(ready_file, "w") as f:
                f.write(str(os.path.getsize(state_file)))
            logger.info(f"[Rank 0] Saved full state dict to {state_file}")
        elif use_skip_loading:
            wait_start = time.time()
            expected_size = None
            while True:
                if os.path.exists(ready_file):
                    try:
                        with open(ready_file) as f:
                            expected_size = int(f.read().strip())
                    except (OSError, ValueError):
                        expected_size = None
                if (
                    expected_size is not None
                    and os.path.exists(state_file)
                    and os.path.getsize(state_file) == expected_size
                ):
                    break
                if time.time() - wait_start > 600:
                    raise TimeoutError(f"Rank {rank} timed out waiting for {ready_file}")
                time.sleep(2)
            loaded_state = torch.load(state_file, map_location="cpu", weights_only=True)
            raw_model.load_state_dict(loaded_state, strict=False)
            del loaded_state
            logger.info(f"[Rank {rank}] Loaded state dict (waited {time.time()-wait_start:.0f}s)")

        dist.barrier()
        if rank == 0:
            for f in (state_file, ready_file):
                try:
                    os.remove(f)
                except OSError:
                    pass

        # Step 2: EP slice — 256 → 32 experts/rank
        for module in raw_model.modules():
            experts = getattr(module, "experts", None)
            if experts is None or not hasattr(experts, "w1") or experts.w1.dim() < 3:
                continue
            if experts.w1.shape[0] == n_experts:
                start = rank * experts_per_rank
                end = start + experts_per_rank
                for w_name in ("w1", "w3", "w2"):
                    param = getattr(experts, w_name)
                    setattr(experts, w_name, torch.nn.Parameter(
                        param.data[start:end].clone()
                    ))
                if hasattr(experts, "num_local_experts"):
                    experts.num_local_experts = experts_per_rank
            module._ep_patched = True
            module._ep_num_experts = n_experts
            module._ep_experts_per_rank = experts_per_rank

        # Step 3: Configure moe_ep
        moe_ep.configure(
            group=dist.group.WORLD,
            rank=rank,
            size=world_size,
            experts_per_rank=experts_per_rank,
        )
        moe_ep.enable()

    # Step 5: Enable grouped-GEMM
    enable_grouped_gemm = bool(
        trainer.config.rollout.drafter.training.get(
            "dsv4_dspark_enable_grouped_gemm", False
        )
    )
    if enable_grouped_gemm:
        from verl_speco.models.dsv4_dspark.backbone import moe_grouped_gemm
        moe_grouped_gemm.enable()
        if rank == 0:
            logger.info("[EP] grouped-GEMM enabled")

    # Step 6: Move to device and set requires_grad
    raw_model.to(trainer.runtime_device)
    n_trainable = 0
    expert_param_ids = set()
    for name, p in raw_model.named_parameters():
        if "embed_tokens" in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
            n_trainable += 1

    # Collect expert param ids (local, no gradient sync needed)
    for module in raw_model.modules():
        if not getattr(module, "_ep_patched", False):
            continue
        experts = getattr(module, "experts", None)
        if experts is not None:
            expert_param_ids.update(id(p) for p in experts.parameters())
    trainer._ep_expert_param_ids = expert_param_ids
    trainer._ep_ddp_world_size = world_size

    if rank == 0:
        logger.info(f"[EP+DDP] {n_trainable} trainable params, "
                   f"{len(expert_param_ids)} expert params (local), "
                   f"{n_trainable - len(expert_param_ids)} non-expert params (DDP sync)")

    # Step 7: Rebuild freqs_cis on CPU
    draft_model = getattr(raw_model, "draft_model", None) or raw_model
    if hasattr(draft_model, "_rebuild_freqs"):
        seq_len = draft_model.freqs_cis.shape[0] if hasattr(draft_model, "freqs_cis") else 1
        draft_model._rebuild_freqs(max(seq_len, 1))
        draft_model._freqs_ok = True
        if rank == 0:
            logger.info(f"[EP-only] Rebuilt freqs_cis (CPU, {draft_model.freqs_cis.shape})")

    # Step 8: Initialize router bias to break routing symmetry before first forward.
    # With random init or fresh checkpoint, bias=0 causes all tokens to route to
    # the same expert, producing 0-token ranks → all_to_all deadlock in moe_ep.
    # Inject rank-aware bias so each rank's experts get roughly equal load.
    _init_router_bias(raw_model, world_size, experts_per_rank, rank)

    logger.info(
        f"[Rank {rank}] EP+DDP applied: EP={world_size}, "
        f"{experts_per_rank} experts/rank, "
        f"non-expert grads synced via all-reduce"
    )


def apply_ep_fsdp2(trainer, raw_model: torch.nn.Module, fsdp_kwargs: dict, fsdp_config, use_skip_loading: bool):
    """Apply FSDP2 with EP -- twinkle-aligned approach.

    Key: FSDP2-wrap the DRAFT MODEL (inner model with layers), NOT the
    training wrapper. The training wrapper calls draft_model.forward()
    which triggers FSDP2 hooks. Training wrapper's own params (if any)
    are not FSDP2-managed.

    Flow:
    1. File-share weights from rank0
    2. EP slice: 256 -> 32 experts/rank
    3. Configure moe_ep + grouped-GEMM
    4. Move to device
    5. shard_experts_as_dtensor: wrap expert params as Shard(0) DTensors
    6. FSDP2 wrap draft_model: per-layer (shared_experts, attn, block) + root
       with ignored_params=expert DTensors
    7. Set self.model = raw_model (training wrapper still calls draft_model)
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    n_experts = int(
        trainer.config.rollout.drafter.training.get(
            "dsv4_dspark_backbone_n_routed_experts", 256
        )
    )
    if n_experts % world_size != 0:
        raise ValueError(
            f"EP: n_routed_experts ({n_experts}) must be divisible by "
            f"world_size ({world_size})"
        )
    experts_per_rank = n_experts // world_size
    mesh = trainer.training_device_mesh
    random_init = bool(
        trainer.config.rollout.drafter.training.get("dsv4_dspark_random_init", False)
    )

    from verl_speco.models.dsv4_dspark.backbone import moe_ep
    pre_configured = moe_ep._EP is not None and moe_ep._EP.size > 1

    if pre_configured and (random_init and use_skip_loading):
        # EP was configured before build_model — model already has 32 experts/rank.
        # No file-share, no EP slice, no moe_ep config needed.
        logger.info(f"[Rank {rank}] Random init + EP pre-configured: skipping file-share + EP slice + moe_ep config")
    else:
        # EP not pre-configured — handle random init (no EP) or checkpoint (file-share + slice)
        if random_init and use_skip_loading:
            # Random init without EP: skip everything, use degenerate MoE path
            logger.info(f"[Rank {rank}] Random init (no EP pre-config): skipping file-share + EP slice + moe_ep config")
        else:
            # Checkpoint path: file-share + EP slice + moe_ep config
            shard_dir = tempfile.gettempdir()
            state_file = os.path.join(shard_dir, "dsv4_dspark_rank0_state.pt")
            ready_file = os.path.join(shard_dir, "dsv4_dspark_rank0_ready.flag")

        for f in (state_file, ready_file):
            try:
                os.remove(f)
            except OSError:
                pass
        dist.barrier()

        if rank == 0:
            state_dict = raw_model.state_dict()
            torch.save(state_dict, state_file)
            del state_dict
            with open(ready_file, "w") as f:
                f.write("ready")
            logger.info(f"[Rank 0] Saved full state dict to {state_file}")
        elif use_skip_loading:
            wait_start = time.time()
            while not os.path.exists(ready_file):
                if time.time() - wait_start > 600:
                    raise TimeoutError(f"Rank {rank} timed out waiting for {ready_file}")
                time.sleep(2)
            loaded_state = torch.load(state_file, map_location="cpu", weights_only=True)
            raw_model.load_state_dict(loaded_state, strict=False)
            del loaded_state
            logger.info(f"[Rank {rank}] Loaded state dict (waited {time.time()-wait_start:.0f}s)")

        dist.barrier()
        if rank == 0:
            for f in (state_file, ready_file):
                try:
                    os.remove(f)
                except OSError:
                    pass

        # Step 2: EP slice -- 256 -> 32 experts/rank
        draft_model_inner = getattr(raw_model, "draft_model", None) or raw_model
        for module in draft_model_inner.modules():
            experts = getattr(module, "experts", None)
            if experts is None or not hasattr(experts, "w1") or experts.w1.dim() < 3:
                continue
            if experts.w1.shape[0] == n_experts:
                start = rank * experts_per_rank
                end = start + experts_per_rank
                for w_name in ("w1", "w3", "w2"):
                    param = getattr(experts, w_name)
                    setattr(experts, w_name, torch.nn.Parameter(
                        param.data[start:end].clone()
                    ))
                if hasattr(experts, "num_local_experts"):
                    experts.num_local_experts = experts_per_rank
            module._ep_patched = True
            module._ep_num_experts = n_experts
            module._ep_experts_per_rank = experts_per_rank

        # Step 3: Configure moe_ep + grouped-GEMM
        from verl_speco.models.dsv4_dspark.backbone import moe_ep
        moe_ep.configure(
            group=dist.group.WORLD,
            rank=rank,
            size=world_size,
            experts_per_rank=experts_per_rank,
        )
        moe_ep.enable()

    enable_grouped_gemm = bool(
        trainer.config.rollout.drafter.training.get(
            "dsv4_dspark_enable_grouped_gemm", False
        )
    )
    if enable_grouped_gemm:
        from verl_speco.models.dsv4_dspark.backbone import moe_grouped_gemm
        moe_grouped_gemm.enable()
        if rank == 0:
            logger.info("[EP] grouped-GEMM enabled")

    # Step 4: Move to device (cast to bf16 first to halve memory, matching speculators)
    raw_model.to(torch.bfloat16)
    raw_model.to(trainer.runtime_device)

    # Step 5: shard_experts_as_dtensor — only under real EP (not random init)
    draft_model = getattr(raw_model, "draft_model", None) or raw_model
    if not random_init and hasattr(draft_model, 'shard_experts_as_dtensor'):
        draft_model.shard_experts_as_dtensor(mesh)
        if rank == 0:
            logger.info("[EP] Experts wrapped as Shard(0) DTensors")

    # Step 6: FSDP2 wrap draft_model per-layer + root
    mp_policy = fsdp_kwargs["mp_policy"]
    reshard = True

    ignored_params = None
    if not random_init and hasattr(draft_model, 'fsdp_ignored_params'):
        ignored_params = draft_model.fsdp_ignored_params() or None

    wrap_modules = []
    if hasattr(draft_model, 'fsdp_wrap_plan'):
        wrap_modules = draft_model.fsdp_wrap_plan()

    # Random init (no EP): don't wrap GroupedExperts separately —
    # _moe_dispatch_torch accesses w1/w2/w3 directly (not via experts()),
    # so FSDP2 pre-forward hook on GroupedExperts won't fire.
    # Let the parent block's FSDP2 wrap handle unshard.
    if random_init:
        wrap_modules = [m for m in wrap_modules
                        if not hasattr(m, 'w1') or not hasattr(m, 'num_local_experts')]

    extra = {"mesh": mesh, "reshard_after_forward": reshard, "mp_policy": mp_policy}
    if ignored_params:
        extra["ignored_params"] = ignored_params

    for mod in wrap_modules:
        fully_shard(mod, **extra)

    fully_shard(raw_model, mesh=mesh, reshard_after_forward=False,
                mp_policy=mp_policy,
                ignored_params=ignored_params if ignored_params else None)

    # Step 7: Rebuild freqs_cis
    if hasattr(draft_model, "_rebuild_freqs"):
        seq_len = draft_model.freqs_cis.shape[0] if hasattr(draft_model, "freqs_cis") else 1
        draft_model._rebuild_freqs(max(seq_len, 1))
        draft_model._freqs_ok = True
        if rank == 0:
            logger.info(f"[EP] Rebuilt freqs_cis (CPU, {draft_model.freqs_cis.shape})")

    # Mark FSDP2 mode (skip _sync_non_expert_grads)
    trainer._ep_ddp_world_size = 0  # 0 = FSDP2 mode (no manual gradient sync)

    # Initialize router bias to break routing symmetry before first forward.
    if random_init:
        _init_router_bias(raw_model, world_size, n_experts, 0)  # all experts local
        logger.info(
            f"[Rank {rank}] FSDP2-only (random init): {n_experts} experts/rank (no EP), "
            f"draft_model FSDP2-wrapped, MoE degenerate path (no all_to_all)"
        )
    else:
        _init_router_bias(raw_model, world_size, experts_per_rank, rank)
        logger.info(
            f"[Rank {rank}] EP+FSDP2 applied: EP={world_size}, "
            f"{experts_per_rank} experts/rank, "
            f"draft_model FSDP2-wrapped, "
            f"{len(ignored_params) if ignored_params else 0} expert params as DTensors"
        )


def sync_non_expert_grads(trainer) -> None:
    """All-reduce (average) gradients of non-expert params across the WORLD group.

    Expert params are local (EP): each rank owns a disjoint set of experts,
    so their gradients must NOT be reduced. Non-expert params (attention,
    norms, shared_experts, router, etc.) are replicated across all ranks;
    their gradients need averaging to produce a consistent update.
    """
    expert_ids = getattr(trainer, "_ep_expert_param_ids", None)
    if not expert_ids:
        return

    world_size = getattr(trainer, "_ep_ddp_world_size", 1)
    if world_size <= 1:
        return

    grads_to_reduce = []
    for p in trainer.model.parameters():
        if id(p) in expert_ids:
            continue
        if p.grad is not None:
            grads_to_reduce.append(p.grad)

    if not grads_to_reduce:
        return

    flat_grad = torch._utils._flatten_dense_tensors(grads_to_reduce)
    dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
    flat_grad.div_(float(world_size))

    split_grads = torch._utils._unflatten_dense_tensors(flat_grad, grads_to_reduce)
    for orig, synced in zip(grads_to_reduce, split_grads):
        orig.copy_(synced)
