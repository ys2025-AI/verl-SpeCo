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
"""Key mapping + dequantization for loading released DSV4 DSpark weights."""
from __future__ import annotations

import re

_HC_SITE = {"hc_attn": "attn_hc", "hc_ffn": "ffn_hc"}


def map_released_key(key: str, n_draft_layers: int = 3) -> str | None:
    if key.endswith(".scale"):
        return None
    if key.endswith(("_offset", "_scale")):
        return None
    if key.startswith("layers."):
        return None
    if key == "embed.weight":
        return "embed_tokens.weight"
    if key == "head.weight":
        return "lm_head.weight"
    if key in ("norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"):
        return None
    m = re.match(r"^mtp\.(\d+)\.(.*)$", key)
    if not m:
        return None
    stage, rest = int(m.group(1)), m.group(2)
    if rest in ("embed.weight", "head.weight", "norm.weight", "main_proj.weight", "main_norm.weight"):
        if rest == "main_proj.weight":
            return "fc.weight"
        if rest == "main_norm.weight":
            return "hidden_norm.weight"
        if rest == "norm.weight":
            return "norm.weight"
        return None
    if rest.startswith("markov_head."):
        return rest
    if rest == "confidence_head.proj.weight":
        return rest
    hh = re.match(r"^hc_head_(fn|base|scale)$", rest)
    if hh:
        return f"hc_head.hc_{hh.group(1)}"
    hc = re.match(r"^hc_(attn|ffn)_(fn|base|scale)$", rest)
    if hc:
        return f"layers.{stage}.{_HC_SITE['hc_' + hc.group(1)]}.{hc.group(2)}"
    if rest.startswith("ffn.gate."):
        return f"layers.{stage}.ffn.router.{rest[len('ffn.gate.'):]}"

    expert_m = re.match(r"^ffn\.experts\.(\d+)\.(w[123])\.weight$", rest)
    if expert_m:
        e, w = int(expert_m.group(1)), expert_m.group(2)
        return f"layers.{stage}.ffn.experts.{w}_idx_{e}"

    if rest.startswith(("attn.", "attn_norm.", "ffn_norm.", "ffn.shared_experts.")):
        return f"layers.{stage}.{rest}"
    return None


# FP4 e2m1fn lookup table: 4-bit index → float value
# Values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}
_FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _unpack_fp4(int8_tensor: torch.Tensor) -> torch.Tensor:
    """Unpack int8-packed FP4 (2 values per byte) → float32.

    High nibble = first value, low nibble = second value.
    """
    import torch
    t = int8_tensor.to(torch.int16)  # avoid overflow with uint8 view
    low = (t & 0x0F).to(torch.int64)
    high = ((t >> 4) & 0x0F).to(torch.int64)
    # Lookup table
    table = torch.tensor(_FP4_TABLE, dtype=torch.float32)
    low_vals = table[low]
    high_vals = table[high]
    # Interleave: [l0, h0, l1, h1, ...] → [l0, h0, l1, h1, ...]
    out = torch.stack([low_vals, high_vals], dim=-1).reshape(*low_vals.shape[:-1], -1)
    return out


def _expand_scale(scale: torch.Tensor, weight_shape: tuple, block: tuple = (128, 128)) -> torch.Tensor:
    """Expand block-wise scale to match weight shape.

    Supports:
      - scale shape == weight shape // block → standard block-wise
      - scale shape[0] == weight shape[0] → per-row block-wise (dim 1 only)
    """
    import torch
    scale = scale.to(torch.float32)
    if scale.shape == (weight_shape[0] // block[0], weight_shape[1] // block[1]):
        # Standard block: expand both dims
        return scale.repeat_interleave(block[0], dim=0).repeat_interleave(block[1], dim=1)
    if scale.dim() == 2 and scale.shape[0] == weight_shape[0]:
        # Per-row block: expand dim 1 only
        rep = weight_shape[1] // scale.shape[1]
        return scale.repeat_interleave(rep, dim=1)
    # Fallback: try element-wise broadcast
    if scale.shape == weight_shape:
        return scale
    # Try expanding with repeat_interleave
    rep0 = weight_shape[0] // scale.shape[0] if scale.dim() >= 1 else 1
    rep1 = weight_shape[1] // scale.shape[1] if scale.dim() >= 2 else 1
    return scale.repeat_interleave(rep0, dim=0).repeat_interleave(rep1, dim=1)


def _dequant_weight(weight, scale, offset, target_dtype=None):
    """Dequantize int8 weight with float scale/offset → target dtype.

    Used for W8A8 format: int8 + f32 scale + f32 offset.
    """
    import torch
    if target_dtype is None:
        target_dtype = torch.bfloat16
    result = (weight.to(torch.float32) * scale + offset).to(target_dtype)
    return result


def _dequant_fp8(weight, scale, target_dtype=None, block=(128, 128)):
    """Dequantize FP8 (e4m3fn) weight with e8m0fnu scale → target dtype.

    weight: float8_e4m3fn
    scale: float8_e8m0fnu, block-wise
    """
    import torch
    if target_dtype is None:
        target_dtype = torch.bfloat16
    w_f32 = weight.to(torch.float32)
    s_f32 = scale.to(torch.float32)
    s_expanded = _expand_scale(s_f32, tuple(w_f32.shape), block)
    return (w_f32 * s_expanded).to(target_dtype)


def _dequant_fp4_packed(weight, scale, target_dtype=None, block=(128, 32)):
    """Dequantize FP4-packed int8 weight with e8m0fnu scale → target dtype.

    weight: int8, shape (out, in//2) — 2 FP4 values packed per byte
    scale: float8_e8m0fnu, block-wise over packed dim
    """
    import torch
    if target_dtype is None:
        target_dtype = torch.bfloat16
    # Unpack FP4 → float32, shape (out, in)
    w_f32 = _unpack_fp4(weight)
    s_f32 = scale.to(torch.float32)
    # Scale is over packed dim; expand to unpacked dim
    s_expanded = _expand_scale(s_f32, tuple(w_f32.shape), block)
    return (w_f32 * s_expanded).to(target_dtype)


def load_released_draft(
    model, checkpoint_path: str, n_draft_layers: int = 3, verbose: bool = False,
    target_dtype=None,
):
    """Load released DSV4 DSpark weights (mtp.* namespace) into our model.

    Auto-detects weight precision and converts to target_dtype:
      - FP8 (float8_e4m3fn) + e8m0fnu scale → target_dtype (default bf16)
      - FP4 (int8-packed) + e8m0fnu scale → target_dtype (default bf16)
      - INT8 + f32 scale/offset → target_dtype (W8A8 format, default bf16)
      - BF16/FP32 → target_dtype (if different, else as-is)

    Args:
        model: DSV4DSparkDraftModel instance
        checkpoint_path: path to checkpoint directory
        n_draft_layers: number of draft layers to load
        verbose: print loading stats
        target_dtype: target dtype for dequantized weights (default: torch.bfloat16)
    """
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import torch
    from safetensors import safe_open

    if target_dtype is None:
        target_dtype = torch.bfloat16

    # Load quantization config if available
    quant_cfg = {}
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        quant_cfg = cfg.get("quantization_config") or {}
        cfg.get("expert_dtype") or ""
    else:
        pass

    block_size = tuple(quant_cfg.get("weight_block_size", [128, 128]))

    index_path = None
    for name in (
        "model.safetensors.index.json",
        "quant_model_weights.safetensors.index.json",
    ):
        p = os.path.join(checkpoint_path, name)
        if os.path.exists(p):
            index_path = p
            break

    if index_path:
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        mtp_keys = [
            k for k in weight_map
            if k.startswith("mtp.") or k in ("embed.weight", "head.weight")
        ]
        shards_needed = sorted({weight_map[k] for k in mtp_keys})
    else:
        st_path = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(st_path):
            st_path = os.path.join(checkpoint_path, "quant_model_weights.safetensors")
        if not os.path.exists(st_path):
            raise FileNotFoundError(f"No safetensors found in {checkpoint_path}")
        shards_needed = [os.path.basename(st_path)]
        weight_map = None
        checkpoint_path = os.path.dirname(st_path)

    model_state = dict(model.named_parameters())
    # Also include named_buffers() — the Router's noaux_tc bias is a
    # register_buffer (not a Parameter), so named_parameters() misses it.
    # Without loading the released bias (~3.0), the router uses bias=0 →
    # expert collapse → 0-token routing → garbage output + all_to_all deadlock.
    buffer_state = dict(model.named_buffers())
    for name, buf in buffer_state.items():
        if name not in model_state and buf.numel() > 0:
            model_state[name] = buf
    loaded = {}
    skipped = []

    def _process_shard(shard_name):
        """Load, auto-detect dtype, dequantize all MTP tensors from one shard."""
        shard_path = os.path.join(checkpoint_path, shard_name)
        if not os.path.exists(shard_path):
            return {}
        results = {}
        with safe_open(shard_path, framework="pt") as f:
            shard_keys = list(f.keys())
            for key in shard_keys:
                if key.endswith(("_offset", "_scale", ".scale")):
                    continue
                if not (key.startswith("mtp.") or key in ("embed.weight", "head.weight")):
                    continue
                mapped = map_released_key(key, n_draft_layers)
                if mapped is None:
                    continue

                tensor = f.get_tensor(key)

                # Auto-detect precision from actual checkpoint contents (not config)
                is_expert = ".experts." in key and key.endswith(".weight")

                if tensor.dtype == torch.float8_e4m3fn:
                    # FP8 (e4m3fn) weight + e8m0fnu block-wise scale
                    scale_key = key.replace(".weight", ".scale")
                    if scale_key in shard_keys:
                        scale = f.get_tensor(scale_key)
                        tensor = _dequant_fp8(tensor, scale, target_dtype, block_size)
                    else:
                        tensor = tensor.to(target_dtype)

                elif tensor.dtype == torch.int8:
                    # Detect scale key format: .scale (FP4/FP8 block) vs _scale (W8A8 per-row)
                    fp4_scale_key = key.replace(".weight", ".scale")
                    w8a8_scale_key = key + "_scale"
                    if is_expert and fp4_scale_key in shard_keys:
                        # FP4-packed int8 + e8m0fnu scale (original checkpoint)
                        scale = f.get_tensor(fp4_scale_key)
                        fp4_block = (block_size[0], block_size[1] // 2)
                        tensor = _dequant_fp4_packed(tensor, scale, target_dtype, fp4_block)
                    elif w8a8_scale_key in shard_keys:
                        # W8A8: int8 + f32 scale/offset
                        scale = f.get_tensor(w8a8_scale_key)
                        offset_key = key + "_offset"
                        offset = f.get_tensor(offset_key) if offset_key in shard_keys else torch.zeros_like(scale)
                        tensor = _dequant_weight(tensor, scale, offset, target_dtype)
                    else:
                        tensor = tensor.to(target_dtype)

                elif tensor.dtype != target_dtype:
                    tensor = tensor.to(target_dtype)

                results[key] = (mapped, tensor)
        return results

    with ThreadPoolExecutor(max_workers=min(4, len(shards_needed))) as pool:
        future_to_shard = {pool.submit(_process_shard, s): s for s in shards_needed}
        all_results = {}
        for future in as_completed(future_to_shard):
            all_results.update(future.result())

    for key, (mapped, tensor) in all_results.items():
        if mapped.endswith("_idx_") or ("_idx_" in mapped and ".experts." in mapped):
            parts = mapped.rsplit("_idx_", 1)
            base = parts[0]
            expert_idx = int(parts[1])
            if base in model_state:
                param = model_state[base]
                slot = min(expert_idx, param.shape[0] - 1)
                if param.shape[1:] == tensor.shape:
                    param.data[slot].copy_(tensor.to(param.dtype))
                    loaded[f"{base}[{slot}]"] = key
                else:
                    skipped.append((mapped, key, tuple(tensor.shape), tuple(param.shape[1:])))
            continue

        if mapped in model_state:
            param = model_state[mapped]
            if param.shape == tensor.shape:
                param.data.copy_(tensor.to(param.dtype))
                loaded[mapped] = key
            else:
                skipped.append((mapped, key, tuple(tensor.shape), tuple(param.shape)))

    if verbose:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("DSV4 DSpark: loaded %d tensors, %d skipped, target_dtype=%s",
                    len(loaded), len(skipped), target_dtype)
        for m, k, ts, ms in skipped[:5]:
            logger.warning("  shape mismatch: %s ← %s: %s != %s", m, k, ts, ms)
    return loaded
