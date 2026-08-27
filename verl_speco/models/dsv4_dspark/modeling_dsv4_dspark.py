# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Subclasses :class:`DSparkDraftModel` and swaps the decoder stack for the"""
from __future__ import annotations

import logging
import os

import torch
from torch import nn

from verl_speco.models.dspark import DSparkDraftModel

from .backbone.block import MhcDecoderBlock
from .backbone.hyper import HyperHead
from .backbone.rotary import precompute_freqs_cis
from .configuration_dsv4_dspark import DSV4DSparkConfig

logger = logging.getLogger(__name__)


class DSV4DSparkDraftModel(DSparkDraftModel):
    config_class = DSV4DSparkConfig
    _no_split_modules = ["MhcDecoderBlock"]

    def __init__(self, config: DSV4DSparkConfig) -> None:
        super().__init__(config)
        bb = _BackboneConfig(config)
        self.backbone_cfg = bb
        self.grad_checkpoint = bool(int(os.environ.get("DSPARK_RECOMPUTE", "0")))

        self.layers = nn.ModuleList(MhcDecoderBlock(bb) for _ in range(bb.n_draft_layers))
        self.hc_head = HyperHead(bb)

        self._rope_dim = bb.rope_head_dim
        self.register_buffer(
            "freqs_cis",
            torch.view_as_real(precompute_freqs_cis(
                bb.rope_head_dim, bb.original_seq_len or 1, 0, bb.rope_theta,
                bb.rope_factor, bb.beta_fast, bb.beta_slow,
            )).contiguous(),
            persistent=False,
        )
        self._init_backbone_params()

    def _init_backbone_params(self) -> None:
        if any(p.is_meta for p in self.parameters()):
            return
        std = 0.02
        for m in [*self.layers, self.hc_head]:
            for name, p in m.named_parameters():
                if p.dim() >= 2 and (".fn" in name or "weight" in name or "hc_fn" in name) and (
                    torch.isnan(p).any() or not p.abs().sum().isfinite() or p.abs().sum() == 0
                ):
                    nn.init.normal_(p, std=std)

    def _rebuild_freqs(self, seqlen: int) -> None:
        bb = self.backbone_cfg
        self.freqs_cis = torch.view_as_real(precompute_freqs_cis(
            self._rope_dim, seqlen, 0, bb.rope_theta,
            bb.rope_factor, bb.beta_fast, bb.beta_slow,
        )).contiguous()

    def _ep_active(self) -> bool:
        from verl_speco.models.dsv4_dspark.backbone import moe_ep
        return moe_ep._EP is not None and moe_ep._EP.size > 1

    def load_embedding(self, model_path: str, embedding_key: str = "model.embed_tokens.weight") -> None:
        import glob as glob_mod
        import json

        from safetensors import safe_open
        from transformers.utils import snapshot_download

        if not os.path.exists(model_path):
            model_path = snapshot_download(repo_id=model_path)

        index_json_path = glob_mod.glob(os.path.join(model_path, "*.index.json"))
        if len(index_json_path) == 0:
            super().load_embedding(model_path, embedding_key)
            return
        if len(index_json_path) > 1:
            picked = None
            for ip in index_json_path:
                try:
                    with open(ip, "r") as f:
                        if embedding_key in json.load(f).get("weight_map", {}):
                            picked = ip
                            break
                except (OSError, json.JSONDecodeError):
                    continue
            if picked is None:
                raise FileNotFoundError(
                    f"{embedding_key!r} not in any index.json under {model_path}"
                )
            index_json_path = [picked]

        with open(index_json_path[0], "r") as f:
            index_json = json.load(f)
        ckpt_file = index_json["weight_map"][embedding_key]
        if ckpt_file.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, ckpt_file), framework="pt") as f:
                self.embed_tokens.weight.copy_(f.get_tensor(embedding_key))
        else:
            state_dict = torch.load(
                os.path.join(model_path, ckpt_file),
                map_location="cpu",
                weights_only=True,
            )
            self.embed_tokens.weight.copy_(state_dict[embedding_key])

    def fsdp_wrap_plan(self) -> list:
        plan = []
        for block in self.layers:
            if not self._ep_active():
                plan.append(block.ffn.experts)
            plan.append(block.ffn.shared_experts)
            plan.append(block.attn)
            plan.append(block)
        return plan

    def fsdp_ignored_params(self) -> set:
        if not self._ep_active():
            return set()
        params = set()
        for block in self.layers:
            params.update(block.ffn.experts.parameters())
        # Also ignore frozen params that shouldn't be FSDP2-cast
        if hasattr(self, 'embed_tokens'):
            params.update(self.embed_tokens.parameters())
        return params

    def shard_experts_as_dtensor(self, mesh) -> None:
        from torch.distributed.tensor import DTensor, Shard
        for module in self.modules():
            if type(module).__name__ != "GroupedExperts":
                continue
            for name in ("w1", "w2", "w3"):
                p = getattr(module, name, None)
                if p is None or isinstance(p.data, DTensor):
                    continue
                dt = DTensor.from_local(p.data, mesh, [Shard(0)], run_check=False)
                setattr(module, name, torch.nn.Parameter(dt, requires_grad=p.requires_grad))

    def _rope_at(self, positions: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "_freqs_ok", False):
            self._freqs_ok = True
            self._rebuild_freqs(max(self.freqs_cis.shape[0], 1))
        if positions.numel() and int(positions.max()) >= self.freqs_cis.shape[0]:
            self._rebuild_freqs(int(positions.max()) + 1)
        return self.freqs_cis.to(positions.device)[positions]

    @staticmethod
    def _mask_to_bias(
        dense_attention_mask: torch.Tensor | None,
        block_mask=None,
    ) -> torch.Tensor | None:
        """verl-SpeCo builds ``dense_attention_mask [bsz, 1, draft_len, ctx+draft]``"""
        if dense_attention_mask is None:
            return None
        mask = dense_attention_mask
        while mask.dim() > 3:
            mask = mask.squeeze(1)
        neg_inf = torch.finfo(mask.float().dtype).min
        return torch.where(mask.bool(), 0.0, neg_inf).to(torch.float32)

    def forward(
        self,
        draft_input_ids: torch.Tensor | None,
        context_feature: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        dense_attention_mask: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The call signature matches :meth:`DFlashDraftModel.forward` so the"""
        if noise_embedding is not None:
            draft_hidden = noise_embedding.to(context_feature.dtype)
        else:
            draft_hidden = self.embed_tokens(draft_input_ids).to(context_feature.dtype)

        draft_hidden.shape[0]

        block_freqs = self._rope_at(draft_position_ids.reshape(-1).long())
        context_freqs = self._rope_at(context_position_ids.reshape(-1).long())
        attn_bias = self._mask_to_bias(dense_attention_mask, block_mask)

        hc = self.backbone_cfg.hc_mult
        streams = draft_hidden.unsqueeze(2).repeat(1, 1, hc, 1)

        from torch.utils.checkpoint import checkpoint

        if self.training:
            for layer in self.layers:
                if hasattr(layer.ffn.router, 'update_load_balance_bias'):
                    layer.ffn.router.update_load_balance_bias()

        for layer_idx, layer in enumerate(self.layers):
            if self.grad_checkpoint and self.training:
                streams = checkpoint(
                    layer,
                    streams,
                    context_feature,
                    block_freqs,
                    context_freqs,
                    attn_bias,
                    use_reentrant=False,
                )
            else:
                streams = layer(
                    streams,
                    context_feature,
                    block_freqs,
                    context_freqs,
                    attn_bias,
                )
            if os.environ.get("MOE_NAN_DBG") == "1":
                import torch.distributed as _d
                _r = _d.get_rank() if _d.is_initialized() else 0
                if _r == 0 and not torch.isfinite(streams).all():
                    _n = int((~torch.isfinite(streams)).sum().item())
                    logger.error(
                        f"[FWD-NaN] after layer {layer_idx}: {_n}/{streams.numel()} non-finite")

        hidden = self.norm(self.hc_head(streams))
        if os.environ.get("NAN_HOOK") == "1":
            import torch.distributed as _d
        return hidden.contiguous()


class _BackboneConfig:
    """Bridges :class:`DSV4DSparkConfig` (a PretrainedConfig subclass) to the"""

    def __init__(self, config: DSV4DSparkConfig) -> None:
        self.hidden_size = int(config.hidden_size)
        self.rms_norm_eps = float(config.rms_norm_eps)
        self.n_draft_layers = int(config.num_hidden_layers)
        self.block_size = int(getattr(config, "block_size", 7))
        self.noise_token_id = int(getattr(config, "mask_token_id", 0) or 0)
        target_layer_ids = getattr(config, "target_layer_ids", None)
        if target_layer_ids is None:
            target_layer_ids = tuple(
                range(int(getattr(config, "num_context_layers", 5) or 5))
            )
        self.target_layer_ids = tuple(int(x) for x in target_layer_ids)
        self.markov_rank = int(getattr(config, "markov_rank", 256))
        self.num_heads = int(config.dsv4_num_heads)
        self.head_dim = int(config.dsv4_head_dim)
        self.rope_head_dim = int(config.dsv4_rope_head_dim)
        self.q_lora_rank = int(config.dsv4_q_lora_rank)
        self.o_lora_rank = int(config.dsv4_o_lora_rank)
        self.o_groups = int(config.dsv4_o_groups)
        self.window_size = int(config.dsv4_window_size)
        self.rope_theta = float(config.dsv4_rope_theta)
        self.rope_factor = float(config.dsv4_rope_factor)
        self.original_seq_len = int(config.dsv4_original_seq_len)
        self.beta_fast = float(config.dsv4_beta_fast)
        self.beta_slow = float(config.dsv4_beta_slow)
        self.n_routed_experts = int(config.dsv4_n_routed_experts)
        self.n_shared_experts = int(config.dsv4_n_shared_experts)
        self.n_activated_experts = int(config.dsv4_n_activated_experts)
        self.moe_inter_dim = int(config.dsv4_moe_inter_dim)
        self.score_func = str(config.dsv4_score_func)
        self.route_scale = float(config.dsv4_route_scale)
        self.swiglu_limit = float(config.dsv4_swiglu_limit)
        self.hc_mult = int(config.dsv4_hc_mult)
        self.hc_sinkhorn_iters = int(config.dsv4_hc_sinkhorn_iters)
        self.hc_eps = float(config.dsv4_hc_eps)

    @property
    def num_target_layers(self) -> int:
        return len(self.target_layer_ids)

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim
