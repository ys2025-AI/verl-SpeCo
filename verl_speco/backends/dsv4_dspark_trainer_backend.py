# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extends :class:`DSparkTrainerBackend` to build the DSV4-native sparse-backbone"""
from __future__ import annotations

import logging
import os
from copy import deepcopy

import torch

from verl_speco.backends.dspark_trainer_backend import (
    DSparkTrainerBackend,
    DSparkTrainingModel,
)
from verl_speco.models.dsv4_dspark import DSV4DSparkConfig, DSV4DSparkDraftModel
from verl_speco.trainer.checkpoint import log_drafter_checkpoint_step

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class DSV4DSparkTrainerBackend(DSparkTrainerBackend):
    """"""

    @property
    def model_type(self):
        return "dsv4_dspark"

    def _normalize_dflash_config(
        self, drafter_config, target_hf_config, normalized_state, spec_model_path
    ):
        training_cfg = self.config.rollout.drafter.training
        if training_cfg.get("dsv4_dspark_num_target_layers", None) is not None:
            drafter_config.num_context_layers = int(
                training_cfg["dsv4_dspark_num_target_layers"]
            )
        if normalized_state:
            fc_key = next((k for k in normalized_state if k.endswith("main_proj.weight")), None)
            if fc_key and fc_key in normalized_state:
                fc_w = normalized_state[fc_key]
                if fc_w.ndim == 2 and fc_w.size(0) > 0:
                    target_hs = int(
                        getattr(getattr(target_hf_config, "text_config", target_hf_config), "hidden_size",
                                getattr(drafter_config, "target_hidden_size", drafter_config.hidden_size))
                    )
                    inferred = fc_w.size(1) // target_hs
                    if inferred > 0 and inferred != getattr(drafter_config, "num_context_layers", 0):
                        logger.info(
                            "DSV4 DSpark: inferring num_context_layers=%d from %s shape %s",
                            inferred, fc_key, tuple(fc_w.shape),
                        )
                        drafter_config.num_context_layers = inferred
        return super()._normalize_dflash_config(
            drafter_config,
            target_hf_config,
            normalized_state,
            spec_model_path,
        )

    def _build_fallback_config(self, target_hf_config):
        training_cfg = self.config.rollout.drafter.training
        target_text_config = getattr(target_hf_config, "text_config", target_hf_config)
        hidden_size_cfg = training_cfg.get("dsv4_dspark_hidden_size", None)
        hidden_size = int(
            hidden_size_cfg
            if hidden_size_cfg is not None
            else target_text_config.hidden_size
        )
        num_context_layers = int(
            training_cfg.get("dsv4_dspark_num_target_layers", 3)
        )
        target_num_hidden_layers = int(
            getattr(target_text_config, "num_hidden_layers", 43)
        )
        mask_token_id_cfg = training_cfg.get("dsv4_dspark_mask_token_id", None)
        mask_token_id = int(
            mask_token_id_cfg
            if mask_token_id_cfg is not None
            else target_text_config.vocab_size - 1
        )
        target_layer_ids = training_cfg.get("dsv4_dspark_target_layer_ids", None)
        if target_layer_ids is None:
            from verl_speco.models.dflash import build_target_layer_ids

            target_layer_ids = build_target_layer_ids(
                num_context_layers, target_num_hidden_layers
            )
        return DSV4DSparkConfig(
            hidden_size=hidden_size,
            intermediate_size=int(
                getattr(target_text_config, "intermediate_size", hidden_size * 4)
            ),
            num_hidden_layers=int(
                training_cfg.get("dsv4_dspark_num_hidden_layers", 3)
            ),
            num_attention_heads=int(target_text_config.num_attention_heads),
            num_key_value_heads=int(
                getattr(
                    target_text_config,
                    "num_key_value_heads",
                    target_text_config.num_attention_heads,
                )
            ),
            vocab_size=int(target_text_config.vocab_size),
            rms_norm_eps=float(getattr(target_text_config, "rms_norm_eps", 1e-6)),
            max_position_embeddings=int(
                getattr(target_text_config, "max_position_embeddings", 32768)
            ),
            rope_theta=float(getattr(target_text_config, "rope_theta", 10000.0)),
            num_target_layers=target_num_hidden_layers,
            num_context_layers=num_context_layers,
            target_hidden_size=int(target_text_config.hidden_size),
            target_num_hidden_layers=target_num_hidden_layers,
            target_layer_ids=target_layer_ids,
            mask_token_id=mask_token_id,
            block_size=int(training_cfg.get("dsv4_dspark_block_size", 7)),
            num_anchors=int(training_cfg.get("dsv4_dspark_num_anchors", 512)),
            markov_rank=int(training_cfg.get("dsv4_dspark_markov_rank", 256)),
            markov_head_type=str(
                training_cfg.get("dsv4_dspark_markov_head_type", "vanilla")
            ),
            confidence_head_alpha=float(
                training_cfg.get("dsv4_dspark_confidence_head_alpha", 0.0)
            ),
            confidence_head_with_markov=bool(
                training_cfg.get("dsv4_dspark_confidence_head_with_markov", True)
            ),
            ce_loss_alpha=float(training_cfg.get("dsv4_dspark_ce_loss_alpha", 0.1)),
            l1_loss_alpha=float(training_cfg.get("dsv4_dspark_l1_loss_alpha", 0.0)),
            tv_loss_alpha=float(training_cfg.get("dsv4_dspark_tv_loss_alpha", 0.0)),
            loss_decay_gamma=float(
                training_cfg.get("dsv4_dspark_loss_decay_gamma", 7.0)
            ),
            dsv4_num_heads=int(training_cfg.get("dsv4_dspark_backbone_num_heads", 64)),
            dsv4_head_dim=int(training_cfg.get("dsv4_dspark_backbone_head_dim", 512)),
            dsv4_rope_head_dim=int(
                training_cfg.get("dsv4_dspark_backbone_rope_head_dim", 64)
            ),
            dsv4_q_lora_rank=int(
                training_cfg.get("dsv4_dspark_backbone_q_lora_rank", 1024)
            ),
            dsv4_o_lora_rank=int(
                training_cfg.get("dsv4_dspark_backbone_o_lora_rank", 1024)
            ),
            dsv4_o_groups=int(training_cfg.get("dsv4_dspark_backbone_o_groups", 8)),
            dsv4_window_size=int(
                training_cfg.get("dsv4_dspark_backbone_window_size", 128)
            ),
            dsv4_rope_theta=float(
                training_cfg.get("dsv4_dspark_backbone_rope_theta", 10000.0)
            ),
            dsv4_rope_factor=float(
                training_cfg.get("dsv4_dspark_backbone_rope_factor", 16.0)
            ),
            dsv4_original_seq_len=int(
                training_cfg.get("dsv4_dspark_backbone_original_seq_len", 65536)
            ),
            dsv4_beta_fast=float(
                training_cfg.get("dsv4_dspark_backbone_beta_fast", 32.0)
            ),
            dsv4_beta_slow=float(
                training_cfg.get("dsv4_dspark_backbone_beta_slow", 1.0)
            ),
            dsv4_n_routed_experts=int(
                training_cfg.get("dsv4_dspark_backbone_n_routed_experts", 256)
            ),
            dsv4_n_shared_experts=int(
                training_cfg.get("dsv4_dspark_backbone_n_shared_experts", 1)
            ),
            dsv4_n_activated_experts=int(
                training_cfg.get("dsv4_dspark_backbone_n_activated_experts", 6)
            ),
            dsv4_moe_inter_dim=int(
                training_cfg.get("dsv4_dspark_backbone_moe_inter_dim", 2048)
            ),
            dsv4_score_func=str(
                training_cfg.get("dsv4_dspark_backbone_score_func", "sqrtsoftplus")
            ),
            dsv4_route_scale=float(
                training_cfg.get("dsv4_dspark_backbone_route_scale", 1.5)
            ),
            dsv4_swiglu_limit=float(
                training_cfg.get("dsv4_dspark_backbone_swiglu_limit", 10.0)
            ),
            dsv4_hc_mult=int(training_cfg.get("dsv4_dspark_backbone_hc_mult", 4)),
            dsv4_hc_sinkhorn_iters=int(
                training_cfg.get("dsv4_dspark_backbone_hc_sinkhorn_iters", 20)
            ),
            dsv4_hc_eps=float(training_cfg.get("dsv4_dspark_backbone_hc_eps", 1e-6)),
        )

    def build_model(self):
        target_model_path = self.config.model.path
        spec_model_path = self.config.rollout.drafter.model_path
        config_path = os.path.join(spec_model_path, "config.json")
        target_hf_config = self._get_target_hf_config()
        normalized_state = None
        skip_loading = getattr(self, "_skip_weight_loading", False)

        if config_path and os.path.exists(config_path):
            drafter_config = DSV4DSparkConfig.from_dsv4_dspark_pretrained(spec_model_path)
            training_cfg = self.config.rollout.drafter.training
            drafter_config.num_hidden_layers = int(
                training_cfg.get("dsv4_dspark_num_hidden_layers", 3)
            )
            if not skip_loading and spec_model_path and os.path.exists(spec_model_path):
                log_drafter_checkpoint_step(
                    logger, spec_model_path, action="Loading DSV4 DSpark drafter weights"
                )
                normalized_state = self._normalize_draft_state_dict(
                    self._load_draft_state_dict(spec_model_path)
                )
        else:
            drafter_config = self._build_fallback_config(target_hf_config)

        if not isinstance(drafter_config, DSV4DSparkConfig):
            raise TypeError(
                f"DSV4 DSpark config is not a DSV4DSparkConfig: {type(drafter_config)}"
            )
        drafter_config = self._normalize_dflash_config(
            drafter_config, target_hf_config, normalized_state, spec_model_path
        )

        # Sync loss params from training_cfg onto the backend instance (the
        # __init__ defaults are stale; build_model reads the real values here).
        self.loss_mode = str(training_cfg.get("dsv4_dspark_loss_mode", "full_vocab"))
        self.ce_loss_alpha = float(training_cfg.get("dsv4_dspark_ce_loss_alpha", 0.1))
        self.l1_loss_alpha = float(training_cfg.get("dsv4_dspark_l1_loss_alpha", 0.0))
        self.tv_loss_alpha = float(training_cfg.get("dsv4_dspark_tv_loss_alpha", 0.0))
        self.l1_chunk_size = int(training_cfg.get("dsv4_dspark_l1_chunk_size", 0))

        used_released_weights = False
        if (
            not skip_loading
            and spec_model_path
            and os.path.exists(spec_model_path)
            and os.path.exists(config_path)
        ):
            draft_model = DSV4DSparkDraftModel(deepcopy(drafter_config))
            state_keys = set(normalized_state.keys()) if normalized_state else set()
            if any(k.startswith("mtp.") for k in state_keys):
                from verl_speco.models.dsv4_dspark.weights import load_released_draft
                n_layers = int(getattr(drafter_config, "num_hidden_layers", 3))
                target_dtype_str = str(self.config.rollout.drafter.training.get(
                    "dsv4_dspark_weight_dtype", "bfloat16"))
                dtype_map = {
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "float32": torch.float32,
                    "fp32": torch.float32,
                    "int8": torch.int8,
                }
                target_dtype = dtype_map.get(target_dtype_str, torch.bfloat16)
                load_released_draft(draft_model, spec_model_path, n_layers,
                                    verbose=True, target_dtype=target_dtype)
                used_released_weights = True
            else:
                self._load_draft_checkpoint(
                    draft_model, spec_model_path, normalized_state=normalized_state
                )
        else:
            draft_model = DSV4DSparkDraftModel(deepcopy(drafter_config))
        if not skip_loading and not used_released_weights:
            draft_model.load_embedding(target_model_path)
        draft_model.freeze_embedding()

        if not skip_loading:
            self.target_lm_head = self._build_target_lm_head(
                target_model_path, target_hf_config
            )
        else:
            from verl_speco.models.target.target_head import TargetHead
            target_text_config = getattr(target_hf_config, "text_config", target_hf_config)
            hidden_size = int(getattr(target_text_config, "hidden_size", drafter_config.hidden_size))
            vocab_size = int(getattr(target_text_config, "vocab_size", drafter_config.vocab_size))
            lm_head_weight = torch.empty(vocab_size, hidden_size, dtype=torch.bfloat16)
            self.target_lm_head = TargetHead(lm_head_weight)
        training_cfg = self.config.rollout.drafter.training
        return DSparkTrainingModel(
            draft_model=draft_model,
            block_size=int(
                training_cfg.get(
                    "dsv4_dspark_block_size", getattr(drafter_config, "block_size", 7)
                )
            ),
            num_anchors=int(
                training_cfg.get(
                    "dsv4_dspark_num_anchors", getattr(drafter_config, "num_anchors", 512)
                )
            ),
            loss_decay_gamma=float(
                training_cfg.get(
                    "dsv4_dspark_loss_decay_gamma",
                    getattr(drafter_config, "loss_decay_gamma", 7.0),
                )
            ),
            loss_mode=str(training_cfg.get("dsv4_dspark_loss_mode", "full_vocab")),
            sampled_ce_negatives=int(
                training_cfg.get("dsv4_dspark_sampled_ce_negatives", 0)
            ),
            ce_loss_alpha=float(
                training_cfg.get(
                    "dsv4_dspark_ce_loss_alpha",
                    getattr(drafter_config, "ce_loss_alpha", 0.1),
                )
            ),
            l1_loss_alpha=float(
                training_cfg.get(
                    "dsv4_dspark_l1_loss_alpha",
                    getattr(drafter_config, "l1_loss_alpha", 0.0),
                )
            ),
            tv_loss_alpha=float(
                training_cfg.get(
                    "dsv4_dspark_tv_loss_alpha", 0.0,
                )
            ),
            confidence_head_alpha=float(
                training_cfg.get("dsv4_dspark_confidence_loss_alpha", 0.0)
            ),
            l1_chunk_size=int(training_cfg.get("dsv4_dspark_l1_chunk_size", 0)),
            debug_log=bool(training_cfg.get("dsv4_dspark_debug_log", False)),
            debug_log_first_n=int(training_cfg.get("dsv4_dspark_debug_log_first_n", 2)),
            debug_log_interval=int(training_cfg.get("dsv4_dspark_debug_log_interval", 100)),
        ), drafter_config

    def preprocess_individual_items(self, items, device, model_config):
        res = {"ids": [], "h_states": [], "masks": [], "target_last_h_states": []}
        max_window = int(
            self.config.rollout.drafter.training.get("dsv4_dspark_max_window", 512)
        )
        pad_id = int(getattr(model_config, "pad_token_id", 0) or 0)
        h_dim = int(
            getattr(model_config, "target_hidden_size", model_config.hidden_size)
        )
        num_context_layers = int(
            getattr(
                model_config,
                "num_context_layers",
                getattr(model_config, "num_target_layers", 3),
            )
        )
        expected_hidden_dim = h_dim * num_context_layers

        for item in items:
            layout = item.get("hidden_states_layout")
            if layout not in (None, "dflash_aux", "dflash_aux_plus_last"):
                raise ValueError(
                    f"DSV4 DSpark expected hidden_states_layout='dflash_aux' or "
                    f"'dflash_aux_plus_last', got {layout!r}."
                )
            ids = item["input_ids"].to(device, non_blocking=True)
            raw_h = item["hidden_states"]
            full_h = (
                torch.cat(raw_h, dim=-1) if isinstance(raw_h, (list, tuple)) else raw_h
            )
            full_h = full_h.to(device, dtype=torch.bfloat16)
            if full_h.size(-1) < expected_hidden_dim:
                raise ValueError(
                    f"DSV4 DSpark expected at least {expected_hidden_dim} hidden dims "
                    f"({num_context_layers} context layers of size {h_dim}), "
                    f"got {full_h.size(-1)}"
                )
            target_last_h = None
            if layout == "dflash_aux_plus_last":
                expected_with_last = expected_hidden_dim + h_dim
                if full_h.size(-1) != expected_with_last:
                    raise ValueError(
                        "DSV4 DSpark hidden_states_layout='dflash_aux_plus_last' expected "
                        f"exactly {expected_with_last} hidden dims, got {full_h.size(-1)}"
                    )
                target_last_h = full_h[..., expected_hidden_dim:expected_with_last]
                full_h = full_h[..., :expected_hidden_dim]
            elif layout == "dflash_aux" and full_h.size(-1) != expected_hidden_dim:
                raise ValueError(
                    f"DSV4 DSpark hidden_states_layout='dflash_aux' expected exactly "
                    f"{expected_hidden_dim} hidden dims, got {full_h.size(-1)}"
                )

            if item.get("loss_mask") is not None:
                item_loss_mask = item["loss_mask"].to(
                    device, dtype=torch.float32, non_blocking=True
                )
            elif "prompts" in item and "responses" in item:
                item_loss_mask = torch.zeros_like(ids, dtype=torch.float32)
                prompt_len = item["prompts"].size(0)
                responses = item["responses"]
                item_loss_mask[prompt_len : prompt_len + responses.size(0)] = (
                    responses != pad_id
                ).float()[: max(0, ids.size(0) - prompt_len)]
            else:
                item_loss_mask = torch.zeros_like(ids, dtype=torch.float32)
                item_loss_mask[:] = 1.0

            valid_len = min(ids.size(0), full_h.size(0), item_loss_mask.size(0))
            ids = ids[:valid_len]
            full_h = full_h[:valid_len]
            item_loss_mask = item_loss_mask[:valid_len]
            nonzero = torch.nonzero(item_loss_mask)
            if nonzero.numel() > 0:
                r_start = nonzero[0, 0]
                start = torch.clamp(
                    r_start - (max_window // 2),
                    min=0,
                    max=max(0, ids.size(0) - max_window),
                ).item()
                end = min(start + max_window, ids.size(0))
            else:
                start, end = max(0, ids.size(0) - max_window), ids.size(0)

            res["ids"].append(ids[start:end])
            res["h_states"].append(full_h[start:end, :expected_hidden_dim])
            res["masks"].append(item_loss_mask[start:end])
            if target_last_h is not None:
                res["target_last_h_states"].append(target_last_h[start:end])
            else:
                res["target_last_h_states"].append(None)
        return res

    def compute_loss(self, model, batch, _current_pad_size):
        if getattr(self, "use_ulysses_sp", False):
            raise NotImplementedError(
                "DSV4 DSpark drafter training does not support Ulysses sequence parallel yet"
            )
        if self.target_lm_head is None:
            raise ValueError("DSV4 DSpark target_lm_head is not initialized")

        draft_model = model.module if hasattr(model, "module") else model
        hidden_states = batch["hidden_states"]
        num_context_layers = draft_model.draft_model.num_context_layers
        per_layer_dim = hidden_states.shape[-1] // num_context_layers
        hidden_states_list = list(hidden_states.split(per_layer_dim, dim=-1))

        loss, accuracy, loss_pp, acc_pp, count_pp, diagnostics = model(
            input_ids=batch["input_ids"],
            hidden_states_list=hidden_states_list,
            loss_mask=batch["loss_mask"],
            lm_head_weight=self.target_lm_head.fc.weight,
            target_last_hidden_states=batch.get("target_last_hidden_states"),
        )
        local_num_tokens = diagnostics.get("ce_weighted_token_count")
        if not torch.is_tensor(local_num_tokens):
            local_num_tokens = count_pp.sum()
        local_num_tokens = local_num_tokens.to(loss.device, dtype=loss.dtype)
        return {
            "total_local_vloss": torch.tensor(0.0, device=batch["input_ids"].device),
            "total_local_ploss": loss * local_num_tokens,
            "local_num_tokens": local_num_tokens,
            "v_weight": 0.0,
            "p_weight": 1.0,
            "accuracy": accuracy.detach(),
            "loss_per_position": loss_pp.detach(),
            "acc_per_position": acc_pp.detach(),
            "count_per_position": count_pp.detach(),
            "diagnostics": {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in diagnostics.items()
            },
        }
