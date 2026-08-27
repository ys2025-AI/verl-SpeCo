set -x

# =============================================================================
# DSV4-Flash DSpark standalone (offline) draft training on Ascend NPU
# =============================================================================
# Requires:
#   - 8× Ascend 910 (64GB), CANN 9.0.0, torch_npu 2.10.0
#   - DSpark checkpoint at MODEL_PATH (bf16 or W8A8)
#   - Feature store at FEATURE_STORE_PATH (dflash_aux_plus_last layout)
#
# Usage:
#   bash examples/run_dsv4-flash_dspark_separate_training.sh
#
# For random init (no checkpoint loading), set:
#   INIT_ON_META=false RANDOM_INIT=true
# For checkpoint loading (continued training), set:
#   INIT_ON_META=false RANDOM_INIT=false
# =============================================================================
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_BUFFSIZE=128 OMP_PROC_BIND=false OMP_NUM_THREADS=4
export DSPARK_MOE_BALANCE=1 DSPARK_MOE_BALANCE_RATE=1e-3
export DSPARK_RECOMPUTE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29725
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1800
export HCCL_SOCKET_IFNAME=enp162s0f0
export HYDRA_FULL_ERROR=1
export ASCEND_GLOBAL_LOG_LEVEL=1

MODEL_PATH=${MODEL_PATH:-/home/model/DeepSeek-V4-Flash-DSpark-W8A8}
FEATURE_STORE_PATH=${FEATURE_STORE_PATH:-/tmp/dsv4_gsm8k_real_hs}
MAX_STEPS=${MAX_STEPS:-9}
NUM_ANCHORS=${NUM_ANCHORS:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
INIT_ON_META=${INIT_ON_META:-false}
RANDOM_INIT=${RANDOM_INIT:-true}
ENABLE_EP=${ENABLE_EP:-true}
ENABLE_GROUPED_GEMM=${ENABLE_GROUPED_GEMM:-false}
EP_NO_FSDP=${EP_NO_FSDP:-false}

python -m verl_speco.draft_train_launcher \
    speco.draft_training.nproc_per_node=${NPROC:-8} speco.draft_training.standalone=false \
    actor_rollout_ref.rollout.drafter.enable=True \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=DSV4_DSPARK \
    actor_rollout_ref.rollout.drafter.model_path=${MODEL_PATH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.rollout.drafter.training.feature_store.path=${FEATURE_STORE_PATH} \
    actor_rollout_ref.rollout.drafter.training.feature_store.shuffle=True \
    actor_rollout_ref.rollout.drafter.training.feature_store.repeat=True \
    actor_rollout_ref.rollout.drafter.training.max_steps=${MAX_STEPS} \
    actor_rollout_ref.rollout.drafter.training.save_final_checkpoint=False \
    actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=${BATCH_SIZE} \
    actor_rollout_ref.rollout.drafter.training.init_on_meta=${INIT_ON_META} \
    +actor_rollout_ref.rollout.drafter.training.dsv4_dspark_random_init=${RANDOM_INIT} \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_hidden_size=4096 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_num_target_layers=3 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_num_hidden_layers=3 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_block_size=6 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_mask_token_id=128799 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_num_anchors=${NUM_ANCHORS} \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_loss_mode=full_vocab \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_loss_decay_gamma=4 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_ce_loss_alpha=0.1 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_tv_loss_alpha=1.8 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_l1_loss_alpha=0.0 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_l1_chunk_size=64 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_markov_rank=256 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_max_window=256 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_backbone_n_routed_experts=256 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_backbone_n_activated_experts=6 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_backbone_moe_inter_dim=2048 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_backbone_hc_mult=4 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_backbone_hc_sinkhorn_iters=20 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_backbone_num_heads=64 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_backbone_head_dim=512 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_backbone_rope_head_dim=64 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_backbone_q_lora_rank=1024 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_backbone_o_lora_rank=1024 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_backbone_o_groups=8 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_backbone_window_size=128 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_backbone_n_shared_experts=1 \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_enable_ep=${ENABLE_EP} \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_enable_grouped_gemm=${ENABLE_GROUPED_GEMM} \
    +actor_rollout_ref.rollout.drafter.training.dsv4_dspark_ep_no_fsdp=${EP_NO_FSDP} \
    actor_rollout_ref.rollout.drafter.training.dsv4_dspark_debug_log=False \
    actor_rollout_ref.rollout.drafter.training.lr=2e-4 \
    actor_rollout_ref.rollout.drafter.training.lr_scheduler_type=cosine \
    actor_rollout_ref.rollout.drafter.training.min_lr_ratio=0.0 \
    actor_rollout_ref.rollout.drafter.training.step=${MAX_STEPS} \
    actor_rollout_ref.actor.strategy=fsdp2 $@
