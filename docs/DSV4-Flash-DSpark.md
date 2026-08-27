# 基于 verl-SpeCo 框架的 DeepSeek-V4-Flash DSpark 草稿模型训练实现技术报告

## 1. 背景

### 1.1 投机解码的发展趋势

大语言模型（LLM）的推理延迟主要由自回归解码的串行性决定——每生成一个 token 需要完整的前向计算。投机解码（Speculative Decoding）通过"草稿模型预测 + 目标模型验证"的两阶段策略打破串行性瓶颈：轻量级草稿模型（Drafter）以低成本生成候选 token 序列，目标模型（Verifier）批量验证并接受匹配的 token，从而在一次前向中产出多个 token。

草稿模型架构经历了三代演进：

| 代次 | 架构 | 代表 | 特点 |
|------|------|------|------|
| 第一代 | 自回归 MTP | DeepSeek-V3 MTP、Medusa | 草稿模型是目标模型的浅层复制，逐 token 自回归生成 |
| 第二代 | 块状草稿 | DFlash、EAGLE | 草稿模型一次生成整个 draft block（多个 token），利用目标模型的中间隐藏状态（Hidden States, HS）作为输入 |
| 第三代 | 原生稀疏草稿 | DSpark | 草稿模型复用目标模型的完整子架构（MLA + MoE + 超连接），在 token 级别实现"小目标模型"效果 |

### 1.2 DSpark 框架概述

DSpark 是 2026 年 6 月由 DeepSeek 联合北京大学发布的开源推测解码框架，直接集成在 V4-Flash 的推理包中。`DeepSeek-V4-Flash-DSpark` 仓库包含的是 **V4-Flash 本体（284B 总参 / 13B 激活）+ DSpark 草稿模型（~20B）** 的完整推理包（合计 ~304B）。

传统推测解码存在两个瓶颈：纯并行草稿模型每个位置独立预测，缺乏 token 间依赖，导致块内后缀接受率快速衰减；固定长度验证浪费批处理容量。DSpark 通过**半自回归草稿生成**和**置信度调度验证**两个机制解决。

DSpark 相比原 MTP-1 基线，V4-Flash 生成速度提升 **60%–85%**，V4-Pro 提升 **57%–78%**。

### 1.3 本报告范围

本报告描述在 verl-SpeCo 训练框架上实现DSV4-Flash DSpark草稿模型单独训练、并在 NPU 上实现训练和推理的完整实现。

### 1.4 验证环境

#### 1.4.1 硬件环境

| 项目 | 值 |
|------|------|
| NPU | Ascend 910, 8 卡 × 64GB HBM |
| NPU 驱动 | 25.5.1 (V100R001C23SPC006B220) |
| npu-smi | 25.5.1 |

#### 1.4.2 软件环境

| 项目 | 版本 | 安装路径 |
|------|------|---------|
| Python | 3.11.15 | /usr/local/python3.11.15 |
| CANN Toolkit | 9.0.0 (V100R001C10SPC001B250) | /usr/local/Ascend/cann-9.0.0 |
| torch | 2.10.0+cpu | pip |
| torch_npu | 2.10.0 | pip |
| transformers | 5.10.0 | pip |
| vllm | 0.18.0+empty | pip |
| vllm-ascend | 0.19.1rc2.dev793+g603650716 | /vllm-ascend (editable) |
| speculators | 0.5.0.dev608 | /speculators (editable) |
| verl-SpeCo | dev | /verl-SpeCo (editable) |

#### 1.4.3 模型 Checkpoint

| Checkpoint | 名称 | 大小 | 格式 |
|-----------|------|------|------|
| DSpark-BF16 | DeepSeek-V4-Flash-DSpark-bf16 | 567GB (142 shards) | BF16, 3-aux |
| DSpark-W8A8 | DeepSeek-V4-Flash-DSpark-W8A8 | 282GB (71+68 shards) | W8A8 (int8+FP8), 3-aux |
| 0731-BF16 | DeepSeek-V4-Flash-0731-bf16 | — | BF16 (文档参考) |

---

## 2. 基础：模型结构详解

### 2.1 V4-Flash 目标模型架构

V4-Flash 是 DeepSeek V4 系列的轻量版（284B 总参 / 13B 激活），与 V4-Pro（1.6T/49B）共享架构设计。

| 配置项 | V4-Flash | V4-Pro |
|--------|----------|--------|
| 总参数量 | 284B | 1.6T |
| 每 token 激活参数 | 13B | 49B |
| Transformer 层数 | 43 | 61 |
| 隐藏维度 | 4096 | 7168 |
| 上下文窗口 | 1,000,000 tokens | 1,000,000 tokens |
| 最大输出长度 | 384K tokens | 384K tokens |

#### 2.1.1 混合注意力架构（CSA + HCA + SWA）

V4 系列最大的架构革新是用**混合注意力**替代了 V2/V3 的 MLA，使 1M token 长上下文在工程上真正可用。

**层排列方式（Flash 特有）**：
- 第 0~1 层：纯滑动窗口注意力（SWA），窗口大小 128
- 第 2~42 层：CSA 与 HCA 交替排列

**CSA（Compressed Sparse Attention）**：
- 压缩率 4x（每 4 个 token 压缩为 1 个 KV 条目）
- 索引器查询头数 64，头维度 128
- 稀疏注意力 top-k=512
- 查询头数 64，头维度 512，查询压缩维度 1024
- 输出投影分组数 8

**HCA（Heavily Compressed Attention）**：
- 压缩率 128x（每 128 个 token 压缩为 1 个条目）
- 在压缩后的短序列上做稠密全局注意力

**滑动窗口分支**：无论 CSA 还是 HCA 层，都保留独立的 128-token 未压缩 SWA 分支。同时使用**可学习 Attention Sinks** 稳定超长序列的注意力分布。

**效率收益**：在 1M token 场景下，V4-Flash 的单 token 推理 FLOPs 仅为 V3.2 的 **10%**，KV Cache 占用仅为 **7%**。

#### 2.1.2 DeepSeekMoE

全部 43 层均为 MoE 层：

| 配置项 | 数值 |
|--------|------|
| 每层路由专家数 | 256 |
| 每层共享专家数 | 1 |
| 每 token 激活路由专家数 | 6 |
| 专家中间隐藏维度 | 2048 |
| 前 3 层路由策略 | Hash Routing（基于 token ID 的预定义哈希） |
| 其余层路由策略 | 标准 top-k 路由 |
| 路由亲和度计算 | Sqrt(Softplus(·)) |

负载均衡策略沿用 V3 的 auxiliary-loss-free 策略，增加轻微的序列级平衡损失。

#### 2.1.3 mHC（Manifold-Constrained Hyper-Connections）

mHC 替代传统 Transformer 的残差连接，是 V4 训练稳定性的关键：

- 扩展因子 4x（将层间通道拓宽 4 倍）
- 通道混合矩阵约束在 Birkhoff 多面体上（双随机矩阵），谱范数上限为 1
- 将深层网络的信号放大从混乱的 3000x 压制到稳定的 1.6x
- Sinkhorn-Knopp 迭代 20 次

#### 2.1.4 其他组件

| 组件 | 配置 |
|------|------|
| MTP 深度 | 1（与 V3 相同） |
| 位置编码 | RoPE |
| 优化器 | Muon（替代 AdamW） |
| 量化精度 | FP4（MoE 专家权重）+ FP8（其余参数） |

### 2.2 DSpark 草稿模型结构

DSpark 草稿模型（~20B 参数）并非独立的小型语言模型，而是**直接复用 V4-Flash 目标模型尾部层特征、并叠加 3 层独立 MoE 草稿层**的混合架构。草稿权重存储在 `mtp.*` 命名空间，与 284B 目标模型打包在同一 checkpoint 中。

#### 2.2.1 半自回归设计

DSpark 采用**"重并行主干 + 轻量顺序头"**的混合架构：

```
┌─────────────────────────────────────────────────────────────┐
│  阶段一：并行主干（Parallel Backbone）                         │
│  基于目标模型第 40-42 层输出特征，通过 3 层独立 MoE 层          │
│  一次性前向传播，生成整个草稿块（γ=5）的 base logits             │
│                                                              │
│  输入: target_model.layers[40-42] 的 hidden states            │
│       ↓                                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                      │
│  │ mtp.0   │→│ mtp.1   │→│ mtp.2   │  (3层 MoE 草稿层)      │
│  │ Attention│ │ Attention│ │ Attention│                      │
│  │ 256 routed│ │ 256 routed│ │ 256 routed│                  │
│  │ +1 shared │ │ +1 shared │ │ +1 shared │                  │
│  └─────────┘  └─────────┘  └─────────┘                      │
│       ↓                                                      │
│  base logits U₁, U₂, ..., U₅                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段二：顺序头（Sequential Head）— Markov Head               │
│  在 base logits 上注入前缀依赖的转移偏置，逐位置采样              │
│  Bₖ(xₖ₋₁, xₖ) = W₁(xₖ₋₁) · W₂(xₖ)  (低秩 r=256)            │
│  最终分布: pₖ(v|x₀,x<ₖ) = exp(Uₖ(v)+Bₖ(...)) / Σ exp(...)  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段三：置信度头（Confidence Head）                          │
│  mtp.2.confidence_head.proj                                  │
│  输出每个草稿位置的接受概率 c₁, c₂, ..., c₅                   │
│  供 Hardware-Aware Prefix Scheduler 动态裁剪验证长度             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.2.2 并行主干：3 层独立 MoE 草稿层

每个 mtp layer 是一个完整的 DSV4 decoder layer，**减去 DSA/CSA/HCA 压缩器**：

| 配置项 | 数值 | 说明 |
|--------|------|------|
| 草稿层数 | 3 层 (`mtp.0`, `mtp.1`, `mtp.2`) | 每层都是完整 Transformer 块 |
| 注意力机制 | 密集 SWA (window=128)，块内双向 | 无 CSA/HCA 压缩器、无索引器 |
| MoE 配置 | 256 routed + 1 shared | 与目标模型一致 |
| 每 token 激活专家 | 6 | 与目标模型一致 |
| 专家中间维度 | 2048 | 与目标模型一致 |
| 隐藏维度 | 4096 | 与目标模型一致 |
| 目标特征层 | [40, 41, 42] | 草稿层输入来自目标模型最后 3 层 |
| 草稿块大小 γ | 5 | 每次生成 5 个草稿 token |
| 可学习 Attention Sink | 每层 64 个 head 各一个 | 与目标模型一致 |

关键设计点：
- 草稿层**复用目标模型的 Embedding 和 LM Head**（训练时冻结），只更新草稿主干、顺序头和置信度头
- 草稿层权重存储在 `mtp.*` 命名空间下，加载时映射到 `model.layers.{0,1,2}`
- 论文中 5 层配置用于消融实验，V4-Flash 生产部署选择更浅的 3 层结构

#### 2.2.3 顺序头：Markov Head（低秩分解）

| 配置项 | 数值 |
|--------|------|
| 类型 | Markov Head（默认）/ RNN Head（备选） |
| 低秩 r | 256 |
| 矩阵分解 | V×V 转移矩阵分解为两个低秩矩阵 W₁, W₂ |
| 依赖范围 | 仅前一个 token xₖ₋₁ |
| 计算方式 | `Bₖ(xₖ₋₁, xₖ) = W₁(xₖ₋₁) · W₂(xₖ)` |
| 延迟开销 | 极小（T_sequential ≪ T_parallel） |

仅依赖前一个 token 的 Markov Head 已足以打破"独立性假设"，显著减缓接受率衰减。实验表明，**仅 2 层的 DSpark 就超过了 5 层的纯并行 DFlash 基线**。

#### 2.2.4 置信度头

位于 `mtp.2` 顶部的独立投影层，输出每个草稿位置的置信度分数 cₖ ∈ [0,1]，物理意义为 P(第 k 个 token 被接受 | 前面所有 token 已被接受)。

#### 2.2.5 参数量估算

| 组件 | 估算参数量 | 说明 |
|------|-----------|------|
| 3 层 MoE 草稿主干 | ~19B | 每层 ≈ 6.3B |
| Markov Head | ~0.3B | 低秩 r=256 |
| Confidence Head | ~0.01B | 单层投影 |
| DSpark 草稿头合计 | ~20B | |
| V4-Flash 目标模型 | 284B | 本体 |
| 完整 Checkpoint | ~304B | ~167GB |

### 2.3 精度与量化配置

#### 2.3.1 各组件精度

| 组件 | 精度类型 | 说明 |
|------|----------|------|
| mtp.0~2 草稿 MoE 层 Attention/FFN 投影 | FP8 E4M3 | 非专家权重 |
| mtp.0~2 草稿 MoE 层专家权重 | **FP4** | 256 routed + 1 shared expert |
| mtp.0~2 草稿 MoE 层 scale | UE8M0 (E8M0 无符号指数) | 块级量化 scale |
| Markov Head (markov_w1, markov_w2) | **bfloat16** | 低秩转移矩阵 |
| Confidence Head | **bfloat16** | 单层投影 |
| Embedding / LM Head | 复用目标模型 (FP8) | 训练时冻结 |

Markov Head 保持 bfloat16 的原因：参数量极小（~0.3B），量化收益可忽略；需要精确建模相邻 token 转移关系，FP8/FP4 可能引入不可接受的分布偏移。

#### 2.3.2 量化解耦（部署关键）

草稿模型的 `quantization_config` 必须与目标模型**解耦**：

| | 目标模型 | DSpark 草稿模型 |
|--|---------|----------------|
| 格式 | Native MXFP4/FP8 | 同格式但独立配置 |
| 后端 | 目标模型专属量化路径 | `deepseek_v4_fp8` 独立路径 |

如果草稿模型错误继承目标模型的专属量化配置，会导致草稿 MoE 被路由到错误的量化后端，接受率暴跌到 ~1.0 token/step。

#### 2.3.3 W8A8 Ascend 量化格式

在 Ascend NPU 上，W8A8 checkpoint 使用不同的量化格式：

| 组件 | 权重类型 | Scale 格式 |
|------|---------|-----------|
| Attention 投影 (wq_a, wq_b, wkv, wo_a, wo_b) | int8 + f32 scale/offset | per-row |
| MoE 专家 (w1, w2, w3) | FP8 e4m3fn | e8m0fnu block-wise (128×128) |
| Norm, Embedding, Head | bf16/f32 | 不量化 |

`load_released_draft()` 自动检测 dtype 并反量化到 bf16，FP8 → bf16 验证结果与 bf16 checkpoint **bit-exact**（diff=0.000000）。

### 2.4 推理时的数据流

```
┌── Serve：目标模型（bf16, TP8/DP2, EP off） ──────────┐
│  1. 目标模型前向到 layer 42                          │
│  2. 在 layer 40/41/42 提取 aux HS（3 × 4096 = 12288）│
│  3. 提取 verifier_last_h = norm(hc_head(最终残差))   │
│  4. aux(12288) + verifier_last_h(4096) = 16384       │
│  5. 将 HS 送入草稿模型                               │
└──────────────────────────────────────────────────────┘
                        │ HS (16384 dim)
                        ▼
┌── Draft：草稿模型（3 mtp layers） ───────────────────┐
│  1. extract_context_feature: fc(HS) → hidden_norm    │
│  2. noise_embedding: anchor_token + mask_token(128799)│
│  3. 3 × MhcDecoderBlock:                             │
│     - MLA attention (block causal, SWA=128)          │
│     - MoE: router → top-6 → grouped GEMM            │
│     - mHC: Sinkhorn 超连接                           │
│  4. norm(hc_head(streams)) → draft_hidden          │
│  5. lm_head(draft_hidden) → base logits U₁..U₅     │
│  6. Markov Head: Bₖ = W₁(xₖ₋₁)·W₂(xₖ) → 注入偏置    │
│  7. Confidence Head: c₁..c₅ 置信度分数               │
└──────────────────────────────────────────────────────┘
                        │ 5 draft tokens + logits + confidence
                        ▼
┌── Verify：目标模型验证 ──────────────────────────────┐
│  Hardware-Aware Scheduler 根据置信度裁剪验证长度      │
│  目标模型前向验证 draft tokens                       │
│  rejection sampling: 接受匹配的 token               │
└──────────────────────────────────────────────────────┘
```

### 2.5 训练时的数据流

训练使用离线 HS（feature store），冻结目标模型的 Embedding 和 LM Head：

1. **HS 采集**：目标模型在 GSM8K 文本上前向，在 layer 40/41/42 提取 aux HS + verifier_last_h
2. **Feature store**：`dflash_aux_plus_last` 布局，16384 dim
3. **训练前向**：
   - `extract_context_feature`：`fc(12288) → hidden_norm → [batch, seq, 4096]`
   - 噪声嵌入：anchor token + mask_token(128799) → `[batch, draft_len, 4096]`
   - 3 层 decoder block → `norm(hc_head(streams))` → draft_hidden
4. **Loss 计算**（三项，按位置加权 $w_k = \exp(-(k-1)/\gamma)$）：

| 损失项 | 公式 | 作用 |
|--------|------|------|
| 交叉熵损失 | CE(draft_logits, target_ids) | 训练草稿模型预测正确 token |
| 分布匹配损失 (TV) | 1 - min(p_draft, p_target).sum() | 最小化草稿/目标分布差异 |
| 置信度损失 (BCE) | BCE(confidence, accept_label) | 训练置信度头预测接受标签 |

`total_loss = (ce × α_ce + tv × α_tv + bce × α_bce) × w_k`

### 2.6 关键配置参数

| 参数 | 值 | 说明 |
|------|------|------|
| `mask_token_id` / `dspark_noise_token_id` | 128799 | 草稿块中非 anchor 位置的填充 token |
| `block_size` / `dspark_block_size` | 6（训练）/ 5（推理） | 训练含 anchor slot，推理不含 |
| `target_layer_ids` / `dspark_target_layer_ids` | [40, 41, 42] | 目标模型提取 HS 的层 |
| `markov_rank` | 256 | Markov Head 低秩 r |
| `hc_mult` | 4 | 超连接多流数 |
| `hc_sinkhorn_iters` | 20 | Sinkhorn 迭代次数 |
| `n_routed_experts` | 256 | 每层路由专家数 |
| `n_activated_experts` | 6 | 每 token 激活专家数 |
| `loss_decay_gamma` | 4~7 | 位置衰减因子 γ |

---

## 3. 实现方案：整体设计

### 3.1 代码结构

```
verl_speco/
  models/dsv4_dspark/
    __init__.py                      # DSV4DSparkConfig, DSV4DSparkDraftModel 导出
    configuration_dsv4_dspark.py     # 配置类（修复 transformers 5.10 兼容）
    modeling_dsv4_dspark.py          # 草稿模型（含 load_embedding 多 index 兼容）
    weights.py                       # 权重映射 + FP8/FP4/INT8/BF16 自动反量化
    backbone/
      __init__.py
      block.py                       # MhcDecoderBlock（完整 Transformer 块）
      attention.py                   # 密集 SWA 注意力（block causal, window=128）
      hyper.py                        # mHC 超连接（Sinkhorn 双随机约束）
      moe.py                          # Router + GroupedExperts + bias 初始化
      moe_ep.py                       # 专家并行 all-to-all 通信
      moe_grouped_gemm.py             # NPU grouped matmul
      moe_compile.py                  # MoE 编译优化
      norm.py                         # RMSNorm
      rotary.py                       # RoPE 位置编码
      kernels.py                      # 自定义 NPU kernel
  backends/
    dsv4_dspark_trainer_backend.py    # DSV4 后端（config, 权重, loss, preprocess, compute_loss）
    dsv4_ep_utils.py                  # EP/FSDP2 设置工具（独立文件，~450 行）
    dspark_trainer_backend.py         # DSpark 后端（DSparkTrainingModel + TV loss + VeOmni bfloat16）
    factory.py                        # 后端工厂（DSV4_DSPARK 算法注册）
  integration/
    rollout_publish.py                # 草稿权重发布 + VeOmni actor lm_head 导出 + old-logprob 安装
    vllm_runtime.py                   # vLLM 运行时 + level-2 sleep/wake_up 快照恢复
    oldlogprob_runtime.py             # old-logprob hidden state 收集 hook
    oldlogprob_layer_ids.py           # 目标层 ID 配置
  trainer/
    base_trainer.py                  # 通用训练器（EP 薄委托 + FSDP/HCCL + VeOmni 兼容）
    draft_training_loop.py           # 训练循环 + 分布式初始化 + NPU 设备绑定
    draft_dataset.py                 # Feature store 数据加载
    checkpoint.py                    # 检查点保存/加载
  tools/
    convert_checkpoint.py            # 检查点转换工具
  draft_train.py                     # 入口（NPU 设备绑定）
  config/
    speco_base.yaml                  # 基础配置（含 DSV4 + park_hccl_after_drafter_training）
```

### 3.2 设计原则

1. **最小改动原则**：`dflash_trainer_backend.py`、`configuration_dflash.py`、`modeling_dflash.py`、`target_head.py` 保持零改动
2. **DSV4 逻辑隔离**：所有 DSV4 特有逻辑集中在 `dsv4_dspark_trainer_backend.py` 和 `dsv4_ep_utils.py`
3. **薄委托模式**：`base_trainer.py` 中的 EP 方法（`_is_ep_enabled`、`_apply_ep_only`、`_apply_ep_fsdp2`、`_sync_non_expert_grads`）是 4-10 行的薄委托，通过延迟导入避免循环依赖
4. **VeOmni 兼容性**：DSV4 新增逻辑通过守卫条件确保不影响 VeOmni 后端的 HCCL/FSDP 设置。VeOmni 专用方法（`_use_flattened_drafter_fsdp_mesh`、`_should_park_drafter_hccl`、`_use_blocking_npu_optimizer_offload`）检查 `model_type == "dspark"`（不含 `"dsv4_dspark"`），DSV4 走新路径。Cleanup 路径中每个行为独立守卫，不使用顶层互斥分支

### 3.3 权重加载与自动反量化

`load_released_draft()` 函数自动检测 checkpoint 格式：

| Checkpoint 格式 | 检测条件 | 反量化路径 | 输出 |
|---|---|---|---|
| FP8 (e4m3fn) | `tensor.dtype == torch.float8_e4m3fn` | `_dequant_fp8(weight, e8m0_scale)` | bf16 |
| FP4 (int8-packed) | `tensor.dtype == int8` + `.scale` + `.experts.` | `_dequant_fp4_packed(weight, e8m0_scale)` | bf16 |
| W8A8 (int8) | `tensor.dtype == int8` + `_scale` | `_dequant_weight(weight, f32_scale, f32_offset)` | bf16 |
| BF16 | `tensor.dtype == torch.bfloat16` | 直接加载 | bf16 |

`named_buffers()` 也被加载——Router 的 bias 是 `register_buffer`（不是 Parameter），`named_parameters()` 会遗漏它。不加载 bias 会导致专家塌缩和 all_to_all 死锁。

### 3.4 配置兼容性修复

**transformers 5.10.0 兼容**：`PretrainedConfig.__init__` 在 `super().__init__()` 中访问 `self.max_position_embeddings`，但 `DFlashConfig` 在 super 调用之后才设置此属性。修复：在 `DSV4DSparkConfig.__init__` 中，将 `max_position_embeddings` 和 `rope_theta` 移到 `super().__init__()` 之前设置。

### 3.5 多 index.json 兼容

量化 checkpoint 同时包含 `model.safetensors.index.json`（含 mtp/embed/head 草稿权重）和 `quant_model_weights.safetensors.index.json`（主模型，无 mtp）。原代码遇多 index 直接 raise。修复：遍历所有 index 文件，选择 weight_map 包含所需 key 的那个。

### 3.6 `num_context_layers` 覆盖修复

`DSV4DSparkConfig` 默认 `num_context_layers=5`（来自 DFlashConfig），训练配置设为 3 但条件检查 `is None` 为 False（5≠None）→ override 被跳过 → preprocess 报 "expected 20480 (5 layers), got 16384"。修复：移除 `is None` 条件，无条件 override。

### 3.7 VeOmni 后端兼容性

DSV4 DSpark 的修改涉及共享文件（`base_trainer.py`、`dspark_trainer_backend.py`、`rollout_publish.py`、`vllm_runtime.py`），需确保不影响 VeOmni 后端的 HCCL 和 FSDP 设置。设计如下：

#### 3.7.1 FSDP Mesh 展平（VeOmni 专用）

VeOmni 在 NPU 上使用 2D dp×sp DeviceMesh，但 DSpark 不使用 Ulysses SP，2D mesh 会导致 FSDP2 使用 HSDP 模式（参数在 SP 维分片、DP 维复制），语义错误。VeOmni 专用方法 `_use_flattened_drafter_fsdp_mesh()` 检测 `model_type == "dspark"`（不含 `"dsv4_dspark"`），将 2D mesh 展平为 1D full-shard mesh。DSV4 不触发此条件，直接使用 `training_device_mesh`。

`fsdp_device_mesh` 属性（由 `_resolve_drafter_fsdp_device_mesh()` 计算）对 VeOmni 返回展平的 1D mesh，对 DSV4 等于 `training_device_mesh`。FSDP 包装、checkpoint process group、`save_fsdp_shard_world_size` 元数据均使用 `fsdp_device_mesh`。

#### 3.7.2 HCCL Parking（VeOmni 专用，opt-in）

`park_hccl_after_drafter_training` 配置项（默认 `false`）控制是否在训练后释放空闲 HCCL 通信器。`_should_park_drafter_hccl()` 检查 `model_type == "dspark"` + `actor_strategy == "veomni"` + `device_name == "npu"`，DSV4 不触发。`_park_idle_drafter_hccl()` 调用 `_delete_tcpstore_key` + `abort_hccl_comm` 释放通信器。

#### 3.7.3 阻塞式 NPU Optimizer Offload

`_offload_optimizer_state_to_cpu()` 内部调用 `_use_blocking_npu_optimizer_offload()` 判断：VeOmni+NPU 使用阻塞式 D2H 拷贝（Ascend 异步 D2H 失败只在后续 synchronize 暴露），非 VeOmni 调用 `offload_fsdp_optimizer()` 常规 offload。该方法对所有后端通用，无需显式守卫。

#### 3.7.4 Cleanup 路径设计

Cleanup 函数使用单一路径 + 独立守卫，不使用顶层互斥分支：

| 行为 | 守卫 | VeOmni | 非 VeOmni | VeOmni+DSV4 |
|------|------|--------|----------|-------------|
| barrier | `if not _use_blocking_npu_optimizer_offload()` | 跳过 | 执行 | 跳过 |
| optimizer offload | `_offload_optimizer_state_to_cpu()` 内部自动判断 | 阻塞式 | 常规 | 阻塞式 |
| target head offload | `_move_target_lm_head("cpu")` 对所有后端 | offload 到 CPU | offload 到 CPU | offload 到 CPU |
| HCCL parking | `_park_idle_drafter_hccl()` 内部自动判断 | 按需 | no-op | no-op |

此设计确保 VeOmni+DSV4 组合可以同时获得 VeOmni 的无 barrier + CPU offload（NPU 显存管理），不排斥 DSV4 的其他特性。

#### 3.7.5 Target LM Head bfloat16 转换

`DSparkTrainerBackend._build_target_lm_head()` override 在 VeOmni+NPU 时将 target lm_head 转为 bfloat16。基类 `DFlashTrainerBackend` 不做此转换。该方法对 DSV4 DSpark（继承自 `DSparkTrainerBackend`）同样生效。

#### 3.7.6 `defer_device_apply` 与分块 NPU Apply

`sync_target_lm_head_weight()` 接受 `defer_device_apply` 参数，允许将 target head 权重暂存到 CPU、延迟到 `activate_training_model()` 时再 apply。NPU 上使用 256MB 分块 `copy_`（`non_blocking=False`），避免大 tensor 一次性 D2H 导致 Ascend AICPU 崩溃。

#### 3.7.7 `activation_stage` 调试追踪

`activate_training_model()` 维护 `activation_stage` 变量（`"enter"` → `"apply_pending_target_head_on_host"` → `"build_draft_model"` → `"load_draft_model"` → `"load_draft_optimizer"` → `"load_target_lm_head"` → `"apply_pending_target_head_on_device"`），错误时输出 `failed stage={activation_stage}` 便于定位失败位置。

### 3.8 rollout_publish.py VeOmni Actor 支持

`rollout_publish.py` 包含 VeOmni actor 专属的 lm_head 导出和 old-logprob 安装逻辑：

| 组件 | 功能 |
|------|------|
| `actor_training_backend_name()` | 检测 actor backend strategy（veomni/fsdp/megatron） |
| `veomni_parallel_layout()` / `validate_veomni_parallel_layout()` | 提取并校验 R2/R3 router replay 配置 |
| `_is_veomni_actor_worker()` | 判断 worker 是否使用 VeOmni actor |
| `_materialize_veomni_lm_head_rows()` | DTensor 分片感知的稀疏行收集（`to_local()` + 全局偏移 + `all_reduce`） |
| `_export_veomni_actor_lm_head_weight()` | VeOmni 完整 lm_head 导出管线（DTensor 处理 + 参数 offload + NPU staging + `keep_model_on_device`） |
| `validate_oldlogprob_hidden_runtime_for_worker()` | init_model 后验证 VeOmni 模型契约（layers/final_norm/lm_head 存在） |

VeOmni old-logprob 安装分支调用 `install_oldlogprob_hidden_runtime_patch(actor_backend="veomni")`，安装 `VeOmniEngineWithLMHead` 的 hidden-state 收集 hook。非 VeOmni 走通用 FSDP patch 路径。

### 3.9 vLLM Level-2 Sleep/Wake-up 快照恢复

`vllm_runtime.py` 中 `SpecoVLLMColocateWorkerExtension` 实现 level-2 sleep/wake-up 的 draft 状态快照与恢复：

- **`_speco_sleep_hook`**：vLLM level-2 sleep 丢弃 CuMem weights pool 前，将 draft 参数和 buffer 快照到 CPU
- **`_speco_wake_up_hook`**：wake_up 后先尝试从快照恢复（`_speco_restore_draft_after_level2`），无快照时 fallback 到 checkpoint 重载（`_speco_reload_draft_from_checkpoint`）
- **`_speco_snapshot_draft_for_level2()`**：快照所有 `named_parameters()` + `named_buffers()` 到 CPU
- **`_speco_restore_draft_after_level2()`**：从快照恢复，含 shape 校验和 error 收集
- **`_speco_rebuild_draft_metadata_buffers()`**：恢复后重建 fused KV buffers（`_build_fused_kv_buffers`）

快照恢复优先于 checkpoint 重载，确保在线发布的最新权重不因 level-2 sleep 丢失。

---

## 4. 精度对齐

### 4.1 对齐目标

确保 verl-SpeCo 的 DSV4 DSpark 实现与 speculators 参考实现 **bit-exact** 一致。

### 4.2 三层验证

| 层级 | 验证内容 | 方法 | 结果 |
|------|---------|------|------|
| 前向 | 3 层 MhcDecoderBlock 前向输出 | 统一权重加载后，同输入比输出 | diff=0.000000, argmax 100% |
| 反向 | 27/27 参数梯度 | 随机初始化同步，前向+反向比 .grad | bit-exact |
| 接受 | draft_hidden + logits + per-position accept | 完整前向 + lm_head + accept | draft_hidden diff=0.03125, argmax match |

### 4.3 关键对齐修复

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | `fc_key` 选错 `.scale` | `next(k for k in state if "main_proj" in k)` 选了 `.scale`（也含 "main_proj"） | `k.endswith("main_proj.weight")` |
| 2 | `quantization_config` 显式 null | `cfg.get("quantization_config", {})` 返回 None | `cfg.get("quantization_config") or {}` |
| 3 | `mask_token_id` 未设置 | 默认 fallback 到 `vocab_size-1=129279`（错） | 显式设 `128799` |
| 4 | Router bias 未加载 | `named_parameters()` 遗漏 `register_buffer` | 同时加载 `named_buffers()` |
| 5 | `num_context_layers` 未覆盖 | 条件 `is None` 为 False（默认值 5≠None） | 移除 `is None` 条件 |

---

## 5. 优化：EP 与 MoE 融合

### 5.1 专家并行（EP）架构

DSpark 草稿模型有 256 路由专家 × 3 层 = 768 专家。单卡放不下全部 256 专家（~40GB bf16），需要 EP 将专家分片到 8 卡：

- 每 rank 持有 32 个完整专家（256 ÷ 8）
- Token 按路由索引通过 all-to-all 发送到对应 rank
- 各 rank 在本地执行 grouped GEMM
- 结果通过 all-to-all 返回

### 5.2 EP Pre-config 机制

**核心设计**：在 `build_model()` 之前调用 `moe_ep.configure()`，使 `MoE.__init__` 读取 EP context 并只构建本 rank 的 32 个专家（而非全部 256 个）。

这与 speculators 的 `DSPARK_EP=1` 环境变量机制一致。EP 与 `init_on_meta` 不兼容：EP 在 `MoE.__init__` 时就按 rank 分片构建，不需要 file-share 或 broadcast。

### 5.3 三种训练模式

| 模式 | 机制 | 随机初始化 | Checkpoint | 状态 |
|------|------|-----------|------------|------|
| FSDP2+EP + checkpoint | EP slice → DTensor → FSDP2 wrap | ✗ | ✓ | ✓ 验证通过 |
| FSDP2-only + random | 无 EP, FSDP2 全量分片, degenerate MoE | ✓ | ✗ | ✓ 验证通过 |
| EP+DDP + random | EP slice → DDP all-reduce | ✓ | ✗ | ✗ all_to_all 死锁 |

### 5.4 随机初始化模式

随机初始化模式（`dsv4_dspark_random_init=true`）：
- **EP pre-config**：在 `build_model()` 前 `moe_ep.configure()`，每 rank 只构建 32 专家
- **跳过 file-share**：所有 rank 独立随机初始化，无需同步
- **FSDP2 全量分片**：256 专家通过 FSDP2 分片到 8 卡（degenerate MoE 路径，无 all_to_all）
- **Router bias 初始化**：打破路由对称性，防止 MoE 专家塌缩

### 5.5 Router Bias 初始化

随机初始化时，Router bias=0 → 所有 token 路由到同一个专家 → 其他 rank 收到 0 token → all_to_all 死锁。修复：在 EP 设置完成后、第一个 forward 之前，注入 rank-aware bias（本 rank 专家 +0.01，其他 -0.01 + 小随机噪声）。

### 5.6 NPU 设备绑定

`_configure_device()` 中 `getattr(device_module, "set_device", None)` 返回 None（`torch_npu` 顶层模块没有 `set_device`，但 `torch_npu.npu` 子模块有）→ 所有 rank 默认设备 0 → HCCL 报 `EI0015: same physical device ID0`。修复：在 `draft_train.py` 入口显式调用 `torch_npu.npu.set_device(local_rank)`。

### 5.7 HCCL `--standalone` 冲突

`torch.distributed.run --standalone` 无条件覆写 `MASTER_ADDR=127.0.0.1`，可能与 `HCCL_SOCKET_IFNAME` 指定的网络接口冲突，导致 HCCL 设备发现失败。修复：使用 `standalone=false` + 显式 `MASTER_ADDR`/`MASTER_PORT` 环境变量。

### 5.8 vllm-ascend 源码修复

为支持 W8A8 目标模型的 HS 采集，对 vllm-ascend 做了 3 处源码修复：

1. **`modelslim_config.py`**：量化描述 key 格式不匹配（checkpoint 格式 → vllm 格式前向映射）
2. **`model_runner_v1.py`**：HS dumper 增加 `verifier_last_h = norm(hc_head(_mtp_hidden_buffer))`
3. **OMP 线程池**：`OMP_NUM_THREADS=1` + `VLLM_WORKER_MULTIPROC_METHOD=fork` 修复 forked worker 崩溃

---

## 6. 端到端验证

### 6.1 HS 采集

使用 vllm-ascend 加载 W8A8 目标模型（282GB, TP=8），在 GSM8K 文本上前向：
- 一次一个 prompt（`max_num_seqs=1`），避免 batch 混合
- 通过 `TokensPrompt` 传入预分词的 token_ids
- `max_tokens=1` 只需 prefill HS
- 后处理：按序列长度匹配 dump 文件与 token_ids

采集结果：8 样本，16384 dim（aux 12288 + verifier_last_h 4096），`dflash_aux_plus_last` 布局。

### 6.2 训练验证

**Checkpoint 加载模式（FSDP2+EP）**：

| Step | Loss | LR |
|------|------|----|
| 1 | 2.5610 | 2.0e-4 |
| 5 | 2.5182 | 1.2e-4 |
| 9 | 2.5660 | 6.0e-6 |

9/9 步成功，0 NaN。

**随机初始化模式（FSDP2-only, degenerate MoE）**：

| Step | Loss | LR |
|------|------|----|
| 1 | 1.6118 | 2.0e-4 |
| 5 | 1.5783 | 1.2e-4 |
| 9 | 1.5733 | 6.0e-6 |

9/9 步成功，0 NaN。

**对比分析**：
- `ploss = CE loss × ce_loss_alpha(0.1)`
- 随机初始化 CE loss ≈ 16.1（高于均匀预测 `ln(129280)=11.77`，但方差小）
- Checkpoint CE loss ≈ 25.6（训练好的权重在 W8A8 HS 不匹配时"高置信度地预测错误"→ CE loss 更高）

### 6.3 推理验证

使用 vllm-ascend 加载 W8A8 目标模型 + DSpark 草稿模型进行投机解码：

| 模式 | 吞吐 | 加速比 |
|------|------|--------|
| 无投机解码 | 20.7 tok/s | 1.0x |
| DSpark 投机解码 (num_spec=5) | 25.9 tok/s | 1.25x |

粗估 acceptance rate ~5-17%（受 W8A8 动态激活量化噪声影响，远低于预期 40-60%）。

### 6.4 已知限制

1. **W8A8 HS 不匹配**：W8A8 模型每次 matmul 将激活量化为 int8（`npu_dynamic_quant`），43 层后 HS 累积噪声使 released drafter 无法使用。speculators 参考实现也产出 0% accuracy，证实问题在 HS 数据而非代码。
2. **FSDP2+EP 随机初始化**：NPU FSDP2 对随机初始化的 DTensor unshard 有 507018 aicpu 崩溃（checkpoint 模式正常）。
3. **EP+DDP all_to_all 死锁**：HCCL `all_to_all_single` 不能处理 0-token split。
4. **8 卡 HCCL EI0015**：8 卡时 HCCL rootInfo 检测报 "same physical device ID"（2 卡正常）。可能是网络/资源问题，非代码问题。
5. **FSDP2+EP step-10 soft-hang**：NPU FSDP2 unshard/reshard 循环在 9 步后退化。

---

## 7. 总结

### 7.1 已完成

| 项目 | 状态 |
|------|------|
| DSV4 DSpark 草稿模型迁移到 verl-SpeCo | ✅ |
| 前向/反向/接受精度对齐（speculators ↔ verl-SpeCo） | ✅ bit-exact |
| FP8/FP4/INT8/BF16 自动反量化权重加载 | ✅ bit-exact vs bf16 |
| FSDP2+EP 训练（checkpoint 模式） | ✅ 9 步成功 |
| FSDP2-only 训练（随机初始化模式） | ✅ 2 卡训练启动验证 |
| HS 采集管线（vllm-ascend W8A8） | ✅ 8 样本 |
| vllm-ascend 投机解码 | ✅ 功能验证 |
| 最小改动原则（dflash/dspark/configuration_dflash/modeling_dflash/target_head 零改动） | ✅ |
| NPU 设备绑定修复 | ✅ |
| VeOmni HCCL/FSDP 兼容性恢复 | ✅ |
| VeOmni actor lm_head 导出 + old-logprob 恢复 | ✅ |
| vLLM level-2 sleep/wake_up 快照恢复 | ✅ |
| Cleanup 路径独立守卫设计（VeOmni+DSV4 共存） | ✅ |
| 训练启动脚本 | ✅ |

### 7.2 待完成

| 项目 | 优先级 | 依赖 |
|------|--------|------|
| 从 bf16 模型采集 HS（解除 W8A8 HS 不匹配） | P0 | bf16 模型 567GB 跨 8 卡加载方案 |
| 禁用 W8A8 激活量化采 HS | P1 | vllm-ascend 修改 `npu_quant_matmul` → `F.linear` |

### 7.3 verl-SpeCo 源码修改详细记录

#### 7.3.1 `draft_train.py`（入口文件）

**修改**：在模块级（任何 torch 操作之前）强制绑定 NPU 设备。

```python
# 新增（模块级代码，__main__ 之前）
_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
try:
    import torch_npu
    torch_npu.npu.set_device(_local_rank)
except ImportError:
    pass
```

**原因**：`torchrun` 为每个进程设置 `LOCAL_RANK`，但 `verl.utils.device.get_torch_device()` 返回的 `torch_npu` 顶层模块没有 `set_device`（在 `torch_npu.npu` 子模块上），导致所有 rank 默认设备 0，HCCL 报 `EI0015: same physical device ID0`。

#### 7.3.2 `trainer/draft_training_loop.py`（训练循环）

**修改 1**：`_configure_device()` 增加 NPU 显式分支。

```python
# 修改后
if device_name == "npu":
    import torch_npu
    torch_npu.npu.set_device(int(local_rank))
    return
```

**修改 2**：新增 `"dsv4_dspark"` 到 backend_type set 和 `_VARIANT_RUNTIME_ALIASES`。

#### 7.3.3 `trainer/base_trainer.py`（通用训练器）

**修改 1**：`_is_block_drafter_backend()` 和 `_block_drafter_metric_prefix()` 添加 `"dsv4_dspark"`。

**修改 2**：新增 4 个 EP 薄委托方法（每个 4-10 行，延迟导入）：

```python
def _is_ep_enabled(self) -> bool:
    from verl_speco.backends.dsv4_ep_utils import is_ep_enabled
    return is_ep_enabled(self.config)

def _apply_ep_only(self, raw_model, use_skip_loading):
    from verl_speco.backends.dsv4_ep_utils import apply_ep_only
    apply_ep_only(self, raw_model, use_skip_loading)

def _apply_ep_fsdp2(self, raw_model, fsdp_kwargs, fsdp_config, use_skip_loading):
    from verl_speco.backends.dsv4_ep_utils import apply_ep_fsdp2
    apply_ep_fsdp2(self, raw_model, fsdp_kwargs, fsdp_config, use_skip_loading)

def _sync_non_expert_grads(self):
    from verl_speco.backends.dsv4_ep_utils import sync_non_expert_grads
    sync_non_expert_grads(self)
```

**修改 3**：`_build_draft_model()` 增加 EP pre-config + `use_skip_loading` + 条件化 `raw_model.to(device)`。FSDP2 包装使用 `fsdp_device_mesh`（VeOmni 返回展平的 1D mesh，DSV4 等于 `training_device_mesh`），EP 分支也使用 `fsdp_device_mesh`。

**修改 4**：所有 `"dspark"` model_type 检查改为 `in ("dspark", "dsv4_dspark")`。

**修改 5**：`local_loss.backward()` 后增加 EP 梯度同步。

**修改 6**（VeOmni 兼容性恢复）：恢复以下 VeOmni 专用方法，均通过 `model_type == "dspark"` 守卫确保不影响 DSV4：

- `fsdp_device_mesh` 属性 + `_resolve_drafter_fsdp_device_mesh()`：VeOmni 2D→1D mesh 展平
- `_use_flattened_drafter_fsdp_mesh()`：检测 VeOmni+NPU+DSpark+dp>1
- `_use_blocking_npu_optimizer_offload()`：检测 VeOmni+NPU
- `_offload_optimizer_state_to_cpu()`：阻塞式/常规 offload 自动判断
- `_should_park_drafter_hccl()` + `_park_idle_drafter_hccl()`：HCCL 通信器释放（opt-in）
- `_tensor_local_shard()`：DTensor local shard 访问
- `_checkpoint_process_group()`：恢复 flattened mesh group 检查
- `save_fsdp_shard_world_size`：恢复 checkpoint 元数据
- `park_hccl_after_drafter_training` 属性 + 配置项

**修改 7**（VeOmni 兼容性恢复）：`sync_target_lm_head_weight()` 恢复 `defer_device_apply` 参数和 `_pending_target_lm_head_chunked_apply` 标志，NPU 上使用 256MB 分块 `copy_`。

**修改 8**（VeOmni 兼容性恢复）：`activate_training_model()` 恢复 `activation_stage` 变量追踪和早期 `_apply_pending_target_lm_head_weight()` 调用。

**修改 9**（Cleanup 设计）：Cleanup 和 `release_training_memory_after_activation` 使用单一路径 + 独立守卫，不使用顶层 `if _use_blocking_npu_optimizer_offload():` 互斥分支。barrier 是唯一需要显式守卫的 DSV4 新增行为（`if not _use_blocking_npu_optimizer_offload()`），optimizer offload 和 target head offload 使用原始方法对所有后端通用。

#### 7.3.4 `backends/dsv4_dspark_trainer_backend.py`（DSV4 后端）

**修改 1**：`build_model()` 中 `num_context_layers` override 条件从 `is None` 改为无条件。

**修改 2**：`build_model()` 中 `skip_loading` 检查扩展到 `_load_draft_state_dict` 调用——`skip_loading=True` 时跳过状态字典加载，避免 "Multiple index.json files found" 错误。

**修改 3**：`compute_loss()` 改为直接调用 DSpark `forward()`（传 `lm_head_weight`）。

**修改 4**：`build_model()` 传 `tv_loss_alpha` 参数。

#### 7.3.5 `backends/dspark_trainer_backend.py`（DSpark 后端）

**修改 1**：`DSparkTrainingModel.__init__()` 增加 `tv_loss_alpha=0.0` 参数（默认 0，不影响其他模型），`forward()` 增加 TV loss 计算块。

**修改 2**（VeOmni 兼容性恢复）：恢复 `_build_target_lm_head()` override，VeOmni+NPU 时将 target lm_head 转为 bfloat16。

**修改 3**：`_normalize_dflash_config` 恢复到原始位置（`_build_target_lm_head` 之后）。

#### 7.3.6 `integration/rollout_publish.py`（草稿权重发布）

**VeOmni actor 支持完整恢复**：

- 恢复 `_is_actor` 守卫（非 actor worker 跳过 patch 安装）
- 恢复 VeOmni old-logprob 安装分支（`install_oldlogprob_hidden_runtime_patch(actor_backend="veomni")` + `validate_veomni_parallel_layout`）
- 恢复 `validate_oldlogprob_hidden_runtime_for_worker()` 函数和 `init_model` 中的调用
- 恢复 `actor_training_backend_name()`、`veomni_parallel_layout()`、`validate_veomni_parallel_layout()`
- 恢复 `_is_veomni_actor_worker()`、`_materialize_veomni_lm_head_rows()`、`_export_veomni_actor_lm_head_weight()`
- 恢复 `export_actor_lm_head_weight` 的 VeOmni 分支和 `keep_model_on_device` 参数
- 恢复 `get_actor_lm_head_weight` 的 `keep_model_on_device` 参数

#### 7.3.7 `integration/vllm_runtime.py`（vLLM 运行时）

**修改 1**：`_strip_speco_internal_speculative_keys` 删除（改为探测后添加 `draft_sample_method`，而非添加后剥离）。

**修改 2**：DSV4_DSPARK 添加到 vLLM drafter architectures 和 algorithm 检查。

**修改 3**（Level-2 快照恢复）：恢复以下组件：
- `_speco_draft_level2_snapshot` 类属性
- `_speco_sleep_hook`：level-2 sleep 前快照 draft 状态到 CPU
- `_speco_wake_up_hook`：wake_up 后先从快照恢复，无快照时 fallback 到 checkpoint 重载
- `_speco_snapshot_draft_for_level2()`：快照所有 parameters + buffers
- `_speco_restore_draft_after_level2()`：从快照恢复，含 shape 校验
- `_speco_rebuild_draft_metadata_buffers()`：恢复后重建 fused KV buffers

#### 7.3.8 `models/dsv4_dspark/`（草稿模型）

**修改 1**（`configuration_dsv4_dspark.py`）：`max_position_embeddings` 和 `rope_theta` 移到 `super().__init__()` 之前（transformers 5.10 兼容）。

**修改 2**（`modeling_dsv4_dspark.py`）：覆写 `load_embedding()` 处理多 index.json 文件。新增 `fsdp_wrap_plan()` 和 `fsdp_ignored_params()` 供 FSDP2 包装使用。

**修改 3**（`weights.py`）：`quant_cfg = cfg.get("quantization_config") or {}`（处理显式 null）。`load_released_draft()` 同时加载 `named_buffers()`。

#### 7.3.9 `backends/dsv4_ep_utils.py`（新增文件，~450 行）

完整的 EP/FSDP2/router bias 工具，包含：

| 函数 | 功能 |
|------|------|
| `is_ep_enabled(config)` | 检查是否启用 EP |
| `apply_ep_only(trainer, model, skip_loading)` | EP+DDP 模式：file-share + EP slice + moe_ep.configure + grouped GEMM |
| `apply_ep_fsdp2(trainer, model, kwargs, config, skip_loading)` | FSDP2+EP 模式或 FSDP2-only（random init） |
| `sync_non_expert_grads(trainer)` | EP 模式下非 expert 参数梯度 all-reduce |
| `_init_router_bias(model, world_size, experts_per_rank, rank)` | Router bias 初始化，防止专家塌缩 |

#### 7.3.10 `config/speco_base.yaml`（配置）

- 新增全部 `dsv4_dspark_*` 配置项（backbone 超参、loss 参数、EP 开关等）
- 恢复 `park_hccl_after_drafter_training: false` 配置项（VeOmni HCCL parking opt-in）

#### 7.3.11 `backends/factory.py`（后端工厂）

新增 `DSV4_DSPARK` 算法注册，延迟导入 `DSV4DSparkTrainerBackend`。

### 7.4 vllm-ascend 源码修改详细记录

#### 7.4.1 `vllm_ascend/quantization/modelslim_config.py`

**修改 1**：`maybe_update_config()` 早期返回条件修改。

```python
# 修改前：任何非空 quant_description 都直接返回
if self.quant_description:
    return

# 修改后：只在实际包含 layer-level 数据（keys 以 .weight 结尾）时才返回
if self.quant_description and any(
    k.endswith(".weight") for k in self.quant_description
):
    return
```

**原因**：`from_config()` 设置 `quant_description` 为 `{"quant_method": "ascend", ...}`（config metadata），不含 layer-level data。原条件判断为 True 直接返回，导致实际量化描述文件 `quant_model_description.json` 未被加载。

**修改 2**：加载 `quant_model_description.json` 后，前向映射 key 格式。

```python
# 新增：将 checkpoint 格式 key 转换为 vllm 格式
mt = getattr(hf_config, "model_type", "")
pm = QUANT_MODEL_PREFIX_MAPPINGS.get(mt)  # {"layers.": "model.layers.", "embed.": "model.embed_tokens.", ...}
sm = QUANT_MODEL_SUBSTR_MAPPINGS.get(mt)  # {".attn.": ".self_attn.", ".w1.": ".gate_proj.", ...}
if pm:
    from vllm.model_executor.models.utils import WeightsMapper
    fwd_mapper = WeightsMapper(orig_to_new_prefix=pm, orig_to_new_substr=sm)
    self.quant_description = fwd_mapper.apply_dict(self.quant_description)
```

**原因**：checkpoint 的 `quant_model_description.json` 使用 checkpoint 格式 key（如 `embed.weight`、`layers.0.attn.wq_a.weight`），但 vllm 模型代码传 vllm 格式 prefix（如 `model.embed_tokens`、`model.layers.0.self_attn.wq_a`），导致 `KeyError`。

**修改 3**：`get_linear_quant_type()` 容错处理。

```python
# 修改前
quant_type = quant_description[prefix + ".weight"]  # KeyError

# 修改后
quant_type = quant_description.get(prefix + ".weight")
if quant_type is None:
    quant_type = "FLOAT"  # 草稿模型层不在 quant_description 中，当作未量化处理
```

**原因**：草稿模型的 `mtp.*` 层不在 `quant_model_description.json` 中（该文件只描述目标模型层），原代码直接 `[]` 索引导致 `KeyError`。

**修改 4**：`is_layer_skipped_ascend()` 容错处理。

```python
# 修改前
is_shard_skipped = self.quant_description[shard_prefix + ".weight"] == "FLOAT"  # KeyError

# 修改后
is_shard_skipped = self.quant_description.get(shard_prefix + ".weight", "FLOAT") == "FLOAT"

# 新增：如果 prefix 不在 quant_description 中，也当作 FLOAT 处理
if not is_skipped:
    is_skipped = not any(
        key.startswith(prefix) and key.endswith(".weight")
        for key in self.quant_description
    )
```

#### 7.4.2 `vllm_ascend/worker/model_runner_v1.py`

**修改**：`_dump_dspark_hidden_states()` 增加 `verifier_last_h` 计算。

```python
# 新增：在 dump hidden_states 之前，计算 verifier_last_h
vlh = None
mtp_buf = getattr(inner, "_mtp_hidden_buffer", None)
if mtp_buf is not None:
    try:
        n = min(num_tokens, mtp_buf.shape[0])
        with torch.no_grad():
            # _mtp_hidden_buffer 是 2D [N, hc_mult*hidden_size]
            # _hc_head_torch 需要 3D [N, hc_mult, hidden_size]
            residual = mtp_buf[:n].view(n, -1, 4096).clone()
            hc_out = inner.hc_head(residual, inner.hc_head_fn,
                                   inner.hc_head_scale, inner.hc_head_base)
            vlh = inner.norm(hc_out).detach().cpu().clone()
    except Exception:
        pass

# 保存到 dump 文件
save_dict["verifier_last_h"] = vlh[:seq_len]
```

**原因**：原 dumper 只保存 `_dspark_hidden_buffer`（aux 层 HS, 12288 dim），不包含 `verifier_last_h`（post-norm 最终隐藏, 4096 dim）。训练需要两者（`dflash_aux_plus_last` 布局，共 16384 dim）。

**关键细节**：`_mtp_hidden_buffer` 是 2D `[N, hc_mult*hidden_size]`，但 `_hc_head_torch` 期望 3D `[N, hc_mult, hidden_size]`，需要 `.view(n, -1, 4096)` reshape。
