# Kimi K3 原始资料索引

> 检索日期：2026-07-29
>
> 原则：架构事实优先使用 Moonshot/Kimi 官方报告、公开 checkpoint 配置和实现；组件机制使用原始论文；媒体报道和第三方解读不作为参数依据。

本轮核对使用的 revision：

```text
MoonshotAI/Kimi-K3:              7c5be9599120d7993748de66a76128614f15f210
Hugging Face moonshotai/Kimi-K3: 9f62e4e9fffbd0a83ddd60e1c209d828994b3569
MoonshotAI/MoonEP:               0f385f038fc33bec22e3bcf5a07a8a22693e754c
```

正文链接保留到官方主分支，便于继续跟进；如果后续参数或实现发生变化，应先与上述 revision 比较，再更新讲义。

## 1. 第一优先级：Kimi K3 本体

完整的参数、公式和实现核对保留在[架构研究笔记](architecture-reference.md)中；面向观众的主讲内容见[架构讲义](../../architecture/kimi-k3.md)与 [HTML 演示](../../architecture/kimi-k3-visualization/index.html)。

### 官方技术报告

- 报告页：[arXiv:2607.24653](https://arxiv.org/abs/2607.24653)
- 官方 PDF：[Kimi K3: Open Frontier Intelligence](https://github.com/MoonshotAI/Kimi-K3/raw/main/k3_tech_report.pdf)
- 官方仓库：[MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)

建议阅读顺序：

| 内容 | 报告位置 | 用途 |
|---|---|---|
| 总体架构图 | Figure 2，p. 3 | 讲 token/depth/channel 三条主线 |
| Hybrid Attention | §2.1，pp. 4–6 | KDA、Gated MLA、NoPE |
| Lower-bounded KDA | Figure 3，p. 5 | 解释算法参数化如何改变 kernel |
| Attention Residuals | §2.2，p. 6 | Full/Block AttnRes |
| Stable LatentMoE | §2.3，pp. 6–9 | latent path、SiTU、QB |
| SiTU-GLU | Figure 4，p. 7 | activation 曲线 |
| Quantile Balancing | Figure 5，p. 8 | router balancing |
| Native Vision | §2.4，pp. 9–10 | MoonViT-V2 |
| K2/K3 对比 | Table 1，p. 11 | 架构变化总表 |
| 1M context curriculum | §3.4，p. 12 | 8K→64K→256K→1M |
| KDA 系统实现 | §5.1，pp. 17–18 | prefill、context parallel |
| MoE training infra | §5.2，pp. 19–21 | MoonEP、memory、overlap |
| Hybrid cache | §5.4.1，pp. 23–24 | KDA state + MLA KV |
| Inference kernels | §5.4.2，pp. 24–25 | KDA/AttnRes/MoE 性能问题 |

使用建议：

- 后续网页最好重画 Figure 2，而不是直接截图。原图信息密度较高，重画时保留“三个 mixing 维度”即可。
- Figure 3、4、5 适合拆成独立交互图，不要全部塞进一页。
- 任何 `2.5×` 结论都要注明是 architecture + data + training recipe 的组合 scaling-law 结果。

### 官方发布博客与模型卡

- [Kimi K3 Tech Blog](https://www.kimi.com/blog/kimi-k3)
- [Kimi K3 Hugging Face model card](https://huggingface.co/moonshotai/Kimi-K3)
- [官方 README](https://github.com/MoonshotAI/Kimi-K3/blob/main/README.md)

适合确认：

- 对外定位、发布时间和开放权重状态；
- 2.8T / 104B / 1M 等摘要参数；
- 官方对 KDA、AttnRes、Stable LatentMoE 的一句话表述。

不适合确认：

- 具体 tensor shape；
- QAT 的精确覆盖范围；
- KDA/MLA 层号；
- inference cache 的真实 layout。

这些内容应回到技术报告、config 和代码。

## 2. 第二优先级：公开 checkpoint 配置与推理实现

### 配置

- [config.json](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
- [configuration_kimi_k3.py](https://huggingface.co/moonshotai/Kimi-K3/blob/main/configuration_kimi_k3.py)

config 可直接核对：

```text
num_hidden_layers = 93
hidden_size = 7168
attn_res_block_size = 12
first_k_dense_replace = 1
num_experts = 896
num_experts_per_token = 16
num_shared_experts = 2
routed_expert_hidden_size = 3584
moe_intermediate_size = 3072
kv_lora_rank = 512
q_lora_rank = 1536
max_position_embeddings = 1048576
```

1-based full-attention layer pattern：

```text
4, 8, 12, ..., 88, 92, 93
```

其余 69 层为 KDA。

### 文本 backbone

- [modeling_kimi_linear.py](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py)

适合核对：

- `KimiDeltaAttention`：short convolution、chunk/recurrent 两种执行路径、lower bound 和 full-rank output gate；
- `KimiMLAAttention`：NoPE MLA 和 output gate；
- `KimiSparseMoeBlock`：router → down projection → expert → RMSNorm → up projection + shared experts；
- `KimiDecoderLayer`：第一层 dense，后续层 MoE；
- `_apply_attn_res`：公开 inference 版 Block AttnRes 的精确数据流。

注意：

- 这是通用 Hugging Face 推理参考实现，不等于 Moonshot 生产 kernel；
- 训练路径中的 MoE 在该文件中明确未实现；
- cache 结构也不应直接当作生产 serving footprint。

### 多模态路径

- [modeling_kimi_k3.py](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_k3.py)
- [kimi_k3_vision_processing.py](https://huggingface.co/moonshotai/Kimi-K3/blob/main/kimi_k3_vision_processing.py)

适合核对：

- MoonViT-V2 的 patch embedding、27-layer encoder；
- 2×2 spatial merge 和 temporal pooling；
- projector；
- image placeholder 被替换成 visual embedding sequence 的方式。

## 3. 第三优先级：核心组件原始论文

### KDA / Hybrid Attention

- 论文：[Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- 代码：[MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- kernel：[MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA)

主要用途：

- delta-rule recurrence；
- KDA 的 fine-grained forget gate；
- chunkwise parallel form；
- 为什么混合 KDA 与 MLA。

引用性能数字时要写清实验对象。`up to 75% KV-cache reduction` 和 `up to 6× decoding throughput at 1M context` 来自 Kimi Linear 48B，不是 Kimi K3 端到端 benchmark。

### Attention Residuals

- 论文：[Attention Residuals](https://arxiv.org/abs/2603.15031)
- 代码/说明：[MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)

主要用途：

- 标准 residual 的 depth bottleneck；
- Full AttnRes 公式；
- Block AttnRes 的 `O(Ld) → O(Nd)`；
- content-dependent depth routing。

### LatentMoE

- 论文：[LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts](https://arxiv.org/abs/2601.18089)

主要用途：

- shared full-width path 与 routed latent-width path 的分离；
- 为什么 latent width 能换取更大的 expert pool / top-k；
- accuracy per FLOP / parameter 的原始设计背景。

K3 的 Normalized LatentMoE、SiTU-GLU、Quantile Balancing 是在该基本结构上的进一步稳定化，不要把它们误写成 LatentMoE 原论文已有的同一套方案。

## 4. 系统资料：为 Roofline 和 MoE 优化章节准备

### MoonEP

- [MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP)

关键词：

```text
dynamic redundant experts
perfect per-rank token balance
static computation shapes
zero-copy dispatch/combine
expert weight prefetch
shared-expert overlap
```

公开实现面向 NVIDIA GPU。它是问题定义和系统设计参考，不是 TPU 可直接复用的实现。

### 报告中的 inference 线索

技术报告 §5.4.2 明确给出三个值得后续验证的判断：

1. KDA prefill 的问题是 chunk 内并行与跨 chunk state propagation；decode 的问题是 recurrent-state traffic 与原地更新。
2. Block AttnRes 在 prefill/decode 都有明显 memory-access 成本，优化重点是 sequence parallel、overlap 和 fusion。
3. 小 batch 下 routed-expert group GEMM 会退化成 weight streaming，官方 GPU 实现因此采用 token-centric decode kernel。

这些是后续 TPU profile 的假设来源，不是 TPU 上已经成立的测量结果。

## 5. K2 对照资料

- [MoonshotAI/Kimi-K2](https://github.com/MoonshotAI/Kimi-K2)
- K3 Technical Report Table 1（p. 11）

对照时优先使用 K3 报告的同表口径，避免把不同 K2 checkpoint 或 K2.5 多模态配置混进来。

## 6. 不作为架构事实来源

下面这些材料可用于了解舆论或部署进展，但不能覆盖官方 config/report：

- 新闻报道和参数转载；
- 未注明 checkpoint revision 的模型聚合网站；
- 第三方“架构复现”；
- `nano-kpu` 等 K3 生成的 demo 项目——它证明的是 agent 能力，不是 Kimi K3 本体的精确硬件结构；
- benchmark 榜单截图。

## 7. 建议后续补齐的本地原始资料

当前仓库先保留稳定链接和阅读索引，避免无选择地复制大文件。正式制作网站前，可再决定是否 vendor 以下资产：

```text
sources/kimi-k3/
├── README.md
├── papers/
│   ├── k3-tech-report.pdf
│   ├── kimi-linear.pdf
│   ├── attention-residuals.pdf
│   └── latent-moe.pdf
├── configs/
│   └── config.json
└── figures/
    ├── architecture-redraw.svg
    ├── kda-decay-redraw.svg
    ├── attnres-redraw.svg
    └── latent-moe-redraw.svg
```

建议只把“可复现页面所需、许可证允许、revision 已固定”的原始文件放入仓库；其余用 URL + commit/revision 固定。
