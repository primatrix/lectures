# GLM5.2 & Kimi K3：Bring-up 与背景

## 主题

本次分享围绕 GLM5.2 与 Kimi K3 展开，从模型架构出发，结合 TPU 的硬件特征和推理阶段差异，讨论模型模块在实际推理系统中的性能表现与优化思路。

分享希望串起下面这条主线：

> 模型为什么采用这些架构 → Prefill 和 Decode 如何执行 → TPU 上的瓶颈在哪里 → 如何以 MoE 为例分析和优化模块性能。

整体以简短 presentation 为目标，预计 15～30 分钟。内容以建立直观认识和解释关键取舍为主，不展开为完整的技术报告。

## 内容与分工

### 1. 架构

- GLM5.2：@炯轩
- Kimi K3：@brian

这一部分介绍两个模型最重要的架构变化，并回答：这些设计分别想解决什么问题？它们对推理系统带来了哪些新的计算、存储或通信需求？

### 2. 硬件：TPU Roofline

- 负责人：@brian

建立后续性能分析所需的硬件背景，包括计算吞吐、内存带宽、算术强度，以及如何判断一个算子更接近 compute-bound 还是 memory-bound。

### 3. Prefill 与 Decode 的区别

- 负责人：@炯轩

从推理执行过程解释两个阶段的主要差异：

- Prefill 一次处理多个输入 token，并行度较高，通常更容易利用矩阵计算能力。
- Decode 按 token 自回归执行，单步计算规模较小，更容易受到权重读取、状态访问、通信和调度开销影响。
- 同一个模块在 Prefill 和 Decode 阶段可能呈现完全不同的性能瓶颈，因此需要分别分析和优化。

### 4. 模块优化：以 MoE 为例

- 负责人：@brian

以 MoE 为例，将前面的模型与硬件背景落到具体模块上，讨论 routing、expert dispatch、矩阵计算、通信以及算子融合等可能的优化方向。

## 共同背景

GLM5.2 和 Kimi K3 都在继续扩大模型规模和上下文长度，但推理性能并不只由参数量或理论计算量决定。模型结构会改变需要保存和读取的状态、可利用的并行度以及设备间的通信方式；推理框架和 kernel 的实现，则决定这些理论设计能否真正转化为性能收益。

因此，本次分享不把“模型架构介绍”和“性能优化”视为两个割裂的话题，而是通过 Prefill、Decode 和 TPU Roofline 建立连接：

1. 先理解模型引入了哪些新模块。
2. 再判断这些模块在 Prefill 和 Decode 中分别执行什么工作。
3. 将计算量、访存量和通信量映射到 TPU Roofline。
4. 最后通过 MoE 案例展示如何形成具体的性能优化判断。

## 写作边界

- 架构部分强调设计动机、核心机制和关键差异，避免堆叠过多公式。
- Roofline 部分服务于后续分析，不展开成完整的硬件课程。
- 性能部分区分公开资料中的理论收益、仓库实现现状和待验证的优化假设。
- 尚未完成 launch 或 benchmark 的内容不写成确定的性能结论。
- 每个模块优先回答“解决什么问题、引入什么代价、可能在哪里优化”。

## 待补充的背景资料

- [GLM-5 系列官方仓库](https://github.com/zai-org/GLM-5)
- [IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201)
- [Kimi K3 官方介绍](https://www.kimi.com/blog/kimi-k3)
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- [Attention Residuals](https://arxiv.org/abs/2603.15031)
- TPU Roofline 与 SGLang-JAX 实际 profile、benchmark 资料（后续补充）

## 已整理材料

- [Kimi K3 HTML 演示](../architecture/kimi-k3-visualization/index.html)
- [Kimi K3 架构讲义](../architecture/kimi-k3.md)
- [Kimi K3 架构研究笔记](../sources/kimi-k3/architecture-reference.md)
- [Kimi K3 原始资料索引](../sources/kimi-k3/README.md)

## 后续文档建议

```text
kimi&glm/
├── background/
│   └── README.md
├── architecture/
│   ├── glm5.2.md
│   ├── kimi-k3.md
│   └── kimi-k3-visualization/
│       └── index.html
├── tpu-roofline.md
├── prefill-vs-decode.md
└── moe-optimization.md
```
