# GLM-5.2：从 MHA 到 DSA 的长上下文架构

> 本节负责人：@炯轩
>
> 配套演示：[GLM-5.2 Attention 可视化](./glm5.2-visualization/index.html)

GLM-5.2 面向长程任务和百万 token 上下文，核心架构围绕注意力状态的保存与访问展开：使用 MLA 压缩历史 K/V 表示，使用 DSA 选择需要参与核心注意力的历史 token，并通过 IndexShare 在相邻层之间复用 token 选择结果。

本节聚焦模型结构本身。Prefill/Decode 的阶段性性能差异、TPU Roofline、MoE 通信和具体 profile 在后续章节讨论。

## 1. 从 QKV Attention 到 KV Cache

给定输入 hidden states `X`，标准注意力首先计算：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

然后执行：

$$
\operatorname{Attn}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_h}}\right)V
$$

在自回归 Decode 的第 `t` 步，当前 query 需要和截至 `t` 的历史 K/V 交互：

$$
o_t=\operatorname{softmax}\left(
\frac{q_tK_{1:t}^{\mathsf T}}{\sqrt{d_h}}
\right)V_{1:t}
$$

因此，推理时会将已经生成 token 的 K/V 保存为 KV cache。每一步只需要计算新 token 的 `q_t、k_t、v_t`，并把新的 `k_t、v_t` 追加到 cache；历史 K/V 不再重复计算，但仍会被后续 Decode step 反复读取。

```text
Prefill：输入序列 → 计算 Q/K/V → 生成输出并写入 KV cache

Decode：当前 hidden state
          → 计算 q_t、k_t、v_t
          → 追加 k_t、v_t
          → q_t 读取历史 K_{1:t}、V_{1:t}
```

长上下文的注意力优化可以沿着两个方向理解：

1. 历史状态应该以什么形式保存？
2. 当前 query 需要访问哪些历史 token？

```text
历史状态的表示：MHA → GQA → MLA
历史 token 的访问：Dense Attention → DSA
```

## 2. 注意力结构的演变

### 2.1 MHA：每个 query head 对应一组 K/V

Multi-Head Attention（MHA）中，query、key、value 都按多个 head 组织。每个 query head 通常对应独立的 K/V head，当前 query 需要访问完整历史中的 K/V。

MHA 的结构直接、表达能力强，但随着上下文长度增长，历史 K/V 的保存和读取压力也会增长。

### 2.2 GQA：多个 query head 共享 K/V

Grouped-Query Attention（GQA）让多个 query head 共享一组 K/V head：

```text
多个 Q heads ──┐
               ├── 共享一组 K/V heads
多个 Q heads ──┘
```

GQA 减少了 K/V head 的副本，保留了较多 query head 的表达能力。它优化的是 K/V 的 head 组织方式，但当前 query 仍然需要访问对应历史中的 token。

### 2.3 MLA：将 K/V 压缩到 latent 表示

Multi-head Latent Attention（MLA）进一步改变 K/V cache 的表示方式。可以用下面一组简化公式表示其低秩 KV 压缩：

$$
c_t^{KV}=W^{DKV}h_t,\qquad
k_t^C=W^{UK}c_t^{KV},\qquad
v_t^C=W^{UV}c_t^{KV}
$$

推理时主要缓存低维的 `c_t^{KV}`，而不是每个 head 的完整 K/V。在同一 MLA 参数化下，显式恢复完整 K/V 与直接在 latent 空间计算是等价的，例如：

$$
q_t^{\mathsf T}W^{UK}c_j^{KV}
=\left((W^{UK})^{\mathsf T}q_t\right)^{\mathsf T}c_j^{KV}
$$

因此可以将部分投影吸收到 query 或输出路径。这里依赖的是矩阵乘法的结合律和分配律，而不是交换律；矩阵乘法一般不可交换，所以 RoPE 的位置相关部分需要通过 decoupled RoPE 单独处理。

MLA 解决的是“历史状态如何保存和表示”的问题：

```text
历史 token
    │
    ├── 共享的 latent KV 表示
    └── 解耦的 RoPE 位置表示
```

### 2.4 DSA：选择需要访问的历史 token

DeepSeek Sparse Attention（DSA）进一步优化“当前 query 访问哪些历史 token”。它先由轻量 indexer 对历史 token 打分，再选择 Top-K 位置，最后只对选中的历史状态执行核心注意力。

```text
MLA：历史状态以什么形式保存？
DSA：当前 query 需要访问哪些历史 token？
```

因此，GLM-5.2 中 MLA 和 DSA 是相互配合的两个层次：MLA 提供紧凑的历史状态表示，DSA 控制核心注意力的历史访问范围。

```text
[图 1：MHA → GQA → MLA → DSA 的架构演变]
MHA：独立 K/V heads
GQA：共享 K/V heads
MLA：latent KV + decoupled RoPE
DSA：Indexer + Top-K + Sparse Core Attention
状态：待绘制。
```

## 3. GLM-5.2 的整体结构

公开资料显示，GLM-5.2 采用 `glm_moe_dsa` 架构，模型规模标识为 744B-A40B，并支持最长约 1M token 的上下文。其整体结构可以概括为：

```text
输入 hidden states
        │
        ▼
78 个 Transformer blocks
        │
        ├── MLA-based Attention
        │      └── DSA：Indexer → Top-K → Sparse Core Attention
        │
        ├── FFN
        │      ├── 前 3 层：Dense FFN
        │      └── 后续层：MoE FFN
        │             ├── 256 个 routed experts
        │             ├── 每 token 选择 8 个
        │             └── 1 个 shared expert
        │
        └── MTP：speculative decoding 路径
```

其中，注意力和 FFN 是两个相对独立的结构优化方向：

- 注意力侧：MLA、DSA 和 IndexShare 面向长上下文历史状态；
- FFN 侧：MoE 通过稀疏专家激活扩大模型容量，具体路由和通信留给 MoE 部分展开。

公开配置中的关键结构参数如下：

| 结构项 | 配置 | 含义 |
|---|---:|---|
| Transformer 层数 | 78 | 注意力和 FFN 的层数 |
| 最大位置长度 | 1,048,576 | 百万上下文的结构背景 |
| routed experts | 256 | MoE 的 routed expert 总数 |
| 每 token 激活专家数 | 8 | 每个 token 的稀疏专家路径 |
| shared experts | 1 | 共享 FFN 路径 |
| index Top-K | 2048 | DSA 选择的历史位置数 |

```text
[图 2：GLM-5.2 单个 Transformer block]
Residual → MLA/DSA → Residual → MoE FFN
Attention 分支标出 Indexer、Top-K、Gather 和 Core Attention；
FFN 分支标出 Router、Experts 和 Combine。
状态：待绘制。
```

## 4. DSA：Indexer、Top-K 与核心注意力

DSA 将历史访问拆成“筛选”和“精确计算”两个阶段：

1. Indexer 对历史 token 计算相关性分数；
2. Top-K 产生需要访问的历史位置；
3. Gather 读取对应的历史状态；
4. Core Attention 在选中的历史状态上完成注意力计算。

```text
当前 Query / 历史状态
          │
          ▼
       Indexer
          │  对历史位置打分
          ▼
        Top-K
          │  输出 token positions
          ▼
        Gather
          │  读取选中的历史状态
          ▼
  Sparse Core Attention
          │
          ▼
       当前层输出
```

DSA 的结构优势在于：核心注意力不再对全部历史 token 执行同等强度的计算和访问。与此同时，Indexer、Top-K 和 Gather 也成为新的执行环节，这些环节在 Prefill 和 Decode 中的表现将在后续性能部分分析。

```text
[图 3：Dense Attention 与 DSA 的访问路径对比]
左侧：当前 query 访问全部历史 K/V；
右侧：Indexer 先选择 Top-K，再 Gather 选中的历史状态。
状态：待绘制。
```

## 5. IndexShare：跨层复用 token 选择

如果每一层都独立运行 Indexer，DSA 的 token 筛选会在层间重复执行。IndexShare 的结构是：由一个 full indexer layer 产生 Top-K token index，后续相邻的 shared layers 复用这组 index，并各自执行本层的 Core Attention。

```text
Full Indexer Layer
        │
        └── 产生 Top-K token index
              │
              ├── Shared Layer：复用 index → Core Attention
              ├── Shared Layer：复用 index → Core Attention
              └── Shared Layer：复用 index → Core Attention
```

公开配置中，78 层包含 full indexer layers 和 shared indexer layers；整体模式可以理解为周期性重新计算 index，其余相邻层复用结果。

IndexShare 共享的是 token 位置索引，不是完整的 attention 输出，也不是完整 KV cache。每个 shared layer 仍然保留自己的 query、注意力计算和输出路径。

```text
[图 4：IndexShare 的跨层结构]
标出 Full Indexer、Shared Layers、Top-K Index Buffer 和各层独立的 Core Attention。
状态：待绘制。
```

## 6. 架构总结与性能衔接

GLM-5.2 的注意力结构可以用三句话总结：

1. MHA、GQA、MLA 逐步改变历史 K/V 的组织和表示方式；
2. DSA 在此基础上选择需要参与核心注意力的历史 token；
3. IndexShare 在层间复用 token 选择结果，减少重复的 Indexer 路径。

从模型结构到后续性能分析的关系是：

```text
MLA / DSA / IndexShare
          │
          ▼
历史状态布局、Indexer、Top-K、Gather、Core Attention
          │
          ▼
Prefill 与 Decode 的不同执行形态
          │
          ▼
Roofline、Kernel 和端到端 Profile
```

### MTP 的位置

MTP（Multi-Token Prediction）主要服务 speculative decoding，属于 Decode 优化路径，而不是 MLA → DSA → IndexShare 这条注意力架构主线。正文只需简要说明其作用，具体的 MTP、KVShare 和 acceptance length 可以放到 Decode 部分或附录。

## 7. 分享页面结构

### Slide 1：从 QKV Attention 到 KV Cache

- QKV Attention 基本公式；
- Decode 读取历史 KV cache 的过程；
- 长上下文中的历史状态问题。

```text
[图：Prefill 写入 KV cache，Decode 读取历史 KV cache]
```

### Slide 2：MHA → GQA → MLA → DSA

- MHA：独立 K/V heads；
- GQA：共享 K/V heads；
- MLA：latent KV + decoupled RoPE；
- DSA：Indexer + Top-K + Sparse Core Attention。

```text
[图：四阶段注意力架构演变]
```

### Slide 3：GLM-5.2 整体结构

- 78 层 Transformer；
- MLA-based Attention + DSA；
- Dense FFN 与 MoE FFN；
- 256 routed experts、8 experts per token、1 shared expert。

```text
[图：GLM-5.2 Transformer block]
```

### Slide 4：DSA 内部流程

- Indexer；
- Top-K；
- Gather；
- Sparse Core Attention。

```text
[图：Dense Attention 与 DSA 数据访问对比]
```

### Slide 5：IndexShare

- Full indexer layer 产生 token index；
- Shared layers 复用 index；
- 各层仍独立执行 Core Attention；
- 共享对象不是完整 KV cache。

```text
[图：IndexShare 跨层复用]
```

### Slide 6：从架构到 Prefill/Decode

```text
MLA / DSA / IndexShare
          │
          ▼
不同阶段的计算、访存和调度形态
          │
          ▼
Prefill / Decode 与后续 Roofline 分析
```

## 参考资料

- [GLM-5 官方仓库](https://github.com/zai-org/GLM-5)
- [GLM-5.2 官方介绍](https://z.ai/blog/glm-5.2)
- [GLM-5.2 Hugging Face config](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)
- [GQA：Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [DeepSeek-V2：A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/html/2405.04434v5)
- [IndexCache：Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201)
- [GLM-5 技术报告](https://arxiv.org/abs/2602.15763)
