# GLM-5.2 与 Kimi K3：长上下文模型架构讲义

> GLM-5.2：@炯轩 · Kimi K3：@brian
>
> 配套演示：[GLM-5.2 & Kimi K3 HTML PPT](./glm5.2-visualization/index.html)

这份讲义面向已经了解 Transformer、Attention、Residual 和 MoE 基础概念的观众。它与 HTML PPT 的 16 个章节一一对应，可作为分享前的预习材料，也可作为讲者稿使用。

整场分享只追一条主线：上下文、网络深度和模型容量继续扩大以后，模型怎样保存信息、选择信息，并避免让每个 token 为全部状态和全部参数付费？

---

## 第一部分：从 Attention 到 GLM-5.2

### Chapter 01：从 activation 得到 Q、K、V

给定一段包含 `S` 个 token 的 hidden states：

$$
X\in\mathbb{R}^{S\times d_{\text{model}}}
$$

Attention 首先沿 feature 维做三次线性投影：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

投影不会改变 token 轴。输入有 `S` 行，得到的 Q、K、V 仍然各有 `S` 行；变化的是每一行的 feature 表示。

这一点是后面理解 KV cache 的基础：历史状态按 token 逐行保存，序列越长，历史行数越多。

**本页结论：** Attention 的纵轴是 token，横轴是 feature；Q、K、V 都保留输入的 token 数。

---

### Chapter 02：一行 Query 如何变成一行输出

只看当前 token 的一行 query：

$$
p_t=\operatorname{softmax}\left(
\frac{q_tK^{\mathsf T}}{\sqrt{d_h}}
\right)
$$

如果当前能看到 `S` 个历史 token，`p_t` 就包含 `S` 个权重。每个权重回答：当前 query 应该从对应的历史 token 读取多少信息？

这些权重再用于混合 Value：

$$
o_t=p_tV
$$

因此，一行 query 先产生一行 token 权重，再把多行 Value 混合成一行输出。权重的长度由可访问的 token 数决定，而不是由 embedding dimension 决定。

**本页结论：** Attention 本质上是“按 query 对历史 token 分配权重，再用同一组权重混合 Value”。

---

### Chapter 03：Decode 的历史来自 KV cache

Prefill 会并行处理整段输入，并把每个 token 的 K/V 写入 KV cache。Decode 每次只增加一个 token，因此只需要计算当前的 `q_t、k_t、v_t`。

当前的 `k_t、v_t` 会追加到 cache；当前 query 则读取截至 `t` 的全部历史：

$$
o_t=\operatorname{softmax}\left(
\frac{q_tK_{1:t}^{\mathsf T}}{\sqrt{d_h}}
\right)V_{1:t}
$$

```text
当前 hidden state
      ├── q_t：读取历史
      ├── k_t：追加到 K cache
      └── v_t：追加到 V cache
```

KV cache 避免重复计算历史 K/V，但没有消除历史读取。上下文越长，cache 容量越大，Decode 每一步需要访问的历史也越长。

**本页结论：** 长上下文的核心压力不仅是 Attention FLOPs，还包括逐 token 增长的状态容量和读取流量。

---

### Chapter 04：MHA 复制多条 Attention 路径

Multi-Head Attention 将 feature space 划分为多个 head。每个 head 独立生成 Q、K、V，独立计算 token 权重，再把所有 head 的输出拼接起来。

```text
Q₁, K₁, V₁ → Attention₁ → O₁
Q₂, K₂, V₂ → Attention₂ → O₂
Q₃, K₃, V₃ → Attention₃ → O₃
Q₄, K₄, V₄ → Attention₄ → O₄
                         ↓
                    Concat + W_O
```

多个 head 可以在不同子空间建模不同关系，但每个历史 token 也需要保存多组 head-specific K/V。

**本页结论：** MHA 增加了并行表达路径，也让 KV cache 带上了 head 维度。

---

### Chapter 05：GQA 合并 K/V heads

Grouped-Query Attention 保留多个 Query/Attention heads，但让一组 Query heads 共享同一组 K/V。

例如，4 个 Query heads 可以只对应 2 组 K/V：

```text
Q₁、Q₂ ──→ Shared KV A ──→ Attention₁、Attention₂
Q₃、Q₄ ──→ Shared KV B ──→ Attention₃、Attention₄
```

GQA 减少了需要保存的 K/V heads，同时保留多条 Query 和 Attention 路径。它压缩的是 head 组织方式，历史 token 行数仍然随上下文增长。

**本页结论：** GQA 让多个 Query heads 共享 K/V，用较少 cache 换取接近 MHA 的多头表达。

---

### Chapter 06：MLA 用 latent 表示压缩 KV cache

Multi-head Latent Attention 不再为每个历史 token 保存完整的 multi-head K/V，而是先将 hidden state 压缩成更窄的 latent：

$$
c_t^{KV}=W^{DKV}h_t
$$

在非吸收式实现中，使用某个历史 token 时，可以再从 latent 恢复对应的 K/V：

$$
k_t^C=W^{UK}c_t^{KV},\qquad
v_t^C=W^{UV}c_t^{KV}
$$

同样是 5 个历史 token，cache 仍有 5 行；变化在于每行不再保存完整的多头 K/V，而是保存更紧凑的 latent 状态。

部分投影还可以利用矩阵乘法结合律吸收到 query 或输出路径。RoPE 的位置相关部分不能简单吸收，因此需要单独处理。

**本页结论：** MLA 没有减少历史 token 数，而是减少每个历史 token 常驻的状态宽度。

---

### Chapter 07：DSA 先选择历史，再做核心 Attention

MLA 回答“历史状态以什么形式保存”，DeepSeek Sparse Attention 回答“当前 query 需要读取哪些历史位置”。

DSA 将历史访问拆成四步：

1. Indexer 对历史 token 计算相关性分数；
2. Top-K 产生需要读取的位置；
3. Gather 读取这些位置的 latent KV；
4. Sparse Core Attention 只在选中位置上计算。

```text
Current Query
     ↓
  Indexer ──→ history scores
     ↓
   Top-K ──→ token positions
     ↓
  Gather ──→ selected latent KV
     ↓
Sparse Core Attention
```

核心 Attention 的读取范围从全部历史收敛到 Top-K，但系统新增了 Indexer、Top-K 和 Gather。这些环节在 Prefill 与 Decode 中可能呈现不同瓶颈。

**本页结论：** DSA 不是把 Attention 直接改成稀疏矩阵，而是先选择 token，再让核心 Attention 精确读取。

---

### Chapter 08：GLM-5.2 如何组装这些模块

GLM-5.2 将 MLA、DSA、IndexShare 与 MoE 放在同一条数据流中：

```text
Hidden
  → MLA latent KV
  → Indexer
  → Top-K positions
  → Gather
  → Sparse Core Attention
  → MoE
  → Output
```

公开配置显示，模型包含 78 个 Transformer layers，最长位置长度为 1,048,576。DSA 的 Top-K 为 2048；MoE 包含 256 个 routed experts，每个 token 激活 8 个，并保留 shared expert。

IndexShare 进一步减少层间重复选择。一个 full indexer layer 产生 Top-K index，后续相邻层复用这些位置，但仍然执行自己的 Core Attention。

```text
Full layer：Indexer → [t₂, t₄, …] → Core Attention
Shared +1：reuse index          → own Core Attention
Shared +2：reuse index          → own Core Attention
Shared +3：reuse index          → own Core Attention
```

IndexShare 共享的是 token index，不是完整 Attention 输出，也不是完整 KV cache。

**本页结论：** MLA 让历史存得更紧，DSA 让核心读得更少，IndexShare 减少重复选择，MoE 让 FFN 稀疏激活。

---

## 第二部分：Kimi K3 的三条信息通道

### Chapter 09：1M Agent trace 会卡在哪里

先不看模块名，想象一个持续运行的 agent。它读需求、浏览代码、调用工具、观察画面、定位错误并继续修改；这些信息不断进入上下文。

上下文从 128K 扩展到 1M 后，会出现三类问题：

- **跨 token：** KV cache 随历史增长，Decode 每一步需要加载更长的状态；
- **跨 depth：** residual 逐层累加，早期表示容易混入累计流；
- **跨 channel：** Dense 扩容会让每个 token 执行更多参数。

三个问题可以合成一句话：

> 怎样让有效信息留下来，又不让状态、访存和计算随 scale 一起失控？

---

### Chapter 10：K3 的共同回答是压缩与选择

Kimi K3 分别改造 token、depth 和 channel 三条信息通道：

| 信息方向 | 扩展问题 | K3 的机制 |
|---|---|---|
| Token | 历史状态逐 token 增长 | KDA 固定状态 + 周期性 Gated MLA |
| Depth | residual 难以选择早期表示 | Block Attention Residuals |
| Channel | 模型容量扩大抬高 token 成本 | Stable LatentMoE |

三条路线的共同点不是“删除信息”，而是先把完整 dense cost 变成更紧凑的状态或路径，再按内容选择真正需要的部分。

---

### Chapter 11：KDA 如何遗忘、修正、写入和读取

Kimi Delta Attention 用固定大小的 recurrent state 替代随序列长度增长的逐 token KV 列表。对一个 attention head，状态更新为：

$$
S_t=
(I-\beta_t k_tk_t^{\mathsf T})
\operatorname{Diag}(\alpha_t)S_{t-1}
+\beta_t k_tv_t^{\mathsf T}
$$

输出为：

$$
o_t=S_t^{\mathsf T}q_t
$$

![KDA 状态更新公式与矩阵示意](./glm5.2-visualization/assets/kda-state-update-tikz.png)

按实际计算顺序，可以把公式拆成四个动作：

1. **遗忘：** `Diag(α_t)S_{t-1}` 对旧状态的 key channels 做不同强度的保留；
2. **修正：** `I-β_tk_tk_t^T` 压低当前 key 方向上的旧关联；
3. **写入：** `β_tk_tv_t^T` 写入当前的 key→value 关联；
4. **读取：** `S_t^Tq_t` 让 query 从更新后的状态得到输出。

普通 linear attention 更接近不断累加 `kv`。KDA 增加遗忘和 delta correction，因此可以覆盖或修正已有方向，而不只是继续追加。

状态尺寸始终是 `d_k×d_v`。无论上下文是 1K、128K 还是 1M token，KDA 层的历史状态量都不按 token 数继续增长。

这不等于无损保存全部历史。KDA 保存的是压缩后的工作状态，精确的 token-level 细节仍需要其他机制补充。

**本页结论：** KDA 用固定状态承接长历史，通过遗忘、修正和写入持续更新内容。

---

### Chapter 12：Hybrid Attention 结合工作状态与全局查阅

K3 不是纯 KDA。论文中的一个 Hybrid block 从下往上包含三组 `KDA + Stable LatentMoE`，再接一组 `Gated MLA + Stable LatentMoE`。

![Kimi K3 Hybrid block](./glm5.2-visualization/assets/kimi-k3-figure2-hybrid-block.png)

```text
KDA + Stable LatentMoE
KDA + Stable LatentMoE
KDA + Stable LatentMoE
Gated MLA + Stable LatentMoE
```

主干重复 23 组 Hybrid block，并在末尾追加一层 Gated MLA。因此共有 69 层 KDA 和 24 层 Gated MLA。

KDA 负责高频的固定状态更新；Gated MLA 周期性保留全局 token-to-token Attention。这里的 Gated MLA 是固定排布，不是模型需要时才触发的外部工具。

**本页结论：** KDA 负责可扩展的工作状态，全局 Attention 周期性补充精确的历史关联。

---

### Chapter 13：AttnRes 让当前层沿网络深度选择信息

普通 residual 将每个子层输出以固定方式加入同一条 residual stream。网络变深后，早期表示会不断混入累计结果，当前层难以单独重新选择某个早期来源。

![Standard、Full 与 Block Attention Residuals](./glm5.2-visualization/assets/attention-residuals-overview.png)

Full Attention Residuals 把 embedding 和全部早期层表示作为 depth sources。当前层使用 learned pseudo-query `w` 生成权重 `α`，再按内容组合这些来源。

Block Attention Residuals 保留 block 内的普通 residual，只在 block representations 之间做 depth attention。这样可以把需要保存的表示从逐层规模降到逐 block 规模。

不要把图中的 `α` 理解为固定层权重。它随当前 token 的表示变化，因此不同 token 可以重新读取不同深度的信息。

**本页结论：** Self Attention 沿 token 选择来源，AttnRes 沿网络 depth 选择来源。

---

### Chapter 14：LatentMoE 只压缩 routed expert 路径

传统 MoE 通过稀疏激活扩大容量，但 Top-K 增大以后，Serving 仍需读取更多专家权重，并在 Expert Parallel 中传输更多 token payload。

LatentMoE 的关键不是把整个模型变窄，而是只压缩送往 routed experts 的路径。

![Kimi K3 Stable LatentMoE](./glm5.2-visualization/assets/kimi-k3-figure2-stable-latentmoe.png)

三条路径应当分开理解：

```text
Router：x[d] → Sigmoid(W_r x) → Top-k indices + weights

Routed：x[d] → W↓ → z[ℓ] → dispatch → experts
        → combine → RMSNorm → W↑ → routed output[d]

Shared：x[d] → shared experts → shared output[d]
```

Router 直接读取 full-width `x[d]`。`W↓` 与 Router 是从 `x` 分出的并行支路，不是先压缩再让 Router 选择。

真正跨设备 dispatch 并进入 routed experts 的是 `z[ℓ]`。因此 routed expert 的主要权重矩阵和 All-to-All payload 都从宽度 `d` 缩到 `ℓ`。

Shared experts 始终处理 full-width 输入。Routed output 经 RMSNorm 和 `W↑` 回到 `d` 维，再与 shared output 相加。

“Stable” 还包括数值稳定设计：RMSNorm 控制 routed aggregate 的尺度，SiTU-GLU 抑制极端乘法激活。

Quantile Balancing 是训练时更新 Router selection bias 的负载均衡方法。它影响训练中的专家分配，不改变上面的推理数据路径，也不保证每个推理 batch 都完全均衡。

**本页结论：** Router 看完整表示，只有专家 payload 进入 latent space；省下的带宽与权重读取预算可以换取更多专家路径。

---

### Chapter 15：Native Vision 的创新在联合训练

MoonViT-V2 相比前代主要改变训练方式，而不是发明另一套视觉推理路径。

![Kimi K3 Native Vision 路径](./glm5.2-visualization/assets/kimi-k3-figure2-vision-path.png)

#### 单一 next-token loss

视觉 token 与文字一起构成上下文，K3 仍然预测下一个文本 token。预测错误只产生一份 next-token loss，这份 loss 同时约束文本生成和前面的视觉表示。

#### 端到端联合训练

Loss 的梯度会经过 K3 backbone 和 MLP projector 回传到 MoonViT-V2。训练不是冻结 text backbone 后只更新 ViT 或 adapter，而是三部分从预训练开始共同学习。

#### MoonViT-V2 从头训练

MoonViT-V2 不使用 SigLIP 等对比学习视觉编码器进行初始化。论文观察到 from-scratch 联合训练的梯度更稳定，视觉表示也会直接围绕 OCR、文字和细粒度结构理解形成。

推理数据流仍然与常见 VLM 相同：

```text
图片 / 视频
    → MoonViT-V2
    → MLP projector
    → visual embeddings
    → 插入 text embedding sequence
    → K3 backbone 自回归推理
```

**本页结论：** Native 的重点是训练目标和主干联合优化；推理时仍由 ViT 理解视觉，再把视觉表示交给文本模型。

---

## 第三部分：从架构走向系统分析

### Chapter 16：架构没有消灭成本，而是重新安排成本

GLM-5.2 和 Kimi K3 选择了不同的长上下文路线，但都把完整 dense cost 重组为更明确的系统对象。

| 模型 | 架构机制 | 系统需要关注的对象 |
|---|---|---|
| GLM-5.2 | MLA | latent KV cache |
| GLM-5.2 | DSA | Indexer、Top-K、Gather |
| GLM-5.2 | IndexShare | 跨层 index 复用 |
| Kimi K3 | KDA | recurrent state update |
| Kimi K3 | AttnRes | block representation reads |
| Kimi K3 | LatentMoE | routing、latent dispatch、All-to-All |
| Kimi K3 | Native Vision | 视觉序列与统一 backbone |

接下来的性能分析需要按阶段展开：

- **Prefill：** 关注并行处理长序列、state scan、Top-K、Gather 和矩阵计算；
- **Decode：** 关注 cache/state 读取、权重 streaming、通信、launch 与调度延迟；
- **TPU Roofline：** 将 FLOPs、HBM bytes 和 collective bytes 映射到硬件上；
- **MoE 优化：** 分析 routing、dispatch/combine、expert GEMM、通信与融合。

架构不是性能结论。它只定义了必须执行的数据流；最终瓶颈仍取决于 batch、序列长度、并行策略、kernel 和部署拓扑。

---

## 总结

GLM-5.2 的主线是：

> 用 MLA 压缩逐 token 历史，再用 DSA 选择核心 Attention 真正读取的位置，并通过 IndexShare 减少层间重复选择。

Kimi K3 的主线是：

> 用 KDA 与 Gated MLA 改造 token 信息流，用 AttnRes 改造 depth 信息流，用 LatentMoE 改造 channel 信息流，再把视觉接入同一个训练目标和文本主干。

理解这些机制后，下一步不应继续罗列架构名词，而应把每个模块拆成计算、状态、访存与通信，再分别分析 Prefill 和 Decode。

---

## References

- [GLM-5.2 官方介绍](https://z.ai/blog/glm-5.2)
- [Kimi K3 Technical Report](https://arxiv.org/abs/2607.24653)
- [Kimi Linear](https://arxiv.org/abs/2510.26692)
- [Attention Residuals](https://arxiv.org/abs/2603.15031)
- [Attention Residuals GitHub](https://github.com/MoonshotAI/Attention-Residuals)
- [LatentMoE](https://arxiv.org/abs/2601.18089)
- [NVIDIA LatentMoE Project](https://research.nvidia.com/labs/nemotron/LatentMoE/)
