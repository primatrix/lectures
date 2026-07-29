# Kimi K3：信息如何走得更远

> 面向已经了解 Transformer、Attention、Residual 和 MoE 基础概念的观众。
>
> 这份文档是 HTML 演示的讲者稿。目标不是复述配置表，而是让观众在分享结束后能够解释：KDA、AttnRes、LatentMoE 和 Native Vision 分别改变了哪一条信息通路，以及它们为什么会影响推理系统。

## 一句话主线

Kimi K3 的重点不是“更大的 Transformer”，而是让信息在三个方向上更有选择地流动：

- 跨 token：长历史不必处处完整重读；
- 跨 depth：深层网络可以重新选择早期表示；
- 跨 channel：不同 token 可以进入不同的专长路径。

KDA 与全局 attention、Attention Residuals、LatentMoE 分别改造这三条通道。Native Vision 则把视觉 token 接入同一条主干信息流。

## 演示入口

[从 Chapter 09 打开合并版 HTML 演示](glm5.2-visualization/index.html#chapter-9)

推荐时长为 12～15 分钟。每一章只回答一个问题；按 `Space` 或 `→` 逐步揭示画面，底部文字就是当前画面的讲述提示。

---

## Chapter 1：先问问题，不先报参数

### 屏幕上的问题

> 怎样让信息走得更远，又不在每一步都支付完整成本？

### 可以这样讲

“我们先不看 Kimi K3 有多少参数、多少层。那些数字能告诉我们模型有多大，却不能告诉我们它为什么这样设计。

想象一个 agent 连续工作很久：它读过大量代码和文档，中间观察过图片，也执行过很多工具。当前这个 token 想继续做对的事情，需要解决三类问题。

第一，它怎样从很长的历史里拿到信息；第二，经过很深的网络以后，它怎样重新使用早期形成的表示；第三，不同内容怎样得到不同的计算。K3 的架构可以沿着这三条信息通道来理解。”

### 转场

“这三条通道其实并不陌生，它们就在每个普通 Transformer block 里。”

---

## Chapter 2：先建立共同坐标系

### 屏幕上的对应关系

```text
Attention       → 跨 token 混合信息
Residual stream → 沿 depth 传递信息
FFN / MoE       → 跨 channel 做变换
```

### 可以这样讲

“Self Attention 决定当前 token 从哪些 token 读取内容；Residual stream 把表示一层一层传下去；FFN 则在同一个 token 内重新组合 channel。

这三种经典结构都很好用，但任务变长、网络变深、模型容量变大以后，它们各自遇到一个扩展问题：

完整 attention 的历史状态会越来越多；普通 residual 把很多层的输出混进同一条累计表示；dense FFN 又让每个 token 做同一套计算。

K3 没有抛弃 Transformer，而是分别改造这三个接口。”

### 观众应该记住

后面每讲一个新名词，都可以先问：它在改 token、depth，还是 channel？

---

## Chapter 3：KDA 是工作记忆，不是完整档案

### 屏幕上的对比

左边是 Full Attention 的完整档案：每个历史 token 都留下可供未来检索的 K/V。

右边是 KDA 的工作记忆：历史持续更新同一份 recurrent state，状态大小不再随着序列长度逐 token 增长。

### 可以这样讲

“完整 attention 像保存一整套逐字档案。优点是未来可以回到任意位置，缺点是历史越长，需要保留和读取的内容越多。

KDA 更像工作记忆。新 token 到来时，它不是再追加一页档案，而是更新现有状态：保留仍有用的信息，修正与新事实冲突的方向，再写入新的内容。当前 query 直接从这份状态里读取结果。

所以 KDA 的收益来自压缩历史，而不是免费保存全部历史。它换掉的是不断增长的逐 token 状态，代价是细节会被概括。”

### 不要讲成

- “KDA 记住了所有历史”——它维护的是有限状态，不是无损档案。
- “KDA 没有 cache”——它仍然有 recurrent state，只是状态不按 token 持续增长。
- “线性 attention 就一定更快”——真正性能还取决于 state 更新、数据布局和 kernel。

### 转场

“既然工作记忆会概括信息，模型怎样找回某个精确细节？这就是为什么 K3 不是纯 KDA。”

---

## Chapter 4：Hybrid Attention 是工作记忆加全局查阅

### 可以这样讲

“K3 同时保留两种互补能力。

多数时候，KDA 持续维护当前最重要的工作记忆；隔一段深度，Gated MLA 重新开放一次全局 token-to-token 查阅，让模型可以直接访问历史中的具体位置。

这不是在 KDA 和 full attention 之间二选一，而是让它们承担不同角色：KDA 负责高频、可扩展的记忆更新，全局 attention 负责低频但精确的历史检索。”

### 一个合适的类比

```text
KDA        = 脑中持续更新的工作记忆
Gated MLA  = 必要时回到原始档案查原文
```

这个类比只用于解释职责差异。KDA state 不是自然语言摘要，MLA 也不是由模型显式触发的外部工具。

### 转场

“前面解决的是沿序列回看。接下来换一个方向：网络很深时，当前层能不能回看早期层？”

---

## Chapter 5：AttnRes 让网络沿深度检索

### 屏幕上的对比

普通 residual 像一根接力棒：此前层的结果不断累加到同一条 hidden stream。

Attention Residuals 像保留多份中间草稿：当前层可以根据当前 token，选择更需要哪一份早期表示。

### 可以这样讲

“普通 residual 的连接方式是固定的：前面层的输出以固定方式累加。网络越深，早期信息越容易被混进一个累计状态里。

AttnRes 把此前的表示变成可选择的 depth sources。当前 token 会给这些来源分配内容相关的权重，再组合出当前层的输入。K3 使用 block 级版本，保留的是分块表示，而不是每一层的所有输出。

直观上，一个代码 token 可能想重新读取较早的语法和变量表示；一个视觉 token 可能想重新读取浅层的空间特征。这里的重点不是模型真的给这些层贴了标签，而是选择会随输入内容变化。”

### 观众应该记住

Self Attention 是“沿 token 选来源”，AttnRes 是“沿网络深度选来源”。

### 转场

“现在 token 能在时间和深度上选择信息。第三个方向，是选择把计算交给谁。”

---

## Chapter 6：LatentMoE 把专家计算放进更窄的空间

### 屏幕上的两条路径

```text
shared path：所有 token 都经过，承载公共变换
routed path：路由器选择少量专家，专家处理压缩后的 latent payload
```

### 可以这样讲

“普通 MoE 的问题不只在专家计算，也在 token 被送往专家时需要搬运多宽的 activation。

LatentMoE 把公共能力和专长能力拆成两条路。shared path 继续处理完整表示；routed path 则先把要送给专家的 payload 压到更窄的 latent space，再 dispatch 给被选中的专家，最后聚合并投影回主干宽度。

这里要把顺序讲准确：router 仍然从完整 token 判断该选哪些专家；被压缩的是发往专家的 payload，而不是 router 的观察范围。

这样做的关键，是把‘拥有更多专长路径’和‘每条路径都搬运完整宽度’解耦。Stable 这部分则处理另一个现实问题：专家变多以后，输出尺度、极端 activation 和负载不均都更难控制，因此需要 normalization、soft-cap 和 balancing。”

### 不要讲成

- “有固定的代码专家、数学专家”——专家分工由训练形成，画面中的标签只能作为示意。
- “只要稀疏就一定省时间”——routing、dispatch/combine、通信和小 batch 的权重读取都可能成为瓶颈。
- “router 在 latent space 里选专家”——公开实现中，router 读取 full-width 输入。

### 转场

“这条路径会自然连接到后面的 MoE 优化：模型层面省下的 dense 计算，到了系统层面变成 routing、通信与 weight streaming。”

---

## Chapter 7：Native Vision 不只是加一个图片入口

### 可以这样讲

“图像和视频先经过 MoonViT-V2，变成 visual tokens，再通过 projector 映射到文本 backbone 使用的 embedding space。之后，视觉和文字 token 进入同一个 K3 主干，也会经过 KDA、AttnRes 和 LatentMoE。

‘Native’ 的重点不是模型能接收图片这么简单，而是视觉能力从训练阶段就进入这条统一信息流。对 agent 来说，这使得‘生成代码—观察渲染结果—继续修改’可以成为同一段长程交互，而不是两个独立模型之间的临时接力。”

### 需要保留的边界

视觉输入仍有独立的 vision encoder 和 projector；“统一”指进入语言主干后的 token stream 与训练目标，不代表图片 patch 和文本 token 从输入端起就完全相同。

---

## Chapter 8：架构没有消灭成本，而是重新安排成本

### 最后一页这样收束

“K3 的共同思路，是把处处执行的完整 dense cost，换成结构化的 state、selection 和 routing：

- KDA 与 Gated MLA 用工作记忆更新和周期性全局查阅，替代每层都持续扩大历史读取；
- AttnRes 用 block representation 的保存与选择，换取沿深度的内容相关信息流；
- LatentMoE 用 latent dispatch、expert routing 和通信，换取更大的稀疏专长容量。

所以一句话记住 Kimi K3：它让信息沿 token、depth、channel 三个方向选择性流动。”

### 接到后续分享

“但选择不是免费的。KDA 带来 recurrent-state 更新，AttnRes 带来 block representation 的读取，LatentMoE 带来 routing 与通信。下一部分我们就把这些成本拆到 Prefill 和 Decode，再放到 TPU Roofline 上看。”

---

## References

- [Kimi K3 Technical Report](https://arxiv.org/abs/2607.24653)
- [Kimi Linear](https://arxiv.org/abs/2510.26692)
- [Attention Residuals](https://arxiv.org/abs/2603.15031)
- [LatentMoE](https://arxiv.org/abs/2601.18089)

## 内容边界

- 主讲不放参数表、层数表、模型代际对比或 benchmark 排名。
- 机制页只保留解释设计取舍所需的概念，不展开完整公式。
- 类比用于建立直觉，不能覆盖“有限状态”“block-level retrieval”“full-width router”等关键边界。
- GPU 报告中的优化经验只作为系统问题来源，不直接写成 TPU 性能结论。
