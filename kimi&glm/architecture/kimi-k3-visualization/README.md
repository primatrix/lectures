# Kimi K3 Information Flow

面向已经了解 Transformer、Attention 和 MoE 基础概念，但不需要深入到论文公式或模型配置的观众。

演示不从参数规模和层数开始，而是围绕三条信息通道展开：

1. 跨 token：KDA 工作记忆与周期性全局 MLA；
2. 跨 depth：Attention Residuals；
3. 跨 channel：Stable LatentMoE；
4. Native Vision 将图像和视频接入同一个信息流。

## 本地预览

这是一个无构建依赖的单文件网页，CSS、JavaScript 和全部演示内容都在 `index.html` 中。

```bash
cd "kimi&glm/architecture/kimi-k3-visualization"
python3 -m http.server 8080
```

访问：

```text
http://localhost:8080
```

## 演示操作

- `Space` / `→`：下一步
- `←`：上一步
- `↓`：下一章
- `↑`：上一章
- 左侧章节导航：直接跳转
- 页面底部按钮：鼠标操作
- 移动端左右滑动：上一步 / 下一步

## 章节结构

1. 核心问题：信息怎样走得更远，又不处处支付完整成本
2. 普通 Transformer 的三条信息通道
3. KDA：完整档案与工作记忆
4. Hybrid Attention：工作记忆与全局查阅
5. AttnRes：对网络深度做 Attention
6. LatentMoE：router 看完整 token，专家处理更窄的 latent payload
7. Native Vision：统一 visual/text token stream
8. 架构到系统：三种 cost shift

## 演示边界

- 不展示模型参数表、层数表或 K2/K3 对比。
- 不展开 KDA recurrence、AttnRes softmax、SiTU-GLU 和 Quantile Balancing 的完整公式。
- 专家用 A/B/C/D 表示；它们的分工由训练形成，不假设存在人工命名的“代码专家”或“数学专家”。
- 性能页只陈述架构引入的新成本，不把 GPU 实现结论直接迁移成 TPU 结论。

## 协作约定

- 保持页面为离线可运行的单文件，不引入 CDN 或在线字体。
- 每章只回答一个问题，标题和底部讲稿优先使用观众能复述的语言。
- 修改步骤时同步更新 `<section data-steps>` 与脚本中的 `captions`。
- 动画只由键盘、按钮或触摸操作触发，不自动播放。
