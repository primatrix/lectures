# GLM-5.2 & Kimi K3 Architecture + Inference Visualization

在线预览：[https://primatrix.github.io/lectures/kimi-glm/architecture/](https://primatrix.github.io/lectures/kimi-glm/architecture/)

GLM-5.2 与 Kimi K3 共用的单文件 HTML 演示。前 18 章从 Attention 基础逐步建立两种长上下文架构；Chapter 19 先给出 TPU v7x 的硬件基线，后 5 章再用 Roofline、HBM 流量与 Arithmetic Intensity 分析 Prefill、Decode、DSA 和 MTP。

## 本地预览

这是一个无构建依赖的单文件网页，CSS 和 JavaScript 都包含在 `index.html` 中。

最简单的方式是直接用浏览器打开 `index.html`。协作开发时建议启动本地静态服务器：

```bash
cd "kimi&glm/architecture/glm5.2-visualization"
python3 -m http.server 8080
```

然后访问：

```text
http://localhost:8080
```

修改 `index.html` 后刷新浏览器即可看到结果，不需要安装依赖或执行编译。

## 演示操作

- `Space` / `→`：下一步
- `←`：上一步
- `↓`：下一章
- `↑`：上一章
- 右侧章节导航：直接跳转到指定章节
- 页面底部按钮：适合鼠标操作或现场演示

## 章节结构

1. Attention：Activation → Q / K / V
2. Attention：单行 Query → 单行输出
3. Decode：新 token 与 KV cache
4. MHA：一条 Attention 路径扩展为多个 heads
5. GQA：多个 Query heads 共享 K/V
6. MLA：完整 multi-head KV 与 latent cache
7. DSA：Indexer、Top-K、Gather 与 Sparse Attention
8. GLM-5.2：MLA、DSA、IndexShare 与 MoE
9. Scaling information flow：sequence、depth、width 的扩展问题与 K3 的三个回答
10. Why Linear Attention：为什么 Softmax 历史不能提前压进固定状态
11. Fixed-state memory：固定状态为什么还需要遗忘与覆盖
12. KDA：遗忘、修正、写入与读取
13. Hybrid Attention：3× KDA + 1× Gated MLA
14. Why AttnRes：标准 residual 的固定等权聚合问题
15. AttnRes Q/K/V：沿 depth axis 的矩阵计算
16. Block AttnRes：沿 K3 Figure 2 的红色 w/α 路径理解跨 block 读取与 12-layer 配置
17. LatentMoE：896 / Top-16 与 routed width 7168 → 3584
18. 总结：衔接 Prefill、Decode、Roofline 与 MoE 优化
19. TPU v7x：官方芯片架构与单芯片算力、HBM、ICI 基线
20. Roofline：Compute Time、HBM Time 与 Arithmetic Intensity
21. Prefill / Decode：Chunk 与 Batch 的权重复用差异
22. GLM-5.2 Model Roofline：resident 与 active parameters
23. Long Context：Attention DP、KV 流量与 DSA
24. MTP：增加一次 target forward 的并行 token 数
25. MoE Cost Model：16K token、EP=32 下的 Compute、HBM 与 ICI 单层理论下界
26. Fused MoE Pipeline：按 expert wave 重叠 ICI、HBM DMA、MXU 与 VPU/VMEM
27. Measured Overlap：性能收益来自跨引擎重叠，当前瓶颈仍是 ICI 带宽与延迟

## 文件说明

```text
glm5.2-visualization/
├── index.html   # 页面结构、样式、交互与全部演示内容
└── README.md    # 本地预览与协作说明
```

配套文字讲义位于同级目录的 [`../README.md`](../README.md)。

## 协作约定

- 保持页面为离线可运行的单文件，不引入 CDN 或在线字体。
- Tensor 图中纵向始终表示 token，横向表示 embedding/head feature。
- 修改章节步骤时，同步更新：
  - 对应 `<section>` 的 `data-steps`
  - 页面底部 `captions` 中该章节的文案数量
  - 左侧章节导航及章节编号
- 动画只由键盘或按钮触发，不增加自动循环播放。
- 文案使用 presentation 口吻：强调结构、shape 和结论，避免描述讲解顺序。

## 提交前检查

1. 使用本地静态服务器打开页面。
2. 从 Chapter 01 逐步播放到 Chapter 27。
3. 确认所有 tensor 的 token 数、shape 标签和视觉行列一致。
4. 确认左右方向键、上下方向键、空格和导航按钮均能正常工作。
5. 确认浏览器控制台没有 JavaScript 错误。
