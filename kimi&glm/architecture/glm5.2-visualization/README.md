# GLM-5.2 & Kimi K3 Architecture Visualization

GLM-5.2 与 Kimi K3 共用的单文件 HTML 演示。前半部分从基础 Attention、Decode 与 KV cache 逐步建立 GLM-5.2 的 MLA、DSA 和 IndexShare；后半部分沿 token、depth、channel 三条信息通道解释 Kimi K3 的 KDA、AttnRes、LatentMoE 与 Native Vision。

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
3. Decode：新 token 与 5-token KV cache
4. MHA：1 个 head 扩展为 4 个并行 heads
5. GQA：4 个 Query heads 共享 2 组 KV
6. MLA：5 个 token 的完整 KV 与 latent cache 对比
7. DSA：Indexer、Top-K、Gather 与 Sparse Attention
8. GLM-5.2：MLA、DSA、IndexShare 与 MoE 的组合优势
9. Kimi K3：token、depth、channel 三条信息通道
10. KDA：完整 K/V 历史与 recurrent state
11. Hybrid Attention：KDA 工作记忆与 Gated MLA 全局查阅
12. Attention Residuals：沿网络深度选择 block representation
13. Stable LatentMoE：full-width router 与 latent expert payload
14. Native Vision：视觉 token 接入同一个 K3 backbone
15. 总结：从模型架构衔接 Prefill、Decode、Roofline 与 MoE 优化

## 文件说明

```text
glm5.2-visualization/
├── index.html   # 页面结构、样式、交互与全部演示内容
└── README.md    # 本地预览与协作说明
```

配套文字讲义位于同级目录的 `../glm5.2.md` 与 `../kimi-k3.md`。

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
2. 从 Chapter 01 逐步播放到 Chapter 16。
3. 确认所有 tensor 的 token 数、shape 标签和视觉行列一致。
4. 确认左右方向键、上下方向键、空格和导航按钮均能正常工作。
5. 确认浏览器控制台没有 JavaScript 错误。
