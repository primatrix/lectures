# 从 SGLang-JAX 看懂 SGLang-TorchTPU

面向已有 SGLang-JAX 经验的内部技术分享，基于 Outline「sglang-torchtpu share」下的三篇子文档，按接入、编译、并行与生态维护四个主题组织。当前共 24 页，重点展开 TorchTPU 的 API、调用链与实现特点。页面为离线单文件 HTML，无构建或 CDN 依赖。

## 本地使用

直接打开 `index.html`，或在仓库根目录运行 `python3 -m http.server 8080`，访问 `http://localhost:8080/sglang-torchtpu/`。

- `←` / `→`、`PageUp` / `PageDown`、空格：翻页。
- `Home` / `End`：首尾页；`O`：目录；`N`：讲者备注；`F`：全屏；`R`：阅读模式。
- 图解中的按钮控制接入方式、编译前端、权重 rank、MLP 阶段及启动步骤。
- 触屏横向滑动翻页；窄屏默认阅读模式。链接中的 `#s编号` 可以定位具体页。
- 浏览器打印可导出全部页面，自动隐藏操作控件和备注。

## 内容依据

资料整理与源码核查日期为 **2026-10-22**。三篇文档提供分享结构；原分享对照三个 `origin/main` 独立快照修正，本次分片 / 通信增补另对照本地源码快照。每页底部“源码”提供固定 commit 的证据。没有新增 TPU 性能实测。

1. [TPU 接入上游：fork、hook 与 platform](https://outline.infiscale-tech.com/doc/01-tpu-forkhook-platform-XZYq21rjcW)
2. [SGLang-TorchTPU 与 SGLang-JAX：复用边界、编译执行链路与维护责任](https://outline.infiscale-tech.com/doc/02-sglang-torchtpu-sglang-jax-PSyFJ5KzgS)
3. [多卡并行：从调度到设备同步](https://outline.infiscale-tech.com/doc/03-6ggcTQc0IF)

核查基线：SGLang-TorchTPU `44705a09565b`、SGLang-JAX `b3e3e16b2d3f`、TorchTPU `9041e2b81bcc`。三个 main 独立获取，不代表已联合运行验证。原仓库分支与工作树保持不变。交互图是机制示意；参数片段不是完整启动配方。

原 19 页版本的详细修正、逐页证据与检查结果见 [源码核查记录](CODE_REVIEW.md) 和 [静态检查汇总](audit-results.json)。本次并行章节补充核对 SGLang-TorchTPU `c32b417ac91f`；沿用的 JAX / runtime 依据仍固定到上面的快照，不代表联合部署验证。当前逐页证据以 HTML 内的“源码”为准。

## 模型代码与设备实现

第 8 页以 Qwen3 dense / Qwen2MLP 说明普通 PyTorch 模型代码、F.linear 和 checkpoint 参数映射的复用。第 9 页以 FP8 MoE 展开加载后转置、packing / scales 布局，以及匹配布局的 Pallas GMM / fused-RS 计算；模型内 QK norm / RoPE 融合与归约交接说明 forward 也可能需要适配。

第 19 页用两组 prefill 请求的 token 数 5、2，比较 TorchTPU 各组交换 metadata 后构建局部输入，与 JAX 统一 padding 后构建全局分片输入。bucket 8 / 16 为示例假设。该页替换通用的一轮执行流程。

本轮浏览器检查见 [模型与组批检查结果](modeling-checks.json)。

## 从 JAX 到 Torch 的并行认知

第 13–17 页以两卡 dense MLP 为线索，用 5 页说明从 JAX 全局数组到当前 Torch 局部 Tensor 路径的变化。正文聚焦分享主线，具体分支和实现条件放在备注与源码链接。

| 页码 | 内容 | 讲解重点 |
| --- | --- | --- |
| 13 | 全局数组 → 局部程序 | 并行层定义跨设备分片与通信，后端编译执行 |
| 14 | 按 rank 持有局部权重 | 每个 worker 绑定设备；用权重图展示 up / down 切片 |
| 15 | 局部计算与显式归约 | 完整 MLP 与 all-reduce 代码始终可见，交互切换 rank 状态 |
| 16 | 显式 Torch 集合通信 API | 原生 all_reduce / all_gather_into_tensor / reduce_scatter_tensor，与 SGLang 封装和 TPU 执行链对照 |
| 17 | 分布式 Pallas | kernel 内通信 API 相通，桥接处对齐布局和归约契约 |

设备绑定与权重分片合并展示；删除独立的正确性反例与归约开关页；Attention-DP 的 batch / bucket 协调要点并入第 18 页“提交与结果等待”。第 18 页聚焦 TorchTPU 的提交与结果等待，第 19 页比较两边的组批与输入组织。交互是机制示意，不模拟 TPU 性能。原并行章节浏览器检查见 [并行章节检查结果](parallel-walkthrough-checks.json)。

## HiCache / P/D 与生态维护

第 20–23 页介绍生态承接与维护边界，第 24 页对照代码组织。入口为 `index.html#s20`，沿用离线 HTML、导航、备注与逐页源码弹窗。

| 页码 | 内容 | 讲解重点 |
| --- | --- | --- |
| 20 | SGLang 生态在 TorchTPU 上的承接 | 公共主体、设备适配、当前未接通路径 |
| 21 | HiCache：公共缓存管理与 TPU 搬运 | Controller / HostPool 接口、GPU / Raiden 恢复链与完成等待 |
| 22 | P/D：公共交付接口与两条传输链路 | Sender / Receiver / KVPoll、push / pull 调用与共同完成流程 |
| 23 | 复用如何转化为长期维护能力 | 上游同步、接口收敛、TPU 适配与组合验证 |

内容依据为 `sglang-torchtpu/docs_myself/04-sglang-ecosystem-hicache-mooncake-reuse.md` 与 `05-pd-transfer-callchains-gpu-tpu.md` 两份报告。TPU 基线固定为 `c32b417ac91f`，报告中的上游对照为 `afd0d7b47b`；整理日期为 2026-10-23。公共层指 fork 中沿用的框架与实现，不表示与最新上游逐字一致。GPU / Raiden P/D 图标明主路径条件；Mooncake L3 的未接通结论限定于报告核实的 TPU fused-KV 路径。没有新增硬件或性能实测。

## 接入、成图与 KV 状态

第 5 页对照平台入口与通用插件入口，第 6 页以 AFTER hook 替换 worker 类为例，第 7 页展示当前接入状态。第 11 页展示通过配置选择的 Dynamo / direct 成图路径，并保留 JAX jit 对照；第 12 页用代码比较 JAX 返回 KV 状态与 TorchTPU mutation / alias 接口，direct 写回细节在备注中。

删除基础服务 / tile 流水页，`fused_moe_rs` 的具体实现保留在第 17 页及备注。后续页码、目录与源码索引同步调整。上一轮浏览器检查见 [接入与状态检查结果](feedback-checks.json)，当前版本见 [模型与组批检查结果](modeling-checks.json)。

## 设计与维护

搜索并阅读了 HTML slides 相关 skills，实际应用 [Anthropic frontend-design](https://github.com/anthropics/skills/blob/main/skills/frontend-design/SKILL.md)：以双实现对照和执行路径为视觉主体，JAX 用蓝色、TorchTPU 用橙色、公共设备层用青色。没有安装全局 skill 或引入第三方运行时。

每个 `section.slide` 包含 `data-title`、`data-chapter`、`data-source` 和 `data-note`；目录、页码、出处及备注由这些字段生成。主题变量位于 CSS 的 `:root`。新增页后无需手动修改总页数。

仓库首页和 GitHub Pages 工作流已接入 `/sglang-torchtpu/`；推送到 `main` 后由既有部署流程发布。
