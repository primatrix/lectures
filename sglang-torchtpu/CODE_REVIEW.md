# HTML 内容源码核查记录

核查日期：2026-10-22。当前对象：`index.html` 的 **19 页**正文、交互、代码节选和讲者备注。面向熟悉 SGLang-JAX 的开发者，以实现与 API 为主，不做技术路线取舍。

## 本轮内容调整

- 并行章节聚焦运行时并行入口、归约交接、Attention-DP 编译契约与 Pallas 算子桥接。
- 删除 DTensor 词汇映射、局部求和教学、DP 名词比较、padding 滑块与设备分组切换。
- 删除“预热要覆盖真实调用的完整签名”“用解析后的 ParallelContext 驱动模型层”“CP 的使用方式落实到 backend 与 KV 接口”三页，不保留它们的正文、备注或页面源码索引。
- 末页只对照服务、模型、编译与 kernel 的代码组织；原支持范围、总结和资料页已删除，来源从每页底部查看。

## DTensor 使用范围的复核

当前 TPU 常用模型推理路径以 rank-local `torch.Tensor`、并行层权重切分、显式 group 通信与 layer boundary 为主；没有证据支持“DTensor 已被广泛使用”。这不等于仓库中没有 DTensor。

- [模型 opt-in 后才应用 PyTorch TP](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/model_executor/model_runner.py#L798)：`python/sglang/srt/model_executor/model_runner.py:798`。
- [TorchNativeLlama 的 opt-in](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/models/torch_native_llama.py#L399)：`python/sglang/srt/models/torch_native_llama.py:399`。
- [DeviceMesh / parallelize_module 辅助实现](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/layers/model_parallel.py#L161)：`python/sglang/srt/layers/model_parallel.py:161`。
- [权重同步中的 DTensor 转换](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/weight_sync/utils.py#L108)：`python/sglang/srt/weight_sync/utils.py:108`。

`ModelRunner.maybe_apply_post_load_model_transforms` 仅在 `tp_size > 1` 且模型声明 `supports_torch_tp` 时调用 `apply_torch_tp`。在本次快照的 `python/sglang/srt` 中，声明 `supports_torch_tp=True` 的模型只有 `TorchNativeLlamaForCausalLM`，其 helper 使用 `DeviceMesh` / `parallelize_module`。`weight_sync/utils.py` 则为权重更新将 DTensor 转为 `full_tensor()`；这是权重输入适配，不代表 forward 使用 DTensor。`torch_native` attention backend 和 `TorchNativeLlama` 模型是不同入口。runtime 的 DTensor 测试不作为 SGLang 主推理路径采用它的证据。

## 本轮 API 核查重点

- `RowParallelLinear` 的归约包含 SP、decode 与 skip 条件；不将 `reduce_results=False` 等同于不会通信。
- Qwen3-MoE 的 `summed_by_compute` 来自 `handles_moe_reduction`，边界跳过已完成的 sum，保留 move 与 write-back。
- Attention-DP 分开保存有效长度与 padded communication metadata，dummy batch 与实际请求需要一致的布局和 flags。
- fused-RS 的 `jax_op` 注册保留 Torch-local shape；Pallas mesh 设备映射不等于模型改用 DTensor。

## 源码范围与 Git 状态

用户提到的目录实际对应 `../sglang-torchtpu`、`../sgl-jax`；另核查底层 `../torch_tpu`。`../sglang-jax` 和 `../sglang-torchpu` 不存在。

| 项目 | 原工作树 HEAD | 本次获取的 origin/main | 提交日期 | 相对原 HEAD 新增提交 |
| --- | --- | --- | --- | --- |
| SGLang-TorchTPU | `c32b417ac91f` | `44705a09565b36ddf4ab6ac7244847ec5887c42c` | 2026-10-21 | 99 |
| SGLang-JAX | `e496419160e3` | `b3e3e16b2d3f0281a9e23f547cd4382f4a166eb6` | 2026-10-20 | 63 |
| TorchTPU runtime | `dd6e5886faa4` | `9041e2b81bcc2385edc668e6e563fa09f1eca8b0` | 2026-09-28 | 974 |

各仓库均有未跟踪的本地文件，因此使用 `git fetch origin main` 后 `git archive origin/main` 导出的独立快照，没有 merge/rebase、切换分支、清理文件或更新原工作树。所有证据链接固定到上表 commit。

这三个 main **不是已联合验证的部署组合**。SGLang-TorchTPU 的 Python wheel 依赖为 `torch_tpu==0.2.0`，Bazel 中的 runtime 源码 pin 为 `caab3f95fb657540be716c81136322f8de45c056`；不把独立获取的 runtime main 等同于这些 pin。见 [wheel 依赖 pin 不等于 runtime main](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/pyproject_tpu.toml#L78)、[Bazel runtime 源码 pin](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/MODULE.bazel#L19)。

## 前一轮核查后保留的实现修正

### 1. JAX main 的 AOT 核查记录（在线编译入口见第 9–10 页）

以下保留源码核查结果；预热与并行编译的独立讲解页已删除。

- `--save-aot`：在 CPU host 针对指定 TPU 拓扑导出；有模型、投机、LoRA、多模态等适用限制。
- `--aot-model-dir`：从 executable store 严格加载；缺失签名不会静默回退编译。
- `SGLANG_JAX_AOT_DISPATCH`：仍默认 `0`；控制可选 `execute_sharded` 快速分发，与离线加载分开。
- `precompile_num_threads` 声明默认 `2`，适用配置可以先并行编译再执行预热；不适用时走串行。

常规 NNX split/merge、jit、donation 和 `replace_all` 描述仍成立。没有把 AOT 入口存在解释成所有模型、布局都能离线导出。

来源：[严格加载 / 编译 / 导出](https://github.com/sgl-project/sglang-jax/blob/b3e3e16b2d3f0281a9e23f547cd4382f4a166eb6/python/sgl_jax/srt/model_executor/compilation_manager.py#L406)、[并行编译适用条件](https://github.com/sgl-project/sglang-jax/blob/b3e3e16b2d3f0281a9e23f547cd4382f4a166eb6/python/sgl_jax/srt/model_executor/model_runner.py#L354)、[离线导出限制](https://github.com/sgl-project/sglang-jax/blob/b3e3e16b2d3f0281a9e23f547cd4382f4a166eb6/python/sgl_jax/srt/model_executor/aot_server.py#L30)、[fast dispatch 默认开关](https://github.com/sgl-project/sglang-jax/blob/b3e3e16b2d3f0281a9e23f547cd4382f4a166eb6/python/sgl_jax/srt/model_executor/aot_dispatch.py#L47)。

### 2. 纠正 entry-point 配置示例并补全双入口（第 5 页）

TOML 示例由 `[entry-points."..."]` 修正为实际的 `[project.entry-points."..."]`。补充 fork 中的 `tpu_hooks`：显式选择 `SGLANG_PLATFORM=tpu` 会排除 `sglang-tpu` 的通用插件，Mode A 经 fork 入口加载；Mode B 经插件入口加载。两入口调用同一个带去重与 TPU 平台检查的注册函数。

来源：[插件的两个 entry-point 组](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/pyproject.toml#L22)、[Mode A 的 fork 入口](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/pyproject.toml#L220)、[插件排除逻辑](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/plugins/__init__.py#L89)、[注册去重与平台门控](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/sglang_tpu/hooks/__init__.py#L262)。

### 3. 明确 wait=False 的含义（第 16 页）

页面中的 `tpu_sync` 是源码对 `torch_tpu._internal.sync.sync.synchronize` 的别名。新 main 的 C++ 路径明确等待 execution started / `is_materializing`，只是没有等待最终结果就绪；不能把 `wait=False` 理解成完全不阻塞 host。结果消费仍须等 copy event，且该事件还会等待复制线程提交并传播错误。

来源：[wait=False 等待启动](https://gitlab.infiscale-tech.com/external/torch_tpu/-/blob/9041e2b81bcc2385edc668e6e563fa09f1eca8b0/csrc/internal/sync/sync_py.cc#L114)、[copy 事件与提交等待](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/sglang_tpu/result_copy_worker.py#L12)。

## 静态工具记录（保留在核查附件，已从分享正文删除）

| 指标 | 原文基线 | 本次 main 实跑 |
| --- | --- | --- |
| core is_tpu 引用 / 文件数 | 395 / 85 | 395 / 85 |
| hooks | 19 | 35 |
| runtime patches | 18 | 19 |
| 被 patch 的 core 属性 | 31 | 36 |
| TPU 测试注册项 | 221 | 230 |
| 注册元数据有问题的项 | 10 | 8 |
| 解析失败 | 0 | 0 |

`support_matrix.py --strict` 退出码为 **2**，问题是词表外标签或缺少 model/topology 标签；没有运行对应 TPU 测试，因此不能称作“8 个 TPU 测试失败”。`check_tpu_branches.py` 退出码 **0**，并提示若干 baseline 可下调；没有修改 baseline。

来源：[静态计数工具](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/tools/plugin/scoreboard.py#L33)、[支持矩阵检查工具](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/scripts/ci/tpu/support_matrix.py#L26)。完整汇总保存在 `audit-results.json`。

## 当前逐页源码索引

| 页 | 内容 | 主要依据 |
| --- | --- | --- |
| 1 | 从 JAX 出发 | [Torch 编译入口](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/sglang_tpu/compilation/decoder_compile_policy.py#L785) |
| 2 | 三个问题贯穿分享 | [插件和 fork 安装关系](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/sglang_tpu/README.md#L7) |
| 3 | 两条栈的复用边界 | [公共 PyTorch 模型](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/models/qwen3.py#L174) |
| 4 | Fork、hook、platform | [Hook 包装语义](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/plugins/hook_registry.py#L350) |
| 5 | Plugin 负责发现与加载 | [插件的两个 entry-point 组](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/pyproject.toml#L22) |
| 6 | Hook 的实际语义 | [BEFORE / AFTER / AROUND / REPLACE](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/plugins/hook_registry.py#L350) |
| 7 | 当前状态与收敛方向 | [仍需 fork + 插件](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/sglang_tpu/README.md#L7) |
| 8 | Modeling 复用到哪里 | [Qwen3 共用模型流程](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/models/qwen3.py#L174) |
| 9 | 从启动到第一轮请求 | [TPU worker 初始化](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/utils/patch_torch.py#L195) |
| 10 | 三条编译前端 | [Dynamo 与 direct 选择](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/sglang_tpu/compilation/decoder_compile_policy.py#L785) |
| 11 | KV 状态如何跨编译边界 | [JAX 状态拆分](https://github.com/sgl-project/sglang-jax/blob/b3e3e16b2d3f0281a9e23f547cd4382f4a166eb6/python/sgl_jax/srt/model_executor/model_runner.py#L275) |
| 12 | TorchTPU 的并行入口 | [模型 opt-in 后才应用 PyTorch TP](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/model_executor/model_runner.py#L798) |
| 13 | 归约由谁完成 | [Qwen3-MoE 归约交接](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/models/qwen3_moe.py#L967) |
| 14 | Attention-DP 的编译契约 | [CPU 元数据对齐](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/managers/scheduler_components/dp_attn.py#L531) |
| 15 | Pallas 分布式算子接入 | [实际 jax_op 注册与局部 shape 契约](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/sglang_tpu/moe/pallas/fused_moe_rs_op.py#L181) |
| 16 | 图内依赖与图外同步 | [CPU / device 建组](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/distributed/parallel_state.py#L381) |
| 17 | Overlap 与 kernel 内流水 | [独立结果复制线程](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/sglang_tpu/result_copy_worker.py#L45) |
| 18 | 串起一轮执行 | [请求接收与广播](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/managers/scheduler_components/request_receiver.py#L112) |
| 19 | 两条实现的代码组织差异 | [公共 PyTorch 模型](https://github.com/sgl-project/sglang-torchtpu/blob/44705a09565b36ddf4ab6ac7244847ec5887c42c/python/sglang/srt/models/qwen3.py#L174) |

## 验证范围

页面每页均有固定 commit 的源码链接，共 63 个；已核查本地快照的文件和行号。浏览器回归覆盖七种桌面 / 移动尺寸、页面导航、剩余交互、源码弹窗、备注、阅读与打印模式；详细结果见 `audit-results.json`。

没有运行 TPU 推理、精度、性能或跨 main 版本兼容测试。静态源码中的入口与测试注册不能作为设备测试已通过的证据。

文案检查还采用了 TorchTPU 仓库的 [docs-review skill](https://gitlab.infiscale-tech.com/external/torch_tpu/-/blob/9041e2b81bcc2385edc668e6e563fa09f1eca8b0/.agents/skills/docs-review/SKILL.md)，应用术语与条件说明规则，保持简体中文。
