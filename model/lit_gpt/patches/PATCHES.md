# 第三方代码运行时补丁记录

## fla.utils.USE_CUDA_GRAPH（2026-08-30）

- **位置**：`model/lit_gpt/kernels.py` 的 `_ensure_fla_compat()`
- **对象**：`third_party/flash-linear-attention`（fla 0.6.0，服务器editable安装）
- **原因**：`third_party/GatedDeltaNet-2/lit_gpt/gdn2_ops/chunk_kda.py` 依赖较新版 fla 的
  `fla.utils.USE_CUDA_GRAPH`（由环境变量 `FLA_USE_CUDA_GRAPH` 控制的triton autotune开关），
  fla 0.6.0 尚无此符号。
- **方式**：运行时注入（monkey-patch），不改动 fla 源码：缺失时按上游同语义
  `os.getenv("FLA_USE_CUDA_GRAPH", "0") == "1"` 注入默认值 False。
- **影响面**：仅 `use_cuda_graph` autotune 参数取默认 False，与上游默认一致，无行为差异。
- **移除条件**：fla 升级到含 `USE_CUDA_GRAPH` 的版本后可删。
