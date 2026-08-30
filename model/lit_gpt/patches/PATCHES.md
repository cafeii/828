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

## vendored fla 版本切换 0.6.0 → 6810828（2026-08-30）

- **对象**：服务器 `third_party/flash-linear-attention`（editable安装源码目录）
- **原因**：gdn2_ops 调用 `chunk_gla_fwd_o_gk(use_exp2=..., transpose_state_layout=...)`，
  这两个参数仅存在于 fla 主干 2026-03-10（#776）~ 2026-04-28 窗口
  （此后 #867 将 exp2 设为默认并删参、#905 改名 `state_v_first`）。
  原 vendored 0.6.0 与最新主干均不兼容。
- **动作**：服务器上切换为 fla-org/flash-linear-attention@6810828fad32316a7ba03f1d5915665e991e484b
  （窗口内最新，自带 USE_CUDA_GRAPH，报告版本号 0.5.1）。
  旧版本保留于 `third_party/flash-linear-attention-0.6.0.bak`。
  editable 的 .pth 指向路径不变，conda 环境零改动。
- **经用户批准**（2026-08-30）。kernels.py 的 USE_CUDA_GRAPH 注入shim保留（此版本下为no-op，防回退）。
