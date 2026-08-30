# 第三方代码运行时补丁记录

## kernel 来源切换：vendored gdn2_ops → fla 0.6.0 公开 API（2026-08-30，现行方案）

- **位置**：`model/lit_gpt/kernels.py`
- **决策**：GDN2/KDA/GDN 的 Triton kernel 统一取自 fla 0.6.0 公开 API
  （`fla.ops.gdn2.chunk_gdn2 / fused_recurrent_gdn2`、`fla.ops.kda.chunk_kda`、
  `fla.ops.gated_delta_rule.chunk_gated_delta_rule`），不再 import
  `third_party/GatedDeltaNet-2/lit_gpt/gdn2_ops`。
- **原因**：vendored gdn2_ops 调用 fla **内部函数**（`chunk_gla_fwd_o_gk` 等）且锚定
  主干未发布窗口 2026-03-10（#776）~ 2026-04-28（`use_exp2` 与 `transpose_state_layout`
  共存的唯一时段；#867 删前者、#905 改名后者），与 fla 0.6.0 及最新主干均不兼容。
  fla 0.6.0 已上游化完整 GDN2 kernel，公开签名与 vendored 版一致
  （仅 `transpose_state_layout` 改名 `state_v_first`，本项目不使用）。
- **影响面**："与 NVIDIA 论文快照 kernel 逐字节一致"不再成立，但公开语义（递归公式、
  门定义）相同，数值正确性由 `tests/test_kernel_parity.py` 三件套把关。
- **经用户批准**（2026-08-30）。

## 已撤销的历史操作（记录备查）

以下两项是 kernel 来源切换前的过渡操作，随切换一并撤销：

1. **fla.utils.USE_CUDA_GRAPH 运行时注入**：为 vendored gdn2_ops 在 fla 0.6.0 下
   缺符号做的 monkey-patch（原 `kernels.py` 的 `_ensure_fla_compat()`）。
   kernel 不再来自 gdn2_ops，注入已删除。
2. **服务器 vendored fla 切换 0.6.0 → 6810828**：为满足 gdn2_ops 的版本窗口，
   服务器 `third_party/flash-linear-attention` 曾被替换为主干 6810828（未提交）。
   已还原为仓库跟踪的 0.6.0；6810828 副本清理见服务器遗留清单。
