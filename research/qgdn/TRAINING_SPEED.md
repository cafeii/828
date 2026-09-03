# QGDN 训练加速报告

## 结论

在不改变 QGDN 数学定义、门控初始化和输出语义的前提下，生产路径固定 DPLR chunk
size 为 32，并用 `torch.compile` 融合 2T 虚拟序列的输入构造。

340M、序列长度 4096、单卡 H800 的 50-step 隔离基准中，优化后的 QGDN 相对原路径
平均提速 `1.89%`，中位 step 时间改善 `2.88%`，峰值显存减少约 `0.050 GB`。
优化后吞吐为同场 GDN 的 `90.87%`。

## 基准结果

| 路径 | 吞吐 (token/s) | 平均 step | 中位 step | 峰值显存 |
|---|---:|---:|---:|---:|
| GDN 对照 | 17,761.3 | 230.61 ms | 226.09 ms | 6.685 GB |
| QGDN，chunk 32 | 15,840.4 | 258.58 ms | 254.82 ms | 6.844 GB |
| QGDN，chunk 32 + 编译输入 | 16,140.2 | 253.78 ms | 247.70 ms | 6.794 GB |

比较使用相同 340M 配置、序列长度 4096、micro batch 1、activation checkpointing 和
BF16 数值策略。每个变体在独立进程中运行，包含 5 个 warmup step 和 50 个测量 step。

## 正确性与并行训练

- 编译前后 6 个构造器输出一致。
- 所有输入梯度有限，最大相对 RMSE 为 `3.77e-6`。
- 完整模型最终 loss 与 eager chunk-32 路径相差 `5.46e-8`。
- CPU 定向测试 15 项、CUDA compiled/eager 前向与反向测试 2 项通过。
- 8 卡 DDP smoke 作业 35175 正常完成；稳态全局吞吐为 96,350.2 token/s，峰值显存
  9.383 GB，8 个 rank 均完成训练、保存和验证。

关闭 DPLR backward recomputation 的结果对测量顺序敏感，未稳定复现，因此生产配置
继续保留 recomputation。当前实现仍通过 2T 虚拟序列表达 Recall 和 Delta；进一步提速
需要物理 T 长度的 rank-2 融合 scan kernel，这属于独立的高风险内核项目。
