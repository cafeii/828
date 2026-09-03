# QGDN 训练加速报告

## 推荐配置

QGDN 340M、序列长度 4096、8 张 H800 的推荐训练配置为：

- micro batch：8
- global batch：128
- activation checkpointing：关闭
- training loss：fused cross entropy
- DDP：`gradient_as_bucket_view=True`
- QGDN DPLR chunk size：32，并编译输入构造

对应命令行参数是：

```text
--micro-batch-size 8 --global-batch-size 128 \
--no-activation-checkpointing --training-loss fused
```

这套配置不改变 QGDN 的状态转移、门控、数据顺序、global batch 或优化器更新次数。

## 8 卡结果

| 配置 | 稳态吞吐 | 预计 10B 时间 | 预计 15B 时间 | 峰值显存 |
|---|---:|---:|---:|---:|
| 旧配置：micro batch 1、checkpoint、PyTorch CE | 121,366 token/s | 22.89 小时 | 34.33 小时 | 9.39 GB/GPU |
| 推荐配置 | 339,844 token/s | 8.17 小时 | 12.26 小时 | 77.07 GB/GPU |

推荐配置的吞吐是旧配置的 2.80 倍。8.17 小时是扣除首次 kernel 编译 step 后的纯训练投影；完整运行还会有启动、验证和保存开销，因此实际时间会略高。

显存已经接近 80 GB 上限。正式训练应使用独占整节点，并避免同时增加保存大张量的监控逻辑。

## 为什么有效

旧配置在每个 rank 上需要 16 次梯度累积，并且 activation checkpointing 会重复计算前向。micro batch 8 将每个 rank 的累积次数降到 2；关闭 checkpointing 用显存换计算；fused cross entropy 避免显式生成 FP32 全词表 loss 中间量。

fused loss 的数值检查结果：loss 与 PyTorch 实现完全一致；logits 梯度相对 RMSE 为 `3.33e-7`，最大绝对误差为 `3.73e-9`，所有值有限。

## 没有采用的改动

将 global batch 从 128 增加到 256 或 512，只把预计 10B 时间从 8.17 小时降到 8.12 或 8.09 小时，收益不到 1%，同时会改变优化轨迹，因此不采用。

此前关闭 DPLR backward recomputation 的收益不稳定，也不采用。

整模型 `torch.compile` 在单卡上快 7.40%，但 8 卡短基准的稳态窗口平均只快 3.65%：吞吐从 339,844 增至 352,265 token/s，10B 投影从 8.17 降至 7.89 小时。它还改变了 BF16 图内归并顺序：初始验证 loss 从 `10.383948684` 变为 `10.386828542`，12-step 验证 loss 从 `10.591185093` 变为 `10.600138903`。由于平均收益低于 5% 且不能与已有 eager 实验严格配对，正式训练不采用整模型编译。诊断脚本保留，供未来同时重跑所有对照时使用。

专用环境内旧 `torchrun` 文件残留了其他环境的 shebang。DDP 基准和后续作业必须使用当前 Python 启动：

```text
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 ...
```

## 复现实验

- 同一专用环境、同一节点的最终 8 卡三路对照：Slurm 35313
- 推荐配置、真实日志频率：Slurm 35292（旧 `torchrun` shebang，仅作交叉参考）
- global batch 128/256/512：Slurm 35295（旧 `torchrun` shebang，仅作批量比率诊断）
- 单卡 micro batch 扫描：Slurm 35261
- 单卡整模型编译对照：Slurm 35300
