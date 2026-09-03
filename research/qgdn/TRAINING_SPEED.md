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

## 虚拟 2T 瓶颈与物理 T 审计

Slurm 35365 用同一张 H800、340M、序列 4096、micro batch 8、关闭 checkpoint 和 fused loss 隔离了 2T 成本：GDN 为 110,755 token/s，Recall 置零但仍走虚拟 2T 的 QGDN 只有 40,562 token/s（36.6%），峰值显存从 54.04 GB 升至 73.23 GB。这证明主瓶颈是通用 DPLR 的 2T 时间展开，而不是 Recall 数值本身。

Slurm 35377（commit `6c8b5a0789da6a419e3ed3512daff4ee0ede5c30`）审计了不构造虚拟行的串行物理 T Triton 原型。它为 `FAILED / 1:0`，JUnit 结果为 58 通过、3 失败：三种更新顺序的 forward 输出与末状态断言先通过，但 backward Triton 在写入 `grad_g` 时因 block/标量类型不匹配而编译失败。三次失败的单项耗时为 363.548、437.586 和 400.792 秒，还暴露了当前静态展开 backward 的编译资源问题。

由于梯度未执行，数值与有限性门槛未通过；整模型 benchmark 没有启动，所以没有物理 T 吞吐或峰值显存数据。`QGDN_USE_PHYSICAL_T` 继续为 `False`。后续应改为物理 T 的并行 chunk/WY 秩二扫描，不采用已在 Slurm 35367 触发 CUDA 越界的 TileLang 快路径。

Commit `17b2201ce73817f615dcd93b8722f92163926086` 已固定下一个 CUDA 实现的 CPU/FP64 合约。每个真实 token 使用一个秩二 `scale * I + U @ V.T` 转移和写入 bias，chunk 之间通过可结合的紧凑仿射组合衔接，全程不构造 K×K 稠密转移。三种更新顺序、query/key recall 和 chunk size 1/3/8 的输出、末状态与全输入梯度均已对齐逐 token FP64 参考；聚焦测试 28 passed。这只证明并行化代数合约正确，尚未产生 CUDA 吞吐或显存结果，所以默认路径不变。

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
- 虚拟 2T 成本与三种更新顺序：Slurm 35365
- TileLang DPLR 快路径 CUDA 越界：Slurm 35367
- 串行物理 T CUDA 门控失败：Slurm 35377（实验 `20260903-193528-qgdn-physical-t-audit-0b29a7`）
