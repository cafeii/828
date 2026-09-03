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

Commit `78eef3486dc51af3657fe1cd59739c48b50af1a7` 将 chunk 内 reads 写成一个单位下三角 2×2 block 系统，用一次 batched `solve_triangular` 并行求出，然后恢复每个物理 token 的状态和输出。CPU/FP64 聚焦测试扩展到 46 passed。H800 作业 35580 又验证了三种更新顺序：3 passed，输出/末状态相对 RMSE `<2e-5`，所有输入梯度均有限且相对 RMSE `<1e-4`。该实现是 CUDA 数值 oracle，未做性能声明；下一步是用融合 WY/扫描 kernel 取代 PyTorch 求解与 Python chunk 循环。

Commit `a85f2a5329c8ece5d0897d7fd77fd5e5f2a2cd9e` 进一步把全部 chunk 的三角系统合并为一次批量求解，只让 compact chunk-end 状态转移保持顺序。CPU/FP64 新增 18 项、chunk/WY 聚焦集合 55 项通过。H800 作业 35593 的三种顺序 CUDA 门控均通过；在 FP32、B=2/T=128/H=4/K=V=64、chunk=16 的算子 forward+backward 上，相对逐 chunk 求解分别加速 `1.226x`、`1.915x`、`2.111x`。但是 peak allocated memory 从约 0.115 GB 增至约 0.169 GB，即 `1.46x`。这说明批量 WY 准备值得继续融合，同时也否决了把当前通用 PyTorch/autograd oracle 直接接入整模型；生产默认仍保持虚拟 2T。

Slurm 35600 对“把 effective-right 与 write-response 合成一次宽 RHS 三角求解”做了同配置 A/B。三种顺序相对双 solve 的速度比分别为 `0.769x`、`0.849x`、`0.942x`，peak allocated memory 均完全不变（`1.000x`）。该微优化已否决，双 solve 继续作为数值 oracle 默认；要降低 0.169 GB 的 oracle 峰值，必须用专用 WY forward/backward kernel 避免完整耦合矩阵与通用 autograd 保存，而不是调整 `solve_triangular` 的调用次数。

Commit `189100afe783d4d1fb701780d1f0b57c55ea4f0e` 固定了无 2C×2C system 的流式 WY 前代数，CPU/FP64 chunk/WY 聚焦集合扩展到 91 passed。Slurm 35622 的 triangular/streaming CUDA 门控 6 项全部通过，但 eager streaming 相对双 solve 只有 `0.376x`、`0.444x`、`0.412x` 速度，peak allocated memory 还增加到 `1.142x`。这否决了 Python token 循环和通用 autograd 版本，却保留了一个清晰的专用 kernel 方案：在单个 program 内保存 rank-2 history，并用重算式或手写 backward 避免保存整段中间图。

Commit `1b589cabfd6c4737bb87af62b199a64aa43023cc` 已实现单个 forward-only Triton WY kernel，不再向 PyTorch 暴露全局 `[B,H,N,2C,2C]` coupling system；提交前 CPU/FP64 rank-2 回归为 99 passed。首次 Slurm 35627 暴露了 chunk=8 的内部 `tl.dot` 归约维小于 16 的编译约束：chunk=16 三种顺序通过，chunk=8 三种顺序失败。Commit `d77f05e93b39506c4a7d7e9479b0e87ec9afb825` 用受 mask 约束的零填充修复后，Slurm 35628 为 `COMPLETED / 0:0`、`run.exitcode=0`、JUnit 6 passed / 0 failed；三种顺序、chunk 8/16 和非整 chunk padding 均通过，最大输出/末状态相对 RMSE 分别为 `4.62e-8` 和 `2.48e-8`，所有输出与状态有限。

35628 的 FP32 B=2/T=128/H=4/K=V=64/chunk=16 forward-only 算子对照中，Triton 相对 triangular 双 solve 的中位数速度比为 Recall→Delta `0.732x`、Delta→Recall `1.280x`、Parallel `1.220x`；peak allocated memory 比统一为 `0.987x`，incremental peak 比为 `0.982x`。Recall→Delta 样本在 2.48–6.66 ms 间明显抖动，当前结果不足以证明三种顺序都有稳定加速。该 kernel 尚无 backward 和有限梯度结果，也不是整模型 benchmark，因此不能启用物理 T 默认路径；下一步是手写/重算 backward 和更稳健的交错 A/B，再融合 chunk-state/output。

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
- 并行秩二 chunk 数值 oracle：Slurm 35580（实验 `20260903-212608-qgdn-rank2-chunk-cuda-67839c`）
- 全 chunk 批量 WY 数值与算子诊断：Slurm 35593（实验 `20260903-214702-qgdn-parallel-wy-cuda-6c110f`）
- 合并 WY RHS 候选否决：Slurm 35600（实验 `20260903-215810-qgdn-wy-fused-rhs-cuda-412446`）
- eager 流式 WY 候选否决：Slurm 35622（实验 `20260903-221056-qgdn-wy-streaming-cuda-f7dfff`）
- Triton WY 首次编译门禁：Slurm 35627（chunk=8 的 dot 维度约束失败；chunk=16 三种顺序通过）
- forward-only Triton WY CUDA 数值与算子诊断：Slurm 35628（6/6 门禁通过；速度信号混合）
