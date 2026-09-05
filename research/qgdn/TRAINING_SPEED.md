# QGDN 训练加速报告

> 2026-09-04 物理 T 已在独立短诊断路径上恢复优化，完整状态、失败路线和启用门槛见
> [PHYSICAL_T_DEFERRED.md](PHYSICAL_T_DEFERRED.md)。已冻结正式训练和生产默认继续使用虚拟 2T，
> `QGDN_USE_PHYSICAL_T=False`。

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

## 10BT 正式训练

2026-09-04 已将推荐配置用于两条同初始化正式训练：Parallel 为 Slurm 36311，Delta→Recall
为 Slurm 36312，冻结代码均为 `7eb73ca89411c54d4fe7a8ffb427df44e7709cfa`。两者使用 seed
3407、`max_steps=19073`（9,999,745,024 prediction tokens）、每 2000 step 验证 1600
sequences、每 1000 step 保存 checkpoint。gamma 与 beta 同方案独立 Xavier 随机初始化，物理 T
保持关闭。

由于提交时没有空闲整节点，没有单独排队一个 8 卡 smoke；相同的初始化与三顺序 CUDA
前向/末状态/反向测试被放在每个正式 allocation 的入口，JUnit 门禁失败会直接阻止训练启动。
初始排队状态分别是 36311 `PENDING (Resources)`、36312 `PENDING (Priority)`。按既有 8 卡
实测，纯训练投影约 8.17 小时；计入首次 kernel 编译、验证和 checkpoint，预计单作业约
8.5–9.2 小时，实际起始时间取决于整节点排队。

首次训练后 validation（step 2000、1,048,576,000 tokens）中，Parallel 的 loss/PPL 为
`3.142662 / 23.16546`，Delta→Recall 为 `3.143802 / 23.19188`。共同 step 2671 的最近
20 点训练 loss 均值也基本相同（`3.049501` vs `3.050772`）。但 gamma 分布明显不同：
Parallel mean/std/饱和率为 `0.44296 / 0.31147 / 8.05%`，Delta→Recall 为
`0.22305 / 0.21491 / 1.57%`。当前所有记录值和梯度均有限，且两者峰值显存均为
`77.0617 GB/GPU`；该差异作为更新顺序行为信号继续跟踪，暂不触发配置调整。

step 4000 的第二次对齐 validation 仍显示两者基本持平：Parallel loss/PPL 为
`2.975204 / 19.59361`，Delta→Recall 为 `2.976156 / 19.61229`。共同 step 4551 的近 20 点
训练 loss 为 `2.93801 / 2.93946`。Parallel 与 Delta→Recall 的 gamma mean/std/饱和率分别
为 `0.42114 / 0.30414 / 6.08%` 和 `0.20319 / 0.20486 / 1.17%`，饱和率均下降。
Delta→Recall 在 dgx01 遭遇 CPU/I/O 共址争用后出现吞吐波动，40 点中位约 `244.8k
token/s`，但最新 step 已恢复到 `295.6k token/s`；数值、显存和 checkpoint 均正常，当前不重启。

step 6000 的第三次对齐 validation 为 Parallel `2.893768 / 18.06124`、Delta→Recall
`2.895291 / 18.08877`。共同 step 6691 的近 20 点训练 loss 为 `2.85331 / 2.85491`；
beta mean/std 分别为 `0.28484 / 0.16481` 与 `0.27335 / 0.16485`，gamma
mean/std/饱和率分别为 `0.40765 / 0.30031 / 5.19%` 与
`0.19141 / 0.19779 / 0.91%`。Delta→Recall 最近 40 点吞吐中位已恢复至 `297.6k
token/s`，但均值仍为 `270.6k`，说明节点争用长尾尚未完全消失。

17:41 dgx01 的 CPU/I/O 共址争用再次恶化，Delta→Recall 最近 40 点吞吐中位/均值降为
`188.3k / 191.9k token/s`；同期 Parallel 为 `306.4k token/s`。共同 step 7501 的近
20 点 loss 为 `2.82758 / 2.82857`，beta mean/std 为 `0.28673 / 0.16486` 与
`0.27436 / 0.16497`，gamma mean/std/饱和率为 `0.40637 / 0.30111 / 5.18%` 与
`0.18623 / 0.19535 / 0.84%`。数值和 checkpoint 均健康；按受扰吞吐仍可在时限内完成，
故不重启或改变科学配置。

step 8000 的第四次共同 validation 仍基本持平：Parallel `2.839141 / 17.10108`，
Delta→Recall `2.839986 / 17.11553`，Parallel PPL 仅低约 `0.085%`。共同 step 8411
的近 20 点 loss 为 `2.79759 / 2.79914`，beta mean/std 为
`0.28944 / 0.16635` 与 `0.27827 / 0.16689`，gamma mean/std/饱和率为
`0.40398 / 0.29998 / 5.04%` 与 `0.18398 / 0.19339 / 0.78%`。Delta→Recall 最近
40 点吞吐中位已回升至 `297.1k token/s`（均值 `271.5k`），无需干预。

为排除历史 GDN 的 micro-batch、checkpointing、loss kernel 和验证集条数差异，已提交同一
冻结 commit 和当前训练配方的 GDN control：实验
`20260904-190833-gdn-aligned-340m-10bt-s3407-1b99be`、Slurm `37118`。其 mbs8/GA2、
fused loss、无 activation checkpointing、seed 3407、1600-sequence validation、10BT
目标均与 Parallel 和 Delta→Recall 一致；当前为 `PENDING (Priority)`，后续只在共同
step/token 与 validation 节点比较三者。

Recall→Delta 的严格对齐训练也已提交：实验
`20260904-194217-qgdn-recall-delta-340m-10bt-s3407-11ef45`、Slurm `37183`，冻结 commit
和 mbs8/GA2、fused loss、无 activation checkpointing、seed 3407、1600-sequence
validation、10BT 配方均与另外三路一致，gamma 为当前 beta-style 独立 Xavier 随机初始化。
当前 `PENDING (Priority)`；后续报告将以 GDN 加三种更新顺序组成四路共同节点比较。

GDN control 已在 dgx25 开始训练，preflight JUnit `6/6`。截至共同 step 1391，近 20 点
loss 为 GDN `3.25359`、Parallel `3.25499`、Delta→Recall `3.25668`；GDN 最近 40 点
吞吐中位 `866.0k token/s`、峰值显存 `56.82 GB`，另外两路约 `306.4k/297.8k
token/s`、`77.06 GB`。GDN alpha/beta mean/std 为 `0.69946 / 0.33857`、
`0.28540 / 0.16762`（gamma 不适用）；Parallel beta 与 gamma mean/std/饱和率为
`0.28319 / 0.16566`、`0.46390 / 0.31074 / 8.90%`；Delta→Recall 为
`0.26978 / 0.16564`、`0.25526 / 0.22255 / 1.48%`。该训练窗口尚不能替代共同 validation。

Recall→Delta 也已在 dgx12 启动并通过 6/6 preflight。共同 step 1831 的近 20 点 loss 为
Recall→Delta `3.15765`、GDN `3.15855`、Parallel `3.16001`、Delta→Recall `3.16071`，
四路最大差 `0.00305`。此时 Recall→Delta beta mean/std 与 gamma
mean/std/饱和率为 `0.27871 / 0.16365`、`0.45872 / 0.31359 / 9.27%`；Parallel 为
`0.27695 / 0.16287`、`0.45344 / 0.31355 / 8.90%`；Delta→Recall 为
`0.26491 / 0.16301`、`0.23995 / 0.22010 / 1.63%`。GDN alpha/beta mean/std 为
`0.69561 / 0.34028`、`0.27701 / 0.16435`，gamma 不适用。

首次四路共同 step-2000 validation 已完成：Recall→Delta、GDN、Parallel、
Delta→Recall 的 loss/PPL 依次为 `3.140643 / 23.11873`、`3.141821 / 23.14598`、
`3.142662 / 23.16546`、`3.143802 / 23.19188`。Recall→Delta 当前相对 GDN 低
`0.001178` loss 和约 `0.118%` PPL；幅度很小且只有一个共同验证点，继续等 step 4000
及以后节点再判断。

在已经覆盖到 step 10000 的三路共同 validation，GDN、Parallel、Delta→Recall 的
loss/PPL 分别为 `2.797239 / 16.39930`、`2.796521 / 16.38754`、
`2.798008 / 16.41192`。Parallel 相对 GDN 只低 `0.000718` loss，Delta→Recall 只高
`0.000769`，当前应视为基本持平。最近 40 点吞吐中位为 GDN `866.8k`、Recall→Delta
`310.0k`、Parallel `306.5k`、Delta→Recall `299.5k token/s`；峰值显存为
`56.82 / 77.03 / 77.06 / 77.06 GB`。四路数值和 checkpoint 均正常，Recall→Delta 的
step-4000 validation 完成后再进行第二次四路验证比较。

Parallel Slurm 36311 已成功完成全部 `19073` step 和 `9,999,745,024` prediction tokens：
Slurm `COMPLETED / 0:0`、`run.exitcode=0`、summary `completed`、preflight JUnit `6/6`。
训练计算/墙钟为 `8.998/9.121 h`，有效训练/墙钟吞吐为 `308.69k/304.53k token/s`，
峰值显存 `77.0617 GB/GPU`；最终 validation loss/PPL 为 `2.696578 / 14.82890`。末步
beta mean/std 为 `0.29295 / 0.17039`，gamma mean/std/饱和率为
`0.38905 / 0.29873 / 4.55%`。小型日志与指标已完整回收，权重按规则保留在远端。

Delta→Recall Slurm 36312 也已成功完成并回收：Slurm `COMPLETED / 0:0`、
`run.exitcode=0`、summary `completed`、preflight JUnit `6/6`，终点同为 step `19073` 和
`9,999,745,024` tokens。训练计算/墙钟为 `11.139/11.379 h`，有效训练/墙钟吞吐为
`249.36k/244.11k token/s`，峰值显存 `77.0617 GB/GPU`。最终 validation loss/PPL 为
`2.698694 / 14.86030`；相对 GDN 高 `0.134%` PPL，相对 Parallel 高 `0.212%` PPL。
末步 beta mean/std 为 `0.27745 / 0.16940`，gamma mean/std/饱和率为
`0.16184 / 0.18106 / 0.481%`，全部数值有限。其运行时吞吐受 dgx01 共址 I/O/CPU 争用
拖累，因此不把这组墙钟差解释为更新顺序本身的纯算子差异。

Recall→Delta 到达 step 8000 后，四路 validation loss/PPL 为 Recall→Delta
`2.837916 / 17.08014`、GDN `2.839041 / 17.09937`、Parallel
`2.839141 / 17.10108`、Delta→Recall `2.839986 / 17.11553`。Recall→Delta 相对
GDN 的 PPL 低约 `0.112%`；它在四个共同验证节点方向一致地领先，但量级仍只有千分之一，
继续等待完整训练终点。

step-10000 时 Recall→Delta 的 loss/PPL 为 `2.795812 / 16.37592`，相对 GDN
`2.797239 / 16.39930` 低 `0.001427` loss、`0.143%` PPL，也小幅优于 Parallel
`16.38754` 和 Delta→Recall `16.41192`。五个共同验证点方向一致，但收益仍只有约千分之一。

step-12000 时 Recall→Delta 的 loss/PPL 为 `2.761304 / 15.82045`，相对 GDN
`2.762260 / 15.83558` 低 `0.000956` loss、`0.096%` PPL；Parallel 与 Delta→Recall
的 PPL 为 `15.83187 / 15.85808`。领先方向仍一致，但较 step 10000 收窄，继续按小信号
处理。作业现至 step `12661`，近 20 点 loss `2.72435`，beta mean/std
`0.29498 / 0.17073`，gamma mean/std/饱和率 `0.40377 / 0.30137 / 4.79%`，稳态吞吐
约 `310.1k token/s`。

step-14000 的 Recall→Delta loss/PPL 为 `2.733998 / 15.39431`，相对 GDN
`2.735018 / 15.41002` 低 `0.001020` loss、`0.102%` PPL；Parallel 与 Delta→Recall
的 PPL 为 `15.40296 / 15.43278`。作业现至 step `14111`，近 20 点 loss
`2.70426`，beta mean/std `0.29606 / 0.17186`，gamma mean/std/饱和率
`0.40063 / 0.30156 / 4.76%`，稳态吞吐约 `310.2k token/s`。七个共同节点方向一致，
但收益仍是约千分之一的小信号。

step-16000 的 Recall→Delta loss/PPL 为 `2.713727 / 15.08540`，相对 GDN
`2.714649 / 15.09932` 低 `0.000922` loss、`0.092%` PPL，相对 Parallel 低
`0.033%` PPL。作业至 step `16661` 的 beta mean/std 为 `0.29524 / 0.17125`，gamma
mean/std/饱和率为 `0.39832 / 0.30204 / 4.81%`，近 20 点 loss `2.67928`、稳态约
`310.9k token/s`。八个节点虽同向，但幅度继续是小于千分之一的信号。

Recall→Delta Slurm 37183 已成功完成并回收：step `19073`、
`9,999,745,024` tokens，Slurm/`run.exitcode` 为 `COMPLETED 0:0 / 0`，JUnit `6/6`。
训练计算/墙钟为 `8.893 / 9.019 h`，对应 `312.33k / 307.99k token/s`，峰值显存
`77.0250 GB/GPU`。最终 loss/PPL `2.696375 / 14.82588`，比 GDN、Parallel、
Delta→Recall 分别低 `0.098% / 0.020% / 0.232%` PPL。末步 beta mean/std
`0.29414 / 0.17227`，gamma mean/std/饱和率 `0.39617 / 0.30182 / 4.76%`，全部
有限。四路严格对齐的最终排名为 Recall→Delta、Parallel、GDN、Delta→Recall，
但第一名对 GDN 的改善仍只有约千分之一，不是明确的大收益。

四路 step-4000 validation 的 loss/PPL 为 Recall→Delta `2.972738 / 19.54537`、GDN
`2.974415 / 19.57818`、Parallel `2.975204 / 19.59361`、Delta→Recall
`2.976156 / 19.61229`。Recall→Delta 连续第二个共同节点领先 GDN，本次低 `0.001677`
loss 和约 `0.168%` PPL；信号一致但幅度仍小。

共同 step 4301 的近 20 点 loss 为 Recall→Delta `2.93770`、GDN `2.93884`、Parallel
`2.93941`、Delta→Recall `2.94062`。Recall→Delta beta 与 gamma mean/std/饱和率为
`0.27915 / 0.16282`、`0.42653 / 0.30771 / 6.77%`；Parallel 为
`0.27715 / 0.16134`、`0.42266 / 0.30638 / 6.50%`；Delta→Recall 为
`0.26539 / 0.16079`、`0.20436 / 0.20538 / 1.20%`。GDN alpha/beta mean/std 为
`0.68817 / 0.34125`、`0.27588 / 0.16160`，gamma 不适用。其余三路仍正常运行。

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

Commit `221892c59d07bfc55fcf5a5206eae9dffe2864c2` 增加了不保存 forward token 中间图的手写重算 backward。CPU/FP64 直接梯度对照覆盖 chunk 1/3/8，完整 rank-2 聚焦集合为 102 passed。Slurm 35642 为 `COMPLETED / 0:0`、`run.exitcode=0`、JUnit 6 passed / 0 failed；三种顺序、chunk 8/16 和尾部 padding 的输出、末状态以及 q/k/v/g/beta/gamma/初态全部梯度有限，最大输出/状态/梯度相对 RMSE 为 `4.62e-8`、`2.48e-8`、`7.98e-7`。

这一步证明了 backward 公式和重算边界，但没有得到可用训练速度：同一 FP32 B=2/T=128/H=4/K=V=64/chunk=16 算子 forward+backward 中，重算路径约 48.1–48.4 ms，triangular oracle 为 8.75–13.0 ms，速度比分别为 `0.270x`、`0.182x`、`0.198x`。peak allocated memory 仅降至 `0.975x`，incremental peak 为 `0.957x`。瓶颈是 Python token 循环与大量 einsum launch；该实现只保留为下一版专用 Triton backward 的数值 oracle，不接入训练，也不改变 `QGDN_USE_PHYSICAL_T=False`。

Commit `f09b36e81f59d73d5767173b48a20bb5e40f4d0c` 又将伴随求解和四组 WY 因子梯度融合到单个 Triton backward program。CPU/FP64 dense-adjoint、逐 token 手写反向和 PyTorch autograd 三方一致，rank-2 聚焦集合为 102 passed。Slurm 35649 为 `COMPLETED / 0:0`、`run.exitcode=0`、JUnit 6 passed / 0 failed；覆盖三种顺序、chunk 8/16、尾部 padding 与全部输入梯度，最大输出/状态/梯度相对 RMSE 为 `4.62e-8`、`2.48e-8`、`7.71e-7`，所有值有限。

融合后同配置 forward+backward 为 8.78、6.37、6.10 ms，相对 triangular 的速度比分别为 Recall→Delta `0.798x`、Delta→Recall `1.068x`、Parallel `1.071x`；peak allocated memory 比为 `0.975x`，incremental peak 比为 `0.957x`。这比 Python 重算的约 48 ms 大幅改善，但第一种顺序仍存在 6.23–10.24 ms 的明显波动，不能据此声称三种顺序稳定加速。下一步先做顺序轮换的交错 A/B，再决定是否值得融合 chunk-state/output；整模型门禁与 `QGDN_USE_PHYSICAL_T=False` 均保持不变。

Commit `7c7d237bbecf3f0832ec3b403d28c0f84946ab75` 将 fused-WY 测量改为顺序轮换、backend 交错、固定输出梯度和每样本 8 次 forward+backward。Slurm 35893 的 JUnit 为 6/6，通过三种顺序的数值与有限梯度门禁。50 组样本的配对中位速度比均约 `1.058x`，bootstrap 中位数 95% 下界分别为 `1.032`、`1.047`、`1.037`；peak allocated 比约 `0.975x`。这确认了 WY kernel 的小幅稳定收益，也说明更大的机会在剩余 chunk-state/output 部分。

Commit `55d66c80b9a8438d55ff8efc2e37d6550b26c3f3` 将块级状态扫描与块内输出恢复改写为专用 Triton forward/backward。块级状态只跨 chunk 递推；输出对所有 chunk 并行计算，不物化 token 级 K×V 状态。prepared-chunk 的 CPU/FP64 密集合约在 chunk 1/3/8 上验证了查询、decay、左右因子、write-read、写入向量、value 和初态全部梯度，原 rank-2 聚焦集继续 102 passed。

Slurm 35894 首先通过 6/6 CUDA 门禁并给出约 `2.21x` 信号，但其三模型顺序与三后端轮换发生周期锁相，因此只保留数值证据。Commit `22a3b27fd0c7f286320199d18edea793e229f8a2` 修复编排并强制每个模型顺序覆盖三种后端排列；Slurm 35895 再次 `COMPLETED / 0:0`、`run.exitcode=0`、JUnit 6 passed / 0 failed。

35895 的 FP32 B=2/T=128/H=4/K=V=64/chunk=16、50 组×8 次 forward+backward 结果为：Recall→Delta、Delta→Recall、Parallel 相对 triangular 的配对中位加速分别为 `2.199x`、`2.212x`、`2.239x`，配对 p10 为 `1.987x`、`2.004x`、`2.012x`，150 个配对全部大于 1。全融合中位时延为 4.143、4.103、4.023 ms，而 triangular 为 9.153、9.106、9.093 ms。peak allocated memory 比约 `0.522x`，incremental peak 比为 `0.111–0.114x`。

数值方面，最大输出、末状态和七组模型输入梯度相对 RMSE 为 `1.80e-7`、`6.23e-8`、`1.84e-6`，全部有限。结果文件与 JUnit 已按 SHA-256 与远端一致回收；回收脚本仍因 manifest 的 `outputs/` 双拼路径返回 1，但实际声明产物都存在。

这仍是 diagnostic 物理-T 算子对 triangular 物理-T oracle 的结果，不是 340M 整模型相对虚拟 2T 的加速。下一步是接入显式 opt-in 训练分支、验证 BF16 与实际形状，再做同卡整模型 A/B；在达到稳定 `>1.25x` 且显存不恶化之前，`QGDN_USE_PHYSICAL_T=False` 不变。

Commit `1335ee690822ab72c736c94c2b6c49363d604e95` 已将全融合 chunk-16 rank-2/WY 路径接入显式 opt-in 训练分支，同时保持默认关闭。Slurm 35896（实验 `20260904-094915-physical-chunk-training-audit-2ce53c`）为 `COMPLETED / 0:0`、`run.exitcode=0`，JUnit 6/6；实际 340M 算子形状 `B=1/T=4096/H=16/K=V=64` 的三顺序 BF16 检查中，输出相对 RMSE 为 `0.00438–0.00458`，末状态为 `0.00256–0.00262`，七组输入梯度最坏为 `0.00881`，全部有限。物理路径 peak allocated 是生产虚拟 2T 的 `0.843x–0.871x`。

因此训练入口与 CUDA 数值门禁已通过，下一步可以进行 micro batch 8、序列 4096、关闭 checkpoint、fused loss 的单卡整模型同配置 A/B。这里的 `0.843x–0.871x` 仍是算子审计的显存比例，不是整模型结果；在整模型三顺序都稳定超过 `1.25x` 且显存不恶化前，`QGDN_USE_PHYSICAL_T=False` 不变。

## 整模型容量修复与 micro batch 4 结论

Slurm 35911 在干净 H800 上给出了 micro batch 8 的可用虚拟 2T 基线：`40,121.94 token/s`、`74.58 GB`、中位 step `0.8169 s`。同配置 no-recompute 物理 T 在后续层 forward OOM，进程使用达 `79.17 GiB`。

两种 mb8 重算方案均无法同时保住速度和容量：Slurm 35933 的整物理算子 checkpoint 将峰值降至 `55.66 GB`，但仅有 `6,856.94 token/s`（虚拟 2T mb8 的 `0.171x`）；Slurm 35959 的 chunk-start-only 重算在 CUDA JUnit 6/6 通过后仍以 `79.11 GiB` 进程占用 OOM，未产生吞吐数据。

降低到 micro batch 4 后，no-recompute 物理 T 确实能运行，但同机直接对照否决了这条训练路径：

| 路径 | Slurm | micro batch | 中位 step | 吞吐 | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| 虚拟 2T | 36061 | 4 | 0.4444 s | 36,863.50 token/s | 39.80 GB |
| no-recompute 物理 T | 36054 | 4 | 2.3731 s | 6,903.17 token/s | 62.62 GB |

两个作业均为 `COMPLETED / 0:0`、`run.exitcode=0`、CUDA JUnit 6/6，loss 有限。物理 T 的同 mb4 吞吐比只有 `0.187x`，中位 step 慢 `5.34x`，峰值显存反而为 `1.573x`。相对当前虚拟 2T mb8 生产基线，物理 mb4 吞吐也只有 `0.172x`，峰值显存为 `0.840x`。

在 8 卡 global batch 128 下，mb4 需要 GA4，而当前 mb8 只需要 GA2。梯度累积次数加倍不会改变上述单卡核心吞吐比，还会额外增加 Python/DDP/优化器边界开销。因此不做三顺序完整稳定 A/B，不做 8 卡 mb4/GA4 smoke，也不改写上面的生产推荐配置。

这组数据还说明：局部物理 T 算子相对 triangular oracle 的 `2.20x–2.24x` 加速没有转化为整模型收益。下一步必须 profiler 分解 18 层中的 WY 准备、chunk-state、output、backward 与调用开销，而不是继续调 batch size 或扩展 checkpoint 组合。`QGDN_USE_PHYSICAL_T=False` 继续保持。

## 并行 output backward 优化

Slurm 36076 将 B=4/T=4096/H=16/K=V=64/chunk=16 的单层物理算子分段，找到了与整模型失速一致的主瓶颈：

| 阶段 | 中位时延 |
|---|---:|
| prepare forward | 1.43 ms |
| WY forward | 3.41 ms |
| state + output forward | 2.52 ms |
| 旧 state + output backward | 92.24 ms |
| WY backward | 5.62 ms |
| prepared-input VJP | 4.54 ms |
| 完整物理算子 forward+backward | 118.33 ms |
| 同形状虚拟 2T 算子 | 15.38 ms |

旧 backward 把每 chunk 的输出反向放在跨 256 chunk 的单 program 逆扫里，整个 B=4/H=16/V=64 形状只有 128 个 program。Commit `e79e48c651961d8a7c0e413f2829cadeff9b8b35` 将输出反向拆成 65,536 个 chunk/value-block 并行 program，另外用轻量紧凑状态逆扫传播跨 chunk 伴随。CPU/FP64 回归为 138 passed / 48 CUDA skipped。

Slurm 36080 的三顺序 CUDA 门禁 6/6 通过，`run.exitcode=0`，全部输出、状态和七组梯度通过既定 BF16 门槛。新 backward 中位从 `92.79 ms` 降至 `19.10 ms`（`4.86x`），物理算子从 `118.33 ms` 降至 `45.07 ms`（`2.63x`）。但同次虚拟 2T 算子为 `15.70 ms`，因此新物理算子仍只有 `0.348x` 相对吞吐。

Slurm 36084 的整模型 mb4 单臂为 `15,819.91 token/s`、`31.52 GB`、中位 step `1.0351 s`，loss 有限且 CUDA JUnit 6/6。这相对旧物理 mb4 的 `6,903.17 token/s / 62.62 GB` 是 `2.29x` 加速和 `0.503x` 显存，但相对同机虚拟 mb4 仍只有 `0.429x` 吞吐和 `0.792x` 显存；相对虚拟 mb8 生产基线吞吐只有 `0.394x`。

因此该拆分作为后续优化基线保留，但当前版本不扩展到三顺序稳定 A/B 或 8 卡 mb4/GA4 smoke。下一步先将新 `19.10 ms` 拆成 parallel output adjoint 和 compact state adjoint 分别计时，再决定减少 atomic/program 数还是将状态逆扫改为分层 associative scan。`QGDN_USE_PHYSICAL_T=False` 不变。

Slurm 36440 在不可变 commit `7e6d198bc719914e1837863916fa280c7b260121` 上完成了
kernel 级拆分：作业 `COMPLETED / 0:0`、`run.exitcode=0`、CUDA JUnit 6/6、
全部数值有限。B=4/T=4096/H=16/K=V=64/chunk=16 的结果为：

| 分支 | 每次 CUDA self time | 占 19.017 ms backward |
|---|---:|---:|
| compact state adjoint | 9.469 ms | 49.8% |
| parallel output adjoint | 9.067 ms | 47.7% |
| 清零 / `fill_` | 0.399 ms | 2.1% |

两个主 kernel 的合计占比为 `51.1% / 48.9%`，状态逆扫略高但没有单支压倒性瓶颈。
同次 WY backward 为 `5.589 ms`，prepared-input VJP 为 `4.607 ms`，物理/虚拟算子时延为
`44.898/15.668 ms = 2.866x`。因此先审计 state/output 的 value-block 粒度：当
`BV=16` 时，V=64 造成 4 份与 V 无关的因子载入/计算和 4 路原子累加。下一个候选
将交错比较 `BV=16/32/64`，并要求实际形状全输入梯度仍通过。WY 内部的
closure/响应重算以及 prepared-input 的通用 autograd 各自只占当前整物理算子约
`12.4%` 和 `10.3%`，本轮不先改动。

Slurm 36443/36445 随后分别通过 CPU/FP64 与实际形状 H800 全梯度门禁，但否决了
state/output 同时加宽：BV16/32/64 的交错中位数为 `19.068/42.285/54.802 ms`。
output adjoint 随 block 加宽降到 `5.50/2.90 ms`，state adjoint 却因 256-chunk 串行
逆扫失去并行度而增至 `36.39/51.48 ms`。

Commit `5b1c92e9de7735f10c229f42b21ba5a7a82cb0f2` 改为 output BV64、state BV16。
Slurm 36448 的 CPU/FP64 回归为 138 passed / 48 CUDA skipped；Slurm 36451 为
`COMPLETED / 0:0`、`run.exitcode=0`、CUDA JUnit 6/6，全部数值有限，实际形状最坏
prepared-gradient 相对 RMSE `2.06e-7`。48 轮去相位 A/B 中 hybrid 为 `12.887 ms`，
对 BV16 的 `19.087 ms` 配对加速中位 `1.482x`（p10 `1.476x`，p90 `1.488x`，48/48
均快），peak allocated 同为 `4,288,676,352` bytes。kernel 级 output 从 `9.135 ms`
降至 `2.900 ms`，state 保持 `9.587 ms`。

完整物理算子中位 `38.673 ms`，相对 Slurm 36440 的 `44.898 ms` 提速约 `1.161x`，
但仍是同次虚拟 2T `15.798 ms` 的 `2.448x` 时延。WY backward 的 `5.625 ms` 几乎全部
落在单一融合 kernel（`5.498 ms`）；prepared-input VJP 的 `4.554 ms` 则主要分散在
mul/div/sum/add/neg。由于算子尚未超过虚拟 2T，本轮不进入整模型 A/B，生产默认继续关闭
物理 T。

Commit `5aaed69` 和 Slurm 36830/36862 又完成了完整 physical/virtual kernel 对照。CPU 为
138 passed / 48 CUDA skipped；CUDA 为 `COMPLETED / 0:0`、`run.exitcode=0`、JUnit 6/6、
全部有限。物理/虚拟中位 `38.689/15.773 ms = 2.453x`。核心结论是“物理 T”没有减少真正
支配 chunk 算法的维度：`T×rank2` 与 `2T×rank1` 都有 8192 个 rank row，物理 chunk16 与
虚拟 chunk32 都是 32 rank rows，并且都要扫描 256 个 chunk。再加上物理全 FP32/IEEE、
虚拟主要 BF16/Tensor Core，prepared/packed 静态输入量反而几乎相等（`471.9/469.8 MB`），
物理 FP32 chunk starts 还是虚拟 BF16 chunk states 的两倍。

kernel 证据进一步定位到 backward 架构：物理 state kernel 把 transition/read 重建和多组
factor/value/decay VJP 留在 256-chunk 串行循环中，耗时 `9.552 ms`；虚拟 DPLR 只在
`dhu` kernel 中传播依赖，把其余 VJP chunk-parallel 化，state kernel 仅 `0.478 ms`。
物理两次 WY forward 加 backward 为 `6.603+5.461=12.064 ms`；虚拟 WY 的
prepare+`wu` 两次和 backward 仅 `0.548+0.321=0.869 ms`，其余 intra algebra 由独立并行
kernel 承担。物理六个具名 Triton kernel 合计 `27.702 ms`，虚拟 13 个具名 DPLR kernel
仅 `11.373 ms`。

因此下一方向从“直接做分层 scan”修正为：先把 dependency-only state adjoint 与每 chunk
transition VJP 分离，复刻虚拟 DPLR 的串行最小化结构；随后才验证 BF16/Tensor Core WY 与
prepared-input VJP 融合。当前算子若要达到虚拟的 `1.25x`，需从 `38.689 ms` 降到
`<12.619 ms`，即至少 `3.07x`，单点微优化不构成可行路线。

专用环境内旧 `torchrun` 文件残留了其他环境的 shebang。DDP 基准和后续作业必须使用当前 Python 启动：

```text
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 ...
```

## 固定 gamma=1 消融结果

Commit `538712c` 新增 Recall→Delta 和 Parallel 的 `recall_gate="fixed"` / `recall_init=1.0`
配置；commit `49898cb` 将 CUDA 数值门禁扩展到 gamma=1 的三种更新顺序。固定 gamma
不参与训练，因此模型参数不含 recall-gate projection；变量统计应恒为
`gamma_mean=1.0`、`gamma_std=0.0`、`gamma_saturated_fraction=1.0`。

Slurm 37379（Recall→Delta）和 37380（Parallel）均已在独占 H800 节点上通过
`10 passed / 0 failed / 0 errors` preflight，包括 gamma=1 的 BF16 输出、末状态和全输入
梯度检查。两个作业都冻结在 `49898cb61973bf619c3256436b0d7bfc43e18326`，
使用与既有 GDN/可学习 gamma 对照一致的 10BT recipe：8×H800、T=4096、mb8/GB128/GA2、
fused loss、无 activation checkpointing、seed 3407、每 2000 step 验证 1600 条序列。
这两路仍使用虚拟 2T，预期吞吐和显存与同顺序可学习 gamma 近似；本消融的目的是
模型效果，不是速度优化。

首批训练样本显示 Recall→Delta 约 `316.5k token/s / 75.68 GB`，Parallel 约
`313.2k token/s / 75.72 GB`；首 step 因 kernel 编译而耗时约 113 秒，不纳入稳态。两路均为
`344,353,984` 参数、`recall_parameters=0`，gamma 指标持续为 `1/0/100%`，暂无 OOM、
非有限数值或配置偏离。

首个 step-2000 validation 已完成，gamma 仍在所有训练记录中严格为 `1/0/100%`。
Recall→Delta loss/PPL 为 `3.144389 / 23.20549`，相对同顺序 beta-style gamma 的
`3.140643 / 23.11873` 高 `0.375%` PPL；Parallel 为 `3.146042 / 23.24387`，相对
同顺序 beta-style gamma 的 `3.142662 / 23.16546` 高 `0.338%` PPL。两路最近 20 点
吞吐约 `316.02k / 313.21k token/s`，峰值显存不变；单个早期节点尚不足以下最终结论。

step-4000 validation 中，固定 gamma=1 Recall→Delta/Parallel 的 loss/PPL 为
`2.977149 / 19.63176`、`2.977474 / 19.63815`，比同顺序 beta-style gamma 分别高
`0.442% / 0.227%` PPL，也都差于 GDN `19.57818`。两路全部训练记录的 gamma 仍严格
为 `1/0/100%`，数值正常；连续两个共同验证点均显示固定强 recall 略差。

step-6000 validation 再次复现：固定 gamma=1 Recall→Delta/Parallel 的 PPL 为
`18.11486 / 18.10973`，比各自 beta-style 对照高 `0.448% / 0.268%`，比 GDN 高
`0.250% / 0.221%`。门控严格为 `1/0/100%` 且无数值异常，因此连续三点的负向信号
不能归因于实现漂移。

最新进度为 Recall→Delta step `7391`、Parallel step `7321`。全部训练记录仍精确保持
gamma `1/0/100%`；两路 beta mean/std 为 `0.25549 / 0.15378`、
`0.25233 / 0.15271`，近 20 点 loss 为 `2.82538 / 2.82573`，稳态吞吐约
`316.4k / 313.2k token/s`。loss、梯度、checkpoint 与日志持续健康。

step-8000 validation 的 Recall→Delta/Parallel loss/PPL 为
`2.842080 / 17.15141`、`2.841971 / 17.14953`。它们比各自 beta-style 对照高
`0.417% / 0.283%` PPL，也比 GDN 高 `0.304% / 0.293%`。两路现至
step `8751 / 8661`，gamma 在全部记录中仍精确为 `1/0/100%`；beta mean/std 为
`0.25606 / 0.15504` 与 `0.25424 / 0.15416`，数值和 checkpoint 均正常。固定强 recall
已在连续四个共同验证点稳定略差。

step-10000 validation 继续显示固定 gamma=1 略差：Recall→Delta/Parallel 的
loss/PPL 为 `2.799789 / 16.44118`、`2.799699 / 16.43970`，相对各自 beta-style
对照高 `0.399% / 0.318%` PPL，相对 GDN 高 `0.255% / 0.246%`。两路现至
step `10191 / 10091`，beta mean/std 为 `0.25772 / 0.15607` 与
`0.25611 / 0.15540`，gamma 仍严格为 `1/0/100%`；loss、grad norm、日志和 checkpoint
全部正常。连续五个共同节点均不支持固定强 recall。

step-12000 validation 继续复现：固定 gamma=1 Recall→Delta/Parallel 的 loss/PPL
为 `2.764880 / 15.87714`、`2.764925 / 15.87785`，相对各自 beta-style 对照高
`0.358% / 0.290%` PPL，相对 GDN 高 `0.262% / 0.267%`。两路现至
step `12761 / 12631`，beta mean/std 为 `0.26120 / 0.15812` 与
`0.25920 / 0.15711`，gamma 继续严格为 `1/0/100%`；近 20 点 loss 为
`2.72785 / 2.72884`。连续六个共同节点均否决固定强 recall 的效果收益。

step-14000 validation 继续复现：固定 gamma=1 Recall→Delta/Parallel 的
loss/PPL 为 `2.737784 / 15.45271`、`2.737444 / 15.44745`，相对各自 beta-style
对照高 `0.379% / 0.289%` PPL，相对 GDN 高 `0.277% / 0.243%`。两路现至
step `14141 / 14001`，beta mean/std 为 `0.26212 / 0.15805` 与
`0.25812 / 0.15711`，gamma 继续全程严格为 `1/0/100%`。连续七个节点均否决
固定强 recall 的效果收益。

两路固定 gamma=1 作业已成功完成。Recall→Delta/Parallel 最终 loss/PPL 为
`2.699919 / 14.87853`、`2.699549 / 14.87302`，相对同顺序 beta-style 高
`0.355% / 0.297%` PPL，相对 GDN 高 `0.257% / 0.220%`。末步 beta mean/std
为 `0.25857 / 0.15874` 与 `0.25554 / 0.15780`，gamma 全程严格为 `1/0/100%`。
训练/墙钟有效吞吐为 `318.15k/313.90k` 与 `315.12k/310.57k token/s`，峰值
显存 `75.6754 / 75.7184 GB/GPU`。终点完整否决固定强 recall。

## 可训练高 gamma 初始化消融结果

Commit `9e381f0` 新增 Recall→Delta 和 Parallel 的 trainable `gamma∼U(0.85,0.95)` 配置。
为保证初始 gate 严格在指定区间，每层每 head 先在 gate 空间采样，再将 logit 写入
可训练 bias；token projection weight 从零开始但保留梯度。该初始器在 forked RNG 中执行，
不改变共享 backbone 的 seed-3407 初始化。CPU 聚焦测试 `7/7` 通过，同时确认 gate
weight/bias 都收到有限非零梯度。

Slurm 37413（Recall→Delta）与 37414（Parallel）已分别启动于 dgx37/dgx38，冻结
commit `9e381f0ecb1a8c2fcd8397446003cf6fdf0530b7`。两路沿用 8×H800、T=4096、
mb8/GB128/GA2、fused loss、无 activation checkpointing 和同一验证口径；依然是虚拟 2T。
两路 H800 preflight 均为 `13/13` 通过，包括 gamma=0.9 的三顺序 BF16 输出、
末状态和全输入梯度检查。

实际启动后，两路共享初始化 hash 完全相同，均含 `328,000` 个可训练 gamma
参数。step 1 的 gamma mean/std/饱和率为 `0.898251 / 0.028024 / 0%`；到共同
step 31，Recall→Delta 为 `0.888909 / 0.031800 / 0.714%`，Parallel 为
`0.888854 / 0.031824 / 0.657%`。目前吞吐约 `312.1k / 309.8k token/s`，峰值显存
`77.03 / 77.06 GB/GPU`，无 OOM 或非有限数值。

到共同 step `1291`，Recall→Delta / Parallel 的近 20 点 loss 为
`3.28581 / 3.28671`，beta mean/std 为 `0.27866 / 0.16357` 与
`0.27854 / 0.16267`；gamma mean/std/饱和率已变为
`0.63251 / 0.30221 / 19.46%` 与 `0.63557 / 0.30007 / 19.11%`。两路仍高度对齐，
loss、grad norm 和 gate 值均有限，但约 19% 的 token gate 在 1.3K step 时已进入饱和区，
说明高初值没有被整体维持，而是迅速产生明显极化。最近 20 点吞吐约
`312.34k / 309.42k token/s`，两路 step-1000 checkpoint 均完整存在。

高 gamma 两路的 step-2000 validation loss/PPL 分别为 Recall→Delta
`3.142448 / 23.16049`、Parallel `3.143302 / 23.18028`。同顺序比较时，它们比
beta-style gamma 分别高 `0.181% / 0.064%` PPL，但比固定 gamma=1 分别低
`0.194% / 0.274%`，首个验证点呈现 beta-style、可训练高 gamma、固定 1 的由好到差排序。

共同 step `2621` 时，高 gamma Recall→Delta / Parallel 的近 20 点 loss 为
`3.05874 / 3.05883`，beta mean/std 为 `0.27061 / 0.15839` 与
`0.26994 / 0.15825`，gamma mean/std/饱和率为 `0.57741 / 0.31262 / 15.75%` 与
`0.57582 / 0.31047 / 15.26%`。饱和率从 step 1291 的约 19% 回落，暂未演变为单调塌缩；
两路所有数值有限，最近 20 点吞吐约 `312.15k / 309.44k token/s`。

共同 step `3851` 时，高 gamma Recall→Delta / Parallel 的近 20 点 loss 为
`2.96783 / 2.96738`，beta mean/std 为 `0.27336 / 0.15795` 与
`0.27076 / 0.15722`，gamma mean/std/饱和率为 `0.55068 / 0.31397 / 13.62%` 与
`0.55053 / 0.31249 / 13.10%`。饱和率延续从 19% 向下回落的趋势，所有指标有限；
两路距 step-4000 validation 仅约 150 step。

高 gamma 的 step-4000 validation loss/PPL 为 Recall→Delta
`2.975552 / 19.60045`、Parallel `2.974974 / 19.58912`。Recall→Delta 相对同顺序
beta-style 高 `0.282%` PPL，而 Parallel 低 `0.023%`；两路均明显好于固定 gamma=1，
但相对 GDN 仍高 `0.114% / 0.056%`。当前证据支持“gamma 可学习很重要”，不支持高
gamma 初始化本身稳定优于 beta-style 初始化。

共同 step `5081` 时，高 gamma Recall→Delta / Parallel 的近 20 点 loss 为
`2.90062 / 2.89972`，gamma mean/std/饱和率为 `0.53337 / 0.31511 / 12.22%` 与
`0.53536 / 0.31298 / 11.78%`；饱和率继续下降，数值和吞吐正常。

step-6000 validation 中，高 gamma Recall→Delta/Parallel 的 loss/PPL 为
`2.895069 / 18.08475`、`2.893995 / 18.06533`。同顺序对 beta-style 的 PPL 差分别为
`+0.281% / +0.023%`，对固定 gamma=1 则低 `0.166% / 0.245%`；相对 GDN 分别为
`+0.083% / -0.024%`。共同 step `6451` 的近 20 点 loss 为 `2.86154 / 2.86057`，
beta mean/std 为 `0.28145 / 0.16091` 与 `0.28003 / 0.16094`，gamma
mean/std/饱和率为 `0.52453 / 0.31461 / 11.28%` 与
`0.52535 / 0.31201 / 10.72%`。gate 饱和继续从早期约 19% 回落；当前结论仍是
可训练 gamma 显著好于固定 1，而高初值相对 beta-style 没有稳定收益。

共同 step `7781` 时，高 gamma Recall→Delta/Parallel 的近 20 点 loss 为
`2.82595 / 2.82504`，beta mean/std 为 `0.28443 / 0.16262` 与
`0.28250 / 0.16274`，gamma mean/std/饱和率为
`0.51811 / 0.31376 / 10.70%` 与 `0.51741 / 0.31204 / 10.31%`。饱和率继续缓慢
下降且没有数值异常；下一次效果判断等待共同 step-8000 validation。

step-8000 validation 中，高 gamma Recall→Delta/Parallel 的 loss/PPL 为
`2.840848 / 17.13029`、`2.839491 / 17.10706`。相对同顺序 beta-style 的 PPL 差为
`+0.294% / +0.035%`，相对固定 gamma=1 则低 `0.123% / 0.248%`，相对 GDN 高
`0.181% / 0.045%`。现至 step `9261 / 9191`，beta mean/std 为
`0.28931 / 0.16458` 与 `0.28734 / 0.16400`，gamma mean/std/饱和率为
`0.51014 / 0.31363 / 10.09%` 与 `0.50839 / 0.31167 / 9.82%`。可训练 gate 的
早期极化继续退潮且数值健康；四个共同验证点仍支持可训练优于固定 1，但高初值没有稳定
优于 beta-style。

step-10000 validation 中，高 gamma Recall→Delta/Parallel 的 loss/PPL 为
`2.798340 / 16.41737`、`2.797475 / 16.40318`。相对同顺序 beta-style 的 PPL 差为
`+0.253% / +0.095%`，相对固定 gamma=1 则低 `0.145% / 0.222%`，相对 GDN 高
`0.110% / 0.024%`。两路现至 step `10521 / 10441`，beta mean/std 为
`0.28998 / 0.16540` 与 `0.28834 / 0.16574`，gamma mean/std/饱和率为
`0.50631 / 0.31321 / 9.94%` 与 `0.50592 / 0.31201 / 9.70%`。五个共同节点均说明
高初值可训练 gate 优于固定 1，但没有超过 beta-style 初始化。

step-12000 validation 中，高 gamma Recall→Delta/Parallel 的 loss/PPL 为
`2.763985 / 15.86294`、`2.763101 / 15.84892`。相对同顺序 beta-style 的 PPL 差为
`+0.269% / +0.108%`，相对固定 gamma=1 则低 `0.089% / 0.182%`，相对 GDN 高
`0.173% / 0.084%`。两路现至 step `13161 / 13061`，beta mean/std 为
`0.29208 / 0.16637` 与 `0.29047 / 0.16630`，gamma mean/std/饱和率为
`0.50085 / 0.31353 / 9.70%` 与 `0.49780 / 0.31186 / 9.31%`。六个共同节点仍然
只支持可训练优于固定 1，不支持高 gamma 初值优于 beta-style。

两路高 gamma 可训练作业已成功完成。Recall→Delta/Parallel 最终 loss/PPL 为
`2.698479 / 14.85712`、`2.697799 / 14.84702`，相对同顺序 beta-style 高
`0.211% / 0.122%` PPL，相对固定 gamma=1 低 `0.144% / 0.175%`，相对 GDN 高
`0.113% / 0.045%`。末步 beta mean/std 为 `0.28953 / 0.16810` 与
`0.28842 / 0.16793`，gamma mean/std/饱和率为 `0.48994 / 0.31519 / 9.56%` 与
`0.48626 / 0.31390 / 9.15%`。训练/墙钟有效吞吐为 `314.18k/309.76k` 与
`311.40k/307.11k token/s`，峰值显存 `77.0250 / 77.0617 GB/GPU`。最终结论是
可训练优于固定 1，但高初值不如 beta-style。

Q-Delta 现已按论文精确递推实现为每个真实 token 一个 rank-1 DPLR row：
`x=k_hat+lambda*q_hat`，`S'=alpha*S+beta*k_hat*(v-alpha*x^T*S)^T`。因此它既不走
QGDN 虚拟 2T，也不依赖已否决的物理 T rank-2 kernel。CPU/FP64 的输出、末状态和全部
输入梯度门禁已通过；完整 10BT 严格对齐实验
`20260905-105931-qdelta-340m-10bt-s3407-95125c` / Slurm `38035` 已提交；入口 CUDA
门禁在 dgx25 通过 `3/3` 后，已使用 8×H800、T4096、mb8、GB128、GA2 开始跑满
19073 step。step 31 的首批有限指标为 loss `7.53594`、grad norm `1.11967`、lambda
mean/std `0.28854/0.05214`、收缩区间违规率 `0`、峰值显存 `66.27 GB/GPU`；热身后
首个有效吞吐样本约 `472.8k token/s`。后续用其真实训练
吞吐和峰值显存分别对比 GDN 与虚拟 2T QGDN，不把通用 DPLR 的理论 rank-1 优势直接当作
整模型速度结论。

负号消融保持同一个 rank-1 DPLR 实现，仅令 `x=k_hat-lambda*q_hat`。commit
`790d93dba29ac1814d4e6647b630a067f8173d45` 的 CPU/FP64 与 QGDN 相关回归为
`145 passed`，正负号参数在相同 seed 下逐位一致。完整 10BT 实验
`20260905-113033-qdelta-minus-340m-10bt-s3407-2ab048` / Slurm `38065` 已在 dgx17 通过
联合 CUDA `5/5` 门禁并开始训练；step 21 的 loss/grad norm 为
`8.23535/1.50980`，lambda mean/std `0.28866/0.05035`，收缩违规率 `0`，热身吞吐
约 `475.8k token/s`，峰值显存 `66.27 GB/GPU`。其余配方与正号任务完全一致，目标同为
19073 step / 9,999,745,024 prediction tokens。

首个共同 step-2000 validation 的正号/负号 loss/PPL 为
`3.139583/23.09423` 与 `3.141828/23.14613`，负号高 `0.2247%` PPL；正号相对 GDN
低 `0.2236%`，负号与 GDN 基本持平。对齐 step 1991 的最近 20 点训练 loss 为
`3.14627/3.14912`，两者均有限且收缩区间违规率为 `0`。最近 20 点单机吞吐约
`472.5k/470.4k token/s`，峰值显存 `66.27/66.27 GB/GPU`；两条运行在不同节点，
不把这点吞吐差解释为符号本身的速度差。

step-4000 的第二个共同 validation 仍由正号领先：正号/负号 loss/PPL 为
`2.971961 / 19.53018` 与 `2.973236 / 19.55509`，负号高 `0.1275%` PPL。正号
相对 GDN 低 `0.2451%` PPL，也比 beta-style Recall→Delta 低 `0.0777%`；负号
相对 GDN 低 `0.1179%`，但比 Recall→Delta 高 `0.0497%`。对齐 step 4101 的
近 20 点训练 loss 为 `2.94383 / 2.94526`，稳态吞吐约
`472.6k / 475.6k token/s`，峰值显存 `66.2665 / 66.2684 GB/GPU`。alpha/beta/lambda
均值和标准差均有限，收缩区间违规率继续为 `0`；两条 step-4000 checkpoint
完整且无 CUDA/NCCL/OOM 错误。连续两个共同节点支持正号，但差距正在收窄，
不提前终止任一作业。

step-6000/8000 的后续共同 validation 继续由正号领先。正号/负号 PPL 依次为
`18.02264/18.03731` 和 `17.06370/17.08113`，负号高 `0.0814%/0.1021%`。正号
相对 GDN 低 `0.2608%/0.2086%`，相对 beta-style Recall→Delta 低
`0.0632%/0.0962%`；负号与 Recall→Delta 的差距只有 `+0.0181%/+0.0058%`。
对齐 step 8081 的近 20 点训练 loss 为 `2.81421/2.81474`，吞吐为
`470.7k/475.8k token/s`，峰值显存仍为 `66.2665/66.2684 GB/GPU`。alpha/beta/lambda
和 alignment 统计均有限，收缩违规率为 `0`，无 CUDA/NCCL/OOM 错误。正号现至
step `10111` 并已完成 step-10000 validation（PPL `16.36178`）；负号现至 step `8081`，
继续跑到同一节点后再解读。

step-10000 的第五个共同 validation 中，正号/负号 loss/PPL 为
`2.794948/16.36178` 与 `2.795717/16.37437`，负号高 `0.0770%` PPL。正号
相对 GDN 低 `0.2288%`，相对 beta-style Recall→Delta 低 `0.0863%`；负号
相对 GDN 低 `0.1520%`，与 Recall→Delta 基本持平。对齐 step 11681 的近 20 点
训练 loss 为 `2.74715/2.74749`，吞吐为 `472.8k/475.4k token/s`，峰值显存仍为
`66.2665/66.2684 GB/GPU`。alpha/beta/lambda/alignment 均有限，收缩违规率为 `0`。
正号现至 step `13701` 并已完成 step-12000 validation（PPL `15.80459`）；负号现至
step `11681`，继续按相同节点配对。

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
- 手写重算 WY backward 数值与算子诊断：Slurm 35642（6/6 全梯度门禁通过；Python/einsum 路径因仅有 0.182x–0.270x 速度而否决）
- 融合 Triton WY backward 数值与算子诊断：Slurm 35649（6/6 全梯度门禁通过；两种顺序约 1.07x，一种受抖动影响为 0.798x）
- 块摊销、顺序轮换的 fused-WY 交错 A/B：Slurm 35893（6/6 门禁；三顺序配对中位约 1.058x，bootstrap 95% 下界均大于 1）
- 全融合状态/输出首次门禁：Slurm 35894（6/6 门禁；数值通过，性能编排发现三周期锁相）
- 去相位全融合状态/输出诊断：Slurm 35895（6/6 门禁；三顺序配对中位 2.199x–2.239x，p10 1.987x–2.012x）
- 训练入口与实际 340M BF16 数值门禁：Slurm 35896（6/6 门禁；最坏输入梯度相对 RMSE 0.00881；算子 peak allocated 为虚拟 2T 的 0.843x–0.871x）
- 干净 H800 整模型 mb8 容量门禁：Slurm 35911（虚拟 2T 40,121.94 token/s；no-recompute 物理 T 在 79.17 GiB OOM）
- 整物理算子 checkpoint 否决：Slurm 35933（55.66 GB，但仅 6,856.94 token/s）
- chunk-start-only 重算容量门禁：Slurm 35959（CUDA JUnit 6/6，整模型仍于 79.11 GiB OOM）
- 同机 micro batch 4 物理/虚拟对照：Slurm 36054 / 36061（6,903.17 vs 36,863.50 token/s；62.62 vs 39.80 GB）
- mb4/T4096 单层分阶段 profiler：Slurm 36076（旧 state+output backward 92.24 ms，占物理算子约 78%）
- 并行 output backward CUDA/性能门禁：Slurm 36080（6/6 数值与全梯度门禁；backward 4.86x，整算子 2.63x）
- 并行 output backward 整模型 mb4 单臂：Slurm 36084（15,819.91 token/s，31.52 GB；同 mb4 虚拟吞吐比 0.429x）
- split backward kernel 级拆分：Slurm 36440（state 9.469 ms / output 9.067 ms / fill 0.399 ms；CUDA JUnit 6/6）
- 联合 BV16/32/64 审计：Slurm 36443/36445（全梯度通过；state/output 同时加宽因 state 逆扫恶化而否决）
- output BV64 / state BV16 hybrid：Slurm 36448/36451（CPU 138 passed；CUDA 6/6；split backward 1.482x；峰值显存持平）
- 完整 physical/virtual kernel 根因对照：Slurm 36830/36862（38.689 vs 15.773 ms；确认 rank-row/chunk 数未下降，串行 state VJP 与 FP32 WY 为关键差距）
- 固定 gamma=1 的 Recall→Delta / Parallel 10BT 消融：Slurm 37379/37380（H800 preflight 均 10/10 通过，训练成功完成并回收）
- 可训练 gamma∼U(0.85,0.95) 的 Recall→Delta / Parallel 10BT 消融：Slurm 37413/37414（dgx37/dgx38，H800 preflight 13/13，训练成功完成并回收）
- 论文 Q-Delta 严格对齐 10BT：Slurm 38035（实验 `20260905-105931-qdelta-340m-10bt-s3407-95125c`；commit `18dc085b`；入口 CUDA 门禁后跑满 19073 step）
- Q-Delta 负号严格配对 10BT：Slurm 38065（实验 `20260905-113033-qdelta-minus-340m-10bt-s3407-2ab048`；commit `790d93d`；入口正负号联合 CUDA 门禁 `5/5`）
