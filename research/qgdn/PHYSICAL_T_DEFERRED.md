# QGDN 物理 T 优化暂存

更新时间：2026-09-04（Asia/Shanghai）

## 暂存决定与本次恢复

物理 T 优化已暂停，代码和实验记录保留，当前训练路线恢复为经过验证的虚拟 2T
generalized-DPLR 实现。生产默认必须保持：

```text
QGDN_USE_PHYSICAL_T=False
```

暂停原因不是数值公式未解决，而是当前物理 T 实现在真实 340M、T=4096 整模型上仍显著
慢于虚拟 2T。后续 FineWeb 训练不应启用物理 T，也不应为了容纳物理 T 将 micro batch
从 8 降到 4。

2026-09-04 13:04 起，只在独立不可变快照和短 H800 诊断中恢复优化。已冻结的
Parallel Slurm 36311 和 Delta→Recall Slurm 36312 不修改、不取消、不重提；不提交新的
FineWeb 正式训练。本次恢复不改变上述生产默认和启用门槛。

## 已完成的实现

- 三种更新顺序 Recall→Delta、Delta→Recall、Parallel 均已表示为每个真实 token 的精确
  rank-2 仿射转移。
- 已建立逐 token dense、紧凑 rank-2、chunk/WY 和训练入口四层参考与对照。
- 已实现专用 Triton WY forward/backward、块级状态扫描和块内输出恢复。
- 已将输出反向从跨 chunk 的状态逆扫中拆出，对所有 chunk 并行执行；跨 chunk 部分只保留
  紧凑状态伴随。
- 物理 T 训练入口仍是显式 opt-in，不影响虚拟 2T 默认路径。

主要代码入口：

- `model/lit_gpt/mixers/qgdn_training_kernel.py`
- `model/lit_gpt/mixers/qgdn_wy_kernel.py`
- `model/lit_gpt/mixers/qgdn_state_output_kernel.py`
- `model/lit_gpt/mixers/qgdn_reference.py`
- `model/lit_gpt/mixers/qgdn_rule.py`

## 正确性证据

- CPU/FP64 回归在当前 split-backward 版本达到 `138 passed / 48 CUDA skipped`。
- Slurm 35896 在实际 340M 算子形状 `B=1/T=4096/H=16/K=V=64` 上通过三种顺序、
  query/key recall、输出、末状态和七组输入梯度的 BF16 门禁；最坏输入梯度相对 RMSE
  为 `0.008812`，所有值有限。
- Slurm 36080 在并行 output backward 后再次通过 CUDA JUnit `6/6`，三种顺序的输出、
  状态和全部输入梯度均通过门槛。

因此，物理 T 的代数与 CUDA 数值路线已经成立；当前否决依据是训练效率，而不是公式错误。

## 性能结论

以下结果均为 H800、340M、sequence length 4096：

| 路径 | micro batch | 吞吐 | 峰值显存 | 说明 |
|---|---:|---:|---:|---|
| 虚拟 2T | 8 | 40,121.94 token/s | 74.58 GB | 当前生产基线，Slurm 35911 |
| 虚拟 2T | 4 | 36,863.50 token/s | 39.80 GB | 同机控制，Slurm 36061 |
| 旧物理 T | 4 | 6,903.17 token/s | 62.62 GB | no-recompute，Slurm 36054 |
| split-backward 物理 T | 4 | 15,819.91 token/s | 31.52 GB | 当前最好物理版本，Slurm 36084 |

split-backward 使物理整模型相对旧版本提速 `2.29x`，峰值显存降为 `0.503x`；但它仍只有
同 micro batch 虚拟 2T 的 `0.429x` 吞吐，以及虚拟 2T micro batch 8 基线的 `0.394x`
吞吐。这个差距不能由梯度累积调整弥补。

分阶段 profiler 的关键变化：

- Slurm 36076：旧 state+output backward 为 `92.24 ms`，完整物理算子为 `118.33 ms`。
- Slurm 36080：并行化后该 backward 为 `19.10 ms`，加速 `4.86x`；完整物理算子为
  `45.07 ms`，加速 `2.63x`。
- 同次虚拟 2T 算子为 `15.70 ms`，所以优化后的物理算子仍慢 `2.87x`。

## 已否决路线

- TileLang 候选曾触发 CUDA 越界，不得直接复用。
- 静态展开整个序列的串行物理 T Triton backward 编译时间和资源开销不可接受。
- eager/Python token 循环和通用 autograd streaming 路径速度不足。
- 合并 triangular solve 的两组 RHS 没有收益。
- 整物理算子 checkpoint 虽能降低显存，但吞吐只有虚拟基线的 `0.171x`。
- chunk-start-only 重算在 micro batch 8 下仍 OOM。
- no-recompute 物理 T 在 micro batch 8 下使用约 `79.17 GiB` 后 OOM。

## 本次重启的起点

当前开发分支起点为 commit `7e6d198bc719914e1837863916fa280c7b260121`。该提交加入了
split backward 的 kernel 事件级 profiler；为它准备的实验
`20260904-120154-split-bwd-kernel-profile-mb4-5e584e` 在当时只完成准备、**没有提交 Slurm**，
因此当时不能当作已有结果。

该快照后于 Slurm 36440 唯一提交并完成：`COMPLETED / 0:0`、`run.exitcode=0`、
CUDA JUnit 6/6，分段结果全部有限。B=4/T=4096/H=16/K=V=64/chunk=16 上，
split backward 中位数 `19.017 ms`，其中 compact state adjoint `9.469 ms` (`49.8%`)，
parallel output adjoint `9.067 ms` (`47.7%`)，清零 `0.399 ms` (`2.1%`)。WY backward 与
prepared-input VJP 分别为 `5.589 ms` 和 `4.607 ms`。完整物理算子 `44.898 ms`，
虚拟 2T 算子 `15.668 ms`。

因 state/output 两支近似五五开，本轮先选择同时作用于两支的 value-block 合并候选：
对 `BV=16/32/64` 做全梯度与去相位 A/B，验证减少 V 无关计算的重复加载及共享梯度的
atomic 竞争是否有实质收益。

联合加宽已经否决。Commit `620a6e0` 的 CPU Slurm 36443 为 138 passed / 48 CUDA
skipped；H800 Slurm 36445 为 `COMPLETED / 0:0`、`run.exitcode=0`、CUDA JUnit 6/6、
全部梯度有限。BV16/32/64 的 backward 中位数为 `19.068/42.285/54.802 ms`：output
确实由约 `9.14 ms` 降至 `5.50/2.90 ms`，但 256-chunk 串行 state 逆扫由约
`9.54 ms` 恶化为 `36.39/51.48 ms`。

Commit `5b1c92e9de7735f10c229f42b21ba5a7a82cb0f2` 保留 state BV16、仅将 output 改为
BV64。CPU Slurm 36448 再次获得 138 passed / 48 skipped；H800 Slurm 36451
（`20260904-132840-physical-bwd-hybrid-cuda-7bbde6`）为 `COMPLETED / 0:0`、
`run.exitcode=0`、CUDA JUnit 6/6、全输入梯度有限，最坏相对 RMSE `2.06e-7`。48 轮
去相位 A/B 中 hybrid `12.887 ms` 对 BV16 `19.087 ms`，配对加速中位 `1.482x`
（p10/p90 `1.476x/1.488x`，48/48 均快），四种配置 peak allocated 都是
`4,288,676,352` bytes。output kernel `9.135 -> 2.900 ms`，state kernel 保持
`9.587 ms`。

完整物理算子从 `44.898 ms` 降到 `38.673 ms`（约 `1.161x`），但同次虚拟 2T 为
`15.798 ms`，物理路径仍有 `2.448x` 时延。WY backward `5.625 ms` 中融合 kernel 占
`5.498 ms`；prepared-input VJP `4.554 ms` 则主要由 mul `1.222 ms`、div
`1.044 ms`、sum `0.594 ms`、add_ `0.584 ms`、neg `0.278 ms` 组成。

Slurm 36830/36862 在 commit `5aaed694f8a3b6c4d71d52d33dfda44740e33edd` 上补全了
两条完整路径的 kernel 对照；CPU 为 138 passed / 48 CUDA skipped，H800 为
`COMPLETED / 0:0`、`run.exitcode=0`、CUDA JUnit 6/6、全部有限。物理/虚拟中位为
`38.689/15.773 ms = 2.453x`。关键结论如下：

- 物理 `T×rank2` 与虚拟 `2T×rank1` 都是 8192 个 rank row；chunk16×rank2 与
  chunk32×rank1 都形成 32-row WY block，且都扫描 256 个 chunk。物理 T 没有降低核心
  block rank 或依赖深度。
- 物理 prepared 张量全为 FP32，Triton dot 使用 IEEE；虚拟 DPLR 主张量与状态缓存主要为
  BF16 并使用 Hopper autotune/Tensor Core。两侧 prepared/packed 静态输入约
  `471.9/469.8 MB`，而物理 FP32 chunk starts `268.4 MB` 是虚拟 BF16 states
  `134.2 MB` 的两倍。
- 物理 state backward `9.552 ms` 仍把可独立的 transition/read 重建和
  left/effective/write/value/decay VJP 放在 256-chunk 串行循环内；虚拟 DPLR 将这些工作拆到
  chunk-parallel kernel，其 dependency-only `dhu` 只有 `0.478 ms`。
- 物理两次 WY forward 共 `6.603 ms`、WY backward `5.461 ms`，还另有 eager prepared
  VJP；虚拟 WY forward/backward 的直接 kernel 总计只有 `0.869 ms`，intra algebra 也由
  BF16/autotuned 并行 kernel 承担。物理六个具名 Triton kernel 总计 `27.702 ms`，虚拟
  13 个 DPLR kernel 为 `11.373 ms`。

因此当前差距是表示与 kernel 架构共同造成的，不是再调一个 block size 就能消除。把物理
算子做到虚拟的 `1.25x` 要求 `<12.619 ms`，相对当前至少还需 `3.07x`；整模型门槛因
Amdahl 定律会更严格。

接下来的优先顺序应为：

1. 先将 dependency-only state adjoint 与每 chunk transition VJP 分离；串行 kernel 只保留
   必须跨 chunk 传播的 K×V adjoint，factor/value/decay 梯度全部 chunk-parallel 化。
2. 在分离后的并行 VJP 与 WY 中审计 BF16 storage、FP32 accumulation 和 Tensor Core dot，
   优先复用/改造 FLA block-WY，只改变 paired-row causal mask。
3. 融合 physical preparation 的 forward/VJP；只有纯 state 依赖仍主导时才尝试分层 scan。
4. 每个候选先通过三种顺序的 CPU/FP64 与 H800 全梯度门禁。
5. 只有实际 T=4096 算子超过虚拟 2T，并且整模型 micro batch 8 不 OOM，才进入完整 A/B。
6. 只有整模型相对虚拟 2T 稳定快 `>1.25x` 且显存不恶化，才能考虑打开默认开关。

## 当前训练路线

当前使用虚拟 2T，下一候选为 `qgdn_parallel_340M`。8 卡配置为：

```text
sequence length = 4096
micro batch = 8 / GPU
global batch = 128 sequences = 524,288 tokens
gradient accumulation = 128 / (8 GPUs * 8 micro batch) = 2
activation checkpointing = off
training loss = fused cross entropy
QGDN_USE_PHYSICAL_T = False
```

正式训练提交前仍应先做一次相同配置的短 8 卡 Parallel smoke；本次归档不提交训练作业。
