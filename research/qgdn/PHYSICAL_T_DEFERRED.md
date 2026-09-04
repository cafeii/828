# QGDN 物理 T 优化暂存

更新时间：2026-09-04（Asia/Shanghai）

## 暂停决定

物理 T 优化已暂停，代码和实验记录保留，当前训练路线恢复为经过验证的虚拟 2T
generalized-DPLR 实现。生产默认必须保持：

```text
QGDN_USE_PHYSICAL_T=False
```

暂停原因不是数值公式未解决，而是当前物理 T 实现在真实 340M、T=4096 整模型上仍显著
慢于虚拟 2T。后续 FineWeb 训练不应启用物理 T，也不应为了容纳物理 T 将 micro batch
从 8 降到 4。

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

## 以后重启时的起点

当前开发分支起点为 commit `7e6d198bc719914e1837863916fa280c7b260121`。该提交加入了
split backward 的 kernel 事件级 profiler；为它准备的实验
`20260904-120154-split-bwd-kernel-profile-mb4-5e584e` **没有提交 Slurm**，不能当作已有结果。

若以后恢复，优先顺序应为：

1. 分离测量 chunk-parallel output adjoint、compact state-adjoint scan、WY backward 和
   prepared-input VJP。
2. 若 output adjoint 主导，减少 program 数、原子竞争和重复读取；若 state scan 主导，改为
   分层 associative reverse scan。
3. 每个候选先通过三种顺序的 CPU/FP64 与 H800 全梯度门禁。
4. 只有实际 T=4096 算子超过虚拟 2T，并且整模型 micro batch 8 不 OOM，才进入完整 A/B。
5. 只有整模型相对虚拟 2T 稳定快 `>1.25x` 且显存不恶化，才能考虑打开默认开关。

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
