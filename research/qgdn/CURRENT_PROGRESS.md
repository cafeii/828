# QGDN 当前进度与接力说明

更新时间：2026-09-03 19:41（Asia/Shanghai）

## 当前目标

当前只做两件事：

1. 保持单一记忆矩阵，实现并比较三种双监督更新顺序：Recall→Delta、Delta→Recall、Parallel。
2. 优化 QGDN 训练速度，重点移除现有实现中的显式虚拟 2T 序列。

目前没有继续 QR-GDN、DT-GDN、JQC-GDN，也没有提交新的 FineWeb 正式预训练。

## 已完成

三种更新顺序已作为独立模型配置接入，均使用精确的秩二仿射递推：

- `qgdn_340M`：Recall→Delta
- `qgdn_delta_then_recall_340M`：Delta→Recall
- `qgdn_parallel_340M`：Parallel

已完成 FP64 参考、状态转移、初末状态、前向与全输入反向验证。H800 作业 35365 通过 58 项 GPU 测试，三个版本均通过 BF16 输出、末状态和梯度检查。

关键提交：

| Commit | 内容 |
|---|---|
| `bdc85ee89f84f59bbd5b8db0c8ba137a775898ed` | 接入三种精确更新顺序 |
| `95394b53d4d7ef52c533302b7269baecd77c4150` | 增加 CUDA 顺序验证和 2T 成本隔离基准 |
| `4277295560fed0568a41d6d9c10b2ae909b0ccec` | 修复动态长度编译 |
| `b765376d201a6f94a8681b24441802e8ffb69e28` | 审计更快的 DPLR 后端候选 |
| `343e75a4c9200f80a6344ba1c977f5e472220998` | 物理 T 融合 kernel 原型 |
| `6c8b5a0789da6a419e3ed3512daff4ee0ede5c30` | 为物理 T 原型增加 CUDA 门控测试与整模型基准 |

## 当前速度结论

同一张 H800、340M、序列 4096、micro batch 8、关闭 checkpoint、fused loss 的短基准如下：

| 模型路径 | 吞吐 | 相对 GDN | 峰值显存 |
|---|---:|---:|---:|
| GDN | 110,755 token/s | 100.0% | 54.04 GB |
| Recall 置零但仍走虚拟 2T | 40,562 token/s | 36.6% | 73.23 GB |
| Recall→Delta | 40,028 token/s | 36.1% | 74.58 GB |
| Delta→Recall | 38,868 token/s | 35.1% | 74.62 GB |
| Parallel | 39,660 token/s | 35.8% | 74.62 GB |

这说明当前主要瓶颈不是 Recall 数值本身，而是虚拟 2T 的通用 DPLR 路径。即使 Recall 完全关闭，仍损失约 63.4% 吞吐，并增加约 19 GB 峰值显存。

这组数据来自 Slurm 35365；它用于比较实现效率，不是语言模型效果实验。

## 已排除的候选

TileLang 快路径在 Slurm 35367 中触发 CUDA `vectorized_gather_kernel index out of bounds`，作业以退出码 1 失败。该候选没有产出可用性能结果，也没有被设为生产默认。

## 正在运行的物理 T 诊断

物理 T 原型直接按 T 个真实 token 执行每步两个低秩修正，不再构造零行、2T 查询/键/值张量或丢弃一半输出。它保持三种更新顺序的原公式，不把两个修正错误地合并成一个秩一更新。

当前原型仍是显式 opt-in，`QGDN_USE_PHYSICAL_T=False`，因此没有改变生产训练路径。

- Slurm：35377
- 实验：`20260903-193528-qgdn-physical-t-audit-0b29a7`
- Commit：`6c8b5a0789da6a419e3ed3512daff4ee0ede5c30`
- 资源：1×H800，16 CPU，128 GB，最长 2 小时
- 内容：先跑完整 QGDN CUDA 测试，再比较虚拟 2T 与三个物理 T 版本的整模型吞吐和显存
- 当前状态：RUNNING

不要在终态审查前启用该 kernel。它的反向会在短 chunk 内重建状态，需要重点检查 Triton 编译资源、梯度误差、有限值和实际吞吐；只要其中一项不合格，就保持现有生产路径不变。

## 下一任务的第一步

1. 用 `experiment.py status`、`sacct`、`run.exitcode`、日志、JUnit 和 `physical-t-audit.json` 审查 Slurm 35377。
2. 无论成功或失败，都用 `sync_results.sh` 回收结果并记录第一处实质性错误。
3. 若正确性通过且整模型明显加速，再做 8 卡 DDP smoke；通过后才考虑把物理 T 设为默认。
4. 若 Triton 资源、数值重建或吞吐不合格，保留失败证据，转为物理 T 的并行 chunk/WY 秩二实现，不回退到有 CUDA 越界的 TileLang 候选。
5. 验证完成后更新本文件和 `TRAINING_SPEED.md`，提交并推送 `QGDN` 分支。

远程开发仓库为 `/work/projects/memos-b3/code/wangzr/828`，分支 `QGDN`，专用环境为 `/work/projects/memos-b3/software/miniconda3/envs/wangzr-qgdn`。GitHub 为 `cafeii/828`。
