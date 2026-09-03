# QGDN 当前进度与接力说明

更新时间：2026-09-03 20:06（Asia/Shanghai）

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

Slurm 35377（实验 `20260903-193528-qgdn-physical-t-audit-0b29a7`，commit `6c8b5a0789da6a419e3ed3512daff4ee0ede5c30`）否决了串行物理 T Triton 原型。作业为 `FAILED / 1:0`，`run.exitcode=1`；JUnit 记录 61 项测试中 58 通过、3 失败。三种更新顺序都先通过了 CUDA forward 输出和末状态容差检查，但 backward kernel 编译时在 `tl.store(grad_g + token_head, dg_t)` 报错：要写入标量指针的 `dg_t` 被推导为 block。每种顺序的失败分别耗时 363.548、437.586 和 400.792 秒，整个 pytest 用时 1335.137 秒，说明当前静态展开 backward 还有明显编译资源问题。

因为 backward 未能编译，梯度有限性和数值误差均未获得；门控脚本未进入整模型 benchmark，因此 `physical-t-audit.json` 缺失，吞吐和峰值显存都没有可报告数值。这不是速度候选通过，不能据此推断它比虚拟 2T 更快。

## 物理 T 路径结论

物理 T 原型直接按 T 个真实 token 执行每步两个低秩修正，不再构造零行、2T 查询/键/值张量或丢弃一半输出。它保持三种更新顺序的原公式，不把两个修正错误地合并成一个秩一更新。

该串行原型仍只是显式 opt-in，`QGDN_USE_PHYSICAL_T=False`，因此生产训练仍走已验证的虚拟 2T DPLR 路径。不修补后直接启用，也不用已越界的 TileLang 候选替换它。下一个有效方向是将每个真实 token 的秩二仿射转移接入并行 chunk/WY 扫描，保留 T 时间长度，同时避免单个 program 内静态展开整个反向递推。

## 下一任务的第一步

1. 设计物理 T 的并行 chunk/WY 秩二实现，不继续扩展当前串行 Triton 递推。
2. 先用 FP64 密集参考检查三种更新顺序的 chunk 转移、输出、末状态和所有输入梯度。
3. 再用 H800 做 CUDA 数值与有限梯度检查；只有通过后才跑同卡、同配置的整模型虚拟 2T 对照。
4. 只有整模型明确加速后才做 8 卡 DDP smoke；在此之前保持 `QGDN_USE_PHYSICAL_T=False`。

远程开发仓库为 `/work/projects/memos-b3/code/wangzr/828`，分支 `QGDN`，专用环境为 `/work/projects/memos-b3/software/miniconda3/envs/wangzr-qgdn`。GitHub 为 `cafeii/828`。
