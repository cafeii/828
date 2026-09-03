# QGDN 当前进度与接力说明

更新时间：2026-09-03 22:03（Asia/Shanghai）

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
| `17b2201ce73817f615dcd93b8722f92163926086` | 定义物理 T 秩二 chunk/WY 紧凑仿射合约 |
| `78eef3486dc51af3657fe1cd59739c48b50af1a7` | 用块下三角求解并行化 chunk 内秩二 reads |
| `ddd4c9a9ed8a0e3907080ecbc2a352ba3d6b5c85` | 增加并行 chunk 的 H800 输出/状态/反向门控 |
| `a85f2a5329c8ece5d0897d7fd77fd5e5f2a2cd9e` | 将全部物理 T chunk 的 WY 准备合并为一次批量求解 |
| `ba5be78c6da8805d57cc7ef040e38847767909a7` | 增加批量 WY CUDA 门控和算子诊断 |
| `0f14b2fa2d20e621b43b0881877466619d4f4247` | 审计合并 WY 两组 RHS 的候选 |
| `d157fcb2fa190ad05d3c7d67f1f52e0cc7ff5b4b` | 根据 H800 结果保留双 solve 默认 |

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

## 并行 chunk/WY 合约

已建立可微的 CPU/FP64 紧凑仿射参考。每个真实 token 直接表示为 `scale * I + U @ V.T` 的秩二转移加一个写入 bias；两个仿射可在不构造 K×K 稠密转移的情况下结合，组合满足结合律。一个 C-token chunk 保持 C 个物理时间行，WY 秩维最多为 2C，不再向公共算子暴露 2T 虚拟序列。

针对 Recall→Delta、Delta→Recall 和 Parallel，query/key recall、chunk size 1/3/8，已验证输出、末状态和 q/k/v/g/beta/gamma/初态的全部梯度与逐 token FP64 参考一致。新增块求解实现先在 CPU 聚焦测试中与紧凑组合参考共同通过 46 项，最严梯度容差为 `8e-11`。

块求解实现会先除去 chunk 内标量 decay prefix，将每个 token 的两个 read 组成单位下三角 2×2 block 系统，一次 `solve_triangular` 得到全部 chunk 内 reads，再累加低秩修正和写入。该路径保持 `[B,T,H,2,K]` 因子和 T 行输出，不构造 2T 序列或 K×K 稠密转移。

H800 作业 35580（实验 `20260903-212608-qgdn-rank2-chunk-cuda-67839c`）为 `COMPLETED / 0:0`，`run.exitcode=0`，stderr 为空，JUnit 为 3 passed / 0 failed。三种顺序的 FP32 CUDA 输出、末状态和所有输入梯度均为有限值；输出/状态相对 RMSE 低于 `2e-5`，梯度低于 `1e-4`。日志和 JUnit 已回收；当前回收脚本因将 manifest 中的 `outputs/pytest.xml` 再次拼到 `outputs/` 下而误报 required output 缺失，但本地和远程的实际 JUnit 文件均存在且可解析。这是数值里程碑，尚未进入整模型吞吐/显存评测。

下一版 reference 已把 chunk 维折叠到 batch 维，一次 `solve_triangular` 同时准备全部独立 chunk 的有效 right 因子和写入响应；块间只剩 compact rank-2C 状态转移，取得各 chunk 起始状态后，再并行恢复所有 chunk 内状态与输出。最后一个不完整 chunk 用恒等转移补齐，测试会显式覆盖这一分支。CPU/FP64 新增 18 项全部通过，相关 chunk/WY 聚焦集合为 55 passed / 3 CUDA skipped，覆盖三种顺序、query/key recall、chunk size 1/3/8、输出、末状态和全部输入梯度。

H800 作业 35593（实验 `20260903-214702-qgdn-parallel-wy-cuda-6c110f`，commit `ba5be78c6da8805d57cc7ef040e38847767909a7`）为 `COMPLETED / 0:0`，`run.exitcode=0`，JUnit 3 passed / 0 failed，三种顺序的 CUDA 输出、末状态和全部输入梯度均有限且通过与逐 token 参考的相对 RMSE 门槛。算子级 FP32、B=2/T=128/H=4/K=V=64、chunk=16 的 forward+backward 中位数显示，批量 WY 相对逐 chunk solve 在 Recall→Delta、Delta→Recall、Parallel 上分别快 `1.226x`、`1.915x`、`2.111x`；对应吞吐为 20,575、25,244、25,091 physical token/s。代价是峰值 allocated memory 均约为对照的 `1.46x`（约 0.169 GB vs 0.115 GB）。这仍是算子 oracle，不是整模型结果；完整耦合矩阵和 autograd 保存带来的显存增长必须在融合 kernel 中消除。

H800 作业 35600（实验 `20260903-215810-qgdn-wy-fused-rhs-cuda-412446`，commit `0f14b2fa2d20e621b43b0881877466619d4f4247`）进一步审计了把 effective-right 和 write-response 拼成一个宽 RHS、只调用一次 `solve_triangular` 的微优化。作业 `COMPLETED / 0:0`，`run.exitcode=0`，JUnit 3 passed / 0 failed；但相对双 solve，三种顺序的速度比分别只有 `0.769x`、`0.849x`、`0.942x`，峰值 allocated memory 比均为 `1.000x`。因此该候选没有速度或显存收益，已在 commit `d157fcb2fa190ad05d3c7d67f1f52e0cc7ff5b4b` 恢复双 solve 默认，仅保留开关用于复现。这个结果说明通用三角求解的 RHS 拼接不是解法，下一步必须直接写专用 WY kernel，并避免 materialize 完整耦合矩阵。

## 下一任务的第一步

1. 将已验证的双 solve 全 chunk 准备改写为专用 WY kernel，避免 materialize 完整 `[B,H,N,2C,2C]` 耦合矩阵和通用 autograd 保存；不再尝试宽 RHS 合并。
2. 接入块间状态 kernel 与块内输出 kernel，并以当前批量 WY 路径作为 CUDA 数值 oracle；随后实现手写反向。
3. 融合 CUDA 数值、有限梯度及显存门槛通过后，再跑同卡、同配置的整模型虚拟 2T 对照。
4. 只有整模型明确加速后才做 8 卡 DDP smoke；在此之前保持 `QGDN_USE_PHYSICAL_T=False`。

远程开发仓库为 `/work/projects/memos-b3/code/wangzr/828`，分支 `QGDN`，专用环境为 `/work/projects/memos-b3/software/miniconda3/envs/wangzr-qgdn`。GitHub 为 `cafeii/828`。
