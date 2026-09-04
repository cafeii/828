# QGDN 当前进度与接力说明

更新时间：2026-09-04 09:22（Asia/Shanghai）

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
| `189100afe783d4d1fb701780d1f0b57c55ea4f0e` | 增加不物化 2C×2C system 的流式 WY 前代数 |
| `d1706b46f021a6d0e6d29d8a2cb87320d3a9f856` | 增加流式 WY CUDA 数值与性能门控 |
| `7c7d237bbecf3f0832ec3b403d28c0f84946ab75` | 增加块摊销、顺序轮换和交错 WY A/B |
| `55d66c80b9a8438d55ff8efc2e37d6550b26c3f3` | 融合物理 T 的块级状态扫描和块内输出恢复 |
| `bb42be3df2fc4e627f266bcd9b75c387c55e20ad` | 增加 triangular / fused-WY / 全融合三后端诊断 |
| `22a3b27fd0c7f286320199d18edea793e229f8a2` | 消除三顺序与三后端轮换的周期锁相 |

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

Commit `189100afe783d4d1fb701780d1f0b57c55ea4f0e` 又实现了不构造 2C×2C system 的流式 forward substitution：每个物理 token 的两个 effective-right 因子只与此前的 rank-2 history 相乘，write-response 则通过 chunk 内零初态递推得到；B/H/chunk 维保持并行。CPU/FP64 新增后端对齐后，相关 chunk/WY 聚焦集合为 91 passed / 6 CUDA skipped，三种顺序、query/key recall、chunk size 1/3/8、输出、末状态和全部输入梯度均正确。

H800 作业 35622（实验 `20260903-221056-qgdn-wy-streaming-cuda-f7dfff`，commit `d1706b46f021a6d0e6d29d8a2cb87320d3a9f856`）为 `COMPLETED / 0:0`，`run.exitcode=0`，JUnit 6 passed / 0 failed，三种顺序的 triangular/streaming CUDA 输出、状态和梯度门禁均通过。但 eager/autograd streaming 相对双 solve 的速度比仅为 `0.376x`、`0.444x`、`0.412x`，peak allocated memory 反而统一为 `1.142x`。因此流式代数可作为专用 kernel 内部算法，但当前 Python 循环和通用 autograd 路径被否决；默认仍是 triangular 双 solve oracle。

Commit `1b589cabfd6c4737bb87af62b199a64aa43023cc` 将这套流式代数写成单个 forward-only Triton WY kernel：每个 B/H/chunk lane 在 program 内构造 token-causal closure，并同时生成 effective-right 与零初态 write-response，不再向 PyTorch 物化全局 `[B,H,N,2C,2C]` system。提交前的 CPU/FP64 rank-2 聚焦回归为 99 passed，说明三种顺序既有的输出、末状态和全输入梯度密集参考合约未被改动。

第一次 H800 编译门禁 35627（实验 `20260903-222547-qgdn-wy-triton-fwd-2d432c`）为 `FAILED / 1:0`，`run.exitcode=1`，JUnit 3 passed / 3 failed：chunk=16 的三种顺序通过，但 chunk=8 的 write RHS 使用了归约维 8 的 `tl.dot`，低于 Triton 要求的 16，故未产生 benchmark JSON。Commit `d77f05e93b39506c4a7d7e9479b0e87ec9afb825` 仅将该内部 token 维零填充到 16，并保持物理 chunk 的因果 mask 不变。

修复后的 H800 作业 35628（实验 `20260903-222951-qgdn-wy-triton-fwd-pad-c60f57`）为 `COMPLETED / 0:0`，`run.exitcode=0`，JUnit 6 passed / 0 failed，覆盖三种顺序、chunk 8/16 和 identity-padded 尾 chunk。相对 triangular oracle，最大输出相对 RMSE 为 `4.62e-8`，最大末状态相对 RMSE 为 `2.48e-8`，全部输出和状态有限。FP32 B=2/T=128/H=4/K=V=64/chunk=16 的 forward-only 中位数速度比分别为 `0.732x`、`1.280x`、`1.220x`；peak allocated memory 比均为 `0.987x`，incremental peak 比为 `0.982x`。首个顺序的 10 次样本有明显 2.48–6.66 ms 波动，因此当前只能确认 CUDA 数值正确和轻微显存下降，不能声称三种顺序都有稳定速度收益。

该 Triton 后端仍是显式 diagnostic 且 forward-only；尚无 backward、有限梯度门禁、整模型吞吐或整模型峰值显存结果。生产默认 `QGDN_USE_PHYSICAL_T=False`，不会因本次 forward 通过而启用。

Commit `221892c59d07bfc55fcf5a5206eae9dffe2864c2` 为该 forward kernel 增加了手写重算 backward。forward 只保存 normalized-left、right、normalized-write 和 values 四个输入；backward 重算 effective-right history 与 chunk 内 write-state history，再显式反传，不保存 forward token 中间图，也不构造 2C×2C system。新增 CPU/FP64 直接梯度对照覆盖 chunk size 1/3/8，完整 rank-2 聚焦集合为 102 passed。

H800 作业 35642（实验 `20260903-224553-qgdn-wy-recompute-bwd-27370d`）为 `COMPLETED / 0:0`，`run.exitcode=0`，JUnit 6 passed / 0 failed，覆盖三种更新顺序、chunk 8/16、尾 chunk padding、输出、末状态以及 q/k/v/g/beta/gamma/初态全部梯度。最大输出、末状态、输入梯度相对 RMSE 分别为 `4.62e-8`、`2.48e-8`、`7.98e-7`，全部有限。

但这版 backward 仍由 Python token 循环和通用 einsum 执行。FP32 B=2/T=128/H=4/K=V=64/chunk=16 的 forward+backward 中位数约为 48.1–48.4 ms，而 triangular oracle 为 8.75–13.0 ms，三种顺序的速度比分别只有 `0.270x`、`0.182x`、`0.198x`。peak allocated memory 比为 `0.975x`，incremental peak 比为 `0.957x`。因此手写公式与重算边界作为 CUDA backward 的可靠 oracle 保留，但当前执行路径因速度被否决，不接入训练默认。

Commit `f09b36e81f59d73d5767173b48a20bb5e40f4d0c` 将上述伴随公式融合进单个 Triton backward program。每个 B/H/chunk lane 在片上重建 `A^-1`，用 `A^-T` 同时反传 effective-right 与 write-response，再将 causal coupling 的梯度映射回 normalized-left、right、normalized-write 和 values。CPU/FP64 新增的 dense-adjoint 公式与逐 token 手写反向、PyTorch autograd 三方一致，完整 rank-2 聚焦集合仍为 102 passed。

H800 作业 35649（实验 `20260903-225903-qgdn-wy-triton-bwd-ec36b5`）为 `COMPLETED / 0:0`，`run.exitcode=0`，JUnit 6 passed / 0 failed。三种更新顺序、chunk 8/16、尾 chunk padding、输出、末状态和全部七组模型输入梯度均有限；最大输出、末状态、输入梯度相对 RMSE 分别为 `4.62e-8`、`2.48e-8`、`7.71e-7`。

同一算子配置的 fused forward+backward 中位数为 Recall→Delta 8.78 ms、Delta→Recall 6.37 ms、Parallel 6.10 ms，相对 triangular 分别为 `0.798x`、`1.068x`、`1.071x`；peak allocated memory 比仍为 `0.975x`，incremental peak 比为 `0.957x`。相比上一版 Python 重算的约 48 ms，launch 开销已经基本消除，但 Recall→Delta 的 10 次样本仍在 6.23–10.24 ms 间波动。当前结论是 fused backward 数值通过、两种顺序有约 7% 算子收益，尚未证明三种顺序稳定加速，也没有整模型结果。

Commit `ab679003af3c8aa2520cc556bf959bad3cc4924b` 和 `7c7d237bbecf3f0832ec3b403d28c0f84946ab75` 随后将计时改为顺序轮换、后端交错、固定输出梯度，并用每样本 8 次 forward+backward 摊销单次调度噪声。Slurm 35893（实验 `20260904-084515-qgdn-wy-block-ab-e88ce0`）为 `COMPLETED / 0:0`、`run.exitcode=0`、JUnit 6 passed / 0 failed。50 组样本中，三种顺序的配对中位速度比均约 `1.058x`，bootstrap 中位数 95% 下界分别为 `1.032`、`1.047`、`1.037`，AB/BA 两个方向的中位数也全部大于 1；peak allocated 比约 `0.975x`，incremental peak 比约 `0.957x`。因此 fused WY 本身的三顺序收益被确认稳定，但幅度只有约 6%。

Commit `55d66c80b9a8438d55ff8efc2e37d6550b26c3f3` 新增了两个专用 Triton 阶段：块级状态 kernel 只跨 chunk 递推，不逐 token 串行；输出 kernel 对全部 chunk 并行计算 query 对 chunk 起始状态、rank-2 WY 更新和直接写入的因果投影，不物化每个 token 的 K×V 状态。配套 backward 反向扫描 chunk，并手写查询、decay、左右因子、write-read、写入向量、value 和初态梯度。独立 prepared-chunk FP64 合约在 chunk 1/3/8 上 3/3 通过，原 rank-2 聚焦集仍为 102 passed。

Slurm 35894 首次验证该全融合路径：`COMPLETED / 0:0`、`run.exitcode=0`、JUnit 6 passed / 0 failed，最大输出、末状态和七组模型输入梯度相对 RMSE 为 `1.80e-7`、`6.23e-8`、`1.82e-6`。它显示约 `2.21x` 的算子速度信号，但审计发现三种模型顺序与三后端使用同周期轮换，导致每个模型顺序内部的后端位置固定；数值证据保留，稳定性能结论由修正版复测取代。

Commit `22a3b27fd0c7f286320199d18edea793e229f8a2` 将后端轮换去相位，并加入必须覆盖全部三种排列的运行时断言。Slurm 35895（实验 `20260904-091430-qgdn-fused-state-output-dephased-e7e1bf`）为 `COMPLETED / 0:0`、`run.exitcode=0`、JUnit 6 passed / 0 failed；每个更新顺序都实际覆盖三种后端排列。B=2/T=128/H=4/K=V=64/chunk=16、50 组且每组 8 次 forward+backward 的结果如下：

| 更新顺序 | triangular 中位数 | 全融合中位数 | 配对中位加速 | 配对 p10 | 全部配对 >1 |
|---|---:|---:|---:|---:|---:|
| Recall→Delta | 9.153 ms | 4.143 ms | 2.199x | 1.987x | 50/50 |
| Delta→Recall | 9.106 ms | 4.103 ms | 2.212x | 2.004x | 50/50 |
| Parallel | 9.093 ms | 4.023 ms | 2.239x | 2.012x | 50/50 |

全融合相对只融合 WY 的配对中位加速仍为 `2.121x`、`2.100x`、`2.135x`。peak allocated memory 相对 triangular 为 `0.522x`，incremental peak 为 `0.111–0.114x`；相对 fused-WY-only 的 peak 比为 `0.532–0.533x`。最大输出、末状态和模型输入梯度相对 RMSE 为 `1.80e-7`、`6.23e-8`、`1.84e-6`，全部有限。

这证明物理 T 的完整参考算子分解已经具备强而稳定的局部速度/显存收益，但它仍运行在 diagnostic reference 接口，尚未接入 340M 训练模块，也不是相对生产虚拟 2T 的整模型结果。`QGDN_USE_PHYSICAL_T=False` 继续保持；下一门禁是训练路径接入、BF16/实际形状数值检查和单卡整模型同配置 A/B。

## 下一任务的第一步

1. 将已验证的全融合物理 T 路径接入 `qgdn_rule` 的显式 opt-in 训练分支，保持三种更新顺序和默认开关不变。
2. 补齐 BF16/实际 340M head shape、序列 4096 和尾 chunk 的输出、末状态、全部参数梯度与有限性门禁；若 kernel envelope 不适合实际形状，先修实现而不是启用默认。
3. 数值通过后跑同卡、同模型、同 batch、同训练配置的单卡整模型“物理 T vs 虚拟 2T”交错 A/B，最低启用门槛为稳定 `>1.25x`，目标约 `1.5x`，且显存不得恶化。
4. 只有整模型通过后才做 8 卡 DDP smoke；在此之前保持 `QGDN_USE_PHYSICAL_T=False`。

远程开发仓库为 `/work/projects/memos-b3/code/wangzr/828`，分支 `QGDN`，专用环境为 `/work/projects/memos-b3/software/miniconda3/envs/wangzr-qgdn`。GitHub 为 `cafeii/828`。
