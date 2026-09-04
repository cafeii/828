# QGDN 当前进度与接力说明

更新时间：2026-09-04（Asia/Shanghai）

## 当前目标

当前路线已经切回虚拟 2T：

1. 保持单一记忆矩阵，使用已验证的虚拟 2T 实现比较三种双监督更新顺序。
2. 下一训练候选为 `qgdn_parallel_340M`，推荐 8 卡、micro batch 8、global batch 128、
   gradient accumulation 2、关闭 activation checkpointing、fused loss。
3. 物理 T 优化已在不触及正式训练的独立短诊断路径上恢复；生产默认仍保持
   `QGDN_USE_PHYSICAL_T=False`。完整接力信息见 [物理 T 优化暂存](PHYSICAL_T_DEFERRED.md)。
4. 新训练的 token-wise gamma 默认改为与 beta 相同的初始化方案：独立
   Xavier-uniform 权重（相同 gain）和零 bias；不再使用零权重加 `logit(0.1)`。
   已完成的 Recall→Delta 10B 结果仍属于旧初始化，不能与新初始化的 Parallel 解释为
   只改变更新顺序的严格配对实验。

目前没有继续 QR-GDN、DT-GDN、JQC-GDN。已在同一新 gamma 初始化下提交
Parallel 与 Delta→Recall 两个 10BT FineWeb 正式预训练，均保持虚拟 2T 默认路径。

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
| `1335ee690822ab72c736c94c2b6c49363d604e95` | 将全融合物理 T 接入显式 opt-in 训练路径 |
| `69d9e6cae3f50cbb4d399bc878f872e382df4be8` | 保持整模型 A/B 父进程 CUDA-free |
| `a1c1bd2be7848e24cfd6b5d7dd4a9fea4778b5a8` | 审计整个物理 T 算子 checkpoint |
| `101f1f2fa9bf84b67813f08c69178adf13fadca3` | 只重算物理 T chunk-start 状态 |
| `b1084a821a69dc654f82fc3102d9e00814cb3daa` | 用单一自定义 autograd 边界约束跨层保存张量 |
| `9ced1c8d7fb8eb83df224de973c0dd17e7298306` | 增加实际 mb4/T4096 物理 T 分阶段 profiler |
| `e79e48c651961d8a7c0e413f2829cadeff9b8b35` | 将每 chunk 输出反向从跨 chunk 状态逆扫中分离并行化 |

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

这证明物理 T 的完整参考算子分解具备强而稳定的局部速度/显存收益；当时它仍运行在 diagnostic reference 接口，也不是相对生产虚拟 2T 的整模型结果。

Commit `1335ee690822ab72c736c94c2b6c49363d604e95` 随后将这条全融合路径接入 `qgdn_rule` 的显式 opt-in chunk 训练分支。新分支支持 query/key recall、三种更新顺序、末状态和尾 chunk；旧的串行物理 T 分支不再由生产入口调用。物理 chunk size 独立固定为 16，默认开关仍为 `QGDN_USE_PHYSICAL_T=False`。提交前显式隐藏 CUDA 的完整 CPU 回归为 135 passed / 48 skipped。

Slurm 35896（实验 `20260904-094915-physical-chunk-training-audit-2ce53c`）为 `COMPLETED / 0:0`、`run.exitcode=0`、JUnit 6 passed / 0 failed。入口级 BF16 测试覆盖 query/key、三种更新顺序、尾 chunk、输出、末状态和 q/k/v/g/beta/gamma/初态全部梯度。实际 340M 算子形状 `B=1/T=4096/H=16/K=V=64` 与生产虚拟 2T 路径的结果如下：

| 更新顺序 | 输出相对 RMSE | 末状态相对 RMSE | 最大输入梯度相对 RMSE | 物理/虚拟 peak allocated |
|---|---:|---:|---:|---:|
| Recall→Delta | 0.004375 | 0.002561 | 0.006391 | 0.843x |
| Delta→Recall | 0.004581 | 0.002620 | 0.008812 | 0.871x |
| Parallel | 0.004394 | 0.002623 | 0.005034 | 0.849x |

所有输出、状态和梯度均有限，远低于输出/状态 0.025、梯度 0.07 的 BF16 门槛。该结果通过了训练入口与实际形状的 CUDA 数值门禁，并在算子作用域相对虚拟 2T 降低约 12.9%–15.7% 峰值分配；它尚未给出整模型吞吐，因此默认开关继续关闭。

## 整模型内存与 micro batch 4 门禁

Slurm 35911 在一张干净 H800 上先完成了虚拟 2T、micro batch 8 基线：`40,121.94 token/s`、`74.58 GB`、中位 step `0.8169 s`，loss 有限。同一作业的第一个 no-recompute 物理 T 进程在后续层 forward 中 OOM：PyTorch 已分配 `78.53 GiB`，进程总使用 `79.17 GiB`，请求额外 64 MiB 失败。

两个 micro batch 8 内存候选也未达标：

- Slurm 35933 的整算子 checkpoint 可运行，CUDA JUnit 6/6、`run.exitcode=0`、loss 有限，峰值显存降到 `55.66 GB`；但吞吐仅 `6,856.94 token/s`，是虚拟 2T mb8 基线的 `0.171x`，因重算开销被否决。
- Slurm 35959 只重算 chunk-start，CUDA JUnit 6/6 先通过，但整模型 mb8 仍在后续层 forward OOM：进程使用 `79.11 GiB`、PyTorch 分配 `78.46 GiB`；`run.exitcode=1`，未产出性能 JSON。

用户建议的 micro batch 4 已在同一 dgx19、同一不可变 commit `69d9e6c`、340M、T=4096、关闭 checkpoint、fused loss 下做了直接对照：

| 路径 | Slurm | 中位 step | 吞吐 | 峰值显存 | loss/门禁 |
|---|---:|---:|---:|---:|---|
| 虚拟 2T mb4 | 36061 | 0.4444 s | 36,863.50 token/s | 39.80 GB | finite；CUDA JUnit 6/6 |
| no-recompute 物理 T mb4 | 36054 | 2.3731 s | 6,903.17 token/s | 62.62 GB | finite；CUDA JUnit 6/6 |

物理 T 只有同 mb4 虚拟 2T 吞吐的 `0.187x`（慢 `5.34x`），显存反而是 `1.573x`。它相对当前生产虚拟 2T mb8 也只有 `0.172x` 吞吐；虽然显存为 `0.840x`，但 8 卡 global batch 128 需要从 mb8/GA2 改为 mb4/GA4，累积开销不会扭转这一差距。

因此 mb4 路径在第一个最快预期的 no-recompute 候选上就同时失败了 `>1.25x` 吞吐门槛和“显存不劣化”门槛。不再执行三顺序×多重复的完整 A/B，也不提交 8 卡 DDP smoke。`QGDN_USE_PHYSICAL_T=False` 保持不变。

## 并行输出 backward 里程碑

Slurm 36076（commit `9ced1c8`）首先用当前单 autograd 边界将 B=4/T=4096/H=16/K=V=64/chunk=16 的物理算子拆段。作业 `COMPLETED / 0:0`、`run.exitcode=0`、CUDA JUnit 6/6、全部量有限。forward 的 prepare/WY/state+output 中位分别只有 `1.43/3.41/2.52 ms`，WY backward 为 `5.62 ms`，prepared-input VJP 为 `4.54 ms`；原 `state+output backward` 却达 `92.24 ms`，占完整物理算子 `118.33 ms` 的约 78%。同形状虚拟 2T 算子只有 `15.38 ms`。

根因是旧 backward 只启动 128 个 B/H/value-block program，每个 program 在内部串行逆扫 256 个 chunk，并把每个 chunk 的输出反向所有矩阵乘也放进这个循环。Commit `e79e48c` 将输出反向改为 65,536 个 chunk/value-block 并行 program，只把紧凑状态伴随传播留在跨 chunk 逆扫中。提交前 CPU/FP64 回归为 138 passed / 48 CUDA skipped。

Slurm 36080 验证了新 backward：`COMPLETED / 0:0`、`run.exitcode=0`、CUDA JUnit 6/6，三种更新顺序的输出、末状态和七组输入梯度门禁均通过。同一 profiler 中：

- `state+output backward`：`92.79 -> 19.10 ms`，加速 `4.86x`；
- 完整物理算子 forward+backward：`118.33 -> 45.07 ms`，加速 `2.63x`；
- 新物理算子仍是虚拟 2T `15.70 ms` 的 `2.87x` 时延，吞吐比为 `0.348x`。

Slurm 36084 随后完成整模型 mb4 单臂：中位 step `1.0351 s`、`15,819.91 token/s`、`31.52 GB`、loss 有限，作业与 JUnit 均为 0 失败。相对旧 no-recompute 物理 mb4，吞吐提高 `2.29x`、峰值显存降到 `0.503x`；但相对同机虚拟 mb4 仍只有 `0.429x` 吞吐，显存为 `0.792x`。相对虚拟 mb8 生产基线的吞吐也只有 `0.394x`。

这是显著的工程改进，但当前候选仍同时达不到同 mb4 `>1.25x` 门槛和 mb4/GA4 相对现有 mb8 的有效吞吐门槛，因此不跑完整三顺序 A/B 或 8 卡 smoke，默认开关不变。

Slurm 36440（实验 `20260904-120154-split-bwd-kernel-profile-mb4-5e584e`，commit
`7e6d198bc719914e1837863916fa280c7b260121`）已完成 split backward 的 kernel 级拆分。
作业为 `COMPLETED / 0:0`、`run.exitcode=0`，CUDA JUnit 6/6，所有分段结果有限。
在 B=4/T=4096/H=16/K=V=64/chunk=16 上，`state+output backward` 中位数为
`19.017 ms`；5 次 profiler 迭代按 CUDA self time 分解为：

- compact state adjoint：`9.469 ms`，占整段 `49.8%`；
- parallel output adjoint：`9.067 ms`，占 `47.7%`；
- `fill_`/清零：`0.399 ms`，占 `2.1%`。

两个主 kernel 在自身 CUDA 时间中分别占 `51.1%` 和 `48.9%`，状态逆扫只是略高，
不能把剩余瓶颈归因于单一分支。同次 WY backward 为 `5.589 ms`，prepared-input VJP
为 `4.607 ms`；完整物理算子为 `44.898 ms`，对虚拟 2T 的 `15.668 ms` 仍是
`2.866x` 时延。

当前选定的下一候选是合并 state/output backward 的 value block：当 V=64 且
`BV=16` 时，两个 kernel 都会将与 value 维无关的 left/effective/write 载入和计算重复
4 次，并对共享梯度做 4 路 atomic add。先对 `BV=32/64` 做实际形状数值与交错 A/B；
若不能稳定降低整段时延则直接否决。

这一联合候选已被实测否决。Commit `620a6e0` 先加入 `BV=16/32/64` 审计；Slurm 36443
完成 CPU/FP64 全输入梯度回归（138 passed / 48 CUDA skipped），Slurm 36445 完成 H800
实际形状门禁（`COMPLETED / 0:0`、`run.exitcode=0`、CUDA JUnit 6/6、全部梯度有限）。
但 state/output 同时加宽时，交错中位数由 `19.068 ms` 恶化为 `42.285/54.802 ms`。
kernel 事件说明两支响应相反：output adjoint 从约 `9.14 ms` 降到 `5.50/2.90 ms`，而
包含 256-chunk 串行逆扫的 state adjoint 从约 `9.54 ms` 增至 `36.39/51.48 ms`。

Commit `5b1c92e9de7735f10c229f42b21ba5a7a82cb0f2` 因此将两支解耦：output 使用覆盖完整
V=64 的 `BV=64`，消除共享梯度 atomic；state 保持 `BV=16` 维持并行度。Slurm 36448
再次完成 CPU/FP64 回归（138 passed / 48 skipped）。Slurm 36451（实验
`20260904-132840-physical-bwd-hybrid-cuda-7bbde6`）在 B=4/T=4096/H=16/K=V=64 上完成
最终 H800 审计：

- 作业 `COMPLETED / 0:0`、`run.exitcode=0`，CUDA JUnit 6/6，输出、状态和全部 prepared
  input 梯度有限；相对 BV16 对照的最坏梯度相对 RMSE 为 `2.06e-7`；
- 48 轮去相位交错 A/B 中，每个后端在四个位置各出现 12 次；hybrid 中位
  `12.887 ms`，对 BV16 的 `19.087 ms` 配对加速中位 `1.482x`，p10/p90 为
  `1.476x/1.488x`，48/48 样本均快；
- state kernel 保持 `9.587 ms`，output kernel 由 `9.135 ms` 降到 `2.900 ms`
  （`3.15x`），清零约 `0.399 ms`；四种配置的 peak allocated 均为
  `4,288,676,352` bytes，没有显存恶化；
- 完整物理算子由 Slurm 36440 的 `44.898 ms` 降至 `38.673 ms`（约 `1.161x`），但
  同次虚拟 2T 为 `15.798 ms`，物理路径仍是 `2.448x` 时延、只有 `0.409x` 吞吐。

同次审计也覆盖了另外两段。WY backward 中位 `5.625 ms`，其中融合
`_qgdn_streaming_wy_bwd_kernel` 约 `5.498 ms`、清零 `0.125 ms`，已经是单一主 kernel。
prepared-input VJP 中位 `4.554 ms`，是通用 autograd 的碎片化点算子链；profiler 每次约含
`1.222 ms` mul、`1.044 ms` div、`0.594 ms` sum、`0.584 ms` in-place add 和
`0.278 ms` neg。若继续处理 prepared-input VJP，应一次融合整条闭式链，而不是微调单个
点算子；state adjoint 是否应直接改为分层 scan，则由下面的完整路径对照重新判断。

hybrid 是有效的算子级改进，但完整物理算子仍未超过虚拟 2T，因此按门禁不启动整模型
A/B、不提交新 FineWeb 训练，也不讨论打开默认路径。`QGDN_USE_PHYSICAL_T=False` 不变。

### 为什么物理 T 仍比虚拟 2T 慢

Commit `5aaed694f8a3b6c4d71d52d33dfda44740e33edd` 为同一次完整 physical/virtual
forward+backward 加入 kernel profile。CPU Slurm 36830 为 `COMPLETED / 0:0`、
`run.exitcode=0`、138 passed / 48 CUDA skipped；H800 Slurm 36862（实验
`20260904-160304-physical-virtual-kernel-cuda-ec494c`）为 `COMPLETED / 0:0`、
`run.exitcode=0`、CUDA JUnit 6/6，所有结果有限。相同 B=4/T=4096/H=16/K=V=64 输入上，
物理与虚拟算子中位为 `38.689/15.773 ms`，即 `2.453x` 时延。

根因不是简单的 kernel launch 数，而是物理表示没有带来预期的工作量下降，同时落入了效率
更低的实现形态：

1. **T 减半没有减少 block rank 或 scan depth。** 每个物理 token 是 rank-2，每个虚拟
   row 是 rank-1；所以两条路径都是每条序列 `8192` 个 rank row。物理 chunk 是
   `16×rank-2=32 rows`，虚拟 chunk 是 `32×rank-1=32 rows`，在 T=4096 上都恰好有
   `256` 个 chunk。WY 的 32×32 闭包、跨 chunk 状态依赖和主要 K/V 矩阵乘规模都没有减半。
2. **rank×2 与 FP32×2 抵消了长度收益。** 当前物理路径将 q/k normalization、rank-2
   factors、value、WY factors 和 chunk states 全部提升为 FP32，并在 Triton dot 中显式使用
   `input_precision="ieee"`；虚拟 DPLR 的 q/k/v/a/b、w/u/h 主要保持 BF16，并调用 Hopper
   autotune kernel。实际 prepared/packed 输入静态字节量约为 `471.9/469.8 MB`；物理 FP32
   chunk starts 为 `268.4 MB`，是虚拟 BF16 chunk states `134.2 MB` 的两倍。
3. **所谓 compact state backward 仍串行执行了大量局部 VJP。** 物理
   `_qgdn_chunk_state_bwd_kernel` 在每个 B/H/BV program 的 256-chunk 逆循环内，不仅传播
   state adjoint，还重建 transition/read，并计算 left/effective/write/value/decay 的梯度，
   每 chunk 有多组 FP32 IEEE dot 和 atomic。它单独耗时 `9.552 ms`。虚拟
   `chunk_dplr_bwd_kernel_dhu` 只保留依赖链，把 intra/WY/output 参数梯度放在 chunk-parallel
   kernel，state adjoint 仅 `0.478 ms`。因此上一版文档把下一步直接定为 hierarchical scan
   还不够准确；应先把可并行的 transition VJP 从串行循环剥离。
4. **WY 与输入 VJP 也失去了成熟融合。** 完整物理调用两次 FP32 streaming-WY forward
   （正常 forward 和 backward 重算），合计 `6.603 ms`，WY backward `5.461 ms`；物理
   state/WY/output 六个具名 Triton kernel 合计 `27.702 ms`。虚拟 DPLR 的 13 个对应具名
   kernel 合计 `11.373 ms`，其中 WY forward prepare+`wu` 两次调用总共 `0.548 ms`、WY
   backward `0.321 ms`。此外物理 prepared-input VJP 仍是 eager autograd 点算子链，而虚拟
   input builder 的 forward/backward 已由 Inductor 融合。

这意味着只优化一个小 kernel 不足以翻盘。若仅要求物理算子达到虚拟算子的 `1.25x`，目标
已经是 `<12.619 ms`，当前还需 `3.07x` 整体提速；20 层整模型还受非 mixer 开销的 Amdahl
约束，实际门槛更严。正确优先级改为：先拆出 dependency-only state adjoint 与 chunk-parallel
transition VJP；再审计 BF16 storage/Tensor Core accumulation 及复用 FLA block-WY 的 paired-row
causal mask；随后融合 physical preparation VJP。只有纯依赖 scan 仍占主导时才做分层 scan。

## 虚拟 2T 正式训练已提交

2026-09-04 在冻结 commit `7eb73ca89411c54d4fe7a8ffb427df44e7709cfa` 上直接提交了两个
10BT 作业；因当时没有空闲 8 卡节点，不另占节点提交独立 smoke，而是在每个正式作业开头执行
gamma 初始化和三顺序虚拟 2T CUDA 输出/状态/反向门禁，门禁失败时不会进入训练：

| 模型 | 实验 | Slurm | 初始状态 |
|---|---|---:|---|
| Parallel | `20260904-124542-qgdn-parallel-340m-10bt-s3407-1268f4` | 36311 | PENDING (Resources) |
| Delta→Recall | `20260904-124920-qgdn-delta-recall-340m-10bt-s3407-retry-a11400` | 36312 | PENDING (Priority) |

两者均为 8×H800、T=4096、micro batch 8、global batch 128、gradient accumulation 2、
fused loss、关闭 activation checkpointing、seed 3407；`max_steps=19073`，即
9,999,745,024 个 prediction tokens。验证每 2000 step、1600 sequences，checkpoint 每 1000
step。gamma 与 beta 使用同方案的独立 Xavier 随机初始化，`QGDN_USE_PHYSICAL_T=False`。

首次训练后对齐检查已完成。两作业内置 CUDA/JUnit 均为 6/6，通过 step 2000 后均有完整
checkpoint，日志持续更新且 loss、grad norm、gamma 统计和吞吐均为有限值。相同 step 2000
的 validation 如下：

| 模型 | validation loss | PPL |
|---|---:|---:|
| Parallel | 3.142662 | 23.16546 |
| Delta→Recall | 3.143802 | 23.19188 |

在共同 step 2671 对齐，最近 20 个日志点的训练 loss 均值为 Parallel `3.049501`、
Delta→Recall `3.050772`，当前尚无显著优化差距。gamma 动态已经分化：Parallel 的
mean/std/饱和率为 `0.44296 / 0.31147 / 8.05%`，Delta→Recall 为
`0.22305 / 0.21491 / 1.57%`。两者仍有限，暂不判定异常；后续继续在相同 token/step 和
validation 边界追踪，而不按墙钟错位比较。

第二个共同 validation（step 4000、2,097,152,000 tokens）继续接近：Parallel 的
loss/PPL 为 `2.975204 / 19.59361`，Delta→Recall 为 `2.976156 / 19.61229`，Parallel
仅领先约 `0.095%` PPL。共同 step 4551 的近 20 点训练 loss 均值也只差 `0.00145`
（`2.93801` vs `2.93946`）。Parallel 的 gamma mean/std/饱和率为
`0.42114 / 0.30414 / 6.08%`，Delta→Recall 为 `0.20319 / 0.20486 / 1.17%`；两边饱和率
均较上一观察下降。

Delta→Recall 所在 dgx01 曾因同时运行大量 CPU scoring 作业出现不规则 step 抖动，最近 40
个日志点吞吐中位约 `244.8k token/s`，但最新 step 已回升到 `295.6k token/s`；同一时段
Parallel 稳定为 `306.3k token/s`。没有 OOM、Traceback、非有限数值或 checkpoint 损坏，且
18 小时上限仍充足，因此保留当前作业继续运行，不为瞬态节点争用丢弃训练进度。

第三个共同 validation（step 6000、3,145,728,000 tokens）仍未拉开：Parallel 的
loss/PPL 为 `2.893768 / 18.06124`，Delta→Recall 为 `2.895291 / 18.08877`，Parallel
PPL 领先约 `0.152%`。共同 step 6691 的近 20 点训练 loss 为 `2.85331 / 2.85491`。
此时 Parallel 与 Delta→Recall 的 beta mean/std 分别为
`0.28484 / 0.16481` 和 `0.27335 / 0.16485`，非常接近；gamma mean/std/饱和率分别为
`0.40765 / 0.30031 / 5.19%` 和 `0.19141 / 0.19779 / 0.91%`。Delta→Recall 的最近
40 点吞吐中位已恢复到 `297.6k token/s`，但均值 `270.6k` 仍反映少量节点争用长尾。

17:41 的后续观察显示 dgx01 争用再次加重：Delta→Recall 最近 40 点吞吐中位/均值降至
`188.3k / 191.9k token/s`，而 Parallel 保持 `306.4k token/s`。节点快照有 36--45 个
blocked process 和约 14% I/O wait，与训练数值无关。共同 step 7501 的近 20 点 loss
仍为 `2.82758 / 2.82857`；beta mean/std 为 `0.28673 / 0.16486` 与
`0.27436 / 0.16497`，gamma mean/std/饱和率为 `0.40637 / 0.30111 / 5.18%` 与
`0.18623 / 0.19535 / 0.84%`。两边 loss、grad norm 和门控统计仍全部有限，checkpoint
持续写入；按当前受扰吞吐 Delta→Recall 纯训练尚需约 9 小时，仍在 18 小时 Slurm 上限内，
因此不丢弃已完成的 39.3% 进度、不重提，仅继续监控节点争用是否解除。

第四个共同 validation（step 8000）仍然等价：Parallel 的 loss/PPL 为
`2.839141 / 17.10108`，Delta→Recall 为 `2.839986 / 17.11553`，Parallel PPL 仅低约
`0.085%`。共同 step 8411 的近 20 点 loss 为 `2.79759 / 2.79914`；beta mean/std 为
`0.28944 / 0.16635` 与 `0.27827 / 0.16689`，gamma mean/std/饱和率为
`0.40398 / 0.29998 / 5.04%` 与 `0.18398 / 0.19339 / 0.78%`。dgx01 的最近 40 点
吞吐中位恢复至 `297.1k token/s`（均值 `271.5k`），节点 I/O wait 降到 6%--8%；
此前的吞吐下降确认是可逆的共址争用，不改训练配置。

为判断 recall rule 是否真正改善语言建模质量，已追加严格对齐的 GDN control：实验
`20260904-190833-gdn-aligned-340m-10bt-s3407-1b99be`、Slurm `37118`，冻结在同一 commit
`7eb73ca89411c54d4fe7a8ffb427df44e7709cfa`。它与两条 QGDN 完全共用数据、seed 3407、
T=4096、global batch 128、micro batch 8、gradient accumulation 2、fused loss、关闭
activation checkpointing、每 2000 step 验证 1600 条序列及 19073-step/10BT 终点。
作业已原子查重并记录，当前 `PENDING (Priority)`；现有半小时监控已扩展为三路同 step/token
对齐，不再把历史 mbs1/GA16、2560-sequence GDN 曲线当作严格配对结论。

同样配方的 Recall→Delta 也已补齐：实验
`20260904-194217-qgdn-recall-delta-340m-10bt-s3407-11ef45`、Slurm `37183`，模型配置
`qgdn_340M` 在冻结快照中明确对应 `recall_then_delta`，并使用当前 beta-style 独立随机
gamma 初始化。它与 GDN、Parallel、Delta→Recall 的 commit、数据、seed、批量、loss、
checkpointing、验证口径和 10BT 终点完全相同；作业已原子查重并记录，当前
`PENDING (Priority)`。半小时监控现扩展为四路严格对齐。

GDN control 随后在 dgx25 获配并进入训练，preflight JUnit `6/6`，截至 step 1391 的
loss、grad norm 和门控统计全部有限；最近 40 点吞吐中位为 `866.0k token/s`，峰值显存
`56.82 GB`。共同 step 1391 的近 20 点训练 loss 为 GDN `3.25359`、Parallel
`3.25499`、Delta→Recall `3.25668`，这是 GDN 略优的早期训练信号，需等待共同 step 2000
validation 再判断。该点 GDN alpha/beta mean/std 为 `0.69946 / 0.33857`、
`0.28540 / 0.16762`（gamma 不适用）；Parallel beta 与 gamma mean/std/饱和率为
`0.28319 / 0.16566`、`0.46390 / 0.31074 / 8.90%`；Delta→Recall 为
`0.26978 / 0.16564`、`0.25526 / 0.22255 / 1.48%`。Recall→Delta 仍正常排队。

Recall→Delta 随后在 dgx12 获配并开始训练，preflight JUnit 同样为 `6/6`，没有
`run.exitcode`、异常日志或非有限指标。首次四路可对齐的 step 1831 近 20 点训练 loss 为
Recall→Delta `3.15765`、GDN `3.15855`、Parallel `3.16001`、Delta→Recall `3.16071`；
最大差仅 `0.00305`，仍属于早期训练波动。该点 Recall→Delta 的 beta mean/std 为
`0.27871 / 0.16365`，gamma
mean/std/饱和率为 `0.45872 / 0.31359 / 9.27%`；Parallel 为
`0.27695 / 0.16287`、`0.45344 / 0.31355 / 8.90%`；Delta→Recall 为
`0.26491 / 0.16301`、`0.23995 / 0.22010 / 1.63%`。GDN alpha/beta mean/std 为
`0.69561 / 0.34028`、`0.27701 / 0.16435`，gamma 不适用。

随后完成的首次四路共同 step-2000 validation 中，Recall→Delta 的 loss/PPL 为
`3.140643 / 23.11873`，GDN 为 `3.141821 / 23.14598`，Parallel 为
`3.142662 / 23.16546`，Delta→Recall 为 `3.143802 / 23.19188`。Recall→Delta 暂时最好，
但相对 GDN 只低 `0.001178` loss、`0.02725` PPL（约 `0.118%`）；这是单个早期验证点，
不能据此确认 recall rule 的稳定收益。

三条已经到达 step 10000 的严格对齐 validation 也显示差距极小：GDN loss/PPL 为
`2.797239 / 16.39930`，Parallel 为 `2.796521 / 16.38754`，Delta→Recall 为
`2.798008 / 16.41192`。相对 GDN，Parallel 的 validation loss 低 `0.000718`，
Delta→Recall 高 `0.000769`；目前不能支持 recall rule 有明确收益或损害的结论。最近 40 点
吞吐中位为 GDN `866.8k`、Recall→Delta `310.0k`、Parallel `306.5k`、Delta→Recall
`299.5k token/s`；峰值显存分别为 `56.82 / 77.03 / 77.06 / 77.06 GB`。四个作业均在
持续更新 checkpoint，保留原配置继续运行。

Parallel Slurm 36311 已自然完成：Slurm `COMPLETED / 0:0`、`run.exitcode=0`，summary 为
`completed`，最终 step `19073`、prediction tokens `9,999,745,024`，preflight JUnit
`6/6`。训练计算时间 `8.998 h`、完整墙钟 `9.121 h`，对应 `308.69k / 304.53k token/s`；
峰值显存 `77.0617 GB/GPU`。最终 validation loss/PPL 为 `2.696578 / 14.82890`。末步
beta mean/std 为 `0.29295 / 0.17039`，gamma mean/std/饱和率为
`0.38905 / 0.29873 / 4.55%`，loss、grad norm 和全部门控统计有限。日志、JUnit、metrics、
run/summary 已回收，三项主要产物与远端 SHA-256 一致；约 4.14 GB checkpoint 和 1.38 GB
final model 按回收规则保留在远端。

Recall→Delta 的 step-4000 validation 进一步给出第二个同方向信号。四路 loss/PPL 为
Recall→Delta `2.972738 / 19.54537`、GDN `2.974415 / 19.57818`、Parallel
`2.975204 / 19.59361`、Delta→Recall `2.976156 / 19.61229`。Recall→Delta 相对 GDN
低 `0.001677` loss、约 `0.168%` PPL；它在 step 2000 与 4000 都暂时领先，但幅度仍小，
需要后续共同验证点确认。

共同 step 4301 的近 20 点训练 loss 为 Recall→Delta `2.93770`、GDN `2.93884`、
Parallel `2.93941`、Delta→Recall `2.94062`。该点 Recall→Delta beta 与 gamma
mean/std/饱和率为 `0.27915 / 0.16282`、`0.42653 / 0.30771 / 6.77%`；Parallel 为
`0.27715 / 0.16134`、`0.42266 / 0.30638 / 6.50%`；Delta→Recall 为
`0.26539 / 0.16079`、`0.20436 / 0.20538 / 1.20%`。GDN alpha/beta mean/std 为
`0.68817 / 0.34125`、`0.27588 / 0.16160`，gamma 不适用。其余三作业继续正常运行：
Delta→Recall/GDN/Recall→Delta 分别到达约 step `16321/17341/4301`，无异常日志或
非有限数值。

GDN control Slurm `37118` 随后自然完成：Slurm `COMPLETED / 0:0`、
`run.exitcode=0`、summary 为 `completed`，最终 step `19073`、prediction tokens
`9,999,745,024`。训练计算时间 `3.174 h`、完整墙钟 `3.277 h`，对应
`875.07k / 847.73k token/s`；峰值显存 `56.8215 GB/GPU`。最终 validation
loss/PPL 为 `2.697354 / 14.84041`，末步 alpha mean/std 为
`0.68316 / 0.34822`，beta mean/std 为 `0.28844 / 0.16865`。日志、JUnit、metrics
与 summary 已回收，大模型产物依规则保留在远端。

Delta→Recall Slurm `36312` 也已自然完成并回收：Slurm `COMPLETED / 0:0`、
`run.exitcode=0`、summary `completed`、preflight JUnit `6/6`，终点为 step `19073` 和
`9,999,745,024` prediction tokens。训练计算/完整墙钟为 `11.139 / 11.379 h`，有效
训练/墙钟吞吐为 `249.36k / 244.11k token/s`；峰值显存 `77.0617 GB/GPU`。最终
validation loss/PPL 为 `2.698694 / 14.86030`，比严格对齐 GDN 高 `0.001340` loss、
`0.134%` PPL，比 Parallel 高 `0.002115` loss、`0.212%` PPL。末步 beta mean/std 为
`0.27745 / 0.16940`，gamma mean/std/饱和率为 `0.16184 / 0.18106 / 0.481%`；loss、
grad norm 和门控统计均有限。dgx01 的共址争用使实际吞吐低于 Parallel，但不影响终态
模型效果比较。

## 固定 gamma=1 消融已启动

为直接检验可学习 gamma 是否将 recall 压弱，commit
`538712cc64d3fa4fa32aa348d8e252a03cb5b62f` 新增 Recall→Delta 和 Parallel 两个
340M 固定门控配置：`recall_gate="fixed"`、`recall_init=1.0`。这是严格的
`gamma_t ≡ 1`，gamma 不是可训练参数，不引入 recall-gate 投影参数。测试还确认
两个模型与各自可学习 gamma 对照的共享 backbone 在 seed 3407 下保持相同初始化。

Commit `49898cb61973bf619c3256436b0d7bfc43e18326` 将 CUDA 顺序门禁扩展到
`gamma=0.1/1.0 × 三种更新顺序`，仍逐一核对 BF16 输出、末状态和全部输入梯度。
提交前 CPU 聚焦回归为 `11 passed / 6 skipped / 174 deselected`。两个不可变快照作业均已
通过 H800 preflight：`10 passed / 0 failed / 0 errors`，其中包含 gamma=1 三顺序的
CUDA 输出、状态和全梯度检查：

| 模型 | 实验 | Slurm | 节点 | 当前状态 |
|---|---|---:|---|---|
| Recall→Delta, gamma=1 | `20260904-231927-qgdn-recall-delta-gamma-one-340m-10bt-s3407-2b34ac` | 37379 | dgx24 | RUNNING |
| Parallel, gamma=1 | `20260904-231927-qgdn-parallel-gamma-one-340m-10bt-s3407-63bf96` | 37380 | dgx25 | RUNNING |

两者均使用与现有严格对齐实验相同的 FineWeb 数据、seed 3407、8×H800、
T=4096、micro batch 8、global batch 128、gradient accumulation 2、fused loss、关闭
activation checkpointing、每 2000 step 验证 1600 条序列，以 step 19073 /
`9,999,745,024` prediction tokens 为终点。生产路径仍是虚拟 2T，
`QGDN_USE_PHYSICAL_T=False`。训练日志中 gamma 统计必须始终严格为
mean `1.0`、std `0.0`、saturated fraction `1.0`；任一偏离都视为配置或实现错误。

首批训练指标已进一步确认配置生效。两路的参数量均为 `344,353,984`、
`recall_parameters=0`，共享初始化 SHA-256 均为
`2ce1c8431c5ab5ff8080b71fe92e84a2ef93791222d5fb7d29dd91a2cf6c36fb`。初始
validation loss/PPL 分别为 Recall→Delta `10.532466 / 37513.86`、Parallel
`10.532462 / 37513.70`。排除首次编译 step 后，Recall→Delta 最新稳态样本约
`316.5k token/s / 75.68 GB`，Parallel 约 `313.2k token/s / 75.72 GB`；loss、grad norm、
beta 与 gamma 统计全部有限，gamma 在所有已观测 step 上严格为 `1.0 / 0.0 / 100%`。

两路已通过首个共同 step-2000 validation，固定 gamma 统计仍逐点严格保持
`1.0 / 0.0 / 100%`，没有 OOM、非有限值或异常日志。Recall→Delta 的 loss/PPL 为
`3.144389 / 23.20549`，比同顺序 beta-style gamma 的 `3.140643 / 23.11873` 高约
`0.375%` PPL；Parallel 为 `3.146042 / 23.24387`，比同顺序 beta-style gamma 的
`3.142662 / 23.16546` 高约 `0.338%` PPL。这是单个早期验证点，只构成固定强 recall
略差的初步信号，需等后续共同节点确认。

## 可训练 gamma∼U(0.85, 0.95) 消融已启动

Commit `9e381f0ecb1a8c2fcd8397446003cf6fdf0530b7` 新增 Recall→Delta 和 Parallel
两个高 gamma 可训练配置。每层、每个 head 在 gate 空间独立采样
`gamma∼U(0.85,0.95)`，再用 logit 写入可训练 bias；token-wise projection weight 从零开始但
仍可训练。因此初始 gamma 对 token 恒定但在 head/layer 之间随机，并且从第一次更新起
可学出 token 依赖；初始值不会因输入激活超出指定区间。

CPU 聚焦回归为 `7 passed / 186 deselected`，确认初值范围、随机性、weight/bias 的
有限非零梯度，以及与同顺序 beta-style gamma 模型的共享 backbone 初始化逐参数一致。
CUDA 门禁同时增加 `gamma=0.9` 作用点，与 `0.1/1.0` 一起覆盖三种更新顺序。

| 模型 | 实验 | Slurm | 节点 | 当前状态 |
|---|---|---:|---|---|
| Recall→Delta, trainable gamma∼U(0.85,0.95) | `20260904-234011-qgdn-recall-delta-gamma-uniform-085-095-340m-10bt-s3407-78f438` | 37413 | dgx37 | RUNNING; preflight 13/13 |
| Parallel, trainable gamma∼U(0.85,0.95) | `20260904-234011-qgdn-parallel-gamma-uniform-085-095-340m-10bt-s3407-374799` | 37414 | dgx38 | RUNNING; preflight 13/13 |

两个作业仍完全沿用对齐的 10BT recipe 和虚拟 2T 默认路径。高 gamma 区间只是
初始条件，训练后 gamma 可自由离开该区间；后续需与同顺序 beta-style 初始化、固定
gamma=1 和 GDN 在相同 step/token 及共同 validation 节点对齐比较。
两路 H800 preflight 均为 `13 passed / 0 failed / 0 errors`，包括 gamma=0.9 下三顺序
BF16 输出、末状态和全输入梯度门禁。

首批训练指标确认了实际 340M 配置：两路均为 `344,681,984` 参数、
`recall_parameters=328,000`，共享初始化 SHA-256 与既有对照同为
`2ce1c8431c5ab5ff8080b71fe92e84a2ef93791222d5fb7d29dd91a2cf6c36fb`。初始
validation loss/PPL 为 Recall→Delta `10.532419 / 37512.09`、Parallel
`10.532419 / 37512.12`。step 1 两路 gamma mean/std/饱和率完全相同，为
`0.898251 / 0.028024 / 0%`；到共同 step 31，Recall→Delta 为
`0.888909 / 0.031800 / 0.714%`，Parallel 为 `0.888854 / 0.031824 / 0.657%`。
这说明 gate 已开始产生 token 依赖，暂无塌缩或异常饱和。同一 step 31 的 loss 为
`7.54924 / 7.54941`，稳态样本约为 `312.1k / 309.8k token/s`，峰值显存为
`77.03 / 77.06 GB/GPU`；loss、grad norm 和 beta/gamma 统计全部有限。

到共同 step `1291`，Recall→Delta 与 Parallel 的近 20 点训练 loss 为
`3.28581 / 3.28671`，仍基本重合；beta mean/std 分别为
`0.27866 / 0.16357` 与 `0.27854 / 0.16267`。gamma 已从高初值快速展宽：两路
mean/std/饱和率分别为 `0.63251 / 0.30221 / 19.46%` 与
`0.63557 / 0.30007 / 19.11%`。所有值和梯度仍有限、两顺序轨迹高度一致，当前不是数值
崩溃；但约五分之一 token gate 已进入饱和区，属于必须继续观察的明显极化，不能再描述为
“暂无异常饱和”。两路最近 20 点吞吐约 `312.34k / 309.42k token/s`，峰值显存不变，
完整 step-1000 checkpoint 已写入远端。

## 当前下一步

1. 物理 T 下一候选先拆分 dependency-only state adjoint 与 chunk-parallel transition VJP；
   在 CPU/FP64 合约中明确验证边界 state adjoint 和全部 factor/value/decay 梯度，再考虑 CUDA。
2. 36311、36312 和 37118 已成功完成并回收，不得重提；冻结保留仍在运行的
   37183/37379/37380/37413/37414，继续检查 loss、grad norm、门控统计、吞吐、峰值显存、
   日志新鲜度和 checkpoint 完整性。固定 gamma 两路必须保持 `1/0/100%` 统计。
3. 对瞬态 Slurm/NCCL/launcher/存储故障可在同一冻结配置上恢复；OOM、非有限数值或需要改变
   micro batch/科学配置时先保留证据并停止，不盲目重跑。
4. 其余五个作业到终态后同样回收 JUnit、退出码、日志、metrics 与 summary；对齐比较
   高区间可训练 gamma、固定 gamma=1、beta-style 可学习 gamma 与 GDN，判断强 recall 是改善还是伤害 loss/PPL。

远程开发仓库为 `/work/projects/memos-b3/code/wangzr/828`，分支 `QGDN`，专用环境为 `/work/projects/memos-b3/software/miniconda3/envs/wangzr-qgdn`。GitHub 为 `cafeii/828`。
