# QR-GDN 审查记录

## 旧方向快照审查

- 快照：`20260902-200157-dtjqc-rank2-state-gpu-pad-retest-902a43`
- 结论：只完成准备，从未提交到 Slurm。
- 依据：manifest 中 `job_id=null`；不存在 `submission.lock` 与 job-id 文件；`squeue`、`sacct` 均无唯一任务名记录。
- 处置：保留为历史，不提交、不恢复。

## 联合理论里程碑

- Commit：`6ac9421c501ef63e6f274689204c4577f6d766b7`
- 固定了联合近端目标、KV 到 QR 的块下三角耦合以及两倍隐状态口径。
- 因果读出修正：主 KV 分支沿用现有 GDN 的更新后读取；QR 残差读取更新前状态，通过零初始化 `tanh` 门融合，避免当前 token 自回显并保证初始化时严格退化为 GDN。

## FP64 参考实现里程碑

- Commit：`2cf36020eec5438cda33b8b57d5d43ca7386f195`
- 命令：`CUDA_VISIBLE_DEVICES= PYTHONPATH=model:third_party/flash-linear-attention python -m pytest tests/test_qr_gdn_reference.py -q`
- 结果：6 passed。
- 覆盖：联合目标直接求解、显式与块仿射递推的前向/状态/梯度一致性、关闭 QR 时的 GDN 精确退化、QR 查询残差收缩、当前 token 无 QR 自回显、分段恢复一致性。
- 首轮测试发现仿射参考中 KV 写入门少了一个广播维度；修复后完整重跑通过。

## 模型接入里程碑

- Commit：`193306906e29621507e27146c28142e4b1b879dd`
- 注册 `qr_gdn_340M` 与 tiny 配置，加入独立的 QR 遗忘门、写入门和零初始化 `tanh` 读出门；状态缓存使用 `(M^{KV}, M^{QR})` 对。
- 共享初始化检查：同 seed 下全部 GDN 共享参数逐位相等；QR 读出为零时完整 tiny GPT 输出逐位相等。
- CPU 验证：QR-GDN 新测试、FP64 参考、既有 QGDN/DT/JQC 回归及配置注册共 27 项通过；GPU 测试尚未执行。
- 首轮模型测试暴露两项测试装配问题：未把 tiny 模型切到 `naive`，以及比较模型二次初始化时未重置测试 RNG；修复测试后重跑通过，不涉及模型公式修改。

## 当前门槛（更新）

模型参考路径已接入，但 chunk 模式仍明确失败关闭。尚未完成并行 kernel、GPU、BF16、8 卡 DDP、吞吐和峰值显存验证，因此正式训练继续被阻止，本轮尚未调用 `sbatch`。

## 物理时间 rank-2 分块代数里程碑

- Commit：`e92d4579e5df29ad0dca09205b0fd56f5a2f3fdc`
- 将联合状态堆叠为 $[M^{KV};M^{QR}]$，每个真实 token 表示为两个同时发生的低秩修正和两个通道各自的标量遗忘；外部值只写入 KV 半区。
- 构造了支持双通道向量遗忘的精确 block-WY 变换，保持物理序列长度为 $T$，每个长度为 $C$ 的 chunk 紧凑秩为 $2C$；没有 `2T` 虚拟序列或 $2K\times2K$ 稠密转移。
- 验证：显式递推与 rank-2 因子的输出、双终态及全部梯度一致；chunk size 1/2/4/8 的终态一致；单 chunk 的全部状态梯度一致。新增 6 项通过，合并 CPU 回归共 33 项通过。
- 当前限制：这一步只固定了并行代数和可微 PyTorch oracle；尚未接入 GPU 状态传播与 chunk 内输出 kernel，正式 `chunk` 路径仍失败关闭。

## GPU 状态传播诊断提交

- Commit：`b51e08f1ff31f0b5ffdfcddf5378118beb320863`
- 实验：`20260903-012356-qr-gdn-state-gpu-f95c41`；Slurm job `34756`。
- 资源：碎片节点 `tko-b3-nv-dgx07` 的 1 GPU、4 CPU、16G、20 分钟，不占用空闲整节点。
- 提交前复核：开发工作树干净、完整唯一任务名无重复、manifest job-id 为空、submission.lock 与 job-id 文件均不存在；已原子创建锁并立即登记 job-id。
- 目的：验证紧凑向量遗忘 chunk 变换的 FP32/BF16 GPU 状态传播、初态和可选终态；不训练模型。
- 正式训练仍受阻：chunk 内输出和反向 kernel 尚未完成。

## GPU 状态传播诊断终态审查（job 34756）

- Slurm：`FAILED`，exit code `1:0`，elapsed `00:01:13`；`run.exitcode=1`，`run.json` 存在，必需产物 `result.json` 缺失。日志已经回收，不能判为成功。
- 测试进度：8 passed、1 failed。第一处实质错误发生在 BF16 测试的 PyTorch oracle：BF16 紧凑因子与 FP32 累积状态直接进入 `einsum`，在调用被测 kernel 的比较之前失败；因此该次结果没有建立 BF16 kernel 错误。
- 修复：oracle 明确把已量化的紧凑变换转为 FP32 后累计，commit `a9af58aaea48bc9149ce7cbc11c44dd882f13cf7`；无 GPU 的开发节点回归为 16 passed、3 GPU-skipped。
- 重试：本次心跳已使用一次 `sbatch`，遵守上限，不在本轮再次提交；下次心跳可用新快照重跑这一项诊断。
- 正式训练仍受阻：状态传播 GPU 诊断尚未通过，且 chunk 内输出与反向 kernel 仍未完成。

## GPU 状态传播诊断重试提交

- 实验：`20260903-020132-qr-gdn-state-gpu-retry-a4c2b3`；Slurm job `34791`；快照 commit `8227ce96135a721695c5f1b3535dd3aec7dc7c3b`。
- 这是 job 34756 的一次诊断重试，包含 BF16 PyTorch oracle dtype 修复。
- 资源：碎片节点 `tko-b3-nv-dgx04` 已占 7/8 GPU，申请剩余 1 GPU、4 CPU、16G、20 分钟。
- 提交前开发树干净，精确完整任务名无重复，manifest/job-id/lock 均为空；已加锁并立即登记 job-id。
- 本轮只执行这一项 `sbatch`，不提交正式训练。

## GPU 状态传播诊断重试终态审查（job 34791）

- Slurm `COMPLETED`，exit code `0:0`，elapsed `00:01:16`；`run.exitcode=0`。
- `run.json`、完整日志和必需产物 `outputs/result.json` 均已核对并回收。
- H800 结果：9 passed；FP32 与 BF16 存储/FP32 累积、带/不带初态、可选终态均与 PyTorch oracle 相符。
- 结构约束：真实物理时间步；没有 `2T` 虚拟序列，也没有稠密 $2K\times2K$ 转移。
- 状态传播 GPU 门槛通过。正式训练仍受阻于 chunk 内输出与反向 kernel，以及后续稳定性、恢复、8 卡 DDP 和吞吐/显存门槛。

## Chunk 内输出前向实现候选

- Commit：`dc9742ca8760a256a6d95368f8cd616c652815bb`。
- 新增 Triton chunk-local 输出 kernel：不同真实 chunk 与 value tile 并行；每个 kernel program 只在固定 chunk 内执行有限 token 递推。没有逐 token Python 循环、`2T` 虚拟序列或稠密 $2K\times2K$ 转移。
- 因果读出保持固定：KV 读取更新后状态，QR 残差读取更新前状态，因此当前 token 无 QR 自回显。
- 新增 FP32/BF16、零/非零初态的 token oracle 对照测试，并把它加入 GPU 审查入口。
- 开发节点验证：Python 编译通过；16 passed、7 GPU-skipped。GPU 编译与数值对照尚待下一次心跳提交诊断，本轮 `sbatch` 配额已用完。
- 反向 kernel 尚未实现，模型 `chunk` 正式路径仍保持失败关闭，正式训练继续阻止。

## Chunk 内输出 GPU 诊断提交

- 实验：`20260903-024229-qr-gdn-chunk-output-gpu-c9bd61`；Slurm job `34840`；快照 commit `4b52c867fc1f210a9161c2cd1a023fbaffd50244`。
- 资源：碎片节点 `tko-b3-nv-dgx07` 已占 7/8 GPU，申请剩余 1 GPU、4 CPU、16G、20 分钟。
- 提交前开发树干净，精确任务名无重复，manifest/job-id/lock 为空；已加锁并登记 job-id。
- 验证范围：既有状态传播与 rank-2 扫描，加上 chunk 内输出的 FP32/BF16、零/非零初态四种组合。
- 本轮只执行这一项 `sbatch`，不提交正式训练。

## Chunk 内输出 GPU 诊断终态审查（job 34840）

- Slurm `COMPLETED`，exit code `0:0`，elapsed `00:01:15`；`run.exitcode=0`。
- `run.json`、日志、`outputs/result.json` 均完整并已回收。
- H800：13 passed。新增 chunk 输出覆盖 FP32/BF16 以及零/非零初态，均与 FP32 逐 token oracle 相符；既有状态传播和 rank-2 扫描也通过。
- 时间维保持真实 $T$，不同物理 chunk 并行；没有逐 token Python 训练循环、`2T` 展开或稠密 $2K\times2K$ 转移。
- 并行前向门槛通过。正式训练仍受阻于反向 kernel，以及稳定性、恢复、8 卡 DDP 和吞吐/显存验证。

## 并行自动微分候选与 GPU 诊断提交

- Commit：`0d2623790f2efd569fb5109a80fdc01364d93f97`。
- 利用块下三角依赖拆成三次生产级 GDN chunk 调用：原生 KV 更新后读、移位 KV 更新得到更新前回忆、移位 QR 更新得到更新前 QR 读出；三次调用都复用 FLA 的正式自定义反向。
- 保持真实 $T$ 个物理位置，无逐 token Python 训练循环、无 `2T` 虚拟序列、无稠密 $2K\times2K$ 转移。QR 读出门为零时，主 KV 调用和标准 GDN 完全相同，开发检查逐位一致。
- 开发检查：FP32/BF16 前向与终态通过；激活专用 Conda 编译器环境后，FP32 反向与 FP64/PyTorch 参考通过。未激活 Conda 时 TileLang 因找不到 `nvcc` 被拒绝，触发 Hopper/Triton 3.5 的已知保护；这确认正式作业必须沿用已验证的 Conda 激活流程，不是模型数学错误。
- 实验：`20260903-033439-qr-gdn-parallel-backward-gpu-599d2d`，Slurm job `34860`，快照 commit 同上。资源为碎片节点 `tko-b3-nv-dgx04`（提交时 5/8 GPU 已占用）的 1 GPU、4 CPU、16G、20 分钟。
- 提交前核对开发树干净、完整唯一任务名无重复、manifest/job-id/lock 为空；已原子加锁并登记 job-id。本轮只执行此一次 `sbatch`。
- 当前候选需要三次完整 chunk 调用，吞吐可能低于 80% 门槛；即使数值诊断通过，也必须先接入模型并基准吞吐/显存，未达标则融合优化，不提交正式训练。

## 并行自动微分 GPU 诊断终态审查（job 34860）

- Slurm `COMPLETED`，exit code `0:0`，elapsed `00:03:31`；`run.exitcode=0`。
- `run.json`、完整日志与 `outputs/result.json` 均已核对并回收。首次回收时 required-output 元数据误写成 `outputs/result.json`，检查器因而在实际文件已存在时报告缺失；将 manifest 纠正为相对 outputs 目录的 `result.json` 后，状态与回收检查均正式通过，未重提作业。
- H800 共 17 passed：既有状态传播、chunk 输出与 rank-2 扫描继续通过；新增三调用并行候选的 FP32/BF16 前向、双终态、FP32 全输入反向，以及 QR 读出关闭时的原生 GDN 逐位一致性均通过。
- 结论：生产 FLA 自定义反向可以承载当前块下三角分解，数值和自动微分门槛通过。正式训练仍未开始；下一步接入模型 `chunk` 路径，再做稳定性/恢复、8 卡 DDP 和 340M 吞吐/峰值显存基准。三次完整 chunk 调用的性能风险仍需实测，未达到 GDN 80% 时必须融合优化。

## 模型并行 chunk 接入候选

- Commit：`70457db631c6c531b89409b225f863525a54d8e7`。
- QR-GDN 训练态 `chunk` 路径已接入三调用并行自动微分实现；短序列评测/增量解码暂用显式参考，直到专用 fused recurrent 双状态 kernel 完成。
- 新增模型级测试：并行 chunk 与 naive 参考输出相符，并检查所有实际梯度有限且 QR 读出投影取得非零梯度。
- CPU/静态回归：10 passed、5 个 CUDA 测试 skipped；Python 编译与 diff 检查通过。模型级 GPU 测试尚未作为 Slurm 诊断运行，本轮已使用 job 34860 的唯一 `sbatch` 配额，留待下一次心跳。
- 正式训练继续阻止；模型级 GPU、稳定性/恢复、8 卡 DDP 和 340M 吞吐/显存门槛均未完成。

## 模型级并行 chunk GPU 诊断提交

- 实验：`20260903-041940-qr-gdn-model-chunk-gpu-9550c1`；Slurm job `34875`；快照 commit `909b99f2baf69edb24c9e9d444427596f8323734`。
- 资源：碎片节点 `tko-b3-nv-dgx04`（提交时 4/8 GPU 已占用）的 1 GPU、4 CPU、16G、20 分钟。
- 验证范围：全部已有 QR-GDN GPU 内核测试，外加模型级 `chunk` 与 `naive` 前向一致性、训练反向有限性和 QR 读出投影非零梯度。
- 提交前开发树干净，完整唯一任务名无重复，manifest/job-id/lock 均为空；已原子加锁并登记 job-id。本轮只执行这一项 `sbatch`，不提交正式训练。

## 模型级并行 chunk GPU 诊断终态审查（job 34875）

- Slurm `COMPLETED`，exit code `0:0`，elapsed `00:01:23`；`run.exitcode=0`。
- `run.json`、日志和必需产物 `outputs/result.json` 完整并已回收。
- H800 共 18 passed：模型级 QR-GDN `chunk` 输出与 naive 参考一致，完整反向梯度有限，QR 读出投影取得非零梯度；先前状态、输出、FP32/BF16 和底层自动微分检查继续通过。
- 模型并行路径门槛通过。正式训练仍未开始；后续门槛为近极端门控稳定性、分段恢复/缓存一致性、340M 吞吐与峰值显存，以及 8 卡 DDP smoke。

## 稳定性与性能门槛工具

- Commit：`1ae71f28271d84a1ae13e4015b69ac267916fad3`。
- 新增 GPU 检查：128 token 单次执行与两个 64 token 分段延续的一致性、近极端遗忘/写入/读出门下的前向和反向、模型级 BF16 autocast 反向有限性。
- 新增同卡 340M 训练步基准：GDN 与 QR-GDN 均使用序列 4096、micro batch 1、BF16 mixed、激活检查点、AdamW 和完整前向/交叉熵/反向/裁剪/更新；报告参数量、FP32 状态字节、逐步耗时、吞吐和峰值显存，并自动判断 QR/GDN 吞吐是否达到 80%。
- Meta 参数检查：GDN 344,353,984；QR-GDN 345,338,304，新增 984,320。Python 编译、CPU 回归和配置构造通过。
- 本轮已使用 job 34875 的唯一 `sbatch`，因此新工具留待下次心跳在 H800 上运行。正式训练继续阻止。

## 稳定性与 340M 性能诊断提交

- 实验：`20260903-045704-qr-gdn-stability-performance-39343a`；Slurm job `34888`；快照 commit `e58a27f76eb4b137ed762f71250a6698e1c500bc`。
- 资源：碎片节点 `tko-b3-nv-dgx03`（提交时 6/8 GPU 已占用）的 1 GPU、8 CPU、64G、1 小时。
- 验证先运行分段恢复、近极端门控和 BF16 模型反向，再在同一 H800 上顺序测量 GDN 340M 与 QR-GDN 340M：sequence length 4096、micro batch 1、BF16 mixed、激活检查点，1 个 warmup 和 3 个正式训练步。
- 吞吐门槛固定为 QR-GDN/GDN ≥ 0.8，同时记录参数、两倍状态字节和峰值显存。低于门槛只记录为需优化，不启动完整训练。
- 提交前开发树干净、完整唯一任务名无重复，manifest/job-id/lock 均为空；已原子加锁并登记 job-id。本轮只执行这一项 `sbatch`。

## 稳定性与 340M 性能诊断终态审查（job 34888）

- Slurm `COMPLETED`，exit code `0:0`，elapsed `00:07:06`；`run.exitcode=0`。`run.json`、日志和 `outputs/result.json` 完整并已回收。
- H800 稳定性套件 8 passed：分段续算、近极端门控前向/反向和 BF16 模型反向均通过，没有新增数值错误。
- 同卡 340M、sequence 4096、micro batch 1、BF16 mixed、激活检查点的实测吞吐：GDN 18,201.02 token/s，QR-GDN 11,465.70 token/s，比例 62.99%，未达到 80% 门槛。
- GDN/QR-GDN 峰值显存分别为 6.6846/6.6897 GB；FP32 递归状态字节分别为 5,242,880/10,485,760，确认 QR 隐状态严格为两倍，当前显存增加主要未体现在模型训练峰值上。
- 正式训练继续阻止。当前三次完整 chunk 调用的性能代价过高，先做融合优化，再重新验证前向、反向与吞吐。

## 两调用性能优化候选

- Commit：`92d5f22c32e10d590f748e742c7e9818e4cbb1e3`。
- 原生 KV chunk 已经产生有效增量 $\delta_t$ 且输出 $M_t^{KV\top}q_t$。利用
  $M_{t-1}^{KV\top}q_t=(M_t^{KV\top}q_t-\langle q_t,k_t\rangle\delta_t)/\alpha_t^{KV}$，直接恢复 QR 写入目标，删除一次移位 KV 全状态扫描。
- 扩展 vendored FLA 自定义自动微分接口，使有效增量可被组合 recurrence 读取，且其外部梯度在状态反向前正确并入；默认 GDN API 行为保持不变。
- 训练路径由三次降为两次生产级 chunk 调用；仍保持真实 $T$、无逐 token Python 循环、无 `2T` 序列和无稠密 $2K\times2K$ 转移。
- CPU/静态回归 16 passed，Python 编译和 diff 检查通过。该代数恢复与扩展反向尚须下一次心跳在 H800 上验证并重测吞吐；本次已提交 job 34888，遵守每次心跳最多一次 `sbatch`，不追加 GPU 作业。

## 两调用候选 GPU 数值与性能复测提交

- 实验：`20260903-054145-qr-gdn-two-call-gpu-perf-4d3e5d`；Slurm job `34895`；快照 commit `415589690f51cbbf28c678eddf5552c696c2af52`，核心实现 commit `92d5f22c32e10d590f748e742c7e9818e4cbb1e3`。
- 资源：best-fit 碎片节点 `tko-b3-nv-dgx04`（提交时 4/8 GPU 已占用），申请 1 GPU、8 CPU、64G、1 小时。
- 先重跑全部并行前向/反向、BF16、极端门控和分段续算检查，再以相同 H800、340M、sequence 4096、micro batch 1、BF16 mixed、激活检查点复测 GDN 与两调用 QR-GDN 的吞吐和峰值显存。
- 提交前开发树干净、完整唯一任务名无重复、manifest/job-id/lock 均为空；已原子加锁并立即登记 job-id。本轮只执行这一项 `sbatch`，正式训练仍受 80% 吞吐、8 卡 DDP 等门槛阻止。

## 两调用候选 GPU 诊断失败审查（job 34895）

- Slurm `FAILED`，exit code `1:0`，elapsed `00:01:49`；`run.exitcode=1`。`run.json` 和完整日志已回收，因测试先失败而没有生成必需的 `result.json`。
- GPU 测试为 5 passed、3 failed。第一处实质错误是 BF16 路径中代数恢复的 `recall` 保持 FP32，第二次 FLA 调用由此形成 BF16/FP32 Triton dot 类型不一致。
- 更关键的设计错误是 $M_{t-1}^{KV\top}q_t=(M_t^{KV\top}q_t-\langle q_t,k_t\rangle\delta_t)/\alpha_t^{KV}$ 在 $\alpha_t^{KV}$ 很小时需要相消后除以小数；极端遗忘测试出现明显放大。这不是可以通过放宽容差接受的路径，因此废弃该恢复公式。
- 没有运行性能基准，也没有提交正式训练。本轮 `sbatch` 配额已由 job 34895 使用，不在同一心跳重试。

## 稳定更新前读出候选

- Commit：`60db87b1ac67e34d33b973e0ad8240a0fc6f3fd2`。
- KV chunk 在一次状态扫描后已有 chunk 初态和有效写入。新增一个移位查询输出 kernel：在位置 $t-1$ 用 $q_t$ 读取 $M_{t-1}^{KV}$，再把结果右移，从而直接得到 $r_t$；不做相消，也不除以 $\alpha_t^{KV}$。
- 自定义反向在移位查询流上复用已验证的 GDN 反向，并在 L2 归一化反向之前把查询、键、值、门控和初态梯度合并。训练仍是两次完整 chunk 状态扫描，另加一个只读输出 kernel；没有逐 token Python 循环、`2T` 展开或稠密 $2K\times2K$ 转移。
- Python 编译、diff 检查和 CPU 回归 16 passed。GPU 前向、BF16、极端门控、反向及吞吐需下一次心跳重新验证。

## 稳定更新前读出候选 GPU 数值与性能复测提交

- 实验：20260903-062036-qr-gdn-stable-preread-gpu-perf-0c76bd；Slurm job 34914；快照 commit 40f0c7d98059a885871a4c9a0ca0aceca5b62042，核心实现 commit 60db87b1ac67e34d33b973e0ad8240a0fc6f3fd2。
- 资源：best-fit 碎片节点 tko-b3-nv-dgx04（提交前 4/8 GPU 已占用），申请 1 GPU、8 CPU、64G、1 小时。
- 先运行全部 GPU 数值、BF16、近极端门控、分段恢复和反向检查，再按相同 H800、340M、sequence 4096、micro batch 1、BF16 mixed、激活检查点复测吞吐和峰值显存。
- 提交前开发树干净，完整唯一任务名无重复，manifest/job-id/lock 均为空；已原子加锁并登记 job-id。本轮只执行这一项 sbatch，正式训练仍受 80% 吞吐与 8 卡 DDP 门槛阻止。

## 稳定更新前读出候选终态审查（job 34914）

- Slurm COMPLETED，exit code 0:0，elapsed 00:03:23；run.exitcode=0。run.json、日志和必需的 result.json 均完整并已回收。
- H800 数值套件 8 passed：FP32/BF16、近极端门控、反向、初末状态和分段续算均通过；稳定读取方案消除了除以小 alpha^KV 的病态恢复。
- 同卡 340M、sequence 4096、micro batch 1、BF16 mixed、激活检查点：GDN 18,314.69 token/s，QR-GDN 12,462.27 token/s，比例 68.05%，仍低于 80% 门槛。
- 相比三调用原型的 62.99%，吞吐比例提高 5.05 个百分点，但第二次完整 QR 状态扫描及附加反向仍过重。峰值显存 GDN/QR-GDN 为 6.6846/6.6713 GB，没有异常增长；递归状态字节仍严格为两倍。
- 正式训练继续阻止。下一步实现联合双状态融合 kernel，避免两套独立 chunk 调用及其重复调度/中间量；完成数值回归后再在后续心跳提交一次 GPU 复测。

## Rank-2 融合前向可行性诊断提交

- 工具 commit eff4c3f0ce4389dde0c7b59c7c9fd1350d9a7c7b；实验 20260903-070207-qr-gdn-fused-forward-profile-581312；Slurm job 34918。
- 在 B1、T4096、H16、K64、V64、BF16、chunk 64 下，分别测量原生 GDN、当前两调用 QR、rank-2 变换构造加融合 kernel、预先构造后的融合 kernel 本体，并核对融合输出与当前路径误差。
- 资源为 best-fit 碎片节点 tko-b3-nv-dgx04 的 1 GPU、4 CPU、32G、30 分钟；提交前开发树干净，唯一任务名、job-id 和 submission.lock 均无重复。本轮只执行这一项 sbatch。

## Rank-2 融合前向可行性诊断终态审查（job 34918）

- Slurm COMPLETED，exit code 0:0，elapsed 00:00:19；run.exitcode=0。run.json、日志和 result.json 完整并已回收。
- B1/T4096/H16/K64/V64/BF16 下，原生 GDN、当前两调用 QR、rank-2 kernel-only、rank-2 完整前向平均耗时分别为 0.534、1.329、1.441、5.312 ms。
- 通用 rank-2 kernel 本体速度仅为当前路径的 92.26%；加入 block-WY 因子构造后仅为 25.02%，峰值临时显存约 1.617 GB。输出误差 max 0.001953、mean 0.000164，在 BF16 预期范围内，但性能不合格。
- 结论：rank-2 结构继续保留作理论与数值参考，不作为正式训练 kernel。生产优化改为保留原生 KV 前向和因果两调用分解，专门融合 KV 正常读出与更新前回忆的两路反向贡献，使 KV 状态梯度只扫描一次；这同时保留关闭 QR 读出时的原生 GDN 精确路径。
- 本轮 sbatch 配额已由 job 34918 使用，不追加 GPU 作业；正式训练仍受 80% 吞吐和 8 卡 DDP 门槛阻止。


## 专用双读反向实现与 GPU 性能诊断提交

- 核心实现 commit `f2eca23807dfeee39f8920ba78f9b191314085b2`：保留原生 KV 前向和稳定的更新前 QR 读取；将 KV 正常读出与移位 QR 回忆的状态梯度合并进一次逆向状态扫描。各自查询梯度仍独立计算，值、写入、遗忘和初态梯度在公共链路内求和。
- Python 编译、diff 检查和 CPU/静态回归通过：16 passed、8 个 CUDA 测试按预期 skipped。
- 实验 `20260903-074232-qr-gdn-dual-read-backward-gpu-perf-e641e7`；Slurm job `34925`；资源为 best-fit 碎片节点 `tko-b3-nv-dgx09`（提交时 4/8 GPU 已占用）的 1 GPU、8 CPU、64G、1 小时。
- 作业先运行全部 H800 数值、BF16、极端门控、分段恢复和反向检查，再用相同 340M、sequence 4096、micro batch 1、BF16 mixed、激活检查点配置重测 GDN/QR-GDN 吞吐与峰值显存。
- 提交前开发树和实验快照均干净，完整唯一任务名、job-id 与 `submission.lock` 无重复；已原子加锁并登记 job-id。本轮只执行这一项 `sbatch`。正式训练仍受 80% 吞吐与 8 卡 DDP 门槛阻止。


## 专用双读反向诊断终态审查（job 34925）

- Slurm `COMPLETED`，exit code `0:0`，elapsed `00:08:09`；`run.exitcode=0`。`run.json`、日志和必需的 `result.json` 完整并已回收。
- H800 数值套件 8 passed：FP32/BF16、完整反向、初末状态、近极端门控和分段续算均通过；共享逆向状态扫描没有改变 QR-GDN 数值语义。
- 同卡 340M、sequence 4096、micro batch 1、BF16 mixed、激活检查点：GDN 18,432.88 token/s，QR-GDN 12,664.14 token/s，比例 68.70%，仍低于 80% 门槛。
- 与上一稳定两调用方案的 68.05% 相比仅提高约 0.65 个百分点，说明重复逆向状态扫描不是主要瓶颈。峰值显存 GDN/QR-GDN 为 6.6846/6.6855 GB，没有异常增长；递归状态仍严格为两倍。
- 正式训练继续阻止。下一候选将合并两路读出的局部 `dv` 与 `dq/k/g` 梯度 kernel，去掉剩余辅助 kernel 和零 `dh` 中间量；如果仍无法达到 80%，需要重新评估双状态机制的计算结构，而不能放宽门槛。

## 双读局部梯度融合实现

- 核心实现 commit `06a2e931420d29f6e189b862789a2f8d2fe80040`：在现有共享逆向状态扫描之上，将主 KV 正常读出与移位 QR 更新前读出的局部 `dv`、`dq/k/g` 贡献分别合并到一次 kernel 调用，删除第二次局部梯度 kernel 和全零 `dh` 张量。
- 梯度恒等关系保持不变：公共状态递推的 `dh` 项只计一次；主/QR 两路读出各自的查询梯度独立输出，它们对 `k` 和衰减门的贡献在同一 kernel 内相加。未改变前向、QR 因果时序或普通 GDN 默认路径。
- Python 编译、`git diff --check` 和 CPU/静态回归通过：16 passed，8 个 CUDA 测试按预期 skipped。H800 编译、FP32/BF16 数值与 340M 吞吐尚待本轮唯一 GPU 诊断；正式训练继续阻止。

## 双读局部梯度融合 GPU 性能诊断提交

- 实验 `20260903-083340-qr-gdn-fused-local-grad-gpu-perf-67c06c`；Slurm job `34936`；源码实现 commit `06a2e931420d29f6e189b862789a2f8d2fe80040`，实验快照 commit `ce5eff548c8e871ddc202dad8f26c549819bc8fd`。
- 按集群 best-fit 规则定向到碎片节点 `tko-b3-nv-dgx07`（提交时 7/8 GPU 已占用；剩余 CPU/内存足够），申请 1 GPU、8 CPU、64G、1 小时。
- 作业先运行 H800 FP32/BF16、近极端门控、完整反向、初末状态与分段续算检查，再在相同 340M、sequence 4096、micro batch 1、BF16 mixed、激活检查点配置下测量 GDN/QR-GDN 吞吐与峰值显存。
- 提交前实验快照干净，完整唯一任务名、`job-id` 和 `submission.lock` 均无重复；已原子加锁并登记 job-id。本轮只提交 job 34936，正式训练继续受 80% 吞吐与 8 卡 DDP 门槛阻止。

## 双读局部梯度融合诊断终态审查（job 34936）

- Slurm `COMPLETED`，exit code `0:0`，elapsed `00:08:45`；`run.exitcode=0`。`run.json`、完整日志和必需的 `outputs/result.json` 均已核对并回收，状态检查先于文件最终落盘造成的短暂缺失提示已消失。
- H800 数值套件 8 passed：FP32/BF16、完整反向、初末状态、近极端门控和分段续算均通过。融合的公共递推项与两路局部读出梯度保持参考语义。
- 同卡 340M、sequence 4096、micro batch 1、BF16 mixed、激活检查点：GDN 18,411.98 token/s，QR-GDN 12,841.61 token/s，比例 69.75%，仍低于 80% 门槛。
- 相比上一版共享逆向扫描的 68.70%，比例提高约 1.04 个百分点；合并局部梯度 launch 的收益有限，主要代价仍来自第二套 QR 状态路径。峰值显存 GDN/QR-GDN 为 6.6846/6.6823 GB，无异常增长；递归状态字节保持严格两倍。
- 正式训练和 8 卡 DDP 继续阻止。下一步只设计专用耦合双状态 kernel，共同调度 KV/QR 状态准备与扫描；不回到已否决的通用 rank-2 block-WY、不做除以小 `alpha` 的恢复，也不放宽 80% 门槛。
