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
