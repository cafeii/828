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
