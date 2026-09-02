# QR-GDN 实现与评测工作流

## 目标

实现双通道 QR-GDN，并与标准 GDN 比较 FineWeb 预训练效果。标准 GDN 使用一块状态矩阵；QR-GDN 使用同尺寸的 $M^{KV}$ 与 $M^{QR}$，总隐状态为 GDN 的两倍。

## 固定机制

KV 通道保存外部关联 $k_t\rightarrow v_t$，QR 通道保存内部关联 $q_t\rightarrow r_t$：

$$
r_t=(M_{t-1}^{KV})^\top q_t.
$$

两块矩阵由同一个联合近端目标产生，门控统一记为 $\alpha_t^{KV}$、$\beta_t^{KV}$、$\alpha_t^{QR}$、$\beta_t^{QR}$。$k_t$ 与 $q_t$ 均做 L2 归一化。

当前 token 的 KV 输出沿用标准 GDN 的更新后读取语义。QR 读出只读取 $M_{t-1}^{QR}$，避免把刚写入的 $q_t\rightarrow r_t$ 当作当前 token 的即时旁路；它通过零初始化的有界残差门加入输出，因此初始化时与 GDN 严格一致。

## 验证门槛

1. FP64 逐 token 参考、联合目标直接求解和块仿射递推相符。
2. 前向、反向、初态、终态、分段恢复和增量解码相符。
3. QR 写入关闭且 QR 读出门为零时严格退化为现有 GDN。
4. 正式训练使用真正的 chunk/associative-scan 或等价融合 kernel。
5. 通过 CPU、GPU、FP32/BF16、稳定性、8 卡 DDP 和吞吐/显存验证后才提交完整训练。

## 正式评测

只运行 QR-GDN seed 3407、42，各 19,073 steps、9,999,745,024 prediction tokens。标准 GDN 复用已审查结果。最终报告整体 loss/PPL、固定位置分桶、双状态成本、门控统计、吞吐、峰值显存和两个 seed 的配对差值。

## 作业与提交纪律

`active_jobs` 是活动作业的唯一权威记录。每次心跳最多一次 `sbatch`；所有正式实验来自干净、已提交、已完整验证的 commit。保留旧实验历史，不恢复 QGDN、DT-GDN 或 JQC-GDN 作业，不 push 或强推。
