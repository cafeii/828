# QGDN

本仓库当前只维护一条主要研究线：**QGDN（Query-Guided DeltaNet）**。它在标准
Gated Delta Network 的全局衰减与 Delta 写入之间加入查询引导的 Recall Rule，恢复
当前查询因衰减而损失的一部分旧读出。

## 当前结论

- 340M 模型在 FineWeb 上完成两个配对 seed、每项约 10B token 的正式训练。
- QGDN 在两个 seed 上均小幅降低验证 loss 和 perplexity；平均 loss 差值为
  `-0.001248`，平均 PPL 差值为 `-0.01844`。
- 位置分桶显示收益主要出现在序列后半段，但两个 seed 只支持描述性判断。
- 单卡 H800 优化后吞吐达到同场 GDN 的 `90.87%`。

完整数字见 [实验结果](research/qgdn/EXPERIMENTS.md)，方法定义见
[QGDN 概览](research/qgdn/QGDN_OVERVIEW_CN.md)，数值策略见
[可复现性说明](research/qgdn/REPRODUCIBILITY.md)。已停止的研究分支只保留在
[其他方向归档](research/OTHER_DIRECTIONS.md)；实现细节仍可从 Git 历史恢复。

## 代码结构

- `model/lit_gpt/mixers/qgdn.py`：QGDN 层和门控参数。
- `model/lit_gpt/mixers/qgdn_rule.py`：生产训练与推理 recurrence。
- `model/lit_gpt/mixers/qgdn_reference.py`：FP32/FP64 逐 token 参考和 rank-2 数学 oracle。
- `tests/test_qgdn.py`：机制、梯度、退化关系和生产 kernel 一致性测试。
- `scripts/qgdn/`：数据、训练、评测、审查和汇总入口。

## 最小验证

```bash
PYTHONPATH=model:third_party/flash-linear-attention \
python -m pytest tests/test_qgdn.py tests/test_qgdn_data.py \
  tests/test_qgdn_prepare_data.py tests/test_qgdn_audit.py -q
```

CUDA kernel 和 8 卡验证必须在 Slurm 分配的 GPU 节点上运行。正式实验使用的模型代码
固定在提交 `f62322a5fd0cdbc1ed45a9753bdfa22a663143d4`；后续训练加速提交不追溯改变
既有实验结果。
