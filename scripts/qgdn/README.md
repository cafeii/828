# QGDN 脚本

本目录只保留可复用入口：

| 脚本 | 用途 |
|---|---|
| `prepare_data.py` / `data.py` | 构建和读取固定 FineWeb token 数据 |
| `audit_fineweb.py` | 只读审查原始数据与 token chunks |
| `train.py` | QGDN/GDN 训练与 checkpoint 恢复 |
| `evaluate.py` | 固定验证集评测 |
| `summarize.py` | 核对身份并生成配对汇总 |
| `validate.py` | CPU/GPU、恢复、DDP 与全尺寸模型验证 |
| `audit_kernel.py` | 生产 QGDN kernel 对参考 recurrence 的 CUDA 审查 |
| `benchmark_training_speed.py` | GDN/QGDN 隔离训练吞吐与显存基准 |
| `evaluate_gate_stats.py` | 终点门控全局 mean/std 统计 |
| `validate_gate_stats_ddp.py` | 门控跨 rank 聚合验证 |
| `run_gate_stats_suite.py` | 两个模型的只读门控统计编排 |
| `evaluate_position_buckets.py` | 固定位置分桶 loss/PPL |
| `gate.py` | 正式训练前的 fail-closed 身份与验证门 |
| `runtime.py` | 统一设备和数值策略 |

训练和 CUDA 审查必须通过 Slurm 在分配节点上执行。历史数据修复、文件晋升、恢复诊断、
计划生成及一次性性能 profile 脚本已删除；相应操作需要时应从 Git 历史中恢复并重新审查。

## QGDN 340M 高吞吐配置

`configs/qgdn_340m_fineweb_10b_fast.args` 固化了已验证的 FineWeb 10B 正式配置：
micro batch 8、global batch 128、关闭 activation checkpointing，并使用 fused cross entropy。
在 8 张 H800 上测得稳态吞吐约 339,844 token/s，10B token 的纯训练时间约 8.17 小时，
峰值显存约 77.07 GB/GPU。

通过 `@` 直接加载配置，并按每次实验传入 seed、数据清单和输出目录：

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/qgdn/train.py @configs/qgdn_340m_fineweb_10b_fast.args \
  --seed 3407 \
  --data-manifest /absolute/path/to/manifest.json \
  --output /absolute/path/outside/source/tree
```

输出目录必须位于不可变源码 checkout 之外。使用当前 Python 的 `torch.distributed.run`，
不要调用可能残留旧环境 shebang 的 `torchrun` 命令。该配置不启用整模型 `torch.compile`。
