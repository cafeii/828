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
