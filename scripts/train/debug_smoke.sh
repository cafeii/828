#!/bin/bash
# 本地/单卡debug冒烟：tiny配置 + 小数据切片，验证训练管线可跑
# 用法: bash scripts/train/debug_smoke.sh <train_data_dir> [val_data_dir]
set -e
cd "$(dirname "$0")/../.."

TRAIN_DATA=${1:?需要train_data_dir（litdata格式）}
VAL_DATA=${2:-}

ARGS=(
  --model_name gdn2_lsr_tiny
  --exp_name debug_smoke_$(date +%m%d%H%M)
  --train_data_dir "$TRAIN_DATA"
  --devices 1
  --max_tokens 20000000
  --micro_batch_size 2
  --global_batch_size 8
  --warmup_tokens 2000000
  --save_step_interval 100
  --eval_step_interval 100
  --eval_iters 10
)
[ -n "$VAL_DATA" ] && ARGS+=(--val_data_dir "$VAL_DATA")

python scripts/pretrain.py "${ARGS[@]}"
