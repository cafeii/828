#!/usr/bin/env bash
# 标准零样本九项任务（lm-eval 0.4.9 上游，独立评估环境）。
# 任务与指标对齐 docs/experiment.md：
#   Lambada(openai) ppl/acc, PIQA acc, Hellaswag acc_norm, Winogrande acc,
#   ARC-easy acc, ARC-challenge acc_norm, OpenBookQA acc, SIQA acc, BoolQ acc
# （FineWeb-test ppl 不在此脚本：用训练 val litdata 单独算，见 run_fineweb_ppl.py）
#
# 用法: bash scripts/eval/run_lm_eval_std.sh <model_name> <ckpt_path> [output_dir]
# 环境变量: EVAL_PY 覆盖 python 解释器（服务器上用 conda 环境的 python）
set -Eeuo pipefail

MODEL_NAME="${1:?需要 model_name，如 gdn2_lsa_340M}"
CKPT_PATH="${2:?需要 ckpt_path}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${3:-${WORKSPACE}/outputs/eval/lm_eval_std/${MODEL_NAME}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${WORKSPACE}/checkpoints/Llama-2-7b-hf}"
EVAL_PY="${EVAL_PY:-${WORKSPACE}/.venv-eval/bin/python}"
# 评估数据全部预下载（服务器无外网），显式走离线缓存
export HF_HOME="${HF_HOME:-${WORKSPACE}/dataset/eval_data/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

[[ "${CKPT_PATH}" = /* ]] || CKPT_PATH="${WORKSPACE}/${CKPT_PATH}"
[[ -f "${CKPT_PATH}" ]] || { echo "[fail] ckpt 不存在: ${CKPT_PATH}" >&2; exit 1; }

# social_iqa_fixed = 上游 social_iqa 换 dataset_path（旧脚本数据集名在 datasets>=3 不可加载），
# 定义在 tasks_override/siqa.yaml，评分口径与上游完全一致
TASKS="lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,social_iqa_fixed,boolq"

mkdir -p "${OUTPUT_DIR}"
"${EVAL_PY}" "${WORKSPACE}/scripts/eval/run_lm_eval.py" \
  --model litgpt \
  --model_args "model_name=${MODEL_NAME},ckpt_path=${CKPT_PATH},tokenizer_path=${TOKENIZER_PATH},max_length=4096" \
  --tasks "${TASKS}" \
  --include_path "${WORKSPACE}/scripts/eval/tasks_override" \
  --num_fewshot 0 \
  --batch_size 8 \
  --device cuda:0 \
  --output_path "${OUTPUT_DIR}" \
  --log_samples \
  2>&1 | tee "${OUTPUT_DIR}/run.log"

echo "[ok] 标准任务评估完成: ${OUTPUT_DIR}"
