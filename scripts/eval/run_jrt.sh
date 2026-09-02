#!/usr/bin/env bash
# JRT 召回类评估（JRT ArXiv Table 1 口径）：6 任务 × {default, twice} = 12 项。
# 参数与官方 prompt_scripts/run_jrt_prompt_hf.sh 完全对齐：
#   context_length=1000, answer_length=50, --cutting_context, --context_key text, limit=全量
# 与官方唯一差异：batch_size=1（本模型 packed 定长训练、不支持 padding mask，生成类任务必须 bs=1）。
#
# 用法: bash scripts/eval/run_jrt.sh <model_name> <ckpt_path> [output_dir]
#   model_name  Config.from_name 可解析的名字，如 gdn2_lsa_340M
#   ckpt_path   fabric 训练 ckpt，如 outputs/<exp-id>/<model>/final-model-ckpt.pth
set -Eeuo pipefail

MODEL_NAME="${1:?需要 model_name，如 gdn2_lsa_340M}"
CKPT_PATH="${2:?需要 ckpt_path}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${3:-${WORKSPACE}/outputs/eval/jrt/${MODEL_NAME}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${WORKSPACE}/checkpoints/Llama-2-7b-hf}"
HARNESS="${WORKSPACE}/third_party/prefix-linear-attention/lm-eval-harness"
# 评估数据全部预下载（服务器无外网），显式走离线缓存
export HF_HOME="${HF_HOME:-${WORKSPACE}/dataset/eval_data/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

[[ "${CKPT_PATH}" = /* ]] || CKPT_PATH="${WORKSPACE}/${CKPT_PATH}"
[[ -f "${CKPT_PATH}" ]] || { echo "[fail] ckpt 不存在: ${CKPT_PATH}" >&2; exit 1; }
[[ -d "${TOKENIZER_PATH}" ]] || { echo "[fail] tokenizer 不存在: ${TOKENIZER_PATH}" >&2; exit 1; }

TASKS=(
  based_swde based_swde_twice
  based_fda based_fda_twice
  based_squad based_squad_twice
  based_drop based_drop_twice
  based_nq_1024 based_nq_1024_twice
  based_triviaqa based_triviaqa_twice
)

mkdir -p "${OUTPUT_DIR}"
cd "${HARNESS}"

for task in "${TASKS[@]}"; do
  echo "[jrt] ${MODEL_NAME} / ${task}"
  # 逐任务独立进程：单任务 OOM/报错不影响其余任务，便于断点续跑
  python -m lm_eval \
    --model litgpt \
    --model_args "model_name=${MODEL_NAME},ckpt_path=${CKPT_PATH},tokenizer_path=${TOKENIZER_PATH},max_length=4096" \
    --tasks "${task}" \
    --device cuda:0 \
    --batch_size 1 \
    --cutting_context \
    --context_length 1000 \
    --answer_length 50 \
    --context_key text \
    --log_samples \
    --output_path "${OUTPUT_DIR}/${task}" \
    2>&1 | tee "${OUTPUT_DIR}/${task}.log"
done

echo "[ok] JRT 评估完成: ${OUTPUT_DIR}"
