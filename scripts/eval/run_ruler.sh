#!/usr/bin/env bash
# RULER 长上下文评测（4 任务 × 4 长度），直接驱动 prepare/call_api/evaluate 三段，
# 不走上游 run.sh（其 config_models.sh 的 tokenizer 自动探测会误判为 nemo，见 patches/PATCHES.md）。
#
# 任务口径 docs/experiment.md: S-NIAH-1/2/3 + MK-NIAH-1
#   -> yaml 名 niah_single_1 / niah_single_2 / niah_single_3 / niah_multikey_1
# 长度网格 plans/2026-09-01-lsa300m-eval.md: 1024/2048/4096/8192（4096=训练长度，8192 看外推）
#
# 用法: bash scripts/eval/run_ruler.sh <model_name> <ckpt_path> [output_dir]
# 环境变量: EVAL_PY 覆盖解释器；NUM_SAMPLES 覆盖每任务样本数（默认 500，与官方一致）
#   QUICK=1  前期快速验证子集（docs/experiment.md：S-NIAH-2/3 × 4 长度），缺省全量
set -Eeuo pipefail

MODEL_NAME="${1:?需要 model_name，如 gdn2_lsr_340M}"
CKPT_PATH="${2:?需要 ckpt_path}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${3:-${WORKSPACE}/outputs/eval/ruler/${MODEL_NAME}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${WORKSPACE}/checkpoints/Llama-2-7b-hf}"
EVAL_PY="${EVAL_PY:-${WORKSPACE}/.venv-eval/bin/python}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"

RULER_SCRIPTS="${WORKSPACE}/third_party/RULER/scripts"
TASKS=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1)
# 前期快速验证子集（docs/experiment.md 评估安排）
[[ "${QUICK:-0}" = "1" ]] && TASKS=(niah_single_2 niah_single_3)
SEQ_LENGTHS=(1024 2048 4096 8192)
# nltk punkt 已预下载（服务器无外网）；tokenizer 从本地路径加载不触网
export NLTK_DATA="${NLTK_DATA:-${WORKSPACE}/dataset/eval_data/nltk_data}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

[[ "${CKPT_PATH}" = /* ]] || CKPT_PATH="${WORKSPACE}/${CKPT_PATH}"
[[ -f "${CKPT_PATH}" ]] || { echo "[fail] ckpt 不存在: ${CKPT_PATH}" >&2; exit 1; }
[[ -x "${EVAL_PY}" ]] || { echo "[fail] EVAL_PY 不可执行: ${EVAL_PY}（服务器上请设 EVAL_PY 指向含 fla/lightning 的 conda python）" >&2; exit 1; }
[[ -f "${RULER_SCRIPTS}/data/synthetic/json/PaulGrahamEssays.json" ]] || {
  echo "[fail] 缺 PaulGrahamEssays.json（S-NIAH-2/3、MK-NIAH-1 依赖），需先本地下载并同步" >&2; exit 1; }

# prepare/call_api 用相对 import（from data.xxx / from litgpt_model），须在 scripts/ 下执行
cd "${RULER_SCRIPTS}"

for SEQ in "${SEQ_LENGTHS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    DATA_DIR="${OUTPUT_DIR}/${SEQ}/data"
    PRED_DIR="${OUTPUT_DIR}/${SEQ}/eval_raw"   # 逐样本预测属原始输出，回收时排除
    mkdir -p "${DATA_DIR}" "${PRED_DIR}"
    echo "[run] seq=${SEQ} task=${TASK}"

    "${EVAL_PY}" data/prepare.py \
      --save_dir "${DATA_DIR}" --benchmark synthetic --task "${TASK}" \
      --tokenizer_path "${TOKENIZER_PATH}" --tokenizer_type hf \
      --max_seq_length "${SEQ}" --num_samples "${NUM_SAMPLES}" \
      --model_template_type base

    "${EVAL_PY}" pred/call_api.py \
      --data_dir "${DATA_DIR}" --save_dir "${PRED_DIR}" \
      --benchmark synthetic --task "${TASK}" \
      --server_type litgpt \
      --model_name_or_path "${MODEL_NAME}:${CKPT_PATH}" \
      --temperature 0.0 --top_p 1.0 --batch_size 1
  done

  "${EVAL_PY}" eval/evaluate.py \
    --data_dir "${OUTPUT_DIR}/${SEQ}/eval_raw" --benchmark synthetic
done

echo "[ok] RULER 评测完成: ${OUTPUT_DIR}"
