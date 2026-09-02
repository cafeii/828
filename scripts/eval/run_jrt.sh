#!/usr/bin/env bash
# JRT 召回类评估（架构对比口径，对齐 MoM README 命令与 GDN/GDN2/MoM 论文表 "truncated to 2K" caption）：
#   6 个 default 任务（无 twice），context_length=2048, answer_length=48, --cutting_context,
#   --context_key text, limit=全量；NQ 用 based_nq_2048 数据集变体（文档长度与截断窗口匹配）。
# 与 MoM 唯一差异：batch_size=1（本模型 packed 定长训练、不支持 padding mask，生成类任务必须 bs=1）。
#
# 评估机制（fork lm_eval/api/task.py: truncate_context + tasks/based_*/task.py）：
#   1) 截断：窗口 = context_length - answer_length = 2000 token，以答案在原文首次出现的
#      位置为中心截取（前半 + 后半各约一半）；文档短于窗口则全保留。
#   2) prompt：cloze 式——截后上下文 + " Key:" 问题后缀（零样本、无指令，面向 base 模型）。
#   3) 生成：各任务硬编码 max_gen_toks=48，遇换行停止（--answer_length 只影响截断缓冲）。
#   4) 判分：contains——生成文本中大小写不敏感包含 gold answer 即对，按样本取均值。
#
# 口径沿革（2026-09-02 核验）：
#   - JRT-Prompt 口径（prompt_scripts/run_jrt_prompt_hf.sh，ArXiv Table 1）：context_length=1000、
#     6 default + 6 twice。该上游脚本自 2024-07-09 起从未变更；GDN README 指向它是为了
#     prompt 格式（警告裸任务名结果差异巨大），而非截断长度。
#   - 架构对比口径（本脚本）：MoM README 命令与论文 Table 1 caption（"truncated to 2K"）互证；
#     GDN/GDN2 Table 4 caption 同为 2K。若需对照 JRT ArXiv Table 1 的 JRT-Prompt 数字，
#     改用 --context_length 1000 --answer_length 50 并追加 twice 任务（based_*_twice）。
#
# 用法: bash scripts/eval/run_jrt.sh <model_name> <ckpt_path> [output_dir]
#   model_name  Config.from_name 可解析的名字，如 gdn2_lsr_340M
#   ckpt_path   fabric 训练 ckpt，如 outputs/<exp-id>/<model>/final-model-ckpt.pth
# 环境变量: EVAL_PY 覆盖 python 解释器（服务器上指向装好 fork lm-eval(-e) 与 fla 的环境，
#   如 conda activate lzc-rnn 后的 python；缺省用当前 PATH 里的 python）
#   QUICK=1  前期快速验证子集（docs/experiment.md：FDA/SWDE/SQuAD），缺省全量
set -Eeuo pipefail

MODEL_NAME="${1:?需要 model_name，如 gdn2_lsr_340M}"
CKPT_PATH="${2:?需要 ckpt_path}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${3:-${WORKSPACE}/outputs/eval/jrt/${MODEL_NAME}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${WORKSPACE}/checkpoints/Llama-2-7b-hf}"
HARNESS="${WORKSPACE}/third_party/prefix-linear-attention/lm-eval-harness"
EVAL_PY="${EVAL_PY:-python}"
# 评估数据全部预下载（服务器无外网），显式走离线缓存
export HF_HOME="${HF_HOME:-${WORKSPACE}/dataset/eval_data/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

[[ "${CKPT_PATH}" = /* ]] || CKPT_PATH="${WORKSPACE}/${CKPT_PATH}"
[[ -f "${CKPT_PATH}" ]] || { echo "[fail] ckpt 不存在: ${CKPT_PATH}" >&2; exit 1; }
[[ -d "${TOKENIZER_PATH}" ]] || { echo "[fail] tokenizer 不存在: ${TOKENIZER_PATH}" >&2; exit 1; }

TASKS=(
  based_swde
  based_fda
  based_squad
  based_drop
  based_nq_2048
  based_triviaqa
)
# 前期快速验证子集（docs/experiment.md 评估安排）
[[ "${QUICK:-0}" = "1" ]] && TASKS=(based_fda based_swde based_squad)

mkdir -p "${OUTPUT_DIR}"
cd "${HARNESS}"

for task in "${TASKS[@]}"; do
  echo "[jrt] ${MODEL_NAME} / ${task}"
  # 逐任务独立进程：单任务 OOM/报错不影响其余任务，便于断点续跑
  "${EVAL_PY}" -m lm_eval \
    --model litgpt \
    --model_args "model_name=${MODEL_NAME},ckpt_path=${CKPT_PATH},tokenizer_path=${TOKENIZER_PATH},max_length=4096" \
    --tasks "${task}" \
    --device cuda:0 \
    --batch_size 1 \
    --cutting_context \
    --context_length 2048 \
    --answer_length 48 \
    --context_key text \
    --log_samples \
    --output_path "${OUTPUT_DIR}/${task}" \
    2>&1 | tee "${OUTPUT_DIR}/${task}.log"
done

echo "[ok] JRT 评估完成: ${OUTPUT_DIR}"
