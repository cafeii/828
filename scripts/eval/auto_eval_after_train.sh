#!/usr/bin/env bash
# 完训即拉起评估：监视训练 job 结束后自动提交 std/ruler/jrt 三个评估 job。
# 由 run-remote-experiment 在提交训练后挂到登录节点后台执行。
# 用法: bash scripts/eval/auto_eval_after_train.sh <TRAIN_JOB_ID> <MODEL_NAME> <CKPT_PATH> <EXP_ID> [EVAL_GPUS_PER_JOB]
set -Eeuo pipefail

TRAIN_JOB_ID="${1:?}"
MODEL_NAME="${2:?}"
CKPT_PATH="${3:?}"
EXP_ID="${4:?}"
WORKSPACE="/work/projects/memos-b3/code/lzc/rnn"
LOG_DIR="${WORKSPACE}/logs/${EXP_ID}"
USER_NAME="刘之辰"
EVAL_GPUS="${5:-1}"

cd "${WORKSPACE}"
mkdir -p "${LOG_DIR}"

echo "[auto-eval] watch train job ${TRAIN_JOB_ID} (${MODEL_NAME})"

# 轮询训练 job 直至从队列消失（squeue 查不到即终态）
while true; do
  st=$(squeue -j "${TRAIN_JOB_ID}" -h -o %T 2>/dev/null || true)
  if [[ -z "${st}" ]]; then
    break
  fi
  sleep 120
done

# 判定训练成败：final ckpt 存在即成功
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[auto-eval][fail] 训练 job ${TRAIN_JOB_ID} 结束但未找到 ${CKPT_PATH}，不拉起评估" >&2
  exit 1
fi
echo "[auto-eval] train done, ckpt ok: ${CKPT_PATH}"

EVAL_OUT="${WORKSPACE}/outputs/${EXP_ID}-eval"
mkdir -p "${EVAL_OUT}"

submit_eval() {
  local suite="$1" gpus="$2"
  local job="${USER_NAME}-Metis-${EXP_ID}-eval-${suite}-${MODEL_NAME##*_340M}"
  local sb="${LOG_DIR}/job-eval-${suite}-${MODEL_NAME}.sbatch"
  cat > "${sb}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${gpus}
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=${LOG_DIR}/%x-%j.out
#SBATCH --error=${LOG_DIR}/%x-%j.err
set -Eeuo pipefail
source /work/projects/memos-b3/software/miniconda3/etc/profile.d/conda.sh
conda activate lzc-rnn
cd ${WORKSPACE}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
case "${suite}" in
  std)
    bash scripts/eval/run_lm_eval_std.sh ${MODEL_NAME} ${CKPT_PATH} ${EVAL_OUT}/std/${MODEL_NAME}
    ;;
  ruler)
    QUICK=1 bash scripts/eval/run_ruler.sh ${MODEL_NAME} ${CKPT_PATH} ${EVAL_OUT}/ruler/${MODEL_NAME}
    ;;
  jrt)
    bash scripts/eval/run_jrt.sh ${MODEL_NAME} ${CKPT_PATH} ${EVAL_OUT}/jrt/${MODEL_NAME}
    ;;
esac
echo "[ok] eval ${suite} ${MODEL_NAME} done"
EOF
  sbatch "${sb}"
}

submit_eval std   "${EVAL_GPUS}"
submit_eval ruler "${EVAL_GPUS}"
submit_eval jrt   "${EVAL_GPUS}"
echo "[auto-eval] submitted 3 eval jobs for ${MODEL_NAME}"
