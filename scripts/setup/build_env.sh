#!/usr/bin/env bash
# build_env.sh — 在 B3 服务器构建 lzc-rnn 训练环境（幂等，可重复执行）
# 用法（Slurm CPU 作业内执行）：bash scripts/setup/build_env.sh
set -Eeuo pipefail
CONDA=/work/projects/memos-b3/software/miniconda3
ENV_NAME=lzc-rnn
WORKSPACE=/work/projects/memos-b3/code/lzc/rnn
PROXY=http://10.144.2.126:8080

source "$CONDA/etc/profile.d/conda.sh"

# 1. conda 环境（不存在才创建）
if ! conda env list | grep -q "^$ENV_NAME "; then
  echo "[1/4] 创建 conda 环境 $ENV_NAME (python 3.11)"
  conda create -y -n "$ENV_NAME" python=3.11
else
  echo "[1/4] $ENV_NAME 已存在，跳过创建"
fi
conda activate "$ENV_NAME"

# 2. 锁定依赖（PyPI 走清华镜像；非交互 shell 不读 bashrc，需显式指定）
echo "[2/4] 安装锁定依赖 requirements-lock.txt"
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  pip install --no-cache-dir -r "$WORKSPACE/scripts/setup/requirements-lock.txt"

# 3. flash-attn 2.8.3：优先 GitHub 预编译 wheel（经 clash 代理），失败回退源码编译
echo "[3/4] 安装 flash-attn"
export HTTPS_PROXY="$PROXY" HTTP_PROXY="$PROXY" https_proxy="$PROXY" http_proxy="$PROXY"
ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
for abi in "$ABI" "$([ "$ABI" = TRUE ] && echo FALSE || echo TRUE)"; do
  WHEEL="flash_attn-2.8.3+cu12torch2.9cxx11abi${abi}-cp311-cp311-linux_x86_64.whl"
  URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/$WHEEL"
  if pip install --no-cache-dir "$URL"; then
    break
  fi
  echo "[warn] $WHEEL 不可用"
  if [ "$abi" != "$ABI" ]; then
    echo "[warn] 预编译 wheel 均不可用，回退源码编译（约 1 小时）"
    pip install --no-cache-dir "flash-attn==2.8.3" --no-build-isolation
  fi
done
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy

# 4. fla（vendored 路径 editable 安装）
echo "[4/4] 安装 fla（third_party editable）"
pip install --no-cache-dir -e "$WORKSPACE/third_party/flash-linear-attention" --no-build-isolation

# 校验
echo "[verify] 导入校验"
python - <<'EOF'
import torch, transformers, lightning, litdata, pyarrow, datasets, fla, flash_attn
print("torch", torch.__version__, "| triton", __import__("triton").__version__)
print("flash_attn", flash_attn.__version__, "| fla", fla.__version__)
print("transformers", transformers.__version__, "| lightning", lightning.__version__)
print("OK")
EOF
