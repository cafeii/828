#!/usr/bin/env python
# JRT（prefix-linear-attention fork harness）评估入口。
# fork 未安装进环境，经 PYTHONPATH 指向 third_party/prefix-linear-attention/lm-eval-harness
# 使用（避免与上游 lm_eval 0.4.9 同模块名冲突）；本脚本注入 transformers 5.x 兼容 shim
# 后转交 fork 的 CLI。参数与 `python -m lm_eval` 完全相同。
# 由 scripts/eval/run_jrt.sh 调用；单测/调试可手动：
#   PYTHONPATH=third_party/prefix-linear-attention/lm-eval-harness \
#   python scripts/eval/run_jrt.py --model litgpt --tasks based_swde ...
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WORKSPACE / "scripts" / "eval"))

# shim 说明见 run_lm_eval.py（evaluate 先导入以触发其 transformers 对象替换副作用）
import evaluate  # noqa: F401
import transformers

if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = transformers.AutoModelForCausalLM

if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
