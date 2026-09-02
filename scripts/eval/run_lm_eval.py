#!/usr/bin/env python
# 上游 lm-eval（0.4.9，独立环境 .venv-eval / 服务器 lzc-eval）跑标准零样本九项任务的入口。
# 上游包里没有我们的模型类，正式插件机制只覆盖任务不覆盖模型，
# 故用本脚本先注册 litgpt 模型再转交官方 CLI。用法与 lm_eval CLI 相同：
#   python scripts/eval/run_lm_eval.py --model litgpt \
#     --model_args model_name=gdn2_lsr_340M,ckpt_path=...,tokenizer_path=... \
#     --tasks piqa,hellaswag --batch_size 8 --output_path outputs/eval/...
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WORKSPACE / "scripts" / "eval"))

from lm_eval.api.registry import register_model  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402


@register_model("litgpt")
class LitGPTLM(HFLM):
    """与 third_party/prefix-linear-attention fork 里的注册保持同构（见其 patches/PATCHES.md）。"""

    def __init__(
        self,
        model_name: str,
        ckpt_path: str,
        tokenizer_path: str,
        max_length: int = 4096,
        device: str = "cuda",
        dtype: str = "bfloat16",
        **kwargs,
    ) -> None:
        import torch
        from wrapper import load_eval_model

        model, tokenizer = load_eval_model(
            model_name=model_name,
            ckpt_path=ckpt_path,
            tokenizer_path=tokenizer_path,
            device=device,
            dtype=getattr(torch, dtype),
        )
        super().__init__(
            pretrained=model,
            backend="causal",
            max_length=max_length,
            tokenizer=tokenizer,
            device=device,
            **kwargs,
        )


if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
