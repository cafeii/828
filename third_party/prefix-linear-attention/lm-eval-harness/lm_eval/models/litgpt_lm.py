# 工作区自训 lit_gpt 模型接入（JRT评估用）：包装为 HFLM 供本 harness 消费。
# model_args 示例:
#   --model litgpt
#   --model_args model_name=gdn2_lsa_340M,ckpt_path=outputs/.../final-model-ckpt.pth,tokenizer_path=checkpoints/Llama-2-7b-hf
# 生成类任务（based_*）模型不支持padding mask，必须 --batch_size 1。
import sys
from pathlib import Path

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

# harness 位于 <workspace>/third_party/prefix-linear-attention/lm-eval-harness，
# 本文件是 .../lm_eval/models/litgpt_lm.py，workspace = 向上5级
_WORKSPACE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_WORKSPACE / "scripts" / "eval"))


@register_model("litgpt")
class LitGPTLM(HFLM):
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
