# 工作区自训 lit_gpt 模型接入 RULER（PATCH，见 ../../patches/PATCHES.md）。
# 形态仿同目录 model_wrappers.py 的 MambaModel：__call__ + process_batch 两个方法即够。
# 本模型 packed 定长训练、不支持 padding mask，故 process_batch 逐条循环，RULER 侧须 BATCH_SIZE=1。
import sys
from pathlib import Path
from typing import Dict, List

import torch

# 本文件位于 <workspace>/third_party/RULER/scripts/pred/litgpt_model.py -> workspace = 向上4级
_WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_WORKSPACE / "scripts" / "eval"))


class LitGPTModel:
    def __init__(
        self,
        name_or_path: str,
        model_name: str = None,
        tokenizer_path: str = None,
        **generation_kwargs,
    ) -> None:
        from wrapper import load_eval_model

        # RULER 只透传 name_or_path（= MODEL_PATH），故按约定用 "<model_name>:<ckpt_path>" 编码，
        # 或由 config_models.sh 显式提供 model_name/tokenizer_path。
        if model_name is None:
            model_name, _, ckpt = name_or_path.partition(":")
            if not ckpt:
                raise ValueError(
                    "name_or_path 需为 '<model_name>:<ckpt_path>'，实得: " + name_or_path
                )
        else:
            ckpt = name_or_path

        if tokenizer_path is None:
            tokenizer_path = str(_WORKSPACE / "checkpoints" / "Llama-2-7b-hf")

        self.device = "cuda"
        self.model, self.tokenizer = load_eval_model(
            model_name=model_name,
            ckpt_path=ckpt,
            tokenizer_path=tokenizer_path,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.generation_kwargs = dict(generation_kwargs)
        self.stop = self.generation_kwargs.pop("stop", None)
        self.max_genlen = self.generation_kwargs.pop("max_new_tokens", 128)
        # RULER 默认贪心（temperature=0 / top_p=1），HF generate 需显式关掉采样
        if float(self.generation_kwargs.pop("temperature", 0.0)) == 0.0:
            self.generation_kwargs["do_sample"] = False
            self.generation_kwargs.pop("top_p", None)
            self.generation_kwargs.pop("top_k", None)

    @torch.inference_mode()
    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=self.max_genlen,
            pad_token_id=self.tokenizer.eos_token_id,
            **self.generation_kwargs,
        )
        text = self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        if self.stop:
            for s in self.stop:
                text = text.split(s)[0]
        return {"text": [text]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        return [self.__call__(prompt, **kwargs) for prompt in prompts]
