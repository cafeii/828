# HF兼容包装：把工作区 lit_gpt 模型（fabric ckpt）暴露为 transformers CausalLM，
# 供 lm-eval(HFLM) / RULER / JRT 三套评估框架统一消费。
# - loglikelihood类任务：forward(input_ids) -> logits（batch右pad安全：causal RNN尾部pad不影响之前时刻）
# - 生成类任务：generate 走 GenerationMixin（GPU环境用fla的Cache/Mixin，与fla全家模型同路径），
#   prefill走chunk kernel、逐token decode自动切fused_recurrent（mixer按T<=64判定）
# - 模型为packed定长训练、无padding支持：attention_mask仅接受None/全1（生成类任务用batch_size=1）

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_WD = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_WD / "model"))

from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel  # noqa: E402
from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: E402

from lit_gpt.config import Config  # noqa: E402
from lit_gpt.model import GPT  # noqa: E402
from lit_gpt.mixers.cache import SimpleCache  # noqa: E402

try:  # GPU评估环境（有fla）：用fla的Cache与GenerationMixin，transformers版本兼容性已由fla处理
    from fla.models.utils import Cache as _CacheCls
    from fla.models.utils import FLAGenerationMixin as _GenMixin
except ImportError:  # 本地CPU开发环境：SimpleCache + 原生Mixin（generate不在CPU测）
    from transformers import GenerationMixin as _GenMixin

    _CacheCls = SimpleCache


class LitGPTConfig(PretrainedConfig):
    model_type = "litgpt_rnn"

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        self.model_name = model_name
        super().__init__(**kwargs)

    @classmethod
    def from_litgpt(cls, lit_config: Config, **kwargs) -> "LitGPTConfig":
        return cls(
            model_name=lit_config.name,
            vocab_size=lit_config.padded_vocab_size,
            hidden_size=lit_config.n_embd,
            num_hidden_layers=lit_config.n_layer,
            max_position_embeddings=32768,  # RNN无位置编码；评估长度由调用侧显式控制
            tie_word_embeddings=False,
            bos_token_id=1,  # Llama-2 tokenizer
            eos_token_id=2,
            pad_token_id=2,
            use_cache=True,
            **kwargs,
        )


class LitGPTForCausalLM(PreTrainedModel, _GenMixin):
    config_class = LitGPTConfig
    base_model_prefix = "gpt"
    _supports_cache_class = True
    _no_split_modules = ["Block"]

    def __init__(self, config: LitGPTConfig, lit_config: Optional[Config] = None):
        super().__init__(config)
        self.gpt = GPT(lit_config if lit_config is not None else Config.from_name(config.model_name))

    def get_input_embeddings(self):
        return self.gpt.transformer.wte

    def get_output_embeddings(self):
        return self.gpt.lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if attention_mask is not None and not bool(attention_mask.bool().all()):
            raise NotImplementedError("模型不支持padding mask：生成类任务请用batch_size=1")
        if use_cache and past_key_values is None:
            past_key_values = _CacheCls()
        logits = self.gpt(input_ids, past_key_values=past_key_values)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


def load_eval_model(
    model_name: str,
    ckpt_path: str,
    tokenizer_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """加载fabric训练ckpt为评估模型。返回 (model, tokenizer)。"""
    lit_config = Config.from_name(model_name)
    model = LitGPTForCausalLM(LitGPTConfig.from_litgpt(lit_config), lit_config)

    sd = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    sd = sd.get("model", sd)
    sd = {
        k.removeprefix("_forward_module.").removeprefix("_orig_mod.").removeprefix("model."): v
        for k, v in sd.items()
    }
    missing, unexpected = model.gpt.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, f"ckpt键不匹配 missing={missing} unexpected={unexpected}"

    model.to(device=device, dtype=dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
