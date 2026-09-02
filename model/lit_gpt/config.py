# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

# 适配自 third_party/GatedDeltaNet-2/lit_gpt/config.py：
# 把 gdn2_per_layer 换成通用的 mixer/mixer_per_layer 分派，并加入 GQA/LSR 相关字段。

from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

import torch
from typing_extensions import Self

from .utils import find_multiple


@dataclass
class Config:
    org: str = "local"
    name: str = "lit-GPT"
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 64
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    local_window: int = -1
    mlp: bool = True
    nope: bool = True  # RNN实验默认无位置编码（attention基线需显式设False）
    mamba_init: bool = False
    n_query_groups: Optional[int] = None  # attention层的GQA分组（见原版注释）
    shared_attention_norm: bool = False
    _norm_class: Literal["LayerNorm", "RMSNorm", "FusedRMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    _mlp_class: Literal["LLaMAMLP"] = "LLaMAMLP"
    intermediate_size: Optional[int] = None
    condense_ratio: int = 1

    # ---- RNN mixer（GDN2/LSR）相关 ----
    mixer: str = "attn"  # "attn" | "gdn2" | "gdn" | "kda"
    mixer_per_layer: int = 1  # 1=全RNN层；N>1=每N层1个RNN其余attn；<=0=纯attn
    num_groups: Optional[int] = None  # RNN组数G；None→n_head（MHA形态）
    head_dim: Optional[int] = None  # RNN头维d_k；None→n_embd//n_head
    expand_v: float = 1.0  # value头维扩展（方案：GQA+expand v_dim）
    num_v_heads: Optional[int] = None  # v头总数（方案：GQA+增加v_head，状态数=v头数）；None→等同组数
    use_lsr: bool = False  # LSR开关：组级潜状态 + 静态P还原
    lsr_latent_dim: Optional[int] = None  # 潜维d_c；None→head_v_dim
    use_short_conv: bool = True
    conv_size: int = 4
    allow_neg_eigval: bool = False

    def __post_init__(self):
        # error checking
        assert self.n_embd % self.n_head == 0
        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head
        if self.num_groups is not None:
            assert self.n_head % self.num_groups == 0, "num_groups必须整除n_head"
        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("The config needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @property
    def mlp_class(self) -> Type:
        # `self._mlp_class` cannot be the type to keep the config json serializable
        from . import model as _model

        return getattr(_model, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self._norm_class == "RMSNorm":
            from .rmsnorm import RMSNorm

            return RMSNorm
        elif self._norm_class == "FusedRMSNorm":
            from .rmsnorm import FusedRMSNorm

            return FusedRMSNorm
        return getattr(torch.nn, self._norm_class)


# 公共骨架超参。参数量对齐（各方案间、与基线论文间）在冒烟阶段打印核对后微调。
_gdn2_340M_base = dict(
    org="local",
    block_size=4096,
    vocab_size=32000,
    padding_multiple=64,
    mixer="gdn2",
    mixer_per_layer=1,
    n_layer=24,
    n_head=16,
    n_embd=1024,
    rotary_percentage=1.0,
    parallel_residual=False,
    bias=False,
    _norm_class="FusedRMSNorm",
    norm_eps=1e-5,
    _mlp_class="LLaMAMLP",
    intermediate_size=2816,
    nope=True,
    mamba_init=True,
    use_short_conv=True,
)

_gdn2_1p3B_base = dict(
    _gdn2_340M_base,
    n_layer=24,
    n_head=32,  # 实验设计：1B档head=32（head_dim=64与300M一致）
    n_embd=2048,
    intermediate_size=5504,
)

configs = [
    # 开发用小配置（CPU/单卡可跑通）
    dict(
        _gdn2_340M_base,
        name="gdn2_lsr_tiny",
        n_layer=2,
        n_head=4,
        n_embd=256,
        intermediate_size=704,
        num_groups=2,
        use_lsr=True,
    ),
    # ---- ~340M 主实验方案（docs/experiment.md：MHA / GQA / GVA / LSR，G=4）----
    dict(_gdn2_340M_base, name="gdn2_mha_340M"),  # num_groups=None → G=H，MHA形态
    dict(_gdn2_340M_base, name="gdn2_gqa_340M", num_groups=4),
    dict(_gdn2_340M_base, name="gdn2_gva_340M", num_groups=4, num_v_heads=16),
    dict(_gdn2_340M_base, name="gdn2_lsr_340M", num_groups=4, use_lsr=True),
    # ---- ~340M GDN/KDA 骨架对照（沿用gdn2骨架超参，只换mixer）----
    dict(_gdn2_340M_base, name="gdn_mha_340M", mixer="gdn"),
    dict(_gdn2_340M_base, name="gdn_gqa_340M", mixer="gdn", num_groups=4),
    dict(_gdn2_340M_base, name="gdn_gva_340M", mixer="gdn", num_groups=4, num_v_heads=16),
    dict(_gdn2_340M_base, name="gdn_lsr_340M", mixer="gdn", num_groups=4, use_lsr=True),
    dict(_gdn2_340M_base, name="kda_mha_340M", mixer="kda"),
    dict(_gdn2_340M_base, name="kda_gqa_340M", mixer="kda", num_groups=4),
    dict(_gdn2_340M_base, name="kda_gva_340M", mixer="kda", num_groups=4, num_v_heads=16),
    dict(_gdn2_340M_base, name="kda_lsr_340M", mixer="kda", num_groups=4, use_lsr=True),
    # ---- ~1.3B（head=32，G=4）----
    dict(_gdn2_1p3B_base, name="gdn2_mha_1.3B"),
    dict(_gdn2_1p3B_base, name="gdn2_gqa_1.3B", num_groups=4),
    dict(_gdn2_1p3B_base, name="gdn2_gva_1.3B", num_groups=4, num_v_heads=32),
    dict(_gdn2_1p3B_base, name="gdn2_lsr_1.3B", num_groups=4, use_lsr=True),
]

name_to_config = {config["name"]: config for config in configs}
