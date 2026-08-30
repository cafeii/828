# GDN 骨架 token mixer。模块结构对齐 mixers/gdn2.py，门参数化/初始化对齐
# third_party/GatedDeltaNet/lit_gpt/gated_delta_net.py（use_mamba_gate=True 路径）。
#
# 递归：S_t = α_t (I - β_t k k^T) S_{t-1} + β_t k v^T
#   α_t = exp(g_t) 为 per-head 标量 decay（log域 g:[B,T,H]，mamba 参数化），β_t 为 per-head 标量。
# GQA/LSA 化的关键约束：β 与 decay 必须组级（若逐头则组内各头 state 不同，P 无法提出），q 保持逐头。
# GDN ≡ GDN2 取 b=β·1（k维广播）、w=β·1（v维广播）、g 标量广播到 k 维（标量 α 与擦除项可交换），
# naive 模式据此复用 naive_gdn2_recurrence；chunk/fused_recurrent 走 fla 的 gated_delta_rule kernel
# （g/beta 形状 [B,T,H]，见 fla.ops.gated_delta_rule 文档）。
# LSA（use_lsa=True）：v 为组级潜向量 c，写入 β k c^T，出口乘静态 P 还原（同 gdn2.py 策略2）。

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F


class GatedDeltaNet(nn.Module):
    """GDN token mixer，统一支持 MHA / GQA / GQA+expand_v / GQA+LSA 形态。

    形状约定（G=num_groups, I=num_heads//G, d_k=head_dim,
    d_v=head_dim*expand_v, d_c=lsa_latent_dim）：
      q: [B,T,H,d_k] 逐头
      k: [B,T,G,d_k] 组级
      v:   [B,T,G,d_s] 组级，d_s = d_c（LSA）或 d_v（GQA）
      g/b: [B,T,G] 组级标量（log-decay / 擦除写入门β）
      递归状态: G份 [d_k, d_s]（策略2下kernel内冗余为H份，数学等价）
      LSA还原: P [H, d_v, d_c]，o = einsum('bthc,hvc->bthv', o_latent, P)
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        num_groups: Optional[int] = None,
        head_dim: int = 128,
        expand_v: float = 1.0,
        num_v_heads: Optional[int] = None,
        use_lsa: bool = False,
        lsa_latent_dim: Optional[int] = None,
        mode: Literal["chunk", "fused_recurrent", "naive"] = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()
        if num_v_heads is not None and num_v_heads != 1:
            raise NotImplementedError("GQA+增加v_head 方案语义待定，暂未实现（见计划遗留问题）。")

        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_groups = num_groups if num_groups is not None else num_heads
        assert num_heads % self.num_groups == 0, "num_groups必须整除num_heads"
        self.heads_per_group = num_heads // self.num_groups
        self.use_lsa = use_lsa
        self.allow_neg_eigval = allow_neg_eigval
        self.use_short_conv = use_short_conv
        self.layer_idx = layer_idx

        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(f"expand_v={expand_v}与head_dim={head_dim}的乘积不是整数。")
        # d_s: 进入递归的value侧维度（LSA为潜维，否则为头维）
        self.latent_dim = (lsa_latent_dim or self.head_v_dim) if use_lsa else self.head_v_dim

        self.key_dim = self.num_heads * self.head_k_dim  # q逐头
        self.gk_dim = self.num_groups * self.head_k_dim  # k组级
        self.gv_dim = self.num_groups * self.latent_dim  # v组级
        self.value_dim = self.num_heads * self.head_v_dim  # 输出侧逐头

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.gk_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.gv_dim, bias=False)

        if use_short_conv:
            from fla.modules import ShortConvolution

            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, bias=conv_bias, activation="silu")
            self.k_conv1d = ShortConvolution(self.gk_dim, conv_size, bias=conv_bias, activation="silu")
            self.v_conv1d = ShortConvolution(self.gv_dim, conv_size, bias=conv_bias, activation="silu")

        # 遗忘门/擦除写入门（组级标量），投影与bias设定对齐GDN原版（use_mamba_gate=True）
        self.gk_proj = nn.Linear(hidden_size, self.num_groups, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_groups, bias=True)

        # 遗忘门参数（组级标量），mamba初始化对齐GDN原版
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_groups, dtype=torch.float32).uniform_(0, 16)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(self.num_groups, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        # LSA静态还原矩阵 P [H, d_v, d_c]
        if use_lsa:
            self.p_mat = nn.Parameter(torch.empty(self.num_heads, self.head_v_dim, self.latent_dim))

        # 输出路径：逐头SiLU门控RMSNorm + 输出投影（g_proj单层对齐GDN原版）
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        try:
            from fla.modules import FusedRMSNormSwishGate

            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        except ImportError:
            from .gdn2 import RMSNormSwishGate

            self.o_norm = RMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2**-2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if module is self and self.use_lsa:
            nn.init.xavier_uniform_(self.p_mat, gain=2**-2.5)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ):
        assert attention_mask is None, "预训练路径不支持padding mask（数据为packed定长块）。"
        B, T, _ = hidden_states.shape
        mode = "fused_recurrent" if (T <= 64 and not self.training and self.mode != "naive") else self.mode
        if self.training:
            assert mode in ("chunk", "naive"), "训练只支持chunk模式（naive仅用于CPU测试）。"

        last_state = None
        if past_key_values is not None:
            from fla.layers.utils import get_layer_cache

            last_state = get_layer_cache(self, past_key_values)

        if self.use_short_conv:
            conv_q, conv_k, conv_v = (last_state["conv_state"] if last_state is not None else (None,) * 3)
            q, conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=use_cache)
            k, conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=use_cache)
            v, conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=use_cache)
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        # 组级标量log遗忘门，fp32保证下游cumsum数值稳定
        g = -self.A_log.float().exp() * F.softplus(self.gk_proj(hidden_states).float() + self.dt_bias)
        b = self.b_proj(hidden_states).sigmoid()

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (g d) -> ... g d", d=self.head_k_dim)
        v = rearrange(v, "... (g d) -> ... g d", d=self.latent_dim)

        # 策略2：组级张量repeat到H份进kernel（组内头共享，头g*I..(g+1)*I-1属于组g）
        if self.heads_per_group > 1:
            k, v = (repeat(x, "... g d -> ... (g i) d", i=self.heads_per_group) for x in (k, v))
            g, b = (repeat(x, "... g -> ... (g i)", i=self.heads_per_group) for x in (g, b))

        if self.allow_neg_eigval:
            b = b * 2.0

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if mode == "chunk":
            from ..kernels import get_chunk_gated_delta_rule

            o, recurrent_state = get_chunk_gated_delta_rule()(
                q=q, k=k, v=v, g=g, beta=b,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.get("cu_seqlens"),
            )
        elif mode == "fused_recurrent":
            from ..kernels import get_fused_recurrent_gated_delta_rule

            o, recurrent_state = get_fused_recurrent_gated_delta_rule()(
                q=q, k=k, v=v, g=g, beta=b,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.get("cu_seqlens"),
            )
        elif mode == "naive":
            from .naive import naive_gdn2_recurrence

            # GDN ≡ GDN2：b=β·1（k维）、w=β·1（v维）、g标量广播到k维
            o, recurrent_state = naive_gdn2_recurrence(
                q, k, v,
                g.unsqueeze(-1).expand(B, T, self.num_heads, self.head_k_dim),
                b.unsqueeze(-1).expand(B, T, self.num_heads, self.head_k_dim),
                b.unsqueeze(-1).expand(B, T, self.num_heads, self.latent_dim),
                initial_state=recurrent_state,
            )
            o = o.to(hidden_states.dtype)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            from fla.layers.utils import update_layer_cache

            update_layer_cache(
                self,
                past_key_values,
                recurrent_state=recurrent_state,
                conv_state=(conv_q, conv_k, conv_v) if self.use_short_conv else None,
                offset=T,
            )

        # LSA出口还原：潜空间读取结果乘静态P回到每头value空间
        if self.use_lsa:
            o = torch.einsum("bthc,hvc->bthv", o, self.p_mat.to(o.dtype))

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        return self.o_proj(o), None, past_key_values
