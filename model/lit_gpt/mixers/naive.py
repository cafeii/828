# 纯 torch 逐 token 递归参考实现（CPU 可跑，无 triton/fla 依赖）。
# 语义与 third_party/GatedDeltaNet-2 的 chunk_gdn2 kernel 对齐
# （use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False）。
# LSR 数学推导见 docs/research.md。

import torch
import torch.nn.functional as F

__all__ = ["naive_gdn2_recurrence", "naive_lsr_forward", "naive_lsr_expanded_forward"]


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    # 至少 fp32；输入已是 fp64 则保持 fp64（供高精度等价性测试）
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def naive_gdn2_recurrence(q, k, v, g, b, w, scale=None, initial_state=None):
    """逐 token 递归的 GDN2 参考实现。

    S_t = (I - k_t (b_t⊙k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t⊙v_t)^T
    o_t = scale * (q_t S_t)

    Diag(exp(g_t)) 作用在 S 的 k 维（行）：S_decayed[i,j] = exp(g)[i] * S[i,j]。
    q/k 进入递归前做 L2 normalize（dim=-1），与 kernel 的
    use_qk_l2norm_in_kernel=True 对齐。全程至少 fp32 计算（fp64 输入保持 fp64）。

    Args:
        q, k: [B, T, H, dk]
        v:    [B, T, H, dv]
        g:    [B, T, H, dk]  log-decay（负值）
        b:    [B, T, H, dk]  擦除门
        w:    [B, T, H, dv]  写入门
        scale: 默认 dk ** -0.5
        initial_state: [B, H, dk, dv]，默认零

    Returns:
        (o: [B, T, H, dv], S: [B, H, dk, dv])
    """
    dtype = _compute_dtype(q)
    q, k, v, g, b, w = (x.to(dtype) for x in (q, k, v, g, b, w))
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    if scale is None:
        scale = dk**-0.5

    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    if initial_state is None:
        S = q.new_zeros(B, H, dk, dv)
    else:
        S = initial_state.to(dtype).clone()

    o = q.new_empty(B, T, H, dv)
    for t in range(T):
        kt = k[:, t]                                   # [B, H, dk]
        S = torch.exp(g[:, t]).unsqueeze(-1) * S       # Diag(exp(g)) 行缩放
        bkS = torch.einsum("bhk,bhkv->bhv", b[:, t] * kt, S)
        S = S - kt.unsqueeze(-1) * bkS.unsqueeze(-2)   # (I - k (b⊙k)^T) S
        S = S + kt.unsqueeze(-1) * (w[:, t] * v[:, t]).unsqueeze(-2)
        o[:, t] = scale * torch.einsum("bhk,bhkv->bhv", q[:, t], S)
    return o, S


def naive_lsr_forward(q, k, c, g, b, w, P, scale=None, initial_state=None):
    """LSR 潜空间递归：组级潜 state T_g 递归 + 出口乘 P。

    组级张量 repeat_interleave 到 H 头（头 g*I..(g+1)*I-1 属于组 g）后走
    naive_gdn2_recurrence（v 口传 c，w 口传 w），得潜输出 o_latent:[B,T,H,dc]，
    再 o = einsum('bthc,hvc->bthv', o_latent, P)。

    Args:
        q:       [B, T, H, dk]
        k, g, b: [B, T, G, dk]
        c, w:    [B, T, G, dc]
        P:       [H, dv, dc]
        initial_state: 潜 state T_0，[B, G, dk, dc]，默认零

    Returns:
        (o: [B, T, H, dv], T_final: [B, G, dk, dc])
        T_final 取每组第一个头的 state（函数内断言组内各头 state 一致）。
    """
    H = q.shape[2]
    G = k.shape[2]
    I = H // G
    k, g, b, c, w = (x.repeat_interleave(I, dim=2) for x in (k, g, b, c, w))
    S0 = None if initial_state is None else initial_state.repeat_interleave(I, dim=1)
    o_latent, S = naive_gdn2_recurrence(q, k, c, g, b, w, scale, S0)
    o = torch.einsum("bthc,hvc->bthv", o_latent, P.to(o_latent.dtype))

    S = S.unflatten(1, (G, I))                          # [B, G, I, dk, dc]
    assert torch.allclose(S, S[:, :, :1].expand_as(S)), "组内各头潜 state 应一致"
    return o, S[:, :, 0]


def naive_lsr_expanded_forward(q, k, c, g, b, w, P, scale=None, initial_state=None):
    """策略 1 参考：入口展开 v_{t,g,i} = P_{g,i} (w_t⊙c_t)，per-head 真实 state 递归。

    P reshape 为 [G, I, dv, dc]，v = einsum('btgc,givc->btgiv') 展平到 H 头，
    组级 k/g/b repeat 到头，w 口传全 1，走 naive_gdn2_recurrence。
    与 naive_lsr_forward 数学等价（P 可从递归提出，见 docs/research.md）。

    Args:
        形状同 naive_lsr_forward；initial_state 为 per-head 真实 state
        S_0:[B, H, dk, dv]（对应潜 T_0 时应取 S_0[h] = T_0[g(h)] @ P[h]^T）。

    Returns:
        (o: [B, T, H, dv], S_final: [B, H, dk, dv])
    """
    B, T, G, dc = c.shape
    H, dv, _ = P.shape
    I = H // G
    c_gated = w * c                                     # [B, T, G, dc]
    Pg = P.reshape(G, I, dv, dc).to(c_gated.dtype)
    v = torch.einsum("btgc,givc->btgiv", c_gated, Pg).reshape(B, T, H, dv)
    k, g, b = (x.repeat_interleave(I, dim=2) for x in (k, g, b))
    return naive_gdn2_recurrence(q, k, v, g, b, torch.ones_like(v), scale, initial_state)
