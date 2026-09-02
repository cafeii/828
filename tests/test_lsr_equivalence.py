# LSR 数学等价性单测（CPU，纯 torch）。数学依据见 docs/research.md。
# 运行：uv run pytest tests/test_lsr_equivalence.py -q（repo 根）

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

try:
    from lit_gpt.mixers.naive import (
        naive_gdn2_recurrence,
        naive_lsr_expanded_forward,
        naive_lsr_forward,
    )
except ImportError:
    # lit_gpt/__init__.py 可能 import 尚不存在的 config；绕开包顶层直接按路径加载
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "naive", Path(__file__).parent.parent / "model" / "lit_gpt" / "mixers" / "naive.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    naive_gdn2_recurrence = _mod.naive_gdn2_recurrence
    naive_lsr_forward = _mod.naive_lsr_forward
    naive_lsr_expanded_forward = _mod.naive_lsr_expanded_forward

B, T, H, G, DK, DV, DC = 2, 8, 8, 2, 16, 16, 16
I = H // G


def make_inputs(dtype=torch.float64, dv=DV, dc=DC, seed=0):
    """构造贴近真实语义的输入：g 为负 log-decay，b/w 在 [0,1]。"""
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, dtype=dtype)

    q = rand(B, T, H, DK)
    k = rand(B, T, G, DK)
    c = rand(B, T, G, dc)
    g = -F.softplus(rand(B, T, G, DK))
    b = rand(B, T, G, DK).sigmoid()
    w = rand(B, T, G, dc).sigmoid()
    P = rand(H, dv, dc) / dc**0.5
    return q, k, c, g, b, w, P


def expand_initial_state(T0, P):
    """潜初始 state T0:[B,G,dk,dc] → per-head 真实初始 state S0[h]=T0[g(h)] @ P[h]^T。"""
    return torch.einsum("bhkc,hvc->bhkv", T0.repeat_interleave(H // G, dim=1), P)


def test_p_factorization():
    """两条路径严格等价：出口乘 P（潜递归）== 入口展开 P（真实 state 递归）。"""
    q, k, c, g, b, w, P = make_inputs(torch.float64)
    o_latent, _ = naive_lsr_forward(q, k, c, g, b, w, P)
    o_expanded, _ = naive_lsr_expanded_forward(q, k, c, g, b, w, P)
    assert torch.allclose(o_latent, o_expanded, atol=1e-10)

    # fp32 下容差放宽
    q, k, c, g, b, w, P = make_inputs(torch.float32, seed=1)
    o_latent, _ = naive_lsr_forward(q, k, c, g, b, w, P)
    o_expanded, _ = naive_lsr_expanded_forward(q, k, c, g, b, w, P)
    assert torch.allclose(o_latent, o_expanded, atol=1e-5)


def test_identity_p_reduces_to_gqa():
    """dc=dv 且 P_{g,i}=I 时，LSR 退化为 GQA（组级 v 直接被组内所有 q 头读取）。"""
    q, k, c, g, b, w, _ = make_inputs(torch.float64)
    P = torch.eye(DV, dtype=torch.float64).expand(H, DV, DC).contiguous()
    o_lsr, _ = naive_lsr_forward(q, k, c, g, b, w, P)

    rep = lambda x: x.repeat_interleave(I, dim=2)
    o_gqa, _ = naive_gdn2_recurrence(q, rep(k), rep(c), rep(g), rep(b), rep(w))
    assert torch.allclose(o_lsr, o_gqa, atol=1e-12)


def test_state_shape():
    """潜 state 是 G 份：T_final 形状 [B,G,dk,dc]（组内各头一致性在函数内断言）。"""
    q, k, c, g, b, w, P = make_inputs(torch.float64)
    o, T_final = naive_lsr_forward(q, k, c, g, b, w, P)
    assert o.shape == (B, T, H, DV)
    assert T_final.shape == (B, G, DK, DC)

    _, S_final = naive_lsr_expanded_forward(q, k, c, g, b, w, P)
    assert S_final.shape == (B, H, DK, DV)
    # 真实 state 与潜 state 满足 S[h] = T[g(h)] @ P[h]^T
    assert torch.allclose(S_final, expand_initial_state(T_final, P), atol=1e-10)


def test_one_step_manual():
    """T=1 时递归与手写单步公式一致，验证门语义（含非零初始 state 的完整公式）。"""
    gen = torch.Generator().manual_seed(42)
    rand = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    q, k = rand(B, 1, H, DK), rand(B, 1, H, DK)
    v = rand(B, 1, H, DV)
    g = -F.softplus(rand(B, 1, H, DK))
    b = rand(B, 1, H, DK).sigmoid()
    w = rand(B, 1, H, DV).sigmoid()
    S0 = rand(B, H, DK, DV)

    o, S1 = naive_gdn2_recurrence(q, k, v, g, b, w, initial_state=S0)

    qn = F.normalize(q[:, 0], p=2, dim=-1)
    kn = F.normalize(k[:, 0], p=2, dim=-1)
    # S_1 = (I - k (b⊙k)^T) Diag(exp(g)) S_0 + k (w⊙v)^T
    Sd = torch.exp(g[:, 0]).unsqueeze(-1) * S0
    S1_manual = (
        Sd
        - torch.einsum("bhk,bhi,bhiv->bhkv", kn, b[:, 0] * kn, Sd)
        + torch.einsum("bhk,bhv->bhkv", kn, w[:, 0] * v[:, 0])
    )
    o_manual = DK**-0.5 * torch.einsum("bhk,bhkv->bhv", qn, S1_manual)
    assert torch.allclose(S1, S1_manual, atol=1e-12)
    assert torch.allclose(o[:, 0], o_manual, atol=1e-12)


def test_initial_state_equivalence():
    """非零初始潜 state：两条路径带同一 T_0（expanded 侧初始 S_0 = T_0 @ P^T）仍等价。"""
    q, k, c, g, b, w, P = make_inputs(torch.float64, seed=7)
    gen = torch.Generator().manual_seed(100)
    T0 = torch.randn(B, G, DK, DC, generator=gen, dtype=torch.float64)

    o_latent, T_final = naive_lsr_forward(q, k, c, g, b, w, P, initial_state=T0)
    o_expanded, S_final = naive_lsr_expanded_forward(
        q, k, c, g, b, w, P, initial_state=expand_initial_state(T0, P)
    )
    assert torch.allclose(o_latent, o_expanded, atol=1e-10)
    assert torch.allclose(S_final, expand_initial_state(T_final, P), atol=1e-10)


# ---- GDN / KDA 骨架形态（β 为组级标量；GDN 为标量 decay，KDA 为 channel-wise decay）----
# GDN: S_t = α(I - βkk^T)S_{t-1} + βkv^T ≡ GDN2 取 b=β·1(k维)、w=β·1(v维)、g 标量广播到 k 维。
# KDA: channel-wise decay ≡ GDN2 同取 b=β·1、w=β·1，g 直接对应。
# 两骨架无独立写门，LSR 写入为 β k c^T，P 可提出性依然严格成立。


def make_scalar_beta_inputs(backbone, dtype=torch.float64, seed=0):
    """GDN/KDA 形态输入：标量门广播成 GDN2 的 b/w/g 后可直接走 naive_gdn2_recurrence。"""
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, dtype=dtype)

    q = rand(B, T, H, DK)
    k = rand(B, T, G, DK)
    c = rand(B, T, G, DC)
    if backbone == "gdn":  # per-head 标量 log-decay，广播到 k 维
        g = (-F.softplus(rand(B, T, G))).unsqueeze(-1).expand(B, T, G, DK)
    else:  # kda: channel-wise log-decay
        g = -F.softplus(rand(B, T, G, DK))
    beta = rand(B, T, G).sigmoid()
    b = beta.unsqueeze(-1).expand(B, T, G, DK)
    w = beta.unsqueeze(-1).expand(B, T, G, DC)
    P = rand(H, DV, DC) / DC**0.5
    return q, k, c, g, b, w, P


@pytest.mark.parametrize("backbone", ["gdn", "kda"])
def test_backbone_p_factorization(backbone):
    """GDN/KDA 形态下策略1（入口展开v=Pc）与策略2（出口乘P）严格等价。"""
    q, k, c, g, b, w, P = make_scalar_beta_inputs(backbone)
    o_latent, _ = naive_lsr_forward(q, k, c, g, b, w, P)
    o_expanded, _ = naive_lsr_expanded_forward(q, k, c, g, b, w, P)
    assert torch.allclose(o_latent, o_expanded, atol=1e-10)

    # fp32 下容差放宽
    q, k, c, g, b, w, P = make_scalar_beta_inputs(backbone, torch.float32, seed=1)
    o_latent, _ = naive_lsr_forward(q, k, c, g, b, w, P)
    o_expanded, _ = naive_lsr_expanded_forward(q, k, c, g, b, w, P)
    assert torch.allclose(o_latent, o_expanded, atol=1e-5)


@pytest.mark.parametrize("backbone", ["gdn", "kda"])
def test_backbone_identity_p_reduces_to_gqa(backbone):
    """GDN/KDA 形态下 dc=dv 且 P=I 时，LSR 退化为 GQA（组级 v 直读）。"""
    q, k, c, g, b, w, _ = make_scalar_beta_inputs(backbone)
    P = torch.eye(DV, dtype=torch.float64).expand(H, DV, DC).contiguous()
    o_lsr, _ = naive_lsr_forward(q, k, c, g, b, w, P)

    rep = lambda x: x.repeat_interleave(I, dim=2)
    o_gqa, _ = naive_gdn2_recurrence(q, rep(k), rep(c), rep(g), rep(b), rep(w))
    assert torch.allclose(o_lsr, o_gqa, atol=1e-12)
