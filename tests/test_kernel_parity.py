# 服务器GPU单测：真实chunk_gdn2 kernel上的LSA策略等价性验证。
# 运行（服务器，lzc-rnn env）: pytest tests/test_kernel_parity.py -q
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from lit_gpt.mixers.naive import naive_gdn2_recurrence  # noqa: E402

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA")

B, T, H, G, DK, DV, DC = 2, 256, 8, 2, 64, 64, 64
I = H // G


def make_inputs(seed=0, dtype=torch.bfloat16, device="cuda"):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, DK, dtype=dtype, device=device)
    k = torch.randn(B, T, G, DK, dtype=dtype, device=device)
    c = torch.randn(B, T, G, DC, dtype=dtype, device=device)
    g = -torch.nn.functional.softplus(torch.randn(B, T, G, DK, device=device))  # fp32 log-decay
    b = torch.rand(B, T, G, DK, dtype=dtype, device=device)
    w = torch.rand(B, T, G, DC, dtype=dtype, device=device)
    P = torch.randn(H, DV, DC, dtype=dtype, device=device) / DC**0.5
    return q, k, c, g, b, w, P


def rep(x):
    """组级→头级 repeat_interleave（与mixers/gdn2.py的repeat语义一致）"""
    return x.repeat_interleave(I, dim=2)


@requires_cuda
def test_chunk_vs_naive():
    """chunk_gdn2 kernel vs naive逐token递归（GQA repeat路径）"""
    from lit_gpt.kernels import get_chunk_gdn2

    q, k, c, g, b, w, _ = make_inputs()
    o_kernel, s_kernel = get_chunk_gdn2()(
        q=q, k=rep(k), v=rep(c), g=rep(g), b=rep(b), w=rep(w),
        output_final_state=True, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
    )
    o_naive, s_naive = naive_gdn2_recurrence(
        q.float(), rep(k).float(), rep(c).float(), rep(g).float(), rep(b).float(), rep(w).float()
    )
    assert torch.allclose(o_kernel.float(), o_naive, atol=5e-2, rtol=5e-2), \
        f"max diff: {(o_kernel.float() - o_naive).abs().max()}"
    assert torch.allclose(s_kernel.float(), s_naive, atol=5e-2, rtol=5e-2)


@requires_cuda
def test_strategy1_vs_strategy2():
    """策略1（入口展开v=Pc）vs 策略2（出口乘P）在真实kernel上等价 → P可提出性"""
    from lit_gpt.kernels import get_chunk_gdn2

    chunk_gdn2 = get_chunk_gdn2()
    q, k, c, g, b, w, P = make_inputs()

    # 策略2：潜空间递归，出口乘P
    o2_latent, _ = chunk_gdn2(
        q=q, k=rep(k), v=rep(c), g=rep(g), b=rep(b), w=rep(w),
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
    )
    o2 = torch.einsum("bthc,hvc->bthv", o2_latent, P)

    # 策略1：写门先折进c，入口展开成per-head真实v，w口传1
    c_gated = w * c  # [B,T,G,DC]
    Pg = P.view(G, I, DV, DC)
    v_expanded = torch.einsum("btgc,givc->btgiv", c_gated, Pg).reshape(B, T, H, DV)
    o1, _ = chunk_gdn2(
        q=q, k=rep(k), v=v_expanded, g=rep(g), b=rep(b), w=torch.ones_like(v_expanded),
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
    )
    assert torch.allclose(o1.float(), o2.float(), atol=5e-2, rtol=5e-2), \
        f"max diff: {(o1.float() - o2.float()).abs().max()}"


@requires_cuda
def test_mixer_layer_forward_backward():
    """整层GatedDeltaNet2（LSA开）前向反向可跑，梯度有限"""
    from lit_gpt.mixers.gdn2 import GatedDeltaNet2

    layer = GatedDeltaNet2(
        hidden_size=256, num_heads=H, num_groups=G, head_dim=DK, use_lsa=True,
    ).cuda().to(torch.bfloat16)
    layer.train()
    x = torch.randn(B, T, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    o, _, _ = layer(x)
    assert o.shape == x.shape
    o.sum().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None and p.grad.isfinite().all(), f"梯度异常: {name}"
    assert layer.p_mat.grad.abs().sum() > 0, "P矩阵没有收到梯度"
