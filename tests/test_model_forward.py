# 整模型级单测（CPU可跑）：各配置实例化、前向反向、loss有限性。
# 运行: uv run pytest tests/test_model_forward.py -q
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from lit_gpt.config import Config, name_to_config  # noqa: E402


def make_tiny(**kw):
    base = dict(
        name="tiny_test", block_size=64, vocab_size=256, padding_multiple=64,
        mixer="gdn2", mixer_per_layer=1, n_layer=2, n_head=4, n_embd=64,
        intermediate_size=176, parallel_residual=False, bias=False,
        _norm_class="RMSNorm", _mlp_class="LLaMAMLP", nope=True, mamba_init=True,
        head_dim=16, use_short_conv=False,  # CPU无fla，关掉short conv
    )
    base.update(kw)
    return Config(**base)


CASES = {
    "mha": dict(),
    "gqa": dict(num_groups=2),
    "gqa_expandv": dict(num_groups=2, expand_v=2.0),
    "gva": dict(num_groups=2, num_v_heads=4),
    "gva_h4": dict(n_head=2, num_groups=2, num_v_heads=8, head_dim=16),  # q-k-v=2-2-8 缩比对应 4-4-16（v>q 路径）
    "lsr": dict(num_groups=2, use_lsr=True),
    "lsr_dc32": dict(num_groups=2, use_lsr=True, lsr_latent_dim=32),
    "lsr_pI": dict(num_groups=2, use_lsr=True, lsr_init_p="identity"),
    # GDN/KDA 骨架（同形态覆盖）
    "gdn_mha": dict(mixer="gdn"),
    "gdn_gqa": dict(mixer="gdn", num_groups=2),
    "gdn_lsr": dict(mixer="gdn", num_groups=2, use_lsr=True),
    "kda_mha": dict(mixer="kda"),
    "kda_gqa": dict(mixer="kda", num_groups=2),
    "kda_lsr": dict(mixer="kda", num_groups=2, use_lsr=True),
}


@pytest.mark.parametrize("case", CASES.keys())
def test_forward_backward(case):
    from lit_gpt.model import GPT

    torch.manual_seed(0)
    config = make_tiny(**CASES[case])
    model = GPT(config)
    # naive模式跑CPU（chunk需triton）
    for block in model.transformer.h:
        block.attn.mode = "naive"
    model.train()
    idx = torch.randint(0, 256, (2, 32))
    logits = model(idx)
    assert logits.shape == (2, 32, config.padded_vocab_size)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1))
    assert loss.isfinite()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(g.isfinite().all() for g in grads)


def test_lsr_state_smaller_than_mha():
    """LSR的递归状态参数化：潜state按G份（通过mixer属性核对形状语义）"""
    from lit_gpt.mixers.gdn2 import GatedDeltaNet2

    m = GatedDeltaNet2(hidden_size=64, num_heads=4, num_groups=2, head_dim=16, use_lsr=True, use_short_conv=False)
    assert m.k_proj.out_features == 2 * 16  # 组级k
    assert m.v_proj.out_features == 2 * 16  # 组级潜v
    assert m.q_proj.out_features == 4 * 16  # 逐头q
    assert m.p_mat.shape == (4, 16, 16)
    assert m.A_log.shape == (2,)  # 组级遗忘门


def test_lsr_identity_init():
    """lsr_init_p='identity'：P=I 精确热启动；xavier 默认值不受影响；d_c≠d_v 时矩形主对角。"""
    from lit_gpt.mixers.gdn2 import GatedDeltaNet2

    kw = dict(hidden_size=64, num_heads=4, num_groups=2, head_dim=16, use_lsr=True, use_short_conv=False)
    m = GatedDeltaNet2(**kw, lsr_init_p="identity")
    assert torch.equal(m.p_mat, torch.eye(16).expand(4, 16, 16))

    m_x = GatedDeltaNet2(**kw)  # 默认 xavier
    assert m_x.lsr_init_p == "xavier" and not torch.equal(m_x.p_mat, m.p_mat)

    m_r = GatedDeltaNet2(**kw, lsr_init_p="identity", lsr_latent_dim=8)  # 矩形 d_v=16 > d_c=8
    expected = torch.zeros(4, 16, 8)
    expected[:, torch.arange(8), torch.arange(8)] = 1.0
    assert torch.equal(m_r.p_mat, expected)

    # P=I 时整层前向与 GQA 数值一致（naive 模式）
    from lit_gpt.mixers.gdn2 import GatedDeltaNet2 as G2

    torch.manual_seed(0)
    lsr = G2(hidden_size=64, num_heads=4, num_groups=2, head_dim=16, use_lsr=True,
             lsr_init_p="identity", use_short_conv=False, mode="naive").double()
    torch.manual_seed(0)
    gqa = G2(hidden_size=64, num_heads=4, num_groups=2, head_dim=16, use_lsr=False,
             use_short_conv=False, mode="naive").double()
    lsr.load_state_dict({k: v for k, v in gqa.state_dict().items() if k in lsr.state_dict()}, strict=False)
    # p_mat 不在 gqa state_dict 里，保持 I
    x = torch.randn(2, 8, 64, dtype=torch.float64)
    o_lsr, _, _ = lsr(x)
    o_gqa, _, _ = gqa(x)
    assert torch.allclose(o_lsr, o_gqa, atol=1e-10), (o_lsr - o_gqa).abs().max()


def test_gva_v_gt_q_path():
    """num_v_heads > num_heads（GVA 4-4-16 缩比 2-2-8）：前向形状/梯度正常，且能恢复 v>q 信息。"""
    from lit_gpt.mixers.gdn2 import GatedDeltaNet2

    torch.manual_seed(0)
    m = GatedDeltaNet2(hidden_size=64, num_heads=2, num_groups=2, head_dim=16,
                       num_v_heads=8, use_short_conv=False, mode="naive").double()
    assert m.q_proj.out_features == 2 * 16 and m.v_proj.out_features == 8 * 16
    x = torch.randn(2, 8, 64, dtype=torch.float64, requires_grad=True)
    o, state, _ = m(x)
    assert o.shape == x.shape
    o.sum().backward()
    assert x.grad is not None and x.grad.isfinite().all()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters())

    # 语义核对：组内 r=J/H 个 v 头对应同一 q 头，输出为均值；且 v>q 下不同 v 头读出确实不同
    from lit_gpt.mixers.naive import naive_gdn2_recurrence
    B, T, H, G, J, DK, DV = 1, 4, 2, 2, 8, 16, 16
    r = J // H
    q = torch.randn(B, T, H, DK, dtype=torch.float64)
    k = torch.randn(B, T, G, DK, dtype=torch.float64)
    v = torch.randn(B, T, J, DV, dtype=torch.float64)
    g = torch.zeros(B, T, G, DK, dtype=torch.float64)
    b = torch.zeros(B, T, G, DK, dtype=torch.float64)
    w = torch.ones(B, T, J, DV, dtype=torch.float64)
    qM = q.repeat_interleave(r, dim=2)
    kM = k.repeat_interleave(J // G, dim=2)
    gM = g.repeat_interleave(J // G, dim=2)
    bM = b.repeat_interleave(J // G, dim=2)
    oM, _ = naive_gdn2_recurrence(qM, kM, v, gM, bM, w)
    o_grouped = oM.view(B, T, H, r, DV).mean(dim=3)
    assert o_grouped.shape == (B, T, H, DV)
    # 组内不同 v 头的读出确实不同（v>q 信息量未被 q 广播抹掉）
    assert not torch.allclose(oM[:, :, 0], oM[:, :, 1], atol=1e-6)


def test_registered_configs_instantiate():
    """config.py注册的所有配置能实例化Config（不建模型，防手误）"""
    for name in name_to_config:
        c = Config.from_name(name)
        assert c.n_head % (c.num_groups or c.n_head) == 0
