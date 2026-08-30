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
    "lsa": dict(num_groups=2, use_lsa=True),
    "lsa_dc32": dict(num_groups=2, use_lsa=True, lsa_latent_dim=32),
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


def test_lsa_state_smaller_than_mha():
    """LSA的递归状态参数化：潜state按G份（通过mixer属性核对形状语义）"""
    from lit_gpt.mixers.gdn2 import GatedDeltaNet2

    m = GatedDeltaNet2(hidden_size=64, num_heads=4, num_groups=2, head_dim=16, use_lsa=True, use_short_conv=False)
    assert m.k_proj.out_features == 2 * 16  # 组级k
    assert m.v_proj.out_features == 2 * 16  # 组级潜v
    assert m.q_proj.out_features == 4 * 16  # 逐头q
    assert m.p_mat.shape == (4, 16, 16)
    assert m.A_log.shape == (2,)  # 组级遗忘门


def test_registered_configs_instantiate():
    """config.py注册的所有配置能实例化Config（不建模型，防手误）"""
    for name in name_to_config:
        c = Config.from_name(name)
        assert c.n_head % (c.num_groups or c.n_head) == 0
