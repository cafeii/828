# 增量解码一致性单测（CPU可跑，naive模式）：
# prefill+逐token增量decode 与 一次性全量forward 的logits应一致。
# 运行: uv run pytest tests/test_eval_wrapper.py -q
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from lit_gpt.config import Config  # noqa: E402
from lit_gpt.mixers.cache import SimpleCache  # noqa: E402
from lit_gpt.model import GPT  # noqa: E402


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
    "gqa": dict(num_groups=2),
    "lsr": dict(num_groups=2, use_lsr=True),
}


def build(case):
    torch.manual_seed(0)
    model = GPT(make_tiny(**CASES[case]))
    for block in model.transformer.h:
        block.attn.mode = "naive"
    model.eval()
    return model


@pytest.mark.parametrize("case", CASES.keys())
def test_incremental_matches_full(case):
    model = build(case)
    torch.manual_seed(1)
    idx = torch.randint(0, 256, (2, 24))

    with torch.no_grad():
        full = model(idx)

        # prefill前16个token，随后逐token decode
        cache = SimpleCache()
        prefill = model(idx[:, :16], past_key_values=cache)
        step_logits = [prefill]
        for t in range(16, 24):
            step_logits.append(model(idx[:, t : t + 1], past_key_values=cache))
        inc = torch.cat(step_logits, dim=1)

    assert inc.shape == full.shape
    torch.testing.assert_close(inc, full, rtol=1e-4, atol=1e-4)
    assert cache.seen_tokens == 24


@pytest.mark.parametrize("case", CASES.keys())
def test_wrapper_forward(case):
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "eval"))
    from wrapper import LitGPTConfig, LitGPTForCausalLM

    lit_config = make_tiny(**CASES[case])
    torch.manual_seed(0)
    wrapper = LitGPTForCausalLM(LitGPTConfig.from_litgpt(lit_config), lit_config)
    for block in wrapper.gpt.transformer.h:
        block.attn.mode = "naive"
    wrapper.eval()

    torch.manual_seed(1)
    idx = torch.randint(0, 256, (1, 20))
    with torch.no_grad():
        full = wrapper(idx).logits
        out = wrapper(idx[:, :12], use_cache=True)
        cache = out.past_key_values
        logits = [out.logits]
        for t in range(12, 20):
            logits.append(wrapper(idx[:, t : t + 1], past_key_values=cache, use_cache=True).logits)
        inc = torch.cat(logits, dim=1)

    torch.testing.assert_close(inc, full, rtol=1e-4, atol=1e-4)
