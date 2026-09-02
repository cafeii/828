import torch

from lit_gpt.config import Config
from lit_gpt.mixers.gdn import GatedDeltaNet
from lit_gpt.mixers.qr_gdn import QueryRecallGatedDeltaNet


def mixer_kwargs():
    return dict(
        hidden_size=64,
        num_heads=4,
        num_groups=4,
        head_dim=16,
        use_short_conv=False,
        mode="naive",
    )


def tiny_config(mixer):
    return Config(
        name=f"{mixer}_qr_test",
        block_size=32,
        vocab_size=256,
        padding_multiple=64,
        mixer=mixer,
        mixer_per_layer=1,
        n_layer=2,
        n_head=2,
        n_embd=128,
        intermediate_size=352,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        nope=True,
        mamba_init=True,
        head_dim=64,
        use_short_conv=False,
    )


def test_shared_mixer_parameters_match_native_gdn_initialization():
    torch.manual_seed(3407)
    gdn = GatedDeltaNet(**mixer_kwargs())
    torch.manual_seed(3407)
    qr = QueryRecallGatedDeltaNet(**mixer_kwargs())
    qr_state = qr.state_dict()
    for name, value in gdn.state_dict().items():
        torch.testing.assert_close(qr_state[name], value, rtol=0, atol=0)
    assert torch.count_nonzero(qr.qr_read_proj.weight) == 0
    assert torch.count_nonzero(qr.qr_read_proj.bias) == 0


def test_qr_model_initial_output_and_shared_parameters_match_gdn():
    from lit_gpt.model import GPT

    torch.manual_seed(42)
    gdn = GPT(tiny_config("gdn"))
    torch.manual_seed(42)
    qr = GPT(tiny_config("qr_gdn"))
    torch.manual_seed(1234)
    gdn.apply(lambda module: gdn._init_weights(module, gdn.config.n_layer))
    torch.manual_seed(1234)
    qr.apply(lambda module: qr._init_weights(module, qr.config.n_layer))
    for model in (gdn, qr):
        for block in model.transformer.h:
            block.attn.mode = "naive"

    qr_state = qr.state_dict()
    for name, value in gdn.state_dict().items():
        torch.testing.assert_close(qr_state[name], value, rtol=0, atol=0)

    tokens = torch.randint(0, 256, (2, 16))
    gdn.eval()
    qr.eval()
    with torch.no_grad():
        expected = gdn(tokens)
        actual = qr(tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qr_model_backward_reaches_read_gate_and_stays_finite():
    from lit_gpt.model import GPT

    torch.manual_seed(7)
    model = GPT(tiny_config("qr_gdn"))
    model.apply(lambda module: model._init_weights(module, model.config.n_layer))
    for block in model.transformer.h:
        block.attn.mode = "naive"
    model.train()
    tokens = torch.randint(0, 256, (2, 16))
    logits = model(tokens)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens.flatten())
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    for block in model.transformer.h:
        assert block.attn.qr_read_proj.weight.grad is not None
        assert torch.count_nonzero(block.attn.qr_read_proj.weight.grad) > 0


def test_qr_gate_moments_use_named_raw_sums_without_changing_output():
    torch.manual_seed(99)
    mixer = QueryRecallGatedDeltaNet(**mixer_kwargs())
    hidden = torch.randn(2, 11, 64)
    baseline = mixer(hidden)[0]
    mixer.reset_gate_stats()
    mixer.collect_gate_stats = True
    observed = mixer(hidden)[0]
    torch.testing.assert_close(observed, baseline, rtol=0, atol=0)
    moments = mixer.gate_moments()
    assert set(moments) == {
        "alpha_kv", "beta_kv", "alpha_qr", "beta_qr", "qr_read"
    }
    expected_count = 2 * 11 * 4
    for raw in moments.values():
        assert raw.dtype == torch.float64
        assert raw.shape == (3,)
        assert raw[2].item() == expected_count
    torch.testing.assert_close(moments["qr_read"][:2], torch.zeros(2, dtype=torch.float64))
