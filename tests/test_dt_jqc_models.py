import torch
import torch.nn.functional as F

from lit_gpt.config import Config
from lit_gpt.model import GPT


def build(name):
    torch.manual_seed(3407)
    cfg = Config.from_name(name, use_short_conv=False, _norm_class="RMSNorm")
    model = GPT(cfg)
    model.apply(lambda module: model._init_weights(module, n_layer=cfg.n_layer))
    return cfg, model


def shared_parameters(model):
    return {name: value for name, value in model.named_parameters() if ".recall_" not in name}


def test_new_models_preserve_shared_gdn_initialization_and_parameter_parity():
    _, gdn = build("gdn_recall_tiny")
    _, dt = build("dt_gdn_recall_tiny")
    _, jqc = build("jqc_gdn_recall_tiny")
    expected = shared_parameters(gdn)
    for model in (dt, jqc):
        actual = shared_parameters(model)
        assert actual.keys() == expected.keys()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert sum(p.numel() for p in dt.parameters()) == sum(p.numel() for p in jqc.parameters())


def test_cpu_forward_backward_and_gate_moments():
    tokens = torch.randint(0, 256, (2, 17))
    targets = torch.randint(0, 256, (2, 17))
    for name in ("dt_gdn_recall_tiny", "jqc_gdn_recall_tiny"):
        cfg, model = build(name)
        for block in model.transformer.h:
            block.attn.mode = "naive"
            block.attn.collect_gate_stats = True
            block.attn.reset_gate_stats()
        logits = model(tokens)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        assert logits.shape == (2, 17, cfg.padded_vocab_size)
        assert loss.isfinite()
        loss.backward()
        assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
        for block in model.transformer.h:
            moments = block.attn.gate_moments()
            assert set(moments) == {"alpha", "beta", "gamma", "gamma_saturated", "forgetting_margin"}
            assert all(item.dtype == torch.float64 and item[2] > 0 for item in moments.values())
            torch.testing.assert_close(
                block.attn.recall_proj.bias.sigmoid(),
                torch.full_like(block.attn.recall_proj.bias, 0.1),
            )
            assert block.attn.recall_proj.weight.grad is not None
            assert block.attn.recall_proj.weight.grad.abs().sum() > 0


def test_unvalidated_chunk_backend_fails_closed():
    _, model = build("dt_gdn_recall_tiny")
    with torch.no_grad():
        try:
            model(torch.randint(0, 256, (1, 7)))
        except NotImplementedError as error:
            assert "rank-two" in str(error)
        else:
            raise AssertionError("DT/JQC must not silently use a reference loop for chunk training")
