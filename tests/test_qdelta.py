"""Formula, initialization, and CUDA parity gates for paper Q-Delta."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from lit_gpt.config import Config
from lit_gpt.model import GPT
from lit_gpt.mixers.qdelta_rule import qdelta_reference, qdelta_rule


def inputs(*, T=17, K=8, V=6, dtype=torch.float64, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(260608804)
    B, H = 2, 3
    q = torch.randn(B, T, H, K, generator=generator, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, generator=generator, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, generator=generator, dtype=dtype, device=device)
    g = -torch.rand(B, T, H, generator=generator, dtype=dtype, device=device) * 0.08 - 0.001
    beta = torch.rand(B, T, H, generator=generator, dtype=dtype, device=device) * 0.8 + 0.1
    lamb = torch.rand(B, T, H, generator=generator, dtype=dtype, device=device) * 0.8 + 0.1
    state = torch.randn(B, H, K, V, generator=generator, dtype=dtype, device=device) * 0.1
    return q, k, v, g, beta, lamb, state


def dense_transition(q, k, v, g, beta, lamb, *, query_sign, initial_state):
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    state = initial_state
    outputs = []
    scale = q.shape[-1] ** -0.5
    eye = torch.eye(q.shape[-1], dtype=q.dtype, device=q.device)
    for t in range(q.shape[1]):
        qt, kt, vt = q[:, t], k[:, t], v[:, t]
        alpha, strength = g[:, t].exp(), beta[:, t]
        mixed = kt + query_sign * lamb[:, t, :, None] * qt
        transition = eye - strength[..., None, None] * torch.einsum("bhk,bhj->bhkj", kt, mixed)
        state = alpha[..., None, None] * torch.einsum("bhkj,bhjv->bhkv", transition, state)
        state = state + strength[..., None, None] * torch.einsum("bhk,bhv->bhkv", kt, vt)
        outputs.append(scale * torch.einsum("bhk,bhkv->bhv", qt, state))
    return torch.stack(outputs, dim=1), state


@pytest.mark.parametrize("query_sign", [1.0, -1.0])
def test_fp64_formula_output_state_and_all_input_gradients(query_sign):
    xs = [x.requires_grad_() for x in inputs()]
    ref = [x.detach().clone().requires_grad_() for x in xs]
    actual = qdelta_reference(*xs[:6], query_sign=query_sign, initial_state=xs[6])
    expected = dense_transition(*ref[:6], query_sign=query_sign, initial_state=ref[6])
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target, rtol=2e-12, atol=2e-12)
    weights = [torch.randn_like(value) for value in actual]
    actual_grads = torch.autograd.grad(sum((value * weight).sum() for value, weight in zip(actual, weights)), xs)
    expected_grads = torch.autograd.grad(sum((value * weight).sum() for value, weight in zip(expected, weights)), ref)
    for value, target in zip(actual_grads, expected_grads):
        assert value.isfinite().all()
        torch.testing.assert_close(value, target, rtol=3e-11, atol=3e-11)


def test_qdelta_model_preserves_gdn_backbone_and_learns_lambda():
    overrides = dict(use_short_conv=False, _norm_class="RMSNorm")
    models = []
    for name in ("gdn_recall_tiny", "qdelta_recall_tiny", "qdelta_minus_recall_tiny"):
        torch.manual_seed(3407)
        config = Config.from_name(name, **overrides)
        model = GPT(config)
        model.apply(lambda module: model._init_weights(module, n_layer=config.n_layer))
        models.append(model)
    gdn, qdelta, qdelta_minus = models
    for candidate in (qdelta, qdelta_minus):
        candidate_parameters = dict(candidate.named_parameters())
        for name, parameter in gdn.named_parameters():
            torch.testing.assert_close(parameter, candidate_parameters[name], rtol=0, atol=0)
        lambda_parameters = [p for n, p in candidate.named_parameters() if ".lambda_proj." in n]
        assert sum(p.numel() for p in lambda_parameters) == 2 * 128 * 2
        for block in candidate.transformer.h:
            block.attn.mode = "naive"
        logits = candidate(torch.randint(0, 256, (2, 13)))
        F.cross_entropy(logits.flatten(0, 1), torch.randint(0, 256, (26,))).backward()
        assert all(
            p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
            for p in lambda_parameters
        )
    for plus, minus in zip(qdelta.parameters(), qdelta_minus.parameters()):
        torch.testing.assert_close(plus, minus, rtol=0, atol=0)
    assert qdelta.transformer.h[0].attn.qdelta_query_sign == 1.0
    assert qdelta_minus.transformer.h[0].attn.qdelta_query_sign == -1.0


@pytest.mark.parametrize("query_sign", [1.0, -1.0])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity requires an allocated GPU")
def test_cuda_output_state_and_all_input_gradients(query_sign):
    raw = inputs(T=65, K=64, V=64, dtype=torch.float32, device="cuda")
    gpu = [x.detach().to(torch.bfloat16 if i < 3 else torch.float32).requires_grad_() for i, x in enumerate(raw)]
    ref = [x.detach().float().requires_grad_() for x in gpu]
    actual = qdelta_rule(
        *gpu[:6], query_sign=query_sign, initial_state=gpu[6], output_final_state=True
    )
    expected = qdelta_reference(*ref[:6], query_sign=query_sign, initial_state=ref[6])
    for value, target in zip(actual, expected):
        relative = (value.float() - target).square().mean().sqrt() / target.square().mean().sqrt().clamp_min(1e-6)
        assert relative < 0.03
    weights = [torch.randn_like(value).float() for value in expected]
    actual_grads = torch.autograd.grad(sum((value.float() * weight).sum() for value, weight in zip(actual, weights)), gpu)
    expected_grads = torch.autograd.grad(sum((value * weight).sum() for value, weight in zip(expected, weights)), ref)
    for value, target in zip(actual_grads, expected_grads):
        assert value.isfinite().all()
        relative = (value.float() - target).square().mean().sqrt() / target.square().mean().sqrt().clamp_min(1e-6)
        assert relative < 0.09
