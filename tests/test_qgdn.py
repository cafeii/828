"""Mechanism tests against dense equations, not just the implementation itself."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from lit_gpt.config import Config
from lit_gpt.model import GPT
from lit_gpt.mixers.qgdn_rule import dplr_inputs, qgdn_reference, qgdn_rule
from lit_gpt.mixers.naive import naive_gdn2_recurrence


def inputs(T=7, K=4, V=3, dtype=torch.float64, device="cpu", gamma_value=None):
    torch.manual_seed(912)
    q, k = [torch.randn(2, T, 2, K, dtype=dtype, device=device) for _ in range(2)]
    v = torch.randn(2, T, 2, V, dtype=dtype, device=device)
    g = -torch.rand(2, T, 2, dtype=dtype, device=device) * 0.8
    beta = torch.rand_like(g)
    gamma = torch.rand_like(g) if gamma_value is None else torch.full_like(g, gamma_value)
    state = torch.randn(2, 2, K, V, dtype=dtype, device=device)
    return [x.requires_grad_() for x in (q, k, v, g, beta, gamma, state)]


def dense(q, k, v, g, beta, gamma, state, recall_mode="query"):
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    eye = torch.eye(q.shape[-1], dtype=q.dtype, device=q.device)
    out = []
    for t in range(q.shape[1]):
        r = q[:, t] if recall_mode == "query" else k[:, t]
        projection = eye if recall_mode == "isotropic" else r.unsqueeze(-1) @ r.unsqueeze(-2)
        D = g[:, t].exp()[..., None, None] * eye + (gamma[:, t] * (1 - g[:, t].exp()))[..., None, None] * projection
        kt = k[:, t]
        edit = eye - beta[:, t, :, None, None] * kt.unsqueeze(-1) @ kt.unsqueeze(-2)
        state = edit @ D @ state + beta[:, t, :, None, None] * kt.unsqueeze(-1) @ v[:, t].unsqueeze(-2)
        out.append((q[:, t].unsqueeze(-2) @ state).squeeze(-2) / q.shape[-1] ** 0.5)
    return torch.stack(out, dim=1), state


@pytest.mark.parametrize("recall_mode", ["query", "key", "isotropic"])
def test_dense_equation_outputs_and_all_gradients(recall_mode):
    xs = inputs()
    actual = qgdn_reference(*xs[:6], initial_state=xs[6], recall_mode=recall_mode)
    expected = dense(*xs, recall_mode=recall_mode)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
    weights = [torch.randn_like(x) for x in actual]
    gradients = [torch.autograd.grad(sum((a * w).sum() for a, w in zip(pair, weights)), xs, retain_graph=True)
                 for pair in (actual, expected)]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("gamma", [0.0, 0.1, 1.0])
def test_virtual_dplr_order(gamma):
    q, k, v, g, beta, gate, h0 = inputs(gamma_value=gamma)
    args = dplr_inputs(q, k, v, g, beta, gate)
    S = h0
    out = []
    for t in range(q.shape[1] * 2):
        # Independent interpretation of FLA's API (a reads; b writes).
        read = torch.einsum("bhk,bhkv->bhv", args["a"][:, t], S)
        S = args["gk"][:, t].exp()[..., None] * S
        S = S + args["b"][:, t, :, :, None] * read[..., None, :]
        S = S + args["k"][:, t, :, :, None] * args["v"][:, t, :, None, :]
        if t % 2:
            out.append(torch.einsum("bhk,bhkv->bhv", args["q"][:, t], S) / q.shape[-1] ** 0.5)
    expected = dense(q, k, v, g, beta, gate, h0)
    for a, b in zip((torch.stack(out, dim=1), S), expected):
        torch.testing.assert_close(a, b, atol=3e-12, rtol=3e-12)


def test_zero_recall_is_original_gdn():
    q, k, v, g, beta, gamma, state = inputs(gamma_value=0)
    actual = qgdn_reference(q, k, v, g, beta, gamma, initial_state=state)
    expected = naive_gdn2_recurrence(q, k, v, g[..., None].expand_as(q), beta[..., None].expand_as(q),
                                    beta[..., None].expand_as(v), initial_state=state)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)


def test_recall_readout_and_nonexpansion():
    q, _, _, g, _, gamma, state = inputs(T=1)
    q = F.normalize(q[:, 0], dim=-1)
    alpha, gamma = g[:, 0].exp(), gamma[:, 0]
    eye = torch.eye(q.shape[-1], dtype=q.dtype)
    D = alpha[..., None, None] * eye + (gamma * (1 - alpha))[..., None, None] * q[..., :, None] * q[..., None, :]
    recalled = D @ state
    read = lambda S: torch.einsum("bhk,bhkv->bhv", q, S)
    torch.testing.assert_close(read(recalled), (alpha + gamma * (1 - alpha))[..., None] * read(state))
    assert torch.linalg.matrix_norm(D, ord=2).max() <= 1
    # gamma=1 protects only Recall, not a subsequent edit at k=q.
    full_D = alpha[..., None, None] * eye + (1 - alpha)[..., None, None] * q[..., :, None] * q[..., None, :]
    torch.testing.assert_close(read(full_D @ state), read(state))
    erased = (eye - q[..., :, None] * q[..., None, :]) @ full_D @ state
    torch.testing.assert_close(read(erased), torch.zeros_like(read(erased)), atol=2e-12, rtol=0)


def test_state_carry_and_causality():
    q, k, v, g, beta, gamma, state = inputs(T=11)
    whole, final = qgdn_reference(q, k, v, g, beta, gamma, initial_state=state)
    first, middle = qgdn_reference(*(x[:, :4] for x in (q, k, v, g, beta, gamma)), initial_state=state)
    last, carried = qgdn_reference(*(x[:, 4:] for x in (q, k, v, g, beta, gamma)), initial_state=middle)
    torch.testing.assert_close(whole, torch.cat((first, last), dim=1))
    torch.testing.assert_close(final, carried)


def test_backbone_initialization_and_gate_gradient():
    models = []
    for mixer in ("gdn", "qgdn"):
        torch.manual_seed(3407)
        cfg = Config.from_name(f"{mixer}_recall_tiny", use_short_conv=False, _norm_class="RMSNorm")
        model = GPT(cfg)
        model.apply(lambda m: model._init_weights(m, n_layer=cfg.n_layer))
        for block in model.transformer.h:
            block.attn.mode = "naive"
        models.append(model)
    shared = dict(models[0].named_parameters())
    for name, param in models[1].named_parameters():
        if name in shared:
            torch.testing.assert_close(param, shared[name], rtol=0, atol=0)
    model = models[1]
    model.gradient_checkpointing = True
    logits = model(torch.randint(0, 256, (2, 13)))
    F.cross_entropy(logits.flatten(0, 1), torch.randint(0, 256, (26,))).backward()
    for block in model.transformer.h:
        grad = block.attn.recall_proj.weight.grad
        assert grad is not None and grad.isfinite().all() and grad.abs().sum() > 0
        assert torch.allclose(block.attn.recall_proj.bias.sigmoid(), torch.full((2,), 0.1))


@pytest.mark.parametrize("T", [17, 65, 257, 4096])
@pytest.mark.parametrize("recall_mode", ["query", "key", "isotropic"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity requires an allocated GPU")
def test_cuda_output_state_and_backward(T, recall_mode):
    xs = inputs(T=T, K=64, V=64, dtype=torch.float32, device="cuda", gamma_value=0.1)
    # Same rounded q/k/v as the actual BF16 kernel; FP32 log decay and gates.
    gpu = [x.detach().to(torch.bfloat16 if i < 3 else torch.float32).requires_grad_() for i, x in enumerate(xs)]
    ref = [x.detach().float().requires_grad_() for x in gpu]
    actual = qgdn_rule(*gpu[:6], initial_state=gpu[6], output_final_state=True, recall_mode=recall_mode)
    expected = qgdn_reference(*ref[:6], initial_state=ref[6], recall_mode=recall_mode)
    for a, b in zip(actual, expected):
        assert (a.float() - b).square().mean().sqrt() / b.square().mean().sqrt() < 0.025
    weights = [torch.randn_like(x).float() for x in expected]
    gradients = [torch.autograd.grad(sum((a.float() * w).sum() for a, w in zip(pair, weights)), inp)
                 for pair, inp in ((actual, gpu), (expected, ref))]
    for a, b in zip(*gradients):
        assert a.isfinite().all()
        assert (a.float() - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-6) < 0.07
