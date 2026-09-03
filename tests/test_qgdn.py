"""Mechanism tests against dense equations, not just the implementation itself."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from lit_gpt.config import Config
from lit_gpt.model import GPT
from lit_gpt.mixers.qgdn_reference import (
    qgdn_rank2_factors,
    qgdn_rank2_reference,
    qgdn_reference,
)
from lit_gpt.mixers.qgdn_rule import dplr_inputs, qgdn_rule
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


UPDATE_ORDERS = ("recall_then_delta", "delta_then_recall", "parallel")


def dense(q, k, v, g, beta, gamma, state, recall_mode="query", update_order="recall_then_delta"):
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    eye = torch.eye(q.shape[-1], dtype=q.dtype, device=q.device)
    out = []
    for t in range(q.shape[1]):
        r = q[:, t] if recall_mode == "query" else k[:, t]
        alpha = g[:, t].exp()[..., None, None]
        kt = k[:, t]
        if recall_mode == "isotropic":
            state = (alpha + gamma[:, t, :, None, None] * (1 - alpha)) * state
            error = v[:, t] - torch.einsum("bhk,bhkv->bhv", kt, state)
            state = state + beta[:, t, :, None, None] * kt[..., None] * error[..., None, :]
        else:
            old_read = torch.einsum("bhk,bhkv->bhv", r, state)
            decayed = alpha * state
            recall_error = old_read - torch.einsum("bhk,bhkv->bhv", r, decayed)
            delta_error = v[:, t] - torch.einsum("bhk,bhkv->bhv", kt, decayed)
            recall_update = gamma[:, t, :, None, None] * r[..., None] * recall_error[..., None, :]
            delta_update = beta[:, t, :, None, None] * kt[..., None] * delta_error[..., None, :]
            if update_order == "recall_then_delta":
                state = decayed + recall_update
                error = v[:, t] - torch.einsum("bhk,bhkv->bhv", kt, state)
                state = state + beta[:, t, :, None, None] * kt[..., None] * error[..., None, :]
            elif update_order == "delta_then_recall":
                state = decayed + delta_update
                error = old_read - torch.einsum("bhk,bhkv->bhv", r, state)
                state = state + gamma[:, t, :, None, None] * r[..., None] * error[..., None, :]
            else:
                state = decayed + recall_update + delta_update
        out.append((q[:, t].unsqueeze(-2) @ state).squeeze(-2) / q.shape[-1] ** 0.5)
    return torch.stack(out, dim=1), state


@pytest.mark.parametrize(
    "recall_mode,update_order",
    [(mode, order) for mode in ("query", "key") for order in UPDATE_ORDERS]
    + [("isotropic", "recall_then_delta")],
)
def test_dense_equation_outputs_and_all_gradients(recall_mode, update_order):
    xs = inputs()
    actual = qgdn_reference(
        *xs[:6], initial_state=xs[6], recall_mode=recall_mode, update_order=update_order
    )
    expected = dense(*xs, recall_mode=recall_mode, update_order=update_order)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
    weights = [torch.randn_like(x) for x in actual]
    gradients = [torch.autograd.grad(sum((a * w).sum() for a, w in zip(pair, weights)), xs, retain_graph=True)
                 for pair in (actual, expected)]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("gamma", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("update_order", UPDATE_ORDERS)
def test_virtual_dplr_order(gamma, update_order):
    q, k, v, g, beta, gate, h0 = inputs(gamma_value=gamma)
    args = dplr_inputs(q, k, v, g, beta, gate, update_order=update_order)
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
    expected = dense(q, k, v, g, beta, gate, h0, update_order=update_order)
    for a, b in zip((torch.stack(out, dim=1), S), expected):
        torch.testing.assert_close(a, b, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("update_order", UPDATE_ORDERS)
def test_zero_recall_is_original_gdn(update_order):
    q, k, v, g, beta, gamma, state = inputs(gamma_value=0)
    actual = qgdn_reference(
        q, k, v, g, beta, gamma, initial_state=state, update_order=update_order
    )
    expected = naive_gdn2_recurrence(q, k, v, g[..., None].expand_as(q), beta[..., None].expand_as(q),
                                    beta[..., None].expand_as(v), initial_state=state)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("recall_mode", ["query", "key"])
@pytest.mark.parametrize("update_order", UPDATE_ORDERS)
def test_physical_time_rank2_matches_qgdn_outputs_state_and_gradients(recall_mode, update_order):
    args = inputs(T=7)
    expected = qgdn_reference(
        *args[:6], initial_state=args[6], recall_mode=recall_mode, update_order=update_order
    )
    weights = [torch.randn_like(value) for value in expected]
    expected_grads = torch.autograd.grad(
        sum((value * weight).sum() for value, weight in zip(expected, weights)), args
    )

    cloned = [x.detach().clone().requires_grad_() for x in args]
    actual = qgdn_rank2_reference(
        *cloned[:6],
        initial_state=cloned[6],
        recall_mode=recall_mode,
        update_order=update_order,
    )
    actual_grads = torch.autograd.grad(
        sum((value * weight).sum() for value, weight in zip(actual, weights)), cloned
    )
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=2e-12, atol=2e-12)
    for value, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(value, reference, rtol=4e-11, atol=4e-11)

    _, _, left, right, _ = qgdn_rank2_factors(
        *cloned[:2], *cloned[3:6], recall_mode=recall_mode, update_order=update_order
    )
    assert left.shape[1] == args[0].shape[1]
    assert left.shape[-2] == 2 and right.shape == left.shape


@pytest.mark.parametrize("update_order", UPDATE_ORDERS)
def test_rank2_factors_remain_finite_at_collinear_extreme_gates(update_order):
    q, _, v, g, _, _, state = inputs(T=3)
    q = F.normalize(q, dim=-1)
    k = q.detach().clone().requires_grad_()
    g = torch.full_like(g, -20.0, requires_grad=True)
    beta = torch.full_like(g, 1 - 1e-8, requires_grad=True)
    gamma = torch.full_like(g, 1 - 1e-8, requires_grad=True)
    actual = qgdn_rank2_reference(
        q,
        k,
        v,
        g,
        beta,
        gamma,
        initial_state=state,
        update_order=update_order,
    )
    expected = dense(
        q,
        k,
        v,
        g,
        beta,
        gamma,
        state,
        update_order=update_order,
    )
    assert all(value.isfinite().all() for value in (*actual, *expected))
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=2e-9, atol=2e-9)


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


@pytest.mark.parametrize("update_order", UPDATE_ORDERS)
def test_state_carry_and_causality(update_order):
    q, k, v, g, beta, gamma, state = inputs(T=11)
    whole, final = qgdn_reference(
        q, k, v, g, beta, gamma, initial_state=state, update_order=update_order
    )
    first, middle = qgdn_reference(
        *(x[:, :4] for x in (q, k, v, g, beta, gamma)),
        initial_state=state,
        update_order=update_order,
    )
    last, carried = qgdn_reference(
        *(x[:, 4:] for x in (q, k, v, g, beta, gamma)),
        initial_state=middle,
        update_order=update_order,
    )
    torch.testing.assert_close(whole, torch.cat((first, last), dim=1))
    torch.testing.assert_close(final, carried)


@pytest.mark.parametrize("recall_mode", ["query", "key"])
@pytest.mark.parametrize("update_order", UPDATE_ORDERS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compile parity requires an allocated GPU")
def test_cuda_compiled_dplr_inputs_match_eager_outputs_and_gradients(recall_mode, update_order):
    xs = inputs(T=257, K=64, V=64, dtype=torch.float32, device="cuda")[:6]
    base = [x.detach().to(torch.bfloat16 if i < 3 else torch.float32) for i, x in enumerate(xs)]
    eager_inputs = [x.clone().requires_grad_() for x in base]
    eager = tuple(dplr_inputs(
        *eager_inputs, recall_mode=recall_mode, update_order=update_order, compiled=False
    ).values())
    weights = [torch.randn_like(value) for value in eager]
    eager_grads = torch.autograd.grad(
        sum((value * weight).float().mean() for value, weight in zip(eager, weights)),
        eager_inputs,
    )

    compiled_inputs = [x.clone().requires_grad_() for x in base]
    compiled = tuple(dplr_inputs(
        *compiled_inputs, recall_mode=recall_mode, update_order=update_order, compiled=True
    ).values())
    compiled_grads = torch.autograd.grad(
        sum((value * weight).float().mean() for value, weight in zip(compiled, weights)),
        compiled_inputs,
    )
    for actual, expected in zip(compiled, eager):
        relative_rmse = (
            (actual.float() - expected.float()).square().mean().sqrt()
            / expected.float().square().mean().sqrt().clamp_min(1e-7)
        )
        # Inductor may reassociate the BF16 factor builder.  The physical
        # recurrence parity test below is the semantic check; this threshold
        # only rejects material compilation drift.
        assert relative_rmse < 5e-5
    for actual, expected in zip(compiled_grads, eager_grads):
        assert actual.isfinite().all()
        relative_rmse = (
            (actual.float() - expected.float()).square().mean().sqrt()
            / expected.float().square().mean().sqrt().clamp_min(1e-7)
        )
        assert relative_rmse < 1e-4


@pytest.mark.parametrize("recall_weight_init,recall_init", [("zero", 0.1), ("beta", 0.5)])
def test_backbone_initialization_and_gate_gradient(recall_weight_init, recall_init):
    models = []
    for mixer in ("gdn", "qgdn"):
        torch.manual_seed(3407)
        cfg = Config.from_name(f"{mixer}_recall_tiny", use_short_conv=False, _norm_class="RMSNorm",
                               recall_weight_init=recall_weight_init, recall_init=recall_init)
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
        assert torch.allclose(block.attn.recall_proj.bias.sigmoid(), torch.full((2,), recall_init))
        if recall_weight_init == "beta":
            assert block.attn.recall_proj.weight.abs().sum() > 0
        assert not torch.equal(block.attn.recall_proj.weight, block.attn.b_proj.weight)


def test_update_orders_share_identical_initialization():
    models = []
    for name in (
        "qgdn_recall_tiny",
        "qgdn_delta_then_recall_tiny",
        "qgdn_parallel_tiny",
    ):
        torch.manual_seed(3407)
        cfg = Config.from_name(name, use_short_conv=False, _norm_class="RMSNorm")
        model = GPT(cfg)
        model.apply(lambda module: model._init_weights(module, n_layer=cfg.n_layer))
        models.append(model)
    expected = dict(models[0].named_parameters())
    for model in models[1:]:
        actual = dict(model.named_parameters())
        assert actual.keys() == expected.keys()
        for name, parameter in actual.items():
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("mixer", ["gdn", "qgdn"])
def test_gate_moments_are_complete_and_observation_is_noninvasive(mixer):
    torch.manual_seed(3407)
    cfg = Config.from_name(f"{mixer}_recall_tiny", use_short_conv=False, _norm_class="RMSNorm")
    model = GPT(cfg)
    model.apply(lambda module: model._init_weights(module, n_layer=cfg.n_layer))
    for block in model.transformer.h:
        block.attn.mode = "naive"
    tokens = torch.randint(0, cfg.padded_vocab_size, (2, 13))
    targets = torch.randint(0, cfg.padded_vocab_size, (2, 13))

    baseline = model(tokens)
    baseline_loss = F.cross_entropy(baseline.flatten(0, 1), targets.flatten())
    baseline_loss.backward()
    baseline_gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)

    for block in model.transformer.h:
        block.attn.reset_gate_stats()
        block.attn.collect_gate_stats = True
    observed = model(tokens)
    observed_loss = F.cross_entropy(observed.flatten(0, 1), targets.flatten())
    observed_loss.backward()
    torch.testing.assert_close(observed, baseline, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, baseline_gradients[name], rtol=0, atol=0)

    expected = {"alpha", "beta"} | ({"gamma", "gamma_saturated", "forgetting_margin"} if mixer == "qgdn" else set())
    first = []
    for block in model.transformer.h:
        moments = block.attn.gate_moments()
        assert set(moments) == expected
        assert all(value.dtype == torch.float64 and value.shape == (3,) for value in moments.values())
        assert all(value[2].item() > 0 for value in moments.values())
        first.append({name: value.clone() for name, value in moments.items()})

    model(tokens)
    for before, block in zip(first, model.transformer.h):
        for name, moments in block.attn.gate_moments().items():
            torch.testing.assert_close(moments, before[name] * 2, rtol=0, atol=0)


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


@pytest.mark.parametrize("update_order", UPDATE_ORDERS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity requires an allocated GPU")
def test_cuda_update_order_output_state_and_backward(update_order):
    xs = inputs(T=257, K=64, V=64, dtype=torch.float32, device="cuda", gamma_value=0.1)
    gpu = [
        x.detach().to(torch.bfloat16 if i < 3 else torch.float32).requires_grad_()
        for i, x in enumerate(xs)
    ]
    ref = [x.detach().float().requires_grad_() for x in gpu]
    actual = qgdn_rule(
        *gpu[:6],
        initial_state=gpu[6],
        output_final_state=True,
        update_order=update_order,
    )
    expected = qgdn_reference(
        *ref[:6], initial_state=ref[6], update_order=update_order
    )
    for value, reference in zip(actual, expected):
        relative_rmse = (
            (value.float() - reference).square().mean().sqrt()
            / reference.square().mean().sqrt().clamp_min(1e-6)
        )
        assert relative_rmse < 0.025
    weights = [torch.randn_like(value).float() for value in expected]
    actual_gradients = torch.autograd.grad(
        sum((value.float() * weight).sum() for value, weight in zip(actual, weights)),
        gpu,
    )
    expected_gradients = torch.autograd.grad(
        sum((value * weight).sum() for value, weight in zip(expected, weights)),
        ref,
    )
    for value, reference in zip(actual_gradients, expected_gradients):
        assert value.isfinite().all()
        relative_rmse = (
            (value.float() - reference).square().mean().sqrt()
            / reference.square().mean().sqrt().clamp_min(1e-6)
        )
        assert relative_rmse < 0.07


@pytest.mark.parametrize("update_order", UPDATE_ORDERS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="physical-T CUDA kernel requires a GPU")
def test_cuda_physical_t_output_state_and_backward(update_order):
    from lit_gpt.qgdn_physical import qgdn_physical, qgdn_physical_forward

    xs = inputs(T=64, K=64, V=64, dtype=torch.float32, device="cuda", gamma_value=0.1)
    gpu = [
        x.detach().to(torch.bfloat16 if i < 3 else torch.float32).requires_grad_()
        for i, x in enumerate(xs)
    ]
    qn = F.normalize(gpu[0].float(), dim=-1).to(torch.bfloat16)
    kn = F.normalize(gpu[1].float(), dim=-1).to(torch.bfloat16)
    output, _, final_state = qgdn_physical_forward(
        qn, kn, *gpu[2:6],
        update_order=update_order,
        initial_state=gpu[6],
        output_final_state=True,
    )
    reference_inputs = [value.detach().float().requires_grad_() for value in gpu]
    reference_output, reference_state = qgdn_reference(
        *reference_inputs[:6], initial_state=reference_inputs[6], update_order=update_order
    )
    for value, reference in ((output, reference_output), (final_state, reference_state)):
        relative_rmse = (
            (value.float() - reference).square().mean().sqrt()
            / reference.square().mean().sqrt().clamp_min(1e-6)
        )
        assert relative_rmse < 0.025

    differentiable_output = qgdn_physical(
        qn, kn, *gpu[2:6],
        update_order=update_order,
        initial_state=gpu[6],
    )
    weight = torch.randn_like(reference_output)
    actual_gradients = torch.autograd.grad(
        (differentiable_output.float() * weight).sum(), gpu
    )
    expected_gradients = torch.autograd.grad(
        (reference_output * weight).sum(), reference_inputs
    )
    for value, reference in zip(actual_gradients, expected_gradients):
        assert value.isfinite().all()
        relative_rmse = (
            (value.float() - reference).square().mean().sqrt()
            / reference.square().mean().sqrt().clamp_min(1e-6)
        )
        assert relative_rmse < 0.1
