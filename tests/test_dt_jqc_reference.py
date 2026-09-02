import torch
import torch.nn.functional as F

from lit_gpt.mixers.dt_jqc_rule import (
    dt_gdn_affine_reference,
    dt_gdn_reference,
    jqc_gdn_affine_reference,
    jqc_gdn_reference,
)


def inputs(requires_grad=False):
    torch.manual_seed(3407)
    B, T, H, K, V = 2, 5, 3, 4, 6
    q = torch.randn(B, T, H, K, dtype=torch.float64, requires_grad=requires_grad)
    k = torch.randn(B, T, H, K, dtype=torch.float64, requires_grad=requires_grad)
    v = torch.randn(B, T, H, V, dtype=torch.float64, requires_grad=requires_grad)
    g = (-torch.rand(B, T, H, dtype=torch.float64)).requires_grad_(requires_grad)
    beta = (0.05 + 0.85 * torch.rand(B, T, H, dtype=torch.float64)).requires_grad_(requires_grad)
    gamma = (0.05 + 0.85 * torch.rand(B, T, H, dtype=torch.float64)).requires_grad_(requires_grad)
    state = torch.randn(B, H, K, V, dtype=torch.float64, requires_grad=requires_grad)
    return q, k, v, g, beta, gamma, state


def test_dt_closed_form_matches_direct_normal_equation():
    q, k, v, g, beta, gamma, state = inputs()
    qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    expected_outputs, current = [], state
    eye = torch.eye(q.shape[-1], dtype=q.dtype)
    for t in range(q.shape[1]):
        qt, kt, vt = qn[:, t], kn[:, t], v[:, t]
        alpha = g[:, t].exp()
        rb = beta[:, t] / (1 - beta[:, t])
        rg = gamma[:, t] / (1 - gamma[:, t])
        base = alpha[..., None, None] * current
        old_read = torch.einsum("bhk,bhkv->bhv", qt, current)
        lhs = eye + rb[..., None, None] * kt[..., :, None] * kt[..., None, :] + rg[..., None, None] * qt[..., :, None] * qt[..., None, :]
        rhs = base + rb[..., None, None] * kt[..., None] * vt[..., None, :] + rg[..., None, None] * qt[..., None] * old_read[..., None, :]
        current = torch.linalg.solve(lhs, rhs)
        expected_outputs.append(q.shape[-1] ** -0.5 * torch.einsum("bhk,bhkv->bhv", qt, current))
    actual_outputs, actual_state = dt_gdn_reference(q, k, v, g, beta, gamma, initial_state=state)
    torch.testing.assert_close(actual_outputs, torch.stack(expected_outputs, dim=1), rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(actual_state, current, rtol=1e-11, atol=1e-11)


def test_dt_affine_matches_closed_form_and_gradients():
    args = inputs(requires_grad=True)
    out_a, state_a = dt_gdn_reference(*args[:-1], initial_state=args[-1])
    grads_a = torch.autograd.grad(out_a.square().sum() + state_a.square().sum(), args)
    cloned = tuple(x.detach().clone().requires_grad_() for x in args)
    out_b, state_b = dt_gdn_affine_reference(*cloned[:-1], initial_state=cloned[-1])
    grads_b = torch.autograd.grad(out_b.square().sum() + state_b.square().sum(), cloned)
    torch.testing.assert_close(out_a, out_b, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(state_a, state_b, rtol=1e-11, atol=1e-11)
    for actual, expected in zip(grads_a, grads_b):
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)


def test_jqc_affine_matches_explicit_update_and_gradients():
    args = inputs(requires_grad=True)
    out_a, state_a = jqc_gdn_reference(*args[:-1], initial_state=args[-1])
    grads_a = torch.autograd.grad(out_a.square().sum() + state_a.square().sum(), args)
    cloned = tuple(x.detach().clone().requires_grad_() for x in args)
    out_b, state_b = jqc_gdn_affine_reference(*cloned[:-1], initial_state=cloned[-1])
    grads_b = torch.autograd.grad(out_b.square().sum() + state_b.square().sum(), cloned)
    torch.testing.assert_close(out_a, out_b, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(state_a, state_b, rtol=1e-11, atol=1e-11)
    for actual, expected in zip(grads_a, grads_b):
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)


def test_jqc_gamma_zero_is_native_gdn():
    q, k, v, g, beta, gamma, state = inputs()
    qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    current, outputs = state, []
    for t in range(q.shape[1]):
        decayed = g[:, t].exp()[..., None, None] * current
        error = v[:, t] - torch.einsum("bhk,bhkv->bhv", kn[:, t], decayed)
        current = decayed + beta[:, t, :, None, None] * kn[:, t, :, :, None] * error[..., None, :]
        outputs.append(q.shape[-1] ** -0.5 * torch.einsum("bhk,bhkv->bhv", qn[:, t], current))
    actual, actual_state = jqc_gdn_reference(q, k, v, g, beta, torch.zeros_like(gamma), initial_state=state)
    torch.testing.assert_close(actual, torch.stack(outputs, dim=1), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_state, current, rtol=1e-12, atol=1e-12)
