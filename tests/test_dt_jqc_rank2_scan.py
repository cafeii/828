import pytest
import torch
import torch.nn.functional as F

from lit_gpt.mixers.dt_jqc_rule import (
    compose_affine,
    dense_affine_elements,
    dense_affine_scan_reference,
    dt_gdn_reference,
    jqc_gdn_reference,
    rank2_factor_reference,
    rank2_factors,
)


def inputs(requires_grad=False, T=7):
    torch.manual_seed(42)
    B, H, K, V = 2, 3, 5, 4
    tensors = [
        torch.randn(B, T, H, K, dtype=torch.float64),
        torch.randn(B, T, H, K, dtype=torch.float64),
        torch.randn(B, T, H, V, dtype=torch.float64),
        -0.8 * torch.rand(B, T, H, dtype=torch.float64),
        0.05 + 0.9 * torch.rand(B, T, H, dtype=torch.float64),
        0.05 + 0.9 * torch.rand(B, T, H, dtype=torch.float64),
        torch.randn(B, H, K, V, dtype=torch.float64),
    ]
    return tuple(x.requires_grad_(requires_grad) for x in tensors)


@pytest.mark.parametrize("method,direct", [("dt", dt_gdn_reference), ("jqc", jqc_gdn_reference)])
def test_rank2_factors_match_method_outputs_state_and_gradients(method, direct):
    args = inputs(requires_grad=True)
    expected = direct(*args[:-1], initial_state=args[-1])
    weights = [torch.randn_like(value) for value in expected]
    expected_grads = torch.autograd.grad(
        sum((value * weight).sum() for value, weight in zip(expected, weights)), args
    )
    cloned = tuple(x.detach().clone().requires_grad_() for x in args)
    actual = rank2_factor_reference(*cloned[:-1], method=method, initial_state=cloned[-1])
    actual_grads = torch.autograd.grad(
        sum((value * weight).sum() for value, weight in zip(actual, weights)), cloned
    )
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=2e-11, atol=2e-11)
    for value, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(value, reference, rtol=3e-10, atol=3e-10)


@pytest.mark.parametrize("method", ["dt", "jqc"])
def test_dense_affine_scan_recovers_rank2_recurrence_without_time_expansion(method):
    q, k, v, g, beta, gamma, state = inputs()
    qn, transition, offset = dense_affine_elements(q, k, v, g, beta, gamma, method=method)
    states = dense_affine_scan_reference(transition, offset, state)
    outputs = q.shape[-1] ** -0.5 * torch.einsum("bthk,bthkv->bthv", qn, states)
    expected_outputs, expected_state = rank2_factor_reference(
        q, k, v, g, beta, gamma, method=method, initial_state=state
    )
    torch.testing.assert_close(outputs, expected_outputs, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(states[:, -1], expected_state, rtol=2e-11, atol=2e-11)
    _, _, _, left, right, _ = rank2_factors(q, k, g, beta, gamma, method=method)
    assert left.shape[1] == q.shape[1] and right.shape[1] == q.shape[1]
    assert left.shape[-2] == 2 and right.shape[-2] == 2


def test_affine_composition_is_associative():
    torch.manual_seed(7)
    shape = (2, 3, 4, 4)
    transitions = [torch.randn(shape, dtype=torch.float64) for _ in range(3)]
    offsets = [torch.randn(2, 3, 4, 5, dtype=torch.float64) for _ in range(3)]
    ba = compose_affine(transitions[1], offsets[1], transitions[0], offsets[0])
    left = compose_affine(transitions[2], offsets[2], *ba)
    cb = compose_affine(transitions[2], offsets[2], transitions[1], offsets[1])
    right = compose_affine(*cb, transitions[0], offsets[0])
    for actual, expected in zip(left, right):
        torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)


def test_dt_near_collinear_system_stays_finite_and_exact():
    q, _, v, g, _, _, state = inputs(T=3)
    q = F.normalize(q, dim=-1)
    k = F.normalize(q + 1e-5 * torch.randn_like(q), dim=-1)
    beta = torch.full_like(g, 0.999)
    gamma = torch.full_like(g, 0.999)
    direct = dt_gdn_reference(q, k, v, g, beta, gamma, initial_state=state)
    factored = rank2_factor_reference(q, k, v, g, beta, gamma, method="dt", initial_state=state)
    assert all(value.isfinite().all() for value in direct + factored)
    for actual, expected in zip(factored, direct):
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)
