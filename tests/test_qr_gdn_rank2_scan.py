import pytest
import torch

from lit_gpt.mixers.qr_gdn_rule import (
    apply_vector_decay_chunk,
    block_wy_rank2_vector_decay,
    qr_gdn_rank2_factors,
    qr_gdn_rank2_reference,
    qr_gdn_reference,
)


def inputs(requires_grad=False, T=8):
    torch.manual_seed(2026)
    B, H, K, V = 2, 3, 5, 4
    tensors = [
        torch.randn(B, T, H, K, dtype=torch.float64),
        torch.randn(B, T, H, K, dtype=torch.float64),
        torch.randn(B, T, H, V, dtype=torch.float64),
        -0.8 * torch.rand(B, T, H, dtype=torch.float64),
        0.05 + 0.9 * torch.rand(B, T, H, dtype=torch.float64),
        -0.8 * torch.rand(B, T, H, dtype=torch.float64),
        0.05 + 0.9 * torch.rand(B, T, H, dtype=torch.float64),
        torch.randn(B, T, H, dtype=torch.float64),
        torch.randn(B, H, K, V, dtype=torch.float64),
        torch.randn(B, H, K, V, dtype=torch.float64),
    ]
    return tuple(x.requires_grad_(requires_grad) for x in tensors)


def test_stacked_rank2_factors_match_explicit_outputs_states_and_gradients():
    args = inputs(requires_grad=True)
    data, state = args[:8], args[8:]
    expected = qr_gdn_reference(*data, initial_state=state)
    weights = [torch.randn_like(expected[0]), *(torch.randn_like(x) for x in expected[1])]
    expected_objective = (expected[0] * weights[0]).sum()
    expected_objective = expected_objective + sum(
        (x * w).sum() for x, w in zip(expected[1], weights[1:])
    )
    expected_grads = torch.autograd.grad(expected_objective, args)

    cloned = tuple(x.detach().clone().requires_grad_() for x in args)
    actual = qr_gdn_rank2_reference(*cloned[:8], initial_state=cloned[8:])
    actual_objective = (actual[0] * weights[0]).sum()
    actual_objective = actual_objective + sum(
        (x * w).sum() for x, w in zip(actual[1], weights[1:])
    )
    actual_grads = torch.autograd.grad(actual_objective, cloned)

    torch.testing.assert_close(actual[0], expected[0], rtol=3e-11, atol=3e-11)
    for value, reference in zip(actual[1], expected[1]):
        torch.testing.assert_close(value, reference, rtol=3e-11, atol=3e-11)
    for value, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(value, reference, rtol=6e-10, atol=6e-10)


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 8])
def test_vector_decay_chunks_match_final_state_without_time_expansion(chunk_size):
    q, k, v, g_kv, beta_kv, g_qr, beta_qr, read_logit, kv, qr = inputs()
    _, expected = qr_gdn_rank2_reference(
        q, k, v, g_kv, beta_kv, g_qr, beta_qr, read_logit, initial_state=(kv, qr)
    )
    _, _, decay, left, right, write = qr_gdn_rank2_factors(
        q, k, g_kv, beta_kv, g_qr, beta_qr
    )
    chunks = block_wy_rank2_vector_decay(
        decay, left, right, write, v, chunk_size=chunk_size
    )
    state = torch.cat((kv, qr), dim=-2)
    for index in range(chunks[0].shape[1]):
        state = apply_vector_decay_chunk(tuple(x[:, index] for x in chunks), state)
    torch.testing.assert_close(state[..., : q.shape[-1], :], expected[0], rtol=5e-11, atol=5e-11)
    torch.testing.assert_close(state[..., q.shape[-1] :, :], expected[1], rtol=5e-11, atol=5e-11)
    assert chunks[1].shape[-1] == 2 * chunk_size
    assert decay.shape[1] == q.shape[1]


def test_vector_decay_single_chunk_matches_final_state_gradients():
    args = inputs(requires_grad=True, T=4)
    q, k, v, g_kv, beta_kv, g_qr, beta_qr, _, kv, qr = args
    read_logit = torch.zeros_like(g_kv)
    _, expected_pair = qr_gdn_rank2_reference(
        q, k, v, g_kv, beta_kv, g_qr, beta_qr, read_logit, initial_state=(kv, qr)
    )
    expected = torch.cat(expected_pair, dim=-2)
    weight = torch.randn_like(expected)
    grad_inputs = (q, k, v, g_kv, beta_kv, g_qr, beta_qr, kv, qr)
    expected_grads = torch.autograd.grad((expected * weight).sum(), grad_inputs)

    cloned = tuple(x.detach().clone().requires_grad_() for x in grad_inputs)
    q, k, v, g_kv, beta_kv, g_qr, beta_qr, kv, qr = cloned
    _, _, decay, left, right, write = qr_gdn_rank2_factors(
        q, k, g_kv, beta_kv, g_qr, beta_qr
    )
    compact = block_wy_rank2_vector_decay(
        decay, left, right, write, v, chunk_size=q.shape[1]
    )
    actual = apply_vector_decay_chunk(tuple(x[:, 0] for x in compact), torch.cat((kv, qr), dim=-2))
    actual_grads = torch.autograd.grad((actual * weight).sum(), cloned)

    torch.testing.assert_close(actual, expected, rtol=5e-11, atol=5e-11)
    for value, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(value, reference, rtol=8e-10, atol=8e-10)
