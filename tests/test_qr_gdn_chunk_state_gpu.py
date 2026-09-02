import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def compact_inputs(storage_dtype, *, initial=True):
    from lit_gpt.mixers.qr_gdn_rule import block_wy_rank2_vector_decay, qr_gdn_rank2_factors

    torch.manual_seed(3407)
    B, T, H, K, V, C = 2, 16, 2, 16, 12, 4
    q = torch.randn(B, T, H, K, device="cuda", dtype=storage_dtype)
    k = torch.randn_like(q)
    v = torch.randn(B, T, H, V, device="cuda", dtype=storage_dtype)
    g_kv = -0.2 * torch.rand(B, T, H, device="cuda")
    g_qr = -0.2 * torch.rand(B, T, H, device="cuda")
    beta_kv = 0.05 + 0.9 * torch.rand(B, T, H, device="cuda")
    beta_qr = 0.05 + 0.9 * torch.rand(B, T, H, device="cuda")
    _, _, decay, left, right, write = qr_gdn_rank2_factors(
        q, k, g_kv, beta_kv, g_qr, beta_qr
    )
    compact = block_wy_rank2_vector_decay(decay, left, right, write, v, chunk_size=C)
    compact = (compact[0], *(x.to(storage_dtype) for x in compact[1:]))
    state = None
    if initial:
        state = torch.randn(B, H, 2 * K, V, device="cuda", dtype=storage_dtype)
    return compact, state


def torch_propagate(compact, initial_state):
    from lit_gpt.mixers.qr_gdn_rule import apply_vector_decay_chunk

    B, chunks, H, D = compact[0].shape
    V = compact[3].shape[-1]
    state = torch.zeros(B, H, D, V, device="cuda", dtype=torch.float32)
    if initial_state is not None:
        state = initial_state.float()
    starts = []
    for index in range(chunks):
        starts.append(state.to(compact[3].dtype))
        state = apply_vector_decay_chunk(tuple(x[:, index] for x in compact), state)
    return torch.stack(starts, dim=1), state.float()


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.bfloat16])
def test_gpu_state_propagation_matches_torch(storage_dtype):
    from lit_gpt.qr_gdn_chunk_state import qr_gdn_chunk_state_fwd

    compact, initial = compact_inputs(storage_dtype)
    expected_starts, expected_final = torch_propagate(compact, initial)
    actual_starts, actual_final = qr_gdn_chunk_state_fwd(
        *compact, initial_state=initial, output_final_state=True
    )
    tolerance = 2e-5 if storage_dtype == torch.float32 else 3e-2
    torch.testing.assert_close(actual_starts.float(), expected_starts.float(), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual_final, expected_final, rtol=tolerance, atol=tolerance)


def test_gpu_zero_initial_and_optional_final_state():
    from lit_gpt.qr_gdn_chunk_state import qr_gdn_chunk_state_fwd

    compact, _ = compact_inputs(torch.float32, initial=False)
    expected_starts, _ = torch_propagate(compact, None)
    actual_starts, final = qr_gdn_chunk_state_fwd(*compact, output_final_state=False)
    assert final is None
    torch.testing.assert_close(actual_starts, expected_starts, rtol=2e-5, atol=2e-5)
