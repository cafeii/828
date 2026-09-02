import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(scalar, u, z, offset, initial_state):
    state = initial_state.float()
    starts = []
    for chunk in range(scalar.shape[1]):
        starts.append(state)
        projected = torch.einsum("bhkm,bhkv->bhmv", z[:, chunk].float(), state)
        state = scalar[:, chunk, :, None, None].float() * state
        state = state + torch.einsum("bhkm,bhmv->bhkv", u[:, chunk].float(), projected)
        state = state + offset[:, chunk].float()
    return torch.stack(starts, dim=1), state


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rank2_chunk_state_forward_matches_reference(dtype):
    from lit_gpt.rank2_chunk_state import rank2_chunk_state_fwd

    torch.manual_seed(123)
    B, N, H, K, M, V = 2, 5, 3, 16, 8, 11
    scalar = (0.94 + 0.04 * torch.rand(B, N, H, device="cuda")).float()
    u = (0.02 * torch.randn(B, N, H, K, M, device="cuda")).to(dtype)
    z = (0.02 * torch.randn(B, N, H, K, M, device="cuda")).to(dtype)
    offset = (0.02 * torch.randn(B, N, H, K, V, device="cuda")).to(dtype)
    initial = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32)

    expected_starts, expected_final = _reference(scalar, u, z, offset, initial)
    starts, final = rank2_chunk_state_fwd(
        scalar, u, z, offset, initial_state=initial, output_final_state=True
    )
    atol = 2e-2 if dtype == torch.bfloat16 else 2e-4
    rtol = 2e-2 if dtype == torch.bfloat16 else 2e-4
    torch.testing.assert_close(starts.float(), expected_starts, atol=atol, rtol=rtol)
    torch.testing.assert_close(final, expected_final, atol=atol, rtol=rtol)


def test_rank2_chunk_state_zero_initial_and_optional_final():
    from lit_gpt.rank2_chunk_state import rank2_chunk_state_fwd

    torch.manual_seed(7)
    B, N, H, K, M, V = 1, 3, 2, 8, 4, 9
    scalar = torch.full((B, N, H), 0.95, device="cuda")
    u = (0.01 * torch.randn(B, N, H, K, M, device="cuda")).bfloat16()
    z = (0.01 * torch.randn(B, N, H, K, M, device="cuda")).bfloat16()
    offset = torch.randn(B, N, H, K, V, device="cuda").bfloat16()
    initial = torch.zeros(B, H, K, V, device="cuda")
    expected_starts, _ = _reference(scalar, u, z, offset, initial)
    starts, final = rank2_chunk_state_fwd(scalar, u, z, offset, output_final_state=False)
    assert final is None
    torch.testing.assert_close(starts.float(), expected_starts, atol=2e-2, rtol=2e-2)
