import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def inputs(dtype, *, initial=True):
    from lit_gpt.mixers.qr_gdn_rule import qr_gdn_rank2_factors

    torch.manual_seed(42)
    B, T, H, K, V, C = 2, 16, 2, 16, 12, 4
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
    g_kv = -0.4 * torch.rand(B, T, H, device="cuda")
    g_qr = -0.4 * torch.rand(B, T, H, device="cuda")
    beta_kv = 0.05 + 0.9 * torch.rand(B, T, H, device="cuda")
    beta_qr = 0.05 + 0.9 * torch.rand(B, T, H, device="cuda")
    qn, _, log_decay, left, right, write = qr_gdn_rank2_factors(
        q, k, g_kv, beta_kv, g_qr, beta_qr
    )
    gate = torch.tanh(torch.randn(B, T, H, device="cuda"))
    state = None
    if initial:
        state = torch.randn(B, H, 2 * K, V, device="cuda", dtype=dtype)
    factors = tuple(x.to(dtype) for x in (qn, log_decay, left, right, write, v, gate))
    return factors, state, C


def torch_reference(factors, initial_state):
    q, log_decay, left, right, write, value, read_gate = (x.float() for x in factors)
    B, T, H, K = q.shape
    V = value.shape[-1]
    state = torch.zeros(B, H, 2 * K, V, device="cuda")
    if initial_state is not None:
        state = initial_state.float()
    outputs = []
    for t in range(T):
        old_qr = state[..., K:, :]
        qr_read = torch.einsum("bhk,bhkv->bhv", q[:, t], old_qr)
        rank_read = torch.einsum("bhrd,bhdv->bhrv", right[:, t], state)
        decay = log_decay[:, t].exp().repeat_interleave(K, dim=-1)
        state = decay[..., None] * state
        state = state + torch.einsum("bhrd,bhrv->bhdv", left[:, t], rank_read)
        state = state + write[:, t, :, :, None] * value[:, t, :, None, :]
        kv_read = torch.einsum("bhk,bhkv->bhv", q[:, t], state[..., :K, :])
        outputs.append((kv_read + read_gate[:, t, :, None] * qr_read) * K**-0.5)
    return torch.stack(outputs, dim=1)


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("initial", [False, True])
def test_chunk_output_matches_token_oracle(storage_dtype, initial):
    from lit_gpt.mixers.qr_gdn_rule import block_wy_rank2_vector_decay
    from lit_gpt.qr_gdn_chunk_output import qr_gdn_chunk_output_fwd
    from lit_gpt.qr_gdn_chunk_state import qr_gdn_chunk_state_fwd

    factors, initial_state, chunk_size = inputs(storage_dtype, initial=initial)
    q, log_decay, left, right, write, value, read_gate = factors
    compact = block_wy_rank2_vector_decay(
        log_decay.float(), left.float(), right.float(), write.float(), value.float(), chunk_size=chunk_size
    )
    compact = tuple(x.to(storage_dtype) for x in compact)
    starts, _ = qr_gdn_chunk_state_fwd(*compact, initial_state=initial_state, output_final_state=False)
    actual = qr_gdn_chunk_output_fwd(
        q, log_decay, left, right, write, value, read_gate, starts, chunk_size=chunk_size
    )
    expected = torch_reference(factors, initial_state)
    tolerance = 3e-5 if storage_dtype == torch.float32 else 5e-2
    torch.testing.assert_close(actual.float(), expected, rtol=tolerance, atol=tolerance)
