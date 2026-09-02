"""Triton state propagation for compact rank-two chunk transforms."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["chunks"])
def _rank2_chunk_state_fwd_kernel(
    scalar,
    u,
    z,
    offset,
    starts,
    initial_state,
    final_state,
    chunks,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    STORE_FINAL: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_m = tl.arange(0, M)
    mask_state = (o_k[:, None] < K) & (o_v[None, :] < V)
    mask_factor = (o_k[:, None] < K) & (o_m[None, :] < M)

    state = tl.zeros((BK, BV), dtype=tl.float32)
    if HAS_INITIAL:
        p_initial = initial_state + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        state = tl.load(p_initial, mask=mask_state, other=0.0).to(tl.float32)

    for i_c in range(chunks):
        chunk_head = (i_b * chunks + i_c) * H + i_h
        p_starts = starts + chunk_head * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_starts, state.to(p_starts.dtype.element_ty), mask=mask_state)

        p_u = u + chunk_head * K * M + o_k[:, None] * M + o_m[None, :]
        p_z = z + chunk_head * K * M + o_k[:, None] * M + o_m[None, :]
        block_u = tl.load(p_u, mask=mask_factor, other=0.0)
        block_z = tl.load(p_z, mask=mask_factor, other=0.0)
        projected = tl.dot(tl.trans(block_z), state.to(block_z.dtype), input_precision="ieee")
        update = tl.dot(block_u, projected.to(block_u.dtype), input_precision="ieee")

        p_offset = offset + chunk_head * K * V + o_k[:, None] * V + o_v[None, :]
        block_offset = tl.load(p_offset, mask=mask_state, other=0.0).to(tl.float32)
        block_scalar = tl.load(scalar + chunk_head).to(tl.float32)
        state = block_scalar * state + update + block_offset

    if STORE_FINAL:
        p_final = final_state + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_final, state, mask=mask_state)


def rank2_chunk_state_fwd(
    scalar: torch.Tensor,
    u: torch.Tensor,
    z: torch.Tensor,
    offset: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Propagate compact transforms and return each physical chunk's input state."""
    if not scalar.is_cuda:
        raise ValueError("rank2_chunk_state_fwd requires CUDA tensors")
    if scalar.ndim != 3 or u.ndim != 5 or z.shape != u.shape or offset.ndim != 5:
        raise ValueError("expected scalar [B,N,H], U/Z [B,N,H,K,M], offset [B,N,H,K,V]")
    B, chunks, H = scalar.shape
    if u.shape[:3] != (B, chunks, H) or offset.shape[:3] != (B, chunks, H):
        raise ValueError("chunk transform batch/chunk/head dimensions must match")
    K, M = u.shape[-2:]
    if offset.shape[-2] != K:
        raise ValueError("factor and offset key dimensions must match")
    V = offset.shape[-1]
    if M not in (2, 4, 8, 16, 32, 64, 128):
        raise ValueError("compact rank must be a supported power of two")
    if K > 256:
        raise ValueError("key dimension above 256 is unsupported")
    expected_initial = (B, H, K, V)
    if initial_state is not None:
        if initial_state.shape != expected_initial or not initial_state.is_cuda:
            raise ValueError(f"initial_state must be a CUDA tensor of shape {expected_initial}")
        initial_state = initial_state.contiguous()

    scalar, u, z, offset = (x.contiguous() for x in (scalar, u, z, offset))
    starts = offset.new_empty((B, chunks, H, K, V))
    final_state = torch.empty(expected_initial, dtype=torch.float32, device=offset.device) if output_final_state else None
    BK = max(triton.next_power_of_2(K), 16)
    BV = 32
    grid = (triton.cdiv(V, BV), B * H)
    _rank2_chunk_state_fwd_kernel[grid](
        scalar,
        u,
        z,
        offset,
        starts,
        initial_state,
        final_state,
        chunks,
        H=H,
        K=K,
        V=V,
        M=M,
        BK=BK,
        BV=BV,
        HAS_INITIAL=initial_state is not None,
        STORE_FINAL=output_final_state,
        num_warps=4,
        num_stages=2,
    )
    return starts, final_state
