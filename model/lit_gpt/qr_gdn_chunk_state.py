"""Triton propagation of compact QR-GDN vector-decay chunk transforms."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["chunks"])
def _qr_gdn_chunk_state_fwd_kernel(
    decay,
    u,
    z,
    offset,
    starts,
    initial_state,
    final_state,
    chunks,
    H: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    BV: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    STORE_FINAL: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_d = tl.arange(0, BD)
    o_v = i_v * BV + tl.arange(0, BV)
    o_m = tl.arange(0, BM)
    mask_state = (o_d[:, None] < D) & (o_v[None, :] < V)
    mask_factor = (o_d[:, None] < D) & (o_m[None, :] < M)

    state = tl.zeros((BD, BV), dtype=tl.float32)
    if HAS_INITIAL:
        p_initial = initial_state + i_bh * D * V + o_d[:, None] * V + o_v[None, :]
        state = tl.load(p_initial, mask=mask_state, other=0.0).to(tl.float32)

    for i_c in range(chunks):
        chunk_head = (i_b * chunks + i_c) * H + i_h
        p_starts = starts + chunk_head * D * V + o_d[:, None] * V + o_v[None, :]
        tl.store(p_starts, state.to(p_starts.dtype.element_ty), mask=mask_state)

        p_u = u + chunk_head * D * M + o_d[:, None] * M + o_m[None, :]
        p_z = z + chunk_head * D * M + o_d[:, None] * M + o_m[None, :]
        block_u = tl.load(p_u, mask=mask_factor, other=0.0)
        block_z = tl.load(p_z, mask=mask_factor, other=0.0)
        projected = tl.dot(tl.trans(block_z), state.to(block_z.dtype), input_precision="ieee")
        update = tl.dot(block_u, projected.to(block_u.dtype), input_precision="ieee")

        p_offset = offset + chunk_head * D * V + o_d[:, None] * V + o_v[None, :]
        block_offset = tl.load(p_offset, mask=mask_state, other=0.0).to(tl.float32)
        row_decay = tl.load(decay + chunk_head * D + o_d, mask=o_d < D, other=0.0).to(tl.float32)
        state = row_decay[:, None] * state + update + block_offset

    if STORE_FINAL:
        p_final = final_state + i_bh * D * V + o_d[:, None] * V + o_v[None, :]
        tl.store(p_final, state, mask=mask_state)


def qr_gdn_chunk_state_fwd(
    decay: torch.Tensor,
    u: torch.Tensor,
    z: torch.Tensor,
    offset: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return each physical chunk's input state and the optional final state."""
    if not decay.is_cuda:
        raise ValueError("qr_gdn_chunk_state_fwd requires CUDA tensors")
    if decay.ndim != 4 or u.ndim != 5 or z.shape != u.shape or offset.ndim != 5:
        raise ValueError("expected decay [B,N,H,D], U/Z [B,N,H,D,M], offset [B,N,H,D,V]")
    B, chunks, H, D = decay.shape
    if u.shape[:4] != (B, chunks, H, D) or offset.shape[:4] != (B, chunks, H, D):
        raise ValueError("chunk transform batch/chunk/head/state dimensions must match")
    M, V = u.shape[-1], offset.shape[-1]
    if M not in (2, 4, 8, 16, 32, 64, 128):
        raise ValueError("compact rank must be a supported power of two")
    if D > 256:
        raise ValueError("stacked state dimension above 256 is unsupported")
    expected_initial = (B, H, D, V)
    if initial_state is not None:
        if initial_state.shape != expected_initial or not initial_state.is_cuda:
            raise ValueError(f"initial_state must be a CUDA tensor of shape {expected_initial}")
        initial_state = initial_state.contiguous()

    decay, u, z, offset = (x.contiguous() for x in (decay, u, z, offset))
    starts = offset.new_empty((B, chunks, H, D, V))
    final_state = (
        torch.empty(expected_initial, dtype=torch.float32, device=offset.device)
        if output_final_state
        else None
    )
    BD = max(triton.next_power_of_2(D), 16)
    BM = max(triton.next_power_of_2(M), 16)
    BV = 32
    grid = (triton.cdiv(V, BV), B * H)
    _qr_gdn_chunk_state_fwd_kernel[grid](
        decay,
        u,
        z,
        offset,
        starts,
        initial_state,
        final_state,
        chunks,
        H=H,
        D=D,
        V=V,
        M=M,
        BM=BM,
        BD=BD,
        BV=BV,
        HAS_INITIAL=initial_state is not None,
        STORE_FINAL=output_final_state,
        num_warps=4,
        num_stages=2,
    )
    return starts, final_state
