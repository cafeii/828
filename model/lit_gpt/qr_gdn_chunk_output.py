"""Triton chunk-local outputs for the coupled QR-GDN recurrence."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qr_gdn_chunk_output_fwd_kernel(
    q,
    log_decay,
    left,
    right,
    write,
    value,
    read_gate,
    chunk_starts,
    output,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_cbh = tl.program_id(1).to(tl.int64)
    chunks = T // C
    i_h = i_cbh % H
    i_bc = i_cbh // H
    i_c = i_bc % chunks
    i_b = i_bc // chunks

    o_d = tl.arange(0, BD)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_state = (o_d[:, None] < D) & (o_v[None, :] < V)
    chunk_head = (i_bc * H + i_h).to(tl.int64)
    p_start = chunk_starts + chunk_head * D * V + o_d[:, None] * V + o_v[None, :]
    state = tl.load(p_start, mask=mask_state, other=0.0).to(tl.float32)

    for i_t in range(C):
        t = i_c * C + i_t
        token_head = ((i_b * T + t) * H + i_h).to(tl.int64)

        # q reads both halves, but QR reads the pre-update state only.
        q_index_qr = o_d - K
        q_qr = tl.load(
            q + token_head * K + q_index_qr,
            mask=(o_d >= K) & (o_d < D),
            other=0.0,
        ).to(tl.float32)
        qr_read = tl.sum(q_qr[:, None] * state, axis=0)

        p_right = right + token_head * 2 * D + o_d
        right_kv = tl.load(p_right, mask=o_d < D, other=0.0).to(tl.float32)
        right_qr = tl.load(p_right + D, mask=o_d < D, other=0.0).to(tl.float32)
        rank_read_kv = tl.sum(right_kv[:, None] * state, axis=0)
        rank_read_qr = tl.sum(right_qr[:, None] * state, axis=0)

        channel = o_d // K
        g = tl.load(log_decay + token_head * 2 + channel, mask=o_d < D, other=0.0)
        state *= tl.exp(g.to(tl.float32))[:, None]

        p_left = left + token_head * 2 * D + o_d
        left_kv = tl.load(p_left, mask=o_d < D, other=0.0).to(tl.float32)
        left_qr = tl.load(p_left + D, mask=o_d < D, other=0.0).to(tl.float32)
        state += left_kv[:, None] * rank_read_kv[None, :]
        state += left_qr[:, None] * rank_read_qr[None, :]

        write_row = tl.load(write + token_head * D + o_d, mask=o_d < D, other=0.0).to(tl.float32)
        value_row = tl.load(value + token_head * V + o_v, mask=o_v < V, other=0.0).to(tl.float32)
        state += write_row[:, None] * value_row[None, :]

        q_kv = tl.load(q + token_head * K + o_d, mask=o_d < K, other=0.0).to(tl.float32)
        kv_read = tl.sum(q_kv[:, None] * state, axis=0)
        gate = tl.load(read_gate + token_head).to(tl.float32)
        result = (kv_read + gate * qr_read) * SCALE
        tl.store(output + token_head * V + o_v, result, mask=o_v < V)


def qr_gdn_chunk_output_fwd(
    q: torch.Tensor,
    log_decay: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    write: torch.Tensor,
    value: torch.Tensor,
    read_gate: torch.Tensor,
    chunk_starts: torch.Tensor,
    *,
    chunk_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute all token outputs from independently supplied chunk-start states.

    The kernel is parallel across physical chunks and value tiles. Its only
    sequential loop is the fixed-size recurrence inside one chunk.
    """
    if not q.is_cuda:
        raise ValueError("qr_gdn_chunk_output_fwd requires CUDA tensors")
    if q.ndim != 4:
        raise ValueError("q must have shape [B,T,H,K]")
    B, T, H, K = q.shape
    D = 2 * K
    expected_token = (B, T, H)
    if log_decay.shape != expected_token + (2,):
        raise ValueError("log_decay must have shape [B,T,H,2]")
    if left.shape != expected_token + (2, D) or right.shape != left.shape:
        raise ValueError("left/right must have shape [B,T,H,2,2K]")
    if write.shape != expected_token + (D,):
        raise ValueError("write must have shape [B,T,H,2K]")
    if value.shape[:3] != expected_token or read_gate.shape != expected_token:
        raise ValueError("value/read_gate token dimensions must match q")
    if chunk_size <= 0 or T % chunk_size:
        raise ValueError("sequence length must be divisible by a positive chunk_size")
    chunks, V = T // chunk_size, value.shape[-1]
    if chunk_starts.shape != (B, chunks, H, D, V):
        raise ValueError("chunk_starts must have shape [B,T/C,H,2K,V]")
    if D > 256:
        raise ValueError("stacked state dimension above 256 is unsupported")
    if chunk_size > 64:
        raise ValueError("chunk_size above 64 is unsupported")
    tensors = (q, log_decay, left, right, write, value, read_gate, chunk_starts)
    if any(not x.is_cuda or x.device != q.device for x in tensors):
        raise ValueError("all inputs must be CUDA tensors on one device")

    q, log_decay, left, right, write, value, read_gate, chunk_starts = (
        x.contiguous() for x in tensors
    )
    output = value.new_empty((B, T, H, V))
    BD = max(triton.next_power_of_2(D), 16)
    BV = 32
    grid = (triton.cdiv(V, BV), B * chunks * H)
    _qr_gdn_chunk_output_fwd_kernel[grid](
        q,
        log_decay,
        left,
        right,
        write,
        value,
        read_gate,
        chunk_starts,
        output,
        T=T,
        H=H,
        K=K,
        V=V,
        C=chunk_size,
        D=D,
        BD=BD,
        BV=BV,
        SCALE=K**-0.5 if scale is None else scale,
        num_warps=4,
        num_stages=2,
    )
    return output
