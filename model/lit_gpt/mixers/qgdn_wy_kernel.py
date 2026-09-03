"""Forward-only Triton primitive for physical-T rank-two WY preparation."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qgdn_streaming_wy_fwd_kernel(
    normalized_left,
    right,
    normalized_write,
    values,
    effective_right,
    write_reads,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    lane = tl.program_id(0).to(tl.int64)
    rank: tl.constexpr = 2
    rows: tl.constexpr = BT * rank
    o_m = tl.arange(0, BM)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    o_t = tl.arange(0, BT)
    m_rows = o_m < rows

    factor_base = lane * rows * K
    p_right = right + factor_base + o_m[:, None] * K + o_k[None, :]
    p_left = normalized_left + factor_base + o_m[:, None] * K + o_k[None, :]
    b_right = tl.load(
        p_right, mask=m_rows[:, None] & (o_k[None, :] < K), other=0.0
    ).to(tl.float32)
    b_left = tl.load(
        p_left, mask=m_rows[:, None] & (o_k[None, :] < K), other=0.0
    ).to(tl.float32)

    # Build (I-C)^-1 entirely inside the program.  Same-token rank pairs are
    # simultaneous, so only strictly earlier physical tokens are causal.
    b_closure = tl.dot(b_right, b_left.T, input_precision="ieee")
    token = o_m // rank
    causal = (token[:, None] > token[None, :]) & (
        m_rows[:, None] & m_rows[None, :]
    )
    b_closure = tl.where(causal, b_closure, 0.0)
    for row in range(1, rows):
        row_mask = o_m == row
        b_row = tl.sum(tl.where(row_mask[:, None], b_closure, 0.0), axis=0)
        b_row = b_row + tl.sum(b_row[:, None] * b_closure, axis=0) * (
            o_m < row
        )
        b_closure = tl.where(row_mask[:, None], b_row, b_closure)
    b_inverse = b_closure + (o_m[:, None] == o_m[None, :])

    b_effective = tl.dot(b_inverse, b_right, input_precision="ieee")
    p_effective = (
        effective_right + factor_base + o_m[:, None] * K + o_k[None, :]
    )
    tl.store(
        p_effective,
        b_effective,
        mask=m_rows[:, None] & (o_k[None, :] < K),
    )

    write_base = lane * BT * K
    p_write = (
        normalized_write + write_base + o_t[:, None] * K + o_k[None, :]
    )
    b_write = tl.load(
        p_write, mask=o_k[None, :] < K, other=0.0
    ).to(tl.float32)
    b_write_coupling = tl.dot(b_right, b_write.T, input_precision="ieee")
    b_write_coupling = tl.where(
        (token[:, None] > o_t[None, :]) & m_rows[:, None],
        b_write_coupling,
        0.0,
    )

    value_base = lane * BT * V
    p_values = values + value_base + o_t[:, None] * V + o_v[None, :]
    b_values = tl.load(
        p_values, mask=o_v[None, :] < V, other=0.0
    ).to(tl.float32)
    b_write_rhs = tl.dot(
        b_write_coupling, b_values, input_precision="ieee"
    )
    b_write_reads = tl.dot(
        b_inverse, b_write_rhs, input_precision="ieee"
    )
    output_base = lane * rows * V
    p_write_reads = (
        write_reads + output_base + o_m[:, None] * V + o_v[None, :]
    )
    tl.store(
        p_write_reads,
        b_write_reads,
        mask=m_rows[:, None] & (o_v[None, :] < V),
    )


def qgdn_streaming_wy_fwd(
    normalized_left: torch.Tensor,
    right: torch.Tensor,
    normalized_write: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return effective-right factors and zero-state write reads.

    Inputs use ``[B,H,N,C,2,K]``, ``[B,H,N,C,K]`` and
    ``[B,H,N,C,V]`` layouts.  This primitive is intentionally forward-only;
    it must not be connected to training until a verified backward exists.
    """
    if not all(x.is_cuda for x in (normalized_left, right, normalized_write, values)):
        raise ValueError("the Triton WY primitive requires CUDA tensors")
    if normalized_left.shape != right.shape or normalized_left.ndim != 6:
        raise ValueError("left and right must share [B,H,N,C,2,K] shape")
    batch, heads, chunks, chunk_size, rank, key_dim = right.shape
    if rank != 2:
        raise ValueError("the QGDN WY primitive requires rank two")
    if normalized_write.shape != (batch, heads, chunks, chunk_size, key_dim):
        raise ValueError("normalized_write has incompatible shape")
    if values.shape[:4] != (batch, heads, chunks, chunk_size):
        raise ValueError("values have incompatible shape")
    if chunk_size not in {8, 16}:
        raise ValueError("the diagnostic kernel supports chunk sizes 8 and 16")
    if any(x.dtype != torch.float32 for x in (normalized_left, right, normalized_write, values)):
        raise ValueError("the diagnostic kernel currently supports float32 only")

    normalized_left = normalized_left.contiguous()
    right = right.contiguous()
    normalized_write = normalized_write.contiguous()
    values = values.contiguous()
    value_dim = values.shape[-1]
    effective_right = torch.empty_like(right)
    write_reads = values.new_empty(
        batch, heads, chunks, chunk_size, rank, value_dim
    )
    block_rows = triton.next_power_of_2(chunk_size * rank)
    block_key = max(16, triton.next_power_of_2(key_dim))
    block_value = max(16, triton.next_power_of_2(value_dim))
    if block_key > 128 or block_value > 128:
        raise ValueError("the diagnostic kernel supports key/value dims up to 128")
    _qgdn_streaming_wy_fwd_kernel[(batch * heads * chunks,)](
        normalized_left,
        right,
        normalized_write,
        values,
        effective_right,
        write_reads,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BM=block_rows,
        BK=block_key,
        BV=block_value,
        num_warps=8,
        num_stages=2,
    )
    return effective_right, write_reads
