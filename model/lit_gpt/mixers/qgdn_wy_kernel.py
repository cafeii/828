"""Fused Triton forward/backward for physical-T rank-two WY preparation."""
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
    BWT: tl.constexpr,
):
    lane = tl.program_id(0).to(tl.int64)
    rank: tl.constexpr = 2
    rows: tl.constexpr = BT * rank
    o_m = tl.arange(0, BM)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    o_t = tl.arange(0, BWT)
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
        p_write,
        mask=(o_t[:, None] < BT) & (o_k[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    b_write_coupling = tl.dot(b_right, b_write.T, input_precision="ieee")
    b_write_coupling = tl.where(
        (token[:, None] > o_t[None, :])
        & m_rows[:, None]
        & (o_t[None, :] < BT),
        b_write_coupling,
        0.0,
    )

    value_base = lane * BT * V
    p_values = values + value_base + o_t[:, None] * V + o_v[None, :]
    b_values = tl.load(
        p_values,
        mask=(o_t[:, None] < BT) & (o_v[None, :] < V),
        other=0.0,
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


@triton.jit
def _qgdn_streaming_wy_bwd_kernel(
    normalized_left,
    right,
    normalized_write,
    values,
    grad_effective,
    grad_write_reads,
    grad_left,
    grad_right,
    grad_normalized_write,
    grad_values,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BWT: tl.constexpr,
):
    lane = tl.program_id(0).to(tl.int64)
    rank: tl.constexpr = 2
    rows: tl.constexpr = BT * rank
    o_m = tl.arange(0, BM)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    o_t = tl.arange(0, BWT)
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

    write_base = lane * BT * K
    p_write = (
        normalized_write + write_base + o_t[:, None] * K + o_k[None, :]
    )
    b_write = tl.load(
        p_write,
        mask=(o_t[:, None] < BT) & (o_k[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    write_causal = (
        (token[:, None] > o_t[None, :])
        & m_rows[:, None]
        & (o_t[None, :] < BT)
    )
    b_write_coupling = tl.dot(b_right, b_write.T, input_precision="ieee")
    b_write_coupling = tl.where(write_causal, b_write_coupling, 0.0)

    value_base = lane * BT * V
    p_values = values + value_base + o_t[:, None] * V + o_v[None, :]
    b_values = tl.load(
        p_values,
        mask=(o_t[:, None] < BT) & (o_v[None, :] < V),
        other=0.0,
    ).to(tl.float32)
    b_write_rhs = tl.dot(
        b_write_coupling, b_values, input_precision="ieee"
    )
    b_write_reads = tl.dot(
        b_inverse, b_write_rhs, input_precision="ieee"
    )

    p_grad_effective = (
        grad_effective + factor_base + o_m[:, None] * K + o_k[None, :]
    )
    b_grad_effective = tl.load(
        p_grad_effective,
        mask=m_rows[:, None] & (o_k[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    grad_write_base = lane * rows * V
    p_grad_write_reads = (
        grad_write_reads
        + grad_write_base
        + o_m[:, None] * V
        + o_v[None, :]
    )
    b_grad_write_reads = tl.load(
        p_grad_write_reads,
        mask=m_rows[:, None] & (o_v[None, :] < V),
        other=0.0,
    ).to(tl.float32)

    # For Y=A^-1 B, dB=A^-T dY and dC=(A^-T dY)Y^T because A=I-C.
    b_adjoint_effective = tl.dot(
        b_inverse.T, b_grad_effective, input_precision="ieee"
    )
    b_adjoint_write = tl.dot(
        b_inverse.T, b_grad_write_reads, input_precision="ieee"
    )
    b_grad_coupling = tl.dot(
        b_adjoint_effective, b_effective.T, input_precision="ieee"
    ) + tl.dot(
        b_adjoint_write, b_write_reads.T, input_precision="ieee"
    )
    b_grad_coupling = tl.where(causal, b_grad_coupling, 0.0)
    b_grad_write_coupling = tl.dot(
        b_adjoint_write, b_values.T, input_precision="ieee"
    )
    b_grad_write_coupling = tl.where(
        write_causal, b_grad_write_coupling, 0.0
    )

    b_grad_right = (
        b_adjoint_effective
        + tl.dot(b_grad_coupling, b_left, input_precision="ieee")
        + tl.dot(b_grad_write_coupling, b_write, input_precision="ieee")
    )
    b_grad_left = tl.dot(
        b_grad_coupling.T, b_right, input_precision="ieee"
    )
    b_grad_write = tl.dot(
        b_grad_write_coupling.T, b_right, input_precision="ieee"
    )
    b_grad_values = tl.dot(
        b_write_coupling.T, b_adjoint_write, input_precision="ieee"
    )

    p_grad_right = grad_right + factor_base + o_m[:, None] * K + o_k[None, :]
    p_grad_left = grad_left + factor_base + o_m[:, None] * K + o_k[None, :]
    tl.store(
        p_grad_right,
        b_grad_right,
        mask=m_rows[:, None] & (o_k[None, :] < K),
    )
    tl.store(
        p_grad_left,
        b_grad_left,
        mask=m_rows[:, None] & (o_k[None, :] < K),
    )
    p_grad_write = (
        grad_normalized_write
        + write_base
        + o_t[:, None] * K
        + o_k[None, :]
    )
    tl.store(
        p_grad_write,
        b_grad_write,
        mask=(o_t[:, None] < BT) & (o_k[None, :] < K),
    )
    p_grad_values = (
        grad_values + value_base + o_t[:, None] * V + o_v[None, :]
    )
    tl.store(
        p_grad_values,
        b_grad_values,
        mask=(o_t[:, None] < BT) & (o_v[None, :] < V),
    )


def _qgdn_streaming_wy_torch_fwd(
    normalized_left: torch.Tensor,
    right: torch.Tensor,
    normalized_write: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute the streaming algebra without a dense coupling system."""
    chunk_size = right.shape[3]
    effective_parts = []
    write_read_parts = []
    write_state = right.new_zeros(*right.shape[:3], right.shape[-1], values.shape[-1])
    for token in range(chunk_size):
        right_token = right[:, :, :, token]
        effective_token = right_token
        for previous, effective_previous in enumerate(effective_parts):
            coupling = torch.einsum(
                "bhnrk,bhnuk->bhnru",
                right_token,
                normalized_left[:, :, :, previous],
            )
            effective_token = effective_token + torch.einsum(
                "bhnru,bhnuk->bhnrk", coupling, effective_previous
            )
        write_read = torch.einsum(
            "bhnrk,bhnkv->bhnrv", right_token, write_state
        )
        write_state = (
            write_state
            + torch.einsum(
                "bhnrk,bhnrv->bhnkv",
                normalized_left[:, :, :, token],
                write_read,
            )
            + normalized_write[:, :, :, token, :, None]
            * values[:, :, :, token, None, :]
        )
        effective_parts.append(effective_token)
        write_read_parts.append(write_read)
    return torch.stack(effective_parts, dim=3), torch.stack(write_read_parts, dim=3)


def _qgdn_streaming_wy_recompute_bwd(
    normalized_left: torch.Tensor,
    right: torch.Tensor,
    normalized_write: torch.Tensor,
    values: torch.Tensor,
    grad_effective: torch.Tensor,
    grad_write_reads: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Manually differentiate WY preparation after recomputing its history."""
    chunk_size = right.shape[3]
    effective_parts, write_read_parts = _qgdn_streaming_wy_torch_fwd(
        normalized_left, right, normalized_write, values
    )
    effective_parts = list(effective_parts.unbind(dim=3))
    write_read_parts = list(write_read_parts.unbind(dim=3))
    grad_effective_parts = [value.clone() for value in grad_effective.unbind(dim=3)]
    grad_left = torch.zeros_like(normalized_left)
    grad_right = torch.zeros_like(right)
    grad_normalized_write = torch.zeros_like(normalized_write)
    grad_values = torch.zeros_like(values)

    # E_t = R_t + sum_{s<t} (R_t L_s^T) E_s.
    for token in range(chunk_size - 1, -1, -1):
        right_token = right[:, :, :, token]
        grad_effective_token = grad_effective_parts[token]
        grad_right_token = grad_effective_token.clone()
        for previous in range(token):
            left_previous = normalized_left[:, :, :, previous]
            effective_previous = effective_parts[previous]
            coupling = torch.einsum(
                "bhnrk,bhnuk->bhnru", right_token, left_previous
            )
            grad_coupling = torch.einsum(
                "bhnrk,bhnuk->bhnru",
                grad_effective_token,
                effective_previous,
            )
            grad_effective_parts[previous] = (
                grad_effective_parts[previous]
                + torch.einsum(
                    "bhnru,bhnrk->bhnuk", coupling, grad_effective_token
                )
            )
            grad_right_token = grad_right_token + torch.einsum(
                "bhnru,bhnuk->bhnrk", grad_coupling, left_previous
            )
            grad_left[:, :, :, previous] += torch.einsum(
                "bhnru,bhnrk->bhnuk", grad_coupling, right_token
            )
        grad_right[:, :, :, token] = grad_right_token

    # Recompute only the C pre-token write states; nothing was saved by forward.
    write_states = []
    write_state = right.new_zeros(*right.shape[:3], right.shape[-1], values.shape[-1])
    for token in range(chunk_size):
        write_states.append(write_state)
        write_state = (
            write_state
            + torch.einsum(
                "bhnrk,bhnrv->bhnkv",
                normalized_left[:, :, :, token],
                write_read_parts[token],
            )
            + normalized_write[:, :, :, token, :, None]
            * values[:, :, :, token, None, :]
        )

    grad_write_state = torch.zeros_like(write_state)
    for token in range(chunk_size - 1, -1, -1):
        left_token = normalized_left[:, :, :, token]
        right_token = right[:, :, :, token]
        write_read = write_read_parts[token]
        write_state_before = write_states[token]
        grad_left[:, :, :, token] += torch.einsum(
            "bhnrv,bhnkv->bhnrk", write_read, grad_write_state
        )
        grad_write_read = grad_write_reads[:, :, :, token] + torch.einsum(
            "bhnrk,bhnkv->bhnrv", left_token, grad_write_state
        )
        grad_normalized_write[:, :, :, token] = torch.einsum(
            "bhnkv,bhnv->bhnk", grad_write_state, values[:, :, :, token]
        )
        grad_values[:, :, :, token] = torch.einsum(
            "bhnk,bhnkv->bhnv",
            normalized_write[:, :, :, token],
            grad_write_state,
        )
        grad_right[:, :, :, token] += torch.einsum(
            "bhnrv,bhnkv->bhnrk", grad_write_read, write_state_before
        )
        grad_write_state = grad_write_state + torch.einsum(
            "bhnrk,bhnrv->bhnkv", right_token, grad_write_read
        )
    return grad_left, grad_right, grad_normalized_write, grad_values


def _qgdn_streaming_wy_cuda_fwd(
    normalized_left: torch.Tensor,
    right: torch.Tensor,
    normalized_write: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the raw Triton forward after validating its diagnostic envelope.

    Inputs use ``[B,H,N,C,2,K]``, ``[B,H,N,C,K]`` and
    ``[B,H,N,C,V]`` layouts.
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
        BWT=max(16, chunk_size),
        num_warps=8,
        num_stages=2,
    )
    return effective_right, write_reads


def _qgdn_streaming_wy_cuda_bwd(
    normalized_left: torch.Tensor,
    right: torch.Tensor,
    normalized_write: torch.Tensor,
    values: torch.Tensor,
    grad_effective: torch.Tensor,
    grad_write_reads: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused in-program adjoint solve and factor gradients."""
    batch, heads, chunks, chunk_size, _, key_dim = right.shape
    value_dim = values.shape[-1]
    expected_grad_write = (*right.shape[:-1], value_dim)
    if grad_effective.shape != right.shape:
        raise ValueError("grad_effective has incompatible shape")
    if grad_write_reads.shape != expected_grad_write:
        raise ValueError("grad_write_reads has incompatible shape")

    normalized_left = normalized_left.contiguous()
    right = right.contiguous()
    normalized_write = normalized_write.contiguous()
    values = values.contiguous()
    grad_effective = grad_effective.contiguous()
    grad_write_reads = grad_write_reads.contiguous()
    grad_left = torch.empty(
        normalized_left.shape,
        dtype=normalized_left.dtype,
        device=normalized_left.device,
    )
    grad_right = torch.empty(
        right.shape, dtype=right.dtype, device=right.device
    )
    grad_normalized_write = torch.empty(
        normalized_write.shape,
        dtype=normalized_write.dtype,
        device=normalized_write.device,
    )
    grad_values = torch.empty(
        values.shape, dtype=values.dtype, device=values.device
    )
    block_rows = triton.next_power_of_2(chunk_size * 2)
    block_key = max(16, triton.next_power_of_2(key_dim))
    block_value = max(16, triton.next_power_of_2(value_dim))
    _qgdn_streaming_wy_bwd_kernel[(batch * heads * chunks,)](
        normalized_left,
        right,
        normalized_write,
        values,
        grad_effective,
        grad_write_reads,
        grad_left,
        grad_right,
        grad_normalized_write,
        grad_values,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BM=block_rows,
        BK=block_key,
        BV=block_value,
        BWT=max(16, chunk_size),
        num_warps=8,
        num_stages=2,
    )
    return grad_left, grad_right, grad_normalized_write, grad_values


class _QGDNStreamingWY(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normalized_left, right, normalized_write, values):
        ctx.save_for_backward(normalized_left, right, normalized_write, values)
        return _qgdn_streaming_wy_cuda_fwd(
            normalized_left, right, normalized_write, values
        )

    @staticmethod
    def backward(ctx, grad_effective, grad_write_reads):
        normalized_left, right, normalized_write, values = ctx.saved_tensors
        if grad_effective is None:
            grad_effective = torch.zeros_like(right)
        if grad_write_reads is None:
            grad_write_reads = values.new_zeros(
                *right.shape[:-1], values.shape[-1]
            )
        return _qgdn_streaming_wy_cuda_bwd(
            normalized_left,
            right,
            normalized_write,
            values,
            grad_effective,
            grad_write_reads,
        )


def qgdn_streaming_wy_fwd(
    normalized_left: torch.Tensor,
    right: torch.Tensor,
    normalized_write: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return WY factors using fused Triton forward and backward kernels."""
    return _QGDNStreamingWY.apply(
        normalized_left, right, normalized_write, values
    )
