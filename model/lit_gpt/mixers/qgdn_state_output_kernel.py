"""Fused physical-T QGDN chunk-state scan and output recovery kernels."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qgdn_chunk_state_fwd_kernel(
    normalized_left,
    effective_right,
    write_reads,
    normalized_write,
    values,
    decay_prefix,
    initial_state,
    chunk_starts,
    final_state,
    N,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BWT: tl.constexpr,
):
    lane = tl.program_id(0).to(tl.int64)
    value_block = tl.program_id(1).to(tl.int64)
    rank: tl.constexpr = 2
    rows: tl.constexpr = BT * rank
    o_m = tl.arange(0, BM)
    o_k = tl.arange(0, BK)
    o_v = value_block * BV + tl.arange(0, BV)
    o_t = tl.arange(0, BWT)
    m_m = o_m < rows
    m_k = o_k < K
    m_v = o_v < V
    m_t = o_t < BT

    state_base = lane * K * V
    p_initial = initial_state + state_base + o_k[:, None] * V + o_v[None, :]
    b_state = tl.load(
        p_initial, mask=m_k[:, None] & m_v[None, :], other=0.0
    ).to(tl.float32)

    for chunk in range(N):
        chunk64 = chunk.to(tl.int64)
        chunk_index = lane * N + chunk64
        p_start = (
            chunk_starts
            + chunk_index * K * V
            + o_k[:, None] * V
            + o_v[None, :]
        )
        tl.store(p_start, b_state, mask=m_k[:, None] & m_v[None, :])

        factor_base = chunk_index * rows * K
        p_left = normalized_left + factor_base + o_m[:, None] * K + o_k[None, :]
        p_effective = effective_right + factor_base + o_m[:, None] * K + o_k[None, :]
        b_left = tl.load(
            p_left, mask=m_m[:, None] & m_k[None, :], other=0.0
        ).to(tl.float32)
        b_effective = tl.load(
            p_effective, mask=m_m[:, None] & m_k[None, :], other=0.0
        ).to(tl.float32)

        rank_value_base = chunk_index * rows * V
        p_write_reads = (
            write_reads
            + rank_value_base
            + o_m[:, None] * V
            + o_v[None, :]
        )
        b_write_reads = tl.load(
            p_write_reads, mask=m_m[:, None] & m_v[None, :], other=0.0
        ).to(tl.float32)
        b_reads = b_write_reads + tl.dot(
            b_effective, b_state, input_precision="ieee"
        )

        write_base = chunk_index * BT * K
        p_write = normalized_write + write_base + o_t[:, None] * K + o_k[None, :]
        b_write = tl.load(
            p_write, mask=m_t[:, None] & m_k[None, :], other=0.0
        ).to(tl.float32)
        value_base = chunk_index * BT * V
        p_values = values + value_base + o_t[:, None] * V + o_v[None, :]
        b_values = tl.load(
            p_values, mask=m_t[:, None] & m_v[None, :], other=0.0
        ).to(tl.float32)

        b_update = tl.dot(b_left.T, b_reads, input_precision="ieee")
        b_update += tl.dot(b_write.T, b_values, input_precision="ieee")
        scale = tl.load(decay_prefix + chunk_index * BT + BT - 1).to(
            tl.float32
        )
        b_state = scale * (b_state + b_update)

    p_final = final_state + state_base + o_k[:, None] * V + o_v[None, :]
    tl.store(p_final, b_state, mask=m_k[:, None] & m_v[None, :])


@triton.jit
def _qgdn_chunk_output_fwd_kernel(
    queries,
    normalized_left,
    effective_right,
    write_reads,
    normalized_write,
    values,
    decay_prefix,
    chunk_starts,
    outputs,
    output_scale,
    N,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BWT: tl.constexpr,
):
    chunk_index = tl.program_id(0).to(tl.int64)
    value_block = tl.program_id(1).to(tl.int64)
    rank: tl.constexpr = 2
    rows: tl.constexpr = BT * rank
    o_m = tl.arange(0, BM)
    o_k = tl.arange(0, BK)
    o_v = value_block * BV + tl.arange(0, BV)
    o_t = tl.arange(0, BWT)
    m_m = o_m < rows
    m_k = o_k < K
    m_v = o_v < V
    m_t = o_t < BT

    p_start = (
        chunk_starts
        + chunk_index * K * V
        + o_k[:, None] * V
        + o_v[None, :]
    )
    b_start = tl.load(
        p_start, mask=m_k[:, None] & m_v[None, :], other=0.0
    ).to(tl.float32)
    query_base = chunk_index * BT * K
    p_queries = queries + query_base + o_t[:, None] * K + o_k[None, :]
    b_queries = tl.load(
        p_queries, mask=m_t[:, None] & m_k[None, :], other=0.0
    ).to(tl.float32)

    factor_base = chunk_index * rows * K
    p_left = normalized_left + factor_base + o_m[:, None] * K + o_k[None, :]
    p_effective = effective_right + factor_base + o_m[:, None] * K + o_k[None, :]
    b_left = tl.load(
        p_left, mask=m_m[:, None] & m_k[None, :], other=0.0
    ).to(tl.float32)
    b_effective = tl.load(
        p_effective, mask=m_m[:, None] & m_k[None, :], other=0.0
    ).to(tl.float32)
    rank_value_base = chunk_index * rows * V
    p_write_reads = (
        write_reads
        + rank_value_base
        + o_m[:, None] * V
        + o_v[None, :]
    )
    b_write_reads = tl.load(
        p_write_reads, mask=m_m[:, None] & m_v[None, :], other=0.0
    ).to(tl.float32)
    b_reads = b_write_reads + tl.dot(
        b_effective, b_start, input_precision="ieee"
    )

    write_base = chunk_index * BT * K
    p_write = normalized_write + write_base + o_t[:, None] * K + o_k[None, :]
    b_write = tl.load(
        p_write, mask=m_t[:, None] & m_k[None, :], other=0.0
    ).to(tl.float32)
    value_base = chunk_index * BT * V
    p_values = values + value_base + o_t[:, None] * V + o_v[None, :]
    b_values = tl.load(
        p_values, mask=m_t[:, None] & m_v[None, :], other=0.0
    ).to(tl.float32)

    b_output = tl.dot(b_queries, b_start, input_precision="ieee")
    b_query_left = tl.dot(b_queries, b_left.T, input_precision="ieee")
    factor_token = o_m // rank
    causal_rank = (
        (o_t[:, None] >= factor_token[None, :])
        & m_t[:, None]
        & m_m[None, :]
    )
    b_query_left = tl.where(causal_rank, b_query_left, 0.0)
    b_output += tl.dot(b_query_left, b_reads, input_precision="ieee")
    b_query_write = tl.dot(b_queries, b_write.T, input_precision="ieee")
    causal_write = (
        (o_t[:, None] >= o_t[None, :])
        & m_t[:, None]
        & m_t[None, :]
    )
    b_query_write = tl.where(causal_write, b_query_write, 0.0)
    b_output += tl.dot(b_query_write, b_values, input_precision="ieee")
    b_decay = tl.load(
        decay_prefix + chunk_index * BT + o_t, mask=m_t, other=0.0
    ).to(tl.float32)
    b_output *= (output_scale * b_decay)[:, None]
    p_outputs = outputs + value_base + o_t[:, None] * V + o_v[None, :]
    tl.store(
        p_outputs,
        b_output,
        mask=m_t[:, None] & m_v[None, :],
    )


@triton.jit
def _qgdn_chunk_state_output_bwd_kernel(
    queries,
    normalized_left,
    effective_right,
    write_reads,
    normalized_write,
    values,
    decay_prefix,
    chunk_starts,
    grad_outputs,
    grad_final_state,
    grad_queries,
    grad_left,
    grad_effective,
    grad_write_reads,
    grad_write,
    grad_values,
    grad_decay,
    grad_initial_state,
    output_scale,
    N,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BWT: tl.constexpr,
):
    lane = tl.program_id(0).to(tl.int64)
    value_block = tl.program_id(1).to(tl.int64)
    rank: tl.constexpr = 2
    rows: tl.constexpr = BT * rank
    o_m = tl.arange(0, BM)
    o_k = tl.arange(0, BK)
    o_v = value_block * BV + tl.arange(0, BV)
    o_t = tl.arange(0, BWT)
    m_m = o_m < rows
    m_k = o_k < K
    m_v = o_v < V
    m_t = o_t < BT

    state_base = lane * K * V
    p_grad_final = (
        grad_final_state
        + state_base
        + o_k[:, None] * V
        + o_v[None, :]
    )
    b_grad_state = tl.load(
        p_grad_final, mask=m_k[:, None] & m_v[None, :], other=0.0
    ).to(tl.float32)

    for reverse_chunk in range(N):
        chunk = N - 1 - reverse_chunk
        chunk64 = chunk.to(tl.int64)
        chunk_index = lane * N + chunk64

        p_start = (
            chunk_starts
            + chunk_index * K * V
            + o_k[:, None] * V
            + o_v[None, :]
        )
        b_start = tl.load(
            p_start, mask=m_k[:, None] & m_v[None, :], other=0.0
        ).to(tl.float32)
        query_base = chunk_index * BT * K
        p_queries = queries + query_base + o_t[:, None] * K + o_k[None, :]
        b_queries = tl.load(
            p_queries, mask=m_t[:, None] & m_k[None, :], other=0.0
        ).to(tl.float32)

        factor_base = chunk_index * rows * K
        p_left = normalized_left + factor_base + o_m[:, None] * K + o_k[None, :]
        p_effective = effective_right + factor_base + o_m[:, None] * K + o_k[None, :]
        b_left = tl.load(
            p_left, mask=m_m[:, None] & m_k[None, :], other=0.0
        ).to(tl.float32)
        b_effective = tl.load(
            p_effective, mask=m_m[:, None] & m_k[None, :], other=0.0
        ).to(tl.float32)
        rank_value_base = chunk_index * rows * V
        p_write_reads = (
            write_reads
            + rank_value_base
            + o_m[:, None] * V
            + o_v[None, :]
        )
        b_write_reads = tl.load(
            p_write_reads, mask=m_m[:, None] & m_v[None, :], other=0.0
        ).to(tl.float32)
        b_reads = b_write_reads + tl.dot(
            b_effective, b_start, input_precision="ieee"
        )

        write_base = chunk_index * BT * K
        p_write = normalized_write + write_base + o_t[:, None] * K + o_k[None, :]
        b_write = tl.load(
            p_write, mask=m_t[:, None] & m_k[None, :], other=0.0
        ).to(tl.float32)
        value_base = chunk_index * BT * V
        p_values = values + value_base + o_t[:, None] * V + o_v[None, :]
        b_values = tl.load(
            p_values, mask=m_t[:, None] & m_v[None, :], other=0.0
        ).to(tl.float32)

        b_query_left = tl.dot(b_queries, b_left.T, input_precision="ieee")
        factor_token = o_m // rank
        causal_rank = (
            (o_t[:, None] >= factor_token[None, :])
            & m_t[:, None]
            & m_m[None, :]
        )
        b_query_left = tl.where(causal_rank, b_query_left, 0.0)
        b_query_write = tl.dot(b_queries, b_write.T, input_precision="ieee")
        causal_write = (
            (o_t[:, None] >= o_t[None, :])
            & m_t[:, None]
            & m_t[None, :]
        )
        b_query_write = tl.where(causal_write, b_query_write, 0.0)

        p_grad_output = (
            grad_outputs + value_base + o_t[:, None] * V + o_v[None, :]
        )
        b_grad_output_raw = tl.load(
            p_grad_output, mask=m_t[:, None] & m_v[None, :], other=0.0
        ).to(tl.float32)
        b_decay = tl.load(
            decay_prefix + chunk_index * BT + o_t, mask=m_t, other=0.0
        ).to(tl.float32)
        b_grad_output = b_grad_output_raw * (
            output_scale * b_decay
        )[:, None]

        b_unscaled_output = tl.dot(
            b_queries, b_start, input_precision="ieee"
        )
        b_unscaled_output += tl.dot(
            b_query_left, b_reads, input_precision="ieee"
        )
        b_unscaled_output += tl.dot(
            b_query_write, b_values, input_precision="ieee"
        )
        b_grad_decay = output_scale * tl.sum(
            b_grad_output_raw * b_unscaled_output, axis=1
        )

        b_grad_query_left = tl.dot(
            b_grad_output, b_reads.T, input_precision="ieee"
        )
        b_grad_query_left = tl.where(causal_rank, b_grad_query_left, 0.0)
        b_grad_query_write = tl.dot(
            b_grad_output, b_values.T, input_precision="ieee"
        )
        b_grad_query_write = tl.where(
            causal_write, b_grad_query_write, 0.0
        )
        b_grad_queries = tl.dot(
            b_grad_output, b_start.T, input_precision="ieee"
        )
        b_grad_queries += tl.dot(
            b_grad_query_left, b_left, input_precision="ieee"
        )
        b_grad_queries += tl.dot(
            b_grad_query_write, b_write, input_precision="ieee"
        )
        b_grad_left = tl.dot(
            b_grad_query_left.T, b_queries, input_precision="ieee"
        )
        b_grad_reads = tl.dot(
            b_query_left.T, b_grad_output, input_precision="ieee"
        )
        b_grad_write = tl.dot(
            b_grad_query_write.T, b_queries, input_precision="ieee"
        )
        b_grad_values = tl.dot(
            b_query_write.T, b_grad_output, input_precision="ieee"
        )
        b_grad_start = tl.dot(
            b_queries.T, b_grad_output, input_precision="ieee"
        )

        b_transition_state = b_start
        b_transition_state += tl.dot(
            b_left.T, b_reads, input_precision="ieee"
        )
        b_transition_state += tl.dot(
            b_write.T, b_values, input_precision="ieee"
        )
        b_grad_scale = tl.sum(
            tl.sum(b_grad_state * b_transition_state, axis=1), axis=0
        )
        scale = tl.load(decay_prefix + chunk_index * BT + BT - 1).to(
            tl.float32
        )
        b_grad_transition = scale * b_grad_state
        b_grad_left += tl.dot(
            b_reads, b_grad_transition.T, input_precision="ieee"
        )
        b_grad_reads += tl.dot(
            b_left, b_grad_transition, input_precision="ieee"
        )
        b_grad_write += tl.dot(
            b_values, b_grad_transition.T, input_precision="ieee"
        )
        b_grad_values += tl.dot(
            b_write, b_grad_transition, input_precision="ieee"
        )
        b_grad_start += b_grad_transition

        b_grad_effective = tl.dot(
            b_grad_reads, b_start.T, input_precision="ieee"
        )
        b_grad_start += tl.dot(
            b_effective.T, b_grad_reads, input_precision="ieee"
        )
        b_grad_decay += tl.where(o_t == BT - 1, b_grad_scale, 0.0)

        p_grad_queries = (
            grad_queries + query_base + o_t[:, None] * K + o_k[None, :]
        )
        tl.atomic_add(
            p_grad_queries,
            b_grad_queries,
            mask=m_t[:, None] & m_k[None, :],
        )
        p_grad_left = grad_left + factor_base + o_m[:, None] * K + o_k[None, :]
        p_grad_effective = (
            grad_effective + factor_base + o_m[:, None] * K + o_k[None, :]
        )
        tl.atomic_add(
            p_grad_left,
            b_grad_left,
            mask=m_m[:, None] & m_k[None, :],
        )
        tl.atomic_add(
            p_grad_effective,
            b_grad_effective,
            mask=m_m[:, None] & m_k[None, :],
        )
        p_grad_rank_values = (
            grad_write_reads
            + rank_value_base
            + o_m[:, None] * V
            + o_v[None, :]
        )
        tl.store(
            p_grad_rank_values,
            b_grad_reads,
            mask=m_m[:, None] & m_v[None, :],
        )
        p_grad_write = (
            grad_write + write_base + o_t[:, None] * K + o_k[None, :]
        )
        tl.atomic_add(
            p_grad_write,
            b_grad_write,
            mask=m_t[:, None] & m_k[None, :],
        )
        p_grad_values = (
            grad_values + value_base + o_t[:, None] * V + o_v[None, :]
        )
        tl.store(
            p_grad_values,
            b_grad_values,
            mask=m_t[:, None] & m_v[None, :],
        )
        tl.atomic_add(
            grad_decay + chunk_index * BT + o_t,
            b_grad_decay,
            mask=m_t,
        )
        b_grad_state = b_grad_start

    p_grad_initial = (
        grad_initial_state
        + state_base
        + o_k[:, None] * V
        + o_v[None, :]
    )
    tl.store(
        p_grad_initial,
        b_grad_state,
        mask=m_k[:, None] & m_v[None, :],
    )


@triton.jit
def _qgdn_chunk_output_bwd_kernel(
    queries,
    normalized_left,
    effective_right,
    write_reads,
    normalized_write,
    values,
    decay_prefix,
    chunk_starts,
    grad_outputs,
    grad_queries,
    grad_left,
    grad_effective,
    grad_write_reads,
    grad_write,
    grad_values,
    grad_decay,
    grad_chunk_starts,
    output_scale,
    N,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BWT: tl.constexpr,
):
    """Differentiate each chunk's outputs independently and in parallel."""
    chunk_index = tl.program_id(0).to(tl.int64)
    value_block = tl.program_id(1).to(tl.int64)
    rank: tl.constexpr = 2
    rows: tl.constexpr = BT * rank
    o_m = tl.arange(0, BM)
    o_k = tl.arange(0, BK)
    o_v = value_block * BV + tl.arange(0, BV)
    o_t = tl.arange(0, BWT)
    m_m = o_m < rows
    m_k = o_k < K
    m_v = o_v < V
    m_t = o_t < BT

    p_start = (
        chunk_starts
        + chunk_index * K * V
        + o_k[:, None] * V
        + o_v[None, :]
    )
    b_start = tl.load(
        p_start, mask=m_k[:, None] & m_v[None, :], other=0.0
    ).to(tl.float32)
    query_base = chunk_index * BT * K
    p_queries = queries + query_base + o_t[:, None] * K + o_k[None, :]
    b_queries = tl.load(
        p_queries, mask=m_t[:, None] & m_k[None, :], other=0.0
    ).to(tl.float32)

    factor_base = chunk_index * rows * K
    p_left = normalized_left + factor_base + o_m[:, None] * K + o_k[None, :]
    p_effective = (
        effective_right + factor_base + o_m[:, None] * K + o_k[None, :]
    )
    b_left = tl.load(
        p_left, mask=m_m[:, None] & m_k[None, :], other=0.0
    ).to(tl.float32)
    b_effective = tl.load(
        p_effective, mask=m_m[:, None] & m_k[None, :], other=0.0
    ).to(tl.float32)
    rank_value_base = chunk_index * rows * V
    p_write_reads = (
        write_reads + rank_value_base + o_m[:, None] * V + o_v[None, :]
    )
    b_write_reads = tl.load(
        p_write_reads, mask=m_m[:, None] & m_v[None, :], other=0.0
    ).to(tl.float32)
    b_reads = b_write_reads + tl.dot(
        b_effective, b_start, input_precision="ieee"
    )

    write_base = chunk_index * BT * K
    p_write = normalized_write + write_base + o_t[:, None] * K + o_k[None, :]
    b_write = tl.load(
        p_write, mask=m_t[:, None] & m_k[None, :], other=0.0
    ).to(tl.float32)
    value_base = chunk_index * BT * V
    p_values = values + value_base + o_t[:, None] * V + o_v[None, :]
    b_values = tl.load(
        p_values, mask=m_t[:, None] & m_v[None, :], other=0.0
    ).to(tl.float32)

    b_query_left = tl.dot(b_queries, b_left.T, input_precision="ieee")
    factor_token = o_m // rank
    causal_rank = (
        (o_t[:, None] >= factor_token[None, :])
        & m_t[:, None]
        & m_m[None, :]
    )
    b_query_left = tl.where(causal_rank, b_query_left, 0.0)
    b_query_write = tl.dot(b_queries, b_write.T, input_precision="ieee")
    causal_write = (
        (o_t[:, None] >= o_t[None, :])
        & m_t[:, None]
        & m_t[None, :]
    )
    b_query_write = tl.where(causal_write, b_query_write, 0.0)

    p_grad_output = grad_outputs + value_base + o_t[:, None] * V + o_v[None, :]
    b_grad_output_raw = tl.load(
        p_grad_output, mask=m_t[:, None] & m_v[None, :], other=0.0
    ).to(tl.float32)
    b_decay = tl.load(
        decay_prefix + chunk_index * BT + o_t, mask=m_t, other=0.0
    ).to(tl.float32)
    b_grad_output = b_grad_output_raw * (output_scale * b_decay)[:, None]

    b_unscaled_output = tl.dot(b_queries, b_start, input_precision="ieee")
    b_unscaled_output += tl.dot(
        b_query_left, b_reads, input_precision="ieee"
    )
    b_unscaled_output += tl.dot(
        b_query_write, b_values, input_precision="ieee"
    )
    b_grad_decay = output_scale * tl.sum(
        b_grad_output_raw * b_unscaled_output, axis=1
    )

    b_grad_query_left = tl.dot(
        b_grad_output, b_reads.T, input_precision="ieee"
    )
    b_grad_query_left = tl.where(causal_rank, b_grad_query_left, 0.0)
    b_grad_query_write = tl.dot(
        b_grad_output, b_values.T, input_precision="ieee"
    )
    b_grad_query_write = tl.where(causal_write, b_grad_query_write, 0.0)
    b_grad_queries = tl.dot(
        b_grad_output, b_start.T, input_precision="ieee"
    )
    b_grad_queries += tl.dot(
        b_grad_query_left, b_left, input_precision="ieee"
    )
    b_grad_queries += tl.dot(
        b_grad_query_write, b_write, input_precision="ieee"
    )
    b_grad_left = tl.dot(
        b_grad_query_left.T, b_queries, input_precision="ieee"
    )
    b_grad_reads = tl.dot(
        b_query_left.T, b_grad_output, input_precision="ieee"
    )
    b_grad_write = tl.dot(
        b_grad_query_write.T, b_queries, input_precision="ieee"
    )
    b_grad_values = tl.dot(
        b_query_write.T, b_grad_output, input_precision="ieee"
    )
    b_grad_start = tl.dot(
        b_queries.T, b_grad_output, input_precision="ieee"
    )
    b_grad_effective = tl.dot(
        b_grad_reads, b_start.T, input_precision="ieee"
    )
    b_grad_start += tl.dot(
        b_effective.T, b_grad_reads, input_precision="ieee"
    )

    p_grad_queries = (
        grad_queries + query_base + o_t[:, None] * K + o_k[None, :]
    )
    if BV >= V:
        tl.store(
            p_grad_queries,
            b_grad_queries,
            mask=m_t[:, None] & m_k[None, :],
        )
    else:
        tl.atomic_add(
            p_grad_queries,
            b_grad_queries,
            mask=m_t[:, None] & m_k[None, :],
        )
    p_grad_left = grad_left + factor_base + o_m[:, None] * K + o_k[None, :]
    p_grad_effective = (
        grad_effective + factor_base + o_m[:, None] * K + o_k[None, :]
    )
    if BV >= V:
        tl.store(
            p_grad_left,
            b_grad_left,
            mask=m_m[:, None] & m_k[None, :],
        )
        tl.store(
            p_grad_effective,
            b_grad_effective,
            mask=m_m[:, None] & m_k[None, :],
        )
    else:
        tl.atomic_add(
            p_grad_left,
            b_grad_left,
            mask=m_m[:, None] & m_k[None, :],
        )
        tl.atomic_add(
            p_grad_effective,
            b_grad_effective,
            mask=m_m[:, None] & m_k[None, :],
        )
    p_grad_rank_values = (
        grad_write_reads
        + rank_value_base
        + o_m[:, None] * V
        + o_v[None, :]
    )
    tl.store(
        p_grad_rank_values,
        b_grad_reads,
        mask=m_m[:, None] & m_v[None, :],
    )
    p_grad_write = grad_write + write_base + o_t[:, None] * K + o_k[None, :]
    if BV >= V:
        tl.store(
            p_grad_write,
            b_grad_write,
            mask=m_t[:, None] & m_k[None, :],
        )
    else:
        tl.atomic_add(
            p_grad_write,
            b_grad_write,
            mask=m_t[:, None] & m_k[None, :],
        )
    p_grad_values = grad_values + value_base + o_t[:, None] * V + o_v[None, :]
    tl.store(
        p_grad_values,
        b_grad_values,
        mask=m_t[:, None] & m_v[None, :],
    )
    if BV >= V:
        tl.store(
            grad_decay + chunk_index * BT + o_t,
            b_grad_decay,
            mask=m_t,
        )
    else:
        tl.atomic_add(
            grad_decay + chunk_index * BT + o_t,
            b_grad_decay,
            mask=m_t,
        )
    p_grad_start = (
        grad_chunk_starts
        + chunk_index * K * V
        + o_k[:, None] * V
        + o_v[None, :]
    )
    tl.store(
        p_grad_start,
        b_grad_start,
        mask=m_k[:, None] & m_v[None, :],
    )


@triton.jit
def _qgdn_chunk_state_bwd_kernel(
    normalized_left,
    effective_right,
    write_reads,
    normalized_write,
    values,
    decay_prefix,
    chunk_starts,
    grad_final_state,
    grad_chunk_starts,
    grad_left,
    grad_effective,
    grad_write_reads,
    grad_write,
    grad_values,
    grad_decay,
    grad_initial_state,
    N,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BWT: tl.constexpr,
):
    """Reverse only the compact inter-chunk transition recurrence."""
    lane = tl.program_id(0).to(tl.int64)
    value_block = tl.program_id(1).to(tl.int64)
    rank: tl.constexpr = 2
    rows: tl.constexpr = BT * rank
    o_m = tl.arange(0, BM)
    o_k = tl.arange(0, BK)
    o_v = value_block * BV + tl.arange(0, BV)
    o_t = tl.arange(0, BWT)
    m_m = o_m < rows
    m_k = o_k < K
    m_v = o_v < V
    m_t = o_t < BT

    state_base = lane * K * V
    p_grad_final = (
        grad_final_state
        + state_base
        + o_k[:, None] * V
        + o_v[None, :]
    )
    b_grad_state = tl.load(
        p_grad_final, mask=m_k[:, None] & m_v[None, :], other=0.0
    ).to(tl.float32)

    for reverse_chunk in range(N):
        chunk = N - 1 - reverse_chunk
        chunk64 = chunk.to(tl.int64)
        chunk_index = lane * N + chunk64
        p_start = (
            chunk_starts
            + chunk_index * K * V
            + o_k[:, None] * V
            + o_v[None, :]
        )
        b_start = tl.load(
            p_start, mask=m_k[:, None] & m_v[None, :], other=0.0
        ).to(tl.float32)
        p_grad_start = (
            grad_chunk_starts
            + chunk_index * K * V
            + o_k[:, None] * V
            + o_v[None, :]
        )
        b_grad_start = tl.load(
            p_grad_start, mask=m_k[:, None] & m_v[None, :], other=0.0
        ).to(tl.float32)

        factor_base = chunk_index * rows * K
        p_left = normalized_left + factor_base + o_m[:, None] * K + o_k[None, :]
        p_effective = (
            effective_right + factor_base + o_m[:, None] * K + o_k[None, :]
        )
        b_left = tl.load(
            p_left, mask=m_m[:, None] & m_k[None, :], other=0.0
        ).to(tl.float32)
        b_effective = tl.load(
            p_effective, mask=m_m[:, None] & m_k[None, :], other=0.0
        ).to(tl.float32)
        rank_value_base = chunk_index * rows * V
        p_write_reads = (
            write_reads
            + rank_value_base
            + o_m[:, None] * V
            + o_v[None, :]
        )
        b_write_reads = tl.load(
            p_write_reads, mask=m_m[:, None] & m_v[None, :], other=0.0
        ).to(tl.float32)
        b_reads = b_write_reads + tl.dot(
            b_effective, b_start, input_precision="ieee"
        )

        write_base = chunk_index * BT * K
        p_write = normalized_write + write_base + o_t[:, None] * K + o_k[None, :]
        b_write = tl.load(
            p_write, mask=m_t[:, None] & m_k[None, :], other=0.0
        ).to(tl.float32)
        value_base = chunk_index * BT * V
        p_values = values + value_base + o_t[:, None] * V + o_v[None, :]
        b_values = tl.load(
            p_values, mask=m_t[:, None] & m_v[None, :], other=0.0
        ).to(tl.float32)

        b_transition_state = b_start
        b_transition_state += tl.dot(
            b_left.T, b_reads, input_precision="ieee"
        )
        b_transition_state += tl.dot(
            b_write.T, b_values, input_precision="ieee"
        )
        b_grad_scale = tl.sum(
            tl.sum(b_grad_state * b_transition_state, axis=1), axis=0
        )
        scale = tl.load(decay_prefix + chunk_index * BT + BT - 1).to(
            tl.float32
        )
        b_grad_transition = scale * b_grad_state
        b_grad_left = tl.dot(
            b_reads, b_grad_transition.T, input_precision="ieee"
        )
        b_grad_reads = tl.dot(
            b_left, b_grad_transition, input_precision="ieee"
        )
        b_grad_write = tl.dot(
            b_values, b_grad_transition.T, input_precision="ieee"
        )
        b_grad_values = tl.dot(
            b_write, b_grad_transition, input_precision="ieee"
        )
        b_grad_start += b_grad_transition
        b_grad_effective = tl.dot(
            b_grad_reads, b_start.T, input_precision="ieee"
        )
        b_grad_start += tl.dot(
            b_effective.T, b_grad_reads, input_precision="ieee"
        )

        p_grad_left = (
            grad_left + factor_base + o_m[:, None] * K + o_k[None, :]
        )
        p_grad_effective = (
            grad_effective
            + factor_base
            + o_m[:, None] * K
            + o_k[None, :]
        )
        if BV >= V:
            b_existing_left = tl.load(
                p_grad_left,
                mask=m_m[:, None] & m_k[None, :],
                other=0.0,
            )
            b_existing_effective = tl.load(
                p_grad_effective,
                mask=m_m[:, None] & m_k[None, :],
                other=0.0,
            )
            tl.store(
                p_grad_left,
                b_existing_left + b_grad_left,
                mask=m_m[:, None] & m_k[None, :],
            )
            tl.store(
                p_grad_effective,
                b_existing_effective + b_grad_effective,
                mask=m_m[:, None] & m_k[None, :],
            )
        else:
            tl.atomic_add(
                p_grad_left,
                b_grad_left,
                mask=m_m[:, None] & m_k[None, :],
            )
            tl.atomic_add(
                p_grad_effective,
                b_grad_effective,
                mask=m_m[:, None] & m_k[None, :],
            )
        p_grad_rank_values = (
            grad_write_reads
            + rank_value_base
            + o_m[:, None] * V
            + o_v[None, :]
        )
        b_existing_reads = tl.load(
            p_grad_rank_values,
            mask=m_m[:, None] & m_v[None, :],
            other=0.0,
        )
        tl.store(
            p_grad_rank_values,
            b_existing_reads + b_grad_reads,
            mask=m_m[:, None] & m_v[None, :],
        )
        p_grad_write = (
            grad_write + write_base + o_t[:, None] * K + o_k[None, :]
        )
        if BV >= V:
            b_existing_write = tl.load(
                p_grad_write,
                mask=m_t[:, None] & m_k[None, :],
                other=0.0,
            )
            tl.store(
                p_grad_write,
                b_existing_write + b_grad_write,
                mask=m_t[:, None] & m_k[None, :],
            )
        else:
            tl.atomic_add(
                p_grad_write,
                b_grad_write,
                mask=m_t[:, None] & m_k[None, :],
            )
        p_grad_values = (
            grad_values + value_base + o_t[:, None] * V + o_v[None, :]
        )
        b_existing_values = tl.load(
            p_grad_values,
            mask=m_t[:, None] & m_v[None, :],
            other=0.0,
        )
        tl.store(
            p_grad_values,
            b_existing_values + b_grad_values,
            mask=m_t[:, None] & m_v[None, :],
        )
        p_grad_scale = grad_decay + chunk_index * BT + BT - 1
        if BV >= V:
            b_existing_scale = tl.load(p_grad_scale)
            tl.store(p_grad_scale, b_existing_scale + b_grad_scale)
        else:
            tl.atomic_add(p_grad_scale, b_grad_scale)
        b_grad_state = b_grad_start

    p_grad_initial = (
        grad_initial_state
        + state_base
        + o_k[:, None] * V
        + o_v[None, :]
    )
    tl.store(
        p_grad_initial,
        b_grad_state,
        mask=m_k[:, None] & m_v[None, :],
    )


def _validate_inputs(
    queries: torch.Tensor,
    decay_prefix: torch.Tensor,
    normalized_left: torch.Tensor,
    effective_right: torch.Tensor,
    write_reads: torch.Tensor,
    normalized_write: torch.Tensor,
    values: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    tensors = (
        queries,
        decay_prefix,
        normalized_left,
        effective_right,
        write_reads,
        normalized_write,
        values,
        initial_state,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("the fused chunk-state/output primitive requires CUDA tensors")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise ValueError("the diagnostic chunk-state/output primitive requires float32")
    if normalized_left.shape != effective_right.shape or normalized_left.ndim != 6:
        raise ValueError("left/effective factors must share [B,H,N,C,2,K]")
    batch, heads, chunks, chunk_size, rank, key_dim = normalized_left.shape
    value_dim = values.shape[-1]
    if rank != 2:
        raise ValueError("the QGDN chunk-state/output primitive requires rank two")
    if chunk_size not in {8, 16}:
        raise ValueError("the diagnostic primitive supports chunk sizes 8 and 16")
    if key_dim > 128 or value_dim > 128:
        raise ValueError("the diagnostic primitive supports key/value dims up to 128")
    if queries.shape != (batch, heads, chunks, chunk_size, key_dim):
        raise ValueError("queries have incompatible shape")
    if decay_prefix.shape != (batch, heads, chunks, chunk_size):
        raise ValueError("decay prefixes have incompatible shape")
    if write_reads.shape != (
        batch,
        heads,
        chunks,
        chunk_size,
        rank,
        value_dim,
    ):
        raise ValueError("write reads have incompatible shape")
    if normalized_write.shape != queries.shape:
        raise ValueError("normalized writes have incompatible shape")
    if values.shape != (batch, heads, chunks, chunk_size, value_dim):
        raise ValueError("values have incompatible shape")
    if initial_state.shape != (batch, heads, key_dim, value_dim):
        raise ValueError("initial state has incompatible shape")
    return batch, heads, chunks, chunk_size, key_dim, value_dim


def _launch_config(chunk_size: int, key_dim: int):
    return (
        triton.next_power_of_2(chunk_size * 2),
        max(16, triton.next_power_of_2(key_dim)),
        max(16, triton.next_power_of_2(chunk_size)),
    )


def _qgdn_chunk_state_cuda_fwd(
    normalized_left,
    effective_right,
    write_reads,
    normalized_write,
    values,
    decay_prefix,
    initial_state,
):
    batch, heads, chunks, chunk_size, _, key_dim = normalized_left.shape
    value_dim = values.shape[-1]
    chunk_starts = initial_state.new_empty(
        batch, heads, chunks, key_dim, value_dim
    )
    final_state = torch.empty_like(initial_state)
    block_rows, block_key, block_time = _launch_config(chunk_size, key_dim)
    state_value_block = 32
    _qgdn_chunk_state_fwd_kernel[
        (batch * heads, triton.cdiv(value_dim, state_value_block))
    ](
        normalized_left,
        effective_right,
        write_reads,
        normalized_write,
        values,
        decay_prefix,
        initial_state,
        chunk_starts,
        final_state,
        N=chunks,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BM=block_rows,
        BK=block_key,
        BV=state_value_block,
        BWT=block_time,
        num_warps=8,
        num_stages=2,
    )
    return chunk_starts, final_state


def _qgdn_chunk_state_output_cuda_fwd(
    queries,
    decay_prefix,
    normalized_left,
    effective_right,
    write_reads,
    normalized_write,
    values,
    initial_state,
    output_scale,
):
    batch, heads, chunks, chunk_size, key_dim, value_dim = _validate_inputs(
        queries,
        decay_prefix,
        normalized_left,
        effective_right,
        write_reads,
        normalized_write,
        values,
        initial_state,
    )
    inputs = [
        tensor.contiguous()
        for tensor in (
            queries,
            decay_prefix,
            normalized_left,
            effective_right,
            write_reads,
            normalized_write,
            values,
            initial_state,
        )
    ]
    (
        queries,
        decay_prefix,
        normalized_left,
        effective_right,
        write_reads,
        normalized_write,
        values,
        initial_state,
    ) = inputs
    chunk_starts, final_state = _qgdn_chunk_state_cuda_fwd(
        normalized_left,
        effective_right,
        write_reads,
        normalized_write,
        values,
        decay_prefix,
        initial_state,
    )
    outputs = values.new_empty(batch, heads, chunks, chunk_size, value_dim)
    block_rows, block_key, block_time = _launch_config(chunk_size, key_dim)
    state_value_block = 32
    _qgdn_chunk_output_fwd_kernel[(batch * heads * chunks, triton.cdiv(value_dim, state_value_block))](
        queries,
        normalized_left,
        effective_right,
        write_reads,
        normalized_write,
        values,
        decay_prefix,
        chunk_starts,
        outputs,
        output_scale,
        N=chunks,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BM=block_rows,
        BK=block_key,
        BV=state_value_block,
        BWT=block_time,
        num_warps=8,
        num_stages=2,
    )
    return outputs, final_state, chunk_starts, tuple(inputs)


def _qgdn_chunk_state_output_cuda_bwd_serial(
    saved_inputs,
    chunk_starts,
    grad_outputs,
    grad_final_state,
    output_scale,
):
    (
        queries,
        decay_prefix,
        normalized_left,
        effective_right,
        write_reads,
        normalized_write,
        values,
        initial_state,
    ) = saved_inputs
    batch, heads, chunks, chunk_size, _, key_dim = normalized_left.shape
    value_dim = values.shape[-1]
    grad_outputs = grad_outputs.contiguous()
    grad_final_state = grad_final_state.contiguous()
    grad_queries = torch.zeros_like(queries)
    grad_decay = torch.zeros_like(decay_prefix)
    grad_left = torch.zeros_like(normalized_left)
    grad_effective = torch.zeros_like(effective_right)
    grad_write_reads = torch.empty_like(write_reads)
    grad_write = torch.zeros_like(normalized_write)
    grad_values = torch.empty_like(values)
    grad_initial_state = torch.empty_like(initial_state)
    block_rows, block_key, block_time = _launch_config(chunk_size, key_dim)
    value_block = 16
    _qgdn_chunk_state_output_bwd_kernel[(batch * heads, triton.cdiv(value_dim, value_block))](
        queries,
        normalized_left,
        effective_right,
        write_reads,
        normalized_write,
        values,
        decay_prefix,
        chunk_starts,
        grad_outputs,
        grad_final_state,
        grad_queries,
        grad_left,
        grad_effective,
        grad_write_reads,
        grad_write,
        grad_values,
        grad_decay,
        grad_initial_state,
        output_scale,
        N=chunks,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BM=block_rows,
        BK=block_key,
        BV=value_block,
        BWT=block_time,
        num_warps=8,
        num_stages=2,
    )
    return (
        grad_queries,
        grad_decay,
        grad_left,
        grad_effective,
        grad_write_reads,
        grad_write,
        grad_values,
        grad_initial_state,
    )


def _qgdn_chunk_state_output_cuda_bwd(
    saved_inputs,
    chunk_starts,
    grad_outputs,
    grad_final_state,
    output_scale,
    *,
    output_value_block=64,
    state_value_block=16,
):
    """Parallelize output adjoints, then scan only compact state adjoints."""
    (
        queries,
        decay_prefix,
        normalized_left,
        effective_right,
        write_reads,
        normalized_write,
        values,
        initial_state,
    ) = saved_inputs
    batch, heads, chunks, chunk_size, _, key_dim = normalized_left.shape
    value_dim = values.shape[-1]
    if output_value_block not in {16, 32, 64}:
        raise ValueError("output_value_block must be one of 16, 32, or 64")
    if state_value_block not in {16, 32, 64}:
        raise ValueError("state_value_block must be one of 16, 32, or 64")
    grad_outputs = grad_outputs.contiguous()
    grad_final_state = grad_final_state.contiguous()
    grad_queries = torch.zeros_like(queries)
    grad_decay = torch.zeros_like(decay_prefix)
    grad_left = torch.zeros_like(normalized_left)
    grad_effective = torch.zeros_like(effective_right)
    grad_write_reads = torch.empty_like(write_reads)
    grad_write = torch.zeros_like(normalized_write)
    grad_values = torch.empty_like(values)
    grad_chunk_starts = torch.empty_like(chunk_starts)
    grad_initial_state = torch.empty_like(initial_state)
    block_rows, block_key, block_time = _launch_config(chunk_size, key_dim)
    _qgdn_chunk_output_bwd_kernel[
        (batch * heads * chunks, triton.cdiv(value_dim, output_value_block))
    ](
        queries,
        normalized_left,
        effective_right,
        write_reads,
        normalized_write,
        values,
        decay_prefix,
        chunk_starts,
        grad_outputs,
        grad_queries,
        grad_left,
        grad_effective,
        grad_write_reads,
        grad_write,
        grad_values,
        grad_decay,
        grad_chunk_starts,
        output_scale,
        N=chunks,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BM=block_rows,
        BK=block_key,
        BV=output_value_block,
        BWT=block_time,
        num_warps=8,
        num_stages=2,
    )
    _qgdn_chunk_state_bwd_kernel[
        (batch * heads, triton.cdiv(value_dim, state_value_block))
    ](
        normalized_left,
        effective_right,
        write_reads,
        normalized_write,
        values,
        decay_prefix,
        chunk_starts,
        grad_final_state,
        grad_chunk_starts,
        grad_left,
        grad_effective,
        grad_write_reads,
        grad_write,
        grad_values,
        grad_decay,
        grad_initial_state,
        N=chunks,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BM=block_rows,
        BK=block_key,
        BV=state_value_block,
        BWT=block_time,
        num_warps=8,
        num_stages=2,
    )
    return (
        grad_queries,
        grad_decay,
        grad_left,
        grad_effective,
        grad_write_reads,
        grad_write,
        grad_values,
        grad_initial_state,
    )


class _QGDNChunkStateOutput(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        queries,
        decay_prefix,
        normalized_left,
        effective_right,
        write_reads,
        normalized_write,
        values,
        initial_state,
        output_scale,
    ):
        outputs, final_state, _chunk_starts, saved_inputs = (
            _qgdn_chunk_state_output_cuda_fwd(
                queries,
                decay_prefix,
                normalized_left,
                effective_right,
                write_reads,
                normalized_write,
                values,
                initial_state,
                output_scale,
            )
        )
        # Chunk starts dominate the tensors retained across model layers.
        # Recreate them once with the compact state kernel during backward.
        ctx.save_for_backward(*saved_inputs)
        ctx.output_scale = output_scale
        return outputs, final_state

    @staticmethod
    def backward(ctx, grad_outputs, grad_final_state):
        saved_inputs = ctx.saved_tensors
        values = saved_inputs[6]
        initial_state = saved_inputs[7]
        if grad_outputs is None:
            grad_outputs = torch.zeros_like(values)
        if grad_final_state is None:
            grad_final_state = torch.zeros_like(initial_state)
        chunk_starts, _ = _qgdn_chunk_state_cuda_fwd(
            saved_inputs[2],
            saved_inputs[3],
            saved_inputs[4],
            saved_inputs[5],
            values,
            saved_inputs[1],
            initial_state,
        )
        gradients = _qgdn_chunk_state_output_cuda_bwd(
            saved_inputs,
            chunk_starts,
            grad_outputs,
            grad_final_state,
            ctx.output_scale,
        )
        return (*gradients, None)


def qgdn_chunk_state_output(
    queries: torch.Tensor,
    decay_prefix: torch.Tensor,
    normalized_left: torch.Tensor,
    effective_right: torch.Tensor,
    write_reads: torch.Tensor,
    normalized_write: torch.Tensor,
    values: torch.Tensor,
    initial_state: torch.Tensor,
    output_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan exact rank-2 chunk maps and recover physical-T outputs."""
    return _QGDNChunkStateOutput.apply(
        queries,
        decay_prefix,
        normalized_left,
        effective_right,
        write_reads,
        normalized_write,
        values,
        initial_state,
        output_scale,
    )
