"""Differentiable QGDN reference recurrences and physical-time affine oracle.

These routines favor readable equations and high-precision verification over
training throughput. Production training uses :mod:`qgdn_rule`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalized(x: torch.Tensor) -> torch.Tensor:
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    return F.normalize(x.to(dtype), dim=-1)


def qgdn_reference(
    q,
    k,
    v,
    g,
    beta,
    gamma,
    *,
    recall_mode="query",
    update_order="recall_then_delta",
    scale=None,
    initial_state=None,
):
    """Apply the differentiable token recurrence in FP32 or FP64."""
    if recall_mode not in {"query", "key", "isotropic"}:
        raise ValueError(recall_mode)
    if update_order not in {"recall_then_delta", "delta_then_recall", "parallel"}:
        raise ValueError(update_order)
    if recall_mode == "isotropic" and update_order != "recall_then_delta":
        raise ValueError("The isotropic control only defines recall_then_delta ordering")
    q, k = _normalized(q), _normalized(k)
    v, g, beta, gamma = (x.to(q.dtype) for x in (v, g, beta, gamma))
    batch, length, heads, key_dim = q.shape
    state = (
        q.new_zeros(batch, heads, key_dim, v.shape[-1])
        if initial_state is None
        else initial_state.to(q.dtype)
    )
    scale = key_dim**-0.5 if scale is None else scale
    outputs = []
    for t in range(length):
        alpha = g[:, t].exp()[..., None, None]
        gamma_t = gamma[:, t, :, None, None]
        eta = gamma_t * (-g[:, t].expm1())[..., None, None]
        recall = q[:, t] if recall_mode == "query" else k[:, t]
        if recall_mode == "isotropic":
            state = (alpha + eta) * state
        else:
            old_read = torch.einsum("bhk,bhkv->bhv", recall, state)
            decayed = alpha * state
            recall_error = old_read - torch.einsum("bhk,bhkv->bhv", recall, decayed)
            delta_error = v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], decayed)
            if update_order == "recall_then_delta":
                state = decayed + gamma_t * recall[..., None] * recall_error[..., None, :]
                delta_error = v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state)
                state = state + beta[:, t, :, None, None] * k[:, t, :, :, None] * delta_error[..., None, :]
            elif update_order == "delta_then_recall":
                state = decayed + beta[:, t, :, None, None] * k[:, t, :, :, None] * delta_error[..., None, :]
                recall_error = old_read - torch.einsum("bhk,bhkv->bhv", recall, state)
                state = state + gamma_t * recall[..., None] * recall_error[..., None, :]
            else:
                state = (
                    decayed
                    + beta[:, t, :, None, None] * k[:, t, :, :, None] * delta_error[..., None, :]
                    + gamma_t * recall[..., None] * recall_error[..., None, :]
                )
        if recall_mode == "isotropic":
            error = v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state)
            state = state + beta[:, t, :, None, None] * k[:, t, :, :, None] * error[..., None, :]
        outputs.append(scale * torch.einsum("bhk,bhkv->bhv", q[:, t], state))
    return torch.stack(outputs, dim=1), state


def qgdn_rank2_factors(
    q, k, g, beta, gamma, *, recall_mode="query", update_order="recall_then_delta"
):
    """Return the exact one-token rank-two affine factors for QGDN."""
    if recall_mode not in {"query", "key"}:
        raise ValueError("rank-two factors support query/key recall only")
    if update_order not in {"recall_then_delta", "delta_then_recall", "parallel"}:
        raise ValueError(update_order)
    qn, kn = _normalized(q), _normalized(k)
    g, beta, gamma = (x.to(qn.dtype) for x in (g, beta, gamma))
    if qn.shape != kn.shape or any(x.shape != qn.shape[:-1] for x in (g, beta, gamma)):
        raise ValueError("incompatible QGDN rank-two factor shapes")

    recall_vector = qn if recall_mode == "query" else kn
    alpha = g.exp()
    recall = gamma * (-g.expm1())
    correlation = (kn * recall_vector).sum(-1)

    left = torch.stack((recall_vector, kn), dim=-2)
    if update_order == "recall_then_delta":
        right = torch.stack(
            (
                recall[..., None] * recall_vector,
                -(alpha * beta)[..., None] * kn
                - (beta * recall * correlation)[..., None] * recall_vector,
            ),
            dim=-2,
        )
        write = beta[..., None] * kn
    elif update_order == "delta_then_recall":
        right = torch.stack(
            (
                recall[..., None] * recall_vector
                + (gamma * alpha * beta * correlation)[..., None] * kn,
                -(alpha * beta)[..., None] * kn,
            ),
            dim=-2,
        )
        write = beta[..., None] * (
            kn - (gamma * correlation)[..., None] * recall_vector
        )
    else:
        right = torch.stack(
            (
                recall[..., None] * recall_vector,
                -(alpha * beta)[..., None] * kn,
            ),
            dim=-2,
        )
        write = beta[..., None] * kn
    return qn, alpha, left, right, write


def qgdn_rank2_reference(
    q,
    k,
    v,
    g,
    beta,
    gamma,
    *,
    recall_mode="query",
    update_order="recall_then_delta",
    scale=None,
    initial_state=None,
):
    """Apply the physical-time rank-two factors as a differentiable oracle."""
    qn, alpha, left, right, write = qgdn_rank2_factors(
        q, k, g, beta, gamma, recall_mode=recall_mode, update_order=update_order
    )
    v = v.to(qn.dtype)
    batch, length, heads, key_dim = qn.shape
    expected = (batch, heads, key_dim, v.shape[-1])
    state = qn.new_zeros(expected) if initial_state is None else initial_state.to(qn.dtype)
    if tuple(state.shape) != expected:
        raise ValueError(f"initial_state must have shape {expected}")
    scale = key_dim**-0.5 if scale is None else scale
    outputs = []
    for t in range(length):
        reads = torch.einsum("bhrk,bhkv->bhrv", right[:, t], state)
        state = alpha[:, t, :, None, None] * state
        state = state + torch.einsum("bhrk,bhrv->bhkv", left[:, t], reads)
        state = state + write[:, t, :, :, None] * v[:, t, :, None, :]
        outputs.append(scale * torch.einsum("bhk,bhkv->bhv", qn[:, t], state))
    return torch.stack(outputs, dim=1), state


def _apply_compact_affine(scale, left, right, bias, state):
    """Apply ``scale * I + left @ right.T`` plus an affine bias."""
    reads = torch.einsum("bhrk,bhkv->bhrv", right, state)
    return (
        scale[..., None, None] * state
        + torch.einsum("bhrk,bhrv->bhkv", left, reads)
        + bias
    )


def _compose_compact_affine(later, earlier):
    """Compose two compact block-WY affine maps without a dense K-by-K matrix.

    Each tuple represents ``scale * state + left @ (right.T @ state) + bias``.
    The returned map is ``later(earlier(state))``.  Rank dimensions concatenate,
    so a chunk of C physical QGDN tokens has rank at most 2C.
    """
    later_scale, later_left, later_right, later_bias = later
    earlier_scale, earlier_left, earlier_right, earlier_bias = earlier
    coupling = torch.einsum(
        "bhrk,bhsk->bhrs", later_right, earlier_left
    )
    earlier_block = (
        later_scale[..., None, None] * earlier_left
        + torch.einsum("bhrk,bhrs->bhsk", later_left, coupling)
    )
    later_block = earlier_scale[..., None, None] * later_left
    left = torch.cat((earlier_block, later_block), dim=-2)
    right = torch.cat((earlier_right, later_right), dim=-2)
    bias_reads = torch.einsum(
        "bhrk,bhkv->bhrv", later_right, earlier_bias
    )
    bias = (
        later_scale[..., None, None] * earlier_bias
        + torch.einsum("bhrk,bhrv->bhkv", later_left, bias_reads)
        + later_bias
    )
    return later_scale * earlier_scale, left, right, bias


def qgdn_rank2_chunk_wy_reference(
    q,
    k,
    v,
    g,
    beta,
    gamma,
    *,
    recall_mode="query",
    update_order="recall_then_delta",
    scale=None,
    initial_state=None,
    chunk_size=16,
):
    """Apply physical-T QGDN through compact rank-two chunk transforms.

    This is the differentiable CPU/FP64 contract for a future parallel CUDA
    kernel.  It keeps one time row per real token and composes each chunk as a
    block-WY-style ``scalar * I + U @ V.T`` affine map.  The Python reference
    deliberately favors transparent algebra over speed; a CUDA implementation
    can evaluate the same associative compositions with a parallel scan.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    qn, alpha, left, right, write = qgdn_rank2_factors(
        q,
        k,
        g,
        beta,
        gamma,
        recall_mode=recall_mode,
        update_order=update_order,
    )
    v = v.to(qn.dtype)
    batch, length, heads, key_dim = qn.shape
    expected = (batch, heads, key_dim, v.shape[-1])
    state = qn.new_zeros(expected) if initial_state is None else initial_state.to(qn.dtype)
    if tuple(state.shape) != expected:
        raise ValueError(f"initial_state must have shape {expected}")
    scale = key_dim**-0.5 if scale is None else scale
    outputs = []

    for chunk_start in range(0, length, chunk_size):
        chunk_state = state
        empty = qn.new_empty(batch, heads, 0, key_dim)
        prefix = (
            qn.new_ones(batch, heads),
            empty,
            empty,
            qn.new_zeros(expected),
        )
        chunk_end = min(chunk_start + chunk_size, length)
        for t in range(chunk_start, chunk_end):
            token_bias = (
                write[:, t, :, :, None] * v[:, t, :, None, :]
            )
            token = (
                alpha[:, t],
                left[:, t],
                right[:, t],
                token_bias,
            )
            prefix = _compose_compact_affine(token, prefix)
            state = _apply_compact_affine(*prefix, chunk_state)
            outputs.append(
                scale * torch.einsum("bhk,bhkv->bhv", qn[:, t], state)
            )
    return torch.stack(outputs, dim=1), state


def qgdn_rank2_chunk_batched_reference(
    q,
    k,
    v,
    g,
    beta,
    gamma,
    *,
    recall_mode="query",
    update_order="recall_then_delta",
    scale=None,
    initial_state=None,
    chunk_size=16,
):
    """Evaluate each physical-T chunk with one block-triangular solve.

    After dividing out the scalar decay prefix, the two reads at every token
    form a unit-lower-triangular block system.  Solving that system exposes all
    intra-chunk reads in parallel; cumulative low-rank updates then recover all
    token states.  This implementation is an autograd-capable CPU/CUDA oracle
    for the eventual fused kernels and never materializes a 2T time axis or a
    dense K-by-K transition.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    qn, alpha, left, right, write = qgdn_rank2_factors(
        q,
        k,
        g,
        beta,
        gamma,
        recall_mode=recall_mode,
        update_order=update_order,
    )
    v = v.to(qn.dtype)
    batch, length, heads, key_dim = qn.shape
    value_dim = v.shape[-1]
    expected = (batch, heads, key_dim, value_dim)
    state = qn.new_zeros(expected) if initial_state is None else initial_state.to(qn.dtype)
    if tuple(state.shape) != expected:
        raise ValueError(f"initial_state must have shape {expected}")
    output_scale = key_dim**-0.5 if scale is None else scale
    outputs = []

    for chunk_start in range(0, length, chunk_size):
        chunk_end = min(chunk_start + chunk_size, length)
        size = chunk_end - chunk_start
        alpha_chunk = alpha[:, chunk_start:chunk_end].permute(0, 2, 1)
        decay_prefix = alpha_chunk.cumprod(dim=-1)
        left_chunk = left[:, chunk_start:chunk_end].permute(0, 2, 1, 3, 4)
        right_chunk = right[:, chunk_start:chunk_end].permute(0, 2, 1, 3, 4)
        normalized_left = left_chunk / alpha_chunk[..., None, None]
        write_chunk = write[:, chunk_start:chunk_end].permute(0, 2, 1, 3)
        normalized_write = write_chunk / decay_prefix[..., None]
        value_chunk = v[:, chunk_start:chunk_end].permute(0, 2, 1, 3)

        # [B,H,t,r,s,u] contains R[t,r]^T U[s,u].  Only s<t belongs
        # to the causal block system; the two same-token ranks are simultaneous.
        coupling = torch.einsum(
            "bhtrk,bhsuk->bhtrsu", right_chunk, normalized_left
        )
        causal = torch.tril(
            torch.ones(size, size, dtype=torch.bool, device=qn.device),
            diagonal=-1,
        )
        coupling = coupling.masked_fill(
            ~causal[None, None, :, None, :, None], 0
        )
        rank = left_chunk.shape[-2]
        system_size = size * rank
        eye = torch.eye(system_size, dtype=qn.dtype, device=qn.device)
        system = eye - coupling.reshape(batch, heads, system_size, system_size)

        initial_reads = torch.einsum(
            "bhtrk,bhkv->bhtrv", right_chunk, state
        )
        write_coupling = torch.einsum(
            "bhtrk,bhsk->bhtrs", right_chunk, normalized_write
        ).masked_fill(~causal[None, None, :, None, :], 0)
        earlier_writes = torch.einsum(
            "bhtrs,bhsv->bhtrv", write_coupling, value_chunk
        )
        rhs = (initial_reads + earlier_writes).reshape(
            batch, heads, system_size, value_dim
        )
        reads = torch.linalg.solve_triangular(
            system, rhs, upper=False, unitriangular=True
        ).reshape(batch, heads, size, rank, value_dim)

        low_rank_updates = torch.einsum(
            "bhtrk,bhtrv->bhtkv", normalized_left, reads
        )
        direct_writes = normalized_write[..., None] * value_chunk[..., None, :]
        normalized_states = state[:, :, None] + (
            low_rank_updates + direct_writes
        ).cumsum(dim=2)
        states = decay_prefix[..., None, None] * normalized_states
        queries = qn[:, chunk_start:chunk_end].permute(0, 2, 1, 3)
        outputs.append(
            output_scale * torch.einsum("bhtk,bhtkv->bhtv", queries, states)
        )
        state = states[:, :, -1]

    return torch.cat(outputs, dim=2).permute(0, 2, 1, 3), state
