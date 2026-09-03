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
    scale=None,
    initial_state=None,
):
    """Apply the differentiable token recurrence in FP32 or FP64."""
    if recall_mode not in {"query", "key", "isotropic"}:
        raise ValueError(recall_mode)
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
        eta = (gamma[:, t] * (-g[:, t].expm1()))[..., None, None]
        recall = q[:, t] if recall_mode == "query" else k[:, t]
        if recall_mode == "isotropic":
            state = (alpha + eta) * state
        else:
            old_read = torch.einsum("bhk,bhkv->bhv", recall, state)
            state = alpha * state + eta * recall[..., None] * old_read[..., None, :]
        error = v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state)
        state = state + beta[:, t, :, None, None] * k[:, t, :, :, None] * error[..., None, :]
        outputs.append(scale * torch.einsum("bhk,bhkv->bhv", q[:, t], state))
    return torch.stack(outputs, dim=1), state


def qgdn_rank2_factors(q, k, g, beta, gamma, *, recall_mode="query"):
    """Return the exact one-token rank-two affine factors for QGDN."""
    if recall_mode not in {"query", "key"}:
        raise ValueError("rank-two factors support query/key recall only")
    qn, kn = _normalized(q), _normalized(k)
    g, beta, gamma = (x.to(qn.dtype) for x in (g, beta, gamma))
    if qn.shape != kn.shape or any(x.shape != qn.shape[:-1] for x in (g, beta, gamma)):
        raise ValueError("incompatible QGDN rank-two factor shapes")

    recall_vector = qn if recall_mode == "query" else kn
    alpha = g.exp()
    recall = gamma * (-g.expm1())
    correlation = (kn * recall_vector).sum(-1)

    left = torch.stack((recall_vector, kn), dim=-2)
    right = torch.stack(
        (
            recall[..., None] * recall_vector,
            -(alpha * beta)[..., None] * kn
            - (beta * recall * correlation)[..., None] * recall_vector,
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
    scale=None,
    initial_state=None,
):
    """Apply the physical-time rank-two factors as a differentiable oracle."""
    qn, alpha, left, right, write = qgdn_rank2_factors(
        q, k, g, beta, gamma, recall_mode=recall_mode
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
