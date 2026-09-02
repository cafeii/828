"""Reference recurrences for DT-GDN and JQC-GDN.

State tensors use [batch, head, key, value] and reads are ``S^T q``.
The routines in this file intentionally use an explicit token loop. They are
correctness oracles for parallel training kernels, never training backends.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalized(x: torch.Tensor) -> torch.Tensor:
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    return F.normalize(x.to(dtype), dim=-1)


def _prepare(q, k, v, g, beta, gamma, initial_state):
    q, k = _normalized(q), _normalized(k)
    v, g, beta, gamma = (x.to(q.dtype) for x in (v, g, beta, gamma))
    if q.shape != k.shape or q.shape[:-1] != v.shape[:-1]:
        raise ValueError("q, k and v must share [B,T,H] dimensions")
    if any(x.shape != q.shape[:-1] for x in (g, beta, gamma)):
        raise ValueError("g, beta and gamma must have shape [B,T,H]")
    B, _, H, K = q.shape
    expected = (B, H, K, v.shape[-1])
    if initial_state is None:
        state = q.new_zeros(expected)
    else:
        if tuple(initial_state.shape) != expected:
            raise ValueError(f"initial_state must have shape {expected}")
        state = initial_state.to(q.dtype)
    return q, k, v, g, beta, gamma, state


def _read(direction: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhk,bhkv->bhv", direction, state)


def dt_gdn_reference(q, k, v, g, beta, gamma, *, scale=None, initial_state=None):
    """Symmetric dual-target proximal update using a stable 2x2 solve."""
    q, k, v, g, beta, gamma, state = _prepare(q, k, v, g, beta, gamma, initial_state)
    K = q.shape[-1]
    scale = K**-0.5 if scale is None else scale
    outputs = []
    for t in range(q.shape[1]):
        qt, kt, vt = q[:, t], k[:, t], v[:, t]
        alpha, bt, gt = g[:, t].exp(), beta[:, t], gamma[:, t]
        correlation = (kt * qt).sum(-1)
        determinant = 1 - bt * gt * correlation.square()
        if torch.any(determinant <= 0):
            raise FloatingPointError("DT-GDN 2x2 system is singular")
        old_q_read = _read(qt, state)
        base = alpha[..., None, None] * state
        residual_k = vt - _read(kt, base)
        residual_q = old_q_read - _read(qt, base)
        c00 = bt / determinant
        c01 = -(bt * gt * correlation) / determinant
        c11 = gt / determinant
        coeff_k = c00[..., None] * residual_k + c01[..., None] * residual_q
        coeff_q = c01[..., None] * residual_k + c11[..., None] * residual_q
        state = base + kt[..., None] * coeff_k[..., None, :] + qt[..., None] * coeff_q[..., None, :]
        outputs.append(scale * _read(qt, state))
    return torch.stack(outputs, dim=1), state


def dt_gdn_affine_reference(q, k, v, g, beta, gamma, *, scale=None, initial_state=None):
    """The same DT-GDN update as ``S_t=A_t S_{t-1}+p_t v_t^T``."""
    q, k, v, g, beta, gamma, state = _prepare(q, k, v, g, beta, gamma, initial_state)
    K = q.shape[-1]
    scale = K**-0.5 if scale is None else scale
    eye = torch.eye(K, dtype=q.dtype, device=q.device)
    outputs = []
    for t in range(q.shape[1]):
        qt, kt, vt = q[:, t], k[:, t], v[:, t]
        alpha, bt, gt = g[:, t].exp(), beta[:, t], gamma[:, t]
        correlation = (kt * qt).sum(-1)
        determinant = 1 - bt * gt * correlation.square()
        if torch.any(determinant <= 0):
            raise FloatingPointError("DT-GDN 2x2 system is singular")
        c00 = bt / determinant
        c01 = -(bt * gt * correlation) / determinant
        c11 = gt / determinant
        uc0 = c00[..., None] * kt + c01[..., None] * qt
        uc1 = c01[..., None] * kt + c11[..., None] * qt
        row0 = -alpha[..., None] * kt
        row1 = (1 - alpha)[..., None] * qt
        transition = alpha[..., None, None] * eye + uc0[..., :, None] * row0[..., None, :] + uc1[..., :, None] * row1[..., None, :]
        state = torch.einsum("bhij,bhjv->bhiv", transition, state) + uc0[..., None] * vt[..., None, :]
        outputs.append(scale * _read(qt, state))
    return torch.stack(outputs, dim=1), state


def jqc_gdn_reference(q, k, v, g, beta, gamma, *, scale=None, initial_state=None):
    """Native GDN write followed by query-addressed consolidation."""
    q, k, v, g, beta, gamma, state = _prepare(q, k, v, g, beta, gamma, initial_state)
    K = q.shape[-1]
    scale = K**-0.5 if scale is None else scale
    outputs = []
    for t in range(q.shape[1]):
        qt, kt, vt = q[:, t], k[:, t], v[:, t]
        alpha, bt, gt = g[:, t].exp(), beta[:, t], gamma[:, t]
        old_q_read = _read(qt, state)
        decayed = alpha[..., None, None] * state
        candidate = decayed + bt[..., None, None] * kt[..., None] * (vt - _read(kt, decayed))[..., None, :]
        state = candidate + gt[..., None, None] * qt[..., None] * (old_q_read - _read(qt, candidate))[..., None, :]
        outputs.append(scale * _read(qt, state))
    return torch.stack(outputs, dim=1), state


def jqc_gdn_affine_reference(q, k, v, g, beta, gamma, *, scale=None, initial_state=None):
    """The same JQC-GDN update as an affine rank-two recurrence."""
    q, k, v, g, beta, gamma, state = _prepare(q, k, v, g, beta, gamma, initial_state)
    K = q.shape[-1]
    scale = K**-0.5 if scale is None else scale
    eye = torch.eye(K, dtype=q.dtype, device=q.device)
    outputs = []
    for t in range(q.shape[1]):
        qt, kt, vt = q[:, t], k[:, t], v[:, t]
        alpha, bt, gt = g[:, t].exp(), beta[:, t], gamma[:, t]
        native = alpha[..., None, None] * (eye - bt[..., None, None] * kt[..., :, None] * kt[..., None, :])
        q_native = torch.einsum("bhk,bhkj->bhj", qt, native)
        transition = native + gt[..., None, None] * qt[..., :, None] * (qt - q_native)[..., None, :]
        correlation = (qt * kt).sum(-1)
        write = bt[..., None] * (kt - gt[..., None] * correlation[..., None] * qt)
        state = torch.einsum("bhij,bhjv->bhiv", transition, state) + write[..., None] * vt[..., None, :]
        outputs.append(scale * _read(qt, state))
    return torch.stack(outputs, dim=1), state
