"""FP64-capable token-loop references for QR-GDN.

State tensors use [batch, head, key, value] and reads are ``M^T x``.
These routines are correctness oracles for a future parallel kernel. They are
not training backends.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


StatePair = Tuple[torch.Tensor, torch.Tensor]


def _normalized(x: torch.Tensor) -> torch.Tensor:
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    return F.normalize(x.to(dtype), dim=-1)


def _read(direction: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhk,bhkv->bhv", direction, state)


def _prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_kv: torch.Tensor,
    beta_kv: torch.Tensor,
    g_qr: torch.Tensor,
    beta_qr: torch.Tensor,
    read_logit: torch.Tensor,
    initial_state: Optional[StatePair],
):
    q, k = _normalized(q), _normalized(k)
    v, g_kv, beta_kv, g_qr, beta_qr, read_logit = (
        x.to(q.dtype) for x in (v, g_kv, beta_kv, g_qr, beta_qr, read_logit)
    )
    if q.shape != k.shape or q.shape[:-1] != v.shape[:-1]:
        raise ValueError("q, k and v must share [B,T,H] dimensions")
    gates = (g_kv, beta_kv, g_qr, beta_qr, read_logit)
    if any(x.shape != q.shape[:-1] for x in gates):
        raise ValueError("all gates and read_logit must have shape [B,T,H]")
    B, _, H, K = q.shape
    expected = (B, H, K, v.shape[-1])
    if initial_state is None:
        kv = q.new_zeros(expected)
        qr = q.new_zeros(expected)
    else:
        if len(initial_state) != 2:
            raise ValueError("initial_state must be an (M_KV, M_QR) pair")
        kv, qr = initial_state
        if tuple(kv.shape) != expected or tuple(qr.shape) != expected:
            raise ValueError(f"both states must have shape {expected}")
        kv, qr = kv.to(q.dtype), qr.to(q.dtype)
    return q, k, v, g_kv, beta_kv, g_qr, beta_qr, read_logit, kv, qr


def qr_gdn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_kv: torch.Tensor,
    beta_kv: torch.Tensor,
    g_qr: torch.Tensor,
    beta_qr: torch.Tensor,
    read_logit: torch.Tensor,
    *,
    scale=None,
    initial_state: Optional[StatePair] = None,
):
    """Apply the two coupled delta updates explicitly.

    The native GDN branch keeps its post-update read. The QR residual reads the
    pre-update QR state, so the current token cannot read back the association
    it just wrote. A zero read logit gives exact native-GDN output semantics.
    """
    q, k, v, g_kv, beta_kv, g_qr, beta_qr, read_logit, kv, qr = _prepare(
        q, k, v, g_kv, beta_kv, g_qr, beta_qr, read_logit, initial_state
    )
    K = q.shape[-1]
    scale = K**-0.5 if scale is None else scale
    outputs = []
    for t in range(q.shape[1]):
        qt, kt, vt = q[:, t], k[:, t], v[:, t]
        old_kv, old_qr = kv, qr
        recalled = _read(qt, old_kv)
        qr_read = _read(qt, old_qr)

        # Keep the native GDN operation order so disabling QR is bitwise exact,
        # rather than only algebraically equivalent in floating point.
        kv = g_kv[:, t].exp()[..., None, None] * old_kv
        erased = _read(beta_kv[:, t, :, None] * kt, kv)
        kv = kv - kt[..., None] * erased[..., None, :]
        kv = kv + kt[..., None] * (beta_kv[:, t, :, None] * vt)[..., None, :]

        output = _read(qt, kv) + read_logit[:, t].tanh()[..., None] * qr_read
        outputs.append(scale * output)

        base_qr = g_qr[:, t].exp()[..., None, None] * old_qr
        qr_error = recalled - _read(qt, base_qr)
        qr = base_qr + beta_qr[:, t, :, None, None] * qt[..., None] * qr_error[..., None, :]

    return torch.stack(outputs, dim=1), (kv, qr)


def qr_gdn_affine_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_kv: torch.Tensor,
    beta_kv: torch.Tensor,
    g_qr: torch.Tensor,
    beta_qr: torch.Tensor,
    read_logit: torch.Tensor,
    *,
    scale=None,
    initial_state: Optional[StatePair] = None,
):
    """Apply the equivalent block-lower-triangular affine recurrence."""
    q, k, v, g_kv, beta_kv, g_qr, beta_qr, read_logit, kv, qr = _prepare(
        q, k, v, g_kv, beta_kv, g_qr, beta_qr, read_logit, initial_state
    )
    K = q.shape[-1]
    scale = K**-0.5 if scale is None else scale
    eye = torch.eye(K, dtype=q.dtype, device=q.device)
    outputs = []
    for t in range(q.shape[1]):
        qt, kt, vt = q[:, t], k[:, t], v[:, t]
        old_kv, old_qr = kv, qr
        alpha_kv = g_kv[:, t].exp()
        alpha_qr = g_qr[:, t].exp()
        bkv, bqr = beta_kv[:, t], beta_qr[:, t]

        transition_kv = alpha_kv[..., None, None] * (
            eye - bkv[..., None, None] * kt[..., :, None] * kt[..., None, :]
        )
        transition_qr = alpha_qr[..., None, None] * (
            eye - bqr[..., None, None] * qt[..., :, None] * qt[..., None, :]
        )
        coupling = bqr[..., None, None] * qt[..., :, None] * qt[..., None, :]

        kv = torch.einsum("bhij,bhjv->bhiv", transition_kv, old_kv)
        kv = kv + bkv[..., None, None] * kt[..., None] * vt[..., None, :]
        qr = torch.einsum("bhij,bhjv->bhiv", transition_qr, old_qr)
        qr = qr + torch.einsum("bhij,bhjv->bhiv", coupling, old_kv)

        output = _read(qt, kv) + read_logit[:, t].tanh()[..., None] * _read(qt, old_qr)
        outputs.append(scale * output)

    return torch.stack(outputs, dim=1), (kv, qr)
