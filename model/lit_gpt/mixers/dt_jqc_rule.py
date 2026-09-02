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


def rank2_factors(q, k, g, beta, gamma, *, method: str):
    """Return canonical rank-two factors without expanding the time axis.

    The recurrence is
      S_t = alpha_t S_{t-1}
            + sum_r left[t,r] (right[t,r]^T S_{t-1})
            + write[t] v_t^T,
    with an explicit rank dimension of size two and the original T tokens.
    """
    if method not in {"dt", "jqc"}:
        raise ValueError(f"unknown method: {method}")
    qn, kn = _normalized(q), _normalized(k)
    g, beta, gamma = (x.to(qn.dtype) for x in (g, beta, gamma))
    if qn.shape != kn.shape or any(x.shape != qn.shape[:-1] for x in (g, beta, gamma)):
        raise ValueError("incompatible rank-two factor shapes")
    alpha = g.exp()
    correlation = (qn * kn).sum(-1)
    if method == "dt":
        determinant = 1 - beta * gamma * correlation.square()
        if torch.any(determinant <= 0):
            raise FloatingPointError("DT-GDN 2x2 system is singular")
        c00 = beta / determinant
        c01 = -(beta * gamma * correlation) / determinant
        c11 = gamma / determinant
        left0 = c00[..., None] * kn + c01[..., None] * qn
        left1 = c01[..., None] * kn + c11[..., None] * qn
        right0 = -alpha[..., None] * kn
        right1 = (1 - alpha)[..., None] * qn
        write = left0
    else:
        left0 = kn
        right0 = -(alpha * beta)[..., None] * kn
        left1 = qn
        right1 = gamma[..., None] * (
            (1 - alpha)[..., None] * qn
            + (alpha * beta * correlation)[..., None] * kn
        )
        write = beta[..., None] * (kn - (gamma * correlation)[..., None] * qn)
    return qn, kn, alpha, torch.stack((left0, left1), dim=-2), torch.stack((right0, right1), dim=-2), write


def rank2_factor_reference(q, k, v, g, beta, gamma, *, method: str, scale=None, initial_state=None):
    """Apply the common rank-two factors as a token-loop correctness oracle."""
    qn, _, alpha, left, right, write = rank2_factors(q, k, g, beta, gamma, method=method)
    v = v.to(qn.dtype)
    B, T, H, K = qn.shape
    expected = (B, H, K, v.shape[-1])
    state = qn.new_zeros(expected) if initial_state is None else initial_state.to(qn.dtype)
    if tuple(state.shape) != expected:
        raise ValueError(f"initial_state must have shape {expected}")
    scale = K**-0.5 if scale is None else scale
    outputs = []
    for t in range(T):
        reads = torch.einsum("bhrk,bhkv->bhrv", right[:, t], state)
        state = alpha[:, t, :, None, None] * state
        state = state + torch.einsum("bhrk,bhrv->bhkv", left[:, t], reads)
        state = state + write[:, t, :, :, None] * v[:, t, :, None, :]
        outputs.append(scale * _read(qn[:, t], state))
    return torch.stack(outputs, dim=1), state


def compact_rank2_chunk_reference(alpha, left, right, write, v):
    """Build an exact scalar-plus-low-rank transform for one physical chunk.

    This is a differentiable CPU oracle for the production WY/Triton kernel.
    It keeps C physical timesteps and appends exactly two factor columns per
    timestep, so the compact rank is 2C without materializing KxK transitions
    or expanding the external time axis to 2C.

    The returned tuple (a, U, Z, E) represents
      S_out = a S_in + U (Z^T S_in) + E.
    """
    if left.shape != right.shape or left.ndim != 5 or left.shape[-2] != 2:
        raise ValueError("left and right must have shape [B,C,H,2,K]")
    if alpha.shape != left.shape[:-2] or write.shape != left.shape[:-2] + left.shape[-1:]:
        raise ValueError("incompatible alpha or write shape")
    if v.shape[:-1] != alpha.shape:
        raise ValueError("v must have shape [B,C,H,V]")

    B, C, H, _, K = left.shape
    Vdim = v.shape[-1]
    scalar = alpha.new_ones((B, H))
    u = left.new_empty((B, H, K, 0))
    z = right.new_empty((B, H, K, 0))
    offset = v.new_zeros((B, H, K, Vdim))

    for t in range(C):
        lt = left[:, t].transpose(-1, -2)
        rt = right[:, t].transpose(-1, -2)
        at = alpha[:, t]

        offset_reads = torch.einsum("bhkr,bhkv->bhrv", rt, offset)
        offset = at[..., None, None] * offset
        offset = offset + torch.einsum("bhkr,bhrv->bhkv", lt, offset_reads)
        offset = offset + write[:, t, :, :, None] * v[:, t, :, None, :]

        if u.shape[-1] == 0:
            appended_z = scalar[..., None, None] * rt
        else:
            interaction = torch.einsum("bhkm,bhkr->bhmr", u, rt)
            appended_z = scalar[..., None, None] * rt
            appended_z = appended_z + torch.einsum("bhkm,bhmr->bhkr", z, interaction)
        u = torch.cat((at[..., None, None] * u, lt), dim=-1)
        z = torch.cat((z, appended_z), dim=-1)
        scalar = at * scalar

    return scalar, u, z, offset


def apply_compact_chunk_reference(compact, state):
    """Apply a compact chunk transform returned above to an input state."""
    scalar, u, z, offset = compact
    if state.shape != offset.shape:
        raise ValueError("state and compact offset must have the same shape")
    projected = torch.einsum("bhkm,bhkv->bhmv", z, state)
    return scalar[..., None, None] * state + torch.einsum("bhkm,bhmv->bhkv", u, projected) + offset


def rank2_chunked_final_state_reference(
    q, k, v, g, beta, gamma, *, method: str, chunk_size: int, initial_state=None
):
    """Apply exact compact rank-two chunk transforms sequentially.

    The Python chunk loop is test-only. The production kernel will construct
    and apply the same compact representation on device.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    qn, _, alpha, left, right, write = rank2_factors(q, k, g, beta, gamma, method=method)
    v = v.to(qn.dtype)
    B, T, H, K = qn.shape
    expected = (B, H, K, v.shape[-1])
    state = qn.new_zeros(expected) if initial_state is None else initial_state.to(qn.dtype)
    if tuple(state.shape) != expected:
        raise ValueError(f"initial_state must have shape {expected}")
    ranks = []
    for start in range(0, T, chunk_size):
        stop = min(start + chunk_size, T)
        compact = compact_rank2_chunk_reference(
            alpha[:, start:stop],
            left[:, start:stop],
            right[:, start:stop],
            write[:, start:stop],
            v[:, start:stop],
        )
        ranks.append(compact[1].shape[-1])
        state = apply_compact_chunk_reference(compact, state)
    return state, ranks


def dense_affine_elements(q, k, v, g, beta, gamma, *, method: str):
    """Materialize dense affine elements only for small scan-algebra tests."""
    qn, _, alpha, left, right, write = rank2_factors(q, k, g, beta, gamma, method=method)
    v = v.to(qn.dtype)
    K = qn.shape[-1]
    eye = torch.eye(K, dtype=qn.dtype, device=qn.device)
    transition = alpha[..., None, None] * eye
    transition = transition + torch.einsum("bthrk,bthrj->bthkj", left, right)
    offset = write[..., :, None] * v[..., None, :]
    return qn, transition, offset


def compose_affine(after_transition, after_offset, before_transition, before_offset):
    """Compose ``after(before(S))``; this binary operator is associative."""
    transition = after_transition @ before_transition
    offset = after_transition @ before_offset + after_offset
    return transition, offset


def dense_affine_scan_reference(transition, offset, initial_state):
    """Inclusive affine scan oracle for small tensors."""
    prefix_transition = torch.eye(
        transition.shape[-1], dtype=transition.dtype, device=transition.device
    ).expand(*transition.shape[:1], *transition.shape[2:3], transition.shape[-1], transition.shape[-1])
    prefix_offset = torch.zeros_like(offset[:, 0])
    states = []
    for t in range(transition.shape[1]):
        prefix_transition, prefix_offset = compose_affine(
            transition[:, t], offset[:, t], prefix_transition, prefix_offset
        )
        states.append(prefix_transition @ initial_state + prefix_offset)
    return torch.stack(states, dim=1)
