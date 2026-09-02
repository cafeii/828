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


def qr_gdn_rank2_factors(q, k, g_kv, beta_kv, g_qr, beta_qr):
    """Factor the stacked 2K-state transition into two simultaneous updates.

    For Z=[M^KV; M^QR], each physical token has a two-channel diagonal
    decay plus exactly two low-rank corrections. The external value write is
    confined to the KV half. No virtual timesteps are introduced.
    """
    qn, kn = _normalized(q), _normalized(k)
    g_kv, beta_kv, g_qr, beta_qr = (
        x.to(qn.dtype) for x in (g_kv, beta_kv, g_qr, beta_qr)
    )
    if qn.shape != kn.shape or any(
        x.shape != qn.shape[:-1] for x in (g_kv, beta_kv, g_qr, beta_qr)
    ):
        raise ValueError("incompatible QR-GDN factor shapes")
    alpha_kv, alpha_qr = g_kv.exp(), g_qr.exp()
    zero = torch.zeros_like(qn)
    left_kv = torch.cat((kn, zero), dim=-1)
    right_kv = torch.cat((-(alpha_kv * beta_kv)[..., None] * kn, zero), dim=-1)
    left_qr = torch.cat((zero, qn), dim=-1)
    right_qr = torch.cat(
        (
            beta_qr[..., None] * qn,
            -(alpha_qr * beta_qr)[..., None] * qn,
        ),
        dim=-1,
    )
    write = torch.cat((beta_kv[..., None] * kn, zero), dim=-1)
    log_decay = torch.stack((g_kv, g_qr), dim=-1)
    left = torch.stack((left_kv, left_qr), dim=-2)
    right = torch.stack((right_kv, right_qr), dim=-2)
    return qn, kn, log_decay, left, right, write


def qr_gdn_rank2_reference(
    q,
    k,
    v,
    g_kv,
    beta_kv,
    g_qr,
    beta_qr,
    read_logit,
    *,
    scale=None,
    initial_state: Optional[StatePair] = None,
):
    """Token-loop oracle using the stacked-state rank-two factors."""
    qn, _, log_decay, left, right, write = qr_gdn_rank2_factors(
        q, k, g_kv, beta_kv, g_qr, beta_qr
    )
    v, read_logit = v.to(qn.dtype), read_logit.to(qn.dtype)
    B, T, H, K = qn.shape
    expected = (B, H, K, v.shape[-1])
    if initial_state is None:
        state = qn.new_zeros((B, H, 2 * K, v.shape[-1]))
    else:
        if len(initial_state) != 2 or any(tuple(x.shape) != expected for x in initial_state):
            raise ValueError(f"both states must have shape {expected}")
        state = torch.cat(tuple(x.to(qn.dtype) for x in initial_state), dim=-2)
    scale = K**-0.5 if scale is None else scale
    outputs = []
    for t in range(T):
        old_qr = state[..., K:, :]
        reads = torch.einsum("bhrd,bhdv->bhrv", right[:, t], state)
        decay = log_decay[:, t].exp().repeat_interleave(K, dim=-1)
        state = decay[..., None] * state
        state = state + torch.einsum("bhrd,bhrv->bhdv", left[:, t], reads)
        state = state + write[:, t, :, :, None] * v[:, t, :, None, :]
        output = _read(qn[:, t], state[..., :K, :])
        output = output + read_logit[:, t].tanh()[..., None] * _read(qn[:, t], old_qr)
        outputs.append(scale * output)
    return torch.stack(outputs, dim=1), (state[..., :K, :], state[..., K:, :])


def block_wy_rank2_vector_decay(log_decay, left, right, write, v, *, chunk_size: int):
    """Build exact compact chunk transforms for two scalar decay channels.

    The returned (decay, U, Z, offset) represents
      state_out = decay * state_in + U @ (Z^T @ state_in) + offset.
    It keeps T physical tokens and a compact rank of 2C per chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if left.shape != right.shape or left.ndim != 5 or left.shape[-2] != 2:
        raise ValueError("left and right must have shape [B,T,H,2,2K]")
    if log_decay.ndim != 4 or log_decay.shape[-1] != 2:
        raise ValueError("log_decay must have shape [B,T,H,2]")
    if log_decay.shape[:-1] != left.shape[:-2]:
        raise ValueError("decay and rank-two factor dimensions must match")
    if write.shape != left.shape[:-2] + left.shape[-1:]:
        raise ValueError("write must have shape [B,T,H,2K]")
    if v.shape[:-1] != log_decay.shape[:-1]:
        raise ValueError("v must have shape [B,T,H,V]")

    v = v.to(left.dtype)
    B, T, H, rank, D = left.shape
    channels = log_decay.shape[-1]
    if D % channels:
        raise ValueError("stacked state dimension must divide evenly into decay channels")
    if T % chunk_size:
        raise ValueError("sequence length must be divisible by chunk_size")
    K = D // channels
    chunks, C, Vdim = T // chunk_size, chunk_size, v.shape[-1]

    g = log_decay.reshape(B, chunks, C, H, channels).permute(0, 1, 3, 2, 4)
    l = left.reshape(B, chunks, C, H, rank, channels, K).permute(0, 1, 3, 2, 4, 5, 6)
    r = right.reshape(B, chunks, C, H, rank, channels, K).permute(0, 1, 3, 2, 4, 5, 6)
    l = l.reshape(B, chunks, H, C * rank, channels, K)
    r = r.reshape(B, chunks, H, C * rank, channels, K)
    p = write.reshape(B, chunks, C, H, channels, K).permute(0, 1, 3, 2, 4, 5)
    values = v.reshape(B, chunks, C, H, Vdim).permute(0, 1, 3, 2, 4)

    prefix = g.cumsum(-2)
    before = prefix - g
    token = torch.arange(C, device=left.device).repeat_interleave(rank)
    factor_prefix = prefix.index_select(-2, token)
    factor_before = before.index_select(-2, token)

    dot_by_channel = torch.einsum("bnhigk,bnhjgk->bnhijg", r, l)
    factor_decay = (factor_before[..., :, None, :] - factor_prefix[..., None, :, :]).exp()
    factor_mask = token[:, None] > token[None, :]
    interactions = (dot_by_channel * factor_decay).sum(-1)
    system = torch.eye(C * rank, dtype=interactions.dtype, device=interactions.device)
    system = system - interactions * factor_mask

    start_reads = r * factor_before.exp()[..., None]
    start_reads = start_reads.reshape(B, chunks, H, C * rank, D)
    effective_reads = torch.linalg.solve_triangular(
        system, start_reads, upper=False, unitriangular=True
    )

    end_prefix = prefix[..., -1, :]
    final_left = l * (end_prefix[..., None, :, None] - factor_prefix[..., :, :, None]).exp()
    final_left = final_left.reshape(B, chunks, H, C * rank, D)

    read_write_by_channel = torch.einsum("bnhigk,bnhcgk->bnhicg", r, p)
    write_decay = (factor_before[..., :, None, :] - prefix[..., None, :, :]).exp()
    write_mask = token[:, None] > torch.arange(C, device=left.device)[None, :]
    read_write = (read_write_by_channel * write_decay).sum(-1) * write_mask
    write_rhs = torch.einsum("bnhic,bnhcv->bnhiv", read_write, values)
    solved_write = torch.linalg.solve_triangular(
        system, write_rhs, upper=False, unitriangular=True
    )

    offset = torch.einsum("bnhid,bnhiv->bnhdv", final_left, solved_write)
    direct_decay = (end_prefix[..., None, :] - prefix).exp()
    direct_write = (p * direct_decay[..., None]).reshape(B, chunks, H, C, D)
    offset = offset + torch.einsum("bnhcd,bnhcv->bnhdv", direct_write, values)

    decay = end_prefix.exp().repeat_interleave(K, dim=-1)
    u = final_left.transpose(-1, -2).contiguous()
    z = effective_reads.transpose(-1, -2).contiguous()
    return decay, u, z, offset


def apply_vector_decay_chunk(compact, state):
    """Apply one transform returned by ``block_wy_rank2_vector_decay``."""
    decay, u, z, offset = compact
    projected = torch.einsum("bhdm,bhdv->bhmv", z, state)
    return decay[..., None] * state + torch.einsum("bhdm,bhmv->bhdv", u, projected) + offset
