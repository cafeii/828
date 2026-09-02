"""Parallel training path for the two-state QR-GDN recurrence.

The block-lower-triangular dependency is evaluated in causal order: first the
KV channel, then the QR channel whose targets are pre-update KV reads. The KV
call exposes its effective delta values, which lets us recover the pre-update
read algebraically and avoid a second KV state scan. The QR channel uses one
more production FLA GDN chunk operator. No token loop, virtual 2T sequence, or
dense 2K transition appears here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _shift_updates(key, value, log_decay, beta):
    """Move update t to slot t+1 and make slot zero an identity update."""
    key_shift = torch.cat((key[:, :1], key[:, :-1]), dim=1)
    value_shift = torch.cat((torch.zeros_like(value[:, :1]), value[:, :-1]), dim=1)
    decay_shift = torch.cat((torch.zeros_like(log_decay[:, :1]), log_decay[:, :-1]), dim=1)
    beta_shift = torch.cat((torch.zeros_like(beta[:, :1]), beta[:, :-1]), dim=1)
    return key_shift, value_shift, decay_shift, beta_shift


def _read(direction: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhk,bhkv->bhv", direction, state)


def qr_gdn_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_kv: torch.Tensor,
    beta_kv: torch.Tensor,
    g_qr: torch.Tensor,
    beta_qr: torch.Tensor,
    read_logit: torch.Tensor,
    *,
    initial_state=None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    """Evaluate QR-GDN with differentiable production chunk kernels.

    A shifted update stream asks the inclusive GDN operator for the state just
    before the current token. This preserves T physical positions. The normal
    KV call remains byte-for-byte the native GDN call when the QR read gate is
    zero; auxiliary reads use scale 1 and the QR residual is scaled only once.
    """
    if q.ndim != 4 or k.shape != q.shape or q.shape[:-1] != v.shape[:-1]:
        raise ValueError("q, k and v must share [B,T,H] dimensions")
    if q.shape[1] == 0:
        raise ValueError("QR-GDN requires a nonempty sequence")
    if q.shape[1] % chunk_size:
        raise ValueError("sequence length must be divisible by chunk_size")
    expected_gate = q.shape[:-1]
    if any(x.shape != expected_gate for x in (g_kv, beta_kv, g_qr, beta_qr, read_logit)):
        raise ValueError("all gates must have shape [B,T,H]")
    if initial_state is None:
        initial_kv = initial_qr = None
    else:
        if len(initial_state) != 2:
            raise ValueError("initial_state must be an (M_KV, M_QR) pair")
        initial_kv, initial_qr = initial_state

    from ..kernels import get_chunk_gated_delta_rule

    rule = get_chunk_gated_delta_rule()
    common = dict(
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
        chunk_size=chunk_size,
    )
    # This call is intentionally identical to native GDN's production path.
    output_kv, final_kv, kv_update = rule(
        q=q,
        k=k,
        v=v,
        g=g_kv,
        beta=beta_kv,
        initial_state=initial_kv,
        output_final_state=output_final_state,
        output_update=True,
        **common,
    )

    # Native GDN writes M_t = alpha_t M_{t-1} + k_t delta_t^T and returns
    # scale * M_t^T q_t. Its effective delta_t is already materialized by the
    # state scan, so the desired pre-update recall follows without another
    # recurrent scan:
    #   M_{t-1}^T q_t = (M_t^T q_t - <q_t,k_t> delta_t) / alpha_t.
    qn = F.normalize(q.float(), dim=-1)
    kn = F.normalize(k.float(), dim=-1)
    qk = (qn * kn).sum(dim=-1)
    scale = q.shape[-1] ** -0.5
    recall = (output_kv.float() / scale - qk[..., None] * kv_update.float())
    recall = recall / g_kv.float().exp()[..., None]

    q_prev, recall_prev, g_qr_prev, beta_qr_prev = _shift_updates(
        q, recall, g_qr, beta_qr
    )
    qr_read, shifted_final_qr = rule(
        q=q,
        k=q_prev,
        v=recall_prev,
        g=g_qr_prev,
        beta=beta_qr_prev,
        scale=1.0,
        initial_state=initial_qr,
        output_final_state=output_final_state,
        **common,
    )

    output = output_kv + scale * read_logit.tanh()[..., None] * qr_read
    final_state = None
    if output_final_state:
        # The shifted QR scan ends at M_{T-1}; apply the last physical update
        # once to return the true M_T required by continuation and decoding.
        q_last = F.normalize(q[:, -1].float(), dim=-1)
        qr_base = g_qr[:, -1].float().exp()[..., None, None] * shifted_final_qr.float()
        qr_error = recall[:, -1].float() - _read(q_last, qr_base)
        final_qr = qr_base + beta_qr[:, -1].float()[..., None, None] * q_last[..., None] * qr_error[..., None, :]
        final_state = final_kv, final_qr
    return output.to(q.dtype), final_state
