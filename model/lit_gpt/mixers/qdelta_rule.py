"""Paper Q-Delta and its sign ablation on the generalized DPLR kernel.

The paper uses a value-first state and

    S_t = alpha S_{t-1}(I - beta (k + lambda q) k^T) + beta v k^T.

Our kernels store the transposed key-first state, so one physical token is
exactly one DPLR row:

    S_t = alpha S_{t-1} - alpha beta k ((k + lambda q)^T S_{t-1})
          + beta k v^T.

The paired sign ablation replaces ``k + lambda q`` with ``k - lambda q``.
Both are rank one and do not use QGDN's virtual-2T construction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


QDELTA_CHUNK_SIZE = 32
QDELTA_COMPILE_INPUTS = True


def _normalized(x: torch.Tensor) -> torch.Tensor:
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    return F.normalize(x.to(dtype), dim=-1)


def _dplr_input_values(q, k, v, g, beta, query_feedback, query_sign):
    qn, kn = _normalized(q), _normalized(k)
    work = qn.dtype
    alpha = g.to(work).exp()
    beta = beta.to(work)
    query_feedback = query_feedback.to(work)
    mixed = kn + query_sign * query_feedback[..., None] * qn
    return (
        qn.to(q.dtype),
        kn.to(q.dtype),
        (beta[..., None] * v.to(work)).to(v.dtype),
        mixed.to(q.dtype),
        (-alpha * beta)[..., None].mul(kn).to(q.dtype),
        g.to(work)[..., None].expand_as(qn),
    )


_compiled_dplr_input_values = torch.compile(_dplr_input_values, fullgraph=True, dynamic=True)


def dplr_inputs(q, k, v, g, beta, query_feedback, *, query_sign=1.0, compiled=False):
    builder = _compiled_dplr_input_values if compiled else _dplr_input_values
    values = builder(q, k, v, g, beta, query_feedback, query_sign)
    return dict(zip(("q", "k", "v", "a", "b", "gk"), values))


def qdelta_reference(
    q, k, v, g, beta, query_feedback, *, query_sign=1.0, scale=None, initial_state=None
):
    """Dense FP64/FP32 reference with state layout ``[B,H,K,V]``."""
    original_dtype = q.dtype
    qn, kn = _normalized(q), _normalized(k)
    work = qn.dtype
    v, g = v.to(work), g.to(work)
    beta, query_feedback = beta.to(work), query_feedback.to(work)
    B, T, H, K = qn.shape
    V = v.shape[-1]
    state = qn.new_zeros(B, H, K, V)
    if initial_state is not None:
        state = state + initial_state.to(work)
    scale = K**-0.5 if scale is None else scale
    outputs = []
    for index in range(T):
        qt, kt, vt = qn[:, index], kn[:, index], v[:, index]
        alpha, strength = g[:, index].exp(), beta[:, index]
        mixed = kt + query_sign * query_feedback[:, index, :, None] * qt
        prediction = torch.einsum("bhk,bhkv->bhv", mixed, state)
        error = vt - alpha[..., None] * prediction
        state = alpha[..., None, None] * state + strength[..., None, None] * torch.einsum(
            "bhk,bhv->bhkv", kt, error
        )
        outputs.append(scale * torch.einsum("bhk,bhkv->bhv", qt, state))
    output = torch.stack(outputs, dim=1)
    return output.to(original_dtype), state


def qdelta_rule(
    q, k, v, g, beta, query_feedback, *, query_sign=1.0, mode="chunk", scale=None,
    initial_state=None, output_final_state=False, cu_seqlens=None,
):
    if query_sign not in {-1.0, 1.0}:
        raise ValueError("query_sign must be +1 or -1")
    if mode not in {"naive", "chunk", "fused_recurrent"}:
        raise ValueError(mode)
    if mode == "naive":
        if cu_seqlens is not None:
            raise NotImplementedError("Reference mode takes equal-length batches")
        output, state = qdelta_reference(
            q, k, v, g, beta, query_feedback, query_sign=query_sign,
            scale=scale, initial_state=initial_state
        )
        return output, state if output_final_state else None
    if not q.is_cuda:
        raise ValueError("Q-Delta chunk/recurrent kernels require CUDA; use mode='naive' on CPU")
    if mode == "fused_recurrent" and torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (q, k, v, g, beta, query_feedback)
    ):
        raise ValueError("fused_recurrent is inference-only; use chunk for gradients")
    from fla.ops.generalized_delta_rule.dplr import (
        chunk_dplr_delta_rule,
        fused_recurrent_dplr_delta_rule,
    )

    op = chunk_dplr_delta_rule if mode == "chunk" else fused_recurrent_dplr_delta_rule
    inputs = dplr_inputs(
        q, k, v, g, beta, query_feedback,
        query_sign=query_sign,
        compiled=mode == "chunk" and QDELTA_COMPILE_INPUTS,
    )
    kwargs = {"chunk_size": QDELTA_CHUNK_SIZE} if mode == "chunk" else {}
    output, state = op(
        **inputs, scale=scale, initial_state=initial_state,
        output_final_state=output_final_state, cu_seqlens=cu_seqlens, **kwargs,
    )
    return output, state
