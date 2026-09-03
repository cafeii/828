"""QGDN reference and GPU implementation, state layout [B,H,K,V].

FLA's DPLR API implements S' = exp(g) S + b (a^T S) + k v^T.
Each real token becomes two virtual steps, in this exact order:
  Recall: g=log(alpha), a=r, b=gamma*(1-alpha)*r, k=v=0.
  Delta:  g=0, a=k, b=-beta*k, write key=k, write value=beta*v.
Only the Delta step produces an output. r=q in the main method.
No inverse alpha, negative-beta GDN trick, or dense K-by-K matrix is used.
This correctness-first chunk backend processes 2T virtual steps; benchmark it.
"""
import torch
import torch.nn.functional as F


def _normalized(x):
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    return F.normalize(x.to(dtype), dim=-1)


def qgdn_reference(q, k, v, g, beta, gamma, *, recall_mode="query", scale=None, initial_state=None):
    """Differentiable token recurrence; at least FP32, FP64 for gradcheck.

    Inputs q,k:[B,T,H,K], v:[B,T,H,V], g,beta,gamma:[B,T,H].
    g<=0 and beta,gamma in [0,1] are caller preconditions.
    """
    if recall_mode not in {"query", "key", "isotropic"}:
        raise ValueError(recall_mode)
    q, k = _normalized(q), _normalized(k)
    v, g, beta, gamma = (x.to(q.dtype) for x in (v, g, beta, gamma))
    B, T, H, K = q.shape
    S = q.new_zeros(B, H, K, v.shape[-1]) if initial_state is None else initial_state.to(q.dtype)
    scale = K ** -0.5 if scale is None else scale
    outputs = []
    for t in range(T):
        alpha = g[:, t].exp()[..., None, None]
        eta = (gamma[:, t] * (-g[:, t].expm1()))[..., None, None]
        r = q[:, t] if recall_mode == "query" else k[:, t]
        if recall_mode == "isotropic":
            S = (alpha + eta) * S
        else:
            old_read = torch.einsum("bhk,bhkv->bhv", r, S)
            S = alpha * S + eta * r[..., None] * old_read[..., None, :]
        error = v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], S)
        S = S + beta[:, t, :, None, None] * k[:, t, :, :, None] * error[..., None, :]
        outputs.append(scale * torch.einsum("bhk,bhkv->bhv", q[:, t], S))
    return torch.stack(outputs, dim=1), S


def qgdn_rank2_factors(q, k, g, beta, gamma, *, recall_mode="query"):
    """Return the exact one-token rank-two affine factors for QGDN.

    For query/key recall, the recurrence is represented at the original T
    physical positions as

      S_t = alpha_t S_{t-1}
            + sum_i left[t,i] (right[t,i]^T S_{t-1})
            + write[t] v_t^T.

    This is a correctness contract for a future fused chunk kernel.  It does
    not expand T to 2T and it does not materialize a K-by-K transition.
    """
    if recall_mode not in {"query", "key"}:
        raise ValueError("rank-two factors support query/key recall only")
    qn, kn = _normalized(q), _normalized(k)
    g, beta, gamma = (x.to(qn.dtype) for x in (g, beta, gamma))
    if qn.shape != kn.shape or any(x.shape != qn.shape[:-1] for x in (g, beta, gamma)):
        raise ValueError("incompatible QGDN rank-two factor shapes")

    r = qn if recall_mode == "query" else kn
    alpha = g.exp()
    recall = gamma * (-g.expm1())
    correlation = (kn * r).sum(-1)

    # (I - beta*k*k^T) (alpha*I + recall*r*r^T)
    # = alpha*I + r*(recall*r)^T
    #   + k*(-alpha*beta*k - beta*recall*(k^T r)*r)^T.
    left = torch.stack((r, kn), dim=-2)
    right = torch.stack(
        (
            recall[..., None] * r,
            -(alpha * beta)[..., None] * kn
            - (beta * recall * correlation)[..., None] * r,
        ),
        dim=-2,
    )
    write = beta[..., None] * kn
    return qn, alpha, left, right, write


def qgdn_rank2_reference(
    q, k, v, g, beta, gamma, *, recall_mode="query", scale=None, initial_state=None
):
    """Apply the physical-T rank-two factors as a differentiable oracle."""
    qn, alpha, left, right, write = qgdn_rank2_factors(
        q, k, g, beta, gamma, recall_mode=recall_mode
    )
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
        outputs.append(scale * torch.einsum("bhk,bhkv->bhv", qn[:, t], state))
    return torch.stack(outputs, dim=1), state


def _interleave(first, second):
    return torch.stack((first, second), dim=2).flatten(1, 2).contiguous()


def dplr_inputs(q, k, v, g, beta, gamma, recall_mode="query"):
    """Build virtual-step tensors. Public for independent algebra/gradient tests."""
    qn, kn = _normalized(q), _normalized(k)
    r = qn if recall_mode == "query" else kn
    dtype = q.dtype
    g = g.to(qn.dtype)
    beta, gamma = beta.to(qn.dtype), gamma.to(qn.dtype)
    eta = gamma * (-g.expm1())
    zeros = torch.zeros_like(q)
    return dict(
        q=_interleave(zeros, qn.to(dtype)),
        k=_interleave(zeros, kn.to(dtype)),
        v=_interleave(torch.zeros_like(v), (v * beta[..., None]).to(dtype)),
        a=_interleave(r.to(dtype), kn.to(dtype)),
        b=_interleave((eta[..., None] * r).to(dtype), (-beta[..., None] * kn).to(dtype)),
        gk=_interleave(g[..., None].expand_as(qn), torch.zeros_like(qn)),
    )


def qgdn_rule(q, k, v, g, beta, gamma, *, recall_mode="query", mode="chunk",
              scale=None, initial_state=None, output_final_state=False, cu_seqlens=None):
    if mode not in {"naive", "chunk", "fused_recurrent"}:
        raise ValueError(mode)
    if recall_mode not in {"query", "key", "isotropic"}:
        raise ValueError(recall_mode)
    if mode == "naive":
        if cu_seqlens is not None:
            raise NotImplementedError("Reference mode takes equal-length batches; test segments separately.")
        o, state = qgdn_reference(q, k, v, g, beta, gamma, recall_mode=recall_mode,
                                 scale=scale, initial_state=initial_state)
        return o.to(q.dtype), state if output_final_state else None
    if not q.is_cuda:
        raise ValueError("QGDN chunk/recurrent kernels require CUDA; use mode='naive' for reference tests.")
    if mode == "fused_recurrent" and torch.is_grad_enabled() and any(
        x.requires_grad for x in (q, k, v, g, beta, gamma)
    ):
        raise ValueError("fused_recurrent is inference-only; use chunk for gradients.")
    if recall_mode == "isotropic":
        # Same trainable gamma, but increases retention in EVERY direction.
        # This control separates query guidance from simply forgetting less.
        from ..kernels import get_chunk_gated_delta_rule, get_fused_recurrent_gated_delta_rule
        # log(alpha + gamma*(1-alpha)), stable without exp(-g).
        effective_alpha = g.float().exp() + gamma.float() * (-g.float().expm1())
        effective_g = effective_alpha.clamp_min(torch.finfo(torch.float32).tiny).log()
        op = get_chunk_gated_delta_rule() if mode == "chunk" else get_fused_recurrent_gated_delta_rule()
        return op(q=q, k=k, v=v, g=effective_g, beta=beta, scale=scale,
                  initial_state=initial_state, output_final_state=output_final_state,
                  use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens)
    from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule, fused_recurrent_dplr_delta_rule
    op = chunk_dplr_delta_rule if mode == "chunk" else fused_recurrent_dplr_delta_rule
    inputs = dplr_inputs(q, k, v, g, beta, gamma, recall_mode)
    o, state = op(**inputs, scale=scale, initial_state=initial_state,
                  output_final_state=output_final_state,
                  cu_seqlens=None if cu_seqlens is None else cu_seqlens * 2)
    return o[:, 1::2].contiguous(), state
