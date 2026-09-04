"""QGDN production recurrence, state layout [B,H,K,V].

FLA's DPLR API implements S' = exp(g) S + b (a^T S) + k v^T.
Each real token becomes two virtual DPLR rows.  The default rows are the
literal Recall-then-Delta operations; the reverse and parallel variants use
exact rank-two factorizations of their physical-time affine transitions.
Only the second row produces an output. r=q in the main method.
No inverse alpha, negative-beta GDN trick, or dense K-by-K matrix is used.
This correctness-first chunk backend processes 2T virtual steps; benchmark it.
"""
import torch
import torch.nn.functional as F


# The exact same generalized-DPLR recurrence is faster at 32 DPLR kernel
# rows on H800.  QGDN still presents 2T virtual rows to this backend, so this
# corresponds to 16 real tokens per DPLR chunk.  Keep the value explicit so
# speed and numerical regressions cannot silently follow an upstream default.
QGDN_TRAIN_CHUNK_SIZE = 32
# Fuse normalization, gate arithmetic, zero filling, and virtual-row packing.
# This preserves the exact 2T recurrence while avoiding a long eager op chain
# in every QGDN layer.  H800 340M/4096 benchmarks validate the compiled graph.
QGDN_COMPILE_DPLR_INPUTS = True
QGDN_DISABLE_DPLR_RECOMPUTE = False
QGDN_USE_PHYSICAL_T = False
# The fused physical-token implementation specializes one exact rank-two WY
# map per 16 real tokens.  Keep this independent of the virtual 2T DPLR row
# count above: changing one backend must not silently retune the other.
QGDN_PHYSICAL_T_CHUNK_SIZE = 16
# The state/output custom backward reconstructs chunk-start states with the
# compact state kernel instead of retaining one [B,H,N,K,V] tape per layer.
QGDN_RECOMPUTE_PHYSICAL_T_CHUNK_STARTS = True
# Optional audited lower bound for the physical log-decay.  Supplying it lets
# FLA select its tensor-core DPLR backend.  The production default remains
# unset until the candidate has passed the numerical and throughput gates.
QGDN_DPLR_LOWER_BOUND = None


def _normalized(x):
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    return F.normalize(x.to(dtype), dim=-1)


def _interleave(first, second):
    return torch.stack((first, second), dim=2).flatten(1, 2).contiguous()


UPDATE_ORDERS = ("recall_then_delta", "delta_then_recall", "parallel")


def _dplr_input_values(q, k, v, g, beta, gamma, recall_from_query, update_order):
    qn, kn = _normalized(q), _normalized(k)
    r = qn if recall_from_query else kn
    dtype = q.dtype
    g = g.to(qn.dtype)
    beta, gamma = beta.to(qn.dtype), gamma.to(qn.dtype)
    alpha = g.exp()
    lost = -g.expm1()
    eta = gamma * lost
    correlation = (kn * r).sum(-1)
    tiny = torch.finfo(alpha.dtype).tiny

    if update_order == "recall_then_delta":
        first_a = r
        first_b = eta[..., None] * r
        second_a = kn
        second_b = -beta[..., None] * kn
        write_key = kn
    elif update_order == "parallel":
        # Factor P - alpha*beta*k*k^T as
        # (I - beta*k*a2^T) P, where P=alpha*I+eta*r*r^T and a2^T P=alpha*k^T.
        denominator = (alpha + eta).clamp_min(tiny)
        second_a = kn - (eta * correlation / denominator)[..., None] * r
        first_a = r
        first_b = eta[..., None] * r
        second_b = -beta[..., None] * kn
        write_key = kn
    elif update_order == "delta_then_recall":
        # Exact two-row factorization of Delta-then-Recall.  P is chosen so
        # that its Sherman-Morrison denominator is strictly positive for the
        # bounded gates; no state-transition inverse is formed.
        z = lost[..., None] * r + (alpha * beta * correlation)[..., None] * kn
        rtz = lost + alpha * beta * correlation.square()
        denominator = (alpha + gamma * rtz).clamp_min(tiny)
        second_a = kn - (gamma * correlation / denominator)[..., None] * z
        first_a = z
        first_b = gamma[..., None] * r
        second_b = -beta[..., None] * kn
        write_key = kn - (gamma * correlation)[..., None] * r
    else:
        raise ValueError(update_order)

    zeros = torch.zeros_like(q)
    return (
        _interleave(zeros, qn.to(dtype)),
        _interleave(zeros, write_key.to(dtype)),
        _interleave(torch.zeros_like(v), (v * beta[..., None]).to(dtype)),
        _interleave(first_a.to(dtype), second_a.to(dtype)),
        _interleave(first_b.to(dtype), second_b.to(dtype)),
        _interleave(g[..., None].expand_as(qn), torch.zeros_like(qn)),
    )


# One graph per semantic branch, with a dynamic token dimension.  Training
# still uses a fixed 4096-token shape, while the validation suite can cover
# several lengths without exhausting Dynamo's global recompile cache.
_compiled_dplr_input_values = torch.compile(_dplr_input_values, fullgraph=True, dynamic=True)


def dplr_inputs(
    q, k, v, g, beta, gamma, recall_mode="query", update_order="recall_then_delta", *, compiled=False
):
    """Build virtual-step tensors. Public for independent algebra/gradient tests."""
    if recall_mode not in {"query", "key"}:
        raise ValueError(recall_mode)
    if update_order not in UPDATE_ORDERS:
        raise ValueError(update_order)
    builder = _compiled_dplr_input_values if compiled else _dplr_input_values
    values = builder(q, k, v, g, beta, gamma, recall_mode == "query", update_order)
    return dict(zip(("q", "k", "v", "a", "b", "gk"), values))


def qgdn_rule(q, k, v, g, beta, gamma, *, recall_mode="query",
              update_order="recall_then_delta", mode="chunk",
              scale=None, initial_state=None, output_final_state=False, cu_seqlens=None):
    if mode not in {"naive", "chunk", "fused_recurrent"}:
        raise ValueError(mode)
    if recall_mode not in {"query", "key", "isotropic"}:
        raise ValueError(recall_mode)
    if update_order not in UPDATE_ORDERS:
        raise ValueError(update_order)
    if recall_mode == "isotropic" and update_order != "recall_then_delta":
        raise ValueError("The isotropic control only defines recall_then_delta ordering")
    if mode == "naive":
        if cu_seqlens is not None:
            raise NotImplementedError("Reference mode takes equal-length batches; test segments separately.")
        from .qgdn_reference import qgdn_reference

        o, state = qgdn_reference(q, k, v, g, beta, gamma, recall_mode=recall_mode,
                                 update_order=update_order, scale=scale, initial_state=initial_state)
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
    if (
        mode == "chunk"
        and QGDN_USE_PHYSICAL_T
        and recall_mode in {"query", "key"}
        and cu_seqlens is None
    ):
        # Exact physical-token path: every chunk's rank-two WY map is prepared
        # in parallel, then fused Triton kernels scan only the compact chunk
        # states and recover all within-chunk outputs.  This replaces the old
        # token-serial physical kernel; it never constructs virtual 2T rows.
        from .qgdn_reference import qgdn_rank2_parallel_wy_reference

        output, state = qgdn_rank2_parallel_wy_reference(
            q,
            k,
            v,
            g,
            beta,
            gamma,
            recall_mode=recall_mode,
            update_order=update_order,
            scale=scale,
            initial_state=initial_state,
            chunk_size=QGDN_PHYSICAL_T_CHUNK_SIZE,
            wy_backend="triton",
            state_backend="triton",
        )
        return output.to(q.dtype), state if output_final_state else None
    from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule, fused_recurrent_dplr_delta_rule
    op = chunk_dplr_delta_rule if mode == "chunk" else fused_recurrent_dplr_delta_rule
    inputs = dplr_inputs(
        q, k, v, g, beta, gamma, recall_mode, update_order,
        compiled=QGDN_COMPILE_DPLR_INPUTS,
    )
    chunk_kwargs = (
        {
            "chunk_size": QGDN_TRAIN_CHUNK_SIZE,
            "disable_recompute": QGDN_DISABLE_DPLR_RECOMPUTE,
            "lower_bound": QGDN_DPLR_LOWER_BOUND,
        }
        if mode == "chunk"
        else {}
    )
    if mode == "chunk" and QGDN_DPLR_LOWER_BOUND is not None:
        # Keep the optimized backend's numerical contract executable.  This
        # is a device-side assertion rather than a clamp, so recurrence values
        # and gradients are unchanged and a violated contract fails loudly.
        torch._assert_async(
            (g >= QGDN_DPLR_LOWER_BOUND).all(),
            "QGDN log-decay is below the audited DPLR lower bound",
        )
    o, state = op(**inputs, scale=scale, initial_state=initial_state,
                  output_final_state=output_final_state,
                  cu_seqlens=None if cu_seqlens is None else cu_seqlens * 2,
                  **chunk_kwargs)
    return o[:, 1::2].contiguous(), state
