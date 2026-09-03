"""QGDN production recurrence, state layout [B,H,K,V].

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


def _normalized(x):
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    return F.normalize(x.to(dtype), dim=-1)


def _interleave(first, second):
    return torch.stack((first, second), dim=2).flatten(1, 2).contiguous()


def _dplr_input_values(q, k, v, g, beta, gamma, recall_from_query):
    qn, kn = _normalized(q), _normalized(k)
    r = qn if recall_from_query else kn
    dtype = q.dtype
    g = g.to(qn.dtype)
    beta, gamma = beta.to(qn.dtype), gamma.to(qn.dtype)
    eta = gamma * (-g.expm1())
    zeros = torch.zeros_like(q)
    return (
        _interleave(zeros, qn.to(dtype)),
        _interleave(zeros, kn.to(dtype)),
        _interleave(torch.zeros_like(v), (v * beta[..., None]).to(dtype)),
        _interleave(r.to(dtype), kn.to(dtype)),
        _interleave((eta[..., None] * r).to(dtype), (-beta[..., None] * kn).to(dtype)),
        _interleave(g[..., None].expand_as(qn), torch.zeros_like(qn)),
    )


_compiled_dplr_input_values = torch.compile(_dplr_input_values, fullgraph=True, dynamic=False)


def dplr_inputs(q, k, v, g, beta, gamma, recall_mode="query", *, compiled=False):
    """Build virtual-step tensors. Public for independent algebra/gradient tests."""
    if recall_mode not in {"query", "key"}:
        raise ValueError(recall_mode)
    builder = _compiled_dplr_input_values if compiled else _dplr_input_values
    values = builder(q, k, v, g, beta, gamma, recall_mode == "query")
    return dict(zip(("q", "k", "v", "a", "b", "gk"), values))


def qgdn_rule(q, k, v, g, beta, gamma, *, recall_mode="query", mode="chunk",
              scale=None, initial_state=None, output_final_state=False, cu_seqlens=None):
    if mode not in {"naive", "chunk", "fused_recurrent"}:
        raise ValueError(mode)
    if recall_mode not in {"query", "key", "isotropic"}:
        raise ValueError(recall_mode)
    if mode == "naive":
        if cu_seqlens is not None:
            raise NotImplementedError("Reference mode takes equal-length batches; test segments separately.")
        from .qgdn_reference import qgdn_reference

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
    inputs = dplr_inputs(
        q, k, v, g, beta, gamma, recall_mode,
        compiled=QGDN_COMPILE_DPLR_INPUTS,
    )
    chunk_kwargs = (
        {
            "chunk_size": QGDN_TRAIN_CHUNK_SIZE,
            "disable_recompute": QGDN_DISABLE_DPLR_RECOMPUTE,
        }
        if mode == "chunk"
        else {}
    )
    o, state = op(**inputs, scale=scale, initial_state=initial_state,
                  output_final_state=output_final_state,
                  cu_seqlens=None if cu_seqlens is None else cu_seqlens * 2,
                  **chunk_kwargs)
    return o[:, 1::2].contiguous(), state
