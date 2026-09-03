"""Physical-token fused QGDN recurrence for the query Recall variants.

The CUDA kernels keep one real token per loop iteration and apply both low-rank
corrections to one FP32 state tile.  No zero rows, 2T queries/values, or unused
Recall-position outputs are materialized.  Backward reconstructs states inside
short chunks from exact FP32 chunk-end checkpoints, then differentiates the
literal update equations in reverse order.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


ORDER_RECALL_THEN_DELTA = tl.constexpr(0)
ORDER_DELTA_THEN_RECALL = tl.constexpr(1)
ORDER_PARALLEL = tl.constexpr(2)
ORDER_IDS = {
    "recall_then_delta": ORDER_RECALL_THEN_DELTA,
    "delta_then_recall": ORDER_DELTA_THEN_RECALL,
    "parallel": ORDER_PARALLEL,
}


@triton.jit(do_not_specialize=["T"])
def _qgdn_physical_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    gamma,
    initial_state,
    output,
    chunk_ends,
    final_state,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHUNK: tl.constexpr,
    ORDER: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    STORE_FINAL: tl.constexpr,
):
    i_bh = tl.program_id(0).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_state = mask_k[:, None] & mask_v[None, :]
    state = tl.zeros((BK, BV), dtype=tl.float32)
    if HAS_INITIAL:
        p_initial = initial_state + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        state += tl.load(p_initial, mask=mask_state, other=0.0).to(tl.float32)

    chunks = T // CHUNK
    for i_c in tl.range(0, chunks):
        for i_t in tl.static_range(0, CHUNK):
            t = i_c * CHUNK + i_t
            token_head = ((i_b * T + t) * H + i_h).to(tl.int64)
            q_t = tl.load(q + token_head * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
            k_t = tl.load(k + token_head * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
            v_t = tl.load(v + token_head * V + o_v, mask=mask_v, other=0.0).to(tl.float32)
            alpha = tl.exp(tl.load(g + token_head).to(tl.float32))
            beta_t = tl.load(beta + token_head).to(tl.float32)
            gamma_t = tl.load(gamma + token_head).to(tl.float32)
            q_read = tl.sum(q_t[:, None] * state, axis=0)
            k_read = tl.sum(k_t[:, None] * state, axis=0)
            correlation = tl.sum(q_t * k_t, axis=0)

            if ORDER == ORDER_RECALL_THEN_DELTA:
                recall_write = gamma_t * (1.0 - alpha) * q_read
                state = alpha * state + q_t[:, None] * recall_write[None, :]
                delta_write = beta_t * (v_t - tl.sum(k_t[:, None] * state, axis=0))
                state += k_t[:, None] * delta_write[None, :]
                result = (alpha * q_read + recall_write) + correlation * delta_write
            elif ORDER == ORDER_DELTA_THEN_RECALL:
                delta_write = beta_t * (v_t - alpha * k_read)
                state = alpha * state + k_t[:, None] * delta_write[None, :]
                q_after_delta = alpha * q_read + correlation * delta_write
                recall_write = gamma_t * (q_read - q_after_delta)
                state += q_t[:, None] * recall_write[None, :]
                result = q_after_delta + recall_write
            else:
                recall_write = gamma_t * (1.0 - alpha) * q_read
                delta_write = beta_t * (v_t - alpha * k_read)
                state = (
                    alpha * state
                    + q_t[:, None] * recall_write[None, :]
                    + k_t[:, None] * delta_write[None, :]
                )
                result = (alpha * q_read + recall_write) + correlation * delta_write
            tl.store(output + token_head * V + o_v, SCALE * result, mask=mask_v)

        chunk_head = ((i_b * chunks + i_c) * H + i_h).to(tl.int64)
        p_chunk = chunk_ends + chunk_head * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_chunk, state, mask=mask_state)

    if STORE_FINAL:
        p_final = final_state + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_final, state, mask=mask_state)


@triton.jit(do_not_specialize=["T"])
def _qgdn_physical_bwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    gamma,
    grad_output,
    chunk_ends,
    grad_q,
    grad_k,
    grad_v,
    grad_g,
    grad_beta,
    grad_gamma,
    grad_initial,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHUNK: tl.constexpr,
    ORDER: tl.constexpr,
    SCALE: tl.constexpr,
    STORE_INITIAL_GRAD: tl.constexpr,
):
    i_bh = tl.program_id(0).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_state = mask_k[:, None] & mask_v[None, :]
    adjoint = tl.zeros((BK, BV), dtype=tl.float32)
    chunks = T // CHUNK

    for reverse_chunk in tl.range(0, chunks):
        i_c = chunks - 1 - reverse_chunk
        chunk_head = ((i_b * chunks + i_c) * H + i_h).to(tl.int64)
        p_chunk = chunk_ends + chunk_head * K * V + o_k[:, None] * V + o_v[None, :]
        state = tl.load(p_chunk, mask=mask_state, other=0.0).to(tl.float32)

        for reverse_token in tl.static_range(0, CHUNK):
            i_t = CHUNK - 1 - reverse_token
            t = i_c * CHUNK + i_t
            token_head = ((i_b * T + t) * H + i_h).to(tl.int64)
            q_t = tl.load(q + token_head * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
            k_t = tl.load(k + token_head * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
            v_t = tl.load(v + token_head * V + o_v, mask=mask_v, other=0.0).to(tl.float32)
            alpha = tl.exp(tl.load(g + token_head).to(tl.float32))
            beta_t = tl.load(beta + token_head).to(tl.float32)
            gamma_t = tl.load(gamma + token_head).to(tl.float32)
            correlation = tl.sum(q_t * k_t, axis=0)

            # Invert alpha*I + L*R^T with the 2x2 Woodbury system.  Loading an
            # exact FP32 chunk endpoint every CHUNK steps bounds reconstruction
            # drift while avoiding a T*K*V state tape.
            eta = gamma_t * (1.0 - alpha)
            if ORDER == ORDER_RECALL_THEN_DELTA:
                r0 = eta * q_t
                r1 = -alpha * beta_t * k_t - beta_t * eta * correlation * q_t
                write = beta_t * k_t
            elif ORDER == ORDER_DELTA_THEN_RECALL:
                r0 = eta * q_t + gamma_t * alpha * beta_t * correlation * k_t
                r1 = -alpha * beta_t * k_t
                write = beta_t * (k_t - gamma_t * correlation * q_t)
            else:
                r0 = eta * q_t
                r1 = -alpha * beta_t * k_t
                write = beta_t * k_t

            residual_state = state - write[:, None] * v_t[None, :]
            y0 = tl.sum(r0[:, None] * residual_state, axis=0)
            y1 = tl.sum(r1[:, None] * residual_state, axis=0)
            m00 = alpha + tl.sum(r0 * q_t, axis=0)
            m01 = tl.sum(r0 * k_t, axis=0)
            m10 = tl.sum(r1 * q_t, axis=0)
            m11 = alpha + tl.sum(r1 * k_t, axis=0)
            determinant = m00 * m11 - m01 * m10
            solved0 = (m11 * y0 - m01 * y1) / determinant
            solved1 = (m00 * y1 - m10 * y0) / determinant
            old_state = (
                residual_state
                - q_t[:, None] * solved0[None, :]
                - k_t[:, None] * solved1[None, :]
            ) / alpha

            do_t = tl.load(
                grad_output + token_head * V + o_v, mask=mask_v, other=0.0
            ).to(tl.float32) * SCALE
            dq_t = tl.sum(state * do_t[None, :], axis=1)
            current = adjoint + q_t[:, None] * do_t[None, :]
            dg_t = 0.0
            dbeta_t = 0.0
            dgamma_t = 0.0
            dv_t = tl.zeros((BV,), dtype=tl.float32)
            dk_t = tl.zeros((BK,), dtype=tl.float32)

            if ORDER == ORDER_RECALL_THEN_DELTA:
                old_q = tl.sum(q_t[:, None] * old_state, axis=0)
                decayed = alpha * old_state
                recall_error = old_q - tl.sum(q_t[:, None] * decayed, axis=0)
                recall_write = gamma_t * recall_error
                recalled = decayed + q_t[:, None] * recall_write[None, :]
                delta_error = v_t - tl.sum(k_t[:, None] * recalled, axis=0)
                delta_write = beta_t * delta_error

                grad_delta_write = tl.sum(k_t[:, None] * current, axis=0)
                dk_t += tl.sum(current * delta_write[None, :], axis=1)
                dbeta_t += tl.sum(grad_delta_write * delta_error, axis=0)
                grad_delta_error = beta_t * grad_delta_write
                dv_t += grad_delta_error
                grad_delta_read = -grad_delta_error
                dk_t += tl.sum(recalled * grad_delta_read[None, :], axis=1)
                grad_recalled = current + k_t[:, None] * grad_delta_read[None, :]

                grad_recall_write = tl.sum(q_t[:, None] * grad_recalled, axis=0)
                dq_t += tl.sum(grad_recalled * recall_write[None, :], axis=1)
                dgamma_t += tl.sum(grad_recall_write * recall_error, axis=0)
                grad_recall_error = gamma_t * grad_recall_write
                grad_old_q = grad_recall_error
                grad_decayed_read = -grad_recall_error
                dq_t += tl.sum(decayed * grad_decayed_read[None, :], axis=1)
                grad_decayed = grad_recalled + q_t[:, None] * grad_decayed_read[None, :]
                dg_t += tl.sum(grad_decayed * old_state, axis=0)
                grad_old = alpha * grad_decayed
                dq_t += tl.sum(old_state * grad_old_q[None, :], axis=1)
                grad_old += q_t[:, None] * grad_old_q[None, :]

            elif ORDER == ORDER_DELTA_THEN_RECALL:
                old_q = tl.sum(q_t[:, None] * old_state, axis=0)
                old_k = tl.sum(k_t[:, None] * old_state, axis=0)
                delta_error = v_t - alpha * old_k
                delta_write = beta_t * delta_error
                delta_state = alpha * old_state + k_t[:, None] * delta_write[None, :]
                delta_q = tl.sum(q_t[:, None] * delta_state, axis=0)
                recall_error = old_q - delta_q
                recall_write = gamma_t * recall_error

                grad_recall_write = tl.sum(q_t[:, None] * current, axis=0)
                dq_t += tl.sum(current * recall_write[None, :], axis=1)
                dgamma_t += tl.sum(grad_recall_write * recall_error, axis=0)
                grad_recall_error = gamma_t * grad_recall_write
                grad_old_q = grad_recall_error
                grad_delta_q = -grad_recall_error
                dq_t += tl.sum(delta_state * grad_delta_q[None, :], axis=1)
                grad_delta_state = current + q_t[:, None] * grad_delta_q[None, :]

                grad_delta_write = tl.sum(k_t[:, None] * grad_delta_state, axis=0)
                dk_t += tl.sum(grad_delta_state * delta_write[None, :], axis=1)
                dbeta_t += tl.sum(grad_delta_write * delta_error, axis=0)
                grad_delta_error = beta_t * grad_delta_write
                dv_t += grad_delta_error
                grad_old_k = -alpha * grad_delta_error
                dg_t += tl.sum(-old_k * grad_delta_error, axis=0)
                dk_t += tl.sum(old_state * grad_old_k[None, :], axis=1)
                dg_t += tl.sum(grad_delta_state * old_state, axis=0)
                grad_old = alpha * grad_delta_state + k_t[:, None] * grad_old_k[None, :]
                dq_t += tl.sum(old_state * grad_old_q[None, :], axis=1)
                grad_old += q_t[:, None] * grad_old_q[None, :]

            else:
                old_q = tl.sum(q_t[:, None] * old_state, axis=0)
                decayed = alpha * old_state
                recall_error = old_q - tl.sum(q_t[:, None] * decayed, axis=0)
                recall_write = gamma_t * recall_error
                delta_error = v_t - tl.sum(k_t[:, None] * decayed, axis=0)
                delta_write = beta_t * delta_error

                grad_recall_write = tl.sum(q_t[:, None] * current, axis=0)
                dq_t += tl.sum(current * recall_write[None, :], axis=1)
                dgamma_t += tl.sum(grad_recall_write * recall_error, axis=0)
                grad_recall_error = gamma_t * grad_recall_write
                grad_old_q = grad_recall_error
                grad_decayed_q = -grad_recall_error
                dq_t += tl.sum(decayed * grad_decayed_q[None, :], axis=1)
                grad_decayed = current + q_t[:, None] * grad_decayed_q[None, :]

                grad_delta_write = tl.sum(k_t[:, None] * current, axis=0)
                dk_t += tl.sum(current * delta_write[None, :], axis=1)
                dbeta_t += tl.sum(grad_delta_write * delta_error, axis=0)
                grad_delta_error = beta_t * grad_delta_write
                dv_t += grad_delta_error
                grad_decayed_k = -grad_delta_error
                dk_t += tl.sum(decayed * grad_decayed_k[None, :], axis=1)
                grad_decayed += k_t[:, None] * grad_decayed_k[None, :]

                dg_t += tl.sum(grad_decayed * old_state, axis=0)
                grad_old = alpha * grad_decayed
                dq_t += tl.sum(old_state * grad_old_q[None, :], axis=1)
                grad_old += q_t[:, None] * grad_old_q[None, :]

            # dg_t above is dL/dalpha.  alpha=exp(g).
            dg_t *= alpha
            tl.store(grad_q + token_head * K + o_k, dq_t, mask=mask_k)
            tl.store(grad_k + token_head * K + o_k, dk_t, mask=mask_k)
            tl.store(grad_v + token_head * V + o_v, dv_t, mask=mask_v)
            tl.store(grad_g + token_head, dg_t)
            tl.store(grad_beta + token_head, dbeta_t)
            tl.store(grad_gamma + token_head, dgamma_t)
            state = old_state
            adjoint = grad_old

    if STORE_INITIAL_GRAD:
        p_grad_initial = grad_initial + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_grad_initial, adjoint, mask=mask_state)


def qgdn_physical_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    *,
    update_order: str,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the physical-T forward and return output, chunk ends, final state."""
    if update_order not in ORDER_IDS:
        raise ValueError(update_order)
    if not q.is_cuda or q.ndim != 4 or k.shape != q.shape or q.shape[:-1] != v.shape[:-1]:
        raise ValueError("q, k and v must be CUDA [B,T,H,D] tensors")
    B, T, H, K = q.shape
    V = v.shape[-1]
    if K != V or K not in (32, 64, 128):
        raise ValueError("physical QGDN currently requires K == V in {32,64,128}")
    if T == 0 or T % chunk_size:
        raise ValueError("sequence length must be nonzero and divisible by chunk_size")
    if any(x.shape != (B, T, H) for x in (g, beta, gamma)):
        raise ValueError("g, beta and gamma must have shape [B,T,H]")
    if initial_state is not None and initial_state.shape != (B, H, K, V):
        raise ValueError("initial_state has the wrong shape")
    if any(not x.is_contiguous() for x in (q, k, v, g, beta, gamma)):
        q, k, v, g, beta, gamma = (x.contiguous() for x in (q, k, v, g, beta, gamma))
    scale = K**-0.5 if scale is None else scale
    output = torch.empty_like(v)
    chunk_ends = torch.empty(
        (B, T // chunk_size, H, K, V), device=q.device, dtype=torch.float32
    )
    final_state = (
        torch.empty((B, H, K, V), device=q.device, dtype=torch.float32)
        if output_final_state else None
    )
    grid = (B * H,)
    _qgdn_physical_fwd_kernel[grid](
        q, k, v, g, beta, gamma, initial_state, output, chunk_ends, final_state,
        T=T, H=H, K=K, V=V,
        BK=triton.next_power_of_2(K), BV=triton.next_power_of_2(V),
        CHUNK=chunk_size, ORDER=ORDER_IDS[update_order], SCALE=scale,
        HAS_INITIAL=initial_state is not None, STORE_FINAL=output_final_state,
        num_warps=8, num_stages=2,
    )
    return output, chunk_ends, final_state


class _PhysicalQGDNFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, g, beta, gamma, initial_state, scale, update_order, chunk_size):
        output, chunk_ends, _ = qgdn_physical_forward(
            q, k, v, g, beta, gamma,
            update_order=update_order,
            scale=scale,
            initial_state=initial_state,
            output_final_state=False,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q, k, v, g, beta, gamma, chunk_ends)
        ctx.has_initial = initial_state is not None
        ctx.initial_dtype = initial_state.dtype if initial_state is not None else None
        ctx.scale = q.shape[-1] ** -0.5 if scale is None else scale
        ctx.update_order = update_order
        ctx.chunk_size = chunk_size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, g, beta, gamma, chunk_ends = ctx.saved_tensors
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grad_g = torch.empty_like(g)
        grad_beta = torch.empty_like(beta)
        grad_gamma = torch.empty_like(gamma)
        grad_initial = (
            torch.empty(
                (q.shape[0], q.shape[2], q.shape[3], v.shape[3]),
                device=q.device,
                dtype=torch.float32,
            )
            if ctx.has_initial else None
        )
        B, T, H, K = q.shape
        V = v.shape[-1]
        _qgdn_physical_bwd_kernel[(B * H,)](
            q, k, v, g, beta, gamma, grad_output.contiguous(), chunk_ends,
            grad_q, grad_k, grad_v, grad_g, grad_beta, grad_gamma, grad_initial,
            T=T, H=H, K=K, V=V,
            BK=triton.next_power_of_2(K), BV=triton.next_power_of_2(V),
            CHUNK=ctx.chunk_size, ORDER=ORDER_IDS[ctx.update_order], SCALE=ctx.scale,
            STORE_INITIAL_GRAD=ctx.has_initial,
            num_warps=8, num_stages=2,
        )
        if grad_initial is not None and ctx.initial_dtype != torch.float32:
            grad_initial = grad_initial.to(ctx.initial_dtype)
        return (
            grad_q, grad_k, grad_v, grad_g, grad_beta, grad_gamma, grad_initial,
            None, None, None,
        )


def qgdn_physical(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    *,
    update_order: str,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Differentiable physical-T QGDN training operator."""
    return _PhysicalQGDNFunction.apply(
        q, k, v, g, beta, gamma, initial_state, scale, update_order, chunk_size
    )
