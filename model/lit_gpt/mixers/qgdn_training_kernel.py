"""Memory-bounded autograd wrapper for fused physical-T QGDN training."""
from __future__ import annotations

import torch


def _prepare_physical_chunks(
    q,
    k,
    v,
    g,
    beta,
    gamma,
    initial_state,
    *,
    recall_mode: str,
    update_order: str,
    chunk_size: int,
):
    from .qgdn_reference import qgdn_rank2_factors

    qn, alpha, left, right, write = qgdn_rank2_factors(
        q,
        k,
        g,
        beta,
        gamma,
        recall_mode=recall_mode,
        update_order=update_order,
    )
    values = v.to(qn.dtype)
    batch, length, heads, key_dim = qn.shape
    value_dim = values.shape[-1]
    expected_state = (batch, heads, key_dim, value_dim)
    state = initial_state.to(qn.dtype)
    if tuple(state.shape) != expected_state:
        raise ValueError(f"initial_state must have shape {expected_state}")

    pad = (-length) % chunk_size

    def pad_time(tensor, value=0):
        if pad == 0:
            return tensor
        padding = tensor.new_full((batch, pad, *tensor.shape[2:]), value)
        return torch.cat((tensor, padding), dim=1)

    padded_length = length + pad
    chunks = padded_length // chunk_size
    rank = left.shape[-2]
    queries = pad_time(qn).reshape(
        batch, chunks, chunk_size, heads, key_dim
    ).permute(0, 3, 1, 2, 4)
    alpha_chunks = pad_time(alpha, 1).reshape(
        batch, chunks, chunk_size, heads
    ).permute(0, 3, 1, 2)
    left_chunks = pad_time(left).reshape(
        batch, chunks, chunk_size, heads, rank, key_dim
    ).permute(0, 3, 1, 2, 4, 5)
    right_chunks = pad_time(right).reshape(
        batch, chunks, chunk_size, heads, rank, key_dim
    ).permute(0, 3, 1, 2, 4, 5)
    write_chunks = pad_time(write).reshape(
        batch, chunks, chunk_size, heads, key_dim
    ).permute(0, 3, 1, 2, 4)
    value_chunks = pad_time(values).reshape(
        batch, chunks, chunk_size, heads, value_dim
    ).permute(0, 3, 1, 2, 4)
    decay_prefix = alpha_chunks.cumprod(dim=-1)
    normalized_left = left_chunks / alpha_chunks[..., None, None]
    normalized_write = write_chunks / decay_prefix[..., None]
    return (
        queries,
        decay_prefix,
        normalized_left,
        right_chunks,
        normalized_write,
        value_chunks,
        state,
    ), length, padded_length


def _raw_physical_forward(prepared, output_scale: float):
    from .qgdn_state_output_kernel import _qgdn_chunk_state_output_cuda_fwd
    from .qgdn_wy_kernel import _qgdn_streaming_wy_cuda_fwd

    (
        queries,
        decay_prefix,
        normalized_left,
        right,
        normalized_write,
        values,
        initial_state,
    ) = prepared
    effective_right, write_reads = _qgdn_streaming_wy_cuda_fwd(
        normalized_left,
        right,
        normalized_write,
        values,
    )
    outputs, final_state, _, _ = _qgdn_chunk_state_output_cuda_fwd(
        queries,
        decay_prefix,
        normalized_left,
        effective_right,
        write_reads,
        normalized_write,
        values,
        initial_state,
        output_scale,
    )
    return outputs, final_state


class _QGDNPhysicalTraining(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        gamma,
        initial_state,
        recall_mode,
        update_order,
        output_scale,
        chunk_size,
    ):
        prepared, length, padded_length = _prepare_physical_chunks(
            q,
            k,
            v,
            g,
            beta,
            gamma,
            initial_state,
            recall_mode=recall_mode,
            update_order=update_order,
            chunk_size=chunk_size,
        )
        outputs, final_state = _raw_physical_forward(prepared, output_scale)
        batch, _, _, _, value_dim = outputs.shape
        heads = outputs.shape[1]
        outputs = outputs.permute(0, 2, 3, 1, 4).reshape(
            batch, padded_length, heads, value_dim
        )
        ctx.save_for_backward(q, k, v, g, beta, gamma, initial_state)
        ctx.recall_mode = recall_mode
        ctx.update_order = update_order
        ctx.output_scale = output_scale
        ctx.chunk_size = chunk_size
        ctx.padded_length = padded_length
        return outputs[:, :length], final_state

    @staticmethod
    def backward(ctx, grad_outputs, grad_final_state):
        from .qgdn_state_output_kernel import (
            _qgdn_chunk_state_cuda_fwd,
            _qgdn_chunk_state_output_cuda_bwd,
        )
        from .qgdn_wy_kernel import (
            _qgdn_streaming_wy_cuda_bwd,
            _qgdn_streaming_wy_cuda_fwd,
        )

        saved = ctx.saved_tensors
        with torch.enable_grad():
            recomputed = [
                tensor.detach().requires_grad_(True) for tensor in saved
            ]
            prepared, _, _ = _prepare_physical_chunks(
                *recomputed,
                recall_mode=ctx.recall_mode,
                update_order=ctx.update_order,
                chunk_size=ctx.chunk_size,
            )

        (
            queries,
            decay_prefix,
            normalized_left,
            right,
            normalized_write,
            values,
            initial_state,
        ) = prepared
        effective_right, write_reads = _qgdn_streaming_wy_cuda_fwd(
            normalized_left,
            right,
            normalized_write,
            values,
        )
        state_inputs = tuple(
            tensor.contiguous()
            for tensor in (
                queries,
                decay_prefix,
                normalized_left,
                effective_right,
                write_reads,
                normalized_write,
                values,
                initial_state,
            )
        )
        chunk_starts, _ = _qgdn_chunk_state_cuda_fwd(
            state_inputs[2],
            state_inputs[3],
            state_inputs[4],
            state_inputs[5],
            state_inputs[6],
            state_inputs[1],
            state_inputs[7],
        )

        if grad_outputs is None:
            batch, length, heads = saved[0].shape[:3]
            value_dim = saved[2].shape[-1]
            grad_outputs = torch.zeros(
                batch,
                length,
                heads,
                value_dim,
                dtype=torch.float32,
                device=saved[0].device,
            )
        batch, length, heads, value_dim = grad_outputs.shape
        if ctx.padded_length != length:
            padded_grad_outputs = grad_outputs.new_zeros(
                batch, ctx.padded_length, heads, value_dim
            )
            padded_grad_outputs[:, :length] = grad_outputs
        else:
            padded_grad_outputs = grad_outputs
        chunk_grad_outputs = padded_grad_outputs.reshape(
            batch, -1, ctx.chunk_size, heads, value_dim
        ).permute(0, 3, 1, 2, 4)
        if grad_final_state is None:
            grad_final_state = torch.zeros_like(initial_state)
        state_gradients = _qgdn_chunk_state_output_cuda_bwd(
            state_inputs,
            chunk_starts,
            chunk_grad_outputs,
            grad_final_state,
            ctx.output_scale,
        )
        wy_gradients = _qgdn_streaming_wy_cuda_bwd(
            normalized_left,
            right,
            normalized_write,
            values,
            state_gradients[3],
            state_gradients[4],
        )
        prepared_gradients = (
            state_gradients[0],
            state_gradients[1],
            state_gradients[2] + wy_gradients[0],
            wy_gradients[1],
            state_gradients[5] + wy_gradients[2],
            state_gradients[6] + wy_gradients[3],
            state_gradients[7],
        )
        with torch.enable_grad():
            input_gradients = torch.autograd.grad(
                prepared,
                recomputed,
                prepared_gradients,
                allow_unused=True,
            )
        input_gradients = tuple(
            gradient if needed else None
            for gradient, needed in zip(
                input_gradients, ctx.needs_input_grad[:7]
            )
        )
        return (*input_gradients, None, None, None, None)


def qgdn_physical_training(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    *,
    recall_mode: str,
    update_order: str,
    scale: float | None,
    initial_state: torch.Tensor | None,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run physical-T QGDN while retaining only original model inputs."""
    if not all(tensor.is_cuda for tensor in (q, k, v, g, beta, gamma)):
        raise ValueError("the fused physical-T training path requires CUDA")
    batch, _, heads, key_dim = q.shape
    state = (
        torch.zeros(
            batch,
            heads,
            key_dim,
            v.shape[-1],
            dtype=torch.float32,
            device=q.device,
        )
        if initial_state is None
        else initial_state
    )
    output_scale = key_dim**-0.5 if scale is None else scale
    return _QGDNPhysicalTraining.apply(
        q,
        k,
        v,
        g,
        beta,
        gamma,
        state,
        recall_mode,
        update_order,
        output_scale,
        chunk_size,
    )
