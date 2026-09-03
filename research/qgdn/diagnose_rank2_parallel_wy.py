"""Benchmark the physical-T parallel-WY oracle on one CUDA device."""
from __future__ import annotations

import argparse
from functools import partial
import json
import statistics
from pathlib import Path

import torch

from lit_gpt.mixers.qgdn_reference import (
    qgdn_rank2_chunk_batched_reference,
    qgdn_rank2_parallel_wy_reference,
)


UPDATE_ORDERS = ("recall_then_delta", "delta_then_recall", "parallel")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def make_inputs(args):
    torch.manual_seed(3407)
    shape = (args.batch_size, args.sequence_length, args.heads)
    q = torch.randn(*shape, args.key_dim, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn(*shape, args.value_dim, device="cuda")
    g = -0.8 * torch.rand(*shape, device="cuda")
    beta = torch.rand_like(g)
    gamma = torch.rand_like(g)
    state = torch.randn(
        args.batch_size,
        args.heads,
        args.key_dim,
        args.value_dim,
        device="cuda",
    )
    return tuple(x.requires_grad_() for x in (q, k, v, g, beta, gamma, state))


def one_step(operator, tensors, update_order, chunk_size):
    output, final_state = operator(
        *tensors[:6],
        initial_state=tensors[6],
        update_order=update_order,
        chunk_size=chunk_size,
    )
    loss = output.float().square().mean() + final_state.float().square().mean()
    gradients = torch.autograd.grad(loss, tensors)
    finite = bool(output.isfinite().all().item())
    finite = finite and bool(final_state.isfinite().all().item())
    finite = finite and all(
        bool(gradient.isfinite().all().item()) for gradient in gradients
    )
    return finite


def benchmark(operator, tensors, update_order, args):
    for _ in range(args.warmup):
        assert one_step(operator, tensors, update_order, args.chunk_size)
    torch.cuda.synchronize()

    times_ms = []
    peak_bytes = []
    incremental_peak_bytes = []
    for _ in range(args.repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        assert one_step(operator, tensors, update_order, args.chunk_size)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
        peak = torch.cuda.max_memory_allocated()
        peak_bytes.append(peak)
        incremental_peak_bytes.append(peak - allocated_before)

    median_ms = statistics.median(times_ms)
    return {
        "median_forward_backward_ms": median_ms,
        "physical_tokens_per_second": (
            args.batch_size * args.sequence_length * 1000 / median_ms
        ),
        "peak_allocated_gb": max(peak_bytes) / 1e9,
        "incremental_peak_allocated_gb": max(incremental_peak_bytes) / 1e9,
        "all_outputs_states_and_gradients_finite": True,
        "times_ms": times_ms,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tensors = make_inputs(args)
    operators = {
        "per_chunk_solve_oracle": qgdn_rank2_chunk_batched_reference,
        "parallel_wy_two_solve_oracle": partial(
            qgdn_rank2_parallel_wy_reference, fuse_wy_rhs=False
        ),
        "parallel_wy_fused_rhs_oracle": partial(
            qgdn_rank2_parallel_wy_reference, fuse_wy_rhs=True
        ),
        "parallel_wy_streaming_oracle": partial(
            qgdn_rank2_parallel_wy_reference, wy_backend="streaming"
        ),
    }
    results = {}
    for update_order in UPDATE_ORDERS:
        results[update_order] = {
            name: benchmark(operator, tensors, update_order, args)
            for name, operator in operators.items()
        }
        baseline = results[update_order]["per_chunk_solve_oracle"]
        two_solve = results[update_order]["parallel_wy_two_solve_oracle"]
        candidate = results[update_order]["parallel_wy_fused_rhs_oracle"]
        candidate["speedup_vs_per_chunk_solve"] = (
            baseline["median_forward_backward_ms"]
            / candidate["median_forward_backward_ms"]
        )
        candidate["speedup_vs_two_solve_parallel_wy"] = (
            two_solve["median_forward_backward_ms"]
            / candidate["median_forward_backward_ms"]
        )
        candidate["peak_memory_ratio_vs_per_chunk_solve"] = (
            candidate["peak_allocated_gb"] / baseline["peak_allocated_gb"]
        )
        candidate["peak_memory_ratio_vs_two_solve_parallel_wy"] = (
            candidate["peak_allocated_gb"] / two_solve["peak_allocated_gb"]
        )
        streaming = results[update_order]["parallel_wy_streaming_oracle"]
        streaming["speedup_vs_two_solve_parallel_wy"] = (
            two_solve["median_forward_backward_ms"]
            / streaming["median_forward_backward_ms"]
        )
        streaming["peak_memory_ratio_vs_two_solve_parallel_wy"] = (
            streaming["peak_allocated_gb"] / two_solve["peak_allocated_gb"]
        )

    payload = {
        "device": torch.cuda.get_device_name(),
        "dtype": "float32",
        "shape": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "heads": args.heads,
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
            "chunk_size": args.chunk_size,
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "scope": "operator oracle forward+backward; not a full-model benchmark",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
