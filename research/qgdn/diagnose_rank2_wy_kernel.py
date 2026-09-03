"""Validate and benchmark Triton-forward/recompute-backward physical-T WY."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from lit_gpt.mixers.qgdn_reference import qgdn_rank2_parallel_wy_reference


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
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
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
    return tuple(
        value.requires_grad_() for value in (q, k, v, g, beta, gamma, state)
    )


def run_step(tensors, update_order, chunk_size, backend, output_grads=None):
    outputs = qgdn_rank2_parallel_wy_reference(
        *tensors[:6],
        initial_state=tensors[6],
        update_order=update_order,
        chunk_size=chunk_size,
        wy_backend=backend,
    )
    if output_grads is None:
        output_grads = tuple(torch.randn_like(value) for value in outputs)
    gradients = torch.autograd.grad(outputs, tensors, output_grads)
    return outputs, gradients, output_grads


def relative_rmse(actual, expected):
    return float(
        (
            (actual - expected).square().mean().sqrt()
            / expected.square().mean().sqrt().clamp_min(1e-7)
        ).item()
    )


def benchmark(tensors, update_order, backend, args):
    output_grads = None
    for _ in range(args.warmup):
        outputs, gradients, output_grads = run_step(
            tensors,
            update_order,
            args.chunk_size,
            backend,
            output_grads,
        )
        assert all(bool(value.isfinite().all().item()) for value in outputs)
        assert all(bool(value.isfinite().all().item()) for value in gradients)
    del outputs, gradients
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
        outputs, gradients, _ = run_step(
            tensors,
            update_order,
            args.chunk_size,
            backend,
            output_grads,
        )
        end.record()
        torch.cuda.synchronize()
        assert all(bool(value.isfinite().all().item()) for value in outputs)
        assert all(bool(value.isfinite().all().item()) for value in gradients)
        times_ms.append(start.elapsed_time(end))
        peak = torch.cuda.max_memory_allocated()
        peak_bytes.append(peak)
        incremental_peak_bytes.append(peak - allocated_before)
        del outputs, gradients

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
    results = {}
    for update_order in UPDATE_ORDERS:
        expected_inputs = tuple(
            value.detach().clone().requires_grad_() for value in tensors
        )
        expected, expected_grads, output_grads = run_step(
            expected_inputs, update_order, args.chunk_size, "triangular"
        )
        actual_inputs = tuple(
            value.detach().clone().requires_grad_() for value in tensors
        )
        actual, actual_grads, _ = run_step(
            actual_inputs,
            update_order,
            args.chunk_size,
            "triton",
            output_grads,
        )
        errors = {
            "output": relative_rmse(actual[0], expected[0]),
            "final_state": relative_rmse(actual[1], expected[1]),
            "input_gradients": [
                relative_rmse(value, reference)
                for value, reference in zip(actual_grads, expected_grads)
            ],
        }
        if max(errors["output"], errors["final_state"]) >= 2e-5:
            raise AssertionError(
                f"{update_order} Triton WY forward RMSE failed: {errors}"
            )
        if max(errors["input_gradients"]) >= 1e-4:
            raise AssertionError(
                f"{update_order} Triton WY gradient RMSE failed: {errors}"
            )
        baseline = benchmark(tensors, update_order, "triangular", args)
        candidate = benchmark(tensors, update_order, "triton", args)
        candidate["speedup_vs_triangular"] = (
            baseline["median_forward_backward_ms"]
            / candidate["median_forward_backward_ms"]
        )
        candidate["peak_memory_ratio_vs_triangular"] = (
            candidate["peak_allocated_gb"] / baseline["peak_allocated_gb"]
        )
        candidate["incremental_peak_memory_ratio_vs_triangular"] = (
            candidate["incremental_peak_allocated_gb"]
            / baseline["incremental_peak_allocated_gb"]
        )
        results[update_order] = {
            "relative_rmse": errors,
            "triangular": baseline,
            "triton_recompute": candidate,
        }

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
        "scope": "operator forward+backward; not a full-model benchmark",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
