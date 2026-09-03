"""Validate and benchmark the forward-only physical-T Triton WY primitive."""
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
    return q, k, v, g, beta, gamma, state


def run_forward(tensors, update_order, chunk_size, backend):
    return qgdn_rank2_parallel_wy_reference(
        *tensors[:6],
        initial_state=tensors[6],
        update_order=update_order,
        chunk_size=chunk_size,
        wy_backend=backend,
    )


def relative_rmse(actual, expected):
    return float(
        (
            (actual - expected).square().mean().sqrt()
            / expected.square().mean().sqrt().clamp_min(1e-7)
        ).item()
    )


def benchmark(tensors, update_order, backend, args):
    for _ in range(args.warmup):
        outputs = run_forward(tensors, update_order, args.chunk_size, backend)
        assert all(bool(value.isfinite().all().item()) for value in outputs)
    del outputs
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
        outputs = run_forward(tensors, update_order, args.chunk_size, backend)
        end.record()
        torch.cuda.synchronize()
        assert all(bool(value.isfinite().all().item()) for value in outputs)
        times_ms.append(start.elapsed_time(end))
        peak = torch.cuda.max_memory_allocated()
        peak_bytes.append(peak)
        incremental_peak_bytes.append(peak - allocated_before)
        del outputs

    median_ms = statistics.median(times_ms)
    return {
        "median_forward_ms": median_ms,
        "physical_tokens_per_second": (
            args.batch_size * args.sequence_length * 1000 / median_ms
        ),
        "peak_allocated_gb": max(peak_bytes) / 1e9,
        "incremental_peak_allocated_gb": max(incremental_peak_bytes) / 1e9,
        "all_outputs_and_states_finite": True,
        "times_ms": times_ms,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tensors = make_inputs(args)
    results = {}
    with torch.no_grad():
        for update_order in UPDATE_ORDERS:
            expected = run_forward(
                tensors, update_order, args.chunk_size, "triangular"
            )
            actual = run_forward(tensors, update_order, args.chunk_size, "triton")
            errors = {
                "output": relative_rmse(actual[0], expected[0]),
                "final_state": relative_rmse(actual[1], expected[1]),
            }
            if max(errors.values()) >= 2e-5:
                raise AssertionError(
                    f"{update_order} Triton WY relative RMSE failed: {errors}"
                )
            baseline = benchmark(
                tensors, update_order, "triangular", args
            )
            candidate = benchmark(tensors, update_order, "triton", args)
            candidate["speedup_vs_triangular"] = (
                baseline["median_forward_ms"] / candidate["median_forward_ms"]
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
                "triton": candidate,
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
        "scope": "forward-only operator oracle; not a training or full-model benchmark",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
