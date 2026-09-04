"""Validate and benchmark fused physical-T WY/state/output forward/backward."""
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
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--iterations-per-sample", type=int, default=8)
    parser.add_argument("--memory-repeats", type=int, default=3)
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
    if backend == "triangular":
        wy_backend, state_backend = "triangular", "torch"
    elif backend == "triton_wy":
        wy_backend, state_backend = "triton", "torch"
    elif backend == "triton_fused":
        wy_backend, state_backend = "triton", "triton"
    else:
        raise ValueError(f"unsupported diagnostic backend: {backend}")
    outputs = qgdn_rank2_parallel_wy_reference(
        *tensors[:6],
        initial_state=tensors[6],
        update_order=update_order,
        chunk_size=chunk_size,
        wy_backend=wy_backend,
        state_backend=state_backend,
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


def percentile(values, quantile):
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(values):
    median = statistics.median(values)
    absolute_deviations = [abs(value - median) for value in values]
    return {
        "median": median,
        "mad": statistics.median(absolute_deviations),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
        "samples": values,
    }


def timed_step(tensors, update_order, backend, args, output_grads):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for iteration in range(args.iterations_per_sample):
        outputs, gradients, _ = run_step(
            tensors,
            update_order,
            args.chunk_size,
            backend,
            output_grads,
        )
        if iteration + 1 < args.iterations_per_sample:
            del outputs, gradients
    end.record()
    torch.cuda.synchronize()
    assert all(bool(value.isfinite().all().item()) for value in outputs)
    assert all(bool(value.isfinite().all().item()) for value in gradients)
    elapsed_ms = start.elapsed_time(end) / args.iterations_per_sample
    del outputs, gradients
    return elapsed_ms


def measure_memory(tensors, update_order, backend, args, output_grads):
    peak_bytes = []
    incremental_peak_bytes = []
    for _ in range(args.memory_repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        outputs, gradients, _ = run_step(
            tensors,
            update_order,
            args.chunk_size,
            backend,
            output_grads,
        )
        torch.cuda.synchronize()
        assert all(bool(value.isfinite().all().item()) for value in outputs)
        assert all(bool(value.isfinite().all().item()) for value in gradients)
        peak = torch.cuda.max_memory_allocated()
        peak_bytes.append(peak)
        incremental_peak_bytes.append(peak - allocated_before)
        del outputs, gradients
    return {
        "peak_allocated_gb": max(peak_bytes) / 1e9,
        "incremental_peak_allocated_gb": max(incremental_peak_bytes) / 1e9,
    }


def benchmark_interleaved(tensors, output_grads_by_order, args):
    backends = ("triangular", "triton_wy", "triton_fused")
    for warmup_index in range(args.warmup):
        order_offset = warmup_index % len(UPDATE_ORDERS)
        orders = UPDATE_ORDERS[order_offset:] + UPDATE_ORDERS[:order_offset]
        for position, update_order in enumerate(orders):
            backend_offset = (warmup_index + position) % len(backends)
            backend_order = backends[backend_offset:] + backends[:backend_offset]
            for backend in backend_order:
                timed_step(
                    tensors,
                    update_order,
                    backend,
                    args,
                    output_grads_by_order[update_order],
                )

    times_ms = {
        update_order: {backend: [] for backend in backends}
        for update_order in UPDATE_ORDERS
    }
    paired_speedups = {
        update_order: {backend: [] for backend in backends[1:]}
        for update_order in UPDATE_ORDERS
    }
    paired_fused_vs_wy = {update_order: [] for update_order in UPDATE_ORDERS}
    backend_sequences = {update_order: [] for update_order in UPDATE_ORDERS}
    for repeat_index in range(args.repeats):
        order_offset = repeat_index % len(UPDATE_ORDERS)
        orders = UPDATE_ORDERS[order_offset:] + UPDATE_ORDERS[:order_offset]
        for position, update_order in enumerate(orders):
            backend_offset = (repeat_index + position) % len(backends)
            backend_order = backends[backend_offset:] + backends[:backend_offset]
            paired_times = {}
            for backend in backend_order:
                paired_times[backend] = timed_step(
                    tensors,
                    update_order,
                    backend,
                    args,
                    output_grads_by_order[update_order],
                )
                times_ms[update_order][backend].append(paired_times[backend])
            for backend in backends[1:]:
                paired_speedups[update_order][backend].append(
                    paired_times["triangular"] / paired_times[backend]
                )
            paired_fused_vs_wy[update_order].append(
                paired_times["triton_wy"] / paired_times["triton_fused"]
            )
            backend_sequences[update_order].append("->".join(backend_order))

    results = {}
    for update_order in UPDATE_ORDERS:
        backend_results = {}
        for backend in backends:
            timing = summarize_samples(times_ms[update_order][backend])
            memory = measure_memory(
                tensors,
                update_order,
                backend,
                args,
                output_grads_by_order[update_order],
            )
            backend_results[backend] = {
                "median_forward_backward_ms": timing["median"],
                "physical_tokens_per_second": (
                    args.batch_size
                    * args.sequence_length
                    * 1000
                    / timing["median"]
                ),
                "timing_ms": timing,
                **memory,
                "all_outputs_states_and_gradients_finite": True,
            }
        baseline = backend_results["triangular"]
        for backend in backends[1:]:
            candidate = backend_results[backend]
            candidate["speedup_vs_triangular"] = (
                baseline["median_forward_backward_ms"]
                / candidate["median_forward_backward_ms"]
            )
            candidate["paired_speedup_vs_triangular"] = summarize_samples(
                paired_speedups[update_order][backend]
            )
            candidate["peak_memory_ratio_vs_triangular"] = (
                candidate["peak_allocated_gb"] / baseline["peak_allocated_gb"]
            )
            candidate["incremental_peak_memory_ratio_vs_triangular"] = (
                candidate["incremental_peak_allocated_gb"]
                / baseline["incremental_peak_allocated_gb"]
            )
        fused = backend_results["triton_fused"]
        wy_only = backend_results["triton_wy"]
        fused["speedup_vs_triton_wy"] = (
            wy_only["median_forward_backward_ms"]
            / fused["median_forward_backward_ms"]
        )
        fused["paired_speedup_vs_triton_wy"] = summarize_samples(
            paired_fused_vs_wy[update_order]
        )
        fused["peak_memory_ratio_vs_triton_wy"] = (
            fused["peak_allocated_gb"] / wy_only["peak_allocated_gb"]
        )
        fused["incremental_peak_memory_ratio_vs_triton_wy"] = (
            fused["incremental_peak_allocated_gb"]
            / wy_only["incremental_peak_allocated_gb"]
        )
        results[update_order] = {
            "triangular": baseline,
            "triton_wy": wy_only,
            "triton_fused_state_output": fused,
            "backend_sequences": backend_sequences[update_order],
        }
    return results


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tensors = make_inputs(args)
    results = {}
    output_grads_by_order = {}
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
            "triton_fused",
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
                f"{update_order} fused physical-T forward RMSE failed: {errors}"
            )
        if max(errors["input_gradients"]) >= 1e-4:
            raise AssertionError(
                f"{update_order} fused physical-T gradient RMSE failed: {errors}"
            )
        output_grads_by_order[update_order] = tuple(
            value.detach() for value in output_grads
        )
        results[update_order] = {"relative_rmse": errors}
        del (
            expected_inputs,
            expected,
            expected_grads,
            actual_inputs,
            actual,
            actual_grads,
            output_grads,
        )

    benchmark_results = benchmark_interleaved(tensors, output_grads_by_order, args)
    for update_order in UPDATE_ORDERS:
        results[update_order].update(benchmark_results[update_order])

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
        "iterations_per_sample": args.iterations_per_sample,
        "memory_repeats": args.memory_repeats,
        "scope": (
            "physical-T operator forward+backward; triangular, fused-WY-only, "
            "and fused-WY-plus-chunk-state/output are order-rotated and "
            "interleaved; timing is block-amortized, memory is measured "
            "separately; not a full-model benchmark"
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
