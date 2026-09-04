"""Profile the fused physical-T training stages at an actual model shape."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

from runtime import configure_device_from_cli, configure_numerics

if __name__ == "__main__":
    configure_device_from_cli()

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))


def summarize(samples: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p10_ms": ordered[max(0, math.ceil(0.1 * len(ordered)) - 1)],
        "p90_ms": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)],
    }


def measure(function, *, warmup: int, measured: int) -> dict:
    for _ in range(warmup):
        result = function()
        torch.cuda.synchronize()
        del result
    samples = []
    for _ in range(measured):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del result
    return summarize(samples)


def measure_interleaved(functions: dict[str, object], *, warmup: int, measured: int) -> dict:
    names = tuple(functions)
    orders = list(itertools.permutations(names))
    for _ in range(warmup):
        for name in names:
            result = functions[name]()
            torch.cuda.synchronize()
            del result
    samples = {name: [] for name in names}
    position_counts = {name: [0] * len(names) for name in names}
    rounds = []
    for sample_index in range(measured):
        order = orders[sample_index % len(orders)]
        round_times = {}
        for position, name in enumerate(order):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = functions[name]()
            end.record()
            end.synchronize()
            elapsed = float(start.elapsed_time(end))
            del result
            samples[name].append(elapsed)
            round_times[name] = elapsed
            position_counts[name][position] += 1
        rounds.append({"order": order, "timings_ms": round_times})
    if measured >= len(orders):
        assert all(all(count > 0 for count in counts) for counts in position_counts.values())
    control = names[0]
    summaries = {name: summarize(values) for name, values in samples.items()}
    paired_speedups = {
        name: [
            round_result["timings_ms"][control] / round_result["timings_ms"][name]
            for round_result in rounds
        ]
        for name in names[1:]
    }
    return {
        "control": control,
        "summaries": summaries,
        "paired_speedups_vs_control": {
            name: {
                **summarize(values),
                "all_above_one": all(value > 1 for value in values),
            }
            for name, values in paired_speedups.items()
        },
        "position_counts": position_counts,
        "rounds": rounds,
    }


def profile_cuda_events(function, *, iterations: int) -> list[dict]:
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as kernel_profiler:
        for _ in range(iterations):
            result = function()
            del result
        torch.cuda.synchronize()
    kernel_events = []
    for event in kernel_profiler.key_averages():
        self_device_time = float(event.self_device_time_total)
        if self_device_time > 0:
            kernel_events.append(
                {
                    "key": event.key,
                    "count": event.count,
                    "self_device_time_total_us": self_device_time,
                    "device_time_total_us": float(event.device_time_total),
                }
            )
    kernel_events.sort(
        key=lambda event: event["self_device_time_total_us"], reverse=True
    )
    return kernel_events


def peak_allocated(function) -> int:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    result = function()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    return peak


def make_inputs(batch: int, length: int, heads: int, width: int):
    generator = torch.Generator(device="cuda").manual_seed(3407)
    shape = (batch, length, heads)
    q = torch.randn(*shape, width, generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(*shape, width, generator=generator, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(*shape, width, generator=generator, device="cuda", dtype=torch.bfloat16)
    g = -0.8 * torch.rand(*shape, generator=generator, device="cuda", dtype=torch.float32)
    beta = torch.rand(*shape, generator=generator, device="cuda", dtype=torch.float32)
    gamma = torch.rand(*shape, generator=generator, device="cuda", dtype=torch.float32)
    initial_state = torch.randn(
        batch,
        heads,
        width,
        width,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    return q, k, v, g, beta, gamma, initial_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=7)
    parser.add_argument("--candidate-measured", type=int, default=30)
    parser.add_argument("--kernel-profile-iterations", type=int, default=3)
    parser.add_argument("--skip-serial", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This diagnostic requires exactly one allocated CUDA GPU")
    if args.sequence_length % args.chunk_size:
        raise ValueError("the diagnostic shape must contain whole chunks")
    if args.measured < 3:
        raise ValueError("at least three measured samples are required")
    if args.kernel_profile_iterations < 1:
        raise ValueError("kernel profile iterations must be positive")
    if args.candidate_measured < 6:
        raise ValueError("candidate measurements must cover all six backend orders")

    numerics = configure_numerics()
    from lit_gpt.mixers.qgdn_rule import qgdn_rule
    from lit_gpt.mixers.qgdn_state_output_kernel import (
        _qgdn_chunk_state_cuda_fwd,
        _qgdn_chunk_state_output_cuda_bwd,
        _qgdn_chunk_state_output_cuda_bwd_serial,
        _qgdn_chunk_state_output_cuda_fwd,
    )
    from lit_gpt.mixers.qgdn_training_kernel import (
        _prepare_physical_chunks,
        qgdn_physical_training,
    )
    from lit_gpt.mixers.qgdn_wy_kernel import (
        _qgdn_streaming_wy_cuda_bwd,
        _qgdn_streaming_wy_cuda_fwd,
    )

    base = make_inputs(args.batch_size, args.sequence_length, args.heads, args.width)
    prepare_kwargs = {
        "recall_mode": "query",
        "update_order": "recall_then_delta",
        "chunk_size": args.chunk_size,
    }

    with torch.no_grad():
        prepared, _, _ = _prepare_physical_chunks(*base, **prepare_kwargs)
        prepared = tuple(tensor.contiguous() for tensor in prepared)
        effective_right, write_reads = _qgdn_streaming_wy_cuda_fwd(
            prepared[2], prepared[3], prepared[4], prepared[5]
        )
        effective_right = effective_right.contiguous()
        write_reads = write_reads.contiguous()
        outputs, final_state, chunk_starts, state_inputs = (
            _qgdn_chunk_state_output_cuda_fwd(
                prepared[0],
                prepared[1],
                prepared[2],
                effective_right,
                write_reads,
                prepared[4],
                prepared[5],
                prepared[6],
                args.width**-0.5,
            )
        )
        grad_outputs = torch.randn_like(outputs)
        grad_final_state = torch.randn_like(final_state)
        state_gradients = _qgdn_chunk_state_output_cuda_bwd(
            state_inputs,
            chunk_starts,
            grad_outputs,
            grad_final_state,
            args.width**-0.5,
        )
        wy_gradients = _qgdn_streaming_wy_cuda_bwd(
            prepared[2],
            prepared[3],
            prepared[4],
            prepared[5],
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

    recomputed = tuple(tensor.detach().requires_grad_(True) for tensor in base)
    with torch.enable_grad():
        differentiable_prepared, _, _ = _prepare_physical_chunks(
            *recomputed, **prepare_kwargs
        )

    physical_inputs = tuple(tensor.detach().requires_grad_(True) for tensor in base)
    virtual_inputs = tuple(tensor.detach().requires_grad_(True) for tensor in base)

    def prepare_forward():
        with torch.no_grad():
            return _prepare_physical_chunks(*base, **prepare_kwargs)[0]

    def wy_forward():
        return _qgdn_streaming_wy_cuda_fwd(
            prepared[2], prepared[3], prepared[4], prepared[5]
        )

    def state_scan_forward():
        return _qgdn_chunk_state_cuda_fwd(
            prepared[2],
            effective_right,
            write_reads,
            prepared[4],
            prepared[5],
            prepared[1],
            prepared[6],
        )

    def state_output_forward():
        return _qgdn_chunk_state_output_cuda_fwd(
            prepared[0],
            prepared[1],
            prepared[2],
            effective_right,
            write_reads,
            prepared[4],
            prepared[5],
            prepared[6],
            args.width**-0.5,
        )

    def state_output_backward():
        return _qgdn_chunk_state_output_cuda_bwd(
            state_inputs,
            chunk_starts,
            grad_outputs,
            grad_final_state,
            args.width**-0.5,
        )

    def state_output_backward_with_block(value_block):
        return _qgdn_chunk_state_output_cuda_bwd(
            state_inputs,
            chunk_starts,
            grad_outputs,
            grad_final_state,
            args.width**-0.5,
            value_block=value_block,
        )

    def state_output_backward_serial():
        return _qgdn_chunk_state_output_cuda_bwd_serial(
            state_inputs,
            chunk_starts,
            grad_outputs,
            grad_final_state,
            args.width**-0.5,
        )

    def wy_backward():
        return _qgdn_streaming_wy_cuda_bwd(
            prepared[2],
            prepared[3],
            prepared[4],
            prepared[5],
            state_gradients[3],
            state_gradients[4],
        )

    def prepared_input_vjp():
        return torch.autograd.grad(
            differentiable_prepared,
            recomputed,
            prepared_gradients,
            allow_unused=True,
            retain_graph=True,
        )

    def physical_rule_forward_backward():
        output, state = qgdn_physical_training(
            *physical_inputs[:6],
            recall_mode="query",
            update_order="recall_then_delta",
            scale=None,
            initial_state=physical_inputs[6],
            chunk_size=args.chunk_size,
        )
        objective = output.float().square().mean() + state.square().mean()
        gradients = torch.autograd.grad(objective, physical_inputs)
        return output, state, gradients

    def virtual_rule_forward_backward():
        output, state = qgdn_rule(
            *virtual_inputs[:6],
            initial_state=virtual_inputs[6],
            output_final_state=True,
            recall_mode="query",
            update_order="recall_then_delta",
        )
        objective = output.float().square().mean() + state.square().mean()
        gradients = torch.autograd.grad(objective, virtual_inputs)
        return output, state, gradients

    functions = {
        "prepare_forward": prepare_forward,
        "wy_forward": wy_forward,
        "state_scan_forward": state_scan_forward,
        "state_plus_output_forward": state_output_forward,
        "state_plus_output_backward": state_output_backward,
        "wy_backward": wy_backward,
        "prepared_input_vjp": prepared_input_vjp,
        "physical_rule_forward_backward": physical_rule_forward_backward,
        "virtual_rule_forward_backward": virtual_rule_forward_backward,
    }
    if not args.skip_serial:
        functions = {
            **dict(list(functions.items())[:4]),
            "state_plus_output_backward_serial": state_output_backward_serial,
            **dict(list(functions.items())[4:]),
        }
    timings = {
        name: measure(function, warmup=args.warmup, measured=args.measured)
        for name, function in functions.items()
    }
    value_block_functions = {
        f"bv{value_block}": (
            lambda value_block=value_block: state_output_backward_with_block(value_block)
        )
        for value_block in (16, 32, 64)
    }
    value_block_numerics = {}
    with torch.no_grad():
        control_gradients = state_output_backward_with_block(16)
        for name, function in value_block_functions.items():
            candidate_gradients = function()
            relative_rmse = []
            all_finite = True
            for candidate, control in zip(candidate_gradients, control_gradients):
                all_finite = all_finite and bool(candidate.isfinite().all())
                error = (
                    (candidate.float() - control.float()).square().mean().sqrt()
                    / control.float().square().mean().sqrt().clamp_min(1e-6)
                )
                relative_rmse.append(float(error))
            maximum_relative_rmse = max(relative_rmse)
            if not all_finite or maximum_relative_rmse >= 0.02:
                raise AssertionError(
                    f"{name} failed the prepared-gradient gate: "
                    f"finite={all_finite}, maximum_relative_rmse={maximum_relative_rmse}"
                )
            value_block_numerics[name] = {
                "all_prepared_gradients_finite": all_finite,
                "prepared_gradient_relative_rmse": relative_rmse,
                "maximum_prepared_gradient_relative_rmse": maximum_relative_rmse,
            }
            del candidate_gradients
        del control_gradients
    value_block_timings = measure_interleaved(
        value_block_functions,
        warmup=args.warmup,
        measured=args.candidate_measured,
    )
    value_block_peaks = {
        name: peak_allocated(function) for name, function in value_block_functions.items()
    }
    profiled_functions = {
        **value_block_functions,
        "wy_backward": wy_backward,
        "prepared_input_vjp": prepared_input_vjp,
    }
    kernel_profiles = {
        name: profile_cuda_events(
            function, iterations=args.kernel_profile_iterations
        )
        for name, function in profiled_functions.items()
    }
    chunks = args.sequence_length // args.chunk_size
    physical_ms = timings["physical_rule_forward_backward"]["median_ms"]
    virtual_ms = timings["virtual_rule_forward_backward"]["median_ms"]
    report = {
        "status": "passed",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numerics": numerics,
        "shape": {
            "batch": args.batch_size,
            "sequence_length": args.sequence_length,
            "heads": args.heads,
            "key_dim": args.width,
            "value_dim": args.width,
            "chunk_size": args.chunk_size,
            "chunks": chunks,
        },
        "structural_concurrency": {
            "wy_programs": args.batch_size * args.heads * chunks,
            "state_scan_programs": args.batch_size * args.heads * 2,
            "parallel_output_backward_programs": {
                str(value_block): (
                    args.batch_size
                    * args.heads
                    * chunks
                    * math.ceil(args.width / value_block)
                )
                for value_block in (16, 32, 64)
            },
            "compact_state_backward_programs": {
                str(value_block): (
                    args.batch_size
                    * args.heads
                    * math.ceil(args.width / value_block)
                )
                for value_block in (16, 32, 64)
            },
            "serial_chunks_per_state_program": chunks,
            "note": "the split backward makes output adjoints chunk-parallel; only the compact state adjoint loops over chunks",
        },
        "timings": timings,
        "value_block_audit": {
            "default_value_block": 32,
            "numerics_vs_bv16": value_block_numerics,
            "interleaved_timings": value_block_timings,
            "peak_allocated_bytes": value_block_peaks,
            "tokens_per_second": {
                name: (
                    args.batch_size
                    * args.sequence_length
                    * 1000
                    / summary["median_ms"]
                )
                for name, summary in value_block_timings["summaries"].items()
            },
        },
        "kernel_profiles": kernel_profiles,
        "physical_to_virtual_rule_time_ratio": physical_ms / virtual_ms,
        "virtual_to_physical_rule_speed_ratio": virtual_ms / physical_ms,
        "physical_rule_tokens_per_second": (
            args.batch_size * args.sequence_length * 1000 / physical_ms
        ),
        "virtual_rule_tokens_per_second": (
            args.batch_size * args.sequence_length * 1000 / virtual_ms
        ),
        "all_finite": bool(
            all(
                tensor.isfinite().all()
                for tensor in (
                    outputs,
                    final_state,
                    *state_gradients,
                    *wy_gradients,
                )
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
