"""Profile exact QGDN DPLR chunk sizes on a representative mixer shape."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from lit_gpt.kernels import get_chunk_gated_delta_rule
from lit_gpt.mixers.qgdn_rule import dplr_inputs


def make_inputs(*, batch: int, length: int, heads: int, key_dim: int, value_dim: int):
    generator = torch.Generator(device="cuda").manual_seed(3407)
    q = torch.randn(batch, length, heads, key_dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(q.shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(batch, length, heads, value_dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    # Cover the observed operating range while including a conservative tail.
    g = -(0.001 + 3.999 * torch.rand(batch, length, heads, device="cuda", generator=generator))
    beta = torch.rand(batch, length, heads, device="cuda", generator=generator)
    gamma = torch.rand(batch, length, heads, device="cuda", generator=generator)
    return q, k, v, g, beta, gamma


def clone_for_grad(inputs):
    return tuple(value.detach().clone().requires_grad_() for value in inputs)


def qgdn_call(inputs, *, chunk_size: int, output_final_state: bool):
    from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule

    virtual = dplr_inputs(*inputs)
    output, state = chunk_dplr_delta_rule(
        **virtual,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )
    return output[:, 1::2].contiguous(), state


def gdn_call(inputs, *, output_final_state: bool):
    q, k, v, g, beta, _ = inputs
    return get_chunk_gated_delta_rule()(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
    )


def loss_and_grad(call, inputs):
    output, _ = call(inputs, output_final_state=False)
    weight = torch.sin(torch.arange(output.numel(), device=output.device, dtype=torch.float32)).reshape(output.shape)
    loss = (output.float() * weight).mean()
    gradients = torch.autograd.grad(loss, inputs)
    return output, gradients


def relative_rmse(actual, expected):
    numerator = (actual.float() - expected.float()).square().mean().sqrt()
    denominator = expected.float().square().mean().sqrt().clamp_min(1e-7)
    return float((numerator / denominator).item())


def timed_step(call, base_inputs, warmup: int, measured: int):
    def run():
        inputs = clone_for_grad(base_inputs)
        output, _ = call(inputs, output_final_state=False)
        loss = output.float().square().mean()
        # GDN intentionally does not consume the QGDN-only gamma input.
        torch.autograd.grad(loss, inputs, allow_unused=True)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    peak = 0
    elapsed = []
    for _ in range(measured):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))
        peak = max(peak, torch.cuda.max_memory_allocated())
    return {
        "milliseconds": elapsed,
        "mean_milliseconds": statistics.mean(elapsed),
        "median_milliseconds": statistics.median(elapsed),
        "peak_memory_gb": peak / 1e9,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.use_deterministic_algorithms(True)
    base = make_inputs(
        batch=args.batch,
        length=args.length,
        heads=args.heads,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
    )

    validation_inputs = clone_for_grad(base)
    reference_output, reference_state = qgdn_call(
        validation_inputs, chunk_size=16, output_final_state=True
    )
    output_weight = torch.randn_like(reference_output, dtype=torch.float32)
    state_weight = torch.randn_like(reference_state, dtype=torch.float32)
    reference_gradients = torch.autograd.grad(
        (reference_output.float() * output_weight).mean()
        + (reference_state.float() * state_weight).mean(),
        validation_inputs,
    )

    results = {}
    for chunk_size in (16, 32, 64):
        inputs = clone_for_grad(base)
        output, state = qgdn_call(inputs, chunk_size=chunk_size, output_final_state=True)
        gradients = torch.autograd.grad(
            (output.float() * output_weight).mean() + (state.float() * state_weight).mean(),
            inputs,
        )
        validation = {
            "finite": bool(
                output.isfinite().all()
                and state.isfinite().all()
                and all(value.isfinite().all() for value in gradients)
            ),
            "output_relative_rmse_vs_chunk16": relative_rmse(output, reference_output),
            "state_relative_rmse_vs_chunk16": relative_rmse(state, reference_state),
            "gradient_relative_rmse_vs_chunk16": [
                relative_rmse(value, reference)
                for value, reference in zip(gradients, reference_gradients)
            ],
        }
        results[f"qgdn_chunk{chunk_size}"] = {
            "validation": validation,
            "timing": timed_step(
                lambda values, output_final_state=False, size=chunk_size: qgdn_call(
                    values, chunk_size=size, output_final_state=output_final_state
                ),
                base,
                args.warmup,
                args.measured,
            ),
        }

    results["gdn"] = {
        "timing": timed_step(gdn_call, base, args.warmup, args.measured),
    }
    q16 = results["qgdn_chunk16"]["timing"]["median_milliseconds"]
    for name, result in results.items():
        timing = result["timing"]
        timing["speedup_vs_qgdn_chunk16"] = q16 / timing["median_milliseconds"]
        timing["tokens_per_second"] = args.batch * args.length / (timing["median_milliseconds"] / 1000)

    report = {
        "status": "measured",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shape": {
            "batch": args.batch,
            "length": args.length,
            "heads": args.heads,
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
        },
        "gate_range": {"g_min": float(base[3].min()), "g_max": float(base[3].max())},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
