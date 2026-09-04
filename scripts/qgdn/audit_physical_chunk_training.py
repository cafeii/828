"""Audit the opt-in fused physical-T training path at the 340M operator shape."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

from runtime import configure_device_from_cli, configure_numerics

if __name__ == "__main__":
    configure_device_from_cli()

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))


UPDATE_ORDERS = ("recall_then_delta", "delta_then_recall", "parallel")
INPUT_NAMES = ("q", "k", "v", "g", "beta", "gamma", "initial_state")


def relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float()
    expected = expected.float()
    denominator = expected.square().mean().sqrt().clamp_min(1e-6)
    return float(((actual - expected).square().mean().sqrt() / denominator).item())


def make_inputs(sequence_length: int, *, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch, heads, width = 1, 16, 64
    shape = (batch, sequence_length, heads)
    q = torch.randn(*shape, width, generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(*shape, width, generator=generator, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(*shape, width, generator=generator, device="cuda", dtype=torch.bfloat16)
    g = -0.8 * torch.rand(*shape, generator=generator, device="cuda", dtype=torch.float32)
    beta = torch.rand(*shape, generator=generator, device="cuda", dtype=torch.float32)
    gamma = torch.rand(*shape, generator=generator, device="cuda", dtype=torch.float32)
    initial_state = torch.randn(
        batch, heads, width, width, generator=generator, device="cuda", dtype=torch.float32
    )
    return [q, k, v, g, beta, gamma, initial_state]


def run_arm(
    base: list[torch.Tensor],
    weights: tuple[torch.Tensor, torch.Tensor],
    *,
    update_order: str,
    physical: bool,
) -> dict:
    rule_module = importlib.import_module("lit_gpt.mixers.qgdn_rule")
    rule_module.QGDN_USE_PHYSICAL_T = physical
    inputs = [value.detach().clone().requires_grad_() for value in base]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output, final_state = rule_module.qgdn_rule(
        *inputs[:6],
        initial_state=inputs[6],
        output_final_state=True,
        recall_mode="query",
        update_order=update_order,
    )
    objective = (output.float() * weights[0]).sum() + (final_state * weights[1]).sum()
    gradients = torch.autograd.grad(objective, inputs)
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()
    finite = all(value.isfinite().all() for value in (output, final_state, *gradients))
    result = {
        "output": output.detach().float().cpu(),
        "final_state": final_state.detach().float().cpu(),
        "gradients": [value.detach().float().cpu() for value in gradients],
        "finite": bool(finite),
        "peak_memory_bytes": peak_memory,
    }
    del inputs, output, final_state, objective, gradients
    torch.cuda.empty_cache()
    return result


def compare_order(sequence_length: int, update_order: str, *, seed: int) -> dict:
    base = make_inputs(sequence_length, seed=seed)
    weight_generator = torch.Generator(device="cuda").manual_seed(seed + 10000)
    weights = (
        torch.randn(
            1,
            sequence_length,
            16,
            64,
            generator=weight_generator,
            device="cuda",
            dtype=torch.float32,
        ),
        torch.randn(
            1,
            16,
            64,
            64,
            generator=weight_generator,
            device="cuda",
            dtype=torch.float32,
        ),
    )
    virtual = run_arm(base, weights, update_order=update_order, physical=False)
    physical = run_arm(base, weights, update_order=update_order, physical=True)
    output_error = relative_rmse(physical["output"], virtual["output"])
    state_error = relative_rmse(physical["final_state"], virtual["final_state"])
    gradient_errors = {
        name: relative_rmse(actual, expected)
        for name, actual, expected in zip(
            INPUT_NAMES, physical["gradients"], virtual["gradients"]
        )
    }
    return {
        "update_order": update_order,
        "finite": virtual["finite"] and physical["finite"],
        "output_relative_rmse": output_error,
        "final_state_relative_rmse": state_error,
        "gradient_relative_rmse": gradient_errors,
        "max_gradient_relative_rmse": max(gradient_errors.values()),
        "peak_memory_bytes": {
            "virtual_2t": virtual["peak_memory_bytes"],
            "physical_t": physical["peak_memory_bytes"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This audit requires exactly one allocated CUDA GPU")
    numerics = configure_numerics()
    cases = [
        compare_order(args.sequence_length, update_order, seed=6100 + index)
        for index, update_order in enumerate(UPDATE_ORDERS)
    ]
    thresholds = {
        "output_relative_rmse": 0.025,
        "final_state_relative_rmse": 0.025,
        "gradient_relative_rmse": 0.07,
    }
    passed = all(
        case["finite"]
        and math.isfinite(case["output_relative_rmse"])
        and math.isfinite(case["final_state_relative_rmse"])
        and math.isfinite(case["max_gradient_relative_rmse"])
        and case["output_relative_rmse"] < thresholds["output_relative_rmse"]
        and case["final_state_relative_rmse"] < thresholds["final_state_relative_rmse"]
        and case["max_gradient_relative_rmse"] < thresholds["gradient_relative_rmse"]
        for case in cases
    )
    report = {
        "status": "passed" if passed else "failed",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numerics": numerics,
        "shape": {
            "batch": 1,
            "sequence_length": args.sequence_length,
            "heads": 16,
            "key_dim": 64,
            "value_dim": 64,
            "qkv_dtype": "bfloat16",
            "gate_state_dtype": "float32",
        },
        "reference": "production virtual 2T DPLR path",
        "candidate": "physical-T chunk-16 fused rank-2 WY/state/output path",
        "thresholds": thresholds,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
