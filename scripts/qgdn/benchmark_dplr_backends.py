"""Audit safe DPLR backends and benchmark 340M QGDN training steps."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "qgdn"))

import torch

from benchmark_training_speed import benchmark_model
from lit_gpt.mixers import qgdn_rule as rule_module
from runtime import configure_numerics


CASES = {
    "triton_chunk32": {"tilelang": False, "chunk_size": 32, "lower_bound": None},
    "tilelang_chunk16": {"tilelang": True, "chunk_size": 16, "lower_bound": -9.0},
    "tilelang_chunk32": {"tilelang": True, "chunk_size": 32, "lower_bound": -5.0},
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = (actual.float() - expected.float()).square().mean().sqrt()
    denominator = expected.float().square().mean().sqrt().clamp_min(1e-7)
    return float((numerator / denominator).item())


def run_rule(case: str, values: tuple[torch.Tensor, ...], weights):
    settings = CASES[case]
    os.environ["FLA_TILELANG"] = "1" if settings["tilelang"] else "0"
    rule_module.QGDN_TRAIN_CHUNK_SIZE = settings["chunk_size"]
    rule_module.QGDN_DPLR_LOWER_BOUND = settings["lower_bound"]
    cloned = tuple(value.detach().clone().requires_grad_() for value in values)
    output, state = rule_module.qgdn_rule(
        *cloned[:6],
        recall_mode="query",
        update_order="recall_then_delta",
        mode="chunk",
        initial_state=cloned[6],
        output_final_state=True,
    )
    gradients = torch.autograd.grad(
        (output.float() * weights[0]).sum() + (state.float() * weights[1]).sum(),
        cloned,
    )
    torch.cuda.synchronize()
    return (output.detach(), state.detach()), tuple(value.detach() for value in gradients)


def validate_backends() -> dict:
    generator = torch.Generator(device="cuda").manual_seed(3407)
    B, T, H, K, V = 2, 128, 2, 64, 64
    values = (
        torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, generator=generator),
        torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, generator=generator),
        torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16, generator=generator),
        -(0.001 + 0.1 * torch.rand(B, T, H, device="cuda", generator=generator)),
        torch.rand(B, T, H, device="cuda", generator=generator),
        torch.rand(B, T, H, device="cuda", generator=generator),
        torch.randn(B, H, K, V, device="cuda", dtype=torch.float32, generator=generator),
    )
    weights = (
        torch.randn(B, T, H, V, device="cuda", dtype=torch.float32, generator=generator),
        torch.randn(B, H, K, V, device="cuda", dtype=torch.float32, generator=generator),
    )
    reference = run_rule("triton_chunk32", values, weights)
    report = {}
    for case in ("tilelang_chunk16", "tilelang_chunk32"):
        actual = run_rule(case, values, weights)
        report[case] = {
            "output_relative_rmse": relative_rmse(actual[0][0], reference[0][0]),
            "state_relative_rmse": relative_rmse(actual[0][1], reference[0][1]),
            "gradient_relative_rmse": [
                relative_rmse(value, expected)
                for value, expected in zip(actual[1], reference[1])
            ],
            "finite": bool(all(value.isfinite().all() for group in actual for value in group)),
        }
    return report


def child(args) -> None:
    settings = CASES[args.only]
    os.environ["FLA_TILELANG"] = "1" if settings["tilelang"] else "0"
    rule_module.QGDN_DPLR_LOWER_BOUND = settings["lower_bound"]
    torch.manual_seed(3407)
    tokens = torch.randint(0, 50304, (args.micro_batch_size, args.sequence_length), device="cuda")
    targets = torch.randint(0, 50304, (args.micro_batch_size, args.sequence_length), device="cuda")
    model = benchmark_model(
        "qgdn_340M",
        tokens,
        targets,
        args.warmup,
        args.measured,
        activation_checkpointing=False,
        loss_implementation="fused",
        qgdn_chunk_size=settings["chunk_size"],
        compile_qgdn_inputs=True,
    )
    write_json(args.output, {
        "case": args.only,
        "settings": settings,
        "model": model,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument("--only", choices=tuple(CASES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This benchmark requires exactly one allocated CUDA GPU")
    configure_numerics(cpu=False)
    if args.only:
        child(args)
        return

    validation = validate_backends()
    children = {}
    for case in CASES:
        child_output = args.output.with_name(f".{args.output.stem}-{case}.json")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output", str(child_output),
            "--sequence-length", str(args.sequence_length),
            "--micro-batch-size", str(args.micro_batch_size),
            "--warmup", str(args.warmup),
            "--measured", str(args.measured),
            "--only", case,
        ]
        subprocess.run(command, check=True)
        children[case] = json.loads(child_output.read_text())
        child_output.unlink()

    models = {case: value["model"] for case, value in children.items()}
    baseline = models["triton_chunk32"]["tokens_per_second"]
    report = {
        "status": "measured",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "device": torch.cuda.get_device_name(0),
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "validation": validation,
        "models": models,
        "speedups_vs_triton_chunk32": {
            case: model["tokens_per_second"] / baseline
            for case, model in models.items()
        },
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
