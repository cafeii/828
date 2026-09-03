"""Benchmark the old and optimized QGDN training configurations on one node."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


CASES = (
    {
        "name": "baseline-b1-checkpoint-torch",
        "micro_batch_size": 1,
        "activation_checkpointing": True,
        "training_loss": "torch",
    },
    {
        "name": "optimized-b8-no-checkpoint-fused",
        "micro_batch_size": 8,
        "activation_checkpointing": False,
        "training_loss": "fused",
    },
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run_logged(command: list[str], stdout_path: Path, stderr_path: Path) -> tuple[int, float]:
    started = time.perf_counter()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        result = subprocess.run(command, text=True, stdout=stdout, stderr=stderr)
    return result.returncode, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--only",
        choices=tuple(config["name"] for config in CASES),
        help="Run one configuration when a focused repeat is sufficient",
    )
    args = parser.parse_args()
    if args.steps < 2 or args.log_every < 1:
        parser.error("steps must be at least 2 and log-every must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    train = Path(__file__).with_name("train.py")
    validate_loss = Path(__file__).with_name("validate_fused_loss.py")
    report_path = args.output_dir / "ddp-scaling.json"
    report = {
        "status": "running",
        "world_size": args.world_size,
        "steps": args.steps,
        "global_batch_size": args.global_batch_size,
        "sequence_length": args.sequence_length,
        "log_every": args.log_every,
        "selected_case": args.only,
        "cases": [],
    }
    write_json(report_path, report)

    parity_output = args.output_dir / "fused-loss-parity.json"
    parity_rc, parity_seconds = run_logged(
        [sys.executable, str(validate_loss), "--output", str(parity_output)],
        args.output_dir / "fused-loss-parity.out",
        args.output_dir / "fused-loss-parity.err",
    )
    report["fused_loss_validation"] = {
        "returncode": parity_rc,
        "seconds": parity_seconds,
        "result": json.loads(parity_output.read_text()) if parity_rc == 0 else None,
    }
    write_json(report_path, report)
    if parity_rc != 0:
        report["status"] = "failed_fused_loss_validation"
        write_json(report_path, report)
        raise RuntimeError("Fused-loss validation failed")

    selected_cases = [
        config for config in CASES if args.only is None or config["name"] == args.only
    ]
    for config in selected_cases:
        run_output = args.output_dir / config["name"]
        command = [
            "torchrun",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={args.world_size}",
            str(train),
            "--model",
            "qgdn_340M",
            "--output",
            str(run_output),
            "--task",
            "mqar",
            "--max-steps",
            str(args.steps),
            "--sequence-length",
            str(args.sequence_length),
            "--global-batch-size",
            str(args.global_batch_size),
            "--micro-batch-size",
            str(config["micro_batch_size"]),
            "--eval-every",
            str(args.steps),
            "--eval-sequences",
            str(args.world_size),
            "--save-every",
            str(args.steps),
            "--log-every",
            str(args.log_every),
            "--training-loss",
            config["training_loss"],
        ]
        if not config["activation_checkpointing"]:
            command.append("--no-activation-checkpointing")
        returncode, process_seconds = run_logged(
            command,
            args.output_dir / f"{config['name']}.out",
            args.output_dir / f"{config['name']}.err",
        )
        case = dict(config, returncode=returncode, process_seconds=process_seconds)
        summary_path = run_output / "summary.json"
        metrics_path = run_output / "metrics.jsonl"
        if returncode == 0 and summary_path.exists() and metrics_path.exists():
            summary = json.loads(summary_path.read_text())
            train_metrics = [
                json.loads(line)
                for line in metrics_path.read_text().splitlines()
                if json.loads(line)["kind"] == "train"
            ]
            first_step_seconds = train_metrics[0]["step_seconds"]
            steady_steps = args.steps - 1
            steady_seconds = summary["train_seconds"] - first_step_seconds
            steady_tps = (
                steady_steps * args.global_batch_size * args.sequence_length
                / steady_seconds
            )
            steady_metrics = [row for row in train_metrics if row["step"] > 1]
            case.update(
                status="completed",
                peak_memory_gb=summary["peak_memory_gb"],
                train_seconds=summary["train_seconds"],
                wall_seconds=summary["wall_seconds"],
                first_step_seconds=first_step_seconds,
                steady_steps=steady_steps,
                steady_seconds=steady_seconds,
                steady_tokens_per_second=steady_tps,
                logged_steady_tokens_per_second_median=(
                    statistics.median(row["tokens_per_second"] for row in steady_metrics)
                    if steady_metrics
                    else None
                ),
                projected_10b_hours=10e9 / steady_tps / 3600,
                projected_15b_hours=15e9 / steady_tps / 3600,
                finite=all(
                    all(math.isfinite(row[key]) for key in ("loss", "grad_norm", "tokens_per_second"))
                    for row in train_metrics
                ),
            )
        else:
            case["status"] = "failed"
            case["stderr_tail"] = (
                args.output_dir / f"{config['name']}.err"
            ).read_text()[-8000:]
        report["cases"].append(case)
        write_json(report_path, report)

    completed = [case for case in report["cases"] if case["status"] == "completed"]
    report["status"] = "completed" if len(completed) == len(selected_cases) else "failed"
    if completed:
        fastest = max(completed, key=lambda case: case["steady_tokens_per_second"])
        report["fastest_case"] = fastest["name"]
        report["fastest_tokens_per_second"] = fastest["steady_tokens_per_second"]
        report["fastest_projected_10b_hours"] = fastest["projected_10b_hours"]
        report["fastest_projected_15b_hours"] = fastest["projected_15b_hours"]
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if report["status"] != "completed":
        raise RuntimeError("One or more DDP benchmark cases failed")


if __name__ == "__main__":
    main()
