"""Measure QGDN micro-batch, checkpointing, and loss-kernel scaling.

Every case runs in a fresh process so CUDA allocator state and OOM failures do
not contaminate later measurements.  This is a throughput diagnostic, not a
training or quality experiment.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_CASES = (
    ("baseline-b1-checkpoint-torch", 1, True, "torch"),
    ("b1-checkpoint-fused", 1, True, "fused"),
    ("b1-no-checkpoint-torch", 1, False, "torch"),
    ("b2-no-checkpoint-torch", 2, False, "torch"),
    ("b4-no-checkpoint-torch", 4, False, "torch"),
    ("b8-no-checkpoint-torch", 8, False, "torch"),
    ("b1-no-checkpoint-fused", 1, False, "fused"),
    ("b2-no-checkpoint-fused", 2, False, "fused"),
    ("b4-no-checkpoint-fused", 4, False, "fused"),
    ("b8-no-checkpoint-fused", 8, False, "fused"),
    ("b2-checkpoint-fused", 2, True, "fused"),
    ("b4-checkpoint-fused", 4, True, "fused"),
    ("b8-checkpoint-fused", 8, True, "fused"),
    ("b16-checkpoint-fused", 16, True, "fused"),
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measured", type=int, default=10)
    args = parser.parse_args()

    benchmark = Path(__file__).with_name("benchmark_training_speed.py")
    cases = []
    for name, micro_batch, checkpointing, loss_implementation in DEFAULT_CASES:
        child_output = args.output.with_name(f".{args.output.stem}-{name}.json")
        command = [
            sys.executable,
            str(benchmark),
            "--output",
            str(child_output),
            "--sequence-length",
            str(args.sequence_length),
            "--micro-batch-size",
            str(micro_batch),
            "--warmup",
            str(args.warmup),
            "--measured",
            str(args.measured),
            "--loss-implementation",
            loss_implementation,
            "--only",
            "qgdn_compiled_inputs",
        ]
        if not checkpointing:
            command.append("--no-activation-checkpointing")
        started = time.perf_counter()
        completed = subprocess.run(command, text=True, capture_output=True)
        case = {
            "name": name,
            "micro_batch_size": micro_batch,
            "activation_checkpointing": checkpointing,
            "loss_implementation": loss_implementation,
            "process_seconds": time.perf_counter() - started,
        }
        if completed.returncode == 0 and child_output.exists():
            child = json.loads(child_output.read_text())
            model = child["model"]
            throughput = model["tokens_per_second"]
            case.update(
                status="completed",
                tokens_per_second=throughput,
                mean_step_seconds=model["mean_step_seconds"],
                median_step_seconds=model["median_step_seconds"],
                peak_memory_gb=model["peak_memory_gb"],
                projected_10b_hours_8gpu_ideal=10e9 / (8 * throughput) / 3600,
                projected_15b_hours_8gpu_ideal=15e9 / (8 * throughput) / 3600,
                finite=model["finite"],
                commit=child["commit"],
                device=child["device"],
                torch=child["torch"],
                cuda=child["cuda"],
                numerics=child["numerics"],
            )
            child_output.unlink()
        else:
            case.update(
                status="failed",
                returncode=completed.returncode,
                stdout_tail=completed.stdout[-4000:],
                stderr_tail=completed.stderr[-8000:],
            )
            child_output.unlink(missing_ok=True)
        cases.append(case)
        write_json(args.output, {"status": "running", "cases": cases})

    completed_cases = [case for case in cases if case["status"] == "completed"]
    fastest = max(completed_cases, key=lambda case: case["tokens_per_second"])
    report = {
        "status": "completed",
        "sequence_length": args.sequence_length,
        "warmup_steps": args.warmup,
        "measured_steps": args.measured,
        "process_isolation": True,
        "cases": cases,
        "fastest_case": fastest["name"],
        "fastest_tokens_per_second": fastest["tokens_per_second"],
        "fastest_peak_memory_gb": fastest["peak_memory_gb"],
        "fastest_projected_10b_hours_8gpu_ideal": fastest[
            "projected_10b_hours_8gpu_ideal"
        ],
        "fastest_projected_15b_hours_8gpu_ideal": fastest[
            "projected_15b_hours_8gpu_ideal"
        ],
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
