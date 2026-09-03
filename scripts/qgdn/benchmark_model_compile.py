"""Compare eager and full-model-compiled QGDN training in fresh processes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run_case(
    name: str,
    output_dir: Path,
    sequence_length: int,
    micro_batch_size: int,
    warmup: int,
    measured: int,
    *,
    compile_model_forward: bool,
) -> dict:
    output = output_dir / f"{name}.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_training_speed.py")),
        "--output",
        str(output),
        "--sequence-length",
        str(sequence_length),
        "--micro-batch-size",
        str(micro_batch_size),
        "--warmup",
        str(warmup),
        "--measured",
        str(measured),
        "--no-activation-checkpointing",
        "--loss-implementation",
        "fused",
        "--only",
        "qgdn_compiled_inputs",
    ]
    if compile_model_forward:
        command.append("--compile-model-forward")
    with (output_dir / f"{name}.out").open("w") as stdout, (
        output_dir / f"{name}.err"
    ).open("w") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
    result = {
        "name": name,
        "returncode": completed.returncode,
        "compile_model_forward": compile_model_forward,
    }
    if completed.returncode == 0 and output.exists():
        result["status"] = "completed"
        result["report"] = json.loads(output.read_text())
    else:
        result["status"] = "failed"
        result["stderr_tail"] = (output_dir / f"{name}.err").read_text()[-8000:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measured", type=int, default=8)
    parser.add_argument("--acceptance-speedup", type=float, default=1.05)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    eager = run_case(
        "eager",
        args.output_dir,
        args.sequence_length,
        args.micro_batch_size,
        args.warmup,
        args.measured,
        compile_model_forward=False,
    )
    compiled = run_case(
        "compiled",
        args.output_dir,
        args.sequence_length,
        args.micro_batch_size,
        args.warmup,
        args.measured,
        compile_model_forward=True,
    )
    report = {
        "status": "completed",
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "activation_checkpointing": False,
        "loss_implementation": "fused",
        "warmup_steps": args.warmup,
        "measured_steps": args.measured,
        "acceptance_speedup": args.acceptance_speedup,
        "cases": {"eager": eager, "compiled": compiled},
        "candidate_accepted": False,
    }
    if eager["status"] == compiled["status"] == "completed":
        eager_model = eager["report"]["model"]
        compiled_model = compiled["report"]["model"]
        report["mean_throughput_speedup"] = (
            compiled_model["tokens_per_second"] / eager_model["tokens_per_second"]
        )
        report["median_step_speedup"] = (
            eager_model["median_step_seconds"] / compiled_model["median_step_seconds"]
        )
        report["peak_memory_delta_gb"] = (
            compiled_model["peak_memory_gb"] - eager_model["peak_memory_gb"]
        )
        report["candidate_accepted"] = (
            report["mean_throughput_speedup"] >= args.acceptance_speedup
            and compiled_model["finite"]
        )
    else:
        report["status"] = "candidate_failed"
    write_json(args.output_dir / "model-compile.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
