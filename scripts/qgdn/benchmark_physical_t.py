"""Benchmark physical-token QGDN against the virtual-2T compatibility path."""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "qgdn"))


ORDER_CONFIGS = {
    "recall_then_delta": "qgdn_340M",
    "delta_then_recall": "qgdn_delta_then_recall_340M",
    "parallel": "qgdn_parallel_340M",
}
MODELS = {
    f"{backend}_{order}": (config, backend == "physical")
    for order, config in ORDER_CONFIGS.items()
    for backend in ("virtual", "physical")
}


def rotate(values: list[str], offset: int) -> list[str]:
    offset %= len(values)
    return values[offset:] + values[:offset]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--only", choices=tuple(MODELS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.only is None:
        if args.repeats < 2:
            raise ValueError("stable A/B requires at least two repeats")
        order_names = list(ORDER_CONFIGS)
        schedule = []
        runs = {
            order: {backend: [] for backend in ("virtual", "physical")}
            for order in order_names
        }
        failures = []
        reference_child = None
        for repeat in range(args.repeats):
            for order in rotate(order_names, repeat):
                backends = (
                    ("virtual", "physical")
                    if repeat % 2 == 0
                    else ("physical", "virtual")
                )
                for backend in backends:
                    name = f"{backend}_{order}"
                    child = args.output.with_name(
                        f".{args.output.stem}-r{repeat}-{name}.json"
                    )
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--output", str(child),
                        "--sequence-length", str(args.sequence_length),
                        "--micro-batch-size", str(args.micro_batch_size),
                        "--warmup", str(args.warmup),
                        "--measured", str(args.measured),
                        "--only", name,
                    ]
                    completed = subprocess.run(command, check=False)
                    schedule.append(
                        {
                            "repeat": repeat,
                            "order": order,
                            "backend": backend,
                            "returncode": completed.returncode,
                        }
                    )
                    if completed.returncode or not child.is_file():
                        failures.append(schedule[-1])
                        break
                    child_report = json.loads(child.read_text())
                    reference_child = reference_child or child_report
                    child_result = child_report["result"]
                    child_result["repeat"] = repeat
                    runs[order][backend].append(child_result)
                    child.unlink()
                if failures:
                    break
            if failures:
                break
        if reference_child is None:
            report = {
                "status": "failed",
                "sequence_length": args.sequence_length,
                "micro_batch_size": args.micro_batch_size,
                "warmup_steps": args.warmup,
                "measured_steps": args.measured,
                "repeats": args.repeats,
                "schedule": schedule,
                "failures": failures,
            }
            write_json(args.output, report)
            raise SystemExit(1)

        comparisons = {}
        for order in order_names:
            virtual = runs[order]["virtual"]
            physical = runs[order]["physical"]
            virtual_by_repeat = {result["repeat"]: result for result in virtual}
            physical_by_repeat = {result["repeat"]: result for result in physical}
            paired_repeats = sorted(virtual_by_repeat.keys() & physical_by_repeat.keys())
            speedups = [
                physical_by_repeat[repeat]["tokens_per_second"]
                / virtual_by_repeat[repeat]["tokens_per_second"]
                for repeat in paired_repeats
            ]
            memory_ratios = [
                physical_by_repeat[repeat]["peak_memory_gb"]
                / virtual_by_repeat[repeat]["peak_memory_gb"]
                for repeat in paired_repeats
            ]
            finite = all(
                result["finite"]
                for backend in ("virtual", "physical")
                for result in runs[order][backend]
            )
            comparisons[order] = {
                "paired_speedups": speedups,
                "median_speedup": statistics.median(speedups) if speedups else None,
                "minimum_speedup": min(speedups) if speedups else None,
                "paired_peak_memory_ratios": memory_ratios,
                "maximum_peak_memory_ratio": max(memory_ratios) if memory_ratios else None,
                "finite": finite,
                "paired_repeats": paired_repeats,
                "complete_pairs": len(paired_repeats),
                "passed": bool(
                    len(paired_repeats) == args.repeats
                    and finite
                    and min(speedups) > 1.25
                    and max(memory_ratios) <= 1.0
                ),
            }
        passed = not failures and all(value["passed"] for value in comparisons.values())
        report = {
            "status": "passed" if passed else "measured_below_gate",
            "commit": reference_child["commit"],
            "device": reference_child["device"],
            "torch": reference_child["torch"],
            "cuda": reference_child["cuda"],
            "sequence_length": args.sequence_length,
            "micro_batch_size": args.micro_batch_size,
            "warmup_steps": args.warmup,
            "measured_steps": args.measured,
            "repeats": args.repeats,
            "activation_checkpointing": False,
            "loss_implementation": "fused",
            "process_isolation": True,
            "pairing": "same update-order and repeat; order rotates, backend order alternates by repeat",
            "gate": {
                "minimum_speedup_strictly_above": 1.25,
                "maximum_peak_memory_ratio": 1.0,
                "all_training_values_finite": True,
            },
            "schedule": schedule,
            "failures": failures,
            "runs": runs,
            "comparisons": comparisons,
        }
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if failures:
            raise SystemExit(1)
        return

    # Keep the orchestration parent completely CUDA-free.  The micro-batch-8
    # virtual baseline sits close to the 80-GB limit, so even a parent's idle
    # CUDA context can invalidate the same-card comparison.
    from runtime import configure_device_from_cli, configure_numerics

    configure_device_from_cli()
    import torch
    from benchmark_training_speed import benchmark_model
    from lit_gpt.config import Config

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This benchmark requires exactly one allocated CUDA GPU")
    configure_numerics(cpu=False)
    torch.manual_seed(117)
    model_name, physical = MODELS[args.only]
    qgdn_rule = importlib.import_module("lit_gpt.mixers.qgdn_rule")
    qgdn_rule.QGDN_USE_PHYSICAL_T = physical
    config = Config.from_name(model_name, block_size=args.sequence_length)
    tokens = torch.randint(
        0,
        config.padded_vocab_size,
        (args.micro_batch_size, args.sequence_length),
        device="cuda",
    )
    targets = torch.roll(tokens, shifts=-1, dims=1)
    result = benchmark_model(
        model_name,
        tokens,
        targets,
        args.warmup,
        args.measured,
        activation_checkpointing=False,
        loss_implementation="fused",
        qgdn_chunk_size=32,
        compile_qgdn_inputs=True,
    )
    report = {
        "status": "measured_child",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "candidate": args.only,
        "physical_t": physical,
        "physical_t_chunk_size": qgdn_rule.QGDN_PHYSICAL_T_CHUNK_SIZE,
        "physical_t_recompute_prepared_tensors": (
            qgdn_rule.QGDN_RECOMPUTE_PHYSICAL_T_PREPARED_TENSORS
        ),
        "result": result,
    }
    write_json(args.output, report)


if __name__ == "__main__":
    main()
