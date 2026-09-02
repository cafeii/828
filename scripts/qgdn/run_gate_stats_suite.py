"""Validate gate-stat aggregation and evaluate several formal endpoint checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=RUN_DIR",
        help="Formal run directory containing run.json, summary.json and checkpoint.pt",
    )
    return parser.parse_args()


def parse_runs(entries):
    result = []
    for entry in entries:
        label, separator, directory = entry.partition("=")
        if not separator or not label or not directory:
            raise ValueError(f"Invalid --run value: {entry!r}")
        result.append((label, Path(directory)))
    if len(result) > torch_visible_device_count():
        raise ValueError("Every endpoint evaluator needs a distinct allocated GPU")
    if len({label for label, _ in result}) != len(result):
        raise ValueError("Run labels must be unique")
    return result


def torch_visible_device_count():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return len([item for item in visible.split(",") if item.strip()]) if visible else 0


def finalize(raw):
    result = {}
    for name, moments in sorted(raw.items()):
        total = moments["sum"]
        square_total = moments["sum_of_squares"]
        count = moments["count"]
        mean = total / count
        variance = max(0.0, square_total / count - mean * mean)
        result[name] = {**moments, "mean": mean, "std": math.sqrt(variance)}
    return result


def pooled_reports(reports):
    pooled = {}
    for report in reports.values():
        family = "qgdn" if report["model"].startswith("qgdn") else "gdn"
        raw = pooled.setdefault(family, {})
        for name, gate in report["gates"].items():
            target = raw.setdefault(name, {"sum": 0.0, "sum_of_squares": 0.0, "count": 0})
            target["sum"] += gate["sum"]
            target["sum_of_squares"] += gate["sum_of_squares"]
            target["count"] += gate["count"]
    return {family: finalize(raw) for family, raw in pooled.items()}


def markdown(report):
    lines = [
        "# Endpoint gate statistics",
        "",
        "Population statistics over the fixed 2,560-sequence validation set at each completed formal endpoint.",
        "",
        "| seed | model | gate | count | mean | population std |",
        "|---:|---|---|---:|---:|---:|",
    ]
    order = ("alpha", "beta", "gamma", "forgetting_margin", "gamma_saturated")
    for label, endpoint in report["endpoints"].items():
        for name in order:
            if name in endpoint["gates"]:
                gate = endpoint["gates"][name]
                lines.append(
                    f"| {endpoint['seed']} | {endpoint['model']} | {name} | {gate['count']} | "
                    f"{gate['mean']:.9f} | {gate['std']:.9f} |"
                )
    lines += ["", "## Pooled across the two completed seeds", "", "| model | gate | count | mean | population std |", "|---|---|---:|---:|---:|"]
    for family, gates in report["pooled_by_model"].items():
        for name in order:
            if name in gates:
                gate = gates[name]
                lines.append(f"| {family} | {name} | {gate['count']} | {gate['mean']:.9f} | {gate['std']:.9f} |")
    lines += [
        "",
        "All standard deviations were computed after merging FP64 sum, sum-of-squares and count. "
        "Endpoint evaluation was read-only and reproduced each formal validation loss.",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    runs = parse_runs(args.run)
    if torch_visible_device_count() < 2:
        raise RuntimeError("At least two Slurm-allocated GPUs are required")
    args.output.mkdir(parents=True, exist_ok=True)

    ddp_output = args.output / "gate-stats-ddp-validation.json"
    ddp_env = os.environ.copy()
    ddp_env["CUDA_VISIBLE_DEVICES"] = ",".join(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[:2])
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            "scripts/qgdn/validate_gate_stats_ddp.py",
            "--output",
            str(ddp_output),
        ],
        cwd=ROOT,
        env=ddp_env,
        check=True,
    )

    processes = []
    physical_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    for index, (label, run_dir) in enumerate(runs):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = physical_devices[index]
        command = [
            sys.executable,
            "-u",
            "scripts/qgdn/evaluate_gate_stats.py",
            "--checkpoint",
            str(run_dir / "checkpoint.pt"),
            "--run-json",
            str(run_dir / "run.json"),
            "--summary",
            str(run_dir / "summary.json"),
            "--data-manifest",
            str(args.data_manifest),
            "--output-json",
            str(args.output / f"gate-stats-{label}.json"),
            "--output-markdown",
            str(args.output / f"gate-stats-{label}.md"),
        ]
        print(json.dumps({"launch": label, "gpu": physical_devices[index], "run": str(run_dir)}), flush=True)
        processes.append((label, subprocess.Popen(command, cwd=ROOT, env=environment)))
    failures = [(label, process.wait()) for label, process in processes if process.wait() != 0]
    if failures:
        raise RuntimeError(f"Endpoint gate-stat evaluation failed: {failures}")

    endpoints = {}
    for label, _ in runs:
        endpoint = json.loads((args.output / f"gate-stats-{label}.json").read_text())
        if endpoint["status"] != "passed" or not endpoint["observation_is_bitwise_noninvasive"]:
            raise RuntimeError(f"Endpoint validation did not pass: {label}")
        endpoints[label] = endpoint
    report = {
        "status": "passed",
        "ddp_validation": json.loads(ddp_output.read_text()),
        "endpoints": endpoints,
        "pooled_by_model": pooled_reports(endpoints),
        "excluded_seed": 2026,
        "exclusion_reason": "cancelled at user request before completion; partial runs are not mixed into endpoint statistics",
    }
    (args.output / "gate-stats-comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output / "gate-stats-comparison.md").write_text(markdown(report))
    print(json.dumps({"status": "passed", "output": str(args.output / "gate-stats-comparison.json")}), flush=True)


if __name__ == "__main__":
    main()
