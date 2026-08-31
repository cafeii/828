"""Compare final checkpoints only after enforcing paired-run invariants."""
import argparse
import json
import statistics
from pathlib import Path


def pairing_key(run):
    args = {k: v for k, v in run["args"].items() if k != "model"}
    config = {k: v for k, v in run["config"].items() if k not in {"name", "mixer"} and not k.startswith("recall_")}
    return dict(args=args, config=config, data=run["data_sha256"], code=run["code_revision"],
                world=run["world_size"], initialization=run["shared_initialization_sha256"], precision=run["precision"],
                numerics=run.get("numerics"))


def summarize(paths):
    rows, baselines, comparisons = [], {}, []
    for path in paths:
        run = json.loads((path / "run.json").read_text())
        result = json.loads((path / "summary.json").read_text())
        if result["status"] != "completed" or result["step"] != run["args"]["max_steps"]:
            raise ValueError(f"Incomplete run: {path}")
        if result["trained_tokens"] != run["planned_tokens"]:
            raise ValueError(f"Token budget mismatch: {path}")
        if result["identity"] != run["identity"]:
            raise ValueError(f"Mismatched run/summary files: {path}")
        row = dict(path=str(path), model=run["config"]["name"], seed=run["args"]["seed"],
                   parameters=result["parameters"], tokens=result["trained_tokens"],
                   validation=result["final_validation"], gpu_hours=result["gpu_hours"],
                   train_seconds=result["train_seconds"], peak_memory_gb=result["peak_memory_gb"])
        key = json.dumps(pairing_key(run), sort_keys=True)
        rows.append((row, run, key))
        if run["config"]["mixer"] == "gdn":
            if key in baselines:
                raise ValueError("Duplicate GDN baseline for the same paired experiment")
            baselines[key] = row
    seen = set()
    for row, run, key in rows:
        if run["config"]["mixer"] == "gdn":
            continue
        if (row["model"], key) in seen:
            raise ValueError("Duplicate treatment run would overweight a seed")
        seen.add((row["model"], key))
        if key not in baselines:
            raise ValueError(f"No exactly matched GDN baseline for {row['path']}")
        baseline = baselines[key]
        if row["validation"]["scored_tokens"] != baseline["validation"]["scored_tokens"]:
            raise ValueError("Validation token counts differ")
        comparison = dict(model=row["model"], seed=row["seed"],
                          loss_reduction=baseline["validation"]["loss"] - row["validation"]["loss"],
                          extra_parameters=row["parameters"] - baseline["parameters"],
                          training_time_ratio=row["train_seconds"] / baseline["train_seconds"])
        if "accuracy" in row["validation"]:
            comparison["accuracy_gain"] = row["validation"]["accuracy"] - baseline["validation"]["accuracy"]
        comparisons.append(comparison)
    aggregates = []
    for model in sorted({c["model"] for c in comparisons}):
        values = [c["loss_reduction"] for c in comparisons if c["model"] == model]
        aggregates.append(dict(model=model, seeds=len(values), mean_loss_reduction=statistics.mean(values),
                               std_loss_reduction=statistics.stdev(values) if len(values) > 1 else None,
                               positive_seeds=sum(v > 0 for v in values)))
    return dict(runs=[r[0] for r in rows], paired_comparisons=comparisons, aggregates=aggregates,
                interpretation="Positive loss_reduction favors QGDN. Report all seeds, controls and costs; small seed counts are exploratory.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = summarize(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
