"""Read-only position-bucket LM evaluation for a completed QGDN-suite checkpoint."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from lit_gpt.config import Config
from lit_gpt.model import GPT
from data import TokenCorpus, load_manifest
from runtime import configure_numerics

BUCKETS = ((1, 256), (257, 512), (513, 1024), (1025, 2048), (2049, 4096))
PRODUCTION_COMMIT = "f62322a5fd0cdbc1ed45a9753bdfa22a663143d4"


def json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def validate_buckets(sequence_length):
    zero_based = []
    expected = 1
    for start, end in BUCKETS:
        if start != expected or end < start:
            raise ValueError("Buckets must be contiguous and nonempty")
        zero_based.append((start - 1, end))
        expected = end + 1
    if expected != sequence_length + 1:
        raise ValueError(f"Buckets end at {expected - 1}, sequence length is {sequence_length}")
    return zero_based


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-json", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--eval-sequences", type=int, default=2560)
    parser.add_argument("--production-commit", default=PRODUCTION_COMMIT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        missing = [name for name in ("checkpoint", "run_json", "summary", "data_manifest", "output_json", "output_markdown") if getattr(args, name) is None]
        if missing:
            parser.error("missing required arguments: " + ", ".join(missing))
    return args


def self_test():
    ranges = validate_buckets(4096)
    assert ranges == [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096)]
    values = torch.arange(4096, dtype=torch.float64)
    parts = [values[start:end].sum() for start, end in ranges]
    assert torch.stack(parts).sum().item() == values.sum().item()
    print(json.dumps({"status": "passed", "buckets": BUCKETS}))


def markdown(report):
    lines = [
        "# GDN seed 3407 position-bucket validation",
        "",
        f"Checkpoint step: {report['checkpoint_step']}; validation sequences: {report['validation_sequences']}.",
        "",
        "| positions | scored tokens | loss | perplexity |",
        "|---|---:|---:|---:|",
    ]
    for bucket in report["buckets"]:
        lines.append(f"| {bucket['start']}-{bucket['end']} | {bucket['scored_tokens']} | {bucket['loss']:.9f} | {bucket['perplexity']:.9f} |")
    lines += [
        "",
        f"Bucket aggregate loss: {report['aggregate']['bucket_reconstructed_loss']:.12f}.",
        f"Formal-style aggregate loss: {report['aggregate']['formal_style_loss']:.12f}.",
        f"Training summary loss: {report['aggregate']['training_summary_loss']:.12f}.",
        "",
        "This is a read-only GDN result. QGDN-GDN bucket differences wait for the matched QGDN final checkpoint.",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("A Slurm-allocated CUDA GPU is required")
    if args.eval_sequences <= 0:
        raise ValueError("eval-sequences must be positive")
    source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    behavior_paths = ["model", "scripts/qgdn/data.py", "scripts/qgdn/runtime.py"]
    changed = subprocess.run(["git", "diff", "--quiet", args.production_commit, "--", *behavior_paths], cwd=ROOT).returncode
    if changed != 0:
        raise RuntimeError("Evaluation source changes production model/data/numerics behavior")
    run = json.loads(args.run_json.read_text())
    summary = json.loads(args.summary.read_text())
    if run["code_revision"] != args.production_commit:
        raise ValueError("Run was not produced by the required production commit")
    if summary["identity"] != run["identity"] or summary["status"] != "completed":
        raise ValueError("run/summary identity or completion status mismatch")
    if summary["step"] != run["args"]["max_steps"] or summary["trained_tokens"] != run["planned_tokens"]:
        raise ValueError("Checkpoint run did not complete the planned budget")
    sequence_length = run["args"]["sequence_length"]
    ranges = validate_buckets(sequence_length)
    if args.eval_sequences != run["args"]["eval_sequences"]:
        raise ValueError("Evaluation sequence count must match formal validation")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    numerics = configure_numerics(cpu=False)
    if numerics != run["numerics"]:
        raise ValueError("Numerics policy differs from training")
    seed = run["args"]["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    config = Config.from_name(run["args"]["model"], block_size=sequence_length)
    if config.mixer != "gdn":
        raise ValueError("This first diagnostic is restricted to the completed GDN checkpoint")
    manifest, paths = load_manifest(args.data_manifest, verify_hashes=False)
    if json_hash(manifest) != run["data_sha256"]:
        raise ValueError("Data manifest identity mismatch")
    corpus = TokenCorpus(paths["val"], sequence_length, seed)
    count = min(args.eval_sequences, corpus.n_blocks)
    if count != args.eval_sequences:
        raise ValueError("Validation corpus has fewer blocks than the formal evaluation")
    model = GPT(config)
    model.apply(lambda module: model._init_weights(module, n_layer=config.n_layer))
    model.gradient_checkpointing = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["identity"] != run["identity"] or checkpoint["step"] != summary["step"]:
        raise ValueError("Checkpoint identity or step mismatch")
    checkpoint.pop("optimizer", None)
    checkpoint.pop("rng", None)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != run["parameters"]:
        raise ValueError("Model parameter count mismatch")
    model.to(device).eval()
    bucket_totals = torch.zeros(len(ranges), 2, device=device, dtype=torch.float64)
    formal_totals = torch.zeros(2, device=device, dtype=torch.float64)
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        for row in range(count):
            x, y = corpus.batch([row], shuffle=False)
            x, y = x.to(device), y.to(device)
            with amp():
                logits = model(x)
            flat_logits, flat_targets = logits.float().flatten(0, 1), y.flatten()
            formal_totals[0] += F.cross_entropy(flat_logits, flat_targets, ignore_index=-100, reduction="sum")
            formal_totals[1] += (flat_targets != -100).sum()
            token_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100, reduction="none").view_as(y)
            for index, (start, end) in enumerate(ranges):
                valid = y[:, start:end] != -100
                bucket_totals[index, 0] += token_loss[:, start:end][valid].sum()
                bucket_totals[index, 1] += valid.sum()
            if (row + 1) % 64 == 0:
                print(json.dumps({"evaluated_sequences": row + 1, "total": count}), flush=True)
    if not torch.isfinite(bucket_totals).all() or not torch.isfinite(formal_totals).all():
        raise FloatingPointError("Nonfinite bucket evaluation totals")
    bucket_sum = bucket_totals[:, 0].sum().item()
    bucket_count = bucket_totals[:, 1].sum().item()
    formal_sum, formal_count = formal_totals.tolist()
    if int(bucket_count) != int(formal_count) or int(formal_count) != summary["final_validation"]["scored_tokens"]:
        raise ValueError("Scored-token count mismatch")
    bucket_loss = bucket_sum / bucket_count
    formal_loss = formal_sum / formal_count
    summary_loss = summary["final_validation"]["loss"]
    if abs(bucket_loss - formal_loss) > 2e-6:
        raise ValueError("Bucket sums do not reconstruct formal-style loss")
    if abs(formal_loss - summary_loss) > 2e-6:
        raise ValueError("Evaluation does not reproduce the training summary loss")
    bucket_records = []
    for (one_start, one_end), totals in zip(BUCKETS, bucket_totals.tolist()):
        loss = totals[0] / totals[1]
        bucket_records.append({"start": one_start, "end": one_end, "nll_sum": totals[0], "scored_tokens": int(totals[1]), "loss": loss, "perplexity": math.exp(loss)})
    report = {
        "status": "passed",
        "model": run["args"]["model"],
        "seed": seed,
        "production_commit": args.production_commit,
        "evaluation_source_revision": source_revision,
        "run_identity": run["identity"],
        "checkpoint_step": summary["step"],
        "trained_tokens": summary["trained_tokens"],
        "validation_seed": run["validation_seed"],
        "validation_sequences": count,
        "precision": run["precision"],
        "numerics": numerics,
        "parameters": parameters,
        "buckets": bucket_records,
        "aggregate": {
            "nll_sum": bucket_sum,
            "scored_tokens": int(bucket_count),
            "bucket_reconstructed_loss": bucket_loss,
            "formal_style_loss": formal_loss,
            "training_summary_loss": summary_loss,
            "bucket_minus_formal_loss": bucket_loss - formal_loss,
            "formal_minus_summary_loss": formal_loss - summary_loss,
            "perplexity": math.exp(bucket_loss),
        },
        "interpretation": "GDN-only read-only endpoint profile; matched QGDN differences are pending.",
    }
    write_text_atomic(args.output_json, json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_text_atomic(args.output_markdown, markdown(report))
    print(json.dumps({"status": "passed", "output_json": str(args.output_json), "aggregate_loss": bucket_loss}), flush=True)


if __name__ == "__main__":
    main()
