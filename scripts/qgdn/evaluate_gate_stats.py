"""Read-only endpoint alpha/beta/gamma statistics on the formal validation set."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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

PRODUCTION_COMMIT = "f62322a5fd0cdbc1ed45a9753bdfa22a663143d4"


def json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--eval-sequences", type=int, default=2560)
    parser.add_argument("--production-commit", default=PRODUCTION_COMMIT)
    return parser.parse_args()


def finalize(moments):
    result = {}
    for name, values in sorted(moments.items()):
        total, square_total, count = values.tolist()
        if count <= 0:
            raise RuntimeError(f"Gate {name} has zero observations")
        mean = total / count
        variance = max(0.0, square_total / count - mean * mean)
        result[name] = {
            "sum": total,
            "sum_of_squares": square_total,
            "count": int(count),
            "mean": mean,
            "std": math.sqrt(variance),
        }
    return result


def merge_layer_moments(model):
    totals = {}
    layers = []
    for index, block in enumerate(model.transformer.h):
        moments = {name: values.detach().cpu() for name, values in block.attn.gate_moments().items()}
        layers.append({"layer": index, "gates": finalize(moments)})
        for name, values in moments.items():
            if name in totals:
                totals[name].add_(values)
            else:
                totals[name] = values.clone()
    return finalize(totals), layers


def markdown(report):
    lines = [
        f"# {report['model']} seed {report['seed']} endpoint gate statistics",
        "",
        f"Checkpoint step {report['checkpoint_step']}; {report['validation_sequences']} fixed validation sequences.",
        "",
        "| gate | count | mean | population std |",
        "|---|---:|---:|---:|",
    ]
    for name in ("alpha", "beta", "gamma", "forgetting_margin", "gamma_saturated"):
        if name in report["gates"]:
            gate = report["gates"][name]
            lines.append(f"| {name} | {gate['count']} | {gate['mean']:.9f} | {gate['std']:.9f} |")
    lines += [
        "",
        f"Reproduced validation loss: {report['validation']['loss']:.12f}; summary loss: {report['validation']['summary_loss']:.12f}.",
        "",
        "Moments are accumulated in FP64 from raw gate elements across all validation tokens, heads and layers. Standard deviation is sqrt(sum(x^2)/count - mean^2).",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A Slurm-allocated CUDA GPU is required")
    source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run = json.loads(args.run_json.read_text())
    summary = json.loads(args.summary.read_text())
    if run["code_revision"] != args.production_commit:
        raise ValueError("Run was not produced by the required production commit")
    if summary["identity"] != run["identity"] or summary["status"] != "completed":
        raise ValueError("Run/summary identity or completion status mismatch")
    if summary["step"] != 19073 or summary["trained_tokens"] != 9999745024:
        raise ValueError("Checkpoint did not complete the formal budget")
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
    config = Config.from_name(run["args"]["model"], block_size=run["args"]["sequence_length"])
    if config.mixer not in {"gdn", "qgdn"}:
        raise ValueError(f"Unsupported mixer: {config.mixer}")
    manifest, paths = load_manifest(args.data_manifest, verify_hashes=False)
    if json_hash(manifest) != run["data_sha256"]:
        raise ValueError("Data manifest identity mismatch")
    corpus = TokenCorpus(paths["val"], run["args"]["sequence_length"], seed)
    count = min(args.eval_sequences, corpus.n_blocks)
    if count != args.eval_sequences:
        raise ValueError("Validation corpus has fewer blocks than requested")

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
    blocks = [block.attn for block in model.transformer.h]

    # Directly verify observation does not perturb logits.
    probe_x, _ = corpus.batch([0], shuffle=False)
    probe_x = probe_x.to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        baseline_logits = model(probe_x)
        for mixer in blocks:
            mixer.reset_gate_stats()
            mixer.collect_gate_stats = True
        observed_logits = model(probe_x)
        for mixer in blocks:
            mixer.collect_gate_stats = False
    if not torch.equal(baseline_logits, observed_logits):
        raise RuntimeError("Gate observation changed model logits")
    del baseline_logits, observed_logits, probe_x

    for mixer in blocks:
        mixer.reset_gate_stats()
        mixer.collect_gate_stats = True
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    with torch.inference_mode():
        for row in range(count):
            x, y = corpus.batch([row], shuffle=False)
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            flat_logits, flat_targets = logits.float().flatten(0, 1), y.flatten()
            totals[0] += F.cross_entropy(flat_logits, flat_targets, ignore_index=-100, reduction="sum")
            totals[1] += (flat_targets != -100).sum()
            if (row + 1) % 64 == 0:
                print(json.dumps({"evaluated_sequences": row + 1, "total": count}), flush=True)
    for mixer in blocks:
        mixer.collect_gate_stats = False
    loss = (totals[0] / totals[1]).item()
    summary_loss = summary["final_validation"]["loss"]
    if int(totals[1].item()) != summary["final_validation"]["scored_tokens"]:
        raise ValueError("Validation token count mismatch")
    if abs(loss - summary_loss) > 2e-6:
        raise ValueError("Instrumented validation does not reproduce summary loss")
    gates, layers = merge_layer_moments(model)
    required = {"alpha", "beta"} | ({"gamma"} if config.mixer == "qgdn" else set())
    if not required.issubset(gates):
        raise ValueError(f"Missing required gates: {required - set(gates)}")
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
        "aggregation": "FP64 raw sum/sum-of-squares/count across all validation tokens, heads and layers; population std",
        "observation_is_bitwise_noninvasive": True,
        "gates": gates,
        "layers": layers,
        "validation": {
            "loss": loss,
            "perplexity": math.exp(loss),
            "summary_loss": summary_loss,
            "loss_minus_summary": loss - summary_loss,
            "scored_tokens": int(totals[1].item()),
        },
    }
    write_atomic(args.output_json, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_atomic(args.output_markdown, markdown(report))
    print(json.dumps({"status": "passed", "model": report["model"], "output": str(args.output_json)}), flush=True)


if __name__ == "__main__":
    main()
