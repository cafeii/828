"""Benchmark native GDN and the QGDN chunk-16/chunk-32 training paths."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "qgdn"))

import torch
import torch.nn.functional as F

from lit_gpt.config import Config
from lit_gpt.model import GPT
from runtime import configure_numerics


def benchmark_model(
    name: str,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    warmup: int,
    measured: int,
    *,
    qgdn_chunk_size: int | None = None,
) -> dict:
    from lit_gpt.mixers import qgdn_rule

    original_chunk_size = qgdn_rule.QGDN_TRAIN_CHUNK_SIZE
    if qgdn_chunk_size is not None:
        qgdn_rule.QGDN_TRAIN_CHUNK_SIZE = qgdn_chunk_size
    try:
        torch.manual_seed(3407)
        config = Config.from_name(name, block_size=tokens.shape[1])
        model = GPT(config)
        model.apply(lambda module: model._init_weights(module, n_layer=config.n_layer))
        model.gradient_checkpointing = True
        model.cuda().train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=4e-4, betas=(0.9, 0.95), fused=True
        )
        durations = []
        for index in range(warmup + measured):
            optimizer.zero_grad(set_to_none=True)
            if index == warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(tokens)
                loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            if index >= warmup:
                durations.append(time.perf_counter() - start)
        result = {
            "model": name,
            "qgdn_chunk_size": qgdn_chunk_size,
            "step_seconds": durations,
            "mean_step_seconds": statistics.mean(durations),
            "median_step_seconds": statistics.median(durations),
            "tokens_per_second": tokens.numel() / statistics.mean(durations),
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
            "final_loss": loss.item(),
            "finite": bool(torch.isfinite(loss)),
        }
        del optimizer, model, logits, loss
        torch.cuda.empty_cache()
        return result
    finally:
        qgdn_rule.QGDN_TRAIN_CHUNK_SIZE = original_chunk_size


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument(
        "--reverse-order",
        action="store_true",
        help="run chunk32 before chunk16 to expose order/cache bias",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This benchmark requires exactly one allocated CUDA GPU")

    numerics = configure_numerics(cpu=False)
    torch.manual_seed(117)
    config = Config.from_name("gdn_control_340M", block_size=args.sequence_length)
    tokens = torch.randint(
        0, config.padded_vocab_size, (1, args.sequence_length), device="cuda"
    )
    targets = torch.roll(tokens, shifts=-1, dims=1)
    runners = {
        "gdn": lambda: benchmark_model(
            "gdn_control_340M", tokens, targets, args.warmup, args.measured
        ),
        "qgdn_chunk16": lambda: benchmark_model(
            "qgdn_340M", tokens, targets, args.warmup, args.measured,
            qgdn_chunk_size=16,
        ),
        "qgdn_chunk32": lambda: benchmark_model(
            "qgdn_340M", tokens, targets, args.warmup, args.measured,
            qgdn_chunk_size=32,
        ),
    }
    order = (
        ["qgdn_chunk32", "qgdn_chunk16", "gdn"]
        if args.reverse_order
        else ["gdn", "qgdn_chunk16", "qgdn_chunk32"]
    )
    models = {name: runners[name]() for name in order}
    gdn = models["gdn"]["tokens_per_second"]
    old = models["qgdn_chunk16"]["tokens_per_second"]
    new = models["qgdn_chunk32"]["tokens_per_second"]
    report = {
        "status": "measured",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sequence_length": args.sequence_length,
        "micro_batch_size": 1,
        "activation_checkpointing": True,
        "warmup_steps": args.warmup,
        "measured_steps": args.measured,
        "measurement_order": order,
        "numerics": numerics,
        "models": models,
        "candidate_speedup_vs_chunk16": new / old,
        "chunk16_to_gdn_ratio": old / gdn,
        "chunk32_to_gdn_ratio": new / gdn,
        "throughput_target": 0.9,
        "throughput_target_passed": new / gdn >= 0.9,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
