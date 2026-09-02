"""Same-H800 QR-GDN stability checks and 340M training-step benchmark."""
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


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def benchmark_model(name: str, tokens: torch.Tensor, targets: torch.Tensor, warmup: int, steps: int) -> dict:
    torch.manual_seed(3407)
    config = Config.from_name(name, block_size=tokens.shape[1])
    model = GPT(config)
    model.apply(lambda module: model._init_weights(module, n_layer=config.n_layer))
    model.gradient_checkpointing = True
    parameters = sum(parameter.numel() for parameter in model.parameters())
    mixer = model.transformer.h[0].attn
    state_elements_per_layer = mixer.num_heads * mixer.head_k_dim * mixer.latent_dim
    state_matrices = 2 if config.mixer == "qr_gdn" else 1
    state_elements = config.n_layer * state_matrices * state_elements_per_layer
    model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, betas=(0.9, 0.95), fused=True)

    durations = []
    for index in range(warmup + steps):
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
        elapsed = time.perf_counter() - start
        if index >= warmup:
            durations.append(elapsed)
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
    result = {
        "model": name,
        "parameters": parameters,
        "state_matrices": state_matrices,
        "state_elements": state_elements,
        "state_bytes_fp32": state_elements * 4,
        "step_seconds": durations,
        "median_step_seconds": statistics.median(durations),
        "mean_step_seconds": statistics.mean(durations),
        "tokens_per_second": tokens.numel() / statistics.mean(durations),
        "peak_memory_gb": peak_memory_gb,
        "final_loss": loss.item(),
    }
    del optimizer, model, logits, loss
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This diagnostic requires exactly one allocated CUDA GPU")
    if args.sequence_length % 64:
        raise ValueError("sequence length must be divisible by the production chunk size")
    numerics = configure_numerics(cpu=False)

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_qr_gdn_parallel_gpu.py", "-q"],
        cwd=ROOT,
        check=False,
        text=True,
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)

    torch.manual_seed(117)
    baseline_config = Config.from_name("gdn_control_340M", block_size=args.sequence_length)
    tokens = torch.randint(
        0,
        baseline_config.padded_vocab_size,
        (1, args.sequence_length),
        device="cuda",
    )
    targets = torch.roll(tokens, shifts=-1, dims=1)
    results = {}
    for name in ("gdn_control_340M", "qr_gdn_340M"):
        results[name] = benchmark_model(name, tokens, targets, args.warmup, args.steps)
    ratio = results["qr_gdn_340M"]["tokens_per_second"] / results["gdn_control_340M"]["tokens_per_second"]
    report = {
        "status": "measured",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "cuda_device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "precision": "bf16-mixed",
        "sequence_length": args.sequence_length,
        "micro_batch_size": 1,
        "activation_checkpointing": True,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "numerics": numerics,
        "models": results,
        "qr_to_gdn_throughput_ratio": ratio,
        "throughput_gate": 0.8,
        "throughput_gate_passed": ratio >= 0.8,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
