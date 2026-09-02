"""Profile the existing two-call QR path against the rank-2 fused forward candidate."""
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

from lit_gpt.kernels import get_chunk_gated_delta_rule
from lit_gpt.mixers.qr_gdn_parallel import qr_gdn_parallel
from lit_gpt.mixers.qr_gdn_rule import block_wy_rank2_vector_decay, qr_gdn_rank2_factors
from lit_gpt.qr_gdn_chunk_output import qr_gdn_chunk_output_fwd
from lit_gpt.qr_gdn_chunk_state import qr_gdn_chunk_state_fwd
from runtime import configure_numerics


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def measure(fn, warmup: int, steps: int) -> dict:
    durations = []
    output = None
    for index in range(warmup + steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if index >= warmup:
            durations.append(elapsed)
    return {
        "mean_seconds": statistics.mean(durations),
        "median_seconds": statistics.median(durations),
        "steps": durations,
        "output": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This diagnostic requires exactly one allocated CUDA GPU")
    if args.sequence_length % 64:
        raise ValueError("sequence length must be divisible by 64")
    numerics = configure_numerics(cpu=False)

    torch.manual_seed(3407)
    B, T, H, K, V = 1, args.sequence_length, 16, 64, 64
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    g_kv = -(0.001 + 0.03 * torch.rand(B, T, H, device="cuda"))
    beta_kv = (0.05 + 0.9 * torch.rand(B, T, H, device="cuda")).float()
    g_qr = -(0.001 + 0.03 * torch.rand(B, T, H, device="cuda"))
    beta_qr = (0.05 + 0.9 * torch.rand(B, T, H, device="cuda")).float()
    read_logit = (0.1 * torch.randn(B, T, H, device="cuda")).float()
    rule = get_chunk_gated_delta_rule()

    def native_gdn():
        return rule(
            q=q,
            k=k,
            v=v,
            g=g_kv,
            beta=beta_kv,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            chunk_size=64,
        )[0]

    def current_qr():
        return qr_gdn_parallel(
            q,
            k,
            v,
            g_kv,
            beta_kv,
            g_qr,
            beta_qr,
            read_logit,
            output_final_state=False,
            chunk_size=64,
        )[0]

    def prepare_compact():
        qn, _, log_decay, left, right, write = qr_gdn_rank2_factors(
            q, k, g_kv, beta_kv, g_qr, beta_qr
        )
        compact = block_wy_rank2_vector_decay(
            log_decay, left, right, write, v.float(), chunk_size=64
        )
        compact = (
            compact[0],
            compact[1].to(torch.bfloat16),
            compact[2].to(torch.bfloat16),
            compact[3].to(torch.bfloat16),
        )
        factors = (
            qn.to(torch.bfloat16),
            log_decay.to(torch.bfloat16),
            left.to(torch.bfloat16),
            right.to(torch.bfloat16),
            write.to(torch.bfloat16),
            v,
            read_logit.tanh().to(torch.bfloat16),
        )
        return compact, factors

    compact, factors = prepare_compact()

    def fused_kernel_only():
        starts, _ = qr_gdn_chunk_state_fwd(
            *compact, initial_state=None, output_final_state=False
        )
        return qr_gdn_chunk_output_fwd(
            *factors, starts, chunk_size=64
        )

    def fused_full():
        current_compact, current_factors = prepare_compact()
        starts, _ = qr_gdn_chunk_state_fwd(
            *current_compact, initial_state=None, output_final_state=False
        )
        return qr_gdn_chunk_output_fwd(
            *current_factors, starts, chunk_size=64
        )

    benchmarks = {}
    outputs = {}
    for name, fn in (
        ("native_gdn", native_gdn),
        ("current_two_call_qr", current_qr),
        ("rank2_kernel_only", fused_kernel_only),
        ("rank2_full_forward", fused_full),
    ):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        measured = measure(fn, args.warmup, args.steps)
        outputs[name] = measured.pop("output")
        measured["tokens_per_second"] = T / measured["mean_seconds"]
        measured["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
        benchmarks[name] = measured

    current = outputs["current_two_call_qr"].float()
    report = {
        "status": "measured",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "cuda_device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shape": {"batch": B, "tokens": T, "heads": H, "key_dim": K, "value_dim": V},
        "dtype": "bfloat16",
        "chunk_size": 64,
        "warmup": args.warmup,
        "steps": args.steps,
        "numerics": numerics,
        "benchmarks": benchmarks,
        "rank2_full_to_current_ratio": (
            benchmarks["current_two_call_qr"]["mean_seconds"]
            / benchmarks["rank2_full_forward"]["mean_seconds"]
        ),
        "rank2_kernel_to_current_ratio": (
            benchmarks["current_two_call_qr"]["mean_seconds"]
            / benchmarks["rank2_kernel_only"]["mean_seconds"]
        ),
        "rank2_vs_current_max_abs": (
            outputs["rank2_full_forward"].float() - current
        ).abs().max().item(),
        "rank2_vs_current_mean_abs": (
            outputs["rank2_full_forward"].float() - current
        ).abs().mean().item(),
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
