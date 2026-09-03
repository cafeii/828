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


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = (actual.float() - expected.float()).square().mean().sqrt()
    denominator = expected.float().square().mean().sqrt().clamp_min(1e-7)
    return float((numerator / denominator).item())


def validate_compiled_builder() -> dict:
    """Compare compiled virtual-row construction with the eager contract."""
    from lit_gpt.mixers.qgdn_rule import dplr_inputs

    generator = torch.Generator(device="cuda").manual_seed(912)
    B, T, H, K, V = 1, 257, 2, 64, 64
    base = (
        torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, generator=generator),
        torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, generator=generator),
        torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16, generator=generator),
        -torch.rand(B, T, H, device="cuda", generator=generator),
        torch.rand(B, T, H, device="cuda", generator=generator),
        torch.rand(B, T, H, device="cuda", generator=generator),
    )
    eager_inputs = tuple(x.detach().clone().requires_grad_() for x in base)
    eager = tuple(dplr_inputs(*eager_inputs, compiled=False).values())
    weights = [torch.randn_like(value) for value in eager]
    eager_grads = torch.autograd.grad(
        sum((value * weight).float().mean() for value, weight in zip(eager, weights)),
        eager_inputs,
    )

    compiled_inputs = tuple(x.detach().clone().requires_grad_() for x in base)
    compiled = tuple(dplr_inputs(*compiled_inputs, compiled=True).values())
    compiled_grads = torch.autograd.grad(
        sum((value * weight).float().mean() for value, weight in zip(compiled, weights)),
        compiled_inputs,
    )
    return {
        "finite": bool(
            all(value.isfinite().all() for value in (*compiled, *compiled_grads))
        ),
        "output_relative_rmse": [
            _relative_rmse(value, reference) for value, reference in zip(compiled, eager)
        ],
        "gradient_relative_rmse": [
            _relative_rmse(value, reference)
            for value, reference in zip(compiled_grads, eager_grads)
        ],
    }


def benchmark_model(
    name: str,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    warmup: int,
    measured: int,
    *,
    qgdn_chunk_size: int | None = None,
    compile_qgdn_inputs: bool = False,
    disable_qgdn_recompute: bool = False,
) -> dict:
    from lit_gpt.mixers import qgdn_rule

    original_chunk_size = qgdn_rule.QGDN_TRAIN_CHUNK_SIZE
    original_compile_inputs = qgdn_rule.QGDN_COMPILE_DPLR_INPUTS
    original_disable_recompute = qgdn_rule.QGDN_DISABLE_DPLR_RECOMPUTE
    if qgdn_chunk_size is not None:
        qgdn_rule.QGDN_TRAIN_CHUNK_SIZE = qgdn_chunk_size
        qgdn_rule.QGDN_COMPILE_DPLR_INPUTS = compile_qgdn_inputs
        qgdn_rule.QGDN_DISABLE_DPLR_RECOMPUTE = disable_qgdn_recompute
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
            "compile_qgdn_inputs": compile_qgdn_inputs,
            "disable_qgdn_recompute": disable_qgdn_recompute,
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
        qgdn_rule.QGDN_COMPILE_DPLR_INPUTS = original_compile_inputs
        qgdn_rule.QGDN_DISABLE_DPLR_RECOMPUTE = original_disable_recompute


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
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="measure each model in a fresh Python/CUDA process",
    )
    parser.add_argument(
        "--recompute-pair",
        action="store_true",
        help="with --isolated, measure only compiled inputs with/without DPLR recompute",
    )
    parser.add_argument(
        "--only", choices=(
            "gdn",
            "qgdn_chunk16",
            "qgdn_chunk32",
            "qgdn_compiled_inputs",
            "qgdn_compiled_no_recompute",
        ),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This benchmark requires exactly one allocated CUDA GPU")
    if args.recompute_pair and not args.isolated:
        raise ValueError("--recompute-pair requires --isolated")

    if args.recompute_pair:
        order = (
            ["qgdn_compiled_no_recompute", "qgdn_compiled_inputs"]
            if args.reverse_order
            else ["qgdn_compiled_inputs", "qgdn_compiled_no_recompute"]
        )
    else:
        order = (
            ["qgdn_compiled_no_recompute", "qgdn_compiled_inputs", "qgdn_chunk32", "qgdn_chunk16", "gdn"]
            if args.reverse_order
            else ["gdn", "qgdn_chunk16", "qgdn_chunk32", "qgdn_compiled_inputs", "qgdn_compiled_no_recompute"]
        )
    if args.isolated:
        if args.only is not None:
            raise ValueError("--isolated and --only are mutually exclusive")
        child_reports = {}
        for name in order:
            child_output = args.output.with_name(f".{args.output.stem}-{name}.json")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output", str(child_output),
                "--sequence-length", str(args.sequence_length),
                "--warmup", str(args.warmup),
                "--measured", str(args.measured),
                "--only", name,
            ]
            subprocess.run(command, check=True)
            child_reports[name] = json.loads(child_output.read_text())
            child_output.unlink()
        models = {name: child_reports[name]["model"] for name in order}
        if args.recompute_pair:
            normal = models["qgdn_compiled_inputs"]["tokens_per_second"]
            no_recompute = models["qgdn_compiled_no_recompute"]["tokens_per_second"]
            report = {
                "status": "measured_isolated_recompute_pair",
                "commit": child_reports[order[0]]["commit"],
                "device": child_reports[order[0]]["device"],
                "torch": child_reports[order[0]]["torch"],
                "cuda": child_reports[order[0]]["cuda"],
                "sequence_length": args.sequence_length,
                "micro_batch_size": 1,
                "activation_checkpointing": True,
                "warmup_steps": args.warmup,
                "measured_steps": args.measured,
                "measurement_order": order,
                "process_isolation": True,
                "numerics": child_reports[order[0]]["numerics"],
                "compiled_builder_validation": child_reports["qgdn_compiled_inputs"].get(
                    "compiled_builder_validation"
                ),
                "models": models,
                "no_recompute_speedup": no_recompute / normal,
                "peak_memory_delta_gb": (
                    models["qgdn_compiled_no_recompute"]["peak_memory_gb"]
                    - models["qgdn_compiled_inputs"]["peak_memory_gb"]
                ),
            }
            write_json(args.output, report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
            return
        gdn = models["gdn"]["tokens_per_second"]
        old = models["qgdn_chunk16"]["tokens_per_second"]
        chunk32 = models["qgdn_chunk32"]["tokens_per_second"]
        new = models["qgdn_compiled_inputs"]["tokens_per_second"]
        no_recompute = models["qgdn_compiled_no_recompute"]["tokens_per_second"]
        report = {
            "status": "measured_isolated",
            "commit": child_reports[order[0]]["commit"],
            "device": child_reports[order[0]]["device"],
            "torch": child_reports[order[0]]["torch"],
            "cuda": child_reports[order[0]]["cuda"],
            "sequence_length": args.sequence_length,
            "micro_batch_size": 1,
            "activation_checkpointing": True,
            "warmup_steps": args.warmup,
            "measured_steps": args.measured,
            "measurement_order": order,
            "process_isolation": True,
            "numerics": child_reports[order[0]]["numerics"],
            "compiled_builder_validation": child_reports["qgdn_compiled_inputs"].get(
                "compiled_builder_validation"
            ),
            "models": models,
            "chunk32_speedup_vs_chunk16": chunk32 / old,
            "compiled_input_speedup_vs_chunk32": new / chunk32,
            "no_recompute_speedup_vs_compiled_inputs": no_recompute / new,
            "candidate_speedup_vs_chunk16": new / old,
            "no_recompute_speedup_vs_chunk16": no_recompute / old,
            "chunk16_to_gdn_ratio": old / gdn,
            "chunk32_to_gdn_ratio": chunk32 / gdn,
            "compiled_input_to_gdn_ratio": new / gdn,
            "no_recompute_to_gdn_ratio": no_recompute / gdn,
            "throughput_target": 0.9,
            "throughput_target_passed": no_recompute / gdn >= 0.9,
        }
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return

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
        "qgdn_compiled_inputs": lambda: benchmark_model(
            "qgdn_340M", tokens, targets, args.warmup, args.measured,
            qgdn_chunk_size=32, compile_qgdn_inputs=True,
        ),
        "qgdn_compiled_no_recompute": lambda: benchmark_model(
            "qgdn_340M", tokens, targets, args.warmup, args.measured,
            qgdn_chunk_size=32, compile_qgdn_inputs=True,
            disable_qgdn_recompute=True,
        ),
    }
    if args.only is not None:
        report = {
            "status": "measured_child",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numerics": numerics,
            "model": runners[args.only](),
        }
        if args.only == "qgdn_compiled_inputs":
            report["compiled_builder_validation"] = validate_compiled_builder()
        write_json(args.output, report)
        return
    models = {name: runners[name]() for name in order}
    gdn = models["gdn"]["tokens_per_second"]
    old = models["qgdn_chunk16"]["tokens_per_second"]
    chunk32 = models["qgdn_chunk32"]["tokens_per_second"]
    new = models["qgdn_compiled_inputs"]["tokens_per_second"]
    no_recompute = models["qgdn_compiled_no_recompute"]["tokens_per_second"]
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
        "chunk32_speedup_vs_chunk16": chunk32 / old,
        "compiled_input_speedup_vs_chunk32": new / chunk32,
        "no_recompute_speedup_vs_compiled_inputs": no_recompute / new,
        "candidate_speedup_vs_chunk16": new / old,
        "no_recompute_speedup_vs_chunk16": no_recompute / old,
        "chunk16_to_gdn_ratio": old / gdn,
        "chunk32_to_gdn_ratio": chunk32 / gdn,
        "compiled_input_to_gdn_ratio": new / gdn,
        "no_recompute_to_gdn_ratio": no_recompute / gdn,
        "throughput_target": 0.9,
        "throughput_target_passed": no_recompute / gdn >= 0.9,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
