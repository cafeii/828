"""Benchmark physical-token QGDN against the virtual-2T compatibility path."""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "qgdn"))

import torch

from benchmark_training_speed import benchmark_model, write_json
from lit_gpt.config import Config
from runtime import configure_numerics


MODELS = {
    "virtual_recall_then_delta": ("qgdn_340M", False),
    "physical_recall_then_delta": ("qgdn_340M", True),
    "physical_delta_then_recall": ("qgdn_delta_then_recall_340M", True),
    "physical_parallel": ("qgdn_parallel_340M", True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument("--only", choices=tuple(MODELS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This benchmark requires exactly one allocated CUDA GPU")

    if args.only is None:
        children = {}
        for name in MODELS:
            child = args.output.with_name(f".{args.output.stem}-{name}.json")
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output", str(child),
                    "--sequence-length", str(args.sequence_length),
                    "--micro-batch-size", str(args.micro_batch_size),
                    "--warmup", str(args.warmup),
                    "--measured", str(args.measured),
                    "--only", name,
                ],
                check=True,
            )
            children[name] = json.loads(child.read_text())
            child.unlink()
        results = {name: value["result"] for name, value in children.items()}
        baseline = results["virtual_recall_then_delta"]["tokens_per_second"]
        report = {
            "status": "measured",
            "commit": children["virtual_recall_then_delta"]["commit"],
            "device": children["virtual_recall_then_delta"]["device"],
            "torch": children["virtual_recall_then_delta"]["torch"],
            "cuda": children["virtual_recall_then_delta"]["cuda"],
            "sequence_length": args.sequence_length,
            "micro_batch_size": args.micro_batch_size,
            "warmup_steps": args.warmup,
            "measured_steps": args.measured,
            "activation_checkpointing": False,
            "loss_implementation": "fused",
            "process_isolation": True,
            "models": results,
            "ratios_to_virtual": {
                name: value["tokens_per_second"] / baseline
                for name, value in results.items()
                if name != "virtual_recall_then_delta"
            },
        }
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return

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
        "result": result,
    }
    write_json(args.output, report)


if __name__ == "__main__":
    main()
