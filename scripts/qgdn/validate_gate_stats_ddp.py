"""Two-rank check that gate moments are globally merged before std is computed."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from lit_gpt.config import Config
from lit_gpt.model import GPT
from train import global_gate_statistics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def local_moments(model):
    result = {}
    for block in model.transformer.h:
        for name, values in block.attn.gate_moments().items():
            values = values.detach().cpu()
            result[name] = result.get(name, torch.zeros_like(values)) + values
    return {name: values.tolist() for name, values in result.items()}


def expected_statistics(gathered):
    totals = {}
    for rank in gathered:
        for name, values in rank.items():
            values = torch.tensor(values, dtype=torch.float64)
            totals[name] = totals.get(name, torch.zeros_like(values)) + values
    result = {}
    for name, (total, square_total, count) in totals.items():
        mean = total / count
        variance = torch.clamp(square_total / count - mean.square(), min=0)
        if name == "gamma_saturated":
            result["gamma_saturated_fraction"] = mean.item()
        else:
            result[f"{name}_mean"] = mean.item()
            result[f"{name}_std"] = variance.sqrt().item()
    return result


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 2:
        raise ValueError("This validation requires exactly two ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    reports = {}
    for mixer_name in ("gdn", "qgdn"):
        torch.manual_seed(3407)
        config = Config.from_name(f"{mixer_name}_recall_tiny", use_short_conv=False, _norm_class="RMSNorm")
        model = GPT(config)
        model.apply(lambda module: model._init_weights(module, n_layer=config.n_layer))
        for block in model.transformer.h:
            block.attn.mode = "naive"
            block.attn.reset_gate_stats()
            block.attn.collect_gate_stats = True
        device = torch.device("cuda", local_rank)
        model.to(device).eval()
        tokens = (torch.arange(26, device=device).reshape(2, 13) + rank * 37) % config.padded_vocab_size
        with torch.inference_mode():
            model(tokens)
        raw = local_moments(model)
        gathered = [None] * world
        dist.all_gather_object(gathered, raw)
        actual = global_gate_statistics(model, world)
        if rank == 0:
            expected = expected_statistics(gathered)
            if actual.keys() != expected.keys():
                raise AssertionError((actual.keys(), expected.keys()))
            for name in actual:
                if not math.isclose(actual[name], expected[name], rel_tol=1e-12, abs_tol=1e-12):
                    raise AssertionError((mixer_name, name, actual[name], expected[name]))
            reports[mixer_name] = actual
    if rank == 0:
        report = {"status": "passed", "world_size": world, "models": reports,
                  "aggregation": "raw FP64 sum/sum-of-squares/count reduced across ranks before population std"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
