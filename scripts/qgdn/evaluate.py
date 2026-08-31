"""Evaluate one trained model at fixed context lengths, including MQAR delay tests."""
import argparse
import contextlib
import math
import sys
from pathlib import Path

from runtime import configure_device_from_cli

if __name__ == "__main__":
    configure_device_from_cli()

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from lit_gpt.config import Config
from lit_gpt.model import GPT
from data import TokenCorpus, load_manifest, mqar_batch
from train import json_hash, write_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="model_final.pt from the paired trainer")
    p.add_argument("--data-manifest", type=Path)
    p.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 8192])
    p.add_argument("--eval-tokens", type=int, default=10485760)
    p.add_argument("--mqar-sequences", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if min(args.lengths) < 1 or min(args.eval_tokens, args.mqar_sequences, args.batch_size) < 1:
        p.error("Lengths and sample budgets must be positive")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, run = Config(**saved["config"]), saved["run"]
    if args.cpu and not config.name.endswith("_tiny"):
        p.error("CPU evaluation is only for tiny integration tests")
    device = torch.device("cpu" if args.cpu else "cuda")
    model = GPT(config).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    if args.cpu:
        for block in model.transformer.h:
            block.attn.mode = "naive"
    task = run["args"]["task"]
    if task == "lm":
        if args.data_manifest is None:
            p.error("LM evaluation needs the original disjoint-data manifest")
        manifest, paths = load_manifest(args.data_manifest)
        if json_hash(manifest) != run["data_sha256"]:
            raise ValueError("Evaluation corpus differs from the registered held-out corpus")
    results = []
    for length in args.lengths:
        if task == "lm":
            corpus = TokenCorpus(paths["val"], length, 172903)
            sequences = min(corpus.n_blocks, args.eval_tokens // length)
        else:
            sequences = args.mqar_sequences
        if sequences == 0:
            raise ValueError("No complete validation sequence within the token budget")
        total_loss, correct, count = 0.0, 0, 0
        # Four positional bins show whether improvements persist late in context.
        bin_loss, bin_count = [0.0] * 4, [0] * 4
        for start in range(0, sequences, args.batch_size):
            rows = range(start, min(sequences, start + args.batch_size))
            x, y = corpus.batch(rows, False) if task == "lm" else mqar_batch(
                rows, length, 172903, config.vocab_size, overwrite=run["args"]["mqar_overwrite"])
            x, y = x.to(device), y.to(device)
            with contextlib.nullcontext() if args.cpu else torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            losses = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), ignore_index=-100, reduction="none").view_as(y)
            valid = y != -100
            total_loss += losses.sum().item()
            count += valid.sum().item()
            correct += ((logits.argmax(-1) == y) & valid).sum().item()
            for b in range(4):
                sl = slice(length * b // 4, length * (b + 1) // 4)
                bin_loss[b] += losses[:, sl].sum().item()
                bin_count[b] += valid[:, sl].sum().item()
        loss = total_loss / count
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite evaluation loss")
        results.append(dict(length=length, sequences=sequences, scored_tokens=count, loss=loss,
                            perplexity=math.exp(loss) if task == "lm" else None,
                            accuracy=correct / count if task == "mqar" else None,
                            position_bin_loss=[s / n if n else None for s, n in zip(bin_loss, bin_count)]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, dict(checkpoint=str(args.checkpoint.resolve()), task=task, results=results,
                                 note="Use matched checkpoints and budgets. Context extrapolation is not evidence of improvement by itself."))


if __name__ == "__main__":
    main()
