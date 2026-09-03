"""Paired GDN/QGDN training: explicit budgets, global data order, strict resume.

Single node CUDA/DDP via ``python -m torch.distributed.run``. No automatic
Slurm submission or downloads.
The CPU option is solely for small integration tests (no ShortConv).
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from runtime import configure_device_from_cli, configure_numerics

if __name__ == "__main__":
    configure_device_from_cli()

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from lit_gpt.config import Config
from lit_gpt import FusedCrossEntropyLoss
from lit_gpt.model import GPT
from data import TokenCorpus, load_manifest, mqar_batch


def json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", choices=["lm", "mqar"], default="lm")
    p.add_argument("--data-manifest", type=Path)
    p.add_argument("--max-steps", type=int, default=19073)
    p.add_argument("--sequence-length", type=int, default=4096)
    p.add_argument("--global-batch-size", type=int, default=128)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--learning-rate", type=float, default=4e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-sequences", type=int, default=2560)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--training-loss",
        choices=("torch", "fused"),
        default="torch",
        help="Fused loss avoids the full FP32 logits buffer and enables larger micro batches",
    )
    p.add_argument("--no-activation-checkpointing", action="store_true")
    p.add_argument("--resume", type=Path, help="Explicit complete checkpoint; any mismatch/error is fatal")
    p.add_argument("--cpu", action="store_true", help="Tiny integration tests only")
    p.add_argument("--stop-after-step", type=int, help="Orderly checkpoint/exit without changing the planned LR schedule")
    p.add_argument("--mqar-overwrite", action="store_true")
    args = p.parse_args()
    for key in ("max_steps", "sequence_length", "global_batch_size", "micro_batch_size", "eval_every",
                "eval_sequences", "save_every", "log_every"):
        if getattr(args, key) <= 0:
            p.error(f"{key} must be positive")
    if args.task == "lm" and args.data_manifest is None:
        p.error("LM training requires a disjoint-data manifest")
    if args.task == "mqar" and args.data_manifest is not None:
        p.error("MQAR does not use a text corpus")
    if args.warmup_steps is None:
        args.warmup_steps = max(1, int(args.max_steps * 0.01))
    if not 0 <= args.warmup_steps <= args.max_steps:
        p.error("Invalid warmup length")
    if args.stop_after_step is not None and not 0 < args.stop_after_step <= args.max_steps:
        p.error("--stop-after-step must lie within the full training budget")
    return args


def learning_rate(args, step):
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / args.warmup_steps
    fraction = (step - args.warmup_steps + 1) / max(1, args.max_steps - args.warmup_steps)
    return args.learning_rate * (args.min_lr_ratio + (1 - args.min_lr_ratio) * (1 + math.cos(math.pi * fraction)) / 2)


def optimizer_groups(model, decay):
    groups = [[], []]
    for p in model.parameters():
        groups[int(p.ndim < 2 or getattr(p, "_no_weight_decay", False))].append(p)
    return [dict(params=groups[0], weight_decay=decay), dict(params=groups[1], weight_decay=0.0)]


def shared_parameter_hash(model):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ".recall_" not in name:
            h.update(name.encode())
            h.update(parameter.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def reduce_sum(tensor, world):
    if world > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def global_gate_statistics(model, world):
    """Merge raw moments across microbatches, layers and DDP ranks."""
    totals = {}
    for block in model.transformer.h:
        if not hasattr(block.attn, "gate_moments"):
            continue
        for name, moments in block.attn.gate_moments().items():
            if name in totals:
                totals[name].add_(moments)
            else:
                totals[name] = moments.clone()
    result = {}
    for name in sorted(totals):
        total, square_total, count = reduce_sum(totals[name], world)
        if count.item() <= 0:
            raise RuntimeError(f"Gate statistics for {name} have zero count")
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
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError("Keep outputs outside the immutable source checkout")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit source changes before training; a dirty checkout is not a reproducible snapshot")
    rank, world, local_rank = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    if args.global_batch_size % (world * args.micro_batch_size):
        raise ValueError("Global batch must be exactly divisible by world size * micro batch")
    if args.cpu and not args.model.endswith("_tiny"):
        raise ValueError("--cpu is restricted to the tiny smoke configurations")
    if args.cpu and args.training_loss != "torch":
        raise ValueError("The fused training loss requires CUDA")
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("No allocated CUDA GPU; do not run training on the login node")
    device = torch.device("cpu" if args.cpu else f"cuda:{local_rank}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    numerics = configure_numerics(cpu=args.cpu)
    if world > 1:
        dist.init_process_group("gloo" if args.cpu else "nccl")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    amp = lambda: contextlib.nullcontext() if args.cpu else torch.autocast("cuda", dtype=torch.bfloat16)
    config = Config.from_name(args.model, block_size=args.sequence_length)
    if args.cpu:
        config.use_short_conv, config._norm_class = False, "RMSNorm"
    if config.mixer not in {"gdn", "qgdn", "dt_gdn", "jqc_gdn"} or (config.num_groups or config.n_head) != config.n_head or config.use_lsa:
        raise ValueError("The paired trainer requires a supported standard-MHA GDN-family model")

    manifest, corpus = None, {}
    if args.task == "lm":
        # Verify full hashes once on rank 0; every rank also checks lengths/partition metadata.
        manifest, paths = load_manifest(args.data_manifest, verify_hashes=rank == 0)
        if manifest["vocab_size"] != config.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match the model")
        corpus = {split: TokenCorpus(path, args.sequence_length, args.seed) for split, path in paths.items()}
    if world > 1:
        dist.barrier()

    # Initialize on CPU in the same order; QGDN's extra gate preserves the RNG stream.
    model = GPT(config)
    model.apply(lambda m: model._init_weights(m, n_layer=config.n_layer))
    if args.cpu:
        for block in model.transformer.h:
            block.attn.mode = "naive"
    model.gradient_checkpointing = not args.no_activation_checkpointing
    initial_shared_hash = shared_parameter_hash(model) if rank == 0 else None
    parameters = sum(p.numel() for p in model.parameters())
    extra_parameters = sum(p.numel() for n, p in model.named_parameters() if ".recall_" in n)
    model.to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model, args.weight_decay), lr=args.learning_rate,
                                 betas=(args.beta1, args.beta2), fused=device.type == "cuda")
    training_model = DistributedDataParallel(
        model,
        device_ids=[local_rank] if not args.cpu else None,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    ) if world > 1 else model
    training_loss = (
        FusedCrossEntropyLoss(inplace_backward=True)
        if args.training_loss == "fused"
        else None
    )
    immutable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                      if k not in {"output", "resume", "stop_after_step", "data_manifest"}}
    code_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run = dict(args=immutable_args, config=asdict(config), world_size=world, code_revision=code_revision,
               data_sha256=json_hash(manifest) if manifest else "mqar-v1", parameters=parameters,
               recall_parameters=extra_parameters, shared_initialization_sha256=initial_shared_hash,
               planned_tokens=args.max_steps * args.global_batch_size * args.sequence_length,
               validation_seed=172903, precision="fp32" if args.cpu else "bf16-mixed",
               checkpointing=model.gradient_checkpointing, numerics=numerics)
    identity = json_hash({k: v for k, v in run.items() if k != "shared_initialization_sha256"})
    step, train_seconds, wall_seconds = 0, 0.0, 0.0
    initial_validation = None
    if args.resume:
        # Never continue with a partially loaded model or reset only the step counter.
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["identity"] != identity:
            raise ValueError("Resume mismatch: code, config, data, topology or training hyperparameters changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, train_seconds, wall_seconds = checkpoint["step"], checkpoint["train_seconds"], checkpoint["wall_seconds"]
        initial_validation = checkpoint["initial_validation"]
        rng = checkpoint["rng"][rank]
        torch.set_rng_state(rng["torch"].cpu())
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if not args.cpu:
            torch.cuda.set_rng_state(rng["cuda"].cpu(), device)
        del checkpoint
        if not args.output.is_dir():
            raise ValueError("Resume into the original output directory")
        if json.loads((args.output / "run.json").read_text())["identity"] != identity:
            raise ValueError("Output directory belongs to a different run")
    elif rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "run.json", dict(run, identity=identity))
    if world > 1:
        dist.barrier()
    session_start = time.perf_counter()
    starting_wall = wall_seconds

    def batch(rows, validation=False):
        if args.task == "lm":
            x, y = corpus["val" if validation else "train"].batch(rows, shuffle=not validation)
        else:
            x, y = mqar_batch(rows, args.sequence_length, 172903 if validation else args.seed,
                             config.vocab_size, overwrite=args.mqar_overwrite)
        # Dataset construction fixes vocabulary, but malformed input must fail loudly.
        if x.min() < 0 or x.max() >= config.vocab_size or y.max() >= config.vocab_size:
            raise ValueError("Input token outside the declared vocabulary")
        return x.to(device), y.to(device)

    def log(record):
        if rank == 0:
            with (args.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        count = min(args.eval_sequences, corpus["val"].n_blocks) if args.task == "lm" else args.eval_sequences
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        rows = list(range(rank, count, world))
        for start in range(0, len(rows), args.micro_batch_size):
            x, y = batch(rows[start:start + args.micro_batch_size], validation=True)
            with amp():
                logits = model(x)
            totals[0] += F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum")
            valid = y != -100
            totals[1] += valid.sum()
            if args.task == "mqar":
                totals[2] += ((logits.argmax(-1) == y) & valid).sum()
        reduce_sum(totals, world)
        if totals[1] == 0:
            raise RuntimeError("Validation scored zero tokens")
        loss = (totals[0] / totals[1]).item()
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite validation loss")
        result = dict(kind="validation", step=step, tokens=step * args.global_batch_size * args.sequence_length,
                      loss=loss, scored_tokens=int(totals[1].item()), sequences=count)
        if args.task == "lm":
            result["perplexity"] = math.exp(loss)
        else:
            result["accuracy"] = (totals[2] / totals[1]).item()
        model.train()
        log(result)
        return result

    def save_checkpoint():
        rng = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())
        if not args.cpu:
            rng["cuda"] = torch.cuda.get_rng_state(device)
        all_rng = [None] * world
        if world > 1:
            dist.all_gather_object(all_rng, rng)
        else:
            all_rng[0] = rng
        if rank == 0:
            temporary = args.output / "checkpoint.pt.tmp"
            torch.save(dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                            train_seconds=train_seconds, wall_seconds=starting_wall + time.perf_counter() - session_start,
                            rng=all_rng, initial_validation=initial_validation), temporary)
            temporary.replace(args.output / "checkpoint.pt")
        if world > 1:
            dist.barrier()

    if step == 0:
        initial_validation = evaluate()
    accumulation = args.global_batch_size // (world * args.micro_batch_size)
    stop = args.stop_after_step or args.max_steps
    if not step <= stop:
        raise ValueError("Checkpoint is beyond the requested stopping step")
    last_validation = None
    while step < stop:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        should_log = step % args.log_every == 0 or step + 1 == stop
        for block in model.transformer.h:
            if hasattr(block.attn, "collect_gate_stats"):
                block.attn.collect_gate_stats = should_log
                if should_log:
                    block.attn.reset_gate_stats()
        lr = learning_rate(args, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for micro in range(accumulation):
            for block in model.transformer.h:
                if hasattr(block.attn, "collect_gate_stats"):
                    block.attn.collect_gate_stats = should_log
            begin = step * args.global_batch_size + (micro * world + rank) * args.micro_batch_size
            x, y = batch(range(begin, begin + args.micro_batch_size))
            sync = training_model.no_sync() if world > 1 and micro + 1 < accumulation else contextlib.nullcontext()
            with sync:
                with amp():
                    logits = training_model(x)
                    loss = (
                        training_loss(logits, y)
                        if training_loss is not None
                        else F.cross_entropy(
                            logits.float().flatten(0, 1),
                            y.flatten(),
                            ignore_index=-100,
                        )
                    )
                # Non-reentrant activation checkpointing recomputes layers during backward.
                # Disable side-effectful observation so every gate element is counted once.
                for block in model.transformer.h:
                    if hasattr(block.attn, "collect_gate_stats"):
                        block.attn.collect_gate_stats = False
                (loss / accumulation).backward()
            loss_sum += loss.detach() / accumulation
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        mean_loss = reduce_sum(loss_sum, world) / world
        if not torch.isfinite(mean_loss):
            raise FloatingPointError("Nonfinite training loss; checkpoint not advanced")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        train_seconds += elapsed
        step += 1
        for block in model.transformer.h:
            if hasattr(block.attn, "collect_gate_stats"):
                block.attn.collect_gate_stats = False
        if should_log:
            record = dict(kind="train", step=step, tokens=step * args.global_batch_size * args.sequence_length,
                          loss=mean_loss.item(), grad_norm=float(norm), lr=lr, step_seconds=elapsed,
                          tokens_per_second=args.global_batch_size * args.sequence_length / elapsed,
                          peak_memory_gb=0 if args.cpu else torch.cuda.max_memory_allocated() / 1e9)
            if args.task == "lm":
                record["data_epochs"] = step * args.global_batch_size / corpus["train"].n_blocks
            record.update(global_gate_statistics(model, world))
            log(record)
        if step % args.eval_every == 0 or step == stop:
            last_validation = evaluate()
        if step % args.save_every == 0 or step == stop:
            save_checkpoint()
    if last_validation is None:
        last_validation = evaluate()
    if rank == 0:
        wall_seconds = starting_wall + time.perf_counter() - session_start
        result = dict(status="completed" if step == args.max_steps else "paused", step=step,
                      trained_tokens=step * args.global_batch_size * args.sequence_length,
                      initial_validation=initial_validation, final_validation=last_validation,
                      parameters=parameters, recall_parameters=extra_parameters,
                      train_seconds=train_seconds, wall_seconds=wall_seconds,
                      gpu_hours=0 if args.cpu else wall_seconds * world / 3600,
                      peak_memory_gb=0 if args.cpu else torch.cuda.max_memory_allocated() / 1e9,
                      identity=identity)
        write_json(args.output / "summary.json", result)
        if step == args.max_steps:
            temporary = args.output / "model_final.pt.tmp"
            torch.save(dict(model=model.state_dict(), config=asdict(config), run=run, step=step), temporary)
            temporary.replace(args.output / "model_final.pt")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
