# 适配自 third_party/GatedDeltaNet-2/pretrain.py（Lightning AI, Apache 2.0）。
# 改动：
# - 数据侧换成 litdata StreamingDataset（scripts/data/prepare_fineweb.py 的产出格式）
# - 模型 import 自 model/lit_gpt（本工作区实现）
# - wandb 可选（--wandb 开启），默认CSV日志；去掉未用的 stream_tok / 时长退出路径

import argparse
import math
import os
import sys
import time
from pathlib import Path

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.loggers import CSVLogger

wd = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(wd / "model"))

from lit_gpt import FusedCrossEntropyLoss
from lit_gpt.config import Config
from lit_gpt.model import GPT, Block
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.utils import chunked_cross_entropy, num_parameters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--train_data_dir", type=str, required=True)
    p.add_argument("--val_data_dir", type=str, default=None)
    p.add_argument("--out_root", type=str, default="outputs/pretrain")
    p.add_argument("--devices", type=int, default=torch.cuda.device_count() or 1)
    p.add_argument("--nodes", type=int, default=1)
    # 训练规模
    p.add_argument("--max_tokens", type=int, default=int(10e9))
    p.add_argument("--micro_batch_size", type=int, default=4)
    p.add_argument("--global_batch_size", type=int, default=512)
    # 优化器
    p.add_argument("--learning_rate", type=float, default=4e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_tokens", type=int, default=int(0.5e9))
    # 日志与保存
    p.add_argument("--log_iter_interval", type=int, default=10)
    p.add_argument("--save_step_interval", type=int, default=500)
    p.add_argument("--eval_step_interval", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def create_dataloader(data_dir, block_size, batch_size, seed, num_workers, shuffle=True):
    from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

    dataset = StreamingDataset(
        input_dir=data_dir,
        item_loader=TokensLoader(block_size=block_size),
        shuffle=shuffle,
        drop_last=True,
        seed=seed,
    )
    return StreamingDataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, drop_last=True
    )


def get_lr(args, it, warmup_iters, max_iters):
    min_lr = args.learning_rate * args.min_lr_ratio
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)


def main():
    args = parse_args()
    out_dir = os.path.join(args.out_root, args.exp_name)
    args.gradient_accumulation_steps = args.global_batch_size // (args.micro_batch_size * args.devices * args.nodes)
    assert args.gradient_accumulation_steps > 0

    loggers = [CSVLogger(out_dir, name="csv")]
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger

        loggers.append(WandbLogger(project="rnn-lsa", name=args.exp_name, id=args.exp_name, save_dir=out_dir))

    if args.devices * args.nodes > 1:
        strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full")
    else:
        strategy = "auto"
    fabric = L.Fabric(
        devices=args.devices, num_nodes=args.nodes, strategy=strategy, precision="bf16-mixed", loggers=loggers
    )
    fabric.launch()
    fabric.seed_everything(args.seed)

    config = Config.from_name(args.model_name)
    resume = os.path.exists(os.path.join(out_dir, "latest-model-ckpt.pth"))
    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        fabric.print(f"Config: {config.__dict__}")
        fabric.print(f"grad_accum={args.gradient_accumulation_steps}, resume={resume}")
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=args.log_iter_interval)

    train_dataloader = create_dataloader(
        args.train_data_dir, config.block_size + 1, args.micro_batch_size, args.seed, args.num_workers
    )
    val_dataloader = (
        create_dataloader(
            args.val_data_dir, config.block_size + 1, args.micro_batch_size, args.seed, 1, shuffle=False
        )
        if args.val_data_dir
        else None
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(lambda m: model._init_weights(m, n_layer=config.n_layer))
    if fabric.global_rank == 0:
        fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
        fabric.print(f"Non-embedding parameters: {num_parameters(model.transformer.h):,}")
        fabric.print(f"Total parameters: {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        fused=(fabric.device.type == "cuda"),
    )
    optimizer = fabric.setup_optimizers(optimizer)
    state = {"model": model, "optimizer": optimizer, "iter_num": 0, "step_count": 0}

    if resume:
        ckpt = os.path.join(out_dir, "latest-model-ckpt.pth")
        fabric.print(f"Resuming from {ckpt}")
        fabric.load(ckpt, state)
        dl_state = os.path.join(out_dir, f"latest-data-state-rank{fabric.global_rank}.pth")
        if os.path.exists(dl_state):
            train_dataloader.load_state_dict(torch.load(dl_state, weights_only=False))

    train(args, out_dir, fabric, state, train_dataloader, val_dataloader, monitor)
    if fabric.device.type == "cuda" and fabric.global_rank == 0:
        fabric.print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(args, out_dir, fabric, state, train_dataloader, val_dataloader, monitor):
    model = state["model"]
    optimizer = state["optimizer"]

    tokens_per_iter = args.micro_batch_size * model.config.block_size
    max_iters = args.max_tokens // fabric.world_size // tokens_per_iter
    warmup_iters = args.warmup_tokens // fabric.world_size // tokens_per_iter
    loss_func = FusedCrossEntropyLoss()
    initial_iter = state["iter_num"]
    total_t0 = time.perf_counter()
    total_lengths = 0

    def save_checkpoint(final=False):
        name = "final" if final else "latest"
        path = os.path.join(out_dir, f"{name}-model-ckpt.pth")
        fabric.print(f"Saving checkpoint to {path!r}")
        if final:
            state["optimizer"] = None
        fabric.save(path, state)
        if not final:
            torch.save(
                train_dataloader.state_dict(),
                os.path.join(out_dir, f"latest-data-state-rank{fabric.global_rank}.pth"),
            )

    for train_data in train_dataloader:
        if state["iter_num"] >= max_iters:
            break
        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].long().contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].long().contiguous()

        lr = get_lr(args, state["iter_num"], warmup_iters, max_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        is_accumulating = (state["iter_num"] + 1) % args.gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = loss_func(logits, targets)
            fabric.backward(loss / args.gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        state["iter_num"] += 1
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        if fabric.global_rank == 0 and state["iter_num"] % args.log_iter_interval == 0:
            tokens_B = tokens_per_iter * state["iter_num"] * fabric.world_size / 1e9
            eta_h = (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"]) / 3600
            fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, lr {lr:.2e},"
                f" iter time {(t1 - iter_t0) * 1000:.2f}ms, trained {tokens_B:.3f}B tokens, ETA {eta_h:.2f}h"
            )
        monitor.on_train_batch_end(
            state["iter_num"] * args.micro_batch_size,
            t1 - total_t0,
            fabric.world_size,
            state["step_count"],
            flops_per_batch=1,
            lengths=total_lengths,
            train_loss=loss.item(),
        )

        if not is_accumulating and state["step_count"] % args.save_step_interval == 0:
            save_checkpoint()
        if (
            val_dataloader is not None
            and not is_accumulating
            and state["step_count"] % args.eval_step_interval == 0
        ):
            t0 = time.perf_counter()
            val_loss = validate(args, fabric, model, val_dataloader)
            if fabric.global_rank == 0:
                fabric.print(f"step {state['step_count']}: val loss {val_loss:.4f}, val time {time.perf_counter()-t0:.1f}s")
                fabric.log_dict({"metric/val_loss": val_loss, "metric/val_ppl": math.exp(val_loss)}, state["step_count"])
            fabric.barrier()

    save_checkpoint(final=True)


@torch.no_grad()
def validate(args, fabric, model, val_dataloader):
    model.eval()
    losses = 0.0
    k = 0
    for k, val_data in enumerate(val_dataloader):
        if k >= args.eval_iters:
            break
        input_ids = val_data[:, 0 : model.config.block_size].long().contiguous()
        targets = val_data[:, 1 : model.config.block_size + 1].long().contiguous()
        logits = model(input_ids)
        losses += chunked_cross_entropy(logits, targets, chunk_size=0).item()
    model.train()
    return losses / max(k, 1)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
