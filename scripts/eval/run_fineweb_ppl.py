#!/usr/bin/env python
# FineWeb-test ppl：在 held-out val litdata（预处理时切出的尾部 1 个 parquet）上算困惑度。
# 口径与训练时 validate() 一致（block_size 截断、chunked CE），但默认遍历全部 val 数据
# 而非 eval_iters 截断，给出确定性的最终数字。跑在训练环境（lzc-rnn / 主 .venv，需 litdata）。
#   python scripts/eval/run_fineweb_ppl.py --model_name gdn2_lsa_340M \
#     --ckpt_path outputs/.../final-model-ckpt.pth --val_data_dir <litdata val 目录>
import argparse
import math
import sys
from pathlib import Path

import torch

_WD = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WD / "model"))
sys.path.insert(0, str(_WD / "scripts" / "eval"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--val_data_dir", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=0, help="0=全量")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from lit_gpt.utils import chunked_cross_entropy
    from wrapper import load_eval_model

    from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

    model, _ = load_eval_model(
        args.model_name, args.ckpt_path,
        tokenizer_path=str(_WD / "checkpoints" / "Llama-2-7b-hf"),
        device=args.device, dtype=torch.bfloat16,
    )
    block_size = model.gpt.config.block_size

    dataset = StreamingDataset(
        input_dir=args.val_data_dir,
        item_loader=TokensLoader(block_size=block_size + 1),
        shuffle=False, drop_last=True, seed=42,
    )
    loader = StreamingDataLoader(dataset, batch_size=args.batch_size, num_workers=2, drop_last=True)

    total_loss, n = 0.0, 0
    with torch.inference_mode():
        for k, batch in enumerate(loader):
            if args.max_batches and k >= args.max_batches:
                break
            batch = batch.to(args.device)
            input_ids = batch[:, 0:block_size].long().contiguous()
            targets = batch[:, 1 : block_size + 1].long().contiguous()
            logits = model.gpt(input_ids)
            total_loss += chunked_cross_entropy(logits, targets, chunk_size=0).item()
            n += 1
            if n % 20 == 0:
                print(f"[{n}] running loss {total_loss / n:.4f}", flush=True)

    loss = total_loss / max(n, 1)
    print(f"fineweb_val: batches={n} loss={loss:.4f} ppl={math.exp(loss):.3f}")


if __name__ == "__main__":
    main()
