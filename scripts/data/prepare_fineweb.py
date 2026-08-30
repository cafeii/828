# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# 适配自 litgpt/data/prepare_slimpajama.py：数据源从 jsonl.zst 换成 FineWeb parquet。
# 2026-08-30: 去掉 litgpt 依赖（服务器env未装），直接用 litdata DataChunkRecipe +
# transformers AutoTokenizer；tokenizer 定为 Llama-2 32k（与模型config的vocab_size=32000对齐）。
# token 存 uint16（vocab 32000 < 65536），训练侧读出后需 cast 到 long（scripts/pretrain.py已处理）。

import os
import time
from pathlib import Path

import numpy as np
from litdata.processing.data_processor import DataChunkRecipe, DataProcessor


class FineWebDataRecipe(DataChunkRecipe):
    is_generator = True

    def __init__(self, tokenizer_path: str, chunk_size: int):
        super().__init__(chunk_size)
        self.tokenizer_path = tokenizer_path
        self._tokenizer = None  # 每个worker进程内惰性构造

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return self._tokenizer

    def prepare_structure(self, input_dir):
        files = sorted(Path(input_dir).rglob("*.parquet"))
        assert files, f"{input_dir} 下没有parquet文件"
        return [str(file) for file in files]

    def prepare_item(self, filepath):
        import pyarrow.parquet as pq

        eos = self.tokenizer.eos_token_id
        parquet = pq.ParquetFile(filepath)
        # 按 row group 分批读，避免单个 2GB 文件解压后撑爆内存
        for batch in parquet.iter_batches(batch_size=1024, columns=["text"]):
            texts = batch.column("text").to_pylist()
            for ids in self.tokenizer(texts, add_special_tokens=False)["input_ids"]:
                yield np.asarray(ids + [eos], dtype=np.uint16)  # bos=False, eos=True


def prepare(
    input_dir: Path = Path("/work/projects/memos-b3/datasets/lzc_rnn/fineweb/sample/10BT"),
    output_dir: Path = Path("/work/projects/memos-b3/datasets/lzc_rnn/fineweb-litdata/10BT/train"),
    tokenizer_path: Path = Path("checkpoints/Llama-2-7b-hf"),
    chunk_size: int = (2049 * 16384),
    num_workers: int = None,
    fast_dev_run: bool = False,
) -> None:
    """把 FineWeb parquet 转成 litdata 预分块格式（供 scripts/pretrain.py 使用）。

    注意：FineWeb 无官方 validation/test。如需 validation，
    请先把个别 parquet 文件复制到单独目录，再对其跑一次本脚本。
    """
    data_recipe = FineWebDataRecipe(tokenizer_path=str(tokenizer_path), chunk_size=chunk_size)
    data_processor = DataProcessor(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        num_workers=num_workers or os.cpu_count(),
        num_downloaders=1,
    )

    start_time = time.time()
    data_processor.run(data_recipe)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, default=Path("/work/projects/memos-b3/datasets/lzc_rnn/fineweb/sample/10BT"))
    p.add_argument("--output_dir", type=Path, default=Path("/work/projects/memos-b3/datasets/lzc_rnn/fineweb-litdata/10BT/train"))
    p.add_argument("--tokenizer_path", type=Path, default=Path("checkpoints/Llama-2-7b-hf"))
    p.add_argument("--chunk_size", type=int, default=2049 * 16384)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    prepare(**vars(p.parse_args()))
