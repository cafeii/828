# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# 适配自 litgpt/data/prepare_slimpajama.py：数据源从 jsonl.zst 换成 FineWeb parquet。

import os
import time
from pathlib import Path

from litgpt.data.prepare_starcoder import DataChunkRecipe
from litgpt.tokenizer import Tokenizer
from litgpt.utils import CLI, extend_checkpoint_dir


class FineWebDataRecipe(DataChunkRecipe):
    is_generator = True

    def __init__(self, tokenizer: Tokenizer, chunk_size: int):
        super().__init__(chunk_size)
        self.tokenizer = tokenizer

    def prepare_structure(self, input_dir):
        files = sorted(Path(input_dir).rglob("*.parquet"))
        return [str(file) for file in files]

    def prepare_item(self, filepath):
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(filepath)
        # 按 row group 分批读，避免单个 2GB 文件解压后撑爆内存
        for batch in parquet.iter_batches(batch_size=1024, columns=["text"]):
            for text in batch.column("text").to_pylist():
                yield self.tokenizer.encode(string=text, bos=False, eos=True)


def prepare(
    input_dir: Path = Path("/work/projects/memos-b3/datasets/lzc_rnn/fineweb/sample/100BT"),
    output_dir: Path = Path("/work/projects/memos-b3/datasets/lzc_rnn/fineweb-litdata/100BT/train"),
    tokenizer_path: Path = Path("checkpoints/Llama-3-8b"),
    chunk_size: int = (2049 * 16384),
    fast_dev_run: bool = False,
) -> None:
    """把 FineWeb parquet 转成 litdata 预分块格式（供 litgpt pretrain 使用）。

    注意：FineWeb 无官方 validation/test。如需 validation，
    请先把个别 parquet 文件移到单独目录，再对其跑一次本脚本。
    """
    from litdata.processing.data_processor import DataProcessor

    tokenizer_path = extend_checkpoint_dir(tokenizer_path)
    tokenizer = Tokenizer(tokenizer_path)
    data_recipe = FineWebDataRecipe(tokenizer=tokenizer, chunk_size=chunk_size)
    data_processor = DataProcessor(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        num_workers=os.cpu_count(),
        num_downloaders=1,
    )

    start_time = time.time()
    data_processor.run(data_recipe)
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    CLI(prepare)
