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


def _default_num_workers() -> int:
    """按Slurm实际分配的核数定worker数。

    os.cpu_count()在Slurm里返回整个节点的核数（B3为224），而作业通常只分到
    其中一部分，直接用会超订、反而变慢。sched_getaffinity才反映cgroup限额。
    """
    n = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if n:
        return int(n)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


class FineWebDataRecipe(DataChunkRecipe):
    is_generator = True

    def __init__(self, tokenizer_path: str, chunk_size: int, row_groups_per_item: int = 16):
        super().__init__(chunk_size)
        self.tokenizer_path = tokenizer_path
        self.row_groups_per_item = row_groups_per_item
        self._tokenizer = None  # 每个worker进程内惰性构造

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return self._tokenizer

    def prepare_structure(self, input_dir):
        import pyarrow.parquet as pq

        files = sorted(Path(input_dir).rglob("*.parquet"))
        assert files, f"{input_dir} 下没有parquet文件"
        # item粒度取 (文件, row_group区间) 而非整文件：10BT只有15个2GB文件，
        # 按文件分发时并行度被文件数卡死（40 worker只有15个在干活）。
        # 10BT共14874个row group，per_item=16 → 937个item，对64 worker约15个/worker，
        # 尾部不均衡可控；单item约32MB，摊得住进程池调度开销。
        items = []
        for file in files:
            num_rg = pq.ParquetFile(file).num_row_groups
            for start in range(0, num_rg, self.row_groups_per_item):
                items.append((str(file), start, min(start + self.row_groups_per_item, num_rg)))
        return items

    def prepare_item(self, item):
        import pyarrow.parquet as pq

        filepath, rg_start, rg_end = item
        # litdata队列传输会把tuple变list、int字符串化，int()做幂等兼容
        rg_start, rg_end = int(rg_start), int(rg_end)
        eos = self.tokenizer.eos_token_id
        parquet = pq.ParquetFile(filepath)
        # 逐 row group 读，避免单个 2GB 文件解压后撑爆内存
        for rg in range(rg_start, rg_end):
            texts = parquet.read_row_group(rg, columns=["text"]).column("text").to_pylist()
            for ids in self.tokenizer(texts, add_special_tokens=False)["input_ids"]:
                yield np.asarray(ids + [eos], dtype=np.uint16)  # bos=False, eos=True


def prepare(
    input_dir: Path = Path("/work/projects/memos-b3/datasets/lzc_rnn/fineweb/sample/10BT"),
    output_dir: Path = Path("/work/projects/memos-b3/datasets/lzc_rnn/fineweb-litdata/10BT/train"),
    tokenizer_path: Path = Path("checkpoints/Llama-2-7b-hf"),
    chunk_size: int = (2049 * 16384),
    num_workers: int = None,
    row_groups_per_item: int = 64,
    fast_dev_run: bool = False,
) -> None:
    """把 FineWeb parquet 转成 litdata 预分块格式（供 scripts/pretrain.py 使用）。

    注意：FineWeb 无官方 validation/test。如需 validation，
    请先把个别 parquet 文件复制到单独目录，再对其跑一次本脚本。
    """
    data_recipe = FineWebDataRecipe(
        tokenizer_path=str(tokenizer_path), chunk_size=chunk_size, row_groups_per_item=row_groups_per_item
    )
    data_processor = DataProcessor(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        num_workers=num_workers or _default_num_workers(),
        num_downloaders=1,
        # 共享节点会清/dev/shm（见9c0c857前科）：spawn子进程按名重建SemLock会
        # FileNotFoundError秒死且litdata误报完成；fork继承已打开对象，免疫清理。
        start_method="fork",
    )

    start_time = time.time()
    data_processor.run(data_recipe)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    import argparse

    # worker进程内tokenizer各自串行，rust侧并行会与进程池争CPU并触发fork警告
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, default=Path("/work/projects/memos-b3/datasets/lzc_rnn/fineweb/sample/10BT"))
    p.add_argument("--output_dir", type=Path, default=Path("/work/projects/memos-b3/datasets/lzc_rnn/fineweb-litdata/10BT/train"))
    p.add_argument("--tokenizer_path", type=Path, default=Path("checkpoints/Llama-2-7b-hf"))
    p.add_argument("--chunk_size", type=int, default=2049 * 16384)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--row_groups_per_item", type=int, default=16)
    p.add_argument("--fast_dev_run", action="store_true")
    prepare(**vars(p.parse_args()))
