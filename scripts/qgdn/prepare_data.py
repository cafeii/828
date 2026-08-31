"""Prepare disjoint FineWeb train/validation files; never submits a Slurm job."""
import argparse
import hashlib
import json
import os
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from data import file_sha256


def tokenize_source(job):
    """One source per process; the parent merges shards in source order."""
    index, source, tokenizer_path, output, batch_size = job
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    path = Path(source)
    before = path.stat()
    source_hash = file_sha256(path)
    shard = Path(output) / f"source-{index:05d}.bin.partial"
    count = documents = 0
    print(f"tokenizing: {path}", flush=True)
    with shard.open("xb") as stream:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["text"]):
            for ids in tokenizer(batch.column(0).to_pylist(), add_special_tokens=False)["input_ids"]:
                ids.append(tokenizer.eos_token_id)
                if min(ids) < 0 or max(ids) >= 32000:
                    raise ValueError("Tokenizer produced an out-of-vocabulary ID")
                stream.write(np.asarray(ids, dtype="<u2").tobytes())
                count += len(ids)
                documents += 1
        stream.flush()
        os.fsync(stream.fileno())
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed during tokenization: {path}")
    print(f"completed: {path.name}, tokens={count}, documents={documents}", flush=True)
    return dict(shard=str(shard), tokens=count, documents=documents,
                source=dict(path=str(path), size=before.st_size, sha256=source_hash))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True, help="Directory of source parquet files")
    p.add_argument("--tokenizer", type=Path, required=True, help="Local, fixed Llama-2 32k tokenizer directory")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--val-files", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=1, help="Parallel source files; token order is unchanged")
    args = p.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        p.error("Worker and batch counts must be positive")
    if args.workers > int(os.environ.get("SLURM_CPUS_PER_TASK", args.workers)):
        p.error("Workers exceed allocated CPUs")
    files = sorted(args.input_dir.resolve().rglob("*.parquet"))
    if not 0 < args.val_files < len(files):
        p.error("--val-files must leave at least one file in each partition")
    # Refuse all overwrites, including an incomplete earlier preparation.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if len(tokenizer) != 32000 or tokenizer.eos_token_id is None:
        raise ValueError("This experiment fixes a 32000-token vocabulary with an EOS token")
    tokenizer_names = {"tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                       "special_tokens_map.json", "added_tokens.json", "config.json"}
    tokenizer_files = {f.name: file_sha256(f)
                       for f in sorted(args.tokenizer.iterdir()) if f.is_file() and f.name in tokenizer_names}
    if not tokenizer_files:
        raise ValueError("Tokenizer files not found")
    manifest = dict(format="qgdn-u16-v1", vocab_size=32000, dtype="<u2",
                    tokenizer_files=tokenizer_files, bos=False, eos=True,
                    split_policy="sorted_source_files_tail_holdout", sources={}, splits={})
    jobs = [(i, str(path), str(args.tokenizer), str(args.output_dir), args.batch_size)
            for i, path in enumerate(files)]
    if args.workers == 1:
        records = [tokenize_source(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            records = list(pool.map(tokenize_source, jobs))
    for split, selected in (("train", records[:-args.val_files]), ("val", records[-args.val_files:])):
        manifest["sources"][split] = [r["source"] for r in selected]
        digest, count, documents = hashlib.sha256(), 0, 0
        temporary = args.output_dir / f"{split}.bin.partial"
        with temporary.open("xb") as stream:
            for record in selected:
                with Path(record["shard"]).open("rb") as shard:
                    while raw := shard.read(8 * 1024 * 1024):
                        stream.write(raw)
                        digest.update(raw)
                count += record["tokens"]
                documents += record["documents"]
            stream.flush()
            os.fsync(stream.fileno())
        if count == 0:
            raise ValueError(f"Empty {split} partition")
        temporary.rename(args.output_dir / f"{split}.bin")
        manifest["splits"][split] = dict(file=f"{split}.bin", tokens=count, documents=documents, sha256=digest.hexdigest())
    if {f["sha256"] for f in manifest["sources"]["train"]} & {f["sha256"] for f in manifest["sources"]["val"]}:
        raise ValueError("Identical source files occur in both partitions")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for record in records:
        Path(record["shard"]).unlink()
    print(f"Complete: {args.output_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
