"""Prepare disjoint FineWeb train/validation files; never submits a Slurm job."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from data import file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True, help="Directory of source parquet files")
    p.add_argument("--tokenizer", type=Path, required=True, help="Local, fixed Llama-2 32k tokenizer directory")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--val-files", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()
    files = sorted(args.input_dir.resolve().rglob("*.parquet"))
    if not 0 < args.val_files < len(files):
        p.error("--val-files must leave at least one file in each partition")
    # Refuse all overwrites, including an incomplete earlier preparation.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import pyarrow.parquet as pq
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
    for split, paths in (("train", files[:-args.val_files]), ("val", files[-args.val_files:])):
        manifest["sources"][split] = []
        digest, count, documents = hashlib.sha256(), 0, 0
        temporary = args.output_dir / f"{split}.bin.partial"
        with temporary.open("wb") as stream:
            for path in paths:
                print(f"{split}: {path}", flush=True)
                manifest["sources"][split].append(dict(path=str(path), size=path.stat().st_size, sha256=file_sha256(path)))
                for batch in pq.ParquetFile(path).iter_batches(batch_size=args.batch_size, columns=["text"]):
                    texts = batch.column(0).to_pylist()
                    for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
                        ids.append(tokenizer.eos_token_id)
                        if min(ids) < 0 or max(ids) >= 32000:
                            raise ValueError("Tokenizer produced an out-of-vocabulary ID")
                        raw = np.asarray(ids, dtype="<u2").tobytes()
                        stream.write(raw)
                        digest.update(raw)
                        count += len(ids)
                        documents += 1
            stream.flush()
            os.fsync(stream.fileno())
        if count == 0:
            raise ValueError(f"Empty {split} partition")
        temporary.rename(args.output_dir / f"{split}.bin")
        manifest["splits"][split] = dict(file=f"{split}.bin", tokens=count, documents=documents, sha256=digest.hexdigest())
    if {f["sha256"] for f in manifest["sources"]["train"]} & {f["sha256"] for f in manifest["sources"]["val"]}:
        raise ValueError("Identical source files occur in both partitions")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Complete: {args.output_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
