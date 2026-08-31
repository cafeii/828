"""Deterministic, globally indexed token streams for paired experiments."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path, verify_hashes=True):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest["format"] != "qgdn-u16-v1":
        raise ValueError("Expected a manifest from scripts/qgdn/prepare_data.py")
    if not 1 < manifest["vocab_size"] <= 65536:
        raise ValueError("Invalid uint16 vocabulary")
    files = []
    for split in ("train", "val"):
        info = manifest["splits"][split]
        file = (path.parent / info["file"]).resolve()
        if file.stat().st_size != 2 * info["tokens"] or info["tokens"] <= 0:
            raise ValueError(f"Invalid/truncated {split} token file: {file}")
        if verify_hashes and file_sha256(file) != info["sha256"]:
            raise ValueError(f"Content hash changed: {file}")
        files.append(file)
    if files[0] == files[1] or manifest["splits"]["train"]["sha256"] == manifest["splits"]["val"]["sha256"]:
        raise ValueError("Train and validation token files must be distinct")
    sources = manifest["sources"]
    if not sources["train"] or not sources["val"]:
        raise ValueError("Both source partitions must be recorded")
    train_hashes = {item["sha256"] for item in sources["train"]}
    if train_hashes & {item["sha256"] for item in sources["val"]}:
        raise ValueError("Overlapping source files across train and validation")
    return manifest, dict(zip(("train", "val"), files))


class TokenCorpus:
    def __init__(self, path, sequence_length, seed):
        self.tokens = np.memmap(path, mode="r", dtype="<u2")
        self.width = sequence_length + 1
        self.n_blocks = len(self.tokens) // self.width
        if not self.n_blocks:
            raise ValueError("Corpus does not contain one complete sequence")
        self.seed = seed
        self.permutations = {}

    def batch(self, rows, shuffle):
        ids = []
        for row in rows:
            epoch, position = divmod(int(row), self.n_blocks)
            if shuffle:
                if epoch not in self.permutations:
                    self.permutations[epoch] = np.random.default_rng(self.seed + epoch).permutation(self.n_blocks)
                    for old in list(self.permutations):
                        if old < epoch - 1:
                            del self.permutations[old]
                position = self.permutations[epoch][position]
            elif epoch:
                raise ValueError("Validation must not wrap or repeat examples")
            begin = position * self.width
            ids.append(np.array(self.tokens[begin:begin + self.width], dtype=np.int64))
        packed = torch.from_numpy(np.stack(ids))
        return packed[:, :-1].contiguous(), packed[:, 1:].contiguous()


def mqar_batch(rows, sequence_length, seed, vocab_size, pairs=8, queries=4, overwrite=False):
    """Causal multi-query associative recall with fresh bindings per example.

    Writes are [WRITE,key,value]; reads are [QUERY,key,value]. Only answer
    positions are scored. Random fillers separate writes and queries. Values
    already answered remain in the causal history; query keys are distinct.
    Optional repeated writes test whether preserving memory prevents overwriting.
    """
    if vocab_size < 256 or pairs > 112 or queries > pairs:
        raise ValueError("MQAR requires vocab>=256, queries<=pairs<=112")
    width = sequence_length + 1
    rewrites = max(1, pairs // 4) if overwrite else 0
    if 3 * (pairs + queries + rewrites) > width:
        raise ValueError("Sequence too short for MQAR payload")
    batches, labels = [], []
    for row in rows:
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(row)]))
        filler_rng = np.random.default_rng(np.random.SeedSequence([seed, int(row), 1]))
        tokens = filler_rng.integers(3, 16, width, dtype=np.int64)
        targets = np.full(sequence_length, -100, dtype=np.int64)
        keys = rng.choice(np.arange(16, 128), size=pairs, replace=False)
        values = rng.integers(128, vocab_size, pairs)
        for i, (key, value) in enumerate(zip(keys, values)):
            tokens[3*i:3*i+3] = [2, key, value]
        for i in range(rewrites):
            values[i] = 128 + (values[i] - 128 + rng.integers(1, vocab_size - 128)) % (vocab_size - 128)
            start = 3 * pairs + 3 * i
            tokens[start:start+3] = [2, keys[i], values[i]]
        query_ids = rng.choice(pairs, size=queries, replace=False)
        start = width - 3 * queries
        for i, index in enumerate(query_ids):
            pos = start + 3 * i
            tokens[pos:pos+3] = [1, keys[index], values[index]]
            targets[pos + 1] = values[index]  # logit at key predicts the following value
        batches.append(tokens[:-1])
        labels.append(targets)
    return torch.from_numpy(np.stack(batches)), torch.from_numpy(np.stack(labels))
