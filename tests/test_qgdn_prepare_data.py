"""Verify parallel preparation preserves exact token order and held-out sources."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def test_parallel_preparation_matches_serial_and_expected_tokens(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    vocabulary = {"<unk>": 0, "<eos>": 1, **{f"w{i}": i + 2 for i in range(31998)}}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    documents = [["w1 w2", "w3"], ["w4 w5 w6"], ["w7 w8"]]
    for i, texts in enumerate(documents):
        pq.write_table(pa.table({"text": texts}), source / f"{i:03d}.parquet")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TOKENIZERS_PARALLELISM": "false"}
    manifests = []
    for workers in (1, 2):
        output = tmp_path / f"out-{workers}"
        command = [sys.executable, str(root / "scripts/qgdn/prepare_data.py"), "--input-dir", str(source),
                   "--tokenizer", str(tokenizer_path), "--output-dir", str(output), "--workers", str(workers)]
        subprocess.run(command, env=env, check=True)
        manifests.append(json.loads((output / "manifest.json").read_text()))
        for split, selected in (("train", documents[:2]), ("val", documents[2:])):
            expected = [token for group in selected for text in group
                        for token in tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]]
            np.testing.assert_array_equal(np.fromfile(output / f"{split}.bin", dtype="<u2"), expected)
        assert not list(output.glob("*.partial"))
        # Existing results are immutable even if a new invocation uses the same arguments.
        assert subprocess.run(command, env=env, capture_output=True).returncode != 0
    assert manifests[0] == manifests[1]
    assert {x["path"] for x in manifests[0]["sources"]["train"]}.isdisjoint(
        x["path"] for x in manifests[0]["sources"]["val"])
