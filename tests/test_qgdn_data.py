import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "qgdn"))
from data import TokenCorpus, file_sha256, load_manifest, mqar_batch
from summarize import pairing_key
from runtime import configure_device_from_cli


def test_cpu_mode_masks_scheduler_visibility_and_gpu_mode_requires_allocation():
    env = {"CUDA_VISIBLE_DEVICES": "0,1", "SLURM_JOB_ID": "123", "QGDN_REQUESTED_GPUS": "0"}
    configure_device_from_cli(["--cpu"], env)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    with pytest.raises(RuntimeError, match="zero GPUs"):
        configure_device_from_cli([], env)
    with pytest.raises(RuntimeError, match="Slurm GPU allocation"):
        configure_device_from_cli([], {})
    gpu_env = {"SLURM_JOB_ID": "123", "SLURM_JOB_GPUS": "0", "QGDN_REQUESTED_GPUS": "1"}
    configure_device_from_cli([], gpu_env)


def test_global_stream_is_independent_of_rank_partition_and_resume(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(1100, dtype="<u2").tofile(path)
    data = TokenCorpus(path, sequence_length=9, seed=3407)
    whole = data.batch(range(96, 128), shuffle=True)[0]
    partitioned = torch.cat([data.batch(range(start, start + 4), shuffle=True)[0] for start in range(96, 128, 4)])
    torch.testing.assert_close(whole, partitioned)
    restored = TokenCorpus(path, sequence_length=9, seed=3407)
    torch.testing.assert_close(data.batch(range(120, 128), True)[0], restored.batch(range(120, 128), True)[0])
    with pytest.raises(ValueError, match="must not wrap"):
        data.batch([data.n_blocks], False)


@pytest.mark.parametrize("overwrite", [False, True])
def test_mqar_answers_are_last_causal_write_and_labels_are_shifted(overwrite):
    x, y = mqar_batch(range(5), 128, 17, 256, overwrite=overwrite)
    assert (y != -100).sum().item() == 20
    for tokens, labels in zip(x.tolist(), y.tolist()):
        memory = {}
        for i in range(len(tokens)):
            if tokens[i] == 2:
                memory[tokens[i+1]] = tokens[i+2]
            if tokens[i] == 1:
                assert labels[i+1] == memory[tokens[i+1]]
    again = mqar_batch(range(5), 128, 17, 256, overwrite=overwrite)
    torch.testing.assert_close(x, again[0])
    assert not torch.equal(x, mqar_batch(range(5), 128, 18, 256, overwrite=overwrite)[0])
    longer = mqar_batch(range(5), 256, 17, 256, overwrite=overwrite)
    torch.testing.assert_close(x[:, :24], longer[0][:, :24])
    torch.testing.assert_close(y[y != -100], longer[1][longer[1] != -100])


def test_manifest_rejects_contamination_and_changed_bytes(tmp_path):
    splits = {}
    for i, split in enumerate(("train", "val")):
        path = tmp_path / f"{split}.bin"
        np.full(32, i, dtype="<u2").tofile(path)
        splits[split] = dict(file=path.name, tokens=32, sha256=file_sha256(path))
    manifest = dict(format="qgdn-u16-v1", vocab_size=256, splits=splits,
                    sources={"train": [dict(sha256="source_a")], "val": [dict(sha256="source_b")]})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    load_manifest(path)
    manifest["sources"]["val"][0]["sha256"] = "source_a"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Overlapping"):
        load_manifest(path)
    manifest["sources"]["val"][0]["sha256"] = "source_b"
    path.write_text(json.dumps(manifest))
    (tmp_path / "val.bin").write_bytes(b"x" * 64)
    with pytest.raises(ValueError, match="hash changed"):
        load_manifest(path)


def test_pairing_rejects_data_and_hyperparameter_changes():
    run = dict(args=dict(model="gdn_control_340M", seed=1, learning_rate=4e-4),
               config=dict(name="gdn_control_340M", mixer="gdn", n_layer=20),
               data_sha256="data", code_revision="code", world_size=2,
               shared_initialization_sha256="shared", precision="bf16-mixed")
    other = json.loads(json.dumps(run))
    other["args"]["model"] = "qgdn_340M"
    other["config"].update(name="qgdn_340M", mixer="qgdn", recall_mode="query")
    assert pairing_key(run) == pairing_key(other)
    other["args"]["learning_rate"] = 8e-4
    assert pairing_key(run) != pairing_key(other)
