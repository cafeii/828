"""Bounded verification suite; run inside an allocated Slurm job, never submits."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from runtime import configure_device_from_cli

if __name__ == "__main__":
    configure_device_from_cli()

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from lit_gpt.config import Config
from lit_gpt.model import GPT
from data import file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--full-model", action="store_true", help="Also run one 4096-token optimizer step of each ~340M model")
    p.add_argument("--cpu", action="store_true", help="Mask CUDA before imports and run only CPU/tiny checks")
    args = p.parse_args()
    if args.cpu and args.full_model:
        p.error("--full-model requires an allocated GPU, not --cpu")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", cuda=torch.cuda.is_available(), gpu_count=torch.cuda.device_count(), commands=[],
                  gpu_parity_verified=False, ddp_verified=False, full_model_verified=False,
                  scope="cuda" if torch.cuda.is_available() else "cpu-reference-and-tiny-training")

    def run(command, env=None):
        command = [str(x) for x in command]
        print("RUN", command, flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        report["commands"].append(command)

    run([sys.executable, "-m", "pytest", "tests/test_qgdn.py", "tests/test_qgdn_data.py", "-q",
         f"--junitxml={args.output / 'tests.xml'}"])
    report["gpu_parity_verified"] = torch.cuda.is_available()
    counts = {}
    with torch.device("meta"):
        for name in ("gdn_control_340M", "qgdn_340M", "qgdn_key_340M", "qgdn_isotropic_340M", "qgdn_fixed_340M"):
            m = GPT(Config.from_name(name))
            counts[name] = sum(p.numel() for p in m.parameters())
    report["parameter_counts"] = counts

    # Real next-token LM loader and validation path, using a tiny synthetic corpus.
    data_root = args.output / "fixture"
    data_root.mkdir()
    manifest = dict(format="qgdn-u16-v1", vocab_size=256, splits={}, sources={})
    for i, split in enumerate(("train", "val")):
        path = data_root / f"{split}.bin"
        np.random.default_rng(i + 23).integers(0, 256, size=129 * 64, dtype=np.uint16).astype("<u2").tofile(path)
        manifest["splits"][split] = dict(file=path.name, tokens=129 * 64, sha256=file_sha256(path))
        manifest["sources"][split] = [dict(sha256=f"synthetic-fixture-{split}")]
    manifest_path = data_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    base = [sys.executable, "scripts/qgdn/train.py", "--sequence-length", "128", "--max-steps", "3",
            "--global-batch-size", "2", "--micro-batch-size", "1", "--eval-sequences", "4",
            "--log-every", "1", "--eval-every", "3", "--save-every", "3"]
    if not torch.cuda.is_available():
        base += ["--cpu"]
    for variant in ("gdn", "qgdn"):
        run(base + ["--model", f"{variant}_recall_tiny", "--task", "lm", "--data-manifest", manifest_path,
                    "--output", args.output / f"lm-{variant}"])
    run([sys.executable, "scripts/qgdn/summarize.py", args.output / "lm-gdn", args.output / "lm-qgdn",
         "--output", args.output / "paired-smoke.json"])
    # Continuous versus interrupted/resumed runs use the same full schedule and data positions.
    paired = base + ["--model", "qgdn_recall_tiny", "--task", "mqar"]
    run(paired + ["--output", args.output / "continuous"])
    run(paired + ["--output", args.output / "resumed", "--stop-after-step", "1"])
    run(paired + ["--output", args.output / "resumed", "--resume", args.output / "resumed/checkpoint.pt"])
    a = torch.load(args.output / "continuous/checkpoint.pt", map_location="cpu", weights_only=False)
    b = torch.load(args.output / "resumed/checkpoint.pt", map_location="cpu", weights_only=False)
    assert a["step"] == b["step"] == 3
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], atol=3e-7, rtol=3e-6)
    report["resume_matches_continuous"] = True
    # A changed seed must not silently restart or partly load a checkpoint.
    mismatch = subprocess.run([str(x) for x in paired + ["--output", args.output / "resumed", "--resume",
                                args.output / "resumed/checkpoint.pt", "--seed", "9"]], cwd=ROOT)
    assert mismatch.returncode != 0
    report["mismatched_resume_rejected"] = True
    run([sys.executable, "scripts/qgdn/evaluate.py", "--checkpoint", args.output / "continuous/model_final.pt",
         "--lengths", "128", "256", "--mqar-sequences", "4", "--output", args.output / "delay-eval.json"]
        + ([] if torch.cuda.is_available() else ["--cpu"]))
    if torch.cuda.device_count() >= 2:
        run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2",
             "scripts/qgdn/train.py", "--model", "qgdn_recall_tiny", "--task", "mqar", "--sequence-length", "128",
             "--max-steps", "2", "--global-batch-size", "4", "--eval-sequences", "4", "--output", args.output / "ddp"])
        report["ddp_verified"] = True
    if args.full_model:
        if not torch.cuda.is_available():
            raise RuntimeError("Full-model smoke requires CUDA")
        for name in ("gdn_control_340M", "qgdn_340M"):
            run([sys.executable, "scripts/qgdn/train.py", "--model", name, "--task", "mqar", "--sequence-length", "4096",
                 "--max-steps", "1", "--global-batch-size", "1", "--eval-sequences", "1", "--output", args.output / name])
        report["full_model_verified"] = True
    report["status"] = "passed"
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
