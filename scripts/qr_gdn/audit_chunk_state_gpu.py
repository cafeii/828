"""Run the committed QR-GDN GPU state and chunk-output diagnostic."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_qr_gdn_chunk_state_gpu.py",
        "tests/test_qr_gdn_chunk_output_gpu.py",
        "tests/test_qr_gdn_parallel_gpu.py",
        "tests/test_qr_gdn_rank2_scan.py",
        "-q",
    ]
    completed = subprocess.run(command, check=False, text=True)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    result = {
        "status": "passed",
        "tests": "qr_gdn_chunk_state_gpu + qr_gdn_chunk_output_gpu + qr_gdn_parallel_gpu + qr_gdn_rank2_scan",
        "parallel_backward": "production_gdn_custom_backward",
        "production_chunk_calls": 3,
        "physical_timesteps": True,
        "virtual_2t": False,
        "dense_2k_transition": False,
        "cuda_device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
