"""Run CUDA correctness tests before benchmarking the physical-token kernel."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_qgdn.py",
            "-q",
            f"--junitxml={args.junit}",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "qgdn" / "benchmark_physical_t.py"),
            "--output", str(args.output),
            "--sequence-length", str(args.sequence_length),
            "--micro-batch-size", str(args.micro_batch_size),
            "--warmup", str(args.warmup),
            "--measured", str(args.measured),
        ],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
