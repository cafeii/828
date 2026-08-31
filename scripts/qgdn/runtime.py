"""Admission checks before importing CUDA-aware libraries in experiment CLIs."""
import argparse
import os
import sys


def configure_device_from_cli(argv=None, environ=None):
    argv = sys.argv[1:] if argv is None else argv
    env = os.environ if environ is None else environ
    if "--help" in argv or "-h" in argv:
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu", action="store_true")
    args, _ = parser.parse_known_args(argv)
    if args.cpu:
        # Set before torch/fla imports, and inherit this setting in every child.
        env["CUDA_VISIBLE_DEVICES"] = ""
        return
    if env.get("QGDN_REQUESTED_GPUS") == "0":
        raise RuntimeError("This job requested zero GPUs. Use --cpu for the tiny validation suite.")
    if not env.get("SLURM_JOB_ID") or not env.get("SLURM_JOB_GPUS"):
        raise RuntimeError("GPU experiment CLIs require a Slurm GPU allocation; use --cpu only for tiny checks.")
