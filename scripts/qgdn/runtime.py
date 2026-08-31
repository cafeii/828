"""Admission checks before importing CUDA-aware libraries in experiment CLIs."""
import argparse
import os
import sys

CUDA_NUMERICS = dict(policy="qgdn-repro-v1", deterministic_algorithms=True,
                     cublas_workspace_config=":4096:8", cudnn_benchmark=False,
                     cudnn_deterministic=True, float32_matmul_precision="high",
                     fused_rmsnorm_num_warps=4)


def configure_device_from_cli(argv=None, environ=None):
    argv = sys.argv[1:] if argv is None else argv
    env = os.environ if environ is None else environ
    if "--help" in argv or "-h" in argv:
        return
    # cuBLAS reads this when its handle is created, before any training matmul.
    env["CUBLAS_WORKSPACE_CONFIG"] = CUDA_NUMERICS["cublas_workspace_config"]
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


def configure_numerics(*, cpu=False):
    """Use one measured numerical policy for both arms of the paired study.

The resume diagnostic found different per-process RMSNorm autotuning choices.
Pin the existing fused kernels, not their equations, before the first forward.
This does not promise determinism for arbitrary custom kernels or hardware;
the repeated-run and resume checks still have to pass on the allocated GPUs.
"""
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUDA_NUMERICS["cublas_workspace_config"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    numerics = dict(CUDA_NUMERICS)
    if cpu:
        numerics.update(cublas_workspace_config=None, fused_rmsnorm_num_warps=None)
        return numerics
    from lit_gpt import rmsnorm

    for name in ("_layer_norm_fwd_1pass_kernel", "_layer_norm_bwd_kernel"):
        tuner = getattr(rmsnorm, name)
        configs = [c for c in tuner.configs if c.num_warps == CUDA_NUMERICS["fused_rmsnorm_num_warps"]]
        if len(configs) != 1:
            raise RuntimeError(f"Unsupported fused RMSNorm configuration: {name}")
        tuner.configs = configs
        tuner.cache.clear()
    return numerics
