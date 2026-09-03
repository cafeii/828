"""Targeted CUDA audit of the production QGDN recurrence against its reference and GDN limit."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from runtime import configure_device_from_cli, configure_numerics

if __name__ == "__main__":
    configure_device_from_cli()

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from lit_gpt.kernels import get_chunk_gated_delta_rule
from lit_gpt.mixers.qgdn_reference import qgdn_reference
from lit_gpt.mixers.qgdn_rule import qgdn_rule


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_rmse(actual, expected):
    actual, expected = actual.float(), expected.float()
    denominator = expected.square().mean().sqrt().clamp_min(1e-6)
    return ((actual - expected).square().mean().sqrt() / denominator).item()


def tensors(length, gamma_value, *, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, length, 2)
    q = torch.randn(*shape, 64, generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(*shape, 64, generator=generator, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(*shape, 64, generator=generator, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(*shape, generator=generator, device="cuda", dtype=torch.float32) * 0.8
    beta = torch.rand(*shape, generator=generator, device="cuda", dtype=torch.float32)
    gamma = torch.full(shape, gamma_value, device="cuda", dtype=torch.float32)
    state = torch.randn(1, 2, 64, 64, generator=generator, device="cuda", dtype=torch.float32)
    return [x.requires_grad_() for x in (q, k, v, g, beta, gamma, state)]


def compare_to_reference(length, gamma_value):
    actual_inputs = tensors(length, gamma_value, seed=1000 + length + int(gamma_value * 10))
    reference_inputs = [x.detach().float().requires_grad_() for x in actual_inputs]
    actual = qgdn_rule(*actual_inputs[:6], initial_state=actual_inputs[6],
                        output_final_state=True, recall_mode="query")
    expected = qgdn_reference(*reference_inputs[:6], initial_state=reference_inputs[6],
                               recall_mode="query")
    weights = [torch.randn_like(x, dtype=torch.float32) for x in expected]
    actual_grads = torch.autograd.grad(
        sum((x.float() * w).sum() for x, w in zip(actual, weights)), actual_inputs)
    expected_grads = torch.autograd.grad(
        sum((x * w).sum() for x, w in zip(expected, weights)), reference_inputs)
    output_errors = [relative_rmse(a, b) for a, b in zip(actual, expected)]
    gradient_errors = [relative_rmse(a, b) for a, b in zip(actual_grads, expected_grads)]
    return dict(length=length, gamma=gamma_value, output_relative_rmse=output_errors,
                gradient_relative_rmse=gradient_errors,
                finite=all(math.isfinite(x) for x in output_errors + gradient_errors))


def compare_zero_to_gdn(length):
    source = tensors(length, 0.0, seed=3000 + length)
    q_inputs = [x.detach().requires_grad_() for x in source]
    g_inputs = [x.detach().requires_grad_() for x in source[:5]] + [source[6].detach().requires_grad_()]
    q_output = qgdn_rule(*q_inputs[:6], initial_state=q_inputs[6],
                          output_final_state=True, recall_mode="query")
    g_output = get_chunk_gated_delta_rule()(
        q=g_inputs[0], k=g_inputs[1], v=g_inputs[2], g=g_inputs[3], beta=g_inputs[4],
        initial_state=g_inputs[5], output_final_state=True, use_qk_l2norm_in_kernel=True)
    weights = [torch.randn_like(x, dtype=torch.float32) for x in g_output]
    q_grads = torch.autograd.grad(
        sum((x.float() * w).sum() for x, w in zip(q_output, weights)),
        (q_inputs[0], q_inputs[1], q_inputs[2], q_inputs[3], q_inputs[4], q_inputs[6]))
    g_grads = torch.autograd.grad(
        sum((x.float() * w).sum() for x, w in zip(g_output, weights)), g_inputs)
    output_errors = [relative_rmse(a, b) for a, b in zip(q_output, g_output)]
    gradient_errors = [relative_rmse(a, b) for a, b in zip(q_grads, g_grads)]
    return dict(length=length, output_relative_rmse=output_errors,
                gradient_relative_rmse=gradient_errors,
                finite=all(math.isfinite(x) for x in output_errors + gradient_errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA audit requires an allocated GPU")
    numerics = configure_numerics()
    reference = [compare_to_reference(length, gamma) for gamma in (0.0, 0.1, 1.0)
                 for length in (17, 65, 257, 4096)]
    zero_limit = [compare_zero_to_gdn(length) for length in (17, 65, 257, 4096)]
    max_reference_output = max(max(x["output_relative_rmse"]) for x in reference)
    max_reference_gradient = max(max(x["gradient_relative_rmse"]) for x in reference)
    max_zero_output = max(max(x["output_relative_rmse"]) for x in zero_limit)
    max_zero_gradient = max(max(x["gradient_relative_rmse"]) for x in zero_limit)
    thresholds = dict(reference_output_relative_rmse=0.025, reference_gradient_relative_rmse=0.07,
                      zero_gdn_output_relative_rmse=0.025, zero_gdn_gradient_relative_rmse=0.07)
    passed = (all(x["finite"] for x in reference + zero_limit)
              and max_reference_output < thresholds["reference_output_relative_rmse"]
              and max_reference_gradient < thresholds["reference_gradient_relative_rmse"]
              and max_zero_output < thresholds["zero_gdn_output_relative_rmse"]
              and max_zero_gradient < thresholds["zero_gdn_gradient_relative_rmse"])
    files = ["model/lit_gpt/mixers/qgdn.py", "model/lit_gpt/mixers/qgdn_rule.py",
             "model/lit_gpt/mixers/qgdn_reference.py",
             "model/lit_gpt/mixers/gdn.py",
             "third_party/flash-linear-attention/fla/ops/generalized_delta_rule/dplr/chunk.py"]
    report = dict(status="passed" if passed else "failed",
                  code_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  numerics=numerics, thresholds=thresholds,
                  maxima=dict(reference_output_relative_rmse=max_reference_output,
                               reference_gradient_relative_rmse=max_reference_gradient,
                               zero_gdn_output_relative_rmse=max_zero_output,
                               zero_gdn_gradient_relative_rmse=max_zero_gradient),
                  reference_cases=reference, zero_recall_vs_gdn_cases=zero_limit,
                  source_sha256={name: sha256(ROOT / name) for name in files})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
