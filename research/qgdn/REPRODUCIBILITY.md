# Numerical reproducibility for the paired study

The GDN and QGDN experiment CLIs use the same `qgdn-repro-v1` numerical policy:

- PyTorch deterministic algorithms enabled.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA-aware imports.
- cuDNN benchmarking disabled and deterministic mode enabled.
- FP32 matmul precision remains `high`; training remains BF16 mixed precision.
- The existing fused backbone RMSNorm forward and backward kernels both use
  `num_warps=4`, instead of choosing a configuration independently in each process.

This changes neither the model equations nor the parameter count, data, optimizer
hyperparameters, or training budget. The policy is recorded in `run.json` and the
checkpoint identity. Evaluation, pilot admission, and paired summaries verify it.
Both arms must be measured with this policy; pilot timings include its cost.

## Evidence for the change

GPU validation job 31967 failed the original continuous/resumed parameter check.
Diagnostics 32264 and 32270 used fresh processes, the real trainer, the tiny QGDN
configuration, seed 3407, BF16 mixed precision, and three optimizer steps.

All training row IDs and input/label hashes matched. The 180 saved model and
optimizer tensors were restored exactly. Nevertheless, the unpinned RMSNorm
autotuner selected different warp counts across processes. In job 32270, the
forward/backward choices were 2/16, 1/4, 2/1, and 8/4 for the two full runs,
the stopped run, and the resumed run, respectively.

The final comparison covers 45 model tensors. Counts below are the tensors
exceeding the existing `atol=3e-7, rtol=3e-6` threshold:

| Diagnostic condition | Full repeat | Continuous vs resumed |
| --- | ---: | ---: |
| Deterministic PyTorch, autotuned fused RMSNorm | 30 | 29 |
| Deterministic PyTorch, fixed 4-warp fused RMSNorm | 0 | 0 |
| Deterministic PyTorch, native backbone RMSNorm | 0 | 0 |

Both fixed and native controls were bitwise identical in these final comparisons.
The native implementation was a diagnostic control only. Production keeps the
fused implementation, with its launch configuration fixed.

## Validation remains mandatory

`validate.py` retains the original tolerance and now compares a full repeat as
well as interrupted/resumed training, checking optimizer states too. It still
requires numerical/gradient parity, mismatched-resume rejection, multi-GPU DDP,
and full-size model smoke checks before any pilot can start.

The tiny diagnostic does not establish bitwise reproducibility for every custom
kernel, GPU type, model size, or distributed execution. Each frozen code revision
must pass the allocated-GPU validation. Do not relax tolerances to hide a failure.

Detailed diagnostic reports and traces remain in the experiment directories for
`20260831-222600-qgdn-resume-diagnostic-8b4224` and
`20260831-224611-qgdn-norm-diagnostic-8b69d0`.
