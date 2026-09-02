# GPU state-forward diagnostic review: job 34295

- **Terminal state:** FAILED (`sacct` exit `1:0`, elapsed `00:00:41`)
- **Run marker:** `run.exitcode=1`
- **Required output:** `pytest.xml` present
- **Snapshot commit:** `a3b5acfd9377a49014a2b91eb719f1c3f81d1523`
- **Import-path fix:** verified; the test imported `lit_gpt` and entered Triton compilation
- **Formal outcome:** failed; no GPU numerical result accepted

The first substantive error was `Triton CompilationError: Input shapes should have M >= 1, N >= 1 and K >= 16`. The small-shape coverage used compact rank `M=8` and `M=4`, while Hopper requires the inner dimension of `tl.dot` to be at least 16.

Commit `e51ccdea2c85ec047a73d6c4d984d5e5ed060499` pads only the internal dot tile dimension to at least 16 and masks loads with the actual `M`; physical sequence length, pointer strides, and the rank-two recurrence are unchanged. Static compilation and all 28 targeted CPU FP64 regressions pass. A GPU retest is required before this kernel can be accepted. No second job was submitted in this heartbeat because job 34295 consumed the one-`sbatch` allowance.
