# GPU state-forward diagnostic 34254

- Experiment: `20260902-181059-dtjqc-rank2-state-gpu-8a21f7`
- Commit: `a3b5acfd9377a49014a2b91eb719f1c3f81d1523`
- Slurm/accounting: `COMPLETED`, `0:0`; wrapper `run.exitcode=0`.
- Required output: present, but contains three pytest failures.
- Formal review: **FAILED**. The first substantive error is `ModuleNotFoundError: No module named lit_gpt`; the kernel was never imported or executed.
- Root cause: the nested `bash -lc` reset the job-provided snapshot `PYTHONPATH`; its `pytest | tee` pipeline also returned the successful `tee` status and hid pytest failure from Slurm.
- Recovery: use direct `python -m pytest ... --junitxml=<required-output>` without a login shell or pipeline. Reuse the same code commit. Retry is deferred because this heartbeat already used its one allowed `sbatch`.
- Logs and output were preserved and collected locally.
