# DT-GDN / JQC-GDN FineWeb comparison

This stage implements two distinct query-directed state updates and compares them with the completed GDN baseline. State layout is `[B,H,K,V]`, reads are `S^T x`, and both keys and queries are L2-normalized.

- **DT-GDN:** the exact joint proximal solution around `B=alpha*S` with write target `(k,v)` and recall target `(q,S^Tq)`. Its stable 2x2 system has determinant `1-beta*gamma*(k^Tq)^2` and must be implemented as an affine rank-two parallel scan.
- **JQC-GDN:** first performs the native GDN update, then consolidates the previous query read at address `q`. Setting `gamma=0` must reproduce GDN exactly. It too must use an affine rank-two parallel scan.

Development gates are FP64 direct-solve and recurrence oracles, forward/backward parity, endpoint-state parity, near-collinearity checks, CPU/GPU tests, BF16/FP32 checks, 8-GPU DDP, and a same-H800 throughput/peak-memory benchmark. Full training is forbidden until both new methods reach at least 80% of GDN throughput without unreasonable memory growth.

Formal runs use only seeds 3407 and 42. Each new method runs both seeds with the existing FineWeb manifest, 19,073 steps, 9,999,745,024 prediction tokens, sequence length 4096, global batch 128, micro batch 1, 8 GPUs, 32 CPUs, 256G, BF16, and the established optimizer, schedule, numerics and validation set. Existing reviewed GDN results are reused after identity checks. No seed 2026, tuning, pilot selection or extra seed is allowed.

At most four unreviewed full jobs may be active. Each wake may submit at most one Slurm job of any kind. `active_jobs` in `state.json` is authoritative. Every submission uses a committed clean worktree, unique compliant name, `submission.lock`, recorded job ID and bounded time. Queue disappearance alone is never success.

All models report alpha/beta/gamma global mean and population standard deviation from FP64 sum, sum-of-squares and count merged across microbatches, layers and DDP ranks. GDN gamma is inapplicable. Reports also retain loss, PPL, throughput, peak memory, parameters and wall time. No gate standard deviations may be averaged.

Commit every validated milestone and at least every two hours while substantive changes remain. Commit only task files; do not push without separate authorization. Preserve all logs and prior results.
