# QGDN training-speed optimization

## Scope

Optimize the existing scalar QGDN training path without changing its recurrence, gates, initialization, model configuration, or output semantics. QR-GDN, DT-GDN, JQC-GDN, and other method-design work is frozen.

The historical, reviewed QGDN implementation is commit `f62322a5fd0cdbc1ed45a9753bdfa22a663143d4`. Later commits may contain monitoring and unrelated historical work, but speed changes must remain exactly equivalent to this QGDN mechanism.

## Fixed recurrence

For normalized query and key, let `c_t = gamma_t (1-alpha_t)`:

`S_rec = alpha_t S_{t-1} + c_t q_t (q_t^T S_{t-1})`

`S_t = S_rec + beta_t k_t (v_t - k_t^T S_rec)^T`

The output reads the post-Delta state: `o_t = d_k^-1/2 S_t^T q_t`.

The exact physical-token affine form is:

`S_t = alpha_t S_{t-1} + L_{t,0}(R_{t,0}^T S_{t-1}) + L_{t,1}(R_{t,1}^T S_{t-1}) + p_t v_t^T`,

where

- `L_{t,0}=q_t`, `R_{t,0}=c_t q_t`;
- `L_{t,1}=k_t`, `R_{t,1}=-alpha_t beta_t k_t-beta_t c_t(k_t^Tq_t)q_t`;
- `p_t=beta_t k_t`.

This removes the conceptual need for the current 2T virtual sequence while preserving the cross term from Recall followed by Delta.

## Optimization order

1. Profile the current QGDN DPLR path and isolate input construction, forward, and backward costs.
2. Benchmark existing DPLR chunk sizes 16, 32, and 64. Adopt a larger chunk only if output, final state, every input gradient, BF16 stability, and full-model training remain within the established numerical policy.
3. Remove avoidable interleave allocations and unused virtual outputs when measurable.
4. Implement a physical-T rank-two fused chunk/scan path only if the lower-risk changes do not reach the target.
5. Validate CPU FP64, CUDA FP32/BF16, continuation/final state, backward, and 8-GPU DDP. Benchmark the same 340M, sequence-4096, micro-batch-1 workload against GDN and the historical QGDN path.

## Acceptance

- No change to QGDN math, gate parameterization, initialization, state layout, or output timing.
- All outputs, final states, and gradients pass the existing tolerances; no NaN/Inf in stress cases.
- Target throughput is at least 90% of GDN on the same H800 setup, or otherwise the best verified improvement with remaining bottlenecks documented.
- Peak memory must not increase materially.
- No full FineWeb pretraining is submitted for speed-only work.
