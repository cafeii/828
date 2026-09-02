# Unified rank-two kernel contract

Both mechanisms use one physical token per recurrence step:

`S_t = alpha_t S_{t-1} + L_{t,0}(R_{t,0}^T S_{t-1}) + L_{t,1}(R_{t,1}^T S_{t-1}) + p_t v_t^T`.

`L` and `R` have shape `[B,T,H,2,K]`; the rank axis is two while the time axis remains exactly `T`. The output is `K^-1/2 q_t^T S_t`. This is the sole contract for the future chunk/scan kernel and prevents a hidden `2T` implementation.

For DT-GDN, let `c=k^Tq`, `d=1-beta*gamma*c^2`, `C00=beta/d`, `C01=-beta*gamma*c/d`, and `C11=gamma/d`. Then:

- `L0=C00 k+C01 q`, `R0=-alpha k`, `p=L0`;
- `L1=C01 k+C11 q`, `R1=(1-alpha)q`.

For JQC-GDN:

- `L0=k`, `R0=-alpha*beta*k`;
- `L1=q`, `R1=gamma*((1-alpha)q+alpha*beta*(q^Tk)k)`;
- `p=beta*(k-gamma*(q^Tk)q)`.

Affine elements compose associatively as `(A2,C2)∘(A1,C1)=(A2A1,A2C1+C2)`. The FP64 tests verify both factorizations against their independent method equations, including all input gradients, and verify an inclusive affine scan against token recurrence. The production implementation must preserve this contract while avoiding dense `K×K` materialization and Python token loops.

## Compact physical-chunk form

Write one token transition as `A=alpha*I+L*R^T`, with `L,R` having two columns. A prefix product is represented exactly as `P=a*I+U*Z^T`. Left-composing one token gives:

`A*P=(alpha*a)*I + [alpha*U,L] [Z,a*R+Z*(U^T*R)]^T`.

Thus each physical token appends two columns and a chunk of length `C` has compact rank `2C`; it does not create a virtual `2T` sequence or a dense `K×K` transition. The affine offset is updated under the same token transition and yields the exact chunk map `S_out=a*S_in+U*(Z^T*S_in)+E`.

The CPU oracle validates this representation for chunk sizes 1, 2, 4 and 7 against the independent token recurrence for DT-GDN and JQC-GDN. It also validates gradients with respect to q, k, v, alpha, beta, gamma and the initial state. The remaining implementation work is to build the same `2C` compact factors and backward pass inside the GPU chunk kernel.

## Vectorized block-WY construction

The block-WY preprocessing now builds every chunk without a token loop. Factor rows are indexed by `(token, direction)` and the strict causal mask compares token indices. Consequently the two directions from one token never interact as if one happened first.

For factor row `i=(t,r)` and factor column `j=(s,u)`, the lower-triangular interaction is zero unless `s<t`; otherwise it is the decay from the end of token `s` to the start of token `t`, multiplied by `R_i^T L_j`. Solving the resulting unit-lower-triangular system of order `2C` yields all read coefficients at once. The same solve includes earlier value writes and directly produces `(a,U,Z,E)` for `S_out=a*S_in+U*(Z^T*S_in)+E`.

This preprocessing is differentiable, uses batched tensor operations, and never materializes a `K×K` transition. FP64 tests cover DT/JQC state parity across one and multiple chunks and gradient parity for every recurrence input. GPU state propagation, token outputs and a custom backward remain the next production-kernel stages.
