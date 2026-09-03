"""Evaluate a semantics-preserving QR-GDN schedule with shared q normalization.

The candidate removes four full shifted copies and normalizes q once.  Its QR
scan consumes adjacent length-(T-1) views: update t is read by q[t+1].  This is
an isolated scheduling diagnostic; production code is not changed here.
"""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "qgdn"))

import torch
import torch.nn.functional as F

from fla.modules.l2norm import l2norm
from lit_gpt.config import Config
from lit_gpt.kernels import get_chunk_gated_delta_rule
from lit_gpt.model import GPT
from runtime import configure_numerics


def _read(direction: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhk,bhkv->bhv", direction, state)


def qr_gdn_shared_norm_sliced(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_kv: torch.Tensor,
    beta_kv: torch.Tensor,
    g_qr: torch.Tensor,
    beta_qr: torch.Tensor,
    read_logit: torch.Tensor,
    *,
    initial_state=None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    """Exact QR-GDN recurrence with shared normalization and adjacent views."""
    if q.ndim != 4 or k.shape != q.shape or q.shape[:-1] != v.shape[:-1]:
        raise ValueError("q, k and v must share [B,T,H] dimensions")
    if q.shape[1] == 0:
        raise ValueError("QR-GDN requires a nonempty sequence")
    expected_gate = q.shape[:-1]
    if any(x.shape != expected_gate for x in (g_kv, beta_kv, g_qr, beta_qr, read_logit)):
        raise ValueError("all gates must have shape [B,T,H]")
    if initial_state is None:
        initial_kv = initial_qr = None
    else:
        if len(initial_state) != 2:
            raise ValueError("initial_state must be an (M_KV, M_QR) pair")
        initial_kv, initial_qr = initial_state

    # These are the same FLA normalization kernels used inside native GDN.  qn
    # fans out to both channels, so autograd merges their gradients before one
    # L2-normalization backward pass.
    qn = l2norm(q)
    kn = l2norm(k)
    rule = get_chunk_gated_delta_rule()
    common = dict(
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
        chunk_size=chunk_size,
    )
    output_kv, final_kv, recall = rule(
        q=qn,
        k=kn,
        v=v,
        g=g_kv,
        beta=beta_kv,
        initial_state=initial_kv,
        output_final_state=output_final_state,
        output_pre_read=True,
        **common,
    )

    B, T, H, K = q.shape
    if initial_qr is None:
        first_read = v.new_zeros((B, 1, H, v.shape[-1]))
    else:
        first_read = _read(qn[:, 0].float(), initial_qr.float())[:, None].to(v.dtype)

    shifted_final_qr = initial_qr
    if T > 1:
        # Slot i performs physical update i and is read by physical query i+1.
        # With B=1 (the formal micro batch), these slices are contiguous views,
        # so no shifted q/value/gate tensors are materialized.
        tail_read, shifted_final_qr = rule(
            q=qn[:, 1:],
            k=qn[:, :-1],
            v=recall[:, :-1],
            g=g_qr[:, :-1],
            beta=beta_qr[:, :-1],
            scale=1.0,
            initial_state=initial_qr,
            output_final_state=output_final_state,
            **common,
        )
        qr_read = torch.cat((first_read, tail_read), dim=1)
    else:
        qr_read = first_read

    scale = K**-0.5
    output = output_kv + scale * read_logit.tanh()[..., None] * qr_read
    final_state = None
    if output_final_state:
        if shifted_final_qr is None:
            shifted_final_qr = q.new_zeros((B, H, K, v.shape[-1]), dtype=torch.float32)
        q_last = F.normalize(q[:, -1].float(), dim=-1)
        qr_base = g_qr[:, -1].float().exp()[..., None, None] * shifted_final_qr.float()
        qr_error = recall[:, -1].float() - _read(q_last, qr_base)
        final_qr = qr_base + beta_qr[:, -1].float()[..., None, None] * q_last[..., None] * qr_error[..., None, :]
        final_state = final_kv, final_qr
    return output.to(q.dtype), final_state


def _inputs(dtype=torch.float32, *, requires_grad=False, T=128):
    torch.manual_seed(3407)
    B, H, K, V = 1, 2, 16, 12
    values = [
        torch.randn(B, T, H, K, device="cuda", dtype=dtype),
        torch.randn(B, T, H, K, device="cuda", dtype=dtype),
        torch.randn(B, T, H, V, device="cuda", dtype=dtype),
        -0.3 * torch.rand(B, T, H, device="cuda", dtype=torch.float32),
        0.05 + 0.9 * torch.rand(B, T, H, device="cuda", dtype=torch.float32),
        -0.3 * torch.rand(B, T, H, device="cuda", dtype=torch.float32),
        0.05 + 0.9 * torch.rand(B, T, H, device="cuda", dtype=torch.float32),
        torch.randn(B, T, H, device="cuda", dtype=torch.float32),
        torch.randn(B, H, K, V, device="cuda", dtype=torch.float32),
        torch.randn(B, H, K, V, device="cuda", dtype=torch.float32),
    ]
    return tuple(x.requires_grad_(requires_grad) for x in values)


def validate_candidate() -> dict:
    module = importlib.import_module("lit_gpt.mixers.qr_gdn_parallel")
    current = module.qr_gdn_parallel

    args = _inputs(requires_grad=True)
    expected = current(*args[:8], initial_state=args[8:], output_final_state=True)
    weights = [torch.randn_like(expected[0]), *(torch.randn_like(x) for x in expected[1])]
    expected_loss = (expected[0] * weights[0]).sum() + sum(
        (x * w).sum() for x, w in zip(expected[1], weights[1:])
    )
    expected_grads = torch.autograd.grad(expected_loss, args)

    cloned = tuple(x.detach().clone().requires_grad_() for x in args)
    actual = qr_gdn_shared_norm_sliced(
        *cloned[:8], initial_state=cloned[8:], output_final_state=True
    )
    actual_loss = (actual[0] * weights[0]).sum() + sum(
        (x * w).sum() for x, w in zip(actual[1], weights[1:])
    )
    actual_grads = torch.autograd.grad(actual_loss, cloned)
    torch.testing.assert_close(actual[0], expected[0], rtol=3e-3, atol=3e-3)
    for value, reference in zip(actual[1], expected[1]):
        torch.testing.assert_close(value, reference, rtol=3e-3, atol=3e-3)
    for value, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(value, reference, rtol=2e-2, atol=2e-2)

    q, k, v, gkv, bkv, gqr, bqr, read, kv, qr = _inputs(T=64)
    zeros = torch.zeros_like(read)
    native, _ = get_chunk_gated_delta_rule()(
        q=q,
        k=k,
        v=v,
        g=gkv,
        beta=bkv,
        initial_state=kv,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    degenerate, _ = qr_gdn_shared_norm_sliced(
        q, k, v, gkv, bkv, gqr, bqr, zeros,
        initial_state=(kv, qr), output_final_state=False,
    )
    torch.testing.assert_close(degenerate, native, rtol=0, atol=0)

    bf16 = _inputs(dtype=torch.bfloat16, requires_grad=True, T=64)
    out, state = qr_gdn_shared_norm_sliced(
        *bf16[:8], initial_state=bf16[8:], output_final_state=True
    )
    loss = out.float().square().mean() + sum(x.square().mean() for x in state)
    grads = torch.autograd.grad(loss, bf16)
    assert torch.isfinite(out).all() and all(torch.isfinite(x).all() for x in (*state, *grads))

    return {
        "fp32_output_max_abs": (actual[0] - expected[0]).abs().max().item(),
        "fp32_state_max_abs": max((x - y).abs().max().item() for x, y in zip(actual[1], expected[1])),
        "fp32_gradient_max_abs": max((x - y).abs().max().item() for x, y in zip(actual_grads, expected_grads)),
        "zero_read_native_bitwise": True,
        "bf16_forward_backward_finite": True,
    }


def benchmark_model(
    name: str,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    warmup: int,
    steps: int,
    *,
    qr_impl=None,
) -> dict:
    qr_module = importlib.import_module("lit_gpt.mixers.qr_gdn_parallel")
    original = qr_module.qr_gdn_parallel
    if qr_impl is not None:
        qr_module.qr_gdn_parallel = qr_impl
    try:
        torch.manual_seed(3407)
        config = Config.from_name(name, block_size=tokens.shape[1])
        model = GPT(config)
        model.apply(lambda module: model._init_weights(module, n_layer=config.n_layer))
        model.gradient_checkpointing = True
        model.cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, betas=(0.9, 0.95), fused=True)
        durations = []
        for index in range(warmup + steps):
            optimizer.zero_grad(set_to_none=True)
            if index == warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(tokens)
                loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if index >= warmup:
                durations.append(elapsed)
        result = {
            "model": name,
            "step_seconds": durations,
            "mean_step_seconds": statistics.mean(durations),
            "median_step_seconds": statistics.median(durations),
            "tokens_per_second": tokens.numel() / statistics.mean(durations),
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
            "final_loss": loss.item(),
        }
        del optimizer, model, logits, loss
        torch.cuda.empty_cache()
        return result
    finally:
        qr_module.qr_gdn_parallel = original


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This diagnostic requires exactly one allocated CUDA GPU")
    if args.sequence_length % 64:
        raise ValueError("sequence length must be divisible by 64")
    numerics = configure_numerics(cpu=False)
    validation = validate_candidate()

    torch.manual_seed(117)
    config = Config.from_name("gdn_control_340M", block_size=args.sequence_length)
    tokens = torch.randint(0, config.padded_vocab_size, (1, args.sequence_length), device="cuda")
    targets = torch.roll(tokens, shifts=-1, dims=1)
    models = {
        "gdn_control_340M": benchmark_model(
            "gdn_control_340M", tokens, targets, args.warmup, args.steps
        ),
        "qr_gdn_current_340M": benchmark_model(
            "qr_gdn_340M", tokens, targets, args.warmup, args.steps
        ),
        "qr_gdn_shared_norm_sliced_340M": benchmark_model(
            "qr_gdn_340M", tokens, targets, args.warmup, args.steps,
            qr_impl=qr_gdn_shared_norm_sliced,
        ),
    }
    baseline = models["gdn_control_340M"]["tokens_per_second"]
    current = models["qr_gdn_current_340M"]["tokens_per_second"]
    candidate = models["qr_gdn_shared_norm_sliced_340M"]["tokens_per_second"]
    report = {
        "status": "measured",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "cuda_device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "precision": "bf16-mixed",
        "sequence_length": args.sequence_length,
        "micro_batch_size": 1,
        "activation_checkpointing": True,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "numerics": numerics,
        "validation": validation,
        "models": models,
        "current_qr_to_gdn_ratio": current / baseline,
        "candidate_qr_to_gdn_ratio": candidate / baseline,
        "candidate_to_current_ratio": candidate / current,
        "throughput_gate": 0.8,
        "throughput_gate_passed": candidate / baseline >= 0.8,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
