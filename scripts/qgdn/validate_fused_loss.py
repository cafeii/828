"""Check the optimized CUDA training loss against the previous PyTorch path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime import configure_device_from_cli, configure_numerics

configure_device_from_cli()

import torch
import torch.nn.functional as F

from lit_gpt import FusedCrossEntropyLoss


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-7)
    return float((error / scale).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=50304)
    args = parser.parse_args()
    numerics = configure_numerics(cpu=False)

    generator = torch.Generator(device="cuda").manual_seed(20260903)
    base = torch.randn(
        2,
        257,
        args.vocab_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    targets = torch.randint(
        0,
        args.vocab_size,
        (2, 257),
        device="cuda",
        generator=generator,
    )
    targets[0, :7] = -100

    reference_logits = base.detach().clone().requires_grad_()
    reference_loss = F.cross_entropy(
        reference_logits.float().flatten(0, 1),
        targets.flatten(),
        ignore_index=-100,
    )
    reference_loss.backward()

    fused_logits = base.detach().clone().requires_grad_()
    fused_loss = FusedCrossEntropyLoss(inplace_backward=True)(fused_logits, targets)
    fused_loss.backward()

    result = {
        "status": "completed",
        "shape": list(base.shape),
        "dtype": str(base.dtype),
        "reference_loss": float(reference_loss.item()),
        "fused_loss": float(fused_loss.item()),
        "loss_absolute_error": abs(float(fused_loss.item() - reference_loss.item())),
        "gradient_relative_rmse": relative_rmse(
            fused_logits.grad, reference_logits.grad
        ),
        "gradient_max_absolute_error": float(
            (fused_logits.grad.float() - reference_logits.grad.float()).abs().max().item()
        ),
        "finite": bool(
            torch.isfinite(fused_loss)
            and torch.isfinite(fused_logits.grad).all()
            and torch.isfinite(reference_logits.grad).all()
        ),
        "numerics": numerics,
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
