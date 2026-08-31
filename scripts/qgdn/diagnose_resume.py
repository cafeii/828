"""Single-GPU tiny-run diagnostic; never substitutes for validate.py.

Run the unchanged trainer in fresh processes, tracing each optimizer step.
Compare independent full runs with interrupted/resumed runs, both with the
normal runtime and with PyTorch deterministic algorithms requested. Custom
kernels may remain nondeterministic; the measured comparisons decide that.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ATOL, RTOL = 3e-7, 3e-6


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def child(trace, deterministic, train_args):
    import torch
    import train

    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This diagnostic requires one allocated CUDA GPU and one process")
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    trace.mkdir(parents=True, exist_ok=False)
    sys.argv = ["train.py", *train_args]
    args = train.parse_args()
    offset = 0
    if args.resume:
        offset = torch.load(args.resume, map_location="cpu", weights_only=False)["step"]
    if args.model != "qgdn_recall_tiny" or args.max_steps != 3 or args.task != "mqar":
        raise ValueError("Only the fixed three-step tiny MQAR probe is permitted")

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu(v) for v in value)
        return value

    tracked, batches = {}, []
    make_model = train.GPT
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = torch.optim.AdamW.step
    original_batch = train.mqar_batch
    original_init = torch.optim.AdamW.__init__

    def model_factory(*a, **kw):
        model = make_model(*a, **kw)
        tracked["model"] = model
        return model

    def optimizer_init(self, *a, **kw):
        original_init(self, *a, **kw)
        tracked["optimizer"] = self

    def record_before_clip(*a, **kw):
        step = offset + tracked.get("completed", 0) + 1
        model = tracked["model"]
        torch.save(cpu(dict(model=model.state_dict(), optimizer=tracked["optimizer"].state_dict(),
                            gradients={n: p.grad for n, p in model.named_parameters()},
                            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state())),
                   trace / f"step-{step}-before.pt")
        return original_clip(*a, **kw)

    def record_after_step(self, *a, **kw):
        result = original_step(self, *a, **kw)
        tracked["completed"] = tracked.get("completed", 0) + 1
        step = offset + tracked["completed"]
        torch.save(cpu(dict(model=tracked["model"].state_dict(), optimizer=self.state_dict())),
                   trace / f"step-{step}-after.pt")
        return result

    def record_batch(rows, sequence_length, seed, vocab_size, **kw):
        rows = list(rows)
        x, y = original_batch(rows, sequence_length, seed, vocab_size, **kw)
        batches.append(dict(rows=rows, seed=seed,
                            x_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                            y_sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest()))
        return x, y

    train.GPT = model_factory
    train.mqar_batch = record_batch
    torch.optim.AdamW.__init__ = optimizer_init
    torch.optim.AdamW.step = record_after_step
    torch.nn.utils.clip_grad_norm_ = record_before_clip
    try:
        train.main()
    finally:
        write_json(trace / "trace.json", dict(deterministic_requested=deterministic,
                   torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   device=torch.cuda.get_device_name(0), completed_steps=tracked.get("completed", 0),
                   starting_step=offset, batches=batches,
                   note="Tracing copies tensors to CPU and changes timings, not training equations."))


def tensor_map(value, prefix=""):
    import torch
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    result = {}
    if isinstance(value, dict):
        for k, v in value.items():
            result.update(tensor_map(v, f"{prefix}/{k}"))
    elif isinstance(value, (list, tuple)):
        for k, v in enumerate(value):
            result.update(tensor_map(v, f"{prefix}/{k}"))
    return result


def compare(a, b):
    import torch
    aa, bb = tensor_map(a), tensor_map(b)
    rows = []
    for name in sorted(aa.keys() & bb.keys()):
        x, y = aa[name], bb[name]
        if x.shape != y.shape or x.dtype != y.dtype:
            rows.append(dict(name=name, incompatible=True))
            continue
        different = (x != y).sum().item()
        if not different:
            continue
        delta = (x.double() - y.double()).abs()
        finite = bool(torch.isfinite(delta).all())
        bad = int((~torch.isclose(x.double(), y.double(), atol=ATOL, rtol=RTOL)).sum())
        rows.append(dict(name=name, different_elements=int(different), beyond_tolerance=bad,
                         finite=finite, max_absolute_error=float(delta.max()) if finite else None))
    return dict(missing_left=sorted(bb.keys() - aa.keys()), missing_right=sorted(aa.keys() - bb.keys()),
                tensor_count=len(aa), differing_tensors=len(rows),
                beyond_tolerance_tensors=sum(r.get("beyond_tolerance", 0) > 0 or r.get("incompatible", False)
                                             for r in rows), differences=rows)


def orchestrate(output):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Request exactly one GPU for this diagnostic")
    if output.resolve().is_relative_to(ROOT):
        raise ValueError("Outputs must be outside the immutable source checkout")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", is_validation_pass=False, atol=ATOL, rtol=RTOL,
                  code_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  cases={}, comparisons={})
    base = ["--model", "qgdn_recall_tiny", "--task", "mqar", "--sequence-length", "128",
            "--max-steps", "3", "--global-batch-size", "2", "--micro-batch-size", "1",
            "--eval-sequences", "4", "--log-every", "1", "--eval-every", "3", "--save-every", "3"]

    def run(mode, tag, run_dir, extra=()):
        name = f"{mode}-{tag}"
        trace = output / name
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child-trace", str(trace)]
        if mode == "deterministic":
            cmd.append("--deterministic")
        cmd += ["--", *base, "--output", str(run_dir), *map(str, extra)]
        env = os.environ.copy()
        if mode == "deterministic":
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        log_path = output / f"{name}.log"
        print("RUN", name, flush=True)
        with log_path.open("w") as log:
            try:
                code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=600).returncode
            except subprocess.TimeoutExpired:
                code = 124
        report["cases"][name] = dict(returncode=code, trace=str(trace), log=str(log_path), command=cmd)
        write_json(output / "diagnosis.json", report)
        return code == 0

    def load(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    for mode in ("default", "deterministic"):
        a, b, split = (output / f"{mode}-{x}-run" for x in ("full-a", "full-b", "split"))
        ok_a = run(mode, "full-a", a)
        ok_b = run(mode, "full-b", b)
        ok_stop = run(mode, "stop", split, ["--stop-after-step", "1"])
        ok_resume = False
        saved_prefix = output / f"{mode}-saved-step-1.pt"
        if ok_stop:
            shutil.copyfile(split / "checkpoint.pt", saved_prefix)
            ok_resume = run(mode, "resume", split, ["--resume", split / "checkpoint.pt"])
        for label, left, right, steps in (
            ("full_repeat", "full-a", "full-b", [1, 2, 3]),
            ("independent_first_step", "full-a", "stop", [1]),
            ("continuous_vs_resume", "full-a", "resume", [2, 3]),
        ):
            for step in steps:
                for point in ("before", "after"):
                    l = output / f"{mode}-{left}" / f"step-{step}-{point}.pt"
                    r = output / f"{mode}-{right}" / f"step-{step}-{point}.pt"
                    if l.exists() and r.exists():
                        report["comparisons"][f"{mode}/{label}/{step}/{point}"] = compare(load(l), load(r))
        restored = output / f"{mode}-resume/step-2-before.pt"
        if saved_prefix.exists() and restored.exists():
            old, new = load(saved_prefix), load(restored)
            report["comparisons"][f"{mode}/saved_vs_restored"] = compare(
                {k: old[k] for k in ("model", "optimizer")}, {k: new[k] for k in ("model", "optimizer")})
        if ok_a and ok_b and ok_resume:
            for label, right in (("full_repeat_final", b), ("resume_final", split)):
                report["comparisons"][f"{mode}/{label}"] = compare(
                    load(a / "checkpoint.pt")["model"], load(right / "checkpoint.pt")["model"])
        write_json(output / "diagnosis.json", report)
    report["status"] = "diagnostic_completed"
    report["interpretation"] = (
        "Inspect default full-repeat differences before attributing divergence to resume. "
        "Deterministic-request errors identify unsupported operations; success alone does not prove determinism. "
        "Saved-vs-restored compares tensors before the next optimizer update; LR scalars are scheduled separately. "
        "This diagnostic cannot authorize pilots or main training."
    )
    write_json(output / "diagnosis.json", report)
    print(json.dumps({k: {x: v[x] for x in ("differing_tensors", "beyond_tolerance_tensors")}
                      for k, v in report["comparisons"].items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child-trace", type=Path)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.child_trace:
        child(args.child_trace, args.deterministic, args.trainer_args[1:] if args.trainer_args[:1] == ["--"] else args.trainer_args)
    elif args.output:
        orchestrate(args.output)
    else:
        parser.error("Provide --output, or the internal --child-trace option")


if __name__ == "__main__":
    main()
