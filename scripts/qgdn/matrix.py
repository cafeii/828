"""Generate a reviewable paired experiment plan. Does not run or submit jobs."""
import argparse
import hashlib
import json
import shlex
from pathlib import Path

VARIANTS = {
    "gdn": "gdn_control_340M",
    "qgdn": "qgdn_340M",
    "key": "qgdn_key_340M",
    "isotropic": "qgdn_isotropic_340M",
    "fixed": "qgdn_fixed_340M",
    "head": "qgdn_head_340M",
    "zero": "qgdn_zero_340M",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["smoke", "pilot", "main"], default="pilot")
    p.add_argument("--task", choices=["lm", "mqar"], default="lm")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=["gdn", "qgdn"])
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--devices", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--data-manifest", type=Path)
    p.add_argument("--out-root", type=Path, required=True, help="Fresh outputs, inside the experiment's outputs directory")
    p.add_argument("--plan-dir", type=Path, required=True)
    args = p.parse_args()
    if args.devices <= 0 or args.micro_batch_size <= 0:
        p.error("Invalid device/micro batch count")
    if args.task == "lm" and args.data_manifest is None:
        p.error("Supply the manifest from prepare_data.py; no implicit training/validation paths")
    if args.task == "mqar" and args.data_manifest is not None:
        p.error("MQAR does not use a text corpus")
    seeds = args.seeds or ([3407, 42, 2026] if args.stage == "main" else [3407])
    steps, length, batch, evaluations = {"smoke": (3, 128, args.devices * args.micro_batch_size, 8),
                                        "pilot": (512, 4096, 128, 2560),
                                        "main": (19073, 4096, 128, 2560)}[args.stage]
    if batch % (args.devices * args.micro_batch_size):
        p.error("Global batch128 must be divisible by devices * micro batch")
    if args.stage == "smoke" and set(args.variants) - {"gdn", "qgdn"}:
        p.error("Tiny smoke supports gdn/qgdn; run the full ablations at pilot/main scale")
    if args.stage == "smoke" and args.task != "mqar":
        p.error("Tiny smoke has vocabulary256; use --task mqar")
    jobs = []
    for seed in dict.fromkeys(seeds):
        for variant in dict.fromkeys(args.variants):
            model = f"{variant}_recall_tiny" if args.stage == "smoke" else VARIANTS[variant]
            name = f"{args.stage}-{variant}-seed{seed}"
            command = ["torchrun", "--standalone", "--nnodes=1", f"--nproc-per-node={args.devices}",
                       "scripts/qgdn/train.py", "--model", model, "--task", args.task,
                       "--output", str(args.out_root / name), "--seed", str(seed),
                       "--max-steps", str(steps), "--sequence-length", str(length),
                       "--global-batch-size", str(batch), "--micro-batch-size", str(args.micro_batch_size),
                       "--eval-sequences", str(evaluations)]
            if args.data_manifest:
                command += ["--data-manifest", str(args.data_manifest.resolve())]
            jobs.append(dict(name=name, variant=variant, seed=seed, model=model,
                             tokens=steps * batch * length, command=command))
    args.plan_dir.mkdir(parents=True, exist_ok=False)
    plan = dict(stage=args.stage, task=args.task, devices=args.devices, seeds=seeds,
                data_manifest_sha256=hashlib.sha256(args.data_manifest.read_bytes()).hexdigest() if args.data_manifest else None,
                steps=steps, sequence_length=length, global_batch_size=batch,
                note="Review only. Submit each command through the personal Slurm skill after resource checks.", jobs=jobs)
    (args.plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    (args.plan_dir / "commands.txt").write_text("\n".join(shlex.join(job["command"]) for job in jobs) + "\n")
    print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
