"""Fail closed before a queued training command if prerequisite checks failed."""
import argparse
import json
import math
from pathlib import Path
import subprocess

from summarize import pairing_key


def check_gate(validation, pilot_dirs=(), *, revision, world_size, max_hours=168):
    if validation.get("status") != "passed" or not all(validation.get(key) for key in
            ("gpu_parity_verified", "ddp_verified", "full_model_verified")):
        raise ValueError("GPU numerical, DDP and full-model validation must all pass")
    if validation.get("full_model_world_size") != world_size or validation.get("ddp_world_size") != world_size:
        raise ValueError("Validation topology differs from planned training")
    if validation.get("code_revision") != revision:
        raise ValueError("Validation code differs from planned training")
    pairs = []
    for directory in pilot_dirs:
        directory = Path(directory)
        run = json.loads((directory / "run.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        if run["code_revision"] != revision or run["world_size"] != world_size:
            raise ValueError("Pilot code/topology differs from planned training")
        if summary["status"] != "completed" or summary["step"] != 512 or run["args"]["max_steps"] != 512:
            raise ValueError("Pilot has not completed its fixed 512-step budget")
        if summary["trained_tokens"] != 512 * 128 * 4096 or run["args"]["task"] != "lm":
            raise ValueError("Pilot task/token budget differs from the registered plan")
        if summary["identity"] != run["identity"]:
            raise ValueError("Pilot summary and run identity differ")
        final_loss = summary["final_validation"]["loss"]
        if not math.isfinite(final_loss) or final_loss >= summary["initial_validation"]["loss"]:
            raise ValueError("Pilot did not reduce its initial validation loss")
        # This is only a safety limit, never a requirement that QGDN beat GDN.
        projected_hours = summary["wall_seconds"] / 512 * 19073 / 3600
        if not math.isfinite(projected_hours) or projected_hours > max_hours * 0.9:
            raise ValueError(f"Pilot projects {projected_hours:.1f}h, too close to {max_hours}h wall limit")
        pairs.append(run)
    if pairs:
        if len(pairs) != 2 or {r["args"]["model"] for r in pairs} != {"gdn_control_340M", "qgdn_340M"}:
            raise ValueError("Need both GDN and QGDN pilots")
        if pairing_key(pairs[0]) != pairing_key(pairs[1]):
            raise ValueError("Pilots are not a matched comparison")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--pilot", type=Path, nargs="*", default=[])
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--max-hours", type=float, default=168)
    p.add_argument("--command", nargs=argparse.REMAINDER, required=True)
    args = p.parse_args()
    if not args.command:
        p.error("Supply a training command")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    check_gate(json.loads(args.validation.read_text()), args.pilot,
               revision=revision, world_size=args.world_size, max_hours=args.max_hours)
    print("Prerequisite validation passed; starting planned command", flush=True)
    subprocess.run(args.command, check=True)


if __name__ == "__main__":
    main()
