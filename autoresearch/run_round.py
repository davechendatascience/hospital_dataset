"""Run ONE round of the gist-network architecture-research loop.

A round = train the architecture currently described in
`autoresearch/current_config.json` for a short fixed budget, evaluate the sim
split (valid) + the real dev split (real_dev) every epoch, dump a few real-
domain prediction overlays, then snapshot the result into
`autoresearch/iter_<n>/report.json` and append it to
`autoresearch/leaderboard.jsonl`.

The loop is human/LLM-in-the-loop by design: this script handles
train+eval+inference+logging; the *architecture-modification* step is done by
Claude between rounds (read the report + overlays, edit current_config.json
and/or the GistFirstNetwork source, then run the next round). See PROTOCOL.md.

    /home/edge-host/Documents/.venv/bin/python autoresearch/run_round.py --round 1

NEVER point --eval-splits / --overlay-split at real_holdout: that set is scored
exactly once, at the very end, by report_holdout.py. Selecting architectures on
it would overfit the design to those images (the same leakage we avoid for
checkpoint selection, one level up).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AR = ROOT / "autoresearch"
VENV = "/home/edge-host/Documents/.venv/bin/python"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--data", default="ward_v2")
    ap.add_argument("--budget-epochs", type=int, default=3,
                    help="Epochs per round. Short on purpose; we compare "
                         "architectures at equal compute, not at convergence.")
    ap.add_argument("--eval-splits", default="valid,real_dev",
                    help="Sim anchor + real dev set. MUST NOT include real_holdout.")
    ap.add_argument("--overlay-split", default="real_dev")
    ap.add_argument("--overlays", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--short-edge", type=int, default=504)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name-base", default="gfn_ward_v2")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Extra flags forwarded verbatim to train_seg_detr.py")
    args = ap.parse_args()

    assert "real_holdout" not in args.eval_splits, \
        "real_holdout must never be an eval split during the search loop"

    cfg = json.loads((AR / "current_config.json").read_text())
    run_name = f"{args.name_base}_r{args.round}"
    run_dir = ROOT / "runs" / "seg_detr" / run_name

    cmd = [
        VENV, str(ROOT / "train_seg_detr.py"),
        "--data", args.data, "--train-data", args.data,
        "--eval-splits", args.eval_splits,
        "--gap-pair", "valid,real_dev",
        "--epochs", str(args.budget_epochs),
        "--batch", str(args.batch),
        "--short-edge", str(args.short_edge),
        "--device", args.device,
        "--dump-overlays", str(args.overlays),
        "--overlay-split", args.overlay_split,
        "--project", "runs/seg_detr", "--name", run_name,
    ] + list(cfg["flags"]) + list(args.extra or [])

    # Snapshot the model source so this variant is reproducible even after the
    # next round edits GistFirstNetwork in place.
    snap_dir = AR / "variants"
    snap_dir.mkdir(exist_ok=True)
    shutil.copy(ROOT / "train_seg_detr.py",
                snap_dir / f"r{args.round}_{cfg['version']}_train_seg_detr.py")

    print(f"[round {args.round}] arch: {cfg['arch_desc']}")
    print("[round] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # Read the last metrics row.
    mcsv = run_dir / "metrics.csv"
    last = None
    with open(mcsv) as f:
        for last in csv.DictReader(f):
            pass

    report = {
        "round": args.round,
        "run_name": run_name,
        "arch_desc": cfg["arch_desc"],
        "version": cfg["version"],
        "flags": cfg["flags"],
        "budget_epochs": args.budget_epochs,
        "eval_splits": args.eval_splits,
        "metrics": last,
        "metrics_csv": str(mcsv),
        "overlays_dir": str(run_dir / "overlays" / f"epoch{args.budget_epochs}"),
        "source_snapshot": str(snap_dir / f"r{args.round}_{cfg['version']}_train_seg_detr.py"),
    }
    idir = AR / f"iter_{args.round:02d}"
    idir.mkdir(exist_ok=True)
    (idir / "report.json").write_text(json.dumps(report, indent=2))
    with open(AR / "leaderboard.jsonl", "a") as f:
        f.write(json.dumps(report) + "\n")

    g = (last or {}).get("sim_real_gap_AP", "?")
    print(f"\n[round {args.round}] DONE — {cfg['arch_desc']}")
    print(f"  valid_AP={  (last or {}).get('valid_AP')}"
          f"  real_dev_AP={(last or {}).get('real_dev_AP')}"
          f"  gap={g}"
          f"  auxF1(real_dev)={(last or {}).get('real_dev_auxF1')}")
    print(f"  report   -> {idir / 'report.json'}")
    print(f"  overlays -> {report['overlays_dir']}")
    print(f"  >>> hand back to Claude: read the report + overlays, then define "
          f"the next variant in {AR / 'current_config.json'}")


if __name__ == "__main__":
    main()
