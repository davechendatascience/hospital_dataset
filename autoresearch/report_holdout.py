"""Score ONE checkpoint on real_holdout — the untouchable final estimate.

Run this exactly once, at the very end of the search, on the architecture the
loop selected (by its real_dev gap). Until then real_holdout is never seen, so
this number is an unbiased estimate of the true sim->real gap rather than
something the architecture search overfit to.

    /home/edge-host/Documents/.venv/bin/python autoresearch/report_holdout.py \
        --weights runs/seg_detr/gfn_ward_v2_rN/weights/epochK \
        --config  autoresearch/iter_NN/report.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = "/home/edge-host/Documents/.venv/bin/python"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True,
                    help="iter_NN/report.json of the SELECTED variant")
    ap.add_argument("--data", default="ward_v2")
    ap.add_argument("--budget-epochs", type=int, default=10,
                    help="Train the selected arch this long before the final "
                         "holdout read (longer than a search round).")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--short-edge", type=int, default=504)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    rep = json.loads(args.config.read_text())
    cmd = [
        VENV, str(ROOT / "train_seg_detr.py"),
        "--data", args.data, "--train-data", args.data,
        # valid = sim anchor, real_holdout = the one-time honest estimate.
        "--eval-splits", "valid,real_holdout",
        "--gap-pair", "valid,real_holdout",
        "--epochs", str(args.budget_epochs),
        "--batch", str(args.batch), "--short-edge", str(args.short_edge),
        "--device", args.device,
        "--project", "runs/seg_detr", "--name", f"{rep['run_name']}_HOLDOUT",
    ] + list(rep["flags"])
    print(f"[holdout] final estimate for: {rep['arch_desc']}")
    print("[holdout] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print("[holdout] done — this is the number to report.")


if __name__ == "__main__":
    main()
