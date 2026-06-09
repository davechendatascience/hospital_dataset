"""End-to-end sim->real domain-adaptation pipeline (orchestrator).

Chains the stages, each a re-runnable script, with the methodology guards we
settled on: held-out real, multi-feature-space gap eval, downstream real AP as
the selection metric.

  0 split    : real photos -> real_train / real_holdout   (run separately)
  1 lora     : LoRA-on-real adapts SD's prior to our ward  (train_lora_real.py)
  2 translate: SD + GT-depth ControlNet [+LoRA] [+IP-Adapter] over sim,
               carrying labels                              (style_transfer_controlnet.py)
  3 measure  : DINOv2 + CLIP MMD/probe, styled vs real_holdout (measure_domain_gap.py)
  4 train    : YOLO on styled-sim, per-epoch train + real AP (train_yolo_da.py)

  .venv/bin/python da_pipeline.py --adapt lora+ip --stages lora,translate,measure,train

--adapt picks the distribution-adaptation knobs: none | lora | ip | lora+ip.
--stages selects which stages to (re)run. Each stage shells out so you can
inspect/re-run any one independently.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent


def run(cmd, title):
    print(f"\n{'='*70}\n[da] {title}\n[da] $ {' '.join(str(c) for c in cmd)}\n{'='*70}",
          flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-data", type=Path, default=Path("ward_v3"))
    ap.add_argument("--real-train", type=Path, default=Path("ward_v1/real_train"))
    ap.add_argument("--real-holdout", type=Path, default=Path("ward_v1/real_holdout"))
    ap.add_argument("--out-styled", type=Path, default=Path("ward_v3_styled"))
    ap.add_argument("--sim-splits", default="train,val")
    ap.add_argument("--adapt", choices=["none", "lora", "ip", "lora+ip"], default="lora+ip",
                    help="distribution-adaptation knobs for the SD backbone")
    ap.add_argument("--stages", default="lora,translate,measure,train",
                    help="comma list: lora,translate,measure,train")
    ap.add_argument("--lora-out", type=Path, default=Path("runs/lora/ward_real"))
    ap.add_argument("--lora-steps", type=int, default=2000)
    ap.add_argument("--gt-depth", action="store_true", default=True)
    ap.add_argument("--seg-scale", type=float, default=0.0)
    ap.add_argument("--ip-scale", type=float, default=0.6)
    ap.add_argument("--task", choices=["detect", "segment"], default="detect")
    ap.add_argument("--yolo-model", default="yolo11s.pt")
    ap.add_argument("--yolo-epochs", type=int, default=50)
    ap.add_argument("--gap-n", type=int, default=300)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    use_lora = "lora" in args.adapt
    use_ip = "ip" in args.adapt
    lora_dir = args.lora_out / "last"
    styled_train = (args.out_styled / "train").resolve()

    # --- 1 LoRA-on-real ---
    if "lora" in stages and use_lora:
        run([PY, ROOT / "train_lora_real.py", "--real-dir", args.real_train,
             "--out", args.lora_out, "--steps", args.lora_steps, "--device", args.device],
            f"stage 1: LoRA-on-real ({args.lora_steps} steps)")

    # --- 2 translate sim -> real ---
    if "translate" in stages:
        cmd = [PY, ROOT / "style_transfer_controlnet.py", "--data", args.sim_data,
               "--split", args.sim_splits, "--seg-scale", args.seg_scale,
               "--apply", "--out", args.out_styled, "--device", args.device]
        if args.gt_depth:
            cmd.append("--gt-depth")
        if use_lora:
            cmd += ["--lora", lora_dir]
        if use_ip:
            cmd += ["--ip-adapter", "--ip-ref-dir", args.real_train, "--ip-scale", args.ip_scale]
        run(cmd, f"stage 2: translate sim->real (adapt={args.adapt})")

    # --- 3 measure the gap (DINOv2 + CLIP, held-out real) ---
    if "measure" in stages:
        for feat in ("dinov2", "clip"):
            run([PY, ROOT / "measure_domain_gap.py", "--data", args.sim_data,
                 "--set", "sim=train", "--set", f"styled={styled_train}",
                 "--set", f"real={args.real_holdout.resolve()}",
                 "--compare-to", "real", "--gap-base", "sim",
                 "--feature", feat, "--max-per-set", args.gap_n, "--device", args.device],
                f"stage 3: gap measure ({feat}) vs held-out real")

    # --- 4 train YOLO on styled, eval real_holdout ---
    if "train" in stages:
        run([PY, ROOT / "train_yolo_da.py", "--train-dir", args.out_styled / "train",
             "--val-dir", args.real_holdout, "--task", args.task,
             "--model", args.yolo_model, "--epochs", args.yolo_epochs, "--device", args.device],
            f"stage 4: YOLO ({args.task}) styled->real, per-epoch train+real AP")

    print("\n[da] pipeline complete.")


if __name__ == "__main__":
    main()
