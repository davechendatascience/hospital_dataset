"""Stage 4-5: train a YOLO detector on a (styled) sim dataset and evaluate on
the held-out REAL split, reporting BOTH the train-set score and the real-test
score every epoch (so the sim->real generalization gap is visible per-epoch).

Train dir and val dir each need images/ + _annotations.coco.json; YOLO labels
are derived from the COCO (bbox for --task detect, polygons for segment) into a
labels/ dir. real_holdout is the val/test domain — selection is on real AP.

    .venv/bin/python train_yolo_da.py \
        --train-dir ward_v3_styled/train --val-dir ward_v1/real_holdout \
        --task detect --model yolo11s.pt --epochs 50
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco_mask

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ROS2_bridge" / "src"))
from fixed_categories import FIXED_CATEGORIES  # type: ignore  # noqa: E402

ID2NAME = {cid: n for n, cid in FIXED_CATEGORIES.items()}
NAMES = [ID2NAME[c] for c in sorted(ID2NAME) if c != 0]   # contiguous 0..N-1


def _polys(m, W, H, eps=0.002):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if len(c) < 3:
            continue
        p = cv2.arcLength(c, True)
        a = cv2.approxPolyDP(c, eps * p, True)
        if len(a) < 3:
            continue
        out.append([min(max(v, 0.0), 1.0) for pt in a.reshape(-1, 2)
                    for v in (pt[0] / W, pt[1] / H)])
    return out


def ensure_labels(split_dir: Path, task: str):
    """Derive YOLO labels/ from the split's COCO json for the given task."""
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())
    meta = {im["id"]: im for im in coco["images"]}
    by_img = defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)
    ldir = split_dir / "labels"
    ldir.mkdir(exist_ok=True)
    for f in ldir.glob("*.txt"):
        f.unlink()
    for img_id, im in meta.items():
        stem = Path(im["file_name"]).stem
        W, H = im["width"], im["height"]
        lines = []
        for a in by_img.get(img_id, []):
            if a["category_id"] == 0:
                continue
            cls = a["category_id"] - 1
            if task == "segment" and isinstance(a.get("segmentation"), dict):
                for poly in _polys(coco_mask.decode(a["segmentation"]), W, H):
                    lines.append(f"{cls} " + " ".join(f"{c:.6f}" for c in poly))
            else:
                x, y, w, h = a["bbox"]
                lines.append(f"{cls} {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}")
        (ldir / f"{stem}.txt").write_text(("\n".join(lines) + "\n") if lines else "")


def write_yaml(path: Path, train_imgs: Path, val_imgs: Path) -> Path:
    lines = [f"path: {path.parent.resolve()}",
             f"train: {train_imgs.resolve()}",
             f"val: {val_imgs.resolve()}",
             f"nc: {len(NAMES)}", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    path.write_text("\n".join(lines) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", type=Path, required=True, help="styled sim split")
    ap.add_argument("--val-dir", type=Path, required=True, help="real_holdout (eval domain)")
    ap.add_argument("--task", choices=["detect", "segment"], default="detect")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/yolo_da")
    ap.add_argument("--name", default="styled_to_real")
    args = ap.parse_args()

    print(f"[yolo-da] deriving YOLO {args.task} labels for train + val splits")
    ensure_labels(args.train_dir, args.task)
    ensure_labels(args.val_dir, args.task)

    out = PROJECT_ROOT / args.project / args.name
    out.mkdir(parents=True, exist_ok=True)
    data_yaml = write_yaml(out / "data.yaml",
                           args.train_dir / "images", args.val_dir / "images")
    # second yaml whose "val" is the TRAIN images, for per-epoch train-set scoring
    train_eval_yaml = write_yaml(out / "data_traineval.yaml",
                                 args.train_dir / "images", args.train_dir / "images")

    from ultralytics import YOLO
    model = YOLO(args.model)

    rows = []
    csv_path = out / "da_metrics.csv"

    def on_epoch(trainer):
        ep = int(trainer.epoch) + 1
        last = Path(trainer.save_dir) / "weights" / "last.pt"
        if not last.is_file():
            return
        # train-set score (held-out val score is what ultralytics already logs)
        tr = YOLO(str(last)).val(data=str(train_eval_yaml), split="val",
                                 imgsz=args.imgsz, batch=args.batch, device=args.device,
                                 verbose=False, plots=False, save_json=False)
        m = trainer.metrics or {}
        key = "metrics/mAP50-95(B)"
        val_map = float(m.get(key, 0.0))
        train_map = float(tr.box.map)
        row = {"epoch": ep, "train_mAP": round(train_map, 4),
               "real_val_mAP": round(val_map, 4),
               "gap": round(train_map - val_map, 4)}
        rows.append(row)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerows(rows)
        print(f"[yolo-da][epoch {ep}] train_mAP={train_map:.4f}  "
              f"real_val_mAP={val_map:.4f}  gap={train_map-val_map:.4f}", flush=True)

    model.add_callback("on_fit_epoch_end", on_epoch)
    print(f"[yolo-da] train={args.train_dir} -> val(real)={args.val_dir}  "
          f"task={args.task} model={args.model}")
    model.train(data=str(data_yaml), task=args.task, epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, device=args.device,
                project=args.project, name=args.name, exist_ok=True)
    print(f"[yolo-da] done. per-epoch train+real scores -> {csv_path}")


if __name__ == "__main__":
    main()
