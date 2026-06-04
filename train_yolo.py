"""
Train a YOLO instance-segmentation model on the ward_v1 dataset.

This script:
  1. Reads <dataset>/train/_annotations.coco.json (and valid/, test/)
  2. Converts each COCO annotation set to YOLO instance-segmentation .txt
     polygon format alongside the existing images (one .txt per image,
     written under <dataset>/<split>/labels/ — Ultralytics convention)
  3. Writes data.yaml describing the splits and class list
  4. Kicks off `ultralytics.YOLO.train()` with sensible defaults for
     hospital-object detection.

Run with the project venv python:

    /home/edge-host/Documents/.venv/bin/python train_yolo.py \
        --data ward_v1 \
        --model yolo11s-seg.pt \
        --epochs 50 --imgsz 1024 --batch 8
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from pycocotools import mask as coco_mask

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ROS2_bridge" / "src"))
from fixed_categories import FIXED_CATEGORIES  # type: ignore  # noqa: E402

SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True,
                   help="Dataset root containing train/, valid/, test/ "
                        "(produced by build_dataset.py)")
    p.add_argument("--model", default="yolo11s-seg.pt",
                   help="Ultralytics model checkpoint. Common picks: "
                        "yolo11n-seg (nano), yolo11s-seg (small, default), "
                        "yolo11m-seg (medium), yolo11l-seg (large)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz",  type=int, default=1024,
                   help="Training image size (longest side); 1024 keeps "
                        "small hospital objects visible without OOM at "
                        "batch 8 on 32-48 GB GPUs")
    p.add_argument("--batch",  type=int, default=8)
    p.add_argument("--device", default="0",
                   help="GPU id, 'cpu', or comma-list for multi-GPU "
                        "(e.g., '0,1'). Default: GPU 0.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="runs/segment",
                   help="Ultralytics project dir for run outputs")
    p.add_argument("--name", default=None,
                   help="Run name (default: timestamp-based)")
    p.add_argument("--rebuild-labels", action="store_true",
                   help="Force re-conversion of COCO -> YOLO labels even if "
                        "they already exist")
    p.add_argument("--skip-train", action="store_true",
                   help="Only convert labels + write data.yaml; don't train")
    return p.parse_args()


# ---------------------------------------------------------------------------
# COCO -> YOLO instance-seg label conversion
# ---------------------------------------------------------------------------

def _rle_or_poly_to_binary_mask(seg, H: int, W: int) -> np.ndarray:
    """Accept any COCO segmentation format and return an HxW uint8 binary mask."""
    if isinstance(seg, dict):
        # RLE
        return coco_mask.decode(seg).astype(np.uint8)
    if isinstance(seg, list) and seg:
        # polygon list of lists OR a single flat list
        if isinstance(seg[0], list):
            rles = coco_mask.frPyObjects(seg, H, W)
            return coco_mask.decode(coco_mask.merge(rles)).astype(np.uint8)
        else:
            rles = coco_mask.frPyObjects([seg], H, W)
            return coco_mask.decode(coco_mask.merge(rles)).astype(np.uint8)
    return np.zeros((H, W), dtype=np.uint8)


def _mask_to_yolo_polygon(mask: np.ndarray, W: int, H: int,
                          eps_pct: float = 0.002) -> list[list[float]]:
    """Find contours in `mask`, simplify with Douglas-Peucker, normalize.
    Returns a list of polygons; each polygon is a flat list [x1,y1,x2,y2,...]
    in 0-1 image coords. Tiny contours are filtered out."""
    if mask.sum() < 4:
        return []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if len(c) < 3:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(c, eps_pct * peri, True)
        if len(approx) < 3:
            continue
        flat = []
        for pt in approx.reshape(-1, 2):
            x, y = float(pt[0]), float(pt[1])
            flat.append(min(max(x / W, 0.0), 1.0))
            flat.append(min(max(y / H, 0.0), 1.0))
        polys.append(flat)
    return polys


def convert_split_to_yolo(split_dir: Path, cat_id_to_yolo: dict,
                          force: bool = False) -> tuple[int, int, int]:
    """Read split_dir/_annotations.coco.json, write per-image .txt labels
    alongside the images under split_dir/labels/. Also moves images into
    split_dir/images/ so Ultralytics' default layout works. Returns
    (n_images, n_annotations_written, n_annotations_dropped)."""
    coco_json = split_dir / "_annotations.coco.json"
    if not coco_json.exists():
        print(f"[yolo] no annotations at {coco_json}; skipping {split_dir.name}")
        return 0, 0, 0
    with open(coco_json) as f:
        coco = json.load(f)

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Move/hard-link images into images/ if they were at split_dir root
    for img in coco["images"]:
        src = split_dir / img["file_name"]
        dst = images_dir / Path(img["file_name"]).name
        if src.exists() and src != dst:
            if dst.exists():
                src.unlink()
            else:
                shutil.move(str(src), str(dst))

    # Group annotations by image
    by_img = {}
    for ann in coco["annotations"]:
        by_img.setdefault(int(ann["image_id"]), []).append(ann)

    n_written = 0
    n_dropped = 0
    for img in coco["images"]:
        img_id = int(img["id"])
        W, H = int(img["width"]), int(img["height"])
        fn = Path(img["file_name"]).name
        stem = Path(fn).stem
        label_path = labels_dir / f"{stem}.txt"
        if label_path.exists() and not force:
            continue
        anns = by_img.get(img_id, [])
        lines = []
        for ann in anns:
            cat_id = int(ann["category_id"])
            if cat_id not in cat_id_to_yolo:
                n_dropped += 1
                continue
            yolo_idx = cat_id_to_yolo[cat_id]
            mask = _rle_or_poly_to_binary_mask(ann.get("segmentation", []), H, W)
            polys = _mask_to_yolo_polygon(mask, W, H)
            if not polys:
                n_dropped += 1
                continue
            # YOLO instance-seg uses ONE polygon per annotation. If a mask
            # produced multiple disjoint polygons we write each as its own
            # row with the same class id (ultralytics handles this fine).
            for poly in polys:
                lines.append(
                    f"{yolo_idx} " + " ".join(f"{c:.6f}" for c in poly)
                )
                n_written += 1
        with open(label_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    return len(coco["images"]), n_written, n_dropped


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------

def write_data_yaml(data_root: Path, yolo_classes: list[str]) -> Path:
    """Write a YOLO data.yaml with absolute paths to the converted splits."""
    p = data_root / "data.yaml"
    lines = [
        f"path: {data_root.resolve()}",
        "train: train/images",
        "val:   valid/images",
        "test:  test/images",
        "",
        f"nc: {len(yolo_classes)}",
        "names:",
    ]
    for i, name in enumerate(yolo_classes):
        lines.append(f"  {i}: {name}")
    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    data_root = args.data.expanduser().resolve()
    if not data_root.is_dir():
        sys.exit(f"--data {data_root} does not exist")

    # Build the canonical category-id -> YOLO-class-index map from
    # fixed_categories.py (sorted by COCO id ascending, then yolo_idx = position).
    # We exclude the synthetic "ward_object" supercategory (id 0).
    sorted_cats = sorted(
        ((name, cid) for name, cid in FIXED_CATEGORIES.items() if cid != 0),
        key=lambda nc: nc[1],
    )
    cat_id_to_yolo = {cid: i for i, (_, cid) in enumerate(sorted_cats)}
    yolo_classes = [name for name, _ in sorted_cats]
    print(f"[yolo] {len(yolo_classes)} classes (yolo_idx 0..{len(yolo_classes)-1})")

    # Convert each split to YOLO format
    totals = {}
    missing = []
    for split in SPLITS:
        if not (data_root / split / "_annotations.coco.json").exists():
            missing.append(split)
            continue
        n_img, n_w, n_d = convert_split_to_yolo(
            data_root / split, cat_id_to_yolo, force=args.rebuild_labels,
        )
        totals[split] = (n_img, n_w, n_d)
        print(f"[yolo] {split}: {n_img} images, {n_w} polygons written, "
              f"{n_d} dropped")
    if missing:
        sys.exit(
            f"[yolo] dataset is incomplete: missing {missing} split(s). "
            f"Wait for build_dataset.py to finish before training."
        )

    yaml_path = write_data_yaml(data_root, yolo_classes)
    print(f"[yolo] wrote data spec: {yaml_path}")

    if args.skip_train:
        print("[yolo] --skip-train set; stopping after label conversion")
        return

    # Train
    print(f"[yolo] starting training: model={args.model}, epochs={args.epochs}, "
          f"imgsz={args.imgsz}, batch={args.batch}, device={args.device}")
    from ultralytics import YOLO
    model = YOLO(args.model)

    # Per-epoch eval on the held-out test split, in addition to the standard
    # val-split eval Ultralytics already runs. We hook on_fit_epoch_end (fires
    # AFTER Ultralytics has written last.pt for the epoch), load last.pt into
    # a fresh YOLO wrapper, and run val() with split='test'. Using last.pt
    # rather than poking trainer.model avoids any interaction with the live
    # EMA / autograd state.
    test_metrics_rows: list[dict] = []

    def _eval_test_split(trainer) -> None:
        epoch = int(trainer.epoch) + 1  # match results.csv 1-indexing
        last_pt = Path(trainer.save_dir) / "weights" / "last.pt"
        if not last_pt.is_file():
            return
        eval_model = YOLO(str(last_pt))
        metrics = eval_model.val(
            data=str(yaml_path),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            verbose=False,
            plots=False,
            save_json=False,
        )
        row = {
            "epoch":     epoch,
            "box_map":   float(metrics.box.map),
            "box_map50": float(metrics.box.map50),
            "box_map75": float(metrics.box.map75),
        }
        seg = getattr(metrics, "seg", None)
        if seg is not None and getattr(seg, "map", None) is not None:
            row["seg_map"]   = float(seg.map)
            row["seg_map50"] = float(seg.map50)
            row["seg_map75"] = float(seg.map75)
        test_metrics_rows.append(row)

        csv_path = Path(trainer.save_dir) / "test_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(test_metrics_rows[0].keys()))
            w.writeheader()
            for r in test_metrics_rows:
                w.writerow(r)
        print("[test@epoch{:d}] ".format(epoch)
              + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"))

    model.add_callback("on_fit_epoch_end", _eval_test_split)

    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        # Sensible defaults for sim-to-real detection: a bit of geometric +
        # photometric augmentation (test set is real photos, so we want the
        # model to be invariant to camera/lighting nuisance).
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        translate=0.1, scale=0.4, fliplr=0.5,
        mosaic=1.0, mixup=0.0, copy_paste=0.0,
        plots=True,
    )


if __name__ == "__main__":
    main()
