"""
Run a trained YOLO instance-segmentation model on the test split and write:
  1. COCO-format predictions JSON (image_id / category_id / bbox / score / RLE)
  2. Optional per-image visualizations (drawn predictions overlaid on the RGB)
  3. Optional COCOeval summary (box-mAP + mask-mAP against the test
     _annotations.coco.json)

Run with the project venv python:

    /home/edge-host/Documents/.venv/bin/python predict_yolo.py \
        --weights runs/segment/train/weights/best.pt \
        --data ward_v1 \
        --save-viz --eval
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ROS2_bridge" / "src"))
from fixed_categories import FIXED_CATEGORIES  # type: ignore  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True,
                   help="Path to trained YOLO weights (e.g. runs/segment/train/weights/best.pt)")
    p.add_argument("--data", type=Path, required=True,
                   help="Dataset root with test/_annotations.coco.json + test/images/")
    p.add_argument("--split", default="test", choices=("train", "valid", "test"))
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir for predictions (default: <data>/<split>_predictions/)")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold for keeping predictions")
    p.add_argument("--iou",  type=float, default=0.6,
                   help="NMS IoU threshold")
    p.add_argument("--device", default="0",
                   help="GPU id, 'cpu', or comma-list")
    p.add_argument("--save-viz", action="store_true",
                   help="Save annotated visualization JPG per test image")
    p.add_argument("--max-viz", type=int, default=None,
                   help="Cap number of visualizations written (default: all)")
    p.add_argument("--eval", action="store_true",
                   help="Run COCOeval (box + mask mAP) against the GT JSON")
    return p.parse_args()


def _palette(n: int):
    """Stable per-class color palette (HSV → BGR for cv2)."""
    out = []
    for i in range(n):
        h = int(180 * i / max(n, 1))
        rgb = cv2.cvtColor(np.uint8([[[h, 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
        out.append(tuple(int(c) for c in rgb))
    return out


def _mask_to_rle(binary_mask: np.ndarray) -> dict:
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _draw_predictions(image_bgr: np.ndarray, boxes_xyxy, masks_bool,
                      cls_ids, scores, class_names, palette,
                      mask_alpha: float = 0.45) -> np.ndarray:
    """Overlay masks (alpha-blended) + boxes + class:score labels on the image."""
    out = image_bgr.copy()
    H, W = out.shape[:2]
    # Masks layer (per-pixel argmax of overlapping colored masks)
    overlay = np.zeros_like(out, dtype=np.uint8)
    for m, cid in zip(masks_bool, cls_ids):
        color = palette[int(cid) % len(palette)]
        overlay[m] = color
    out = cv2.addWeighted(out, 1.0, overlay, mask_alpha, 0)
    # Boxes + labels
    for (x1, y1, x2, y2), cid, sc in zip(boxes_xyxy, cls_ids, scores):
        color = palette[int(cid) % len(palette)]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_names[int(cid)]}:{float(sc):.2f}"
        ((tw, th), bl) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (int(x1), int(y1) - th - 4),
                      (int(x1) + tw, int(y1)), color, -1)
        cv2.putText(out, label, (int(x1), int(y1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    args.weights = args.weights.expanduser().resolve()
    args.data    = args.data.expanduser().resolve()
    split_dir    = args.data / args.split
    gt_json      = split_dir / "_annotations.coco.json"
    if not gt_json.is_file():
        sys.exit(f"missing {gt_json}")

    out_dir = args.out or (args.data / f"{args.split}_predictions")
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz" if args.save_viz else None
    if viz_dir:
        viz_dir.mkdir(exist_ok=True)

    # Class mapping: YOLO class index (0..N-1) -> COCO category_id
    sorted_cats = sorted(
        ((name, cid) for name, cid in FIXED_CATEGORIES.items() if cid != 0),
        key=lambda nc: nc[1],
    )
    yolo_to_cat_id = [cid for _, cid in sorted_cats]
    class_names   = [name for name, _ in sorted_cats]
    print(f"[predict] {len(class_names)} classes")

    # ------------- load model + run inference -------------
    from ultralytics import YOLO
    print(f"[predict] loading {args.weights}")
    model = YOLO(str(args.weights))

    # Resolve image dir for this split (Ultralytics convention from train_yolo.py)
    images_dir = split_dir / "images"
    if not images_dir.is_dir():
        # Fallback: images at split root (test split before train_yolo restructured)
        images_dir = split_dir

    # Build file_name -> coco image_id map from the GT JSON so predictions can
    # be tied back to GT image_ids for evaluation.
    with open(gt_json) as f:
        gt = json.load(f)
    fname_to_id = {Path(img["file_name"]).name: int(img["id"]) for img in gt["images"]}

    # Predict all images in the dir at once — Ultralytics streams batches.
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not image_paths:
        sys.exit(f"no images under {images_dir}")
    print(f"[predict] running inference on {len(image_paths)} images")

    palette = _palette(len(class_names))
    coco_predictions = []
    n_viz = 0

    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        retina_masks=True,   # mask resampled to native image resolution
        verbose=False,
        stream=True,
    )

    # Ultralytics stream yields results in input order; zip with original paths
    # because r.path can be auto-renamed (e.g. "image723.jpg") and won't match
    # the COCO GT file_names.
    for img_path, r in zip(image_paths, results):
        fname = img_path.name
        if fname not in fname_to_id:
            print(f"  [warn] no GT entry for {fname}; skipping")
            continue
        image_id = fname_to_id[fname]
        H, W     = r.orig_shape

        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()        # (N, 4)
        scores     = r.boxes.conf.cpu().numpy()        # (N,)
        cls_ids    = r.boxes.cls.cpu().numpy().astype(int)

        if r.masks is not None:
            masks = r.masks.data.cpu().numpy().astype(bool)  # (N, H, W)
        else:
            masks = np.zeros((len(boxes_xyxy), H, W), dtype=bool)

        for i in range(len(boxes_xyxy)):
            cid    = int(cls_ids[i])
            if cid < 0 or cid >= len(yolo_to_cat_id):
                continue
            cat_id = yolo_to_cat_id[cid]
            x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[i]]
            w, h = x2 - x1, y2 - y1
            entry = {
                "image_id":    image_id,
                "category_id": cat_id,
                "bbox":        [x1, y1, w, h],
                "score":       float(scores[i]),
            }
            if masks[i].any():
                entry["segmentation"] = _mask_to_rle(masks[i])
            coco_predictions.append(entry)

        if viz_dir is not None and (args.max_viz is None or n_viz < args.max_viz):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                drawn = _draw_predictions(
                    img_bgr, boxes_xyxy, masks, cls_ids, scores,
                    class_names, palette,
                )
                out_path = viz_dir / (img_path.stem + "_pred.jpg")
                cv2.imwrite(str(out_path), drawn, [cv2.IMWRITE_JPEG_QUALITY, 90])
                n_viz += 1

    pred_path = out_dir / "predictions.coco.json"
    with open(pred_path, "w") as f:
        json.dump(coco_predictions, f)
    print(f"[predict] wrote {pred_path}  ({len(coco_predictions)} predictions)")
    if viz_dir:
        print(f"[predict] wrote {n_viz} visualizations under {viz_dir}")

    # ------------- COCOeval (optional) -------------
    if args.eval:
        if not coco_predictions:
            print("[eval] no predictions to evaluate")
            return
        print("[eval] running COCOeval against ground truth...")
        cocoGt = COCO(str(gt_json))
        try:
            cocoDt = cocoGt.loadRes(str(pred_path))
        except Exception as e:
            sys.exit(f"[eval] loadRes failed: {e}")
        for iou_type in ("bbox", "segm"):
            print(f"\n--- {iou_type} ---")
            ev = COCOeval(cocoGt, cocoDt, iou_type)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()


if __name__ == "__main__":
    main()
