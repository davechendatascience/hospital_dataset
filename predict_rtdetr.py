"""
Run a trained RT-DETR (ViT-encoder transformer detector) on a split and write:
  1. COCO-format predictions JSON (image_id / category_id / bbox / score)
  2. Optional per-image visualizations (predicted boxes overlaid on the RGB)
  3. Optional COCOeval summary (box mAP against the split _annotations.coco.json)

Run with the project venv python:

    /home/edge-host/Documents/.venv/bin/python predict_rtdetr.py \
        --weights runs/detect/rtdetr_sim/weights/best.pt \
        --data ward_v1 --split real_test \
        --save-viz --eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ROS2_bridge" / "src"))
from fixed_categories import FIXED_CATEGORIES  # type: ignore  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", default="real_test",
                   choices=("train", "valid", "test", "real_test", "real_train"))
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: <data>/<split>_rtdetr_predictions/)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--device", default="0")
    p.add_argument("--save-viz", action="store_true")
    p.add_argument("--max-viz", type=int, default=None)
    p.add_argument("--eval", action="store_true",
                   help="Run COCOeval (box mAP) against the GT JSON in the split")
    return p.parse_args()


def build_class_maps() -> tuple[list[str], dict[int, int]]:
    """Returns (yolo_classes, yolo_idx -> coco_category_id) matching the
    mapping used by train_yolo.py / train_rtdetr.py."""
    sorted_cats = sorted(
        ((name, cid) for name, cid in FIXED_CATEGORIES.items() if cid != 0),
        key=lambda nc: nc[1],
    )
    yolo_classes = [name for name, _ in sorted_cats]
    yolo_to_coco = {i: cid for i, (_, cid) in enumerate(sorted_cats)}
    return yolo_classes, yolo_to_coco


def color_for(cls_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(cls_id * 9973 + 1)
    return tuple(int(x) for x in rng.integers(64, 255, size=3))


def draw_predictions(img: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                     classes: np.ndarray, names: list[str]) -> np.ndarray:
    out = img.copy()
    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
        c = color_for(int(cls))
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        label = f"{names[int(cls)]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (int(x1), int(y1) - th - 4),
                      (int(x1) + tw + 4, int(y1)), c, -1)
        cv2.putText(out, label, (int(x1) + 2, int(y1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    data_root = args.data.expanduser().resolve()
    split_dir = data_root / args.split
    images_dir = split_dir / "images"
    if not images_dir.is_dir():
        sys.exit(f"[rtdetr-predict] missing {images_dir}")

    out_dir = args.out or (data_root / f"{args.split}_rtdetr_predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz"
    if args.save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    yolo_classes, yolo_to_coco = build_class_maps()

    # Build coco file_name -> image_id map (so predictions reference the GT
    # image ids, which COCOeval requires).
    gt_path = split_dir / "_annotations.coco.json"
    fn_to_image_id: dict[str, int] = {}
    if gt_path.exists():
        with open(gt_path) as f:
            gt = json.load(f)
        for im in gt["images"]:
            fn_to_image_id[Path(im["file_name"]).name] = int(im["id"])
    else:
        print(f"[rtdetr-predict] no GT json at {gt_path}; predictions will use "
              f"synthetic image ids")

    from ultralytics import RTDETR
    model = RTDETR(str(args.weights))

    image_paths = sorted([p for p in images_dir.iterdir()
                          if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    print(f"[rtdetr-predict] {len(image_paths)} images in {images_dir}")

    coco_preds: list[dict] = []
    next_synth_id = 10_000_000
    n_viz_saved = 0

    # Stream predictions one image at a time to keep memory bounded.
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
        stream=True,
    )
    for res in results:
        img_path = Path(res.path)
        img_name = img_path.name
        if img_name in fn_to_image_id:
            img_id = fn_to_image_id[img_name]
        else:
            img_id = next_synth_id
            next_synth_id += 1

        if res.boxes is None or len(res.boxes) == 0:
            if args.save_viz and (args.max_viz is None or n_viz_saved < args.max_viz):
                img = cv2.imread(str(img_path))
                cv2.imwrite(str(viz_dir / img_name), img)
                n_viz_saved += 1
            continue

        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
            w = float(x2 - x1)
            h = float(y2 - y1)
            coco_preds.append({
                "image_id": img_id,
                "category_id": int(yolo_to_coco[int(c)]),
                "bbox": [float(x1), float(y1), w, h],
                "score": float(s),
            })

        if args.save_viz and (args.max_viz is None or n_viz_saved < args.max_viz):
            img = cv2.imread(str(img_path))
            viz = draw_predictions(img, xyxy, conf, cls, yolo_classes)
            cv2.imwrite(str(viz_dir / img_name), viz)
            n_viz_saved += 1

    preds_path = out_dir / "predictions_coco.json"
    with open(preds_path, "w") as f:
        json.dump(coco_preds, f)
    print(f"[rtdetr-predict] wrote {len(coco_preds)} predictions to {preds_path}")
    if args.save_viz:
        print(f"[rtdetr-predict] wrote {n_viz_saved} visualizations to {viz_dir}")

    if args.eval:
        if not gt_path.exists():
            print(f"[rtdetr-predict] --eval set but no GT at {gt_path}")
            return
        if not coco_preds:
            print(f"[rtdetr-predict] no predictions produced; skipping COCOeval")
            return
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(preds_path))
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        # Persist the summary stats
        stats = {
            "AP":      float(evaluator.stats[0]),
            "AP50":    float(evaluator.stats[1]),
            "AP75":    float(evaluator.stats[2]),
            "AP_S":    float(evaluator.stats[3]),
            "AP_M":    float(evaluator.stats[4]),
            "AP_L":    float(evaluator.stats[5]),
            "AR_1":    float(evaluator.stats[6]),
            "AR_10":   float(evaluator.stats[7]),
            "AR_100":  float(evaluator.stats[8]),
            "AR_S":    float(evaluator.stats[9]),
            "AR_M":    float(evaluator.stats[10]),
            "AR_L":    float(evaluator.stats[11]),
        }
        with open(out_dir / "cocoeval.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[rtdetr-predict] wrote eval summary to {out_dir / 'cocoeval.json'}")


if __name__ == "__main__":
    main()
