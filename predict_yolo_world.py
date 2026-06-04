"""
Run a YOLO-World checkpoint on a COCO-format split and write:
  1. COCO-format predictions JSON (bbox-only — YOLO-World is detection)
  2. Optional per-image visualizations
  3. Optional pycocotools COCOeval (bbox-mAP) against the split's GT

Works with either the pretrained checkpoint (yolov8x-worldv2.pt) for a
zero-shot baseline, or a PEFT-fine-tuned weight (runs/.../best.pt).

Run:
    /home/edge-host/Documents/.venv/bin/python predict_yolo_world.py \\
        --weights runs/world/peft_smoke/weights/best.pt \\
        --data ward_v1 --split real_test \\
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
from fixed_categories import FIXED_CATEGORIES  # noqa: E402


_PROMPT_OVERRIDES = {
    "TV":                       "television",
    "iv_pole":                  "IV pole",
    "tv":                       "television",
    "hospital_bed":             "hospital bed",
    "bedside_monitor":          "bedside patient monitor",
    "bedside_table":            "bedside table",
    "overbed_table":            "overbed table",
    "air_vent":                 "air vent",
    "alcohol_spray_bottle":     "alcohol spray bottle",
    "bed_curtain":              "bed privacy curtain",
    "companion_chair":          "companion chair",
    "ear_thermometer":          "ear thermometer",
    "gas_manifold":             "medical gas wall manifold",
    "medical_gloves":           "medical gloves",
    "medical_package":          "medical package",
    "medical_waste_container":  "medical waste container",
    "oxygen_flowmeter":         "oxygen flowmeter",
    "remote_control":           "remote control",
    "soiled_linen_bin":         "soiled linen bin",
    "tissue_dispenser":         "tissue dispenser",
    "toilet_handle":            "toilet handle",
    "toilet":                   "toilet",
    "weight_scale":             "weight scale",
    "waste_bin":                "waste bin",
    "door_handle":              "door handle",
    "light_switch":             "light switch",
    "suction_jar":              "suction jar",
    "suction_knob":             "suction knob",
    "ward_object":              "ward_object",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=Path("yolov8x-worldv2.pt"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", default="real_test",
                   help="Subdir of --data with _annotations.coco.json + images/")
    p.add_argument("--out", type=Path, default=None,
                   help="Default: <data>/<split>_yoloworld_predictions/")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf",  type=float, default=0.15)
    p.add_argument("--iou",   type=float, default=0.5)
    p.add_argument("--device", default="0")
    p.add_argument("--save-viz", action="store_true")
    p.add_argument("--max-viz", type=int, default=None)
    p.add_argument("--eval", action="store_true")
    return p.parse_args()


def _palette(n):
    out = []
    for i in range(n):
        h = int(180 * i / max(n, 1))
        rgb = cv2.cvtColor(np.uint8([[[h, 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
        out.append(tuple(int(c) for c in rgb))
    return out


def _draw(img_bgr, boxes_xyxy, cls_idxs, scores, names, palette):
    out = img_bgr.copy()
    for (x1, y1, x2, y2), ci, sc in zip(boxes_xyxy, cls_idxs, scores):
        color = palette[int(ci) % len(palette)]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{names[int(ci)]}:{float(sc):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (int(x1), int(y1) - th - 4),
                      (int(x1) + tw, int(y1)), color, -1)
        cv2.putText(out, label, (int(x1), int(y1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    args.data = args.data.expanduser().resolve()
    split_dir = args.data / args.split
    gt_json   = split_dir / "_annotations.coco.json"
    if not gt_json.is_file():
        sys.exit(f"missing {gt_json}")

    out_dir = args.out or (args.data / f"{args.split}_yoloworld_predictions")
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz" if args.save_viz else None
    if viz_dir:
        viz_dir.mkdir(exist_ok=True)

    # Canonical class order (same as train_yolo_world.py)
    sorted_cats = sorted(
        ((name, cid) for name, cid in FIXED_CATEGORIES.items() if cid != 0),
        key=lambda nc: nc[1],
    )
    class_idx_to_cat_id = [cid for _, cid in sorted_cats]
    class_names = [name for name, _ in sorted_cats]
    prompts = [_PROMPT_OVERRIDES.get(n, n.replace("_", " ")) for n in class_names]
    print(f"[world] {len(prompts)} prompts; sample: {prompts[:5]}")

    images_dir = split_dir / "images"
    if not images_dir.is_dir():
        images_dir = split_dir
    image_paths = sorted([*images_dir.glob("*.png"), *images_dir.glob("*.jpg")])
    if not image_paths:
        sys.exit(f"no images under {images_dir}")
    print(f"[world] {len(image_paths)} images to process")

    with open(gt_json) as f:
        gt = json.load(f)
    fname_to_id = {Path(im["file_name"]).name: int(im["id"]) for im in gt["images"]}

    from ultralytics import YOLOWorld
    print(f"[world] loading {args.weights}")
    model = YOLOWorld(str(args.weights))
    model.set_classes(prompts)
    palette = _palette(len(class_names))
    predictions = []
    n_viz = 0

    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        device=args.device, verbose=False, stream=True,
        batch=1, workers=0, half=True,
    )

    for img_path, r in zip(image_paths, results):
        fname = img_path.name
        if fname not in fname_to_id:
            continue
        image_id = fname_to_id[fname]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        scores     = r.boxes.conf.cpu().numpy()
        cls_ids    = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes_xyxy)):
            ci = int(cls_ids[i])
            if ci < 0 or ci >= len(class_idx_to_cat_id):
                continue
            x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[i]]
            predictions.append({
                "image_id":    image_id,
                "category_id": class_idx_to_cat_id[ci],
                "bbox":        [x1, y1, x2 - x1, y2 - y1],
                "score":       float(scores[i]),
            })
        if viz_dir is not None and (args.max_viz is None or n_viz < args.max_viz):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                drawn = _draw(img_bgr, boxes_xyxy, cls_ids, scores, class_names, palette)
                cv2.imwrite(str(viz_dir / (img_path.stem + "_pred.jpg")), drawn,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                n_viz += 1

    pred_path = out_dir / "predictions.coco.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f)
    print(f"[world] wrote {pred_path} ({len(predictions)} predictions)")
    if viz_dir:
        print(f"[world] wrote {n_viz} viz under {viz_dir}")

    if args.eval:
        if not predictions:
            print("[eval] no predictions to evaluate"); return
        cocoGt = COCO(str(gt_json))
        cocoDt = cocoGt.loadRes(str(pred_path))
        print("\n--- bbox ---")
        ev = COCOeval(cocoGt, cocoDt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()

        print("\n--- per-class AP@[.5:.95] (bbox) ---")
        precisions = ev.eval["precision"]   # (T, R, K, A, M)
        cat_ids = cocoGt.getCatIds()
        cats = {c["id"]: c["name"] for c in cocoGt.dataset["categories"]}
        rows = []
        for k, cid in enumerate(cat_ids):
            p = precisions[:, :, k, 0, -1]
            p_valid = p[p > -1]
            ap = float(p_valid.mean()) if p_valid.size else float("nan")
            n_gt = len(cocoGt.getAnnIds(catIds=[cid]))
            rows.append((cats.get(cid, "?"), n_gt, ap))
        rows.sort(key=lambda r: -r[1])
        print(f"{'class':25s} {'gt_n':>6s} {'AP':>6s}")
        for name, n, ap in rows:
            if n > 0 or (ap == ap):
                print(f"{name:25s} {n:6d} {ap:6.3f}")


if __name__ == "__main__":
    main()
