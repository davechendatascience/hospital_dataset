"""Write per-image YOLO label files from each split's COCO json.

For every image in <split>/_annotations.coco.json, emits one .txt per image in:
  <split>/labels_bbox/<stem>.txt   class cx cy w h           (YOLO detection)
  <split>/labels_seg/<stem>.txt    class x1 y1 x2 y2 ...     (YOLO instance-seg)
  <split>/labels/<stem>.txt        = labels_seg (YOLO default layout)

YOLO class index = COCO category_id - 1 (ids are 1..43 for the real classes).
Seg polygons come from the RLE masks; bbox from the COCO bbox. No images moved.

    /home/edge-host/Documents/.venv/bin/python coco_to_yolo_labels.py \
        --data ward_v3 --splits train,val
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco_mask


def mask_to_polys(m, W, H, eps_pct=0.002):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
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
            flat.append(min(max(float(pt[0]) / W, 0.0), 1.0))
            flat.append(min(max(float(pt[1]) / H, 0.0), 1.0))
        polys.append(flat)
    return polys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("ward_v3"))
    ap.add_argument("--splits", default="train,val")
    args = ap.parse_args()

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        sdir = args.data / split
        coco = json.loads((sdir / "_annotations.coco.json").read_text())
        for d in ("labels", "labels_bbox", "labels_seg"):
            (sdir / d).mkdir(parents=True, exist_ok=True)
        meta = {im["id"]: im for im in coco["images"]}
        by_img = defaultdict(list)
        for a in coco["annotations"]:
            by_img[a["image_id"]].append(a)

        n_box = n_seg = 0
        for img_id, im in meta.items():
            stem = Path(im["file_name"]).stem
            W, H = im["width"], im["height"]
            bbox_lines, seg_lines = [], []
            for a in by_img.get(img_id, []):
                cls = a["category_id"] - 1
                x, y, w, h = a["bbox"]
                bbox_lines.append(f"{cls} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                                  f"{w / W:.6f} {h / H:.6f}")
                seg = a.get("segmentation")
                if isinstance(seg, dict) and "counts" in seg:
                    for poly in mask_to_polys(coco_mask.decode(seg), W, H):
                        seg_lines.append(f"{cls} " + " ".join(f"{c:.6f}" for c in poly))
            nl = lambda L: ("\n".join(L) + "\n") if L else ""
            (sdir / "labels_bbox" / f"{stem}.txt").write_text(nl(bbox_lines))
            (sdir / "labels_seg" / f"{stem}.txt").write_text(nl(seg_lines))
            (sdir / "labels" / f"{stem}.txt").write_text(nl(seg_lines))
            n_box += len(bbox_lines); n_seg += len(seg_lines)
        print(f"[yolo-labels] {split}: {len(meta)} images -> "
              f"labels/ labels_bbox/ labels_seg/  ({n_box} boxes, {n_seg} seg polys)")
    print("[yolo-labels] done")


if __name__ == "__main__":
    main()
