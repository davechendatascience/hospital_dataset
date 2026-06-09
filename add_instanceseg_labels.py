"""Write COCO instance-SEGMENTATION labels (RLE masks) into ward_v3 train/val,
derived from the raw Isaac instance-segmentation (no re-render, no filtering).

For each image already present in <split>/images (rgb_frame_<idx>.png), reads
_raw/instance_segmentation_<idx>.png (uint16 per-pixel instance IDs) +
instance_segmentation_semantics_mapping_<idx>.json (id -> class), and for every
labelled instance writes a COCO annotation with an RLE `segmentation` mask +
tight bbox. Class names map to ids via fixed_categories. Overwrites the
bbox-only _annotations.coco.json in place.

    /home/edge-host/Documents/.venv/bin/python add_instanceseg_labels.py \
        --data ward_v3 --raw ward_v3/_raw --splits train,val
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ROS2_bridge" / "src"))
from fixed_categories import FIXED_CATEGORIES  # type: ignore  # noqa: E402

SKIP_CLASSES = {"BACKGROUND", "UNLABELLED"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("ward_v3"))
    ap.add_argument("--raw", type=Path, default=Path("ward_v3/_raw"))
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--min-pixels", type=int, default=16)
    args = ap.parse_args()

    categories = [{"id": cid, "name": name, "supercategory": "ward_object"}
                  for name, cid in FIXED_CATEGORIES.items()]

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        sdir = args.data / split
        rgbs = sorted(glob.glob(str(sdir / "images" / "*.png")))
        images, annotations = [], []
        ann_id = 1
        n_inst = n_drop = 0
        for img_i, rgb in enumerate(rgbs, 1):
            stem = Path(rgb).stem                       # rgb_frame_<idx>
            idx = re.search(r"(\d+)$", stem).group(1)
            inst_png = args.raw / f"instance_segmentation_{idx}.png"
            sem_map = args.raw / f"instance_segmentation_semantics_mapping_{idx}.json"
            if not (inst_png.is_file() and sem_map.is_file()):
                n_drop += 1
                continue
            seg = np.array(Image.open(inst_png))        # (H,W) uint16
            H, W = seg.shape[:2]
            mapping = json.loads(sem_map.read_text())
            images.append({"id": img_i, "file_name": Path(rgb).name,
                           "width": int(W), "height": int(H)})
            for inst_id in np.unique(seg):
                info = mapping.get(str(int(inst_id)))
                if not info:
                    continue
                cls = info.get("class")
                if cls in SKIP_CLASSES or cls not in FIXED_CATEGORIES:
                    continue
                m = (seg == inst_id).astype(np.uint8)
                if m.sum() < args.min_pixels:
                    continue
                rle = coco_mask.encode(np.asfortranarray(m))
                rle["counts"] = rle["counts"].decode("ascii")
                ys, xs = np.where(m)
                x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                annotations.append({
                    "id": ann_id, "image_id": img_i,
                    "category_id": FIXED_CATEGORIES[cls],
                    "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                    "area": int(m.sum()), "iscrowd": 0,
                    "segmentation": rle,
                })
                ann_id += 1
                n_inst += 1
            if img_i % 250 == 0:
                print(f"\r  {split} {img_i}/{len(rgbs)}", end="", flush=True)
        out = sdir / "_annotations.coco.json"
        out.write_text(json.dumps({"info": {}, "images": images,
                                   "annotations": annotations,
                                   "categories": categories}))
        print(f"\n[seg] {split}: {len(images)} images, {n_inst} instance masks "
              f"-> {out} (skipped {n_drop} frames missing raw seg)")
    print("[seg] done")


if __name__ == "__main__":
    main()
