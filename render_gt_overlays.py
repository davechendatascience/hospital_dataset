"""Render ground-truth COCO instance masks over a dataset's images.

Companion to predict_seg_detr.py (same overlay style) but drawing the LABELS,
not predictions -- the quick way to verify annotation alignment per split
(e.g. that ward_v3 labels carried onto the Cosmos-styled ward_v4 frames).

    .venv/bin/python render_gt_overlays.py --data ward_v4 \
        --splits train,valid,real_dev,real_holdout --n 3 --out ward_v4_gt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

PROJECT = Path(__file__).resolve().parent


def render_split(split_dir: Path, out_dir: Path, n: int, seed: int) -> int:
    ann_file = split_dir / "_annotations.coco.json"
    img_dir = split_dir / "images"
    if not ann_file.is_file() or not img_dir.is_dir():
        print(f"[gt] {split_dir}: missing images/ or annotations -- skipped")
        return 0
    coco = COCO(str(ann_file))
    names = {c["id"]: c["name"] for c in coco.dataset["categories"]}
    ids = [i for i in coco.imgs
           if (img_dir / coco.imgs[i]["file_name"]).is_file()]
    random.Random(seed).shuffle(ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    palette = rng.randint(60, 255, (64, 3))
    done = 0
    for img_id in ids[:n]:
        info = coco.imgs[img_id]
        im = Image.open(img_dir / info["file_name"]).convert("RGB")
        base = np.array(im, dtype=np.float32)
        draw_jobs = []
        for k, a in enumerate(coco.imgToAnns.get(img_id, [])):
            if a["category_id"] == 0 or not a.get("segmentation"):
                continue
            m = coco.annToMask(a).astype(bool)
            if m.shape != base.shape[:2] or not m.any():
                continue
            color = palette[k % len(palette)]
            base[m] = 0.45 * base[m] + 0.55 * color
            ys, xs = np.nonzero(m)
            draw_jobs.append((int(xs.mean()), int(ys.mean()),
                              names.get(a["category_id"], "?"), tuple(color)))
        vis = Image.fromarray(base.astype(np.uint8))
        d = ImageDraw.Draw(vis)
        for x, y, txt, color in draw_jobs:
            w = d.textlength(txt)
            d.rectangle([x - 2, y - 11, x + w + 2, y + 2], fill=(0, 0, 0))
            d.text((x, y - 10), txt, fill=color)
        dst = out_dir / f"gt_{Path(info['file_name']).stem}.jpg"
        vis.save(dst, quality=92)
        print(f"[gt] {split_dir.name}/{info['file_name']}: "
              f"{len(draw_jobs)} instances -> {dst}")
        done += 1
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=PROJECT / "ward_v4")
    ap.add_argument("--splits", default="train,valid,real_dev,real_holdout")
    ap.add_argument("--n", type=int, default=3, help="images per split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=PROJECT / "ward_v4_gt")
    args = ap.parse_args()

    total = 0
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        total += render_split(args.data / split, args.out / split,
                              args.n, args.seed)
    print(f"[gt] {total} overlays -> {args.out}")


if __name__ == "__main__":
    main()
