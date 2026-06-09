"""Rebuild ward_v3 train/val (2700/300) consistently from _raw + sim.json,
without disturbing files that already exist.

For each split (deterministic seed-42 partition, matching split_coco):
  * _annotations.coco.json  -> rewritten from sim.json, filtered to the split
    (so the annotations always match the images actually present).
  * images/<file_name>      -> copied from _raw/rgb_<idx>.png if missing.
  * depth/<stem>.png        -> normalized from _raw/distance_to_camera_<idx>.npy
                               if missing (8-bit, near=bright).
Existing files are left untouched (train images/depth stay as-is).

    /home/edge-host/Documents/.venv/bin/python materialize_ward_v3.py \
        --coco ward_v3/jsonDataset/sim.json --raw ward_v3/_raw --out ward_v3
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def depth_png(npy_path: Path, invert=True) -> Image.Image:
    d = np.load(npy_path).astype(np.float32)
    finite = np.isfinite(d) & (d < 1e6)
    if finite.any():
        d = np.where(finite, d, d[finite].max())
        lo, hi = d.min(), d.max()
        norm = (d - lo) / (hi - lo + 1e-6)
        if invert:
            norm = 1.0 - norm
        img = (norm * 255).astype(np.uint8)
    else:
        img = np.zeros(d.shape, np.uint8)
    return Image.fromarray(img, mode="L")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", type=Path, default=Path("ward_v3/jsonDataset/sim.json"))
    ap.add_argument("--raw", type=Path, default=Path("ward_v3/_raw"))
    ap.add_argument("--out", type=Path, default=Path("ward_v3"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    coco = json.loads(args.coco.read_text())
    images = sorted(coco["images"], key=lambda im: im["file_name"])
    order = [re.search(r"(\d+)\.png$", im["file_name"]).group(1) for im in images]
    rng = random.Random(args.seed)
    shuf = order[:]; rng.shuffle(shuf)
    n_val = int(round(len(shuf) * args.val_frac))
    val_idx = set(shuf[:n_val])

    def idx_of(im):
        return re.search(r"(\d+)\.png$", im["file_name"]).group(1)

    for split in ("train", "val"):
        keep = [im for im in images if (idx_of(im) in val_idx) == (split == "val")]
        keep_ids = {im["id"] for im in keep}
        anns = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
        sdir = args.out / split
        (sdir / "images").mkdir(parents=True, exist_ok=True)
        (sdir / "depth").mkdir(parents=True, exist_ok=True)
        # annotations (always rewritten so they match the present images)
        (sdir / "_annotations.coco.json").write_text(json.dumps({
            "info": coco.get("info", {}), "images": keep,
            "annotations": anns, "categories": coco["categories"]}))
        n_img = n_dep = 0
        for im in keep:
            fn = im["file_name"]                  # rgb_frame_<idx>.png
            idx = idx_of(im)
            stem = Path(fn).stem
            dst_img = sdir / "images" / fn
            if not dst_img.exists():
                src = args.raw / f"rgb_{idx}.png"
                if src.exists():
                    Image.open(src).convert("RGB").save(dst_img); n_img += 1
            dst_dep = sdir / "depth" / f"{stem}.png"
            if not dst_dep.exists():
                src = args.raw / f"distance_to_camera_{idx}.npy"
                if src.exists():
                    depth_png(src).save(dst_dep); n_dep += 1
        print(f"[mat] {split}: {len(keep)} imgs / {len(anns)} anns in json; "
              f"copied {n_img} new images, {n_dep} new depth")
    print("[mat] done")


if __name__ == "__main__":
    main()
