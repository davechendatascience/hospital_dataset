"""Add depth IMAGES to existing train/val splits, paired with the RGB.

Reads each split image's frame index, loads the matching
_raw/distance_to_camera_<idx>.npy (radial depth, metric), normalises it to an
8-bit PNG (per-image min-max, near=bright), and writes <split>/depth/<same
stem>.png so images/X.png <-> depth/X.png pair by filename. RGB untouched.

    /home/edge-host/Documents/.venv/bin/python add_depth_to_splits.py \
        --data ward_v3 --raw ward_v3/_raw --splits train,val
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
from PIL import Image


def depth_png(npy_path: Path, invert=True) -> Image.Image:
    d = np.load(npy_path).astype(np.float32)
    finite = np.isfinite(d) & (d < 1e6)
    if finite.any():
        d = np.where(finite, d, d[finite].max())   # background -> farthest
        lo, hi = d.min(), d.max()
        norm = (d - lo) / (hi - lo + 1e-6)
        if invert:
            norm = 1.0 - norm                       # near = bright
        img = (norm * 255).astype(np.uint8)
    else:
        img = np.zeros(d.shape, np.uint8)
    return Image.fromarray(img, mode="L")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("ward_v3"))
    ap.add_argument("--raw", type=Path, default=Path("ward_v3/_raw"))
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--depth-src", default="distance_to_camera")
    args = ap.parse_args()

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        img_dir = args.data / split / "images"
        depth_dir = args.data / split / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        rgbs = sorted(glob.glob(str(img_dir / "*.png")))
        n_ok = n_miss = 0
        for p in rgbs:
            stem = Path(p).stem
            idx = re.search(r"(\d+)$", stem).group(1)
            npy = args.raw / f"{args.depth_src}_{idx}.npy"
            if not npy.is_file():
                n_miss += 1
                continue
            depth_png(npy).save(depth_dir / f"{stem}.png")
            n_ok += 1
            if n_ok % 250 == 0:
                print(f"\r  {split} {n_ok}/{len(rgbs)}", end="", flush=True)
        print(f"\n[depth] {split}: wrote {n_ok} depth PNGs -> {depth_dir} "
              f"(missing {n_miss})")
    print("[depth] done")


if __name__ == "__main__":
    main()
