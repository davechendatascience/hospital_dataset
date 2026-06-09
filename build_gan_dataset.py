"""Turn a raw Isaac render (_raw/) into a clean RGB+depth dataset for
depth-conditioned CUT training.

Reads <render>/_raw/{rgb_NNNN.png, distance_to_camera_NNNN.npy} and writes:
    <out>/images/NNNN.png   RGB (3ch)
    <out>/depth/NNNN.png    depth, per-image min-max normalised to 8-bit
                            (near = bright; inf/NaN background -> farthest)

Per-image normalisation gives scene-relative depth (like Depth-Anything /
ControlNet depth), so the conditioning channel is robust without knowing the
absolute room scale. RGB and depth share the frame index so CUT can stack them.

    /home/edge-host/Documents/.venv/bin/python build_gan_dataset.py \
        --raw ward_v3_depth/_raw --out ward_v3_depth/sim
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="render's _raw/ dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depth-src", default="distance_to_camera",
                    choices=["distance_to_camera", "distance_to_image_plane"])
    ap.add_argument("--no-invert", action="store_true",
                    help="far=bright instead of the default near=bright")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of frames -> val split (0 = flat, no split)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rgbs = sorted(glob.glob(str(args.raw / "rgb_*.png")))
    # deterministic train/val assignment by frame
    import random
    idxs = [re.search(r"rgb_(\d+)\.png$", p).group(1) for p in rgbs]
    rng = random.Random(args.seed)
    val_set = set()
    if args.val_frac > 0:
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * args.val_frac))
        val_set = set(shuffled[:n_val])
    splits = ["train", "val"] if args.val_frac > 0 else ["."]
    for s in splits:
        (args.out / s / "images").mkdir(parents=True, exist_ok=True)
        (args.out / s / "depth").mkdir(parents=True, exist_ok=True)

    n = {"train": 0, "val": 0, ".": 0}
    n_skip = 0
    for rgb_path, idx in zip(rgbs, idxs):
        dpath = args.raw / f"{args.depth_src}_{idx}.npy"
        if not dpath.is_file():
            n_skip += 1
            continue
        split = "." if args.val_frac <= 0 else ("val" if idx in val_set else "train")
        Image.open(rgb_path).convert("RGB").save(
            args.out / split / "images" / f"{idx}.png")
        d = np.load(dpath).astype(np.float32)
        finite = np.isfinite(d) & (d < 1e6)
        if finite.any():
            d = np.where(finite, d, d[finite].max())   # background -> farthest
            lo, hi = d.min(), d.max()
            norm = (d - lo) / (hi - lo + 1e-6)
            if not args.no_invert:
                norm = 1.0 - norm                       # near = bright
            img = (norm * 255).astype(np.uint8)
        else:
            img = np.zeros(d.shape, np.uint8)
        Image.fromarray(img, mode="L").save(args.out / split / "depth" / f"{idx}.png")
        n[split] += 1
        if sum(n.values()) % 250 == 0:
            print(f"\r  {sum(n.values())}/{len(rgbs)}", end="", flush=True)
    print(f"\n[gan-data] -> {args.out}  train={n['train']} val={n['val']} "
          f"flat={n['.']}  (skipped {n_skip} missing-depth)")


if __name__ == "__main__":
    main()
