"""Build extra input channels (depth, seg) for a dataset's splits, so CUT can
be conditioned on structure.

Depth: Depth-Anything-V2 monocular estimate from the RGB (no Isaac re-render;
works identically on sim and real, which matters since CUT trains on both
domains). Seg: rasterized from the split's COCO masks (one colour per class).

Writes <split>/depth/<stem>.png (8-bit grayscale) and, with --seg,
<split>/seg/<stem>.png (RGB). Leaves images/ and labels untouched.

    /home/edge-host/Documents/.venv/bin/python build_channels.py \
        --data ward_v1 --splits train,valid,test1,test2 --depth --seg
"""
from __future__ import annotations

import argparse
import colorsys
import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def list_images(split_dir: Path):
    cand = split_dir / "images" if (split_dir / "images").is_dir() else split_dir
    out = []
    for e in ("png", "jpg", "jpeg", "bmp", "webp"):
        out += glob.glob(str(cand / f"*.{e}")) + glob.glob(str(cand / f"*.{e.upper()}"))
    return sorted(set(out))


def palette(n=64):
    pal = np.zeros((n, 3), np.uint8)
    for i in range(1, n):
        r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.65, 0.95)
        pal[i] = (int(r * 255), int(g * 255), int(b * 255))
    return pal


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("ward_v1"))
    ap.add_argument("--splits", default="train,valid,test1,test2")
    ap.add_argument("--depth", action="store_true", default=True)
    ap.add_argument("--no-depth", dest="depth", action="store_false")
    ap.add_argument("--seg", action="store_true", default=False)
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    dev = (f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu"
           else "cpu")
    dev_idx = int(args.device) if dev.startswith("cuda") else -1

    depth_pipe = None
    if args.depth:
        from transformers import pipeline
        print(f"[ch] depth estimator: {args.depth_model}")
        depth_pipe = pipeline("depth-estimation", model=args.depth_model, device=dev_idx)

    coco_cache = {}
    pal = palette()

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        sdir = args.data / split
        files = list_images(sdir)
        if not files:
            print(f"[ch] {split}: no images, skipping"); continue
        if args.depth:
            (sdir / "depth").mkdir(exist_ok=True)
        if args.seg:
            (sdir / "seg").mkdir(exist_ok=True)
            from pycocotools.coco import COCO
            coco = COCO(str(sdir / "_annotations.coco.json"))
            by_stem = {Path(v["file_name"]).stem: k for k, v in coco.imgs.items()}
        print(f"[ch] {split}: {len(files)} images "
              f"(depth={args.depth} seg={args.seg})", flush=True)

        # depth in batches
        if args.depth:
            for i in range(0, len(files), args.batch):
                chunk = files[i:i + args.batch]
                imgs = [Image.open(f).convert("RGB") for f in chunk]
                res = depth_pipe(imgs)
                res = res if isinstance(res, list) else [res]
                for f, r in zip(chunk, res):
                    r["depth"].save(sdir / "depth" / (Path(f).stem + ".png"))
                if (i + args.batch) % 256 < args.batch:
                    print(f"\r  depth {min(i+args.batch,len(files))}/{len(files)}",
                          end="", flush=True)
            print()
        # seg from labels
        if args.seg:
            for f in files:
                stem = Path(f).stem
                w, h = Image.open(f).size
                seg = np.zeros((h, w, 3), np.uint8)
                img_id = by_stem.get(stem)
                if img_id is not None:
                    for a in coco.imgToAnns.get(img_id, []):
                        if a["category_id"] == 0:
                            continue
                        m = coco.annToMask(a)
                        seg[m > 0] = pal[a["category_id"] % len(pal)]
                Image.fromarray(seg).save(sdir / "seg" / (stem + ".png"))
        print(f"[ch] {split}: done")
    print("[ch] all splits done")


if __name__ == "__main__":
    main()
