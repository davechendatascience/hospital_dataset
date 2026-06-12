"""Overlay GT instance masks + class names directly on BasicWriter output.

For every rgb_<idx>.png under a replicator _raw dir, reads the sibling
instance_segmentation_<idx>.png (raw uint32 IDs) and
instance_segmentation_semantics_mapping_<idx>.json, normalizes each entry
onto fixed_categories.py (case fixes + LABEL_ALIASES) and writes
gt_<idx>.jpg NEXT TO the raw frame -- the immediate way to eyeball that the
authored stage labels came through the online generation correctly.

    .venv/bin/python overlay_raw_gt.py --raw ward_data/ward_dataset/_train_render/_raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "ROS2_bridge" / "src"))
from fixed_categories import class_from_entry  # noqa: E402


def overlay_frame(rgb_path: Path, dst: Path) -> int:
    idx = rgb_path.stem.split("_", 1)[1]
    inst_png = rgb_path.parent / f"instance_segmentation_{idx}.png"
    sem_map = rgb_path.parent / f"instance_segmentation_semantics_mapping_{idx}.json"
    if not inst_png.is_file() or not sem_map.is_file():
        return -1
    inst = np.asarray(Image.open(inst_png))
    if inst.ndim == 3:
        inst = inst[..., 0]
    mapping = json.loads(sem_map.read_text())

    base = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
    rng = np.random.RandomState(7)
    palette = rng.randint(60, 255, (64, 3))
    draw_jobs = []
    k = 0
    for inst_id_str, entry in mapping.items():
        try:
            inst_id = int(inst_id_str)
        except ValueError:
            continue
        cls = class_from_entry(entry)
        if cls is None:
            continue
        m = inst == inst_id
        if not m.any():
            continue
        color = palette[k % len(palette)]
        base[m] = 0.45 * base[m] + 0.55 * color
        ys, xs = np.nonzero(m)
        draw_jobs.append((int(xs.mean()), int(ys.mean()), cls, tuple(color)))
        k += 1
    vis = Image.fromarray(base.astype(np.uint8))
    d = ImageDraw.Draw(vis)
    for x, y, txt, color in draw_jobs:
        w = d.textlength(txt)
        d.rectangle([x - 2, y - 11, x + w + 2, y + 2], fill=(0, 0, 0))
        d.text((x, y - 10), txt, fill=color)
    vis.save(dst, quality=92)
    return len(draw_jobs)


def generate(raw_dir: Path, limit: int | None = None) -> int:
    rgbs = sorted(raw_dir.rglob("rgb_*.png"))
    if limit:
        rgbs = rgbs[:limit]
    done = 0
    for p in rgbs:
        idx = p.stem.split("_", 1)[1]
        n = overlay_frame(p, p.parent / f"gt_{idx}.jpg")
        if n >= 0:
            done += 1
            if done <= 5 or done % 200 == 0:
                print(f"[gt] {p.name}: {n} instances -> gt_{idx}.jpg", flush=True)
    print(f"[gt] {done}/{len(rgbs)} overlays written next to raw frames")
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True,
                    help="BasicWriter output dir (the _raw folder)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    generate(args.raw, args.limit)


if __name__ == "__main__":
    main()
