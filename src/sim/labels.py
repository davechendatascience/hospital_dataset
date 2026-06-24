"""Fix mislabeled Isaac assets in the ward_v3 COCO annotations (exact, per-prim).

The semantic name rules misclassified four asset groups (verified visually,
sim-vs-real contact sheets):

  * Sink_Mirror      sink         -> mirror        (it IS the mirror panel)
  * toilethandle*    toilet_handle-> door_handle   (door levers; the REAL set
                                                    uses toilet_handle for the
                                                    grab bars by the toilet)
  * bucket           waste_bin    -> REMOVE        (wall wire basket, not in
                                                    the taxonomy; poisoned the
                                                    waste_bin class)
  * access_sensor*   door_handle  -> REMOVE        (access keypads, not in the
                                                    taxonomy)

Existing annotations are corrected EXACTLY by replaying the _raw per-frame
instance-ID segmentation (color -> prim path): an annotation of a suspect
class is re-assigned/deleted when >50% of its mask pixels belong to a target
prim. Split stems map 1:1 to raw indices (rgb_frame_<idx> <-> *_<idx>).

    .venv/bin/python fix_isaac_labels.py --splits ward_v3/train,ward_v3/val
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

PROJECT = Path(__file__).resolve().parents[2]

# prim (2nd path segment under /World) -> new class name, or None = delete
TARGETS = {
    "Sink_Mirror": "mirror",
    "toilethandle": "door_handle",
    "toilethandle2": "door_handle",
    "bucket": None,
    "access_sensor": None,
    "access_sensor2": None,
    "access_sensor3": None,
}
# only annotations of these (current) classes are candidates
SUSPECT_CLASSES = {"sink", "toilet_handle", "waste_bin", "door_handle"}


def prim_key(path: str) -> str | None:
    parts = path.split("/")
    return parts[2] if len(parts) > 2 and parts[1] == "World" else None


def target_masks(raw: Path, idx: str):
    """{prim_name: bool mask} for target prims present in this frame."""
    seg_p = raw / f"instance_id_segmentation_{idx}.png"
    map_p = raw / f"instance_id_segmentation_mapping_{idx}.json"
    if not seg_p.is_file() or not map_p.is_file():
        return {}
    mapping = json.loads(map_p.read_text())
    colors = {}                                  # prim -> [rgba, ...]
    for color_s, prim_path in mapping.items():
        k = prim_key(str(prim_path))
        if k in TARGETS:
            colors.setdefault(k, []).append(ast.literal_eval(color_s))
    if not colors:
        return {}
    seg = np.array(Image.open(seg_p))            # (H, W, 4)
    out = {}
    for k, cols in colors.items():
        m = np.zeros(seg.shape[:2], bool)
        for c in cols:
            m |= (seg == np.array(c, np.uint8)).all(-1)
        if m.any():
            out[k] = m
    return out


def fix_split(split_dir: Path, raw: Path, apply: bool):
    ann_file = split_dir / "_annotations.coco.json"
    coco = json.loads(ann_file.read_text())
    name2id = {c["name"]: c["id"] for c in coco["categories"]}
    id2name = {v: k for k, v in name2id.items()}
    suspect_ids = {name2id[n] for n in SUSPECT_CLASSES if n in name2id}
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    stats, drop = {}, set()
    for im in coco["images"]:
        anns = [a for a in by_img.get(im["id"], [])
                if a["category_id"] in suspect_ids]
        if not anns:
            continue
        idx = Path(im["file_name"]).stem.split("_")[-1]
        tmasks = target_masks(raw, idx)
        if not tmasks:
            continue
        for a in anns:
            m = coco_mask.decode(a["segmentation"]).astype(bool)
            area = max(int(m.sum()), 1)
            for prim, tm in tmasks.items():
                if tm.shape != m.shape:
                    continue
                if int((m & tm).sum()) / area <= 0.5:
                    continue
                new = TARGETS[prim]
                old = id2name[a["category_id"]]
                key = (old, new or "REMOVED", prim)
                stats[key] = stats.get(key, 0) + 1
                if new is None:
                    drop.add(a["id"])
                else:
                    a["category_id"] = name2id[new]
                break

    if apply:
        backup = ann_file.with_suffix(".coco.json.bak")
        if not backup.exists():
            shutil.copy2(ann_file, backup)
        coco["annotations"] = [a for a in coco["annotations"]
                               if a["id"] not in drop]
        ann_file.write_text(json.dumps(coco))
    print(f"\n[{split_dir}] {'APPLIED' if apply else 'dry-run'}:")
    for (old, new, prim), n in sorted(stats.items()):
        print(f"  {old:14s} -> {new:12s} (prim {prim})  x{n}")
    print(f"  total changed={sum(stats.values())}, deleted={len(drop)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="ward_v3/train,ward_v3/val")
    ap.add_argument("--raw", type=Path, default=PROJECT / "ward_v3/_raw")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for s in args.splits.split(","):
        fix_split(PROJECT / s.strip(), args.raw, apply=not args.dry_run)


if __name__ == "__main__":
    main()
