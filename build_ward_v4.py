"""Assemble ward_v4: the Cosmos-styled ward_v3 train frames + carried-over labels.

Takes every styled frame currently in cosmos_jobs/outputs/ (the resumable batch
keeps adding more -- re-run this script anytime to pick up new ones), pairs it
with its ward_v3 annotations (same stem; Cosmos outputs 1920x1080 like the sim
input and the controls pin object positions, so the labels stay valid), and
writes a self-contained split:

    ward_v4/train/images/<stem>.jpg
    ward_v4/train/_annotations.coco.json      (file_name -> .jpg, same W/H)

Train with the existing trainer (real holdout as the eval domain):
    .venv/bin/python train_yolo_da.py --train-dir ward_v4/train \
        --val-dir ward_v3/test --task detect --model yolo11s.pt --epochs 50
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--styled-dir", type=Path, default=PROJECT / "cosmos_jobs/outputs",
                    help="Cosmos batch output dir (top-level <stem>.jpg files).")
    ap.add_argument("--source-split", type=Path, default=PROJECT / "ward_v3/train",
                    help="Split whose COCO annotations match the styled stems.")
    ap.add_argument("--out", type=Path, default=PROJECT / "ward_v4/train")
    ap.add_argument("--link", action="store_true",
                    help="Symlink images instead of copying.")
    args = ap.parse_args()

    styled = {p.stem: p for p in args.styled_dir.glob("*.jpg")
              if "_control_" not in p.name and "_mask_" not in p.name}
    if not styled:
        raise SystemExit(f"no styled frames in {args.styled_dir}")

    coco = json.loads((args.source_split / "_annotations.coco.json").read_text())
    keep_imgs, keep_ids = [], set()
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        if stem in styled:
            im = dict(im)
            im["file_name"] = f"{stem}.jpg"
            keep_imgs.append(im)
            keep_ids.add(im["id"])
    keep_anns = [a for a in coco["annotations"] if a["image_id"] in keep_ids]

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for f in img_dir.glob("*.jpg"):
        f.unlink()
    for im in keep_imgs:
        stem = Path(im["file_name"]).stem
        dst = img_dir / im["file_name"]
        if args.link:
            dst.symlink_to(styled[stem].resolve())
        else:
            shutil.copy2(styled[stem], dst)

    out_coco = {"images": keep_imgs, "annotations": keep_anns,
                "categories": coco["categories"]}
    (args.out / "_annotations.coco.json").write_text(json.dumps(out_coco))
    print(f"[ward_v4] {len(keep_imgs)} styled frames "
          f"({len(styled) - len(keep_imgs)} styled without annotations skipped), "
          f"{len(keep_anns)} annotations -> {args.out}")


if __name__ == "__main__":
    main()
