"""Split ward_v1/test/ into real_train/ and real_test/ with multi-label
stratification so every (sufficiently-represented) class lands in both splits.

Iterative-stratification implementation follows Sechidis et al. 2011:
process samples sorted by rarest contained class first; for each sample,
assign to the split with the largest remaining quota for the rarest label.

Output (under ward_v1/):
    real_train/{images,labels}/  ~70% of test images (~510 / 728)
    real_test/{images,labels}/   ~30% of test images (~218 / 728)
    real_train/_annotations.coco.json
    real_test/_annotations.coco.json

Images and YOLO .txt labels are symlinked (no duplication).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


def stratified_split(image_label_sets: dict[int, set[int]],
                     test_frac: float,
                     seed: int) -> tuple[set[int], set[int]]:
    """Iterative stratification. Returns (train_ids, test_ids)."""
    rng = random.Random(seed)
    # Per-class targets for the test split.
    cls_counts = Counter()
    for labels in image_label_sets.values():
        for c in labels:
            cls_counts[c] += 1
    test_target = {c: cls_counts[c] * test_frac for c in cls_counts}
    test_remaining = dict(test_target)
    train_remaining = {c: cls_counts[c] - test_target[c] for c in cls_counts}

    # Sort images by their rarest label (rare-first); ties broken randomly.
    img_order = sorted(
        image_label_sets.keys(),
        key=lambda i: (min((cls_counts[c] for c in image_label_sets[i]), default=10**9),
                       rng.random()),
    )

    train_ids, test_ids = set(), set()
    for img_id in img_order:
        labels = image_label_sets[img_id]
        if not labels:
            # Background-only image: assign by overall ratio.
            (test_ids if rng.random() < test_frac else train_ids).add(img_id)
            continue
        # For each split, compute "desire" = sum over sample's labels of
        # remaining quota in that split. Pick the higher-desire split.
        test_desire  = sum(max(test_remaining[c], 0.0)  for c in labels)
        train_desire = sum(max(train_remaining[c], 0.0) for c in labels)
        if test_desire > train_desire or (test_desire == train_desire and rng.random() < test_frac):
            test_ids.add(img_id)
            for c in labels:
                test_remaining[c] -= 1
        else:
            train_ids.add(img_id)
            for c in labels:
                train_remaining[c] -= 1

    # Hard guarantee: every class with >=2 total instances appears in test.
    # If a class ended up with 0 test instances, move one image containing
    # that class from train to test (prefer an image that contains only
    # already-covered classes to minimize collateral movement).
    test_cls_counts = Counter()
    for i in test_ids:
        for c in image_label_sets[i]:
            test_cls_counts[c] += 1
    for c in cls_counts:
        if cls_counts[c] >= 2 and test_cls_counts[c] == 0:
            candidates = [i for i in train_ids if c in image_label_sets[i]]
            if not candidates:
                continue
            # Prefer the candidate whose other labels already have the most
            # coverage in test (so moving it doesn't deplete train of rare classes).
            candidates.sort(
                key=lambda i: -sum(test_cls_counts[cc] for cc in image_label_sets[i] if cc != c)
            )
            mover = candidates[0]
            train_ids.discard(mover); test_ids.add(mover)
            for cc in image_label_sets[mover]:
                test_cls_counts[cc] += 1
    return train_ids, test_ids


def subset_coco(coco: dict, keep_image_ids: set[int]) -> dict:
    out = {k: v for k, v in coco.items() if k not in ("images", "annotations")}
    out["images"] = [im for im in coco["images"] if int(im["id"]) in keep_image_ids]
    out["annotations"] = [a for a in coco["annotations"]
                          if int(a["image_id"]) in keep_image_ids]
    return out


def materialize_split(name: str, image_ids: set[int],
                      coco: dict, src_root: Path, dst_root: Path,
                      cat_id_to_name: dict, src_split: str = "test") -> dict:
    split_dir = dst_root / name
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Symlink per-file (real dir, file-level links — works around ultralytics
    # symlink-resolution edge cases we hit with directory-level symlinks).
    src_imgs = src_root / src_split / "images"
    src_lbls = src_root / src_split / "labels"
    n_img = n_lbl_found = n_lbl_missing = 0
    for im in coco["images"]:
        if int(im["id"]) not in image_ids:
            continue
        fname = Path(im["file_name"]).name
        stem = Path(fname).stem
        src_img = src_imgs / fname
        if src_img.is_file():
            dst_img = img_dir / fname
            if dst_img.is_symlink() or dst_img.exists():
                dst_img.unlink()
            dst_img.symlink_to(src_img.resolve())
            n_img += 1
        src_lbl = src_lbls / f"{stem}.txt"
        if src_lbl.is_file():
            dst_lbl = lbl_dir / f"{stem}.txt"
            if dst_lbl.is_symlink() or dst_lbl.exists():
                dst_lbl.unlink()
            dst_lbl.symlink_to(src_lbl.resolve())
            n_lbl_found += 1
        else:
            n_lbl_missing += 1

    sub = subset_coco(coco, image_ids)
    (split_dir / "_annotations.coco.json").write_text(json.dumps(sub))

    # Per-class instance counts in this split
    per_class = Counter()
    for a in sub["annotations"]:
        per_class[int(a["category_id"])] += 1
    return {
        "name": name, "n_images": n_img, "n_labels": n_lbl_found,
        "n_labels_missing": n_lbl_missing, "n_anns": len(sub["annotations"]),
        "per_class": per_class, "cat_id_to_name": cat_id_to_name,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("ward_v1"),
                   help="Dataset root (containing the source split)")
    p.add_argument("--src-split", default="test",
                   help="Source split under --data to divide (default: test)")
    p.add_argument("--names", default="real_train,real_test",
                   help="Output split names '<keep>,<heldout>'; the second "
                        "receives --test-frac of the images.")
    p.add_argument("--test-frac", type=float, default=0.30,
                   help="Fraction of source images placed in the second "
                        "(held-out) split")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    name_keep, name_held = [s.strip() for s in args.names.split(",")]
    root = args.data.resolve()
    coco_path = root / args.src_split / "_annotations.coco.json"
    coco = json.loads(coco_path.read_text())

    image_label_sets = defaultdict(set)
    for im in coco["images"]:
        image_label_sets[int(im["id"])]  # init empty
    for a in coco["annotations"]:
        image_label_sets[int(a["image_id"])].add(int(a["category_id"]))
    image_label_sets = dict(image_label_sets)

    train_ids, test_ids = stratified_split(image_label_sets, args.test_frac, args.seed)
    print(f"split: {len(train_ids)} {name_keep}  |  {len(test_ids)} {name_held}  "
          f"(target test_frac={args.test_frac:.2f}, actual={len(test_ids)/len(image_label_sets):.3f})")

    cat_id_to_name = {int(c["id"]): c["name"] for c in coco["categories"]}

    stats = []
    for name, ids in ((name_keep, train_ids), (name_held, test_ids)):
        s = materialize_split(name, ids, coco, root, root, cat_id_to_name, args.src_split)
        stats.append(s)
        print(f"  {name}: images={s['n_images']}, labels={s['n_labels']} "
              f"(missing={s['n_labels_missing']}), annotations={s['n_anns']}")

    # Class balance sanity check
    print("\nper-class instance counts (cat_id  name             train  test):")
    train_pc = stats[0]["per_class"]; test_pc = stats[1]["per_class"]
    all_cats = sorted(set(train_pc) | set(test_pc))
    zero_in_test = []
    for cid in all_cats:
        name = cat_id_to_name.get(cid, "?")
        t = train_pc.get(cid, 0); v = test_pc.get(cid, 0)
        flag = "  ⚠ zero-in-test" if v == 0 and t > 0 else ""
        if v == 0 and t > 0:
            zero_in_test.append(name)
        print(f"  {cid:3d}  {name:22s} {t:6d} {v:6d}{flag}")
    if zero_in_test:
        print(f"\nWARN: {len(zero_in_test)} classes ended up with 0 test instances: {zero_in_test}")


if __name__ == "__main__":
    main()
