"""
Train RT-DETR (transformer detector with a ViT-style encoder) on the sim
splits of the ward_v1 dataset.

This script:
  1. Reads the existing YOLO instance-seg labels (polygons) under
     ward_v1/{train,valid}/labels and derives bounding-box labels
     into a parallel directory ward_v1/{train,valid}/labels_bbox.
  2. Writes a detection-only data.yaml at ward_v1/data_rtdetr.yaml that
     points Ultralytics at those bbox labels.
  3. Kicks off `ultralytics.RTDETR.train()` with sim-to-real augs.

Run with the project venv python:

    /home/edge-host/Documents/.venv/bin/python train_rtdetr.py \
        --data ward_v1 --model rtdetr-l.pt \
        --epochs 50 --imgsz 640 --batch 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ROS2_bridge" / "src"))
from fixed_categories import FIXED_CATEGORIES  # type: ignore  # noqa: E402

SIM_SPLITS = ("train", "valid")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True,
                   help="Dataset root containing train/, valid/, real_test/")
    p.add_argument("--model", default="rtdetr-l.pt",
                   help="Ultralytics RT-DETR checkpoint (rtdetr-l.pt or rtdetr-x.pt)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640,
                   help="RT-DETR official input is 640; raise if VRAM allows")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="rtdetr_sim")
    p.add_argument("--save-period", type=int, default=1,
                   help="Save a checkpoint every N epochs (default: 1 -> every "
                        "epoch). Ultralytics writes epoch{N}.pt alongside "
                        "last.pt/best.pt under weights/.")
    p.add_argument("--eval-split", default="real_test",
                   help="Split under <data>/ to run per-epoch eval on, in "
                        "addition to the standard val-split eval. Set to "
                        "'none' to disable.")
    p.add_argument("--rebuild-labels", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args()


def polygons_to_bbox_labels(src_dir: Path, dst_dir: Path,
                            force: bool = False) -> tuple[int, int]:
    """Read src_dir/*.txt (YOLO polygon format) and write bbox-format labels
    to dst_dir/*.txt. Each polygon row `cls x1 y1 x2 y2 ...` becomes one bbox
    row `cls cx cy w h` (normalized). Returns (files_written, polys_dropped).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    dropped = 0
    for poly_file in sorted(src_dir.glob("*.txt")):
        out_file = dst_dir / poly_file.name
        if out_file.exists() and not force:
            continue
        out_lines: list[str] = []
        for raw in poly_file.read_text().splitlines():
            parts = raw.strip().split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                dropped += 1
                continue
            if len(coords) < 6 or len(coords) % 2 != 0:
                dropped += 1
                continue
            xs = coords[0::2]
            ys = coords[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w = max(0.0, x_max - x_min)
            h = max(0.0, y_max - y_min)
            if w < 1e-6 or h < 1e-6:
                dropped += 1
                continue
            cx = x_min + w / 2.0
            cy = y_min + h / 2.0
            out_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        out_file.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
        written += 1
    return written, dropped


def swap_labels_to_bbox(split_dir: Path) -> None:
    """Back up split_dir/labels -> labels_seg and symlink labels -> labels_bbox.

    Idempotent: skips if labels is already the right symlink.
    """
    live = split_dir / "labels"
    bbox_dir = split_dir / "labels_bbox"
    backup = split_dir / "labels_seg"
    if not bbox_dir.is_dir():
        return
    if live.is_symlink():
        target = live.resolve()
        if target != bbox_dir.resolve():
            live.unlink()
            live.symlink_to(bbox_dir.resolve())
        return
    if live.is_dir():
        if not backup.exists():
            live.rename(backup)
        else:
            import shutil as _sh
            _sh.rmtree(live)
        live.symlink_to(bbox_dir.resolve())


def write_data_yaml(data_root: Path, yolo_classes: list[str],
                    eval_split: str | None) -> Path:
    """Write a detection data.yaml. If eval_split is provided, includes a
    `test:` entry so Ultralytics' val(split='test') hits it per epoch."""
    p = data_root / "data_rtdetr.yaml"
    lines = [
        f"path: {data_root.resolve()}",
        "train: train/images",
        "val:   valid/images",
    ]
    if eval_split:
        lines.append(f"test:  {eval_split}/images")
    lines += ["", f"nc: {len(yolo_classes)}", "names:"]
    for i, name in enumerate(yolo_classes):
        lines.append(f"  {i}: {name}")
    p.write_text("\n".join(lines) + "\n")
    return p


def main() -> None:
    args = parse_args()
    data_root = args.data.expanduser().resolve()
    if not data_root.is_dir():
        sys.exit(f"--data {data_root} does not exist")

    sorted_cats = sorted(
        ((name, cid) for name, cid in FIXED_CATEGORIES.items() if cid != 0),
        key=lambda nc: nc[1],
    )
    yolo_classes = [name for name, _ in sorted_cats]
    print(f"[rtdetr] {len(yolo_classes)} classes")

    # Splits to prepare = sim splits + optional eval split (e.g. real_test)
    eval_split = None if args.eval_split.lower() == "none" else args.eval_split
    splits_to_prep = list(SIM_SPLITS)
    if eval_split and eval_split not in splits_to_prep:
        splits_to_prep.append(eval_split)

    for split in splits_to_prep:
        split_dir = data_root / split
        if not (split_dir / "images").is_dir():
            if split == eval_split:
                print(f"[rtdetr] eval split {split} has no images/ — disabling "
                      f"per-epoch eval on it")
                eval_split = None
                continue
            sys.exit(f"[rtdetr] missing {split_dir}/images — run train_yolo.py "
                     f"first to lay out the dataset")
        poly_src = split_dir / "labels"
        if poly_src.is_symlink():
            poly_src = split_dir / "labels_seg"
        if not poly_src.is_dir():
            if split == eval_split:
                print(f"[rtdetr] eval split {split} has no polygon labels — "
                      f"disabling per-epoch eval on it")
                eval_split = None
                continue
            sys.exit(f"[rtdetr] missing polygon labels at {split_dir}/labels "
                     f"or {split_dir}/labels_seg — run train_yolo.py first")
        n_w, n_d = polygons_to_bbox_labels(
            poly_src, split_dir / "labels_bbox", force=args.rebuild_labels,
        )
        swap_labels_to_bbox(split_dir)
        # Stale cache from polygon-era runs must be removed; Ultralytics will
        # rebuild from the bbox labels on next access.
        cache = split_dir / "labels.cache"
        if cache.exists():
            cache.unlink()
        print(f"[rtdetr] {split}: {n_w} bbox label files written, {n_d} polys dropped")

    yaml_path = write_data_yaml(data_root, yolo_classes, eval_split)
    print(f"[rtdetr] wrote data spec: {yaml_path}"
          + (f" (eval split: {eval_split})" if eval_split else ""))

    if args.skip_train:
        return

    print(f"[rtdetr] starting training: model={args.model}, epochs={args.epochs}, "
          f"imgsz={args.imgsz}, batch={args.batch}, device={args.device}, "
          f"save_period={args.save_period}")
    from ultralytics import RTDETR
    model = RTDETR(args.model)

    # Per-epoch eval on the held-out split, mirroring train_yolo.py. We hook
    # on_fit_epoch_end (fires AFTER Ultralytics writes last.pt), load last.pt
    # into a fresh RTDETR wrapper, and run val() with split='test'. Using
    # last.pt rather than poking trainer.model avoids any interaction with
    # the live EMA / autograd state.
    test_metrics_rows: list[dict] = []

    def _eval_test_split(trainer) -> None:
        if not eval_split:
            return
        epoch = int(trainer.epoch) + 1
        last_pt = Path(trainer.save_dir) / "weights" / "last.pt"
        if not last_pt.is_file():
            return
        eval_model = RTDETR(str(last_pt))
        metrics = eval_model.val(
            data=str(yaml_path),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            verbose=False,
            plots=False,
            save_json=False,
        )
        row = {
            "epoch":     epoch,
            "box_map":   float(metrics.box.map),
            "box_map50": float(metrics.box.map50),
            "box_map75": float(metrics.box.map75),
        }
        test_metrics_rows.append(row)
        csv_path = Path(trainer.save_dir) / f"{eval_split}_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(test_metrics_rows[0].keys()))
            w.writeheader()
            for r in test_metrics_rows:
                w.writerow(r)
        print(f"[{eval_split}@epoch{epoch:d}] "
              + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"))

    if eval_split:
        model.add_callback("on_fit_epoch_end", _eval_test_split)

    model.train(
        save_period=args.save_period,
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        # Sim-to-real friendly augs: colour + mild geometry; mosaic is helpful
        # for small-object recall but RT-DETR is sensitive to extreme scale
        # jitter, so we keep scale moderate.
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        translate=0.1, scale=0.4, fliplr=0.5,
        mosaic=1.0, mixup=0.0,
        plots=True,
    )


if __name__ == "__main__":
    main()
