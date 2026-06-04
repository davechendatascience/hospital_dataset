"""
PEFT fine-tune YOLO-World on stylized sim + real ward photos.

Strategy:
  - PEFT lever: partial freezing (--freeze N freezes first N model modules).
    Default 20 leaves the neck + head + text-fusion trainable, backbone
    frozen. The CLIP text features themselves are not retrained.
  - Domain-balanced batches: real_train images are symlink-replicated to
    match the sim count, so the random sampler gives ~50/50 sim/real per
    batch (same effect as a WeightedRandomSampler without subclassing
    Ultralytics internals).
  - Per-epoch eval on the held-out real_test split via on_fit_epoch_end
    callback, written to runs/.../test_metrics.csv alongside results.csv.

Run:
    /home/edge-host/Documents/.venv/bin/python train_yolo_world.py \\
        --data ward_v1 \\
        --weights yolov8x-worldv2.pt \\
        --epochs 30 --imgsz 1024 --batch 8 --freeze 20 \\
        --project /home/edge-host/Documents/GitHub/hospital_dataset/runs/world \\
        --name peft
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ROS2_bridge" / "src"))
from fixed_categories import FIXED_CATEGORIES  # noqa: E402

# Natural-language prompt overrides for CLIP text encoder
# (mirrors predict_yolo_world.py; kept inline so this script is self-contained).
_PROMPT_OVERRIDES = {
    "TV":                       "television",
    "iv_pole":                  "IV pole",
    "tv":                       "television",
    "hospital_bed":             "hospital bed",
    "bedside_monitor":          "bedside patient monitor",
    "bedside_table":            "bedside table",
    "overbed_table":            "overbed table",
    "air_vent":                 "air vent",
    "alcohol_spray_bottle":     "alcohol spray bottle",
    "bed_curtain":              "bed privacy curtain",
    "companion_chair":          "companion chair",
    "ear_thermometer":          "ear thermometer",
    "gas_manifold":             "medical gas wall manifold",
    "medical_gloves":           "medical gloves",
    "medical_package":          "medical package",
    "medical_waste_container":  "medical waste container",
    "oxygen_flowmeter":         "oxygen flowmeter",
    "remote_control":           "remote control",
    "soiled_linen_bin":         "soiled linen bin",
    "tissue_dispenser":         "tissue dispenser",
    "toilet_handle":            "toilet handle",
    "toilet":                   "toilet",
    "weight_scale":             "weight scale",
    "waste_bin":                "waste bin",
    "door_handle":              "door handle",
    "light_switch":             "light switch",
    "suction_jar":              "suction jar",
    "suction_knob":             "suction knob",
    "ward_object":              "ward_object",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("ward_v1"),
                   help="Original dataset root (with real_train/, real_test/ "
                        "produced by split_real_test.py)")
    p.add_argument("--sim", type=Path,
                   default=Path("ward_v1_styled_flat"),
                   help="Stylized sim dataset (flat layout with train/, valid/)")
    p.add_argument("--mix-dir", type=Path,
                   default=Path("ward_v1_mixed_flat"),
                   help="Output dir for the mixed (sim+real) training layout")
    p.add_argument("--weights", default="yolov8x-worldv2.pt",
                   help="YOLO-World checkpoint")
    p.add_argument("--freeze", type=int, default=20,
                   help="Freeze the first N model modules. For yolov8x-world "
                        "this leaves the head + last C2f blocks trainable.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz",  type=int, default=1024)
    p.add_argument("--batch",  type=int, default=8)
    p.add_argument("--lr0",    type=float, default=5e-4,
                   help="Initial LR. PEFT on YOLO-World wants 1e-4..5e-4.")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="runs/world")
    p.add_argument("--name",    default=None)
    p.add_argument("--rebuild-mix", action="store_true",
                   help="Rebuild the mixed dataset dir from scratch")
    p.add_argument("--skip-train", action="store_true",
                   help="Only build the mixed dataset + data.yaml, don't train")
    return p.parse_args()


def build_mixed_layout(mix_dir: Path, sim_dir: Path, data_dir: Path,
                       force: bool) -> tuple[int, int, int, int]:
    """Build a Ultralytics-flat dataset that combines:
       train/ = stylized sim train + real_train (oversampled to ~50/50)
       valid/ = stylized sim valid (unchanged early-stop signal)
       test/  = real_test held-out
    Uses real dirs + file-level symlinks (avoids the dir-symlink edge case
    we hit before with Ultralytics' path resolution).

    Returns (n_sim_train, n_real_train_unique, n_real_train_replicated, oversample_factor).
    """
    sim_train = sim_dir / "train" / "images"
    sim_valid = sim_dir / "valid" / "images"
    sim_train_lbl = sim_dir / "train" / "labels"
    sim_valid_lbl = sim_dir / "valid" / "labels"
    real_train_img = data_dir / "real_train" / "images"
    real_train_lbl = data_dir / "real_train" / "labels"
    real_test_img  = data_dir / "real_test"  / "images"
    real_test_lbl  = data_dir / "real_test"  / "labels"

    for p in (sim_train, sim_valid, real_train_img, real_test_img):
        if not p.is_dir():
            sys.exit(f"missing required input dir: {p}")

    if force and mix_dir.exists():
        shutil.rmtree(mix_dir)

    # train: sim + oversampled real
    train_img_dst = mix_dir / "train" / "images"
    train_lbl_dst = mix_dir / "train" / "labels"
    train_img_dst.mkdir(parents=True, exist_ok=True)
    train_lbl_dst.mkdir(parents=True, exist_ok=True)

    # Symlink sim train images & labels (preserve original basenames).
    sim_imgs = sorted([*sim_train.glob("*.png"), *sim_train.glob("*.jpg")])
    for f in sim_imgs:
        dst = train_img_dst / f.name
        if not dst.exists():
            dst.symlink_to(f.resolve())
        lbl = sim_train_lbl / (f.stem + ".txt")
        if lbl.is_file():
            dst_lbl = train_lbl_dst / lbl.name
            if not dst_lbl.exists():
                dst_lbl.symlink_to(lbl.resolve())

    # Oversample real_train so total real ≈ total sim. The on-disk replication
    # is functionally identical to a WeightedRandomSampler — each replica is a
    # distinct sample with its own random augmentation, so the model sees
    # different pixels per epoch despite the same underlying image.
    real_imgs = sorted([*real_train_img.glob("*.png"), *real_train_img.glob("*.jpg")])
    n_sim = len(sim_imgs)
    n_real_unique = len(real_imgs)
    factor = max(1, round(n_sim / max(n_real_unique, 1)))
    n_real_replicated = 0
    for f in real_imgs:
        lbl = real_train_lbl / (f.stem + ".txt")
        if not lbl.is_file():
            continue
        for k in range(factor):
            img_name = f"real_{f.stem}_rep{k:02d}{f.suffix}"
            lbl_name = f"real_{f.stem}_rep{k:02d}.txt"
            dst_img = train_img_dst / img_name
            dst_lbl = train_lbl_dst / lbl_name
            if not dst_img.exists():
                dst_img.symlink_to(f.resolve())
            if not dst_lbl.exists():
                dst_lbl.symlink_to(lbl.resolve())
            n_real_replicated += 1

    # valid: stylized sim valid (file-level symlinks)
    val_img_dst = mix_dir / "valid" / "images"
    val_lbl_dst = mix_dir / "valid" / "labels"
    val_img_dst.mkdir(parents=True, exist_ok=True)
    val_lbl_dst.mkdir(parents=True, exist_ok=True)
    for f in sorted([*sim_valid.glob("*.png"), *sim_valid.glob("*.jpg")]):
        dst = val_img_dst / f.name
        if not dst.exists():
            dst.symlink_to(f.resolve())
        lbl = sim_valid_lbl / (f.stem + ".txt")
        if lbl.is_file():
            dst_lbl = val_lbl_dst / lbl.name
            if not dst_lbl.exists():
                dst_lbl.symlink_to(lbl.resolve())

    # test: real_test held-out
    test_img_dst = mix_dir / "test" / "images"
    test_lbl_dst = mix_dir / "test" / "labels"
    test_img_dst.mkdir(parents=True, exist_ok=True)
    test_lbl_dst.mkdir(parents=True, exist_ok=True)
    for f in sorted([*real_test_img.glob("*.png"), *real_test_img.glob("*.jpg")]):
        dst = test_img_dst / f.name
        if not dst.exists():
            dst.symlink_to(f.resolve())
        lbl = real_test_lbl / (f.stem + ".txt")
        if lbl.is_file():
            dst_lbl = test_lbl_dst / lbl.name
            if not dst_lbl.exists():
                dst_lbl.symlink_to(lbl.resolve())

    return n_sim, n_real_unique, n_real_replicated, factor


def write_data_yaml(mix_dir: Path, class_names: list[str]) -> Path:
    p = mix_dir / "data.yaml"
    lines = [
        f"path: {mix_dir.resolve()}",
        "train: train/images",
        "val:   valid/images",
        "test:  test/images",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")
    p.write_text("\n".join(lines) + "\n")
    return p


def main() -> None:
    args = parse_args()
    data_dir = args.data.expanduser().resolve()
    sim_dir  = args.sim.expanduser().resolve()
    mix_dir  = args.mix_dir.expanduser().resolve()
    project  = Path(args.project).expanduser().resolve()  # absolute, bypasses ultralytics settings.yaml runs_dir

    # 1) Class set & prompts (same convention as predict_yolo_world.py)
    sorted_cats = sorted(
        ((name, cid) for name, cid in FIXED_CATEGORIES.items() if cid != 0),
        key=lambda nc: nc[1],
    )
    class_names = [name for name, _ in sorted_cats]
    prompts = [_PROMPT_OVERRIDES.get(n, n.replace("_", " ")) for n in class_names]
    print(f"[world-peft] {len(class_names)} classes; sample prompts: {prompts[:5]}")

    # 2) Build mixed dataset layout
    n_sim, n_real_u, n_real_rep, factor = build_mixed_layout(
        mix_dir, sim_dir, data_dir, force=args.rebuild_mix
    )
    print(f"[world-peft] mixed layout @ {mix_dir}")
    print(f"  sim train images:       {n_sim}")
    print(f"  real_train unique:      {n_real_u}  (replicated {factor}× → {n_real_rep})")
    print(f"  effective sim:real:     {n_sim}:{n_real_rep}  "
          f"(={n_sim/(n_sim+n_real_rep):.2f}:{n_real_rep/(n_sim+n_real_rep):.2f})")

    yaml_path = write_data_yaml(mix_dir, class_names)
    print(f"[world-peft] data.yaml → {yaml_path}")

    if args.skip_train:
        print("[world-peft] --skip-train; stopping.")
        return

    # 3) Load YOLO-World, set our class prompts, train with freeze=N
    from ultralytics import YOLOWorld
    model = YOLOWorld(args.weights)
    model.set_classes(prompts)
    print(f"[world-peft] model={args.weights}, freeze first {args.freeze} modules")

    # 4) Per-epoch held-out real_test eval (mirrors train_yolo.py callback)
    test_rows: list[dict] = []

    def _eval_real_test(trainer) -> None:
        epoch = int(trainer.epoch) + 1
        last_pt = Path(trainer.save_dir) / "weights" / "last.pt"
        if not last_pt.is_file():
            return
        eval_model = YOLOWorld(str(last_pt))
        eval_model.set_classes(prompts)
        metrics = eval_model.val(
            data=str(yaml_path), split="test",
            imgsz=args.imgsz, batch=args.batch, device=args.device,
            workers=args.workers, verbose=False, plots=False, save_json=False,
        )
        row = {
            "epoch":     epoch,
            "box_map":   float(metrics.box.map),
            "box_map50": float(metrics.box.map50),
            "box_map75": float(metrics.box.map75),
        }
        # dedupe Ultralytics' end-of-training extra fire
        if test_rows and test_rows[-1]["epoch"] == epoch:
            return
        test_rows.append(row)
        csv_path = Path(trainer.save_dir) / "test_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
            w.writeheader()
            for r in test_rows:
                w.writerow(r)
        print(f"[real_test@epoch{epoch}] "
              + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"))

    model.add_callback("on_fit_epoch_end", _eval_real_test)

    # 5) Train. Augmentation: photometric stronger than train_yolo.py because
    # the real domain has more lighting/color variation than even the styled sim.
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.name,
        freeze=args.freeze,
        lr0=args.lr0,
        # Photometric aug (heavier than vanilla) — mostly hits the real subset
        # which lives at the rare end of the sampler distribution.
        hsv_h=0.03, hsv_s=0.6, hsv_v=0.4,
        translate=0.1, scale=0.4, fliplr=0.5,
        mosaic=1.0, mixup=0.0, copy_paste=0.0,
        plots=True,
    )


if __name__ == "__main__":
    main()
