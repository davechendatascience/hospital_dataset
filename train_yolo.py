"""
Domain-AGNOSTIC YOLO11 instance-segmentation training (single phase).

Reframe (per docs/sim2real_data_efficiency.pdf §10.5 and the discussion):
domain adaptation here is NOT "warp sim into real". The goal is a model whose
features are AGNOSTIC to the sim/real domain, so it works on EITHER. We get
there by:

  1. Joint SUPERVISED training on labelled sim (train) AND labelled real
     (real_dev) in one mixed dataloader -- the task head learns from both
     domains' labels (real_dev is oversampled so it isn't drowned by sim).
  2. A DOMAIN-INVARIANCE pressure on a backbone feature: a DANN
     gradient-reversal domain head (and/or an unbiased MMD term) computed
     from the SAME mixed batch, using per-image domain labels read from the
     file paths -- pushing the backbone to make sim and real indistinguishable.
  3. A final measurement on BOTH domains (sim valid + real holdout) so we can
     SEE whether the model is actually domain-agnostic, not just adapted.

real_holdout is never trained on or used for checkpoint selection -> it stays
an honest test of cross-domain generalization.

Two-line mental model:
    task loss  = seg/det loss on (sim + real_dev) labels        -> good on both
    domain loss= DANN/MMD making features sim/real-agnostic     -> generalizes

    .venv/bin/python train_yolo.py --data ward_data/ward_dataset_v3 \
        --model yolo11s-seg.pt --epochs 60 --imgsz 1024 --batch 16 \
        --dann --mmd --cls-prior --real-oversample 8 --name v3_domain_agnostic
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import cv2
from pycocotools import mask as coco_mask

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ROS2_bridge" / "src"))
from fixed_categories import FIXED_CATEGORIES  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True,
                   help="Dataset root (build_dataset.py layout) with the sim "
                        "and real splits as subdirs.")
    p.add_argument("--model", default="yolo11s-seg.pt",
                   help="Ultralytics seg checkpoint (yolo11n/s/m/l/x-seg). For "
                        "a small real set prefer n/s.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz",  type=int, default=1024)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="runs/segment")
    p.add_argument("--name", default=None)
    p.add_argument("--rebuild-labels", action="store_true",
                   help="Force re-conversion of COCO -> YOLO labels.")
    p.add_argument("--skip-train", action="store_true",
                   help="Only convert labels + write data.yaml.")
    # ---- which subdirs are sim vs real ----
    p.add_argument("--sim-train", default="train",
                   help="Labelled SIM split to train on.")
    p.add_argument("--real-train", default="real_dev",
                   help="Labelled REAL split to train on (jointly with sim).")
    p.add_argument("--sim-val", default="valid",
                   help="SIM split for per-epoch validation / selection.")
    p.add_argument("--real-holdout", default="real_holdout",
                   help="REAL split for monitoring + final test. NEVER trained "
                        "on or selected on.")
    p.add_argument("--real-oversample", type=int, default=8,
                   help="Repeat the real_train images this many times in the "
                        "mixed train set so real isn't drowned by sim (and the "
                        "domain branch sees balanced batches). 1 = no oversample.")
    # ---- domain-invariance objectives ----
    p.add_argument("--dann", action="store_true",
                   help="Domain-adversarial head (gradient reversal) making the "
                        "backbone feature sim/real-indistinguishable.")
    p.add_argument("--dann-weight", type=float, default=1.0,
                   help="Max DANN weight (lambda ramps 0->this).")
    p.add_argument("--dann-ramp", type=int, default=1000,
                   help="Steps to ramp the DANN lambda from 0 to max.")
    p.add_argument("--mmd", action="store_true",
                   help="Add an unbiased per-location MMD between the sim and "
                        "real images IN each mixed batch (see "
                        "docs/rkhs-mmd-domain-adaptation.md).")
    p.add_argument("--mmd-weight", type=float, default=1.0)
    p.add_argument("--align-layer", type=int, default=10,
                   help="Backbone layer to tap (YOLO11 10 = C2PSA, stride 32).")
    p.add_argument("--align-locations", type=int, default=1024,
                   help="Per-domain spatial locations sampled for the MMD.")
    # ---- A2 measured class prior ----
    p.add_argument("--cls-prior", action="store_true",
                   help="Init the detect-head cls bias from measured per-class "
                        "frequencies (from the sim train split).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# COCO -> YOLO instance-seg label conversion (unchanged, proven)
# ---------------------------------------------------------------------------

def _rle_or_poly_to_binary_mask(seg, H: int, W: int) -> np.ndarray:
    if isinstance(seg, dict):
        return coco_mask.decode(seg).astype(np.uint8)
    if isinstance(seg, list) and seg:
        if isinstance(seg[0], list):
            rles = coco_mask.frPyObjects(seg, H, W)
            return coco_mask.decode(coco_mask.merge(rles)).astype(np.uint8)
        rles = coco_mask.frPyObjects([seg], H, W)
        return coco_mask.decode(coco_mask.merge(rles)).astype(np.uint8)
    return np.zeros((H, W), dtype=np.uint8)


def _mask_to_yolo_polygon(mask: np.ndarray, W: int, H: int,
                          eps_pct: float = 0.002) -> list[list[float]]:
    if mask.sum() < 4:
        return []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if len(c) < 3:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(c, eps_pct * peri, True)
        if len(approx) < 3:
            continue
        flat = []
        for pt in approx.reshape(-1, 2):
            flat.append(min(max(float(pt[0]) / W, 0.0), 1.0))
            flat.append(min(max(float(pt[1]) / H, 0.0), 1.0))
        polys.append(flat)
    return polys


def convert_split_to_yolo(split_dir: Path, cat_id_to_yolo: dict,
                          force: bool = False) -> tuple[int, int, int]:
    """Read split_dir/_annotations.coco.json, write per-image .txt labels under
    split_dir/labels/, move images into split_dir/images/. Returns
    (n_images, n_polys_written, n_dropped)."""
    coco_json = split_dir / "_annotations.coco.json"
    if not coco_json.exists():
        print(f"[yolo] no annotations at {coco_json}; skipping {split_dir.name}")
        return 0, 0, 0
    coco = json.loads(coco_json.read_text())
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for img in coco["images"]:
        src = split_dir / img["file_name"]
        dst = images_dir / Path(img["file_name"]).name
        if src.exists() and src != dst:
            (src.unlink() if dst.exists() else shutil.move(str(src), str(dst)))
    by_img: dict = {}
    for ann in coco["annotations"]:
        by_img.setdefault(int(ann["image_id"]), []).append(ann)
    n_written = n_dropped = 0
    for img in coco["images"]:
        W, H = int(img["width"]), int(img["height"])
        stem = Path(img["file_name"]).stem
        label_path = labels_dir / f"{stem}.txt"
        if label_path.exists() and not force:
            continue
        lines = []
        for ann in by_img.get(int(img["id"]), []):
            cat_id = int(ann["category_id"])
            if cat_id not in cat_id_to_yolo:
                n_dropped += 1
                continue
            polys = _mask_to_yolo_polygon(
                _rle_or_poly_to_binary_mask(ann.get("segmentation", []), H, W), W, H)
            if not polys:
                n_dropped += 1
                continue
            for poly in polys:
                lines.append(f"{cat_id_to_yolo[cat_id]} "
                             + " ".join(f"{c:.6f}" for c in poly))
                n_written += 1
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(coco["images"]), n_written, n_dropped


def write_data_yaml(data_root: Path, classes: list[str], train_list: list[str],
                    val_rel: str, test_rel: str) -> Path:
    """data.yaml with a LIST train (sim + oversampled real) and single val/test."""
    p = data_root / "data_da.yaml"
    lines = [f"path: {data_root.resolve()}", "train:"]
    lines += [f"  - {t}" for t in train_list]
    lines += [f"val:  {val_rel}", f"test: {test_rel}", "", f"nc: {len(classes)}",
              "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(classes)]
    p.write_text("\n".join(lines) + "\n")
    return p


def _measure_cls_prior(coco_json: Path, cat_id_to_yolo: dict, nc: int):
    """A2: per-class expected count PER IMAGE (YOLO class order), from sim."""
    coco = json.loads(Path(coco_json).read_text())
    n_img = max(len(coco.get("images", [])), 1)
    counts = [0.0] * nc
    for ann in coco.get("annotations", []):
        yi = cat_id_to_yolo.get(int(ann["category_id"]))
        if yi is not None:
            counts[yi] += 1.0
    return [c / n_img for c in counts]


# ---------------------------------------------------------------------------
# Domain-invariance: unbiased MMD + the model/trainer (module-level = picklable)
# ---------------------------------------------------------------------------
import torch  # noqa: E402

# Config the (Ultralytics-constructed) trainer/model read at build time.
_DA_CFG: dict = {}


def _mmd(a, b):
    """Unbiased U-statistic squared-MMD for a sum of characteristic RBF
    kernels (median-heuristic bandwidths) on L2-normalized features. Diagonal
    self-terms excluded (0 in expectation when the two samples match). See
    docs/rkhs-mmd-domain-adaptation.md."""
    a = torch.nn.functional.normalize(a.float(), dim=1)
    b = torch.nn.functional.normalize(b.float(), dim=1)
    n, m = a.shape[0], b.shape[0]
    x = torch.cat([a, b], 0)
    d2 = torch.cdist(x, x).pow(2)
    med = d2.detach().flatten().median().clamp_min(1e-6)
    k = sum(torch.exp(-d2 / (g * med)) for g in (0.5, 1.0, 2.0))
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    cross = 2.0 * kxy.mean()
    if n < 2 or m < 2:
        return (kxx.mean() + kyy.mean() - cross).clamp_min(0.0)
    sxx = kxx.sum() - kxx.diagonal().sum()
    syy = kyy.sum() - kyy.diagonal().sum()
    return (sxx / (n * (n - 1)) + syy / (m * (m - 1)) - cross).clamp_min(0.0)


try:
    import torch.nn as _nn
    import torch.nn.functional as _F
    from ultralytics.models.yolo.segment import SegmentationTrainer as _SegTrainer
    from ultralytics.nn.tasks import SegmentationModel as _SegModel
    from ultralytics.nn.modules.head import Detect as _Detect
    from ultralytics.utils import RANK as _RANK

    def _feat_hook(module, inp, out):
        module._da_feat = out          # named (picklable) hook; cleared each loss()

    class _GradReverse(torch.autograd.Function):
        """Identity forward; sign-flipped, scaled gradient backward (DANN)."""
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.lam * g, None

    class DomainAgnosticSegModel(_SegModel):
        """SegmentationModel whose loss adds domain-invariance terms (DANN /
        MMD) computed from the MIXED sim+real batch, using per-image domain
        labels read from the file paths. The task (seg) loss is the standard
        supervised loss over whatever labelled images the batch contains, so
        training jointly on sim+real already learns both domains."""

        def init_da(self, cfg):
            self._da = dict(cfg)
            self._da_layer = min(cfg.get("layer", 10), len(self.model) - 1)
            self.model[self._da_layer].register_forward_hook(_feat_hook)
            self._da_step = 0
            self._da_terms = [t for t in ("dann", "mmd") if cfg.get(t)]
            self._domain_head = None
            if cfg.get("dann"):
                was = self.training
                self.eval()
                with torch.no_grad():
                    self.predict(torch.zeros(1, 3, 64, 64))
                c = self.model[self._da_layer]._da_feat.shape[1]
                self.model[self._da_layer]._da_feat = None
                self._domain_head = _nn.Sequential(
                    _nn.AdaptiveAvgPool2d(1), _nn.Flatten(),
                    _nn.Linear(c, 128), _nn.ReLU(), _nn.Linear(128, 1))
                self.train(was)

        def _domain_of_batch(self, batch, n, device):
            """Per-image domain: 1.0 if the file path is under the real-train
            split, else 0.0 (sim). Falls back to all-sim if paths absent."""
            tag = self._da.get("real_tag", "/real")
            files = batch.get("im_file") or []
            if len(files) != n:
                return torch.zeros(n, device=device)
            return torch.tensor([1.0 if tag in str(f) else 0.0 for f in files],
                                device=device)

        def loss(self, batch, preds=None):
            loss, items = super().loss(batch, preds)   # forward -> hook stores batch feat
            a = getattr(self, "_da", None)
            if a is None:
                return loss, items                      # stripped at save -> vanilla
            layer = self.model[self._da_layer]
            feat = layer._da_feat
            logs = {}
            if self.training and feat is not None:
                bs = feat.shape[0]
                dom = self._domain_of_batch(batch, bs, feat.device)
                sim_m = dom < 0.5
                real_m = dom >= 0.5

                if a.get("dann"):                       # domain-adversarial
                    ramp = min(1.0, self._da_step / float(a.get("dann_ramp", 1000)))
                    lam = a["dann"] * ramp
                    logit = self._domain_head(_GradReverse.apply(feat, lam)).squeeze(1)
                    dloss = _F.binary_cross_entropy_with_logits(logit, dom)
                    loss = loss + bs * dloss
                    logs["dann"] = dloss.detach()

                if a.get("mmd") and sim_m.any() and real_m.any():   # need both domains
                    def perloc(f, k):
                        x = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).float()
                        idx = torch.randperm(x.shape[0], device=x.device)[:k]
                        return x[idx]
                    k = a["locations"]
                    mmd2 = _mmd(perloc(feat[sim_m], k), perloc(feat[real_m], k))
                    loss = loss + a["mmd"] * bs * mmd2
                    logs["mmd"] = mmd2.detach()

                self._da_step += 1
            layer._da_feat = None                       # save-safe
            for t in self._da_terms:                    # keep loss_items length fixed
                v = logs.get(t, items.new_zeros(()))
                items = torch.cat([items, v.reshape(1).to(items)])
            return loss, items

    class DomainAgnosticTrainer(_SegTrainer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if _DA_CFG.get("cls_prior_counts") is not None:
                self.add_callback("on_pretrain_routine_end", _set_cls_prior)

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = DomainAgnosticSegModel(
                cfg, nc=self.data["nc"], ch=self.data["channels"],
                verbose=verbose and _RANK == -1)
            if weights:
                model.load(weights)
            model.init_da(_DA_CFG)
            return model

        def get_validator(self):
            v = super().get_validator()
            for t in _DA_CFG.get("_terms", []):
                if t not in self.loss_names:
                    self.loss_names = (*self.loss_names, t)
            return v

        def save_model(self):
            # strip DA state so the checkpoint reloads as a plain, vanilla
            # SegmentationModel (no phantom loss items on reload).
            mods = [self.model]
            if getattr(self, "ema", None) is not None and self.ema.ema is not None:
                mods.append(self.ema.ema)
            stash = []
            for m in mods:
                m = getattr(m, "module", m)
                stash.append((m, getattr(m, "_da", None), getattr(m, "_da_layer", None)))
                if getattr(m, "_da", None) is not None:
                    m._da = None
                lyr = getattr(m, "_da_layer", None)
                if lyr is not None:
                    m.model[lyr]._forward_hooks.clear()
                    m.model[lyr]._da_feat = None
            try:
                super().save_model()
            finally:
                for m, da, lyr in stash:
                    if da is not None:
                        m._da = da
                    if lyr is not None:
                        m.model[lyr].register_forward_hook(_feat_hook)

    def _set_cls_prior(trainer):
        counts = _DA_CFG.get("cls_prior_counts")
        imgsz = _DA_CFG.get("imgsz", 1024)
        det = next((m for m in trainer.model.modules() if isinstance(m, _Detect)), None)
        if det is None or counts is None:
            return
        exp = torch.tensor(counts, dtype=torch.float32)
        cls_heads = det.cls_head if hasattr(det, "cls_head") else det.cv3
        for conv, s in zip(cls_heads, det.stride):
            n_loc = (imgsz / float(s)) ** 2
            conv[-1].bias.data[:det.nc] = torch.log(exp.to(conv[-1].bias) / n_loc + 1e-9)
        print(f"[yolo][cls-prior] head cls-bias initialized from measured "
              f"frequencies (nc={det.nc})", flush=True)

    _DA_AVAILABLE = True
except Exception as _da_import_err:
    _DA_AVAILABLE = False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    data_root = args.data.expanduser().resolve()
    if not data_root.is_dir():
        sys.exit(f"--data {data_root} does not exist")

    sorted_cats = sorted(((n, c) for n, c in FIXED_CATEGORIES.items() if c != 0),
                         key=lambda nc: nc[1])
    cat_id_to_yolo = {cid: i for i, (_, cid) in enumerate(sorted_cats)}
    yolo_classes = [n for n, _ in sorted_cats]
    print(f"[yolo] {len(yolo_classes)} classes")

    # Splits we need: sim_train + real_train (train on both) ; sim_val ; holdout.
    splits = []
    for s in (args.sim_train, args.real_train, args.sim_val, args.real_holdout):
        if s not in splits:
            splits.append(s)
    totals = {}
    for split in splits:
        if not (data_root / split / "_annotations.coco.json").exists():
            sys.exit(f"[yolo] missing split '{split}' under {data_root}")
        n_img, n_w, n_d = convert_split_to_yolo(
            data_root / split, cat_id_to_yolo, force=args.rebuild_labels)
        totals[split] = (n_img, n_w, n_d)
        print(f"[yolo] {split}: {n_img} images, {n_w} polygons, {n_d} dropped")

    # Mixed labelled TRAIN = sim + real (oversampled). real_holdout excluded.
    train_list = [f"{args.sim_train}/images"]
    train_list += [f"{args.real_train}/images"] * max(1, args.real_oversample)
    yaml_path = write_data_yaml(data_root, yolo_classes, train_list,
                                f"{args.sim_val}/images",
                                f"{args.real_holdout}/images")
    n_sim, n_real = totals[args.sim_train][0], totals[args.real_train][0]
    real_frac = (args.real_oversample * n_real) / (n_sim + args.real_oversample * n_real)
    print(f"[yolo] mixed train: {n_sim} sim + {n_real}x{args.real_oversample} real "
          f"(~{real_frac*100:.0f}% real) | val={args.sim_val} | "
          f"test/holdout={args.real_holdout}")
    print(f"[yolo] wrote {yaml_path}")

    if args.skip_train:
        print("[yolo] --skip-train set; stopping after conversion")
        return

    from ultralytics import YOLO
    model = YOLO(args.model)

    # ---- per-epoch monitor on the REAL holdout (test split). Measurement
    # only -- best.pt is still selected on the sim val, and the holdout is
    # never trained on -> honest cross-domain signal during training. ----
    rows: list[dict] = []

    def _eval_holdout(trainer) -> None:
        epoch = int(trainer.epoch) + 1
        last_pt = Path(trainer.save_dir) / "weights" / "last.pt"
        if not last_pt.is_file():
            return
        m = YOLO(str(last_pt)).val(data=str(yaml_path), split="test",
                                   imgsz=args.imgsz, batch=args.batch,
                                   device=args.device, workers=args.workers,
                                   verbose=False, plots=False, save_json=False)
        row = {"epoch": epoch, "real_box_map": float(m.box.map),
               "real_box_map50": float(m.box.map50)}
        seg = getattr(m, "seg", None)
        if seg is not None and getattr(seg, "map", None) is not None:
            row["real_seg_map"] = float(seg.map)
            row["real_seg_map50"] = float(seg.map50)
        rows.append(row)
        with open(Path(trainer.save_dir) / "real_holdout_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("[real-holdout@epoch{}] ".format(epoch)
              + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"))

    model.add_callback("on_fit_epoch_end", _eval_holdout)

    train_extra = {}
    da_on = args.dann or args.mmd or args.cls_prior
    if da_on:
        if not _DA_AVAILABLE:
            sys.exit("DA requested but the trainer failed to import "
                     "(ultralytics internals).")
        cfg = dict(imgsz=args.imgsz, layer=args.align_layer,
                   locations=args.align_locations,
                   real_tag=f"/{args.real_train}/")
        if args.dann:
            cfg["dann"] = args.dann_weight
            cfg["dann_ramp"] = args.dann_ramp
        if args.mmd:
            cfg["mmd"] = args.mmd_weight
        if args.cls_prior:
            cfg["cls_prior_counts"] = _measure_cls_prior(
                data_root / args.sim_train / "_annotations.coco.json",
                cat_id_to_yolo, len(yolo_classes))
        cfg["_terms"] = [t for t in ("dann", "mmd") if cfg.get(t)]
        _DA_CFG.clear()
        _DA_CFG.update(cfg)
        train_extra["trainer"] = DomainAgnosticTrainer
        print(f"[yolo][DA] domain-invariance terms={cfg['_terms']} "
              f"cls_prior={cfg.get('cls_prior_counts') is not None} "
              f"real_tag={cfg['real_tag']}")

    model.train(
        data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, workers=args.workers,
        project=args.project, name=args.name,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        translate=0.1, scale=0.4, fliplr=0.5,
        mosaic=1.0, mixup=0.0, copy_paste=0.0,
        plots=True, **train_extra,
    )

    # ---- FINAL domain-agnostic measurement: best.pt on BOTH domains ----
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print("\n[domain-agnostic report] best.pt on each domain "
          "(val=sim, test=real holdout):")
    report = {}
    for tag, split in (("sim", "val"), ("real", "test")):
        m = YOLO(str(best)).val(data=str(yaml_path), split=split,
                                imgsz=args.imgsz, batch=args.batch,
                                device=args.device, workers=args.workers,
                                verbose=False, plots=False, save_json=False)
        seg = getattr(m, "seg", None)
        report[tag] = {
            "box_map": float(m.box.map), "box_map50": float(m.box.map50),
            "seg_map": float(seg.map) if seg is not None else float("nan"),
            "seg_map50": float(seg.map50) if seg is not None else float("nan"),
        }
        print(f"  {tag:4s} ({split}): box_mAP50-95={report[tag]['box_map']:.4f} "
              f"seg_mAP50-95={report[tag]['seg_map']:.4f} "
              f"seg_mAP50={report[tag]['seg_map50']:.4f}")
    gap = report["sim"]["seg_map"] - report["real"]["seg_map"]
    print(f"  sim->real seg gap = {gap:.4f}  (smaller = more domain-agnostic)")
    with open(save_dir / "domain_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[domain-agnostic report] -> {save_dir/'domain_report.json'}")


if __name__ == "__main__":
    main()
