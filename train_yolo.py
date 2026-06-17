"""
Train a YOLO instance-segmentation model on the ward_v1 dataset.

This script:
  1. Reads <dataset>/train/_annotations.coco.json (and valid/, test/)
  2. Converts each COCO annotation set to YOLO instance-segmentation .txt
     polygon format alongside the existing images (one .txt per image,
     written under <dataset>/<split>/labels/ — Ultralytics convention)
  3. Writes data.yaml describing the splits and class list
  4. Kicks off `ultralytics.YOLO.train()` with sensible defaults for
     hospital-object detection.

Run with the project venv python:

    /home/edge-host/Documents/.venv/bin/python train_yolo.py \
        --data ward_v1 \
        --model yolo11s-seg.pt \
        --epochs 50 --imgsz 1024 --batch 8
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
from PIL import Image
from pycocotools import mask as coco_mask

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ROS2_bridge" / "src"))
from fixed_categories import FIXED_CATEGORIES  # type: ignore  # noqa: E402

SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True,
                   help="Dataset root containing train/, valid/, test/ "
                        "(produced by build_dataset.py)")
    p.add_argument("--model", default="yolo11s-seg.pt",
                   help="Ultralytics model checkpoint. Common picks: "
                        "yolo11n-seg (nano), yolo11s-seg (small, default), "
                        "yolo11m-seg (medium), yolo11l-seg (large)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz",  type=int, default=1024,
                   help="Training image size (longest side); 1024 keeps "
                        "small hospital objects visible without OOM at "
                        "batch 8 on 32-48 GB GPUs")
    p.add_argument("--batch",  type=int, default=8)
    p.add_argument("--device", default="0",
                   help="GPU id, 'cpu', or comma-list for multi-GPU "
                        "(e.g., '0,1'). Default: GPU 0.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="runs/segment",
                   help="Ultralytics project dir for run outputs")
    p.add_argument("--name", default=None,
                   help="Run name (default: timestamp-based)")
    p.add_argument("--rebuild-labels", action="store_true",
                   help="Force re-conversion of COCO -> YOLO labels even if "
                        "they already exist")
    p.add_argument("--skip-train", action="store_true",
                   help="Only convert labels + write data.yaml; don't train")
    # ---- split overrides (e.g. fine-tune on real_dev, eval on real_holdout) ----
    p.add_argument("--train-split", default="train",
                   help="Subdir under --data to TRAIN on (e.g. real_dev for "
                        "the A1 real fine-tune phase).")
    p.add_argument("--val-split", default="valid",
                   help="Subdir to validate on (e.g. real_holdout).")
    p.add_argument("--test-split", default="test",
                   help="Subdir for the per-epoch test eval (e.g. real_holdout).")
    # ---- domain adaptation (MMD feature alignment) ----
    p.add_argument("--align-real", type=Path, default=None,
                   help="Dir of UNLABELED real images. Adds an unbiased-MMD "
                        "loss pulling a backbone feature map's distribution "
                        "toward these real images each step (same RKHS method "
                        "as train_seg_detr; see docs/rkhs-mmd-domain-"
                        "adaptation.md). Off unless set.")
    p.add_argument("--align-weight", type=float, default=1.0,
                   help="Weight of the MMD alignment loss (tune: YOLO's base "
                        "loss is batch-scaled, so the align term is too).")
    p.add_argument("--align-batch", type=int, default=8,
                   help="Real images per alignment step.")
    p.add_argument("--align-layer", type=int, default=10,
                   help="Backbone layer index to align (YOLO11 default 10 = "
                        "C2PSA, deepest backbone feature, stride 32).")
    p.add_argument("--align-locations", type=int, default=1024,
                   help="Per-side spatial locations sampled from the feature "
                        "map for the MMD. Per-LOCATION features are used (not "
                        "global-average-pooled): pooled conv features collapse "
                        "in high-D and give a degenerate ~0 MMD.")
    # ---- A1: decaying sim-anchor (L2-SP), for the real fine-tune phase ----
    p.add_argument("--anchor", action="store_true",
                   help="A1: add a decaying L2 penalty pulling weights toward "
                        "their loaded (sim-pretrained) values. Use in phase 2 "
                        "(fine-tune on labeled real, --model = the sim ckpt).")
    p.add_argument("--anchor-tau0", type=float, default=None,
                   help="A1 crossover sample-count tau0 (prior=data weight at "
                        "N=tau0). Default ~ N_real/4. lambda(N)=lambda0*tau0/"
                        "(tau0+N).")
    p.add_argument("--anchor-lambda0", type=float, default=1.0,
                   help="A1 anchor strength scale (set so the anchor term is "
                        "~0.1-1x the seg loss at N=0).")
    p.add_argument("--anchor-fisher", action="store_true",
                   help="A1 variant: Fisher-weighted (EWC) anchor instead of "
                        "isotropic L2-SP (per-param diagonal Fisher from sim).")
    # ---- A2: measured class-prior head-bias init ----
    p.add_argument("--cls-prior", action="store_true",
                   help="A2: init the detect head cls bias from measured "
                        "per-class frequencies (computed from --data/train).")
    # ---- A3: freeze backbone + adapter optimizer ----
    p.add_argument("--freeze", type=int, default=0,
                   help="A3: freeze the first N model layers (YOLO11 backbone "
                        "= 0..10; use 11 to train head+BN-affine only).")
    p.add_argument("--adapter-lr", type=float, default=None,
                   help="A3: if set, AdamW on the trainable params at this LR "
                        "(adapter fine-tune). Pairs with --freeze.")
    # ---- A4: AdaBN ----
    p.add_argument("--adabn", action="store_true",
                   help="A4: reset BatchNorm running stats at epoch 0 so real "
                        "images repopulate them (low-order channel-stat shift).")
    # ---- A5: DANN domain-adversarial branch ----
    p.add_argument("--dann", action="store_true",
                   help="A5: gradient-reversal domain head on the tapped "
                        "backbone feature (needs --align-real for the real "
                        "images). Learned upgrade of the MMD term.")
    p.add_argument("--dann-weight", type=float, default=1.0,
                   help="A5 max domain-loss weight (lambda ramps 0->this).")
    # ---- A9: depth auxiliary head (sim phase; RGB-only at test) ----
    p.add_argument("--depth-aux", action="store_true",
                   help="A9: auxiliary head predicting GT depth (from "
                        "<split>/depth/) on the tapped feature, sim phase only. "
                        "Privileged-information distillation; no test-time cost.")
    p.add_argument("--depth-aux-weight", type=float, default=0.5,
                   help="A9 weight of the depth-prediction MSE loss.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# COCO -> YOLO instance-seg label conversion
# ---------------------------------------------------------------------------

def _rle_or_poly_to_binary_mask(seg, H: int, W: int) -> np.ndarray:
    """Accept any COCO segmentation format and return an HxW uint8 binary mask."""
    if isinstance(seg, dict):
        # RLE
        return coco_mask.decode(seg).astype(np.uint8)
    if isinstance(seg, list) and seg:
        # polygon list of lists OR a single flat list
        if isinstance(seg[0], list):
            rles = coco_mask.frPyObjects(seg, H, W)
            return coco_mask.decode(coco_mask.merge(rles)).astype(np.uint8)
        else:
            rles = coco_mask.frPyObjects([seg], H, W)
            return coco_mask.decode(coco_mask.merge(rles)).astype(np.uint8)
    return np.zeros((H, W), dtype=np.uint8)


def _mask_to_yolo_polygon(mask: np.ndarray, W: int, H: int,
                          eps_pct: float = 0.002) -> list[list[float]]:
    """Find contours in `mask`, simplify with Douglas-Peucker, normalize.
    Returns a list of polygons; each polygon is a flat list [x1,y1,x2,y2,...]
    in 0-1 image coords. Tiny contours are filtered out."""
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
            x, y = float(pt[0]), float(pt[1])
            flat.append(min(max(x / W, 0.0), 1.0))
            flat.append(min(max(y / H, 0.0), 1.0))
        polys.append(flat)
    return polys


def convert_split_to_yolo(split_dir: Path, cat_id_to_yolo: dict,
                          force: bool = False) -> tuple[int, int, int]:
    """Read split_dir/_annotations.coco.json, write per-image .txt labels
    alongside the images under split_dir/labels/. Also moves images into
    split_dir/images/ so Ultralytics' default layout works. Returns
    (n_images, n_annotations_written, n_annotations_dropped)."""
    coco_json = split_dir / "_annotations.coco.json"
    if not coco_json.exists():
        print(f"[yolo] no annotations at {coco_json}; skipping {split_dir.name}")
        return 0, 0, 0
    with open(coco_json) as f:
        coco = json.load(f)

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Move/hard-link images into images/ if they were at split_dir root
    for img in coco["images"]:
        src = split_dir / img["file_name"]
        dst = images_dir / Path(img["file_name"]).name
        if src.exists() and src != dst:
            if dst.exists():
                src.unlink()
            else:
                shutil.move(str(src), str(dst))

    # Group annotations by image
    by_img = {}
    for ann in coco["annotations"]:
        by_img.setdefault(int(ann["image_id"]), []).append(ann)

    n_written = 0
    n_dropped = 0
    for img in coco["images"]:
        img_id = int(img["id"])
        W, H = int(img["width"]), int(img["height"])
        fn = Path(img["file_name"]).name
        stem = Path(fn).stem
        label_path = labels_dir / f"{stem}.txt"
        if label_path.exists() and not force:
            continue
        anns = by_img.get(img_id, [])
        lines = []
        for ann in anns:
            cat_id = int(ann["category_id"])
            if cat_id not in cat_id_to_yolo:
                n_dropped += 1
                continue
            yolo_idx = cat_id_to_yolo[cat_id]
            mask = _rle_or_poly_to_binary_mask(ann.get("segmentation", []), H, W)
            polys = _mask_to_yolo_polygon(mask, W, H)
            if not polys:
                n_dropped += 1
                continue
            # YOLO instance-seg uses ONE polygon per annotation. If a mask
            # produced multiple disjoint polygons we write each as its own
            # row with the same class id (ultralytics handles this fine).
            for poly in polys:
                lines.append(
                    f"{yolo_idx} " + " ".join(f"{c:.6f}" for c in poly)
                )
                n_written += 1
        with open(label_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    return len(coco["images"]), n_written, n_dropped


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------

def write_data_yaml(data_root: Path, yolo_classes: list[str],
                    train_split="train", val_split="valid",
                    test_split="test") -> Path:
    """Write a YOLO data.yaml with absolute paths to the chosen splits."""
    p = data_root / "data.yaml"
    lines = [
        f"path: {data_root.resolve()}",
        f"train: {train_split}/images",
        f"val:   {val_split}/images",
        f"test:  {test_split}/images",
        "",
        f"nc: {len(yolo_classes)}",
        "names:",
    ]
    for i, name in enumerate(yolo_classes):
        lines.append(f"  {i}: {name}")
    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

# --------------------------------------------------------------------------- #
# Domain adaptation: MMD feature alignment (the RKHS method, ported to YOLO)
# --------------------------------------------------------------------------- #
# Config is stashed module-level because Ultralytics constructs the trainer/
# model for us; the trainer reads this in get_model(). Only used when
# --align-real is set.
_ALIGN_CFG: dict = {}


def _mmd(a, b):
    """Unbiased U-statistic estimator of squared MMD ‖μ_P−μ_Q‖²_H for a sum of
    characteristic RBF kernels (median-heuristic bandwidths) on L2-normalized
    features. Diagonal self-terms excluded (correct estimator; 0 in
    expectation when P=Q). Mirrors train_seg_detr._mmd. See
    docs/rkhs-mmd-domain-adaptation.md."""
    import torch
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
    sum_xx = kxx.sum() - kxx.diagonal().sum()
    sum_yy = kyy.sum() - kyy.diagonal().sum()
    return (sum_xx / (n * (n - 1)) + sum_yy / (m * (m - 1)) - cross).clamp_min(0.0)


class _RealLoader:
    """Infinite, shuffled loader of UNLABELED real images, letterboxed to
    `imgsz` and normalized exactly like Ultralytics (RGB, CHW, /255)."""

    def __init__(self, image_dir: Path, imgsz: int, batch: int, device,
                 seed: int = 0):
        import random as _r
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.paths = sorted(str(p) for p in Path(image_dir).iterdir()
                            if p.suffix.lower() in exts)
        if not self.paths:
            raise SystemExit(f"--align-real: no images in {image_dir}")
        self.imgsz, self.batch, self.device = imgsz, batch, device
        self._rng = _r.Random(seed)
        self._rng.shuffle(self.paths)
        self._i = 0

    def _letterbox(self, im):
        h, w = im.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas[top:top + nh, left:left + nw] = im
        return canvas

    def __next__(self):
        import torch
        batch = []
        for _ in range(self.batch):
            if self._i >= len(self.paths):
                self._i = 0
                self._rng.shuffle(self.paths)
            im = cv2.imread(self.paths[self._i]); self._i += 1
            if im is None:
                continue
            im = self._letterbox(im)[:, :, ::-1]          # BGR->RGB
            t = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)
            batch.append(t.float() / 255.0)
        x = torch.stack(batch, 0).to(self.device, non_blocking=True)
        return x


# Alignment trainer/model are defined at MODULE level (not in a factory) so
# the model class is picklable -- Ultralytics torch.saves the whole model into
# the checkpoint. Imports are guarded so the non-aligned path still works if
# something is off; main() checks _ALIGN_AVAILABLE before using them.
import torch  # noqa: E402

try:
    import math as _math
    import torch.nn as _nn
    import torch.nn.functional as _F
    from ultralytics.models.yolo.segment import SegmentationTrainer as _SegTrainer
    from ultralytics.nn.tasks import SegmentationModel as _SegModel
    from ultralytics.nn.modules.head import Detect as _Detect
    from ultralytics.utils import RANK as _RANK
    from ultralytics.data import YOLODataset as _YOLODataset

    def _letterbox_gray(d, size):
        """Aspect-preserving, centered letterbox of a grayscale array to
        (size,size), matching Ultralytics' LetterBox geometry (pad far=0)."""
        h, w = d.shape[:2]
        r = min(size / h, size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        d = cv2.resize(d, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((size, size), np.uint8)
        top, left = (size - nh) // 2, (size - nw) // 2
        canvas[top:top + nh, left:left + nw] = d
        return canvas

    class DepthYOLODataset(_YOLODataset):
        """A9: also yields the paired GT depth (letterboxed to imgsz) as
        label['depth'] in [0,1]. Requires geometric augmentation OFF (main
        enforces it) so the plain centered letterbox keeps depth pixel-aligned
        with the letterboxed RGB."""

        def __getitem__(self, index):
            sample = super().__getitem__(index)
            imgf = Path(self.im_files[index])
            dpath = imgf.parent.parent / "depth" / (imgf.stem + ".png")
            if dpath.is_file():
                d = cv2.imread(str(dpath), cv2.IMREAD_GRAYSCALE)
                d = _letterbox_gray(d, self.imgsz)
                sample["depth"] = torch.from_numpy(d).float().unsqueeze(0) / 255.0
            else:
                sample["depth"] = torch.zeros(1, self.imgsz, self.imgsz)
            return sample

        @staticmethod
        def collate_fn(batch):
            depths = [b.pop("depth", None) for b in batch]
            out = _YOLODataset.collate_fn(batch)
            if all(d is not None for d in depths):
                out["depth"] = torch.stack(depths, 0)
            return out

    def _feat_hook(module, inp, out):
        # Capture the tapped feature ON THE SUBMODULE via a named module-level
        # function (not a closure/lambda) so the model stays picklable. Cleared
        # after each loss() so no tensor lingers at save/deepcopy.
        module._align_feat = out

    class _GradReverse(torch.autograd.Function):
        """A5: identity forward, sign-flipped (and scaled) gradient backward."""
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.lam * g, None

    class AlignSegmentationModel(_SegModel):
        """SegmentationModel + composable domain-adaptation loss terms:
        A1 decaying sim-anchor (L2-SP), A5 DANN domain head, MMD feature
        alignment, A9 depth auxiliary head. Each is gated by _ALIGN_CFG and
        contributes (a) a loss addition and (b) a logged column."""

        def init_align(self, cfg):
            self._da = {"loader": None, **cfg}
            self._da_layer = min(cfg.get("layer", 10), len(self.model) - 1)
            self.model[self._da_layer].register_forward_hook(_feat_hook)
            self._da_step = 0
            self._anchor_ref = None
            # fixed-order list of the active logged terms (-> loss_names cols)
            self._da_terms = [t for t in ("mmd", "dann", "anchor", "depth")
                              if cfg.get({"depth": "depth_aux"}.get(t, t))]
            # build the DANN / depth heads EAGERLY (before build_optimizer, so
            # their params join the optimizer) via a tiny dummy forward to read
            # the tapped feature's channel count.
            self._domain_head = None
            self._depth_head = None
            if cfg.get("dann") or cfg.get("depth_aux"):
                was = self.training
                self.eval()
                with torch.no_grad():
                    self.predict(torch.zeros(1, 3, 64, 64))
                c = self.model[self._da_layer]._align_feat.shape[1]
                self.model[self._da_layer]._align_feat = None
                if cfg.get("dann"):
                    self._domain_head = _nn.Sequential(
                        _nn.AdaptiveAvgPool2d(1), _nn.Flatten(),
                        _nn.Linear(c, 128), _nn.ReLU(), _nn.Linear(128, 1))
                if cfg.get("depth_aux"):
                    self._depth_head = _nn.Sequential(
                        _nn.Conv2d(c, c // 2, 3, padding=1), _nn.GELU(),
                        _nn.Conv2d(c // 2, 1, 1))
                self.train(was)

        def snapshot_anchor(self):
            """A1: record theta_sim from the loaded weights as the anchor ref."""
            if self._da.get("anchor"):
                self._anchor_ref = {n: p.detach().clone()
                                    for n, p in self.named_parameters()
                                    if p.requires_grad}

        def loss(self, batch, preds=None):
            loss, items = super().loss(batch, preds)   # forward -> hook stores sim feat
            a = getattr(self, "_da", None)
            if a is None:
                return loss, items                      # stripped at save -> vanilla model
            layer = self.model[self._da_layer]
            logs = {}
            if self.training:
                bs = batch["img"].shape[0]
                sim_feat = layer._align_feat
                dev = sim_feat.device
                need_real = bool(a.get("mmd") or a.get("dann"))
                real_feat = None
                if need_real:
                    if a["loader"] is None:
                        a["loader"] = _RealLoader(a["real_dir"], a["imgsz"],
                                                  a["real_batch"], dev)
                    real = next(a["loader"])
                    _ = self.forward(real)              # hook stores real feat
                    real_feat = layer._align_feat

                if a.get("mmd"):                        # MMD (per-location)
                    def perloc(f, n):
                        x = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).float()
                        idx = torch.randperm(x.shape[0], device=x.device)[:n]
                        return x[idx]
                    n = a["locations"]
                    mmd2 = _mmd(perloc(sim_feat, n), perloc(real_feat, n))
                    loss = loss + a["mmd"] * bs * mmd2
                    logs["mmd"] = mmd2.detach()

                if a.get("dann"):                       # A5: gradient-reversal domain head
                    ramp = min(1.0, self._da_step / float(a.get("dann_ramp", 1000)))
                    lam = a["dann"] * ramp
                    sl = self._domain_head(_GradReverse.apply(sim_feat, lam)).squeeze(1)
                    rl = self._domain_head(_GradReverse.apply(real_feat, lam)).squeeze(1)
                    dloss = (_F.binary_cross_entropy_with_logits(sl, torch.zeros_like(sl))
                             + _F.binary_cross_entropy_with_logits(rl, torch.ones_like(rl)))
                    loss = loss + bs * dloss
                    logs["dann"] = dloss.detach()

                if a.get("anchor") and self._anchor_ref is not None:   # A1
                    tau0 = a["anchor_tau0"]
                    lam = a["anchor"] * tau0 / (tau0 + self._da_step)   # decays on 1/N tail
                    fisher = a.get("anchor_fisher_diag")
                    pen = sim_feat.new_zeros(())
                    for nm, p in self.named_parameters():
                        ref = self._anchor_ref.get(nm)
                        if ref is None or not p.requires_grad:
                            continue
                        d2 = (p - ref.to(p.device)) ** 2   # ref snapshotted on CPU
                        pen = pen + ((fisher[nm].to(p.device) * d2).sum()
                                     if fisher and nm in fisher else d2.sum())
                    loss = loss + lam * pen
                    logs["anchor"] = (lam * pen).detach()

                if a.get("depth_aux") and "depth" in batch:            # A9
                    pred = self._depth_head(sim_feat)
                    tgt = batch["depth"].to(pred)
                    if tgt.dim() == 3:
                        tgt = tgt.unsqueeze(1)
                    tgt = _F.interpolate(tgt, size=pred.shape[-2:],
                                         mode="bilinear", align_corners=False)
                    dl = _F.mse_loss(pred, tgt)
                    loss = loss + a["depth_aux"] * bs * dl
                    logs["depth"] = dl.detach()

                self._da_step += 1
            layer._align_feat = None                    # save-safe: no tensor on the module
            # append one logged item per active term (real value in train, 0 in
            # eval) so loss_items length always matches the extended loss_names.
            for t in self._da_terms:
                v = logs.get(t, items.new_zeros(()))
                items = torch.cat([items, v.reshape(1).to(items)])
            return loss, items

    class AlignSegmentationTrainer(_SegTrainer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if _ALIGN_CFG.get("cls_prior_counts") is not None:
                self.add_callback("on_pretrain_routine_end", _set_cls_prior)
            if _ALIGN_CFG.get("adabn"):
                self.add_callback("on_train_epoch_start", _adabn_reset)

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = AlignSegmentationModel(
                cfg, nc=self.data["nc"], ch=self.data["channels"],
                verbose=verbose and _RANK == -1)
            if weights:
                model.load(weights)                     # load BEFORE the anchor snapshot
            model.init_align(_ALIGN_CFG)
            model.snapshot_anchor()                     # theta_sim = loaded weights (A1)
            return model

        def build_dataset(self, img_path, mode="train", batch=None):
            # A9: in the sim TRAIN phase, use the depth-yielding dataset so the
            # loss can supervise the auxiliary depth head. Val/other modes use
            # the stock dataset.
            if mode == "train" and _ALIGN_CFG.get("depth_aux"):
                gs = max(int((getattr(self.model, "module", self.model)).stride.max()), 32)
                return DepthYOLODataset(
                    img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
                    augment=True, hyp=self.args, rect=False,
                    cache=self.args.cache or None,
                    single_cls=self.args.single_cls or False, stride=gs, pad=0.0,
                    prefix="train: ", task=self.args.task,
                    classes=self.args.classes, data=self.data,
                    fraction=self.args.fraction)
            return super().build_dataset(img_path, mode, batch)

        def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9,
                            decay=1e-5, iterations=1e5):
            # A3: when an adapter LR is set, AdamW over only the trainable
            # params (backbone frozen via --freeze) -- head + BN-affine adapt.
            alr = _ALIGN_CFG.get("adapter_lr")
            if alr is None:
                return super().build_optimizer(model, name, lr, momentum, decay, iterations)
            trainable = [p for p in (getattr(model, "module", model)).parameters()
                         if p.requires_grad]
            return torch.optim.AdamW(trainable, lr=alr, weight_decay=decay)

        def get_validator(self):
            v = super().get_validator()
            for t in _ALIGN_CFG.get("_terms", []):
                if t not in self.loss_names:
                    self.loss_names = (*self.loss_names, t)
            return v

        def save_model(self):
            # Strip the DA state (hook + captured feature + config + anchor ref)
            # from the live model AND the EMA so the saved checkpoint is a plain
            # SegmentationModel (reloading it for the per-epoch test eval then
            # won't append phantom loss items). The DANN/depth heads stay as
            # inert submodules; predict() ignores them.
            mods = [self.model]
            if getattr(self, "ema", None) is not None and self.ema.ema is not None:
                mods.append(self.ema.ema)
            stash = []
            for m in mods:
                m = getattr(m, "module", m)
                stash.append((m, getattr(m, "_da", None),
                              getattr(m, "_anchor_ref", None),
                              getattr(m, "_da_layer", None)))
                if getattr(m, "_da", None) is not None:
                    m._da = None
                if getattr(m, "_anchor_ref", None) is not None:
                    m._anchor_ref = None
                lyr = getattr(m, "_da_layer", None)
                if lyr is not None:
                    m.model[lyr]._forward_hooks.clear()
                    m.model[lyr]._align_feat = None
            try:
                super().save_model()
            finally:
                for m, da, ref, lyr in stash:
                    if da is not None:
                        m._da = da
                    if ref is not None:
                        m._anchor_ref = ref
                    if lyr is not None:
                        m.model[lyr].register_forward_hook(_feat_hook)

    def _set_cls_prior(trainer):
        # A2: init the Detect/Segment head cls bias from measured per-class
        # expected counts (computed from --data/train). Runs once after the
        # model/strides are ready, before training.
        counts = _ALIGN_CFG.get("cls_prior_counts")
        imgsz = _ALIGN_CFG.get("imgsz", 1024)
        det = next((m for m in trainer.model.modules() if isinstance(m, _Detect)), None)
        if det is None or counts is None:
            return
        exp = torch.tensor(counts, dtype=torch.float32)
        cls_heads = det.cls_head if hasattr(det, "cls_head") else det.cv3
        for conv, s in zip(cls_heads, det.stride):
            n_loc = (imgsz / float(s)) ** 2
            conv[-1].bias.data[:det.nc] = torch.log(exp.to(conv[-1].bias) / n_loc + 1e-9)
        print(f"[yolo][A2] set cls-bias prior from measured frequencies "
              f"(nc={det.nc})", flush=True)

    def _adabn_reset(trainer):
        # A4: at epoch 0, forget sim-domain BN stats and switch to cumulative
        # moving average so the real images repopulate running mean/var.
        if trainer.epoch != 0:
            return
        nreset = 0
        for m in trainer.model.modules():
            if isinstance(m, _nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None
                nreset += 1
        print(f"[yolo][A4] AdaBN: reset {nreset} BatchNorm layers on real data",
              flush=True)

    _ALIGN_AVAILABLE = True
except Exception as _align_import_err:                  # ultralytics import issue
    _ALIGN_AVAILABLE = False


def _measure_cls_prior(coco_json: Path, cat_id_to_yolo: dict, nc: int):
    """A2: per-class expected count PER IMAGE, in YOLO class order (0..nc-1),
    measured from a COCO json (e.g. the sim train split)."""
    coco = json.loads(Path(coco_json).read_text())
    n_img = max(len(coco.get("images", [])), 1)
    counts = [0.0] * nc
    for ann in coco.get("annotations", []):
        yi = cat_id_to_yolo.get(int(ann["category_id"]))
        if yi is not None:
            counts[yi] += 1.0
    return [c / n_img for c in counts]


def main() -> None:
    args = parse_args()
    data_root = args.data.expanduser().resolve()
    if not data_root.is_dir():
        sys.exit(f"--data {data_root} does not exist")

    # Build the canonical category-id -> YOLO-class-index map from
    # fixed_categories.py (sorted by COCO id ascending, then yolo_idx = position).
    # We exclude the synthetic "ward_object" supercategory (id 0).
    sorted_cats = sorted(
        ((name, cid) for name, cid in FIXED_CATEGORIES.items() if cid != 0),
        key=lambda nc: nc[1],
    )
    cat_id_to_yolo = {cid: i for i, (_, cid) in enumerate(sorted_cats)}
    yolo_classes = [name for name, _ in sorted_cats]
    print(f"[yolo] {len(yolo_classes)} classes (yolo_idx 0..{len(yolo_classes)-1})")

    # Convert each (possibly overridden) split to YOLO format
    splits = []
    for s in (args.train_split, args.val_split, args.test_split):
        if s not in splits:
            splits.append(s)
    totals = {}
    missing = []
    for split in splits:
        if not (data_root / split / "_annotations.coco.json").exists():
            missing.append(split)
            continue
        n_img, n_w, n_d = convert_split_to_yolo(
            data_root / split, cat_id_to_yolo, force=args.rebuild_labels,
        )
        totals[split] = (n_img, n_w, n_d)
        print(f"[yolo] {split}: {n_img} images, {n_w} polygons written, "
              f"{n_d} dropped")
    if missing:
        sys.exit(
            f"[yolo] dataset is incomplete: missing {missing} split(s) under "
            f"{data_root}."
        )

    yaml_path = write_data_yaml(data_root, yolo_classes,
                                args.train_split, args.val_split, args.test_split)
    print(f"[yolo] wrote data spec: {yaml_path}  "
          f"(train={args.train_split}, val={args.val_split}, test={args.test_split})")

    if args.skip_train:
        print("[yolo] --skip-train set; stopping after label conversion")
        return

    # Train
    print(f"[yolo] starting training: model={args.model}, epochs={args.epochs}, "
          f"imgsz={args.imgsz}, batch={args.batch}, device={args.device}")
    from ultralytics import YOLO
    model = YOLO(args.model)

    # Per-epoch eval on the held-out test split, in addition to the standard
    # val-split eval Ultralytics already runs. We hook on_fit_epoch_end (fires
    # AFTER Ultralytics has written last.pt for the epoch), load last.pt into
    # a fresh YOLO wrapper, and run val() with split='test'. Using last.pt
    # rather than poking trainer.model avoids any interaction with the live
    # EMA / autograd state.
    test_metrics_rows: list[dict] = []

    def _eval_test_split(trainer) -> None:
        epoch = int(trainer.epoch) + 1  # match results.csv 1-indexing
        last_pt = Path(trainer.save_dir) / "weights" / "last.pt"
        if not last_pt.is_file():
            return
        eval_model = YOLO(str(last_pt))
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
        seg = getattr(metrics, "seg", None)
        if seg is not None and getattr(seg, "map", None) is not None:
            row["seg_map"]   = float(seg.map)
            row["seg_map50"] = float(seg.map50)
            row["seg_map75"] = float(seg.map75)
        test_metrics_rows.append(row)

        csv_path = Path(trainer.save_dir) / "test_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(test_metrics_rows[0].keys()))
            w.writeheader()
            for r in test_metrics_rows:
                w.writerow(r)
        print("[test@epoch{:d}] ".format(epoch)
              + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"))

    model.add_callback("on_fit_epoch_end", _eval_test_split)

    # ---- Domain adaptation (A1 anchor / A2 cls-prior / A3 freeze+adapter /
    # A4 AdaBN / A5 DANN / MMD / A9 depth-aux). All composable + gated. ----
    da_active = any([args.align_real is not None, args.anchor, args.dann,
                     args.cls_prior, args.adabn, args.adapter_lr is not None,
                     args.depth_aux])
    train_extra = {}
    if args.freeze:
        train_extra["freeze"] = args.freeze          # Ultralytics built-in (A3)
    if da_active:
        if not _ALIGN_AVAILABLE:
            sys.exit("DA requested but the alignment trainer failed to import "
                     "(ultralytics internals). See startup error.")
        cfg = dict(imgsz=args.imgsz, layer=args.align_layer,
                   locations=args.align_locations, real_batch=args.align_batch)
        if args.dann and args.align_real is None:
            sys.exit("--dann needs --align-real (the unlabeled real images).")
        if args.align_real is not None:
            align_dir = args.align_real.expanduser().resolve()
            if not align_dir.is_dir():
                sys.exit(f"--align-real {align_dir} does not exist")
            cfg["real_dir"] = align_dir
            cfg["mmd"] = args.align_weight            # MMD feature-alignment term
        if args.dann:
            cfg["dann"] = args.dann_weight
        if args.anchor:
            n_real = totals.get(args.train_split, (0,))[0]
            cfg["anchor"] = args.anchor_lambda0
            cfg["anchor_tau0"] = (args.anchor_tau0 if args.anchor_tau0 is not None
                                  else max(1.0, n_real / 4.0))
            cfg["anchor_fisher"] = args.anchor_fisher
            if args.anchor_fisher:
                print("[yolo][A1] --anchor-fisher: diagonal Fisher is not "
                      "estimated (needs a sim-data pass); using isotropic "
                      "L2-SP.", flush=True)
        if args.depth_aux:
            cfg["depth_aux"] = args.depth_aux_weight
        if args.cls_prior:
            cfg["cls_prior_counts"] = _measure_cls_prior(
                data_root / "train" / "_annotations.coco.json",
                cat_id_to_yolo, len(yolo_classes))
        if args.adabn:
            cfg["adabn"] = True
        if args.adapter_lr is not None:
            cfg["adapter_lr"] = args.adapter_lr
        cfg["_terms"] = [t for t in ("mmd", "dann", "anchor", "depth")
                         if cfg.get({"depth": "depth_aux"}.get(t, t))]
        _ALIGN_CFG.clear()
        _ALIGN_CFG.update(cfg)
        train_extra["trainer"] = AlignSegmentationTrainer
        print(f"[yolo][DA] active terms={cfg['_terms']} "
              f"cls_prior={cfg.get('cls_prior_counts') is not None} "
              f"adabn={bool(cfg.get('adabn'))} adapter_lr={cfg.get('adapter_lr')} "
              f"freeze={args.freeze} anchor_tau0={cfg.get('anchor_tau0')}")

    # Geometric augmentation: ON normally; OFF for A9 depth-aux so the GT
    # depth (loaded with a plain letterbox) stays pixel-aligned with the
    # augmented RGB (depth doesn't ride through mosaic/affine/flip).
    if args.depth_aux:
        aug = dict(translate=0.0, scale=0.0, fliplr=0.0, flipud=0.0,
                   degrees=0.0, shear=0.0, perspective=0.0,
                   mosaic=0.0, mixup=0.0, copy_paste=0.0)
        print("[yolo][A9] depth-aux: geometric augmentation disabled to keep "
              "depth aligned (photometric HSV kept).")
    else:
        aug = dict(translate=0.1, scale=0.4, fliplr=0.5,
                   mosaic=1.0, mixup=0.0, copy_paste=0.0)

    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        # Sensible defaults for sim-to-real detection: a bit of geometric +
        # photometric augmentation (test set is real photos, so we want the
        # model to be invariant to camera/lighting nuisance).
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        **aug,
        plots=True,
        **train_extra,
    )


if __name__ == "__main__":
    main()
