"""Render instance-segmentation predictions from a train_seg_detr.py checkpoint.

Rebuilds the model exactly like training (custom backbones aren't loadable via
plain from_pretrained -- the saved config holds a dummy backbone), loads the
epoch's safetensors, predicts on a few images and writes colored mask overlays
(class name + score at each instance) next to the originals.

    .venv/bin/python predict_seg_detr.py \
        --ckpt runs/seg_detr/<name>/weights/epoch4 --backbone dinov2 \
        --images ward_v4/real_holdout/images --n 4 --out predictions
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

import train_seg_detr as T

PROJECT = Path(__file__).resolve().parents[2]


def build(args):
    ns = SimpleNamespace(
        model=args.model, dino_name=args.dino_name,
        freeze_backbone=True, gfn_gate=True, gfn_latents=64,
        gfn_aux_weight=0.0, lejepa_ckpt=args.lejepa_ckpt)
    if args.backbone == "dinov2":
        return T.build_dinov2_mask2former(ns)
    if args.backbone == "gfn":
        return T.build_gfn_mask2former(ns)
    if args.backbone == "lejepa":
        return T.build_lejepa_mask2former(ns)
    from transformers import AutoModelForUniversalSegmentation
    return AutoModelForUniversalSegmentation.from_pretrained(
        args.model, id2label=T.ID2LABEL, label2id=T.LABEL2ID,
        num_labels=T.NUM_LABELS, ignore_mismatched_sizes=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="runs/seg_detr/<name>/weights/epochN dir")
    ap.add_argument("--backbone", choices=["swin", "dinov2", "gfn", "lejepa"],
                    default="dinov2")
    ap.add_argument("--model", default="facebook/mask2former-swin-tiny-coco-instance")
    ap.add_argument("--dino-name", default="facebook/dinov2-base")
    ap.add_argument("--lejepa-ckpt", type=Path, default=None)
    ap.add_argument("--images", type=Path,
                    default=PROJECT / "ward_v4/real_holdout/images")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <ckpt run dir>/predictions")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from safetensors.torch import load_file
    from transformers import AutoImageProcessor

    model = build(args)
    sd = load_file(args.ckpt / "model.safetensors")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[predict] loaded {args.ckpt} (missing={len(missing)}, "
          f"unexpected={len(unexpected)})")
    assert not unexpected, unexpected[:5]
    dev = torch.device(args.device)
    model.to(dev).eval()
    processor = AutoImageProcessor.from_pretrained(args.ckpt)

    paths = sorted(p for p in args.images.iterdir()
                   if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
    random.Random(args.seed).shuffle(paths)
    paths = paths[:args.n]
    out = args.out or args.ckpt.parent.parent / "predictions"
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(7)
    palette = rng.randint(60, 255, (64, 3))
    for p in paths:
        im = Image.open(p).convert("RGB")
        inputs = processor(images=im, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = model(**inputs)
        res = processor.post_process_instance_segmentation(
            outputs, target_sizes=[im.size[::-1]], threshold=args.score)[0]
        seg = res["segmentation"]
        seg = seg.cpu().numpy() if torch.is_tensor(seg) else np.array(seg)
        base = np.array(im, dtype=np.float32)
        draw_jobs = []
        for k, info in enumerate(res["segments_info"]):
            m = seg == info["id"]
            if not m.any():
                continue
            color = palette[k % len(palette)]
            base[m] = 0.45 * base[m] + 0.55 * color
            ys, xs = np.nonzero(m)
            name = T.ID2LABEL.get(info["label_id"], str(info["label_id"]))
            draw_jobs.append((int(xs.mean()), int(ys.mean()),
                              f"{name} {info['score']:.2f}", tuple(color)))
        vis = Image.fromarray(base.astype(np.uint8))
        d = ImageDraw.Draw(vis)
        for x, y, txt, color in draw_jobs:
            w = d.textlength(txt)
            d.rectangle([x - 2, y - 11, x + w + 2, y + 2], fill=(0, 0, 0))
            d.text((x, y - 10), txt, fill=color)
        dst = out / f"pred_{p.stem}.jpg"
        vis.save(dst, quality=92)
        print(f"[predict] {p.name}: {len(draw_jobs)} instances "
              f">= {args.score} -> {dst}")
    print(f"[predict] done -> {out}")


if __name__ == "__main__":
    main()
