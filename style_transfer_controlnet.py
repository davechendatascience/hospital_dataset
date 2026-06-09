"""Stage 2 of the DA pipeline: translate sim -> real with a frozen Stable
Diffusion backbone + ControlNet, optionally with a LoRA that adapts SD's prior
to OUR real ward (from train_lora_real.py).

Realism comes from SD's pretrained prior (+ the LoRA-on-real shift); geometry is
held by ControlNet conditioned on the sim's GROUND-TRUTH depth (ward_v3) and/or
a segmentation map rasterized from the COCO masks. Output is photoreal RGB that
keeps the sim labels, so it's a drop-in labeled training set.

Modes:
  --preview N         : write N [sim|depth|seg|styled] panels
  --preview-labels N  : write N [sim+masks | styled+masks] label-accuracy checks
  --apply             : stylize whole split(s) -> <out>/<split>/{images,labels,
                        labels_bbox,labels_seg,_annotations.coco.json}

    .venv/bin/python style_transfer_controlnet.py --data ward_v3 --split train \
        --gt-depth --seg-scale 0 --lora runs/lora/ward_real/last --apply --out ward_v3_styled
"""
from __future__ import annotations

import argparse
import colorsys
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

DEFAULT_PROMPT = ("a realistic photograph of a hospital ward room interior, "
                  "natural lighting, DSLR photo, high detail")
DEFAULT_NEG = ("cartoon, illustration, render, cgi, 3d, lowres, blurry, "
               "deformed, distorted geometry, extra objects")


def class_palette(n: int = 64) -> np.ndarray:
    pal = np.zeros((n, 3), np.uint8)
    for i in range(1, n):
        r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.65, 0.95)
        pal[i] = (int(r * 255), int(g * 255), int(b * 255))
    return pal


def mask_to_yolo_polys(m, W, H, eps_pct=0.002):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if len(c) < 3:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        ap = cv2.approxPolyDP(c, eps_pct * peri, True)
        if len(ap) < 3:
            continue
        flat = []
        for pt in ap.reshape(-1, 2):
            flat += [min(max(float(pt[0]) / W, 0.0), 1.0),
                     min(max(float(pt[1]) / H, 0.0), 1.0)]
        polys.append(flat)
    return polys


class CocoLabels:
    """filename-stem -> COCO image + per-instance (mask, cid, bbox); seg source."""
    def __init__(self, split_dir: Path):
        from pycocotools.coco import COCO
        self.coco = COCO(str(split_dir / "_annotations.coco.json"))
        self.by_stem = {Path(im["file_name"]).stem: i for i, im in self.coco.imgs.items()}
        self.cats = self.coco.cats

    def masks_for(self, stem):
        img_id = self.by_stem.get(stem)
        if img_id is None:
            return []
        return [(self.coco.annToMask(a), a["category_id"])
                for a in self.coco.imgToAnns.get(img_id, []) if a["category_id"] != 0]

    def yolo_targets(self, stem):
        img_id = self.by_stem.get(stem)
        if img_id is None:
            return 0, 0, []
        info = self.coco.imgs[img_id]
        W, H = int(info["width"]), int(info["height"])
        out = [(a["category_id"] - 1, a["bbox"], self.coco.annToMask(a))
               for a in self.coco.imgToAnns.get(img_id, []) if a["category_id"] != 0]
        return W, H, out


class MultiControlStyler:
    def __init__(self, args, device, labels=None):
        from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                               UniPCMultistepScheduler)
        from diffusers.utils import logging as dlog
        dlog.set_verbosity_error()
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        dev_idx = int(args.device) if device.type == "cuda" else -1
        self.depth = None
        if not args.gt_depth:
            from transformers import pipeline as hf_pipeline
            print(f"[cnet] depth estimator: {args.depth_model}")
            self.depth = hf_pipeline("depth-estimation", model=args.depth_model, device=dev_idx)
        else:
            print("[cnet] using ground-truth depth from <split>/depth/")
        print(f"[cnet] controlnet depth={args.controlnet_depth} seg={args.controlnet_seg} "
              f"base={args.sd_model}")
        cn = [ControlNetModel.from_pretrained(args.controlnet_depth, torch_dtype=dtype),
              ControlNetModel.from_pretrained(args.controlnet_seg, torch_dtype=dtype)]
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            args.sd_model, controlnet=cn, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device)
        if args.lora:
            self.pipe.load_lora_weights(str(args.lora))
            print(f"[cnet] loaded LoRA (real-adapted prior): {args.lora}")
        self.args, self.dev, self.labels = args, device, labels
        self.scales = [args.depth_scale, args.seg_scale]
        self.palette = class_palette()

    def depth_map(self, pil, size, stem=None):
        if self.args.gt_depth and stem is not None:
            gt = self.args.data / self.args.split / "depth" / f"{stem}.png"
            if gt.is_file():
                return Image.open(gt).convert("RGB").resize((size, size), Image.BICUBIC)
        return self.depth(pil)["depth"].convert("RGB").resize((size, size), Image.BICUBIC)

    def seg_map(self, pil, stem, size):
        w, h = pil.size
        seg = np.zeros((h, w, 3), np.uint8)
        if self.labels is not None:
            for mask, cid in self.labels.masks_for(stem):
                seg[mask > 0] = self.palette[cid % len(self.palette)]
        return Image.fromarray(seg).resize((size, size), Image.NEAREST)

    @torch.no_grad()
    def stylize(self, pil, stem, seed):
        w, h = pil.size
        depth = self.depth_map(pil, self.args.gen_size, stem)
        seg = self.seg_map(pil, stem, self.args.gen_size)
        g = torch.Generator(device=self.dev).manual_seed(seed)
        out = self.pipe(prompt=self.args.prompt, negative_prompt=self.args.neg_prompt,
                        image=[depth, seg], controlnet_conditioning_scale=self.scales,
                        num_inference_steps=self.args.steps, guidance_scale=self.args.guidance,
                        generator=g).images[0]
        return out.resize((w, h), Image.BICUBIC), depth, seg


def list_images(split_dir: Path):
    import glob
    cand = split_dir / "images" if (split_dir / "images").is_dir() else split_dir
    out = []
    for e in ("png", "jpg", "jpeg", "bmp", "webp"):
        out += glob.glob(str(cand / f"*.{e}")) + glob.glob(str(cand / f"*.{e.upper()}"))
    return sorted(set(out))


def draw_labels(base, masks_cids, palette, cats):
    from PIL import ImageDraw
    arr = np.array(base.convert("RGB")).astype(np.float32)
    for mask, cid in masks_cids:
        arr[mask.astype(bool)] = 0.5 * arr[mask.astype(bool)] + 0.5 * palette[cid % len(palette)]
    out = Image.fromarray(arr.astype(np.uint8)); draw = ImageDraw.Draw(out)
    for mask, cid in masks_cids:
        ys, xs = np.where(mask > 0)
        if len(xs):
            draw.text((int(xs.min()), max(0, int(ys.min()) - 10)),
                      cats.get(cid, {}).get("name", str(cid)),
                      fill=tuple(int(c) for c in palette[cid % len(palette)]))
    return out


def run_preview_labels(styler, files, out_dir, n):
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = styler.labels.cats
    for i, p in enumerate(files[:n]):
        pil = Image.open(p).convert("RGB"); stem = Path(p).stem
        styled, _, _ = styler.stylize(pil, stem, 1000 + i)
        masks = styler.labels.masks_for(stem)
        w, h = pil.size; canvas = Image.new("RGB", (w * 2, h))
        canvas.paste(draw_labels(pil, masks, styler.palette, cats), (0, 0))
        canvas.paste(draw_labels(styled, masks, styler.palette, cats), (w, 0))
        canvas.save(out_dir / f"{stem}_labelcheck.png")
    print(f"[cnet] wrote {min(n, len(files))} label-checks -> {out_dir}")


def run_preview(styler, files, out_dir, n):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(files[:n]):
        pil = Image.open(p).convert("RGB")
        styled, depth, seg = styler.stylize(pil, Path(p).stem, 1000 + i)
        w, h = pil.size; canvas = Image.new("RGB", (w * 4, h))
        for j, im in enumerate([pil, depth.resize(pil.size), seg.resize(pil.size), styled]):
            canvas.paste(im, (w * j, 0))
        canvas.save(out_dir / f"{Path(p).stem}_panel.png")
    print(f"[cnet] wrote {min(n, len(files))} previews [sim|depth|seg|styled] -> {out_dir}")


def run_apply(styler, args):
    src, dst = args.data / args.split, args.out / args.split
    for d in ("images", "labels", "labels_bbox", "labels_seg"):
        (dst / d).mkdir(parents=True, exist_ok=True)
    if (src / "_annotations.coco.json").is_file():
        shutil.copy(src / "_annotations.coco.json", dst / "_annotations.coco.json")
    files = list_images(src)
    if args.limit:
        files = files[:args.limit]
    print(f"[cnet] stylizing {len(files)} -> {dst} (image+labels one at a time)", flush=True)
    for i, p in enumerate(files):
        stem = Path(p).stem
        styled, _, _ = styler.stylize(Image.open(p).convert("RGB"), stem, i)
        styled.save(dst / "images" / f"{stem}.png")
        W, H, targets = styler.labels.yolo_targets(stem)
        bb, sg = [], []
        for cls, (bx, by, bw, bh), mask in targets:
            bb.append(f"{cls} {(bx+bw/2)/W:.6f} {(by+bh/2)/H:.6f} {bw/W:.6f} {bh/H:.6f}")
            for poly in mask_to_yolo_polys(mask, W, H):
                sg.append(f"{cls} " + " ".join(f"{c:.6f}" for c in poly))
        nl = lambda L: ("\n".join(L) + "\n") if L else ""
        (dst / "labels_bbox" / f"{stem}.txt").write_text(nl(bb))
        (dst / "labels_seg" / f"{stem}.txt").write_text(nl(sg))
        (dst / "labels" / f"{stem}.txt").write_text(nl(sg))
        if (i + 1) % 25 == 0:
            print(f"\r  {i+1}/{len(files)}", end="", flush=True)
    print(f"\n[cnet] done -> {dst}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("ward_v3"))
    p.add_argument("--split", default="train")
    p.add_argument("--sd-model", default="sd-legacy/stable-diffusion-v1-5")
    p.add_argument("--lora", type=Path, default=None, help="LoRA weights dir (train_lora_real.py)")
    p.add_argument("--controlnet-depth", default="lllyasviel/sd-controlnet-depth")
    p.add_argument("--controlnet-seg", default="lllyasviel/sd-controlnet-seg")
    p.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--gt-depth", action="store_true", default=False,
                   help="use ground-truth depth from <split>/depth/ instead of estimating")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--neg-prompt", default=DEFAULT_NEG)
    p.add_argument("--gen-size", type=int, default=512)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--depth-scale", type=float, default=1.0)
    p.add_argument("--seg-scale", type=float, default=0.5,
                   help="0 = depth-only (avoids seg color-leak; dense GT depth is enough)")
    p.add_argument("--device", default="0")
    p.add_argument("--preview", type=int, default=0)
    p.add_argument("--preview-labels", type=int, default=0)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("ward_v3_styled"))
    return p.parse_args()


def main():
    args = parse_args()
    device = (torch.device(f"cuda:{args.device}")
              if torch.cuda.is_available() and args.device != "cpu" else torch.device("cpu"))
    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    styler = MultiControlStyler(args, device, labels=None)
    if args.preview_labels or args.preview:
        args.split = splits[0]
        styler.labels = CocoLabels(args.data / splits[0])
        files = list_images(args.data / splits[0])
        if args.preview_labels:
            run_preview_labels(styler, files, Path("runs/cnet_preview"), args.preview_labels)
        else:
            run_preview(styler, files, Path("runs/cnet_preview"), args.preview)
    elif args.apply:
        for s in splits:
            args.split = s
            styler.labels = CocoLabels(args.data / s)
            run_apply(styler, args)
    else:
        print("[cnet] pass --preview N, --preview-labels N, or --apply")


if __name__ == "__main__":
    main()
