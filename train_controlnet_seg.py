"""Train a ControlNet conditioned on the CLASS-ID segmentation map, on the sim
data, so the model becomes OBJECT-AWARE: it learns "seg region of class C ->
render a class-C object" instead of painting a holistic scene.

SD (vae/text/unet) is frozen; only the ControlNet branch trains. Conditioning =
a per-class-id colour map rasterized from the COCO masks (same palette the
styler uses at inference). Target = the sim RGB. After this, the styler loads
this ControlNet (--controlnet-seg <out>/last) so the class-id seg actually
drives object identity; geometry comes from the frozen depth ControlNet, and
realism from real-LoRA + IP-Adapter on top.

    .venv/bin/python train_controlnet_seg.py --data ward_v3 --split train \
        --out runs/cnet_seg/ward --steps 4000
"""
from __future__ import annotations

import argparse
import colorsys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


def class_palette(n: int = 64) -> np.ndarray:
    pal = np.zeros((n, 3), np.uint8)
    for i in range(1, n):
        r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.65, 0.95)
        pal[i] = (int(r * 255), int(g * 255), int(b * 255))
    return pal


class SegToRGB(Dataset):
    """(class-id seg map [0,1], sim RGB [-1,1]) pairs from a split's COCO."""
    def __init__(self, split_dir: Path, res: int):
        from pycocotools.coco import COCO
        self.coco = COCO(str(split_dir / "_annotations.coco.json"))
        self.ids = list(self.coco.imgs)
        self.img_dir = split_dir / "images"
        self.res = res
        self.palette = class_palette()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        info = self.coco.imgs[img_id]
        rgb = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        H, W = info["height"], info["width"]
        seg = np.zeros((H, W, 3), np.uint8)
        for a in self.coco.imgToAnns.get(img_id, []):
            cid = a["category_id"]
            if cid == 0:
                continue
            m = self.coco.annToMask(a)
            seg[m > 0] = self.palette[cid % len(self.palette)]
        rgb = rgb.resize((self.res, self.res), Image.BICUBIC)
        seg = Image.fromarray(seg).resize((self.res, self.res), Image.NEAREST)
        return (TF.to_tensor(rgb) - 0.5) / 0.5, TF.to_tensor(seg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("ward_v3"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--sd-model", default="sd-legacy/stable-diffusion-v1-5")
    ap.add_argument("--out", type=Path, default=Path("runs/cnet_seg/ward"))
    ap.add_argument("--caption", default="a hospital ward room interior")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    dev = (torch.device(f"cuda:{args.device}")
           if torch.cuda.is_available() and args.device != "cpu" else torch.device("cpu"))
    args.out.mkdir(parents=True, exist_ok=True)

    from diffusers import (AutoencoderKL, ControlNetModel, DDPMScheduler,
                           UNet2DConditionModel)
    from transformers import CLIPTextModel, CLIPTokenizer

    print(f"[cnet-seg] base={args.sd_model}  data={args.data/args.split}")
    tok = CLIPTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer")
    text = CLIPTextModel.from_pretrained(args.sd_model, subfolder="text_encoder").to(dev)
    vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae").to(dev)
    unet = UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet").to(dev)
    sched = DDPMScheduler.from_pretrained(args.sd_model, subfolder="scheduler")
    for m in (text, vae, unet):
        m.requires_grad_(False)
    text.eval(); vae.eval(); unet.eval()

    controlnet = ControlNetModel.from_unet(unet).to(dev)
    controlnet.train()
    n_cn = sum(p.numel() for p in controlnet.parameters())
    print(f"[cnet-seg] ControlNet params: {n_cn/1e6:.0f}M (trainable)")
    opt = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

    ds = SegToRGB(args.data / args.split, args.res)
    print(f"[cnet-seg] pairs: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    with torch.no_grad():
        ids = tok(args.caption, padding="max_length", truncation=True,
                  max_length=tok.model_max_length, return_tensors="pt").input_ids.to(dev)
        cap_embed = text(ids)[0]

    def save(tag):
        controlnet.save_pretrained(str(args.out / tag))
        print(f"[cnet-seg] saved -> {args.out/tag}", flush=True)

    sf = vae.config.scaling_factor
    step = 0
    t0 = time.time()
    while step < args.steps:
        for rgb, seg in loader:
            rgb, seg = rgb.to(dev), seg.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    lat = vae.encode(rgb).latent_dist.sample() * sf
                noise = torch.randn_like(lat)
                t = torch.randint(0, sched.config.num_train_timesteps,
                                  (lat.shape[0],), device=dev).long()
                noisy = sched.add_noise(lat, noise, t)
                ce = cap_embed.expand(lat.shape[0], -1, -1)
                down, mid = controlnet(noisy, t, encoder_hidden_states=ce,
                                       controlnet_cond=seg, return_dict=False)
                pred = unet(noisy, t, encoder_hidden_states=ce,
                            down_block_additional_residuals=down,
                            mid_block_additional_residual=mid).sample
                loss = F.mse_loss(pred.float(), noise.float())
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 25 == 0:
                ips = step / (time.time() - t0)
                print(f"\r  step {step}/{args.steps} loss={loss.item():.4f} "
                      f"{ips:.1f} it/s ETA {(args.steps-step)/max(ips,1e-9):.0f}s",
                      end="", flush=True)
            if step % args.save_every == 0:
                print(); save("last")
            if step >= args.steps:
                break
    print(); save("last")
    print(f"[cnet-seg] done ({step} steps, {time.time()-t0:.0f}s) -> {args.out/'last'}")


if __name__ == "__main__":
    main()
