"""LoRA fine-tune Stable Diffusion's U-Net on the real ward photos, to shift the
generative prior toward OUR real distribution (stage 1 of the DA pipeline).

This is the data-efficient distribution-adaptation lever: SD already knows
"realistic images"; a LoRA trained on ~500 real photos pulls its prior toward
*this* ward. The trained LoRA is then loaded by style_transfer_controlnet.py
(SD + LoRA + ground-truth-depth ControlNet) so translated sim looks like our
real, not generic SD-real.

Trained by the standard diffusion objective (noise prediction) on the real
images with a fixed caption — NOT on any DINOv2/CLIP-MMD, so the MMD metric
stays an honest held-out evaluator. real_holdout is never seen here.

    .venv/bin/python train_lora_real.py --real-dir ward_v1/real_train \
        --out runs/lora/ward_real --steps 2000
"""
from __future__ import annotations

import argparse
import glob
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class RealImages(Dataset):
    def __init__(self, img_dir: Path, res: int):
        cand = img_dir / "images" if (img_dir / "images").is_dir() else img_dir
        self.files = []
        for e in ("png", "jpg", "jpeg", "bmp", "webp"):
            self.files += glob.glob(str(cand / f"*.{e}")) + glob.glob(str(cand / f"*.{e.upper()}"))
        self.files = sorted(set(self.files))
        self.tf = transforms.Compose([
            transforms.Resize(res, antialias=True),
            transforms.RandomCrop(res),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-dir", type=Path, default=Path("ward_v1/real_train"))
    p.add_argument("--sd-model", default="sd-legacy/stable-diffusion-v1-5")
    p.add_argument("--out", type=Path, default=Path("runs/lora/ward_real"))
    p.add_argument("--caption", default="a photo of a hospital ward room interior")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    dev = (torch.device(f"cuda:{args.device}")
           if torch.cuda.is_available() and args.device != "cpu" else torch.device("cpu"))
    args.out.mkdir(parents=True, exist_ok=True)

    from diffusers import (AutoencoderKL, DDPMScheduler, StableDiffusionPipeline,
                           UNet2DConditionModel)
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    print(f"[lora] base={args.sd_model}  real={args.real_dir}  rank={args.rank}")
    tok = CLIPTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer")
    text = CLIPTextModel.from_pretrained(args.sd_model, subfolder="text_encoder").to(dev)
    vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae").to(dev)
    unet = UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet").to(dev)
    sched = DDPMScheduler.from_pretrained(args.sd_model, subfolder="scheduler")
    for m in (text, vae, unet):
        m.requires_grad_(False)
    vae.eval(); text.eval()

    # LoRA only on the U-Net attention projections
    unet.add_adapter(LoraConfig(
        r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in lora_params)
    print(f"[lora] trainable LoRA params: {n_lora/1e6:.2f}M")
    opt = torch.optim.AdamW(lora_params, lr=args.lr)

    ds = RealImages(args.real_dir, args.res)
    print(f"[lora] real images: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    # fixed caption embedding (same prompt for all)
    with torch.no_grad():
        ids = tok(args.caption, padding="max_length", truncation=True,
                  max_length=tok.model_max_length, return_tensors="pt").input_ids.to(dev)
        cap_embed = text(ids)[0]

    def save_lora(tag):
        sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionPipeline.save_lora_weights(str(args.out / tag), unet_lora_layers=sd)
        print(f"[lora] saved -> {args.out/tag}", flush=True)

    unet.train()
    step = 0
    sf = vae.config.scaling_factor
    import time
    t0 = time.time()
    while step < args.steps:
        for imgs in loader:
            imgs = imgs.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    lat = vae.encode(imgs).latent_dist.sample() * sf
                noise = torch.randn_like(lat)
                t = torch.randint(0, sched.config.num_train_timesteps,
                                  (lat.shape[0],), device=dev).long()
                noisy = sched.add_noise(lat, noise, t)
                ce = cap_embed.expand(lat.shape[0], -1, -1)
                pred = unet(noisy, t, encoder_hidden_states=ce).sample
                target = noise if sched.config.prediction_type == "epsilon" \
                    else sched.get_velocity(lat, noise, t)
                loss = F.mse_loss(pred.float(), target.float())
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 50 == 0:
                ips = step / (time.time() - t0)
                print(f"\r  step {step}/{args.steps}  loss={loss.item():.4f}  "
                      f"{ips:.1f} it/s  ETA {(args.steps-step)/max(ips,1e-9):.0f}s",
                      end="", flush=True)
            if step % args.save_every == 0:
                print(); save_lora("last")
            if step >= args.steps:
                break
    print(); save_lora("last")
    print(f"[lora] done ({step} steps, {time.time()-t0:.0f}s) -> {args.out/'last'}")


if __name__ == "__main__":
    main()
