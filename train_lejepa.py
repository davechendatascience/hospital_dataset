"""LeJEPA self-supervised pretraining on the ward domain (sim + styled + real).

Reimplementation of LeJEPA (LeCun & Balestriero, arXiv:2511.08544): a JEPA
trained with just two terms and NO anti-collapse heuristics (no stop-gradient,
no EMA teacher, no whitening):

  1. prediction loss  -- MSE between the embeddings of two augmented views;
  2. SIGReg           -- Sketched Isotropic Gaussian Regularization: project
     the batch embeddings onto M random unit directions and push each 1-D
     marginal toward N(0,1) with an Epps-Pulley characteristic-function test
     (computed on a fixed t-grid -> linear in batch size, differentiable).

The paper's key claim we exploit: in-domain SSL pretraining on a specialized
corpus can beat transfer from giant generic backbones. Here the corpus is OUR
ward in all three appearances (Isaac sim renders, Cosmos-styled frames, real
webcam photos), so the backbone learns ward semantics with sim and real in ONE
embedding space -- exactly what the sim2real detector / domain metrics want.

Real-data discipline: ward_v3/test is hash-split; only the even half joins the
SSL pool (dev), the odd half stays untouched for evaluation purity.

    .venv/bin/python train_lejepa.py --epochs 400 --batch 192 --name ward_r50
"""
from __future__ import annotations

import argparse
import math
import zlib
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

PROJECT = Path(__file__).resolve().parent
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ------------------------------------------------------------------- data --
def collect_images(dirs, include_real_dev=True):
    paths = []
    for d in dirs:
        d = Path(d)
        if d.is_dir():
            paths += [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT]
    if include_real_dev:
        real = PROJECT / "ward_v3/test/images"
        if real.is_dir():
            real_all = [p for p in sorted(real.iterdir())
                        if p.suffix.lower() in IMG_EXT]
            dev = [p for p in real_all
                   if zlib.crc32(p.stem.encode()) % 2 == 0]
            paths += dev
            print(f"[lejepa] real photos: {len(dev)}/{len(real_all)} in the SSL "
                  f"pool (hash-even dev half; odd half stays held out)")
    return paths


class TwoViews(Dataset):
    def __init__(self, paths, size=224):
        self.paths = paths
        self.aug = T.Compose([
            T.RandomResizedCrop(size, scale=(0.25, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(9, (0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        return self.aug(im), self.aug(im)


# ----------------------------------------------------------------- SIGReg --
class SIGReg(nn.Module):
    """Epps-Pulley statistic of M random 1-D projections against N(0,1),
    integrated over a fixed t-grid with N(0,1) weights (linear in batch)."""

    def __init__(self, n_dirs=256, t_max=4.0, t_pts=33):
        super().__init__()
        self.n_dirs = n_dirs
        t = torch.linspace(-t_max, t_max, t_pts)
        w = torch.exp(-0.5 * t ** 2) / math.sqrt(2 * math.pi)   # N(0,1) pdf
        w = w * (t[1] - t[0])                                    # trapezoid dt
        self.register_buffer("t", t)
        self.register_buffer("w", w)
        self.register_buffer("phi", torch.exp(-0.5 * t ** 2))   # target ECF

    def forward(self, z):
        # z: (n, d) -- raw embeddings; isotropy target N(0, I)
        u = torch.randn(z.shape[1], self.n_dirs, device=z.device, dtype=z.dtype)
        u = u / u.norm(dim=0, keepdim=True)
        p = z @ u                                  # (n, M) projections
        tp = p.unsqueeze(-1) * self.t              # (n, M, K)
        c = torch.cos(tp).mean(0)                  # (M, K) Re ECF
        s = torch.sin(tp).mean(0)                  # (M, K) Im ECF
        ep = ((c - self.phi) ** 2 + s ** 2) * self.w
        return ep.sum(-1).mean()


# ------------------------------------------------------------------ train --
def build_backbone(arch):
    net = getattr(models, arch)(weights=None)
    if hasattr(net, "fc"):                          # resnets
        dim = net.fc.in_features
        net.fc = nn.Identity()
    elif hasattr(net, "heads"):                     # torchvision ViTs
        dim = net.heads.head.in_features
        net.heads = nn.Identity()
    else:
        raise SystemExit(f"unsupported arch {arch}")
    return net, dim


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="*", default=[
        PROJECT / "ward_v3/train/images",     # sim renders
        PROJECT / "ward_v4/train/images",     # Cosmos-styled
        PROJECT / "ward_v3/val/images",       # sim val renders
        PROJECT / "ward_10k/_train_render/_raw",  # placement-DR leftovers
    ])
    ap.add_argument("--no-real", action="store_true",
                    help="exclude the real dev half from the SSL pool")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 1.5e-3 * batch/256")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--lambda-sigreg", type=float, default=1.0)
    ap.add_argument("--n-dirs", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--name", default="ward_lejepa")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    paths = collect_images(args.data, include_real_dev=not args.no_real)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit("no images found")
    print(f"[lejepa] {len(paths)} images | arch={args.arch} "
          f"batch={args.batch} epochs={args.epochs}")

    out = PROJECT / "lejepa_runs" / args.name
    out.mkdir(parents=True, exist_ok=True)

    ds = TwoViews(paths, size=args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=args.workers, pin_memory=True,
                    persistent_workers=args.workers > 0)

    dev = torch.device(args.device)
    net, dim = build_backbone(args.arch)
    net = net.to(dev)
    sig = SIGReg(n_dirs=args.n_dirs).to(dev)
    lr = args.lr or 1.5e-3 * args.batch / 256
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=args.wd)
    steps_per_epoch = max(len(dl), 1)
    total = args.epochs * steps_per_epoch
    warm = args.warmup * steps_per_epoch

    def lr_at(step):
        if step < warm:
            return step / max(warm, 1)
        f = (step - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * f))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    log = open(out / "log.csv", "a")
    log.write("epoch,loss,pred,sigreg,std,lr,sec\n")

    step = 0
    for ep in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        agg = {"loss": 0.0, "pred": 0.0, "sig": 0.0, "std": 0.0, "n": 0}
        for v1, v2 in dl:
            v1, v2 = v1.to(dev, non_blocking=True), v2.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = net(torch.cat([v1, v2], 0)).float()
            z1, z2 = z.chunk(2, 0)
            pred = F.mse_loss(z1, z2)
            sg = 0.5 * (sig(z1) + sig(z2))
            loss = pred + args.lambda_sigreg * sg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            agg["loss"] += loss.item(); agg["pred"] += pred.item()
            agg["sig"] += sg.item()
            agg["std"] += z.std(0).mean().item(); agg["n"] += 1
        n = max(agg["n"], 1)
        line = (f"{ep},{agg['loss']/n:.4f},{agg['pred']/n:.4f},"
                f"{agg['sig']/n:.5f},{agg['std']/n:.3f},"
                f"{sched.get_last_lr()[0]:.2e},{time.time()-t0:.1f}")
        log.write(line + "\n"); log.flush()
        print(f"[lejepa] ep {ep:4d}/{args.epochs} "
              f"loss={agg['loss']/n:.4f} pred={agg['pred']/n:.4f} "
              f"sigreg={agg['sig']/n:.5f} std={agg['std']/n:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if ep % args.ckpt_every == 0 or ep == args.epochs:
            torch.save({"arch": args.arch, "dim": dim, "epoch": ep,
                        "model": net.state_dict()}, out / f"ckpt_ep{ep}.pt")
    torch.save({"arch": args.arch, "dim": dim, "epoch": args.epochs,
                "model": net.state_dict()}, out / "ckpt_final.pt")
    print(f"[lejepa] done -> {out}/ckpt_final.pt")


if __name__ == "__main__":
    main()
