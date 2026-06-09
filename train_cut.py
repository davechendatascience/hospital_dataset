"""CUT (Contrastive Unpaired Translation, Park et al. ECCV'20) — sim -> real
appearance translation for the ward dataset, metric-gated on the DINOv2
distribution gap.

WHY CUT (not CycleGAN): one generator + PatchNCE contrastive loss instead of
two generators + a cycle loss. ~half the compute, and the patchwise contrastive
objective preserves content/shape better than cycle-consistency — which is what
keeps the COCO masks valid when we run the trained G back over the labeled sim
images.

Trains on the LABELED splits: sim = <data>/train/images (carries masks), real =
<data>/test/images. Unpaired (no correspondence needed). Only colour/texture is
changed, so masks/boxes stay valid.

METRIC-GATED: every --eval-every epochs, translate a fixed sim subset and
measure the DINOv2 distribution distance to real (reusing measure_domain_gap):
probe accuracy toward 50% and MMD toward 0 = the gap closing. Early-stops on
probe-accuracy patience instead of a fixed epoch count.

Train:
    /home/edge-host/Documents/.venv/bin/python train_cut.py --data ward_v1 \
        --crop 256 --eval-every 20 --device 0

Apply (produce the labeled stylized dataset for GFN training):
    /home/edge-host/Documents/.venv/bin/python train_cut.py --data ward_v1 \
        --apply --weights runs/cut/ward_v1/weights/best_G.pt \
        --out ward_v1_styled
"""
from __future__ import annotations

import argparse
import csv
import functools
import glob
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
IMG_EXTS = ("png", "jpg", "jpeg", "bmp", "webp")


# --------------------------------------------------------------------------- #
# Networks (ResNet generator + PatchGAN discriminator + PatchSampleF MLP)
# --------------------------------------------------------------------------- #
def init_weights(net: nn.Module, gain: float = 0.02) -> None:
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, norm):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), norm(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), norm(dim),
        )

    def forward(self, x):
        return x + self.conv(x)


class ResnetGenerator(nn.Module):
    """9-block ResNet generator (CycleGAN/CUT standard). `forward(x, layers,
    encode_only=True)` returns the encoder features CUT samples patches from."""

    def __init__(self, ngf: int = 64, n_blocks: int = 9):
        super().__init__()
        norm = functools.partial(nn.InstanceNorm2d, affine=False)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), norm(ngf), nn.ReLU(True)]
        for i in range(2):                       # downsample
            m = 2 ** i
            model += [nn.Conv2d(ngf * m, ngf * m * 2, 3, 2, 1),
                      norm(ngf * m * 2), nn.ReLU(True)]
        for _ in range(n_blocks):                # transform
            model += [ResnetBlock(ngf * 4, norm)]
        for i in range(2):                       # upsample (resize+conv, not
            m = 2 ** (2 - i)                     # transpose-conv: no checkerboard)
            model += [nn.Upsample(scale_factor=2, mode="nearest"),
                      nn.ReflectionPad2d(1), nn.Conv2d(ngf * m, ngf * m // 2, 3),
                      norm(ngf * m // 2), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x, layers=None, encode_only=False):
        if layers:
            feats, feat = [], x
            for i, layer in enumerate(self.model):
                feat = layer(feat)
                if i in layers:
                    feats.append(feat)
                if i == max(layers) and encode_only:
                    return feats
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """70x70 PatchGAN."""

    def __init__(self, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        norm = functools.partial(nn.InstanceNorm2d, affine=False)
        seq = [nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * prev, ndf * mult, 4, 2, 1),
                    norm(ndf * mult), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * prev, ndf * mult, 4, 1, 1),
                norm(ndf * mult), nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(ndf * mult, 1, 4, 1, 1)]
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)


class PatchSampleF(nn.Module):
    """Projects sampled patch features through a per-layer 2-layer MLP (lazily
    built once feature dims are known), L2-normalised."""

    def __init__(self, nc: int = 256):
        super().__init__()
        self.nc = nc
        self.mlp_init = False

    def _build(self, feats, device):
        for i, f in enumerate(feats):
            mlp = nn.Sequential(nn.Linear(f.shape[1], self.nc), nn.ReLU(),
                                nn.Linear(self.nc, self.nc)).to(device)
            setattr(self, f"mlp_{i}", mlp)
        self.mlp_init = True

    def forward(self, feats, num_patches=256, patch_ids=None):
        if not self.mlp_init:
            self._build(feats, feats[0].device)
        out_feats, out_ids = [], []
        for i, feat in enumerate(feats):
            B, C, H, W = feat.shape
            flat = feat.permute(0, 2, 3, 1).flatten(1, 2)          # (B, HW, C)
            if patch_ids is not None:
                pid = patch_ids[i]
            else:
                pid = torch.randperm(flat.shape[1], device=feat.device)[
                    :min(num_patches, flat.shape[1])]
            sample = flat[:, pid, :].flatten(0, 1)                  # (B*P, C)
            sample = getattr(self, f"mlp_{i}")(sample)
            out_feats.append(F.normalize(sample, dim=1))
            out_ids.append(pid)
        return out_feats, out_ids


class PatchNCELoss(nn.Module):
    def __init__(self, nce_T: float = 0.07):
        super().__init__()
        self.T = nce_T
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, feat_q, feat_k, num_patches):
        # feat_*: (B*P, dim), row-major so index = b*P + p. Negatives are the
        # other patches WITHIN the same image, so reshape to (B, P, dim).
        N, dim = feat_q.shape
        B = N // num_patches
        feat_k = feat_k.detach()
        l_pos = (feat_q * feat_k).sum(1, keepdim=True)              # (N,1) positives
        q = feat_q.view(B, num_patches, dim)
        k = feat_k.view(B, num_patches, dim)
        l_neg = torch.bmm(q, k.transpose(1, 2))                    # (B,P,P)
        eye = torch.eye(num_patches, device=q.device, dtype=torch.bool)[None]
        l_neg.masked_fill_(eye, -10.0)
        l_neg = l_neg.reshape(N, num_patches)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        return self.ce(logits, torch.zeros(N, dtype=torch.long, device=q.device))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def list_images(d: Path) -> list[str]:
    cand = d / "images" if (d / "images").is_dir() else d
    out = []
    for e in IMG_EXTS:
        out += glob.glob(str(cand / f"*.{e}")) + glob.glob(str(cand / f"*.{e.upper()}"))
    return sorted(set(out))


class UnpairedImages(Dataset):
    def __init__(self, sim_dir: Path, real_dir: Path, load: int, crop: int):
        self.sim = list_images(sim_dir)
        self.real = list_images(real_dir)
        if crop and crop > 0:
            tf = [transforms.Resize(load, antialias=True),
                  transforms.RandomCrop(crop)]
        else:
            # crop<=0 -> FULL RESOLUTION: use the native image (all 1920x1080,
            # both dims divisible by 4, so the generator tiles cleanly). batch
            # must be 1 at this size.
            tf = []
        tf += [transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
        self.tf = transforms.Compose(tf)

    def __len__(self):
        return max(len(self.sim), len(self.real))

    def __getitem__(self, i):
        a = Image.open(self.sim[i % len(self.sim)]).convert("RGB")
        b = Image.open(self.real[random.randrange(len(self.real))]).convert("RGB")
        return self.tf(a), self.tf(b)


# --------------------------------------------------------------------------- #
# CUT model wrapper
# --------------------------------------------------------------------------- #
class CUT:
    def __init__(self, args, device):
        self.args = args
        self.dev = device
        self.amp = getattr(args, "amp", False) and device.type == "cuda"
        self.G = ResnetGenerator().to(device)
        self.D = NLayerDiscriminator().to(device)
        self.F = PatchSampleF(nc=256).to(device)
        init_weights(self.G); init_weights(self.D)
        self.layers = [int(x) for x in args.nce_layers.split(",")]
        self.nce = [PatchNCELoss(args.nce_T).to(device) for _ in self.layers]
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.opt_F = None  # built after F is lazily initialised

        # --- direct distribution loss: MMD in CLIP feature space ----------- #
        # We minimise MMD between G(sim) and real in CLIP features, NOT DINOv2 —
        # DINOv2 is the held-out validator, so we never optimise and grade with
        # the same statistic. CLIP is frozen; gradients flow G -> img -> CLIP.
        self.lambda_mmd = args.lambda_mmd
        self.mmd_real_n = args.mmd_real_n
        self.mmd_sigmas = [0.25, 0.5, 1.0, 2.0, 4.0]   # multi-bandwidth RBF
        self.clip = None
        self.real_clip = None                          # cached real CLIP feats
        if self.lambda_mmd > 0:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32").to(device).eval()
            for p in self.clip.parameters():
                p.requires_grad_(False)
            self._clip_mean = torch.tensor(
                [0.48145466, 0.4578275, 0.40821073], device=device)[None, :, None, None]
            self._clip_std = torch.tensor(
                [0.26862954, 0.26130258, 0.27577711], device=device)[None, :, None, None]

    def gan_loss(self, pred, is_real):  # LSGAN
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return F.mse_loss(pred, target)

    def calc_nce(self, src, tgt):
        feat_k = self.G(src, self.layers, encode_only=True)
        feat_q = self.G(tgt, self.layers, encode_only=True)
        k_pool, ids = self.F(feat_k, self.args.num_patches, None)
        q_pool, _ = self.F(feat_q, self.args.num_patches, ids)
        total = 0.0
        for fq, fk, crit in zip(q_pool, k_pool, self.nce):
            total = total + crit(fq, fk, self.args.num_patches)
        return total / len(self.layers)

    def clip_embed(self, imgs):
        """imgs in [-1,1], (B,3,h,w) -> L2-normalised CLIP image features
        (differentiable; CLIP frozen)."""
        x = (imgs + 1.0) * 0.5
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False)
        x = (x - self._clip_mean) / self._clip_std
        # explicit vision_model + projection (version-robust; get_image_features
        # returns a ModelOutput rather than a tensor in this transformers build)
        pooled = self.clip.vision_model(pixel_values=x).pooler_output
        return F.normalize(self.clip.visual_projection(pooled), dim=1)

    @staticmethod
    def mmd2(x, y, sigmas):
        """Unbiased multi-bandwidth RBF MMD^2 (diagonal dropped). x carries
        grad; y (real) is detached. Bandwidth = median heuristic (detached)."""
        xx = torch.cdist(x, x) ** 2
        yy = torch.cdist(y, y) ** 2
        xy = torch.cdist(x, y) ** 2
        with torch.no_grad():
            med = torch.median(torch.cat([xx.flatten(), yy.flatten(),
                                          xy.flatten()])) + 1e-8
        m, n = x.shape[0], y.shape[0]
        total = 0.0
        for s in sigmas:
            g = 1.0 / (2.0 * s * med)
            Kxx = torch.exp(-g * xx) - torch.diag(torch.diagonal(torch.exp(-g * xx)))
            Kyy = torch.exp(-g * yy) - torch.diag(torch.diagonal(torch.exp(-g * yy)))
            Kxy = torch.exp(-g * xy)
            total = total + (Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1))
                             - 2.0 * Kxy.mean())
        return total / len(sigmas)

    def mmd_loss(self, fake_B):
        if self.clip is None or self.real_clip is None:
            return torch.zeros((), device=fake_B.device)
        styled = self.clip_embed(fake_B)
        n = min(self.mmd_real_n, self.real_clip.shape[0])
        idx = torch.randperm(self.real_clip.shape[0], device=fake_B.device)[:n]
        return self.mmd2(styled, self.real_clip[idx], self.mmd_sigmas)

    def data_dependent_init(self, real_A, real_B):
        """One forward to instantiate F's MLPs, then build its optimizer."""
        with torch.no_grad():
            _ = self.G(real_A)
        self.calc_nce(real_A, self.G(real_A))   # builds F
        self.opt_F = torch.optim.Adam(self.F.parameters(), lr=self.args.lr,
                                      betas=(0.5, 0.999))

    def step(self, real_A, real_B):
        a = self.args
        ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=self.amp)
        with ac:
            real = torch.cat([real_A, real_B], 0) if a.nce_idt else real_A
            fake = self.G(real)
        fake_B = fake[:real_A.size(0)]
        idt_B = fake[real_A.size(0):] if a.nce_idt else None

        # --- D ---
        for p in self.D.parameters():
            p.requires_grad_(True)
        self.opt_D.zero_grad()
        with ac:
            loss_D = 0.5 * (self.gan_loss(self.D(fake_B.detach()), False)
                            + self.gan_loss(self.D(real_B), True))
        loss_D.backward(); self.opt_D.step()

        # --- G + F ---
        for p in self.D.parameters():
            p.requires_grad_(False)
        self.opt_G.zero_grad(); self.opt_F.zero_grad()
        with ac:
            loss_GAN = self.gan_loss(self.D(fake_B), True) * a.lambda_gan
            loss_NCE = self.calc_nce(real_A, fake_B) * a.lambda_nce
            if a.nce_idt:
                loss_NCE = 0.5 * (loss_NCE + self.calc_nce(real_B, idt_B) * a.lambda_nce)
            loss_MMD = self.mmd_loss(fake_B) * a.lambda_mmd
            loss_G = loss_GAN + loss_NCE + loss_MMD
        loss_G.backward(); self.opt_G.step(); self.opt_F.step()
        return {"D": float(loss_D), "G_GAN": float(loss_GAN),
                "NCE": float(loss_NCE), "MMD": float(loss_MMD)}

    @torch.no_grad()
    def translate(self, pil: Image.Image, short: int) -> Image.Image:
        self.G.eval()
        w, h = pil.size
        s = short / min(w, h)
        nw, nh = (round(w * s) // 4) * 4, (round(h * s) // 4) * 4
        x = transforms.functional.to_tensor(pil.resize((nw, nh), Image.BICUBIC))
        x = ((x - 0.5) / 0.5).unsqueeze(0).to(self.dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
            y = self.G(x)
        y = y.float().squeeze(0).clamp(-1, 1).cpu()
        y = (y * 0.5 + 0.5)
        out = transforms.functional.to_pil_image(y).resize((w, h), Image.BICUBIC)
        self.G.train()
        return out


# --------------------------------------------------------------------------- #
# Metric-gated eval (reuse measure_domain_gap)
# --------------------------------------------------------------------------- #
def build_metric(args, device):
    import measure_domain_gap as mdg
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    def embed(files):
        return mdg.embed_set(files, dino, proc, device, 32, "dinov2")

    real_files = list_images(args.data / "test")
    real_emb = embed(real_files)
    sim_files = list_images(args.data / "train")
    rng = random.Random(0)
    eval_sim = rng.sample(sim_files, min(args.eval_n, len(sim_files)))
    return mdg, embed, real_emb, eval_sim


def dump_samples(cut, files, out_dir, epoch, short):
    """Save side-by-side [original | styled] for a FIXED set of sim scenes each
    epoch, so the translation can be watched evolving over training."""
    d = out_dir / "samples" / f"epoch{epoch:03d}"
    d.mkdir(parents=True, exist_ok=True)
    for p in files:
        orig = Image.open(p).convert("RGB")
        styled = cut.translate(orig, short)
        w, h = styled.size
        canvas = Image.new("RGB", (w * 2, h))
        canvas.paste(orig.resize((w, h), Image.BICUBIC), (0, 0))
        canvas.paste(styled, (w, 0))
        canvas.save(d / (Path(p).stem + ".png"))


def run_metric(cut, mdg, embed, real_emb, eval_sim, tmp_dir, device, short):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for f in tmp_dir.glob("*.png"):
        f.unlink()
    files = []
    for p in eval_sim:
        out = cut.translate(Image.open(p).convert("RGB"), short)
        dst = tmp_dir / (Path(p).stem + ".png")
        out.save(dst); files.append(str(dst))
    styled_emb = embed(files)
    mmd = mdg.mmd_rbf(styled_emb, real_emb)
    pad, acc = mdg.proxy_a_distance(styled_emb, real_emb, device)
    return {"MMD": mmd, "PAD": pad, "probe_acc": acc}


# --------------------------------------------------------------------------- #
# Apply: produce the labeled stylized dataset
# --------------------------------------------------------------------------- #
def apply_translation(args, device):
    """Convert EVERY sim image (all requested sim splits, all images) to real
    style, preserving each split's COCO annotations + labels so the output is a
    complete drop-in labeled dataset."""
    cut = CUT(args, device)
    sd = torch.load(args.weights, map_location=device)
    cut.G.load_state_dict(sd)
    cut.G.eval()
    splits = [s.strip() for s in args.apply_splits.split(",") if s.strip()]
    grand = 0
    for split in splits:
        src, dst = args.data / split, args.out / split
        (dst / "images").mkdir(parents=True, exist_ok=True)
        files = list_images(src)
        print(f"[apply] {split}: translating ALL {len(files)} sim images -> "
              f"{dst/'images'} (short edge {args.apply_short}, output resized "
              f"to original WxH so masks stay valid)")
        for i, p in enumerate(files):
            pil = Image.open(p).convert("RGB")
            cut.translate(pil, args.apply_short).save(
                dst / "images" / (Path(p).stem + ".png"))
            if (i + 1) % 200 == 0:
                print(f"\r  {split} {i+1}/{len(files)}", end="", flush=True)
        print()
        grand += len(files)
        # Carry labels over so the stylized split is a drop-in LABELED dataset.
        if (src / "_annotations.coco.json").is_file():
            shutil.copy(src / "_annotations.coco.json", dst / "_annotations.coco.json")
        if (src / "labels").is_dir():
            shutil.copytree(src / "labels", dst / "labels", dirs_exist_ok=True)
    print(f"[apply] done. {grand} stylized images across {splits} under {args.out}")


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("ward_v1"))
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size. MMD needs >1 to estimate a distribution; "
                        "16 @ crop 256 fits and gives a usable MMD per step.")
    p.add_argument("--load", type=int, default=286, help="resize before crop")
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda-gan", type=float, default=1.0)
    p.add_argument("--lambda-nce", type=float, default=1.0)
    p.add_argument("--lambda-mmd", type=float, default=10.0,
                   help="Weight of the CLIP-feature MMD distribution loss "
                        "(0 = plain CUT). MMD^2 is small, so it needs a large "
                        "weight to matter vs GAN/NCE.")
    p.add_argument("--mmd-real-n", type=int, default=128,
                   help="Real CLIP features sampled per step for the MMD loss.")
    p.add_argument("--nce-idt", action="store_true", default=True,
                   help="identity NCE on real (CUT default on; FastCUT off)")
    p.add_argument("--nce-layers", default="0,4,8,12,16")
    p.add_argument("--num-patches", type=int, default=256)
    p.add_argument("--nce-T", type=float, default=0.07)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--iters-per-epoch", type=int, default=1000,
                   help="Cap iters per epoch so metric-gating is fine-grained "
                        "(0 = full sim set = ~4250). Dataset reshuffles each "
                        "epoch, so all sim images are still seen over time.")
    p.add_argument("--log-every", type=int, default=100,
                   help="Print a per-iter progress line every N iters.")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-n", type=int, default=400, help="sim images per metric eval")
    p.add_argument("--eval-short", type=int, default=256)
    p.add_argument("--sample-n", type=int, default=6,
                   help="Fixed sim scenes saved as [original|styled] each epoch.")
    p.add_argument("--sample-every", type=int, default=1)
    p.add_argument("--patience", type=int, default=4,
                   help="stop after this many evals with no probe-acc improvement")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", default=True,
                   help="bf16 autocast (faster + less memory; on by default)")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/cut")
    p.add_argument("--name", default="ward_v1")
    # apply mode
    p.add_argument("--apply", action="store_true", help="translate train -> --out")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("ward_v1_styled"))
    p.add_argument("--apply-splits", default="train,valid",
                   help="Sim splits to convert in --apply mode (every image in "
                        "each). Default train,valid = all sim images.")
    p.add_argument("--apply-short", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    device = (torch.device(f"cuda:{args.device}")
              if torch.cuda.is_available() and args.device != "cpu"
              else torch.device("cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # constant 1920x1080 -> fastest convs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.apply:
        if not args.weights or not args.weights.is_file():
            raise SystemExit("--apply needs --weights pointing at a trained G .pt")
        apply_translation(args, device)
        return

    out_dir = (Path(args.project) / args.name if Path(args.project).is_absolute()
               else PROJECT_ROOT / args.project / args.name)
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)
    print(f"[cut] out: {out_dir}  device={device}")

    ds = UnpairedImages(args.data / "train", args.data / "test", args.load, args.crop)
    print(f"[cut] sim={len(ds.sim)} real={len(ds.real)}  batch={args.batch}")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    cut = CUT(args, device)
    if cut.clip is not None:
        print(f"[cut] caching real CLIP features for the MMD loss "
              f"({len(ds.real)} imgs, lambda_mmd={args.lambda_mmd}) ...")
        tf = transforms.Compose([
            transforms.Resize(args.crop, antialias=True),
            transforms.CenterCrop(args.crop),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
        feats = []
        with torch.no_grad():
            for i in range(0, len(ds.real), 32):
                b = torch.stack([tf(Image.open(p).convert("RGB"))
                                 for p in ds.real[i:i + 32]]).to(device)
                feats.append(cut.clip_embed(b))
        cut.real_clip = torch.cat(feats, 0)
        print(f"[cut] real CLIP features: {tuple(cut.real_clip.shape)}")
    print("[cut] building metric (DINOv2) + caching real embeddings ...")
    mdg, embed, real_emb, eval_sim = build_metric(args, device)
    # fixed set of scenes to visualise each epoch (seed != eval, varied scenes)
    sample_files = random.Random(1).sample(ds.sim, min(args.sample_n, len(ds.sim)))

    csv_path = out_dir / "metrics.csv"
    fields = ["epoch", "loss_D", "loss_G_GAN", "loss_NCE", "loss_MMD",
              "val_MMD", "PAD", "probe_acc"]
    rows = []
    best_acc, since_improve = 1.0, 0
    init_done = False
    epoch_len = (min(len(loader), args.iters_per_epoch) if args.iters_per_epoch
                 else len(loader))

    for epoch in range(1, args.max_epochs + 1):
        agg = {"D": 0.0, "G_GAN": 0.0, "NCE": 0.0, "MMD": 0.0}
        n = 0
        t0 = time.time()
        for a, b in loader:
            if args.iters_per_epoch and n >= args.iters_per_epoch:
                break
            a, b = a.to(device), b.to(device)
            if not init_done:
                cut.data_dependent_init(a, b)
                init_done = True
            losses = cut.step(a, b)
            for k in agg:
                agg[k] += losses[k]
            n += 1
            if n % args.log_every == 0:
                ips = n / (time.time() - t0)
                eta = (epoch_len - n) / max(ips, 1e-9)
                print(f"\r  epoch {epoch} {n}/{epoch_len}  {ips:4.1f} it/s  "
                      f"D={agg['D']/n:.3f} G={agg['G_GAN']/n:.3f} "
                      f"NCE={agg['NCE']/n:.3f} MMD={agg['MMD']/n:.4f}  "
                      f"ETA {eta:4.0f}s", end="", flush=True)
        print()
        agg = {k: v / max(n, 1) for k, v in agg.items()}
        print(f"[cut] epoch {epoch}/{args.max_epochs}  "
              f"D={agg['D']:.3f} G_GAN={agg['G_GAN']:.3f} NCE={agg['NCE']:.3f} "
              f"MMD={agg['MMD']:.4f} ({time.time()-t0:.0f}s)", flush=True)

        if epoch % args.sample_every == 0:
            dump_samples(cut, sample_files, out_dir, epoch, args.eval_short)

        if epoch % args.eval_every == 0 or epoch == args.max_epochs:
            m = run_metric(cut, mdg, embed, real_emb, eval_sim,
                           out_dir / "styled_eval", device, args.eval_short)
            print(f"[cut][metric@{epoch}] val_MMD(DINOv2)={m['MMD']:.4f} "
                  f"PAD={m['PAD']:.3f} probe_acc={m['probe_acc']:.3f}  "
                  f"(baseline sim->real MMD 0.091/probe 1.000; floor ~0/0.5)",
                  flush=True)
            rows.append({"epoch": epoch, "loss_D": round(agg["D"], 4),
                         "loss_G_GAN": round(agg["G_GAN"], 4),
                         "loss_NCE": round(agg["NCE"], 4),
                         "loss_MMD": round(agg["MMD"], 4),
                         "val_MMD": round(m["MMD"], 4), "PAD": round(m["PAD"], 4),
                         "probe_acc": round(m["probe_acc"], 4)})
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
            torch.save(cut.G.state_dict(), out_dir / "weights" / "last_G.pt")
            if m["probe_acc"] < best_acc - 1e-3:
                best_acc, since_improve = m["probe_acc"], 0
                torch.save(cut.G.state_dict(), out_dir / "weights" / "best_G.pt")
                print(f"[cut] new best probe_acc={best_acc:.3f} -> best_G.pt")
            else:
                since_improve += 1
                if since_improve >= args.patience:
                    print(f"[cut] probe_acc plateaued ({args.patience} evals) — "
                          f"early stop at epoch {epoch}. best={best_acc:.3f}")
                    break
    print(f"[cut] done. metrics -> {csv_path}; best_G -> {out_dir/'weights'/'best_G.pt'}")
    print(f"[cut] next: --apply --weights {out_dir/'weights'/'best_G.pt'} "
          f"--out ward_v1_styled")


if __name__ == "__main__":
    main()
