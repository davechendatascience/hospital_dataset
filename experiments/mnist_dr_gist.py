"""
Corrected Gist-First probe (see docs/global-first-architecture.md).

The previous single-source run showed the toy was UNFAITHFUL: it stripped out the
domain-robust feature source and gave the gist nothing invariant to read, and the
FiLM gate was too weak to be genuinely top-down. This version fixes all three
issues that critique identified:

  1. DOMAIN RANDOMIZATION. Train on K random textures, test on HELD-OUT textures
     (same generative family, disjoint samples). Sweep K to measure how few
     textures each architecture needs to generalize -- the real question is
     "sample-efficiency of texture-invariance", not "beat a CNN on flat->textured".
  2. REAL TOP-DOWN OPERATOR. The gist conditions the *queries* of a cross-attention
     readout into the features (gist-conditioned cross-attention), instead of a
     channel-global FiLM affine. The gist now steers what each readout pulls.
  3. GIST-CONSISTENCY LOSS. Two differently-textured views of the same digit must
     produce the same gist -> explicit pressure for texture-invariant semantics.

Models compared at each K: cnn (bottom-up prior), vit (plain), gfn (corrected),
gfn_nocons (gfn with consistency off -> isolates the loss vs the architecture).

Hypothesis: gfn reaches high held-out accuracy at SMALLER K than vit; gfn vs
gfn_nocons shows whether the consistency loss is the active ingredient.

    /home/edge-host/Documents/.venv/bin/python experiments/mnist_dr_gist.py \
        --epochs 6 --ks 1,4,16 --device 0
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# Texture banks + compositing (MNIST-M-style |digit - bg|)
# --------------------------------------------------------------------------- #
def gen_textures(n: int, seed: int) -> torch.Tensor:
    """n reproducible procedural color textures (n,3,28,28) in [0,1]: a sum of
    two upsampled random color fields (low + mid frequency)."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        low = torch.rand(1, 3, 4, 4, generator=gen)
        mid = torch.rand(1, 3, 8, 8, generator=gen)
        l = F.interpolate(low, 28, mode="bilinear", align_corners=False)[0]
        m = F.interpolate(mid, 28, mode="bilinear", align_corners=False)[0]
        out.append((0.6 * l + 0.4 * m).clamp(0, 1))
    return torch.stack(out)


def composite(x_gray: torch.Tensor, bank: torch.Tensor, noise: float = 0.05) -> torch.Tensor:
    """Composite digits onto random textures drawn from `bank`, then standardize."""
    B = x_gray.shape[0]
    dev = x_gray.device
    idx = torch.randint(0, bank.shape[0], (B,), device=dev)
    bg = bank[idx]
    out = (x_gray.expand(-1, 3, -1, -1) - bg).abs()
    if noise > 0:
        out = (out + noise * torch.randn_like(out)).clamp(0, 1)
    return (out - MEAN.to(dev)) / STD.to(dev)


# --------------------------------------------------------------------------- #
# Models (all return (logits, aux_or_None, gist_or_None))
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.25),
                                  nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
                                  nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.net(x)), None, None


class PlainViT(nn.Module):
    def __init__(self, n_classes: int = 10, dim: int = 64, heads: int = 4,
                 depth: int = 3, patch: int = 4):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        n_tok = (28 // patch) ** 2
        self.pos = nn.Parameter(torch.randn(1, n_tok, dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, 4 * dim, batch_first=True,
                                       norm_first=True) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_classes))

    def forward(self, x):
        t = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos
        for b in self.blocks:
            t = b(t)
        return self.head(t.mean(1)), None, None


class CrossAttnBlock(nn.Module):
    """Queries q read key/value kv (cross-attention) + FFN, norm-first."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.nq = nn.LayerNorm(dim)
        self.nkv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))

    def forward(self, q, kv):
        kv = self.nkv(kv)
        q = q + self.attn(self.nq(q), kv, kv, need_weights=False)[0]
        return q + self.ffn(self.n2(q))


class GistFirstV2(nn.Module):
    """Corrected GFN: features -> global gist -> gist-CONDITIONED cross-attention
    readout (real top-down). gist is also supervised (aux) and, in training, made
    texture-invariant via a consistency loss (handled in the train loop)."""

    def __init__(self, n_classes: int = 10, dim: int = 64, heads: int = 4,
                 num_latents: int = 16, patch: int = 4, readout_depth: int = 2):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        n_tok = (28 // patch) ** 2
        self.pos = nn.Parameter(torch.randn(1, n_tok, dim) * 0.02)
        self.encoder = nn.TransformerEncoderLayer(
            dim, heads, 4 * dim, batch_first=True, norm_first=True)
        # gist (Perceiver bottleneck)
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.gist_cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gist_norm = nn.LayerNorm(dim)
        self.gist_self = nn.TransformerEncoderLayer(
            dim, heads, 4 * dim, batch_first=True, norm_first=True)
        self.gist_pool = nn.LayerNorm(dim)
        self.aux_head = nn.Linear(dim, n_classes)
        # gist-conditioned top-down readout
        self.readout_pos = nn.Parameter(torch.randn(1, n_tok, dim) * 0.02)
        self.g_to_query = nn.Linear(dim, dim)
        self.readout = nn.ModuleList(
            [CrossAttnBlock(dim, heads) for _ in range(readout_depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_classes))

    def _features(self, x):
        t = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos
        return self.encoder(t)

    def _gist(self, feats):
        q = self.latents.unsqueeze(0).expand(feats.shape[0], -1, -1)
        z, _ = self.gist_cross(q, feats, feats, need_weights=False)
        z = self.gist_self(self.gist_norm(z + q))
        return self.gist_pool(z.mean(1))

    def gist_only(self, x):
        return self._gist(self._features(x))

    def forward(self, x):
        feats = self._features(x)                       # F (bottom-up)
        g = self._gist(feats)                            # global gist FIRST
        aux = self.aux_head(g)
        # gist conditions the readout queries -> top-down reads of F
        q = self.readout_pos.expand(x.shape[0], -1, -1) + self.g_to_query(g).unsqueeze(1)
        for blk in self.readout:
            q = blk(q, feats)
        return self.head(q.mean(1)), aux, g


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, bank, device, seed: int = 7) -> float:
    model.eval()
    torch.manual_seed(seed)                              # same held-out draws for all models
    correct = total = 0
    for x, y in loader:
        x = composite(x.to(device), bank)
        y = y.to(device)
        logits, _, _ = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def train_model(model, kind, train_loader, test_loader, train_bank, test_bank,
                device, epochs, aux_w, cons_w):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    use_cons = (kind == "gfn") and cons_w > 0
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            v1 = composite(x, train_bank)
            opt.zero_grad()
            logits, aux, g1 = model(v1)
            loss = F.cross_entropy(logits, y)
            if aux is not None and aux_w > 0:
                loss = loss + aux_w * F.cross_entropy(aux, y)
            if use_cons:                                  # second textured view, match gists
                g2 = model.gist_only(composite(x, train_bank))
                loss = loss + cons_w * F.mse_loss(
                    F.normalize(g1, dim=-1), F.normalize(g2, dim=-1))
            loss.backward()
            opt.step()
            run += loss.item()
    return evaluate(model, test_loader, test_bank, device)   # held-out texture acc


def make_model(kind):
    return {"cnn": lambda: SmallCNN(),
            "vit": lambda: PlainViT(),
            "gfn": lambda: GistFirstV2(),
            "gfn_nocons": lambda: GistFirstV2()}[kind]()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--ks", default="1,4,16", help="comma-separated K values to sweep")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default="0")
    p.add_argument("--data", default="./data")
    p.add_argument("--aux-weight", type=float, default=0.5)
    p.add_argument("--cons-weight", type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(0)
    device = (torch.device(f"cuda:{args.device}")
              if torch.cuda.is_available() and args.device != "cpu"
              else torch.device("cpu"))
    ks = [int(k) for k in args.ks.split(",")]

    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(args.data, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(args.data, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, args.batch, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, 512, shuffle=False, num_workers=4)

    test_bank = gen_textures(64, seed=99999).to(device)   # held-out, shared by all
    kinds = ["cnn", "vit", "gfn_nocons", "gfn"]
    table = {kind: {} for kind in kinds}

    for K in ks:
        train_bank = gen_textures(K, seed=1000 + K).to(device)
        for kind in kinds:
            torch.manual_seed(0)                          # same init seed per cell
            model = make_model(kind)
            cw = 0.0 if kind == "gfn_nocons" else args.cons_weight
            acc = train_model(model, "gfn" if kind.startswith("gfn") else kind,
                              train_loader, test_loader, train_bank, test_bank,
                              device, args.epochs, args.aux_weight, cw)
            n = sum(pp.numel() for pp in model.parameters())
            table[kind][K] = acc
            print(f"[K={K:<3d}] {kind:11s} params={n:>8,d} held-out_acc={acc:.2f}%",
                  flush=True)

    print("\n=========== HELD-OUT TEXTURE ACCURACY vs K (train on sim only) ===========")
    hdr = "K".ljust(5) + "".join(f"{k:>12s}" for k in kinds)
    print(hdr)
    for K in ks:
        print(f"{K:<5d}" + "".join(f"{table[k][K]:>11.2f}%" for k in kinds))
    print("\nRead: which model climbs to high held-out accuracy at the SMALLEST K.")
    print("gfn vs vit -> architecture; gfn vs gfn_nocons -> consistency loss.")


if __name__ == "__main__":
    main()
