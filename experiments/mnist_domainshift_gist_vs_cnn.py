"""
Controlled sim->real probe for the Gist-First Network idea
(see docs/global-first-architecture.md). The aligned version of the MNIST test:
inject a *texture domain shift with preserved digit semantics*, the cheap analog
of the ward synthetic->real gap.

Both domains use the SAME generative process -- an MNIST-M-style absolute-
difference composite of the digit onto a background -- differing ONLY in the
background:
  * sim   (train): background is a FLAT random color (no spatial texture).
  * real  (test) : background is spatial color TEXTURE + noise.
So digit shape/identity is invariant; only the low-level appearance shifts. A
model that commits to global shape first should keep more accuracy on `real`.

Train on sim only. Report sim acc (in-domain), real acc (shifted) and the gap
for: CNN (bottom-up baseline), Gist-First (gate+aux), and the same net with
gate/aux off (plain tiny ViT).

    /home/edge-host/Documents/.venv/bin/python \
        experiments/mnist_domainshift_gist_vs_cnn.py --epochs 8 --device 0
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


def make_domain(x_gray: torch.Tensor, domain: str) -> torch.Tensor:
    """x_gray: (B,1,28,28) in [0,1] -> standardized (B,3,28,28) RGB composite.

    out = |digit - background|  (MNIST-M blend). `sim` uses a flat random color
    background; `real` uses spatial color texture + noise. Semantics (digit
    shape) identical; only background texture differs."""
    B, _, H, W = x_gray.shape
    dev = x_gray.device
    digit = x_gray.expand(-1, 3, -1, -1)
    if domain == "sim":
        bg = torch.rand(B, 3, 1, 1, device=dev).expand(-1, -1, H, W)
    elif domain == "real":
        low = torch.rand(B, 3, 7, 7, device=dev)                 # low-freq color field
        bg = F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)
        bg = (bg + 0.15 * torch.randn(B, 3, H, W, device=dev)).clamp(0, 1)
    else:
        raise ValueError(domain)
    out = (digit - bg).abs()
    return (out - MEAN.to(dev)) / STD.to(dev)


# --------------------------------------------------------------------------- #
# Models (3-channel input)
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
        return self.head(self.net(x)), None


class GistFirstClassifier(nn.Module):
    def __init__(self, n_classes: int = 10, dim: int = 64, heads: int = 4,
                 num_latents: int = 16, patch: int = 4, use_gate: bool = True):
        super().__init__()
        self.use_gate = use_gate
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        n_tok = (28 // patch) ** 2
        self.pos = nn.Parameter(torch.randn(1, n_tok, dim) * 0.02)
        self.encoder = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=4 * dim, batch_first=True, norm_first=True)
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.gist_cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gist_norm = nn.LayerNorm(dim)
        self.gist_self = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=4 * dim, batch_first=True, norm_first=True)
        self.aux_head = nn.Linear(dim, n_classes)
        self.film = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.refine = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=4 * dim, batch_first=True, norm_first=True)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_classes))

    def forward(self, x):
        t = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos
        t = self.encoder(t)
        B = t.shape[0]
        q = self.latents.unsqueeze(0).expand(B, -1, -1)
        z, _ = self.gist_cross(q, t, t, need_weights=False)
        z = self.gist_self(self.gist_norm(z + q))
        g = z.mean(dim=1)
        aux_logits = self.aux_head(g)
        if self.use_gate:
            gamma, beta = self.film(g).chunk(2, dim=-1)
            t = (1 + gamma).unsqueeze(1) * t + beta.unsqueeze(1)
        t = self.refine(t)
        return self.head(t.mean(dim=1)), aux_logits


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, domain: str, seed: int = 1234) -> float:
    model.eval()
    torch.manual_seed(seed)  # same random backgrounds for every model -> fair
    correct = total = 0
    for x, y in loader:
        x = make_domain(x.to(device), domain)
        y = y.to(device)
        logits, _ = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def train_model(model, train_loader, test_loader, device, epochs, aux_weight, tag):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for x, y in train_loader:
            x = make_domain(x.to(device), "sim")           # train on SIM only
            y = y.to(device)
            opt.zero_grad()
            logits, aux = model(x)
            loss = F.cross_entropy(logits, y)
            if aux is not None and aux_weight > 0:
                loss = loss + aux_weight * F.cross_entropy(aux, y)
            loss.backward()
            opt.step()
            run += loss.item()
        sim = evaluate(model, test_loader, device, "sim")
        real = evaluate(model, test_loader, device, "real")
        print(f"[{tag}] epoch {ep}/{epochs} loss={run/len(train_loader):.4f} "
              f"sim={sim:.2f}% real={real:.2f}% gap={sim-real:.2f}")
    return {"tag": tag, "params": n_params,
            "sim": evaluate(model, test_loader, device, "sim"),
            "real": evaluate(model, test_loader, device, "real")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default="0")
    p.add_argument("--data", default="./data")
    p.add_argument("--aux-weight", type=float, default=0.5)
    args = p.parse_args()

    torch.manual_seed(0)
    device = (torch.device(f"cuda:{args.device}")
              if torch.cuda.is_available() and args.device != "cpu"
              else torch.device("cpu"))

    tf = transforms.ToTensor()  # raw [0,1]; colorization happens in make_domain
    train_ds = datasets.MNIST(args.data, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(args.data, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, args.batch, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, 512, shuffle=False, num_workers=4)

    results = []
    print("=== Small CNN (bottom-up baseline) ===")
    results.append(train_model(SmallCNN(), train_loader, test_loader, device,
                               args.epochs, 0.0, "cnn"))
    print("\n=== Gist-First (gate + aux) ===")
    results.append(train_model(GistFirstClassifier(use_gate=True), train_loader,
                               test_loader, device, args.epochs, args.aux_weight, "gist"))
    print("\n=== Gist-First (gate OFF, aux OFF = plain tiny ViT) ===")
    results.append(train_model(GistFirstClassifier(use_gate=False), train_loader,
                               test_loader, device, args.epochs, 0.0, "vit"))

    print("\n================ SUMMARY (train on sim only) ================")
    print(f"{'model':6s} {'params':>9s} {'sim':>8s} {'real':>8s} {'gap':>7s}")
    for r in results:
        print(f"{r['tag']:6s} {r['params']:>9,d} {r['sim']:>7.2f}% "
              f"{r['real']:>7.2f}% {r['sim']-r['real']:>6.2f}")
    print("\nThe thesis predicts: gist has the SMALLEST sim->real gap.")


if __name__ == "__main__":
    main()
