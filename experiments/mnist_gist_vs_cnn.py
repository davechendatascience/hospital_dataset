"""
MNIST go/no-go for the Gist-First Network idea (see docs/global-first-architecture.md).

This is NOT the sim2real test -- MNIST has no domain gap. It answers two cheap
questions before we spend hours on the ward run:
  1. Does the global-first mechanism (gist computed first -> FiLM-gates the
     lower-level features -> refine -> classify) actually train, and does it
     reach small-CNN accuracy on a clean task? (capacity / trainability)
  2. As a *preview* of the real thesis, when we shift the test distribution
     (invert contrast, add noise) -- something the model never saw in training --
     does committing to a global gist first degrade less than a plain CNN?

Both models are small and size-matched. Train on clean MNIST only; report clean
accuracy plus two never-seen shifts.

    /home/edge-host/Documents/.venv/bin/python experiments/mnist_gist_vs_cnn.py \
        --epochs 3 --device 0
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    """Standard bottom-up baseline: conv -> conv -> pool -> fc."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                  # 28 -> 14
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                  # 14 -> 7
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.25),
                                  nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
                                  nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.net(x)), None  # (logits, aux=None)


class GistFirstClassifier(nn.Module):
    """Faithful miniature of GistFirstNetwork for classification.

    patch-embed conv -> thin token encoder -> K latents cross-attend tokens into
    a global gist g (supervised by an aux head = global-first signal) -> g
    FiLM-gates the tokens -> a refine block reads the gated tokens -> pool ->
    classify. With gate/aux off it degenerates to a plain tiny ViT.
    """

    def __init__(self, n_classes: int = 10, dim: int = 64, heads: int = 4,
                 num_latents: int = 16, patch: int = 4, use_gate: bool = True):
        super().__init__()
        self.use_gate = use_gate
        self.patch_embed = nn.Conv2d(1, dim, patch, patch)    # 28 -> 7x7 tokens
        n_tok = (28 // patch) ** 2
        self.pos = nn.Parameter(torch.randn(1, n_tok, dim) * 0.02)
        self.encoder = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=4 * dim, batch_first=True, norm_first=True)

        # gist: Perceiver-style global bottleneck
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.gist_cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gist_norm = nn.LayerNorm(dim)
        self.gist_self = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=4 * dim, batch_first=True, norm_first=True)
        self.aux_head = nn.Linear(dim, n_classes)

        # FiLM gate from gist (zero-init -> starts as identity)
        self.film = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        # top-down refine over gated tokens, then classify
        self.refine = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=4 * dim, batch_first=True, norm_first=True)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_classes))

    def forward(self, x):
        t = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos  # (B,N,D)
        t = self.encoder(t)

        # gist computed FIRST
        B = t.shape[0]
        q = self.latents.unsqueeze(0).expand(B, -1, -1)
        z, _ = self.gist_cross(q, t, t, need_weights=False)
        z = self.gist_self(self.gist_norm(z + q))
        g = z.mean(dim=1)                                              # (B,D)
        aux_logits = self.aux_head(g)

        # gist gates the lower-level tokens (global-first -> local-later)
        if self.use_gate:
            gamma, beta = self.film(g).chunk(2, dim=-1)
            t = (1 + gamma).unsqueeze(1) * t + beta.unsqueeze(1)

        t = self.refine(t)
        logits = self.head(t.mean(dim=1))
        return logits, aux_logits


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def shift(x: torch.Tensor, kind: str) -> torch.Tensor:
    """Distribution shifts the model never saw in training (inputs are in the
    standardized space; operate then re-standardize-ish by clamping range)."""
    if kind == "clean":
        return x
    if kind == "invert":          # contrast inversion (white digit on black -> black on white)
        return -x
    if kind == "noise":           # additive Gaussian
        return x + 0.8 * torch.randn_like(x)
    raise ValueError(kind)


@torch.no_grad()
def evaluate(model, loader, device, kind: str) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(shift(x, kind))
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
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, aux = model(x)
            loss = F.cross_entropy(logits, y)
            if aux is not None and aux_weight > 0:
                loss = loss + aux_weight * F.cross_entropy(aux, y)
            loss.backward()
            opt.step()
            run += loss.item()
        acc = evaluate(model, test_loader, device, "clean")
        print(f"[{tag}] epoch {ep}/{epochs} loss={run/len(train_loader):.4f} "
              f"clean_acc={acc:.2f}%")
    return {
        "tag": tag, "params": n_params,
        "clean": evaluate(model, test_loader, device, "clean"),
        "invert": evaluate(model, test_loader, device, "invert"),
        "noise": evaluate(model, test_loader, device, "noise"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default="0")
    p.add_argument("--data", default="./data")
    p.add_argument("--aux-weight", type=float, default=0.5)
    args = p.parse_args()

    torch.manual_seed(0)
    device = (torch.device(f"cuda:{args.device}")
              if torch.cuda.is_available() and args.device != "cpu"
              else torch.device("cpu"))

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
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

    print("\n================ SUMMARY ================")
    print(f"{'model':6s} {'params':>9s} {'clean':>8s} {'invert':>8s} {'noise':>8s}")
    for r in results:
        print(f"{r['tag']:6s} {r['params']:>9,d} {r['clean']:>7.2f}% "
              f"{r['invert']:>7.2f}% {r['noise']:>7.2f}%")
    print("\nclean = capacity check (should reach ~99% like the CNN)")
    print("invert/noise = unseen shifts; preview of the global-first robustness thesis")


if __name__ == "__main__":
    main()
