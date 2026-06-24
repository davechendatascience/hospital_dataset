"""Measure the sim->real DISTRIBUTION gap between whole image sets.

Per-image style transfer (AdaIN in style_transfer_ward.py) matches one real
image's channel stats at a time — it never tells you how far the *distribution*
of sim (or stylized-sim) images is from the *distribution* of real images. This
module answers that, over the whole dataset, in the feature space the GFN
actually consumes (frozen DINOv2 CLS embeddings; --feature clip for an
Inception/CLIP-comparable number).

For each named image set it computes a pooled DINOv2 embedding per image, then
between sets reports three complementary distribution distances:

  * MMD   — unbiased RBF maximum-mean-discrepancy (robust on small sets, no
            Gaussian assumption).
  * FD    — Fréchet distance (FID-style: ||mu1-mu2||^2 + Tr(C1+C2-2 sqrt(C1 C2)))
            but on DINOv2 features, so it's comparable across our runs.
  * PAD   — proxy-A-distance 2(1-2e): a linear probe's sim-vs-real error e;
            the most interpretable ("a probe still tells them apart 88% of the
            time").

Headline: how much a translation closes the gap, D(sim,real) vs
D(styled,real), with D(real_a, real_b) as the irreducible real-vs-real floor.

NOTHING here is "trained" in the heavy sense: it's one DINOv2 forward pass over
the images + closed-form stats + a few-second logistic probe. (The thing with
real training time is upgrading the *transfer* itself — CUT/CycleGAN — which is
separate.)

    /home/edge-host/Documents/.venv/bin/python measure_domain_gap.py --data ward_v1
    # custom sets:
    ... --set sim=train --set real=test --set styled=../ward_v1_styled/train
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMG_EXTS = ("png", "jpg", "jpeg", "bmp", "webp")

# ward_v1 default comparison: the two LABELED splits we actually train/eval on
# — sim train vs real test. Add a styled set (--set styled=...) once it exists;
# the CycleGAN/CUT translator is trained on these same labeled splits.
WARD_V1_SETS = {
    "sim":  "train",   # 4250 synthetic renders (labeled)
    "real": "test",    # 728 real photos (labeled)
}


def list_images(path: Path) -> list[str]:
    """Accept a split dir (uses <dir>/images if present), or a raw image dir,
    recursively."""
    cand = path / "images" if (path / "images").is_dir() else path
    files: list[str] = []
    for ext in IMG_EXTS:
        files += glob.glob(str(cand / "**" / f"*.{ext}"), recursive=True)
        files += glob.glob(str(cand / "**" / f"*.{ext.upper()}"), recursive=True)
    return sorted(set(files))


# --------------------------------------------------------------------------- #
# Embedding (frozen DINOv2 CLS, or CLIP image embedding)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed_set(files: list[str], model, processor, device, batch: int,
              feature: str) -> np.ndarray:
    embs = []
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        imgs = [Image.open(f).convert("RGB") for f in chunk]
        if feature == "clip":
            inp = processor(images=imgs, return_tensors="pt").to(device)
            e = model.get_image_features(**inp)
        else:  # dinov2: pooled CLS token (Dinov2Model.pooler_output)
            inp = processor(images=imgs, return_tensors="pt").to(device)
            out = model(**inp)
            e = out.pooler_output if getattr(out, "pooler_output", None) is not None \
                else out.last_hidden_state[:, 0]
        embs.append(e.float().cpu().numpy())
        print(f"\r  embedded {min(i + batch, len(files))}/{len(files)}",
              end="", flush=True)
    print()
    return np.concatenate(embs, 0)


def load_or_embed(name: str, files: list[str], cache_dir: Path, tag: str,
                  embed_fn) -> np.ndarray:
    key = hashlib.md5((tag + "|" + "|".join(files)).encode()).hexdigest()[:12]
    cache = cache_dir / f"{name}_{tag}_{key}.npy"
    if cache.is_file():
        print(f"[{name}] cache hit ({cache.name})")
        return np.load(cache)
    print(f"[{name}] embedding {len(files)} images ...")
    emb = embed_fn(files)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    return emb


# --------------------------------------------------------------------------- #
# Distribution distances
# --------------------------------------------------------------------------- #
def _rbf_gram(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return np.exp(-gamma * np.maximum(d2, 0))


def mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased RBF-MMD^2 with the median-heuristic bandwidth on the pooled
    pairwise sq-distances of a subsample (so it's scale-free)."""
    z = np.concatenate([x, y], 0)
    m = min(len(z), 1000)
    idx = np.random.default_rng(0).choice(len(z), m, replace=False)
    s = z[idx]
    d2 = (s * s).sum(1)[:, None] + (s * s).sum(1)[None, :] - 2 * s @ s.T
    med = np.median(d2[d2 > 0])
    gamma = 1.0 / (med + 1e-12)
    nx, ny = len(x), len(y)
    kxx = _rbf_gram(x, x, gamma); np.fill_diagonal(kxx, 0)
    kyy = _rbf_gram(y, y, gamma); np.fill_diagonal(kyy, 0)
    kxy = _rbf_gram(x, y, gamma)
    return float(kxx.sum() / (nx * (nx - 1)) + kyy.sum() / (ny * (ny - 1))
                 - 2 * kxy.mean())


def frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import linalg
    mu1, mu2 = x.mean(0), y.mean(0)
    c1 = np.cov(x, rowvar=False)
    c2 = np.cov(y, rowvar=False)
    covmean, _ = linalg.sqrtm(c1 @ c2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1 + c2 - 2 * covmean))


def proxy_a_distance(x: np.ndarray, y: np.ndarray, device, seed: int = 0) -> float:
    """Train a logistic probe to tell set x (label 0) from set y (label 1) on a
    train half, measure error e on a held-out half; PAD = 2(1 - 2e)."""
    rng = np.random.default_rng(seed)
    # Balance class sizes so chance = 50%; otherwise (e.g. 400 styled vs 728
    # real) a probe can score "well" just by exploiting the size imbalance and
    # the PAD becomes meaningless near the matched-distribution floor.
    n = min(len(x), len(y))
    x = x[rng.choice(len(x), n, replace=False)]
    y = y[rng.choice(len(y), n, replace=False)]
    feats = np.concatenate([x, y], 0).astype(np.float32)
    labels = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.float32)
    # standardize (probe should use distribution shape, not raw feature scale)
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    perm = rng.permutation(len(feats))
    feats, labels = feats[perm], labels[perm]
    half = len(feats) // 2
    Xtr = torch.tensor(feats[:half], device=device)
    ytr = torch.tensor(labels[:half], device=device)
    Xte = torch.tensor(feats[half:], device=device)
    yte = torch.tensor(labels[half:], device=device)
    clf = torch.nn.Linear(feats.shape[1], 1).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(300):
        opt.zero_grad()
        loss = lossf(clf(Xtr).squeeze(1), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = (clf(Xte).squeeze(1) > 0).float()
        err = (pred != yte).float().mean().item()
    err = min(err, 0.5)
    return 2.0 * (1.0 - 2.0 * err), 1.0 - err  # (PAD, probe accuracy)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("ward_v1"),
                    help="Dataset root; split names in --set resolve under it.")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=PATH",
                    help="A named image set. PATH is a split under --data or an "
                         "absolute dir. Repeat. Default: the ward_v1 preset.")
    ap.add_argument("--feature", choices=["dinov2", "clip"], default="dinov2")
    ap.add_argument("--model", default=None,
                    help="HF id (default: dinov2-base, or clip-vit-base-patch32)")
    ap.add_argument("--compare-to", default="real",
                    help="Set name every other set is compared against.")
    ap.add_argument("--gap-base", default="sim",
                    help="Set whose gap defines 100%% (for the closed-fraction).")
    ap.add_argument("--max-per-set", type=int, default=0,
                    help="Cap images per set (0 = all). Use a small value for a "
                         "quick check; full sets for the real number.")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--cache-dir", type=Path, default=Path("runs/domain_gap/cache"))
    args = ap.parse_args()

    sets = dict(s.split("=", 1) for s in args.set) if args.set else dict(WARD_V1_SETS)
    device = (torch.device(f"cuda:{args.device}")
              if torch.cuda.is_available() and args.device != "cpu"
              else torch.device("cpu"))

    # resolve files per set (skip empty/missing — e.g. styled before generation)
    resolved: dict[str, list[str]] = {}
    for name, p in sets.items():
        path = Path(p) if Path(p).is_absolute() else (args.data / p)
        files = list_images(path)
        if args.max_per_set > 0:
            files = files[:args.max_per_set]
        if not files:
            print(f"[{name}] no images at {path} — skipping")
            continue
        resolved[name] = files
        print(f"[{name}] {len(files)} images <- {path}")
    if args.compare_to not in resolved:
        raise SystemExit(f"--compare-to '{args.compare_to}' has no images")

    # model
    model_id = args.model or ("openai/clip-vit-base-patch32" if args.feature == "clip"
                              else "facebook/dinov2-base")
    print(f"[model] {args.feature}: {model_id} on {device}")
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(model_id)
    if args.feature == "clip":
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained(model_id).to(device).eval()
    else:
        model = AutoModel.from_pretrained(model_id).to(device).eval()
    tag = f"{args.feature}_{Path(model_id).name}_n{args.max_per_set}"

    embs = {name: load_or_embed(
                name, files, args.cache_dir, tag,
                lambda fs: embed_set(fs, model, processor, device, args.batch,
                                     args.feature))
            for name, files in resolved.items()}

    ref = args.compare_to
    print(f"\n=== distribution distance to '{ref}' "
          f"(feature={args.feature}, lower = closer) ===")
    print(f"{'set':<12}{'MMD':>12}{'Frechet':>12}{'PAD':>10}{'probe_acc':>11}")
    dist = {}
    for name, e in embs.items():
        if name == ref:
            continue
        mmd = mmd_rbf(e, embs[ref])
        fd = frechet_distance(e, embs[ref])
        pad, acc = proxy_a_distance(e, embs[ref], device)
        dist[name] = {"MMD": mmd, "Frechet": fd, "PAD": pad}
        print(f"{name:<12}{mmd:>12.4f}{fd:>12.3f}{pad:>10.3f}{acc:>11.3f}")

    # how much of the sim->real gap a translation closes
    base = args.gap_base
    if base in dist:
        print(f"\n=== fraction of the '{base}'->'{ref}' gap closed "
              f"(1 - D(set,{ref}) / D({base},{ref})) ===")
        for name, d in dist.items():
            if name == base:
                continue
            fr = {k: (1 - d[k] / dist[base][k]) if dist[base][k] else float("nan")
                  for k in d}
            print(f"  {name:<12} " + "  ".join(f"{k}:{v:+.2%}" for k, v in fr.items()))
        print(f"\n(real-vs-real sets, if present, are the irreducible floor — "
              f"D should be ~0 there; a styled set wants to approach that floor.)")


if __name__ == "__main__":
    main()
