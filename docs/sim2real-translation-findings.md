# Findings: pixel-level sim→real translation does not close the frozen-feature gap

**Date:** 2026-06-08
**Context:** Attempt to shrink the ward sim→real gap by translating sim renders
to look real (so the GFN / detectors train on real-looking data). Measured the
gap as an **unbiased MMD²** + a **balanced classifier-two-sample probe** on
frozen-encoder embeddings (`measure_domain_gap.py`); translators in `train_cut.py`.

## Metric anchors (ward_v1, sim=train 4250, real=test 728)

| feature space | floor (real-vs-real) | ceiling (sim-vs-real, no translation) |
|---|--:|--:|
| **DINOv2-base** | MMD² ≈ 0 (−0.0005), probe 0.50 | MMD² = 0.091, probe ~1.00 |
| **CLIP ViT-B/32** | MMD² ≈ 0 (−0.0005) | MMD² = 0.175 |

The unbiased estimator goes slightly negative at the floor — the signature that
it's honest (the biased V-statistic is always ≥ 0). The balanced probe gives
0.50 at the floor (no high-dim overfitting at these sample sizes).

## Experiments

1. **CUT @256 (adversarial + PatchNCE), validate DINOv2.**
   DINOv2 MMD went 0.091 → **0.109** (epochs 5,10), probe stuck at **0.998**.
   Translation changed color/texture (visually convincing, geometry preserved)
   but *widened* the DINOv2 gap — the 256-scale softness adds a synthetic
   high-frequency signature DINOv2 separates on. Plateaued.

2. **CUT @512 + resize-conv generator** (test the "real images are sharper"
   hypothesis). Launched, abandoned before a metric in favor of (3); the
   resolution lever was plausible but we chose to test the MMD-loss idea first.

3. **CUT + direct CLIP-feature MMD loss @256, batch 16** (optimize CLIP-MMD,
   **validate held-out DINOv2-MMD**). Decisive:

   | epoch | CLIP-MMD (optimized) | CLIP gap closed | DINOv2-MMD (held-out) | probe |
   |--:|--:|--:|--:|--:|
   | 3 | 0.064 | 63% | 0.104 | 1.000 |
   | 6 | 0.045 | 74% | 0.095 | 1.000 |
   | 9 | 0.047 | ~73% (plateau) | 0.095 | 0.998 |

   The generator matched real's **CLIP** distribution (74% of that gap closed)
   while the **DINOv2** gap stayed at the baseline and the probe stayed ~perfectly
   separable. Triple-confirmed (evals 3/6/9).

## Conclusion

- **Pixel-level translation — adversarial *or* moment-matching — does not make
  sim indistinguishable from real in a strong frozen feature space.** The domain
  signal lives in extractor-specific statistics; matching one frozen encoder's
  distribution (CLIP) does **not** transfer to another (DINOv2).
- **GAN/NCE losses are not evidence of transfer.** They sat at healthy
  equilibrium (D≈0.25, NCE≈1.6) throughout, while nothing transferred. Likewise
  a falling *training* MMD (its own optimization target) is necessary-not-
  sufficient.
- **Methodology win:** validating in a *held-out* feature space caught the
  CLIP-specific overfitting. Optimizing and validating DINOv2-MMD with the same
  statistic would have produced a great-looking-but-meaningless number — the
  same Goodhart trap as selecting checkpoints/architectures on the test set.

## Pivot

Stop trying to make pixels match a feature distribution. Instead **align the
features the GFN actually consumes, during GFN training** (unsupervised domain
adaptation): add an MMD/CORAL alignment loss between sim and real (unlabeled)
backbone/gist features in `train_seg_detr.py`, and **validate on downstream
real-test AP** — the one metric that can't be gamed by matching any single
extractor's moments. This also attacks the round-1 finding directly: the GFN
gist's real-transfer auxF1 was 0.85 (sim) → 0.50 (real); alignment targets that
gap where it matters. See `autoresearch/PROTOCOL.md` and `docs/gfn_template.py`.
