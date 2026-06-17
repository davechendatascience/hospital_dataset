# Sim2real for the ward detector: a domain-AGNOSTIC, single-phase recipe

Background reading: `docs/sim2real_data_efficiency.pdf` (Landay 2026) and
[`rkhs-mmd-domain-adaptation.md`](rkhs-mmd-domain-adaptation.md).

## The framing (why the earlier attempts were the wrong target)
- **MMD-only, sim-trained** (first attempt): aligned features toward
  *unlabeled* real and never used the real labels. It optimized a global
  statistic and left the real labels — our most informative signal — on the
  floor.
- **Decaying-anchor two-phase** (the report's headline): pretrain on sim, then
  *adapt* the weights toward real. Good, but it frames the goal as "move the
  model to the real domain."

The goal we actually want is **domain-agnostic**: features that don't care
whether an image is sim or real, so the *one* model is good on **both**. That
is exactly what a DANN domain-confusion objective produces (report §10.5: "the
backbone is pushed to produce features that cannot distinguish sim from
real"), and the report notes that *mixing* sim+real batches is correct there
because the **domain label is the supervision**.

## The recipe (single phase) — `train_yolo.py`
1. **Joint supervised training** on labelled **sim** (`train`) **+** labelled
   **real** (`real_dev`) in one mixed dataloader. real_dev is **oversampled**
   (`--real-oversample`, default 8 → ~27% of the mix) so it isn't drowned by
   the 8,500 sim images and the domain branch sees balanced batches. The
   task/seg head thus learns from **both** domains' labels.
2. **Domain-invariance** computed from the *same mixed batch*, using per-image
   domain labels read from the file paths (no second loader, no extra forward):
   - `--dann` — gradient-reversal **domain head** (BCE sim-vs-real, λ ramped
     `--dann-ramp`): backbone descends the task loss but *ascends* the domain
     loss → features become indistinguishable across domains. `dann` column.
   - `--mmd` — unbiased per-location **MMD** between the sim and real images in
     each batch (split by domain). `mmd` column. Composes with DANN.
3. `--cls-prior` — init the detect-head class bias from measured per-class
   frequencies (from the sim train split) — stabilizes the dense-detector
   early loss (RetinaNet/YOLO `bias_init` specialized to our measured rates).
4. **Measure domain-agnosticism at the end**: after training, `best.pt` is
   evaluated on **both** the sim `valid` split and the **real holdout**, and a
   `sim→real seg gap` is printed + written to `domain_report.json`. A small
   gap = genuinely domain-agnostic, not just sim-overfit.

**Data discipline:** `real_holdout` is **never** trained on and **never** used
for checkpoint selection (selection is on the sim `valid`); it stays an honest
cross-domain test. real_dev labels *are* used — as a supervised training
target (verified: 2,386 real annotations, all with segmentation masks).

## Command
```
.venv/bin/python train_yolo.py --data ward_data/ward_dataset_v3 --model yolo11s-seg.pt --epochs 60 --imgsz 1024 --batch 16 --workers 8 --name v3_domain_agnostic --dann --mmd --cls-prior --real-oversample 8
```
Splits are configurable: `--sim-train train --real-train real_dev --sim-val valid --real-holdout real_holdout` (defaults shown). Watch the `dann`/`mmd` columns during training and `real_holdout_metrics.csv` for the real number per epoch; the final `domain_report.json` is the headline.

## Implementation notes
- The model/trainer are module-level (picklable); a `save_model` override
  strips the DA hook/state so checkpoints reload as a plain `SegmentationModel`.
- The DANN head is built before the optimizer (a tiny dummy forward reads the
  tapped feature's channel count) so its params are optimized.
- DA loss terms append a 0 in eval so `loss_items` length stays consistent with
  the extended `loss_names` (the validator also calls `model.loss`).
- MMD per-location is used (not global-pooled): pooled conv features collapse
  in high-D to a degenerate ~0 MMD (see the RKHS doc §4); it only fires on
  batches that contain both domains (oversampling ensures they do).

## Tuning
- `--real-oversample`: higher → more real per batch (stronger real signal +
  balanced domains) but more re-use of the 396 real images (overfit risk).
- `--dann-weight` / `--dann-ramp`: the domain pressure; ramp avoids
  destabilizing early training.
- Per the report, prefer a **smaller** variant (n/s) for this data scale.

## What was dropped vs the report's full menu
The report also lists adapt-to-real items — decaying L2-SP anchor (A1),
freeze+adapter (A3), AdaBN (A4), depth-aux (A9). Those frame the goal as
"adapt to real" and are two-phase; this rewrite deliberately refocuses on the
**domain-agnostic, single-phase** target per the project decision. They can be
re-added later from git history if an adapt-to-real ablation is wanted.
