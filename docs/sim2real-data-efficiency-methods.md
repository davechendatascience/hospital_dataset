# Data-efficient sim2real: methods adapted from `sim2real_data_efficiency.pdf`

Our first transfer attempt aligned features with MMD while treating the
labeled real images as *unlabeled* and training sim-only. The report
(`docs/sim2real_data_efficiency.pdf`, Landay 2026) reframes the problem:
**treat the simulator as a decaying prior, not as data to mix or merely
align to, and spend the scarce real labels on a short anchored fine-tune.**
All of its concrete YOLO11 suggestions are implemented in `train_yolo.py` as
composable, independently-toggleable flags (each adds a loss term and/or a
logged column / callback). Verified end-to-end (train → validate → save →
reload) on `ward_dataset_v2` + `ward_dataset_v3/real_dev`.

## Why MMD-only underperformed
- It discarded the real **labels** we have (`real_dev`: 396 imgs / 2,386 anns).
  The report's whole regime is "abundant sim + scarce *labeled* real" — the
  biggest win is fine-tuning on those labels under a sim prior.
- MMD is a single global statistic and the *weak* form of the idea; the report
  calls DANN its "higher-capacity cousin" and the prior/anchor the bigger lever.

## The components (flag → what it does)

| Flag(s) | Method | Effect |
|---|---|---|
| `--anchor --anchor-lambda0 L [--anchor-tau0 T]` | **A1** decaying sim-anchor (L2-SP) | MAP under `θ~N(θ_sim,·)`: penalty `λ(N)·‖θ−θ_sim‖²`, `λ(N)=λ0·τ0/(τ0+N)` decays on the 1/N tail. Snapshots the loaded (sim) weights as the anchor. `anchor` column. |
| `--cls-prior` | **A2** measured class prior | Inits the Detect head cls-bias from per-class frequencies measured from `<data>/train` (we render from a 3D replicator → known rates). Stabilizes the dense-detector early loss. |
| `--freeze N --adapter-lr LR` | **A3** freeze + adapter | Freeze first `N` layers (backbone = 0..10; use 11), AdamW on the trainable head+BN-affine — caps params the scarce real set must fit. |
| `--adabn` | **A4** AdaBN | Resets BatchNorm running stats at epoch 0 (momentum→cumulative) so real images repopulate them — kills the low-order channel-stat shift for ~free. |
| `--align-real DIR --dann [--dann-weight w]` | **A5** DANN | Gradient-reversal domain head on the tapped backbone feature (`--align-layer`); BCE(sim/real) with `λ` ramp. The learned upgrade of the MMD term. `dann` column. |
| `--align-real DIR [--align-weight w]` | MMD | Unbiased per-location MMD pulling the tapped feature toward the real images (see [`rkhs-mmd-domain-adaptation.md`](rkhs-mmd-domain-adaptation.md)). `mmd` column. |
| `--depth-aux [--depth-aux-weight w]` | **A9** depth distillation | Auxiliary head predicting the GT depth (`<split>/depth/`) on the tapped feature, sim phase only; RGB-only at test. Forces geometric aug OFF so depth stays letterbox-aligned. `depth` column. |
| `--train-split / --val-split / --test-split` | split override | Point train at `real_dev`, val/test at `real_holdout` for the fine-tune phase. |

Implementation notes: the model class is module-level + picklable; a
`save_model` override strips the DA hook/state so saved checkpoints reload as
plain `SegmentationModel`; DANN/depth heads are built eagerly (before the
optimizer) via a tiny dummy forward; all DA loss terms append a 0 in eval so
`loss_items` length stays consistent with the extended `loss_names`.

## The two-phase workflow (the report's recommended shape)

**Phase 1 — pretrain on sim** with the alignment/aux signals (→ `best.pt`):
```
.venv/bin/python train_yolo.py --data ward_data/ward_dataset_v3 --model yolo11s-seg.pt --epochs 50 --imgsz 1024 --batch 16 --workers 8 --name v3_sim_pretrain --align-real ward_data/ward_dataset_v3/real_dev/images --dann --dann-weight 1.0 --cls-prior --depth-aux --test-split real_holdout
```

**Phase 2 — anchored fine-tune on labeled real_dev, test on real_holdout** (fresh optimizer = a second `train()` call):
```
.venv/bin/python train_yolo.py --data ward_data/ward_dataset_v3 --model runs/segment/v3_sim_pretrain/weights/best.pt --epochs 40 --imgsz 1024 --batch 8 --workers 8 --name v3_real_finetune --train-split real_dev --val-split real_holdout --test-split real_holdout --anchor --anchor-lambda0 1.0 --adabn --freeze 11 --adapter-lr 0.001 --cls-prior
```

**If you try one thing first** (the report's advice): just Phase 2 — the
decaying anchor is the single highest-value change and it directly uses the
real labels the MMD route threw away.

## Caveats / limitations
- **A1 Fisher (EWC) variant** (`--anchor-fisher`) is not estimated — full EWC
  needs a sim-task gradient pass to get the diagonal Fisher; the flag currently
  falls back to isotropic L2-SP (the report's default) with a printed note.
- **A9 depth-aux** disables geometric augmentation (mosaic/affine/flip) so the
  letterboxed depth stays aligned with the RGB; only ~80% of v3 frames have GT
  depth (render warmup), the rest contribute a zero target.
- Tune `--anchor-lambda0` so the `anchor` column is ~0.1–1× the seg loss early.
- Per the report, prefer the **smaller** YOLO11 variant (n/s) for ~400 real
  images, and couple capacity to anchor strength.
