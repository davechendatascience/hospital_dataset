# Hospital Ward Sim‑to‑Real

Sim‑to‑real pipeline for **hospital‑ward object detection / instance segmentation**.
Goal: render a synthetic ward in **Isaac Sim**, translate the renders into
**photorealistic images that match a specific real ward**, and train a detector on
that styled‑but‑labeled data so it transfers to real handheld‑webcam photos.

The translation is done with **NVIDIA Cosmos‑Transfer2.5** (a world‑foundation model),
which was the first approach in this project to produce realistic, our‑ward‑matching
output while keeping the synthetic labels valid.

```
Isaac Sim render ──► RGB + GT depth + instance‑seg + COCO/YOLO labels   (replicator_dataset.py)
        │
        ├─ seg (class‑id map) ─┐
        ├─ GT depth ───────────┤ controls
        ├─ edge (on the fly) ──┘
        ├─ guided foreground mask (anchor labeled objects)
        ├─ prompt = the frame's actual object classes (from labels)
        └─ style ref = real photo content‑matched by object inventory
                          │
                          ▼
            Cosmos‑Transfer2.5  ──►  photoreal, our‑ward‑style, label‑aligned frames
                          │            (gen_cosmos_jobs.py → cosmos_jobs/run_all.sh)
                          ▼
            train detector (YOLO) on styled sim, evaluate on the REAL holdout
                          │            (train_yolo_da.py)
                          ▼
            domain‑gap measured throughout in DINOv2 / CLIP feature space
                                       (measure_domain_gap.py)
```

## Approaches tried (what worked)

| approach | result |
|---|---|
| **CUT / CycleGAN** (incl. depth‑conditioned, +CLIP‑MMD loss) | ✗ plateaued — a small GAN has no real‑image prior; the DINOv2 gap/probe never closed. |
| **ControlNet‑SD** (depth/seg control + LoRA‑on‑real + IP‑Adapter) | ~ partial — photoreal but only ~28 % of the MMD gap closed (generic‑SD look, not *our* ward). |
| **Cosmos‑Transfer2.5** (2B, multimodal control) | ✓ photoreal **and** matches our ward; structure/labels preserved by the controls + guided mask. |

The GAN and ControlNet scripts were removed once Cosmos worked (see git history). The
negative‑result write‑up is in `docs/sim2real-translation-findings.md`.

## Environments

Two Python envs (neither committed):
- **`.venv`** → symlink to `/home/edge-host/Documents/.venv` — the main ML env (torch 2.12+cu130,
  transformers, diffusers, pycocotools, ultralytics, cv2). Used for everything except Cosmos.
- **`~/cosmos-transfer2.5/.venv`** — the Cosmos env (torch 2.9+cu130), built with `uv sync --extra=cu130`.
- **Isaac Sim** at `~/isaac-sim` — run the renderer with `~/isaac-sim/python.sh`.

Hardware: NVIDIA **GB10** (DGX Spark, aarch64, CUDA 13). Cosmos‑Transfer2.5 officially supports
DGX Spark on cu130.

## Data layout (gitignored — code only is tracked)

```
ward_v3/
  _raw/          Isaac BasicWriter output: rgb_*, distance_to_camera_*.npy, instance_segmentation_*, ...
  train/         2700 sim frames: images/ depth/ labels/ labels_bbox/ labels_seg/ _annotations.coco.json
  val/           300 sim frames (same structure)
  test/          728 REAL ward photos + COCO/YOLO labels (the real domain / style refs)
cosmos_jobs/
  configs/       one Cosmos JSON per sim frame      seg/  depth/  fgmask/   (control inputs)
  outputs/       styled results <stem>.jpg          run_all.sh  manifest.csv
```

Classes: 44 ward categories defined in `ROS2_bridge/src/fixed_categories.py`.

## Pipeline

### 1. Render the sim (Isaac Sim)
```bash
~/isaac-sim/python.sh replicator_dataset.py --stage <ward.usd> --out ward_v3 \
    --frames 3000 --extra-channels --headless        # RGB + GT depth + instance‑seg + labels
# --cosmos      : CosmosWriter (clip multimodal control) instead of BasicWriter
# --trajectory  : smooth Catmull‑Rom camera fly‑through (coherent clips; auto‑on with --cosmos)
```

### 2. Build the labeled dataset (main `.venv`)
```bash
add_instanceseg_labels.py   # _raw instance‑seg → COCO RLE masks
add_depth_to_splits.py      # _raw depth.npy → normalized depth PNGs per split
coco_to_yolo_labels.py      # COCO → YOLO bbox + seg labels
materialize_ward_v3.py      # assemble train/val splits from _raw
split_real_test.py          # split the real photos into train/holdout
```

### 3. Measure the sim→real gap (any time)
```bash
.venv/bin/python measure_domain_gap.py --data ward_v3 \
    --set sim=train --set styled=<styled_dir> --set real=ward_v3/test \
    --compare-to real --gap-base sim --feature dinov2     # also --feature clip
```
Unbiased MMD² + a balanced classifier‑two‑sample probe in DINOv2/CLIP space.
Real‑vs‑real floor ≈ MMD 0.035 / probe 0.59; styled wants to approach that floor.

### 4. Translate sim → real with Cosmos‑Transfer2.5
```bash
# generate one config per frame (controls + guided mask + content‑matched real ref + prompt)
.venv/bin/python gen_cosmos_jobs.py --sim-dir ward_v3/train/images --out cosmos_jobs --vary-style
# run all (resumable: skips rendered frames, survives per‑frame aborts) — ~40 s/frame
nohup bash cosmos_jobs/run_all.sh > ~/cosmos_batch.log 2>&1 &
```

**The recipe** (`gen_cosmos_jobs.py` defaults):
- **Controls:** `seg 0.6` (class‑id label map → object regions) + `depth 0.8` (GT geometry) +
  `edge 1.0` (contours, on the fly). No `vis` (vis keeps sim color/texture).
- **Guided generation:** foreground mask (union of object masks) anchors the labeled objects;
  `guided_generation_step_threshold ≈ 10` — anchor structure early, release for realistic restyle late.
- **Prompt:** scene‑agnostic, lists the frame's *actual* object classes from its COCO labels
  (no brittle ward/corridor/bathroom classifier — the controls define the scene).
- **Style ref (`image_context_path`):** the real photo whose object inventory best matches the
  frame (Jaccard); `--vary-style` samples among the top matches + varies seed + lighting.
- **`guidance` 3** (modest, so structure isn't overpowered).

Notes / limits:
- Cosmos has **no per‑region class→appearance binding** (seg is spatial; the prompt is scene‑level).
  Object *identity* is biased by the prompt + anchored by the guided mask, not hard‑bound.
  True per‑region class control would need post‑training/LoRA on the seg branch.
- `guided_generation_step_threshold` is an **integer step count** (default 10, of ~35), not a fraction.

### 5. Train + evaluate the detector
```bash
.venv/bin/python train_yolo_da.py --train-dir ward_v3_styled/train \
    --val-dir ward_v3/test --task detect --model yolo11s.pt --epochs 50
```
Trains on the styled sim, evaluates on the **held‑out real** set, logging per‑epoch
`train_mAP`, `real_val_mAP`, and the gap. **Real‑test AP is the selection metric**;
the DINOv2/CLIP gap is a guide, not the endpoint.

## Other scripts
- `da_pipeline.py` — orchestrator that chains stages (split → translate → measure → train).
- `train_rtdetr.py`, `train_yolo.py`, `train_yolo_world.py`, `train_seg_detr.py`, `predict_*.py`
  — detector/segmentor variants and inference.
- `inspect_stage.py`, `inspect_ward.sh`, `debug_campose.py` — render/scene inspection.
- `autoresearch/` — earlier global‑first‑backbone research loop. `experiments/` — MNIST domain‑shift probes.
- `docs/` — Cosmos/Replicator setup notes, global‑first architecture, sim2real findings.

## Status
Cosmos‑Transfer2.5 is installed and validated end‑to‑end; `cosmos_jobs/` holds the 2700
ward_v3 configs. The full batch (`run_all.sh`) is the long‑running step (~30 h on one GB10);
the styled output then feeds the detector.
