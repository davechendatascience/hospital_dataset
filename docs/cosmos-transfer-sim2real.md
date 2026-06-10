# Cosmos‑Transfer2.5 for Ward Sim‑to‑Real — Approach & Breakthroughs

Working notes on how we use **NVIDIA Cosmos‑Transfer2.5‑2B** to turn synthetic Isaac‑Sim
ward renders into photorealistic, *our‑ward*-matching images while keeping the synthetic
labels valid — and the sequence of fixes/insights that got it working.

Implementation: `gen_cosmos_jobs.py` (builds one Cosmos job per sim frame) →
`cosmos_jobs/run_all.sh` (resumable batch). Cosmos repo + env: `~/cosmos-transfer2.5`.

---

## 1. Why Cosmos (after GAN and ControlNet failed)

| approach | outcome | why |
|---|---|---|
| CUT / CycleGAN (+depth, +CLIP‑MMD loss) | ✗ gap never closed | a small generator + ~728‑image discriminator has no real‑image prior |
| ControlNet‑SD (depth/seg + LoRA‑on‑real + IP‑Adapter) | ~ only ~28 % of the DINOv2‑MMD gap closed | photoreal, but a *generic* SD look — not our specific ward |
| **Cosmos‑Transfer2.5‑2B** | ✓ photoreal **and** our‑ward | a world‑foundation prior + multimodal spatial control + a real style reference |

Cosmos was the first method to produce output that both looks real and matches our ward,
because structure is pinned by control maps while appearance comes from a real reference image
and a billion‑scale generative prior — see `sim2real-translation-findings.md` for the negatives.

## 2. How Cosmos‑Transfer2.5 works (the levers we use)

Run mode: **image‑to‑image**, one frame per job (`max_frames: 1`,
`num_video_frames_per_chunk: 1`), via `examples/inference.py -i <cfg>.json -o <out>`.

- **`video_path`** — the sim RGB frame (the thing being restyled).
- **Control branches** (each `{control_path?, control_weight}`; spatial, ControlNet‑style):
  - `seg` — class‑id colour map rasterized from the sim COCO → object **regions**.
  - `depth` — ground‑truth Isaac depth → **geometry**.
  - `edge` — contours (computed on the fly) → shape without colour.
  - `vis` (blur) — keeps the **input's colour/texture**; we deliberately **omit** it (it drags sim colours in).
- **`image_context_path`** — a **real ward photo** as the style reference (the IP‑Adapter‑like
  appearance driver: colour, materials, lighting).
- **`prompt`** — text describing the scene/objects. Encoded by **Cosmos‑Reason1‑7B** (used as the
  *text encoder*, not an automatic prompt‑rewriter — there is no scene‑reasoning upsampler in this path).
- **`guidance`** — classifier‑free guidance (default 3; modest so structure isn't overpowered).
- **Guided generation** (optional, important — see §4): `guided_generation_mask` (a foreground mask)
  anchors masked regions to the sim during early denoising; `guided_generation_step_threshold`
  controls how long.

**Critical limitation:** Cosmos has **no per‑region class→appearance binding**. The seg map is
purely *spatial* (no colour→“overbed table” legend), and the prompt is *scene‑level*. So object
**identity** is *biased* (prompt) and *anchored in structure* (guided mask), but never hard‑bound to
a region. True per‑class control would require **post‑training/LoRA on the seg control branch**.

## 3. Breakthroughs & fixes (chronological)

### 3.1 Depth control must be 3‑channel
Cosmos's control reader does `einops.rearrange(x, "t h w c -> t c h w")`, i.e. it needs **HWC**.
Our GT depth PNGs are grayscale (`mode L`, 2‑D) → it crashed the whole batch
(`Wrong shape: expected 4 dims … (1,1080,1920)`). **Fix:** write the depth control as a
3‑channel RGB copy (`Image.open(d).convert("RGB")`). The seg map was already RGB, so only depth needed it.

### 3.2 Label‑driven prompts (correct object *types*)
A generic scene template made Cosmos render a *cabinet* where the labels said **overbed table**
(the prompt literally said "bedside cabinet"). **Fix:** build each frame's prompt from its **own
COCO classes** (`overbed table, IV pole, telephone, …`). Naming the real objects biases Cosmos to
render the right ones.

### 3.3 Guided generation = anchoring the labeled foreground
To stop the model reinterpreting objects, feed a **foreground mask** (union of the frame's instance
masks) as `guided_generation_mask`. Details that bit us:
- Format is **`.npz` with key `arr_0`, shape `(T,H,W)`** (single‑channel; `foreground_labels=None`
  ⇒ any non‑zero = foreground). A PNG is rejected ("not a mp4 or npz file").
- **`guided_generation_step_threshold` is an integer step count** (default 10, out of ~35 denoise
  steps) — *not* a 0–1 fraction. (External advice quoting "0.3" maps to ~10 steps here.)

### 3.4 The guided‑vs‑realism tension
Guided generation anchors masked regions **to the sim input**, so it preserves structure/identity
but pulls the **foreground appearance toward sim** (overriding the real style ref there). Measured on
one ward frame:
- **steps 25 (strong):** identity locked, but sim‑ish (green table top, navy bed).
- **steps 12 (moderate):** good balance — clearly a table, mostly realistic.
- **steps 0 (off):** most photoreal, but object identity can drift.

So strength is a dial between *identity* (high) and *realism* (low).

### 3.5 Why we still need guided masks (context can mislead)
Without guidance, the **`image_context_path` (style ref) can dominate** and make Cosmos
*generate from the reference's content* rather than the sim's actual scene → wrong objects/scene.
The guided foreground mask anchors *this* frame's structure so the reference only **restyles**.
Conclusion: keep guided **on**, but **moderate** (steps ≈ 10) — anchor structure early, release for
realistic restyle late.

### 3.6 Drop our scene classifier — the controls already define the scene
We had a heuristic ward/corridor/bathroom classifier (by object presence) that **mislabeled**
frames (e.g. a sink in a corridor → "bathroom"; a medical‑gas headwall → "corridor"), which then
attached the wrong prompt **and** the wrong style ref. Since Cosmos has no auto‑reasoner *and* the
**seg+depth controls + guided mask already encode the real scene structure**, we removed the
classifier entirely:
- **Prompt** is now scene‑agnostic — "a realistic Taiwanese hospital interior, with `<the frame's
  actual objects>`; `<shared materials>`." The controls dictate the scene.
- **Validated:** frame `0006` now renders a correct **medical‑gas headwall** (was wrongly "corridor")
  and `0008` a correct **utility/corridor with bins** (was wrongly "bathroom").

### 3.7 Content‑matched style reference (object‑inventory Jaccard)
How to pick the real `image_context_path` per frame, robustly:
- **Not** DINOv2 nearest‑neighbour — sim ≠ real in DINOv2 space, so it misassigns.
- **Not** a scene label — brittle (see 3.6).
- **Yes:** index every real test photo by its **set of object classes**, then for each sim frame
  pick the real photo with the highest **Jaccard** overlap of class sets (top‑K; `--vary-style`
  samples among the top‑8 for variety). Both sides have COCO labels, so we match on the actual
  objects present. (Requires the real refs to be labeled — `ward_v3/test`, 728 annotated photos.)

## 4. The current recipe (`gen_cosmos_jobs.py` defaults)

| lever | value | rationale |
|---|---|---|
| `seg` control_weight | **0.6** | keep object regions without imprinting the palette colours |
| `depth` control_weight | **0.8** | strong geometry/viewpoint anchor |
| `edge` control_weight | **1.0** | contours/shape without colour (on the fly) |
| `vis` | **off** | vis preserves the *sim's* colour/texture — the opposite of what we want |
| `guided_generation_step_threshold` | **10** (of ~35) | anchor structure early, release for realistic restyle late |
| guided mask | union of object masks, `.npz arr_0 (1,H,W)` | anchors labeled foreground; stops context‑misgeneration |
| `guidance` | **3** | modest, so structure isn't overpowered by text |
| `prompt` | scene‑agnostic + the frame's **actual COCO objects** | correct object types; no brittle scene class |
| `image_context_path` | real photo, **object‑inventory matched** (Jaccard) | appropriate appearance per frame |
| `--vary-style` | sample top‑8 refs + random seed + lighting modifier | style diversity for a training set |

Per‑frame the pipeline writes: `cosmos_jobs/seg/<stem>.png`, `depth/<stem>.png` (RGB),
`fgmask/<stem>.npz`, and `configs/<stem>.json`. The batch (`run_all.sh`) is **resumable** — it
skips frames whose output exists and restarts on the remainder if a frame aborts.

## 5. What's validated / what's open

**Validated:** end‑to‑end on ward/headwall/corridor/bathroom frames — photoreal, our‑ward style,
correct scene & object types, labels positionally preserved (seg+depth), and the guided mask prevents
the style ref from hijacking content.

**Open:**
- Per‑region class→appearance is still only *biased*, not bound. For hard guarantees → **post‑train /
  LoRA the seg branch** on our taxonomy (Cosmos provides a vid2vid post‑training config).
- The full 2700‑frame batch is ~30 h on one GB10 (single GPU, sequential, ~40 s/frame).
- Final selection metric is **detector AP on the real holdout** (`train_yolo_da.py`), not the
  DINOv2/CLIP gap (that's only a guide).
