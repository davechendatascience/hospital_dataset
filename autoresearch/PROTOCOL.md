# Gist-network architecture autoresearch loop

Search the architecture space of `GistFirstNetwork` (in `../train_seg_detr.py`)
for the inductive bias that **shrinks the sim→real gap**, not the one that
maximizes any single AP. Training loss going down is free; every decoder here
can do it. The only lever is encoding the problem's structure into the
architecture precisely.

## What the loop optimizes

- **Selection metric:** `sim_real_gap_AP = valid_AP − real_dev_AP` (lower is
  better), with `real_dev_AP` itself needing to not collapse. `valid` is sim
  (same renderer as train); `real_dev` is 478 real photos.
- **Fast early signal:** `real_dev_auxF1` — the gist head's image-level
  object-inventory prediction on real images. It converges far faster than mask
  AP, so it tells us early whether the *global commitment* transfers even when
  the mask decoder hasn't caught up. If auxF1 is high but mask AP/gap is bad,
  the encoder transfers and the **decoder** is the leak → bias the top-down
  path next, not the encoder.
- **Qualitative:** `overlays/epochK/*.png` — predicted masks on real images.
  Read these every round; failure *modes* (misses thin metal poles, merges
  bed+curtain, only fires on sim-textured objects) suggest the next bias better
  than the scalar does.

## Data discipline (do not break this)

| split          | role                                   | loop may select on it? |
|----------------|----------------------------------------|------------------------|
| `train`        | 3400 sim images, training only         | —                      |
| `valid`        | 600 sim images, sim anchor             | yes                    |
| `real_dev`     | 478 real images, the dev gap           | **yes**                |
| `real_holdout` | 250 real images, final honest estimate | **never** until the end |
| `test`         | full 728 real (= real_dev ∪ holdout)   | avoid (contains holdout)|

`real_holdout` is scored **once**, at the very end, via `report_holdout.py`, on
the variant the loop already picked by its `real_dev` gap. Touching it earlier
makes the final number a fiction.

## One round

```
/home/edge-host/Documents/.venv/bin/python autoresearch/run_round.py --round N
```

This trains the arch in `current_config.json` for `--budget-epochs` (default 3),
evals `valid,real_dev` each epoch, dumps overlays, and writes
`iter_NN/report.json` + appends `leaderboard.jsonl`. Rounds are compared at
**equal compute** (same budget), so a win is "better bias under fixed training,"
not "trained longer."

## Division of labor

1. **Human:** run the round command above (training runs on your machine).
2. **Claude (between rounds):** read `iter_NN/report.json` + the overlays,
   compare against `leaderboard.jsonl`, form **one** hypothesis-driven change,
   apply it by editing `current_config.json` (flag-level knobs) and/or the
   `GistFirstNetwork` source (structural changes), bump `version`/`arch_desc`,
   and state the hypothesis + the falsifiable prediction (which metric should
   move and why). Then hand back for the next round.

Each change must keep the backbone contract (`.channels`, returns
`.feature_maps`) and be **one** move, so the gap delta is attributable.

## Search menu (hypothesis-driven moves, not random knobs)

- **Gist capacity / structure:** latents `K`, gist self-attn depth, read
  multiple DINOv2 layers (multi-scale gist) instead of only the last block,
  use DINOv2-with-registers.
- **Coupling (gist → spatial):** gate type (FiLM vs cross-attention
  conditioning vs additive seed only), where the gist injects (coarsest level
  only vs all levels vs the Mask2Former decoder queries), gate strength.
- **Global-first supervision:** aux weight, target (presence multilabel vs
  counts vs coarse spatial prior), curriculum (strong early, decay).
- **Backbone:** which DINOv2 layers feed F, frozen vs last-block-tuned (expect
  higher sim AP, wider gap — a control, not a fix).
- **Decoder-side bias** (when auxF1 transfers but masks don't): query init from
  the gist, mask-feature normalization, etc.

Built-in ablation controls already wired as flags: `--no-gfn-gate` (sever the
global commitment from the spatial path) and `--gfn-aux-weight 0` (no
global-first supervision). A real bias should beat both on the gap.

## Stop conditions

- gap ≤ target (set one explicitly when we start), or
- patience: K consecutive rounds with no `real_dev` gap improvement over the
  current best, or
- compute budget hit.

Then: pick the best-by-`real_dev`-gap variant, run `report_holdout.py` once,
and report the `valid − real_holdout` gap as the honest result.

## Caveat

Each variant trains from its own init, so a round conflates "better bias" with
"easier to optimize in N epochs." Equal budget + (optionally) a second seed on
finalists mitigates it; keep it in mind before over-reading a small gap delta.
