# Round 1 analysis — baseline GFN (1 epoch)

Arch: frozen DINOv2-base, FiLM gate on, K=64 latents, 2 gist self-attn layers,
aux weight 0.5. Budget: 1 epoch on ward_v2/train (3400). Eval: valid (sim, 600)
+ real_dev (real, 478).

## Numbers

| metric        | valid (sim) | real_dev (real) | retention |
|---------------|------------:|----------------:|----------:|
| mask AP       |       0.421 |           0.184 |       44% |
| mask AP50     |       0.692 |           0.286 |       41% |
| mask AP75     |       0.399 |           0.189 |       47% |
| gist auxF1    |       0.847 |           0.502 |       59% |

**sim→real gap (valid−real_dev mask AP) = 0.237.** Even at 1 epoch the bottom-up
collapse the design note predicted is visible: real is ~44% of sim. valid AP=0.42
after a single epoch is high only because the Mask2Former decoder is COCO-warm-started.

## Where the leak is (the key read)

The gist transfers **relatively** better than the masks (59% vs 44% retention),
which by the PROTOCOL rule points at the spatial/decoder path as the larger
leak. BUT the gist's **absolute** real transfer is only auxF1=0.50 — the global
object-inventory commitment is wrong about half the time on real images, and the
entire top-down path is conditioned on it. A half-wrong gist cannot gate the
spatial path correctly, so the gist is the *foundational* bottleneck even though
the decoder leaks more on top of it. Both leak; fix the gist first.

Why would the gist leak when it sits on frozen (domain-invariant) DINOv2? Because
the gist *pooling* (latents + cross/self-attn + the aux Linear) is trained on sim
only and reads just the **last** DINOv2 stage — the most abstract, most
sim-fittable block. The domain-robustness of DINOv2 is being partly discarded by
a sim-tuned global read.

## Qualitative failure modes (8 real_dev overlays)

**Transfers well** — large, colour/shape-distinct objects: companion_chair,
bedside_table, curtain, sink (pink), hospital_bed, IV pole, monitors, wall clock,
blackboard. These carry the 0.286 AP50.

**Breaks on real:**
1. **White-on-white sanitary ware** — toilet missed *entirely* (white on white
   tile), shower seat missed. In sim these were material-randomised, so the model
   never had to localise them from shape alone; on real there's no
   colour/texture boundary and it has no shape prior to fall back on.
2. **Thin / specular metal** — grab bars and mirror frame come back as
   fragmented blobs (frame_02704), IV rails partial. High-frequency, reflective,
   under-resolved at short-edge 504.
3. **Boundary sloppiness on big objects** — TV mask covers only part and bleeds
   (frame_01487); door masks are spotty (frame_05102).

Over-segmentation/hallucination is low — failures are **misses + boundary**, not
false positives. The misses (1) and (2) are exactly cases a *correct, strong
semantic+shape prior* should recover — which is the GFN thesis — but only if the
gist that supplies the prior is right, and right now it's 0.50 on real.

## Recommended next move (round 2)

**Multi-scale gist**: feed the gist mid + late DINOv2 stages (e.g.
`gist_source_stages=[-4, -1]`) instead of the last block only, so the global
commitment draws on more domain-robust mid-level structure rather than the most
abstract / most sim-tuned features. Single change, directly targets the leading
bottleneck (gist real-transfer), already verified to run in `docs/gfn_template.py`.

**Falsifiable prediction:** real_dev_auxF1 rises from 0.50 toward sim's 0.85 and
the gap drops below 0.237. If auxF1 does **not** move, gist input-scale isn't the
bottleneck → pivot to (a) stronger global-first supervision (aux weight 0.5→1.0,
flag-only, fastest to try) then (b) richer gist→spatial coupling (`gate_type=xattn`,
the §5.1 GistGate) to attack the decoder leak directly.

Requires a small port of `gist_source_stages` from the template into
`train_seg_detr.py`'s GistFirstNetwork (production currently reads last stage only).

## Caveats

1 epoch — absolute APs will rise with more training and the gap may shift; the
**auxF1 transfer gap (0.85 vs 0.50)** is the more stable signal and is what the
round-2 move targets. Each variant trains from its own init, so confirm a real
gap delta over noise before over-reading.
