# Research Note: A Global-First ("Reverse Hierarchy") Backbone for Sim→Real Ward Perception

**Status:** design note / pre-implementation
**Author:** ychendavid (with Claude)
**Date:** 2026-06-04
**Context:** Follow-on to the frozen-DINOv2 experiment in `train_seg_detr.py`. The
recurring goal across this line of work is *high-level feature learning that
transfers from synthetic ward renders to real photos*. Our own runs show every
bottom-up detector collapses sim→real (Mask2Former-Swin: 0.58 valid → 0.20
real_test; YOLO11-seg: 0.58 → 0.10), and the only model that holds up
(YOLO-World, ~0.67 real) does so partly by *training on real data*, which is the
crutch we want to remove.

This note (1) decomposes CNN / ViT / DETR down to their atomic compute, (2)
shows that all three are **bottom-up in representation** even when their
*receptive field* is global, (3) defines precisely what "global features first,
low-level features later" should mean, (4) grounds it in prior art, and (5)
proposes a concrete architecture — the **Gist-First Network (GFN)** — plus an
experiment plan on `ward_v1`.

---

## 1. The question

> Pull out the graph of DETR and ViT, find the most basic component (for YOLO
> it's a conv layer), and redesign so the network **attends to global features
> first and low-level features later.**

The instinct is right and worth stating sharply: today's vision nets commit to
*texture/edges* before *semantics*. Under domain shift the early commitments are
exactly the brittle ones. If a net instead committed to "this is an IV pole"
(domain-invariant) **before** it looked hard at pixels, the low-level
domain noise would arrive *after* the decision that matters and could be
down-weighted. That is the whole thesis.

---

## 2. The atoms: what these networks are actually made of

Verified against the installed stack (transformers 5.9, the same classes our
training scripts use).

### 2.1 YOLO / CNN — atom = **convolution**

```
image ─▶ [Conv k×k stride s] ─▶ BN ─▶ act ─▶ … (repeat, downsample) ─▶ feature pyramid
         └─ local weighted sum, weight-shared, receptive field grows with depth ─┘
```

- Atom: `y = act(BN(W ⊛ x + b))`. **Local** (k×k support), translation-equivariant.
- Receptive field is *small early, large late* — globality is an emergent
  property of depth.
- Representation: edges/texture (early) → parts → objects (late). **Bottom-up.**

### 2.2 ViT — atoms = **a patchify conv + (self-attention, MLP)**

Introspected from `Dinov2Model`:

```
patch_embeddings.projection = Conv2d(kernel=14, stride=14)   # the ONE conv
block = norm1 ─▶ attention ─▶ ls1 ─▶ +  ─▶ norm2 ─▶ mlp ─▶ ls2 ─▶ +
                 │                              │
                 └ token-mixing (global)        └ channel-mixing (per-token)
```

- **Atom A — patch embed:** a single strided conv. The image is turned into
  tokens by *exactly the YOLO atom*, used once.
- **Atom B — self-attention:** `softmax(QKᵀ/√d)·V`. Content-based **global**
  routing — every token can read every token *from layer 1*.
- **Atom C — MLP:** per-token channel mixing.

Crucial subtlety: **ViT's receptive field is global immediately, but its
*representation* is still bottom-up.** Empirically, early ViT heads attend
locally and encode low-level structure; semantic abstraction emerges only in
late layers. Global *wiring* ≠ global-first *features*. This is the trap the
last experiment half-fell into: we picked a global-RF backbone but still let it
build features low→high.

### 2.3 DETR family — atoms = **backbone conv + encoder self-attn + decoder cross-attn**

Introspected from `Mask2FormerMaskedAttentionDecoderLayer`:

```
image ─▶ CNN/ViT backbone ─▶ feature memory  F  (dense, H×W tokens)
                                   │
        ┌──────────────────────────┘
        ▼
  decoder layer × L:
     self_attn   : queries ↔ queries          (which objects, dedup)
     cross_attn  : queries ↔ F                 (read image at object locations)
     ffn
        ▲
   N learned object queries (a small, global set)
```

- **Atom D — cross-attention:** a *small* set of learned queries reads a *dense*
  feature memory. This is the one genuinely "top-down-ish" primitive already in
  the zoo: global queries pull from features.
- But DETR still runs a full bottom-up backbone first, *then* queries read its
  top layer. The queries never steer the *backbone's* low-level reads.

### 2.4 The unifying picture

| Net   | Atom(s)                         | Receptive field | **Representation order** |
|-------|---------------------------------|-----------------|--------------------------|
| YOLO  | conv                            | local→global    | **bottom-up**            |
| ViT   | patch-conv + self-attn + MLP    | global from L1  | **bottom-up**            |
| DETR  | backbone + self-attn + cross-attn | global         | **bottom-up**, then global read |

Everyone is bottom-up in *representation*. The lever we want is not "make the
receptive field global" (ViT already did) — it's **invert the order in which
abstraction levels are decided.**

---

## 3. What "global-first, local-later" precisely means

"Global" overloads three independent axes. Be explicit about which we invert:

1. **Receptive field** (how far a unit can see). ViT is already global. *Not the
   target.*
2. **Spatial resolution** (coarse grid → fine grid). Standard nets go fine→coarse
   in the encoder, then coarse→fine in a decoder (U-Net/FPN). *Partially exists.*
3. **Representational abstraction** (texture/edges ↔ object identity/semantics).
   Everyone goes low→high. **This is the axis we invert.**

**Target design:** decide a low-dimensional, global, *semantic* "gist" of the
scene first; then run a **top-down, coarse-to-fine** pass that injects that gist
into progressively higher-resolution, lower-level features — so every local read
is *conditioned on* an already-formed semantic commitment.

---

## 4. Prior art (so we build, not reinvent)

This idea is not from nowhere; the contribution would be the *specific
composition + the sim2real framing*, not the primitive.

- **Reverse Hierarchy Theory** (Ahissar & Hochstein, 2004) — the neuroscience
  backbone of this note. "Vision at a glance" is a fast feed-forward sweep to
  high-level cortex giving gist/category first; "vision with scrutiny" is a
  *top-down return* to low-level areas for detail. We are proposing RHT as an
  architecture.
- **Perceiver / Perceiver IO** (Jaegle et al., 2021) — a small latent array
  cross-attends to the input; a global latent bottleneck, then decode. This is
  our "gist encoder" primitive.
- **Slot Attention / object-centric** (Locatello 2020) — global slots compete
  for input → object-level abstraction first. (We argued earlier this hurts mask
  AP on its own; here it's an ingredient, not the whole model.)
- **Deformable DETR** (Zhu 2021), **Cascade R-CNN** (Cai 2018) — coarse-to-fine
  *spatial* refinement with iterative box updates. Reuse the refinement
  machinery; add semantic-gist conditioning.
- **Predictive coding / feedback nets** (Rao & Ballard 1999; PredNet, Lotter
  2017; CORnet; "Feedback Networks", Zamir 2017) — top-down predictions modulate
  bottom-up processing, often recurrently. Source for the *gating* mechanism.
- **GLOM** (Hinton 2021) — part-whole hierarchies reconciled by bottom-up +
  top-down + lateral agreement.

Gap we'd fill: a **single-pass, trainable backbone** that (a) forms a global
semantic gist via a Perceiver-style bottleneck, (b) refines coarse→fine with the
gist *gating* each low-level read, and (c) is explicitly *supervised* global-first,
and we evaluate it on **sim→real transfer** rather than on clean in-domain AP.

---

## 5. Proposed architecture — Gist-First Network (GFN)

### 5.1 New atom

The atom we add is a **gist-gated cross-attention block**: a higher-resolution
spatial state reads image features, but its queries/keys are *modulated by the
global gist* (FiLM-style conditioning), so the semantic commitment controls what
low-level evidence is admitted.

```
GistGate(state S_l, image feats F_l, gist z):
    γ, β = MLP(z)                       # FiLM params from the global gist
    Q    = γ ⊙ Wq·S_l + β               # queries steered by "what we think this is"
    S_l' = S_l + CrossAttn(Q, K=Wk·F_l, V=Wv·F_l)
    S_l' = S_l' + FFN(LN(S_l'))
    return S_l'
```

### 5.2 The graph

```
            image
              │
       patch-conv (the YOLO atom, once)            ── cheap, stays
              │
        shallow token encoder  (2–4 ViT blocks)    ── thin "what's roughly here"
              │  F_hi (fine tokens, kept as a side memory)
              ▼
   ┌───────────────────────────────────────────────┐
   │  GIST ENCODER (Perceiver-style)                │
   │   K global latents  ⟵ cross-attn ⟵  F_hi        │
   │   self-attn over latents × few                 │
   │   →  z  (the scene gist; ~K×D, K small)         │
   └───────────────────────────────────────────────┘
              │  z  ──▶ aux head: image-level multi-label class loss  ◀── GLOBAL-FIRST SUPERVISION
              ▼
   TOP-DOWN COARSE→FINE REFINEMENT  (l = coarse → fine)
        S_coarse  = broadcast(z) onto a small grid
        for each level l (upsample ×2 each step):
            S_l = GistGate(upsample(S_{l-1}), F_l, z)   # low-level reads, gist-gated
        → pyramid {S_4, S_8, S_16, S_32}  (high→low res, like Swin/FPN output)
              │
              ▼
   Mask2Former pixel decoder + masked-attention decoder   ── UNCHANGED
              │
        boxes / masks / classes
```

Read it top to bottom and the inversion is explicit: **z (global semantics) is
computed and supervised before the top-down stack ever reads fine detail**, and
every fine read is gated by z.

### 5.3 Why it drops into our codebase

The refinement stack emits a 4-level pyramid with the **same contract** as the
DINOv2 Simple-Feature-Pyramid backbone I already added (`Dinov2SimpleFPN`:
exposes `.channels`, returns `.feature_maps`). So GFN is a *backbone swap* behind
the existing `--backbone` switch in `train_seg_detr.py`; the Mask2Former pixel
decoder + mask head + COCO mask-AP eval are reused verbatim, and numbers stay
directly comparable to the Swin (0.20) and frozen-DINOv2 baselines.

---

## 6. Why this should help sim→real (the mechanism)

- **Domain shift lives in the low-frequency/texture statistics** the *early*
  layers latch onto. By forming a semantic gist `z` from a thin encoder + global
  bottleneck and **supervising it image-level**, the model is pushed to make its
  category commitment from coarse, domain-robust structure first.
- The **gist-gate** means low-level features enter only *after* and *through* the
  semantic decision: FiLM(z) can suppress texture channels that disagree with the
  committed class. Texture-domain noise is admitted late and conditionally,
  instead of dictating early features.
- This is the architectural version of the frozen-DINOv2 result we expect: the
  representation that decides "what" never gets to overfit sim texture — but here
  it's *trained*, not borrowed, and it's *gated*, not just frozen.

---

## 7. Experiment plan on `ward_v1`

Train sim-only, eval `valid`(sim) + `real_test`(real); the metric is the
**sim→real gap**, not in-domain AP. All comparisons reuse the existing harness.
**GFN is implemented** as `--backbone gfn` in `train_seg_detr.py` (gist sourced
from frozen DINOv2; `--gfn-gate/--no-gfn-gate`, `--gfn-aux-weight`,
`--gfn-latents`). Concrete runs (each: sim-only train, eval valid+real_test):

```bash
PY=/home/edge-host/Documents/.venv/bin/python
COMMON="--data ward_v1 --epochs 30 --batch 4 --short-edge 504 --device 0 \
        --eval-splits valid,test,real_test --project runs/seg_detr"

# B1  frozen DINOv2 + Simple Feature Pyramid (no gist)
$PY train_seg_detr.py $COMMON --backbone dinov2 --name b1_dinov2

# G1  GFN: gist gate ON + global-first aux loss ON   (the proposal)
$PY train_seg_detr.py $COMMON --backbone gfn --gfn-gate --gfn-aux-weight 0.5 --name g1_gfn

# A1  ablate the gate (gist drives only the aux loss; pyramid == B1)
$PY train_seg_detr.py $COMMON --backbone gfn --no-gfn-gate --gfn-aux-weight 0.5 --name a1_nogate

# A2  ablate global-first supervision (gate on, no gist loss)
$PY train_seg_detr.py $COMMON --backbone gfn --gfn-gate --gfn-aux-weight 0.0 --name a2_noaux
```

| # | Run | Flags | Purpose |
|---|-----|-------|---------|
| B0 | Mask2Former-Swin | `--backbone swin` | bottom-up baseline; gap ≈ 0.38 |
| B1 | Frozen DINOv2 + SFPN | `--backbone dinov2` | frozen high-level features |
| G1 | **GFN, gate + aux ON** | `--backbone gfn` | the proposal |
| A1 | GFN, **gate OFF** | `--no-gfn-gate` | does *gating* matter? (vs B1) |
| A2 | GFN, **aux OFF** | `--gfn-aux-weight 0` | does *global-first supervision* matter? |
| A3 | GFN, reverse stack (fine→coarse) | *not yet wired* | controls for "it's just a U-Net" |

Primary readout: `real_test_AP` and `(valid_AP − real_test_AP)` from each run's
`metrics.csv`. Success = G1 shrinks the gap vs B0/B1; A1–A2 attribute *which*
ingredient did it. Sanity check already verified: `--no-gfn-gate --gfn-aux-weight 0`
reduces GFN to the plain DINOv2+SFPN path.

Secondary probes:
- Linear-probe the gist `z` for image-level class on sim vs real — does the gist
  stay domain-invariant?
- Per-class: does the gain concentrate on texture-confusable classes (gauze,
  curtain, linen) vs shape-defined ones (IV pole, bed)?

---

## 8. Risks & open questions

- **Trainability.** Coarse→fine top-down stacks can be unstable; the gist may
  collapse to a constant. Mitigations: deep supervision at each level, start the
  gist head as a strong auxiliary loss then anneal.
- **From-scratch vs pretrained.** Our wins so far came from *pretraining*. GFN's
  gist encoder could be initialized from a frozen DINOv2/CLIP image embedding
  (gist = pooled foundation feature) — likely the strongest variant, and it keeps
  the "borrowed high-level prior" that already works while adding the top-down
  gate. Consider this the default G1 rather than fully-from-scratch.
- **Compute.** The extra Perceiver bottleneck + FiLM is cheap; the top-down
  stack roughly doubles decoder-side FLOPs. Fits on the GB10 at the batch sizes
  we already use.
- **Is the gist too lossy?** K latents may bottleneck localization. K is a knob;
  ablate K ∈ {16, 64, 256}.
- **Honest novelty check.** Each primitive exists (Perceiver bottleneck, FiLM,
  coarse-to-fine DETR, deep supervision). The bet is the *composition + the
  global-first supervision + the sim2real evaluation*. If G1 doesn't beat B1, the
  cleaner takeaway may be "frozen foundation features already capture the
  global-first benefit," which is itself a publishable negative result for our
  setting.

---

## 9. Single-file research template

For iterating on the architecture itself (without the training/eval harness),
`docs/gfn_template.py` is a self-contained condensation of the GFN: the
`GistFirstNetwork` module, the Mask2Former integration, and a runnable
self-test, with every perturbation point tagged `# >>> RESEARCH KNOB` and all
knobs hoisted into a single `GFNConfig`. It exposes more knobs than the
production `train_seg_detr.py` (multi-scale gist via `gist_source_stages`;
`gate_type ∈ {film, seed, none, xattn}` where `xattn` is the §5.1 gist-gated
cross-attention upgrade). Run `python docs/gfn_template.py` for the contract
self-test. The autoresearch loop (`autoresearch/PROTOCOL.md`) drives changes to
these knobs round by round.

## 10. Status & next step

**Done:** GFN is implemented as `--backbone gfn` in `train_seg_detr.py`, reusing
the `Dinov2SimpleFPN` backbone contract (`.channels` + `.feature_maps`). The gist
is computed by K Perceiver-style latents cross-attending the frozen-DINOv2 tokens;
the FiLM gate is zero-initialised (starts as identity, learns to gate); the
global-first aux loss is a BCE multi-label head on the gist, wired into the train
loop and weighted by `--gfn-aux-weight`. Smoke-tested: gate-on/aux-on trains with
a descending aux loss; gate-off/aux-off reduces exactly to DINOv2+SFPN.

**Next:** run B1/G1/A1/A2 above to 30 epochs and compare the sim→real gap; add the
secondary probes (linear-probe the gist on sim vs real; per-class gain breakdown).
If G1 ≳ B1, wire the richer top-down variant (per-level cross-attention reads
instead of FiLM-only, and the A3 fine→coarse control).

## References

- Ahissar & Hochstein (2004), *The reverse hierarchy theory of visual perceptual learning.*
- Jaegle et al. (2021), *Perceiver / Perceiver IO.*
- Locatello et al. (2020), *Object-Centric Learning with Slot Attention.*
- Zhu et al. (2021), *Deformable DETR.*  Cai & Vasconcelos (2018), *Cascade R-CNN.*
- Rao & Ballard (1999), *Predictive coding.*  Lotter et al. (2017), *PredNet.*  Zamir et al. (2017), *Feedback Networks.*
- Hinton (2021), *How to represent part-whole hierarchies in a neural network (GLOM).*
- Carion et al. (2020), *DETR.*  Cheng et al. (2022), *Mask2Former.*  Oquab et al. (2023), *DINOv2.*
