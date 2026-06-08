"""Gist-First Network (GFN) — single-file research template.

A self-contained, hackable condensation of the global-first backbone described
in docs/global-first-architecture.md and implemented (for training) inside
../train_seg_detr.py. Everything you need to read, understand, and modify the
architecture lives in THIS file; the production training/eval harness imports
the same logic but is interleaved with data loading and COCO eval, which makes
it harder to iterate on the model itself.

THESIS (why this shape): bottom-up nets commit to texture/edges before
semantics, and texture is exactly what shifts sim->real. GFN forms a global
semantic "gist" FIRST (Perceiver bottleneck over frozen DINOv2 tokens,
supervised image-level), then runs a top-down coarse->fine pyramid in which the
gist GATES every low-level read. The category commitment is made from
domain-robust structure; pixel/texture noise arrives late and conditionally.

BACKBONE CONTRACT (what Mask2Former needs from us): expose `.channels`
(list[int], one per pyramid level) and return an object with `.feature_maps`
(tuple of 4 tensors, high->low resolution, each `.channels[i]` channels). Match
that and GFN drops in behind the `--backbone` switch with the pixel decoder +
mask head + COCO eval reused verbatim.

ALL research knobs are in `GFNConfig` below, and every place worth perturbing is
tagged `# >>> RESEARCH KNOB`. The search menu in ../autoresearch/PROTOCOL.md maps
1:1 onto these.

Run the self-test (instantiates the real model, dummy forward, checks the
contract + prints trainable/frozen params):

    /home/edge-host/Documents/.venv/bin/python docs/gfn_template.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config — the entire research surface in one place
# --------------------------------------------------------------------------- #
@dataclass
class GFNConfig:
    # --- frozen foundation features F ---
    dino_name: str = "facebook/dinov2-base"
    freeze_backbone: bool = True          # >>> RESEARCH KNOB: frozen vs fine-tune ViT
    # >>> RESEARCH KNOB (multi-scale gist): which ViT stages feed the GIST.
    # The spatial pyramid always uses the last stage; the gist may read more.
    # e.g. [-1] = last block only (production); [-4, -1] = mid + late tokens.
    gist_source_stages: list[int] = field(default_factory=lambda: [-1])

    # --- gist encoder (Perceiver-style global bottleneck) ---
    num_latents: int = 64                 # >>> RESEARCH KNOB: K (gist capacity)
    gist_layers: int = 2                  # >>> RESEARCH KNOB: gist self-attn depth

    # --- coupling: how the gist drives the spatial path ---
    # "film"  : FiLM-gate every pyramid level + seed the coarsest with the gist (production)
    # "seed"  : only seed the coarsest level, no per-level gate
    # "none"  : gist touches ONLY the aux loss; pyramid == plain DINOv2+SFPN (ablation A1)
    # "xattn" : (extension stub) per-level gist-gated cross-attention — see GistGate
    gate_type: str = "film"               # >>> RESEARCH KNOB: coupling mechanism

    # --- global-first supervision ---
    num_labels: int = 43
    aux_target: str = "presence"          # >>> RESEARCH KNOB: "presence" | (ext) "counts"

    # bookkeeping
    hidden_size: int | None = None        # filled from the DINOv2 config
    num_heads: int | None = None


def group_norm(c: int) -> nn.GroupNorm:
    """GroupNorm with the largest divisor group count <= 32 (channel-count safe)."""
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


# --------------------------------------------------------------------------- #
# (Extension) the richer coupling atom from design-note §5.1.
# Production uses FiLM-only; this is the "gist-gated cross-attention" upgrade:
# a spatial state reads image features through queries steered by the gist.
# Wire it into GistFirstNetwork.forward by setting gate_type="xattn" and
# replacing the FiLM `gate()` calls. Left as a clearly-marked, untrained stub.
# --------------------------------------------------------------------------- #
class GistGate(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.film = nn.Linear(dim, 2 * dim)
        self.q = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))

    def forward(self, state: torch.Tensor, feats: torch.Tensor,
                gist: torch.Tensor) -> torch.Tensor:
        # state, feats: (B, C, H, W); gist: (B, C)
        B, C, H, W = state.shape
        s = state.flatten(2).transpose(1, 2)          # (B, HW, C)
        f = feats.flatten(2).transpose(1, 2)
        gamma, beta = self.film(gist).chunk(2, dim=-1)
        q = (1 + gamma[:, None, :]) * self.q(s) + beta[:, None, :]
        s = s + self.attn(q, f, f, need_weights=False)[0]
        s = s + self.ffn(self.norm(s))
        return s.transpose(1, 2).reshape(B, C, H, W)


# --------------------------------------------------------------------------- #
# The Gist-First Network
# --------------------------------------------------------------------------- #
class GistFirstNetwork(nn.Module):
    """Frozen DINOv2 -> global gist (computed & supervised FIRST) -> top-down
    Simple Feature Pyramid gated by the gist. After forward(), `last_aux_logits`
    holds the image-level class logits for the global-first loss."""

    def __init__(self, cfg: GFNConfig):
        super().__init__()
        from transformers import AutoBackbone, AutoConfig

        dcfg = AutoConfig.from_pretrained(cfg.dino_name)
        H = cfg.hidden_size = dcfg.hidden_size
        heads = cfg.num_heads = dcfg.num_attention_heads
        n_stage = dcfg.num_hidden_layers
        self.cfg = cfg

        # Frozen dense features F. We request every stage the gist might read,
        # plus the last (which always feeds the spatial pyramid).
        stages = sorted({(s % n_stage) + 1 for s in cfg.gist_source_stages}
                        | {n_stage})
        self._gist_stage_names = [f"stage{(s % n_stage) + 1}"
                                  for s in cfg.gist_source_stages]
        self._last_stage_name = f"stage{n_stage}"
        self.dino = AutoBackbone.from_pretrained(
            cfg.dino_name, out_features=[f"stage{s}" for s in stages])
        self.freeze = cfg.freeze_backbone
        if self.freeze:
            for p in self.dino.parameters():
                p.requires_grad_(False)
            self.dino.eval()

        # --- gist encoder: K latents cross-attend F, then self-attend ---
        self.latents = nn.Parameter(torch.randn(cfg.num_latents, H) * 0.02)
        self.gist_cross = nn.MultiheadAttention(H, heads, batch_first=True)
        self.gist_cross_norm = nn.LayerNorm(H)
        enc = nn.TransformerEncoderLayer(H, heads, dim_feedforward=4 * H,
                                         batch_first=True, norm_first=True)
        self.gist_self = nn.TransformerEncoder(enc, num_layers=cfg.gist_layers)
        self.gist_pool_norm = nn.LayerNorm(H)
        self.aux_head = nn.Linear(H, cfg.num_labels)   # global-first supervision

        # --- FiLM gate from the gist; zero-init => starts as identity so
        #     training begins at the plain-pyramid solution and learns to gate.
        self.film = nn.Linear(H, 2 * H)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.gate_blocks = None
        if cfg.gate_type == "xattn":                   # >>> RESEARCH KNOB
            self.gate_blocks = nn.ModuleList(
                [GistGate(H, heads) for _ in range(4)])

        # --- Simple Feature Pyramid over the last-stage features (ViTDet) ---
        self.fpn4 = nn.Sequential(
            nn.ConvTranspose2d(H, H // 2, 2, 2), group_norm(H // 2), nn.GELU(),
            nn.ConvTranspose2d(H // 2, H // 4, 2, 2),
        )
        self.fpn8 = nn.ConvTranspose2d(H, H // 2, 2, 2)
        self.fpn16 = nn.Identity()
        self.fpn32 = nn.MaxPool2d(2, 2)

        def head(c_in: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, H, 1, bias=False), group_norm(H),
                nn.Conv2d(H, H, 3, padding=1, bias=False), group_norm(H),
            )

        self.out4, self.out8 = head(H // 4), head(H // 2)
        self.out16, self.out32 = head(H), head(H)
        self.channels = [H, H, H, H]          # backbone contract: 4 levels
        self.last_aux_logits = None

    def train(self, mode: bool = True) -> "GistFirstNetwork":
        super().train(mode)
        if self.freeze:
            self.dino.eval()                  # keep frozen ViT deterministic
        return self

    # --- gist: K latents -> global vector ---
    def _gist(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        z, _ = self.gist_cross(q, tokens, tokens, need_weights=False)
        z = self.gist_cross_norm(z + q)
        z = self.gist_self(z)                 # (B, K, H)
        return self.gist_pool_norm(z.mean(dim=1))   # (B, H)

    def forward(self, pixel_values: torch.Tensor, **kw) -> SimpleNamespace:
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            fmaps = self.dino(pixel_values).feature_maps
        # index the requested feature maps by their position in out_features
        names = list(self.dino.out_features)
        feat = fmaps[names.index(self._last_stage_name)]      # (B, H, gh, gw)

        # gist tokens: concat tokens from the requested stage(s) (multi-scale gist)
        gist_tokens = torch.cat(
            [fmaps[names.index(n)].flatten(2).transpose(1, 2)
             for n in self._gist_stage_names], dim=1)         # (B, sum N, H)
        g = self._gist(gist_tokens)                           # GLOBAL GIST FIRST
        # aux logits in BOTH modes: train uses them for the global-first loss,
        # eval reads them as the gist-transfer probe (see PROTOCOL.md).
        self.last_aux_logits = self.aux_head(g)

        use_gate = self.cfg.gate_type in ("film", "xattn")
        gamma, beta = self.film(g).chunk(2, dim=-1)

        def film(m: torch.Tensor) -> torch.Tensor:
            if self.cfg.gate_type not in ("film",):
                return m
            return (1 + gamma[:, :, None, None]) * m + beta[:, :, None, None]

        f32 = self.fpn32(feat)
        if self.cfg.gate_type in ("film", "seed"):            # seed coarsest with gist
            f32 = f32 + g[:, :, None, None]

        levels = [self.out4(self.fpn4(feat)), self.out8(self.fpn8(feat)),
                  self.out16(self.fpn16(feat)), self.out32(f32)]
        if self.cfg.gate_type == "xattn":                     # extension path
            levels = [blk(lvl, lvl, g) for blk, lvl in zip(self.gate_blocks, levels)]
        else:
            levels = [film(lvl) for lvl in levels]
        return SimpleNamespace(feature_maps=tuple(levels))


# --------------------------------------------------------------------------- #
# Integration: slot GFN into Mask2Former (decoder warm-started from a COCO ckpt)
# --------------------------------------------------------------------------- #
def build_gfn_mask2former(cfg: GFNConfig, id2label: dict,
                          decoder_ckpt: str = "facebook/mask2former-swin-tiny-coco-instance"):
    """Build a Mask2Former whose pixel-decoder input projections are sized to H
    channels, swap in GFN as the encoder, and warm-start every non-encoder
    tensor (decoder, pixel decoder) from the COCO checkpoint."""
    from transformers import (AutoConfig, Dinov2Config, Mask2FormerConfig,
                              Mask2FormerForUniversalSegmentation)

    H = AutoConfig.from_pretrained(cfg.dino_name).hidden_size
    label2id = {v: k for k, v in id2label.items()}

    # dummy 4-stage backbone_config only sizes the pixel-decoder projections to H
    dummy = Dinov2Config(hidden_size=H, num_hidden_layers=4,
                         num_attention_heads=max(1, H // 64), patch_size=14,
                         image_size=518,
                         out_features=["stage1", "stage2", "stage3", "stage4"])
    base = Mask2FormerConfig.from_pretrained(decoder_ckpt)
    base.backbone_config, base.backbone = dummy, None
    base.use_timm_backbone = base.use_pretrained_backbone = False
    base.id2label, base.label2id = id2label, label2id
    base.num_labels = len(id2label)
    model = Mask2FormerForUniversalSegmentation(base)

    src = Mask2FormerForUniversalSegmentation.from_pretrained(decoder_ckpt).state_dict()
    tgt = model.state_dict()
    for k, v in tgt.items():
        if "pixel_level_module.encoder" in k:
            continue                          # encoder is GFN, not the dummy
        if k in src and src[k].shape == v.shape:
            tgt[k] = src[k]
    model.load_state_dict(tgt)
    model.model.pixel_level_module.encoder = GistFirstNetwork(cfg)
    return model


# --------------------------------------------------------------------------- #
# Self-test: real model, dummy forward, contract + param report
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    cfg = GFNConfig()
    print(f"[gfn] building encoder ({cfg.dino_name}, gate={cfg.gate_type}, "
          f"K={cfg.num_latents}, gist_stages={cfg.gist_source_stages}) ...")
    enc = GistFirstNetwork(cfg).eval()

    # DINOv2 patch size is 14; use a multiple of 14 so the ViT tiles cleanly.
    x = torch.randn(2, 3, 14 * 18, 14 * 32)         # (B, 3, 252, 448)
    with torch.no_grad():
        out = enc(x)

    assert hasattr(out, "feature_maps") and len(out.feature_maps) == 4
    assert enc.channels == [cfg.hidden_size] * 4
    print(f"[gfn] H={cfg.hidden_size}  channels={enc.channels}")
    for i, fm in enumerate(out.feature_maps):
        print(f"   level {i}: {tuple(fm.shape)}")          # high->low res
    assert enc.last_aux_logits.shape == (2, cfg.num_labels)
    print(f"[gfn] aux logits: {tuple(enc.last_aux_logits.shape)} (global-first head)")

    tr = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    fr = sum(p.numel() for p in enc.parameters() if not p.requires_grad)
    print(f"[gfn] params: {tr/1e6:.1f}M trainable, {fr/1e6:.1f}M frozen (ViT)")
    print("[gfn] contract OK — drops into Mask2Former via build_gfn_mask2former().")


if __name__ == "__main__":
    _selftest()
