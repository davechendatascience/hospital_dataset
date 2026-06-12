"""
Train a DETR-family *instance-segmentation* model on the sim splits of the
ward_v1 dataset and evaluate mask mAP on valid + test + real_test every epoch.

Unlike Ultralytics RT-DETR (detection only, no mask head), the DETR-family
segmentation models live in HuggingFace `transformers`. They share an almost
identical API, so this one script can train any of them by name. We start with
Mask2Former + Swin-Tiny ("DETR with swin-tiny"):

    facebook/mask2former-swin-tiny-coco-instance

Backbones (`--backbone`):
  * swin (default): the COCO-pretrained Mask2Former above, fully fine-tuned.
  * dinov2: a *frozen* self-supervised DINOv2 ViT + a ViTDet-style Simple
    Feature Pyramid, with the Mask2Former decoder warm-started from --model.
    This isolates the sim->real transfer of high-level (semantic, texture-
    invariant) features: the ViT never sees a gradient, so it cannot memorise
    sim-specific texture. Train on sim only and compare real_test AP to the
    swin baseline. Example:

        /home/edge-host/Documents/.venv/bin/python train_seg_detr.py \
            --data ward_v1 --backbone dinov2 \
            --dino-name facebook/dinov2-base \
            --epochs 30 --batch 4 --short-edge 504 --device 0

Input format: the existing COCO instance-seg annotations
(ward_v1/<split>/_annotations.coco.json, RLE masks). Category ids in the COCO
files are 0=ward_object (background) and 1..43 for the real classes; we map the
real classes to contiguous model labels 0..42 (label = coco_id - 1) so results
line up 1:1 with the YOLO / RT-DETR runs.

Evaluation: standard COCO mask AP via pycocotools (segm), computed against the
untouched ground-truth json of each split, so the numbers are directly
comparable to any other COCO segm evaluation.

Run with the project venv python, e.g. a 2-epoch smoke test:

    /home/edge-host/Documents/.venv/bin/python train_seg_detr.py \
        --data ward_v1 \
        --model facebook/mask2former-swin-tiny-coco-instance \
        --epochs 2 --batch 2 --short-edge 512 --device 0

Add --max-train-samples 200 for a fast sanity run on a subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ROS2_bridge" / "src"))
from fixed_categories import FIXED_CATEGORIES  # type: ignore  # noqa: E402

# Contiguous model labels 0..42 for the 43 real classes (drop background id 0).
# model_label = coco_category_id - 1  <->  coco_category_id = model_label + 1
ID2LABEL = {cid - 1: name for name, cid in FIXED_CATEGORIES.items() if cid != 0}
LABEL2ID = {name: lid for lid, name in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

TRAIN_SPLIT = "train"
DEFAULT_EVAL_SPLITS = ("valid", "test", "real_test")


# --------------------------------------------------------------------------- #
# DINOv2 backbone (frozen ViT + ViTDet Simple Feature Pyramid)
# --------------------------------------------------------------------------- #
def _group_norm(c: int) -> nn.GroupNorm:
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


class Dinov2SimpleFPN(nn.Module):
    """A plain DINOv2 ViT (single-scale, stride-14 features) wrapped in a
    ViTDet-style Simple Feature Pyramid so Mask2Former gets the 4-level pyramid
    its pixel decoder expects.

    The four outputs are ordered high->low resolution (~strides [4,8,16,32]),
    each with `hidden_size` channels so they line up 1:1 with the pixel
    decoder's input projections. When `freeze=True` the ViT runs under no_grad
    and is kept in eval mode, so only the FPN + Mask2Former decoder train and
    the pretrained (domain-invariant) features are preserved verbatim.

    Conforms to the minimal HF backbone contract Mask2Former relies on:
    exposes `.channels` and returns an object with `.feature_maps`.
    """

    def __init__(self, name: str, freeze: bool = True):
        super().__init__()
        from transformers import AutoBackbone, AutoConfig

        cfg = AutoConfig.from_pretrained(name)
        n_layers = cfg.num_hidden_layers
        H = cfg.hidden_size
        # Single feature map from the last block feeds the pyramid (ViTDet).
        self.dino = AutoBackbone.from_pretrained(name, out_features=[f"stage{n_layers}"])
        self.freeze = freeze
        if freeze:
            for p in self.dino.parameters():
                p.requires_grad_(False)
            self.dino.eval()

        # Simple Feature Pyramid: upsample x4 / x2, identity, downsample x2.
        self.fpn4 = nn.Sequential(
            nn.ConvTranspose2d(H, H // 2, 2, 2), _group_norm(H // 2), nn.GELU(),
            nn.ConvTranspose2d(H // 2, H // 4, 2, 2),
        )
        self.fpn8 = nn.ConvTranspose2d(H, H // 2, 2, 2)
        self.fpn16 = nn.Identity()
        self.fpn32 = nn.MaxPool2d(2, 2)

        def head(c_in: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, H, 1, bias=False), _group_norm(H),
                nn.Conv2d(H, H, 3, padding=1, bias=False), _group_norm(H),
            )

        self.out4 = head(H // 4)
        self.out8 = head(H // 2)
        self.out16 = head(H)
        self.out32 = head(H)
        self.channels = [H, H, H, H]

    def train(self, mode: bool = True) -> "Dinov2SimpleFPN":
        super().train(mode)
        if self.freeze:  # keep the frozen ViT deterministic (no drop_path etc.)
            self.dino.eval()
        return self

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> SimpleNamespace:
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            feat = self.dino(pixel_values).feature_maps[0]  # (B, H, gh, gw)
        feature_maps = (
            self.out4(self.fpn4(feat)),
            self.out8(self.fpn8(feat)),
            self.out16(self.fpn16(feat)),
            self.out32(self.fpn32(feat)),
        )
        return SimpleNamespace(feature_maps=feature_maps)


class LejepaResNetPyramid(nn.Module):
    """A ResNet-50 pretrained with OUR in-domain LeJEPA (train_lejepa.py) as the
    Mask2Former backbone. ResNets are natively pyramidal -- layer1..layer4 give
    strides 4/8/16/32 -- so each level is just projected to a common width H
    for the pixel decoder. With freeze=True the ResNet (incl. BatchNorm running
    stats) is frozen in eval mode: the test is whether features pretrained
    jointly on sim+styled+real survive the sim->real shift better than generic
    pretrained backbones. Same contract as Dinov2SimpleFPN."""

    def __init__(self, ckpt_path: str, H: int = 768, freeze: bool = True):
        super().__init__()
        from torchvision.models import resnet50

        net = resnet50(weights=None)
        net.fc = nn.Identity()
        ck = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = net.load_state_dict(ck["model"], strict=False)
        assert not unexpected, f"unexpected keys in lejepa ckpt: {unexpected[:5]}"
        self.ckpt_epoch = ck.get("epoch", "?")
        # `res` prefix is load-bearing: the optimizer's _is_backbone() matcher
        # uses it to separate frozen backbone params from the new projections
        self.res = nn.Sequential()
        self.res.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.res.layer1, self.res.layer2 = net.layer1, net.layer2
        self.res.layer3, self.res.layer4 = net.layer3, net.layer4
        self.freeze = freeze
        if freeze:
            for p in self.res.parameters():
                p.requires_grad_(False)
            self.res.eval()

        def head(c_in: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, H, 1, bias=False), _group_norm(H),
                nn.Conv2d(H, H, 3, padding=1, bias=False), _group_norm(H),
            )

        self.out4, self.out8 = head(256), head(512)
        self.out16, self.out32 = head(1024), head(2048)
        self.channels = [H, H, H, H]

    def train(self, mode: bool = True) -> "LejepaResNetPyramid":
        super().train(mode)
        if self.freeze:   # keep frozen BN running stats / deterministic feats
            self.res.eval()
        return self

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> SimpleNamespace:
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            x = self.res.stem(pixel_values)
            c2 = self.res.layer1(x)
            c3 = self.res.layer2(c2)
            c4 = self.res.layer3(c3)
            c5 = self.res.layer4(c4)
        return SimpleNamespace(feature_maps=(
            self.out4(c2), self.out8(c3), self.out16(c4), self.out32(c5)))


class GistFirstNetwork(nn.Module):
    """Global-first ("reverse hierarchy") backbone. See
    docs/global-first-architecture.md.

    A frozen DINOv2 supplies dense features F. K learned latents cross-attend F
    into a global semantic gist g (computed FIRST, supervised image-level via an
    auxiliary head). g then drives the spatial path top-down: it seeds the
    coarsest pyramid level and FiLM-gates every level of a Simple Feature
    Pyramid built from F, so each low-level read is conditioned on the semantic
    commitment. With `use_gate=False` the gist touches only the aux loss and the
    pyramid reduces to Dinov2SimpleFPN (ablation A1).

    Same backbone contract as Dinov2SimpleFPN: exposes `.channels` and returns an
    object with `.feature_maps`. After forward(), `self.last_aux_logits` holds the
    image-level class logits (or None in eval) for the global-first loss, read by
    the training loop.
    """

    def __init__(self, name: str, num_labels: int, freeze: bool = True,
                 use_gate: bool = True, num_latents: int = 64, gist_layers: int = 2):
        super().__init__()
        from transformers import AutoBackbone, AutoConfig

        cfg = AutoConfig.from_pretrained(name)
        H = cfg.hidden_size
        heads = cfg.num_attention_heads
        self.dino = AutoBackbone.from_pretrained(
            name, out_features=[f"stage{cfg.num_hidden_layers}"])
        self.freeze = freeze
        if freeze:
            for p in self.dino.parameters():
                p.requires_grad_(False)
            self.dino.eval()
        self.use_gate = use_gate

        # --- gist encoder: Perceiver-style global bottleneck over F ---
        self.latents = nn.Parameter(torch.randn(num_latents, H) * 0.02)
        self.gist_cross = nn.MultiheadAttention(H, heads, batch_first=True)
        self.gist_cross_norm = nn.LayerNorm(H)
        enc_layer = nn.TransformerEncoderLayer(
            H, heads, dim_feedforward=4 * H, batch_first=True, norm_first=True)
        self.gist_self = nn.TransformerEncoder(enc_layer, num_layers=gist_layers)
        self.gist_pool_norm = nn.LayerNorm(H)
        self.aux_head = nn.Linear(H, num_labels)

        # FiLM gate from the gist; zero-init -> starts as an identity gate so
        # training begins from the plain-pyramid solution and learns to gate.
        self.film = nn.Linear(H, 2 * H)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        # --- Simple Feature Pyramid (identical to Dinov2SimpleFPN) ---
        self.fpn4 = nn.Sequential(
            nn.ConvTranspose2d(H, H // 2, 2, 2), _group_norm(H // 2), nn.GELU(),
            nn.ConvTranspose2d(H // 2, H // 4, 2, 2),
        )
        self.fpn8 = nn.ConvTranspose2d(H, H // 2, 2, 2)
        self.fpn16 = nn.Identity()
        self.fpn32 = nn.MaxPool2d(2, 2)

        def head(c_in: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, H, 1, bias=False), _group_norm(H),
                nn.Conv2d(H, H, 3, padding=1, bias=False), _group_norm(H),
            )

        self.out4 = head(H // 4)
        self.out8 = head(H // 2)
        self.out16 = head(H)
        self.out32 = head(H)
        self.channels = [H, H, H, H]
        self.last_aux_logits = None

    def train(self, mode: bool = True) -> "GistFirstNetwork":
        super().train(mode)
        if self.freeze:
            self.dino.eval()
        return self

    def _gist(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        z, _ = self.gist_cross(q, tokens, tokens, need_weights=False)
        z = self.gist_cross_norm(z + q)
        z = self.gist_self(z)                       # (B, K, H)
        return self.gist_pool_norm(z.mean(dim=1))   # (B, H) global gist

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> SimpleNamespace:
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            feat = self.dino(pixel_values).feature_maps[0]   # (B, H, gh, gw)
        tokens = feat.flatten(2).transpose(1, 2)             # (B, N, H)
        g = self._gist(tokens)                               # global gist FIRST
        # Compute aux logits in BOTH train and eval: training uses them for the
        # global-first loss; eval reads them as the gist-transfer probe (does
        # the global object-inventory commitment survive the sim->real shift?).
        self.last_aux_logits = self.aux_head(g)

        gamma, beta = self.film(g).chunk(2, dim=-1)          # (B, H), (B, H)

        def gate(m: torch.Tensor) -> torch.Tensor:
            if not self.use_gate:
                return m
            return (1 + gamma[:, :, None, None]) * m + beta[:, :, None, None]

        f32 = self.fpn32(feat)
        if self.use_gate:                                    # S_coarse <- broadcast(z)
            f32 = f32 + g[:, :, None, None]
        # Heads project each level to H channels; the FiLM gate (also H) is
        # applied to the head outputs, so the gist conditions every level.
        feature_maps = (
            gate(self.out4(self.fpn4(feat))),
            gate(self.out8(self.fpn8(feat))),
            gate(self.out16(self.fpn16(feat))),
            gate(self.out32(f32)),
        )
        return SimpleNamespace(feature_maps=feature_maps)


def _build_m2f_shell(args, H: int):
    """Build a Mask2Former whose pixel decoder is sized to H-channel features and
    whose decoder is warm-started from the COCO checkpoint --model. The encoder
    is a throwaway dummy ViT to be swapped for a real H-channel backbone."""
    from transformers import (Dinov2Config, Mask2FormerConfig,
                              Mask2FormerForUniversalSegmentation)

    # Dummy 4-stage backbone_config: only used so Mask2Former sizes its four
    # pixel-decoder input projections to H channels. The real backbone is swapped
    # in by the caller.
    dummy = Dinov2Config(
        hidden_size=H, num_hidden_layers=4, num_attention_heads=max(1, H // 64),
        patch_size=14, image_size=518,
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )
    base = Mask2FormerConfig.from_pretrained(args.model)
    base.backbone_config = dummy
    base.backbone = None
    base.use_timm_backbone = False
    base.use_pretrained_backbone = False
    base.id2label = ID2LABEL
    base.label2id = LABEL2ID
    base.num_labels = NUM_LABELS
    model = Mask2FormerForUniversalSegmentation(base)

    # Warm-start everything except the backbone and shape-mismatched tensors
    # (the 80->43 class head and the pixel-decoder input projections) from the
    # COCO Mask2Former, so only the backbone prior + class head start fresh.
    src = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).state_dict()
    tgt = model.state_dict()
    copied = 0
    for k, v in tgt.items():
        if "pixel_level_module.encoder" in k:
            continue
        if k in src and src[k].shape == v.shape:
            tgt[k] = src[k]
            copied += 1
    model.load_state_dict(tgt)
    return model, copied


def build_lejepa_mask2former(args) -> "nn.Module":
    """Mask2Former with a frozen in-domain LeJEPA ResNet-50 backbone."""
    H = 768
    model, copied = _build_m2f_shell(args, H)
    enc = LejepaResNetPyramid(str(args.lejepa_ckpt), H=H,
                              freeze=args.freeze_backbone)
    model.model.pixel_level_module.encoder = enc
    print(f"[seg-detr] lejepa backbone {args.lejepa_ckpt} "
          f"(ssl epoch {enc.ckpt_epoch}, H={H}, freeze={args.freeze_backbone}); "
          f"decoder warm-started ({copied} tensors) from {args.model}")
    return model


def build_dinov2_mask2former(args) -> "nn.Module":
    """Mask2Former with a frozen DINOv2 + Simple Feature Pyramid backbone."""
    from transformers import AutoConfig
    H = AutoConfig.from_pretrained(args.dino_name).hidden_size
    model, copied = _build_m2f_shell(args, H)
    model.model.pixel_level_module.encoder = Dinov2SimpleFPN(
        args.dino_name, freeze=args.freeze_backbone)
    print(f"[seg-detr] dinov2 backbone {args.dino_name} (H={H}, "
          f"freeze={args.freeze_backbone}); decoder warm-started "
          f"({copied} tensors) from {args.model}")
    return model


def build_gfn_mask2former(args) -> "nn.Module":
    """Mask2Former with the Gist-First Network backbone (global-first design)."""
    from transformers import AutoConfig
    H = AutoConfig.from_pretrained(args.dino_name).hidden_size
    model, copied = _build_m2f_shell(args, H)
    model.model.pixel_level_module.encoder = GistFirstNetwork(
        args.dino_name, num_labels=NUM_LABELS, freeze=args.freeze_backbone,
        use_gate=args.gfn_gate, num_latents=args.gfn_latents)
    print(f"[seg-detr] gfn backbone {args.dino_name} (H={H}, "
          f"freeze={args.freeze_backbone}, gate={args.gfn_gate}, "
          f"latents={args.gfn_latents}, aux_weight={args.gfn_aux_weight}); "
          f"decoder warm-started ({copied} tensors) from {args.model}")
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, required=True,
                   help="EVAL dataset root containing <split>/images + "
                        "<split>/_annotations.coco.json (canonical real splits)")
    p.add_argument("--train-data", default=None,
                   help="Comma-separated dataset root(s) to draw the TRAIN split "
                        "from; their train/ folders are concatenated. Defaults to "
                        "--data. Examples: 'ward_v1_styled_flat' (styled only), "
                        "'ward_v1,ward_v1_styled_flat' (sim+styled = mixed).")
    p.add_argument("--model", default="facebook/mask2former-swin-tiny-coco-instance",
                   help="HF model id (Mask2Former / MaskFormer instance-seg checkpoint). "
                        "For --backbone dinov2 this is the decoder warm-start source.")
    p.add_argument("--backbone", choices=["swin", "dinov2", "gfn", "lejepa"],
                   default="swin",
                   help="swin = COCO Mask2Former (backbone frozen by default; "
                        "--no-freeze-backbone to fully fine-tune). "
                        "dinov2 = frozen DINOv2 ViT + Simple Feature Pyramid. "
                        "gfn = Gist-First Network (global-first: gist from frozen "
                        "DINOv2 gates a top-down pyramid). All warm-start the "
                        "decoder from --model.")
    p.add_argument("--lejepa-ckpt", type=Path, default=None,
                   help="(lejepa) checkpoint from train_lejepa.py; default: the "
                        "newest lejepa_runs/*/ckpt_*.pt.")
    p.add_argument("--dino-name", default="facebook/dinov2-base",
                   help="(dinov2) HF DINOv2 id, e.g. facebook/dinov2-base, "
                        "facebook/dinov2-large, facebook/dinov2-with-registers-base.")
    p.add_argument("--freeze-backbone", dest="freeze_backbone",
                   action="store_true", default=True,
                   help="freeze the backbone (swin encoder / DINOv2 ViT); train "
                        "only pyramid + decoder + heads (default).")
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone",
                   action="store_false",
                   help="also fine-tune the backbone (ablation: higher sim AP, "
                        "expected lower real AP).")
    p.add_argument("--gfn-gate", dest="gfn_gate", action="store_true", default=True,
                   help="(gfn) FiLM-gate the pyramid with the gist + seed the "
                        "coarsest level from it (default on).")
    p.add_argument("--no-gfn-gate", dest="gfn_gate", action="store_false",
                   help="(gfn) ablation A1: disable gating; gist drives only the "
                        "aux loss and the pyramid reduces to plain DINOv2+SFPN.")
    p.add_argument("--gfn-aux-weight", type=float, default=0.5,
                   help="(gfn) weight of the image-level multi-label gist loss "
                        "(global-first supervision). Set 0 for ablation A2.")
    p.add_argument("--gfn-latents", type=int, default=64,
                   help="(gfn) number of global gist latents K.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1,
                   help="LR multiplier for the (pretrained) backbone params")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=0.01,
                   help="Max grad norm (Mask2Former official uses 0.01)")
    p.add_argument("--short-edge", type=int, default=512,
                   help="Image processor shortest edge. Lower = faster/less VRAM "
                        "(native images are 1920x1080).")
    p.add_argument("--long-edge", type=int, default=1333)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval-splits", default=",".join(DEFAULT_EVAL_SPLITS),
                   help="Comma-separated splits to evaluate every epoch")
    p.add_argument("--gap-pair", default="valid,real_dev",
                   help="'<sim_split>,<real_split>' whose AP difference is "
                        "logged each epoch as the sim->real gap column.")
    p.add_argument("--dump-overlays", type=int, default=0,
                   help="If >0, save this many predicted-mask overlays per "
                        "epoch from --overlay-split (qualitative inspection).")
    p.add_argument("--overlay-split", default=None,
                   help="Split to draw overlays from (default: first eval split).")
    p.add_argument("--align-real", type=Path, default=None,
                   help="dir of UNLABELED real images (e.g. ward_v4/real_dev/"
                        "images): adds an MMD loss pulling the encoder's pooled "
                        "pyramid features on the sim batch toward this set's "
                        "distribution (unsupervised domain adaptation). Needs "
                        "trainable encoder params (dinov2/gfn/lejepa "
                        "projections, or --no-freeze-backbone swin).")
    p.add_argument("--align-weight", type=float, default=0.1,
                   help="weight of the MMD alignment loss.")
    p.add_argument("--align-batch", type=int, default=8,
                   help="real images per alignment step (more = steadier MMD).")
    p.add_argument("--align-local", type=int, default=256,
                   help="ALSO align per-LOCATION features: MMD over this many "
                        "spatial positions sampled from the --align-local-level "
                        "pyramid map (0 = global-only). Touches local structure "
                        "the image-pooled term misses.")
    p.add_argument("--align-local-level", type=int, default=1,
                   help="pyramid level for local alignment (0=stride4 .. 3=stride32; "
                        "earlier = lower-level structure, less class-sharp).")
    p.add_argument("--align-ema", type=float, default=0.1,
                   help="EMA update rate for per-domain feature means; adds an "
                        "L2 penalty between EMA-stabilized means to tame "
                        "small-batch MMD noise (0 = off).")
    p.add_argument("--eval-batch", type=int, default=2)
    p.add_argument("--score-thresh", type=float, default=0.5,
                   help="Min score for a predicted mask to count in eval")
    p.add_argument("--max-train-samples", type=int, default=0,
                   help="If >0, train on only the first N images (smoke test)")
    p.add_argument("--max-eval-samples", type=int, default=0,
                   help="If >0, evaluate on only the first N images per split")
    p.add_argument("--eval-first", action="store_true",
                   help="Run eval once before training (epoch 0 baseline)")
    p.add_argument("--project", default="runs/seg_detr")
    p.add_argument("--name", default="mask2former_swin_tiny")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CocoInstanceSeg(Dataset):
    """Yields raw items; tensorization happens in the collate fns so the image
    processor can pad a whole batch consistently."""

    def __init__(self, split_dir: Path, max_samples: int = 0):
        self.images_dir = split_dir / "images"
        if not self.images_dir.is_dir():
            self.images_dir = split_dir   # images in split root (build_dataset)
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.is_file():
            raise FileNotFoundError(f"missing {ann_path}")
        self.ann_path = ann_path
        self.coco = COCO(str(ann_path))
        self.img_ids = sorted(self.coco.imgs.keys())
        if max_samples > 0:
            self.img_ids = self.img_ids[:max_samples]

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.img_ids[idx]
        info = self.coco.imgs[img_id]
        h, w = info["height"], info["width"]
        image = Image.open(self.images_dir / info["file_name"]).convert("RGB")

        anns = self.coco.imgToAnns.get(img_id, [])
        inst_map = np.zeros((h, w), dtype=np.int32)
        inst2sem: dict[int, int] = {}
        for i, ann in enumerate(anns):
            cid = ann["category_id"]
            if cid == 0:  # background supercategory, never an instance
                continue
            # annToMask handles all COCO seg formats (polygon / uncompressed
            # RLE / compressed RLE) uniformly, unlike coco_mask.decode which is
            # RLE-only. This dataset mixes formats across splits.
            m = self.coco.annToMask(ann)  # (h, w) uint8
            inst_id = i + 1
            inst_map[m > 0] = inst_id
            inst2sem[inst_id] = cid - 1  # -> contiguous model label

        return {
            "image": image,
            "inst_map": inst_map,
            "inst2sem": inst2sem,
            "image_id": int(img_id),
            "orig_size": (h, w),
        }


def make_train_collate(processor):
    def collate(batch: list[dict]) -> dict:
        images = [b["image"] for b in batch]
        maps = [b["inst_map"] for b in batch]
        mappings = [b["inst2sem"] for b in batch]
        enc = processor(
            images=images,
            segmentation_maps=maps,
            instance_id_to_semantic_id=mappings,
            return_tensors="pt",
        )
        return {
            "pixel_values": enc["pixel_values"],
            "pixel_mask": enc["pixel_mask"],
            "mask_labels": enc["mask_labels"],
            "class_labels": enc["class_labels"],
        }
    return collate


def make_eval_collate(processor):
    def collate(batch: list[dict]) -> dict:
        images = [b["image"] for b in batch]
        enc = processor(images=images, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"],
            "pixel_mask": enc["pixel_mask"],
            "image_ids": [b["image_id"] for b in batch],
            "orig_sizes": [b["orig_size"] for b in batch],
        }
    return collate


# --------------------------------------------------------------------------- #
# Evaluation (COCO mask AP)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_split(model, processor, ds: CocoInstanceSeg, device, args) -> dict:
    model.eval()
    loader = DataLoader(
        ds, batch_size=args.eval_batch, shuffle=False, num_workers=args.workers,
        collate_fn=make_eval_collate(processor), pin_memory=True,
    )
    results: list[dict] = []

    # Gist-transfer probe (gfn only): does the aux head predict each image's
    # object inventory under the domain shift? Micro-F1 of the image-level
    # multilabel prediction vs the GT class set. Trains/converges far faster
    # than mask AP, so it's the reliable early signal for the research loop.
    encoder = getattr(getattr(getattr(model, "model", None),
                              "pixel_level_module", None), "encoder", None)
    aux_capable = encoder is not None and hasattr(encoder, "aux_head")
    aux_tp = aux_fp = aux_fn = 0
    gt_label_sets: dict[int, set[int]] = {}
    if aux_capable:
        for img_id in ds.img_ids:
            gt_label_sets[img_id] = {
                a["category_id"] - 1 for a in ds.coco.imgToAnns.get(img_id, [])
                if a["category_id"] != 0
            }

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        if aux_capable and encoder.last_aux_logits is not None:
            preds = (encoder.last_aux_logits.sigmoid() > 0.5)
            for row, img_id in zip(preds, batch["image_ids"]):
                pred_set = set(torch.nonzero(row).flatten().tolist())
                gt = gt_label_sets.get(img_id, set())
                aux_tp += len(pred_set & gt)
                aux_fp += len(pred_set - gt)
                aux_fn += len(gt - pred_set)
        processed = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=batch["orig_sizes"],
            threshold=args.score_thresh,
            return_binary_maps=True,
        )
        for res, img_id in zip(processed, batch["image_ids"]):
            seg = res["segmentation"]  # (num_instances, H, W) binary, or empty
            segs_info = res["segments_info"]
            if seg is None or len(segs_info) == 0:
                continue
            seg = seg.cpu().numpy().astype(np.uint8)
            for k, sinfo in enumerate(segs_info):
                rle = coco_mask.encode(np.asfortranarray(seg[k]))
                rle["counts"] = rle["counts"].decode("ascii")
                results.append({
                    "image_id": img_id,
                    "category_id": int(sinfo["label_id"]) + 1,  # back to coco id
                    "segmentation": rle,
                    "score": float(sinfo["score"]),
                })

    aux_f1 = None
    if aux_capable and (aux_tp + aux_fp + aux_fn) > 0:
        prec = aux_tp / (aux_tp + aux_fp) if (aux_tp + aux_fp) else 0.0
        rec = aux_tp / (aux_tp + aux_fn) if (aux_tp + aux_fn) else 0.0
        aux_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    if not results:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "n_preds": 0, "aux_f1": aux_f1}

    coco_gt = ds.coco
    coco_dt = coco_gt.loadRes(results)
    sink = StringIO()
    with redirect_stdout(sink):
        ev = COCOeval(coco_gt, coco_dt, iouType="segm")
        if args.max_eval_samples > 0:
            ev.params.imgIds = ds.img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {
        "AP": float(ev.stats[0]),
        "AP50": float(ev.stats[1]),
        "AP75": float(ev.stats[2]),
        "n_preds": len(results),
        "aux_f1": aux_f1,
    }


# --------------------------------------------------------------------------- #
# Qualitative inference dump (predicted instance masks over the RGB)
# --------------------------------------------------------------------------- #
def _palette(n: int) -> list[tuple]:
    import colorsys
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.61803) % 1.0,
                                                            0.65, 0.95))
            for i in range(max(n, 1))]


@torch.no_grad()
def dump_overlays(model, processor, ds: "CocoInstanceSeg", device, args,
                  epoch: int, out_dir: Path) -> None:
    """Save the first N images of `ds` with predicted instance masks + class
    labels composited over them, so each research round has qualitative real-
    domain failure modes to inspect (not just the scalar gap)."""
    from PIL import ImageDraw
    model.eval()
    pal = _palette(NUM_LABELS)
    n = min(args.dump_overlays, len(ds))
    odir = out_dir / "overlays" / f"epoch{epoch}"
    odir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        item = ds[i]
        image = item["image"].convert("RGB")
        enc = processor(images=[image], return_tensors="pt")
        outputs = model(pixel_values=enc["pixel_values"].to(device),
                        pixel_mask=enc["pixel_mask"].to(device))
        res = processor.post_process_instance_segmentation(
            outputs, target_sizes=[item["orig_size"]],
            threshold=args.score_thresh, return_binary_maps=True)[0]
        arr = np.array(image)
        seg, info = res["segmentation"], res["segments_info"]
        boxes = []
        if seg is not None and len(info):
            seg = seg.cpu().numpy().astype(bool)
            for k, si in enumerate(info):
                color = np.array(pal[si["label_id"] % NUM_LABELS])
                m = seg[k]
                arr[m] = (0.5 * arr[m] + 0.5 * color).astype(np.uint8)
                ys, xs = np.where(m)
                if len(xs):
                    boxes.append((int(xs.min()), int(ys.min()),
                                  ID2LABEL.get(si["label_id"], str(si["label_id"])),
                                  float(si["score"]),
                                  tuple(int(c) for c in color)))
        canvas = Image.fromarray(arr)
        draw = ImageDraw.Draw(canvas)
        for x, y, label, score, color in boxes:
            draw.text((x, max(0, y - 10)), f"{label} {score:.2f}", fill=color)
        stem = Path(ds.coco.imgs[ds.img_ids[i]]["file_name"]).stem
        canvas.save(odir / f"{stem}.png")
    print(f"[seg-detr] wrote {n} prediction overlays -> {odir}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    data_root = args.data.expanduser().resolve()
    if not data_root.is_dir():
        sys.exit(f"--data {data_root} does not exist")

    device = (torch.device(f"cuda:{args.device}")
              if torch.cuda.is_available() and args.device != "cpu"
              else torch.device("cpu"))
    out_dir = (Path(args.project) / args.name if Path(args.project).is_absolute()
               else PROJECT_ROOT / args.project / args.name)
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)
    print(f"[seg-detr] output dir: {out_dir}")
    print(f"[seg-detr] {NUM_LABELS} classes, device={device}, model={args.model}")

    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(args.model)
    # Native frames are 1920x1080; cap the resize for speed / VRAM.
    processor.size = {"shortest_edge": args.short_edge, "longest_edge": args.long_edge}
    # Our instance maps use 0 for background; tell the processor to treat 0 as
    # ignore so it isn't looked up in instance_id_to_semantic_id.
    processor.ignore_index = 0
    processor.do_reduce_labels = False

    if args.backbone in ("dinov2", "gfn"):
        # DINOv2 has a patch size of 14; the default size_divisor of 32 would
        # pad to grids the ViT can't tile. Pad to a multiple of 14 instead.
        processor.size_divisor = 14
        model = (build_gfn_mask2former(args) if args.backbone == "gfn"
                 else build_dinov2_mask2former(args))
    elif args.backbone == "lejepa":
        if args.lejepa_ckpt is None:
            cands = sorted(PROJECT_ROOT.glob("lejepa_runs/*/ckpt_*.pt"),
                           key=lambda p: p.stat().st_mtime)
            if not cands:
                raise SystemExit("--backbone lejepa: no lejepa_runs/*/ckpt_*.pt "
                                 "found; pass --lejepa-ckpt")
            args.lejepa_ckpt = cands[-1]
        model = build_lejepa_mask2former(args)
    else:
        from transformers import AutoModelForUniversalSegmentation
        model = AutoModelForUniversalSegmentation.from_pretrained(
            args.model,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,  # re-init class head 80 -> 43
        )
        if args.freeze_backbone:
            # freeze the (COCO-pretrained) Swin encoder; train only the pixel
            # decoder + transformer decoder + heads -- same regime as the
            # frozen dinov2/gfn paths so backbone comparisons are like-for-like
            enc = model.model.pixel_level_module.encoder
            for p_ in enc.parameters():
                p_.requires_grad_(False)
            enc.eval()
            print("[seg-detr] swin backbone FROZEN "
                  "(--no-freeze-backbone to fine-tune it)", flush=True)
    model.to(device)

    # ----- data ----------------------------------------------------------- #
    # Training split(s) can come from one or more roots (sim, styled, or both);
    # eval always comes from --data so the real-domain metric stays canonical.
    if args.train_data:
        train_roots = [Path(r.strip()).expanduser() for r in
                       args.train_data.split(",") if r.strip()]
        train_roots = [r if r.is_absolute() else (PROJECT_ROOT / r) for r in train_roots]
    else:
        train_roots = [data_root]
    sub_train = [CocoInstanceSeg(r / TRAIN_SPLIT) for r in train_roots]
    for r, ds in zip(train_roots, sub_train):
        print(f"[seg-detr] train source {r.name}/{TRAIN_SPLIT}: {len(ds)} images")
    train_ds = ConcatDataset(sub_train) if len(sub_train) > 1 else sub_train[0]
    if args.max_train_samples > 0:
        train_ds = Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))
    print(f"[seg-detr] train total: {len(train_ds)} images "
          f"from {len(train_roots)} root(s)")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        collate_fn=make_train_collate(processor), pin_memory=True, drop_last=True,
    )

    eval_split_names = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
    eval_dss: dict[str, CocoInstanceSeg] = {}
    for s in eval_split_names:
        sdir = data_root / s
        if not (sdir / "images").is_dir() or not (sdir / "_annotations.coco.json").is_file():
            print(f"[seg-detr] skip eval split '{s}' (missing images/annotations)")
            continue
        eval_dss[s] = CocoInstanceSeg(sdir, args.max_eval_samples)
        print(f"[seg-detr] eval '{s}': {len(eval_dss[s])} images")

    # ----- optimizer (lower LR on the pretrained backbone) ---------------- #
    # For dinov2 only the ViT itself is the (low-LR) "backbone"; the newly
    # initialised Simple Feature Pyramid must train at the full head LR. For
    # swin the whole encoder is the pretrained backbone.
    def _is_backbone(name: str) -> bool:
        if args.backbone in ("dinov2", "gfn"):
            return "pixel_level_module.encoder.dino" in name
        if args.backbone == "lejepa":
            return "pixel_level_module.encoder.res" in name
        return "pixel_level_module.encoder" in name

    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if _is_backbone(n) else head_params).append(p)
    param_groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:  # empty when the dinov2 ViT is frozen
        param_groups.append(
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda"
    gfn_aux_weight = args.gfn_aux_weight if args.backbone == "gfn" else 0.0

    # ----- overlay + gap config ------------------------------------------- #
    overlay_split = args.overlay_split or (next(iter(eval_dss)) if eval_dss else None)
    overlay_ds = eval_dss.get(overlay_split) if overlay_split else None
    gap_sim, gap_real = [s.strip() for s in (args.gap_pair.split(",") + ["", ""])[:2]]
    log_gap = gap_sim in eval_dss and gap_real in eval_dss

    # ----- metrics csv ---------------------------------------------------- #
    csv_path = out_dir / "metrics.csv"
    fieldnames = ["epoch", "train_loss"]
    for s in eval_dss:
        fieldnames += [f"{s}_AP", f"{s}_AP50", f"{s}_AP75", f"{s}_auxF1"]
    if log_gap:
        fieldnames.append("sim_real_gap_AP")
    rows: list[dict] = []

    def run_eval(epoch: int, train_loss: float) -> None:
        row = {"epoch": epoch, "train_loss": round(train_loss, 5)}
        for s, ds in eval_dss.items():
            m = evaluate_split(model, processor, ds, device, args)
            row[f"{s}_AP"] = round(m["AP"], 4)
            row[f"{s}_AP50"] = round(m["AP50"], 4)
            row[f"{s}_AP75"] = round(m["AP75"], 4)
            row[f"{s}_auxF1"] = (round(m["aux_f1"], 4)
                                 if m.get("aux_f1") is not None else "")
            aux_str = (f" auxF1={m['aux_f1']:.4f}"
                       if m.get("aux_f1") is not None else "")
            print(f"[eval@epoch{epoch}] {s:10s} "
                  f"AP={m['AP']:.4f} AP50={m['AP50']:.4f} AP75={m['AP75']:.4f}"
                  f"{aux_str} (n_preds={m['n_preds']})")
        if log_gap:
            row["sim_real_gap_AP"] = round(row[f"{gap_sim}_AP"]
                                           - row[f"{gap_real}_AP"], 4)
            print(f"[eval@epoch{epoch}] sim->real gap "
                  f"(AP {gap_sim}-{gap_real}) = {row['sim_real_gap_AP']:.4f}")
        rows.append(row)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        if args.dump_overlays > 0 and overlay_ds is not None:
            dump_overlays(model, processor, overlay_ds, device, args,
                          epoch, out_dir)

    if args.eval_first:
        print("[seg-detr] baseline eval (epoch 0, untrained class head)")
        run_eval(0, float("nan"))

    # ----- real-distribution alignment (unsupervised DA) ------------------ #
    align_iter = None
    if args.align_real is not None:
        align_paths = sorted(p for p in Path(args.align_real).iterdir()
                             if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
        if not align_paths:
            raise SystemExit(f"--align-real: no images in {args.align_real}")
        enc_has_grad = any(p.requires_grad for p in
                           model.model.pixel_level_module.encoder.parameters())
        if not enc_has_grad:
            print("[seg-detr] WARNING: --align-real set but the encoder has no "
                  "trainable params (fully frozen swin?) -> alignment is a "
                  "no-op. Use dinov2/gfn/lejepa or --no-freeze-backbone.")

        class _RealImgs(Dataset):
            def __len__(self):
                return len(align_paths)

            def __getitem__(self, i):
                return Image.open(align_paths[i]).convert("RGB")

        def _real_collate(ims):
            enc = processor(images=ims, return_tensors="pt")
            return enc["pixel_values"]

        _real_loader = DataLoader(_RealImgs(), batch_size=args.align_batch,
                                  shuffle=True, drop_last=True,
                                  num_workers=2, collate_fn=_real_collate)

        def _cycle(dl):
            while True:
                for b in dl:
                    yield b
        align_iter = _cycle(_real_loader)
        print(f"[seg-detr] real-alignment ON: {len(align_paths)} unlabeled "
              f"images from {args.align_real}, weight={args.align_weight}, "
              f"batch={args.align_batch}")

    def _pool_pyramid(fmaps):
        return torch.cat([f.float().mean(dim=(2, 3)) for f in fmaps], dim=1)

    def _mmd(a, b):
        """Multi-bandwidth RBF MMD^2 (biased) on L2-normalized features."""
        a = torch.nn.functional.normalize(a, dim=1)
        b = torch.nn.functional.normalize(b, dim=1)
        x = torch.cat([a, b], 0)
        d2 = torch.cdist(x, x).pow(2)
        med = d2.detach().flatten().median().clamp_min(1e-6)
        k = sum(torch.exp(-d2 / (g * med)) for g in (0.5, 1.0, 2.0))
        n = a.shape[0]
        kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
        return kxx.mean() + kyy.mean() - 2 * kxy.mean()

    def _local_feats(fmaps, level, n):
        f = fmaps[min(level, len(fmaps) - 1)].float()   # (B, C, H, W)
        x = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])
        idx = torch.randperm(x.shape[0], device=x.device)[:n]
        return x[idx]

    _ema = {"s": None, "t": None}

    def _align_loss(f_sim, f_real):
        """global pooled MMD + local per-location MMD + EMA-mean L2."""
        a = _pool_pyramid(f_sim)
        b = _pool_pyramid(f_real)
        total = _mmd(a, b)
        if args.align_local > 0:
            total = total + _mmd(
                _local_feats(f_sim, args.align_local_level, args.align_local),
                _local_feats(f_real, args.align_local_level, args.align_local))
        if args.align_ema > 0:
            mu_s = torch.nn.functional.normalize(a, dim=1).mean(0)
            mu_t = torch.nn.functional.normalize(b, dim=1).mean(0)
            r = args.align_ema
            if _ema["s"] is None:
                _ema["s"], _ema["t"] = mu_s.detach(), mu_t.detach()
            es = (1 - r) * _ema["s"] + r * mu_s        # grad via current batch
            et = (1 - r) * _ema["t"] + r * mu_t
            total = total + (es - et).pow(2).sum()
            _ema["s"] = ((1 - r) * _ema["s"] + r * mu_s.detach())
            _ema["t"] = ((1 - r) * _ema["t"] + r * mu_t.detach())
        return total

    # ----- train loop ----------------------------------------------------- #
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.backbone == "swin" and args.freeze_backbone:
            # keep the frozen Swin encoder in eval mode (model.train() would
            # re-enable its stochastic depth); dinov2/gfn handle this in their
            # own .train() overrides
            model.model.pixel_level_module.encoder.eval()
        running = 0.0
        running_aux = 0.0
        running_align = 0.0
        n_steps = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            mask_labels = [m.to(device) for m in batch["mask_labels"]]
            class_labels = [c.to(device) for c in batch["class_labels"]]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_amp):
                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
                loss = outputs.loss
                # Global-first supervision: image-level multi-label loss on the
                # gist, derived from the per-instance class labels in the batch.
                if gfn_aux_weight > 0:
                    aux_logits = model.model.pixel_level_module.encoder.last_aux_logits
                    if aux_logits is not None:
                        target = torch.zeros_like(aux_logits, dtype=torch.float32)
                        for i, cl in enumerate(class_labels):
                            if cl.numel():
                                target[i, cl.long()] = 1.0
                        aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            aux_logits.float(), target)
                        loss = loss + gfn_aux_weight * aux_loss
                        running_aux += aux_loss.item()
                if align_iter is not None:
                    real_pv = next(align_iter).to(device)
                    enc = model.model.pixel_level_module.encoder
                    f_sim = enc(pixel_values).feature_maps
                    f_real = enc(real_pv).feature_maps
                    align = _align_loss(f_sim, f_real)
                    loss = loss + args.align_weight * align
                    running_align += align.item()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += loss.item()
            n_steps += 1
            if hasattr(pbar, "set_postfix"):
                post = {"loss": f"{running / n_steps:.4f}"}
                if gfn_aux_weight > 0:
                    post["aux"] = f"{running_aux / n_steps:.4f}"
                pbar.set_postfix(**post)

        train_loss = running / max(n_steps, 1)
        aux_note = (f" aux={running_aux / max(n_steps, 1):.4f}"
                    if gfn_aux_weight > 0 else "")
        if align_iter is not None:
            aux_note += f" align_mmd={running_align / max(n_steps, 1):.4f}"
        print(f"[seg-detr] epoch {epoch} done: train_loss={train_loss:.4f}{aux_note} "
              f"({time.time() - t0:.0f}s)")

        # save checkpoint (HF format) each epoch
        ckpt = out_dir / "weights" / f"epoch{epoch}"
        model.save_pretrained(ckpt)
        processor.save_pretrained(ckpt)

        run_eval(epoch, train_loss)

    print(f"[seg-detr] done. metrics -> {csv_path}")


if __name__ == "__main__":
    main()
