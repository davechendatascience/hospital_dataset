"""
Build a complete train / val / test COCO instance-segmentation dataset.

Layout produced:

    <out>/
      train/
        _annotations.coco.json     (file_name relative to this dir)
        img_<idx>.png ...
      valid/
        _annotations.coco.json
        img_<idx>.png ...
      test/
        _annotations.coco.json
        <orig real test files>

train + valid are freshly rendered by `replicator_dataset.py` (two separate
seeds so the splits don't overlap). Each rendered frame's instance segmentation
mask is RLE-encoded into the COCO annotation.

test is COPIED from the real test recordings in Ward_dataset0518; category
IDs are remapped to match the train/valid taxonomy (`fixed_categories.py`).

Run with the project venv python (NOT isaac-sim python — we need pycocotools):

    /home/edge-host/Documents/.venv/bin/python build_dataset.py \
        --train-frames 4000 --val-frames 600 \
        --out my_ward_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ROS2_bridge" / "src"))
from fixed_categories import FIXED_CATEGORIES  # type: ignore  # noqa: E402

DEFAULT_STAGE          = PROJECT_ROOT / "Collected_Ward0505" / "Ward0505.usd"
DEFAULT_REAL_TEST_ROOT = Path("/home/edge-host/Documents/Ward_dataset0518")
DEFAULT_ISAAC_PYTHON   = Path.home() / "isaac-sim" / "python.sh"
DEFAULT_REPLICATOR     = PROJECT_ROOT / "replicator_dataset.py"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--total-images", type=int, required=True,
                   help="Total image count for train + valid (test comes from "
                        "the real Ward_dataset0518 and is separate). Train/val "
                        "split is controlled by --val-ratio. The script will "
                        "OVER-render by --oversample to absorb CLIP/dark-frac "
                        "drops, then trim the survivors to exactly this many.")
    p.add_argument("--val-ratio", type=float, default=0.15,
                   help="Fraction of --total-images that goes to the val split")
    p.add_argument("--oversample", type=float, default=3.0,
                   help="Render this many times more than the target count so "
                        "we still have enough frames after filtering. Default "
                        "3.0 matches the observed ~33%% filter survival rate "
                        "on the current Ward0505 stage; bump up if you keep "
                        "falling short of --total-images.")
    p.add_argument("--keep-intermediates", action="store_true",
                   help="Don't delete _train_render/ and _val_render/ after "
                        "conversion. Useful for debugging or re-running the "
                        "post-process via --skip-render.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output dataset root (will be created if missing)")
    p.add_argument("--stage", type=Path, default=DEFAULT_STAGE,
                   help="USD scene to render (default: Ward0505.usd)")
    p.add_argument("--real-test-root", type=Path, default=DEFAULT_REAL_TEST_ROOT,
                   help="Path to Ward_dataset0518 (root containing test/ subdir)")
    p.add_argument("--isaac-sim-python", type=Path, default=DEFAULT_ISAAC_PYTHON,
                   help="Isaac Sim python launcher (default: ~/isaac-sim/python.sh)")
    p.add_argument("--replicator-script", type=Path, default=DEFAULT_REPLICATOR,
                   help="Path to replicator_dataset.py")
    p.add_argument("--resolution", nargs=2, type=int, default=[1920, 1080],
                   metavar=("W", "H"))
    p.add_argument("--hfov", type=float, default=70.0,
                   help="Camera horizontal FOV (deg)")
    p.add_argument("--rt-subframes", type=int, default=8,
                   help="Path-trace sub-frames per render (higher = cleaner)")
    p.add_argument("--randomize-materials", action="store_true",
                   help="Forward to replicator_dataset.py: per-frame perturb "
                        "each object's ORIGINAL material in place (diffuse-"
                        "texture tint, colour jitter, roughness re-rolls). "
                        "No texture bank needed.")
    p.add_argument("--randomize-placement", action="store_true",
                   help="Forward to replicator_dataset.py: per-frame MEANINGFUL "
                        "placement DR -- free-standing furniture moves as rigid "
                        "clusters to collision-free floor spots (same room), wall "
                        "objects slide on their wall, fixtures stay. Labels follow "
                        "automatically (rendered from the moved scene).")
    p.add_argument("--placement-shift", type=float, default=0.8,
                   help="Max XY cluster translation (m) for --randomize-placement.")
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--val-seed",   type=int, default=43)
    p.add_argument("--skip-render", action="store_true",
                   help="Reuse existing _train_render/ and _val_render/ "
                        "output without rendering again")
    p.add_argument("--skip-test", action="store_true",
                   help="Don't build the test split (only train/valid)")
    p.add_argument("--min-mask-area", type=int, default=16,
                   help="Drop instance annotations whose mask area is below "
                        "this many pixels (filters degenerate detections)")
    # ---- Frame-level pruning (sim splits only) ----
    # Default: a single CLIP-based "looks like a hospital room?" check that
    # drops frames where the camera ended up looking into a dark gap between
    # walls or another semantically-void scene.
    p.add_argument("--prune-disable", action="store_true",
                   help="Skip pruning; keep every rendered frame")
    p.add_argument("--prune-min-objects", type=int, default=1,
                   help="Sanity floor: drop frames with fewer than this many "
                        "labeled objects (default 1)")
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32",
                   help="HuggingFace CLIP model id used for the void filter")
    p.add_argument("--clip-margin", type=float, default=0.0,
                   help="Drop a frame if max(bad_prompt_sim) > "
                        "max(good_prompt_sim) - margin. Higher = drop more.")
    p.add_argument("--clip-batch", type=int, default=16,
                   help="CLIP image-embedding batch size")
    p.add_argument("--unexplained-dark-max", type=float, default=0.20,
                   help="Drop frames where this fraction of the image is "
                        "near-black AND NOT touching any labeled object's "
                        "segmentation mask (blob-level). Set 1.0 to disable.")
    p.add_argument("--max-object-dark-ratio", type=float, default=0.50,
                   help="Drop frames where ANY labeled object has more than "
                        "this fraction of its segmentation-mask pixels in "
                        "the dark range. Catches frames where a hospital "
                        "bed / toilet / etc. is in deep shadow and thus "
                        "unrecognizable. Set 1.0 to disable.")
    p.add_argument("--min-object-coverage", type=float, default=0.03,
                   help="Drop frames where the SUM of labeled-object "
                        "segmentation areas covers less than this fraction "
                        "of the image. Catches wall-staring views where "
                        "the camera ended up too close to (or inside) a "
                        "wall and almost nothing useful is in frame. "
                        "Set 0.0 to disable.")
    p.add_argument("--dark-thresh", type=int, default=25,
                   help="Per-pixel mean-RGB threshold below which a pixel "
                        "counts as near-black.")
    return p.parse_args()


# Prompts the CLIP void-filter compares against. Easy to extend later.
GOOD_PROMPTS = [
    "a photo of the inside of a hospital ward room with furniture",
    "a photo of a hospital room with a bed and medical equipment",
    "a photo of a hospital bathroom with toilet, sink, and shower",
    "a photo of a hospital front room with a door and sink",
    "a photo of hospital medical equipment in a room",
]
BAD_PROMPTS = [
    # Generic void / darkness
    "a mostly black image",
    "a dark hollow void between walls",
    "a dark abstract image with no clear subject",
    "looking into darkness through a gap",
    "an empty dark corner",
    # Targeted at "half the frame is dark" failure mode that the generic
    # prompts miss when one corner clearly shows a hospital object
    "a mostly black image with a small bright region",
    "a hospital photo where most of the frame is dark",
    "an image cut in half by darkness",
    "a photo with one small bright object surrounded by black",
    "an underexposed photo of a partial scene",
]


# ---------------------------------------------------------------------------
# Step 1 — invoke the renderer
# ---------------------------------------------------------------------------

def run_replicator(args: argparse.Namespace, split: str, n_frames: int,
                   seed: int) -> Path:
    """Invoke replicator_dataset.py for one split. Returns the _raw/ subdir."""
    render_root = args.out / f"_{split}_render"
    raw_dir = render_root / "_raw"
    if args.skip_render and raw_dir.exists() and any(raw_dir.iterdir()):
        print(f"[render] {split}: --skip-render and {raw_dir} populated; skipping")
        return raw_dir
    render_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.isaac_sim_python),
        str(args.replicator_script),
        "--stage",    str(args.stage),
        "--out",      str(render_root),
        "--tag",      split,
        "--frames",   str(n_frames),
        "--seed",     str(seed),
        "--resolution", str(args.resolution[0]), str(args.resolution[1]),
        "--hfov",     str(args.hfov),
        "--rt-subframes", str(args.rt_subframes),
    ]
    if getattr(args, "randomize_materials", False):
        cmd += ["--randomize-materials"]
    if getattr(args, "randomize_placement", False):
        cmd += ["--randomize-placement", "--placement-shift", str(args.placement_shift)]
    print(f"[render] {split}: invoking replicator with {n_frames} frames "
          f"(seed={seed}) -> {render_root}")
    print("        $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return raw_dir


# ---------------------------------------------------------------------------
# Step 2 — convert one _raw/ dir into a COCO instance-seg JSON
# ---------------------------------------------------------------------------

# -------- CLIP "is this a hospital interior?" void filter -----------------
_CLIP_CACHE: dict = {}


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_clip(model_id: str, device: str):
    key = (model_id, device)
    if key not in _CLIP_CACHE:
        from transformers import CLIPModel, CLIPProcessor
        print(f"[clip] loading {model_id} on {device}")
        clip = CLIPModel.from_pretrained(model_id).to(device).eval()
        proc = CLIPProcessor.from_pretrained(model_id)
        _CLIP_CACHE[key] = (clip, proc)
    return _CLIP_CACHE[key]


def _as_tensor(x):
    import torch
    if torch.is_tensor(x):
        return x
    for attr in ("text_embeds", "image_embeds",
                 "pooler_output", "last_hidden_state"):
        v = getattr(x, attr, None)
        if torch.is_tensor(v):
            return v
    raise RuntimeError(f"can't extract tensor from {type(x).__name__}")


def clip_void_filter(rgb_files, prune_cfg, device) -> list[bool]:
    """CLIP-only pre-pass: drop frames where the CLIP embedding looks more
    like a 'dark void' prompt than any 'hospital interior' prompt (after
    margin). The unexplained-dark spatial check happens later inside
    convert_raw_to_coco, where we have access to the instance segmentation."""
    if prune_cfg["disable"] or not rgb_files:
        return [True] * len(rgb_files)

    import torch
    clip, proc = _get_clip(prune_cfg["clip_model"], device)
    margin = prune_cfg["clip_margin"]
    batch_size = prune_cfg["clip_batch"]

    with torch.no_grad():
        gt = proc(text=GOOD_PROMPTS, return_tensors="pt", padding=True).to(device)
        bt = proc(text=BAD_PROMPTS,  return_tensors="pt", padding=True).to(device)
        ge = _as_tensor(clip.get_text_features(**gt))
        be = _as_tensor(clip.get_text_features(**bt))
        ge = ge / ge.norm(dim=-1, keepdim=True)
        be = be / be.norm(dim=-1, keepdim=True)

    keep, n_drop = [], 0
    for i in range(0, len(rgb_files), batch_size):
        batch = rgb_files[i:i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        with torch.no_grad():
            inp = proc(images=imgs, return_tensors="pt").to(device)
            feat = _as_tensor(clip.get_image_features(**inp))
            feat = feat / feat.norm(dim=-1, keepdim=True)
            gs = (feat @ ge.T).max(dim=-1).values
            bs = (feat @ be.T).max(dim=-1).values
        for j in range(len(batch)):
            diff = float(gs[j]) - float(bs[j])
            k = diff > margin
            keep.append(k)
            if not k:
                n_drop += 1
    print(f"[clip] kept {sum(keep)}/{len(rgb_files)} frames "
          f"({n_drop} dropped as 'void/darkness', margin={margin})")
    return keep


def _try_link(src: Path, dst: Path) -> None:
    """Hard-link if possible (instant, no copy); else fall back to copy."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _rle_encode_mask(binary_mask: np.ndarray) -> dict:
    """Encode HxW uint8 binary mask as COCO RLE (JSON-serializable)."""
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    # encode() returns bytes for "counts"; convert to ascii for JSON
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


# ---------------------------------------------------------------------------
# GPU spatial filter — replaces the CPU/numpy/scipy per-frame work in pass 1.
# Decode is still CPU (PIL), but every boolean/sum/reduction runs on cuda.
# scipy.label round-trips one small mask back to CPU for connected components
# (PyTorch has no native CC and rolling our own with iterative dilation is
# slower than the round trip).
# ---------------------------------------------------------------------------

def _gpu_check_frame(rgb_path, inst_png, sem_map_path, prune_cfg,
                     device, category_map, min_mask_area):
    """Per-frame spatial filter check on GPU.
    Returns: (passed, anns_meta, H, W, drop_reason).
        passed     : bool
        anns_meta  : list[dict] with class_name, inst_id, area, bbox  (only if passed)
        H, W       : image dims (0,0 if io failure)
        drop_reason: string in {"no_inst_files","min_objects","unexplained_dark",
                                "object_too_dark","low_coverage",""}
    """
    import torch
    if not (inst_png.exists() and sem_map_path.exists()):
        return False, None, 0, 0, "no_inst_files"

    inst_np = np.asarray(Image.open(inst_png))
    if inst_np.ndim == 3:
        inst_np = inst_np[..., 0]
    inst = torch.from_numpy(inst_np.astype(np.int32)).to(device, non_blocking=True)
    H, W = int(inst.shape[0]), int(inst.shape[1])

    rgb_np = np.asarray(Image.open(rgb_path).convert("RGB"))
    rgb = torch.from_numpy(rgb_np).to(device, non_blocking=True)

    with open(sem_map_path) as f:
        id_to_class = json.load(f)

    pairs = []  # [(inst_id, class_name)] for valid classes
    for k, v in id_to_class.items():
        try:
            iid = int(k)
        except ValueError:
            continue
        cls = (v or {}).get("class", "")
        if cls in ("BACKGROUND", "UNLABELLED", "") or cls not in category_map:
            continue
        pairs.append((iid, cls))

    if not pairs:
        return False, None, H, W, "min_objects"

    iids = torch.tensor([p[0] for p in pairs], device=device, dtype=inst.dtype)
    # (N, H, W) bool mask stack — at 1080p w/ N<=15 this is < 30 MB on GPU.
    masks = inst.unsqueeze(0) == iids.view(-1, 1, 1)
    areas = masks.sum(dim=(1, 2))  # (N,)

    big = areas >= min_mask_area
    if not big.any():
        return False, None, H, W, "min_objects"
    masks = masks[big]
    areas = areas[big]
    pairs = [p for p, b in zip(pairs, big.cpu().tolist()) if b]

    if len(pairs) < prune_cfg["min_objects"]:
        return False, None, H, W, "min_objects"

    gray = rgb.float().mean(dim=-1)
    dark_mask = gray < prune_cfg["dark_thresh"]
    object_mask = masks.any(dim=0)

    # Unexplained-dark (blob-level). scipy.label requires CPU; one quick round
    # trip on a single uint8 array per frame is cheaper than rolling out
    # iterative dilation in torch.
    if bool(dark_mask.any()):
        from scipy.ndimage import label as _cc_label
        dark_cpu = dark_mask.cpu().numpy()
        obj_cpu = object_mask.cpu().numpy()
        labeled, _ = _cc_label(dark_cpu)
        overlap_ids = np.unique(labeled[obj_cpu])
        overlap_ids = overlap_ids[overlap_ids != 0]
        explained = np.isin(labeled, overlap_ids)
        unexp_frac = float((dark_cpu & ~explained).sum()) / float(H * W)
    else:
        unexp_frac = 0.0
    if unexp_frac > prune_cfg["unexplained_dark_max"]:
        return False, None, H, W, "unexplained_dark"

    # Per-instance dark ratio — vectorized: (N, H, W) & dark -> (N,) -> /(N,)
    per_inst_dark = (
        (masks & dark_mask.unsqueeze(0)).sum(dim=(1, 2)).float()
        / areas.float()
    )
    if float(per_inst_dark.max()) > prune_cfg["max_object_dark_ratio"]:
        return False, None, H, W, "object_too_dark"

    coverage = float(areas.float().sum()) / float(H * W)
    if coverage < prune_cfg["min_object_coverage"]:
        return False, None, H, W, "low_coverage"

    # Bboxes per instance. ys.min()/xs.max() per row is a tiny CPU loop; we
    # have at most ~15 instances per frame so it's not worth vectorizing.
    anns_meta = []
    masks_cpu = masks.cpu().numpy()
    areas_cpu = areas.cpu().tolist()
    for i, (iid, cls) in enumerate(pairs):
        m = masks_cpu[i]
        ys, xs = np.where(m)
        if xs.size == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        anns_meta.append({
            "class_name": cls,
            "inst_id": iid,
            "area": int(areas_cpu[i]),
            "bbox": [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
        })
    return True, anns_meta, H, W, ""


def convert_raw_to_coco(raw_dir: Path, split_dir: Path,
                       category_map: dict, min_mask_area: int,
                       prune_cfg: dict, target_count=None, seed=0) -> dict:
    """Read BasicWriter outputs, produce COCO with RLE instance masks. Apply
    the CLIP void filter (drops frames whose CLIP embedding looks more like
    a 'dark void between walls' than a 'hospital interior') and a sanity
    floor on labeled object count.

    Per-frame inputs (from BasicWriter):
      rgb_<idx>.png
      instance_segmentation_<idx>.png                       (raw uint32 IDs)
      instance_segmentation_semantics_mapping_<idx>.json    (id -> class)
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    rgb_files = sorted(raw_dir.rglob("rgb_*.png"))
    if not rgb_files:
        print(f"[coco] no RGB files under {raw_dir}; skipping this split")
        return {"images": [], "annotations": [], "categories": []}

    # ---- CLIP pre-pass on the WHOLE split (no trim yet) ----
    clip_keep = clip_void_filter(rgb_files, prune_cfg, _device())

    drops = {"clip_void": 0, "no_inst_files": 0, "min_objects": 0,
             "unexplained_dark": 0, "object_too_dark": 0, "low_coverage": 0}
    skipped_ann = {"bg": 0, "oor": 0, "tiny": 0}

    # ---- Pass 1: apply ALL spatial filters on every CLIP survivor ----
    # On CUDA we use the vectorized GPU check (_gpu_check_frame); per-frame
    # is roughly 2-3x faster because per-instance mask building, isin, and
    # dark/coverage reductions all run as torch ops on cuda. We don't keep
    # binary masks in memory between passes — pass 2 re-reads the inst_map
    # for the kept frames to RLE-encode them.
    survivors = []
    filter_device = _device()
    use_gpu = (filter_device == "cuda") and (not prune_cfg["disable"])
    if use_gpu:
        print(f"[filter] pass-1 spatial check on cuda")
    for i, rgb_path in enumerate(rgb_files):
        if not clip_keep[i]:
            drops["clip_void"] += 1
            continue
        stem = rgb_path.stem            # rgb_0000
        idx = stem.split("_", 1)[1]
        inst_png     = rgb_path.parent / f"instance_segmentation_{idx}.png"
        inst_sem_map = rgb_path.parent / f"instance_segmentation_semantics_mapping_{idx}.json"

        if use_gpu:
            passed, anns_meta, H, W, reason = _gpu_check_frame(
                rgb_path, inst_png, inst_sem_map, prune_cfg,
                filter_device, category_map, min_mask_area,
            )
            if not passed:
                drops[reason] = drops.get(reason, 0) + 1
                continue
            survivors.append({
                "rgb_path": rgb_path, "idx": idx, "inst_png": inst_png,
                "H": H, "W": W, "anns_meta": anns_meta,
            })
            continue

        # ---- CPU fallback (used when CUDA unavailable or prune disabled) ----
        if not inst_png.exists() or not inst_sem_map.exists():
            drops["no_inst_files"] += 1
            continue
        inst_map = np.asarray(Image.open(inst_png))
        if inst_map.ndim != 2:
            inst_map = inst_map[..., 0]
        H, W = inst_map.shape
        with open(inst_sem_map) as f:
            id_to_class = json.load(f)
        anns_meta = []
        valid_inst_ids = []
        for inst_id_str, info in id_to_class.items():
            try:
                inst_id = int(inst_id_str)
            except ValueError:
                continue
            class_name = (info or {}).get("class", "")
            if class_name in ("BACKGROUND", "UNLABELLED", ""):
                skipped_ann["bg"] += 1
                continue
            if class_name not in category_map:
                skipped_ann["oor"] += 1
                continue
            inst_pixels = (inst_map == inst_id)
            area = int(inst_pixels.sum())
            if area < min_mask_area:
                skipped_ann["tiny"] += 1
                continue
            ys, xs = np.where(inst_pixels)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            anns_meta.append({
                "class_name": class_name, "inst_id": inst_id, "area": area,
                "bbox": [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
            })
            valid_inst_ids.append(inst_id)
        if not prune_cfg["disable"]:
            if len(anns_meta) < prune_cfg["min_objects"]:
                drops["min_objects"] += 1
                continue
            arr = np.asarray(Image.open(rgb_path).convert("RGB"))
            gray = arr.mean(axis=-1)
            dark_mask = gray < prune_cfg["dark_thresh"]
            if not dark_mask.any():
                unexplained_frac = 0.0
            elif not valid_inst_ids:
                unexplained_frac = float(dark_mask.sum()) / dark_mask.size
            else:
                from scipy.ndimage import label as _cc_label
                object_mask = np.isin(inst_map, valid_inst_ids)
                labeled_dark, _ = _cc_label(dark_mask)
                overlap_ids = np.unique(labeled_dark[object_mask])
                overlap_ids = overlap_ids[overlap_ids != 0]
                explained = np.isin(labeled_dark, overlap_ids)
                unexplained_frac = float((dark_mask & ~explained).sum()) / dark_mask.size
            if unexplained_frac > prune_cfg["unexplained_dark_max"]:
                drops["unexplained_dark"] += 1
                continue
            worst_ratio = 0.0
            if valid_inst_ids and dark_mask.any():
                for inst_id in valid_inst_ids:
                    inst_pixels = (inst_map == inst_id)
                    n = int(inst_pixels.sum())
                    if n == 0:
                        continue
                    r = float((inst_pixels & dark_mask).sum()) / n
                    if r > worst_ratio:
                        worst_ratio = r
            if worst_ratio > prune_cfg["max_object_dark_ratio"]:
                drops["object_too_dark"] += 1
                continue
            total_obj_area = sum(a["area"] for a in anns_meta)
            coverage = total_obj_area / float(H * W)
            if coverage < prune_cfg["min_object_coverage"]:
                drops["low_coverage"] += 1
                continue
        survivors.append({
            "rgb_path": rgb_path, "idx": idx, "inst_png": inst_png,
            "H": H, "W": W, "anns_meta": anns_meta,
        })

    # ---- Trim survivors to target_count (random sample, deterministic) ----
    if target_count is not None and len(survivors) > target_count:
        import random as _r
        rng_ = _r.Random(seed)
        rng_.shuffle(survivors)
        survivors = sorted(survivors[:target_count], key=lambda s: s["idx"])
        print(f"[trim] downsampled {len(survivors) + (len(survivors) - target_count if False else 0)} "
              f"-> target {target_count}")
    elif target_count is not None and len(survivors) < target_count:
        print(f"[trim] WARN: only {len(survivors)} survivors but target is "
              f"{target_count}. Increase --oversample to render more.")
    print(f"[trim] using {len(survivors)} frames for the COCO output")

    # ---- Pass 2: write images + RLE-encode masks for the chosen survivors ----
    coco = {
        "info": {"description": f"Ward replicator dataset split={split_dir.name}"},
        "images": [], "annotations": [],
        "categories": [
            {"id": cid, "name": name, "supercategory": "ward_object"}
            for name, cid in category_map.items()
        ],
    }
    ann_id = 1
    for new_image_id, s in enumerate(survivors, start=1):
        img_name = f"img_{s['idx']}.png"
        target_img = split_dir / img_name
        _try_link(s["rgb_path"], target_img)
        coco["images"].append({
            "id": new_image_id, "file_name": img_name,
            "width": s["W"], "height": s["H"],
        })
        # Reload instance map only for this kept frame to derive masks
        inst_map = np.asarray(Image.open(s["inst_png"]))
        if inst_map.ndim != 2:
            inst_map = inst_map[..., 0]
        for a in s["anns_meta"]:
            bin_mask = (inst_map == a["inst_id"])
            rle = _rle_encode_mask(bin_mask)
            coco["annotations"].append({
                "id": ann_id, "image_id": new_image_id,
                "category_id": category_map[a["class_name"]],
                "bbox": a["bbox"], "area": a["area"],
                "iscrowd": 0, "segmentation": rle,
            })
            ann_id += 1

    json_path = split_dir / "_annotations.coco.json"
    with open(json_path, "w") as f:
        json.dump(coco, f)
    total_drops = sum(drops.values())
    print(f"[coco] {json_path}: {len(coco['images'])} images kept, "
          f"{len(coco['annotations'])} anns")
    print(f"[coco]   dropped {total_drops} frames (out of {len(rgb_files)} rendered): "
          + ", ".join(f"{k}={v}" for k, v in drops.items()))
    print(f"[coco]   per-ann skips (within kept frames): "
          + ", ".join(f"{k}={v}" for k, v in skipped_ann.items()))
    return coco


# ---------------------------------------------------------------------------
# Step 3 — copy + remap the real test set
# ---------------------------------------------------------------------------

_NORMALIZE_NAME = re.compile(r"[^a-z0-9]")


def _norm(name: str) -> str:
    return _NORMALIZE_NAME.sub("", name.lower())


def _build_name_aliases(category_map: dict) -> dict:
    """Map normalized-name -> canonical category_id from `category_map`.
    Lets us absorb minor naming drift between the two datasets (e.g.
    'IV_Pole' vs 'iv_pole')."""
    out = {}
    for name, cid in category_map.items():
        out[_norm(name)] = cid
    return out


def _find_real_test_root(root: Path) -> tuple[Path, Path]:
    """Return (image_dir, coco_json_path) for the original test split."""
    # Roboflow-style: <root>/test/_annotations.coco.json + <root>/test/*.jpg
    rb = root / "test"
    rb_json = rb / "_annotations.coco.json"
    if rb_json.exists():
        return rb, rb_json
    # Alternative: <root>/ground_truth/rgbDataset/test_rgb/*.jpg + test.json
    alt_dir = root / "ground_truth" / "rgbDataset" / "test_rgb"
    alt_json = alt_dir / "test.json"
    if alt_json.exists():
        return alt_dir, alt_json
    raise FileNotFoundError(
        f"Could not locate real test COCO under {root}. Tried:\n"
        f"  {rb_json}\n  {alt_json}")


def build_test_from_real(real_root: Path, dst_dir: Path,
                         category_map: dict) -> dict:
    src_img_dir, src_json = _find_real_test_root(real_root)
    print(f"[test] real test src: imgs={src_img_dir}  coco={src_json}")
    with open(src_json) as f:
        src_coco = json.load(f)

    alias_to_id = _build_name_aliases(category_map)

    # Map old category id -> new (canonical) category id by name normalization.
    old_to_new = {}
    unmatched_cats = []
    for cat in src_coco.get("categories", []):
        old_id = int(cat["id"])
        new_id = alias_to_id.get(_norm(cat["name"]))
        if new_id is None:
            unmatched_cats.append(cat["name"])
            continue
        old_to_new[old_id] = new_id
    if unmatched_cats:
        print(f"[test] WARNING: {len(unmatched_cats)} test categories don't "
              f"match fixed_categories.py and will be dropped: "
              f"{', '.join(unmatched_cats[:10])}"
              f"{'...' if len(unmatched_cats) > 10 else ''}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    new_coco = {
        "info": {"description": "Ward real test (remapped category ids)"},
        "images": [], "annotations": [],
        "categories": [
            {"id": cid, "name": name, "supercategory": "ward_object"}
            for name, cid in category_map.items()
        ],
    }
    image_id_map = {}
    new_image_id = 1
    for image in src_coco.get("images", []):
        src_file_name = image["file_name"]
        # Resolve actual image path:
        src_path = (src_img_dir / Path(src_file_name).name)
        if not src_path.exists():
            src_path = src_img_dir / src_file_name
        if not src_path.exists():
            print(f"[test] skipping missing image: {src_file_name}")
            continue
        dst_name = src_path.name
        dst_path = dst_dir / dst_name
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)

        image_id_map[int(image["id"])] = new_image_id
        new_image = dict(image)
        new_image["id"] = new_image_id
        new_image["file_name"] = dst_name
        new_coco["images"].append(new_image)
        new_image_id += 1

    n_dropped = 0
    new_ann_id = 1
    for ann in src_coco.get("annotations", []):
        old_image_id = int(ann["image_id"])
        if old_image_id not in image_id_map:
            n_dropped += 1
            continue
        old_cat_id = int(ann["category_id"])
        new_cat_id = old_to_new.get(old_cat_id)
        if new_cat_id is None:
            n_dropped += 1
            continue
        new_ann = dict(ann)
        new_ann["id"] = new_ann_id
        new_ann["image_id"] = image_id_map[old_image_id]
        new_ann["category_id"] = new_cat_id
        new_coco["annotations"].append(new_ann)
        new_ann_id += 1

    json_path = dst_dir / "_annotations.coco.json"
    with open(json_path, "w") as f:
        json.dump(new_coco, f)
    print(f"[test] {json_path}: {len(new_coco['images'])} images, "
          f"{len(new_coco['annotations'])} anns (dropped {n_dropped})")
    return new_coco


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.out             = args.out.expanduser().resolve()
    args.stage           = args.stage.expanduser().resolve()
    args.real_test_root  = args.real_test_root.expanduser().resolve()
    args.isaac_sim_python = args.isaac_sim_python.expanduser().resolve()
    args.replicator_script = args.replicator_script.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- Split totals + oversample for rendering ----
    val_target   = max(int(round(args.total_images * args.val_ratio)), 1)
    train_target = args.total_images - val_target
    train_render_n = max(int(round(train_target * args.oversample)), train_target)
    val_render_n   = max(int(round(val_target   * args.oversample)), val_target)
    print(f"[plan] total={args.total_images} -> train={train_target}, "
          f"val={val_target} (val_ratio={args.val_ratio})")
    print(f"[plan] oversample x{args.oversample}: render {train_render_n} train, "
          f"{val_render_n} val; trim to target after filtering")

    # 1) Render train + valid via Isaac Sim
    train_raw = run_replicator(args, "train", train_render_n, args.train_seed)
    val_raw   = run_replicator(args, "val",   val_render_n,   args.val_seed)

    # 2) Convert renders to COCO with RLE instance masks (with CLIP void filter)
    prune_cfg = {
        "disable":                args.prune_disable,
        "min_objects":            args.prune_min_objects,
        "clip_model":             args.clip_model,
        "clip_margin":            args.clip_margin,
        "clip_batch":             args.clip_batch,
        "dark_thresh":            args.dark_thresh,
        "unexplained_dark_max":   args.unexplained_dark_max,
        "max_object_dark_ratio":  args.max_object_dark_ratio,
        "min_object_coverage":    args.min_object_coverage,
    }
    if not args.prune_disable:
        print(f"[prune] CLIP({args.clip_model}) margin={args.clip_margin} | "
              f"unexplained-dark<={args.unexplained_dark_max} | "
              f"per-obj dark<={args.max_object_dark_ratio} | "
              f"min-coverage>={args.min_object_coverage} | "
              f"(dark<{args.dark_thresh}) | min_objects>={args.prune_min_objects}")
    convert_raw_to_coco(train_raw, args.out / "train", FIXED_CATEGORIES,
                        args.min_mask_area, prune_cfg,
                        target_count=train_target, seed=args.train_seed)
    convert_raw_to_coco(val_raw,   args.out / "valid", FIXED_CATEGORIES,
                        args.min_mask_area, prune_cfg,
                        target_count=val_target,   seed=args.val_seed)

    # 3) Copy + remap the real test set
    if not args.skip_test:
        build_test_from_real(args.real_test_root, args.out / "test",
                             FIXED_CATEGORIES)

    # 4) Clean up preprocessing artifacts unless --keep-intermediates
    if not args.keep_intermediates:
        for sub in ("_train_render", "_val_render"):
            p = args.out / sub
            if p.exists():
                shutil.rmtree(p)
                print(f"[clean] removed {p}")

    print(f"\n[done] dataset at {args.out}")
    print(f"       train: {args.out/'train'}/_annotations.coco.json")
    print(f"       valid: {args.out/'valid'}/_annotations.coco.json")
    print(f"       test : {args.out/'test'}/_annotations.coco.json")


if __name__ == "__main__":
    main()
