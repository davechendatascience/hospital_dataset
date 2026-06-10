"""Generate per-image Cosmos-Transfer2.5 job configs for sim->real transfer.

One JSON per sim image (image-to-image, max_frames=1) with:
  * a per-image PROMPT  -> the prompt of the room it most resembles
  * image_context_path  -> that room's real reference photo (style guide)
  * edge control        -> geometry preserved (computed on the fly)

Room assignment is deterministic: each sim image is embedded with DINOv2 and
assigned to the nearest of the 3 FIXED real-room reference photos (no kmeans
instability). Emits <out>/configs/<stem>.json + manifest.csv + run_all.sh
(loops cosmos examples/inference.py over every config).

    .venv/bin/python gen_cosmos_jobs.py --sim-dir ward_v3/train/images \
        --out cosmos_jobs --control edge
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np

# realistic lighting/time modifiers appended to the prompt for style variance
STYLE_MODIFIERS = [
    "bright natural daylight from the window",
    "soft overcast daylight",
    "warm late-afternoon light",
    "cool early-morning light",
    "even fluorescent ceiling lighting",
    "dim evening lighting with the ceiling lights on",
]

# scene classification from COCO object NAMES (shared by sim frames + test refs)
WARD_OBJ = {"hospital_bed", "bed_curtain", "overbed_table", "bedside_monitor"}
BATH_OBJ = {"toilet", "toilet_handle", "shower", "sink", "mirror"}
BIN_OBJ = {"soiled_linen_bin", "waste_bin", "medical_waste_container"}


def scene_of(names: set) -> str:
    if names & WARD_OBJ:          # a bed dominates -> ward
        return "ward"
    if names & BATH_OBJ:          # no bed + bathroom fixture
        return "bathroom"
    if names & BIN_OBJ:           # no bed + bins
        return "corridor"
    return "ward"

PROJECT = Path(__file__).resolve().parent
TEST = PROJECT / "ward_v3" / "test" / "images"
COSMOS_REPO = Path("/home/edge-host/cosmos-transfer2.5")

# 3 scene types in the footage: WARD room, CORRIDOR (utility nook with two bins),
# BATHROOM. Each: representative real photo (style ref, from ward_v3/test) + a
# scene-type prompt. Sim images are assigned to the nearest ref by DINOv2.
ROOMS = [
    {
        "name": "ward",
        "ref": TEST / "WIN_20260331_11_10_31_Pro_frame_00630_png.rf.75d81cbd26be74104e3509f5df918c1f.jpg",
        "prompt": ("A realistic photograph of a Taiwanese hospital ward patient room: an "
                   "electric care bed with a deep purple-navy mattress and white plastic side "
                   "rails, a stainless-steel IV pole, a wall-mounted vital-signs monitor on an "
                   "articulated arm, a white louvered air-conditioner, a UV germicidal lamp, a "
                   "beige wall telephone, a white bedside cabinet, a mint-green privacy curtain, "
                   "cream and beige walls with wood-grain laminate wainscot, light wood laminate "
                   "floor, a window with sheer curtains."),
    },
    {
        "name": "corridor",
        "ref": TEST / "WIN_20260331_12_10_38_Pro_frame_00625_png.rf.232c56cd353d169117ca36485026ff84.jpg",
        "prompt": ("A realistic photograph of a Taiwanese hospital corridor and utility nook: a "
                   "stainless-steel rolling soiled-linen bin holding a fabric laundry bag, and a "
                   "cream pedal-operated biomedical-waste bin with a red plastic liner and a "
                   "biohazard label, beside wood-grain laminate wall panels and a sliding door, "
                   "light wood laminate floor, cream walls."),
    },
    {
        "name": "bathroom",
        "ref": TEST / "WIN_20260331_11_26_06_Pro_frame_04423_png.rf.ebcab57482d5df6214098c18da6fce1a.jpg",
        "prompt": ("A realistic photograph of a Taiwanese hospital ward ensuite bathroom: white "
                   "square ceramic wall tiles and beige floor tiles, a white toilet, "
                   "stainless-steel L-shaped and flip-up grab bars, a chrome toilet-paper "
                   "holder, a wall-mounted ceramic sink with a chrome faucet and a framed "
                   "mirror, red nurse-call buttons."),
    },
]


def list_images(d: Path):
    import glob
    out = []
    for e in ("png", "jpg", "jpeg", "bmp", "webp"):
        out += glob.glob(str(d / f"*.{e}")) + glob.glob(str(d / f"*.{e.upper()}"))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-dir", type=Path, default=PROJECT / "ward_v3/train/images")
    ap.add_argument("--out", type=Path, default=PROJECT / "cosmos_jobs")
    ap.add_argument("--seg-weight", type=float, default=1.0,
                    help="seg (label) control weight; feeds the class-id seg map rasterized "
                         "from the sim COCO -> preserves object regions on the output.")
    ap.add_argument("--depth-weight", type=float, default=0.5,
                    help="depth control weight; feeds ground-truth Isaac depth -> geometry. "
                         "Set 0 to disable depth control.")
    ap.add_argument("--depth-dir", type=Path, default=None,
                    help="dir of depth PNGs (default <sim-dir>/../depth).")
    ap.add_argument("--vary-style", action="store_true", default=False,
                    help="style variance (deterministic by stem): per-image random real "
                         "style-ref from the scene pool + random seed + a lighting modifier.")
    ap.add_argument("--test-dir", type=Path, default=PROJECT / "ward_v3/test",
                    help="real test set (with COCO) -> per-scene style-reference pools.")
    ap.add_argument("--captions", type=Path, default=None,
                    help="JSON {stem: prompt} from caption_images.py -> per-image prompt "
                         "(falls back to the room template prompt if a stem is missing).")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    sim_files = list_images(args.sim_dir)
    if args.limit:
        sim_files = sim_files[:args.limit]
    for r in ROOMS:
        if not Path(r["ref"]).is_file():
            raise SystemExit(f"missing room reference image: {r['ref']}")

    # Scene assignment is by the sim's OWN COCO labels (reliable) -- NOT DINOv2 nearest
    # real-ref (sim != real in DINOv2 space, so it misclassifies). classify() below.

    caps = {}
    if args.captions and Path(args.captions).is_file():
        caps = json.loads(Path(args.captions).read_text())
        print(f"[cosmos-jobs] loaded {len(caps)} per-image captions from {args.captions}")

    # control inputs: SEG (class-id label map rasterized from the sim COCO -> preserves
    # object regions) + DEPTH (ground-truth Isaac depth -> geometry). Both fed to Cosmos.
    import colorsys
    from pycocotools.coco import COCO
    from PIL import Image
    pal = np.zeros((64, 3), np.uint8)
    for i in range(1, 64):
        r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.65, 0.95)
        pal[i] = (int(r * 255), int(g * 255), int(b * 255))
    coco = COCO(str(args.sim_dir.parent / "_annotations.coco.json"))
    stem2id = {Path(im["file_name"]).stem: i for i, im in coco.imgs.items()}
    depth_dir = args.depth_dir or (args.sim_dir.parent / "depth")

    # label-based scene classifier (ward / corridor / bathroom) from the sim COCO
    nm = {c["id"]: c["name"] for c in coco.dataset["categories"]}
    ROOM_IDX = {r["name"]: k for k, r in enumerate(ROOMS)}

    def classify(img_id):
        names = {nm[a["category_id"]] for a in coco.imgToAnns.get(img_id, [])}
        return scene_of(names)

    # per-scene REAL style-reference pools: single rep by default; full pool if --vary-style
    ref_pools = {r["name"]: [str(Path(r["ref"]).resolve())] for r in ROOMS}
    if args.vary_style:
        tcoco = COCO(str(args.test_dir / "_annotations.coco.json"))
        tnm = {c["id"]: c["name"] for c in tcoco.dataset["categories"]}
        timg = args.test_dir / "images"
        pools = {"ward": [], "corridor": [], "bathroom": []}
        for tid, tim in tcoco.imgs.items():
            names = {tnm[a["category_id"]] for a in tcoco.imgToAnns.get(tid, [])}
            fp = timg / tim["file_name"]
            if fp.is_file():
                pools[scene_of(names)].append(str(fp.resolve()))
        for k, v in pools.items():
            if v:
                ref_pools[k] = v
        print("[cosmos-jobs] style-ref pools: " +
              ", ".join(f"{k}={len(ref_pools[k])}" for k in ("ward", "corridor", "bathroom")))

    cfg_dir = args.out / "configs"; seg_dir = args.out / "seg"
    cfg_dir.mkdir(parents=True, exist_ok=True); seg_dir.mkdir(parents=True, exist_ok=True)
    out_root = (args.out / "outputs").resolve()
    manifest, counts, n_cap, n_seg, n_depth = [], [0, 0, 0], 0, 0, 0
    for p in sim_files:
        stem = Path(p).stem
        ri = ROOM_IDX[classify(stem2id.get(stem))]
        room = ROOMS[ri]; counts[ri] += 1
        rng = random.Random(stem)                   # deterministic per-image variance
        prompt = caps.get(stem, room["prompt"])     # per-image caption, else scene-type prompt
        if stem in caps:
            n_cap += 1
        ref = str(Path(room["ref"]).resolve())
        seed = args.seed
        if args.vary_style:                         # sample real ref + seed + lighting
            ref = rng.choice(ref_pools[room["name"]])
            seed = rng.randrange(1, 1_000_000)
            if stem not in caps:
                prompt = f"{prompt} {rng.choice(STYLE_MODIFIERS)}."
        controls = {}
        img_id = stem2id.get(stem)
        if img_id is not None:                      # SEG (label) control map
            info = coco.imgs[img_id]; H, W = info["height"], info["width"]
            seg = np.zeros((H, W, 3), np.uint8)
            for a in coco.imgToAnns.get(img_id, []):
                if a["category_id"] != 0:
                    seg[coco.annToMask(a) > 0] = pal[a["category_id"] % 64]
            segp = (seg_dir / f"{stem}.png").resolve()
            Image.fromarray(seg).save(segp)
            controls["seg"] = {"control_path": str(segp), "control_weight": args.seg_weight}
            n_seg += 1
        dpath = depth_dir / f"{stem}.png"           # DEPTH control
        if args.depth_weight > 0 and dpath.is_file():
            controls["depth"] = {"control_path": str(dpath.resolve()),
                                 "control_weight": args.depth_weight}
            n_depth += 1
        cfg = {
            "name": f"{room['name']}_{stem}",
            "prompt": prompt,
            "video_path": str(Path(p).resolve()),     # single image
            "max_frames": 1, "num_video_frames_per_chunk": 1,
            "image_context_path": ref,
            "seed": seed,
            **controls,
        }
        (cfg_dir / f"{stem}.json").write_text(json.dumps(cfg, indent=2))
        manifest.append({"stem": stem, "room": room["name"], "config": str(cfg_dir / f"{stem}.json")})

    with open(args.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "room", "config"]); w.writeheader(); w.writerows(manifest)
    print(f"[cosmos-jobs] controls: seg={n_seg} (w={args.seg_weight}), "
          f"depth={n_depth} (w={args.depth_weight})")

    # BATCH runner: pass ALL configs to one inference.py call so the 7B model is
    # loaded ONCE and every image runs sequentially (each with its own prompt).
    # (-i accepts multiple param files -> batch inference, per inference.py --help.)
    run = args.out / "run_all.sh"
    run.write_text(
        "#!/usr/bin/env bash\n"
        "# Cosmos-Transfer2.5 batch: model loaded once, every image (own prompt) in turn.\n"
        f"set -e\ncd {COSMOS_REPO}\n"
        f'./.venv/bin/python examples/inference.py -i {cfg_dir.resolve()}/*.json '
        f'-o {out_root}\n')
    run.chmod(0o755)

    print(f"[cosmos-jobs] {len(sim_files)} images -> {cfg_dir}  "
          f"(prompts: {n_cap} per-image caption, {len(sim_files)-n_cap} room-template)")
    print(f"[cosmos-jobs] room assignment (image_context_path style ref): " +
          ", ".join(f"{ROOMS[i]['name']}={counts[i]}" for i in range(3)))
    print(f"[cosmos-jobs] outputs -> {out_root}")
    print(f"[cosmos-jobs] run all:  bash {run}")


if __name__ == "__main__":
    main()
