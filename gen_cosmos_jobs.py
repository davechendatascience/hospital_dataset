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

# Scene-agnostic: the seg+depth controls + the guided foreground mask define the
# actual scene (ward/corridor/bathroom), so we DON'T classify it ourselves (that was
# error-prone). The prompt lists the frame's real objects (from COCO labels); the
# style ref is content-matched from the real set by object inventory.
SHARED_MATERIALS = ("cream and beige walls with wood-grain laminate wainscot or white "
                    "ceramic wall tiles, light wood-laminate or tiled floors, stainless-steel "
                    "and white fixtures, mint-green privacy curtains, soft natural lighting")

_ACRONYM = {"iv": "IV", "tv": "TV", "uv": "UV"}


def _humanize(name: str) -> str:
    return " ".join(_ACRONYM.get(w, w) for w in name.split("_"))


def build_prompt(class_names: set) -> str:
    """Scene-agnostic framing + the ACTUAL objects in this frame (from COCO labels)."""
    objs = sorted(_humanize(n) for n in class_names if n and n != "ward_object")
    obj_str = ", ".join(objs) if objs else "hospital equipment"
    return (f"A realistic photograph of a Taiwanese hospital interior, with "
            f"{obj_str}; {SHARED_MATERIALS}.")


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
    ap.add_argument("--seg-weight", type=float, default=0.6,
                    help="seg (label) control weight; feeds the class-id seg map rasterized "
                         "from the sim COCO -> preserves object regions on the output.")
    ap.add_argument("--depth-weight", type=float, default=0.8,
                    help="depth control weight; feeds ground-truth Isaac depth -> geometry. "
                         "Set 0 to disable depth control.")
    ap.add_argument("--depth-dir", type=Path, default=None,
                    help="dir of depth PNGs (default <sim-dir>/../depth).")
    ap.add_argument("--guided", dest="guided", action="store_true", default=True,
                    help="guided generation: anchor the labeled foreground (union of object "
                         "masks) during denoising so object structure/identity is preserved.")
    ap.add_argument("--no-guided", dest="guided", action="store_false")
    ap.add_argument("--guided-steps", type=int, default=10,
                    help="guided_generation_step_threshold out of ~35 (anchor structure early, "
                         "release for restyle late). ~10 balances anchoring + realism.")
    ap.add_argument("--edge-weight", type=float, default=1.0,
                    help="edge control weight (computed on the fly; contours without color).")
    ap.add_argument("--guidance", type=float, default=3.0,
                    help="classifier-free guidance (modest so structure isn't overpowered).")
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
    # Scene is NOT classified here -- the seg+depth controls + the guided foreground
    # mask define the actual scene; the style ref is content-matched by object inventory.

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

    nm = {c["id"]: c["name"] for c in coco.dataset["categories"]}

    # real style-reference index: each real test photo -> its object-class set, so each sim
    # frame is content-matched to the real photo whose objects best match (Jaccard) -- no
    # brittle scene classification; the seg+depth controls + guided mask define the scene.
    tcoco = COCO(str(args.test_dir / "_annotations.coco.json"))
    tnm = {c["id"]: c["name"] for c in tcoco.dataset["categories"]}
    timg = args.test_dir / "images"
    real_refs = []
    for tid, tim in tcoco.imgs.items():
        names = frozenset(tnm[a["category_id"]] for a in tcoco.imgToAnns.get(tid, [])
                          if a["category_id"] != 0)
        fp = timg / tim["file_name"]
        if fp.is_file():
            real_refs.append((str(fp.resolve()), names))
    print(f"[cosmos-jobs] {len(real_refs)} real style refs (content-matched by object inventory)")

    def match_refs(present, k=8):
        def jac(s):
            u = len(present | s)
            return (len(present & s) / u) if u else 0.0
        return [r[0] for r in sorted(real_refs, key=lambda x: jac(x[1]), reverse=True)[:k]]

    cfg_dir = args.out / "configs"; seg_dir = args.out / "seg"; depth_out = args.out / "depth"
    fg_dir = args.out / "fgmask"
    for d in (cfg_dir, seg_dir, depth_out, fg_dir):
        d.mkdir(parents=True, exist_ok=True)
    out_root = (args.out / "outputs").resolve()
    manifest, n_cap, n_seg, n_depth, n_guided = [], 0, 0, 0, 0
    for p in sim_files:
        stem = Path(p).stem
        img_id = stem2id.get(stem)
        present = {nm[a["category_id"]] for a in coco.imgToAnns.get(img_id, [])
                   if a["category_id"] != 0}          # actual object classes in this frame
        rng = random.Random(stem)                   # deterministic per-image variance
        # prompt names the ACTUAL objects (from labels); scene-agnostic framing
        prompt = caps.get(stem) or build_prompt(present)
        if stem in caps:
            n_cap += 1
        cand = match_refs(present)                  # top-K content-matched real refs
        ref = cand[0]
        seed = args.seed
        if args.vary_style:                         # vary among matched refs + seed + lighting
            ref = rng.choice(cand)
            seed = rng.randrange(1, 1_000_000)
            prompt = f"{prompt} {rng.choice(STYLE_MODIFIERS)}."
        controls = {}
        guided = {}
        if img_id is not None:                      # SEG (label) control map + FG mask
            info = coco.imgs[img_id]; H, W = info["height"], info["width"]
            seg = np.zeros((H, W, 3), np.uint8)
            fg = np.zeros((H, W), np.uint8)         # foreground = union of object masks
            for a in coco.imgToAnns.get(img_id, []):
                if a["category_id"] != 0:
                    m = coco.annToMask(a)
                    seg[m > 0] = pal[a["category_id"] % 64]
                    fg[m > 0] = 255
            segp = (seg_dir / f"{stem}.png").resolve()
            Image.fromarray(seg).save(segp)
            controls["seg"] = {"control_path": str(segp), "control_weight": args.seg_weight}
            n_seg += 1
            if args.guided:                          # anchor labeled foreground during denoise
                # guided mask must be .npz with 'arr_0' shape (T,H,W); non-zero = foreground
                fgp = (fg_dir / f"{stem}.npz").resolve()
                np.savez(str(fgp), fg[None].astype(np.uint8))   # (1,H,W)
                guided = {"guided_generation_mask": str(fgp),
                          "guided_generation_step_threshold": args.guided_steps}
                n_guided += 1
        dpath = depth_dir / f"{stem}.png"           # DEPTH control
        if args.depth_weight > 0 and dpath.is_file():
            # GT depth is grayscale (mode L); Cosmos's control reader needs HWC/3-channel
            d3 = (depth_out / f"{stem}.png").resolve()
            Image.open(dpath).convert("RGB").save(d3)
            controls["depth"] = {"control_path": str(d3),
                                 "control_weight": args.depth_weight}
            n_depth += 1
        if args.edge_weight > 0:                     # EDGE control (computed on the fly)
            controls["edge"] = {"control_weight": args.edge_weight}
        cfg = {
            "name": stem,
            "prompt": prompt,
            "video_path": str(Path(p).resolve()),     # single image
            "max_frames": 1, "num_video_frames_per_chunk": 1,
            "image_context_path": ref,
            "guidance": args.guidance,
            "seed": seed,
            **controls,
            **guided,
        }
        (cfg_dir / f"{stem}.json").write_text(json.dumps(cfg, indent=2))
        manifest.append({"stem": stem, "ref": Path(ref).name, "config": str(cfg_dir / f"{stem}.json")})

    with open(args.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "ref", "config"]); w.writeheader(); w.writerows(manifest)
    print(f"[cosmos-jobs] controls: seg={n_seg} (w={args.seg_weight}), depth={n_depth} "
          f"(w={args.depth_weight}), edge (w={args.edge_weight}), guidance={args.guidance}, "
          f"guided-fg-mask={n_guided} (steps={args.guided_steps})")

    # RESUMABLE batch runner: each pass runs inference (model loaded once) on the configs
    # whose output jpg doesn't exist yet; if a frame aborts the batch, the loop restarts
    # on the remaining ones. So a crash mid-run only costs one model reload, not progress.
    run = args.out / "run_all.sh"
    run.write_text(
        "#!/usr/bin/env bash\n"
        "# Cosmos-Transfer2.5 RESUMABLE batch (skips already-rendered frames).\n"
        f"cd {COSMOS_REPO}\n"
        f'CFGDIR="{cfg_dir.resolve()}"; OUT="{out_root}"; mkdir -p "$OUT"\n'
        "while :; do\n"
        "  todo=()\n"
        '  for c in "$CFGDIR"/*.json; do\n'
        '    n=$(basename "$c" .json)\n'
        '    [ -f "$OUT/$n.jpg" ] || todo+=("$c")\n'
        "  done\n"
        '  [ ${#todo[@]} -eq 0 ] && { echo "[run_all] all rendered."; break; }\n'
        '  echo "[run_all] $(date +%H:%M:%S) rendering ${#todo[@]} remaining frames ..."\n'
        '  ./.venv/bin/python examples/inference.py -i "${todo[@]}" -o "$OUT" \\\n'
        '    || echo "[run_all] batch aborted on a frame; resuming remaining ..."\n'
        "done\n")
    run.chmod(0o755)

    print(f"[cosmos-jobs] {len(sim_files)} images -> {cfg_dir}  "
          f"(prompts: {n_cap} caption, {len(sim_files)-n_cap} object-list; refs content-matched)")
    print(f"[cosmos-jobs] outputs -> {out_root}")
    print(f"[cosmos-jobs] run all:  bash {run}")


if __name__ == "__main__":
    main()
