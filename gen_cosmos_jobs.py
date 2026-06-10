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
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent
TEST = PROJECT / "ward_v3" / "test" / "images"
COSMOS_REPO = Path("/home/edge-host/cosmos-transfer2.5")

# 3 rooms: representative real photo (style ref) + its prompt (from DINOv2
# clustering of ward_v3/test + manual captioning).
ROOMS = [
    {
        "name": "bathroom",
        "ref": TEST / "WIN_20260331_11_26_06_Pro_frame_04423_png.rf.ebcab57482d5df6214098c18da6fce1a.jpg",
        "prompt": ("A realistic photograph of a hospital ward ensuite bathroom. White "
                   "square ceramic wall tiles, beige floor tiles; a white toilet; "
                   "stainless-steel L-shaped and flip-up grab bars; a chrome toilet-paper "
                   "holder; a wall-mounted ceramic sink with chrome faucet and a framed "
                   "mirror; red nurse-call buttons; an open doorway to the ward with a care "
                   "bed, mint-green curtain and light wood laminate floor. Soft daylight, "
                   "slightly wide-angle, high detail."),
    },
    {
        "name": "headwall",
        "ref": TEST / "WIN_20260331_11_05_48_Pro_frame_04401_png.rf.e8625b1d4e153b4bde6f719191fe8096.jpg",
        "prompt": ("A realistic close-up photograph of a hospital ward headwall. Cream "
                   "upper wall with beige wood-grain wainscot; a large white louvered "
                   "air-conditioner in a wood frame; a wall-mounted vital-signs monitor on "
                   "an articulated arm; a white UV germicidal lamp; a beige corded wall "
                   "telephone; a white bedside cabinet; a medical gas panel with a green "
                   "oxygen flowmeter, a suction regulator with a clear collection jar, a "
                   "pressure gauge and colored wall outlets; a pale mint-green privacy "
                   "curtain. Soft daylight, high detail."),
    },
    {
        "name": "fullroom",
        "ref": TEST / "WIN_20260331_11_10_31_Pro_frame_00630_png.rf.75d81cbd26be74104e3509f5df918c1f.jpg",
        "prompt": ("A realistic wide photograph of a hospital ward patient room. Cream "
                   "walls with beige wood-grain wainscot; a suspended grid ceiling; a white "
                   "louvered air-conditioner in a wood frame; a wall-mounted vitals monitor "
                   "on an arm; a UV germicidal lamp; a beige wall telephone; a "
                   "stainless-steel IV pole with hooks; a white bedside cabinet; an electric "
                   "care bed with a deep purple mattress and white plastic side rails; a "
                   "mint-green privacy curtain; a window with sheer curtains and bright "
                   "daylight; a wood door; light wood laminate floor. Slightly wide-angle, "
                   "high detail."),
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
    ap.add_argument("--control", default="edge", choices=["edge", "depth", "seg"])
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

    # DINOv2 embeddings: 3 room refs + all sim images -> nearest-room assignment
    import measure_domain_gap as mdg
    from transformers import AutoImageProcessor, AutoModel
    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
    ref_emb = np.asarray(mdg.embed_set([str(r["ref"]) for r in ROOMS], dino, proc, dev, 8, "dinov2"))
    sim_emb = np.asarray(mdg.embed_set(sim_files, dino, proc, dev, 32, "dinov2"))
    # cosine distance -> nearest room
    refn = ref_emb / (np.linalg.norm(ref_emb, axis=1, keepdims=True) + 1e-9)
    simn = sim_emb / (np.linalg.norm(sim_emb, axis=1, keepdims=True) + 1e-9)
    assign = (simn @ refn.T).argmax(1)          # index into ROOMS per sim image

    cfg_dir = args.out / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_root = (args.out / "outputs").resolve()
    manifest, counts = [], [0, 0, 0]
    for p, ri in zip(sim_files, assign):
        room = ROOMS[int(ri)]
        counts[int(ri)] += 1
        stem = Path(p).stem
        cfg = {
            "name": f"{room['name']}_{stem}",
            "prompt": room["prompt"],
            "video_path": str(Path(p).resolve()),     # single image
            "max_frames": 1,
            "num_video_frames_per_chunk": 1,
            "image_context_path": str(Path(room["ref"]).resolve()),
            "seed": args.seed,
            args.control: {},                          # control computed on the fly
        }
        (cfg_dir / f"{stem}.json").write_text(json.dumps(cfg, indent=2))
        manifest.append({"stem": stem, "room": room["name"], "config": str(cfg_dir / f"{stem}.json")})

    with open(args.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "room", "config"]); w.writeheader(); w.writerows(manifest)

    # loop runner: runs Cosmos-Transfer2.5 per config (one image at a time)
    run = args.out / "run_all.sh"
    run.write_text(
        "#!/usr/bin/env bash\n"
        "# Run Cosmos-Transfer2.5 per-image (one config at a time).\n"
        f"set -e\ncd {COSMOS_REPO}\n"
        f'for cfg in {cfg_dir.resolve()}/*.json; do\n'
        f'  name=$(basename "$cfg" .json)\n'
        f'  echo "[cosmos] $name"\n'
        f'  ./.venv/bin/python examples/inference.py -i "$cfg" -o "{out_root}/$name"\n'
        "done\n")
    run.chmod(0o755)

    print(f"[cosmos-jobs] {len(sim_files)} images -> {cfg_dir}")
    print(f"[cosmos-jobs] room assignment: " +
          ", ".join(f"{ROOMS[i]['name']}={counts[i]}" for i in range(3)))
    print(f"[cosmos-jobs] outputs -> {out_root}")
    print(f"[cosmos-jobs] run all:  bash {run}")


if __name__ == "__main__":
    main()
