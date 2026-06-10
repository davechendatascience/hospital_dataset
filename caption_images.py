"""Customized VLM captioner: turn each sim image into a per-image prompt for
Cosmos-Transfer2.5 (a realistic-ward description, not a generic template).

A vision-language model (Qwen2-VL) views each frame and writes a concise prompt
describing the scene AS A REAL hospital-ward photograph -- the objects, their
materials/colors, wall/floor finishes, and lighting -- explicitly told to ignore
that the input is a render. Output: {stem: prompt} JSON consumed by
gen_cosmos_jobs.py --captions.

    .venv/bin/python caption_images.py --img-dir ward_v3/train/images \
        --out cosmos_jobs/captions.json
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import torch
from PIL import Image

INSTRUCTION = (
    "You are writing a prompt for a photorealistic image generator. Look at this "
    "hospital ward scene and describe it as ONE concise paragraph for generating a "
    "REALISTIC photograph of a Taiwanese hospital ward room. Name the visible objects "
    "(e.g. care bed, mattress, IV pole, vital-signs monitor, air-conditioner, wall "
    "phone, bedside cabinet, privacy curtain, sink, toilet, grab bars), and describe "
    "their materials and colors, the wall and floor finishes, and the lighting. Do NOT "
    "mention that it is a 3D render, simulation, or CGI. Output only the prompt paragraph."
)


def list_images(d: Path):
    out = []
    for e in ("png", "jpg", "jpeg", "bmp", "webp"):
        out += glob.glob(str(d / f"*.{e}")) + glob.glob(str(d / f"*.{e.upper()}"))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img-dir", type=Path, default=Path("ward_v3/train/images"))
    ap.add_argument("--out", type=Path, default=Path("cosmos_jobs/captions.json"))
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--max-side", type=int, default=768, help="downscale long side for speed")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    files = list_images(args.img_dir)
    if args.limit:
        files = files[:args.limit]
    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    print(f"[caption] loading {args.model} ...", flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(args.model)

    # resume: keep existing captions
    args.out.parent.mkdir(parents=True, exist_ok=True)
    caps = {}
    if args.out.is_file():
        caps = json.loads(args.out.read_text())
        print(f"[caption] resuming; {len(caps)} already captioned")

    t0 = time.time()
    todo = [f for f in files if Path(f).stem not in caps]
    print(f"[caption] {len(todo)} images to caption (of {len(files)})", flush=True)
    for i, p in enumerate(todo, 1):
        img = Image.open(p).convert("RGB")
        if max(img.size) > args.max_side:
            s = args.max_side / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.BICUBIC)
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": INSTRUCTION}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        cap = proc.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                skip_special_tokens=True)[0].strip().replace("\n", " ")
        caps[Path(p).stem] = cap
        if i % 25 == 0 or i == len(todo):
            args.out.write_text(json.dumps(caps, indent=1))
            ips = i / (time.time() - t0)
            print(f"\r  {i}/{len(todo)}  {ips:.2f} img/s  ETA {(len(todo)-i)/max(ips,1e-9):.0f}s",
                  end="", flush=True)
    args.out.write_text(json.dumps(caps, indent=1))
    print(f"\n[caption] done: {len(caps)} captions -> {args.out}")
    # show one sample
    k = next(iter(caps))
    print(f"[caption] sample [{k}]: {caps[k][:240]}")


if __name__ == "__main__":
    main()
