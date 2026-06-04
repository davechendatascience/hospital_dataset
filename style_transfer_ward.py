#!/usr/bin/env python3
"""Stylize Ward_dataset0518 train/valid images with AdaIN, using a random
test_rgb image as the style reference per source image. Outputs a sibling
dataset Ward_dataset0518_styled/ mirroring the rgbDataset/ subtree;
test_rgb/ images and all COCO JSON annotations are copied unchanged.
AdaIN only edits color/texture, so bounding boxes remain valid."""

import argparse
import glob
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ------------------------------- AdaIN net ---------------------------------
# Architecture matches naoto0804/pytorch-AdaIN so the released weights load
# directly. Encoder is VGG19 (normalised) up to relu4_1 (index 31).

def build_decoder():
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )


def build_vgg():
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),                          # relu4_1 -> index 30
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
    )


def adain(content_feat, style_feat, eps=1e-5):
    # per-channel mean/std match (Huang & Belongie 2017)
    cN, cC = content_feat.shape[:2]
    c_mean = content_feat.view(cN, cC, -1).mean(-1).view(cN, cC, 1, 1)
    c_std = content_feat.view(cN, cC, -1).std(-1).view(cN, cC, 1, 1) + eps
    sN, sC = style_feat.shape[:2]
    s_mean = style_feat.view(sN, sC, -1).mean(-1).view(sN, sC, 1, 1)
    s_std = style_feat.view(sN, sC, -1).std(-1).view(sN, sC, 1, 1) + eps
    return s_std * (content_feat - c_mean) / c_std + s_mean


# ---------------------------- I/O helpers ----------------------------------

IMG_EXTS = ('jpg', 'jpeg', 'png', 'bmp', 'webp')


def list_images(folder):
    out = []
    for ext in IMG_EXTS:
        out.extend(glob.glob(os.path.join(folder, f'*.{ext}')))
        out.extend(glob.glob(os.path.join(folder, f'*.{ext.upper()}')))
    return sorted(out)


def load_image(path, max_size, down_scale=None):
    img = Image.open(path).convert('RGB')
    orig_size = img.size  # (W, H)
    w, h = img.size
    if max_size and max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        w, h = max(int(round(w * scale)), 8), max(int(round(h * scale)), 8)
    if down_scale:
        # round to nearest multiple of down_scale; needed by CAP-VSTNet RevResNet
        w = max((w // down_scale) * down_scale, down_scale)
        h = max((h // down_scale) * down_scale, down_scale)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.BICUBIC)
    return img, orig_size


to_tensor = transforms.ToTensor()


def img_to_tensor(img, device):
    return to_tensor(img).unsqueeze(0).to(device, non_blocking=True)


def tensor_to_pil(t):
    t = t.detach().clamp(0, 1).cpu()
    return transforms.functional.to_pil_image(t.squeeze(0))


# ---------------------- scene category mapping ------------------------------
# Train/valid filenames look like '<ObjectPrefix><LightingSuffix>_rgb_frame_*.png'
# e.g. 'BedcurtainColoredlight051101_rgb_frame_12_283.png'. The ObjectPrefix
# tells us the scene; the lighting suffix is just a render condition and is
# stripped before category lookup.

CATEGORIES = ('Ward', 'Bathroom', 'Frontroom')

_LIGHTING_SUFFIX = re.compile(
    r'(Default|Coloredlight|ColoredLight|GreyStudio)(_?\d+)?$'
)

BASE_TO_CAT = {
    # Bathroom
    'Shower':          'Bathroom',
    'Sink':            'Bathroom',   # plain Sink; FrontroomSink handled below
    'Toilet':          'Bathroom',
    'ToiletDoor':      'Bathroom',
    # Frontroom
    'FrontroomSink':   'Frontroom',
    'Door':            'Frontroom',  # un-prefixed 'Door' (Warddoor/ToiletDoor split out)
    # Ward (everything else falls here via the default below)
    'Airvent':         'Ward',
    'Bed2':            'Ward',
    'BedDataset':      'Ward',
    'Bedcurtain':      'Ward',
    'Bedsidetable':    'Ward',
    'Cabinet':         'Ward',
    'Cabinet2':        'Ward',
    'Companionchair':  'Ward',
    'Curtain':         'Ward',
    'Curtain2':        'Ward',
    'Oxygenflowmeter': 'Ward',
    'Stool':           'Ward',
    'Stoolunfold':     'Ward',
    'TV':              'Ward',
    'Warddoor':        'Ward',
}


def base_prefix(filename):
    """Strip lighting suffix; e.g. 'AirventColoredlight' -> 'Airvent',
    'BedDatasetGreyStudio' -> 'BedDataset', 'FrontroomSinkColoredlight_051101'
    -> 'FrontroomSink'."""
    name = os.path.basename(filename).split('_rgb_')[0]
    return _LIGHTING_SUFFIX.sub('', name)


def filename_category(filename):
    """Return scene category for a train/valid image, or None if unknown."""
    base = base_prefix(filename)
    return BASE_TO_CAT.get(base)


# -------------------- CLIP zero-shot classifier ----------------------------
# Used to label every test image into one of CATEGORIES once, then cached
# to disk so subsequent runs don't recompute.

CLIP_PROMPTS = {
    'Ward':      [
        'a photo of a hospital ward patient room with beds and curtains',
        'a photo of a hospital ward with a bed and an IV pole',
        'a photo of a patient room inside a hospital',
    ],
    'Bathroom':  [
        'a photo of a hospital bathroom with a toilet and a sink',
        'a photo of a hospital bathroom with a shower',
        'a photo of a hospital restroom',
    ],
    'Frontroom': [
        'a photo of a hospital reception or front room',
        'a photo of a hospital corridor near a sink and door',
        'a photo of an entry area inside a hospital',
    ],
}


def classify_test_with_clip(test_paths, device, cache_path):
    """Return dict {basename: category}. Caches results to cache_path."""
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if set(cached) >= {os.path.basename(p) for p in test_paths}:
            print(f'loaded cached CLIP labels from {cache_path}')
            return cached
        else:
            print(f'cache {cache_path} is stale; re-running CLIP')

    from transformers import CLIPModel, CLIPProcessor
    model_id = 'openai/clip-vit-base-patch32'
    print(f'loading CLIP {model_id} on {device}')
    clip = CLIPModel.from_pretrained(model_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_id)

    # average the per-prompt text embedding for each category
    cat_list = list(CLIP_PROMPTS.keys())
    prompts_flat, owners = [], []
    for cat in cat_list:
        for p in CLIP_PROMPTS[cat]:
            prompts_flat.append(p)
            owners.append(cat)
    def _as_tensor(x):
        # transformers 5.x sometimes returns a ModelOutput; unwrap to tensor
        if torch.is_tensor(x):
            return x
        for attr in ('text_embeds', 'image_embeds', 'pooler_output', 'last_hidden_state'):
            v = getattr(x, attr, None)
            if torch.is_tensor(v):
                return v
        raise RuntimeError(f'cannot extract tensor from CLIP output: {type(x).__name__}')

    with torch.no_grad():
        tok = proc(text=prompts_flat, return_tensors='pt', padding=True).to(device)
        txt = _as_tensor(clip.get_text_features(**tok))
        txt = txt / txt.norm(dim=-1, keepdim=True)
    cat_emb = torch.stack(
        [txt[[i for i, o in enumerate(owners) if o == c]].mean(0) for c in cat_list]
    )
    cat_emb = cat_emb / cat_emb.norm(dim=-1, keepdim=True)

    labels = {}
    batch_size = 16
    for i in tqdm(range(0, len(test_paths), batch_size), desc='CLIP classify test'):
        chunk = test_paths[i:i + batch_size]
        imgs = [Image.open(p).convert('RGB') for p in chunk]
        with torch.no_grad():
            inp = proc(images=imgs, return_tensors='pt').to(device)
            feat = _as_tensor(clip.get_image_features(**inp))
            feat = feat / feat.norm(dim=-1, keepdim=True)
            sims = feat @ cat_emb.T  # (B, n_cat)
            idx = sims.argmax(dim=-1).tolist()
        for p, j in zip(chunk, idx):
            labels[os.path.basename(p)] = cat_list[j]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(labels, f, indent=1, sort_keys=True)
        print(f'wrote CLIP labels to {cache_path}')

    # free CLIP weights before AdaIN starts
    del clip, proc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return labels


# ---------------------------- backends -------------------------------------
# Each `make_*_backend` returns (transfer_fn, down_scale).
# transfer_fn(content_pil, style_pil) -> stylized_pil
# down_scale is the spatial alignment constraint the network needs on H/W
# (1 for AdaIN; 4 for CAP-VSTNet's RevResNet with nStrides=[1,2,2]).


def make_adain_backend(weights_dir, device, alpha):
    dec_path = os.path.join(weights_dir, 'decoder.pth')
    vgg_path = os.path.join(weights_dir, 'vgg_normalised.pth')
    for path in (dec_path, vgg_path):
        if not os.path.isfile(path):
            sys.exit(f'AdaIN: missing weight file: {path}')
    vgg_full = build_vgg()
    vgg_full.load_state_dict(torch.load(vgg_path, map_location='cpu', weights_only=True))
    encoder = nn.Sequential(*list(vgg_full.children())[:31])  # up to relu4_1
    decoder = build_decoder()
    decoder.load_state_dict(torch.load(dec_path, map_location='cpu', weights_only=True))
    encoder.to(device).eval()
    decoder.to(device).eval()
    for m in (encoder, decoder):
        for p_ in m.parameters():
            p_.requires_grad_(False)

    @torch.no_grad()
    def transfer(content_pil, style_pil):
        c_t = img_to_tensor(content_pil, device)
        s_t = img_to_tensor(style_pil, device)
        cf = encoder(c_t)
        sf = encoder(s_t)
        feat = adain(cf, sf)
        feat = alpha * feat + (1.0 - alpha) * cf
        out_t = decoder(feat)
        return tensor_to_pil(out_t)

    return transfer, 1, 'AdaIN'


def make_capvst_backend(weights_path, capvst_src, device, alpha):
    """CAP-VSTNet photo mode: RevResNet (reversible flow) + cWCT transfer.
    Photo mode preserves fine patterns/edges because the encoder is
    information-preserving; cWCT then matches second-order statistics of
    deep features for atmospheric color/lighting transfer."""
    if capvst_src not in sys.path:
        sys.path.insert(0, capvst_src)
    from models.RevResNet import RevResNet  # noqa: E402
    from models.cWCT import cWCT             # noqa: E402

    rev = RevResNet(
        nBlocks=[10, 10, 10], nStrides=[1, 2, 2],
        nChannels=[16, 64, 256], in_channel=3,
        mult=4, hidden_dim=16, sp_steps=2,
    )
    sd = torch.load(weights_path, map_location='cpu', weights_only=False)
    rev.load_state_dict(sd['state_dict'])
    rev.to(device).eval()
    for p_ in rev.parameters():
        p_.requires_grad_(False)
    cwct = cWCT()
    down_scale = int(rev.down_scale)

    @torch.no_grad()
    def transfer(content_pil, style_pil):
        c_t = img_to_tensor(content_pil, device)
        s_t = img_to_tensor(style_pil, device)
        z_c = rev(c_t, forward=True)
        z_s = rev(s_t, forward=True)
        if alpha is not None and alpha < 1.0:
            z_cs = cwct.interpolation(z_c, styl_feat_list=[z_s],
                                      alpha_s_list=[1.0], alpha_c=1.0 - alpha)
        else:
            z_cs = cwct.transfer(z_c, z_s, None, None)
        out_t = rev(z_cs, forward=False)
        return tensor_to_pil(out_t)

    return transfer, down_scale, 'CAP-VSTNet(photo)'


# ---------------------------- main work ------------------------------------

def stylize_folder(src_dir, dst_dir, style_pools, transfer_fn, *,
                   work_size, down_scale, force, limit, rng, label,
                   fallback_pool, backend_name):
    """style_pools: dict category -> list[path]. For each source image we use
    its filename-derived category to draw a random style ref; if the category
    is unknown or empty we fall back to fallback_pool. transfer_fn(content_pil,
    style_pil) -> stylized_pil hides the backend (AdaIN or CAP-VSTNet)."""
    os.makedirs(dst_dir, exist_ok=True)
    src_paths = list_images(src_dir)
    if limit:
        src_paths = src_paths[:limit]
    n_written = 0
    n_skipped = 0
    cat_counts = Counter()
    unknown_prefixes = Counter()
    pbar = tqdm(src_paths, desc=f'{backend_name} {label}')
    for p in pbar:
        out_path = os.path.join(dst_dir, os.path.basename(p))
        if (not force) and os.path.exists(out_path):
            n_skipped += 1
            continue
        cat = filename_category(p)
        if cat is None:
            unknown_prefixes[base_prefix(p)] += 1
            pool = fallback_pool
            chosen_cat = 'UNKNOWN->fallback'
        else:
            pool = style_pools.get(cat) or fallback_pool
            chosen_cat = cat if style_pools.get(cat) else f'{cat}->fallback(empty pool)'
        cat_counts[chosen_cat] += 1
        try:
            content_img, orig_size = load_image(p, work_size, down_scale)
            style_path = pool[rng.randrange(len(pool))]
            style_img, _ = load_image(style_path, work_size, down_scale)
            out_img = transfer_fn(content_img, style_img)
            # restore original (W, H) so existing COCO bboxes stay valid.
            # LANCZOS (not BILINEAR) — preserves edge detail when upsampling
            # back from work_size to native, which otherwise looks blurry.
            if out_img.size != orig_size:
                out_img = out_img.resize(orig_size, Image.LANCZOS)
            # preserve extension/format from the source (PNG -> PNG, JPG -> JPG, ...)
            ext = os.path.splitext(p)[1].lower()
            if ext in ('.jpg', '.jpeg'):
                out_img.save(out_path, quality=95)
            else:
                out_img.save(out_path)
            n_written += 1
        except Exception as e:
            tqdm.write(f'[skip] {p}: {e}')
    pbar.close()
    if unknown_prefixes:
        tqdm.write(f'[{label}] unknown filename prefixes (fell back to all-test pool): '
                   + ', '.join(f'{k}={v}' for k, v in unknown_prefixes.most_common()))
    tqdm.write(f'[{label}] style-ref category usage: '
               + ', '.join(f'{k}={v}' for k, v in cat_counts.most_common()))
    return len(src_paths), n_written, n_skipped


def mirror_folder(src_dir, dst_dir, *, force, label):
    os.makedirs(dst_dir, exist_ok=True)
    src_paths = list_images(src_dir)
    for p in tqdm(src_paths, desc=f'copy {label}'):
        dst = os.path.join(dst_dir, os.path.basename(p))
        if (not force) and os.path.exists(dst):
            continue
        shutil.copy2(p, dst)
    return len(src_paths)


def copy_json_files(src_root, dst_root, splits):
    for split in splits:
        src_dir = os.path.join(src_root, split)
        dst_dir = os.path.join(dst_root, split)
        os.makedirs(dst_dir, exist_ok=True)
        for j in glob.glob(os.path.join(src_dir, '*.json')):
            shutil.copy2(j, os.path.join(dst_dir, os.path.basename(j)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src',  default='/home/edge-host/Documents/Ward_dataset0518',
                   help='original dataset root (containing ground_truth/rgbDataset/)')
    p.add_argument('--dst',  default='/home/edge-host/Documents/Ward_dataset0518_styled',
                   help='output sibling dataset root')
    p.add_argument('--backend', choices=('capvst', 'adain'), default='capvst',
                   help='style transfer model: capvst = CAP-VSTNet photo mode '
                        '(reversible flow, photorealistic, preserves fine patterns); '
                        'adain = legacy AdaIN (faster but smudges details)')
    p.add_argument('--weights-dir',
                   default='/home/edge-host/Documents/.venv/share/adain_weights',
                   help='AdaIN backend: dir containing decoder.pth and vgg_normalised.pth')
    p.add_argument('--capvst-ckpt',
                   default='/home/edge-host/Documents/.venv/share/capvst/checkpoints/photo_image.pt',
                   help='CAP-VSTNet backend: path to photo_image.pt')
    p.add_argument('--capvst-src',
                   default='/home/edge-host/Documents/.venv/share/capvst-src',
                   help='CAP-VSTNet backend: path to cloned CAP-VSTNet repo')
    p.add_argument('--work-size', type=int, default=768,
                   help='longest side at which the style net runs (output is '
                        'resized back to source dims afterwards). 512 is fast; '
                        '768 gives more detail (recommended for CAP-VSTNet photo).')
    p.add_argument('--alpha', type=float, default=1.0,
                   help='style strength: 1.0 = full test style, 0.0 = identity')
    p.add_argument('--limit', type=int, default=None,
                   help='per-folder cap (use for smoke tests)')
    p.add_argument('--splits', nargs='+', default=['train_rgb', 'valid_rgb'],
                   help='folders under ground_truth/rgbDataset/ to stylize')
    p.add_argument('--style-split', default='test_rgb',
                   help='folder under ground_truth/rgbDataset/ used as the style pool')
    p.add_argument('--copy-style-split', action='store_true', default=True,
                   help='also mirror the style split (test_rgb) into dst')
    p.add_argument('--force', action='store_true',
                   help='overwrite outputs even if they already exist')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--clip-cache',
                   default='/home/edge-host/Documents/.venv/share/adain_weights/test_clip_labels.json',
                   help='cache file for CLIP-classified test labels (pass empty string to disable)')
    p.add_argument('--ignore-category', action='store_true',
                   help='disable category matching; pick a uniformly random test image per source (legacy behavior)')
    args = p.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    rgb_src = os.path.join(args.src, 'ground_truth', 'rgbDataset')
    rgb_dst = os.path.join(args.dst, 'ground_truth', 'rgbDataset')
    if not os.path.isdir(rgb_src):
        sys.exit(f'expected rgbDataset subtree at {rgb_src}')

    # ---- load style transfer backend ----
    if args.backend == 'capvst':
        print(f'using backend CAP-VSTNet (photo mode), ckpt={args.capvst_ckpt}, '
              f'src={args.capvst_src}, device={args.device}')
        transfer_fn, down_scale, backend_name = make_capvst_backend(
            args.capvst_ckpt, args.capvst_src, args.device, args.alpha
        )
    else:
        print(f'using backend AdaIN, weights_dir={args.weights_dir}, device={args.device}')
        transfer_fn, down_scale, backend_name = make_adain_backend(
            args.weights_dir, args.device, args.alpha
        )
    print(f'  backend={backend_name}, work_size={args.work_size}, '
          f'down_scale={down_scale}, alpha={args.alpha}')

    # ---- gather style refs ----
    style_dir = os.path.join(rgb_src, args.style_split)
    style_paths = list_images(style_dir)
    if not style_paths:
        sys.exit(f'no style images found under {style_dir}')
    print(f'{len(style_paths)} style refs in {args.style_split}, work_size={args.work_size}, alpha={args.alpha}')

    # ---- classify test refs into scene categories with CLIP ----
    if args.ignore_category:
        style_pools = {cat: [] for cat in CATEGORIES}  # never used
        fallback_pool = style_paths
        print('--ignore-category set: drawing style refs uniformly from all test images')
    else:
        cache = args.clip_cache or None
        labels = classify_test_with_clip(style_paths, args.device, cache)
        style_pools = defaultdict(list)
        for sp in style_paths:
            cat = labels.get(os.path.basename(sp))
            if cat in CATEGORIES:
                style_pools[cat].append(sp)
        style_pools = dict(style_pools)
        fallback_pool = style_paths
        print('CLIP scene labels for test_rgb:')
        for cat in CATEGORIES:
            n = len(style_pools.get(cat, []))
            print(f'  {cat:10s}: {n} test images')

    # ---- stylize content splits ----
    t0 = time.time()
    totals = {}
    for split in args.splits:
        src_dir = os.path.join(rgb_src, split)
        dst_dir = os.path.join(rgb_dst, split)
        if not os.path.isdir(src_dir):
            print(f'[warn] skipping missing split {src_dir}')
            continue
        n_tot, n_new, n_skip = stylize_folder(
            src_dir, dst_dir, style_pools, transfer_fn,
            work_size=args.work_size, down_scale=down_scale,
            force=args.force, limit=args.limit, rng=rng, label=split,
            fallback_pool=fallback_pool, backend_name=backend_name,
        )
        totals[split] = (n_tot, n_new, n_skip)

    # ---- mirror style split + all jsons (geometry preserved -> annotations valid) ----
    if args.copy_style_split:
        src_dir = os.path.join(rgb_src, args.style_split)
        dst_dir = os.path.join(rgb_dst, args.style_split)
        n_style = mirror_folder(src_dir, dst_dir, force=args.force, label=args.style_split)
        totals[args.style_split] = (n_style, n_style, 0)

    json_splits = sorted(set(args.splits + ([args.style_split] if args.copy_style_split else [])))
    copy_json_files(rgb_src, rgb_dst, json_splits)

    print('\n--- summary ---')
    for split, (n_tot, n_new, n_skip) in totals.items():
        print(f'  {split:12s}: total={n_tot}  written={n_new}  skipped(existing)={n_skip}')
    print(f'  elapsed: {time.time()-t0:.1f}s')
    print(f'  output dataset: {args.dst}')


if __name__ == '__main__':
    main()
