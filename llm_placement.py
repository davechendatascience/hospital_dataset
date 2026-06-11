"""LLM layout designer for the ward: Qwen-VL proposes arrangements, the
placement engine validates them.

Why hybrid: a VLM is good at SEMANTIC arrangement ("chair beside the bed for a
visitor", "bins tucked by the door") but unreliable at emitting collision-free
coordinates, and far too slow to run once per rendered frame (25k+ frames).
So the split is:
  * the LLM designs a POOL of qualitatively different layouts (this script),
    seeing a top-down floor plan image + the scene's degrees of freedom;
  * placement_dr.apply_layout() validates every proposed DOF through the exact
    same z-aware, grandfathered collision machinery as the rule engine, and
    anything invalid falls back to the original pose (reported);
  * the renderer samples layouts from the pool per frame (+ rider scatter),
    keeping label validity and throughput.

Usage (defaults match the ward_10k build):
    .venv/bin/python llm_placement.py \
        --inputs ward_10k/_train_render/_placement_inputs.json \
        --out llm_layouts --num-layouts 10
Outputs: <out>/layouts.json, <out>/original.png, <out>/layout_NN.png previews.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw

import placement_dr as P

SCENARIOS = [
    "a family member is visiting: place the companion chair close beside the bed",
    "floor-cleaning day: move light furniture away from the room centre, toward walls",
    "post-procedure monitoring: overbed table pulled over the mid-bed, IV pole at the bed head",
    "night configuration: chair and stool stowed near the wall, bins by the door side",
    "physical-therapy session: maximise clear floor space in the middle of the room",
    "restocking round: bins pulled out to be reachable, scale near the corridor side",
    "two nurses working at the bedside: both bed sides clear, table at the bed foot",
    "tidy minimal ward: everything aligned neatly along the walls",
    "patient reading in bed: overbed table at the chest, chair at the window side",
    "wheelchair transfer setup: wide clear approach to the near side of the bed",
]


# ------------------------------------------------------------ scene -> DOF --
def load_scene(inputs_path):
    d = json.loads(Path(inputs_path).read_text())
    objs = d["objects"]
    for o in objs:
        o["aabb"] = (tuple(o["aabb"][0]), tuple(o["aabb"][1]))
        o["translate"] = tuple(o["translate"])
        o["size"] = tuple(o["size"])
    return objs, d["rooms"], d["floor_z"], d["wall_aabbs"]


def short_ids(ctx):
    """Stable short ids the LLM can reference -> mapping to engine DOFs."""
    ids = {}                      # id -> ("floor", path) | ("sat", path) | ("wall", gi)
    counts = {}
    for o in ctx["floor_movers"]:
        c = o["class"]
        counts[c] = counts.get(c, 0) + 1
        ids[f"{c}_{counts[c]}"] = ("floor", o["path"])
    for b in ctx["bays"]:
        for p in b["members"]:
            if ctx["roles"][p] == "satellite":
                c = ctx["obj_by_path"][p]["class"]
                counts[c] = counts.get(c, 0) + 1
                ids[f"{c}_{counts[c]}"] = ("sat", p)
    for gi, g in enumerate(ctx["wall_groups"]):
        if g["in_bay"]:
            continue
        c = ctx["obj_by_path"][g["members"][0]]["class"]
        counts[c] = counts.get(c, 0) + 1
        ids[f"wall_{c}_{counts[c]}"] = ("wall", gi)
    return ids


def dof_description(ctx, ids):
    lines = []
    if ctx["bays"]:
        b = ctx["bays"][0]
        lines.append(f'- "bay_slide": one number in [-0.8, 0.8] (m). Slides the whole '
                     f'bed bay (bed + its wall equipment) along its wall '
                     f'(the {b["axis"]} axis). Dense walls may only allow small values.')
    sat = [i for i, (k, _) in ids.items() if k == "sat"]
    if sat:
        lines.append(f'- "sat_slides": {{id: number in [-0.5, 0.5]}} -- slides that '
                     f'item ALONG the bed relative to the bay. ids: {sat}')
    wall = [i for i, (k, _) in ids.items() if k == "wall"]
    if wall:
        lines.append(f'- "wall_slides": {{id: number in [-0.2, 0.2]}} -- slides that '
                     f'wall item along its wall. ids: {wall}')
    floor = [i for i, (k, _) in ids.items() if k == "floor"]
    if floor:
        rows = []
        for i in floor:
            o = ctx["obj_by_path"][ids[i][1]]
            cx = 0.5 * (o["aabb"][0][0] + o["aabb"][1][0])
            cy = 0.5 * (o["aabb"][0][1] + o["aabb"][1][1])
            rows.append(f'    {i}: now at ({cx:.2f}, {cy:.2f}), '
                        f'footprint {o["size"][0]:.2f}x{o["size"][1]:.2f} m')
        lines.append('- "floor": {id: [x, y]} -- new world-space CENTRE for that '
                     'free-standing item:\n' + "\n".join(rows))
    return "\n".join(lines)


def params_from_ids(ctx, ids, raw):
    """Translate the LLM's id-keyed JSON into engine params. Tolerant: models
    sometimes emit id entries at the TOP level instead of nested under their
    DOF key, so id-keyed entries anywhere are accepted."""
    params = {"bay_slides": {}, "sat_slides": {}, "wall_slides": {}, "floor": {}}

    def eat(i, v):
        if i not in ids:
            return
        kind, ref = ids[i]
        if kind == "floor" and isinstance(v, (list, tuple)) and len(v) == 2:
            params["floor"][ref] = (float(v[0]), float(v[1]))
        elif kind == "sat" and isinstance(v, (int, float)):
            params["sat_slides"][ref] = float(v)
        elif kind == "wall" and isinstance(v, (int, float)):
            params["wall_slides"][ref] = float(v)

    if isinstance(raw.get("bay_slide"), (int, float)):
        params["bay_slides"][0] = float(raw["bay_slide"])
    for key in ("sat_slides", "wall_slides", "floor"):
        sub = raw.get(key)
        if isinstance(sub, dict):
            for i, v in sub.items():
                eat(i, v)
    for i, v in raw.items():                  # top-level id entries
        if i not in ("bay_slide", "sat_slides", "wall_slides", "floor"):
            eat(i, v)
    return params


# ------------------------------------------------------------ floor plan ----
def draw_plan(ctx, walls, positions=None, path="plan.png", scale=70):
    """Top-down floor plan; positions (path->xyz) overrides object centres."""
    xs, ys = [], []
    for r in ctx["by_room"].values():
        xs += [r["xmin"], r["xmax"]]; ys += [r["ymin"], r["ymax"]]
    for o in ctx["objs"]:
        f = P._foot(o["aabb"]); xs += [f[0], f[2]]; ys += [f[1], f[3]]
    x0, x1, y0, y1 = min(xs) - 0.4, max(xs) + 0.4, min(ys) - 0.4, max(ys) + 0.4
    W, H = int((x1 - x0) * scale), int((y1 - y0) * scale)
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)

    def px(x, y):                       # world -> image (y flipped)
        return ((x - x0) * scale, (y1 - y) * scale)

    def rect(f, **kw):
        (ax, ay), (bx, by) = px(f[0], f[3]), px(f[2], f[1])
        dr.rectangle([ax, ay, bx, by], **kw)

    for w in walls:                      # wall meshes
        rect(P._foot((tuple(w[0]), tuple(w[1]))), fill=(190, 190, 190))
    for r in ctx["by_room"].values():    # room bounds
        (ax, ay), (bx, by) = px(r["xmin"], r["ymax"]), px(r["xmax"], r["ymin"])
        dr.rectangle([ax, ay, bx, by], outline=(120, 120, 220), width=2)

    colors = {"bed": (220, 90, 90), "satellite": (240, 160, 60),
              "wall": (150, 110, 200), "floor": (90, 170, 90),
              "surface": (230, 210, 80), "fixture": (170, 170, 170)}
    for o in ctx["objs"]:
        role = ctx["roles"][o["path"]]
        f = P._foot(o["aabb"])
        if positions and o["path"] in positions:
            t0 = ctx["pos0"][o["path"]]
            p1 = positions[o["path"]]
            f = P._rect_shift(f, p1[0] - t0[0], p1[1] - t0[1])
        outline = (60, 60, 60) if o["class"] != "door" else (140, 70, 30)
        rect(f, fill=colors[role], outline=outline)
        cx, cy = 0.5 * (f[0] + f[2]), 0.5 * (f[1] + f[3])
        label = o["class"][:14]
        dr.text((px(cx, cy)[0] - 3.0 * len(label), px(cx, cy)[1] - 5),
                label, fill=(0, 0, 0))
    img.save(path)
    return img


# ------------------------------------------------------------------ LLM -----
def load_qwen(model_id, device):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device)
    return proc, model


def ask_llm(proc, model, plan_img, prompt, max_new_tokens=600):
    import torch
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    text = proc.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    inputs = proc(text=[text], images=[plan_img], return_tensors="pt"
                  ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=0.9, top_p=0.95)
    out = out[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(out, skip_special_tokens=True)[0]


def parse_json(txt):
    """Last parseable balanced {...} block (reasoning models may emit prose
    or scratch JSON first)."""
    best = None
    i = 0
    while True:
        m = txt.find("{", i)
        if m < 0:
            break
        depth = 0
        end = None
        for j in range(m, len(txt)):
            if txt[j] == "{":
                depth += 1
            elif txt[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            break
        try:
            cand = json.loads(re.sub(r",\s*([}\]])", r"\1", txt[m:end + 1]))
            if isinstance(cand, dict):
                best = cand
        except json.JSONDecodeError:
            pass
        i = m + 1
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path,
                    default=Path("ward_10k/_train_render/_placement_inputs.json"))
    ap.add_argument("--out", type=Path, default=Path("llm_layouts"))
    ap.add_argument("--num-layouts", type=int, default=10)
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    objs, rooms, floor_z, walls = load_scene(args.inputs)
    ctx = P.build_ctx(objs, rooms, floor_z, wall_boxes=walls, verbose=True)
    ids = short_ids(ctx)
    args.out.mkdir(parents=True, exist_ok=True)
    plan = draw_plan(ctx, walls, path=str(args.out / "original.png"))
    ward = next(iter(ctx["by_room"].values()))
    dof = dof_description(ctx, ids)

    # worked example with REAL ids + plausible non-zero values (anti-parroting:
    # the values are deliberately arbitrary; the prompt says to choose your own)
    fl = [i for i, (k, _) in ids.items() if k == "floor"]
    st = [i for i, (k, _) in ids.items() if k == "sat"]
    ex_floor = {}
    for n, i in enumerate(fl):
        o = ctx["obj_by_path"][ids[i][1]]
        cx = 0.5 * (o["aabb"][0][0] + o["aabb"][1][0])
        cy = 0.5 * (o["aabb"][0][1] + o["aabb"][1][1])
        ex_floor[i] = [round(cx + (0.3 if n % 2 else -0.3), 2),
                       round(cy + (0.2 if n % 3 else -0.2), 2)]
    example = json.dumps({"bay_slide": 0.05,
                          "sat_slides": {st[0]: -0.2} if st else {},
                          "wall_slides": {},
                          "floor": ex_floor})

    print(f"[llm-placement] loading {args.model} ...", flush=True)
    proc, model = load_qwen(args.model, args.device)

    base_prompt = (
        "You are arranging furniture in a hospital ward for a synthetic-data "
        "renderer. The image is the TOP-DOWN floor plan: x increases to the "
        "right, y increases UPWARD, units are meters. Gray = walls, brown "
        "outline = doors (keep clear!), red = bed, orange = bed-side tables / "
        "IV pole, purple = wall-mounted equipment, green = movable floor "
        "items, light gray = fixed fixtures.\n"
        f"Main room bounds: x in [{ward['xmin']:.2f}, {ward['xmax']:.2f}], "
        f"y in [{ward['ymin']:.2f}, {ward['ymax']:.2f}].\n\n"
        "You control ONLY these degrees of freedom:\n"
        f"{dof}\n\n"
        "Hard rules: no two objects may overlap; keep at least 5 cm clearance; "
        "never block a door; keep items inside the room. A separate validator "
        "rejects any move that violates these, so prefer safe, clearly-valid "
        "positions.\n\n"
        "SCENARIO: <<SCENARIO>>\n\n"
        "Reply with ONLY a JSON object (no prose, no reasoning, no code "
        "fences) with exactly these keys: bay_slide, sat_slides, wall_slides, "
        "floor.\n"
        f"- \"floor\" MUST contain an entry for EVERY one of these ids: "
        f"{[i for i, (k, _) in ids.items() if k == 'floor']} -- give each a "
        "new [x, y] centre that realizes the scenario (reuse the current "
        "centre only if that item truly should not move).\n"
        f"- valid sat ids: {[i for i, (k, _) in ids.items() if k == 'sat']}; "
        f"valid wall ids: {[i for i, (k, _) in ids.items() if k == 'wall']}.\n"
        "- use ONLY ids from those lists, exactly as written. Do not invent "
        "ids and do not output the literal word 'id'.\n"
        "- at least THREE floor items must move by 0.3 m or more -- a layout "
        "that keeps everything at its current centre is invalid.\n"
        f"Example shape (with MADE-UP values -- choose your own!): "
        f"{example}\n"
    )

    results, total_dof, total_rej = [], 0, 0
    for k in range(args.num_layouts):
        scenario = SCENARIOS[k % len(SCENARIOS)]
        raw = None
        for attempt in range(args.retries + 1):
            txt = ask_llm(proc, model, plan,
                          base_prompt.replace("<<SCENARIO>>", scenario))
            raw = parse_json(txt)
            if raw is not None:
                break
        if raw is None:
            print(f"[llm-placement] layout {k}: JSON parse failed, skipped")
            continue
        params = params_from_ids(ctx, ids, raw)
        n_dof = (len(params["bay_slides"]) + len(params["sat_slides"]) +
                 len(params["wall_slides"]) + len(params["floor"]))
        positions, rejected = P.apply_layout(ctx, params, seed=k)
        total_dof += n_dof
        total_rej += len(rejected)
        draw_plan(ctx, walls, positions=positions,
                  path=str(args.out / f"layout_{k:02d}.png"))
        results.append({"scenario": scenario, "raw": raw,
                        "proposed_dof": n_dof, "rejected": rejected,
                        "positions": {p: list(v) for p, v in positions.items()}})
        print(f"[llm-placement] layout {k}: {n_dof} DOFs proposed, "
              f"{len(rejected)} rejected {rejected if rejected else ''}")

    (args.out / "layouts.json").write_text(json.dumps(results, indent=1))
    ok = sum(1 for r in results if r["proposed_dof"] > 0)
    print(f"[llm-placement] {len(results)} layouts ({ok} non-trivial) -> "
          f"{args.out}/layouts.json; DOFs proposed={total_dof}, "
          f"rejected={total_rej} "
          f"({100.0 * (1 - total_rej / max(total_dof, 1)):.0f}% accepted)")


if __name__ == "__main__":
    main()
