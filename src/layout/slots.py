"""Data-driven slot layout: author rooms + typed placement slots, then POPULATE
them into the exact object_targets / rooms shapes the renderer and placement_dr
already consume (Phase 1 of the slot redesign -- see docs/ plan).

The idea is to make a slot SPEC the seed of the scene instead of an authored
USD layout. replicator_dataset.py today reads authored prim poses to build
`object_targets`, then calls placement_dr.generate() to jitter them. This
module produces the SAME `object_targets` (path, class, centroid, aabb, size,
translate, room) and `rooms` ({name, xmin, xmax, ymin, ymax}) from a JSON spec,
so everything downstream -- collision-aware jitter, camera aiming, COCO export
-- is untouched.

Crucially we do NOT tell placement_dr what kind of slot an object came from:
we only emit physically sensible geometry, and placement_dr RE-INFERS the role
(bed / wall / floor / surface / fixture) from class + geometry exactly as it
does for the authored ward. So the coupling between the two modules is just the
object_targets dict -- nothing more.

Slot types (these decide how WE compute the seed geometry; the role is then
re-inferred downstream):
  * wall    -- mounted flush to a room wall at a height; slides along the wall
  * floor   -- stands on the floor inside a rectangular region of the room
  * surface -- rests on top of another slot's object (a receptacle)
  * bay     -- a hospital_bed flush to a headwall plus its satellites

Any slot may carry an explicit `center` ([x,y,z], or `bed_center` for a bay's
bed and `center` on each satellite). When present it OVERRIDES the parametric
placement (edge/t/rect/dperp) and positions the object exactly there -- this is
how template_from_targets captures an authored layout so --layout-spec
reproduces it. To re-position a captured item, edit its `center` or delete it
to fall back to the parametric fields.

Coordinate convention matches placement_dr / replicator: world XY plane, +Z up,
`floor_z` is the floor height. A wall is identified by an `edge` [axis, side]
where `axis` in {"x","y"} is the PINNED coordinate and `side` in {"min","max"}
says which room extreme it sits at; the wall RUNS along the perpendicular axis,
and `t` in [0,1] is the position along that run between the room's min and max.

Pure Python (placement_dr imported only for the optional collision check), so
the whole populate step is unit-testable without Isaac.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# allow running this module directly (python src/layout/slots.py ...): put the
# src/ dir on the path so `from layout import engine` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_AXIS_I = {"x": 0, "y": 1}
_ELEVATED = 0.35   # z_bottom (m above floor) over which an item reads as wall-mounted


# ----------------------------------------------------------------- spec I/O --
def load_spec(path):
    """Load and lightly validate a slot spec JSON. -> dict."""
    spec = json.loads(Path(path).read_text())
    for key in ("rooms", "slots", "assets"):
        if key not in spec:
            raise ValueError(f"slot spec missing top-level '{key}'")
    spec.setdefault("floor_z", 0.0)
    by_name = {r["name"] for r in spec["rooms"]}
    for s in spec["slots"]:
        if s["type"] in ("wall", "floor", "bay") and s.get("room") not in by_name:
            raise ValueError(f"slot {s['id']!r} references unknown room "
                             f"{s.get('room')!r}")
    return spec


def _pick_asset(spec, cls, rng):
    """Choose one asset variant for a class. -> (usd, (sx, sy, sz))."""
    variants = spec["assets"].get(cls)
    if not variants:
        raise ValueError(f"no asset registered for class {cls!r}")
    a = variants[rng.randrange(len(variants))]
    sx, sy, sz = a["size"]
    return a.get("usd", ""), (float(sx), float(sy), float(sz))


# ---------------------------------------------------------------- geometry --
def _room_by_name(spec):
    return {r["name"]: r for r in spec["rooms"]}


def _wall_frame(room, edge):
    """Resolve a wall into the values placement of an item flush to it needs.

    -> dict(axis, side, run, pinned, run_lo, run_hi, sign) where `axis` is the
    pinned coordinate, `run` the perpendicular (wall length) axis, `pinned` the
    world value of the wall plane, [run_lo, run_hi] the room span along `run`,
    and `sign` = +1 if the room interior is in the +axis direction (side=min)."""
    axis, side = edge[0], edge[1]
    run = "y" if axis == "x" else "x"
    pinned = room[f"{axis}{side}"]
    return {"axis": axis, "side": side, "run": run,
            "pinned": pinned,
            "run_lo": room[f"{run}min"], "run_hi": room[f"{run}max"],
            "sign": 1.0 if side == "min" else -1.0}


def _aabb_against_wall(frame, run_center, size, z_bottom, floor_z):
    """AABB ((mn),(mx)) for an item flush to a wall, centred at `run_center`
    along the wall, its back face on the wall plane, bottom at floor_z+z_bottom."""
    sx, sy, sz = size
    depth = size[_AXIS_I[frame["axis"]]]          # extent perpendicular to wall
    run_half = 0.5 * size[_AXIS_I[frame["run"]]]  # half extent along the wall
    if frame["side"] == "min":
        a0, a1 = frame["pinned"], frame["pinned"] + depth
    else:
        a0, a1 = frame["pinned"] - depth, frame["pinned"]
    r0, r1 = run_center - run_half, run_center + run_half
    z0 = floor_z + z_bottom
    z1 = z0 + sz
    if frame["axis"] == "x":
        mn, mx = (a0, r0, z0), (a1, r1, z1)
    else:
        mn, mx = (r0, a0, z0), (r1, a1, z1)
    return (mn, mx)


def _aabb_at(cx, cy, size, z_bottom, floor_z):
    """Axis-aligned box centred at (cx, cy) on the floor (or raised)."""
    sx, sy, sz = size
    z0 = floor_z + z_bottom
    return ((cx - sx / 2, cy - sy / 2, z0),
            (cx + sx / 2, cy + sy / 2, z0 + sz))


def _aabb_centered(center, size):
    """Axis-aligned box of `size` centred on a full 3D `center`. Used when a
    slot carries an explicit `center` (dump-captured authored position) so the
    object is reproduced exactly instead of re-derived from edge/t/rect."""
    cx, cy, cz = center
    sx, sy, sz = size
    return ((cx - sx / 2, cy - sy / 2, cz - sz / 2),
            (cx + sx / 2, cy + sy / 2, cz + sz / 2))


def _on_top(parent_aabb, size, inset, rng):
    """AABB for an item resting on top of `parent_aabb`, scattered within the
    top face inset by `inset`. Falls back to the centre if it doesn't fit."""
    (pmn, pmx) = parent_aabb
    sx, sy, sz = size
    top = pmx[2]
    x0, x1 = pmn[0] + sx / 2 + inset, pmx[0] - sx / 2 - inset
    y0, y1 = pmn[1] + sy / 2 + inset, pmx[1] - sy / 2 - inset
    if x1 > x0 and y1 > y0:
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
    else:
        cx = 0.5 * (pmn[0] + pmx[0])
        cy = 0.5 * (pmn[1] + pmx[1])
    return ((cx - sx / 2, cy - sy / 2, top),
            (cx + sx / 2, cy + sy / 2, top + sz))


def _rect_overlap(a, b, margin=0.0):
    """a, b are (mn, mx) AABBs -- XY footprint overlap only."""
    return not (a[1][0] + margin <= b[0][0] or b[1][0] + margin <= a[0][0] or
                a[1][1] + margin <= b[0][1] or b[1][1] + margin <= a[0][1])


def _make_target(path, cls, aabb, room, usd="", group=None, xform=None,
                 affordance=None):
    (mn, mx) = aabb
    centroid = (0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]),
                0.5 * (mn[2] + mx[2]))
    # `translate` is the value placement_dr treats as pos0 and that the renderer
    # writes to the prim's translate op. For a synthetic (cube) object the pivot
    # is the AABB centre. For a real referenced asset the authored prim's pivot
    # is offset from its AABB centre (`xform["pivot_off"]`); keeping that offset
    # makes the asset's AABB centre land on the slot centre while the prim's
    # translate stays the pivot -- exactly how authored prims behave, so jitter
    # (which shifts translate and aabb by the same delta) stays consistent.
    translate = centroid
    if xform and xform.get("pivot_off"):
        po = xform["pivot_off"]
        translate = (centroid[0] + po[0], centroid[1] + po[1],
                     centroid[2] + po[2])
    return {
        "path": path,
        "class": cls,
        "centroid": centroid,
        "aabb": ((float(mn[0]), float(mn[1]), float(mn[2])),
                 (float(mx[0]), float(mx[1]), float(mx[2]))),
        "size": (float(mx[0] - mn[0]), float(mx[1] - mn[1]),
                 float(mx[2] - mn[2])),
        "translate": translate,
        "room": room,
        "usd": usd,
        # `_group` ties objects whose footprints are MEANT to coexist (a bay's
        # bed + satellites + their riders). Pairs sharing a group are exempt from
        # the seed overlap report -- placement_dr grandfathers them anyway. The
        # leading underscore marks it internal; to_placement_inputs() drops it.
        "_group": group,
        # `_xform` carries the authored rotation/scale (quat + scale) so the
        # renderer can replay the asset's orientation; None -> axis-aligned.
        "_xform": xform,
        # explicit, per-instance support mode (floor|wall|surface) from the slot
        # type -- AUTHORITATIVE over the class default in placement_affordances.
        "affordance": affordance,
    }


# ---------------------------------------------------------------- populate --
def populate(spec, seed=0):
    """Turn a slot spec into (object_targets, rooms, floor_z).

    Slots are processed wall/floor/bay first, then `surface` (which references
    a parent slot by id). Floor and surface items are rejection-sampled so they
    don't overlap already-placed objects; a slot that can't be satisfied is
    skipped with a recorded reason. -> (targets, rooms, floor_z, dropped)."""
    rng = random.Random(seed)
    floor_z = float(spec["floor_z"])
    rooms_by = _room_by_name(spec)
    rooms = [{"name": r["name"], "xmin": r["xmin"], "xmax": r["xmax"],
              "ymin": r["ymin"], "ymax": r["ymax"]} for r in spec["rooms"]]

    targets = []
    by_slot = {}            # slot id -> primary object_target (for surface refs)
    placed = []             # AABBs already committed (for overlap rejection)
    dropped = []

    def _commit(t):
        targets.append(t)
        placed.append(t["aabb"])
        return t

    walls = [s for s in spec["slots"] if s["type"] == "wall"]
    bays = [s for s in spec["slots"] if s["type"] == "bay"]
    floors = [s for s in spec["slots"] if s["type"] == "floor"]
    surfaces = [s for s in spec["slots"] if s["type"] == "surface"]

    # ---- wall slots: flush to the wall at height z, centred at t along it ----
    for s in walls:
        room = rooms_by[s["room"]]
        frame = _wall_frame(room, s["edge"])
        run_center = frame["run_lo"] + s["t"] * (frame["run_hi"] - frame["run_lo"])
        cls = s["class"] if "class" in s else s["allowed"][0]
        usd, size = _pick_asset(spec, cls, rng)
        z_bottom = float(s.get("z", _ELEVATED + 0.4))   # default safely "wall"
        if s.get("center") is not None:
            aabb = _aabb_centered(s["center"], size)
        else:
            aabb = _aabb_against_wall(frame, run_center, size, z_bottom, floor_z)
        by_slot[s["id"]] = _commit(
            _make_target(f"/World/Generated/{s['id']}", cls, aabb,
                         s["room"], usd, group=s["id"], xform=s.get("xform"),
                         affordance="wall"))

    # ---- bays: bed flush to headwall + satellites at relative offsets --------
    for s in bays:
        room = rooms_by[s["room"]]
        frame = _wall_frame(room, s["edge"])
        run_center = frame["run_lo"] + s["t"] * (frame["run_hi"] - frame["run_lo"])
        bed_cls = s.get("bed", "hospital_bed")
        usd, bed_size = _pick_asset(spec, bed_cls, rng)
        if s.get("bed_center") is not None:
            bed_aabb = _aabb_centered(s["bed_center"], bed_size)
        else:
            bed_aabb = _aabb_against_wall(frame, run_center, bed_size, 0.0, floor_z)
        by_slot[s["id"]] = _commit(
            _make_target(f"/World/Generated/{s['id']}/bed", bed_cls, bed_aabb,
                         s["room"], usd, group=s["id"],
                         xform=s.get("bed_xform"), affordance="floor"))
        # satellites: drun = along the wall from bed centre, dperp = OUT from
        # the wall into the room (so the table/pole stands beside the bed).
        for i, sat in enumerate(s.get("satellites", [])):
            scls = sat["class"]
            su, ssize = _pick_asset(spec, scls, rng)
            drun = float(sat.get("drun", 0.0))
            dperp = float(sat.get("dperp", 0.6))
            zb = float(sat.get("z", 0.0))
            rc = run_center + drun
            # perpendicular centre: start at wall plane, move inward by dperp
            depth = ssize[_AXIS_I[frame["axis"]]]
            pc = frame["pinned"] + frame["sign"] * (dperp + depth / 2)
            if frame["axis"] == "x":
                cx, cy = pc, rc
            else:
                cx, cy = rc, pc
            if sat.get("center") is not None:
                aabb = _aabb_centered(sat["center"], ssize)
            else:
                aabb = _aabb_at(cx, cy, ssize, zb, floor_z)
            sat_t = _commit(_make_target(
                f"/World/Generated/{s['id']}/{scls}_{i}", scls, aabb,
                s["room"], su, group=s["id"], xform=sat.get("xform"),
                affordance="floor"))
            # addressable so `surface` slots can rest items on a satellite (the
            # overbed table) instead of only the bed: "<bay_id>/<class>".
            by_slot[f"{s['id']}/{scls}_{i}"] = sat_t
            by_slot.setdefault(f"{s['id']}/{scls}", sat_t)

    # ---- floor slots: stand inside a rectangle, avoiding placed objects ------
    for s in floors:
        room_name = s["room"]
        cls = s["class"] if "class" in s else s["allowed"][0]
        usd, size = _pick_asset(spec, cls, rng)
        if s.get("center") is not None:      # faithful: reproduce authored spot
            by_slot[s["id"]] = _commit(_make_target(
                f"/World/Generated/{s['id']}", cls, _aabb_centered(s["center"],
                size), room_name, usd, group=s["id"], xform=s.get("xform"),
                affordance="floor"))
            continue
        x0, y0, x1, y1 = s["rect"]
        hx, hy = size[0] / 2, size[1] / 2
        chosen = None
        for _ in range(s.get("attempts", 40)):
            cx = rng.uniform(x0 + hx, x1 - hx) if x1 - hx > x0 + hx else 0.5 * (x0 + x1)
            cy = rng.uniform(y0 + hy, y1 - hy) if y1 - hy > y0 + hy else 0.5 * (y0 + y1)
            aabb = _aabb_at(cx, cy, size, 0.0, floor_z)
            if not any(_rect_overlap(aabb, q, margin=0.03) for q in placed):
                chosen = aabb
                break
        if chosen is None:
            dropped.append(f"floor slot {s['id']!r} ({cls}): no collision-free spot")
            continue
        by_slot[s["id"]] = _commit(_make_target(
            f"/World/Generated/{s['id']}", cls, chosen, room_name, usd,
            group=s["id"], xform=s.get("xform"), affordance="floor"))

    # ---- surface slots: rest on a parent slot's top face ---------------------
    for s in surfaces:
        parent = by_slot.get(s["on"])
        if parent is None:
            dropped.append(f"surface slot {s['id']!r}: parent {s['on']!r} not placed")
            continue
        cls = s["class"] if "class" in s else s["allowed"][0]
        usd, size = _pick_asset(spec, cls, rng)
        if s.get("center") is not None:       # faithful: reproduce authored spot
            by_slot[s["id"]] = _commit(_make_target(
                f"/World/Generated/{s['id']}", cls, _aabb_centered(s["center"],
                size), parent["room"], usd,
                group=parent.get("_group") or s["on"], xform=s.get("xform"),
                affordance="surface"))
            continue
        inset = float(s.get("inset", 0.02))
        aabb = None
        sibling_tops = [q for q in placed
                        if abs(q[0][2] - parent["aabb"][1][2]) < 1e-6]
        for _ in range(s.get("attempts", 20)):
            cand = _on_top(parent["aabb"], size, inset, rng)
            if not any(_rect_overlap(cand, q, margin=0.01) for q in sibling_tops):
                aabb = cand
                break
        if aabb is None:
            aabb = _on_top(parent["aabb"], size, inset, rng)   # accept overlap
        # a rider joins its support's group: it MEANS to overlap the support
        # (and, for a satellite support, whatever that satellite overhangs).
        by_slot[s["id"]] = _commit(_make_target(
            f"/World/Generated/{s['id']}", cls, aabb, parent["room"], usd,
            group=parent.get("_group") or s["on"], xform=s.get("xform"),
            affordance="surface"))

    return targets, rooms, floor_z, dropped


# -------------------------------------------------------- optional validate --
def check_overlaps(targets, floor_z, margin=0.0):
    """Report Z-aware footprint overlaps among the populated objects using
    placement_dr's own collision primitive, so the seed is consistent with what
    the jitter solver assumes. Pairs sharing a `_group` (a bay's bed +
    satellites + their riders) are MEANT to coexist and are skipped -- what
    remains is genuine authoring error (e.g. two bays placed on top of each
    other, a chair inside a wall slot). -> list of (pathA, pathB) strings."""
    from layout import engine as pdr
    blks = [(t["path"], pdr._blk(t["aabb"]), t.get("_group")) for t in targets]
    hits = []
    for i in range(len(blks)):
        for j in range(i + 1, len(blks)):
            gi, gj = blks[i][2], blks[j][2]
            if gi is not None and gi == gj:
                continue                    # intended intra-group coexistence
            if pdr._blk_hit(blks[i][1], blks[j][1], margin=margin):
                hits.append(f"{blks[i][0]} <-> {blks[j][0]}")
    return hits


def template_from_targets(object_targets, rooms, floor_z, wall_boxes=None,
                          usd_of=None, xform_of=None):
    """Reverse a populated/authored layout into a STARTER slot spec, in the
    SAME world coordinate frame as the input. Role inference is delegated to
    placement_dr.build_ctx (the jitter engine), so the emitted slot types match
    exactly how that engine would treat each object: beds + their satellites
    become `bay` slots, elevated wall items `wall` slots, floor furniture
    `floor` slots, on-support items `surface` slots; fixtures are skipped (they
    stay in the authored stage). Each object_target needs 'translate' + 'room'.
    -> a spec dict ready to json.dump and hand-edit. Asset USDs are left blank
    (so spawn_layout proxies them with cubes until you fill in real paths)."""
    from layout import engine as pdr
    ctx = pdr.build_ctx(object_targets, rooms, floor_z, wall_boxes=wall_boxes,
                        verbose=False)
    roles, bays = ctx["roles"], ctx["bays"]
    obp, by_room, support_of = (ctx["obj_by_path"], ctx["by_room"],
                                ctx["support_of"])
    usd_of, xform_of = usd_of or {}, xform_of or {}

    slots, assets, path_to_slot = [], {}, {}

    def _reg(o):
        assets.setdefault(o["class"], [{"usd": usd_of.get(o["path"], ""),
                                        "size": [round(v, 3) for v in o["size"]]}])

    def _t_along(room, run, c):
        lo, hi = room[f"{run}min"], room[f"{run}max"]
        return round((c[_AXIS_I[run]] - lo) / (hi - lo), 4) if hi > lo else 0.5

    bay_members = set()
    for bi, b in enumerate(bays):
        room = by_room[b["key"][0]]
        axis, side = b["key"][1]
        run = b["axis"]                       # slide axis == wall run direction
        sign = 1.0 if side == "min" else -1.0
        pinned = room[f"{axis}{side}"]
        bed = obp[b["members"][0]]
        bed_c = bed["centroid"]
        bay_id = f"bay_{bi}"
        _reg(bed)
        path_to_slot[bed["path"]] = bay_id
        bay_members.update(b["members"])
        sats = []
        for p in b["members"][1:]:
            o = obp[p]
            if roles[p] == "satellite":
                _reg(o)
                c = o["centroid"]
                depth = o["size"][_AXIS_I[axis]]
                sats.append({
                    "class": o["class"],
                    "drun": round(c[_AXIS_I[run]] - bed_c[_AXIS_I[run]], 3),
                    "dperp": round(sign * (c[_AXIS_I[axis]] - pinned) - depth / 2, 3),
                    "z": round(o["aabb"][0][2] - floor_z, 3),
                    "center": [round(v, 4) for v in c]})
                if p in xform_of:
                    sats[-1]["xform"] = xform_of[p]
                path_to_slot[p] = f"{bay_id}/{o['class']}"
            elif roles[p] == "wall":          # overhead group -> own wall slot
                _emit_wall(o, room, floor_z, slots, _reg, path_to_slot,
                           f"{bay_id}_w{len(slots)}", xform_of)
        bay = {"id": bay_id, "type": "bay", "room": room["name"],
               "edge": [axis, side], "t": _t_along(room, run, bed_c),
               "bed": bed["class"], "bed_center": [round(v, 4) for v in bed_c],
               "satellites": sats}
        if bed["path"] in xform_of:
            bay["bed_xform"] = xform_of[bed["path"]]
        slots.append(bay)

    wi = fi = 0
    for o in object_targets:
        p = o["path"]
        if p in bay_members or p not in roles:
            continue
        room = by_room.get(o.get("room"))
        if roles[p] == "wall" and room is not None:
            _emit_wall(o, room, floor_z, slots, _reg, path_to_slot,
                       f"wall_{wi}", xform_of)
            wi += 1
        elif roles[p] == "floor" and room is not None:
            _reg(o)
            (mn, mx) = o["aabb"]
            m = 0.4
            fl = {"id": f"floor_{fi}", "type": "floor",
                  "room": room["name"], "class": o["class"],
                  "rect": [round(mn[0] - m, 3), round(mn[1] - m, 3),
                           round(mx[0] + m, 3), round(mx[1] + m, 3)],
                  "center": [round(v, 4) for v in o["centroid"]]}
            if p in xform_of:
                fl["xform"] = xform_of[p]
            slots.append(fl)
            path_to_slot[p] = f"floor_{fi}"
            fi += 1

    # surfaces last: they reference an already-emitted support by slot id
    si = 0
    for o in object_targets:
        p = o["path"]
        if roles.get(p) != "surface":
            continue
        on = path_to_slot.get(support_of.get(p))
        if on is None:
            continue                          # support is a fixture -> skip
        _reg(o)
        sf = {"id": f"surf_{si}", "type": "surface", "on": on,
              "class": o["class"], "center": [round(v, 4) for v in o["centroid"]]}
        if p in xform_of:
            sf["xform"] = xform_of[p]
        slots.append(sf)
        si += 1

    return {
        "floor_z": floor_z,
        "rooms": [{"name": r["name"], "xmin": round(r["xmin"], 3),
                   "xmax": round(r["xmax"], 3), "ymin": round(r["ymin"], 3),
                   "ymax": round(r["ymax"], 3)} for r in rooms],
        "slots": slots,
        "assets": assets,
    }


def _emit_wall(o, room, floor_z, slots, reg, path_to_slot, sid, xform_of=None):
    from layout import engine as pdr
    reg(o)
    edge, run, _ = pdr._nearest_edge(room, o["aabb"])
    axis, side = edge
    lo, hi = room[f"{run}min"], room[f"{run}max"]
    c = o["centroid"]
    t = round((c[_AXIS_I[run]] - lo) / (hi - lo), 4) if hi > lo else 0.5
    w = {"id": sid, "type": "wall", "room": room["name"],
         "edge": [axis, side], "t": t,
         "z": round(o["aabb"][0][2] - floor_z, 3), "class": o["class"],
         "center": [round(v, 4) for v in c]}
    if xform_of and o["path"] in xform_of:
        w["xform"] = xform_of[o["path"]]
    slots.append(w)
    path_to_slot[o["path"]] = sid


def to_placement_inputs(targets, rooms, floor_z, wall_aabbs=None):
    """Same dict shape replicator_dataset.py dumps as _placement_inputs.json, so
    a populated layout is a drop-in seed for placement_dr.generate()."""
    return {
        "objects": [{k: t[k] for k in ("path", "class", "centroid", "aabb",
                                        "size", "translate", "room")}
                    for t in targets],
        "rooms": rooms,
        "floor_z": floor_z,
        "wall_aabbs": wall_aabbs or [],
    }


# ------------------------------------------------------------------- demo ----
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Populate a slot spec and validate it.")
    p.add_argument("spec", help="path to a slot layout JSON")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jitter", type=int, default=0,
                   help="also run placement_dr.generate for N frames as a smoke test")
    args = p.parse_args()

    spec = load_spec(args.spec)
    targets, rooms, floor_z, dropped = populate(spec, seed=args.seed)
    print(f"[slot] populated {len(targets)} objects into {len(rooms)} room(s) "
          f"(floor_z={floor_z})")
    by_cls = {}
    for t in targets:
        by_cls[t["class"]] = by_cls.get(t["class"], 0) + 1
    for c, n in sorted(by_cls.items()):
        print(f"   {n:2d} x {c}")
    for d in dropped:
        print(f"[slot] DROPPED: {d}")

    hits = check_overlaps(targets, floor_z)
    if hits:
        print(f"[slot] {len(hits)} seed overlap(s):")
        for h in hits:
            print(f"   {h}")
    else:
        print("[slot] seed layout is collision-free")

    if args.jitter:
        from layout import engine as placement_dr
        seq = placement_dr.generate(targets, rooms, floor_z,
                                    n_frames=args.jitter, seed=args.seed,
                                    verbose=True)
        moved = sum(1 for s in seq.values() if len(set(s)) > 1)
        print(f"[slot] placement_dr jittered {moved}/{len(seq)} objects across "
              f"{args.jitter} frames")
