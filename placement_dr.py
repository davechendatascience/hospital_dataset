"""Hierarchical, geometry-aware placement randomization for the ward renderer (v2).

v1 failed in two visible ways: wall-mounted items slid into EACH OTHER (no
collision among them), and furniture could drift off its wall. v2 rebuilds the
policy as a room-level hierarchy:

  ROOM (ward / bathroom / frontroom -- objects never change room)
   |- BED BAY(s): hospital_bed + its satellites (overbed/bedside table, IV
   |    pole) + the wall-equipment groups mounted ABOVE the bed (monitor, gas
   |    manifold, flowmeter, suction, wall phone). The whole bay slides as ONE
   |    rigid unit ALONG the headwall (1D), staying flush to the wall, avoiding
   |    doors, fixtures and other wall groups. This mirrors how real wards
   |    differ: the bed bay shifts along the wall, its equipment with it.
   |- WALL GROUPS: remaining wall equipment, clustered by attachment (XY gap
   |    <= link_gap), slides along ITS wall only (the perpendicular coordinate
   |    is never touched, so items stay exactly ON the wall plane) with 1D
   |    interval collision against every other occupant of that wall.
   |- FLOOR furniture (chair, stool, scale, bins, lone tables): local 2D
   |    re-placement around the original spot, collision-checked against
   |    walls, door keep-outs, fixtures, bays and each other.
   |- SURFACE items (remote, thermometer, packages, ...): support detected by
   |    geometry (resting on some object's top face), re-scattered ON that
   |    support's top (inset by the item's size, pairwise non-overlap) and
   |    RIDING the support's movement.
   |- FIXTURES (toilet, sink, shower, doors, windows, curtains, switches,
        vents, ...) and any unrecognized class: never move.

Collision is Z-AWARE: a blocker only conflicts with a move if their footprints
overlap in XY *and* their height ranges intersect. Without this, overhead
fixtures (bed curtain rail, air vents) would freeze the furniture below them --
in reality a bed slides under its curtain rail, an overbed table slides under
the wall phone. Door keep-outs span full height on purpose.

Every randomized move is rejection-sampled against the constraint set and
falls back to the ORIGINAL pose (always valid by construction), so a frame can
degrade to less variation but never to an implausible arrangement. Collision
bookkeeping is conservative: while placing item N, the not-yet-placed items
count as obstacles at their ORIGINAL spots, so "everyone stays put" is always
a consistent solution and the system cannot deadlock into overlap.

Pure Python (no Isaac imports) so the whole policy is unit-testable. The
caller provides object_targets augmented with 'translate' (original world
translate) and 'room', the room XY AABBs, floor_z, and optionally the wall
mesh AABBs for clip checks. Returns {prim_path: [(x, y, z) per frame]}.
Orientation is never authored, so nothing can tip or yaw implausibly.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

BED_CLASSES = {"hospital_bed"}
SATELLITE_CLASSES = {"overbed_table", "bedside_table"}
WALL_CLASSES = {"bedside_monitor", "gas_manifold", "oxygen_flowmeter",
                "suction_jar", "suction_knob", "telephone", "tissue_dispenser",
                "sanitizer", "hook"}
FLOOR_CLASSES = {"companion_chair", "stool", "weight_scale", "waste_bin",
                 "medical_waste_container", "soiled_linen_bin", "iv_pole",
                 "paperbox"}
SURFACE_CLASSES = {"remote_control", "ear_thermometer", "medical_package",
                   "gauze", "medical_gloves", "syringe", "stethoscope",
                   "alcohol_spray_bottle", "sanitizer", "paperbox"}

_SUPPORT_MIN_AREA = 0.08   # m^2 XY footprint to qualify as a support surface
_ELEVATED = 0.35           # z_bottom this far above floor => wall-mounted


# ---------------------------------------------------------------- geometry --
def _foot(aabb):
    (mn, mx) = aabb
    return [mn[0], mn[1], mx[0], mx[1]]


def _blk(aabb):
    """Z-aware blocker: (XY footprint rect, z_lo, z_hi)."""
    (mn, mx) = aabb
    return ([mn[0], mn[1], mx[0], mx[1]], mn[2], mx[2])


def _rect_overlap(a, b, margin=0.0):
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or
                a[3] + margin <= b[1] or b[3] + margin <= a[1])


def _rect_shift(r, dx, dy):
    return [r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy]


def _blk_shift(b, dx, dy):
    return (_rect_shift(b[0], dx, dy), b[1], b[2])


def _blk_hit(a, b, margin=0.0):
    """XY overlap AND height-range intersection (touching in Z = no conflict)."""
    return (_rect_overlap(a[0], b[0], margin)
            and a[1] < b[2] - 1e-9 and b[1] < a[2] - 1e-9)


def _xy_gap(a, b):
    fa, fb = _foot(a), _foot(b)
    dx = max(fb[0] - fa[2], fa[0] - fb[2], 0.0)
    dy = max(fb[1] - fa[3], fa[1] - fb[3], 0.0)
    return math.hypot(dx, dy)


def _iv_overlap(a, b, margin=0.0):
    return a[0] < b[1] + margin and b[0] < a[1] + margin


def _zov(a0, a1, b0, b1):
    return a0 < b1 - 1e-9 and b0 < a1 - 1e-9


def _interval(aabb, axis):
    i = 0 if axis == "x" else 1
    return (aabb[0][i], aabb[1][i])


def _nearest_edge(room, aabb):
    """Nearest room edge to this AABB -> (edge, slide_axis, distance). The
    edge identifies the wall; sliding happens along the wall's direction, and
    the perpendicular coordinate is never modified (stays flush)."""
    (mn, mx) = aabb
    cands = [
        (abs(mn[0] - room["xmin"]), ("x", "min")),
        (abs(room["xmax"] - mx[0]), ("x", "max")),
        (abs(mn[1] - room["ymin"]), ("y", "min")),
        (abs(room["ymax"] - mx[1]), ("y", "max")),
    ]
    dist, edge = min(cands, key=lambda c: c[0])
    return edge, ("y" if edge[0] == "x" else "x"), dist


def _find_support(item, candidates):
    """The object whose top face this item rests on (z_bottom within
    [-6cm, +12cm] of the top, XY centre inside the footprint), preferring the
    highest such top. None if the item isn't resting on a labeled object."""
    (imn, imx) = item["aabb"]
    icx, icy = 0.5 * (imn[0] + imx[0]), 0.5 * (imn[1] + imx[1])
    best, best_top = None, None
    for c in candidates:
        if c["path"] == item["path"]:
            continue
        (cmn, cmx) = c["aabb"]
        if (cmx[0] - cmn[0]) * (cmx[1] - cmn[1]) < _SUPPORT_MIN_AREA:
            continue
        top = cmx[2]
        if not (top - 0.06 <= imn[2] <= top + 0.12):
            continue
        if not (cmn[0] - 0.05 <= icx <= cmx[0] + 0.05 and
                cmn[1] - 0.05 <= icy <= cmx[1] + 0.05):
            continue
        if best is None or top > best_top:
            best, best_top = c, top
    return best


def _near(a, b, gap):
    return _xy_gap(a["aabb"], b["aabb"]) <= gap


def _components(items, link_gap):
    """Connected components of items by XY AABB gap <= link_gap."""
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _xy_gap(items[i]["aabb"], items[j]["aabb"]) <= link_gap:
                parent[find(i)] = find(j)
    comp = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(items[i])
    return list(comp.values())


# ------------------------------------------------------------------ policy --
def generate(object_targets, rooms, floor_z, n_frames, seed=1000,
             max_shift=0.8, wall_slide=0.2, sat_slide=0.5, attempts=40,
             wall_boxes=None, door_clearance=0.5, link_gap=0.25, verbose=True):
    """-> {path: [(x, y, z)] * n_frames}. Each target must carry 'translate'
    (original world translate) and 'room' (room name from the renderer)."""
    rng = random.Random(seed)
    objs = sorted(object_targets, key=lambda o: o["path"])
    for o in objs:
        if "translate" not in o:
            raise ValueError(f"object_targets entry missing 'translate': {o['path']}")
    by_room = {r["name"]: r for r in rooms}
    obj_by_path = {o["path"]: o for o in objs}
    pos0 = {o["path"]: tuple(o["translate"]) for o in objs}

    # ---- role resolution (class + geometry; unknown/unsure -> fixture) ----
    beds = [o for o in objs
            if o["class"] in BED_CLASSES and o.get("room") in by_room]
    supports = [o for o in objs
                if (o["aabb"][1][0] - o["aabb"][0][0]) *
                   (o["aabb"][1][1] - o["aabb"][0][1]) >= _SUPPORT_MIN_AREA]
    roles, support_of = {}, {}
    for o in objs:
        path, cls = o["path"], o["class"]
        zb = o["aabb"][0][2] - floor_z
        if o.get("room") not in by_room:
            roles[path] = "fixture"
            continue
        if cls in BED_CLASSES:
            roles[path] = "bed"
            continue
        if cls in SURFACE_CLASSES:
            sup = _find_support(o, supports)
            if sup is not None:
                roles[path] = "surface"
                support_of[path] = sup["path"]
                continue
        if cls in SATELLITE_CLASSES:
            roles[path] = ("satellite"
                           if any(_near(o, b, 2.0) and b.get("room") == o.get("room")
                                  for b in beds) else "floor")
            continue
        if cls == "iv_pole" and any(_near(o, b, 1.5) and
                                    b.get("room") == o.get("room") for b in beds):
            roles[path] = "satellite"
            continue
        if cls in WALL_CLASSES and zb > _ELEVATED:
            roles[path] = "wall"
            continue
        if cls in (FLOOR_CLASSES | WALL_CLASSES):
            roles[path] = "floor" if zb <= _ELEVATED else "fixture"
            continue
        roles[path] = "fixture"

    # satellites attach to their nearest bed
    sat_bed = {}
    for o in objs:
        if roles[o["path"]] != "satellite":
            continue
        cands = [b for b in beds if b.get("room") == o.get("room")]
        if not cands:
            roles[o["path"]] = "floor"
            continue
        b = min(cands, key=lambda b: _xy_gap(o["aabb"], b["aabb"]))
        sat_bed[o["path"]] = b["path"]

    # ---- wall groups: attached components per (room, edge) wall ----
    wall_key = {}
    for o in objs:
        if roles[o["path"]] == "wall":
            edge, axis, _ = _nearest_edge(by_room[o["room"]], o["aabb"])
            wall_key[o["path"]] = (o["room"], edge, axis)
    by_key = defaultdict(list)
    for o in objs:
        if o["path"] in wall_key:
            by_key[wall_key[o["path"]]].append(o)
    wall_groups = []
    for key, items in by_key.items():
        for members in _components(items, link_gap):
            axis = key[2]
            iv = (min(_interval(m["aabb"], axis)[0] for m in members),
                  max(_interval(m["aabb"], axis)[1] for m in members))
            z0 = min(m["aabb"][0][2] for m in members)
            z1 = max(m["aabb"][1][2] for m in members)
            wall_groups.append({"key": key, "iv": iv, "z": (z0, z1),
                                "in_bay": False,
                                "members": [m["path"] for m in members]})

    # ---- bed bays: bed + satellites + overhead wall groups, 1D along wall --
    bays = []
    for b in beds:
        room = by_room[b["room"]]
        edge, axis, _ = _nearest_edge(room, b["aabb"])
        key = (b["room"], edge, axis)
        members = [b["path"]] + [p for p, bp in sat_bed.items() if bp == b["path"]]
        biv = _interval(b["aabb"], axis)
        biv_inf = (biv[0] - 0.6, biv[1] + 0.6)
        for g in wall_groups:
            if not g["in_bay"] and g["key"] == key and _iv_overlap(g["iv"], biv_inf):
                g["in_bay"] = True
                members += g["members"]
        iv = (min(_interval(obj_by_path[p]["aabb"], axis)[0] for p in members),
              max(_interval(obj_by_path[p]["aabb"], axis)[1] for p in members))
        # collision uses INDIVIDUAL member blockers (union covers the empty
        # floor BETWEEN members and falsely blocks everything in that span)
        blks = [_blk(obj_by_path[p]["aabb"]) for p in members]
        m_ivs = [(_interval(obj_by_path[p]["aabb"], axis),
                  obj_by_path[p]["aabb"][0][2], obj_by_path[p]["aabb"][1][2])
                 for p in members]
        bays.append({"key": key, "axis": axis, "members": members,
                     "iv": iv, "blks": blks, "m_ivs": m_ivs,
                     "z": (min(b[1] for b in blks), max(b[2] for b in blks))})

    # ---- static obstacles (Z-aware blockers) ----
    fixtures = [o for o in objs if roles[o["path"]] == "fixture"]
    static_blks = [_blk(o["aabb"]) for o in fixtures]
    for o in fixtures:
        if o["class"] == "door":            # full-height keep-out around doors
            f = _foot(o["aabb"])
            static_blks.append(([f[0] - door_clearance, f[1] - door_clearance,
                                 f[2] + door_clearance, f[3] + door_clearance],
                                floor_z, floor_z + 2.3))
    all_keys = {g["key"] for g in wall_groups} | {b["key"] for b in bays}
    wall_static = defaultdict(list)         # key -> [(interval, z0, z1)]
    for key in all_keys:
        rname, edge, axis = key
        room = by_room[rname]
        for o in fixtures:
            if o.get("room") != rname:
                continue
            _e, _a, dist = _nearest_edge(room, o["aabb"])
            if _e == edge and dist <= 0.6:
                lo, hi = _interval(o["aabb"], axis)
                pad = 0.4 if o["class"] == "door" else 0.05
                wall_static[key].append(((lo - pad, hi + pad),
                                         o["aabb"][0][2], o["aabb"][1][2]))

    def axis_span(room, axis):
        return ((room["xmin"], room["xmax"]) if axis == "x"
                else (room["ymin"], room["ymax"]))

    floor_movers = sorted((o for o in objs if roles[o["path"]] == "floor"),
                          key=lambda o: -(o["size"][0] * o["size"][1]))
    surface_items = [o for o in objs if roles[o["path"]] == "surface"]
    orig_blks = {o["path"]: _blk(o["aabb"]) for o in floor_movers}

    # wall-mesh clip checks: a NEW pose may only touch walls the ORIGINAL pose
    # already touched (a bed flush to its headwall keeps touching it while it
    # slides; it may not run into a perpendicular wall it never touched).
    wb_blks = [_blk(w) for w in (wall_boxes or [])]

    def _near_walls(blk, reach):
        ex = {i for i, w in enumerate(wb_blks) if _blk_hit(blk, w, margin=0.02)}
        r = blk[0]
        grown = [r[0] - reach, r[1] - reach, r[2] + reach, r[3] + reach]
        return [w for i, w in enumerate(wb_blks)
                if i not in ex and _rect_overlap(grown, w[0])
                and _zov(blk[1], blk[2], w[1], w[2])]

    bay_walls = [[(bi_f, w) for bi_f, blk in enumerate(b["blks"])
                  for w in _near_walls(blk, max_shift + 0.05)] for b in bays]

    # per-member 1D wall checks for each bay: member interval vs the wall's
    # OTHER occupants, only where their height ranges intersect, and with
    # pairs that already coexist in the original layout GRANDFATHERED (the
    # air vent was always above the bed; that is not a new collision).
    for b in bays:
        occ = list(wall_static[b["key"]])
        occ += [(g["iv"], g["z"][0], g["z"][1]) for g in wall_groups
                if not g["in_bay"] and g["key"] == b["key"]]
        stat = static_blks + list(orig_blks.values())
        reach = max_shift + sat_slide + 0.1
        pairs_m = []
        for mi, mb in enumerate(b["blks"]):
            miv, mz0, mz1 = b["m_ivs"][mi]
            pairs = []
            for (qiv, qz0, qz1) in occ:
                if not _zov(mz0, mz1, qz0, qz1):
                    continue
                if _iv_overlap(miv, qiv, margin=0.05):
                    continue                      # grandfathered coexistence
                pairs.append((miv, qiv))
            # static 3D blockers reduce to slide-axis interval pairs too (the
            # perpendicular coordinate never changes during a slide): keep
            # only blockers that overlap the member in Z and in the
            # perpendicular axis, lie within slide reach, and do NOT already
            # coexist at s=0.
            r = mb[0]
            for q in stat:
                if not _zov(mb[1], mb[2], q[1], q[2]):
                    continue
                g = q[0]
                if b["axis"] == "x":
                    if g[3] + 0.03 <= r[1] or r[3] + 0.03 <= g[1]:
                        continue              # perpendicular never overlaps
                    a, qiv = (r[0], r[2]), (g[0], g[2])
                else:
                    if g[2] + 0.03 <= r[0] or r[2] + 0.03 <= g[0]:
                        continue
                    a, qiv = (r[1], r[3]), (g[1], g[3])
                if qiv[1] + 0.03 < a[0] - reach or qiv[0] - 0.03 > a[1] + reach:
                    continue                  # beyond slide reach
                if _iv_overlap(a, qiv, margin=0.03):
                    continue                  # grandfathered coexistence
                pairs.append((a, qiv))
            pairs_m.append(pairs)
        b["pairs_m"] = pairs_m
    floor_walls = {o["path"]: _near_walls(orig_blks[o["path"]], max_shift + 0.05)
                   for o in floor_movers}

    if verbose:
        hist = defaultdict(int)
        for r in roles.values():
            hist[r] += 1
        print("[placement-dr] roles: " +
              ", ".join(f"{k}={v}" for k, v in sorted(hist.items())))
        for b in bays:
            print(f"[placement-dr] bay @ {b['key'][0]} wall {b['key'][1]}: "
                  f"{len(b['members'])} members slide along {b['axis']}")
        print(f"[placement-dr] independent wall groups: "
              f"{sum(1 for g in wall_groups if not g['in_bay'])}, "
              f"floor movers: {len(floor_movers)}, "
              f"surface riders: {len(surface_items)}")

    # ------------------------------------------------------------ frames --
    out = {o["path"]: [] for o in objs}
    riders_of = defaultdict(list)
    for o in surface_items:
        riders_of[support_of[o["path"]]].append(o)

    for _f in range(n_frames):
        delta = {}

        # 1) bed bays -- rigid 1D slide along the headwall
        placed_bays = {}
        for bi, b in enumerate(bays):
            key = b["key"]
            room = by_room[key[0]]
            lo, hi = axis_span(room, b["axis"])
            lo, hi = min(lo, b["iv"][0]), max(hi, b["iv"][1])
            dyn = []                       # other bays sharing this wall (1D)
            dyn_blks = []                  # other bays' member blockers (3D)
            for bj, ob in enumerate(bays):
                if bj == bi:
                    continue
                if ob["key"] == key:
                    dyn.append(placed_bays[bj][1] if bj in placed_bays
                               else ob["iv"])
                if bj in placed_bays:
                    s_j = placed_bays[bj][0]
                    dxy_j = (s_j, 0.0) if ob["axis"] == "x" else (0.0, s_j)
                    dyn_blks += [_blk_shift(q, *dxy_j) for q in ob["blks"]]
                else:
                    dyn_blks += ob["blks"]
            s = 0.0
            for _ in range(attempts):
                cand = rng.uniform(-max_shift, max_shift)
                iv = (b["iv"][0] + cand, b["iv"][1] + cand)
                if iv[0] < lo or iv[1] > hi:
                    continue
                if any(_iv_overlap((a[0] + cand, a[1] + cand), q, margin=0.03)
                       for pm in b["pairs_m"] for a, q in pm):
                    continue
                if any(_iv_overlap(iv, q, margin=0.05) for q in dyn):
                    continue
                dxy = (cand, 0.0) if b["axis"] == "x" else (0.0, cand)
                moved = [_blk_shift(q, *dxy) for q in b["blks"]]
                if dyn_blks and any(_blk_hit(mq, q, margin=0.03)
                                    for mq in moved for q in dyn_blks):
                    continue
                if any(_rect_overlap(moved[fi][0], w[0])
                       for fi, w in bay_walls[bi]):
                    continue
                s = cand
                break
            placed_bays[bi] = (s, (b["iv"][0] + s, b["iv"][1] + s))
            dxy = (s, 0.0) if b["axis"] == "x" else (0.0, s)
            for p in b["members"]:
                delta[p] = dxy

            # satellites additionally jitter ALONG the bed (an overbed table's
            # position down the bed length varies a lot in reality), staying
            # at the bed, respecting the same non-grandfathered pairs, and
            # never NEWLY overlapping another bay member (overbed-table-over-
            # bed is grandfathered; bedside-table-into-bed is not).
            bediv = b["m_ivs"][0][0]            # members[0] is the bed
            cur = {mi2: _blk_shift(b["blks"][mi2], *dxy)
                   for mi2 in range(len(b["members"]))}
            for mi, p in enumerate(b["members"]):
                if roles[p] != "satellite":
                    continue
                mb = b["blks"][mi]
                miv = b["m_ivs"][mi][0]
                t = 0.0
                for _ in range(20):
                    c2 = rng.uniform(-sat_slide, sat_slide)
                    tot = s + c2
                    niv = (miv[0] + tot, miv[1] + tot)
                    if niv[0] < bediv[0] + s - 0.45 or \
                       niv[1] > bediv[1] + s + 0.45:
                        continue              # must stay AT the bed
                    if any(_iv_overlap((a[0] + tot, a[1] + tot), q,
                                       margin=0.03)
                           for a, q in b["pairs_m"][mi]):
                        continue
                    dxy2 = (tot, 0.0) if b["axis"] == "x" else (0.0, tot)
                    nb = _blk_shift(mb, *dxy2)
                    if any(_blk_hit(nb, cur[mj], margin=0.02)
                           and not _blk_hit(mb, b["blks"][mj], margin=0.02)
                           for mj in range(len(b["members"])) if mj != mi):
                        continue
                    t = c2
                    break
                dxy2 = ((s + t, 0.0) if b["axis"] == "x" else (0.0, s + t))
                cur[mi] = _blk_shift(mb, *dxy2)
                delta[p] = dxy2

        # 2) independent wall groups -- 1D slide, never overlapping the wall
        g_now = {}
        for gi, g in enumerate(wall_groups):
            if g["in_bay"]:
                continue
            key = g["key"]
            room = by_room[key[0]]
            lo, hi = axis_span(room, key[2])
            lo, hi = min(lo, g["iv"][0]), max(hi, g["iv"][1])
            obstacles = [iv for (iv, qz0, qz1) in wall_static[key]
                         if _zov(g["z"][0], g["z"][1], qz0, qz1)]
            for gj, g2 in enumerate(wall_groups):
                if gj == gi or g2["in_bay"] or g2["key"] != key:
                    continue
                if _zov(g["z"][0], g["z"][1], g2["z"][0], g2["z"][1]):
                    obstacles.append(g_now.get(gj, g2["iv"]))
            for bj, ob in enumerate(bays):
                if ob["key"] == key:
                    obstacles.append(placed_bays[bj][1])
            s = 0.0
            for _ in range(attempts):
                cand = rng.uniform(-wall_slide, wall_slide)
                iv = (g["iv"][0] + cand, g["iv"][1] + cand)
                if iv[0] < lo or iv[1] > hi:
                    continue
                if any(_iv_overlap(iv, q, margin=0.03) for q in obstacles):
                    continue
                s = cand
                break
            g_now[gi] = (g["iv"][0] + s, g["iv"][1] + s)
            dxy = (s, 0.0) if key[2] == "x" else (0.0, s)
            for p in g["members"]:
                delta[p] = dxy

        # 3) floor furniture -- local 2D re-place with full collision
        placed_blks = list(static_blks)
        for b in bays:                      # members at their FINAL deltas
            for mi, p in enumerate(b["members"]):
                placed_blks.append(_blk_shift(b["blks"][mi], *delta[p]))
        pending = dict(orig_blks)
        for o in floor_movers:
            p = o["path"]
            room = by_room[o["room"]]
            del pending[p]
            f0, z0, z1 = orig_blks[p]
            rxlo, rxhi = min(room["xmin"], f0[0]), max(room["xmax"], f0[2])
            rylo, ryhi = min(room["ymin"], f0[1]), max(room["ymax"], f0[3])
            obstacles = placed_blks + list(pending.values())
            dx = dy = 0.0
            for _ in range(attempts):
                cdx = rng.uniform(-max_shift, max_shift)
                cdy = rng.uniform(-max_shift, max_shift)
                foot = _rect_shift(f0, cdx, cdy)
                if foot[0] < rxlo or foot[2] > rxhi or \
                   foot[1] < rylo or foot[3] > ryhi:
                    continue
                cb = (foot, z0, z1)
                if any(_blk_hit(cb, q, margin=0.03) for q in obstacles):
                    continue
                if any(_rect_overlap(foot, w[0]) for w in floor_walls[p]):
                    continue
                dx, dy = cdx, cdy
                break
            placed_blks.append((_rect_shift(f0, dx, dy), z0, z1))
            delta[p] = (dx, dy)

        # 4) surface items -- ride the support + re-scatter on its top
        for sup_path, items in riders_of.items():
            sup = obj_by_path[sup_path]
            sdx, sdy = delta.get(sup_path, (0.0, 0.0))
            (smn, smx) = sup["aabb"]
            inset = 0.15 if sup["class"] in BED_CLASSES else 0.02
            placed_r = []
            # not-yet-placed riders block at their ORIGINAL spots, so a rider
            # falling back to its original never collides with a later one
            pending_r = {o["path"]: _foot(o["aabb"]) for o in items}
            for o in items:
                p = o["path"]
                del pending_r[p]
                hx = 0.5 * (o["aabb"][1][0] - o["aabb"][0][0])
                hy = 0.5 * (o["aabb"][1][1] - o["aabb"][0][1])
                ccx = 0.5 * (o["aabb"][0][0] + o["aabb"][1][0])
                ccy = 0.5 * (o["aabb"][0][1] + o["aabb"][1][1])
                x0, x1 = smn[0] + hx + inset, smx[0] - hx - inset
                y0, y1 = smn[1] + hy + inset, smx[1] - hy - inset
                ncx, ncy = ccx, ccy
                if x1 > x0 and y1 > y0:
                    for _ in range(20):
                        tx, ty = rng.uniform(x0, x1), rng.uniform(y0, y1)
                        rect = [tx - hx, ty - hy, tx + hx, ty + hy]
                        if any(_rect_overlap(rect, q, margin=0.01)
                               for q in placed_r + list(pending_r.values())):
                            continue
                        ncx, ncy = tx, ty
                        break
                placed_r.append([ncx - hx, ncy - hy, ncx + hx, ncy + hy])
                delta[p] = (ncx - ccx + sdx, ncy - ccy + sdy)

        # 5) emit this frame's world translates
        for o in objs:
            ox, oy, oz = pos0[o["path"]]
            dx, dy = delta.get(o["path"], (0.0, 0.0))
            out[o["path"]].append((ox + dx, oy + dy, oz))

    return out
