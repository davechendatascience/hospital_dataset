"""Hierarchical, geometry-aware placement randomization for the ward renderer (v2).

v1 failed in two visible ways: wall-mounted items slid into EACH OTHER (no
collision among them), and furniture could drift off its wall. v2 rebuilds the
policy as a room-level hierarchy:

  ROOM (ward / bathroom / frontroom -- objects never change room)
   |- BED BAY(s): hospital_bed + its satellites (overbed/bedside table, IV
   |    pole) + the wall-equipment groups mounted ABOVE the bed (monitor, gas
   |    manifold, flowmeter, suction, wall phone). The whole bay slides as ONE
   |    rigid unit ALONG the headwall (1D), staying flush to the wall, avoiding
   |    doors, fixtures and other wall groups. Satellites additionally jitter
   |    ALONG the bed (member-aware, grandfathered collision).
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
the wall phone. Door keep-outs span full height on purpose. Pairs that already
coexist in the ORIGINAL layout are GRANDFATHERED (the air vent was always
above the bed; that is not a new collision).

Every randomized move is rejection-sampled against the constraint set and
falls back to the ORIGINAL pose (always valid by construction), so a frame can
degrade to less variation but never to an implausible arrangement. Collision
bookkeeping is conservative: while placing item N, the not-yet-placed items
count as obstacles at their ORIGINAL spots, so "everyone stays put" is always
a consistent solution and the system cannot deadlock into overlap.

Two entry points share ALL of the above machinery:
  * generate(...)      -- engine SAMPLES every degree of freedom (rng)
  * apply_layout(...)  -- a caller (e.g. an LLM layout designer) PROPOSES the
                          degrees of freedom; the engine validates each one and
                          falls back to the original pose for any that violate
                          constraints, returning what was rejected and why.

Pure Python (no Isaac imports) so the whole policy is unit-testable. The
caller provides object_targets augmented with 'translate' (original world
translate) and 'room', the room XY AABBs, floor_z, and optionally the wall
mesh AABBs for clip checks. Positions are {prim_path: (x, y, z)}; orientation
is never authored, so nothing can tip or yaw implausibly.
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


def _seg_rect(p0, p1, rect, margin=0.0):
    """Liang-Barsky: does the XY segment p0->p1 intersect the AABB rect
    [xmin, ymin, xmax, ymax]? Used to keep a re-placed item on its own side of
    the walls (a move whose path crosses a wall would land in another room)."""
    xmin, ymin = rect[0] - margin, rect[1] - margin
    xmax, ymax = rect[2] + margin, rect[3] + margin
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - xmin), (dx, xmax - p0[0]),
                 (-dy, p0[1] - ymin), (dy, ymax - p0[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False          # parallel and outside this slab
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
    return t0 <= t1


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


def _wall_run_extent(group, wall_boxes, obj_by_path):
    """The run-axis (slide-direction) span of the real wall mesh a wall group is
    attached to. -> (lo, hi) or None if no walls / none match. Picks the wall
    mesh that straddles the group's pinned coordinate and overlaps its run
    interval, nearest in the pinned axis."""
    if not wall_boxes:
        return None
    edge, run = group["key"][1], group["key"][2]
    pin_i = 0 if edge[0] == "x" else 1
    run_i = 0 if run == "x" else 1
    members = [obj_by_path[p]["aabb"] for p in group["members"]]
    pin_c = 0.5 * (min(a[0][pin_i] for a in members) +
                   max(a[1][pin_i] for a in members))
    run_lo, run_hi = group["iv"]
    best, best_d = None, 1e9
    for (mn, mx) in wall_boxes:
        if not (mn[pin_i] - 0.4 <= pin_c <= mx[pin_i] + 0.4):
            continue                      # wall doesn't pass through the group
        if mx[run_i] < run_lo - 0.1 or mn[run_i] > run_hi + 0.1:
            continue                      # wall doesn't span the group's run
        d = abs(0.5 * (mn[pin_i] + mx[pin_i]) - pin_c)
        if d < best_d:
            best_d, best = d, (float(mn[run_i]), float(mx[run_i]))
    return best


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


# ------------------------------------------------------------- scene build --
def _build_ctx(object_targets, rooms, floor_z, max_shift=0.8, wall_slide=0.2,
               sat_slide=0.5, wall_boxes=None, door_clearance=0.5,
               link_gap=0.25, global_frac=0.0, verbose=True):
    """Resolve roles, groups, bays and every precomputed constraint table.
    The returned ctx is consumed by _solve() (one call per frame/layout)."""
    objs = sorted(object_targets, key=lambda o: o["path"])
    for o in objs:
        if "translate" not in o:
            raise ValueError(f"object_targets entry missing 'translate': {o['path']}")
    by_room = {r["name"]: r for r in rooms}
    obj_by_path = {o["path"]: o for o in objs}
    pos0 = {o["path"]: tuple(o["translate"]) for o in objs}

    # ---- role resolution: the support mode is DECLARED, not guessed from z ----
    # Each object's support comes from its per-instance `affordance` (set by the
    # dump from the authored scene) or, failing that, the explicit class table in
    # placement_affordances. The old z-threshold guess misread low wall gear as
    # floor; support is now authoritative. "floor" support is then refined into
    # bed / satellite / floor for MOTION grouping (a bay slides as a unit).
    from layout import affordances as _paff
    beds = [o for o in objs
            if o["class"] in BED_CLASSES and o.get("room") in by_room]
    supports = [o for o in objs
                if (o["aabb"][1][0] - o["aabb"][0][0]) *
                   (o["aabb"][1][1] - o["aabb"][0][1]) >= _SUPPORT_MIN_AREA]
    roles, support_of = {}, {}
    for o in objs:
        path, cls = o["path"], o["class"]
        if o.get("room") not in by_room:
            roles[path] = "fixture"
            continue
        support = o.get("affordance") or _paff.support_of(cls)
        if support == "fixed":
            roles[path] = "fixture"
            continue
        if support == "wall":
            roles[path] = "wall"
            continue
        if support == "surface":
            sup = _find_support(o, supports)
            roles[path] = "surface" if sup is not None else "floor"
            if sup is not None:
                support_of[path] = sup["path"]
            continue
        # support == "floor": refine into bed / satellite / floor.
        # A SURFACE-class item declared 'floor' but sitting ELEVATED actually
        # rests on a surface the dump couldn't detect (an unmodeled counter /
        # shelf); keep it put rather than dropping it to the floor or sliding it.
        if _paff.support_of(cls) == "surface" and \
           (o["aabb"][0][2] - floor_z) > _ELEVATED:
            roles[path] = "fixture"
            continue
        if cls in BED_CLASSES:
            roles[path] = "bed"
            continue
        if (cls in SATELLITE_CLASSES or cls == "iv_pole") and \
           any(_near(o, b, 2.0) and b.get("room") == o.get("room") for b in beds):
            roles[path] = "satellite"
            continue
        roles[path] = "floor"

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

    # ---- wall groups: CONNECTED clusters first, then ONE wall per cluster ----
    # Keying by nearest edge BEFORE clustering splits a unit that straddles a
    # corner: a suction jar + knob 13 cm apart can land on different edges and
    # then slide INTO each other (they're attached in reality). Clustering by
    # connectivity first, then choosing one edge for the whole cluster's union
    # AABB, keeps such a unit a single rigid group.
    wall_by_room = defaultdict(list)
    for o in objs:
        if roles[o["path"]] == "wall":
            wall_by_room[o["room"]].append(o)
    wall_groups = []
    for rname, items in wall_by_room.items():
        room = by_room[rname]
        for members in _components(items, link_gap):
            # the wall is set by the cluster's DOMINANT (largest-footprint)
            # member -- a union AABB of an elongated cluster can read closer to a
            # perpendicular wall and flip the whole group onto the wrong one.
            dom = max(members, key=lambda m: (m["aabb"][1][0] - m["aabb"][0][0]) *
                      (m["aabb"][1][1] - m["aabb"][0][1]))
            edge, axis, _ = _nearest_edge(room, dom["aabb"])
            key = (rname, edge, axis)
            iv = (min(_interval(m["aabb"], axis)[0] for m in members),
                  max(_interval(m["aabb"], axis)[1] for m in members))
            z0 = min(m["aabb"][0][2] for m in members)
            z1 = max(m["aabb"][1][2] for m in members)
            wall_groups.append({"key": key, "iv": iv, "z": (z0, z1),
                                "in_bay": False,
                                "members": [m["path"] for m in members]})

    # real-wall extent per group: the run-axis span of the actual wall mesh the
    # group is against, so wall items randomize ALONG THEIR REAL WALL (not the
    # centroid-room edge, which can run past the wall into a corner / doorway).
    for g in wall_groups:
        g["wall_span"] = _wall_run_extent(g, wall_boxes, obj_by_path)

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
                     "z": (min(q[1] for q in blks), max(q[2] for q in blks))})

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
    floor_walls = {o["path"]: _near_walls(orig_blks[o["path"]], max_shift + 0.05)
                   for o in floor_movers}
    # for LONG-RANGE moves (global sampling / LLM proposals) the near-list is
    # not enough: keep all wall blocks + each mover's originally-touched set
    floor_wall_exempt = {}
    for o in floor_movers:
        blk = orig_blks[o["path"]]
        floor_wall_exempt[o["path"]] = {
            i for i, w in enumerate(wb_blks) if _blk_hit(blk, w, margin=0.02)}
    # GRANDFATHER: obstacles the ORIGINAL pose already conflicts with don't
    # count (the linen bin lives inside the door keep-out in the real ward --
    # "stay there" or "shift 10 cm" must not be rejected because of it)
    floor_gf = {}
    bay_member_blk = {}
    for b in bays:
        for mi, p in enumerate(b["members"]):
            bay_member_blk[p] = b["blks"][mi]
    for o in floor_movers:
        blk = orig_blks[o["path"]]
        gf = set()
        for si, q in enumerate(static_blks):
            if _blk_hit(blk, q, margin=0.03):
                gf.add(f"static:{si}")
        for p2, q in bay_member_blk.items():
            if p2 != o["path"] and _blk_hit(blk, q, margin=0.03):
                gf.add(p2)
        for o2 in floor_movers:
            if o2["path"] != o["path"] and \
               _blk_hit(blk, orig_blks[o2["path"]], margin=0.03):
                gf.add(o2["path"])
        floor_gf[o["path"]] = gf

    # per-member 1D wall + reduced-static checks for each bay, grandfathered
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
            # perpendicular coordinate never changes during a slide)
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

    riders_of = defaultdict(list)
    for o in surface_items:
        riders_of[support_of[o["path"]]].append(o)

    # candidate hosts for PLACEMENT RANDOMIZATION of surface items, from the
    # host-equivalence classes (paperbox -> on bed | bedside_table | overbed_
    # table). A host is ("on", host_path) or ("floor", room). Falls back to the
    # geometrically-detected support if the class declares no host group.
    from layout import affordances as _paff
    hosts_by_class = defaultdict(list)
    for o in objs:
        if _paff.provides_surface(o["class"]):
            hosts_by_class[o["class"]].append(o["path"])
    surface_hosts = {}
    for o in surface_items:
        cands = []
        for kind, sel in _paff.host_specs_for(o["class"]):
            if kind == "on":
                cands += [("on", hp) for hp in hosts_by_class.get(sel, [])]
            elif kind == "floor" and (sel in by_room or sel == "*"):
                cands.append(("floor", sel))
        if not cands and o["path"] in support_of:
            cands = [("on", support_of[o["path"]])]
        surface_hosts[o["path"]] = cands

    return {
        "objs": objs, "by_room": by_room, "obj_by_path": obj_by_path,
        "pos0": pos0, "roles": roles, "support_of": support_of,
        "wall_groups": wall_groups, "bays": bays,
        "static_blks": static_blks, "wall_static": wall_static,
        "floor_movers": floor_movers, "orig_blks": orig_blks,
        "bay_walls": bay_walls, "floor_walls": floor_walls,
        "wb_blks": wb_blks, "floor_wall_exempt": floor_wall_exempt,
        "floor_gf": floor_gf,
        "riders_of": riders_of, "surface_hosts": surface_hosts,
        "max_shift": max_shift, "wall_slide": wall_slide,
        "sat_slide": sat_slide, "global_frac": global_frac,
    }


def _axis_span(room, axis):
    return ((room["xmin"], room["xmax"]) if axis == "x"
            else (room["ymin"], room["ymax"]))


# ------------------------------------------------------------- one layout --
def _solve(ctx, rng, attempts=40, params=None):
    """Produce ONE collision-valid layout: {path: (x, y, z)}.

    params=None        -> every degree of freedom is rng-sampled (generate()).
    params={...}       -> proposed values are validated ONCE each and rejected
                          (-> original pose) if they violate constraints:
        {"bay_slides":  {bay_index: s},          # along the headwall
         "sat_slides":  {path: t},               # along the bed, extra to bay
         "wall_slides": {group_index: s},        # along the group's wall
         "floor":       {path: (x, y)}}          # new world centre XY
    Returns (positions, rejected) -- rejected is a list of strings."""
    proposal_mode = params is not None
    params = params or {}
    rejected = []
    bays = ctx["bays"]
    wall_groups = ctx["wall_groups"]
    by_room = ctx["by_room"]
    roles = ctx["roles"]
    obj_by_path = ctx["obj_by_path"]
    max_shift, wall_slide, sat_slide = (ctx["max_shift"], ctx["wall_slide"],
                                        ctx["sat_slide"])
    delta = {}

    # 1) bed bays -- rigid 1D slide along the headwall
    placed_bays = {}
    for bi, b in enumerate(bays):
        key = b["key"]
        room = by_room[key[0]]
        lo, hi = _axis_span(room, b["axis"])
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

        def _bay_ok(cand):
            iv = (b["iv"][0] + cand, b["iv"][1] + cand)
            if iv[0] < lo or iv[1] > hi:
                return False
            if any(_iv_overlap((a[0] + cand, a[1] + cand), q, margin=0.03)
                   for pm in b["pairs_m"] for a, q in pm):
                return False
            if any(_iv_overlap(iv, q, margin=0.05) for q in dyn):
                return False
            dxy = (cand, 0.0) if b["axis"] == "x" else (0.0, cand)
            moved = [_blk_shift(q, *dxy) for q in b["blks"]]
            if dyn_blks and any(_blk_hit(mq, q, margin=0.03)
                                for mq in moved for q in dyn_blks):
                return False
            if any(_rect_overlap(moved[fi][0], w[0])
                   for fi, w in ctx["bay_walls"][bi]):
                return False
            return True

        s = 0.0
        if proposal_mode:
            prop = params.get("bay_slides", {}).get(bi)
            if prop is not None:
                cand = max(-max_shift, min(max_shift, float(prop)))
                if _bay_ok(cand):
                    s = cand
                else:
                    rejected.append(f"bay_slides[{bi}]={prop:.2f}")
        else:
            for _ in range(attempts):
                cand = rng.uniform(-max_shift, max_shift)
                if _bay_ok(cand):
                    s = cand
                    break
        placed_bays[bi] = (s, (b["iv"][0] + s, b["iv"][1] + s))
        dxy = (s, 0.0) if b["axis"] == "x" else (0.0, s)
        for p in b["members"]:
            delta[p] = dxy

        # satellites additionally jitter ALONG the bed, staying at the bed,
        # respecting the same non-grandfathered pairs, and never NEWLY
        # overlapping another bay member (overbed-table-over-bed stays
        # grandfathered; bedside-table-into-bed is not).
        bediv = b["m_ivs"][0][0]            # members[0] is the bed
        cur = {mi2: _blk_shift(b["blks"][mi2], *dxy)
               for mi2 in range(len(b["members"]))}
        for mi, p in enumerate(b["members"]):
            if roles[p] != "satellite":
                continue
            mb = b["blks"][mi]
            miv = b["m_ivs"][mi][0]

            def _sat_ok(c2):
                tot = s + c2
                niv = (miv[0] + tot, miv[1] + tot)
                if niv[0] < bediv[0] + s - 0.45 or \
                   niv[1] > bediv[1] + s + 0.45:
                    return False              # must stay AT the bed
                if any(_iv_overlap((a[0] + tot, a[1] + tot), q, margin=0.03)
                       for a, q in b["pairs_m"][mi]):
                    return False
                dxy2 = (tot, 0.0) if b["axis"] == "x" else (0.0, tot)
                nb = _blk_shift(mb, *dxy2)
                if any(_blk_hit(nb, cur[mj], margin=0.02)
                       and not _blk_hit(mb, b["blks"][mj], margin=0.02)
                       for mj in range(len(b["members"])) if mj != mi):
                    return False
                return True

            t = 0.0
            if proposal_mode:
                prop = params.get("sat_slides", {}).get(p)
                if prop is not None:
                    c2 = max(-sat_slide, min(sat_slide, float(prop)))
                    if _sat_ok(c2):
                        t = c2
                    else:
                        rejected.append(f"sat_slides[{obj_by_path[p]['class']}]"
                                        f"={prop:.2f}")
            else:
                for _ in range(20):
                    c2 = rng.uniform(-sat_slide, sat_slide)
                    if _sat_ok(c2):
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
        # randomize ALONG THE REAL WALL extent when known, else the room span
        if g.get("wall_span"):
            lo, hi = g["wall_span"]
        else:
            lo, hi = _axis_span(room, key[2])
        lo, hi = min(lo, g["iv"][0]), max(hi, g["iv"][1])
        obstacles = [iv for (iv, qz0, qz1) in ctx["wall_static"][key]
                     if _zov(g["z"][0], g["z"][1], qz0, qz1)]
        for gj, g2 in enumerate(wall_groups):
            if gj == gi or g2["in_bay"] or g2["key"] != key:
                continue
            if _zov(g["z"][0], g["z"][1], g2["z"][0], g2["z"][1]):
                obstacles.append(g_now.get(gj, g2["iv"]))
        for bj, ob in enumerate(bays):
            if ob["key"] == key:
                obstacles.append(placed_bays[bj][1])

        def _grp_ok(cand):
            iv = (g["iv"][0] + cand, g["iv"][1] + cand)
            if iv[0] < lo or iv[1] > hi:
                return False
            return not any(_iv_overlap(iv, q, margin=0.03) for q in obstacles)

        s = 0.0
        if proposal_mode:
            prop = params.get("wall_slides", {}).get(gi)
            if prop is not None:
                cand = max(-wall_slide, min(wall_slide, float(prop)))
                if _grp_ok(cand):
                    s = cand
                else:
                    rejected.append(f"wall_slides[{gi}]={prop:.2f}")
        else:
            # land at a RANDOM position along the wall. When the real wall is
            # known, sweep its FULL extent (true placement randomization); with
            # no walls, fall back to a wall_slide-bounded jitter.
            if g.get("wall_span"):
                clo, chi = lo - g["iv"][0], hi - g["iv"][1]
            else:
                clo = max(-wall_slide, lo - g["iv"][0])
                chi = min(wall_slide, hi - g["iv"][1])
            for _ in range(attempts):
                cand = rng.uniform(clo, chi) if chi > clo else 0.0
                if _grp_ok(cand):
                    s = cand
                    break
        g_now[gi] = (g["iv"][0] + s, g["iv"][1] + s)
        dxy = (s, 0.0) if key[2] == "x" else (0.0, s)
        for p in g["members"]:
            delta[p] = dxy

    # 3) floor furniture -- local 2D re-place with full collision
    placed_blks = [(q, f"static:{si}")
                   for si, q in enumerate(ctx["static_blks"])]
    for b in bays:                      # members at their FINAL deltas
        for mi, p in enumerate(b["members"]):
            placed_blks.append((_blk_shift(b["blks"][mi], *delta[p]), p))
    # wall equipment is a Z-AWARE obstacle for floor furniture: a chair must not
    # overlap a low wall panel (gas manifold reaching the floor), but CAN sit
    # under a high monitor (no z-overlap). Without this, floor items walked
    # through low wall gear.
    for gi, g in enumerate(wall_groups):
        if g["in_bay"]:
            continue
        dxy = (g_now[gi][0] - g["iv"][0], 0.0) if g["key"][2] == "x" \
            else (0.0, g_now[gi][0] - g["iv"][0])
        for p in g["members"]:
            placed_blks.append((_blk_shift(_blk(obj_by_path[p]["aabb"]), *dxy), p))
    pending = {p: (q, p) for p, q in ctx["orig_blks"].items()}
    for o in ctx["floor_movers"]:
        p = o["path"]
        room = by_room[o["room"]]
        del pending[p]
        f0, z0, z1 = ctx["orig_blks"][p]
        rxlo, rxhi = min(room["xmin"], f0[0]), max(room["xmax"], f0[2])
        rylo, ryhi = min(room["ymin"], f0[1]), max(room["ymax"], f0[3])
        gf = ctx["floor_gf"][p]
        obstacles = [(q, tag) for q, tag in
                     placed_blks + list(pending.values())
                     if tag not in gf]

        def _floor_ok(cdx, cdy, full_walls=False):
            foot = _rect_shift(f0, cdx, cdy)
            if foot[0] < rxlo or foot[2] > rxhi or \
               foot[1] < rylo or foot[3] > ryhi:
                return False
            if any(_blk_hit((foot, z0, z1), q, margin=0.03)
                   for q, _tag in obstacles):
                return False
            if full_walls:
                # long-range move: the precomputed near-list only covers walls
                # around the ORIGINAL spot -- check every wall mesh instead
                ex = ctx["floor_wall_exempt"][p]
                cand = (foot, z0, z1)
                if any(_blk_hit(cand, w) for i, w in enumerate(ctx["wb_blks"])
                       if i not in ex):
                    return False
                # CONFINEMENT: the move's PATH must not cross a wall (else the
                # item would teleport into the next room through a gap). Walls
                # the item already straddles (ex) don't count.
                nc = (cx0 + cdx, cy0 + cdy)
                if any(_seg_rect((cx0, cy0), nc, w[0]) and _zov(z0, z1, w[1], w[2])
                       for i, w in enumerate(ctx["wb_blks"]) if i not in ex):
                    return False
            elif any(_rect_overlap(foot, w[0]) for w in ctx["floor_walls"][p]):
                return False
            return True

        cx0 = 0.5 * (o["aabb"][0][0] + o["aabb"][1][0])
        cy0 = 0.5 * (o["aabb"][0][1] + o["aabb"][1][1])
        hx = 0.5 * (o["aabb"][1][0] - o["aabb"][0][0])
        hy = 0.5 * (o["aabb"][1][1] - o["aabb"][0][1])
        dx = dy = 0.0
        if proposal_mode:
            prop = params.get("floor", {}).get(p)
            if prop is not None:
                cdx, cdy = float(prop[0]) - cx0, float(prop[1]) - cy0
                if _floor_ok(cdx, cdy, full_walls=True):
                    dx, dy = cdx, cdy
                else:
                    rejected.append(f"floor[{o['class']}]=({prop[0]:.2f},{prop[1]:.2f})")
        else:
            for _ in range(attempts):
                if ctx["global_frac"] > 0.0 and rng.random() < ctx["global_frac"]:
                    # GLOBAL re-place: anywhere in the room (collision-checked)
                    tx = rng.uniform(rxlo + hx, rxhi - hx)
                    ty = rng.uniform(rylo + hy, ryhi - hy)
                    cdx, cdy = tx - cx0, ty - cy0
                    if _floor_ok(cdx, cdy, full_walls=True):
                        dx, dy = cdx, cdy
                        break
                else:
                    cdx = rng.uniform(-max_shift, max_shift)
                    cdy = rng.uniform(-max_shift, max_shift)
                    if _floor_ok(cdx, cdy):
                        dx, dy = cdx, cdy
                        break
        placed_blks.append(((_rect_shift(f0, dx, dy), z0, z1), p))
        delta[p] = (dx, dy)

    # 4) surface items -- PLACEMENT RANDOMIZATION onto a random ALLOWED host.
    # Host equivalence (placement_affordances): a paperbox lands on the bed OR a
    # bedside table OR an overbed table -- whichever a random draw picks -- at a
    # random free spot on its top (or on a room floor for "floor" hosts). z is
    # set so the item rests on that host, so it can move BETWEEN hosts of
    # different heights (unlike the old ride-your-one-support scatter).
    delta_z = {}
    on_host = defaultdict(list)         # host_path -> rects already placed on it
    for o in (oo for oo in ctx["objs"] if roles[oo["path"]] == "surface"):
        p = o["path"]
        (omn, omx) = o["aabb"]
        hx, hy = 0.5 * (omx[0] - omn[0]), 0.5 * (omx[1] - omn[1])
        ocx, ocy, obot = 0.5 * (omn[0] + omx[0]), 0.5 * (omn[1] + omx[1]), omn[2]
        order = list(ctx["surface_hosts"].get(p, []))
        rng.shuffle(order)
        for kind, sel in order:
            if kind == "on":
                host = obj_by_path.get(sel)
                if host is None:
                    continue
                hdx, hdy = delta.get(sel, (0.0, 0.0))
                (hmn, hmx) = host["aabb"]
                inset = 0.15 if host["class"] in BED_CLASSES else 0.02
                x0, x1 = hmn[0] + hdx + hx + inset, hmx[0] + hdx - hx - inset
                y0, y1 = hmn[1] + hdy + hy + inset, hmx[1] + hdy - hy - inset
                top = hmx[2]
            else:                       # ("floor", room)
                room = by_room[sel]
                x0, x1 = room["xmin"] + hx, room["xmax"] - hx
                y0, y1 = room["ymin"] + hy, room["ymax"] - hy
                top = floor_z
            if x1 <= x0 or y1 <= y0:
                continue
            sz = omx[2] - omn[2]
            host_path = sel if kind == "on" else None
            done = False
            for _ in range(25):
                tx, ty = rng.uniform(x0, x1), rng.uniform(y0, y1)
                rect = [tx - hx, ty - hy, tx + hx, ty + hy]
                key = sel if kind == "on" else f"floor:{sel}"
                if any(_rect_overlap(rect, q, margin=0.01) for q in on_host[key]):
                    continue
                # Z-aware: don't land under an overhang (a remote on the bed must
                # avoid the strip the overbed table cantilevers over) or inside
                # any other furniture; the host itself is exempt.
                cand = (rect, top, top + sz)
                if any(tag != host_path and _blk_hit(cand, q, margin=0.01)
                       for q, tag in placed_blks):
                    continue
                on_host[key].append(rect)
                delta[p] = (tx - ocx, ty - ocy)
                delta_z[p] = top - obot       # rest the item's bottom on `top`
                done = True
                break
            if done:
                break

    # 5) emit world translates (surfaces may also change z to sit on their host)
    out = {}
    for o in ctx["objs"]:
        ox, oy, oz = ctx["pos0"][o["path"]]
        dx, dy = delta.get(o["path"], (0.0, 0.0))
        out[o["path"]] = (ox + dx, oy + dy, oz + delta_z.get(o["path"], 0.0))
    return out, rejected


# -------------------------------------------------------------- public API --
def generate(object_targets, rooms, floor_z, n_frames, seed=1000,
             max_shift=0.8, wall_slide=0.2, sat_slide=0.5, attempts=40,
             wall_boxes=None, door_clearance=0.5, link_gap=0.25,
             global_frac=0.35, verbose=True):
    """-> {path: [(x, y, z)] * n_frames}, every degree of freedom rng-sampled.
    global_frac: probability that a floor item is re-placed ANYWHERE in its
    room (collision-checked) instead of jittered near its original spot."""
    rng = random.Random(seed)
    ctx = _build_ctx(object_targets, rooms, floor_z, max_shift=max_shift,
                     wall_slide=wall_slide, sat_slide=sat_slide,
                     wall_boxes=wall_boxes, door_clearance=door_clearance,
                     link_gap=link_gap, global_frac=global_frac,
                     verbose=verbose)
    out = {o["path"]: [] for o in ctx["objs"]}
    for _f in range(n_frames):
        pos, _ = _solve(ctx, rng, attempts=attempts)
        for p, xyz in pos.items():
            out[p].append(xyz)
    return out


def apply_layout(ctx, params, seed=0):
    """Validate ONE proposed layout (e.g. from an LLM layout designer) through
    the exact same constraint machinery. Unproposed degrees of freedom stay at
    the ORIGINAL pose; rider scatter uses `seed`. -> (positions, rejected)."""
    rng = random.Random(seed)
    return _solve(ctx, rng, params=params or {})


def validate_grounding(object_targets, floor_z, tol=0.05):
    """Analytic gravity check: every object DECLARED to rest on the floor or a
    surface must actually have something directly under it; if its bottom floats
    above that, the placement is invalid (it would fall). Wall-attached and
    fixed objects are exempt (held up by the wall / built in). This is the cheap
    deterministic equivalent of dropping each free body under physics and seeing
    whether it settles where it was placed. -> list of (path, class, reason)."""
    from layout import affordances as _paff
    supports = [o for o in object_targets
                if _paff.provides_surface(o["class"]) or
                (o["aabb"][1][0] - o["aabb"][0][0]) *
                (o["aabb"][1][1] - o["aabb"][0][1]) >= _SUPPORT_MIN_AREA]
    bad = []
    for o in object_targets:
        cls = o["class"]
        support = o.get("affordance") or _paff.support_of(cls)
        zb = o["aabb"][0][2]
        if support in ("wall", "fixed"):
            continue
        # an elevated surface-class item declared 'floor' rests on a surface the
        # dump couldn't detect (unmodeled counter/shelf) -- it stays put, exempt
        if support == "floor" and _paff.support_of(cls) == "surface" and \
           (zb - floor_z) > _ELEVATED:
            continue
        if support == "floor":
            if zb > floor_z + tol:
                bad.append((o["path"], cls, f"floats {zb - floor_z:.2f}m above floor"))
        elif support == "surface":
            if _find_support(o, supports) is None:
                bad.append((o["path"], cls, "no support object under it"))
    return bad


def build_ctx(object_targets, rooms, floor_z, **kw):
    """Public wrapper so layout designers (llm_placement.py) can build the
    scene context once and call apply_layout() per proposal."""
    return _build_ctx(object_targets, rooms, floor_z, **kw)
