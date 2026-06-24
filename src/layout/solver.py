"""Constraint-based placement solver (Infinigen-style) for the Isaac ward.

Replaces placement_dr's greedy rejection sampling with SIMULATED ANNEALING over
a soft-constraint energy, operating directly on Isaac object_targets + rooms +
real wall AABBs + the explicit affordances -- NO Blender.

Two ideas, same as Infinigen Indoors:

  1. HARD constraints by construction. Every object lives in a feasible ZONE: a
     wall item slides ON its wall (flush, fixed height), a floor item moves on
     the room floor, a surface item sits on a host top. Sampling/perturbing
     never leaves the zone, so support + attachment are ALWAYS satisfied -- the
     things we kept hand-patching (grounding, wall adherence) are free.

  2. SOFT constraints by energy. What remains -- objects not overlapping, not
     buried in walls, beds against a wall, satellites near their bed, doorway
     clearances -- are penalty terms summed into an energy the annealer
     minimizes JOINTLY. Adding a new "style" constraint = adding a term, not
     patching a sampler.

Multi-stage like Infinigen: large furniture (floor/wall/bed) is annealed first,
then small surface objects are dropped onto the settled hosts.

Drop-in for placement_dr.generate:  generate(...) -> {path: [(x,y,z)]*n_frames}.
Reuses placement_dr's geometry primitives so there's one source of truth.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from layout import engine as pdr
from layout import affordances as paff

_ELEVATED = 0.35


# ============================================================ zones (hard) ==
def _nearest_wall(center, walls):
    """Wall AABB nearest to an XY point, plus its run axis. -> (wall, run, perp)."""
    best, bd = None, 1e9
    for w in walls:
        (mn, mx) = w
        dx = max(mn[0] - center[0], 0.0, center[0] - mx[0])
        dy = max(mn[1] - center[1], 0.0, center[1] - mx[1])
        d = math.hypot(dx, dy)
        if d < bd:
            bd, best = d, w
    if best is None:
        return None, None, None
    (mn, mx) = best
    run = "x" if (mx[0] - mn[0]) >= (mx[1] - mn[1]) else "y"
    perp = "y" if run == "x" else "x"
    return best, run, perp


def _room_of(cx, cy, rooms, fallback):
    """The SMALLEST room AABB containing (cx, cy) -- geometric room assignment.
    The spec's prim-name room can be wrong (a bin physically in the bathroom
    labelled Ward); 'smallest box that contains it' fixes that and disambiguates
    the overlapping centroid-derived room boxes. Falls back to the labelled room."""
    best, best_a = None, 1e18
    for r in rooms:
        if r["xmin"] <= cx <= r["xmax"] and r["ymin"] <= cy <= r["ymax"]:
            a = (r["xmax"] - r["xmin"]) * (r["ymax"] - r["ymin"])
            if a < best_a:
                best_a, best = a, r
    return best if best is not None else fallback


def build_zones(targets, rooms, floor_z, walls, ctx):
    """One feasible zone per object, from its role + affordance + geometry."""
    by_room = {r["name"]: r for r in rooms}
    roles = ctx["roles"]
    zones = {}
    for o in targets:
        p, cls = o["path"], o["class"]
        role = roles.get(p, "fixture")
        (mn, mx) = o["aabb"]
        hx, hy = 0.5 * (mx[0] - mn[0]), 0.5 * (mx[1] - mn[1])
        sz = mx[2] - mn[2]
        cx0, cy0 = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
        cz = 0.5 * (mn[2] + mx[2])
        # assign the room GEOMETRICALLY (smallest box containing it), not by the
        # spec's prim-name room which can be wrong / ambiguous under overlap.
        room = _room_of(cx0, cy0, rooms, by_room.get(o.get("room")))

        if role in ("fixture", "surface") or room is None:
            # fixtures don't move; surfaces are placed in stage 2
            zones[p] = {"kind": "fixed", "c": (cx0, cy0, cz), "half": (hx, hy),
                        "sz": sz, "role": role}
            continue

        if role == "wall" and walls:
            wall, run, perp = _nearest_wall((cx0, cy0), walls)
            if wall is not None:
                (wmn, wmx) = wall
                ri = 0 if run == "x" else 1
                pi = 0 if perp == "x" else 1
                half_perp = hx if perp == "x" else hy
                # SNAP flush to the wall's room-facing surface: put the item's
                # back face on the wall, on whichever side it was authored.
                auth_perp = cx0 if perp == "x" else cy0
                if auth_perp >= 0.5 * (wmn[pi] + wmx[pi]):
                    perp_val = wmx[pi] + half_perp
                else:
                    perp_val = wmn[pi] - half_perp
                half_run = hx if run == "x" else hy
                lo, hi = wmn[ri] + half_run, wmx[ri] - half_run
                # bound the slide to this object's ROOM (a wall can span into the
                # next room; the item must not slide out of its own room)
                if room is not None:
                    rlo = (room["xmin"] if run == "x" else room["ymin"]) + half_run
                    rhi = (room["xmax"] if run == "x" else room["ymax"]) - half_run
                    lo, hi = max(lo, min(rlo, cx0 if run == "x" else cy0)), \
                        min(hi, max(rhi, cx0 if run == "x" else cy0))
                zones[p] = {"kind": "wall", "run": run, "perp": perp,
                            "perp_val": perp_val, "lo": lo, "hi": hi, "z": cz,
                            "half": (hx, hy), "sz": sz, "wall": wall, "role": role,
                            "cls": cls}
                continue
        # floor / bed / satellite (and wall items with no real walls) -> floor.
        # The centroid-derived room AABBs OVERLAP (a small bathroom sits inside
        # the ward's box), so a ward item's zone would include bathroom floor.
        # Forbid it from any MORE-SPECIFIC (smaller-area) room it doesn't belong
        # to -> a point belongs to the smallest room box containing it.
        avoid = []
        if room is not None:
            my_area = (room["xmax"] - room["xmin"]) * (room["ymax"] - room["ymin"])
            for r in rooms:
                if r["name"] == room["name"]:
                    continue
                if (r["xmax"] - r["xmin"]) * (r["ymax"] - r["ymin"]) < my_area:
                    avoid.append([r["xmin"], r["ymin"], r["xmax"], r["ymax"]])
        zones[p] = {"kind": "floor", "room": room, "z": floor_z + sz / 2,
                    "half": (hx, hy), "sz": sz, "c0": (cx0, cy0), "role": role,
                    "cls": cls, "avoid_rooms": avoid}
    return zones


def sample_zone(z, rng):
    """A random feasible center (cx, cy, cz) in the zone."""
    if z["kind"] == "fixed":
        return z["c"]
    if z["kind"] == "wall":
        run_v = rng.uniform(z["lo"], z["hi"]) if z["hi"] > z["lo"] else \
            0.5 * (z["lo"] + z["hi"])
        if z["run"] == "x":
            return (run_v, z["perp_val"], z["z"])
        return (z["perp_val"], run_v, z["z"])
    # floor
    r, (hx, hy) = z["room"], z["half"]
    cx = rng.uniform(r["xmin"] + hx, r["xmax"] - hx) if r["xmax"] - hx > r["xmin"] + hx else 0.5 * (r["xmin"] + r["xmax"])
    cy = rng.uniform(r["ymin"] + hy, r["ymax"] - hy) if r["ymax"] - hy > r["ymin"] + hy else 0.5 * (r["ymin"] + r["ymax"])
    return (cx, cy, z["z"])


def perturb_zone(z, c, rng, scale):
    """A nearby feasible center, step size ~ scale (annealing temperature)."""
    if z["kind"] == "fixed":
        return c
    if z["kind"] == "wall":
        ri = 0 if z["run"] == "x" else 1
        run_v = min(z["hi"], max(z["lo"], c[ri] + rng.gauss(0, scale)))
        if z["run"] == "x":
            return (run_v, z["perp_val"], z["z"])
        return (z["perp_val"], run_v, z["z"])
    r, (hx, hy) = z["room"], z["half"]
    cx = min(r["xmax"] - hx, max(r["xmin"] + hx, c[0] + rng.gauss(0, scale)))
    cy = min(r["ymax"] - hy, max(r["ymin"] + hy, c[1] + rng.gauss(0, scale)))
    return (cx, cy, z["z"])


def _aabb(c, z):
    hx, hy = z["half"]
    hz = 0.5 * z["sz"]
    return ((c[0] - hx, c[1] - hy, c[2] - hz), (c[0] + hx, c[1] + hy, c[2] + hz))


def _overlap_vol(a, b):
    ox = min(a[1][0], b[1][0]) - max(a[0][0], b[0][0])
    oy = min(a[1][1], b[1][1]) - max(a[0][1], b[0][1])
    oz = min(a[1][2], b[1][2]) - max(a[0][2], b[0][2])
    return ox * oy * oz if ox > 0 and oy > 0 and oz > 0 else 0.0


# ===================================================== HARD constraints ==
# Overlap, wall-penetration and wall-crossing are INVIOLABLE: a move that
# creates one is rejected outright (not penalized), so the configuration is
# always feasible -- items simply cannot overlap. Only genuine PREFERENCES
# (below) are traded off by the annealer.
def _feasible(p, c, centers, zones, walls, static, group_of, authored):
    z = zones[p]
    blk = pdr._blk(_aabb(c, z))
    # NO overlap with any other object -- objects simply cannot interpenetrate.
    # (No group exemption: a bedside table must sit BESIDE the bed, not in it.)
    for q, cq in centers.items():
        if q == p:
            continue
        if pdr._blk_hit(blk, pdr._blk(_aabb(cq, zones[q])), margin=0.0):
            return False
    # no overlap with fixtures (toilet/sink/door)
    for sp, sab in static:
        if pdr._blk_hit(blk, pdr._blk(sab), margin=0.0):
            return False
    # not buried in a wall (the item's own wall is exempt -- it's flush to it)
    own = z.get("wall")
    for w in walls or []:
        if w is own:
            continue
        if pdr._blk_hit(blk, pdr._blk(w), margin=0.0):
            return False
    if z["kind"] == "floor":
        # must not stray into a more-specific room it doesn't belong to (the
        # ward/bathroom AABB overlap), even though that path crosses no wall
        for ar in z.get("avoid_rooms", ()):
            if pdr._rect_overlap(blk[0], ar):
                return False
        # must not move to the far side of a wall (stay in its room)
        if walls:
            c0 = authored[p]
            for w in walls:
                wr = [w[0][0], w[0][1], w[1][0], w[1][1]]
                if pdr._rect_overlap(blk[0], wr):
                    continue
                if pdr._zov(blk[1], blk[2], w[0][2], w[1][2]) and \
                   pdr._seg_rect(c0, (c[0], c[1]), wr):
                    return False
    return True


# ===================================================== SOFT preferences ==
W_AGAINST = 4.0        # beds hug a wall
W_NEARBED = 2.0        # satellites stay next to their bed
W_WALLHUG = 3.0        # free-standing floor furniture (bins/chairs) prefers a wall
DOOR_CLEARANCE = 0.5   # keep-out (m) around doors so furniture won't block them


def _wall_gap(c, half, walls):
    """Min XY gap from a footprint (centre c, half-extents) to any wall; 0 if
    flush against one."""
    x0, y0, x1, y1 = c[0] - half[0], c[1] - half[1], c[0] + half[0], c[1] + half[1]
    best = 1e9
    for w in walls:
        dx = max(w[0][0] - x1, x0 - w[1][0], 0.0)
        dy = max(w[0][1] - y1, y0 - w[1][1], 0.0)
        best = min(best, math.hypot(dx, dy))
    return 0.0 if best > 1e8 else best


def _soft_energy(p, c, z, walls, bed_centers):
    e = 0.0
    if z["role"] == "bed" and walls:
        wall, run, perp = _nearest_wall((c[0], c[1]), walls)
        if wall is not None:
            pi = 0 if perp == "x" else 1
            e += W_AGAINST * min(abs(c[pi] - z["half"][pi] - wall[1][pi]),
                                 abs(c[pi] + z["half"][pi] - wall[0][pi]))
    if z["role"] == "satellite" and bed_centers:
        d = min(math.hypot(c[0] - b[0], c[1] - b[1]) for b in bed_centers)
        e += W_NEARBED * max(0.0, d - 1.0)
    # free-standing floor furniture prefers to sit against a wall rather than in
    # the open middle of the room (realism); satellites/beds handled above.
    if z["role"] == "floor" and walls:
        e += W_WALLHUG * _wall_gap(c, z["half"], walls)
    return e


def _total_soft(centers, movable, zones, walls):
    bed_centers = [centers[p] for p in movable if zones[p]["role"] == "bed"]
    return sum(_soft_energy(p, centers[p], zones[p], walls, bed_centers)
               for p in movable)


def _residual_overlaps(centers, movable, zones, group_of, static):
    """How many object pairs still interpenetrate (only the rare init fallback
    can leave any, since every accepted move is overlap-free)."""
    n = 0
    for i, p in enumerate(movable):
        for q in movable[i + 1:]:
            if _overlap_vol(_aabb(centers[p], zones[p]),
                            _aabb(centers[q], zones[q])) > 1e-9:
                n += 1
        for sp, sab in static:
            if _overlap_vol(_aabb(centers[p], zones[p]), sab) > 1e-9:
                n += 1
    return n


# ============================================================== annealing ==
def _project(c, z):
    """Authored centre projected into the zone, for a feasible-ish init."""
    if z["kind"] == "wall":
        ri = 0 if z["run"] == "x" else 1
        run_v = min(z["hi"], max(z["lo"], c[ri]))
        return (run_v, z["perp_val"], z["z"]) if z["run"] == "x" \
            else (z["perp_val"], run_v, z["z"])
    if z["kind"] == "floor":
        r, (hx, hy) = z["room"], z["half"]
        return (min(r["xmax"] - hx, max(r["xmin"] + hx, c[0])),
                min(r["ymax"] - hy, max(r["ymin"] + hy, c[1])), z["z"])
    return c


def _solve_stage1(movable, zones, walls, group_of, rng, steps, t0, t1, static,
                  authored):
    # PROCEDURAL placement: add objects ONE BY ONE into free space. Order =
    # SUPPORT SURFACES first (bed, bedside/overbed tables, counters) because
    # small items get placed ON them, then the rest largest-first.
    centers = {}
    order = sorted(movable, key=lambda p: (
        not paff.provides_surface(zones[p].get("cls", "")),
        -(zones[p]["half"][0] * zones[p]["half"][1])))
    for p in order:
        placed = None
        for _ in range(150):
            c = sample_zone(zones[p], rng)
            if _feasible(p, c, centers, zones, walls, static, group_of, authored):
                placed = c
                break
        centers[p] = placed if placed is not None else _project(authored[p], zones[p])
    bed_centers = [centers[p] for p in movable if zones[p]["role"] == "bed"]

    for i in range(steps):
        frac = i / max(steps - 1, 1)
        T = t0 * (t1 / t0) ** frac
        p = movable[rng.randrange(len(movable))]
        old = centers[p]
        # big jumps (global resample) for variety + local refinement
        new = sample_zone(zones[p], rng) if rng.random() < 0.3 \
            else perturb_zone(zones[p], old, rng, max(0.05, T))
        if not _feasible(p, new, centers, zones, walls, static, group_of, authored):
            continue                          # HARD reject -- stays feasible
        de = (_soft_energy(p, new, zones[p], walls, bed_centers) -
              _soft_energy(p, old, zones[p], walls, bed_centers))
        if de <= 0 or rng.random() < math.exp(-de / max(T, 1e-6)):
            centers[p] = new
            if zones[p]["role"] == "bed":
                bed_centers = [centers[b] for b in movable if zones[b]["role"] == "bed"]
    return centers


def _place_surfaces(targets, ctx, centers, zones, rng):
    """Stage 2: drop small surface items onto their settled hosts (host
    equivalence from placement_affordances), Z-aware against overhangs."""
    roles = ctx["roles"]
    obp = ctx["obj_by_path"]
    # final aabbs of everything placed so far (for overhang avoidance)
    placed = []
    for p, c in centers.items():
        placed.append((p, _aabb(c, zones[p])))
    host_now = {}                            # host path -> current top aabb
    for p, ab in placed:
        host_now[p] = ab
    on_host = defaultdict(list)
    out = {}
    for o in targets:
        p = o["path"]
        if roles.get(p) != "surface":
            continue
        (omn, omx) = o["aabb"]
        hx, hy = 0.5 * (omx[0] - omn[0]), 0.5 * (omx[1] - omn[1])
        sz = omx[2] - omn[2]
        cands = list(ctx["surface_hosts"].get(p, []))
        rng.shuffle(cands)
        done = False
        for kind, sel in cands:
            if kind != "on" or sel not in host_now:
                continue
            (hmn, hmx) = host_now[sel]
            inset = 0.15 if obp[sel]["class"] in pdr.BED_CLASSES else 0.02
            x0, x1 = hmn[0] + hx + inset, hmx[0] - hx - inset
            y0, y1 = hmn[1] + hy + inset, hmx[1] - hy - inset
            if x1 <= x0 or y1 <= y0:
                continue
            top = hmx[2]
            for _ in range(25):
                tx, ty = rng.uniform(x0, x1), rng.uniform(y0, y1)
                rect = [tx - hx, ty - hy, tx + hx, ty + hy]
                if any(pdr._rect_overlap(rect, q, margin=0.01) for q in on_host[sel]):
                    continue
                cand = (rect, top, top + sz)
                if any(tag != sel and pdr._blk_hit(cand, pdr._blk(ab), margin=0.01)
                       for tag, ab in placed):
                    continue
                on_host[sel].append(rect)
                out[p] = (tx, ty, top + sz / 2)
                done = True
                break
            if done:
                break
        if not done:                         # nowhere free -> keep authored
            out[p] = (0.5 * (omn[0] + omx[0]), 0.5 * (omn[1] + omx[1]),
                      0.5 * (omn[2] + omx[2]))
    return out


# ================================================================= public ==
def generate(object_targets, rooms, floor_z, n_frames, seed=1000,
             wall_boxes=None, steps=3000, t0=0.6, t1=0.03, restarts=3,
             verbose=True):
    """-> {path: [(x, y, z)] * n_frames}. Same output contract as
    placement_dr.generate, so the renderer/test are unchanged. (x,y,z) is the
    object's world TRANSLATE; pivot offsets are reapplied from _xform if any."""
    ctx = pdr.build_ctx(object_targets, rooms, floor_z,
                        wall_boxes=wall_boxes, verbose=False)
    zones = build_zones(object_targets, rooms, floor_z, wall_boxes, ctx)
    group_of = {o["path"]: o.get("_group") for o in object_targets}
    movable = [o["path"] for o in object_targets
               if zones[o["path"]]["kind"] in ("floor", "wall")]
    # fixtures are immovable OBSTACLES the annealer must avoid (a chair can't sit
    # inside the toilet); surfaces are placed in stage 2 so skip them here.
    static = []
    for o in object_targets:
        z = zones[o["path"]]
        if z["kind"] != "fixed" or z["role"] == "surface":
            continue
        ab = _aabb(z["c"], z)
        if o["class"] == "door":
            # full-height keep-out around doors so furniture doesn't block the
            # doorway / swing -- an item flush to a door's inner side is valid by
            # non-overlap but wrong; the clearance forbids it.
            cl = DOOR_CLEARANCE
            ab = ((ab[0][0] - cl, ab[0][1] - cl, ab[0][2]),
                  (ab[1][0] + cl, ab[1][1] + cl, ab[1][2]))
        static.append((o["path"], ab))
    authored = {o["path"]: (0.5 * (o["aabb"][0][0] + o["aabb"][1][0]),
                            0.5 * (o["aabb"][0][1] + o["aabb"][1][1]),
                            0.5 * (o["aabb"][0][2] + o["aabb"][1][2]))
                for o in object_targets}
    by_path = {o["path"]: o for o in object_targets}

    if verbose:
        kinds = defaultdict(int)
        for z in zones.values():
            kinds[z["kind"]] += 1
        print(f"[solver] zones: {dict(kinds)}; annealing {len(movable)} objects, "
              f"{steps} steps/frame")

    rng = random.Random(seed)
    out = {o["path"]: [] for o in object_targets}
    for f in range(n_frames):
        fr = random.Random(rng.random())
        # SA is stochastic -> run a few restarts; prefer the one with the fewest
        # residual overlaps (the rare init fallback), then the best preferences.
        best, best_score = None, (1e9, 1e9)
        for _ in range(max(1, restarts)):
            cs = _solve_stage1(movable, zones, wall_boxes, group_of, fr,
                               steps, t0, t1, static, authored)
            score = (_residual_overlaps(cs, movable, zones, group_of, static),
                     _total_soft(cs, movable, zones, wall_boxes))
            if score < best_score:
                best_score, best = score, cs
        centers = best
        surf = _place_surfaces(object_targets, ctx, centers, zones, fr)
        for o in object_targets:
            p = o["path"]
            if p in centers:
                c = centers[p]
            elif p in surf:
                c = surf[p]
            else:                            # fixed
                (mn, mx) = o["aabb"]
                c = (0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]),
                     0.5 * (mn[2] + mx[2]))
            # translate = AABB-centre + pivot offset (so a real asset's bbox
            # lands on the solved centre), matching slot_layout / placement_dr
            xf = o.get("_xform") or {}
            po = xf.get("pivot_off") or (0.0, 0.0, 0.0)
            out[p].append((c[0] + po[0], c[1] + po[1], c[2] + po[2]))
    return out
