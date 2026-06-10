"""Meaningful placement randomization for the ward renderer (pure Python, no Isaac).

Given the labeled `object_targets` (each augmented with its original world
`translate`), the detected `rooms` (with XY bounds), and `floor_z`, produce a
per-object per-frame WORLD translate so each rendered frame is a *different but
plausible* arrangement of the SAME room:

  * FREE-STANDING furniture (bed, overbed/bedside table, IV pole, chair, bins) is
    grouped into proximity CLUSTERS. Each cluster is rigidly TRANSLATED to a new
    spot on the floor within its room, collision-checked against the other
    clusters (rejection sampling; falls back to no move). Rigid translation keeps
    the cluster internally coherent (bed + its table + IV stay together) and keeps
    every object's orientation, so nothing tips over or clips.
  * SURFACE items (remote, thermometer, sanitizer, ...) ride with their nearest
    cluster (same translation), so things on the overbed table move with it.
  * WALL-mounted objects (monitor, phone, gas panel, ...) SLIDE along their wall
    (the axis parallel to the nearest room edge) by a small bounded offset.
  * FIXTURES (toilet, sink, door, window, curtains) do not move.

Returns {prim_path: [(x, y, z), ... one per frame]}. Orientation is left untouched
(we only author translate), which is why this is collision-safe and plausible.
"""
from __future__ import annotations

import random
from collections import defaultdict

# class -> placement category
_FREE = {
    "hospital_bed", "overbed_table", "bedside_table", "iv_pole", "companion_chair",
    "stool", "weight_scale", "soiled_linen_bin", "waste_bin", "medical_waste_container",
}
_WALL = {
    "bedside_monitor", "telephone", "light_switch", "gas_manifold", "oxygen_flowmeter",
    "suction_jar", "suction_knob", "air_vent", "mirror", "shower", "TV", "hook",
    "tissue_dispenser", "window",
}
_SURFACE = {
    "remote_control", "ear_thermometer", "paperbox", "medical_package", "gauze",
    "medical_gloves", "syringe", "stethoscope", "alcohol_spray_bottle", "sanitizer",
}
# everything else (toilet, sink, door, door_handle, toilet_handle, curtain,
# bed_curtain, ...) is treated as a FIXTURE and left in place.


def categorize(cls: str) -> str:
    if cls in _FREE:
        return "free"
    if cls in _WALL:
        return "wall"
    if cls in _SURFACE:
        return "surface"
    return "fixture"


def _xy_aabb(aabb):
    (mn, mx) = aabb
    return [mn[0], mn[1], mx[0], mx[1]]            # x0,y0,x1,y1


def _shift_aabb(b, dx, dy):
    return [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]


def _overlap(a, b, margin=0.0):
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or
                a[3] + margin <= b[1] or b[3] + margin <= a[1])


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _build_clusters(free_objs, link_dist):
    """Connected components of free objects by XY centroid distance < link_dist."""
    n = len(free_objs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = free_objs[i]["centroid"], free_objs[j]["centroid"]
            if (ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2 < link_dist ** 2:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(free_objs[i])
    return list(groups.values())


def generate(object_targets, rooms, floor_z, n_frames, seed=1000,
             max_shift=0.8, wall_slide=0.25, link_dist=2.0, attempts=40):
    """-> {path: [(x,y,z)] * n_frames}. Requires each target to carry 'translate'
    (original world translate) and 'room' (room name). Falls back gracefully."""
    rng = random.Random(seed)
    by_room_bounds = {r["name"]: r for r in rooms}

    # split objects by category
    cats = {o["path"]: categorize(o["class"]) for o in object_targets}
    pos = {o["path"]: o["translate"] for o in object_targets}          # originals
    out = {o["path"]: [] for o in object_targets}

    # cluster the free objects per room; attach surface items to nearest cluster
    clusters = []        # list of dict(room, members[paths], foot(xy aabb of free members))
    for room in rooms:
        free = [o for o in room["members"] if cats[o["path"]] == "free"]
        for grp in _build_clusters(free, link_dist):
            members = [o["path"] for o in grp]
            foot = _union([_xy_aabb(o["aabb"]) for o in grp])
            cx = 0.5 * (foot[0] + foot[2]); cy = 0.5 * (foot[1] + foot[3])
            clusters.append({"room": room["name"], "members": list(members),
                             "free_foot": foot, "cx": cx, "cy": cy,
                             "surf": []})
    # surface items -> nearest cluster centroid in the same room
    for o in object_targets:
        if cats[o["path"]] != "surface":
            continue
        cand = [c for c in clusters if c["room"] == o.get("room")]
        if not cand:
            continue
        ox, oy = o["centroid"][0], o["centroid"][1]
        c = min(cand, key=lambda c: (c["cx"] - ox) ** 2 + (c["cy"] - oy) ** 2)
        c["surf"].append(o["path"])

    # per-cluster combined footprint (free members + attached surface items)
    foot_of = {o["path"]: _xy_aabb(o["aabb"]) for o in object_targets}
    for c in clusters:
        boxes = [foot_of[p] for p in c["members"] + c["surf"] if p in foot_of]
        c["foot"] = _union(boxes) if boxes else c["free_foot"]

    # per-frame layout
    for _f in range(n_frames):
        placed = []                       # footprints already committed this frame
        for c in clusters:
            r = by_room_bounds[c["room"]]
            dx = dy = 0.0
            for _ in range(attempts):
                tdx = rng.uniform(-max_shift, max_shift)
                tdy = rng.uniform(-max_shift, max_shift)
                # keep the cluster CENTRE inside the (centroid-derived) room bounds
                # -- objects originally sit within this span, safely inside the walls
                ncx, ncy = c["cx"] + tdx, c["cy"] + tdy
                if not (r["xmin"] <= ncx <= r["xmax"] and r["ymin"] <= ncy <= r["ymax"]):
                    continue
                sf = _shift_aabb(c["foot"], tdx, tdy)
                # no overlap with clusters already placed this frame
                if any(_overlap(sf, q, margin=0.05) for q in placed):
                    continue
                dx, dy = tdx, tdy
                break
            placed.append(_shift_aabb(c["foot"], dx, dy))
            for p in c["members"] + c["surf"]:
                x, y, z = pos[p]
                out[p].append((x + dx, y + dy, z))

        # wall objects: slide along the nearest wall (axis parallel to nearest edge)
        for o in object_targets:
            if cats[o["path"]] != "wall":
                continue
            r = by_room_bounds.get(o.get("room"))
            x, y, z = pos[o["path"]]
            if r is None:
                out[o["path"]].append((x, y, z)); continue
            # distance to each room edge -> slide along the parallel axis
            dxmin, dxmax = abs(o["centroid"][0] - r["xmin"]), abs(o["centroid"][0] - r["xmax"])
            dymin, dymax = abs(o["centroid"][1] - r["ymin"]), abs(o["centroid"][1] - r["ymax"])
            s = rng.uniform(-wall_slide, wall_slide)
            if min(dxmin, dxmax) < min(dymin, dymax):      # near an X-wall -> slide in Y
                ny = min(max(y + s, r["ymin"]), r["ymax"]); out[o["path"]].append((x, ny, z))
            else:                                          # near a Y-wall -> slide in X
                nx = min(max(x + s, r["xmin"]), r["xmax"]); out[o["path"]].append((nx, y, z))

        # fixtures: unchanged
        for o in object_targets:
            if cats[o["path"]] == "fixture":
                out[o["path"]].append(tuple(pos[o["path"]]))

    return out
