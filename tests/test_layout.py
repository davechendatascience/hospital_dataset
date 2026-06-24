"""Pure-Python plausibility unit test for the slot layout + placement DR.

Generates N "settings" (frames) from a slot spec WITHOUT Isaac and checks each
for geometric sense, so layout bugs surface offline. Also renders a top-down
PNG per setting for visual inspection.

    python3 tests/test_layout.py [spec.json] [--frames 10] [--seed 1000] [--plots DIR]

Checks per setting (delta = new translate - seed translate, applied to the seed
AABB since placement_dr shifts translate and aabb together; z is never changed):
  * in-room      : an object that started inside its room must not leave it
  * wall-flush   : a wall-mounted object's perpendicular-to-wall coord is fixed
  * floor-z      : floor objects keep their bottom on the floor
  * on-support   : a surface object stays over its support's top face
  * no-overlap   : no NEW footprint overlap (seed overlaps are grandfathered)
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # repo/src
from layout import slots as slot_layout            # noqa: E402
from layout import engine as pdr                   # noqa: E402

TOL = 0.06   # 6 cm slack for float / room-AABB-from-centroid fuzz


def _shift_aabb(aabb, dx, dy, dz=0.0):
    (mn, mx) = aabb
    return ((mn[0] + dx, mn[1] + dy, mn[2] + dz),
            (mx[0] + dx, mx[1] + dy, mx[2] + dz))


def _in_room(foot_mn, foot_mx, room, tol):
    return (foot_mn[0] >= room["xmin"] - tol and foot_mx[0] <= room["xmax"] + tol and
            foot_mn[1] >= room["ymin"] - tol and foot_mx[1] <= room["ymax"] + tol)


def _xy_gap(a, b):
    """XY gap between two AABBs ((mn,mx) each); 0 if they overlap."""
    dx = max(b[0][0] - a[1][0], a[0][0] - b[1][0], 0.0)
    dy = max(b[0][1] - a[1][1], a[0][1] - b[1][1], 0.0)
    return math.hypot(dx, dy)


def _z_overlap(a, b):
    return a[0][2] < b[1][2] - 1e-9 and b[0][2] < a[1][2] - 1e-9


def _overlap_depth(a, b):
    """XY interpenetration depth (min of x- and y-overlap); 0 if disjoint."""
    ox = min(a[1][0], b[1][0]) - max(a[0][0], b[0][0])
    oy = min(a[1][1], b[1][1]) - max(a[0][1], b[0][1])
    return min(ox, oy) if ox > 0 and oy > 0 else 0.0


def check_setting(targets, ctx, rooms, floor_z, pos, fidx, walls=None,
                  teleport_m=1.5, wall_tol=0.18):
    """-> list of violation strings for one generated setting."""
    by_path = {t["path"]: t for t in targets}
    by_room = {r["name"]: r for r in rooms}
    roles = ctx["roles"]
    support_of = ctx["support_of"]
    # path -> its wall group's PINNED axis (edge[0]); bays carry wall members too
    wall_axis_of = {}
    for g in ctx["wall_groups"]:
        for mp in g["members"]:
            wall_axis_of[mp] = g["key"][1][0]
    for b in ctx["bays"]:
        for mp in b["members"]:
            if roles.get(mp) == "wall":
                wall_axis_of[mp] = b["key"][1][0]
    v = []

    # new AABBs (seed aabb shifted by the translate delta)
    new_aabb = {}
    for t in targets:
        p = t["path"]
        ox, oy, oz = t["translate"]
        nx, ny, nz = pos[p]
        new_aabb[p] = _shift_aabb(t["aabb"], nx - ox, ny - oy, nz - oz)

    for t in targets:
        p, cls, role = t["path"], t["class"], roles.get(t["path"], "?")
        (mn, mx) = new_aabb[p]
        room = by_room.get(t.get("room"))

        # wide moves are WANTED now -- what matters is the move stays VALID:
        # its path must not cross a real wall (teleporting into the next room)
        # and the object must not end up buried in a wall. Wall-role items are
        # meant to touch their wall, so they're exempt from penetration.
        ox, oy, _ = t["translate"]
        nx, ny, _ = pos[p]
        if walls:
            scx, scy = 0.5 * (t["aabb"][0][0] + t["aabb"][1][0]), \
                       0.5 * (t["aabb"][0][1] + t["aabb"][1][1])
            ncx, ncy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
            for w in walls:
                wr = (w[0][0], w[0][1], w[1][0], w[1][1])
                if not _z_overlap((mn, mx), w):
                    continue
                # path crossed a wall it didn't start touching -> left the room
                if (_xy_gap((t["aabb"]), w) > 0.02 and
                        pdr._seg_rect((scx, scy), (ncx, ncy), wr)):
                    v.append(f"f{fidx} CROSSED-WALL {cls:17s} {p} move path "
                             f"crosses a wall (left its room)")
                    break
                if role != "wall" and _overlap_depth((mn, mx), w) > 0.08 \
                        and _overlap_depth(t["aabb"], w) <= 0.08:
                    v.append(f"f{fidx} IN-WALL      {cls:18s} {p} buried in a "
                             f"wall ({_overlap_depth((mn, mx), w):.2f}m)")
                    break

        # wall items must sit on a REAL wall (not just inside the room AABB)
        if walls and role == "wall":
            on = any(_xy_gap((mn, mx), w) <= wall_tol and _z_overlap((mn, mx), w)
                     for w in walls)
            if not on:
                gap = min((_xy_gap((mn, mx), w) for w in walls), default=9.9)
                v.append(f"f{fidx} OFF-REAL-WALL {cls:17s} {p} {gap:.2f}m from "
                         f"nearest real wall")

        # in-room (only if it began inside)
        if room is not None:
            (smn, smx) = t["aabb"]
            began_in = _in_room(smn, smx, room, TOL)
            if began_in and not _in_room(mn, mx, room, TOL):
                v.append(f"f{fidx} OUT-OF-ROOM  {cls:18s} {p} "
                         f"foot x[{mn[0]:.2f},{mx[0]:.2f}] y[{mn[1]:.2f},{mx[1]:.2f}] "
                         f"room x[{room['xmin']:.2f},{room['xmax']:.2f}] "
                         f"y[{room['ymin']:.2f},{room['ymax']:.2f}]")

        # floor objects keep their bottom on the floor
        if role == "floor" and abs(mn[2] - floor_z) > TOL:
            v.append(f"f{fidx} FLOATING     {cls:18s} {p} z_bottom={mn[2]:.2f} "
                     f"floor_z={floor_z:.2f}")

        # wall objects: when REAL walls are known, "on the wall" is the honest
        # invariant (the OFF-REAL-WALL check below), so the perpendicular-drift
        # heuristic (which assumes the centroid-edge axis) is skipped -- it
        # false-flags an item that correctly slid along its REAL wall. Only used
        # as a fallback when no walls are available.
        if role == "wall" and not walls:
            ox, oy, _ = t["translate"]
            nx, ny, _ = pos[p]
            perp = wall_axis_of.get(p)
            if perp is None:
                edge, _run, _ = pdr._nearest_edge(by_room[t["room"]], t["aabb"])
                perp = edge[0]
            moved = abs(nx - ox) if perp == "x" else abs(ny - oy)
            if moved > TOL:
                v.append(f"f{fidx} WALL-DRIFT   {cls:18s} {p} moved {moved:.2f}m "
                         f"perpendicular to its {perp}-wall")

        # surface objects: with host randomization they may sit on ANY allowed
        # host, so check the new centre rests over SOME support's top (or floor),
        # not the originally-detected one.
        if role == "surface":
            cx, cy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
            from layout import affordances as _paff
            over = any(
                abs(mn[2] - new_aabb[q["path"]][1][2]) <= 0.06 and
                new_aabb[q["path"]][0][0] - TOL <= cx <= new_aabb[q["path"]][1][0] + TOL and
                new_aabb[q["path"]][0][1] - TOL <= cy <= new_aabb[q["path"]][1][1] + TOL
                for q in targets if _paff.provides_surface(q["class"]))
            on_floor = abs(mn[2] - floor_z) <= 0.06
            if not over and not on_floor:
                v.append(f"f{fidx} OFF-SUPPORT  {cls:18s} {p} centre "
                         f"({cx:.2f},{cy:.2f}) rests on nothing")

    # NEW collisions (seed overlaps are grandfathered)
    seed_blk = {t["path"]: pdr._blk(t["aabb"]) for t in targets}
    seed_hit = set()
    paths = [t["path"] for t in targets]
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if pdr._blk_hit(seed_blk[paths[i]], seed_blk[paths[j]], margin=0.0):
                seed_hit.add((paths[i], paths[j]))
    now_blk = {p: pdr._blk(new_aabb[p]) for p in paths}
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            a, b = paths[i], paths[j]
            if (a, b) in seed_hit:
                continue
            if roles.get(a) == "satellite" and roles.get(b) == "satellite":
                pass  # still report; satellites shouldn't newly collide either
            if pdr._blk_hit(now_blk[a], now_blk[b], margin=0.0):
                v.append(f"f{fidx} NEW-OVERLAP  {by_path[a]['class']} <-> "
                         f"{by_path[b]['class']}  ({a} | {b})")
    return v


def plot_setting(targets, ctx, rooms, pos, path_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    roles = ctx["roles"]
    color = {"bed": "#c44", "satellite": "#e90", "wall": "#39c",
             "floor": "#3a3", "surface": "#a3a", "fixture": "#999", "?": "#999"}
    fig, ax = plt.subplots(figsize=(7, 7))
    for r in rooms:
        ax.add_patch(Rectangle((r["xmin"], r["ymin"]), r["xmax"] - r["xmin"],
                               r["ymax"] - r["ymin"], fill=False, ec="k", lw=1.5))
        ax.text(r["xmin"] + 0.1, r["ymax"] - 0.3, r["name"], fontsize=8)
    for t in targets:
        p = t["path"]
        ox, oy, _ = t["translate"]
        nx, ny, _ = pos[p]
        (mn, mx) = t["aabb"]
        rx, ry = mn[0] + (nx - ox), mn[1] + (ny - oy)
        role = roles.get(p, "?")
        ax.add_patch(Rectangle((rx, ry), mx[0] - mn[0], mx[1] - mn[1],
                               fc=color.get(role, "#999"), ec="k", lw=0.5, alpha=0.6))
        ax.text(rx, ry, t["class"][:10], fontsize=5)
    ax.set_aspect("equal"); ax.autoscale_view()
    ax.set_title(title, fontsize=9)
    fig.savefig(path_png, dpi=90, bbox_inches="tight")
    plt.close(fig)


_ROLE_COLOR = {"bed": "#c44", "satellite": "#e90", "wall": "#39c",
               "floor": "#3a3", "surface": "#a3a", "fixture": "#999", "?": "#999"}


# face vertex indices into an 8-corner box (z-bottom ring 0-3, z-top ring 4-7).
# The dump captures oriented corners in this same order (see replicator_dataset
# .oriented_corner_offsets) so faces connect correctly under any rotation.
_FACE_IDX = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]


def _faces_from_corners(v):
    return [[v[i] for i in f] for f in _FACE_IDX]


def _box_faces(mn, mx):
    """6 quad faces of an axis-aligned box, for Poly3DCollection."""
    x0, y0, z0 = mn
    x1, y1, z1 = mx
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    return _faces_from_corners(v)


def plot_setting_3d(targets, ctx, rooms, floor_z, pos, path_png, title,
                    ceiling=2.4, walls=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    roles = ctx["roles"]
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # real wall geometry (grey, semi-transparent) so 'on a wall' is verifiable
    for w in walls or []:
        ax.add_collection3d(Poly3DCollection(
            _box_faces(w[0], w[1]), alpha=0.12, facecolor="#777",
            edgecolor="#555", linewidths=0.2))

    xs, ys = [], []
    for r in rooms:
        xs += [r["xmin"], r["xmax"]]
        ys += [r["ymin"], r["ymax"]]
        # room as a wireframe box (floor -> ceiling) so vertical context is clear
        for (a, b) in [((r["xmin"], r["ymin"]), (r["xmax"], r["ymin"])),
                       ((r["xmax"], r["ymin"]), (r["xmax"], r["ymax"])),
                       ((r["xmax"], r["ymax"]), (r["xmin"], r["ymax"])),
                       ((r["xmin"], r["ymax"]), (r["xmin"], r["ymin"]))]:
            for z in (floor_z, floor_z + ceiling):
                ax.plot([a[0], b[0]], [a[1], b[1]], [z, z], color="k", lw=0.6)
            ax.plot([a[0], a[0]], [a[1], a[1]],
                    [floor_z, floor_z + ceiling], color="k", lw=0.4, alpha=0.4)

    for t in targets:
        p = t["path"]
        ox, oy, _ = t["translate"]
        nx, ny, _ = pos[p]
        (mn, mx) = t["aabb"]
        dx, dy = nx - ox, ny - oy
        mn2 = (mn[0] + dx, mn[1] + dy, mn[2])
        mx2 = (mx[0] + dx, mx[1] + dy, mx[2])
        # oriented box from captured corners (true rotation) if available, else
        # the axis-aligned AABB. Corners are offsets from the object's centre.
        xf = t.get("_xform")
        if xf and xf.get("corners"):
            cx, cy = 0.5 * (mn2[0] + mx2[0]), 0.5 * (mn2[1] + mx2[1])
            cz = 0.5 * (mn2[2] + mx2[2])
            verts = [(cx + o[0], cy + o[1], cz + o[2]) for o in xf["corners"]]
            faces = _faces_from_corners(verts)
        else:
            faces = _box_faces(mn2, mx2)
        col = _ROLE_COLOR.get(roles.get(p, "?"), "#999")
        pc = Poly3DCollection(faces, alpha=0.55,
                              facecolor=col, edgecolor="k", linewidths=0.3)
        ax.add_collection3d(pc)
        ax.text(mn2[0], mn2[1], mx2[2], t["class"][:10], fontsize=5)

    ax.set_xlim(min(xs), max(xs)); ax.set_ylim(min(ys), max(ys))
    ax.set_zlim(floor_z, floor_z + ceiling)
    # equal aspect so heights aren't visually distorted
    ax.set_box_aspect((max(xs) - min(xs), max(ys) - min(ys), ceiling))
    ax.view_init(elev=22, azim=-60)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title, fontsize=9)
    fig.savefig(path_png, dpi=95, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", default="ward_layout_example.json")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--plots", default=None, help="dir to write layout PNGs")
    ap.add_argument("--view", choices=("2d", "3d", "both"), default="both",
                    help="which plot(s) to write per setting (default both)")
    ap.add_argument("--ceiling", type=float, default=2.4,
                    help="room height (m) for the 3D wireframe box")
    # jitter params -- match what you pass to the renderer so the test validates
    # the SAME configuration you'll render
    ap.add_argument("--max-shift", type=float, default=0.8)
    ap.add_argument("--wall-slide", type=float, default=0.2)
    ap.add_argument("--sat-slide", type=float, default=0.5)
    ap.add_argument("--global-frac", type=float, default=0.35)
    args = ap.parse_args()

    spec = slot_layout.load_spec(args.spec)
    walls = [((w[0][0], w[0][1], w[0][2]), (w[1][0], w[1][1], w[1][2]))
             for w in spec.get("walls", [])] or None
    if walls is None:
        print("[warn] spec has no 'walls' -- wall-adherence/penetration can't be "
              "checked (re-dump with the updated replicator_dataset.py to add them)")
    targets, rooms, floor_z, dropped = slot_layout.populate(spec, seed=args.seed)
    for d in dropped:
        print(f"[populate] DROPPED: {d}")
    ctx = pdr.build_ctx(targets, rooms, floor_z, verbose=False)

    # advisory: where a declared per-instance affordance disagrees with the
    # class default in placement_affordances (e.g. a gas_manifold declared
    # 'floor' though the class is usually wall-mounted) -- check it's intended
    from layout import affordances as paff
    for t in targets:
        decl = t.get("affordance")
        if decl and decl != paff.support_of(t["class"]):
            print(f"[affordance] {t['class']} declared '{decl}' but class default "
                  f"is '{paff.support_of(t['class'])}' -- verify it's correct")

    # grounding: every floor/surface object must rest on something (declared
    # affordance), else it's floating -- the analytic "would it fall?" check
    grounding = pdr.validate_grounding(targets, floor_z)
    if grounding:
        print(f"[grounding] {len(grounding)} unsupported object(s) — would fall:")
        for path, cls, why in grounding:
            print(f"   {cls:20s} {why}  ({path})")
    else:
        print("[grounding] all floor/surface objects rest on something")

    seq = pdr.generate(targets, rooms, floor_z, n_frames=args.frames,
                       seed=args.seed, verbose=False, max_shift=args.max_shift,
                       wall_slide=args.wall_slide, sat_slide=args.sat_slide,
                       global_frac=args.global_frac,
                       wall_boxes=list(walls) if walls else None)
    frames = [{p: seq[p][f] for p in seq} for f in range(args.frames)]

    if args.plots:
        Path(args.plots).mkdir(parents=True, exist_ok=True)

    total = 0
    for f, pos in enumerate(frames):
        vs = check_setting(targets, ctx, rooms, floor_z, pos, f, walls=walls)
        total += len(vs)
        tag = "OK" if not vs else f"{len(vs)} ISSUE(S)"
        print(f"--- setting {f}: {tag} ---")
        for s in vs:
            print("  " + s)
        if args.plots:
            base = Path(args.plots) / f"setting_{f:02d}"
            if args.view in ("2d", "both"):
                plot_setting(targets, ctx, rooms, pos,
                             f"{base}.png", f"setting {f} ({tag})")
            if args.view in ("3d", "both"):
                plot_setting_3d(targets, ctx, rooms, floor_z, pos,
                                f"{base}_3d.png", f"setting {f} 3D ({tag})",
                                ceiling=args.ceiling, walls=walls)
    print(f"\n[summary] {args.frames} settings, {total} total violation(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
