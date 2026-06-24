"""Procedural shell construction: room polygons -> real walls/floors/ceilings.

This is our own re-implementation of the part of Infinigen-Indoors that turns an
abstract floor plan into solid geometry -- its `BlueprintSolidifier` (extrude the
room polygons into walls, boolean-cut the door/window openings) driven by the
`PredefinedFloorPlanSolver` JSON contract (room rectangles + door/window
segments). See docs/infinigen_procedural_report.md  2 / 6.

Why we need it: until now "rooms" were centroid-derived AABBs and "walls" were
reverse-engineered from the authored Isaac geometry. That is backwards -- the
shell should be GENERATED first, then furniture solved into it. shell.py builds
the shell procedurally and hands the solver exactly the structures it already
consumes:

    rooms      : [{name, type, xmin, ymin, xmax, ymax}]   (interior floor rects)
    wall_boxes : [((minx,miny,minz),(maxx,maxy,maxz))]    (solid full-height runs)
    floor_z    : float

so layout.solver.generate(...) runs against a procedural shell with ZERO change.

Geometry is pure Python (no pxr) so it is unit-testable on any host, exactly like
solver.py. USD emission (emit_usd) imports pxr lazily and runs inside Isaac Sim.

Openings (doors/windows) are NOT booleans: USD has no cheap boolean here and we
don't need one. A wall run with a hole is emitted as side panels flanking the
gap + a lintel above (+ a sill below, for windows) -- the same rectangular-panel
decomposition you would build by hand. The solver still sees the run as a solid
full-height obstacle (so nothing crosses a wall); doorway passability is handled
by the opening clearances we also return.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

# default solidify parameters -- mirror Infinigen's ranges
#   wall_height ~ U(2.8, 3.2)   wall_thickness ~ U(0.2, 0.3)
WALL_HEIGHT = 3.0
WALL_THICKNESS = 0.2
FLOOR_SLAB = 0.05          # floor/ceiling slab thickness (m)
DOOR_WIDTH = 0.95
DOOR_HEIGHT = 2.1
WINDOW_WIDTH = 1.2
WINDOW_HEIGHT = 1.2
WINDOW_SILL = 0.9          # window bottom above floor
DOOR_CLEARANCE = 0.6       # keep-out depth in front of a doorway (m)


# ----------------------------------------------------------------- spec model
@dataclass
class Opening:
    """A door or window on a wall LINE (axis-aligned).

    `line` is the fixed coord of the wall it sits on (an x for a vertical wall,
    a y for a horizontal wall); `orient` matches the wall ('h' runs along x,
    'v' runs along y); `center` is the position ALONG the run; width/height/sill
    size the hole. `kind` is 'door' or 'window'."""
    kind: str
    orient: str            # 'h' | 'v'
    line: float
    center: float
    width: float
    height: float
    sill: float = 0.0      # bottom of the hole above floor (0 for doors)

    @property
    def lo(self):
        return self.center - 0.5 * self.width

    @property
    def hi(self):
        return self.center + 0.5 * self.width


@dataclass
class Room:
    name: str
    type: str
    rect: tuple            # (x0, y0, x1, y1) interior floor extent

    @property
    def cx(self):
        return 0.5 * (self.rect[0] + self.rect[2])

    @property
    def cy(self):
        return 0.5 * (self.rect[1] + self.rect[3])


@dataclass
class Shell:
    """The solidified result. `panels` are 3D boxes for USD; the rest is the
    solver contract."""
    rooms: list                       # solver room dicts
    wall_boxes: list                  # solid full-height AABBs (solver obstacles)
    floor_z: float
    wall_height: float
    wall_thickness: float
    panels: list = field(default_factory=list)     # {kind,name,box=(mn,mx)}
    openings: list = field(default_factory=list)    # Opening, with world clearance
    _rooms_in: list = field(default_factory=list)   # source Room objects

    def door_keepouts(self):
        """AABB keep-outs in front of every door, for the solver to avoid (so
        furniture never blocks a doorway). -> [((mnx,mny,mnz),(mxx,mxy,mxz))]."""
        outs = []
        t = self.wall_thickness
        z0, z1 = self.floor_z, self.floor_z + DOOR_HEIGHT
        for op in self.openings:
            if op.kind != "door":
                continue
            cl = DOOR_CLEARANCE
            if op.orient == "h":               # wall runs along x -> clear in y
                outs.append(((op.lo, op.line - t - cl, z0),
                             (op.hi, op.line + t + cl, z1)))
            else:                               # wall runs along y -> clear in x
                outs.append(((op.line - t - cl, op.lo, z0),
                             (op.line + t + cl, op.hi, z1)))
        return outs


# ------------------------------------------------------------- interval union
def _union(intervals):
    """Merge overlapping/touching [lo,hi] intervals -> sorted disjoint list."""
    if not intervals:
        return []
    ivs = sorted(intervals)
    out = [list(ivs[0])]
    for lo, hi in ivs[1:]:
        if lo <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(iv) for iv in out]


def _subtract(run, holes):
    """run=(lo,hi); holes=[(lo,hi)] -> sub-intervals of run NOT covered by holes."""
    segs = [run]
    for hlo, hhi in sorted(holes):
        nxt = []
        for lo, hi in segs:
            if hhi <= lo or hlo >= hi:          # no overlap
                nxt.append((lo, hi))
                continue
            if hlo > lo:
                nxt.append((lo, hlo))
            if hhi < hi:
                nxt.append((hhi, hi))
        segs = nxt
    return [s for s in segs if s[1] - s[0] > 1e-4]


# --------------------------------------------------------------- solidify core
def solidify(rooms, openings=(), wall_height=WALL_HEIGHT,
             wall_thickness=WALL_THICKNESS, floor_z=0.0, verbose=True):
    """Turn room rectangles (+ openings) into a solid Shell.

    rooms    : [Room] or [{name,type,rect}]
    openings : [Opening] or [{kind,orient,line,center,width,height,sill?}]
    """
    rooms = [r if isinstance(r, Room) else Room(**r) for r in rooms]
    openings = [o if isinstance(o, Opening) else Opening(**o) for o in openings]
    t, h = wall_thickness, wall_height
    half = 0.5 * t
    z0 = floor_z

    # 1) every room edge -> a wall segment on a (orient, line) key. A vertical
    #    edge (x fixed) runs along y; a horizontal edge (y fixed) runs along x.
    by_line = defaultdict(list)            # (orient, round(line)) -> [(lo,hi)]
    key_line = {}
    for r in rooms:
        x0, y0, x1, y1 = r.rect
        by_line[("v", round(x0, 3))].append((y0, y1)); key_line[("v", round(x0, 3))] = x0
        by_line[("v", round(x1, 3))].append((y0, y1)); key_line[("v", round(x1, 3))] = x1
        by_line[("h", round(y0, 3))].append((x0, x1)); key_line[("h", round(y0, 3))] = y0
        by_line[("h", round(y1, 3))].append((x0, x1)); key_line[("h", round(y1, 3))] = y1

    # 2) per line, union the segments into solid wall RUNS (shared interior walls
    #    and colinear exterior walls merge automatically).
    runs = []                              # (orient, line, lo, hi)
    for (orient, _k), segs in by_line.items():
        line = key_line[(orient, _k)]
        for lo, hi in _union(segs):
            runs.append((orient, line, lo, hi))

    # 3) bucket openings onto their run.
    op_on = defaultdict(list)
    for op in openings:
        op_on[(op.orient, round(op.line, 3))].append(op)

    wall_boxes, panels = [], []
    for orient, line, lo, hi in runs:
        # solid full-height run -> solver obstacle (keeps furniture from crossing)
        if orient == "h":
            box = ((lo, line - half, z0), (hi, line + half, z0 + h))
        else:
            box = ((line - half, lo, z0), (line + half, hi, z0 + h))
        wall_boxes.append(box)

        ops = [o for o in op_on.get((orient, round(line, 3)), [])
               if lo - 1e-6 <= o.center <= hi + 1e-6]
        holes = [(o.lo, o.hi) for o in ops]
        # side panels = the run minus the opening spans (full height)
        for slo, shi in _subtract((lo, hi), holes):
            panels.append(_panel("wall", f"wall_{orient}", orient, line, half,
                                 slo, shi, z0, z0 + h))
        # lintel above every opening; sill below every window
        for o in ops:
            top = z0 + o.sill + o.height
            if top < z0 + h - 1e-4:
                panels.append(_panel("wall", "lintel", orient, line, half,
                                     o.lo, o.hi, top, z0 + h))
            if o.sill > 1e-4:
                panels.append(_panel("wall", "sill", orient, line, half,
                                     o.lo, o.hi, z0, z0 + o.sill))

    # 4) floor + ceiling slab per room (per-room so types can be re-materialed).
    for r in rooms:
        x0, y0, x1, y1 = r.rect
        panels.append({"kind": "floor", "name": f"floor_{r.name}",
                       "box": ((x0, y0, z0 - FLOOR_SLAB), (x1, y1, z0))})
        panels.append({"kind": "ceiling", "name": f"ceil_{r.name}",
                       "box": ((x0, y0, z0 + h), (x1, y1, z0 + h + FLOOR_SLAB))})

    # 5) solver room dicts.
    room_dicts = [{"name": r.name, "type": r.type,
                   "xmin": r.rect[0], "ymin": r.rect[1],
                   "xmax": r.rect[2], "ymax": r.rect[3]} for r in rooms]

    if verbose:
        nd = sum(1 for o in openings if o.kind == "door")
        nw = sum(1 for o in openings if o.kind == "window")
        print(f"[shell] {len(rooms)} room(s), {len(runs)} wall run(s) -> "
              f"{len(wall_boxes)} solid boxes, {len(panels)} USD panels "
              f"({nd} door(s), {nw} window(s)); h={h} t={t} floor_z={z0}")

    return Shell(rooms=room_dicts, wall_boxes=wall_boxes, floor_z=z0,
                 wall_height=h, wall_thickness=t, panels=panels,
                 openings=openings, _rooms_in=rooms)


def _panel(kind, name, orient, line, half, lo, hi, zlo, zhi):
    if orient == "h":
        box = ((lo, line - half, zlo), (hi, line + half, zhi))
    else:
        box = ((line - half, lo, zlo), (line + half, hi, zhi))
    return {"kind": kind, "name": name, "box": box}


# ------------------------------------------------------------------ USD output
def emit_usd(stage, shell, root="/World/Shell", wall_mat=None,
             floor_mat=None, ceil_mat=None):
    """Author the shell as USD Cube prims under `root`. Lazily imports pxr so the
    module stays importable (and testable) outside Isaac Sim. Returns the root
    prim path. A cube is unit-sized at origin; we scale/translate to each box."""
    from pxr import Usd, UsdGeom, Gf  # noqa: F401  (Isaac-only)

    UsdGeom.Xform.Define(stage, root)
    mats = {"wall": wall_mat, "floor": floor_mat, "ceiling": ceil_mat,
            "lintel": wall_mat, "sill": wall_mat}
    counts = defaultdict(int)
    for pn in shell.panels:
        (mn, mx) = pn["box"]
        size = [mx[i] - mn[i] for i in range(3)]
        ctr = [0.5 * (mn[i] + mx[i]) for i in range(3)]
        nm = pn["name"]
        i = counts[nm]; counts[nm] += 1
        path = f"{root}/{nm}_{i}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*ctr))
        xf.AddScaleOp().Set(Gf.Vec3f(*[max(s, 1e-4) for s in size]))
        mat = mats.get(pn["kind"])
        if mat is not None:
            from pxr import UsdShade
            UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(mat)
    return root


# ------------------------------------------------------------- 2D schematic
def schematic(shell, path):
    """Top-down PNG of the procedural shell: room rects, solid wall runs (grey),
    doors (green gaps), windows (cyan). Mirrors tests/test_layout.py style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(9, 9))
    for r in shell.rooms:
        ax.add_patch(Rectangle((r["xmin"], r["ymin"]),
                               r["xmax"] - r["xmin"], r["ymax"] - r["ymin"],
                               fill=True, alpha=0.10, ec="0.5", lw=0.8))
        ax.text(0.5 * (r["xmin"] + r["xmax"]), 0.5 * (r["ymin"] + r["ymax"]),
                f"{r['name']}\n({r['type']})", ha="center", va="center",
                fontsize=8, color="0.25")
    for pn in shell.panels:
        if pn["kind"] not in ("wall",):
            continue
        (mn, mx) = pn["box"]
        ax.add_patch(Rectangle((mn[0], mn[1]), mx[0] - mn[0], mx[1] - mn[1],
                               fc="0.35", ec="none"))
    for op in shell.openings:
        col = "tab:green" if op.kind == "door" else "tab:cyan"
        if op.orient == "h":
            ax.plot([op.lo, op.hi], [op.line, op.line], color=col, lw=4,
                    solid_capstyle="butt")
        else:
            ax.plot([op.line, op.line], [op.lo, op.hi], color=col, lw=4,
                    solid_capstyle="butt")
    ax.set_aspect("equal"); ax.autoscale_view()
    ax.set_title("procedural shell (grey=walls, green=doors, cyan=windows)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


# --------------------------------------------------------- sample floor plans
def default_ward_plan():
    """A demonstrator plan in the spirit of the real Ward0524: a main ward with
    an en-suite bathroom in a corner and a corridor along one side. Coordinates
    in metres, ward origin at (0,0)."""
    rooms = [
        Room("ward",     "ward",     (0.0, 0.0, 6.0, 5.0)),
        Room("bathroom", "bathroom", (6.0, 0.0, 8.4, 2.6)),
        Room("corridor", "corridor", (0.0, 5.0, 8.4, 6.6)),
    ]
    openings = [
        # ward <-> corridor (door on the shared y=5.0 wall)
        Opening("door", "h", 5.0, 3.0, DOOR_WIDTH, DOOR_HEIGHT),
        # ward <-> bathroom (door on the shared x=6.0 wall)
        Opening("door", "v", 6.0, 1.3, DOOR_WIDTH, DOOR_HEIGHT),
        # corridor exterior door (x=8.4 wall)
        Opening("door", "v", 8.4, 5.8, DOOR_WIDTH, DOOR_HEIGHT),
        # ward exterior windows on the south (y=0) wall
        Opening("window", "h", 0.0, 1.8, WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_SILL),
        Opening("window", "h", 0.0, 4.2, WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_SILL),
    ]
    return rooms, openings


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(description="Procedural shell demo / smoke test")
    ap.add_argument("--out", default=None, help="schematic PNG path")
    ap.add_argument("--solve", action="store_true",
                    help="run layout.solver against the generated shell")
    args = ap.parse_args()

    rooms, openings = default_ward_plan()
    shell = solidify(rooms, openings)
    print(f"[shell] rooms      = {[r['name'] for r in shell.rooms]}")
    print(f"[shell] wall_boxes = {len(shell.wall_boxes)} solid runs")
    print(f"[shell] door keep-outs = {len(shell.door_keepouts())}")

    out = args.out or os.path.join(
        os.path.dirname(__file__), "..", "..", "scratch_shell.png")
    try:
        schematic(shell, out)
        print(f"[shell] schematic -> {os.path.abspath(out)}")
    except Exception as e:                        # matplotlib optional
        print(f"[shell] schematic skipped: {e}")

    if args.solve:
        # tiny synthetic furniture set to prove the solver runs on the shell.
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from layout import solver
        fz = shell.floor_z

        def box(path, cls, room, w, d, hgt, z0):
            r = next(rr for rr in shell.rooms if rr["name"] == room)
            cx, cy = 0.5 * (r["xmin"] + r["xmax"]), 0.5 * (r["ymin"] + r["ymax"])
            return {"path": path, "class": cls, "room": room,
                    "translate": (cx, cy, z0 + hgt / 2), "size": (w, d, hgt),
                    "aabb": ((cx - w / 2, cy - d / 2, z0),
                             (cx + w / 2, cy + d / 2, z0 + hgt))}

        targets = [
            box("/bed", "hospital_bed", "ward", 2.0, 0.9, 0.6, fz),
            box("/bedside", "bedside_table", "ward", 0.5, 0.5, 0.7, fz),
            box("/chair", "chair", "ward", 0.5, 0.5, 0.9, fz),
            box("/bin", "trash_can", "ward", 0.4, 0.4, 0.6, fz),
            box("/toilet", "toilet", "bathroom", 0.5, 0.7, 0.4, fz),
        ]
        paths = solver.generate(targets, shell.rooms, fz, n_frames=1,
                                wall_boxes=shell.wall_boxes, steps=1500)
        print("[shell] solver ran on procedural shell:")
        for p, locs in paths.items():
            x, y, z = locs[0]
            print(f"          {p:12s} -> ({x:5.2f}, {y:5.2f}, {z:5.2f})")
