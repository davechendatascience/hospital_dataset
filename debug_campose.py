"""Offline (no Isaac boot) replication of replicator_dataset.py's camera-pose
sampling, to debug where bad frames' cameras actually sit.

Run with the bundled pxr:
  USDLIBS=~/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.la64.r.cp311
  PYTHONPATH=$USDLIBS LD_LIBRARY_PATH=$USDLIBS/bin:$USDLIBS/lib \
      ~/isaac-sim/kit/python/bin/python3 debug_campose.py 0012 0015 0032 ...
"""
import math
import random
import re
import sys
from pathlib import Path

from pxr import Usd, UsdGeom, Gf

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ROS2_bridge" / "src"))
from fixed_categories import FIXED_CATEGORIES  # noqa: E402

STAGE = str(PROJECT_ROOT / "Collected_Ward0505" / "Ward0505.usd")
SEED = 42
FRAMES = 100
HEIGHT_MIN, HEIGHT_MAX = 1.2, 1.7
MIN_CAM_DISTANCE = 0.5
MAX_OBJ_DIM = 10.0

# ---- import SEMANTIC_RULES + ROOM_NAME_RULES from the real script ----------
src = (PROJECT_ROOT / "replicator_dataset.py").read_text()


def _exec_block(start_marker, end_marker, ns):
    i = src.index(start_marker)
    j = src.index(end_marker, i)
    exec(src[i:j], ns)


ns = {"re": re}
_exec_block("SEMANTIC_RULES = [", "def apply_semantics", ns)
_exec_block("ROOM_NAME_RULES = [", "def build_rooms_from_object_names", ns)
SEMANTIC_RULES = ns["SEMANTIC_RULES"]
ROOM_NAME_RULES = ns["ROOM_NAME_RULES"]
DEFAULT_ROOM = ns["DEFAULT_ROOM"]

stage = Usd.Stage.Open(STAGE)

# ---- mirror deactivate_existing_replicator (affects stage.Traverse sets) ---
to_deactivate = []
for prim in stage.Traverse():
    t = prim.GetTypeName()
    path = str(prim.GetPath())
    if t in ("RenderProduct", "RenderVar"):
        to_deactivate.append(path)
    elif path.startswith("/Replicator") or path.startswith("/OmniGraph"):
        to_deactivate.append(path)
    elif path == "/Graph" or path.startswith("/Graph/"):
        to_deactivate.append(path)
    elif t == "Camera":
        to_deactivate.append(path)
for path in to_deactivate:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid() and prim.IsActive():
        prim.SetActive(False)

bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                           [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

# ---- mirror apply_semantics (matching + AABB only, no Semantics applied) ---
object_targets = []
matched_paths = []
for top in stage.GetPrimAtPath("/World").GetChildren():
    name = top.GetName()
    match = None
    for pattern, cls in SEMANTIC_RULES:
        if pattern.search(name):
            match = cls
            break
    if match is None or match not in FIXED_CATEGORIES:
        continue
    box = bcache.ComputeWorldBound(top).ComputeAlignedBox()
    mn, mx = box.GetMin(), box.GetMax()
    if any(abs(v) > 1e6 for v in (mn[0], mn[1], mn[2], mx[0], mx[1], mx[2])):
        continue
    sx, sy, sz = float(mx[0] - mn[0]), float(mx[1] - mn[1]), float(mx[2] - mn[2])
    if max(sx, sy) > MAX_OBJ_DIM:
        continue
    matched_paths.append(str(top.GetPath()))
    object_targets.append({
        "path": str(top.GetPath()),
        "class": match,
        "centroid": (float((mn[0] + mx[0]) * 0.5),
                     float((mn[1] + mx[1]) * 0.5),
                     float((mn[2] + mx[2]) * 0.5)),
        "aabb": ((float(mn[0]), float(mn[1]), float(mn[2])),
                 (float(mx[0]), float(mx[1]), float(mx[2]))),
        "size": (sx, sy, sz),
    })

# floor_z from object AABB union (mirrors compute_object_aabb path)
floor_z = min(ot["aabb"][0][2] for ot in object_targets)
print(f"{len(object_targets)} labeled targets, floor_z={floor_z:.2f}")

# ---- mirror find_wall_aabbs ------------------------------------------------
wall_re = re.compile(r"wall", re.I)
wall_aabbs = []
for prim in stage.Traverse():
    if not wall_re.search(prim.GetName()):
        continue
    t = prim.GetTypeName()
    if t not in ("Mesh", "Xform", "Scope"):
        continue
    try:
        box = bcache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn, mx = box.GetMin(), box.GetMax()
    except Exception:
        continue
    if any(abs(v) > 1e6 for v in (mn[0], mn[1], mn[2], mx[0], mx[1], mx[2])):
        continue
    sx, sy = float(mx[0] - mn[0]), float(mx[1] - mn[1])
    if t != "Mesh" and min(sx, sy) > 0.6:
        continue
    wall_aabbs.append(((float(mn[0]), float(mn[1]), float(mn[2])),
                       (float(mx[0]), float(mx[1]), float(mx[2]))))
print(f"{len(wall_aabbs)} wall AABBs")


def _point_in_aabb(p, aabb):
    pmin, pmax = aabb
    return (pmin[0] <= p[0] <= pmax[0] and
            pmin[1] <= p[1] <= pmax[1] and
            pmin[2] <= p[2] <= pmax[2])


def _segment_intersects_aabb(p1, p2, aabb):
    eps = 1e-4
    pmin, pmax = aabb
    tmin, tmax = eps, 1.0 - eps
    for i in range(3):
        di = p2[i] - p1[i]
        if abs(di) < 1e-9:
            if p1[i] < pmin[i] or p1[i] > pmax[i]:
                return False
        else:
            t1 = (pmin[i] - p1[i]) / di
            t2 = (pmax[i] - p1[i]) / di
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return True


def _line_of_sight_blocked(cam_pos, target_pos):
    for w in wall_aabbs:
        if _segment_intersects_aabb(cam_pos, target_pos, w):
            return True
    return False


def _inside_any_wall(point):
    for w in wall_aabbs:
        if _point_in_aabb(point, w):
            return True
    return False


def _inside_any_object(point, aabbs, margin=0.10):
    for mn, mx in aabbs:
        if (mn[0] - margin <= point[0] <= mx[0] + margin and
                mn[1] - margin <= point[1] <= mx[1] + margin and
                mn[2] - margin <= point[2] <= mx[2] + margin):
            return True
    return False


# ---- mirror collect_blocker_aabbs -------------------------------------------
def collect_blocker_aabbs(stage, labeled_paths, max_side=4.0, max_depth=3):
    skip = set(labeled_paths)
    out = []
    info = []

    def visit(prim, depth):
        if str(prim.GetPath()) in skip:
            return
        if prim.GetTypeName() not in ("Mesh", "Xform", "Scope", ""):
            return
        try:
            box = bcache.ComputeWorldBound(prim).ComputeAlignedBox()
            mn, mx = box.GetMin(), box.GetMax()
        except Exception:
            return
        if any(abs(v) > 1e6 for v in (mn[0], mn[1], mn[2], mx[0], mx[1], mx[2])):
            return
        sx, sy = float(mx[0] - mn[0]), float(mx[1] - mn[1])
        if max(sx, sy) <= max_side:
            out.append(((float(mn[0]), float(mn[1]), float(mn[2])),
                        (float(mx[0]), float(mx[1]), float(mx[2]))))
            info.append(str(prim.GetPath()))
        elif depth < max_depth:
            for child in prim.GetChildren():
                visit(child, depth + 1)

    for top in stage.GetPrimAtPath("/World").GetChildren():
        visit(top, 0)
    return out, info


blocker_aabbs, blocker_paths = collect_blocker_aabbs(stage, matched_paths)
no_go_aabbs = [ot["aabb"] for ot in object_targets] + blocker_aabbs
no_go_paths = [ot["path"] for ot in object_targets] + blocker_paths
print(f"{len(no_go_aabbs)} no-go AABBs ({len(object_targets)} labeled + "
      f"{len(blocker_aabbs)} unlabeled)")


# ---- mirror build_rooms_from_object_names -----------------------------------
def assign_room(prim_name):
    for pattern, room in ROOM_NAME_RULES:
        if pattern.search(prim_name):
            return room
    return DEFAULT_ROOM


grouped = {"Ward": [], "Bathroom": [], "Frontroom": []}
for obj in object_targets:
    grouped.setdefault(assign_room(obj["path"].rsplit("/", 1)[-1]), []).append(obj)
rooms = []
for room_name, members in grouped.items():
    if not members:
        continue
    xs = [m["centroid"][0] for m in members]
    ys = [m["centroid"][1] for m in members]
    rooms.append({
        "name": room_name, "members": members,
        "xmin": min(xs) - 0.5, "xmax": max(xs) + 0.5,
        "ymin": min(ys) - 0.5, "ymax": max(ys) + 0.5,
    })
for r in rooms:
    print(f"  {r['name']:10s} {len(r['members'])} members  "
          f"X[{r['xmin']:.2f}..{r['xmax']:.2f}] Y[{r['ymin']:.2f}..{r['ymax']:.2f}]")


# ---- mirror sample_camera_pose ----------------------------------------------
def sample_camera_pose(rng):
    rooms_with_members = [r for r in rooms if r["members"]]
    weights = [len(r["members"]) for r in rooms_with_members]
    room = rng.choices(rooms_with_members, weights=weights, k=1)[0]
    target_obj = rng.choice(room["members"])
    target = target_obj["centroid"]
    pos = None
    rejections = []
    for _ in range(50):
        cam_x = rng.uniform(room["xmin"], room["xmax"])
        cam_y = rng.uniform(room["ymin"], room["ymax"])
        cam_z = floor_z + rng.uniform(HEIGHT_MIN, HEIGHT_MAX)
        cand = (cam_x, cam_y, cam_z)
        d2 = (target[0] - cam_x) ** 2 + (target[1] - cam_y) ** 2
        if d2 < MIN_CAM_DISTANCE ** 2:
            pos = cand
            rejections.append("too_close")
            continue
        if _inside_any_wall(cand) or _inside_any_object(cand, no_go_aabbs):
            pos = cand
            rejections.append("inside")
            continue
        if _line_of_sight_blocked(cand, target):
            pos = cand
            rejections.append("los")
            continue
        pos = cand
        break
    jit = 0.15
    look = (target[0] + rng.uniform(-jit, jit),
            target[1] + rng.uniform(-jit, jit),
            target[2] + rng.uniform(-jit, jit))
    return pos, look, room["name"], target_obj, rejections


rng = random.Random(SEED)
poses = [sample_camera_pose(rng) for _ in range(FRAMES)]

n_rej = sum(1 for p in poses if p[4])
n_exhausted = sum(1 for p in poses if len(p[4]) >= 50)
print(f"\n{n_rej}/{FRAMES} frames needed at least one resample; "
      f"{n_exhausted} exhausted all 50 attempts (unchecked fallback)")
from collections import Counter
reasons = Counter(r for p in poses for r in p[4])
print(f"rejection reasons: {dict(reasons)}")
# fattest blockers (potential over-rejectors)
fat = sorted(zip(blocker_paths, blocker_aabbs),
             key=lambda b: -(b[1][1][0]-b[1][0][0]) * (b[1][1][1]-b[1][0][1]))[:8]
print("fattest unlabeled blockers (XY footprint):")
for path, (mn, mx) in fat:
    print(f"   ({mx[0]-mn[0]:5.2f} x {mx[1]-mn[1]:5.2f} x {mx[2]-mn[2]:5.2f}) {path}")

# ---- containment report for requested frames --------------------------------
ask = sys.argv[1:] or []
for idx in ask:
    i = int(idx)
    pos, look, room_name, tobj, rejections = poses[i]
    print(f"\n=== frame {idx} ===")
    print(f"  room={room_name}  target={tobj['class']} @ {tobj['path']}")
    print(f"  cam=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  "
          f"look=({look[0]:.2f}, {look[1]:.2f}, {look[2]:.2f})")
    print(f"  rejections during sampling: {rejections or 'none'}")
    # What contains the camera? Walk EVERY prim (depth<=4) and test its AABB.
    containers = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        depth = path.count("/")
        if depth > 4:
            continue
        if prim.GetTypeName() not in ("Mesh", "Xform", "Scope", ""):
            continue
        try:
            box = bcache.ComputeWorldBound(prim).ComputeAlignedBox()
            mn, mx = box.GetMin(), box.GetMax()
        except Exception:
            continue
        if any(abs(v) > 1e6 for v in (mn[0], mn[1], mn[2], mx[0], mx[1], mx[2])):
            continue
        if _point_in_aabb(pos, ((mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2]))):
            sx, sy, sz = mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]
            containers.append((path, prim.GetTypeName(), sx, sy, sz))
    print(f"  contained by {len(containers)} prim AABB(s):")
    for path, t, sx, sy, sz in containers[:15]:
        print(f"     {t:6s} ({sx:5.2f} x {sy:5.2f} x {sz:5.2f})  {path}")
