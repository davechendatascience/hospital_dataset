"""
Standalone Isaac Sim Replicator dataset generator for Ward0505.

Run with:
    ~/isaac-sim/python.sh replicator_dataset.py \
        --stage Collected_Ward0505/Ward0505.usd \
        --out  output_dataset \
        --frames 200

The goal of this script is to fix the sim->real composition gap by:
  - using a webcam-like camera (70 deg HFOV, 1080p) instead of close-up macro shots
  - placing the camera at hand-held height inside the room and walking a random
    trajectory (the original test set was recorded with a hand-held webcam,
    `WIN_<timestamp>_Pro_frame_*.jpg`)
  - randomizing ceiling lighting per frame (temperature + intensity)
  - muting the per-asset DomeLights and the `Grey_Studio` DistantLight that
    produce the synthetic "studio render" look
  - trusting the semantics AUTHORED in the stage (Semantic Schema Editor);
    labels are only normalized onto fixed_categories.FIXED_CATEGORIES at COCO
    conversion (case fixes + LABEL_ALIASES), never rewritten on prims
  - writing in the same `rgbDataset/<tag>_rgb/` + `jsonDataset/<tag>.json` layout
    expected by ROS2_bridge/src/from_ward_to_roboflow_dataset.py
"""

# argparse must run BEFORE SimulationApp starts, otherwise -h becomes useless.
import argparse
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_STAGE = str(PROJECT_ROOT / "Collected_Ward0505" / "Ward0505.usd")
DEFAULT_OUT   = str(PROJECT_ROOT / "Ward_dataset_v2")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",  default=DEFAULT_STAGE, help="USD scene to render")
    p.add_argument("--out",    default=DEFAULT_OUT,   help="Output dataset root")
    p.add_argument("--tag",    default="ward",        help="Tag used in output paths")
    p.add_argument("--frames", type=int, default=200, help="Total frames to generate")
    p.add_argument("--resolution", nargs=2, type=int, default=[1920, 1080],
                   metavar=("W", "H"), help="Render resolution (test set is 1920x1080)")
    p.add_argument("--hfov", type=float, default=70.0,
                   help="Horizontal FOV in degrees (test webcam is ~70)")
    p.add_argument("--height-min", type=float, default=1.2,
                   help="Minimum camera height (meters)")
    p.add_argument("--height-max", type=float, default=1.7,
                   help="Maximum camera height (meters)")
    p.add_argument("--rt-subframes", type=int, default=8,
                   help="Path-trace sub-frames per render (higher = cleaner, slower)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", dest="headless", action="store_false",
                   help="Open the Isaac Sim GUI (debug only; very slow)")
    p.add_argument("--debug-stage", action="store_true",
                   help="After setup, dump every active light + camera + render product")
    p.add_argument("--max-obj-dim", type=float, default=10.0,
                   help="Reject prims whose world AABB has any horizontal side "
                        "longer than this (m). Filters 'contaminated' assets "
                        "whose imported geometry accidentally spans the scene.")
    p.add_argument("--min-cam-distance", type=float, default=0.5,
                   help="Min camera-to-target distance after sampling (m).")
    p.add_argument("--floor-grid-size", type=float, default=0.10,
                   help="Cell size (m) for the 2D floor plan that walls are "
                        "rasterized into. Smaller = finer room boundaries.")
    p.add_argument("--wall-dilation-cells", type=int, default=1,
                   help="Dilate wall pixels by this many cells before flood "
                        "fill. Closes thin wall gaps that would otherwise "
                        "leak between rooms; 1 cell = `--floor-grid-size` m.")
    p.add_argument("--room-min-area", type=float, default=2.0,
                   help="Discard connected regions smaller than this (m^2). "
                        "Filters out tiny slivers between meshes.")
    p.add_argument("--save-layout-png", action="store_true", default=True,
                   help="Write <out>/_room_layout.png so you can verify the "
                        "detected rooms visually.")
    # --- Online material domain-randomization (perturb originals) ---
    p.add_argument("--randomize-materials", action="store_true", default=False,
                   help="Per-frame perturb the labeled objects' ORIGINAL "
                        "materials in place: per-channel tint on the baked "
                        "diffuse texture, colour jitter for untextured shaders, "
                        "and roughness re-rolls. Objects keep their identity; "
                        "geometry is untouched, so masks/boxes stay valid. "
                        "Off by default.")
    p.add_argument("--randomize-placement", action="store_true", default=False,
                   help="Per-frame meaningful placement DR: free-standing furniture "
                        "(bed + overbed table + IV pole + chair + bins) moves as "
                        "rigid proximity CLUSTERS to new collision-free spots on the "
                        "floor (orientation preserved), surface items ride their "
                        "cluster, wall-mounted objects slide along their wall, "
                        "fixtures stay. Objects stay in their own room. See placement_dr.py.")
    p.add_argument("--placement-shift", type=float, default=0.8,
                   help="Max XY translation (m) per cluster for --randomize-placement.")
    p.add_argument("--placement-seed", type=int, default=0,
                   help="Seed for placement DR (default: --seed).")
    p.add_argument("--placement-global-frac", type=float, default=0.35,
                   help="Probability that a floor item is re-placed ANYWHERE "
                        "in its room (collision-checked) instead of jittered "
                        "near its original spot. 0 = local-only (v2 behavior).")
    p.add_argument("--extra-channels", action="store_true", default=False,
                   help="Also export ground-truth DEPTH (distance_to_camera + "
                        "distance_to_image_plane), surface NORMALS, and stable "
                        "INSTANCE-ID segmentation alongside RGB. Pixel-aligned "
                        "control channels for style transfer (depth-conditioned "
                        "CUT / ControlNet). Off by default; RGB/labels unchanged.")
    p.add_argument("--cosmos", action="store_true", default=False,
                   help="Use CosmosWriter instead of BasicWriter: export clip-based "
                        "multimodal control data (rgb/depth/segmentation/edges/"
                        "shaded_seg + mp4s) for NVIDIA Cosmos-Transfer sim2real, with "
                        "use_instance_id=True for cross-frame object identity. Skips "
                        "the COCO post-step (Cosmos output is clip-structured). "
                        "Implies --trajectory (Cosmos is a video model).")
    p.add_argument("--trajectory", action="store_true", default=False,
                   help="Render a SMOOTH camera fly-through (Catmull-Rom spline through "
                        "valid keyframe poses) instead of independent random poses, so "
                        "frames form a temporally-coherent clip. Auto-on with --cosmos.")
    p.add_argument("--traj-keys", type=int, default=6,
                   help="Number of valid keyframe poses anchoring the trajectory spline.")
    return p.parse_args()


args = parse_args()

# ---- Boot Isaac Sim ----
from isaacsim import SimulationApp  # noqa: E402
sim_app = SimulationApp({"headless": args.headless})

# ---- All Isaac/Omni imports MUST be after SimulationApp() ----
import json                                           # noqa: E402
import random                                         # noqa: E402
import datetime                                       # noqa: E402
from collections import Counter                       # noqa: E402
import numpy as np                                    # noqa: E402
import omni.usd                                       # noqa: E402
import omni.replicator.core as rep                    # noqa: E402
from pxr import Usd, UsdGeom, UsdLux, Sdf, Gf         # noqa: E402

# Load your taxonomy so the same class names are used everywhere.
sys.path.insert(0, str(PROJECT_ROOT / "ROS2_bridge" / "src"))
from fixed_categories import (                        # noqa: E402
    FIXED_CATEGORIES, normalize_label, class_from_entry)

random.seed(args.seed)
np.random.seed(args.seed)

# ============================================================================
# Step 1 — open stage
# ============================================================================
# Relative stage paths make omni.usd double-anchor relative payload refs
# (Collected_Ward0505/Collected_Ward0505/...) -> assets silently fail to load
# and every object renders empty/unlabeled. Always open by absolute path.
args.stage = str(Path(args.stage).resolve())
print(f"[boot] opening stage: {args.stage}")
ctx = omni.usd.get_context()
ok = ctx.open_stage(args.stage)
if not ok:
    print(f"ERROR: failed to open stage {args.stage}")
    sim_app.close()
    sys.exit(1)
stage: Usd.Stage = ctx.get_stage()


# ============================================================================
# Step 2 — clean up the existing Replicator + per-asset lights
# Your stage has 269 RenderProduct prims and 73 per-asset DomeLights left over
# from earlier per-object capture sessions. We deactivate them before adding
# our own.
# ============================================================================
def deactivate_existing_replicator(stage: Usd.Stage) -> int:
    n = 0
    for prim in stage.Traverse():
        t = prim.GetTypeName()
        path = str(prim.GetPath())
        # Old render products / vars / writers leftover from previous sessions
        if t in ("RenderProduct", "RenderVar"):
            prim.SetActive(False)
            n += 1
        # Replicator graph nodes
        elif path.startswith("/Replicator") or path.startswith("/OmniGraph"):
            prim.SetActive(False)
            n += 1
        # ROS_Camera / PushGraph OmniGraphs left in your USD: they reference
        # missing nodes and spam Could-not-find-OmniGraph-node warnings every
        # frame. Deactivating them is safe — we don't need ROS publishing here.
        elif path == "/Graph" or path.startswith("/Graph/"):
            prim.SetActive(False)
            n += 1
        # Hide every pre-existing Camera prim so its wireframe gizmo doesn't
        # show up in renders. Our actual render camera is created LATER by
        # rep.create.camera() and will be unaffected by this.
        elif t == "Camera":
            prim.SetActive(False)
            try:
                UsdGeom.Imageable(prim).MakeInvisible()
            except Exception:
                pass
            n += 1
    return n


def mute_studio_and_per_asset_lights(stage: Usd.Stage, ambient_dome_intensity=120.0):
    """Tone down the lights that gave the scene a "Grey Studio" look while
    KEEPING the room's actual lighting at its authored intensity (so the
    camera isn't shooting in the dark). The rule is naming-AGNOSTIC -- it works
    across scene versions whose ceiling fixtures are named differently
    (Ward0505 used /World/RectLight*; Ward0524 uses /World/Ceil*,
    /World/DiskLight*, /World/Hospital_SunLight, ...):

      - any TOP-LEVEL /World light (direct child, e.g. /World/DiskLight,
        /World/Hospital_SunLight, /World/RectLight) is the room's intended
        lighting -> KEPT at its authored intensity.
      - /Environment/Grey_Studio/DomeLight -> reduced to `ambient_dome_intensity`
        (soft ambient fill).
      - /Environment/Grey_Studio/DistantLight -> muted (synthetic studio sun).
      - nested per-asset lights (/World/<asset>/.../env_light DomeLights left
        over from per-object capture) -> muted to 0.
    """
    muted = []
    kept = []
    ambient = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.GetTypeName() not in (
            "DomeLight", "DistantLight", "RectLight", "DiskLight",
            "SphereLight", "CylinderLight",
        ):
            continue
        # Top-level /World light = a direct child of /World (depth 2). That is
        # the room lighting in every scene version; keep it as authored.
        if path.startswith("/World/") and path.count("/") == 2:
            kept.append(path)
            continue
        light = UsdLux.LightAPI(prim)
        try:
            if path.endswith("Grey_Studio/DomeLight"):
                light.GetIntensityAttr().Set(float(ambient_dome_intensity))
                ambient.append(path)
            else:
                # Grey_Studio DistantLight + nested per-asset env_lights -> zero
                light.GetIntensityAttr().Set(0.0)
                muted.append(path)
        except Exception as e:
            print(f"  (could not adjust {path}: {e})")
    return muted, kept, ambient


print("[clean] deactivating leftover render products + replicator graphs")
n_dead = deactivate_existing_replicator(stage)
print(f"  deactivated {n_dead} stale prims")

print("[clean] muting per-asset env_lights and Grey_Studio DistantLight")
muted, kept, ambient = mute_studio_and_per_asset_lights(stage)
KEPT_LIGHTS = kept            # room lights -> per-frame color/exposure DR below
print(f"  muted {len(muted)} lights")
print(f"  kept {len(kept)} room lights (authored intensity): {kept}")
print(f"  ambient fill at low intensity: {ambient}")
if not KEPT_LIGHTS:
    print("  WARNING: no top-level /World lights kept -> scene may render dark!")


# ============================================================================
# Step 3 — derive the actual room bounds from /CollisionMesh
# Whole-stage AABB is dominated by infinite dome lights so we use a smaller
# prim that's a reliable proxy for the floor + walls.
# ============================================================================
def world_bounds_of(stage: Usd.Stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                               [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = bcache.ComputeWorldBound(prim).ComputeAlignedBox()
    return box.GetMin(), box.GetMax()


room_bounds = world_bounds_of(stage, "/CollisionMesh")
if room_bounds is None:
    # Fallback: compute over /World and trim z-extent to a plausible ceiling
    room_bounds = world_bounds_of(stage, "/World")
    if room_bounds is None:
        print("ERROR: cannot determine room bounds")
        sim_app.close()
        sys.exit(1)

bmin, bmax = room_bounds
print(f"[room] AABB min=({bmin[0]:.2f}, {bmin[1]:.2f}, {bmin[2]:.2f})  "
      f"max=({bmax[0]:.2f}, {bmax[1]:.2f}, {bmax[2]:.2f})")
print(f"[room] size=({bmax[0]-bmin[0]:.2f}, {bmax[1]-bmin[1]:.2f}, "
      f"{bmax[2]-bmin[2]:.2f}) m")


# ============================================================================
# Step 4 — collect the semantics AUTHORED in the stage. The ward USD carries
# hand-set labels (Semantic Schema Editor, new UsdSemantics
# `semantics:labels:*` API) on ~100 prims, including NESTED ones (cabinet
# doors, toilet grab bars). Those are the single source of truth: nothing is
# written here, Replicator reads them natively. We only gather camera /
# placement targets from the labeled TOP-LEVEL prims and report anything
# that doesn't normalize into fixed_categories.py.
# ============================================================================
import re                                              # noqa: E402


def authored_labels(prim) -> list[str]:
    """All semantic labels authored on this prim (new + legacy API)."""
    out = []
    for attr in prim.GetAttributes():
        n = attr.GetName()
        if n.startswith("semantics:labels:"):
            out += [str(v) for v in (attr.Get() or [])]
        elif n.startswith("semantic:") and n.endswith(":params:semanticData"):
            out += [t.strip() for t in str(attr.Get() or "").split(",")
                    if t.strip()]
    return out


def bridge_to_class_taxonomy(prim, labels) -> None:
    """Mirror the authored label onto the legacy 'class' taxonomy IN-SESSION.

    The Semantic Schema Editor wrote `semantics:labels:<instance>` with
    per-object instance names ("Mirror", "handle1", ...) -- NOT the 'class'
    taxonomy that Replicator's annotators filter on by default, so every
    hand-set label is invisible to the writer. (Feeding all ~94 taxonomies
    via BasicWriter(semantic_types=...) sets a giant global filter predicate
    that crashes this build.) The label VALUE is copied verbatim; the stage
    is never saved, so the authored labels stay the source of truth."""
    from pxr import Semantics  # type: ignore
    sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    sem.CreateSemanticTypeAttr().Set("class")
    sem.CreateSemanticDataAttr().Set(",".join(labels))


def collect_semantics(stage: Usd.Stage):
    counts = {}
    unmatched_top = []     # top prims with no authored label anywhere
    nested_only = []       # top prims labeled only in their subtree
    off_taxonomy = []      # (path, raw_label) that normalize_label can't map
    matched_paths = []     # for room-AABB / debugging
    object_targets = []    # list of dict(path, class, centroid, aabb, size)
    oversized = []         # rejected for MAX_OBJ_DIM filter

    bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                               [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    # Full sweep: bridge every authored label (including NESTED ones --
    # cabinet doors, grab bars) onto the 'class' taxonomy the writer reads.
    bridged = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World"),
                           Usd.TraverseInstanceProxies()):
        labs = authored_labels(p)
        if labs:
            bridge_to_class_taxonomy(p, labs)
            bridged += 1
    print(f"[semantics] bridged {bridged} authored labels onto the 'class' "
          f"taxonomy (in-session only; stage not saved)")

    for top in stage.GetPrimAtPath("/World").GetChildren():
        labels = authored_labels(top)
        if not labels:
            # No label on the top prim itself; nested labels (cabinet doors,
            # grab bars, wardwall/window) still render + convert natively —
            # they just aren't camera/placement targets.
            for p in Usd.PrimRange(top, Usd.TraverseInstanceProxies()):
                if p != top and authored_labels(p):
                    nested_only.append(top.GetName())
                    break
            else:
                unmatched_top.append(top.GetName())
            continue
        match = None
        for lab in labels:
            match = normalize_label(lab)
            if match is not None:
                break
        path = str(top.GetPath())
        if match is None:
            off_taxonomy.append((path, ",".join(labels)))
            continue
        # World AABB sanity-filter (contaminated assets would make terrible
        # camera targets / placement members).
        try:
            box = bcache.ComputeWorldBound(top).ComputeAlignedBox()
            mn, mx = box.GetMin(), box.GetMax()
            if any(abs(v) > 1e6 for v in (mn[0], mn[1], mn[2], mx[0], mx[1], mx[2])):
                oversized.append((path, match, "infinite/degenerate"))
                continue
            sx, sy, sz = float(mx[0] - mn[0]), float(mx[1] - mn[1]), float(mx[2] - mn[2])
            if max(sx, sy) > args.max_obj_dim:
                oversized.append((path, match, f"({sx:.1f} x {sy:.1f} x {sz:.1f}) m"))
                continue
        except Exception as e:
            oversized.append((path, match, f"AABB failed: {e}"))
            continue
        counts[match] = counts.get(match, 0) + 1
        matched_paths.append(path)
        centroid = (
            float((mn[0] + mx[0]) * 0.5),
            float((mn[1] + mx[1]) * 0.5),
            float((mn[2] + mx[2]) * 0.5),
        )
        object_targets.append({
            "path": path,
            "class": match,
            "centroid": centroid,
            "aabb": ((float(mn[0]), float(mn[1]), float(mn[2])),
                     (float(mx[0]), float(mx[1]), float(mx[2]))),
            "size": (sx, sy, sz),
        })

    if nested_only:
        print(f"[semantics] {len(nested_only)} top prims labeled only in their "
              f"subtree (rendered, not camera targets): "
              f"{', '.join(sorted(nested_only))}")
    if off_taxonomy:
        print(f"[semantics] {len(off_taxonomy)} top prims with labels outside "
              f"the taxonomy (rendered but DROPPED at COCO conversion):")
        for path, raw in off_taxonomy:
            print(f"     {raw:24s} {path}")
    return counts, unmatched_top, matched_paths, object_targets, oversized


print("[semantics] reading authored class labels from the stage (/World)")
counts, unmatched, matched_paths, object_targets, oversized = collect_semantics(stage)
for k, v in sorted(counts.items()):
    print(f"  {v:3d} x {k}")
if unmatched:
    print(f"  unmatched top-level prims ({len(unmatched)}): "
          f"{', '.join(unmatched[:20])}{'...' if len(unmatched) > 20 else ''}")
if oversized:
    print(f"  [SKIPPED] {len(oversized)} prim(s) failed --max-obj-dim "
          f"({args.max_obj_dim} m) sanity filter:")
    for path, cls, reason in oversized[:20]:
        print(f"     {path}  class={cls}  reason={reason}")

# Sanity dump: every labeled target with its size, so the user can spot
# 'contaminated' assets (e.g. a hospital_bed whose mesh tree includes an
# overbed_table baked in -> larger than expected).
print("[semantics] per-target world AABB sizes (sorted by max dim desc):")
for ot in sorted(object_targets, key=lambda o: -max(o["size"])):
    sx, sy, sz = ot["size"]
    print(f"     {ot['class']:22s} size=({sx:5.2f} x {sy:5.2f} x {sz:5.2f}) m  "
          f"{ot['path']}")

# Overlap diagnostic: list pairs of labeled targets whose 3D AABBs overlap
# substantially. Adjacent objects (bed + bedside table) will trip lightly; a
# bed and an over-bed table sharing 80%+ of their volume usually means the
# asset is duplicated or two assets are stacked at the same world transform.
def _aabb_overlap_ratio(a, b):
    """Volume(a ∩ b) / min(Volume(a), Volume(b))."""
    (amn, amx), (bmn, bmx) = a, b
    ix0, iy0, iz0 = max(amn[0], bmn[0]), max(amn[1], bmn[1]), max(amn[2], bmn[2])
    ix1, iy1, iz1 = min(amx[0], bmx[0]), min(amx[1], bmx[1]), min(amx[2], bmx[2])
    if ix1 <= ix0 or iy1 <= iy0 or iz1 <= iz0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0) * (iz1 - iz0)
    va = max((amx[0]-amn[0]) * (amx[1]-amn[1]) * (amx[2]-amn[2]), 1e-9)
    vb = max((bmx[0]-bmn[0]) * (bmx[1]-bmn[1]) * (bmx[2]-bmn[2]), 1e-9)
    return inter / min(va, vb)


print("[semantics] labeled-asset overlap (>= 50% volume sharing -> likely "
      "duplicate/contaminated asset):")
flagged_any = False
for i in range(len(object_targets)):
    for j in range(i + 1, len(object_targets)):
        r = _aabb_overlap_ratio(object_targets[i]["aabb"], object_targets[j]["aabb"])
        if r >= 0.5:
            print(f"     {object_targets[i]['path']:60s} vs "
                  f"{object_targets[j]['path']:60s}  overlap={r:.2f}")
            flagged_any = True
if not flagged_any:
    print("     (none; assets look clean)")


# ============================================================================
# Step 3b — derive room INTERIOR bounds from the labeled object prims.
# /CollisionMesh from step 3 covers the global collision world (way too big).
# Hospital objects all live inside the room, so the AABB containing them is a
# better approximation of where the camera should sample positions.
# ============================================================================
def compute_object_aabb(stage, prim_paths):
    bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                               [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    mins, maxs = [], []
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        box = bcache.ComputeWorldBound(prim).ComputeAlignedBox()
        bmin, bmax = box.GetMin(), box.GetMax()
        # ignore degenerate / unbounded results
        if any(abs(v) > 1e6 for v in (bmin[0], bmin[1], bmin[2],
                                       bmax[0], bmax[1], bmax[2])):
            continue
        mins.append(bmin)
        maxs.append(bmax)
    if not mins:
        return None
    rmin = Gf.Vec3d(min(m[0] for m in mins),
                    min(m[1] for m in mins),
                    min(m[2] for m in mins))
    rmax = Gf.Vec3d(max(m[0] for m in maxs),
                    max(m[1] for m in maxs),
                    max(m[2] for m in maxs))
    return rmin, rmax


obj_bounds = compute_object_aabb(stage, matched_paths)
if obj_bounds is None:
    print("[room] WARNING: could not compute object AABB; falling back to "
          "/CollisionMesh bounds (camera will likely spawn outside the room)")
    floor_z = float(bmin[2])
else:
    omin, omax = obj_bounds
    print(f"[room] object-derived interior min=({omin[0]:.2f}, {omin[1]:.2f}, "
          f"{omin[2]:.2f}) max=({omax[0]:.2f}, {omax[1]:.2f}, {omax[2]:.2f})")
    print(f"[room] interior size=({omax[0]-omin[0]:.2f}, "
          f"{omax[1]-omin[1]:.2f}, {omax[2]-omin[2]:.2f}) m")
    # Replace the (wrong) /CollisionMesh bounds with the object-derived ones.
    bmin, bmax = omin, omax
    # Floor height = bottom of the object cluster; camera height is measured
    # relative to that, not relative to whatever Z the collision mesh started at.
    floor_z = float(bmin[2])
    print(f"[room] floor_z={floor_z:.2f}; camera height window will be "
          f"[{floor_z + args.height_min:.2f}, {floor_z + args.height_max:.2f}]")


# ============================================================================
# Step 3c — locate wall prims and pre-compute their world AABBs so we can
# reject camera positions that lie inside a wall, or where the line of sight
# from the camera to the target object passes through a wall (which would put
# the camera in a different room than the target).
# ============================================================================
_WALL_NAME_RE = re.compile(r"wall", re.I)


def find_wall_aabbs(stage):
    """Return list of axis-aligned world bounding boxes for wall MESHES.

    Leaf meshes only: group Xforms like /World/WallAssets or /World/wardwall
    have AABBs spanning ~half the room interior, and treating those as solid
    walls made every camera candidate test "inside a wall" (the rejection
    loop then exhausted its attempts and fell back to unchecked samples).
    Thin Xform/Scope prims (a single wrapped wall) are still accepted via the
    slab test."""
    bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                               [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    aabbs = []
    for prim in stage.Traverse():
        if not _WALL_NAME_RE.search(prim.GetName()):
            continue
        t = prim.GetTypeName()
        if t not in ("Mesh", "Xform", "Scope"):
            continue
        try:
            box = bcache.ComputeWorldBound(prim).ComputeAlignedBox()
            mn, mx = box.GetMin(), box.GetMax()
            if any(abs(v) > 1e6 for v in (mn[0], mn[1], mn[2], mx[0], mx[1], mx[2])):
                continue
            sx, sy = float(mx[0] - mn[0]), float(mx[1] - mn[1])
            # Group prims must look like an actual wall slab (one thin
            # horizontal dimension); meshes are taken as-is.
            if t != "Mesh" and min(sx, sy) > 0.6:
                continue
            aabbs.append((
                (float(mn[0]), float(mn[1]), float(mn[2])),
                (float(mx[0]), float(mx[1]), float(mx[2])),
            ))
        except Exception:
            continue
    return aabbs


wall_aabbs = find_wall_aabbs(stage)
print(f"[walls] found {len(wall_aabbs)} wall AABBs for line-of-sight tests")


def _point_in_aabb(p, aabb):
    pmin, pmax = aabb
    return (pmin[0] <= p[0] <= pmax[0] and
            pmin[1] <= p[1] <= pmax[1] and
            pmin[2] <= p[2] <= pmax[2])


def _segment_intersects_aabb(p1, p2, aabb):
    """Slab test: does the closed segment p1->p2 intersect the AABB? We trim
    a small epsilon off both ends so a target object that touches the wall
    doesn't trip a false positive."""
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
    """True if `point` lies inside (or within `margin` m of) any of the given
    world AABBs. Used to reject camera positions that would put the eye
    inside a bed/cabinet/cart mesh."""
    for mn, mx in aabbs:
        if (mn[0] - margin <= point[0] <= mx[0] + margin and
                mn[1] - margin <= point[1] <= mx[1] + margin and
                mn[2] - margin <= point[2] <= mx[2] + margin):
            return True
    return False


def collect_blocker_aabbs(stage, labeled_paths, max_side=4.0, max_depth=3):
    """World AABBs of UNLABELED scene geometry under /World (CabinetAssets,
    industrialpipingmodel, ...) that the camera must not spawn inside —
    labeled objects are handled separately and cameras inside unlabeled
    furniture render pitch-black frames. Group prims whose footprint exceeds
    `max_side` m are recursed into (one level at a time) so a room-spanning
    group like GroundPlane doesn't blanket the whole interior."""
    bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                               [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    skip = set(labeled_paths)
    out = []

    def visit(prim, depth):
        if str(prim.GetPath()) in skip:
            return
        if prim.GetTypeName() not in ("Mesh", "Xform", "Scope", ""):
            return  # lights, graphs, materials, ...
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
        elif depth < max_depth:
            for child in prim.GetChildren():
                visit(child, depth + 1)

    for top in stage.GetPrimAtPath("/World").GetChildren():
        visit(top, 0)
    return out


blocker_aabbs = collect_blocker_aabbs(stage, matched_paths)
# Camera no-go volumes: every labeled object + every unlabeled blocker.
no_go_aabbs = [ot["aabb"] for ot in object_targets] + blocker_aabbs
print(f"[camera] {len(no_go_aabbs)} no-go AABBs for camera placement "
      f"({len(object_targets)} labeled + {len(blocker_aabbs)} unlabeled)")


# ============================================================================
# Step 5 — Replicator graph
# ============================================================================
def hfov_to_focal_length(hfov_deg: float, horizontal_aperture: float) -> float:
    return horizontal_aperture / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))


# Existing scene camera uses aperture 20.955 x 15.29; we adopt the same so
# Replicator's intrinsics line up with what the human-recorded test set sees.
H_APERTURE = 20.955
V_APERTURE = 15.29
focal = hfov_to_focal_length(args.hfov, H_APERTURE)
print(f"[camera] HFOV={args.hfov} deg -> focal_length={focal:.3f} (aperture "
      f"{H_APERTURE}x{V_APERTURE} mm)")


# Prim-name -> room rules. Order matters: most specific patterns first. The
# Ward0505.usd naming convention encodes each asset's room in its top-level
# prim name; we exploit that rather than relying on wall geometry (which
# doesn't fully partition the floor plan).
ROOM_NAME_RULES = [
    (re.compile(r"frontroom",       re.I), "Frontroom"),
    (re.compile(r"\btoilet|shower", re.I), "Bathroom"),
    (re.compile(r"sink_mirror",     re.I), "Bathroom"),
    (re.compile(r"^sink\d*$",       re.I), "Bathroom"),
]
DEFAULT_ROOM = "Ward"


def assign_room(prim_name):
    for pattern, room in ROOM_NAME_RULES:
        if pattern.search(prim_name):
            return room
    return DEFAULT_ROOM


def build_rooms_from_object_names(object_targets, pad_m=0.5):
    """Group labeled objects into rooms by prim-name pattern; each room's
    XY AABB is the union of its members' centroids (with a small inflation
    so the camera has space to stand back)."""
    grouped = {"Ward": [], "Bathroom": [], "Frontroom": []}
    for obj in object_targets:
        prim_name = obj["path"].rsplit("/", 1)[-1]
        room = assign_room(prim_name)
        grouped.setdefault(room, []).append(obj)
    out = []
    for room_name, members in grouped.items():
        if not members:
            continue
        xs = [m["centroid"][0] for m in members]
        ys = [m["centroid"][1] for m in members]
        out.append({
            "name": room_name,
            "members": members,
            "xmin": min(xs) - pad_m,
            "xmax": max(xs) + pad_m,
            "ymin": min(ys) - pad_m,
            "ymax": max(ys) + pad_m,
            "area_m2": (max(xs) - min(xs) + 2 * pad_m)
                       * (max(ys) - min(ys) + 2 * pad_m),
        })
    return out


def detect_rooms_from_walls(stage, object_targets, args, floor_z):
    """Build a top-down 2D floor plan, rasterize all walls onto it (filtered
    to those whose Z-extent actually spans camera eye-level), flood-fill the
    non-wall pixels to recover each room as a connected region. Returns:
        rooms_list  : [room_dict, ...]   (each has mask, AABB, members, area)
        wall_mask   : HxW bool array      (debug; True = wall)
        plan_xmin   : world-X at column 0
        plan_ymin   : world-Y at row 0
        grid_size   : meters per cell
    """
    if not object_targets:
        return [], None, 0.0, 0.0, 0.0

    grid_size = args.floor_grid_size

    # 1) Floor-plan extent: pad object AABB so walls just outside the object
    #    cloud are included.
    obj_xs = [m["centroid"][0] for m in object_targets]
    obj_ys = [m["centroid"][1] for m in object_targets]
    pad = 3.0
    plan_xmin = min(obj_xs) - pad
    plan_xmax = max(obj_xs) + pad
    plan_ymin = min(obj_ys) - pad
    plan_ymax = max(obj_ys) + pad

    W = int(math.ceil((plan_xmax - plan_xmin) / grid_size)) + 1
    H = int(math.ceil((plan_ymax - plan_ymin) / grid_size)) + 1
    wall_mask = np.zeros((H, W), dtype=bool)

    # 2) Find walls (and any structural panel-like prim). We DO NOT trust the
    #    full-stage AABB filter because referenced USDZ walls can have huge
    #    extents in their local frame; we rely on the world-space AABB only.
    bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                               [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    wall_re = re.compile(r"wall", re.I)
    walls_used = 0
    walls_skipped_height = 0
    walls_skipped_huge   = 0
    for prim in stage.Traverse():
        if not wall_re.search(prim.GetName()):
            continue
        if prim.GetTypeName() not in ("Mesh", "Xform", "Scope"):
            continue
        try:
            box = bcache.ComputeWorldBound(prim).ComputeAlignedBox()
            mn, mx = box.GetMin(), box.GetMax()
        except Exception:
            continue
        if any(abs(v) > 1e6 for v in (mn[0], mn[1], mn[2], mx[0], mx[1], mx[2])):
            walls_skipped_huge += 1
            continue
        # Z filter — only walls that actually span the camera's eye-level
        # band (floor_z + height_min .. height_max) count as blocking. This
        # ignores skirting boards / ceiling trim that are above or below the
        # camera and wouldn't visually separate rooms at eye level.
        wall_top = mx[2] - floor_z
        wall_bot = mn[2] - floor_z
        if wall_top < args.height_min or wall_bot > args.height_max:
            walls_skipped_height += 1
            continue
        x0 = max(0, int(math.floor((mn[0] - plan_xmin) / grid_size)))
        x1 = min(W - 1, int(math.ceil((mx[0] - plan_xmin) / grid_size)))
        y0 = max(0, int(math.floor((mn[1] - plan_ymin) / grid_size)))
        y1 = min(H - 1, int(math.ceil((mx[1] - plan_ymin) / grid_size)))
        if x0 <= x1 and y0 <= y1:
            wall_mask[y0:y1 + 1, x0:x1 + 1] = True
            walls_used += 1

    print(f"[rooms] floor plan {W}x{H} cells "
          f"({W*grid_size:.1f} x {H*grid_size:.1f} m) at {grid_size} m/cell")
    print(f"[rooms] walls used={walls_used}, "
          f"skipped-by-height={walls_skipped_height}, "
          f"skipped-huge={walls_skipped_huge}")

    # 3) Dilate wall mask to close tiny gaps between adjacent wall segments
    for _ in range(args.wall_dilation_cells):
        d = wall_mask.copy()
        d[1:]    |= wall_mask[:-1]
        d[:-1]   |= wall_mask[1:]
        d[:, 1:] |= wall_mask[:, :-1]
        d[:, :-1]|= wall_mask[:, 1:]
        wall_mask = d

    # 4) Flood fill non-wall cells via iterative BFS
    visited = np.zeros_like(wall_mask)
    rooms = []
    min_cells = max(int(args.room_min_area / (grid_size * grid_size)), 1)
    from collections import deque as _deque
    for sy in range(H):
        for sx in range(W):
            if wall_mask[sy, sx] or visited[sy, sx]:
                continue
            mask = np.zeros_like(wall_mask)
            q = _deque([(sy, sx)])
            visited[sy, sx] = True
            mn_x = mx_x = sx
            mn_y = mx_y = sy
            count = 0
            while q:
                y, x = q.popleft()
                mask[y, x] = True
                count += 1
                if x < mn_x: mn_x = x
                if x > mx_x: mx_x = x
                if y < mn_y: mn_y = y
                if y > mx_y: mx_y = y
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and \
                       not visited[ny, nx] and not wall_mask[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if count < min_cells:
                continue
            rooms.append({
                "mask": mask,
                "xmin": plan_xmin + mn_x * grid_size,
                "xmax": plan_xmin + mx_x * grid_size,
                "ymin": plan_ymin + mn_y * grid_size,
                "ymax": plan_ymin + mx_y * grid_size,
                "grid_origin": (plan_xmin, plan_ymin),
                "grid_size": grid_size,
                "area_m2": count * grid_size * grid_size,
                "members": [],
            })
    rooms.sort(key=lambda r: -r["area_m2"])

    # 5) Assign each object to whichever room its XY centroid lands in
    unassigned = 0
    for obj in object_targets:
        c = obj["centroid"]
        ix = int((c[0] - plan_xmin) / grid_size)
        iy = int((c[1] - plan_ymin) / grid_size)
        if not (0 <= iy < H and 0 <= ix < W):
            unassigned += 1
            continue
        for room in rooms:
            if room["mask"][iy, ix]:
                room["members"].append(obj)
                break
        else:
            # Sat on a wall cell — find the nearest room by simple search
            best = None
            best_d = float("inf")
            for room in rooms:
                ys, xs = np.where(room["mask"])
                if len(xs) == 0:
                    continue
                # Quick centroid distance
                rcx = float(xs.mean())
                rcy = float(ys.mean())
                d = (ix - rcx) ** 2 + (iy - rcy) ** 2
                if d < best_d:
                    best_d = d
                    best = room
            if best is not None:
                best["members"].append(obj)
            else:
                unassigned += 1

    if unassigned:
        print(f"[rooms] {unassigned} object(s) couldn't be assigned to any room")

    return rooms, wall_mask, plan_xmin, plan_ymin, grid_size


def save_layout_png(rooms, wall_mask, plan_xmin, plan_ymin, grid_size,
                    object_targets, out_path):
    """Top-down PNG: walls (black) + room AABBs (per-room color, semi-trans)
    + object centroids (black dots)."""
    H, W = wall_mask.shape
    img = np.full((H, W, 3), 235, dtype=np.uint8)
    palette = {
        "Ward":      (210, 120, 120),
        "Bathroom":  (120, 200, 130),
        "Frontroom": (130, 150, 230),
    }
    # Fill each room's AABB rectangle with its tint, lightly transparent over
    # the white background (simulated by blending toward the palette color).
    for room in rooms:
        x0 = max(0, int((room["xmin"] - plan_xmin) / grid_size))
        x1 = min(W - 1, int((room["xmax"] - plan_xmin) / grid_size))
        y0 = max(0, int((room["ymin"] - plan_ymin) / grid_size))
        y1 = min(H - 1, int((room["ymax"] - plan_ymin) / grid_size))
        color = palette.get(room["name"], (180, 180, 180))
        if x0 < x1 and y0 < y1:
            tile = img[y0:y1 + 1, x0:x1 + 1].astype(np.int32)
            blend = (tile + np.array(color, dtype=np.int32)) // 2
            img[y0:y1 + 1, x0:x1 + 1] = blend.astype(np.uint8)
    # Walls on top (so they remain visible over the room tints)
    img[wall_mask] = (20, 20, 20)
    # Object centroids
    for obj in object_targets:
        c = obj["centroid"]
        x = int((c[0] - plan_xmin) / grid_size)
        y = int((c[1] - plan_ymin) / grid_size)
        if 0 <= y < H and 0 <= x < W:
            img[max(0, y - 2):min(H, y + 3),
                max(0, x - 2):min(W, x + 3)] = (0, 0, 0)
    # Flip Y so the image reads like a top-down architectural plan (north up)
    img = img[::-1]
    Image.fromarray(img).save(out_path)
    print(f"[rooms] wrote layout preview -> {out_path}")


# Pillow only inside this scope; SimulationApp already loaded it via Replicator.
from PIL import Image  # noqa: E402

# Run the wall flood-fill ONLY for the visualization (so you can see where
# walls landed); the actual room assignment used for camera placement comes
# from prim names, which encode the room reliably in Ward0505.
_flood_rooms, wall_mask, plan_xmin, plan_ymin, grid_size = \
    detect_rooms_from_walls(stage, object_targets, args, floor_z)

rooms = build_rooms_from_object_names(object_targets, pad_m=0.5)
print(f"[rooms] {len(rooms)} room(s) assigned via prim-name patterns:")
for room in rooms:
    cls_counts = Counter(m["class"] for m in room["members"])
    summary = ", ".join(f"{n}x{cls}" for cls, n in cls_counts.most_common(6))
    if len(cls_counts) > 6:
        summary += f", +{len(cls_counts) - 6} more"
    sx = room["xmax"] - room["xmin"]
    sy = room["ymax"] - room["ymin"]
    print(f"  {room['name']:10s}: {len(room['members']):3d} objs  "
          f"XY [{room['xmin']:6.2f}..{room['xmax']:6.2f}] x "
          f"[{room['ymin']:6.2f}..{room['ymax']:6.2f}]  ({sx:.1f}x{sy:.1f} m)")
    print(f"              classes: {summary}")

if args.save_layout_png and wall_mask is not None:
    layout_path = Path(args.out) / "_room_layout.png"
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    save_layout_png(rooms, wall_mask, plan_xmin, plan_ymin, grid_size,
                    object_targets, layout_path)

def sample_camera_pose(rng, centroid_of=None, extra_no_go=None):
    """Pick a room (weighted by labeled-object count), sample a camera
    position inside that room's XY AABB at eye level, aim at one of the
    room's member objects. Room assignment is by prim-name pattern so the
    camera is guaranteed to be in the same room as its target, even though
    the USD walls don't form a closed geometric partition.

    centroid_of: optional {path: (x,y,z)} of MOVED centroids for this frame
    (placement DR) so the camera aims at where the object actually is.
    extra_no_go: optional moved AABBs to reject camera positions inside."""
    rooms_with_members = [r for r in rooms if r["members"]]
    if not rooms_with_members:
        mx_ = 0.3
        x = rng.uniform(bmin[0] + mx_, bmax[0] - mx_)
        y = rng.uniform(bmin[1] + mx_, bmax[1] - mx_)
        z = floor_z + rng.uniform(args.height_min, args.height_max)
        return (x, y, z), (
            rng.uniform(bmin[0] + mx_, bmax[0] - mx_),
            rng.uniform(bmin[1] + mx_, bmax[1] - mx_),
            floor_z + rng.uniform(0.4, 1.7),
        ), True

    weights = [len(r["members"]) for r in rooms_with_members]
    room = rng.choices(rooms_with_members, weights=weights, k=1)[0]
    target_obj = rng.choice(room["members"])
    target = target_obj["centroid"]
    if centroid_of:
        target = centroid_of.get(target_obj["path"], target)

    # Rejection-sample the camera position: far enough from the target AND
    # not inside a wall mesh or any labeled object's AABB (a camera inside
    # geometry renders cut-open/black frames). Fall back to the last sample
    # if no candidate passes — the frame filter will prune it downstream.
    pos = None
    for _ in range(50):
        cam_x = rng.uniform(room["xmin"], room["xmax"])
        cam_y = rng.uniform(room["ymin"], room["ymax"])
        cam_z = floor_z + rng.uniform(args.height_min, args.height_max)
        cand = (cam_x, cam_y, cam_z)
        d2 = (target[0] - cam_x) ** 2 + (target[1] - cam_y) ** 2
        if d2 < args.min_cam_distance ** 2:
            pos = cand
            continue
        if _inside_any_wall(cand) or _inside_any_object(cand, no_go_aabbs):
            pos = cand
            continue
        if extra_no_go and _inside_any_object(cand, extra_no_go):
            pos = cand
            continue
        if _line_of_sight_blocked(cand, target):
            # Room AABBs are centroid-derived and can spill past a wall; a
            # blocked sight line means the camera landed in the wrong room.
            pos = cand
            continue
        pos = cand
        accepted = True
        break
    else:
        accepted = False
    jit = 0.15
    look = (
        target[0] + rng.uniform(-jit, jit),
        target[1] + rng.uniform(-jit, jit),
        target[2] + rng.uniform(-jit, jit),
    )
    return pos, look, accepted


def _catmull_rom(points, n):
    """Smooth Catmull-Rom spline through `points` (list of 3-tuples) -> n samples
    that pass through the keyframes (C1-continuous, no corners)."""
    import numpy as _np
    pts = _np.asarray(points, float)
    if len(pts) == 1:
        return [tuple(float(v) for v in pts[0])] * n
    P = _np.vstack([pts[0], pts, pts[-1]])      # pad ends so the spline hits them
    segs = len(pts) - 1
    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) * segs
        s = min(int(u), segs - 1)
        t = u - s
        p0, p1, p2, p3 = P[s], P[s + 1], P[s + 2], P[s + 3]
        t2, t3 = t * t, t * t * t
        pt = 0.5 * ((2 * p1) + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
        out.append(tuple(float(v) for v in pt))
    return out


def sample_camera_trajectory(rng, n_frames, n_keys):
    """Smooth camera fly-through: sample n_keys VALID keyframe poses (reusing
    sample_camera_pose's room/occlusion/line-of-sight checks) and Catmull-Rom
    interpolate position + look-at over n_frames. Temporally-coherent clip for
    video models (Cosmos-Transfer) instead of jumpy independent poses."""
    key_pos, key_look, tries = [], [], 0
    while len(key_pos) < n_keys and tries < n_keys * 80:
        tries += 1
        p, t, ok = sample_camera_pose(rng)
        if ok and p is not None:
            key_pos.append(p); key_look.append(t)
    while len(key_pos) < 2:                     # guarantee a spline is possible
        p, t, _ = sample_camera_pose(rng)
        key_pos.append(p); key_look.append(t)
    print(f"[camera][trajectory] {len(key_pos)} valid keyframes -> "
          f"{n_frames}-frame Catmull-Rom fly-through")
    return _catmull_rom(key_pos, n_frames), _catmull_rom(key_look, n_frames)


# ---- Meaningful placement DR (v2): hierarchical room -> bed bay / wall groups /
# floor furniture / surface items (placement_dr.py). Computed BEFORE camera
# sampling so each frame's camera aims at the MOVED scene, not stale centroids.
PLACEMENT_SEQ = {}
_ot_by_path = {}
if args.randomize_placement:
    import placement_dr
    _pseed = args.placement_seed or args.seed
    for _ot in object_targets:
        _prim = stage.GetPrimAtPath(_ot["path"])
        _M = UsdGeom.Xformable(_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        _t = _M.ExtractTranslation()
        _ot["translate"] = (float(_t[0]), float(_t[1]), float(_t[2]))
        _ot["room"] = assign_room(_ot["path"].rsplit("/", 1)[-1])
    _dump = {
        "objects": [{k: _ot[k] for k in
                     ("path", "class", "centroid", "aabb", "size",
                      "translate", "room")} for _ot in object_targets],
        "rooms": [{k: r[k] for k in ("name", "xmin", "xmax", "ymin", "ymax")}
                  for r in rooms],
        "floor_z": floor_z,
        "wall_aabbs": wall_aabbs,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "_placement_inputs.json").write_text(json.dumps(_dump))
    _seq_all = placement_dr.generate(
        object_targets, rooms, floor_z, n_frames=args.frames,
        seed=_pseed, max_shift=args.placement_shift, wall_boxes=wall_aabbs,
        global_frac=args.placement_global_frac)
    PLACEMENT_SEQ = {p: s for p, s in _seq_all.items() if len(set(s)) > 1}
    _ot_by_path = {o["path"]: o for o in object_targets}
    print(f"[placement-dr] {len(PLACEMENT_SEQ)}/{len(_seq_all)} objects move "
          f"across {args.frames} frames (max_shift={args.placement_shift} m, "
          f"global_frac={args.placement_global_frac}, seed={_pseed})", flush=True)


def _moved_view(_f):
    """Per-frame camera aim data under placement DR: {path: moved centroid}
    + the moved AABBs (extra camera no-go volumes for this frame)."""
    _cof, _xng = {}, []
    for _p, _s in PLACEMENT_SEQ.items():
        _o = _ot_by_path[_p]
        _ox, _oy, _oz = _o["translate"]
        _dx, _dy, _dz = (_s[_f][0] - _ox, _s[_f][1] - _oy, _s[_f][2] - _oz)
        _c = _o["centroid"]
        _cof[_p] = (_c[0] + _dx, _c[1] + _dy, _c[2] + _dz)
        (_amn, _amx) = _o["aabb"]
        _xng.append(((_amn[0] + _dx, _amn[1] + _dy, _amn[2] + _dz),
                     (_amx[0] + _dx, _amx[1] + _dy, _amx[2] + _dz)))
    return _cof, _xng


rng = random.Random(args.seed)
if args.trajectory or args.cosmos:
    if PLACEMENT_SEQ:
        print("[placement-dr] NOTE: trajectory/cosmos mode renders a coherent "
              "clip but placement changes the scene EVERY frame; consider "
              "omitting --randomize-placement for clips.", flush=True)
    # Smooth fly-through -> temporally-coherent clip for Cosmos-Transfer (video).
    camera_positions, camera_targets = sample_camera_trajectory(
        rng, args.frames, args.traj_keys)
else:
    camera_positions = []
    camera_targets   = []
    _n_fallback = 0
    for _f in range(args.frames):
        if PLACEMENT_SEQ:
            # aim at the MOVED scene of this frame, avoid the moved volumes
            _cof, _xng = _moved_view(_f)
            p, t, ok = sample_camera_pose(rng, centroid_of=_cof,
                                          extra_no_go=_xng)
        else:
            p, t, ok = sample_camera_pose(rng)
        camera_positions.append(p)
        camera_targets.append(t)
        if not ok:
            _n_fallback += 1
    print(f"[camera] {args.frames - _n_fallback}/{args.frames} poses passed all "
          f"placement checks; {_n_fallback} fell back to an unchecked sample "
          f"(high fallback count -> no-go volumes are over-rejecting)")


# Camera + render product live at stage scope (not inside a new layer) so the
# writer attach + orchestrator can find them.
cam = rep.create.camera(
    position=camera_positions[0],
    look_at=camera_targets[0],
    focal_length=focal,
    horizontal_aperture=H_APERTURE,
    # Default clipping_range is (1.0, 1e6) in WORLD units; this stage is in
    # meters, so the default near plane sat 1 m in front of the camera and
    # sliced open any geometry closer than that ("partial object rendering").
    clipping_range=(0.05, 100000.0),
)
rp = rep.create.render_product(cam, resolution=tuple(args.resolution))

# TODO(future): background-diversity submodule. Procedural primitives, 2D
# wall decals, or real-image overlay-paste should live in a separate module
# once we've researched how to automate it properly. Intentionally NOT adding
# clutter here so the v1 dataset stays clean and reproducible.

# Replicator's create.camera in 5.1 doesn't accept vertical_aperture as a kwarg;
# the vertical aperture is implicit from the horizontal aperture + render
# product aspect ratio. Set it explicitly on the USD prim so semantic FOV
# matches what the test webcam captures.
try:
    cam_prim_path = cam.node.get_attribute("inputs:primPath").get()
    if cam_prim_path:
        cam_prim = stage.GetPrimAtPath(cam_prim_path)
        if cam_prim.IsValid():
            from pxr import UsdGeom as _UG
            _cam = _UG.Camera(cam_prim)
            _cam.GetVerticalApertureAttr().Set(V_APERTURE)
            print(f"[camera] set vertical_aperture={V_APERTURE} on {cam_prim_path}")
except Exception as e:
    print(f"[camera] (could not set vertical_aperture explicitly: {e})")

# ---- Online material domain-randomization (perturb originals) ------------ #
# Keeps every labeled object's OWN baked material and jitters it in place each
# frame instead of rebinding procedural textures: per-channel tint on the
# diffuse texture (UsdUVTexture inputs:scale), diffuseColor jitter around the
# authored colour for untextured shaders, and roughness re-rolls. Objects keep
# their identity; geometry is never touched, so masks/boxes stay exactly valid.
MATERIAL_DR = bool(args.randomize_materials)
if MATERIAL_DR and not matched_paths:
    print("[material-dr] no labeled object prims found -> disabling material DR")
    MATERIAL_DR = False
if MATERIAL_DR:
    import re as _re
    from pxr import UsdShade  # noqa: E402

    def _ensure_input(shader_prim, name, type_name, default):
        # rep.modify.attribute only randomizes attributes that already exist on
        # the prim, and the GLB/Tripo converters rarely author these inputs ->
        # author the neutral default once so the per-frame trigger can hit it.
        attr = shader_prim.GetAttribute(f"inputs:{name}")
        if not attr or attr.Get() is None:
            UsdShade.Shader(shader_prim).CreateInput(name, type_name).Set(default)

    tex_scale_paths = []  # UsdUVTexture feeding diffuseColor -> tint via inputs:scale
    rough_paths = []      # UsdPreviewSurface shaders -> re-roll inputs:roughness
    color_ranges = []     # untextured UsdPreviewSurface -> (path, lo, hi) jitter
    mdl_tint_paths = []   # MDL shaders (OmniPBR etc.) -> tint via diffuse_tint

    # Mirrors must NOT be perturbed: tint / roughness / colour jitter destroys
    # the reflective MDL and the mirror renders as a flat tinted panel instead
    # of a reflection. Collect every material bound to a mirror prim and skip
    # those below (handles mirrors that share a material with other prims too).
    _mirror_roots = [ot["path"] for ot in object_targets if ot["class"] == "mirror"]
    _mirror_mats = set()
    for _mroot in _mirror_roots:
        mp = stage.GetPrimAtPath(_mroot)
        if not mp.IsValid():
            continue
        for prim in Usd.PrimRange(mp):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            mmat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if mmat:
                _mirror_mats.add(mmat.GetPath())

    # Resolve each labeled object's bound material via the binding API (the
    # materials live inside the referenced .usdz subtrees, but this also
    # catches ones bound from a shared /World/Looks scope).
    _seen_mats = set()
    for _root in matched_paths:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(_root)):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not mat or mat.GetPath() in _seen_mats:
                continue
            _seen_mats.add(mat.GetPath())
            if mat.GetPath() in _mirror_mats:    # keep mirrors reflective
                continue
            try:
                surf, _, _ = mat.ComputeSurfaceSource(["mdl"])
            except TypeError:  # older USD: single render-context token
                surf, _, _ = mat.ComputeSurfaceSource("mdl")
            if not surf:
                continue
            sprim = surf.GetPrim()
            if surf.GetIdAttr().Get() == "UsdPreviewSurface":
                _ensure_input(sprim, "roughness", Sdf.ValueTypeNames.Float, 0.5)
                rough_paths.append(str(sprim.GetPath()))
                dc = surf.GetInput("diffuseColor")
                if dc and dc.HasConnectedSource():
                    # Perturb ONLY the texture node feeding diffuseColor --
                    # blanket-scaling every UsdUVTexture would corrupt
                    # normal/roughness maps on multi-map materials.
                    src, _, _ = dc.GetConnectedSource()
                    tprim = src.GetPrim()
                    if UsdShade.Shader(tprim).GetIdAttr().Get() == "UsdUVTexture":
                        _ensure_input(tprim, "scale", Sdf.ValueTypeNames.Float4,
                                      Gf.Vec4f(1.0))
                        tex_scale_paths.append(str(tprim.GetPath()))
                else:
                    # Constant-colour shader: jitter AROUND the authored colour
                    # (full-gamut randomization would erase object identity).
                    c = dc.Get() if dc else None
                    c = c if c is not None else Gf.Vec3f(0.5)
                    _ensure_input(sprim, "diffuseColor",
                                  Sdf.ValueTypeNames.Color3f, Gf.Vec3f(c))
                    lo = tuple(max(0.0, v * 0.7 - 0.05) for v in c)
                    hi = tuple(min(1.0, v * 1.3 + 0.05) for v in c)
                    color_ranges.append((str(sprim.GetPath()), lo, hi))
            elif sprim.GetAttribute("info:mdl:sourceAsset"):
                _ensure_input(sprim, "diffuse_tint",
                              Sdf.ValueTypeNames.Color3f, Gf.Vec3f(1.0))
                _ensure_input(sprim, "reflection_roughness_constant",
                              Sdf.ValueTypeNames.Float, 0.5)
                mdl_tint_paths.append(str(sprim.GetPath()))

    def _pattern(paths):
        return "^(" + "|".join(_re.escape(p) for p in paths) + ")$"

    # flush=True: sim_app.close() hard-exits without flushing stdio, so any
    # buffered tail prints are silently lost from redirected logs.
    print(f"[material-dr] perturbing originals: {len(tex_scale_paths)} textured "
          f"+ {len(color_ranges)} flat-colour + {len(mdl_tint_paths)} MDL "
          f"shaders across {len(_seen_mats)} bound materials "
          f"({len(_mirror_mats)} mirror material(s) excluded -> stay reflective)",
          flush=True)
    if not (tex_scale_paths or color_ranges or mdl_tint_paths):
        print("[material-dr] no perturbable shaders found -> disabling material DR")
        MATERIAL_DR = False

# Per-frame randomization: camera pose + ceiling-light intensity/color temp
with rep.trigger.on_frame(num_frames=args.frames, rt_subframes=args.rt_subframes):
    if PLACEMENT_SEQ:
        # author each object's per-frame WORLD translate (orientation untouched)
        for _ppath, _seq in PLACEMENT_SEQ.items():
            with rep.get.prims(path_pattern="^" + re.escape(_ppath) + "$",
                               ignore_case=False):
                rep.modify.pose(position=rep.distribution.sequence(_seq))
    if MATERIAL_DR:
        # Jitter each object's ORIGINAL material in place every frame. Each
        # matched prim draws its own independent sample (same per-prim
        # behaviour as the ceiling lights below).
        # NOTE: ignore_case=False everywhere -- rep.get.prims defaults to
        # case-INsensitive regex, which made exact paths under /World/toilet
        # also match the case-twin /World/Toilet (different asset instance,
        # different authored attrs) and spam write errors.
        if tex_scale_paths:
            # Per-channel multiplier on the baked diffuse texture: covers
            # brightness + tint + mild hue shifts without losing the texture.
            with rep.get.prims(path_pattern=_pattern(tex_scale_paths),
                               ignore_case=False):
                rep.modify.attribute("inputs:scale", rep.distribution.uniform(
                    (0.55, 0.55, 0.55, 1.0), (1.45, 1.45, 1.45, 1.0)))
        for _cpath, _clo, _chi in color_ranges:
            # One node per shader: the jitter range is centred on THAT
            # shader's authored colour, so it can't be a shared distribution.
            with rep.get.prims(path_pattern=_pattern([_cpath]),
                               ignore_case=False):
                rep.modify.attribute("inputs:diffuseColor",
                                     rep.distribution.uniform(_clo, _chi))
        if rough_paths:
            with rep.get.prims(path_pattern=_pattern(rough_paths),
                               ignore_case=False):
                rep.modify.attribute("inputs:roughness",
                                     rep.distribution.uniform(0.20, 0.95))
        if mdl_tint_paths:
            with rep.get.prims(path_pattern=_pattern(mdl_tint_paths),
                               ignore_case=False):
                rep.modify.attribute("inputs:diffuse_tint",
                                     rep.distribution.uniform(
                                         (0.55, 0.55, 0.55), (1.45, 1.45, 1.45)))
                rep.modify.attribute("inputs:reflection_roughness_constant",
                                     rep.distribution.uniform(0.20, 0.95))
    with cam:
        rep.modify.pose(
            position=rep.distribution.sequence(camera_positions),
            look_at=rep.distribution.sequence(camera_targets),
        )
    # Per-frame lighting DR on the kept room lights (color TEMPERATURE +
    # a gentle MULTIPLICATIVE intensity jitter around each light's authored
    # value). We do NOT set an absolute intensity band: the kept fixtures span
    # very different scales (e.g. SphereLight 500, DistantLight 2500, DiskLight
    # 60000), so a fixed band would wash some out and blow others up. The
    # `intensity_multiplier` attribute scales each light relative to its own
    # authored intensity, which is scale-safe across fixture types.
    if KEPT_LIGHTS:
        import re as _re_l
        ceiling = rep.get.prims(
            path_pattern="(" + "|".join(_re_l.escape(p) for p in KEPT_LIGHTS) + ")$")
        with ceiling:
            # exposure is in STOPS: final intensity = intensity * 2**exposure,
            # so this is a per-light multiplicative jitter (~0.62x .. 1.32x)
            # that is correct regardless of each fixture's base intensity.
            rep.modify.attribute("exposure",
                                 rep.distribution.uniform(-0.7, 0.4))
            rep.modify.attribute("color", rep.distribution.choice([
                (1.00, 0.85, 0.65),  # warm white (~2700 K)
                (1.00, 0.93, 0.83),  # neutral warm (~3500 K, fluorescent)
                (1.00, 0.97, 0.92),  # neutral white (~4000 K)
                (0.95, 0.95, 0.95),  # cool white (~5000 K, LED panel)
                (0.92, 0.94, 1.00),  # cool daylight (~5500 K, window mix)
            ]))

    # Independently vary the ambient dome light each frame. Brighter dome
    # mimics a sunlit window; darker dome mimics evening/night corridor.
    dome = rep.get.prims(path_pattern="/Environment/Grey_Studio/DomeLight")
    with dome:
        rep.modify.attribute("intensity",
                             rep.distribution.uniform(300.0, 600.0))
        rep.modify.attribute("color", rep.distribution.choice([
            (1.00, 1.00, 1.00),  # neutral
            (0.95, 0.97, 1.00),  # cool-tinted (overcast)
            (1.00, 0.95, 0.85),  # warm-tinted (golden hour)
            (0.85, 0.90, 1.00),  # cooler twilight
        ]))


# Writer: BasicWriter outputs RGB + COCO-ish data; we'll fold into your
# rgbDataset/jsonDataset layout in step 6 below.
# Absolute path: Replicator's disk backend redirects RELATIVE output_dir to a
# default root (~/omni.replicator_out), which then doesn't match where the
# post-process step looks -> "0 RGB files". Resolve so frames land under --out.
if args.cosmos:
    # ---- CosmosWriter: clip-based multimodal control data for Cosmos-Transfer.
    # Outputs rgb/depth/segmentation/shaded_seg/edges (+ mp4s) under a clip layout.
    # use_instance_id=True embeds per-object identity in the segmentation stream so
    # identity is preserved across frames (see docs/cosmos3_isaacsim_replicator_*).
    cosmos_out = (Path(args.out) / "_cosmos").resolve()
    cosmos_out.mkdir(parents=True, exist_ok=True)
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=str(cosmos_out))
    writer = rep.WriterRegistry.get("CosmosWriter")
    writer.initialize(backend=backend, use_instance_id=True)
    writer.attach(rp)
    print(f"[render][cosmos] CosmosWriter (use_instance_id=True) -> {cosmos_out}",
          flush=True)
    print(f"[render] generating {args.frames} frames into {cosmos_out}")
else:
    # Writer: BasicWriter outputs RGB + COCO-ish data; folded into your
    # rgbDataset/jsonDataset layout in step 6 below.
    # Absolute path: Replicator's disk backend redirects RELATIVE output_dir to a
    # default root (~/omni.replicator_out), which then doesn't match where the
    # post-process step looks -> "0 RGB files". Resolve so frames land under --out.
    raw_out = (Path(args.out) / "_raw").resolve()
    raw_out.mkdir(parents=True, exist_ok=True)
    writer = rep.WriterRegistry.get("BasicWriter")
    _writer_kwargs = dict(
        output_dir=str(raw_out),
        rgb=True,
        bounding_box_2d_tight=True,
        instance_segmentation=True,
        semantic_segmentation=True,
        # semantic_segmentation PNG is colorized so you can VISUALLY confirm masks.
        # instance_segmentation PNG stays in raw uint32 IDs so the COCO converter
        # can derive per-instance binary masks reliably (each pixel = instance id).
        colorize_semantic_segmentation=True,
        colorize_instance_segmentation=False,
    )
    if args.extra_channels:
        # Ground-truth, pixel-aligned control channels (Isaac, not estimated).
        _writer_kwargs.update(
            distance_to_camera=True,        # radial depth (m) -> *.npy per frame
            distance_to_image_plane=True,   # planar depth (m)
            normals=True,                   # surface normals (RGB-encoded)
            instance_id_segmentation=True,  # stable per-object IDs
        )
        print("[render] --extra-channels: also writing depth + normals + instance_id",
              flush=True)
    writer.initialize(**_writer_kwargs)
    writer.attach([rp])
    print(f"[render] generating {args.frames} frames into {raw_out}")

# rep.orchestrator.run() is async; we need to actually pump frames in
# standalone mode. Prefer run_until_complete() when available; fall back to
# polling on sim_app.update().
def _run_orchestrator_sync():
    if hasattr(rep.orchestrator, "run_until_complete"):
        rep.orchestrator.run_until_complete()
        return
    # Polling fallback (older / newer 5.x variants without sync wrapper)
    import time as _t
    rep.orchestrator.run()
    t0 = _t.time()
    # Wait until orchestrator starts (it's enqueued asynchronously)
    while not rep.orchestrator.get_is_started():
        sim_app.update()
        if _t.time() - t0 > 30:
            print("WARN: orchestrator never started after 30s")
            break
    # Then wait until it finishes
    t1 = _t.time()
    while rep.orchestrator.get_is_started():
        sim_app.update()
        if _t.time() - t1 > 1800:   # 30 min hard cap
            print("WARN: orchestrator timed out after 30 min")
            break

_run_orchestrator_sync()
if args.cosmos:
    rep.orchestrator.wait_until_complete()
    try:
        writer.detach()
    except Exception:
        pass
    cosmos_out = (Path(args.out) / "_cosmos").resolve()
    _clips = sorted(cosmos_out.glob("*"))
    print(f"[render][cosmos] done -> {cosmos_out}  ({len(_clips)} entries); "
          f"modality subfolders + mp4s for Cosmos-Transfer")
else:
    print(f"[render] orchestrator done; rendered files now under {raw_out}")
    # A quick directory sanity-check so we can see if anything was actually written
    _written = list(raw_out.rglob("rgb_*.png"))
    print(f"[render] wrote {len(_written)} RGB files; "
          f"example: {_written[0] if _written else '(none!)'}")


# ============================================================================
# Step 6 — convert BasicWriter output -> your rgbDataset/jsonDataset layout
# ============================================================================
def basic_writer_to_coco(raw_dir: Path, out_root: Path, tag: str,
                         category_map: dict) -> dict:
    """Read BasicWriter's per-frame bounding_box_2d_tight + RGB files; emit:
      <out_root>/rgbDataset/<tag>_rgb/rgb_frame_<i>.png
      <out_root>/jsonDataset/<tag>.json   (COCO instance format)
    """
    rgb_dir = out_root / "rgbDataset" / f"{tag}_rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    json_dir = out_root / "jsonDataset"
    json_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {
            "description": f"Ward replicator dataset tag={tag}",
            "date_created": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cid, "name": name, "supercategory": "ward_object"}
            for name, cid in category_map.items()
        ],
    }
    ann_id = 1

    # BasicWriter file naming (Isaac Sim 5.1 default):
    #   rgb_<NNNN>.png
    #   bounding_box_2d_tight_<NNNN>.npy
    #   bounding_box_2d_tight_labels_<NNNN>.json
    # In 5.x BasicWriter sometimes nests files under a `RenderProduct_*` subdir.
    # Search recursively so we don't care which layout it picked.
    rgb_files = sorted(raw_dir.rglob("rgb_*.png"))
    if not rgb_files:
        print(f"[coco] no RGB files found anywhere under {raw_dir}; nothing to do.")
        print(f"[coco] (run with --debug-stage to verify the writer attached)")
        return coco
    print(f"[coco] found {len(rgb_files)} RGB files under {raw_dir}")
    for i, rgb_path in enumerate(rgb_files):
        stem = rgb_path.stem  # rgb_0000
        idx = stem.split("_", 1)[1]
        # Sibling files (same parent as the RGB png)
        bbox_npy  = rgb_path.parent / f"bounding_box_2d_tight_{idx}.npy"
        label_json = rgb_path.parent / f"bounding_box_2d_tight_labels_{idx}.json"
        if not bbox_npy.exists() or not label_json.exists():
            # Try without padding e.g. bounding_box_2d_tight_5.npy
            try:
                raw_idx = str(int(idx))
                bbox_npy  = rgb_path.parent / f"bounding_box_2d_tight_{raw_idx}.npy"
                label_json = rgb_path.parent / f"bounding_box_2d_tight_labels_{raw_idx}.json"
            except ValueError:
                pass
        if not bbox_npy.exists() or not label_json.exists():
            print(f"[coco] skip {rgb_path.name}: missing bbox/label sidecars")
            continue

        # Copy / link RGB into your expected layout
        target = rgb_dir / f"rgb_frame_{idx}.png"
        if not target.exists():
            os.link(rgb_path, target) if hasattr(os, "link") else \
                target.write_bytes(rgb_path.read_bytes())

        # Read image size from one frame (assume all the same)
        from PIL import Image as _PIL
        with _PIL.open(rgb_path) as im:
            W, H = im.size

        image_id = i + 1
        coco["images"].append({
            "id": image_id,
            "file_name": f"rgb_frame_{idx}.png",
            "width":  W,
            "height": H,
        })

        boxes = np.load(bbox_npy)  # structured array
        with open(label_json) as f:
            labels = json.load(f)  # { "<sem_id>": {"class": "<name>"} }

        for row in boxes:
            sid = int(row["semanticId"])
            x_min = int(row["x_min"])
            y_min = int(row["y_min"])
            x_max = int(row["x_max"])
            y_max = int(row["y_max"])
            w, h = x_max - x_min, y_max - y_min
            if w <= 1 or h <= 1:
                continue
            class_name = class_from_entry(labels.get(str(sid)))
            if class_name is None or class_name not in category_map:
                continue
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_map[class_name],
                "bbox": [x_min, y_min, w, h],
                "area": int(w * h),
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

    out_json = json_dir / f"{tag}.json"
    with open(out_json, "w") as f:
        json.dump(coco, f)
    print(f"[coco] wrote {out_json}  ({len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations)")
    return coco


if args.cosmos:
    print(f"[done][cosmos] clip data under {Path(args.out) / '_cosmos'}")
    print(f"      feed the modality folders/mp4s to Cosmos-Transfer for sim2real.")
else:
    print("[post] converting BasicWriter output -> rgbDataset/jsonDataset layout")
    basic_writer_to_coco(
        raw_dir=Path(args.out) / "_raw",
        out_root=Path(args.out),
        tag=args.tag,
        category_map=FIXED_CATEGORIES,
    )
    # GT overlays next to every raw frame (instance masks + class names) so
    # label alignment can be eyeballed straight from the writer output.
    try:
        from overlay_raw_gt import generate as _gt_generate
        print("[post] writing gt_<idx>.jpg overlays next to raw frames")
        _gt_generate(Path(args.out) / "_raw")
    except Exception as _e:
        print(f"[post] GT overlay generation failed (non-fatal): {_e}")
    print(f"[done] dataset under {args.out}")
    print(f"      then run:  python -m src.from_ward_to_roboflow_dataset \\")
    print(f"                    --input-root {args.out}")

sim_app.close()
