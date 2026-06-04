"""
One-shot inspector for an Isaac Sim USD stage.

Run with:
    ~/isaac-sim/python.sh inspect_stage.py /home/edge-host/Documents/GitHub/hospital_dataset/Collected_Ward0505/Ward0505.usd

Prints the information I need before I write the main Replicator generator:
  - up-axis / units / scene scale
  - world AABB (so we know how to place the camera)
  - top-level prims and an excerpt of their subtrees
  - every semantic label found on prims (and which prims carry each)
  - all lights and their type / intensity / color
  - any animated cameras already in the scene
  - per-prim counts of types we care about (Mesh, Xform, Light, Camera, Material)

Goal: ~3 minutes to run, output fits in a paste.
"""

from isaacsim import SimulationApp  # must be imported FIRST in Isaac Sim 5.x
import sys
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("stage", help="Path to the USD stage to inspect")
    p.add_argument("--max-prims-per-section", type=int, default=40,
                   help="Max prim paths to list per category (truncates long output)")
    return p.parse_args()


# Parse args BEFORE the SimulationApp starts so --help works cleanly
args = parse_args()

# Headless Kit startup. The boolean "renderer" is unused; Isaac just needs to
# initialize so we can use omni.usd / pxr.
simulation_app = SimulationApp({"headless": True})

# These imports must come AFTER SimulationApp() — that's when Kit loads.
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, UsdLux, Sdf, Gf  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402


def banner(title):
    print("\n" + "=" * 8 + " " + title + " " + "=" * 8)


def short(s, n=80):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


# --- open the stage ---
ctx = omni.usd.get_context()
ok = ctx.open_stage(args.stage)
if not ok:
    print(f"ERROR: could not open stage: {args.stage}")
    simulation_app.close()
    sys.exit(1)
stage: Usd.Stage = ctx.get_stage()

banner("STAGE METADATA")
print(f"path      : {args.stage}")
print(f"up axis   : {UsdGeom.GetStageUpAxis(stage)}")
print(f"meters/unit: {UsdGeom.GetStageMetersPerUnit(stage):.6f}")
print(f"time codes: start={stage.GetStartTimeCode()} end={stage.GetEndTimeCode()} "
      f"fps={stage.GetTimeCodesPerSecond()}")

# --- world AABB ---
banner("WORLD AABB")
try:
    bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                               [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    world = bcache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedBox()
    bmin, bmax = world.GetMin(), world.GetMax()
    size = bmax - bmin
    print(f"min  : ({bmin[0]:.3f}, {bmin[1]:.3f}, {bmin[2]:.3f})")
    print(f"max  : ({bmax[0]:.3f}, {bmax[1]:.3f}, {bmax[2]:.3f})")
    print(f"size : ({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})  (in stage units)")
except Exception as e:
    print(f"could not compute world AABB: {e}")

# --- top-level structure ---
banner("TOP-LEVEL PRIMS UNDER /")
for p in stage.GetPseudoRoot().GetChildren():
    print(f"  {p.GetPath()}  ({p.GetTypeName() or '<typeless>'})")

# --- type counts across the whole stage ---
banner("PRIM TYPE COUNTS (whole stage)")
type_counts: Counter = Counter()
all_prims = 0
for prim in stage.Traverse():
    all_prims += 1
    type_counts[str(prim.GetTypeName() or "<typeless>")] += 1
print(f"total prims: {all_prims}")
for tname, n in type_counts.most_common():
    print(f"  {n:6d}  {tname}")

# --- semantic labels ---
# In Replicator / Isaac Sim, a "Semantics" schema (singular) stores
# (semantic_type, semantic_data) pairs. Sometimes labels are also encoded as
# customData["semantics"] or as `Semantics_*` API. We try the standard one.
banner("SEMANTIC LABELS")
semantic_to_prims = defaultdict(list)
labeled_count = 0
for prim in stage.Traverse():
    # Standard: Semantics API on the prim
    api_names = prim.GetAppliedSchemas()
    if "SemanticsAPI" in api_names or any("Semantics" in n for n in api_names):
        labeled_count += 1
        # Read the typeName / data attributes
        type_attr = prim.GetAttribute("semantic:Semantics:params:semanticType")
        data_attr = prim.GetAttribute("semantic:Semantics:params:semanticData")
        s_type = type_attr.Get() if type_attr and type_attr.HasAuthoredValue() else None
        s_data = data_attr.Get() if data_attr and data_attr.HasAuthoredValue() else None
        if s_type or s_data:
            key = (str(s_type), str(s_data))
            semantic_to_prims[key].append(str(prim.GetPath()))
        else:
            # Multi-instance Semantics — iterate over instances
            for name in api_names:
                if name.startswith("SemanticsAPI:"):
                    inst = name.split(":", 1)[1]
                    t = prim.GetAttribute(f"semantic:{inst}:params:semanticType").Get()
                    d = prim.GetAttribute(f"semantic:{inst}:params:semanticData").Get()
                    if t or d:
                        semantic_to_prims[(str(t), str(d))].append(str(prim.GetPath()))
print(f"prims carrying Semantics API: {labeled_count}")
print(f"distinct (semanticType, semanticData) pairs: {len(semantic_to_prims)}")
items = sorted(semantic_to_prims.items(), key=lambda kv: -len(kv[1]))
for (stype, sdata), prims in items[: args.max_prims_per_section]:
    sample = short(", ".join(prims[:3]), 110)
    print(f"  ({stype!s:20s}, {sdata!s:35s}) -> {len(prims):4d} prims  e.g. {sample}")
if len(items) > args.max_prims_per_section:
    print(f"  ... {len(items) - args.max_prims_per_section} more semantic groups")

# --- lights ---
banner("LIGHTS")
light_types = ("DistantLight", "DomeLight", "RectLight", "DiskLight",
               "SphereLight", "CylinderLight", "GeometryLight")
for prim in stage.Traverse():
    tname = str(prim.GetTypeName() or "")
    if tname in light_types:
        light = UsdLux.LightAPI(prim) if hasattr(UsdLux, "LightAPI") else None
        try:
            intensity = light.GetIntensityAttr().Get() if light else None
            color = light.GetColorAttr().Get() if light else None
            color_str = (f"({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})"
                         if color is not None else "?")
        except Exception:
            intensity = "?"
            color_str = "?"
        print(f"  {prim.GetPath()}  type={tname}  intensity={intensity}  color={color_str}")

# --- cameras already in the scene ---
banner("CAMERAS IN STAGE")
n_cams = 0
for prim in stage.Traverse():
    if prim.GetTypeName() == "Camera":
        cam = UsdGeom.Camera(prim)
        focal = cam.GetFocalLengthAttr().Get()
        hap = cam.GetHorizontalApertureAttr().Get()
        vap = cam.GetVerticalApertureAttr().Get()
        # Compute horizontal FOV (degrees) for the curious
        import math
        try:
            hfov = math.degrees(2 * math.atan((hap / 2.0) / focal))
        except Exception:
            hfov = None
        print(f"  {prim.GetPath()}  focal={focal}  aperture={hap}x{vap}  "
              f"HFOV≈{hfov:.1f}°" if hfov else
              f"  {prim.GetPath()}  focal={focal}  aperture={hap}x{vap}")
        n_cams += 1
if n_cams == 0:
    print("  (no Camera prims found)")

# --- candidates that probably correspond to fixed_categories.py classes ---
# This is rough: we just scan top-level Xform prim *names* and group them.
banner("XFORM-LIKE PRIM NAMES (probable assets)")
# Strip USD instance counters so e.g. Bed_01, Bed_02 collapse to "Bed"
import re
name_re = re.compile(r"^([A-Za-z][A-Za-z0-9]*?)([_\-]?\d+)?$")
name_counts: Counter = Counter()
for prim in stage.Traverse():
    if prim.GetTypeName() not in ("Xform", "Scope", "Mesh"):
        continue
    n = prim.GetName()
    m = name_re.match(n)
    base = m.group(1) if m else n
    name_counts[base] += 1
# Hide tiny generic things; show the meaningful ones
top = [kv for kv in name_counts.most_common() if len(kv[0]) > 2 and kv[1] >= 1]
for base, n in top[: args.max_prims_per_section]:
    print(f"  {n:4d}  {base}")
if len(top) > args.max_prims_per_section:
    print(f"  ... {len(top) - args.max_prims_per_section} more name groups")

# --- existing render products / replicator graphs ---
banner("EXISTING /Replicator OR /OmniGraph PRIMS")
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if path.startswith("/Replicator") or path.startswith("/OmniGraph"):
        print(f"  {path}  ({prim.GetTypeName() or '<typeless>'})")

print("\nDone. Paste this whole output back to me so I can write the generator.")
simulation_app.close()
