"""Dump the semantics AUTHORED in the ward USD stage (hand-set in the
Semantic Schema Editor) -- no Isaac app, no rendering, plain pxr.

For every prim that carries a Semantics API attribute, print its path,
semanticType, semanticData (the class label), and whether the label exists in
fixed_categories.py. Run with Isaac's bundled pxr:

    P=/home/edge-host/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.la64.r.cp311
    PYTHONPATH=$P LD_LIBRARY_PATH=$P/bin \
        /home/edge-host/isaac-sim/kit/python/bin/python3 inspect_semantics.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from pxr import Usd

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "ROS2_bridge" / "src"))
from fixed_categories import FIXED_CATEGORIES  # noqa: E402

STAGE = sys.argv[1] if len(sys.argv) > 1 else str(
    PROJECT / "Collected_Ward0505" / "Ward0505.usd")

stage = Usd.Stage.Open(STAGE)
print(f"[stage] {STAGE}")

rows = []          # (prim_path, depth_under_world, sem_type, label)
for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
    for attr in prim.GetAttributes():
        n = attr.GetName()
        if n.startswith("semantic:") and n.endswith(":params:semanticData"):
            label = attr.Get()
            t_attr = prim.GetAttribute(
                n.replace(":semanticData", ":semanticType"))
            sem_type = t_attr.Get() if t_attr else "?"
            parts = prim.GetPath().pathString.split("/")
            depth = len(parts) - 2  # /World/X -> 1
            rows.append((prim.GetPath().pathString, depth,
                         str(sem_type), str(label)))

print(f"\n[authored] {len(rows)} prims carry a semantic label")
by_label = Counter(r[3] for r in rows)
print("\n== label counts ==")
for label, n in sorted(by_label.items()):
    ok = "ok" if label in FIXED_CATEGORIES else "NOT IN TAXONOMY"
    print(f"  {n:4d} x {label:28s} [{ok}]")

print("\n== per-prim (depth>1 = nested under a top-level asset) ==")
for path, depth, sem_type, label in sorted(rows):
    flag = "" if label in FIXED_CATEGORIES else "   <-- NOT IN TAXONOMY"
    nest = "  NESTED" if depth > 1 else ""
    print(f"  d{depth} {sem_type:6s} {label:26s} {path}{nest}{flag}")

# top-level prims with NO authored semantics anywhere in their subtree
tops = stage.GetPrimAtPath("/World").GetChildren()
covered = {r[0].split("/")[2] for r in rows if len(r[0].split("/")) > 2}
bare = [t.GetName() for t in tops if t.GetName() not in covered]
print(f"\n== top-level prims with no authored semantics in subtree "
      f"({len(bare)}) ==")
for n in sorted(bare):
    print(f"  {n}")
