# Infinigen procedural generation — investigation & porting report

**Goal of this doc.** We want to *emulate Infinigen's procedural process in our own engine* (the
placement code under `src/layout/`, or a successor) so we can build out whole rooms — walls,
ceilings, floors, and furniture placement — and feed our own **`.usdz` hospital assets** into it,
while keeping our Isaac/USD + Cosmos pipeline. This report is the result of reading the Infinigen
source (`princeton-vl/infinigen`, cloned to `~/Documents/GitHub/infinigen`, v1.19.1 / Blender 4.2).

It answers three questions:
1. How does Infinigen's procedural process actually work (so we can port it)?
2. If we already have `.usdz` assets, can we use them?
3. Can it piece **wall structures, ceilings, and floors** into different room settings?

> Platform note: Infinigen is Blender-based and x86_64/Mac only. On our aarch64 DGX Spark it runs
> only via **Docker x86_64 emulation** (image `infinigen-emu`, see [[infinigen-evaluation]] memory),
> CPU-only and slow. That's fine for *studying/validating* it; it is **not** a production path here.
> This report treats Infinigen as a **reference design**, not a dependency.

---

## TL;DR

- **Yes, Infinigen builds the entire shell procedurally** — a room-adjacency graph → 2D room
  polygons → a "solidify" step that extrudes walls to height/thickness and **boolean-cuts doors and
  windows**, generating floor + ceiling + wall meshes with semantic face tags. Multi-story by z-offset.
- **Yes, our `.usdz` assets can be used** — Infinigen has a first-class `StaticAssetFactory` that
  imports external meshes (`.usd` via `bpy.ops.wm.usd_import`, plus obj/fbx/glb/blend/...). You give
  it a folder, a target dimension, an orientation, and `tag_support=True`; the solver then places it
  like any procedural asset. Infinigen also **exports scenes to USDC** with an `--omniverse`/Isaac path.
- **Yes, walls/ceilings/floors are pieced into room settings** — that's exactly the
  `BlueprintSolidifier` step, and there's a `PredefinedFloorPlanSolver` that takes a **JSON/YAML of
  room polygons + door/window line-segments**, which is the cleanest hook for *our* layouts.
- **Our `src/layout/solver.py` already implements a subset of Infinigen's solver** (zones = degrees of
  freedom, hard non-overlap by rejection, simulated annealing for soft preferences). The gap to close
  is the **richer constraint DSL/relations**, **multi-stage add/remove of objects**, and the
  **shell construction**. See the porting roadmap (§5–6).

---

## 1. The end-to-end pipeline

```
RoomGraph (adjacency topology, room types)
   └─ infinigen/core/constraints/example_solver/room/graph.py  (GraphMaker)
        ↓
2D room polygons (shapely), recursive box-cutting + backtracking assignment
   └─ room/segment.py  (SegmentMaker)  → State.objs[name] = ObjectState(polygon=...)
        ↓
Floor-plan optimization (simulated annealing: extrude/swap rooms)
   └─ room/floor_plan.py (FloorPlanSolver) + room/solver.py (FloorPlanMoves)
        ↓
SHELL MESH: walls/floor/ceiling + door/window/opening cutouts (boolean)
   └─ room/solidifier.py (BlueprintSolidifier)  ← the "build the room" step
        ↓
OBJECT PLACEMENT: constraint DSL + simulated-annealing solver
   └─ core/constraints/ (DSL) + example_solver/{solve,annealing,moves,geometry/dof}.py
        ↓
Export → USDC (+ --omniverse for Isaac, with solve_state.json physics)
   └─ infinigen/tools/export.py ; core/sim/exporters/usd_exporter.py
```

Entry point: `infinigen_examples/generate_indoors.py`; constraint program for homes:
`infinigen_examples/constraints/home.py`.

---

## 2. Building the shell: walls, ceilings, floors → room settings

This is the part most directly relevant to "piece wall/ceiling/floor into room settings." It's all
in `infinigen/core/constraints/example_solver/room/`.

### 2.1 Floorplan → 2D room polygons
- **`base.py` `RoomGraph`** — adjacency topology; rooms named `"{Type}_{level}/{idx}"`
  (e.g. `Bedroom_0/0`). `room_type(name)`, `room_level(name)` parse it.
- **`graph.py` `GraphMaker`** — grows the room graph node-by-node under constraints; `get_typical_areas()`
  picks per-room-type areas that satisfy the constraints, which sets overall building dimensions.
- **`segment.py` `SegmentMaker`** — recursive box-cutting of a contour into room polygons, assigned to
  graph nodes by backtracking. Output: each room is a `shapely.Polygon` (XY), with `SharedEdge`
  relations to neighbours.
- **`floor_plan.py` `FloorPlanSolver.solve()`** → `(State, room_types, dimensions)`; uses
  **`solver.py` `FloorPlanMoves.perturb_state()`** (simulated annealing) to extrude rooms in/out and
  swap adjacent rooms.

### 2.2 Solidify: polygons → actual mesh (the key recipe)
`solidifier.py` `BlueprintSolidifier.make_room(state, name)`:
1. `polygon2obj(segmentize(polygon, door_width))` → flat 2D floor mesh.
2. `SOLIDIFY (thickness=wall_height, offset=-1)` → extrude walls up to ceiling height.
3. `SOLIDIFY (thickness=wall_thickness/2, use_even_offset)` → give walls physical thickness.
   Result per room: floor at `z≈wall_thickness/2`, ceiling at `z≈wall_height-wall_thickness/2`,
   walls between, with interior faces tagged.

**Doors / windows / openings** are made as **cutter solids** then boolean-subtracted:
- `make_door_cutter(mls, direction)` — cube `~(door_width, wall_thickness+, door_size)` placed
  randomly along a wall edge, z-centered on the door.
- `make_window_cutter(mls, is_panoramic)` — cube at window height; panoramic ≈ full-height.
- `make_open_cutter`, `make_entrance_cutter` — full openings / building entrance.
- Applied via `modify_mesh(room, "BOOLEAN", object=cutter, operation="DIFFERENCE")`.
Which connector each shared edge gets (none/open/door/window/panoramic) is a **rules matrix keyed by
the two room types** (`combined_rooms` in `solidifier.py`).

**Semantic face tags** (Blender face attributes) written per room — directly usable downstream:
`SupportSurface` (floor, `z<wall_thickness/2`), `Ceiling`, `Wall`, `Interior` (non-exterior),
`Visible`.

**Parameters** (`constraint_language/constants.py`, `RoomConstants`):
`wall_thickness ~ U(0.2,0.3)`, `wall_height ~ U(2.8,3.2)`, `door_width ≈ 1.0–1.1`,
`door_size ~ U(2.0,2.4)`, window height/size/margins, `unit=0.5` grid.

### 2.3 Multi-story & predefined layouts
- Multi-story: one `BlueprintSolidifier` per level; rooms offset `location.z += wall_height*level`;
  `fixed_contour` reuses the lower floor's outline.
- **`predefined.py` `PredefinedFloorPlanSolver`** — accepts a **dict/JSON/YAML/pickle** of:
  ```
  { "rooms":   { "Bedroom_0/0": {"shape": <Polygon>}, ... },
    "doors":   { "door_0":   {"shape": <LineString>} },
    "windows": { "window_0": {"shape": <LineString>, "is_panoramic": false} },
    "opens":   { "open_0":   {"shape": <LineString>} } }
  ```
  → applies cutters directly. **This is the cleanest contract for feeding *our* ward floorplan**
  (and mirrors what we'd build natively).

**Takeaway for us:** shell construction is conceptually simple and portable to USD — extrude room
polygons to wall/floor/ceiling boxes, boolean-cut door/window rectangles, tag faces. No Blender
required to replicate the *idea*; only the convenience of its modifiers.

---

## 3. The placement brain: constraint DSL + solver

All under `infinigen/core/constraints/`. This is the part our `src/layout/solver.py` already partially
mirrors.

### 3.1 The DSL (`constraint_language/`)
- **`Problem`** = `{constraints: dict[str,BoolExpression]}` (hard) + `{score_terms: dict[str,ScalarExpression]}` (soft).
- **Object sets**: `scene()`, `tagged(objs, tags)`, `related_to(child, parent, relation)`, `count()`,
  `in_range(low, high)`, set ops; quantifiers `.all(lambda r: ...)` (ForAll) and `.mean(...)` (avg soft score).
- **Spatial relations** (`relations.py`): `StableAgainst(child_tags, parent_tags, margin, check_z)`
  (rests-on / against, the workhorse), `SupportedBy`, `Touching`, `CoPlanar`, `RoomNeighbour`,
  plus query relations (`Traverse`, `CutFrom`, `SharedEdge`). Each can be negated.
- **Metrics** for soft terms: `distance`, `accessibility_cost`, `angle_alignment_cost`,
  `freespace_2d`, `min_dist_2d`, `volume`.
- Example (kitchen sink, `constraints/home.py`): a `Sink` from `SinkFactory`, `StableAgainst` the
  countertop's `SupportSurface` with `margin=0.001`, `CoPlanar` front-to-front, constrained to
  0–1 per counter. Reads almost like English.

### 3.2 The solver (`example_solver/`)
- **Multi-stage** (`solve.py`): rooms → large furniture → medium → small objects. Each stage filters
  the constraints to a `Domain` (tags + relations) and solves only those.
- **Simulated annealing** (`annealing.py`): exponential cooling
  `T_k = T0·r^k, r=(Tf/T0)^(1/steps)`; **Metropolis with violation priority** — accept if hard-violation
  count drops, reject if it rises, else standard `exp(-ΔE/T)`. Lazy memoized evaluation (only
  re-score constraints touched by a move).
- **Moves** (`moves/`, weighted by decaying schedules): `Addition` (sample an asset factory + assign
  relations), `Deletion`, `TranslateMove`, `RotateMove`, `ReinitPoseMove`, `Resample`,
  `RelationPlaneChange`/`RelationTargetChange` (re-pick which wall/parent).
- **Degrees of freedom** (`geometry/dof.py`): *each relation removes a DOF.* Translation freedom is a
  rank-deficient 3×3 projection matrix (`dof_matrix_translation`), rotation freedom is an axis
  (`dof_rotation_axis`) or `None`. Moves sample noise and **project through the DOF matrix** so the
  object stays on its surface — **this is exactly our "zone" concept.**
- **State** (`state_def.py`): per object `{obj, polygon, generator, tags, relations, dof matrices}`.
- **Bounds** (`reasoning/constraint_bounding.py`): `count(domain).in_range(low,high)` drives how many
  objects of each kind to add/remove in a stage.

### 3.3 How this maps to *our* `src/layout/solver.py`
| Infinigen | Us (today) | Gap to port |
|---|---|---|
| DOF projection matrices | zones (wall/floor/surface/fixed) | generalize to N-relation DOF |
| hard constraints via violation-priority MH | hard non-overlap by **rejection** | add relation-based hard constraints |
| soft `score_terms` (accessibility, alignment) | `W_AGAINST`, `W_NEARBED`, `W_WALLHUG` | add accessibility/clearance/alignment terms |
| `Addition`/`Deletion` (variable object count) | fixed object set | add object add/remove (presence) |
| relation DSL + tags | `affordances.py` (support + host classes) | grow into a small relation DSL |
| multi-stage (rooms→large→small) | single stage + stage-2 surfaces | formalize stages |
| `StableAgainst`/`RelationPlaneChange` | wall snapping + host pick | add "re-pick wall/parent" moves |

We already independently arrived at the **DOF/zone + hard-constraint + SA** core. The valuable things
to lift are the **declarative relation/tag model**, **variable object counts**, and **soft-cost terms**
(accessibility especially — keeps furniture reachable, which we lack).

---

## 4. Using our `.usdz` assets

Infinigen has a first-class external-asset path: **`StaticAssetFactory`**
(`infinigen/assets/static_assets/base.py`), with an import map that includes USD:

```python
import_map = { "usd": bpy.ops.wm.usd_import, "obj": ..., "fbx": ..., "glb": ..., "blend": ... }
```

Workflow (`docs/StaticAssets.md`, `static_assets/static_category.py`):
1. Drop meshes in `infinigen/assets/static_assets/source/<Category>/` (multiple files → random pick).
2. `Static<Category>Factory = static_category_factory(path, z_dim=..., rotation_euler=..., tag_support=True)`.
3. Register in `static_assets/__init__.py`; tag it in `constraints/semantics.py`
   (`used_as[Semantics.Furniture] |= {StaticHospitalBedFactory}`); add placement constraints.
4. Generate; optionally `infinigen.tools.export -f usdc [--omniverse]` → Isaac.

**Requirements the solver needs from each asset:**
- **Dimensions** — read automatically from the imported mesh's bounding box (`obj.dimensions`).
- **Scale** — one target dimension (`z_dim=1.8` etc.) scales uniformly.
- **Orientation convention** — **+X front, −X back, +Z up, −Z bottom**; otherwise pass `rotation_euler`.
- **Placement surfaces** — `tag_support=True` runs `tag_support_surfaces()` to mark +Z faces so other
  objects can be placed on top.
- **Semantics** — a `Semantics.*` tag mapping the asset to a role (Furniture/Storage/Sink/...).

**USD import/export specifics:**
- **Import**: `.usd` is directly supported (`bpy.ops.wm.usd_import`). ⚠️ **`.usdz` caveat**: the import
  map keys on the file *extension* `"usd"`, and `usd_import` reads `.usdz` (zip container) in Blender
  generally — but our factory would need the `usdz` extension wired in (or convert `.usdz`→`.usd`).
  **To verify on first use.**
- **Export**: full scenes export to **USDC only** (`tools/export.py`, `bpy.ops.wm.usd_export`,
  `root_prim_path="/World"`, textures baked). `--omniverse` prepares it for **Isaac Sim/Omniverse**,
  emitting a `solve_state.json` for physics; there's a `core/sim/exporters/usd_exporter.py`
  `USDBuilder` (physics, materials, COACD collision).

So there are **two ways our usdz assets connect to Infinigen** — see §5.

---

## 5. Two adoption paths

**Path A — Use Infinigen wholesale, feed it our usdz.**
Register our `.usdz` hospital assets as `StaticAssetFactory`s, write hospital room/placement
constraints, generate on an **x86_64 box** (or emulated, slowly), export **USDC → Isaac**, then
Cosmos as today.
- *Pros:* get Infinigen's full solver + shell + materials immediately; least new code.
- *Cons:* Blender dependency; aarch64 can't run it natively (x86 box/cloud or emulation); must map
  Infinigen's annotations to *our* hospital taxonomy for Cosmos; `.usdz` import to verify; heavyweight.

**Path B — Port Infinigen's design into our own pure-Python engine (over USD/Isaac).**
Extend `src/layout/` to (a) build the **shell** (rooms→walls/floors/ceilings + door/window cutouts as
USD prims) and (b) grow the **solver** with Infinigen's relation DSL, variable object counts, and
soft-cost terms.
- *Pros:* keeps our Isaac+Cosmos pipeline and label taxonomy; no Blender; runs natively on the DGX
  Spark; our usdz assets are already first-class.
- *Cons:* we reimplement the shell construction and a richer DSL (but we already have the solver core).

**Recommendation — hybrid:** pursue **Path B** as the product direction, and keep an **Infinigen
reference instance** (emulated or on a cheap x86 box) to *generate ground-truth scenes we compare our
port against*. Concretely: port the **shell** first (highest value, currently missing — our rooms are
fake centroid AABBs), then enrich the **solver** with relations + accessibility cost.

---

## 6. Porting roadmap (mapped to our code)

1. **Shell construction → `src/layout/shell.py` (new).**
   Input: room polygons + door/window segments (mirror `PredefinedFloorPlanSolver`'s JSON contract).
   Output: USD prims — wall boxes (extrude polygon edges by `wall_thickness`, height `wall_height`),
   floor + ceiling slabs, door/window cutouts (boolean or gap), with semantic labels. This replaces
   our centroid-derived "rooms" with real walls and is the single biggest correctness win (recall the
   wall-adherence / room-leak bugs in [[slot-layout-redesign]]).
2. **Solver enrichment → `src/layout/solver.py` + `affordances.py`.**
   - Generalize zones to **N-relation DOF** (Infinigen's `dof_matrix_translation`/`dof_rotation_axis`).
   - Add a small **relation/tag DSL** (StableAgainst / on-surface / against-wall / near / coplanar) on
     top of our affordance model.
   - Add **soft-cost terms**: `accessibility_cost` (reachability), alignment, spacing.
   - Add **variable object counts** (add/remove via `count.in_range`) for composition variety.
3. **Assets.** Our dump already captures usdz path + transform; add **support-surface tags** and
   **dimensions/semantics** metadata so the solver can place riders generally (we hit this exact gap).
4. **Validation.** Run Infinigen indoor (emulated/x86) on a comparable single room; compare our port's
   layouts qualitatively + on our offline checks (`tests/test_layout.py`).

---

## 7. Key Infinigen file map (reference)

| Area | File |
|---|---|
| Indoor entry point | `infinigen_examples/generate_indoors.py` |
| Home constraint program | `infinigen_examples/constraints/home.py`, `semantics.py` |
| Constraint DSL | `infinigen/core/constraints/constraint_language/{relations,result,set_reasoning,constants,rooms}.py` |
| Solver | `infinigen/core/constraints/example_solver/{solve,annealing}.py` |
| Moves | `…/example_solver/moves/{addition,deletion,pose,reassignment}.py` |
| DOF / geometry | `…/example_solver/geometry/dof.py`, `state_def.py` |
| Evaluation / reasoning | `…/evaluator/evaluate.py`, `…/reasoning/{constraint_domain,constraint_bounding}.py` |
| **Shell: rooms/walls/floors** | `…/example_solver/room/{base,graph,segment,floor_plan,solver,solidifier,contour,predefined}.py` |
| Asset factory | `infinigen/core/placement/factory.py` |
| **Static (external) assets** | `infinigen/assets/static_assets/{base,static_category}.py`, `docs/StaticAssets.md` |
| Support-surface tagging | `infinigen/core/tagging.py` (`tag_support_surfaces`) |
| Export USD / Isaac | `infinigen/tools/export.py`, `core/sim/exporters/usd_exporter.py`, `docs/ExportingTo{ExternalFileFormats,Simulators}.md` |

## 8. How to run Infinigen here (reference only)

```bash
# image already built (Docker x86_64 emulation; see infinigen-evaluation memory)
docker run --rm --platform linux/amd64 \
  -v ~/Documents/GitHub/infinigen-out:/infinigen/outputs infinigen-emu \
  python -m infinigen_examples.generate_indoors --seed 0 --task coarse \
  --output_folder outputs/<name> -g fast_solve.gin singleroom.gin \
  -p compose_indoors.terrain_enabled=False \
  restrict_solving.restrict_parent_rooms=\[\"Bathroom\"\]
```
CPU-only under emulation (no GPU/Cycles); use for studying outputs, not throughput.
