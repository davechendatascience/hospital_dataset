"""Explicit placement affordances -- the single, declared source of truth for
HOW each object class is held up and whether things can rest on it.

This replaces the fragile z-threshold role INFERENCE in placement_dr (which, for
example, misread a low-mounted gas_manifold as floor furniture and teleported it
across the room). Every object now declares its support mode up front:

  support:
    "floor"   -- stands on the floor; under gravity it settles at floor level.
    "wall"    -- attached to a wall at a height; gravity does NOT apply (it would
                 fall if it weren't bolted on). Slides only along its wall.
    "surface" -- rests on TOP of some support object (table/bed/counter); under
                 gravity it settles onto that support's top face.
    "fixed"   -- built-in fixture (toilet, sink, door); never moves.

  provides_surface: True if OTHER objects may be placed on this one's top
                    (a table, an overbed table, a bed, a counter).

This directly answers the three things to "specify clearly":
  * attached to walls          -> support == "wall"
  * can be placed on a surface -> support == "surface" (and its host has
                                  provides_surface == True)
  * placed on the floor        -> support == "floor"

A grounding check (placement_dr / test_slot_layout) is then just: every non-wall,
non-fixed object must have something directly under it (floor or a support top);
if its bottom floats above that, the placement is INVALID -- the analytic
equivalent of "let physics drop it and see if it falls".

NOTE: this CLASS table is the DEFAULT. A per-instance `affordance` carried on a
slot (set by the dump from the authored scene) OVERRIDES it -- the same class
can be wall-mounted in one spot and sitting on a counter in another (e.g. a
sanitizer), so the instance wins when known.
"""

# support, provides_surface
_FLOOR = ("floor", False)
_FLOOR_SUPPORT = ("floor", True)      # stands on floor AND holds things on top
_WALL = ("wall", False)
_SURFACE = ("surface", False)
_FIXED = ("fixed", False)
_FIXED_SUPPORT = ("fixed", True)      # built-in but holds things (counter, sink rim)

AFFORDANCES = {
    # --- floor-standing furniture ---
    "hospital_bed":            _FLOOR_SUPPORT,
    "companion_chair":         _FLOOR,
    "stool":                   _FLOOR,
    "weight_scale":            _FLOOR,   # the floor scale you stand on
    "iv_pole":                 _FLOOR,
    "waste_bin":               _FLOOR,
    "medical_waste_container": _FLOOR,
    "soiled_linen_bin":        _FLOOR,
    "overbed_table":           _FLOOR_SUPPORT,
    "bedside_table":           _FLOOR_SUPPORT,

    # --- wall-mounted equipment (gravity does NOT apply) ---
    "bedside_monitor":         _WALL,
    "gas_manifold":            _WALL,
    "oxygen_flowmeter":        _WALL,
    "suction_jar":             _WALL,
    "suction_knob":            _WALL,
    "telephone":               _WALL,
    "tissue_dispenser":        _WALL,
    "sanitizer":               _WALL,
    "hook":                    _WALL,
    "light_switch":            _WALL,
    "mirror":                  _WALL,
    "air_vent":                _WALL,
    "curtain":                 _WALL,
    "bed_curtain":             _WALL,
    "TV":                      _WALL,

    # --- small items that rest on a surface (table/bed/counter top) ---
    "remote_control":          _SURFACE,
    "ear_thermometer":         _SURFACE,
    "medical_package":         _SURFACE,
    "gauze":                   _SURFACE,
    "medical_gloves":          _SURFACE,
    "syringe":                 _SURFACE,
    "stethoscope":             _SURFACE,
    "alcohol_spray_bottle":    _SURFACE,
    "paperbox":                _SURFACE,
    "scale":                   _SURFACE,   # small counter/handheld scale

    # --- built-in fixtures (never move) ---
    "toilet":                  _FIXED,
    "sink":                    _FIXED_SUPPORT,
    "shower":                  _FIXED,
    "door":                    _FIXED,
    "door_handle":             _FIXED,
    "window":                  _FIXED,
}

_DEFAULT = _FLOOR    # unknown class -> assume it stands on the floor


def support_of(cls):
    """-> "floor" | "wall" | "surface" | "fixed" for a class (default floor)."""
    return AFFORDANCES.get(cls, _DEFAULT)[0]


def provides_surface(cls):
    """-> True if other objects may rest on this class's top face."""
    return AFFORDANCES.get(cls, _DEFAULT)[1]


# slot-type (slot_layout) <-> support mode, so the spec's per-instance slot
# overrides the class default.
SLOT_TO_SUPPORT = {"wall": "wall", "floor": "floor", "surface": "surface",
                   "bay": "floor"}


# ---------------------------------------------------------------------------
# HOST EQUIVALENCE CLASSES -- the "style choice" constraints.
#
# A surface object doesn't rest on ONE specific thing; it rests on any member of
# an INTERCHANGEABLE set of hosts. "Anything that can sit on the bed can sit on
# the bedside table" => bed + bedside_table + overbed_table are one host group;
# a paperbox targeting that group is randomly placed on ANY of them. Floors of
# different rooms can be grouped the same way ("frontroom counter ~ bathroom
# floor"), so a host can be a support-surface class OR a room floor.
#
# A host entry is either:
#   ("on", "<class>")   -- on top of any object of that class (a receptacle)
#   ("floor", "<room>") -- on the floor of that room ("*" = any room)
# ---------------------------------------------------------------------------
HOST_GROUPS = {
    # flat tops around the bed -- interchangeable for small items
    "bed_top": [("on", "hospital_bed"), ("on", "bedside_table"),
                ("on", "overbed_table")],
    # low surfaces a small item can equally sit on: a counter top OR a floor
    "low_surface": [("on", "sink"), ("floor", "Bathroom"), ("floor", "Frontroom")],
}

# surface object class -> the host group(s) it may be randomly placed in
SURFACE_PLACEMENT = {
    "remote_control":       ["bed_top"],
    "ear_thermometer":      ["bed_top"],
    "medical_package":      ["bed_top"],
    "paperbox":             ["bed_top"],
    "gauze":                ["bed_top"],
    "medical_gloves":       ["bed_top"],
    "syringe":              ["bed_top"],
    "stethoscope":          ["bed_top"],
    "alcohol_spray_bottle": ["bed_top", "low_surface"],
}


def host_specs_for(cls):
    """-> list of host specs ("on",<class>) / ("floor",<room>) a surface object
    of `cls` may be placed on, expanded from its host groups. Empty -> fall back
    to whatever it geometrically rests on."""
    out = []
    for group in SURFACE_PLACEMENT.get(cls, []):
        out.extend(HOST_GROUPS.get(group, []))
    return out
