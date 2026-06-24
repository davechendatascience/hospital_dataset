FIXED_CATEGORIES = {
  "ward_object": 0,
  "TV": 1,
  "air_vent": 2,
  "alcohol_spray_bottle": 3,
  "bed_curtain": 4,
  "bedside_monitor": 5,
  "bedside_table": 6,
  "cabinet": 7,
  "companion_chair": 8,
  "curtain": 9,
  "door": 10,
  "door_handle": 11,
  "ear_thermometer": 12,
  "gas_manifold": 13,
  "gauze": 14,
  "hook": 15,
  "hospital_bed": 16,
  "iv_pole": 17,
  "light_switch": 18,
  "medical_gloves": 19,
  "medical_package": 20,
  "medical_waste_container": 21,
  "mirror": 22,
  "overbed_table": 23,
  "oxygen_flowmeter": 24,
  "paperbox": 25,
  "remote_control": 26,
  "sanitizer": 27,
  "scale": 28,
  "shower": 29,
  "sink": 30,
  "soiled_linen_bin": 31,
  "stethoscope": 32,
  "stool": 33,
  "suction_jar": 34,
  "suction_knob": 35,
  "syringe": 36,
  "telephone": 37,
  "tissue_dispenser": 38,
  "toilet": 39,
  "toilet_handle": 40,
  "waste_bin": 41,
  "weight_scale": 42,
  "window": 43
}


# Labels authored in the ward USD (Semantic Schema Editor) that don't match a
# taxonomy name exactly. Keys are lowercase authored labels.
LABEL_ALIASES = {
    "light_switcher": "light_switch",
    "solid_linen_bin": "soiled_linen_bin",   # asset typo
    "bedcurtain": "bed_curtain",
}

_LOWER_TO_CANONICAL = {k.lower(): k for k in FIXED_CATEGORIES}


def normalize_label(label):
    """Map an authored semantic label onto its FIXED_CATEGORIES name.

    Tolerates case differences ('Mirror' -> 'mirror', 'tv' -> 'TV') and the
    LABEL_ALIASES above. Returns None when the label has no taxonomy match
    (e.g. 'handle', 'cabinet_door', 'access_sensor').
    """
    s = str(label).strip()
    if s in FIXED_CATEGORIES:
        return s
    low = LABEL_ALIASES.get(s.lower(), s.lower())
    return _LOWER_TO_CANONICAL.get(low)


def class_from_entry(entry):
    """Extract a taxonomy class from a Replicator labels/semantics-mapping
    JSON entry. Tolerates the old-API layout ({'class': 'a,b'}) and new
    UsdSemantics layouts by scanning every string value for the first
    comma-separated token that normalizes into the taxonomy."""
    if not isinstance(entry, dict):
        return None
    for v in entry.values():
        if not isinstance(v, str):
            continue
        for tok in v.split(","):
            cls = normalize_label(tok)
            if cls is not None:
                return cls
    return None


# Two-level supercategory tree (the COCO `categories[].supercategory` field).
# Groups the 43 leaf classes; standard COCO eval stays per-leaf-category, but
# the supercategory documents the hierarchy (e.g. cabinet & its parts live
# under "furniture") and enables coarse-grained analysis. Edit the groups here
# -- the per-class lookup and the dataset builders derive from this.
SUPERCATEGORY_TREE = {
    "furniture":         ["hospital_bed", "bedside_table", "overbed_table",
                          "companion_chair", "stool", "cabinet"],
    "medical_equipment": ["bedside_monitor", "oxygen_flowmeter", "gas_manifold",
                          "iv_pole", "suction_jar", "suction_knob", "scale",
                          "weight_scale", "stethoscope", "ear_thermometer"],
    "consumable":        ["alcohol_spray_bottle", "sanitizer", "gauze",
                          "medical_gloves", "medical_package", "syringe",
                          "paperbox", "tissue_dispenser"],
    "waste_container":   ["waste_bin", "medical_waste_container",
                          "soiled_linen_bin"],
    "bathroom_fixture":  ["toilet", "toilet_handle", "sink", "shower"],
    "structure_fixture": ["door", "door_handle", "window", "mirror",
                          "light_switch", "air_vent", "hook"],
    "textile":           ["curtain", "bed_curtain"],
    "electronics":       ["TV", "telephone", "remote_control"],
}
_CLASS_TO_SUPER = {c: sup for sup, cs in SUPERCATEGORY_TREE.items() for c in cs}


def supercategory_of(name):
    """COCO supercategory for a leaf class name (default 'ward_object')."""
    return _CLASS_TO_SUPER.get(name, "ward_object")
