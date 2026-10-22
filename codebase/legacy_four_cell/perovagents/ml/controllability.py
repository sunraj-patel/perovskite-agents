"""Formal feature ontology for the ML layer.

Every model feature is described along four axes:

    controllability
        experimentally_controllable   input a researcher sets directly
                                      (materials chosen, thickness, anneal)
        indirectly_controllable       only reachable through other inputs
                                      (derived / computed descriptors)
        non_controllable              measurement-derived, metadata,
                                      missingness flags — explanatory only

    type
        continuous | categorical | binary | derived

    role
        material | process | device | measurement | metadata

    valid_range  [lo, hi] declared safe intervention range for continuous
                 inputs (used to clamp what_if sweeps and optimize grids);
                 None when no declared bound exists.

`what_if` / `optimize` only accept features that are experimentally or
indirectly controllable AND continuous AND have no binary/structural
semantics; non-controllable features (Voc, Jsc, stability columns, metadata)
are rejected as preconditions, so a sweep can never operate on an output
variable or a fabricated measurement.

Classification is deterministic and name-driven (no LLM) so the ontology is
stable and auditable at run time.
"""
import re
from typing import Dict, List, Optional

CONTROLLABLE = {"material_composition", "material_property",
                "processing_parameter", "device_architecture"}
NON_CONTROLLABLE = {"measurement_derived", "metadata", "missingness_indicator"}

# feature-name → category rules (checked in order of precedence)
_RULES: List = [
    # missingness indicators first
    (re.compile(r"_na$"), "missingness_indicator"),
    # measurement- / output-derived columns (JV_, Stability_, EQE_, PL…)
    (re.compile(r"^(JV|Stability|EQE|Cell_area|Stabilised)", re.I),
     "measurement_derived"),
    (re.compile(r"(Voc|Jsc|FF|JV_default)$", re.I), "measurement_derived"),
    (re.compile(r"(_pl|PL_|photolumin|hysteresis|measured|fill_factor|open_circuit|"
                r"short_circuit)", re.I), "measurement_derived"),
    # metadata
    (re.compile(r"^(Ref_|data_|Split|seed|test|train|val_)", re.I), "metadata"),
    (re.compile(r"(doi|journal|ID|date|year)$", re.I), "metadata"),
    # device / architecture structure (controllable choice of stack)
    (re.compile(r"^((Cell|Substrate|Backcontact|Architecture|Stack)_|"
                r".*architecture)", re.I), "device_architecture"),
    (re.compile(r"^(HTL|ETL|Perovskite|Pvk|Active)_(HOMO|LUMO|band_gap|"
                r"electron_mobility|hole_mobility|dielectric|radius|"
                r"tolerance_factor|octahedral_factor|tau_descriptor)", re.I),
     "material_property"),
    # material / composition
    (re.compile(r"_(fraction|ratio|stoich|comp)$|^(a_|b_|halide_)", re.I),
     "material_composition"),
    (re.compile(r"(MA|FA|Cs|Rb|Pb|Sn|Bi|I)$", re.I), "material_composition"),
    # processing parameters
    (re.compile(r"(thickness|annear|anneal|quench|temperatur|temp_|time|"
                r"cooling|concentration|spin|deposit|solvent|coating|"
                r"nucleation|seed_ink|precursor)", re.I), "processing_parameter"),
    # default: processing-ish (conservative, treat unknown as controllable)
    ("", "processing_parameter"),
]

# continuous knobs that look numeric but should be binary/structural
_BINARY_STRUCTURAL = re.compile(
    r"^(has_|is_|used_|Add_|Encapsulation)|(dimension|3d_|2d_|capping|"
    r"_y$|_x$|flexible|encapsulated|%|doping$|passivation_layer)", re.I)

# names that denote *derived* descriptors computed by the feature pipeline
_DERIVED = re.compile(
    r"(gap_diff|barrier|_factor|tau_|ion_count|tolerance|descriptor|"
    r"_range$|log_abs|ratio$|signed$)", re.I)

# declared safe intervention ranges for well-known continuous knobs
# (nm, degC, min, ...). Only features with a declared (or later-trained) bound
# get a clamped sweep; everything else is unbounded-but-honest (None).
DECLARED_RANGES = {
    "Perovskite_thickness_nm": [50.0, 1200.0],
    "ETL_thickness_nm": [10.0, 200.0],
    "HTL_thickness_nm": [10.0, 400.0],
    "Perov_anneal_temp_C": [50.0, 220.0],
    "Perov_anneal_time_min": [5.0, 180.0],
}


def classify_feature(name: str) -> str:
    """First rule whose pattern matches the feature name."""
    for pattern, cat in _RULES:
        if isinstance(pattern, re.Pattern):
            if pattern.search(name):
                return cat
        else:
            return cat
    return "processing_parameter"


def is_controllable(name: str) -> bool:
    """True for features a researcher can choose or set directly."""
    return classify_feature(name) in CONTROLLABLE


def controllable_features(names: List[str]) -> List[str]:
    """Filter a list down to controllable features."""
    return [n for n in names if is_controllable(n)]


def classify_many(names: List[str]) -> Dict[str, str]:
    """Per-name category map (batch version of `classify_feature`)."""
    return {n: classify_feature(n) for n in names}


def is_binary_structural(name: str) -> bool:
    """Structural/binary flags (has_X, dimension, capping, ...) are not
    sweepable numeric knobs even when heuristic category says otherwise."""
    return bool(_BINARY_STRUCTURAL.search(name))


def feature_type(name: str) -> str:
    """Heuristic dtype: derived > binary > continuous > categorical."""
    n = name.lower()
    if _DERIVED.search(n) or name.endswith("_na"):
        return "derived"
    if _BINARY_STRUCTURAL.search(n):
        return "binary"
    if re.search(r"(thickness|nm|_c$|_min$|_temp|mobility|gap|homo|lumo|"
                 r"radius|fraction|barrier|time|watt|_mv|_v$|_n$|_k|_g$)", n):
        return "continuous"
    return "categorical"


def _role(name: str, category: str) -> str:
    if _DERIVED.search(name):
        return "material"
    if category == "measurement_derived":
        return "measurement"
    if category == "metadata":
        return "metadata"
    if category in ("material_composition", "material_property"):
        return "material"
    if category == "processing_parameter":
        return "process"
    if category == "device_architecture":
        return "device"
    return "process"


def valid_range(name: str) -> Optional[List[float]]:
    """Declared safe [lo, hi] intervention range, or None if unbounded."""
    return DECLARED_RANGES.get(name)


def describe_feature(name: str,
                     cat_features: Optional[set] = None) -> Dict:
    """Full ontology record for one feature name."""
    category = classify_feature(name)
    if category in NON_CONTROLLABLE:
        controllability = "non_controllable"
    elif _DERIVED.search(name) or category == "material_property":
        controllability = "indirectly_controllable"
    elif is_binary_structural(name):
        controllability = "non_controllable"
    else:
        controllability = "experimentally_controllable"
    ftype = "categorical" if (cat_features and name in cat_features) \
        else feature_type(name)
    if ftype == "binary":
        controllability = "non_controllable"
    return {
        "feature": name,
        "controllability": controllability,
        "type": ftype,
        "role": _role(name, category),
        "category": category,
        "valid_range": valid_range(name),
    }


def is_sweepable(name: str, cat_features: Optional[set] = None) -> bool:
    """True when a what_if/optimize intervention is scientifically meaningful
    for this feature: continuous, controllable, and not a structural flag."""
    rec = describe_feature(name, cat_features)
    if rec["type"] != "continuous":
        return False
    if rec["controllability"] not in ("experimentally_controllable",
                                      "indirectly_controllable"):
        return False
    if is_binary_structural(name):
        return False
    return True