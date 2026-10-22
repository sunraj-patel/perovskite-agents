"""Claim typing.

Every actionable result a tool produces is tagged with a `claim_type` so the
answer generator is forced to use the *right epistemic register* for it:

    model_prediction      book PCE from a fitted ML model (JV_* are never
                          candidates; interval + model name must be reported)
    model_association     SHAP feature importance -> the *model* identifies /
                          associates a feature with higher PCE; NOT causal
    model_sensitivity     what_if / optimize sweeps -> under the model,
                          changing X moves predicted PCE A% -> B%
    database_evidence     aggregate over real device records from the
                          Perovskite Database (n_records + PDB provenance)
    material_property_evidence  computed property from the Materials Project
    literature_evidence   claims grounded in a retrieved bibliographic source
                          (DOI must be cited)
    verification_evidence a passed/failed consistency check against a cite
    computation           arithmetic / unit conversions (calculator, python)
    empirical_fit         symbolic shift/scale fit (never asserted beyond its
                          fitted range)

The mapping is deterministic by (tool, action); a single result dict can carry
one primary claim_type. `CLAIM_LANGUAGE` is injected into the synthesis prompt
so the LLM narrative cannot mix these registers.
"""
from typing import List, Optional

CLAIM_TYPES = (
    "model_prediction", "model_association", "model_sensitivity",
    "database_evidence", "material_property_evidence", "literature_evidence",
    "verification_evidence", "computation", "empirical_fit",
)

# (tool, action) -> claim_type. Missing pairs default to "computation".
TOOL_ACTION_CLAIM = {
    ("calculator", "eval"): "computation",
    ("calculator", "pce"): "computation",
    ("calculator", "convert"): "computation",
    ("python_exec", "exec"): "computation",
    ("literature", "search"): "literature_evidence",
    ("literature", "fetch"): "literature_evidence",
    ("extraction", "extract"): "literature_evidence",
    ("doi", "resolve"): "literature_evidence",
    ("citations", "verify"): "verification_evidence",
    ("pdb", "query"): "database_evidence",
    ("material", "lookup"): "material_property_evidence",
    ("materials_project", "lookup"): "material_property_evidence",
    ("ml", "predict"): "model_prediction",
    ("ml", "explain_global"): "model_association",
    ("ml", "explain_local"): "model_association",
    ("ml", "what_if"): "model_sensitivity",
    ("ml", "optimize"): "model_sensitivity",
    ("scaps", "get_sweep"): "model_sensitivity",
    ("symbolic", "fit"): "empirical_fit",
}

# claim_type -> required narrative register (drives the synthesis prompt).
CLAIM_LANGUAGE = {
    "model_prediction": (
        'Prediction claims: "the model predicts PCE ~ {value}% (<model>, '
        '95% interval [lo, hi])". Never call a prediction a measurement.'),
    "model_association": (
        'Importance claims: SHAP importance is NOT causal. Say "the model '
        'identifies / associates {feature} with higher predicted PCE", never '
        '"increasing {feature} will improve the device".'),
    "model_sensitivity": (
        'Sensitivity/intervention claims: "under the model, changing '
        '{parameter} moves predicted PCE from A% to B% (95% interval)" . '
        'Do not claim an experimental outcome.'),
    "database_evidence": (
        'Database claims: "across {n} records in the Perovskite Database '
        '(DB1.0), {stat}". Cite DOI sources when present.'),
    "material_property_evidence": (
        'Property claims: "computed {property} for {material} from the '
        'Materials Project = {value}" (source always named).'),
    "literature_evidence": (
        'Literature claims: "per {doi} / {source}, ...". Only claim what the '
        'retrieved passage actually states.'),
    "verification_evidence": (
        'Verification claims: report PASS/WARN/FAIL of a stated claim vs a '
        'cited source, never silently adopt the source as ground truth.'),
    "computation": (
        'Computation claims: report the arithmetic result exactly, with '
        'units. No physics interpretation beyond what the inputs allow.'),
    "empirical_fit": (
        'Fit claims: symbolic/empirical relation valid only over the fitted '
        'feature range; never extrapolate beyond it.'),
}

ACTION_CLAIM = {f"{t}.{a}": c for (t, a), c in TOOL_ACTION_CLAIM.items()}


def claim_type_for(tool: str, action: str) -> str:
    """Look up the claim_type for a (tool, action) pair; unknown pairs
    default to "computation"."""
    return TOOL_ACTION_CLAIM.get((tool, action or ""),
                                 TOOL_ACTION_CLAIM.get((tool, ""), "computation"))


def allowed_types() -> List[str]:
    """All claim_types the epistemic gate understands."""
    return list(CLAIM_TYPES)


def primary_claim_type(result: dict, fallback: Optional[str] = None) -> str:
    """Pick the most informative claim_type already present on a result dict
    (used for raw/legacy tool calls that bypassed typed actions)."""
    seen = result.get("claim_type")
    if seen and seen in CLAIM_TYPES:
        return seen
    rt = result.get("result_type")
    rt_map = {
        "prediction": "model_prediction",
        "explanation": "model_association",
        "what_if": "model_sensitivity",
        "optimization": "model_sensitivity",
        "count": "database_evidence",
        "scalar": "database_evidence",
        "table": "database_evidence",
        "distribution": "database_evidence",
        "correlation": "database_evidence",
        "literature_passages": "literature_evidence",
        "material_property": "material_property_evidence",
        "symbolic_fit": "empirical_fit",
        "verification": "verification_evidence",
    }
    if rt in rt_map:
        return rt_map[rt]
    if fallback and fallback in CLAIM_TYPES:
        return fallback
    return "computation" if rt != "what_if" else "model_sensitivity"