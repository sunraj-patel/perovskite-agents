"""Four-layer scientific validation pipeline.

The agent validates at every stage — not just after execution:

    Layer 1  Schema validation      plan-time: action exists, params typed/enum
    Layer 2  Execution validation   did the tool actually run successfully?
    Layer 3  Scientific validity    physics bounds, structural shape, units
    Layer 4  Task sufficiency       did we return what the question required?

Layer 1 lives in `ToolRegistry.validate_action/validate_params`
(core/registry.py); layer 2 is the `status` field of `ExecutionResult`;
layer 3 is `validate_scientific` here; layer 4 is `verify_task`
(agent/verification.py). This module only implements layer 3.
"""
from typing import Any, Dict, List, Tuple

from ..config import PHYSICS_BOUNDS

VALID_ARCHITECTURES = {"p-i-n", "pin", "n-i-p", "nip", "planar", "meso"}


def _num(v) -> float:
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def validate_scientific(result: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Layer-3 checks. Returns (ok, errors, warnings).

    Checks:
      * physics bounds from config (PCE, Voc, Jsc, FF)
      * missing numbers where a result_type demands one (structural)
      * negative counts, empty tables
      * a few absolute domain constraints (band gap > 0, mobility > 0,
        FF within [0,1], valid architecture string)
    Non-blocking: a violating value is flagged, not deleted.
    """
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(result, dict):
        return False, ["result is not a dict"], warnings

    # --- physics bounds ----------------------------------------------------
    for col, b in PHYSICS_BOUNDS.items():
        for key in (col, col.split("_")[-1]):
            v = _num(result.get(key))
            if v is None:
                continue
            if v < b["min"] or v > b["max"]:
                warnings.append(f"{col}={v} outside physics bounds "
                                f"[{b['min']}, {b['max']}]: {b['warn']}")

    # predictions / aggregate values
    for key in ("predicted_pce", "value"):
        v = _num(result.get(key))
        if v is None:
            continue
        if key == "predicted_pce" and (v < 0 or v > PHYSICS_BOUNDS["JV_default_PCE"]["max"]):
            warnings.append(f"predicted_pce={v} outside plausible range")
        elif key == "value" and v < 0:
            warnings.append(f"value={v} is negative")

    # --- structural ----------------------------------------------------------
    for k in ("n", "rows_after", "n_records", "n_hits", "count"):
        v = _num(result.get(k))
        if v is not None and v < 0:
            errors.append(f"{k}={v} is negative")

    rt = result.get("result_type")
    if rt == "table":
        rows = result.get("data") or result.get("rows") or []
        if not rows:
            warnings.append("query returned an empty table")
    if rt == "scalar" and "value" not in result:
        errors.append("scalar result is missing its value")
    if rt == "prediction" and "predicted_pce" not in result:
        errors.append("prediction result is missing predicted_pce")

    # --- domain constraints ---------------------------------------------------
    gap = result.get("band_gap") or result.get("gap")
    if isinstance(gap, (int, float)) and gap <= 0:
        errors.append(f"band gap {gap} <= 0 is unphysical")

    for mob in ("electron_mobility", "hole_mobility", "mobility"):
        v = _num(result.get(mob))
        if v is not None and v < 0:
            errors.append(f"{mob}={v} is negative")

    ff = result.get("FF") or result.get("ff")
    if isinstance(ff, (int, float)) and not (0.0 <= ff <= 1.0):
        warnings.append(f"FF={ff} outside [0,1] (is it a fraction?)")

    arch = result.get("architecture")
    if isinstance(arch, str) and arch.lower() not in VALID_ARCHITECTURES \
            and arch not in ("PIN", "NIP", "Overall"):
        warnings.append(f"architecture '{arch}' is not a recognized value")

    return not errors, errors, warnings