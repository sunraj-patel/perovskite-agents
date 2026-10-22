"""Output contracts: canonical output tokens and how to recognise them in a
tool result.

Every action in the registry declares the *output tokens* it can produce
(``outputs``). The task analyzer declares the tokens a question *requires*.
The verifier matches the two by testing each required token against the actual
result dict. A token match is deterministic — no LLM involved.

Canonical tokens (used across schemas and the verifier):

    prediction | interval | uncertainty | ood_status | architecture
    feature_name | importance | direction
    parameter_value | predicted_pce
    count | aggregate | table | distribution | columns
    records | papers
    passages | material_property | properties | doi_metadata | citation_support
    sweep | equation | calculation | exec_output
"""
from typing import Any, Dict, List


def _feature_records(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find the list of per-feature records inside a result, if any."""
    for key in ("shap_top_features", "features", "recommendations"):
        v = result.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    data = result.get("data")
    if isinstance(data, list):
        rows = [x for x in data if isinstance(x, dict)]
        if rows and any(("name" in r or "feature" in r or "parameter" in r)
                        for r in rows):
            return rows
    return []


def _has_numeric(result: Dict[str, Any], *keys) -> bool:
    for k in keys:
        v = result.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return True
    return False


def _rows_numeric(result: Dict[str, Any], *keys) -> bool:
    """True when every row in a data/-style list carries a numeric key."""
    for key in ("data", "values", "series"):
        v = result.get(key)
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            if all(_has_numeric(x, *keys) for x in v):
                return True
    return False


def _rows_interval(result: Dict[str, Any]) -> bool:
    return all(_has_numeric(r, "interval_lo", "interval_hi")
               for r in _feature_records(result)) or _rows_numeric(
                   result, "interval_lo", "interval_hi")


def _feature_with_value(result: Dict[str, Any], key: str) -> bool:
    return any(_has_numeric(rec, key, "shap", "value") for rec in _feature_records(result))


def _feature_with_direction(result: Dict[str, Any]) -> bool:
    for rec in _feature_records(result):
        if "direction" in rec:
            return True
        for k in ("shap", "importance", "value"):
            v = rec.get(k)
            if isinstance(v, (int, float)) and v != 0:
                return True
    return False


OUTPUT_MATCHERS = {
    # ---- prediction / explanation contracts ------------------------------
    "prediction": lambda r: _has_numeric(r, "predicted_pce")
                            or _rows_numeric(r, "predicted_pce"),
    "interval": lambda r: "pce_95_interval" in r
                          or ("interval_low" in r and "interval_high" in r)
                          or _rows_interval(r),
    "uncertainty": lambda r: "interval_spread_percent" in r or "interval" in r
                              or _rows_interval(r),
    "ood_status": lambda r: "hood" in r,
    "architecture": lambda r: "architecture" in r,
    "feature_name": lambda r: bool(_feature_records(r)),
    "importance": lambda r: _feature_with_value(r, "importance"),
    "direction": lambda r: _feature_with_direction(r),
    "parameter_value": lambda r: bool(_feature_records(r))
                                or "parameter_value" in r or "values" in r
                                or _rows_numeric(r, "parameter_value"),
    "predicted_pce": lambda r: _has_numeric(r, "predicted_pce")
                               or _rows_numeric(r, "predicted_pce"),
    # ---- database query contracts ---------------------------------------
    "count": lambda r: r.get("result_type") == "count",
    "aggregate": lambda r: r.get("result_type") == "scalar"
                           and _has_numeric(r, "value"),
    "table": lambda r: r.get("result_type") == "table",
    "distribution": lambda r: r.get("result_type") == "distribution",
    "columns": lambda r: r.get("result_type") == "columns",
    "records": lambda r: "n_records" in r or "rows_after" in r,
    "papers": lambda r: isinstance(r.get("n_papers"), int) and r["n_papers"] > 0,
    # ---- literature / materials contracts --------------------------------
    "passages": lambda r: bool(r.get("hits")),
    "material_property": lambda r: ("property" in r and ("value" in r or "values" in r))
                                   or bool(r.get("properties")),
    "properties": lambda r: bool(r.get("properties")),
    "doi_metadata": lambda r: ("doi" in r or "title" in r),
    "citation_support": lambda r: ("supported" in r or "verdict" in r
                                   or "verifications" in r),
    # ---- physics / numeric contracts -------------------------------------
    "sweep": lambda r: r.get("result_type") == "what_if"
                       or "sweep_name" in r or "series" in r,
    "equation": lambda r: r.get("result_type") in ("fit", "symbolic_fit")
                          or "equation" in r,
    "calculation": lambda r: ("result" in r) or _has_numeric(r, "result"),
    "exec_output": lambda r: ("output" in r or "stdout" in r),
}


def match_output(token: str, result: Dict[str, Any]) -> bool:
    """True when `result` demonstrably carries `token`."""
    if not isinstance(result, dict):
        return False
    f = OUTPUT_MATCHERS.get(token)
    if f is None:
        # unknown token: treat as met iff the raw dict carries that key
        return token in result
    try:
        return bool(f(result))
    except Exception:
        return False


def match_outputs(tokens, steps: List[Dict[str, Any]]) -> Dict[str, bool]:
    """Match a list of tokens against executed step results.

    A token is satisfied if *any* step's result carries it. Returns
    {token: satisfied} preserving token order of first occurrence.
    """
    satisfied: Dict[str, bool] = {}
    for t in tokens or []:
        if t in satisfied:
            continue
        satisfied[t] = any(match_output(t, s.get("result") or {})
                           for s in steps if s.get("status") in (None, "success"))
    return satisfied