"""SQL gate checks for cell (a): Query + Result Verification (S2).

Operates on (sql_text, result_rows, schema) triples. The schema is
{table: [columns]} as served by the SQL layer (S4). All checks are pure
and deterministic.

Simplifications (DECISIONS.md D8): single-statement SELECT/WITH only; the
identifier parser uses a keyword stoplist plus AS-alias tracking rather
than a full SQL grammar.
"""
import re
from typing import Dict, List, Optional

from ..cells.schemas import CheckResult, VerificationReport
from ..config import PHYSICS_BOUNDS
from .base import check, report

# SQL keywords / functions never treated as identifiers.
_KEYWORDS = {
    "select", "from", "where", "and", "or", "not", "in", "is", "null",
    "like", "between", "as", "on", "join", "inner", "left", "right",
    "outer", "full", "cross", "group", "by", "order", "having", "limit",
    "offset", "distinct", "all", "count", "avg", "mean", "min", "max",
    "sum", "std", "median", "asc", "desc", "with", "union", "case",
    "when", "then", "else", "end", "cast", "coalesce", "nullif",
    # type names (CAST targets), never identifiers
    "real", "integer", "int", "text", "float", "double", "numeric",
    "boolean", "bool", "blob",
}

_PCE_KEYS = ("pce", "JV_default_PCE", "predicted_pce")
_EG_KEYS = ("band_gap", "eg", "perovskite_band_gap", "Eg")
_VOC_KEYS = ("voc", "JV_default_Voc", "Voc")


def _strip_literals(sql: str) -> str:
    no_str = re.sub(r"'[^']*'", " ", sql)
    return re.sub(r'"[^"]*"', " ", no_str)


def _identifiers(sql: str) -> List[str]:
    """Word tokens minus keywords/numbers; AS-aliases tracked separately."""
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", _strip_literals(sql))
    return [t for t in toks
            if t.lower() not in _KEYWORDS and not t[0].isdigit()]


def _aliases(sql: str) -> List[str]:
    return re.findall(r"(?i)\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", sql)


def check_select_only(sql: str) -> CheckResult:
    """Reject non-SELECT statements and stacked statements."""
    s = sql.strip().rstrip(";").strip()
    if not re.match(r"(?i)^(select|with)\b", s):
        return check("select_only", False,
                     "only SELECT/WITH statements allowed", sql[:120])
    if ";" in s:
        return check("select_only", False,
                     "stacked statements not allowed", sql[:120])
    for kw in ("insert", "update", "delete", "drop", "alter", "create",
               "replace", "attach", "detach", "pragma"):
        if re.search(r"(?i)\b%s\b" % kw, s):
            return check("select_only", False,
                         f"forbidden keyword: {kw}", sql[:120])
    return check("select_only", True, "single SELECT statement")


def check_known_tables(sql: str, schema: Dict[str, List[str]]) -> CheckResult:
    """Every FROM/JOIN target must be a real table."""
    tables = re.findall(r"(?i)\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)", sql)
    known = {t.lower() for t in schema}
    unknown = [t for t in tables if t.lower() not in known]
    if unknown:
        return check("known_tables", False,
                     f"unknown tables: {unknown}", unknown)
    return check("known_tables", True, f"tables ok: {tables or ['(none)']}")


def check_known_columns(sql: str,
                        schema: Dict[str, List[str]]) -> CheckResult:
    """Every referenced identifier must resolve to a table, column or alias."""
    cols = {c.lower() for cs in schema.values() for c in cs}
    tabs = {t.lower() for t in schema}
    aliased = {a.lower() for a in _aliases(sql)}
    unknown = []
    for tok in _identifiers(sql):
        parts = tok.split(".")
        if len(parts) == 2 and parts[0].lower() in tabs:
            if parts[1].lower() not in cols:
                unknown.append(tok)
        elif tok.lower() not in cols | tabs | aliased:
            unknown.append(tok)
    if unknown:
        return check("known_columns", False,
                     f"unknown columns: {sorted(set(unknown))}",
                     sorted(set(unknown)))
    return check("known_columns", True, "columns resolve against schema")


def _result_numbers(result: Dict, keys) -> List[float]:
    vals = []
    rows = result.get("rows", [])
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k in keys:
            if k in r and isinstance(r[k], (int, float)):
                vals.append(float(r[k]))
    for k in keys:
        v = result.get(k)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def check_units_sane(result: Dict) -> CheckResult:
    """Units/value ranges sane; PCE in (0,1] is rejected as a likely
    fraction-instead-of-percent error (conservative: genuine sub-1% PCE
    values must arrive with an explicit unit note — DECISIONS.md D8)."""
    bounds = PHYSICS_BOUNDS.get("JV_default_PCE", {})
    pce_max = bounds.get("max", 30.0)
    for v in _result_numbers(result, _PCE_KEYS):
        if 0 < v <= 1.0:
            return check("units_sane", False,
                         f"PCE={v} looks like a fraction, expected percent",
                         {"pce": v})
        if not (0.0 <= v <= pce_max):
            return check("units_sane", False,
                         f"PCE={v} outside [0, {pce_max}]", {"pce": v})
    for v in _result_numbers(result, _EG_KEYS):
        if not (0.3 <= v <= 4.0):
            return check("units_sane", False,
                         f"band_gap={v} eV implausible", {"band_gap": v})
    for v in _result_numbers(result, _VOC_KEYS):
        if not (0.0 <= v <= 2.0):
            return check("units_sane", False,
                         f"Voc={v} V implausible", {"voc": v})
    return check("units_sane", True, "units and ranges sane")


def check_shape(result: Dict, expected_cols: Optional[List[str]] = None,
                min_rows: int = 1) -> CheckResult:
    """Non-empty result, expected columns present, no all-NULL columns."""
    rows = result.get("rows", [])
    if not isinstance(rows, list) or len(rows) < min_rows:
        return check("shape", False,
                     f"expected >= {min_rows} rows, got {len(rows) if isinstance(rows, list) else 0}",
                     {"n_rows": len(rows) if isinstance(rows, list) else 0})
    cols = list(rows[0].keys()) if isinstance(rows[0], dict) else []
    if expected_cols:
        missing = [c for c in expected_cols if c not in cols]
        if missing:
            return check("shape", False, f"missing columns: {missing}",
                         missing)
    all_null = [c for c in cols
                if all((r.get(c) is None) for r in rows if isinstance(r, dict))]
    if all_null:
        return check("shape", False, f"all-NULL columns: {all_null}",
                     all_null)
    return check("shape", True,
                 f"{len(rows)} rows x {len(cols)} cols, no all-NULL column")


def verify_sql(sql: str, result: Dict, schema: Dict[str, List[str]],
               expected_cols: Optional[List[str]] = None,
               min_rows: int = 1) -> VerificationReport:
    """Full gate for one SQL attempt: query checks then result checks."""
    return report([
        check_select_only(sql),
        check_known_tables(sql, schema),
        check_known_columns(sql, schema),
        check_units_sane(result),
        check_shape(result, expected_cols, min_rows),
    ])
