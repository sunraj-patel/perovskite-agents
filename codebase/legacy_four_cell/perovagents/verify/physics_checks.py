"""Physics gate checks for cell (d): Convergence + Physical Check (S2).

Operates on SCAPS-style result rows
{parameter, value, unit, Voc, Jsc, FF, PCE, converged?...} plus the run
provenance. Pure and deterministic.

Shockley-Queisser table: approximate AM1.5G detailed-balance PCE caps used
only as an upper-bound sanity flag; the exact tabulation should be replaced
by a citable source (DECISIONS.md D8). Values near the boundary are not
test-relevant; tests use PCE=45% (must fail) and PCE=20% (must pass).
"""
from typing import Dict, List, Optional

from ..cells.schemas import CheckResult, VerificationReport
from .base import check, report

# Approx. S-Q PCE caps (%) vs absorber band gap (eV). Coarse, upper-bound
# use only; linear interpolation between anchors (see module docstring).
_SQ_TABLE = [
    (1.00, 30.1), (1.10, 32.2), (1.20, 33.3), (1.34, 33.7),
    (1.42, 33.5), (1.50, 32.7), (1.55, 32.2), (1.60, 31.6),
    (1.70, 30.1), (1.80, 28.4),
]
_SQ_GLOBAL_MAX = 33.7  # the detailed-balance maximum, at ~1.34 eV


def sq_limit_pce(band_gap_ev: Optional[float]) -> float:
    """Upper-bound PCE (%) for a band gap; global max when unknown."""
    if band_gap_ev is None:
        return _SQ_GLOBAL_MAX
    eg = float(band_gap_ev)
    if eg <= _SQ_TABLE[0][0]:
        return _SQ_TABLE[0][1]
    for (x0, y0), (x1, y1) in zip(_SQ_TABLE, _SQ_TABLE[1:]):
        if x0 <= eg <= x1:
            f = (eg - x0) / (x1 - x0)
            return y0 + f * (y1 - y0)
    return _SQ_TABLE[-1][1]


def check_bounds(row: Dict,
                 band_gap_ev: Optional[float] = None) -> CheckResult:
    """Parameters positive/plausible; PCE within the S-Q cap for the gap."""
    for key in ("value",):
        v = row.get(key)
        if v is not None and not (isinstance(v, (int, float)) and v > 0):
            return check("bounds", False,
                         f"parameter value not positive: {v!r}", row)
    eg = band_gap_ev
    if eg is None and "band_gap" in row:
        try:
            eg = float(row["band_gap"])
        except (TypeError, ValueError):
            eg = None
    if eg is not None and not (0.3 <= float(eg) <= 4.0):
        return check("bounds", False, f"band gap {eg} eV implausible",
                     {"band_gap": eg})
    pce = row.get("PCE")
    if pce is not None:
        cap = sq_limit_pce(eg)
        if float(pce) > cap:
            return check("bounds", False,
                         f"PCE={pce}% exceeds S-Q cap {cap:.1f}% "
                         f"(Eg={eg} eV)", {"PCE": pce, "cap": cap})
        if float(pce) < 0:
            return check("bounds", False, f"PCE={pce}% negative",
                         {"PCE": pce})
    return check("bounds", True, "parameters and PCE within physical limits")


def check_consistency(row: Dict, pin: float = 100.0,
                      tol_pp: float = 0.5) -> CheckResult:
    """PCE agrees with Voc*Jsc*FF at 1-sun Pin (default 100 mW/cm2)
    within tolerance (percentage points)."""
    try:
        voc, jsc, ff, pce = (float(row[k]) for k in ("Voc", "Jsc", "FF", "PCE"))
    except (KeyError, TypeError, ValueError):
        return check("consistency", False,
                     "Voc/Jsc/FF/PCE must all be numeric",
                     {k: row.get(k) for k in ("Voc", "Jsc", "FF", "PCE")})
    expect = voc * jsc * ff / pin * 100.0
    if abs(expect - pce) > tol_pp:
        return check("consistency", False,
                     f"PCE={pce}% vs Voc*Jsc*FF={expect:.2f}% "
                     f"(>{tol_pp}pp apart)",
                     {"PCE": pce, "implied": round(expect, 2)})
    return check("consistency", True,
                 f"PCE consistent with JV ({expect:.2f}%)")


def check_iv_monotone(iv: Optional[List[Dict]] = None) -> CheckResult:
    """J-V monotonicity where an IV series is present; explicitly skipped
    (not silently passed) when no IV data exists."""
    if not iv:
        return check("iv_monotone", True, "skipped: no IV series provided")
    vs = [float(p["v"]) for p in iv]
    js = [float(p["j"]) for p in iv]
    if any(b < a for a, b in zip(vs, vs[1:])):
        return check("iv_monotone", False, "voltage not ascending",
                     {"v": vs})
    if any(b > a for a, b in zip(js, js[1:])):
        return check("iv_monotone", False, "current not non-increasing",
                     {"j": js})
    return check("iv_monotone", True, "IV monotone as expected")


def check_complete(result: Dict,
                   requested: Optional[List[str]] = None) -> CheckResult:
    """The run finished with all requested outputs present (non-None)."""
    want = requested or ["Voc", "Jsc", "FF", "PCE"]
    missing = [k for k in want if result.get(k) is None]
    if missing:
        return check("complete", False, f"missing outputs: {missing}",
                     missing)
    return check("complete", True, f"outputs present: {want}")


def check_converged(result: Dict) -> CheckResult:
    """Solver convergence explicitly confirmed. Assumed/precomputed
    convergence (no solver evidence) fails — see AUDIT C3."""
    if result.get("convergence_failure"):
        return check("converged", False, "solver reported convergence failure",
                     result.get("convergence_failure"))
    if result.get("converged") is True:
        return check("converged", True, "solver convergence confirmed")
    return check("converged", False,
                 "convergence not explicitly confirmed",
                 {"converged": result.get("converged")})


def verify_physics(result: Dict,
                   requested: Optional[List[str]] = None,
                   band_gap_ev: Optional[float] = None,
                   iv: Optional[List[Dict]] = None) -> VerificationReport:
    """Full gate for one SCAPS attempt."""
    return report([
        check_bounds(result, band_gap_ev),
        check_consistency(result),
        check_iv_monotone(iv),
        check_complete(result, requested),
        check_converged(result),
    ])
