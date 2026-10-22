"""ML gate checks for cell (c): Model + OOD + Controllability Check (S2).

Operates on the result dicts `MLPredictorTool` emits. Pure and
deterministic. Note `check_model_attached` FAILS on current predictor
output (model filename only, no validation metrics) — that is intentional:
it drives S5 to attach the recorded group-aware validation performance
required by spec section 4c (DECISIONS.md D8).
"""
from typing import Dict, List, Optional

from ..cells.schemas import CheckResult, VerificationReport
from ..ml.controllability import is_sweepable
from .base import check, report


def check_model_attached(result: Dict) -> CheckResult:
    """The model artifact is identified AND its recorded validation
    performance (metric, score, split) is attached. Honest group-aware
    scores only — never leaked or random-split (spec 4c)."""
    prov = result.get("provenance", {}) if isinstance(result, dict) else {}
    model = prov.get("model", "")
    perf = prov.get("validation")
    if not model:
        return check("model_attached", False, "no model id in provenance",
                     prov)
    if not isinstance(perf, dict) or not perf.get("metric") \
            or perf.get("score") is None or not perf.get("split"):
        return check("model_attached", False,
                     f"model {model!r} has no recorded validation performance",
                     {"model": model})
    if str(perf.get("split", "")).lower() in ("random", "iid", "shuffle"):
        return check("model_attached", False,
                     f"split {perf['split']!r} is not group-aware", perf)
    return check("model_attached", True,
                 f"{model} + {perf['metric']}={perf['score']} ({perf['split']})")


def check_in_domain(result: Dict) -> CheckResult:
    """The query input lies inside the training domain. An OOD flag fails
    the gate (spec: flag or reject — rejection chosen, DECISIONS.md D8)."""
    hood = result.get("hood", {}) if isinstance(result, dict) else {}
    if not hood.get("applied", False):
        return check("in_domain", True, "OOD check not applied")
    if hood.get("ood"):
        return check("in_domain", False,
                     "input outside training domain",
                     {"fraction_out": hood.get("fraction_out"),
                      "n_out": hood.get("n_out_of_envelope")})
    return check("in_domain", True, "input inside training domain")


def check_controllable(parameter: Optional[str]) -> CheckResult:
    """What-if / Optimize varies only experimentally controllable,
    continuous inputs — never derived, measured or structural quantities."""
    if not parameter:
        return check("controllable", True, "no intervention parameter")
    if is_sweepable(parameter):
        return check("controllable", True,
                     f"{parameter} is sweepable")
    return check("controllable", False,
                 f"{parameter} is not an experimentally controllable input",
                 {"parameter": parameter})


def verify_ml(result: Dict,
              parameter: Optional[str] = None) -> VerificationReport:
    """Full gate for one ML attempt."""
    return report([
        check_model_attached(result),
        check_in_domain(result),
        check_controllable(parameter),
    ])
