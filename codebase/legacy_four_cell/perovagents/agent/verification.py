"""Task-sufficiency verifier (validation layer 4).

After a plan runs, the verifier asks the hardest question: *did the executed
steps actually return what the question requires?* It matches the task
specification's requested output tokens against the executed step results and
issues a verdict:

    SUFFICIENT   every required output token is present in some step result
    RECOVERABLE  some required outputs are missing but a re-plan plausibly
                 covers them (missing requirement is used as planner feedback)
    UNRESOLVED   nothing useful came back (all steps failed / no evidence)

This is the piece the old `synthesize → give up` design was missing: the agent
now *knows* when an incorrectly-chosen tool produced the wrong evidence class
(e.g. a prediction without feature importance) and can re-plan instead of
silently answering off-target.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.contracts import match_outputs
from .models import Plan, PlanStep
from .task import TaskSpec


@dataclass
class Verdict:
    status: str = "UNRESOLVED"          # SUFFICIENT | RECOVERABLE | UNRESOLVED
    missing_requirements: List[str] = field(default_factory=list)
    checked: Dict[str, bool] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self):
        return {"status": self.status,
                "missing_requirements": list(self.missing_requirements),
                "checked": dict(self.checked),
                "diagnostics": list(self.diagnostics)}


_EXPLAIN_TOKENS = {"feature_name", "importance", "direction"}
_PREDICT_TOKENS = {"prediction", "interval"}
_INTERVENE_TOKENS = {"parameter_value", "predicted_pce"}

# Epistemic compatibility: which claim_types may carry the FINAL answer for a
# given task. A result that looks numeric (e.g. a PCE mean) but has the wrong
# claim class must never pass — even if it "has a number".
TASK_ALLOWED_CLAIMS = {
    "prediction": {"model_prediction"},
    "feature_importance": {"model_association"},
    "interpretation": {"model_association", "model_prediction"},
    "sensitivity_analysis": {"model_sensitivity"},
    "optimization": {"model_prediction", "model_sensitivity"},
    "database_query": {"database_evidence"},
    "comparison": {"database_evidence"},
    "factual_lookup": {"material_property_evidence"},
    "literature_retrieval": {"literature_evidence"},
    "literature_extraction": {"literature_evidence"},
    "verification": {"verification_evidence"},
    "calculation": {"computation"},
    "simulation": {"computation", "model_sensitivity"},
    "multi_tool_investigation": set(),    # open by construction
}


def _step_dicts(plan: Plan, step_results) -> List[Dict]:
    """Merge plan intent with executed outcomes into dicts the matcher wants."""
    out = []
    for idx in range(len(step_results)):
        sr = step_results[idx] if idx < len(step_results) else {}
        out.append({"status": bool(sr.get("success", True)) or None,
                    "result": sr.get("result") or {},
                    "expected_outputs": sr.get("expected_outputs") or []})
    return out


def _none_found_intervention(step_results) -> bool:
    """True when an ml.optimize step honestly reported no improvement."""
    return any((s.get("result") or {}).get("optimization_status") == "none_found"
               for s in step_results)


def verify_task(task_spec: TaskSpec, plan: Plan,
                step_results: List[Dict]) -> Verdict:
    """Grade the executed steps against the task specification.

    `step_results` are the executed StepResult dicts (with embedded `result`).
    """
    ver = Verdict()

    # required tokens: those the analyzer demanded + those the *required*
    # steps themselves promised to produce
    required: List[str] = []
    seen = set()
    for t in (task_spec.requested_output or []):
        if t not in seen:
            seen.add(t)
            required.append(t)
    for s in step_results:
        if not s.get("required_for_final_answer", True):
            continue
        for t in s.get("expected_outputs") or []:
            if t not in seen:
                seen.add(t)
                required.append(t)

    # task-flag requirements (semantic layer). Explanation tokens apply to
    # pure explain tasks; intervention tasks (what_if/optimize) ask for
    # parameter_value/predicted_pce and do not force feature-level output.
    if task_spec.requires_prediction and not task_spec.requires_intervention:
        for t in _PREDICT_TOKENS:
            if t not in seen:
                required.append(t)
    if task_spec.requires_explanation and not task_spec.requires_intervention:
        for t in _EXPLAIN_TOKENS:
            if t not in seen:
                required.append(t)
    if task_spec.requires_intervention:
        for t in _INTERVENE_TOKENS | _PREDICT_TOKENS:
            if t not in seen:
                required.append(t)

    if _none_found_intervention(step_results):
        # an honest optimize that found no improvement delivers predicted
        # PCE + interval but by construction no sweep parameter_value (this
        # applies whether the token came from requested_output or plan-side
        # expected_outputs); demanding one would force a fabricated
        # recommendation.
        required = [t for t in required if t != "parameter_value"]

    matched = match_outputs(required,
                            [{"status": "success" if s.get("success", False) else "error",
                              "result": s.get("result") or {}} for s in step_results])
    ver.checked = {t: matched.get(t, False) for t in required}

    missing = [t for t in required if not matched.get(t, False)]
    n_ok = sum(1 for s in step_results if s.get("success", False))
    n_fail = len(step_results) - n_ok

    # claim-type gate: the final-answer carrier must be epistemically
    # compatible with the task. A plausible-looking number from the wrong
    # evidence class is a FAIL, never a pass.
    allowed = TASK_ALLOWED_CLAIMS.get(task_spec.task_type)
    if allowed:
        finals = [s for s in step_results
                  if s.get("required_for_final_answer", True)]
        compatible = [s for s in finals if s.get("success")
                      and s.get("claim_type") in allowed]
        if not compatible and any(s.get("success") for s in finals):
            got = sorted({s.get("claim_type", "?") for s in finals
                          if s.get("success")})
            ver.diagnostics.append(
                "wrong evidence class: task '%s' requires claim_type in %s "
                "but the executed steps produced [%s]; re-plan with an action "
                "that emits %s (e.g. ml.explain_global for "
                "feature_name/importance/direction, ml.what_if for "
                "parameter_value/predicted_pce)."
                % (task_spec.task_type, sorted(allowed), ", ".join(got),
                   sorted(allowed)))
            if "compatible_claim_level" not in missing:
                missing.append("compatible_claim_level")
            ver.checked["compatible_claim_level"] = False

    ver.missing_requirements = missing
    if n_ok and not missing:
        ver.status = "SUFFICIENT"
    elif n_fail == len(step_results) and len(step_results):
        ver.status = "UNRESOLVED"
        ver.diagnostics.append("every planned step failed; no usable evidence")
    elif missing:
        ver.status = "RECOVERABLE"
        if not any("wrong evidence class" in d for d in ver.diagnostics):
            ver.diagnostics.append(
                "required outputs missing: " + ", ".join(missing[:6])
                + "; re-plan with actions that produce these tokens")
    else:
        ver.status = "UNRESOLVED"
    return ver