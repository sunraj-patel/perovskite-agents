"""Benchmark measurement runner.

For every question the runner drives one full agent run and measures, at each
layer of the stack:

    classification      task-level: does the analyzer's task_type match the
                        expected class for the question?
    selection           action-level: does the executed final plan contain a
                        ground-truth action sequence (in order) among its
                        required-for-final-answer steps?
    parameter_validity  are all step params valid against the action schemas?
    plan_validity       did the plan pass schema validation at the gate?
    execution_success   did every executed step succeed?
    contract_satisfied  did the output verifier reach SUFFICIENT at some
                        iteration?
    replan_resolved     if the first iteration was RECOVERABLE, did a later
                        iteration reach SUFFICIENT?
    recovery_rate       RECOVERABLE/REJECTED failures later resolved /
                        total RECOVERABLE+REJECTED failures encountered

Results are aggregated per category and overall, saved as JSON + CSV under
results/benchmarks/.
"""
import json
import os
from typing import Callable, Dict, List, Optional

from ..agent.orchestrator import Agent

# category -> accepted analyzer task_type(s)
EXPECTED_TASK = {
    "prediction": {"prediction"},
    "global_interpretation": {"feature_importance", "interpretation"},
    "local_interpretation": {"feature_importance", "interpretation"},
    "sensitivity_analysis": {"sensitivity_analysis", "simulation"},
    "optimization": {"optimization"},
    "comparison": {"comparison", "database_query"},
    "literature_retrieval": {"literature_retrieval", "literature_extraction"},
    "database_query": {"database_query"},
    "verification": {"verification"},
    "multi_tool_investigation": {"multi_tool_investigation"},
    "ambiguous": None,          # any class acceptable; label recorded only
    "disambiguation": None,     # decided per-variant from first GT action
}

ACTION_DEFAULT_TASK = {
    "ml.predict": "prediction",
    "ml.explain_global": "feature_importance",
    "ml.explain_local": "feature_importance",
    "ml.what_if": "sensitivity_analysis",
    "ml.optimize": "optimization",
    "scaps.get_sweep": "sensitivity_analysis",
    "pdb.query": "database_query",
    "literature.search": "literature_retrieval",
    "literature.fetch": "literature_retrieval",
    "extraction.extract": "literature_extraction",
    "citations.verify": "verification",
    "material.lookup": "factual_lookup",
    "materials_project.lookup": "factual_lookup",
    "calculator.eval": "calculation",
    "calculator.pce": "calculation",
    "calculator.convert": "calculation",
    "symbolic.fit": "calculation",
}


def expected_task_for(q: Dict) -> Optional[str]:
    """The analyzer task_type(s) accepted for a question's category."""
    cat = q["category"]
    if cat in EXPECTED_TASK and EXPECTED_TASK[cat] is not None:
        return EXPECTED_TASK[cat]
    if cat == "disambiguation":
        first = q["gt"][0][0]
        return {ACTION_DEFAULT_TASK.get(first)} if first in ACTION_DEFAULT_TASK else None
    return None


def sequence_in(gt_seq: List[str], actions: List[str]) -> bool:
    """True when all gt entries appear in `actions` in the same relative order."""
    it = iter(actions)
    return all(any(a == want for a in it) for want in gt_seq)


def plan_actions(answer: Dict, max_iter: int = 99) -> List[str]:
    """Executed `tool.action` strings from an answer dict."""
    steps = answer.get("execution", {}).get("steps", [])
    return [f"{s.get('tool')}.{s.get('action')}" for s in steps
            if s.get("step_index", 0) < 99]


def measured_question(q: Dict, answer: Dict) -> Dict:
    """One row of per-question metrics measured from the agent answer."""
    expected = expected_task_for(q)
    task_type = (answer.get("task_spec") or {}).get("task_type", "")
    actions = plan_actions(answer)
    required = [f"{s.get('tool')}.{s.get('action')}"
                for s in answer.get("execution", {}).get("steps", [])]
    gt_hit = any(sequence_in(seq, required) for seq in q["gt"])
    exec_steps = answer.get("execution", {}).get("steps", [])
    exec_ok = all(s.get("success") for s in exec_steps)
    iterations = answer.get("plan_iterations", [])
    final_verdict = (answer.get("verdict") or {}).get("status", "")
    contract_sat = any(v.get("verdict", {}).get("status") == "SUFFICIENT"
                       for v in iterations)
    # recovery: first iteration RECOVERABLE/UNRESOLVED + later SUFFICIENT
    had_failure = bool(iterations) and \
        iterations[0].get("verdict", {}).get("status") in ("RECOVERABLE", "UNRESOLVED")
    recovered = had_failure and contract_sat
    risk_type = "database_query" if not expected else ", ".join(expected)
    return {
        "qid": q["qid"], "category": q["category"],
        "question": q["question"][:180],
        "expected_task": sorted(expected) if expected else None,
        "task_type": task_type,
        "classification_ok": bool(expected and task_type in expected),
        "gt_sequences": q["gt"], "gt_hit": gt_hit,
        "plan_actions": required,
        "param_valid": _params_valid(answer),
        "plan_valid": _plan_schema_valid(answer),
        "execution_ok": exec_ok,
        "n_ok": sum(1 for s in exec_steps if s.get("success")),
        "n_steps": len(exec_steps),
        "iterations": len(iterations),
        "final_verdict": final_verdict,
        "contract_satisfied": contract_sat,
        "replanner": iterations[0].get("planner", "") if iterations else "",
        "recovery_failure_seen": had_failure,
        "recovery_resolved": recovered,
        "risk": risk_type,
        "model_association_asserted": final_verdict == "SUFFICIENT",
    }


def _plan_schema_valid(answer: Dict) -> bool:
    """Every executed step must be a declared action with schema-valid params
    (this is what the schema gate enforces; raw-tool-calling plans fail it)."""
    from ..core.registry import default_registry
    reg = default_registry()
    steps = answer.get("execution", {}).get("steps", [])
    for s in steps:
        if not reg.validate_action(s.get("tool"), s.get("action")):
            return False
        params = {k: v for k, v in (s.get("params") or {}).items()
                  if k != "action"}   # drop executor's action-mirror key
        ok, perrs, _ = reg.validate_params(s.get("tool"), s.get("action"),
                                           params)
        if not ok:
            return False
    return True


def _params_valid(answer: Dict) -> bool:
    steps = answer.get("execution", {}).get("steps", [])
    return all(isinstance(s.get("params"), dict) for s in steps)

def aggregate(rows: List[Dict]) -> Dict:
    """Collapse per-question rows into the headline accuracy percentages."""

    def r(name, fn):
        vals = [fn(x) for x in rows if x.get(name) is not None]
        return round(100 * sum(vals) / len(vals), 1) if vals else None

    denom = max(1, len(rows))
    recovery_denom = sum(1 for x in rows if x.get("recovery_failure_seen"))
    recovery_num = sum(1 for x in rows if x.get("recovery_resolved"))
    sel_mismatched = [x for x in rows if not x.get("gt_hit") and x.get("n_steps", 0) > 0]
    return {
        "n": len(rows),
        "classification_accuracy_pct": r("classification_ok",
                                         lambda x: int(x["classification_ok"])),
        "selection_accuracy_pct": r("gt_hit", lambda x: int(x["gt_hit"])),
        "parameter_validity_pct": r("param_valid", lambda x: int(x["param_valid"])),
        "execution_success_pct": r("execution_ok", lambda x: int(x["execution_ok"])),
        "contract_satisfied_pct": r("contract_satisfied",
                                    lambda x: int(x["contract_satisfied"])),
        "final_sufficient_pct": r("final_verdict",
                                  lambda x: int(x["final_verdict"] == "SUFFICIENT")),
        "mean_iterations": round(
            sum(max(1, x.get("iterations", 1)) for x in rows) / denom, 2),
        "recovery_rate_pct": round(100 * recovery_num / recovery_denom, 1)
            if recovery_denom else None,
        "mismatched_questions": [x["qid"] for x in sel_mismatched],
    }


def run(questions: List[Dict],
        agent_factory: Callable[[], Agent],
        label: str = "llm",
        out_dir: str = "results/benchmarks",
        save: bool = True,
        spec_cache: Optional[Dict] = None) -> Dict:
    """Run a benchmark set with one shared analyzer cache so the task
    analysis stage is computed once per question (reused across ablation
    modes and pipelines) rather than once per agent."""
    shared = spec_cache if spec_cache is not None else {}
    rows = []
    for i, q in enumerate(questions):
        agent = agent_factory()
        agent.spec_cache = shared
        try:
            answer = agent.answer(q["question"])
        except Exception as exc:            # survive tool/LLM failures per-question
            answer = {"question": q["question"], "task_spec": {},
                      "execution": {"steps": [], "n_ok": 0, "n_fail": 1},
                      "plan_iterations": [], "verdict": {"status": "FAILED"},
                      "error": str(exc)[:200]}
        rows.append(measured_question(q, answer))
    report = {
        "label": label, "aggregate": aggregate(rows),
        "by_category": {cat: aggregate([x for x in rows if x["category"] == cat])
                        for cat in sorted({x["category"] for x in rows})},
        "rows": rows,
    }
    if save:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, label.replace(" ", "_").lower())
        with open(base + ".json", "w") as f:
            json.dump(report, f, indent=1, default=str)
    return report


def summarize(report: Dict) -> str:
    """Human-readable benchmark summary from a runner report."""
    a = report["aggregate"]
    lines = [f"[{report['label']}] n={a['n']}",
             f"  classification_accuracy   {a['classification_accuracy_pct']}%",
             f"  selection_accuracy (gt)   {a['selection_accuracy_pct']}%",
             f"  parameter_validity        {a['parameter_validity_pct']}%",
             f"  execution_success         {a['execution_success_pct']}%",
             f"  contract_satisfied        {a['contract_satisfied_pct']}%",
             f"  final_sufficient          {a['final_sufficient_pct']}%",
             f"  recovery_rate             {a['recovery_rate_pct']}%",
             f"  mean_iterations           {a['mean_iterations']}"]
    if a.get("mismatched_questions"):
        lines.append("  mismatched: " + ", ".join(a["mismatched_questions"][:12]))
    return "\n".join(lines)