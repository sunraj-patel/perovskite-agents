"""Intent -> action -> verifier recovery chain for interpretation questions.

Locks the regression reported in review:
  "What are the most relevant parameters for improving an n-i-p device?"
must route analyzer -> feature_importance, rules/LLM planner -> ml.explain_global,
and the verifier must REJECT a database_evidence result for that task even when
the DB row superficially "has numbers" or even matching column names.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perovagents.agent.rules import RulePlanner
from perovagents.agent.task import TaskSpec, analyze_task
from perovagents.agent.verification import verify_task
from perovagents.agent.models import Plan, PlanStep

Q = "What are the most relevant parameters for improving an n-i-p device?"


def test_analyzer_classifies_interpretation_question():
    spec = analyze_task(Q, use_llm=False)
    assert spec.task_type == "feature_importance"
    assert spec.architecture == "n-i-p"
    assert {"feature_name", "importance", "direction"} <= set(spec.requested_output)
    assert spec.requires_explanation
    assert not spec.requires_prediction


def test_rules_planner_selects_explain_global_not_database():
    plan = RulePlanner().plan(Q)
    assert plan.steps and plan.steps[0].tool == "ml"
    assert plan.steps[0].action == "explain_global"
    assert plan.steps[0].params.get("architecture") == "nip"
    assert all(s.tool != "pdb" for s in plan.steps)


def test_rules_planner_still_defaults_to_db_for_factual():
    plan = RulePlanner().plan("How many devices use SnO2 as ETL?")
    assert plan.steps[0].tool == "pdb"


def _dict_step(success=True, claim="database_evidence", rt="table",
               data=None, columns=None):
    return {"step_index": 0, "tool": "pdb", "action": "query",
            "params": {"operation": "top_k"}, "success": success,
            "error": "", "claim_type": claim,
            "required_for_final_answer": True,
            "expected_outputs": ["feature_name", "importance", "direction",
                                 "records"],
            "result": {"result_type": rt,
                       "columns": columns or ["feature_name", "importance",
                                              "direction"],
                       "data": data or [{"feature_name": "X", "importance": 1}]}}


def _feature_importance_spec():
    return TaskSpec(task_type="feature_importance", architecture="n-i-p",
                    requested_output=["feature_name", "importance", "direction"],
                    requires_explanation=True)


def test_verifier_rejects_database_evidence_for_interpretation():
    step = _dict_step()                      # DB table that "has numbers"
    plan = Plan(question=Q, planner="rules",
                steps=[PlanStep(tool="pdb", action="query",
                                params={}, expected_outputs=[])])
    v = verify_task(_feature_importance_spec(), plan, [step])
    # claim-type gate must win over plausible-looking DB columns
    assert v.status == "RECOVERABLE"
    assert any("wrong evidence class" in d for d in v.diagnostics)
    assert v.checked.get("compatible_claim_level") is False


def test_verifier_accepts_model_association_for_interpretation():
    features = {"result_type": "explanation", "scope": "global",
                "architecture": "NIP", "method": "shap",
                "features": [{"name": "Perovskite_thickness_nm",
                              "importance": 0.42, "direction": "positive",
                              "shap": 0.40}]}
    step = {"step_index": 0, "tool": "ml", "action": "explain_global",
            "params": {"architecture": "NIP"}, "success": True,
            "error": "", "claim_type": "model_association",
            "required_for_final_answer": True,
            "expected_outputs": ["feature_name", "importance", "direction"],
            "result": features}
    plan = Plan(question=Q, planner="rules",
                steps=[PlanStep(tool="ml", action="explain_global",
                                params={"architecture": "NIP"}, expected_outputs=[])])
    v = verify_task(_feature_importance_spec(), plan, [step])
    assert v.status == "SUFFICIENT"


def test_verifier_accepts_optimize_none_found():
    spec = TaskSpec(task_type="optimization",
                    requested_output=["parameter_value", "predicted_pce",
                                      "interval"],
                    requires_optimization=True, requires_prediction=True)
    step = {"step_index": 0, "tool": "ml", "action": "optimize",
            "params": {}, "success": True, "error": "",
            "claim_type": "model_sensitivity",
            "required_for_final_answer": True,
            "expected_outputs": ["predicted_pce", "interval"],
            "result": {"result_type": "optimization",
                       "optimization_status": "none_found",
                       "prediction": 12.3, "predicted_pce": 12.3,
                       "interval": [10.0, 14.6], "pce_95_interval": [10.0, 14.6]}}
    plan = Plan(question="improve this device", planner="rules",
                steps=[PlanStep(tool="ml", action="optimize", params={},
                                expected_outputs=[])])
    v = verify_task(spec, plan, [step])
    assert v.status == "SUFFICIENT"