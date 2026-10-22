"""Offline smoke tests for the framework.

Run with:  python tests/test_smoke.py

Covers: registry construction, Tool knowledge graph, calculator, PDB query
(needs the real CSV, so it is skipped if the data file is missing), planner
rule fallback, and the full Agent pipeline with rule planner.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perovagents import Agent
from perovagents import config
from perovagents.core import default_registry
from perovagents.core.provenance import ProvenanceStore
from perovagents.agent import RulePlanner


class TestRegistry(unittest.TestCase):
    def test_constructs(self):
        reg = default_registry()
        names = reg.names()
        self.assertIn("pdb", names)
        self.assertIn("calculator", names)
        self.assertIn("literature", names)
        self.assertIn("scaps", names)
        self.assertIn("ml", names)

    def test_tool_knowledge_graph(self):
        from perovagents.core import ToolKnowledgeGraph
        reg = default_registry()
        kg = ToolKnowledgeGraph(reg)
        self.assertIn("pdb", kg.related("ml"))


class TestCalculator(unittest.TestCase):
    def test_ev_nm_roundtrip(self):
        from perovagents.core.calculator import CalculatorTool
        t = CalculatorTool()
        r = t.call({"action": "convert", "value": 1.5, "from": "eV", "to": "nm"})
        self.assertAlmostEqual(r["result"], 826.56, delta=1)

    def test_pce_formula(self):
        from perovagents.core.calculator import CalculatorTool
        r = CalculatorTool().call({"action": "pce", "voc": 1.1, "jsc": 25.0, "ff": 0.8})
        self.assertAlmostEqual(r["pce_percent"], 22.0, delta=0.01)


class TestPlanner(unittest.TestCase):
    def test_rules_pick_scaps(self):
        plan = RulePlanner().plan("does absorber thickness affect PCE in the physics model?")
        tools = [s.tool for s in plan.steps]
        self.assertIn("scaps", tools)


@unittest.skipUnless(os.path.exists(config.PDB_CSV), "PDB CSV not present")
class TestPDB(unittest.TestCase):
    def test_count(self):
        from perovagents.materials.pdb import PDBTool
        t = PDBTool()
        r = t.call({"operation": "count",
                    "filters": [{"column": "HTL_stack_sequence", "op": "contains",
                                 "value": "Spiro"}]})
        self.assertEqual(r["result_type"], "count")
        self.assertGreater(r["value"], 0)


class TestAgent(unittest.TestCase):
    def test_offline_pipeline(self):
        agent = Agent(pipeline="rules")
        out = agent.answer("what is the average PCE for devices with SnO2 ETL?")
        self.assertIn("trace_id", out)
        self.assertIn("answer", out)
        self.assertEqual(out["execution"]["n_ok"], 1)

    def test_trace_embeds_step_results(self):
        agent = Agent(pipeline="rules")
        out = agent.answer("count records with Spiro as HTL")
        s0 = out["execution"]["steps"][0]
        self.assertIn("result", s0, "trace steps must carry the full result dict")
        self.assertEqual(s0["result"]["result_type"], "count")
        self.assertGreater(s0["result"]["value"], 0)

    def test_step_callbacks_fire(self):
        agent = Agent(pipeline="rules")
        events = []
        out = agent.answer(
            "what is the average PCE for SnO2 devices?",
            on_plan=lambda p: events.append(("plan", p["planner"])),
            on_step_start=lambda i, s: events.append(("start", i, s["tool"])),
            on_step_done=lambda i, s: events.append(("done", i, s["tool"], s["success"])))
        self.assertEqual(events[0][0], "plan")
        self.assertIn("start", [e[0] for e in events])
        self.assertIn("done", [e[0] for e in events])
        self.assertTrue(events[-1][3])


@unittest.skipUnless(os.path.exists(os.path.join(config.RESULTS_DIR, "ml", "model_Overall.cbm")),
                     "ML model artifacts not present")
class TestMLPredictor(unittest.TestCase):
    def test_predict(self):
        from perovagents.ml.predictor import MLPredictorTool
        t = MLPredictorTool()
        r = t.call({"description": "SnO2 ETL, PTAA HTL, CsMAFA perovskite 500 nm thick",
                    "architecture": "p-i-n", "action": "predict"})
        self.assertEqual(r["result_type"], "prediction")
        self.assertGreater(r["predicted_pce"], 0)
        self.assertEqual(r["architecture"], "PIN")
        lo, hi = r["pce_95_interval"]
        self.assertLessEqual(lo, r["predicted_pce"])
        self.assertGreaterEqual(hi, r["predicted_pce"])

    def test_arch_alias_model_resolution(self):
        from perovagents.ml.predictor import MLPredictorTool
        t = MLPredictorTool()
        r = t.call({"description": "p-i-n cell with SnO2 ETL, PTAA HTL and CsMAFA perovskite",
                    "architecture": "pin", "action": "predict"})
        self.assertEqual(r["provenance"]["model"], "model_PIN.cbm")


class TestActionRegistry(unittest.TestCase):
    """Redesign: actions are first-class, schema-validated, never flag-driven."""

    def test_ml_exposes_discrete_actions(self):
        reg = default_registry()
        ml = next(s for s in reg.available() if s.name == "ml")
        self.assertEqual(sorted(ml.actions.keys()),
                         ["explain_global", "explain_local", "optimize",
                          "predict", "what_if"])

    def test_ml_predict_rejects_explain_flag(self):
        # the old silent-failure path: "explain" was a param, now it is rejected
        reg = default_registry()
        ok, errs, _ = reg.validate_params("ml", "predict",
                                          {"explain": True, "description": "x"})
        self.assertFalse(ok)
        self.assertTrue(any("explain" in e for e in errs))

    def test_pdb_query_requires_operation(self):
        reg = default_registry()
        ok, errs, _ = reg.validate_params("pdb", "query", {})
        self.assertFalse(ok)
        self.assertTrue(any("operation" in e.lower() for e in errs))

    def test_unknown_action_rejected(self):
        reg = default_registry()
        self.assertFalse(reg.validate_action("ml", "explain"))

    def test_knowledge_graph_action_compatibility(self):
        from perovagents.core import ToolKnowledgeGraph
        kg = ToolKnowledgeGraph(default_registry())
        # ml.explain_global -> ml.what_if action-compat edge produces a self loop
        self.assertIn("ml", kg.related("ml"))
        self.assertIn("pdb", kg.related("ml"))


class TestControllability(unittest.TestCase):
    def test_classification(self):
        from perovagents.ml.controllability import is_controllable, classify_feature
        self.assertTrue(is_controllable("Perovskite_thickness_nm"))
        self.assertTrue(is_controllable("Perov_anneal_temp_C"))
        self.assertFalse(is_controllable("JV_default_PCE"))
        self.assertFalse(is_controllable("Ref_doi"))
        self.assertFalse(is_controllable("Stability_PCE_retention_after_1000h"))
        self.assertEqual(classify_feature("Perovskite_thickness_nm"),
                         "processing_parameter")


@unittest.skipUnless(os.path.exists(os.path.join(config.RESULTS_DIR, "ml", "model_Overall.cbm")),
                     "ML model artifacts not present")
class TestMLActions(unittest.TestCase):
    def test_explain_global_returns_feature_contract(self):
        from perovagents.ml.predictor import MLPredictorTool
        r = MLPredictorTool().call({"action": "explain_global", "architecture": "overall"})
        self.assertIn("features", r)
        self.assertTrue(all({"name", "importance", "direction"} <= set(f)
                            for f in r["features"]))

    def test_explain_local(self):
        from perovagents.ml.predictor import MLPredictorTool
        r = MLPredictorTool().call({"action": "explain_local",
                                    "description": "p-i-n cell with SnO2 ETL, "
                                                   "PTAA HTL and CsMAFA perovskite"})
        self.assertIn("features", r)
        self.assertGreaterEqual(len(r["features"]), 1)

    def test_what_if_fuzzy_parameter(self):
        from perovagents.ml.predictor import MLPredictorTool
        r = MLPredictorTool().call({"action": "what_if",
                                    "description": "p-i-n device with SnO2 ETL, "
                                                   "PTAA HTL and 400 nm perovskite",
                                    "parameter": "perovskite thickness",
                                    "values": [400, 500, 600]})
        self.assertEqual(r["parameter"], "Perovskite_thickness_nm")
        self.assertEqual(r["result_type"], "what_if")
        self.assertGreaterEqual(len(r["data"]), 2)
        self.assertTrue(all(x["predicted_pce"] > 0 for x in r["data"]))

    def test_what_if_rejects_measurement_derived(self):
        from perovagents.ml.predictor import MLPredictorTool
        with self.assertRaises(ValueError):
            MLPredictorTool().call({"action": "what_if",
                                    "description": "p-i-n cell with SnO2 ETL",
                                    "parameter": "JV_default_PCE",
                                    "values": [18, 19, 20]})

    def test_what_if_rejects_binary_flag(self):
        from perovagents.ml.predictor import MLPredictorTool
        with self.assertRaises(ValueError):
            MLPredictorTool().call({"action": "what_if",
                                    "description": "p-i-n cell with SnO2 ETL",
                                    "parameter": "Perovskite_dimension_3D_with_2D"
                                                 "_capping_layer",
                                    "values": [0, 1]})

    def test_optimize_returns_ranked_recommendations(self):
        from perovagents.ml.predictor import MLPredictorTool
        r = MLPredictorTool().call({"action": "optimize",
                                    "description": "p-i-n cell with SnO2 ETL, "
                                                   "PTAA HTL and CsMAFA perovskite"})
        self.assertEqual(r["result_type"], "optimization")
        for rec in r["recommendations"]:
            self.assertGreaterEqual(rec["delta_predicted_pce"], 0.0)


class TestVerification(unittest.TestCase):
    def test_sufficient_when_tokens_present(self):
        from perovagents.agent.models import Plan
        from perovagents.agent.task import TaskSpec
        spec = TaskSpec(task_type="feature_importance",
                        requested_output=["feature_name", "importance", "direction"],
                        requires_explanation=True)
        plan = Plan(question="q", steps=[])
        from perovagents.agent.verification import verify_task
        v = verify_task(spec, plan, [
            {"success": True, "claim_type": "model_association",
             "result": {"result_type": "explanation",
                        "features": [{"name": "a",
                                      "importance": 0.4,
                                      "direction": "positive"}]},
             "required_for_final_answer": True, "expected_outputs": []}])
        self.assertEqual(v.status, "SUFFICIENT", v.missing_requirements)

    def test_unresolved_when_all_steps_fail(self):
        from perovagents.agent.models import Plan
        from perovagents.agent.task import TaskSpec
        from perovagents.agent.verification import verify_task
        spec = TaskSpec(objective="q")
        v = verify_task(spec, Plan(question="q"), [{"success": False, "result": None}])
        self.assertEqual(v.status, "UNRESOLVED")

    def test_analyzer_flags_intervention(self):
        from perovagents.agent.task import analyze_task
        a = analyze_task("what should i change to improve the pce of my "
                         "n-i-p device?", llm=None)
        self.assertEqual(a.task_type, "optimization")
        self.assertTrue(a.requires_intervention)
        self.assertTrue(a.requires_explanation)


if __name__ == "__main__":
    unittest.main(verbosity=2)