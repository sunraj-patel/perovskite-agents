"""S6 planner tests: cell-aware tasks, subset selection, deps, prompts."""
import unittest

from perovagents.agent.task import TASK_TYPES
from perovagents.cells.schemas import CELLS
from perovagents.control import planner as P
from perovagents.prompts import load as load_prompt


class FakeLLM:
    available = True

    def __init__(self, text):
        self.text = text

    def chat(self, prompt, temperature=0.0):
        return self.text


class TestRulesPlanner(unittest.TestCase):
    def test_database_question_selects_pdb_only(self):
        plan = P.plan_rules("What is the average PCE for devices with SnO2?")
        self.assertEqual(len(plan.tasks), 1)  # subset, not all four
        t = plan.tasks[0]
        self.assertEqual(t.cell, "pdb_sql")
        self.assertIn("aggregation", t.params)
        self.assertEqual(t.depends_on, [])

    def test_prediction_selects_ml_only(self):
        plan = P.plan_rules("Predict the PCE of an n-i-p cell with SnO2.")
        self.assertEqual(len(plan.tasks), 1)
        self.assertEqual(plan.tasks[0].cell, "ml")
        self.assertEqual(plan.tasks[0].params["action"], "predict")

    def test_comparison_fans_out(self):
        plan = P.plan_rules("Compare n-i-p vs p-i-n average PCE.")
        cells = [t.cell for t in plan.tasks]
        self.assertGreaterEqual(len(cells), 2)
        self.assertIn("pdb_sql", cells)

    def test_every_task_has_depends_on_list(self):
        for q in ["What is the average PCE?", "Predict PCE for MAPbI3.",
                  "Compare SnO2 vs TiO2.", "How does thickness affect PCE?"]:
            for t in P.plan_rules(q).tasks:
                self.assertIsInstance(t.depends_on, list)

    def test_what_if_params(self):
        plan = P.plan_rules(
            "What if the perovskite thickness increases? Sensitivity?")
        ml = [t for t in plan.tasks if t.cell == "ml"]
        self.assertTrue(ml)
        self.assertEqual(ml[0].params["action"], "what_if")
        self.assertIn("values", ml[0].params)

    def test_unknown_defaults_to_pdb(self):
        plan = P.plan_rules("blorpy glorp quux?")
        self.assertTrue(plan.tasks)
        self.assertEqual(plan.tasks[0].cell, "pdb_sql")


class TestLLMPlanner(unittest.TestCase):
    GOOD = ('[{"id": "a", "cell": "pdb_sql", "request": "avg PCE",'
            ' "params": {"sql": "SELECT 1"}, "depends_on": []},'
            ' {"id": "b", "cell": "ml", "request": "predict",'
            ' "params": {"action": "predict"}, "depends_on": ["a"]}]')

    def test_parses_tasks_and_deps(self):
        plan = P.plan_llm("q", FakeLLM(self.GOOD))
        self.assertEqual(plan.planner, "llm")
        by_id = {t.id: t for t in plan.tasks}
        self.assertEqual(by_id["b"].depends_on, ["a"])

    def test_bad_json_falls_back_to_rules(self):
        plan = P.plan_llm("What is the average PCE?", FakeLLM("nope"))
        self.assertEqual(plan.planner, "rules")

    def test_unknown_cell_skipped_then_fallback(self):
        plan = P.plan_llm("What is the average PCE?",
                          FakeLLM('[{"id": "a", "cell": "warp_drive"}]'))
        self.assertEqual(plan.planner, "rules")

    def test_dangling_dep_falls_back_to_rules(self):
        plan = P.plan_llm("What is the average PCE?",
                          FakeLLM('[{"id": "a", "cell": "pdb_sql",'
                                  ' "depends_on": ["ghost"]}]'))
        self.assertEqual(plan.planner, "rules")

    def test_entry_prefers_llm_then_rules(self):
        plan = P.plan("Predict PCE.", llm=FakeLLM(self.GOOD),
                      pipeline="llm")
        self.assertEqual(plan.planner, "llm")
        plan = P.plan("Predict PCE.", llm=None, pipeline="rules")
        self.assertEqual(plan.planner, "rules")


class TestPrompts(unittest.TestCase):
    def test_all_prompts_load(self):
        for name in ("planner", "executor", "pdb_agent", "literature_agent",
                     "ml_agent", "scaps_agent"):
            text = load_prompt(name)
            self.assertTrue(len(text) > 100, name)

    def test_missing_prompt_falls_back(self):
        self.assertEqual(load_prompt("nope", fallback="fb"), "fb")

    def test_planner_prompt_names_cells(self):
        text = load_prompt("planner")
        for cell in list(CELLS) + ["depends_on"]:
            self.assertIn(cell, text)


if __name__ == "__main__":
    unittest.main()
