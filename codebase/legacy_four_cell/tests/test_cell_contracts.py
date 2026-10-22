"""S1 contract tests: cell schemas are pure data with correct defaults.

Covers spec section 5 model shapes, VerificationReport pass/fail logic,
RunState dependency readiness, and full task_type -> cell routing coverage.
"""
import unittest

from perovagents.cells.schemas import (
    Attempt, CELLS, Critique, FailedResult, Plan, RunState, Task,
    TASK_CELL_ROUTES, ValidatedResult, VerificationReport, CheckResult,
    cells_for,
)
from perovagents.agent.task import TASK_TYPES


class TestCellContracts(unittest.TestCase):
    def test_task_defaults(self):
        t = Task(id="t1", cell="ml", request="predict PCE")
        self.assertEqual(t.params, {})
        self.assertEqual(t.depends_on, [])
        d = t.to_dict()
        self.assertEqual(d["cell"], "ml")

    def test_plan_round_trip(self):
        p = Plan(query="q", tasks=[Task(id="a", cell="pdb_sql", request="r")])
        d = p.to_dict()
        self.assertEqual(d["tasks"][0]["id"], "a")

    def test_report_passes_only_when_all_checks_pass(self):
        self.assertFalse(VerificationReport().passed)  # empty never passes
        ok = VerificationReport([CheckResult("a", True), CheckResult("b", True)])
        self.assertTrue(ok.passed)
        bad = VerificationReport([CheckResult("a", True, offending={"x": 1}),
                                  CheckResult("b", False, "nope")])
        self.assertFalse(bad.passed)
        self.assertEqual(bad.failed_checks(), ["b"])

    def test_critique_from_report(self):
        rep = VerificationReport([CheckResult("schema", False, "bad col")])
        c = Critique(failed_checks=rep.failed_checks(), explanation="x")
        self.assertEqual(c.failed_checks, ["schema"])

    def test_attempt_links_call_result_report(self):
        a = Attempt(n=1, tool_call={"op": "count"},
                    raw_result={"value": 3},
                    report=VerificationReport([CheckResult("c", True)]))
        self.assertTrue(a.report.passed)

    def test_validated_and_failed_results(self):
        v = ValidatedResult(task_id="t", cell="ml", payload={"pce": 20.1})
        self.assertIn("provenance", v.to_dict())
        f = FailedResult(task_id="t", cell="ml", attempts=[Attempt(n=1)])
        d = f.to_dict()
        self.assertEqual(len(d["attempts"]), 1)
        self.assertIsNone(d["last_critique"])

    def test_runstate_ready_respects_dependencies(self):
        plan = Plan(query="q", tasks=[
            Task(id="pdb", cell="pdb_sql", request="r"),
            Task(id="ml", cell="ml", request="r", depends_on=["pdb"]),
        ])
        st = RunState(plan=plan)
        self.assertEqual(st.ready(), ["pdb"])
        st.mark("pdb", "validated")
        st.results["pdb"] = ValidatedResult(task_id="pdb", cell="pdb_sql")
        self.assertEqual(st.ready(), ["ml"])
        st.mark("ml", "running")
        self.assertEqual(st.ready(), [])
        with self.assertRaises(ValueError):
            st.mark("ml", "bogus")

    def test_failed_dependency_never_unblocks(self):
        plan = Plan(query="q", tasks=[
            Task(id="a", cell="ml", request="r"),
            Task(id="b", cell="scaps", request="r", depends_on=["a"]),
        ])
        st = RunState(plan=plan)
        st.mark("a", "failed")
        st.results["a"] = FailedResult(task_id="a", cell="ml")
        self.assertEqual(st.ready(), [])

    def test_routing_covers_every_task_type(self):
        uncovered = [t for t in TASK_TYPES if not cells_for(t)]
        self.assertEqual(uncovered, [])
        for t in TASK_TYPES:
            for c in cells_for(t):
                self.assertIn(c, CELLS + ("utility",))

    def test_sensitivity_prefers_ml_scaps_second(self):
        self.assertEqual(cells_for("sensitivity_analysis"), ["ml", "scaps"])

    def test_unknown_task_type_routes_nowhere(self):
        self.assertEqual(cells_for("nope"), [])


if __name__ == "__main__":
    unittest.main()
