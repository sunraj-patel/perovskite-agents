"""S9 acceptance: multi-cell cited answer + complete trace (spec §9).

Complements the step suites that already prove the other §9 items:
skeleton/retry-cap (test_cell_loop), gate units (test_verify_gates,
test_sql_tool), parallel + dependency chain (test_executor_v2), per-cell
e2e (test_cells_e2e). Old-path regression is the pre-existing 40-test
suite, run together with everything in CI order below.
"""
import json
import unittest

from perovagents.cells.schemas import Plan, Task
from perovagents.control.app import build_cells
from perovagents.control.config import load_config
from perovagents.control.executor import ExecutorV2
from perovagents.control.synthesis import synthesize


class TestMultiCellAcceptance(unittest.TestCase):
    def test_cited_answer_and_complete_trace(self):
        cfg = load_config()
        ex = ExecutorV2(cells=build_cells(cfg),
                        trace_dir="/tmp/perovagents_s9_traces")
        plan = Plan(query="SnO2 multi-cell acceptance", tasks=[
            Task(id="pdb", cell="pdb_sql", request="avg PCE SnO2",
                 params={"aggregation": {"func": "mean",
                                         "column": "JV_default_PCE"},
                         "filters": [{"column": "Cell_stack_sequence",
                                      "op": "contains", "value": "SnO2"}],
                         "expected_cols": ["value"]}),
            Task(id="lit", cell="literature",
                 request="SnO2 ETL perovskite", params={"k": 2}),
            Task(id="ml", cell="ml", request="predict",
                 params={"action": "predict", "architecture": "overall",
                         "description": "n-i-p SnO2 MAPbI3 500nm"}),
            Task(id="sc", cell="scaps", request="sweep",
                 params={"sweep_name": "pvk_d"}),
        ])
        out = ex.run(plan)
        s = out["summary"]
        # pdb + literature + ml validate (ml has recorded group-aware
        # metrics); scaps fails closed (precomputed rows lack solver
        # evidence) and is named, not dropped.
        self.assertEqual(s["status"]["pdb"], "validated")
        self.assertEqual(s["status"]["lit"], "validated")
        self.assertEqual(s["status"]["ml"], "validated")
        self.assertEqual(s["status"]["sc"], "failed")
        self.assertIn("converged", s["failed"]["sc"])

        synth = synthesize(plan.query, out["state"])
        self.assertIn("SELECT", synth["answer"])   # SQL cited
        self.assertRegex(synth["answer"], r"10\.\S+")  # paper cited
        self.assertIn("model_Overall.cbm", synth["answer"])  # model cited
        self.assertNotIn("ml", synth["failed"])
        self.assertIn("sc", synth["failed"])
        self.assertTrue(synth["sources"])

        # trace replays the whole run
        events = [json.loads(line) for line in
                  open(s["trace"], encoding="utf-8")]
        kinds = [e["event"] for e in events]
        for want in ("plan", "dispatch", "validated", "failed", "done"):
            self.assertIn(want, kinds)
        verdicts = [e for e in events
                    if e["event"] in ("validated", "failed")]
        self.assertEqual(len(verdicts), 4)
        for e in events:
            if e["event"] == "validated":
                self.assertTrue(e["report"]["passed"])


class TestEntryPoints(unittest.TestCase):
    def test_console_script_target_exists(self):
        from perovagents.cli import main
        self.assertTrue(callable(main))

    def test_cells_flag_parses_and_defaults_unchanged(self):
        from perovagents.__main__ import build_parser
        args = build_parser().parse_args(["--cells", "q?"])
        self.assertTrue(args.cells)
        self.assertEqual(args.pipeline, "llm")
        args = build_parser().parse_args(["q?"])
        self.assertFalse(args.cells)


if __name__ == "__main__":
    unittest.main()
