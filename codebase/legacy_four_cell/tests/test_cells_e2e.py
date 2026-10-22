"""S5 end-to-end: one real query per cell through agent->tool->gate.

- PDB/SQL: real AVG query over the full SQLite mirror -> ValidatedResult.
- Literature: real corpus search -> claims -> ValidatedResult.
- ML: real predict validates with recorded group-aware metrics
  (results/ml/validation.json); missing metrics file still fails closed
  (fixture test proves the wiring).
- SCAPS: real precomputed sweep rows pass bounds/consistency/completeness
  after FF normalization, but `converged` honestly fails (no solver
  evidence on precomputed rows) -> FailedResult naming it. A recorded
  live-solver row passes the full gate; a live end-to-end (wine) runs
  under PEROVAGENTS_LIVE_SCAPS=1.
"""
import json
import os
import unittest

from perovagents.cells.base import DEFAULT_MAX_ATTEMPTS
from perovagents.cells.literature import LiteratureCell
from perovagents.cells.ml import MLCell
from perovagents.cells.pdb_sql import PDBSQLCell
from perovagents.cells.schemas import FailedResult, Task, ValidatedResult
from perovagents.cells.scaps import SCAPSCell
from perovagents.tools.sql_tool import default_db_path


class TestPDBCellE2E(unittest.TestCase):
    def test_real_avg_query_validates(self):
        cell = PDBSQLCell(db_path=default_db_path())
        task = Task(
            id="pdb1", cell="pdb_sql",
            request="average PCE for devices with SnO2 ETL",
            params={"aggregation": {"func": "mean",
                                    "column": "JV_default_PCE"},
                    "filters": [{"column": "ETL_stack_sequence",
                                 "op": "contains", "value": "SnO2"}],
                    "expected_cols": ["value"], "limit": 100})
        out = cell.run(task)
        self.assertIsInstance(out, ValidatedResult, out.to_dict())
        self.assertGreater(out.payload["rows"][0]["value"], 5.0)


class TestLiteratureCellE2E(unittest.TestCase):
    def test_real_search_validates(self):
        cell = LiteratureCell()
        task = Task(id="lit1", cell="literature",
                    request="SnO2 electron transport layer perovskite",
                    params={"k": 3})
        out = cell.run(task)
        self.assertIsInstance(out, ValidatedResult, str(out.to_dict())[:500])
        self.assertGreater(out.payload["n_hits"], 0)


class TestMLCellE2E(unittest.TestCase):
    DESC = ("n-i-p device with SnO2 ETL, Spiro-MeOTAD HTL, MAPbI3 "
            "perovskite 500 nm")

    def test_real_predict_validates_with_recorded_metrics(self):
        cell = MLCell()
        task = Task(id="ml1", cell="ml",
                    request="predict PCE for " + self.DESC,
                    params={"action": "predict", "architecture": "overall",
                            "description": self.DESC})
        out = cell.run(task)
        self.assertIsInstance(out, ValidatedResult, str(out.to_dict())[:600])
        val = out.payload.get("provenance", {}).get("validation", {})
        self.assertEqual(val.get("split"), "group-doi-5fold")
        self.assertGreater(val.get("score", 0), 0)

    def test_missing_metrics_still_fails_closed(self):
        cell = MLCell(metrics_path="/tmp/perovagents_nope.json")
        task = Task(id="ml1b", cell="ml",
                    request="predict PCE for " + self.DESC,
                    params={"action": "predict", "architecture": "overall",
                            "description": self.DESC})
        out = cell.run(task)
        self.assertIsInstance(out, FailedResult)
        self.assertEqual(len(out.attempts), DEFAULT_MAX_ATTEMPTS)
        self.assertEqual(out.last_critique.failed_checks,
                         ["model_attached"])

    def test_fixture_metrics_prove_pass_wiring(self):
        path = "/tmp/perovagents_fixture_metrics.json"
        with open(path, "w") as f:
            json.dump({"Overall": {"metric": "rmse", "score": 0.0,
                                   "split": "fixture-only"}}, f)
        cell = MLCell(metrics_path=path)
        task = Task(id="ml2", cell="ml",
                    request="predict PCE for " + self.DESC,
                    params={"action": "predict", "architecture": "overall",
                            "description": self.DESC})
        out = cell.run(task)
        self.assertIsInstance(out, ValidatedResult, str(out.to_dict())[:600])
        os.remove(path)


class TestSCAPSCellE2E(unittest.TestCase):
    def test_precomputed_sweep_fails_closed_on_convergence(self):
        cell = SCAPSCell()
        task = Task(id="sc1", cell="scaps",
                    request="band-gap sensitivity sweep",
                    params={"sweep_name": "pvk_Eg"})
        out = cell.run(task)
        self.assertIsInstance(out, FailedResult)
        names = out.last_critique.failed_checks
        self.assertIn("converged", names)
        # every other named check passed on real rows: only converged fails
        self.assertEqual(sorted(set(names)), ["converged"])

    def test_recorded_live_row_passes_full_gate(self):
        """Recorded real solver output (wine scaps3309.exe, 2026-09-29,
        pvk_d=500nm): proves live-shaped rows validate; not invented."""
        from perovagents.verify.physics_checks import verify_physics
        row = {"parameter": "perovskite_thickness", "value": 500.0,
               "unit": "nm", "converged": True,
               "convergence_failure": False, "Voc": 1.381731,
               "Jsc": 24.61148062, "FF": 0.905279, "PCE": 30.7853}
        rep = verify_physics(row)
        self.assertTrue(rep.passed,
                        [c.to_dict() for c in rep.checks if not c.passed])

    @unittest.skipUnless(os.environ.get("PEROVAGENTS_LIVE_SCAPS") == "1",
                         "live SCAPS needs wine (~3 min); opt-in only")
    def test_live_solver_point_validates(self):
        cell = SCAPSCell(max_attempts=1)
        out = cell.run(Task(
            id="sc-live", cell="scaps", request="thickness point",
            params={"parameter": "pvk_d", "values": [500.0], "live": True}))
        self.assertIsInstance(out, ValidatedResult, str(out.to_dict())[:400])
        self.assertTrue(out.report.passed)


class TestLLMRevision(unittest.TestCase):
    """A cell agent with an LLM revises (not replays) after a critique."""

    def test_pdb_agent_revises_bad_column(self):
        from perovagents.tools.sql_tool import build_sqlite

        class FakeLLM:
            def chat(self, prompt, temperature=0.0):
                assert "known_columns" in prompt  # critique fed to agent
                return '{"sql": "SELECT JV_default_PCE FROM devices LIMIT 2"}'

        mirror = "/tmp/perovagents_rev_test.db"
        build_sqlite(max_rows=200, db_path=mirror)
        cell = PDBSQLCell(db_path=mirror, max_attempts=2, llm=FakeLLM())
        out = cell.run(Task(
            id="rev", cell="pdb_sql", request="avg PCE",
            params={"sql": "SELECT unicorn_column FROM devices",
                    "expected_cols": ["JV_default_PCE"]}))
        self.assertIsInstance(out, ValidatedResult, str(out.to_dict())[:400])
        self.assertEqual(out.provenance["attempts"], 2)

    def test_unparsable_revision_falls_back_deterministically(self):
        from perovagents.cells.llm_agent import llm_tool_call
        from perovagents.cells.schemas import Attempt
        from perovagents.verify.base import check, report

        class BadLLM:
            def chat(self, prompt, temperature=0.0):
                return "not json at all"

        att = Attempt(n=1, tool_call={"sql": "SELECT 1"},
                      report=report([check("c", False, "bad")]))
        out = llm_tool_call(BadLLM(), "pdb_agent",
                            Task(id="t", cell="pdb_sql", request="r"),
                            [att], {"sql": "SELECT 1"})
        self.assertEqual(out, {"sql": "SELECT 1"})


if __name__ == "__main__":
    unittest.main()
