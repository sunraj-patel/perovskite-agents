"""S8 synthesis tests: validated-only answers, citations, failed naming."""
import unittest

from perovagents.cells.schemas import (FailedResult, Plan, RunState, Task,
                                       ValidatedResult)
from perovagents.control.synthesis import synthesize
from perovagents.verify.base import check, report


def _v(tid, cell, payload):
    return ValidatedResult(task_id=tid, cell=cell, payload=payload,
                           report=report([check("c", True, "ok")]))


def _state(results):
    plan = Plan(query="q", tasks=[])
    st = RunState(plan=plan)
    for tid, res in results.items():
        st.results[tid] = res
        st.mark(tid, "validated" if isinstance(res, ValidatedResult)
                else "failed")
    return st


class TestSynthesis(unittest.TestCase):
    def test_failed_results_excluded_but_named(self):
        st = _state({
            "p": _v("p", "pdb_sql",
                    {"rows": [{"value": 21.5}],
                     "provenance": {"sql": "SELECT 1"}}),
            "m": FailedResult(task_id="m", cell="ml", attempts=[]),
        })
        st.errors["m"] = "no recorded validation performance"
        out = synthesize("avg PCE?", st)
        self.assertIn("21.5", out["answer"])
        self.assertNotIn("ml_prediction", out["answer"])
        self.assertIn("m", out["failed"])
        self.assertIn("no recorded validation performance",
                      out["answer"])

    def test_every_claim_cited(self):
        st = _state({
            "l": _v("l", "literature", {"claims": [
                {"text": "SnO2 works",
                 "citations": [{"paper_id": "10/x", "locator": "p2",
                                "passage": "SnO2 works"}]}]}),
            "m": _v("m", "ml",
                    {"predicted_pce": 20.1,
                     "pce_95_interval": [18.0, 22.0],
                     "provenance": {"model": "model_Overall.cbm"}}),
        })
        out = synthesize("q", st)
        self.assertIn("10/x#p2", out["answer"])
        self.assertIn("model_Overall.cbm", out["answer"])
        self.assertTrue(out["sources"])

    def test_empty_state_makes_no_claims(self):
        st = _state({"m": FailedResult(task_id="m", cell="ml",
                                       attempts=[])})
        st.errors["m"] = "gate failed"
        out = synthesize("q", st)
        self.assertIn("No validated results", out["answer"])
        self.assertIn("empty answer", out["warnings"][0])

    def test_deterministic(self):
        st = _state({"p": _v("p", "pdb_sql",
                              {"rows": [{"value": 1}],
                               "provenance": {"sql": "SELECT 1"}})})
        self.assertEqual(synthesize("q", st)["answer"],
                         synthesize("q", st)["answer"])

    def test_every_rendered_section_cites(self):
        # regression: explanation-only ML, empty literature/scaps payloads
        # rendered lines but appended no source (cited_pct < 100%).
        st = _state({
            "m": _v("m", "ml", {"result_type": "explanation",
                                "provenance": {"model": "model_X.cbm"}}),
            "l": _v("l", "literature", {"claims": []}),
            "s": _v("s", "scaps", {"parameter": "pvk_d", "data": []}),
        })
        out = synthesize("q", st)
        self.assertTrue(out["sources"])
        self.assertIn("model_X.cbm", " ".join(out["sources"]))


if __name__ == "__main__":
    unittest.main()
