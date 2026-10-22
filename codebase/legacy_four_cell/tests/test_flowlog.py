"""Flow-log tests: every query/plan/step/verdict/response is persisted.

Run with:  python tests/test_flowlog.py

Covers the clean end-to-end flow log introduced so an entire run (question,
task spec, plans, each tool step with params+result, verdicts, final answer)
can be replayed from a single JSONL file. The legacy path (Agent.answer) and
the cells path (answer_cells) must both emit `flow_log`.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFlowLogLegacyPath(unittest.TestCase):
    def test_agent_answer_writes_complete_flow_log(self):
        import perovagents.config as config
        from perovagents import Agent
        from perovagents.core.flowlog import read_flow
        from perovagents.core.provenance import ProvenanceStore

        with tempfile.TemporaryDirectory() as tmp:
            config.RESULTS_DIR = tmp
            store = ProvenanceStore(persist_dir=os.path.join(tmp, "provenance"))
            agent = Agent(pipeline="rules", store=store,
                          flow_dir=os.path.join(tmp, "flows"))
            result = agent.answer("what is the average PCE for SnO2 devices?")
            self.assertIn("flow_log", result)
            path = os.path.join(tmp, result["flow_log"])
            # flow_log is rel to RESULTS_DIR which we overrode to tmp
            if not os.path.exists(path):
                path = os.path.join(tmp, "flows",
                                    os.path.basename(result["flow_log"]))
            self.assertTrue(os.path.exists(path), f"missing flow log {path}")

            events = read_flow(path)
            kinds = [e["event"] for e in events]
            for want in ("query", "plan", "step", "answer", "done"):
                self.assertIn(want, kinds)
            # at least one tool step logged with params + result
            step = next(e for e in events if e["event"] == "step")
            s = step.get("step", {})
            self.assertEqual(step["iteration"], 0)
            self.assertIn("tool", s)
            self.assertIn("action", s)
            self.assertIn("params", s)
            self.assertIn("result", s)
            # answer event carries the actual response text
            ans = next(e for e in events if e["event"] == "answer")
            self.assertIsInstance(ans.get("answer"), str)
            self.assertTrue(ans["answer"])

    def test_flow_plan_iterations_log_feedback(self):
        # RECOVERABLE verdict should trigger a replan -> a second plan event
        # with feedback attached. Rule pipeline has replan budget 0, so the
        # guaranteed behavior is: every executed plan is logged.
        import perovagents.config as config
        from perovagents import Agent
        from perovagents.core.flowlog import read_flow
        from perovagents.core.provenance import ProvenanceStore

        with tempfile.TemporaryDirectory() as tmp:
            config.RESULTS_DIR = tmp
            store = ProvenanceStore(persist_dir=os.path.join(tmp, "provenance"))
            agent = Agent(pipeline="rules", store=store,
                          flow_dir=os.path.join(tmp, "flows"))
            agent.answer("how many records use PTAA?")
            flows_dir = os.path.join(tmp, "flows")
            path = os.path.join(flows_dir,
                                sorted(os.listdir(flows_dir))[0])
            events = read_flow(path)
            plans = [e for e in events if e["event"] == "plan"]
            self.assertGreaterEqual(len(plans), 1)
            self.assertEqual(plans[0]["iteration"], 0)
            self.assertIn("planner", plans[0])


class TestFlowLogCellsPath(unittest.TestCase):
    def test_answer_cells_writes_flow_log(self):
        import perovagents.config as config
        from perovagents.control.app import answer_cells

        with tempfile.TemporaryDirectory() as tmp:
            config.RESULTS_DIR = tmp
            from perovagents.control.config import load_config
            cfg = load_config()
            cfg["trace"]["dir"] = os.path.join(tmp, "traces")
            result = answer_cells(
                "compare pce of Planar vs Meso n-i-p", pipeline="rules",
                cfg=cfg, trace_dir=cfg["trace"]["dir"])
            self.assertIn("flow_log", result)
            path = os.path.join(tmp, result["flow_log"])
            self.assertTrue(os.path.exists(path),
                            f"missing flow log {path}")
            from perovagents.core.flowlog import read_flow
            events = read_flow(path)
            kinds = [e["event"] for e in events]
            for want in ("query", "plan", "answer", "done"):
                self.assertIn(want, kinds)
            self.assertTrue(any(e["event"] == "step" for e in events),
                            "cell path should log per-task step outcomes")


if __name__ == "__main__":
    unittest.main(verbosity=2)