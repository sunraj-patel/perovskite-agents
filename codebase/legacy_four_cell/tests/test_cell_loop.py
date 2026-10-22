"""S3 cell-loop tests (spec section 9: skeleton + retry cap).

A dummy cell with a failing-then-passing gate must loop exactly once and
emit a ValidatedResult; an always-failing gate must end in FailedResult
after max_attempts. Tool exceptions count as attempts, never crashes.
"""
import unittest

from perovagents.cells.base import DomainCell
from perovagents.cells.schemas import (FailedResult, Task, ValidatedResult)
from perovagents.verify.base import check, report


def _gate_ok_once_factory():
    state = {"calls": 0}

    def gate(raw, call):
        state["calls"] += 1
        if state["calls"] == 1:
            return report([check("shape", False, "empty on first try")])
        return report([check("shape", True, "rows present")])
    return gate


class TestCellSkeleton(unittest.TestCase):
    def test_fail_once_then_pass_loops_exactly_once(self):
        seen = []

        def agent(task, history):
            seen.append([a.report.passed for a in history])
            return {"op": "count"}

        cell = DomainCell(agent_fn=agent,
                          tool_fn=lambda call: {"rows": [{"n": 3}]},
                          gate_fn=_gate_ok_once_factory())
        out = cell.run(Task(id="t1", cell="pdb_sql", request="count"))
        self.assertIsInstance(out, ValidatedResult)
        self.assertEqual(out.provenance["attempts"], 2)
        # agent saw the failed first attempt before revising
        self.assertEqual(seen, [[], [False]])
        self.assertTrue(out.report.passed)

    def test_always_fail_ends_in_failed_result(self):
        cell = DomainCell(
            agent_fn=lambda task, history: {"op": "count"},
            tool_fn=lambda call: {"rows": []},
            gate_fn=lambda raw, call: report(
                [check("shape", False, "always empty")]),
            max_attempts=3)
        out = cell.run(Task(id="t2", cell="ml", request="x"))
        self.assertIsInstance(out, FailedResult)
        self.assertEqual(len(out.attempts), 3)
        self.assertEqual(out.last_critique.failed_checks, ["shape"])

    def test_retry_budget_respected(self):
        cell = DomainCell(
            agent_fn=lambda task, history: {},
            tool_fn=lambda call: {},
            gate_fn=lambda raw, call: report(
                [check("c", False, "no")]),
            max_attempts=1)
        out = cell.run(Task(id="t", cell="scaps", request="x"))
        self.assertIsInstance(out, FailedResult)
        self.assertEqual(len(out.attempts), 1)

    def test_retried_call_differs_after_critique(self):
        """The agent revises its call from the failed check (unknown
        column -> known column); the retry must differ from attempt 1."""
        from perovagents.verify.base import report as make_report
        from perovagents.verify.sql_checks import check_known_columns
        schema = {"devices": ["JV_default_PCE"]}
        calls = []

        def agent(task, history):
            if not history:
                call = {"sql": "SELECT unicorn_column FROM devices"}
            else:
                failed = history[-1].report.failed_checks()
                self.assertIn("known_columns", failed)
                call = {"sql": "SELECT JV_default_PCE FROM devices"}
            calls.append(call)
            return call

        def gate(raw, call):
            return make_report(
                [check_known_columns(call["sql"], schema)])

        cell = DomainCell(agent_fn=agent,
                          tool_fn=lambda call: {"rows": [{"JV_default_PCE": 1}]},
                          gate_fn=gate)
        out = cell.run(Task(id="t9", cell="pdb_sql", request="avg PCE"))
        self.assertIsInstance(out, ValidatedResult)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0], calls[1])
        self.assertIn("JV_default_PCE", calls[1]["sql"])

    def test_tool_exception_becomes_critique_not_crash(self):
        calls = {"n": 0}
        histories = []

        def agent(task, history):
            histories.append(list(history))
            return {}

        def tool(call):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return {"rows": [{"n": 1}]}

        cell = DomainCell(
            agent_fn=agent,
            tool_fn=tool,
            gate_fn=lambda raw, call: report([check("c", True, "ok")]))
        out = cell.run(Task(id="t", cell="literature", request="x"))
        self.assertIsInstance(out, ValidatedResult)
        self.assertEqual(out.provenance["attempts"], 2)
        # the agent saw the tool-error attempt with its critique content
        first = histories[1][0]
        self.assertFalse(first.report.passed)
        self.assertEqual(first.report.failed_checks(), ["tool_ok"])
