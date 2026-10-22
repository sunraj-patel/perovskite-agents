"""S7 executor tests: parallel dispatch, dependencies, failures, trace."""
import json
import os
import threading
import time
import unittest

from perovagents.cells.base import DomainCell
from perovagents.cells.schemas import Plan, Task
from perovagents.control.config import load_config
from perovagents.control.executor import ExecutorV2
from perovagents.verify.base import check, report

TRACE = "/tmp/perovagents_s7_traces"


def _ok_cell(cell_name="dummy", delay=0.0):
    def agent(task, history):
        if delay:
            time.sleep(delay)
        return {}

    def tool(call):
        return {"rows": [{"n": 1}]}

    def gate(raw, call):
        return report([check("c", True, "ok")])

    c = DomainCell(agent, tool, gate)
    c.cell = cell_name
    return c


def _fail_cell(cell_name="dummy"):
    c = DomainCell(lambda t, h: {},
                   lambda call: {"rows": []},
                   lambda raw, call: report(
                       [check("c", False, "always")]),
                   max_attempts=2)
    c.cell = cell_name
    return c


class TestParallelDispatch(unittest.TestCase):
    def test_independent_tasks_overlap_in_time(self):
        entered, proceed = [], threading.Event()

        def agent(task, history):
            entered.append(task.id)
            if len(entered) == 2:
                proceed.set()
            self.assertTrue(proceed.wait(timeout=10),
                            "tasks did not overlap: ran sequentially")
            return {}

        def make(name):
            c = DomainCell(agent, lambda call: {"rows": [{"n": 1}]},
                           lambda raw, call: report(
                               [check("c", True, "ok")]))
            c.cell = name
            return c

        ex = ExecutorV2(cells={"a": make("a"), "b": make("b")},
                        trace_dir=TRACE)
        t0 = time.time()
        out = ex.run(Plan(query="q", tasks=[
            Task(id="t1", cell="a", request="r"),
            Task(id="t2", cell="b", request="r")]))
        self.assertLess(time.time() - t0, 9.0)  # sequential would take >= 10s
        self.assertEqual(out["summary"]["validated"], ["t1", "t2"])

    def test_real_pdb_and_literature_run_parallel(self):
        from perovagents.cells.literature import LiteratureCell
        from perovagents.cells.pdb_sql import PDBSQLCell
        from perovagents.tools.sql_tool import default_db_path
        ex = ExecutorV2(
            cells={"pdb_sql": PDBSQLCell(db_path=default_db_path()),
                   "literature": LiteratureCell()},
            trace_dir=TRACE)
        out = ex.run(Plan(query="q", tasks=[
            Task(id="p", cell="pdb_sql", request="avg PCE SnO2",
                 params={"aggregation": {"func": "mean",
                                         "column": "JV_default_PCE"},
                         "filters": [{"column": "Cell_stack_sequence",
                                      "op": "contains", "value": "SnO2"}],
                         "expected_cols": ["value"]}),
            Task(id="l", cell="literature",
                 request="SnO2 ETL perovskite", params={"k": 2})]))
        self.assertEqual(out["summary"]["validated"], ["l", "p"])
        self.assertTrue(out["summary"]["trace"].endswith(".jsonl"))


class TestDependencies(unittest.TestCase):
    def test_dep_payloads_passed_along_chain(self):
        seen = {}

        def agent(task, history):
            seen[task.id] = dict(task.params.get("dep_results", {}))
            return {}

        def make(name):
            c = DomainCell(agent, lambda call: {"rows": [{"n": 1}]},
                           lambda raw, call: report(
                               [check("c", True, "ok")]))
            c.cell = name
            return c

        ex = ExecutorV2(cells={"a": make("a"), "b": make("b"),
                               "c": make("c")})
        out = ex.run(Plan(query="q", tasks=[
            Task(id="pdb", cell="a", request="r"),
            Task(id="ml", cell="b", request="r", depends_on=["pdb"]),
            Task(id="sc", cell="c", request="r", depends_on=["ml"])]))
        self.assertEqual(out["summary"]["validated"], ["ml", "pdb", "sc"])
        self.assertEqual(set(seen["ml"]), {"pdb"})
        self.assertEqual(set(seen["sc"]), {"ml"})
        self.assertIn("n", seen["ml"]["pdb"]["rows"][0])

    def test_real_pdb_ml_scaps_chain_order(self):
        from perovagents.cells.literature import LiteratureCell  # noqa
        from perovagents.cells.ml import MLCell
        from perovagents.cells.pdb_sql import PDBSQLCell
        from perovagents.cells.scaps import SCAPSCell
        from perovagents.tools.sql_tool import default_db_path
        ex = ExecutorV2(
            cells={"pdb_sql": PDBSQLCell(db_path=default_db_path()),
                   "ml": MLCell(),
                   "scaps": SCAPSCell()},
            trace_dir=TRACE)
        out = ex.run(Plan(query="q", tasks=[
            Task(id="pdb", cell="pdb_sql", request="avg",
                 params={"aggregation": {"func": "mean",
                                         "column": "JV_default_PCE"},
                         "expected_cols": ["value"]}),
            Task(id="ml", cell="ml", request="predict",
                 params={"action": "predict", "architecture": "overall",
                         "description": "n-i-p SnO2 MAPbI3"},
                 depends_on=["pdb"]),
            Task(id="sc", cell="scaps", request="sweep",
                 params={"sweep_name": "pvk_d"}, depends_on=["ml"])]))
        # pdb + ml validate (ml has recorded group-aware metrics);
        # scaps fails closed (precomputed rows lack solver evidence).
        s = out["summary"]
        self.assertEqual(s["status"]["pdb"], "validated")
        self.assertEqual(s["status"]["ml"], "validated")
        self.assertEqual(s["status"]["sc"], "failed")
        self.assertIn("converged", s["failed"]["sc"])
        events = [json.loads(line) for line in
                  open(s["trace"], encoding="utf-8")]
        order = [(e["event"], e.get("task")) for e in events]
        self.assertLess(order.index(("validated", "pdb")),
                        order.index(("dispatch", "ml")))
        self.assertLess(order.index(("validated", "ml")),
                        order.index(("dispatch", "sc")))


class TestFailureAndTimeout(unittest.TestCase):
    def test_failed_cell_skips_dependents(self):
        ex = ExecutorV2(cells={"a": _fail_cell("a"),
                               "b": _ok_cell("b")})
        out = ex.run(Plan(query="q", tasks=[
            Task(id="a1", cell="a", request="r"),
            Task(id="b1", cell="b", request="r", depends_on=["a1"])]))
        s = out["summary"]
        self.assertEqual(s["status"]["a1"], "failed")
        self.assertEqual(s["status"]["b1"], "skipped")
        self.assertIn("a1", s["failed"])  # surfaced, never dropped

    def test_task_timeout(self):
        ex = ExecutorV2(cells={"a": _ok_cell("a", delay=3.0)},
                        timeout_s=0.3)
        out = ex.run(Plan(query="q", tasks=[
            Task(id="a1", cell="a", request="r")]))
        self.assertEqual(out["summary"]["status"]["a1"], "failed")
        self.assertIn("timeout", out["summary"]["failed"]["a1"])

    def test_utility_task(self):
        ex = ExecutorV2(
            cells={}, utility_runner=lambda task: {"result": 2})
        out = ex.run(Plan(query="q", tasks=[
            Task(id="u", cell="utility", request="1+1",
                 params={"action": "eval"})]))
        self.assertEqual(out["summary"]["validated"], ["u"])


class TestTraceAndConfig(unittest.TestCase):
    def test_trace_replays_run(self):
        ex = ExecutorV2(cells={"a": _ok_cell("a")}, trace_dir=TRACE,
                        seed=7)
        out = ex.run(Plan(query="q", tasks=[
            Task(id="a1", cell="a", request="r")]))
        events = [json.loads(line) for line in
                  open(out["summary"]["trace"], encoding="utf-8")]
        kinds = [e["event"] for e in events]
        for want in ("plan", "dispatch", "validated", "done"):
            self.assertIn(want, kinds)
        plans = [e for e in events if e["event"] == "plan"]
        self.assertEqual(plans[0]["seed"], 7)
        verdicts = [e for e in events if e["event"] == "validated"]
        self.assertTrue(verdicts[0]["report"]["passed"])

    def test_config_defaults_and_override(self):
        cfg = load_config("/tmp/perovagents_nope.yaml")
        self.assertEqual(cfg["cells"]["max_attempts"], 3)
        self.assertIn("mirror", cfg["db"])
        path = "/tmp/perovagents_cfg_test.yaml"
        with open(path, "w") as f:
            f.write("cells:\n  max_attempts: 5\n")
        cfg = load_config(path)
        self.assertEqual(cfg["cells"]["max_attempts"], 5)
        self.assertEqual(cfg["cells"]["timeout_s"], 300.0)  # default kept
        os.remove(path)

    def test_shipped_config_yaml_loads(self):
        from perovagents.control.config import default_path
        cfg = load_config(default_path())
        self.assertEqual(cfg["cells"]["max_attempts"], 3)


if __name__ == "__main__":
    unittest.main()
