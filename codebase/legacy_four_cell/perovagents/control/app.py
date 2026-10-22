"""New entry point: planner v2 -> executor v2 -> synthesis v2 (S9).

`answer_cells()` is the cell-architecture counterpart of `Agent.answer`.
The old entry point (`python -m perovagents`, `Agent.answer`, frontend,
benchmarks) is untouched and keeps working; switch over only after the
section 9 acceptance suite passes.
"""
from typing import Any, Dict, Optional

from ..cells.literature import LiteratureCell
from ..cells.ml import MLCell
from ..cells.pdb_sql import PDBSQLCell
from ..cells.scaps import SCAPSCell
from .. import config as _config
from . import planner as planner_v2
from .config import load_config
from .executor import ExecutorV2
from .synthesis import synthesize


def build_cells(cfg: Dict[str, Any], llm=None) -> Dict[str, Any]:
    """Instantiate the four domain cells from config. Passing `llm`
    enables LLM revision on the "No" branch; without it agents replay
    their deterministic call."""
    db = cfg["db"]
    return {
        "pdb_sql": PDBSQLCell(
            db_path=db["mirror"],
            max_attempts=cfg["cells"]["max_attempts"],
            row_limit=db["row_limit"],
            timeout_s=db["query_timeout_s"],
            llm=llm),
        "literature": LiteratureCell(
            max_attempts=cfg["cells"]["max_attempts"], llm=llm),
        "ml": MLCell(max_attempts=cfg["cells"]["max_attempts"],
                     metrics_path=cfg["ml"]["metrics"], llm=llm),
        "scaps": SCAPSCell(max_attempts=cfg["cells"]["max_attempts"],
                           llm=llm),
    }


def _utility_runner(task) -> Dict[str, Any]:
    """Executor-level helpers (calculator today; DECISIONS.md D15)."""
    from ..core.calculator import CalculatorTool
    name = (task.params or {}).get("tool", "calculator")
    if name != "calculator":
        raise ValueError(f"unsupported utility {name!r}")
    out = CalculatorTool().call(dict(task.params or {}))
    if "error" in out:
        raise ValueError(out["error"])
    return out


def answer_cells(question: str, pipeline: str = "rules", llm=None,
                 cfg: Optional[Dict[str, Any]] = None,
                 trace_dir: Optional[str] = None) -> Dict[str, Any]:
    """Full cell-architecture run: plan -> dispatch -> synthesize."""
    from ..core.flowlog import FlowLog
    from ..core.provenance import default_store

    cfg = cfg or load_config()
    store = default_store()
    trace_id = store.new_trace_id()
    flow = FlowLog(run_id=trace_id)
    plan = planner_v2.plan(question, llm=llm, pipeline=pipeline)
    flow.query(question, pipeline=pipeline,
               mode="cells v2")
    flow.plan(0, plan.planner, plan.to_dict())
    ex = ExecutorV2(cells=build_cells(cfg, llm=llm),
                    utility_runner=_utility_runner,
                    timeout_s=cfg["cells"]["timeout_s"],
                    trace_dir=trace_dir or cfg["trace"]["dir"],
                    seed=42)
    out = ex.run(plan)
    order = {t.id: i for i, t in enumerate(out["state"].plan.tasks)}
    for tid, res in out["state"].results.items():  # per-task outcomes
        import json as _json
        try:
            payload = _json.loads(_json.dumps(res.to_dict(), default=str))
        except Exception:
            payload = {"_unserializable": str(res)}
        flow.step(0, order.get(tid, 0), {
            "task_id": tid, "cell": res.cell,
            "request": next((t.request for t in out["state"].plan.tasks
                             if t.id == tid), ""),
            "params": next((t.params for t in out["state"].plan.tasks
                            if t.id == tid), {}),
            "status": out["summary"]["status"].get(tid, "?"),
            "result": payload})
    for t in out["state"].plan.tasks:
        if t.id not in out["state"].results:
            flow.step(0, order.get(t.id, 0), {
                "task_id": t.id, "cell": t.cell, "request": t.request,
                "params": t.params, "status": "skipped"})
    synthesis = synthesize(question, out["state"])
    flow.answer(synthesis["answer"], synthesis["sources"],
                synthesis["warnings"], contract=None)
    flow.done(total_elapsed_seconds=None, n_ok=None, n_fail=None,
              verdict=None, summary=dict(out["summary"]))
    import os as _os
    flow_log = _os.path.relpath(flow.path, _config.RESULTS_DIR)
    return {"question": question, "plan": plan.to_dict(),
            "summary": out["summary"], "synthesis": synthesis,
            "answer": synthesis["answer"], "sources": synthesis["sources"],
            "warnings": synthesis["warnings"],
            "failed": synthesis["failed"],
            "trace": out["summary"]["trace"],
            "flow_log": flow_log}
