"""Cell (a): PDB / SQL (S5).

Agent: uses planner/LLM-authored SQL from `task.params["sql"]` when present,
else a small deterministic builder from structured params
(columns/filters/aggregation/group_by/limit). LLM authoring plugs into the
agent slot at S6; the deterministic path keeps the cell testable offline.
Tool: `SQLTool` (S4, read-only). Gate: `verify_sql` (S2).
"""
from typing import Any, Dict, List

from ..tools.sql_tool import SQLTool
from ..verify.sql_checks import verify_sql
from .base import DomainCell
from .llm_agent import llm_tool_call
from .schemas import Task, VerificationReport

_OPS = {"eq": "=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<=",
        "contains": "LIKE"}
_AGG = {"mean": "AVG", "avg": "AVG", "min": "MIN", "max": "MAX",
        "count": "COUNT", "sum": "SUM"}


def build_sql(params: Dict[str, Any]) -> str:
    """Deterministic SELECT builder for structured params (offline path)."""
    cols = params.get("columns") or ["JV_default_PCE"]
    agg = params.get("aggregation")
    group_by = params.get("group_by")
    if agg:
        func = _AGG.get(str(agg.get("func", "mean")).lower(), "AVG")
        target = agg.get("column", "JV_default_PCE")
        sel = [f"{func}({target}) AS value"]
        if group_by:
            sel.insert(0, group_by)
    else:
        sel = list(cols)
    where = []
    for f in params.get("filters") or []:
        op = _OPS.get(str(f.get("op", "contains")).lower(), "LIKE")
        col, val = f["column"], f["value"]
        if op == "LIKE":
            where.append(f"{col} LIKE '%{val}%'")
        elif isinstance(val, (int, float)):
            where.append(f"{col} {op} {val}")
        else:
            safe = str(val).replace("'", "''")
            where.append(f"{col} {op} '{safe}'")
    sql = f"SELECT {', '.join(sel)} FROM devices"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if group_by and agg:
        sql += f" GROUP BY {group_by}"
    sql += f" LIMIT {int(params.get('limit', 100))}"
    return sql


class PDBSQLCell(DomainCell):
    """PDB/SQL domain cell."""
    cell = "pdb_sql"

    def __init__(self, db_path=None, max_attempts: int = 3,
                 row_limit: int = 1000, timeout_s: float = 10.0,
                 llm=None):
        self.tool = SQLTool(db_path=db_path, row_limit=row_limit,
                            timeout_s=timeout_s)
        self.llm = llm
        super().__init__(agent_fn=self._agent, tool_fn=self._tool,
                         gate_fn=self._gate, max_attempts=max_attempts)

    # -- slots ---------------------------------------------------------
    def _agent(self, task: Task, history: List) -> Dict[str, Any]:
        fallback = {"sql": task.params.get("sql") or build_sql(task.params),
                    "expected_cols": task.params.get("expected_cols")}
        if history and self.llm is not None:
            return llm_tool_call(self.llm, "pdb_agent", task, history,
                                 fallback)
        return fallback

    def _tool(self, call: Dict[str, Any]) -> Dict[str, Any]:
        return self.tool.call(call)

    def _gate(self, raw: Dict[str, Any],
              call: Dict[str, Any]) -> VerificationReport:
        return verify_sql(call.get("sql", ""), raw, self.tool.schema,
                          expected_cols=call.get("expected_cols"))
