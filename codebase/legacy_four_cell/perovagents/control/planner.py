"""Planner v2: cell-aware planning (spec section 2.1, S6).

Output is a `cells.schemas.Plan` of `Task{cell, request, params,
depends_on}` — never bare tool calls. Two backends:

- rules (default, offline): keyword task analysis (`analyze_task` with
  `use_llm=False`), route via `cells_for`, first cell only for single-cell
  types (subset selection — never all four by default), fan-out for
  `comparison` / `multi_tool_investigation`. Params come from the existing
  deterministic helpers (`RulePlanner._ml_action/_infer_arch/_whatif_spec`,
  material extraction). The rules backend emits no `depends_on` (independent
  tasks; DECISIONS.md D12) — dependency chains are built by the LLM backend
  or by hand (S7 tests).
- llm: `prompts/planner.md` + question -> JSON task list. Unknown cells,
  unknown dependency ids, or unparsable output fall back to rules.

Pure planning in both backends: no domain tools are called here.
"""
from typing import Any, Dict, List, Optional

from ..agent.rules import RulePlanner, extract_material
from ..agent.task import analyze_task, extract_materials
from ..cells.schemas import CELL_AND_UTILITY, Plan, Task, cells_for
from ..llm import extract_json
from ..prompts import load as load_prompt

_PROMPT = load_prompt("planner", fallback="Plan cell tasks as JSON.")

_SWEEP_KEYWORDS = [
    (("band gap", "bandgap", "eg ", "1.5", "1.6", "1.7"), "pvk_Eg"),
    (("thickness", "thick", " nm"), "pvk_d"),
    (("mobility", "mun", "mup"), "pvk_mun"),
    (("affinity", "HTL", "htl"), "htl_chi"),
]

_counter = {"n": 0}


def _tid(prefix: str = "t") -> str:
    _counter["n"] += 1
    return f"{prefix}{_counter['n']}"


def _pdb_params(question: str, spec, rules: RulePlanner) -> Dict[str, Any]:
    """Structured params for the deterministic SQL builder (S5)."""
    mats = extract_materials(question)
    params: Dict[str, Any] = {"limit": 100}
    if rules._wants_count(question):
        params.update({
            "aggregation": {"func": "count", "column": "JV_default_PCE"},
            "expected_cols": ["value"]})
    else:
        params.update({
            "columns": ["Cell_architecture", "ETL_stack_sequence",
                        "HTL_stack_sequence", "JV_default_PCE",
                        "Ref_DOI_number"],
            "aggregation": {"func": "mean", "column": "JV_default_PCE"},
            "expected_cols": ["value"]})
    if mats:
        params["filters"] = [{"column": "Cell_stack_sequence",
                              "op": "contains", "value": m} for m in mats]
    return params


def _ml_params(question: str, spec, rules: RulePlanner) -> Dict[str, Any]:
    action = rules._ml_action(question.lower())
    # Align with the analyzed task type: bare "what if" phrasing misses the
    # ML_ACTION_RULES patterns, so intervention tasks default correctly.
    if action == "predict" and spec.task_type == "sensitivity_analysis":
        action = "what_if"
    elif action == "predict" and spec.task_type == "optimization":
        action = "optimize"
    params: Dict[str, Any] = {"action": action,
                              "architecture": rules._infer_arch(
                                  question.lower()),
                              "description": question[:500]}
    if action == "what_if":
        param, values = rules._whatif_spec(question.lower())
        params["parameter"] = param
        params["values"] = values
    return params


def _scaps_params(question: str) -> Dict[str, Any]:
    ql = question.lower()
    for keywords, sweep in _SWEEP_KEYWORDS:
        if any(k in ql for k in keywords):
            return {"sweep_name": sweep}
    return {"sweep_name": "pvk_Eg"}


def plan_rules(question: str) -> Plan:
    """Deterministic cell-task planning (offline-safe)."""
    rules = RulePlanner()
    spec = analyze_task(question, llm=None, use_llm=False)
    cells = cells_for(spec.task_type)
    tasks: List[Task] = []
    if spec.task_type in ("comparison", "multi_tool_investigation"):
        wanted = [c for c in
                  ("pdb_sql", "literature", "ml", "scaps") if c in cells]
    else:
        wanted = cells[:1]  # subset: first (preferred) cell only
    for cell in wanted:
        if cell == "pdb_sql":
            params = _pdb_params(question, spec, rules)
        elif cell == "ml":
            params = _ml_params(question, spec, rules)
        elif cell == "literature":
            params = {"query": question[:300], "k": 5}
        elif cell == "scaps":
            params = _scaps_params(question)
        else:  # utility
            params = {"action": "eval",
                      "expression": extract_material(question) or ""}
        tasks.append(Task(id=_tid(), cell=cell,
                          request=f"{spec.task_type}: {question[:200]}",
                          params=params))
    if not tasks:  # unknown task type: factual fallback, like RulePlanner
        tasks.append(Task(id=_tid(), cell="pdb_sql",
                          request=f"database_query: {question[:200]}",
                          params=_pdb_params(question, spec, rules)))
    return Plan(query=question, tasks=tasks, planner="rules")


def plan_llm(question: str, llm, task_spec=None) -> Plan:
    """LLM cell-task planning with rules fallback."""
    prompt = (_PROMPT + "\n\nQUESTION:\n" + question)[:6000]
    try:
        text = llm.chat(prompt, temperature=0.0)
        parsed = extract_json(text)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        parsed = parsed.get("tasks", [])
    if not isinstance(parsed, list) or not parsed:
        return plan_rules(question)
    tasks: List[Task] = []
    ids = set()
    for i, raw in enumerate(parsed):
        if not isinstance(raw, dict):
            continue
        cell = str(raw.get("cell", ""))
        if cell not in CELL_AND_UTILITY:
            continue
        tid = str(raw.get("id", "") or _tid())
        tasks.append(Task(
            id=tid, cell=cell,
            request=str(raw.get("request", question[:200])),
            params=dict(raw.get("params", {}) or {}),
            depends_on=[d for d in (raw.get("depends_on", []) or [])
                        if isinstance(d, str)]))
        ids.add(tid)
    # dependency ids must reference known tasks, else fall back to rules
    for t in tasks:
        if any(d not in ids for d in t.depends_on):
            return plan_rules(question)
    if not tasks:
        return plan_rules(question)
    return Plan(query=question, tasks=tasks, planner="llm")


def plan(question: str, llm=None, pipeline: str = "rules") -> Plan:
    """Plan entry: LLM backend when requested and available, else rules."""
    if pipeline == "llm" and llm is not None and llm.available:
        try:
            return plan_llm(question, llm)
        except Exception:
            pass
    return plan_rules(question)
