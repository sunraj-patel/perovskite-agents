"""Synthesis v2: final answer from validated results only (S8, spec §7).

Deterministic (no LLM narration — auditability first; DECISIONS.md D14):
- every quantitative statement carries its source (SQL text, paper id +
  locator, model id, sweep name);
- failed cells are named with their reason; gaps are never papered over;
- truncate notes and empty-result notes become warnings, not silent drops.
"""
from typing import Any, Dict, List, Tuple

from ..cells.schemas import FailedResult, RunState, ValidatedResult


def _cite(task_id: str, source: str) -> str:
    return f"{task_id}:{source}"


def _render_pdb(task_id: str, payload: Dict) -> Tuple[List[str], List[str]]:
    lines, cites = [], []
    rows = payload.get("rows", [])
    sql = (payload.get("provenance", {}) or {}).get("sql", "")
    if not rows:
        return [f"[{task_id}] the database returned no rows ({sql})."], \
            [_cite(task_id, "sql")]
    first = rows[0]
    if set(first) == {"value"} or (len(first) == 1 and "value" in first):
        lines.append(f"[{task_id}] result: {first['value']} ({sql}).")
    else:
        lines.append(f"[{task_id}] {len(rows)} record(s) "
                     f"({', '.join(map(str, first.keys()))}) ({sql}).")
        for r in rows[:5]:
            lines.append("  - " + ", ".join(f"{k}={v}" for k, v in r.items()))
    if payload.get("truncated"):
        lines.append(f"[{task_id}] note: rows capped at "
                     f"{payload.get('row_limit')} (truncated).")
    cites.append(_cite(task_id, "sql"))
    return lines, cites


def _render_literature(task_id: str, payload: Dict) -> Tuple[List[str],
                                                             List[str]]:
    lines, cites = [], []
    for claim in payload.get("claims", [])[:8]:
        for cit in claim.get("citations", [])[:2]:
            loc = f"{cit.get('paper_id', '?')}#{cit.get('locator', '?')}"
            lines.append(f"[{task_id}] per {loc}: "
                         f"{str(claim.get('text', ''))[:220]}")
            cites.append(_cite(task_id, loc))
    if not lines:
        lines.append(f"[{task_id}] no supported claims retrieved.")
        cites.append(_cite(task_id, "literature-search"))
    return lines, cites


def _render_ml(task_id: str, payload: Dict) -> Tuple[List[str], List[str]]:
    lines, cites = [], []
    prov = payload.get("provenance", {}) or {}
    model = prov.get("model", "unknown model")
    if "predicted_pce" in payload:
        iv = payload.get("pce_95_interval", [])
        lines.append(f"[{task_id}] model predicts PCE "
                     f"{payload['predicted_pce']}%"
                     + (f" (95% interval {iv}, {model})" if iv
                        else f" ({model})") + ".")
        cites.append(_cite(task_id, model))
    for rec in (payload.get("recommendations", []) or [])[:3]:
        lines.append(f"[{task_id}] candidate: {rec} ({model}).")
        cites.append(_cite(task_id, model))
    if not lines:
        lines.append(f"[{task_id}] ML returned no prediction ({model}).")
    cites.append(_cite(task_id, model))  # every ML section cites its model
    return lines, cites


def _render_scaps(task_id: str, payload: Dict) -> Tuple[List[str], List[str]]:
    lines, cites = [], []
    data = payload.get("data", [])
    param = payload.get("parameter", "parameter")
    if data:
        first, last = data[0], data[-1]
        lines.append(f"[{task_id}] SCAPS sweep over {param}: PCE "
                     f"{first.get('PCE')}% -> {last.get('PCE')}% "
                     f"({len(data)} points, physics sensitivity, not "
                     f"experimental reproduction).")
        cites.append(_cite(task_id, f"scaps:{param}"))
    else:
        lines.append(f"[{task_id}] SCAPS returned no sweep points.")
        cites.append(_cite(task_id, f"scaps:{param}"))
    return lines, cites


def _render_utility(task_id: str, payload: Dict) -> Tuple[List[str],
                                                          List[str]]:
    return [f"[{task_id}] computed: {payload.get('result')}."], \
        [_cite(task_id, "calculation")]


_RENDER = {"pdb_sql": _render_pdb, "literature": _render_literature,
           "ml": _render_ml, "scaps": _render_scaps,
           "utility": _render_utility}


def synthesize(question: str, state: RunState) -> Dict[str, Any]:
    """Build the final answer dict from run state (validated-only)."""
    lines = [f"Question: {question}", ""]
    cites: List[str] = []
    warnings: List[str] = []
    for tid, res in state.results.items():
        if not isinstance(res, ValidatedResult):
            continue
        part, cite = _RENDER.get(res.cell, _render_utility)(
            tid, res.payload)
        lines.extend(part)
        cites.extend(cite)
    failed = {tid: state.errors.get(tid, "")
              for tid, r in state.results.items()
              if isinstance(r, FailedResult)}
    skipped = sorted(tid for tid, s in state.task_status.items()
                     if s == "skipped")
    if failed:
        lines.append("")
        lines.append("Cells that failed (not used in this answer):")
        for tid, reason in failed.items():
            lines.append(f"  - {tid}: {reason[:220]}")
    if skipped:
        lines.append(f"Skipped (dependency failed): {', '.join(skipped)}.")
    if not cites:
        lines.append("")
        lines.append("No validated results; no claims are made.")
        warnings.append("empty answer: nothing validated")
    sources = sorted(set(cites))
    return {"answer": "\n".join(lines), "sources": sources,
            "warnings": warnings, "failed": failed, "skipped": skipped}
