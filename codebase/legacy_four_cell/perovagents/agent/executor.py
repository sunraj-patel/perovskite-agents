"""Plan executor.

Turns a `Plan` (list of tool calls) into executed `StepResult`s. For every
step:

  * resolves the tool via the registry (unknown tool -> failed step, agent
    continues or aborts by policy)
  * runs the tool with timeouts
  * runs the sanity checker over the result and records warnings
  * writes a provenance record

By default the executor **continues** after a failed step and reports the
failure, since partial evidence is often still useful; the integrator can set
`stop_on_error=True` for strict SQL-like behavior.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..core.registry import ToolRegistry
from ..core.sanity import SanityChecker, SanityReport, default_checker
from ..core.provenance import ProvenanceStore, default_store
from ..core.claims import claim_type_for
from .models import Plan

log = logging.getLogger(__name__)


@dataclass
class StepResult:
    tool: str
    action: str
    params: dict
    result: Any = None
    success: bool = True
    error: str = ""
    elapsed_seconds: float = 0.0
    provenance: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    purpose: str = ""
    expected_outputs: List[str] = field(default_factory=list)
    required_for_final_answer: bool = True
    step_index: int = 0
    claim_type: str = "computation"

    def to_dict(self, include_result=False):
        d = {"step_index": self.step_index, "tool": self.tool,
             "action": self.action, "params": self.params, "success": self.success,
             "error": self.error, "elapsed_seconds": self.elapsed_seconds,
             "provenance": self.provenance, "warnings": self.warnings,
             "validation": self.validation,
             "purpose": self.purpose,
             "expected_outputs": list(self.expected_outputs),
             "required_for_final_answer": self.required_for_final_answer,
             "claim_type": self.claim_type,
             "result_summary": summarize(self.result)}
        if include_result and isinstance(self.result, dict):
            d["result"] = self.result
        return d


@dataclass
class PlanResult:
    question: str
    planner: str
    steps: List[StepResult] = field(default_factory=list)
    trace_id: str = ""
    total_elapsed_seconds: float = 0.0

    def to_dict(self, include_result=False):
        return {"question": self.question, "planner": self.planner,
                "trace_id": self.trace_id,
                "total_elapsed_seconds": round(self.total_elapsed_seconds, 3),
                "n_ok": sum(1 for s in self.steps if s.success),
                "n_fail": sum(1 for s in self.steps if not s.success),
                "steps": [s.to_dict(include_result=include_result) for s in self.steps]}

    def ok(self) -> bool:
        return bool(self.steps) and all(s.success for s in self.steps)


def summarize(result: Any) -> str:
    """Collapse a tool result into a one-line summary for traces."""
    if result is None:
        return "null"
    if isinstance(result, dict):
        rt = result.get("result_type", "")
        if rt == "count":
            return f"count={result.get('value', 0)}"
        if rt == "scalar":
            return f"{result.get('column','agg')}={result.get('value', None)}"
        if rt == "table":
            return f"table({len(result.get('data', []))} rows)"
        if rt == "literature_passages":
            return f"passages({len(result.get('hits', []))})"
        if rt == "prediction":
            p = result.get("predicted_pce")
            return f"predicted PCE={p}%" if p is not None else "prediction"
        if rt in ("explanation",):
            return f"shap {result.get('scope','')} top={len(result.get('features', []))}"
        if rt == "what_if":
            return (f"what_if {result.get('parameter')}: "
                    f"{len(result.get('data', []))} values "
                    f"PCE={result.get('data', [{}])[0] and result['data'][0].get('predicted_pce')}%..")
        if rt == "optimization":
            return (f"optimize: {len(result.get('recommendations', []))} "
                    f"recommendations")
        if rt == "material":
            return f"{result.get('material','')}:{result.get('property','')}={result.get('value', None)}"
        if result.get("status") == "error":
            return f"error: {str(result.get('error',''))[:80]}"
        return str(result)[:120]
    return str(result)[:120]


def render_for_synthesis(sd: dict, max_chars: int = 4500) -> str:
    """Rich, LLM-facing rendering of ONE executed step for the synthesis
    prompt. Unlike `summarize` (a one-line trace summary), this includes the
    actual output values: feature importances/directions, predicted PCEs,
    what-if rows, optimization recommendations, table head, etc."""
    tool = sd.get("tool", "?")
    action = sd.get("action", "?")
    params = sd.get("params") or {}
    res = sd.get("result")
    head = f"STEP {tool}.{action}({params})"
    if not sd.get("success"):
        return f"{head} -> ERROR {sd.get('error')}"
    if not isinstance(res, dict):
        return f"{head} -> {res}"
    rt = res.get("result_type", "")
    body = ""
    if rt == "explanation":
        feats = res.get("features") or []
        rows = [f"  {f.get('name')}: importance={f.get('importance')} "
                f"direction={f.get('direction')} shap={f.get('shap')}"
                for f in feats]
        flags = []
        if res.get("n_design_samples"):
            flags.append(f"n_design_samples={res['n_design_samples']}")
        ctr = [n for n, c in (res.get("controllable") or {}).items() if c]
        if ctr:
            flags.append("controllable=" + ", ".join(ctr))
        body = f" {res.get('method')} {res.get('scope')} on model " \
               f"{res.get('architecture')}" + (" (" + "; ".join(flags) + ")" if flags else "") \
               + " features (importance direction shap):\n" \
               + ("\n".join(rows) if rows else "  (no features)")
    elif rt == "prediction":
        body = (f" predicted PCE={res.get('predicted_pce')}% "
                f"95% interval={res.get('pce_95_interval')} "
                f"spread={res.get('interval_spread_percent')}ppt "
                f"architecture={res.get('architecture')} n_bags={res.get('n_bags')}")
    elif rt == "what_if":
        rows = res.get("data") or []
        body = (f" what_if on {res.get('parameter')} (valid_range="
                f"{res.get('valid_range')}, clamped={res.get('clamped_to_valid_range')})"
                f" over {res.get('architecture')}:\n" +
                "\n".join(f"  value={r.get('parameter_value')} -> "
                          f"PCE={r.get('predicted_pce')}% "
                          f"[{r.get('interval_lo')}, {r.get('interval_hi')}]"
                          for r in rows))
    elif rt == "optimization":
        recs = res.get("recommendations") or []
        if res.get("optimization_status") == "none_found":
            body = (f" optimize: HONEST EMPTY result; kept parameters "
                    f"{list(res.get('kept_parameters', []) or [])} "
                    f"predicted PCE={res.get('prediction')} interval="
                    f"{res.get('interval')} — no change improved PCE.")
        else:
            lines = []
            for r in recs:
                lines.append(
                    f"  {r.get('parameter')}: {r.get('direction')} to "
                    f"{r.get('recommended_value')} ({r.get('kind')}, "
                    f"{r.get('category')}) -> PCE {r.get('best_predicted_pce')}% "
                    f"(base {r.get('base_predicted_pce')}%, "
                    f"delta +{r.get('delta_predicted_pce')}ppt)")
            body = f"{len(recs)} optimization recommendations:\n" + \
                   ("\n".join(lines) or "  (none)")
    elif rt == "table":
        data = res.get("data") or []
        cols = res.get("columns") or (list(data[0].keys()) if data else [])
        head_rows = "\n".join(
            "  " + " | ".join(str(r.get(c, "")) for c in cols) for r in data[:15])
        body = f" table({len(data)} rows, columns={cols[:12]})" + \
               (f":\n{head_rows}" if head_rows else "")
    elif rt == "literature_passages":
        hits = res.get("hits") or []
        lines = [f"  [{h.get('doi', '?')}] {(str(h.get('text') or h.get('snippet') or ''))[:220]}"
                 for h in hits[:5]]
        body = f" {len(hits)} passages from {res.get('n_results')} sources:\n" + \
               ("\n".join(lines) or "  (none)")
    elif rt in ("count", "scalar", "material"):
        body = f" {res.get('value', res)}"
    else:
        body = f" {str(res)[:max_chars]}"
    return (head + body)[: max_chars + len(head)]


class Executor:
    def __init__(self, registry: ToolRegistry, checker: SanityChecker = None,
                 store: ProvenanceStore = None, timeout: int = 120,
                 stop_on_error: bool = False,
                 structured_actions: bool = True):
        self.registry = registry
        self.checker = checker or default_checker()
        self.prov = store or default_store()
        self.timeout = timeout
        self.stop_on_error = stop_on_error
        self.structured_actions = structured_actions

    def execute_plan(self, question: str, plan: Plan, trace_id: str = "",
                     on_step_start=None, on_step_done=None) -> PlanResult:
        """Run every step. Optional callbacks let UIs render live progress:

          on_step_start(i, plan_step) -> None   # before step i runs
          on_step_done(i, StepResult) -> None   # after step i finishes
        """
        trace_id = trace_id or self.prov.new_trace_id()
        result = PlanResult(question=question, planner=plan.planner, trace_id=trace_id)
        t0 = time.time()
        for i, step in enumerate(plan.steps):
            if on_step_start is not None:
                on_step_start(i, step)
            sr = self._execute_step(step, i, trace_id)
            if on_step_done is not None:
                on_step_done(i, sr)
            result.steps.append(sr)
            log.info("[%s] step %d/%d %s.%s %s in %.2fs: %s",
                     trace_id, i + 1, len(plan.steps), sr.tool, sr.action,
                     "ok" if sr.success else "ERROR", sr.elapsed_seconds,
                     summarize(sr.result)[:80])
            if not sr.success and self.stop_on_error:
                log.warning("[%s] stop_on_error: aborting after step %d",
                            trace_id, i)
                break
        result.total_elapsed_seconds = time.time() - t0
        return result

    def _execute_step(self, step, i: int, trace_id: str) -> StepResult:
        tool = self.registry.get(step.tool)
        t0 = time.time()
        if tool is None:
            sr = StepResult(tool=step.tool, action=step.action, params=step.params,
                            success=False, error=f"unknown tool '{step.tool}'",
                            purpose=step.purpose, expected_outputs=step.expected_outputs,
                            required_for_final_answer=step.required_for_final_answer,
                            step_index=i)
            self.prov.record(trace_id, i, step.tool, step.params, "error", 0.0, None, [])
            return sr
        # --- legacy raw-tool-calling path (ablation-specific) ---------------
        if not self.structured_actions:
            params = dict(step.params) if step.params else {}
            try:
                er = tool.execute(params)
            except Exception as exc:
                er = type("ER", (), {"status": "error", "error": f"{exc}",
                                     "result": None,
                                     "validation": SanityReport()})()
            elapsed = time.time() - t0
            warnings = list(getattr(getattr(er, "validation", None),
                                    "warnings", []) or [])
            src = self._sources_from(getattr(er, "result", None))
            self.prov.record(trace_id, i, step.tool, params, er.status, elapsed,
                             getattr(er, "result", None), src)
            return StepResult(
                tool=step.tool, action=step.action,
                params=params, result=getattr(er, "result", None),
                success=er.status == "success", error=getattr(er, "error", ""),
                elapsed_seconds=elapsed, provenance=(getattr(er, "provenance", {}) or {}),
                warnings=warnings,
                validation=getattr(getattr(er, "validation", None), "to_dict", lambda: {})()
                if isinstance(getattr(er, "validation", None), SanityReport) else {},
                purpose=step.purpose, expected_outputs=step.expected_outputs,
                required_for_final_answer=step.required_for_final_answer,
                claim_type="computation", step_index=i)

        params = dict(step.params) if step.params else {}
        # action is first-class (never smothered back into a 'flag' param), but
        # legacy tools dispatch on params['action'], so mirror it for them
        if params.get("action") is None:
            params["action"] = step.action
        er = tool.execute(params, action=step.action, trace_id=trace_id)
        elapsed = time.time() - t0
        # layer-2/3 validation surfaced on the step
        warnings = list(er.validation.warnings) if er.status == "success" else []

        # layer-3 structural/physics errors become step warnings too
        warnings.extend(getattr(er.validation, "errors", []) or [])

        sources = self._sources_from(er.result)
        self.prov.record(trace_id, i, step.tool, params, er.status, elapsed,
                         er.result, sources)
        return StepResult(
            tool=step.tool, action=step.action, params=params,
            result=er.result, success=er.status == "success", error=er.error,
            elapsed_seconds=elapsed, provenance=(er.provenance or {}),
            warnings=warnings,
            validation=er.validation.to_dict(),
            purpose=step.purpose, expected_outputs=step.expected_outputs,
            required_for_final_answer=step.required_for_final_answer,
            claim_type=claim_type_for(step.tool, step.action),
            step_index=i)

    @staticmethod
    def _sources_from(result: Any) -> List[str]:
        sources = []
        if isinstance(result, dict):
            p = result.get("provenance") or {}
            for key in ("dois", "source", "source_file", "corpus", "model"):
                v = p.get(key)
                if isinstance(v, list):
                    sources.extend([str(s) for s in v])
                elif v:
                    sources.append(str(v))
            if result.get("hits"):
                for h in result["hits"]:
                    if isinstance(h, dict) and h.get("doi"):
                        sources.append(str(h["doi"]))
        return sources


# legacy alias kept for import compatibility
SanityReportShim = SanityReport