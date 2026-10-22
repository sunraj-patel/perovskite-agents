"""Agent orchestrator.

The high-level entry point of PerovAgents. Given a question it runs the
full agentic loop:

    task analyzer        -> TaskSpec (what evidence is required)
    planner              -> Plan of tool+action steps (semantic, validated)
    plan validator       -> registry schema gate (reject, don't coerce)
    executor             -> deterministic tool invocation
    output validator     -> verifier: do results carry the required outputs?
        SUFFICIENT   -> synthesis
        RECOVERABLE  -> diagnose missing requirement, re-plan (<= MAX_REPLANS)
        UNRESOLVED   -> unresolved response with the evidence we do have
    synthesizer          -> answer constrained by the answer contract

The same loop (without LLM planning) runs in offline/rule mode, so the
framework always completes.
"""
import hashlib
import json
import logging
import os
from typing import Optional

from .. import config  # noqa: F401
from ..core.flowlog import FlowLog
from ..core.provenance import ProvenanceStore, default_store
from ..core.registry import ToolRegistry
from ..core.sanity import SanityChecker, default_checker
from ..core.claims import CLAIM_LANGUAGE, claim_type_for
from ..llm import LLM
from .executor import Executor, PlanResult, render_for_synthesis, summarize
from .planner import LLMPlanner, validate_plan
from .rules import RulePlanner
from .task import TaskSpec, analyze_task
from .verification import Verdict, verify_task

log = logging.getLogger(__name__)

MAX_REPLANS = 2

# Ablation modes A-G (layer-by-layer build-up of the MRS stack):
#   A raw            raw LLM tool-calling: no analyzer, no action schemas, no
#                    schema gate, no output verifier, no replanning
#   B schemas        + first-class actions with per-action input schemas
#   C analyzer       + task analyzer (TaskSpec) feeds the planner prompt
#   D gate           + pre-execution schema/rule validation gate (reject, do
#                    not coerce)
#   E verifier       + output-contract verifier (verdict computed, no replan)
#   F replan         + iterative re-planning until SUFFICIENT (<= MAX_REPLANS)
#   G full (=full)   + task-analyzer->planner contract + claim types + ontology
MODES = {
    "A": {"use_analyzer": False, "structured_actions": False,
          "schema_gate": False, "use_verifier": False,
          "use_replan": False, "enforce_task": False},
    "B": {"use_analyzer": False, "structured_actions": True,
          "schema_gate": False, "use_verifier": False,
          "use_replan": False, "enforce_task": False},
    "C": {"use_analyzer": True, "structured_actions": True,
          "schema_gate": False, "use_verifier": False,
          "use_replan": False, "enforce_task": False},
    "D": {"use_analyzer": True, "structured_actions": True,
          "schema_gate": True, "use_verifier": False,
          "use_replan": False, "enforce_task": False},
    "E": {"use_analyzer": True, "structured_actions": True,
          "schema_gate": True, "use_verifier": True,
          "use_replan": False, "enforce_task": False},
    "F": {"use_analyzer": True, "structured_actions": True,
          "schema_gate": True, "use_verifier": True,
          "use_replan": True, "enforce_task": False},
    "G": {"use_analyzer": True, "structured_actions": True,
          "schema_gate": True, "use_verifier": True,
          "use_replan": True, "enforce_task": True},
    "full": {"use_analyzer": True, "structured_actions": True,
             "schema_gate": True, "use_verifier": True,
             "use_replan": True, "enforce_task": True},
}

SYNTHESIS_PROMPT = """You are a perovskite solar cell research assistant.

Answer the user's question with scientific discipline, using ONLY the tool
results below. Report the actual numbers returned by the tools (never claim
results are missing when values are present); if evidence is genuinely
insufficient, say so and state what would settle it.

You must honor the answer contract:

ANSWER CONTRACT
{CONTRACT}

LANGUAGE RULES (non-negotiable):
- Prediction vs explanation vs intervention are different claims.
- SHAP feature importance is NOT causal: say "the model identifies / is
  associated with / contributes to the model prediction", never "changing X
  will increase PCE" unless an intervention result (what_if/optimize) is
  present.
- what_if/optimize results: "under the model, increasing X changes predicted
  PCE from A% to B%".
- Do not recommend experimental changes from feature importance alone.
- Cite the model/DB provenance names when reporting numbers.

CLAIM-TYPE LANGUAGE (use the register matching each result type):
{CLAIM_LANGUAGE}

Task specification:
{SPEC}

Question: {QUESTION}

Tool results:
{STEPS}

Verdict from the output validator:
{VERDICT}

Answer:"""

OFFLINE_FALLBACK = True


class Agent:
    def __init__(self, registry: Optional[ToolRegistry] = None,
                 llm: Optional[LLM] = None,
                 pipeline: str = "llm",           # "llm" | "rules"
                 store: Optional[ProvenanceStore] = None,
                 checker: Optional[SanityChecker] = None,
                 max_steps: int = 5,
                 timeout: int = 120,
                 max_replans: int = MAX_REPLANS,
                 mode: str = "full",         # ablation mode A-G | full
                 fast: bool = False,         # deterministic analyzer+synthesis
                 spec_cache: Optional[dict] = None,
                 flow_dir: Optional[str] = None):
        from ..core.registry import default_registry
        flags = MODES.get(mode, MODES["full"])
        self.mode = mode
        self.use_analyzer = flags["use_analyzer"]
        self.structured_actions = flags["structured_actions"]
        self.schema_gate = flags["schema_gate"]
        self.use_verifier = flags["use_verifier"]
        self.use_replan = flags["use_replan"]
        self.enforce_task = flags["enforce_task"]
        self.fast = fast
        self.spec_cache = spec_cache
        self.flow_dir = flow_dir
        self.registry = registry or default_registry()
        self.llm = llm or LLM()
        self.pipeline = pipeline
        self.store = store or default_store()
        self.checker = checker or default_checker()
        self.max_steps = max_steps
        self.max_replans = max_replans if pipeline == "llm" else 0
        self.executor = Executor(self.registry, checker=self.checker,
                                 store=self.store, timeout=timeout,
                                 structured_actions=self.structured_actions)
        self.llm_planner = LLMPlanner(self.llm, self.registry,
                                      max_steps=max_steps,
                                      legacy=not self.structured_actions)
        self.rule_planner = RulePlanner()

    # ---- planning --------------------------------------------------------
    def plan_question(self, question: str, task_spec=None,
                      feedback: Optional[str] = None):
        """Produce a plan: LLM planner when available, otherwise rules."""
        if self.pipeline == "llm" and self.llm.usable:
            try:
                return self.llm_planner.plan(question, self.registry,
                                             task_spec=task_spec,
                                             feedback=feedback)
            except Exception as exc:
                log.warning("LLM planner failed (%s); falling back to rules", exc)
        return self.rule_planner.plan(question)

    # ---- full pipeline ---------------------------------------------------
    def answer(self, question: str, on_plan=None, on_step_start=None,
               on_step_done=None) -> dict:
        """End-to-end answer for one question: analyze -> plan -> gate ->
        execute -> verify, with optional replanning to SUFFICIENT."""
        trace_id = self.store.new_trace_id()
        flow = FlowLog(run_id=trace_id, dir=self.flow_dir)
        flow.query(question, pipeline=self.pipeline, mode=self.mode)
        log.info("[%s] query=%r pipeline=%s mode=%s keys=%d",
                 trace_id, question[:120], self.pipeline, self.mode,
                 self.llm.num_keys)
        if self.use_analyzer:
            if self.spec_cache is not None and question in self.spec_cache:
                task_spec = self.spec_cache[question]
            else:
                task_spec = analyze_task(question, self.llm,
                                         use_llm=not self.fast)
                if self.spec_cache is not None:
                    self.spec_cache[question] = task_spec
        else:
            from .task import TaskSpec as _TS
            task_spec = _TS(objective=question)
        log.info("[%s] task_spec type=%s requires=%s",
                 trace_id, task_spec.task_type,
                 {"prediction": task_spec.requires_prediction,
                  "explanation": task_spec.requires_explanation,
                  "intervention": task_spec.requires_intervention})

        feedback: Optional[str] = None
        plans, plan_iterations, verdicts = [], [], []
        step_results: list = []
        total_elapsed: float = 0.0
        final_plan = None
        final_verdict: Optional[Verdict] = None

        flow.task_spec(task_spec.to_dict())
        replan_budget = self.max_replans if self.use_replan else 0
        for iteration in range(replan_budget + 1):
            plan = self.plan_question(question, task_spec, feedback)
            log.info("[%s] iter=%d plan planner=%s steps=%s note=%s",
                     trace_id, iteration, plan.planner,
                     [f"{s.tool}.{s.action}" for s in plan.steps],
                     (plan.note or "")[:80])
            if plan.note and self.pipeline == "llm":
                # LLM rejected -> deterministic fallback (already noted)
                log.info("[%s] iter=%d planner note: %s",
                         trace_id, iteration, plan.note[:200])
                pass

            # layer 1: schema validation gate (reject, never silently coerce)
            if self.schema_gate:
                if self.enforce_task:
                    ok, verrs = validate_plan(plan, self.registry,
                                              task_spec=task_spec)
                else:
                    ok, verrs = validate_plan(plan, self.registry)
                if not ok:
                    log.warning("[%s] iter=%d gate REJECTED: %s",
                                trace_id, iteration, "; ".join(verrs[:5]))
                    if iteration < replan_budget:
                        feedback = ("Your plan failed validation. Fix: "
                                    + "; ".join(verrs[:5]))
                        continue
                    plan = self.rule_planner.plan(question)
                    ok, verrs = validate_plan(plan, self.registry)
                    log.warning("[%s] iter=%d gate forced rules plan (ok=%s)",
                                trace_id, iteration, ok)

            # avoid deterministic replan loops (rule planner would repeat)
            if iteration > 0 and plans and plan.to_dict() == plans[-1].to_dict():
                log.info("[%s] iter=%d identical plan -> stop replanning",
                         trace_id, iteration)
                break

            if on_plan is not None:
                on_plan(plan.to_dict())
            plans.append(plan)
            flow.plan(iteration, plan.planner, plan.to_dict(), feedback)

            execution = self.run(plan, question, trace_id=trace_id,
                                 on_step_start=on_step_start,
                                 on_step_done=on_step_done,
                                 include_result=True)
            total_elapsed += execution.total_elapsed_seconds
            iter_steps = [s.to_dict(include_result=True) for s in execution.steps]
            for i, sd in enumerate(iter_steps):
                sd["step_index"] = len(step_results)   # global index
                sd["plan_iteration"] = iteration
                flow.step(iteration=iteration, step_index=sd["step_index"],
                          step=sd)
                log.info("[%s] step=%d %s.%s -> %s (%.2fs) %s",
                         trace_id, sd["step_index"], sd["tool"], sd["action"],
                         "ok" if sd["success"] else "ERROR",
                         sd.get("elapsed_seconds", 0.0),
                         sd.get("result_summary", "")[:80])
                step_results.append(sd)

            final_plan = plan
            if self.use_verifier:
                verdict = verify_task(task_spec, plan,
                                      [s.to_dict(include_result=True)
                                       for s in execution.steps])
                plan_iterations.append({"iteration": iteration,
                                        "planner": plan.planner,
                                        "verdict": verdict.to_dict(),
                                        "note": plan.note})
                flow.verdict(iteration, verdict.to_dict())
                log.info("[%s] iter=%d verdict=%s missing=%s",
                         trace_id, iteration, verdict.status,
                         verdict.missing_requirements)
                verdicts.append(verdict)
                final_verdict = verdict

                if verdict.status == "SUFFICIENT":
                    log.info("[%s] verdict SUFFICIENT -> synthesize", trace_id)
                    break
                if verdict.status == "RECOVERABLE" and iteration < replan_budget:
                    log.info("[%s] verdict RECOVERABLE -> replan %d/%d",
                             trace_id, iteration + 1, replan_budget)
                    feedback = (
                        "The executed plan did not provide the required outputs "
                        f"({', '.join(verdict.missing_requirements[:6])}). Re-plan "
                        "using actions whose `outputs` include these tokens "
                        "(e.g. ml.explain_global for feature_name/importance/"
                        "direction, ml.what_if for parameter_value/predicted_pce).")
                    continue
                log.info("[%s] verdict %s -> no replan (budget=%d)",
                         trace_id, verdict.status, replan_budget)
                break  # UNRESOLVED, or out of replan budget
            else:
                break  # no output verifier (ablation E-) : one shot

        final_plan = final_plan or self.rule_planner.plan(question)
        log.info("[%s] synthesizing (steps=%d, llm=%s)",
                 trace_id, len(step_results), self.llm.usable)
        answer_text = self._synthesize(
            question, task_spec, step_results,
            final_verdict or Verdict(status="UNRESOLVED",
                                     diagnostics=["no evidence gathered"]),
            plans)
        provenance = self.store.snapshot(trace_id)
        sources = provenance["sources"]
        warnings = [w for sd in step_results for w in sd.get("warnings", [])]
        claims = self._claims(question, task_spec, plans, step_results,
                              final_verdict if final_verdict else
                              Verdict(status="UNRESOLVED",
                                      diagnostics=["no evidence gathered"]),
                              answer_text)
        flow.answer(answer_text, sources, warnings,
                    self._answer_contract(question, task_spec))
        flow.done(total_elapsed_seconds=round(total_elapsed, 3),
                  n_ok=sum(1 for s in step_results if s["success"]),
                  n_fail=sum(1 for s in step_results if not s["success"]),
                  verdict=(final_verdict.to_dict() if final_verdict else {}),
                  flow_path=self._flow_relpath(flow.path))
        log.info("[%s] done: %d ok / %d fail in %.2fs, verdict=%s, "
                 "flow=%s", trace_id,
                 sum(1 for s in step_results if s["success"]),
                 sum(1 for s in step_results if not s["success"]),
                 total_elapsed,
                 final_verdict.status if final_verdict else "NONE",
                 self._flow_relpath(flow.path))

        return {
            "question": question,
            "pipeline": self.pipeline,
            "trace_id": trace_id,
            "flow_log": self._flow_relpath(flow.path),
            "task_spec": task_spec.to_dict(),
            "plan": plans[0].to_dict() if plans else final_plan.to_dict(),
            "plans": [p.to_dict() for p in plans],
            "execution": {"question": question, "planner":
                          (final_plan.planner if final_plan else "rules"),
                          "trace_id": trace_id,
                          "total_elapsed_seconds": round(total_elapsed, 3),
                          "n_ok": sum(1 for s in step_results if s["success"]),
                          "n_fail": sum(1 for s in step_results if not s["success"]),
                          "steps": step_results},
            "verdict": final_verdict.to_dict() if final_verdict else {},
            "plan_iterations": plan_iterations,
            "answer_contract": self._answer_contract(question, task_spec),
            "answer": answer_text,
            "claims": claims,
            "sources": sources,
            "warnings": warnings,
            "provenance": provenance,
            "provenance_graph": self._provenance_graph(step_results),
        }

    # ---- stepwise control (for UIs) --------------------------------------
    def run(self, plan, question: str, trace_id: str = "",
            on_step_start=None, on_step_done=None,
            include_result: bool = False) -> PlanResult:
        """Execute an existing plan step-by-step."""
        def start(i, step):
            if on_step_start is not None:
                on_step_start(i, step.to_dict())
        def done(i, sr):
            if on_step_done is not None:
                on_step_done(i, sr.to_dict(include_result=include_result))
        return self.executor.execute_plan(
            question, plan, trace_id=trace_id,
            on_step_start=start, on_step_done=done)

    def synthesize(self, question: str, execution: PlanResult) -> str:
        """Produce a narrative answer from an already-executed plan
        (no task spec / verdict available -> a permissive synthesis)."""
        task_spec = TaskSpec(objective=question)
        steps = [s.to_dict(include_result=True) for s in execution.steps]
        return self._synthesize(question, task_spec, steps,
                                Verdict(status="UNRESOLVED",
                                        diagnostics=["synthesize() was called "
                                                     "without a verifier"]), None)

    def _synthesize(self, question, task_spec, step_results, verdict, plans) -> str:
        detail = self._execution_detail(step_results)
        contract = self._answer_contract(question, task_spec)
        if self.llm.usable and not self.fast:
            prompt = SYNTHESIS_PROMPT \
                .replace("{CONTRACT}", json.dumps(contract, indent=1)) \
                .replace("{SPEC}", json.dumps(task_spec.to_dict(), indent=1)) \
                .replace("{CLAIM_LANGUAGE}",
                         self._claim_language(step_results)) \
                .replace("{QUESTION}", question) \
                .replace("{STEPS}", "\n".join(f"- {d}" for d in detail)
                         or "(no tool results)") \
                .replace("{VERDICT}", json.dumps(verdict.to_dict(), indent=1))
            text = self.llm.chat(prompt, temperature=0.0)
            if not text.startswith("[LLM ERROR]") and text.strip():
                return text
        lines = [f"Offline summary for: {question}"]
        lines.append(f"[verdict {verdict.status}]\n")
        for d in detail:
            lines.append(f"  * {d}")
        lines.append("  (LLM not available; no narrative synthesized)")
        return "\n".join(lines)

    @staticmethod
    def _claim_language(step_results) -> str:
        """Claim-type grammar for the claim_types actually produced."""
        used = sorted({s.get("claim_type", claim_type_for(s.get("tool", ""),
                                                          s.get("action", "")))
                       for s in step_results})
        return "\n".join(f"- {CLAIM_LANGUAGE[ct]}" for ct in used
                         if ct in CLAIM_LANGUAGE) or \
            "- Report computed/database numbers exactly; no interpretation beyond the data."

    # ---- answer contract (constrains the synthesizer) ---------------------
    @staticmethod
    def _answer_contract(question: str, task_spec: TaskSpec) -> dict:
        requested = list(task_spec.requested_output)
        if task_spec.requires_prediction and "prediction" not in requested:
            requested.append("prediction")
        if task_spec.requires_explanation:
            for t in ("feature_name", "importance", "direction"):
                if t not in requested:
                    requested.append(t)
        if task_spec.requires_intervention:
            for t in ("parameter_value", "predicted_pce"):
                if t not in requested:
                    requested.append(t)
        forbidden = [
            "claiming SHAP/feature importance is causal evidence",
            "recommending experimental changes from feature importance alone",
            "claiming a what-if result is a measured experimental outcome"] \
            if not task_spec.requires_intervention else [
            "claiming a model sensitivity is a measured experimental outcome"]
        return {"question": question, "task_type": task_spec.task_type,
                "required_claims": list(requested),
                "forbidden_claims": forbidden,
                "evidence_requirements": {
                    "numeric": True, "citation": False,
                    "model_provenance": True}}

    # ---- claim records (point-7 provenance chains) -----------------------
    @staticmethod
    def _claims(question, task_spec, plans, step_results, verdict,
                answer_text) -> list:
        """One record per executed evidence-producing step, tracing the full
        chain question -> task_spec -> plan -> tool/action -> params ->
        result -> validation -> claim, plus the overall verdict record.
        `claim_type` is the epistemic class the result belongs to; the answer
        text must honour CLAIM_LANGUAGE for that class."""
        records, seen = [], set()
        for sd in step_results:
            tool, action = sd.get("tool", ""), sd.get("action", "")
            it = sd.get("plan_iteration", 0)
            plan_label = "rules" if not (0 <= it < len(plans)) \
                else plans[it].planner
            chain = {
                "question": question,
                "task_spec": task_spec.to_dict().get("task_type"),
                "plan": plan_label,
                "tool": tool,
                "action": action,
                "params": sd.get("params") or {},
                "claim_type": sd.get("claim_type",
                                     claim_type_for(tool, action)),
                "result_summary": sd.get("result_summary", ""),
                "validation": sd.get("validation"),
                "success": sd.get("success", False),
                "error": sd.get("error", ""),
                "provenance_sources": list(sd.get("provenance", {}).keys()),
            }
            key = (tool, action, str(sd.get("params")))
            if key in seen:
                continue
            seen.add(key)
            records.append({"id": f"claim{len(records)}", "kind": "evidence",
                            "chain": chain})
        records.append({
            "id": f"claim{len(records)}",
            "kind": "verdict",
            "chain": {
                "question": question,
                "claim_type": "verification_evidence",
                "verdict": verdict.to_dict(),
                "validation": {
                    "status": verdict.status,
                    "missing": verdict.missing_requirements},
                "answer_prefix": answer_text[:160],
            }})
        return records
    @staticmethod
    def _provenance_graph(step_results) -> dict:
        nodes = [{"id": "question", "type": "input"}]
        edges = [{"from": "question", "to": "plan"}]
        for i, sd in enumerate(step_results):
            src = sd.get("provenance") or {}
            nodes.append({
                "id": f"step{i}",
                "type": "tool_call",
                "tool": sd.get("tool"),
                "action": sd.get("action"),
                "params": sd.get("params"),
                "claim_type": sd.get("claim_type",
                                     claim_type_for(sd.get("tool", ""),
                                                    sd.get("action", ""))),
                "result_hash": _hash(sd.get("result") or sd.get("result_summary", "")),
                "validation": sd.get("validation"),
                "status": "error" if not sd.get("success") else "ok",
                "sources": src})
            edges.append({"from": "plan", "to": f"step{i}"})
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _execution_detail(step_results):
        out = []
        for s in step_results:
            out.append(render_for_synthesis(s))
        return out

    @staticmethod
    def _flow_relpath(path: str) -> str:
        """Shorten an absolute flow-log path to something readable/persistent."""
        try:
            return os.path.relpath(path, config.RESULTS_DIR)
        except Exception:
            return path

    def tools(self):
        return self.registry.available()


def _hash(obj):
    try:
        blob = json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        blob = str(obj)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]