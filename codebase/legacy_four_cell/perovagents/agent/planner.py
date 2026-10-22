"""LLM planner.

Turns a *task specification* + user question into a minimal, valid sequence of
tool actions. The planner only *plans*; it never runs anything.

The system prompt is assembled from the registry's per-action schemas and the
task analyzer's task types, so action capability is decidable from the catalog:

    * prediction and interpretation are different operations
    * feature importance requires an explanation action, not prediction
    * "what should I change" requires intervention (what_if/optimize)

`plan()` validates each step against the registry (unknown action, wrong param
type, invalid enum, missing required param => rejection with corrective
feedback), instead of silently coercing arbitrary output. On failure a
`PlannerError` carrying `feedback` is raised so the orchestrator can re-run the
planner with the correction.
"""
import json
import logging
from typing import List, Optional

from .. import config  # noqa: F401
from ..core.registry import ToolRegistry
from ..llm import LLM, extract_json
from .models import Plan, PlanStep, PlannerError
from .task import TASK_TYPES

log = logging.getLogger(__name__)

SYSTEM_TEMPLATE = """You are the planning agent for PerovAgents, an agentic
research assistant for perovskite solar cells.

Your job is to convert the user's scientific question into the minimum
sufficient sequence of valid scientific tool ACTIONS.

The TaskSpec produced by the Task Analyzer is authoritative. Do not
reinterpret or override its task_type, architecture, requested outputs, or
evidence requirements unless you identify a contradiction, in which case
return a planning error. You decide only the sequence of tool operations.

IMPORTANT: the tool "ml" is not one capability. It is a family of distinct
scientific operations. You must select the correct ACTION, not merely the ML
tool.

ML operations:
- predict: predict/estimate a target value such as PCE (model_prediction).
- explain_global: identify globally influential features using model
  attribution such as SHAP (model_association).
- explain_local: explain why a particular sample receives its prediction.
- what_if: evaluate predicted response to changing a specified parameter.
- optimize: search allowed controllable parameters for configurations that
  improve the model objective.

Interpretation rules (the analyzer's task_type decides the class; these
phrases are merely examples of that same class):
- "most important/relevant parameters", "which features matter",
  "what drives/influences PCE" -> ml.explain_global
- "why is this device predicted to have ...", "what is driving this device's
  prediction" -> ml.explain_local
- "what happens if I increase/decrease/change X", "how sensitive is PCE to X"
  -> ml.what_if
- "what should I change to improve PCE", "which parameters should I optimize"
  -> ml.optimize
- "what is the predicted PCE", "estimate PCE" -> ml.predict

Do not require the user to explicitly mention SHAP, prediction, sensitivity,
or optimization terminology.

Scientific claim restrictions:
- SHAP / global feature importance describes model-derived association /
  importance, not causality.
- what_if describes model-predicted sensitivity, not experimental causality.
- optimize may only modify features declared experimentally controllable by
  the feature ontology.
- Never optimize measurement-derived, metadata, output, or structural
  bookkeeping features.

Tool-selection rules:
- Do not use a database query merely because the question mentions
  "parameters" or "device".
- Do not use ml.predict when the requested output is feature importance.
- Do not use pdb when the TaskSpec requests model interpretation unless the
  database is explicitly required as additional evidence.
- Use the minimum sufficient number of actions.
- Every action must explicitly support the requested task_type (its declared
  task_types must contain the analyzed task_type).
- Never invent tool names, actions, parameters, or output fields.

Available tools (each action declares the task_types it serves and the outputs
it produces):
{TOOLS_JSON}

For each step return exactly this shape:
{{"tool": "...", "action": "...", "params": {{...}}, "purpose": "...",
  "expected_outputs": ["tokens the result must carry, e.g. prediction, interval,
  feature_name, importance, direction, parameter_value, count, aggregate, ...],
  "required_for_final_answer": true/false}}

Return ONLY a JSON array of steps.

Task specification from the analyzer:
{TASK_JSON}

This task specification is AUTHORITATIVE: your final-answer action's declared
task_types must include the analyzed task_type. If the JSON is empty and you
must self-derive the intent, state the task_type you inferred in your
reasoning — do not silently choose a different answer class.

Question: {QUESTION}
Plan (JSON array only):"""


LEGACY_TEMPLATE = """You are a perovskite solar cell research assistant with
access to these tools: {tools} You plan ONE or more tool calls to answer the
user's question. For each step return exactly:
{{"tool": "<tool name>", "params": {{...}}, "reason": "..."}}.
Use an 'action' key inside params when the tool needs to select an operation.
Return ONLY a JSON array of steps.

Question: {QUESTION}
Steps (JSON array only):"""


def semantic_errors(plan, registry: ToolRegistry, task_type: str = "") -> list:
    """Task-analyzer -> planner contract check.

    The final-answer steps of a plan must serve the analyzed task_type (i.e.
    some action's declared `task_types` must contain it). Context steps that
    don't carry the final answer are tolerated; `multi_tool_investigation` is
    open by construction. This is what stops the planner from quietly
    substituting one answer class for another after the analyzer decides.
    """
    if not task_type or task_type == "multi_tool_investigation":
        return []
    required = [s for s in plan.steps if s.required_for_final_answer]
    if not required:
        return ["no step is marked required_for_final_answer; the plan is "
                "ambiguous about which result carries the answer"]
    if any(registry.action_supports_task(s.tool, s.action, task_type)
           for s in required):
        return []
    return [
        f"{s.tool}.{s.action} does not serve the analyzed task "
        f"'{task_type}': its declared task_types exclude it. Pick an action "
        f"whose task_types include '{task_type}'."
        for s in required if not registry.validate_action(s.tool, s.action)
        or not registry.action_supports_task(s.tool, s.action, task_type)]


class LLMPlanner:
    name = "llm"

    def __init__(self, llm: LLM = None, registry: ToolRegistry = None,
                 max_steps: int = 5, max_attempts: int = 2,
                 legacy: bool = False):
        self.llm = llm or LLM()
        self.registry = registry
        self.max_steps = max_steps
        self.max_attempts = max_attempts
        self.legacy = legacy

    def _catalog_json(self, registry):
        import json
        return json.dumps(registry.catalog_dicts(), indent=1)

    def plan(self, question: str, registry: ToolRegistry = None,
             task_spec=None, feedback: Optional[str] = None,
             attempt: int = 0) -> Plan:
        """Produce a validated plan. Raises PlannerError(feedback) when the
        plan cannot be validated so the orchestrator can retry or fall back."""
        self.registry = registry or self.registry
        if self.registry is None:
            raise PlannerError("no registry available for planning")
        if not self.llm.usable:
            return self._offline_plan(question, registry or self.registry)

        prompt = self._prompt(question, registry or self.registry,
                              task_spec=task_spec, feedback=feedback)
        text = self.llm.chat(prompt, temperature=0.0)
        if str(text).startswith("[LLM ERROR]") or str(text).startswith("[OFFLINE]"):
            # Not a parse failure: the engine is down (quota/circuit breaker).
            # Raise a concise, truthful error so the orchestrator's single
            # "falling back to rules" line carries the real reason.
            raise PlannerError("LLM unavailable (%s)" % str(text).strip())
        parsed = extract_json(text)
        steps, errors = self._coerce_and_validate(parsed, registry or self.registry,
                                                  task_spec=task_spec)
        if steps:
            return Plan(question=question, planner=self.name, steps=steps)
        msg = "LLM produced an invalid plan: " + "; ".join(errors[:4])
        log.warning("%s (attempt %d)", msg, attempt)
        raise PlannerError(msg)

    def retryable(self, question: str, registry: ToolRegistry,
                  task_spec=None, feedback: Optional[str] = None,
                  attempt: int = 0) -> Plan:
        """Best-effort wrapper: returns a validated plan or falls back to the
        deterministic rule planner with a note."""
        try:
            return self.plan(question, registry, task_spec=task_spec,
                             feedback=feedback, attempt=attempt)
        except PlannerError as exc:
            fallback = self._offline_plan(question, registry)
            fallback.note = f"LLM plan rejected ({exc}); rule fallback used"
            return fallback

    def _offline_plan(self, question: str, registry: ToolRegistry) -> Plan:
        from .rules import RulePlanner
        return RulePlanner().plan(question)

    def _prompt(self, question, registry, task_spec=None, feedback=None):
        if self.legacy:
            names = ", ".join(s.name for s in registry.available())
            prompt = LEGACY_TEMPLATE.replace("{tools}", names) \
                .replace("{QUESTION}", question)
        else:
            task_json = (task_spec.to_dict() if task_spec is not None else {})
            prompt = SYSTEM_TEMPLATE \
                .replace("{TOOLS_JSON}", self._catalog_json(registry)) \
                .replace("{TASK_JSON}", json.dumps(task_json, indent=1)) \
                .replace("{QUESTION}", question)
        if feedback:
            prompt += ("\n\nYour previous plan was rejected. Correct exactly "
                       "these problems and return the full revised plan:\n"
                       + str(feedback)[:1200])
        return prompt

    # ---- coercion + validation (never silent) ----------------------------
    def _coerce_and_validate(self, parsed, registry: ToolRegistry,
                             task_spec=None):
        if self.legacy:
            return self._raw_coerce(parsed)
        if not isinstance(parsed, list):
            return [], ["LLM response was not a JSON array of steps"
                        + (f" (got {type(parsed).__name__})" if parsed is not None
                           else " — LLM returned an error/empty payload")]
        steps, errors = [], []
        raw_list = parsed if isinstance(parsed, list) else []
        for raw in raw_list[: self.max_steps]:
            if not isinstance(raw, dict) or "tool" not in raw:
                errors.append("a step is missing its 'tool' key")
                continue
            tool = str(raw["tool"])
            action = str(raw.get("action") or "")
            params = dict(raw.get("params") or {})
            params.pop("action", None)          # action is a first-class field
            if not action:
                specs = list(registry.available())
                spec = next((s for s in specs if s.name == tool), None)
                if spec is None:
                    errors.append(f"unknown tool '{tool}'")
                    continue
                if len(spec.actions) != 1:
                    errors.append(f"step for '{tool}' must declare an action")
                    continue
                action = next(iter(spec.actions))
            if not registry.validate_action(tool, action):
                errors.append(f"unknown action '{action}' for tool '{tool}'")
                continue
            ok, perrs, _w = registry.validate_params(tool, action, params)
            if not ok:
                errors.extend(f"{tool}.{action}: {e}" for e in perrs)
                continue
            spec = registry.action_spec(tool, action)
            outputs = raw.get("expected_outputs") or list(spec.outputs)
            if not outputs:
                errors.append(f"step for {tool}.{action} declares no "
                              f"expected outputs")
                continue
            steps.append(PlanStep(
                tool=tool, action=action, params=params,
                reason=str(raw.get("reason") or "")[:160],
                purpose=str(raw.get("purpose") or raw.get("reason") or "")[:200],
                expected_outputs=[str(o) for o in outputs[:8]],
                required_for_final_answer=bool(raw.get(
                    "required_for_final_answer", True))))
        return steps, errors

    def _raw_coerce(self, parsed):
        """Legacy raw-tool-calling coercion (ablation mode A): accepts any
        tool/params the LLM emits, with no schema or action-first checking."""
        steps, errors = [], []
        if not isinstance(parsed, list):
            return [], ["legacy plan is not a list"]
        for raw in parsed[: self.max_steps]:
            if not isinstance(raw, dict) or "tool" not in raw:
                errors.append("a step is missing its 'tool' key")
                continue
            params = dict(raw.get("params") or {})
            action = str(params.pop("action", "") or "")
            steps.append(PlanStep(
                tool=str(raw["tool"]), action=action, params=params,
                reason=str(raw.get("reason") or "")[:160],
                purpose=str(raw.get("reason") or "")[:200],
                expected_outputs=list(raw.get("expected_outputs")
                                      or raw.get("outputs") or [])[:8],
                required_for_final_answer=True))
        return steps, errors


def default_llm_planner(registry: ToolRegistry = None) -> LLMPlanner:
    """Construct an LLM planner over the default (or given) registry."""
    return LLMPlanner(registry=registry)


def validate_plan(plan: Plan, registry: ToolRegistry,
                  task_spec=None) -> tuple:
    """Re-validate an existing plan. Returns (ok, errors).

    Also enforces the task-analyzer contract when `task_spec` is given:
    the final-answer step of an LLM-made plan must serve the analyzed
    task_type. Deterministic rule plans are domain-authoritative and are
    exempt from the semantic rejection (their intent is coded, not derived
    from a task_type the analyzer assigned separately).
    """
    errors = []
    for step in plan.steps:
        if not registry.validate_action(step.tool, step.action):
            errors.append(f"{step.tool}.{step.action}: unknown action")
            continue
        ok, perrs, _ = registry.validate_params(step.tool, step.action, step.params)
        if not ok:
            errors.extend(f"{step.tool}.{step.action}: {e}" for e in perrs)
    if task_spec is not None and plan.planner != "rules":
        errors.extend(semantic_errors(plan, registry,
                                      task_spec.task_type))
    return not errors, errors