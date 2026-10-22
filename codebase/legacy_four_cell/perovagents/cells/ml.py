"""Cell (c): ML (S5).

Agent: selects operation + parameters deterministically from task params
(`action` defaults to `predict`); LLM selection plugs in at S6.
Tool: `MLPredictorTool.call`. After the call the adapter attaches recorded
validation performance from a metrics file when one exists
(`metrics_path` param, default `results/ml/validation.json`); with no
metrics file the S2 `model_attached` check honestly fails (DECISIONS.md).
Gate: `verify_ml` (S2).
"""
import json
import os
from typing import Any, Dict, List

from .. import config
from ..cells.schemas import VerificationReport
from ..ml.predictor import MLPredictorTool
from ..verify.ml_checks import verify_ml
from .base import DomainCell
from .schemas import Task


def default_metrics_path() -> str:
    """Location of recorded group-aware validation metrics (may not exist)."""
    return os.path.join(config.ensure_results("ml"), "validation.json")


def attach_validation(result: Dict[str, Any],
                      metrics_path: str) -> Dict[str, Any]:
    """Attach {metric, score, split} for the result's model arch when a
    metrics file records them; otherwise leave the result untouched so the
    gate fails closed on missing validation evidence."""
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
    except (OSError, ValueError):
        return result
    arch = str(result.get("architecture", "Overall"))
    entry = (metrics.get(arch) or metrics.get("Overall") or {})
    if isinstance(entry, dict) and entry.get("metric") \
            and entry.get("score") is not None and entry.get("split"):
        prov = dict(result.get("provenance", {}))
        prov["validation"] = {"metric": entry["metric"],
                              "score": entry["score"],
                              "split": entry["split"]}
        result = dict(result)
        result["provenance"] = prov
    return result


class MLCell(DomainCell):
    """ML domain cell."""
    cell = "ml"

    def __init__(self, max_attempts: int = 3,
                 metrics_path: str = None, llm=None):
        self.predictor = MLPredictorTool()
        self.metrics_path = metrics_path or default_metrics_path()
        self.llm = llm
        super().__init__(agent_fn=self._agent, tool_fn=self._tool,
                         gate_fn=self._gate, max_attempts=max_attempts)

    # -- slots ---------------------------------------------------------
    def _agent(self, task: Task, history: List) -> Dict[str, Any]:
        fallback = dict(task.params)
        fallback.setdefault("action", "predict")
        fallback.setdefault("architecture", "overall")
        if history and self.llm is not None:
            from .llm_agent import llm_tool_call
            return llm_tool_call(self.llm, "ml_agent", task, history,
                                 fallback)
        return fallback

    def _tool(self, call: Dict[str, Any]) -> Dict[str, Any]:
        result = self.predictor.call(dict(call))
        if isinstance(result, dict) and result.get("status") != "error":
            result = attach_validation(
                result, call.get("metrics_path", self.metrics_path))
        return result

    def _gate(self, raw: Dict[str, Any],
              call: Dict[str, Any]) -> VerificationReport:
        if raw.get("status") == "error":
            from ..verify.base import check, report
            return report([check("tool_ok", False,
                                 f"predictor error: {raw.get('error', '')}",
                                 raw.get("error", ""))])
        return verify_ml(raw, parameter=call.get("parameter"))
