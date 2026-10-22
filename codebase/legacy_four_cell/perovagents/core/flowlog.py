"""Flow log — one clean JSONL file per user question, capturing the entire
run end to end.

The provenance store records *what happened* (tool name, params, status,
result hash, sources). The flow log additionally keeps the *content*: every
planned step, every tool result, every verdict and the final response, in the
order they happened, so a run can be replayed or audited from the file alone.

Event vocabulary (one JSON line per event):
  query      the user's question + pipeline/mode
  task_spec  the analyzed task specification (dict)
  plan       one planning pass: iteration, planner, full plan dict
  step       one executed tool step: full step dict (params + result)
  verdict    one verifier verdict per plan iteration
  answer     final response text + sources + warnings + answer contract
  done       run totals (elapsed, n_ok/n_fail, final verdict)

Writes are append-only and thread-safe (locked). Large dicts are truncated
at MAX_EVENT_CHARS by replacing the oversized value with a preview + marker,
so pathological outputs never balloon the file but are still visible.
"""
import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from .. import config

MAX_EVENT_CHARS = 200_000


def _render(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


class FlowLog:
    """Per-run flow log persisted as ``flows/flow_<run_id>.jsonl``."""

    def __init__(self, run_id: Optional[str] = None,
                 dir: Optional[str] = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.dir = dir or config.ensure_results("flows")
        self.path = os.path.join(self.dir, f"flow_{self.run_id}.jsonl")
        self._lock = threading.Lock()

    # ---- low level -------------------------------------------------------
    def _append(self, event: str, **fields) -> None:
        line = {"t": time.time(), "event": event, "run_id": self.run_id}
        line.update(fields)
        blob = _render(line)
        if len(blob) > MAX_EVENT_CHARS:
            truncated = {}
            for k, v in fields.items():
                if isinstance(v, (dict, list)):
                    preview = _render(v)
                    if len(preview) > MAX_EVENT_CHARS // 4:
                        truncated[k] = {
                            "_truncated": True,
                            "preview": preview[: MAX_EVENT_CHARS // 4],
                        }
                        continue
                truncated[k] = v
            line = {"t": line["t"], "event": event, "run_id": self.run_id,
                    **_shrink(truncated)}
            blob = _render(line)
        try:
            with self._lock:
                os.makedirs(self.dir, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(blob + "\n")
        except Exception:
            pass  # flow logging must never crash a query

    # ---- high-level events -----------------------------------------------
    def query(self, question: str, pipeline: str = "", mode: str = "") -> None:
        self._append("query", question=question, pipeline=pipeline, mode=mode)

    def task_spec(self, spec: Optional[Dict[str, Any]] = None) -> None:
        self._append("task_spec", spec=spec or {})

    def plan(self, iteration: int = 0, planner: str = "",
             plan: Optional[Dict[str, Any]] = None,
             feedback: Optional[str] = None) -> None:
        self._append("plan", iteration=iteration, planner=planner,
                     plan=plan or {}, feedback=feedback)

    def step(self, iteration: int = 0, step_index: int = 0,
             step: Optional[Dict[str, Any]] = None) -> None:
        self._append("step", iteration=iteration, step_index=step_index,
                     step=step or {})

    def verdict(self, iteration: int = 0,
                verdict: Optional[Dict[str, Any]] = None) -> None:
        self._append("verdict", iteration=iteration, verdict=verdict or {})

    def answer(self, answer: str = "", sources: Optional[List[str]] = None,
               warnings: Optional[List[str]] = None,
               contract: Optional[Dict[str, Any]] = None) -> None:
        self._append("answer", answer=answer, sources=list(sources or []),
                     warnings=list(warnings or []), answer_contract=contract or {})

    def done(self, **summary) -> None:
        self._append("done", **summary)


def _shrink(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate a copy of fields so a JSON render fits in one event budget."""
    out = {}
    for k, v in fields.items():
        if isinstance(v, (dict, list)):
            preview = _render(v)
            out[k] = preview[: (MAX_EVENT_CHARS // 2) - 64]
        else:
            out[k] = v
    return out


def read_flow(path: str) -> List[Dict[str, Any]]:
    """Load a flow log back into an ordered list of event dicts."""
    events: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events