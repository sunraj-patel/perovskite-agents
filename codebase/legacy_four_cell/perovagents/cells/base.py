"""Generic domain-cell loop (spec section 3, S3).

One implementation reused four times: Agent -> Tool -> Gate, with the
gate's Critique routed back to the same cell's agent ("No" branch) until
it passes ("Yes" branch) or `max_attempts` is exhausted (FailedResult).

Cell-specific code (S5) only fills the three slots: `agent_fn` (LLM),
`tool_fn` (deterministic), `gate_fn` (deterministic checks first). The
agent receives the full attempt history so it does not repeat itself.
Tool exceptions count as failed attempts with a critique — they never
crash the run.
"""
from typing import Callable, Dict, List, Optional, Union

from .schemas import (Attempt, Critique, FailedResult, Task, ValidatedResult,
                      VerificationReport)
from ..verify.base import check, report

#: Default retry budget per cell (spec section 3).
DEFAULT_MAX_ATTEMPTS = 3


class DomainCell:
    """Base cell. Subclasses set `cell` and wire the three slots."""

    cell = "base"

    def __init__(self,
                 agent_fn: Callable[[Task, List[Attempt]], Dict],
                 tool_fn: Callable[[Dict], Dict],
                 gate_fn: Callable[[Dict, Dict], VerificationReport],
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS):
        self.agent_fn = agent_fn
        self.tool_fn = tool_fn
        self.gate_fn = gate_fn
        self.max_attempts = max(1, int(max_attempts))

    # ---- one attempt -------------------------------------------------
    def _attempt(self, task: Task, history: List[Attempt],
                 n: int) -> Attempt:
        try:
            tool_call = self.agent_fn(task, history) or {}
        except Exception as exc:  # agent errors also become critiques
            return Attempt(n=n, tool_call={}, raw_result={},
                           report=report([check(
                               "agent_ok", False,
                               f"agent raised: {exc}", str(exc)[:200])]))
        try:
            raw = self.tool_fn(tool_call) or {}
        except Exception as exc:  # tool errors become critiques, not crashes
            return Attempt(n=n, tool_call=tool_call, raw_result={},
                           report=report([check(
                               "tool_ok", False,
                               f"tool raised: {exc}", str(exc)[:200])]))
        try:
            rep = self.gate_fn(raw, tool_call)
        except Exception as exc:  # a crashing gate is a failed attempt too
            rep = report([check(
                "gate_ok", False,
                f"gate raised: {exc}", str(exc)[:200])])
        return Attempt(n=n, tool_call=tool_call, raw_result=raw, report=rep)

    @staticmethod
    def _critique(rep: VerificationReport) -> Critique:
        notes = "; ".join(f"{c.name}: {c.message}" for c in rep.checks
                          if not c.passed)
        return Critique(failed_checks=rep.failed_checks(),
                        explanation=notes or "gate failed without detail",
                        suggested_change="")

    # ---- main loop ---------------------------------------------------
    def run(self, task: Task,
            history: Optional[List[Attempt]] = None
            ) -> Union[ValidatedResult, FailedResult]:
        """Run the cell to a ValidatedResult or, after `max_attempts`,
        a FailedResult carrying every attempt and the last critique."""
        attempts: List[Attempt] = list(history or [])
        last_critique: Optional[Critique] = None
        while len(attempts) < self.max_attempts:
            att = self._attempt(task, attempts, len(attempts) + 1)
            attempts.append(att)
            if att.report.passed:
                return ValidatedResult(
                    task_id=task.id, cell=self.cell,
                    payload=dict(att.raw_result),
                    report=att.report,
                    provenance={"cell": self.cell, "task_id": task.id,
                                "attempts": len(attempts),
                                "tool_call": dict(att.tool_call)})
            last_critique = self._critique(att.report)
        return FailedResult(task_id=task.id, cell=self.cell,
                            attempts=attempts, last_critique=last_critique)
