"""Agent models: plan representation + shared dataclasses."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PlanStep:
    """One tool call in a plan.

    Carries the planner's *intent* for the step (purpose, expected output
    tokens, whether the final answer depends on it) so the verifier can check
    task sufficiency after execution — not just whether the tool 'ran'.
    """
    tool: str
    action: str = ""
    params: dict = field(default_factory=dict)
    reason: str = ""
    purpose: str = ""
    expected_outputs: List[str] = field(default_factory=list)
    required_for_final_answer: bool = True

    def to_dict(self):
        return {"tool": self.tool, "action": self.action,
                "params": self.params, "reason": self.reason,
                "purpose": self.purpose,
                "expected_outputs": list(self.expected_outputs),
                "required_for_final_answer": self.required_for_final_answer}


@dataclass
class Plan:
    """An ordered sequence of steps for one question."""
    question: str
    steps: list = field(default_factory=list)
    planner: str = "unknown"
    note: str = ""

    def to_dict(self):
        return {"question": self.question, "planner": self.planner,
                "note": self.note,
                "steps": [s.to_dict() for s in self.steps]}


class PlannerError(Exception):
    """Raised when a plan cannot be produced or validated."""