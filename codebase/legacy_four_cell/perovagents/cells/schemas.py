"""Cell data contracts (spec section 5).

Typed models shared by the planner, the executor and the four domain cells.
These are pure data — no behaviour, no imports beyond stdlib — so later steps
(S3 cells, S6 planner, S7 executor) can adopt them without touching the old
`agent/models.py` types, which stay untouched until cutover (S9).

Naming follows the spec: Task, Plan, Attempt, Critique, ValidatedResult,
FailedResult, RunState, VerificationReport. `CheckResult` is one row of a
`VerificationReport`: individually named, individually reported checks.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# The four domain cells from the architecture diagram. "utility" is not a
# cell: it marks executor-run helper calls (calculator, python_exec,
# symbolic) that need no verification gate (see DECISIONS.md D2).
CELLS = ("pdb_sql", "literature", "ml", "scaps")
CELL_AND_UTILITY = CELLS + ("utility",)

# Task lifecycle states owned by the executor (spec section 2.2).
TASK_STATUSES = ("pending", "running", "validated", "failed", "skipped")


@dataclass
class Task:
    """One unit of work for one cell. `depends_on` holds other task ids
    whose validated results must be passed in as input before this runs."""
    id: str
    cell: str
    request: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "cell": self.cell, "request": self.request,
                "params": dict(self.params), "depends_on": list(self.depends_on)}


@dataclass
class Plan:
    """Cell-aware plan: a query decomposed into cell-targeted tasks."""
    query: str
    tasks: List[Task] = field(default_factory=list)
    planner: str = "unknown"
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "planner": self.planner,
                "note": self.note,
                "tasks": [t.to_dict() for t in self.tasks]}


@dataclass
class CheckResult:
    """Outcome of one named verification check."""
    name: str
    passed: bool
    message: str = ""
    offending: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": self.passed,
                "message": self.message, "offending": self.offending}


@dataclass
class VerificationReport:
    """The gate's verdict: every check listed with pass/fail, message and
    offending values. A report passes only when every check passes."""
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def failed_checks(self) -> List[str]:
        """Names of the checks that failed (feeds Critique.failed_checks)."""
        return [c.name for c in self.checks if not c.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed,
                "checks": [c.to_dict() for c in self.checks]}


@dataclass
class Attempt:
    """One trip through a cell: tool call, raw result, gate report."""
    n: int
    tool_call: Dict[str, Any] = field(default_factory=dict)
    raw_result: Dict[str, Any] = field(default_factory=dict)
    report: VerificationReport = field(default_factory=VerificationReport)

    def to_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "tool_call": dict(self.tool_call),
                "raw_result": dict(self.raw_result),
                "report": self.report.to_dict()}


@dataclass
class Critique:
    """What the gate sends back to the cell's agent on the 'No' branch."""
    failed_checks: List[str] = field(default_factory=list)
    explanation: str = ""
    suggested_change: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"failed_checks": list(self.failed_checks),
                "explanation": self.explanation,
                "suggested_change": self.suggested_change}


@dataclass
class ValidatedResult:
    """A result that passed its cell's gate, with provenance attached."""
    task_id: str
    cell: str
    payload: Dict[str, Any] = field(default_factory=dict)
    report: VerificationReport = field(default_factory=VerificationReport)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "cell": self.cell,
                "payload": dict(self.payload),
                "report": self.report.to_dict(),
                "provenance": dict(self.provenance)}


@dataclass
class FailedResult:
    """A cell that exhausted its retry budget. The executor must surface
    this explicitly and never silently drop it."""
    task_id: str
    cell: str
    attempts: List[Attempt] = field(default_factory=list)
    last_critique: Optional[Critique] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "cell": self.cell,
                "attempts": [a.to_dict() for a in self.attempts],
                "last_critique": (self.last_critique.to_dict()
                                  if self.last_critique else None)}


@dataclass
class RunState:
    """Executor-owned run state (spec section 2.2)."""
    plan: Plan = field(default_factory=lambda: Plan(query=""))
    task_status: Dict[str, str] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def mark(self, task_id: str, status: str) -> None:
        """Record a task status transition (must be a known status)."""
        if status not in TASK_STATUSES:
            raise ValueError(f"unknown task status {status!r}")
        self.task_status[task_id] = status

    def ready(self) -> List[str]:
        """Task ids with no unmet dependencies (validated deps only)."""
        done = {tid for tid, r in self.results.items()
                if isinstance(r, ValidatedResult)}
        out = []
        for t in self.plan.tasks:
            if self.task_status.get(t.id, "pending") != "pending":
                continue
            if all(d in done for d in t.depends_on):
                out.append(t.id)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"plan": self.plan.to_dict(),
                "task_status": dict(self.task_status),
                "results": {k: (v.to_dict() if hasattr(v, "to_dict") else v)
                            for k, v in self.results.items()},
                "errors": dict(self.errors)}


# task_type (TaskSpec vocabulary, agent/task.py:TASK_TYPES) -> candidate
# cells in preference order. Multi-cell types fan out; "utility" means the
# executor runs a helper tool directly with no gate (DECISIONS.md D2/D3).
TASK_CELL_ROUTES: Dict[str, List[str]] = {
    "database_query": ["pdb_sql"],
    "factual_lookup": ["pdb_sql"],
    "literature_retrieval": ["literature"],
    "literature_extraction": ["literature"],
    "verification": ["literature"],
    "prediction": ["ml"],
    "interpretation": ["ml"],
    "feature_importance": ["ml"],
    "sensitivity_analysis": ["ml", "scaps"],
    "optimization": ["ml"],
    "simulation": ["scaps"],
    "comparison": ["pdb_sql", "literature", "ml", "scaps"],
    "multi_tool_investigation": ["pdb_sql", "literature", "ml", "scaps"],
    "calculation": ["utility"],
}


def cells_for(task_type: str) -> List[str]:
    """Candidate cells for a TaskSpec task_type (empty = unknown type)."""
    return list(TASK_CELL_ROUTES.get(task_type, []))
