"""Shared constructors for gate checks (S2).

Every check in this package returns a `CheckResult`; every gate runner
returns a `VerificationReport`. Pure functions only — no I/O, no LLM calls
here (the literature support/context judgements accept an injectable
`judge` callable, defaulting to the deterministic token-overlap scorer).
"""
from typing import Any, List

from ..cells.schemas import CheckResult, VerificationReport


def check(name: str, passed: bool, message: str = "",
          offending: Any = None) -> CheckResult:
    """Build one named check outcome."""
    return CheckResult(name=name, passed=bool(passed),
                       message=message, offending=offending)


def report(checks: List[CheckResult]) -> VerificationReport:
    """Bundle check outcomes into a gate report."""
    return VerificationReport(checks=list(checks))
