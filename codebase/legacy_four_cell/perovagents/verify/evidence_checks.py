"""Evidence gate checks for cell (b): Evidence + Citation Verification (S2).

A claim is {"text": ..., "citations": [{"paper_id", "locator",
"passage", "number"(optional), "context"(optional dict)}]}.

Deterministic checks (resolvable citation, number-in-source, context tags)
are pure. `support` and the context fallback accept an injectable `judge`
callable `judge(claim_text, passage) -> (supported: bool, note: str)` so
unit tests stay deterministic; the default judge is the token-overlap
scorer already used by `literature/citations.py`, and an LLM judge can be
injected by the cell (S5) per spec section 10 (DECISIONS.md D8).
"""
import re
from typing import Callable, Dict, List, Optional, Tuple

from ..cells.schemas import CheckResult, VerificationReport
from .base import check, report


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def default_judge(claim_text: str,
                  passage: str) -> Tuple[bool, str]:
    """Token-overlap support scorer (mirrors citations.verify bands:
    PASS > 0.6, WARN > 0.3). Gate passes at WARN-or-better."""
    ct = {t for t in _tokens(claim_text) if len(t) > 3}
    if not ct:
        return False, "claim has no content tokens"
    pt = set(_tokens(passage))
    score = len(ct & pt) / len(ct)
    if score > 0.6:
        return True, f"strong token support ({score:.2f})"
    if score > 0.3:
        return True, f"weak token support ({score:.2f})"
    return False, f"insufficient token support ({score:.2f})"


def check_citation_resolvable(citation: Dict) -> CheckResult:
    """Every claim carries a paper id plus a locator."""
    pid = str(citation.get("paper_id", "")).strip()
    loc = str(citation.get("locator", "")).strip()
    if not pid:
        return check("citation_resolvable", False, "missing paper_id",
                     citation)
    if not loc:
        return check("citation_resolvable", False,
                     f"missing locator for {pid}", {"paper_id": pid})
    return check("citation_resolvable", True, f"{pid}#{loc}")


def _norm_num(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def check_number_in_source(number: object, passage: str) -> CheckResult:
    """An extracted number must appear verbatim (whitespace-insensitive)
    in the cited passage text/table."""
    if number is None:
        return check("number_in_source", True, "no number claimed")
    cand = {str(number), f"{float(number):g}"} if isinstance(
        number, (int, float)) else {str(number)}
    hay = _norm_num(passage)
    for c in cand:
        if _norm_num(c) and _norm_num(c) in hay:
            return check("number_in_source", True,
                         f"{c!r} present in source")
    return check("number_in_source", False,
                 f"{number!r} not found in cited passage",
                 {"number": number})


def check_support(claim_text: str, passage: str,
                  judge: Optional[Callable[[str, str],
                                           Tuple[bool, str]]] = None) -> CheckResult:
    """The cited passage actually supports the claim (injectable judge)."""
    fn = judge or default_judge
    supported, note = fn(claim_text, passage)
    return check("support", supported, note,
                 None if supported else {"claim": claim_text[:160]})


def check_context(question_ctx: Dict, claim_ctx: Dict) -> CheckResult:
    """Claim conditions (architecture, composition, setting) match the
    question. Missing claim-side context fails: context must be explicit
    to be verifiable (DECISIONS.md D8). Non-dict inputs are treated as
    absent (planner-shaped, never crashing)."""
    if not isinstance(question_ctx, dict):
        question_ctx = {}
    if not isinstance(claim_ctx, dict):
        claim_ctx = {}
    problems = []
    for key in ("architecture", "composition", "setting"):
        want = str(question_ctx.get(key, "") or "").strip().lower()
        got = str(claim_ctx.get(key, "") or "").strip().lower()
        if not want:
            continue  # question constrains nothing on this axis
        if not got:
            problems.append(f"{key}: claim states no context")
        elif want not in got and got not in want:
            problems.append(f"{key}: question={want!r} vs claim={got!r}")
    if problems:
        return check("context", False, "; ".join(problems),
                     {"question": question_ctx, "claim": claim_ctx})
    return check("context", True, "claim context matches question")


def verify_claim(claim: Dict, question_ctx: Optional[Dict] = None,
                 judge: Optional[Callable] = None) -> VerificationReport:
    """Full gate for one claim across all its citations (all must pass)."""
    checks: List[CheckResult] = []
    for cit in claim.get("citations", []):
        checks.append(check_citation_resolvable(cit))
        if claim.get("number") is not None:
            checks.append(check_number_in_source(claim["number"],
                                                 cit.get("passage", "")))
        checks.append(check_support(claim.get("text", ""),
                                    cit.get("passage", ""), judge))
        checks.append(check_context(question_ctx or {},
                                    cit.get("context", {}) or {}))
    if not claim.get("citations"):
        checks.append(check("citation_resolvable", False,
                            "claim has no citations", claim.get("text", "")[:80]))
    return report(checks)
