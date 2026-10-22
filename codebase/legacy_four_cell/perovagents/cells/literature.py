"""Cell (b): Literature (S5).

Agent: formulates the search strategy deterministically — query text from
`task.params["query"]` (else the task request), top-k and question context
passed through. LLM strategy refinement plugs into the agent slot at S6.
Tool chain: Search/RAG over the markdown+embeddings corpus; each hit becomes
a claim whose citation carries paper id + section locator + passage text
(the extraction/fulltext tools join the chain at S6 planner level; the
search tool already returns sections and snippets).
Gate: `verify_claim` per claim (S2), merged into one report.
"""
from typing import Any, Dict, List

from ..cells.schemas import VerificationReport
from ..literature.search import LiteratureSearchTool
from ..verify.base import check, report
from ..verify.evidence_checks import verify_claim
from .base import DomainCell
from .schemas import Task


class LiteratureCell(DomainCell):
    """Literature domain cell.

    `judge`: None (deterministic token-overlap default), a
    `judge(claim, passage) -> (supported, note)` callable, or `"llm"`
    for an LLM-backed support judge (needs model access).
    """
    cell = "literature"

    def __init__(self, max_attempts: int = 3, judge=None, llm=None):
        from ..llm import get_llm
        self.search = LiteratureSearchTool()
        if judge == "llm":
            judge = self._llm_judge(llm or get_llm())
        self.judge = judge
        self.llm = llm
        super().__init__(agent_fn=self._agent, tool_fn=self._tool,
                         gate_fn=self._gate, max_attempts=max_attempts)

    @staticmethod
    def _llm_judge(llm):
        """YES/NO support judge over an LLM client (6/6 on the eval
        battery vs 4/8 for the token default; DECISIONS.md D17)."""
        def judge(claim_text: str, passage: str):
            ans = llm.chat(
                "Does the PASSAGE support the CLAIM? Answer with exactly "
                f"one word: YES or NO.\nPASSAGE: {passage[:1500]}\n"
                f"CLAIM: {claim_text[:500]}",
                temperature=0.0).strip().upper()
            if ans.startswith("YES"):
                return True, "llm judge: supported"
            if ans.startswith("NO"):
                return False, "llm judge: not supported"
            return False, f"llm judge: unparseable ({ans[:60]})"
        return judge

    # -- slots ---------------------------------------------------------
    def _agent(self, task: Task, history: List) -> Dict[str, Any]:
        ctx = task.params.get("question_ctx", {})
        if not isinstance(ctx, dict):
            ctx = {}  # planner sometimes sends a string; gate needs a map
        fallback = {"query": task.params.get("query") or task.request,
                    "k": int(task.params.get("k", 5)),
                    "question_ctx": ctx}
        if history and self.llm is not None:
            from .llm_agent import llm_tool_call
            return llm_tool_call(self.llm, "literature_agent", task,
                                 history, fallback)
        return fallback

    def _tool(self, call: Dict[str, Any]) -> Dict[str, Any]:
        hits = self.search.search(call["query"], k=call["k"])
        claims = [{"text": h["snippet"],
                   "citations": [{"paper_id": h["doi"],
                                  "locator": h.get("section") or "snippet",
                                  "passage": h["snippet"],
                                  "context": {}}]}
                  for h in hits]
        return {"claims": claims, "n_hits": len(hits),
                "question_ctx": call["question_ctx"],
                "provenance": {"kind": "literature_cell",
                               "query": call["query"], "n_hits": len(hits)}}

    def _gate(self, raw: Dict[str, Any],
              call: Dict[str, Any]) -> VerificationReport:
        claims = raw.get("claims", [])
        if not claims:
            return report([check("claims_present", False,
                                 "no claims extracted from retrieval")])
        checks = []
        for claim in claims:
            rep = verify_claim(claim, raw.get("question_ctx", {}),
                               judge=self.judge)
            checks.extend(rep.checks)
        return report(checks)
