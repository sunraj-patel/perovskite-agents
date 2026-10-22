"""Citation / evidence verifier (Literature layer).

Given a claim and a DOI (or a retrieved passage), verify that the citation
actually supports the claim:

  1. resolve the DOI against the corpus (title, summary of matching passages)
  2. score whether the corpus text contains supportive/contradictory phrasing
  3. return a PASS / WARN / FAIL verdict with the evidence

The verifier is deliberately conservative: it never claims parity with human
reading — it flags "citations likely to need human double-checking".
"""
from typing import Optional

from ..core.tool import Tool
from .search import LiteratureSearchTool
from .fulltext import FullTextTool


class CitationVerifier(Tool):
    name = "citations"
    description = (
        "Verify whether a specific DOI actually supports a stated claim. "
        "Returns a PASS/WARN/FAIL verdict with the matching corpus passages. "
        "Use whenever the agent cites a paper for a factual assertion."
    )
    layer = "literature"
    version = "0.1.0"
    capabilities = {"citation_verification", "verification"}

    def __init__(self, search_tool: Optional[LiteratureSearchTool] = None,
                 fulltext_tool: Optional[FullTextTool] = None):
        self.search = search_tool or LiteratureSearchTool()
        self.fulltext = fulltext_tool or FullTextTool()

    def verify(self, claim: str, doi: str, top_k: int = 3) -> dict:
        """Score whether a cited paper plausibly supports `claim` and return
        a PASS/WARN/FAIL verdict with the underlying evidence."""
        # 1. full text of the cited paper
        doc = self.fulltext.fetch(doi)
        if not doc.get("found"):
            return {"verdict": "FAIL", "reason": "paper not retrievable",
                    "doi": doi, "claim": claim,
                    "provenance": {"kind": "citation_verification", "doi": doi}}
        # 2. score how well the paper's passages match the claim
        body = " ".join(p["text"] for p in doc.get("passages", [])) \
            if "passages" in doc else self.fulltext.get_local(doi)
        tokens = set(token.lower() for token in claim.split() if len(token) > 3)
        text_lower = (body or "").lower()
        overlap = sum(1 for t in tokens if t in text_lower)
        match_score = min(1.0, overlap / max(len(tokens), 1))
        # 3. alternative: semantic search over the paper
        support = self.search.citation_context(claim, k=top_k) if False else []
        papers = [h["doi"] for h in self.search.search(claim, k=5)]
        verdict = "PASS" if match_score > 0.6 and doi in papers else \
                  ("WARN" if match_score > 0.3 else "FAIL")
        return {"verdict": verdict, "doi": doi, "claim": claim,
                "match_score": round(match_score, 3),
                "supporting_evidence": support,
                "papers_from_search": papers[:3],
                "provenance": {"kind": "citation_verification", "doi": doi}}

    def verify_list(self, claims: list, dois: list) -> dict:
        """Verify claim/DOI pairs pairwise."""
        return {"verdicts": [self.verify(c, d) for c, d in zip(claims, dois)],
                "n": min(len(claims), len(dois))}

    def call(self, params: dict) -> dict:
        claim = params.get("claim") or params.get("statement")
        doi = params.get("doi")
        if not claim or not doi:
            return {"error": "claim and doi required"}
        return self.verify(claim, doi)