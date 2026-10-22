"""Full-text PDF retrieval and reading (Literature layer).

Retrieves a paper's full text by DOI (or filename) from the local corpus and
returns it as clean prose. Falls back to a DOI->PDF download attempt from
Crossref + Unpaywall when the document is not cached locally (best-effort,
requires network).

Heavy lifting (`requests`) is imported lazily so offline use never breaks.
"""
import glob
import os
import re
from typing import Optional

from .. import config
from ..core.tool import Tool
from .search import doi_from_filename, extract_title, split_passages

# normalize a DOI to corpus-style filename
def doi_to_filename(doi: str) -> str:
    return doi.replace("/", "_") + ".md"


class FullTextTool(Tool):
    name = "fulltext"
    description = (
        "Retrieve the full text of a paper by DOI (or corpus filename) and "
        "optionally reduce it to the passages answering a query. Returns "
        "sections plus the source title and DOI. May attempt a network fetch "
        "if the paper is not cached locally."
    )
    layer = "literature"
    version = "0.1.0"
    capabilities = {"fulltext", "pdf", "literature"}

    def __init__(self):
        self._body_cache = {}
        self._docs = {}

    # ---- local corpus ----------------------------------------------------
    def _index_local(self):
        if self._docs:
            return
        for path in sorted(glob.glob(config.CORPUS_GLOB, recursive=True)):
            doi = doi_from_filename(path)
            self._docs[doi] = {"path": path, "doi": doi}

    def get_local(self, doi: str) -> Optional[str]:
        """Full text from the local corpus (fuzzy DOI matching), or None."""
        self._index_local()
        doi = doi.strip().replace("//", "/")
        for key, meta in self._docs.items():
            if key == doi or doi in key or key in doi:
                return open(meta["path"], errors="ignore").read()
        return None

    # ---- network fallback (lazy) -----------------------------------------
    def _fetch_network(self, doi: str) -> Optional[str]:
        """Best-effort PDF->text via Unpaywall/Crossref (needs `fitz` + `requests`)."""
        try:
            import fitz  # PyMuPDF
            import requests
        except ImportError:
            return None
        try:
            # 1) Crossref for metadata
            meta = requests.get(f"https://api.crossref.org/works/{doi}",
                                timeout=15, headers={"User-Agent": "perovagents/0.1"})
            if meta.status_code != 200:
                return None
            rmeta = meta.json()["message"]
            # 2) Unpaywall for open-access PDF
            up = requests.get("https://api.unpaywall.org/v2/" + doi,
                              params={"email": "perovagents@localhost"},
                              timeout=15).json()
            url = (up.get("best_oa_location") or {}).get("url_for_pdf")
            if not url:
                return None
            pdf = requests.get(url, timeout=30).content
            doc = fitz.open(stream=pdf, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            text = re.sub(r"\n{3,}", "\n\n", text.strip())
            return "# " + rmeta.get("title", [doi])[0] + "\n\n" + text
        except Exception:
            return None

    # ---- public API ------------------------------------------------------
    def fetch(self, doi: str, query: Optional[str] = None) -> dict:
        """Retrieve a paper's full text: local corpus first, then a network
        PDF fetch. Optionally filters to passages matching `query`."""
        doi = doi.strip()
        text = self.get_local(doi) or self._fetch_network(doi)
        if not text:
            return {"found": False, "doi": doi,
                    "reason": "not in local corpus and network fetch unavailable"}
        title = extract_title(text)
        result = {"found": True, "doi": doi, "title": title,
                  "n_chars": len(text), "sections": len(split_passages(text))}
        if query:
            hits = [{"section": h, "text": b} for h, b in split_passages(text)
                    if query.lower() in b.lower()]
            result["matching_passages"] = hits[:5]
        result["provenance"] = {"kind": "fulltext", "doi": doi,
                                "source": "local-corpus" if self.get_local(doi) else "network"}
        return result

    def call(self, params: dict) -> dict:
        doi = params.get("doi") or params.get("document") or params.get("id")
        if not doi:
            return {"error": "doi required"}
        return self.fetch(doi, query=params.get("query"))