"""DOI / metadata resolver (Literature layer).

Resolves a DOI to bibliographic metadata (title, journal, authors, year,
Abstract when open-access). Uses Crossref REST API with a local cache under
`results/literature/doi_cache.json` so lookups are fast and offline-friendly.

Also resolves to a *static landing* (publisher page) and, when the corpus
contains the paper, reports the local copy so retrieval can short-cut.
"""
import json
import os
from typing import Optional

from .. import config
from ..core.tool import Tool
from .search import doi_from_filename

CACHE_PATH = os.path.join(config.ensure_results("literature"), "doi_cache.json")


def _load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            return json.load(open(CACHE_PATH))
        except Exception:
            return {}
    return {}


class DOIMetadataTool(Tool):
    name = "doi"
    description = (
        "Resolve a DOI to bibliographic metadata (title, authors, journal, "
        "year) via Crossref, with caching. Returns a clean citation record and "
        "notes whether the paper is in the local corpus."
    )
    layer = "literature"
    version = "0.1.0"
    capabilities = {"doi_resolution", "metadata"}

    def resolve(self, doi: str) -> dict:
        """Crossref lookup for one DOI (cached on disk) + local-corpus flag."""
        doi = doi.strip()
        cache = _load_cache()
        if doi in cache:
            rec = cache[doi]
            rec["cached"] = True
            return rec
        try:
            import requests
            r = requests.get(f"https://api.crossref.org/works/{doi}",
                             timeout=15, headers={"User-Agent": "perovagents/0.1 (mailto:perovagents@localhost)"})
            if r.status_code != 200:
                return {"found": False, "doi": doi, "reason": f"crossref status {r.status_code}"}
            m = r.json()["message"]
            authors = [f"{a.get('given','')} {a.get('family','')}".strip()
                       for a in m.get("author", [])][:12]
            rec = {"found": True, "doi": doi,
                   "title": (m.get("title") or [""])[0],
                   "container": (m.get("container-title") or [""])[0],
                   "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                   "authors": authors,
                   "abstract": (m.get("abstract") or "").strip()[:800],
                   "in_local_corpus": self._in_corpus(doi)}
            cache[doi] = rec
            with open(CACHE_PATH, "w") as f:
                json.dump(cache, f, indent=1)
            rec["cached"] = False
            return rec
        except Exception as exc:
            return {"found": False, "doi": doi, "reason": f"network error: {exc}"}

    def _in_corpus(self, doi: str) -> bool:
        return os.path.exists(os.path.join(config.CORPUS_DIR, doi.replace("/", "_") + ".md"))

    def call(self, params: dict) -> dict:
        doi = params.get("doi")
        if not doi:
            return {"error": "doi required"}
        rec = self.resolve(doi)
        rec.setdefault("provenance", {"kind": "doi_resolution", "doi": doi.strip()})
        return rec