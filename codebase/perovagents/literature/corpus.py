"""Literature corpus: passages of the paper collection, searchable.

Loads the passage index cached by earlier runs (`results/literature/index.json`)
or builds it from the markdown papers in `config.CORPUS_DIR`. Search is
*semantic* (sentence-transformers vectors) when the model and the cached vectors
are available, and falls back to *keyword* scoring (TF-IDF cosine) otherwise. Every
result says which method produced it, so nothing is silently degraded.
"""
import glob
import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from .. import config
from .search import doi_from_filename, extract_title, split_passages

_STOP = set("the of and a an in to for with on by is are was were be as at from that this it its "
            "or which than then their these those has have had not can also using used use".split())


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]


def normalize(text: str) -> str:
    """Canonical form for verbatim-quote matching (case, markdown, quotes, dashes, spacing)."""
    t = str(text).lower()
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    t = re.sub(r"[*_`\\]|<[^>]+>", "", t)
    return re.sub(r"\s+", " ", t).strip()


class Corpus:
    def __init__(self, index_path: Optional[str] = None, corpus_glob: Optional[str] = None):
        self.index_path = index_path or os.path.join(config.ensure_results("literature"), "index.json")
        self.corpus_glob = corpus_glob or config.CORPUS_GLOB
        self._passages: Optional[List[Dict[str, Any]]] = None
        self._docs: Dict[str, Dict[str, str]] = {}
        self._vectors: Optional[np.ndarray] = None
        self._model = None
        self._model_failed = False
        self._df: Optional[Counter] = None
        self._toks: Optional[List[Counter]] = None

    # ---- loading ------------------------------------------------------------
    def _load(self) -> None:
        if self._passages is not None:
            return
        passages: List[Dict[str, Any]] = []
        if os.path.exists(self.index_path):
            with open(self.index_path, encoding="utf-8") as f:
                data = json.load(f)
            passages, self._docs = data["passages"], data.get("docs", {})
            vec = os.path.join(os.path.dirname(self.index_path), "vectors.npy")
            if os.path.exists(vec):
                v = np.load(vec)
                if len(v) == len(passages):
                    self._vectors = v
        else:
            for path in sorted(glob.glob(self.corpus_glob, recursive=True)):
                text = open(path, errors="ignore").read()
                doi = doi_from_filename(path)
                self._docs[doi] = {"doi": doi, "title": extract_title(text), "path": path}
                for head, body in split_passages(text):
                    passages.append({"doi": doi, "title": self._docs[doi]["title"],
                                     "section": head, "text": body})
        for i, p in enumerate(passages):
            p["id"] = i
        self._passages = passages
        if not self._docs:
            for p in passages:
                self._docs.setdefault(p["doi"], {"doi": p["doi"], "title": p.get("title", "")})

    @property
    def passages(self) -> List[Dict[str, Any]]:
        self._load()
        return self._passages

    def papers(self) -> List[Dict[str, Any]]:
        self._load()
        counts = Counter(p["doi"] for p in self._passages)
        return [{"doi": d, "title": self._docs.get(d, {}).get("title", ""), "n_passages": n}
                for d, n in sorted(counts.items())]

    # ---- search ---------------------------------------------------------------
    def _semantic(self, query: str) -> Optional[np.ndarray]:
        if self._vectors is None or self._model_failed:
            return None
        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            q = np.array(self._model.encode([query], normalize_embeddings=True)[0], dtype=np.float32)
            return self._vectors @ q
        except Exception:
            self._model_failed = True
            return None

    def _keyword_scores(self, query: str) -> np.ndarray:
        if self._toks is None:
            self._toks = [Counter(_tokens(p["text"] + " " + p.get("section", ""))) for p in self._passages]
            self._df = Counter(t for c in self._toks for t in c)
        n = len(self._passages)
        q = Counter(_tokens(query))
        idf = {t: math.log((n + 1) / (self._df.get(t, 0) + 1)) + 1 for t in q}
        scores = np.zeros(n, dtype=np.float32)
        qnorm = math.sqrt(sum((c * idf[t]) ** 2 for t, c in q.items())) or 1.0
        for i, c in enumerate(self._toks):
            dot = sum(cnt * idf[t] * c[t] * idf[t] for t, cnt in q.items() if t in c)
            if dot:
                scores[i] = dot / (qnorm * math.sqrt(sum((v * 1.0) ** 2 for v in c.values())))
        return scores

    def search(self, query: str, k: int = 6, doi: Optional[str] = None) -> Dict[str, Any]:
        self._load()
        if not self._passages:
            return {"error": "the literature corpus is empty (no papers indexed)"}
        sem = self._semantic(query)
        method = "semantic" if sem is not None else "keyword"
        scores = sem if sem is not None else self._keyword_scores(query)
        if doi:
            mask = np.array([p["doi"] == doi for p in self._passages])
            if not mask.any():
                return {"error": f"no paper with DOI {doi!r} in the corpus; use list_papers"}
            scores = np.where(mask, scores, -1.0)
        order = [i for i in np.argsort(-scores)[:k] if scores[i] > 0]
        hits = [{"doi": self._passages[i]["doi"], "paper_title": self._passages[i].get("title", ""),
                 "section": self._passages[i].get("section", ""), "passage_id": int(i),
                 "score": round(float(scores[i]), 3), "text": self._passages[i]["text"][:700]}
                for i in order]
        out: Dict[str, Any] = {"query": query, "method": method, "n_hits": len(hits), "hits": hits}
        if not hits:
            out["hint"] = "no passage matched; try different or fewer words, or synonyms"
        return out

    def read_paper(self, doi: str, query: Optional[str] = None, max_passages: int = 8) -> Dict[str, Any]:
        self._load()
        key = doi.strip().lower().replace("https://doi.org/", "")
        matches = [d for d in self._docs if d.lower() == key] or \
                  [d for d in self._docs if key in d.lower()]
        if not matches:
            return {"error": f"no paper with DOI {doi!r} in the corpus; use list_papers"}
        d = matches[0]
        ps = [p for p in self._passages if p["doi"] == d]
        if query:
            sc = self._keyword_scores(query)
            ps = sorted(ps, key=lambda p: -sc[p["id"]])[:max_passages]
            ps = sorted(ps, key=lambda p: p["id"])
        else:
            ps = ps[:max_passages]
        return {"doi": d, "title": self._docs[d].get("title", ""),
                "n_passages_total": sum(1 for p in self._passages if p["doi"] == d),
                "n_passages_returned": len(ps),
                "passages": [{"section": p.get("section", ""), "text": p["text"][:900]} for p in ps]}
