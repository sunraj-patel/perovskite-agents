"""Literature search tool (Literature layer).

Semantic search over the project's corpus of PSC papers. The corpus is a set
of markdown/PDF full texts in `config.CORPUS_DIR`. Passages are chunked,
embedded (sentence-transformers), and retrieved by cosine similarity. Every hit
carries its source DOI and paper title so answers can cite evidence.

Lazy: embeddings are only computed on first use and cached under
`results/literature/index.json`.
"""
import glob
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .. import config
from ..backend import Backend


def doi_from_filename(fname: str) -> str:
    """Convert a corpus filename like `10.1002_adfm.201604733.md` to a DOI."""
    base = os.path.basename(fname)
    if base.endswith(".md"):
        base = base[:-3]
    return base.replace("_", "/")


def extract_title(text: str) -> str:
    """Pull the first markdown H1 as the paper title."""
    m = re.search(r"#\s+(?:\*\*)?(.+)", text)
    return m.group(1).strip() if m else ""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_passages(text: str, max_len: int = 600):
    """Split a paper's text into (heading, body) passages with overlap."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks, cur, head = [], [], "Intro"
    for line in text.splitlines():
        hm = re.match(r"^(#{1,4})\s+(.*)$", line)
        if hm:
            if cur:
                chunks.append((head, _clean(" ".join(cur))))
                cur = []
            head = hm.group(2).strip(" *#")
            continue
        cur.append(line)
    if cur:
        chunks.append((head, _clean(" ".join(cur))))
    out = []
    for h, body in chunks:
        if len(body) < 140 or not re.search(r"[A-Za-z]{3,}", body):
            continue
        if len(body) <= max_len:
            out.append((h, body))
        else:
            for i in range(0, len(body), max_len // 2):
                seg = body[i:i + max_len]
                if len(seg) < 140:
                    break
                out.append((h, seg))
    return out


class LiteratureSearchTool(Backend):
    name = "literature"
    description = (
        "Semantic search over the perovskite solar cell literature corpus (full "
        "papers). Find mechanistic explanations, reported materials, processing "
        "recipes, experimental claims. Returns relevant passages with their DOI "
        "so answers can cite primary literature."
    )
    layer = "literature"
    version = "0.2.0"
    capabilities = {"semantic_search", "citations", "literature"}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._passages: Optional[list] = None
        self._vectors: Optional[np.ndarray] = None
        self._docs: Optional[dict] = None

    # ---- corpus loading --------------------------------------------------
    @property
    def passages(self):
        if self._passages is None:
            self._load()
        return self._passages

    @property
    def vectors(self):
        """Embedding matrix for all passages (built lazily)."""
        if self._vectors is None:
            self._load()
        return self._vectors

    def _load(self):
        cache = os.path.join(config.ensure_results("literature"), "index.json")
        if os.path.exists(cache):
            data = json.load(open(cache))
            self._passages = data["passages"]
            self._docs = data.get("docs", {})
            vec_file = os.path.join(config.ensure_results("literature"), "vectors.npy")
            if os.path.exists(vec_file):
                self._vectors = np.load(vec_file)
                return
        docs, passages = {}, []
        for path in sorted(glob.glob(config.CORPUS_GLOB, recursive=True)):
            text = open(path, errors="ignore").read()
            doi = doi_from_filename(path)
            docs[doi] = {"path": path, "doi": doi, "title": extract_title(text)}
            for head, body in split_passages(text):
                passages.append({"doi": doi, "title": docs[doi]["title"],
                                 "section": head, "text": body})
        self._docs, self._passages = docs, passages
        self._vectors = self._embed([p["text"] for p in passages])
        with open(cache, "w") as f:                      # persist passages (not vectors)
            json.dump({"passages": passages, "docs": docs}, f)
        np.save(os.path.join(config.ensure_results("literature"), "vectors.npy"),
                self._vectors)

    def _embed(self, texts: List[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.model_name)
        return np.array(model.encode(texts, batch_size=64, normalize_embeddings=True),
                        dtype=np.float32)

    def _embed_one(self, text: str) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.model_name)
        return np.array(model.encode([text], normalize_embeddings=True)[0],
                        dtype=np.float32)

    # ---- search API ------------------------------------------------------
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Cosine-similarity top-k passages for a query (DOI+title+section)."""
        qv = self._embed_one(query)
        sims = self.vectors @ qv
        order = np.argsort(-sims)[:k]
        return [{
            "doi": self.passages[i]["doi"],
            "paper_title": self.passages[i]["title"],
            "section": self.passages[i]["section"],
            "snippet": self.passages[i]["text"][:600],
            "score": float(sims[i]),
        } for i in order]

    def citation_context(self, query: str, k: int = 3) -> Dict:
        """Top-k passages shaped as citation evidence for a claim."""
        hits = self.search(query, k=k)
        return {"evidence": [{"doi": h["doi"], "paper": h["paper_title"],
                              "text": h["snippet"]} for h in hits],
                "n_evidence": len(hits)}

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query") or params.get("q")
        if not query:
            return {"error": "query required"}
        k = int(params.get("k", 5))
        hits = self.search(query, k=k)
        return {"result_type": "literature_passages", "query": query,
                "n_hits": len(hits), "hits": hits,
                "provenance": {"kind": "literature_search",
                               "corpus": config.CORPUS_DIR,
                               "n_papers": len(self._docs or {})}}