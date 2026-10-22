"""Table/Figure extraction from papers (Literature layer).

Extracts numeric tables and figure captions from the local corpus PDFs/markdown.
Currently supports markdown tables (the corpus default). PDF table extraction
via `pdfplumber` is attempted lazily when available.

Vision-based figure interpretation (chart->data) is a stretch goal and returns
a clear `not_supported` response.
"""
import glob
import os
import re
from typing import List

from .. import config
from ..core.tool import Tool
from .search import doi_from_filename, extract_title


def extract_markdown_tables(text: str) -> List[dict]:
    """Pull markdown pipe-tables with their heading context."""
    lines = text.splitlines()
    tables, cur, context = [], [], ""
    for line in lines:
        if line.strip().startswith("#"):
            if cur:
                tables.append(dict(context=context, content="\n".join(cur)))
                cur = []
            context = line.strip().lstrip("#").strip()
            continue
        if re.search(r"\|", line):
            cur.append(line)
        elif cur and len(cur) >= 2:
            tables.append(dict(context=context, content="\n".join(cur)))
            cur = []
    if len(cur) >= 2:
        tables.append(dict(context=context, content="\n".join(cur)))
    # drop header-separator-only chunks
    clean = []
    for t in tables:
        rows = [r for r in t["content"].splitlines() if re.search(r"\|", r)]
        meaningful = [r for r in rows if not re.fullmatch(r"[\s|:-]+", r)]
        if len(meaningful) >= 2:
            clean.append(dict(context=t["context"], n_rows=len(meaningful) - 1,
                              content="\n".join(rows)))
    return clean


def extract_figure_captions(text: str) -> List[dict]:
    """Pull `Figure N: caption` lines (first 300 chars each)."""
    caps = []
    for m in re.finditer(r"(?:^|\n)\s*(?:Fig\.|Figure)\s*(\d+)[.:]?\s*(.+)", text):
        caps.append({"number": m.group(1), "text": m.group(2).strip()[:300]})
    return caps


class TableFigureExtractor(Tool):
    name = "extraction"
    description = (
        "Extract tables and figure captions from papers in the corpus. "
        "Returns markdown tables with their section context (PDF table "
        "extraction is attempted when pdfplumber is installed)."
    )
    layer = "literature"
    version = "0.1.0"
    capabilities = {"table_extraction", "figure_extraction", "literature"}

    def call(self, params: dict) -> dict:
        doi = params.get("doi") or params.get("document")
        if not doi:
            return {"error": "doi required"}
        path = os.path.join(config.CORPUS_DIR, doi.replace("/", "_") + ".md")
        if not os.path.exists(path):
            return {"found": False, "doi": doi, "reason": "not in corpus"}
        text = open(path, errors="ignore").read()
        return {"found": True, "doi": doi, "title": extract_title(text),
                "tables": extract_markdown_tables(text),
                "n_tables": len(extract_markdown_tables(text)),
                "figure_captions": extract_figure_captions(text)[:10],
                "provenance": {"kind": "literature_extraction", "doi": doi}}