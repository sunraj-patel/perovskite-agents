"""config.yaml loader (S7).

`load_config(path)` overlays the YAML file on code defaults (mirroring
`config.py`), so old imports, the CLI and the frontend keep working with no
file present. Nested dicts merge key-wise; unknown keys are ignored.
"""
import os
from typing import Any, Dict, Optional

from .. import config as defaults

CONFIG_NAME = "config.yaml"


def _defaults() -> Dict[str, Any]:
    return {
        "llm": {"model": defaults.LLM_MODEL,
                "temperature": defaults.LLM_TEMPERATURE},
        "cells": {"max_attempts": 3, "timeout_s": 300.0},
        "db": {"mirror": os.path.join(defaults.RESULTS_DIR, "sqlite",
                                       "perovskite.db"),
               "csv": defaults.PDB_CSV, "row_limit": 1000,
               "query_timeout_s": 10.0},
        "embeddings": {"corpus_dir": defaults.CORPUS_DIR,
                       "model": "all-MiniLM-L6-v2"},
        "scaps": {"path": defaults.SCAPS_DIR,
                  "existing_results": defaults.SCAPS_EXISTING_RESULTS},
        "ml": {"artifacts": os.path.join(defaults.RESULTS_DIR, "ml"),
               "metrics": os.path.join(defaults.RESULTS_DIR, "ml",
                                        "validation.json")},
        "trace": {"dir": os.path.join(defaults.RESULTS_DIR, "traces")},
    }


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif k in out and v is not None:
            out[k] = v
    return out


def default_path() -> str:
    """perovagents/config.yaml next to this package's config.py."""
    return os.path.join(os.path.dirname(defaults.__file__), CONFIG_NAME)


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load config.yaml over defaults (missing file = defaults)."""
    cfg = _defaults()
    path = path or default_path()
    if path and os.path.exists(path):
        import yaml
        with open(path, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg
