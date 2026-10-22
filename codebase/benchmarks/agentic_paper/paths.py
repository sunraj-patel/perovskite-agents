"""Paths and environment for the benchmark.

All outputs live under ``benchmarks/agentic_paper/``. The scientific data the
tools read (SQLite mirror of the Perovskite Database, literature corpus,
SCAPS precomputed sweeps) are large and are NOT part of this snapshot; they are
located through environment variables, defaulting to the development machine.
Override any of them before running::

    export PEROVAGENTS_PDB_SQLITE=/path/to/perovskite.db
    export PEROVAGENTS_CORPUS_DIR=/path/to/markdown_papers
    export PEROVAGENTS_SCAPS_DIR=/path/to/scaps_dir   # holds sweep_results.csv

No secrets are read or written here. API keys are resolved at run time by
``providers.py`` and are never logged.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                       # codebase/
sys.path.insert(0, str(REPO))

QUESTIONS_DIR = HERE / "questions"
REFERENCE_DIR = HERE / "reference"
CONFIG_DIR = HERE / "config"
CASE_DIR = HERE / "case_studies"

# Benchmark version. v2 = first batch (default names, closed, preserved); v3 = framework fixes +
# instrumented providers; v4 = v3 + neutral protocol-compatible direct-baseline prompt + new unseen
# database/literature questions. Each version writes under its own dir so results are never pooled.
VERSION = os.environ.get("BENCH_VERSION", "v4")
_OUT = HERE if VERSION == "v2" else HERE / VERSION
RUNS_DIR = _OUT / "runs"                        # raw episodes + traces + manifest
SCORED_DIR = _OUT / "scored"
TABLES_DIR = _OUT / "tables"
FIG_DIR = _OUT / "figures"
AUDIT_DIR = _OUT / "audit"
FIGAUDIT_DIR = _OUT / "figures_audit"
QUESTIONS_FILE = QUESTIONS_DIR / f"questions_{VERSION}.json"
SPLIT_FILE = CONFIG_DIR / f"split_{VERSION}.json"
FREEZE_PATH = HERE / ("FREEZE.json" if VERSION == "v2" else f"FREEZE_{VERSION}.json")

# Development-machine defaults for the large external data (override via env).
_UGP = Path("/Users/sunraj/Desktop/College/S6/UGP")
_DEFAULTS = {
    "PEROVAGENTS_PDB_SQLITE": _UGP / "perovagents/results/sqlite/perovskite.db",
    "PEROVAGENTS_CORPUS_DIR": _UGP / "sandbox/markdown",
    "PEROVAGENTS_SCAPS_DIR": _UGP / "opencode_sandbox/scaps",
    "PEROVAGENTS_PDB_CSV": _UGP / "opencode_sandbox/perovskite_db.csv",
    "PEROVAGENTS_ENRICHED_CSV": _UGP / "opencode_sandbox/dataset_json_enriched.csv",
}


def setup_env():
    """Point the framework at its data and DISABLE every unlabelled fallback.

    * PEROVAGENTS_OPENCODE_FALLBACK=0 : the framework's own LLM class would
      otherwise fall back from Gemini to ``opencode serve`` and then to a
      deterministic responder. The benchmark never uses that class for
      generation (providers.py is strict), but the switch is set as a guard.
    """
    for k, v in _DEFAULTS.items():
        os.environ.setdefault(k, str(v))
    os.environ["PEROVAGENTS_OPENCODE_FALLBACK"] = "0"
    for d in (RUNS_DIR, SCORED_DIR, TABLES_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
