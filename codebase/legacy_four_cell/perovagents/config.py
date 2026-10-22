"""PerovAgents — central configuration.

Every path and API key in the framework reads from this module. Defaults point
at the repository layout (the data lives outside the package):

    <repo root>/opencode_sandbox/perovskite_db.csv        (PDB)
    <repo root>/opencode_sandbox/dataset_json_enriched   (enriched)
    <repo root>/sandbox/markdown                       (literature corpus)
    <repo root>/sandbox/materials_*.csv                (lit. property tables)

All values can be overridden with environment variables so the framework can be
pointed at fresh data without touching code.
"""
import os

# ----------------------------------------------------------------------------
# Paths (auto-derived from this file's location)
# ----------------------------------------------------------------------------
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)          # perovagents/
REPO_ROOT = os.path.dirname(PROJECT_ROOT)            # UGP/

RESULTS_DIR = os.path.realpath(os.environ.get(
    "PEROVAGENTS_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results")))

# Data files ---------------------------------------------------------------
def _env_or(path, env):
    return os.environ.get(env, path) or path

PDB_CSV = _env_or(os.path.join(REPO_ROOT, "opencode_sandbox", "perovskite_db.csv"),
                  "PEROVAGENTS_PDB_CSV")
ENRICHED_CSV = _env_or(
    os.path.join(REPO_ROOT, "opencode_sandbox", "dataset_json_enriched.csv"),
    "PEROVAGENTS_ENRICHED_CSV")

LIT_PROPERTIES_CSV_1 = os.path.join(REPO_ROOT, "sandbox", "materials_structured.csv")
LIT_PROPERTIES_CSV_2 = os.path.join(REPO_ROOT, "sandbox", "materials_properties.csv")

CORPUS_DIR = _env_or(os.path.join(REPO_ROOT, "sandbox", "markdown"),
                     "PEROVAGENTS_CORPUS_DIR")
CORPUS_GLOB = os.path.join(CORPUS_DIR, "**", "*.md")

# SCAPS --------------------------------------------------------------------
SCAPS_DIR = os.path.join(REPO_ROOT, "opencode_sandbox", "scaps")
SCAPS_EXISTING_RESULTS = os.path.join(SCAPS_DIR, "sweep_results.csv")
SCAPS_MAKE_DEF = os.path.join(SCAPS_DIR, "make_perovskite_def.py")
SCAPS_WINE = "/Users/sunraj/Applications/Age of Empires 2 Definitive Ed.app/Contents/SharedSupport/wine/bin/wine64"
SCAPS_PREFIX = os.path.join(SCAPS_DIR, "prefix")

# LLM ----------------------------------------------------------------------
LLM_MODEL = os.environ.get("PEROVAGENTS_LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = 0.0
CHATBOT_ENV_FILE = os.path.join(REPO_ROOT, "chatbot", ".env")
KEYS_MD_FILE = os.path.expanduser("~/keys.md")

# opencode `serve` fallback was removed (Gemini-only). Keys rotate in a
# simple round-robin circle over every configured key.

# ----------------------------------------------------------------------------
# Physics bounds — used by the scientific sanity checker (core/sanity.py)
# ----------------------------------------------------------------------------
PHYSICS_BOUNDS = {
    "JV_default_PCE": {"min": 0.0, "max": 30.0,
                       "warn": "PCE > 30% is likely an error"},
    "JV_default_Voc": {"min": 0.0, "max": 1.4,
                       "warn": "Voc > 1.4V may indicate tandem/error"},
    "JV_default_Jsc": {"min": 0.0, "max": 32.0,
                       "warn": "Jsc > 32 mA/cm2 is unusually high"},
    "JV_default_FF":  {"min": 0.2, "max": 0.95,
                       "warn": "FF outside 0.2-0.95 indicates failure"},
}

# Performance aliases: user-facing name -> canonical DB column --------------
PERF_ALIAS = {
    "PCE": "JV_default_PCE", "power conversion efficiency": "JV_default_PCE",
    "efficiency": "JV_default_PCE",
    "Voc": "JV_default_Voc", "open circuit voltage": "JV_default_Voc",
    "Jsc": "JV_default_Jsc", "short circuit current": "JV_default_Jsc",
    "FF": "JV_default_FF", "fill factor": "JV_default_FF",
}

ARCH_ALIAS = {
    "n-i-p": "nip", "nip": "nip", "p-i-n": "pin", "pin": "pin",
    "planar": None, "meso": None,
}


def ensure_results(sub=None):
    """Return (and create) a results subdirectory."""
    path = RESULTS_DIR if sub is None else os.path.join(RESULTS_DIR, sub)
    os.makedirs(path, exist_ok=True)
    return path