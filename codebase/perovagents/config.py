"""PerovAgents configuration: every path and limit in one place.

All values can be overridden with environment variables, so the framework can
be pointed at other data without touching code.

    <project root>/results/sqlite/perovskite.db   Perovskite Database (SQLite mirror)
    <project root>/results/ml/                    trained models + validation.json
    <project root>/results/literature/            literature passages + embeddings
    <repo root>/opencode_sandbox/scaps/           SCAPS-1D sweeps (+ live-run tooling)
"""
import os
import shutil

# ---------------------------------------------------------------- paths
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)            # perovagents/  (repo dir)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)              # parent of the repo

RESULTS_DIR = os.path.realpath(os.environ.get(
    "PEROVAGENTS_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results")))


def _path(env, default):
    return os.environ.get(env) or default


# Perovskite Database: the raw CSV is only needed to (re)build the SQLite mirror.
PDB_CSV = _path("PEROVAGENTS_PDB_CSV",
                os.path.join(REPO_ROOT, "opencode_sandbox", "perovskite_db.csv"))
PDB_SQLITE = _path("PEROVAGENTS_PDB_SQLITE",
                   os.path.join(RESULTS_DIR, "sqlite", "perovskite.db"))

# Enriched table + literature property tables (used by the ML feature pipeline).
ENRICHED_CSV = _path("PEROVAGENTS_ENRICHED_CSV",
                     os.path.join(REPO_ROOT, "opencode_sandbox",
                                  "dataset_json_enriched.csv"))
LIT_PROPERTIES_CSV_1 = _path("PEROVAGENTS_LIT_PROPS_1",
                             os.path.join(REPO_ROOT, "sandbox", "materials_structured.csv"))
LIT_PROPERTIES_CSV_2 = _path("PEROVAGENTS_LIT_PROPS_2",
                             os.path.join(REPO_ROOT, "sandbox", "materials_properties.csv"))

# Literature corpus (markdown full texts); passages + embeddings are cached
# under RESULTS_DIR/literature.
CORPUS_DIR = _path("PEROVAGENTS_CORPUS_DIR", os.path.join(REPO_ROOT, "sandbox", "markdown"))
CORPUS_GLOB = os.path.join(CORPUS_DIR, "**", "*.md")

# ML artifacts (models, feature manifests, validation metrics).
MODEL_DIR = _path("PEROVAGENTS_ML_MODEL_DIR", os.path.join(RESULTS_DIR, "ml"))
ML_VALIDATION_JSON = os.path.join(MODEL_DIR, "validation.json")

# SCAPS-1D. Precomputed sweeps are read from a CSV; live runs need wine + SCAPS.
SCAPS_DIR = _path("PEROVAGENTS_SCAPS_DIR", os.path.join(REPO_ROOT, "opencode_sandbox", "scaps"))
SCAPS_EXISTING_RESULTS = _path("PEROVAGENTS_SCAPS_RESULTS",
                               os.path.join(SCAPS_DIR, "sweep_results.csv"))
SCAPS_MAKE_DEF = os.path.join(SCAPS_DIR, "make_perovskite_def.py")


def _find_wine() -> str:
    """Locate a wine64 that can actually start scaps3309.exe.

    PATH lookup is not good enough here: Homebrew's `wine` formula on recent
    macOS installs only the `wine-stable` *wrapper* (a shell stub that re-execs
    a bottled wine the user may not have), and picking that up silently breaks
    every live run. The CrossOver runtime shipped with Age of Empires II is a
    complete, self-contained wine64 that already has SCAPS working, so it is
    tried first and PATH is only a last resort.
    """
    override = os.environ.get("PEROVAGENTS_SCAPS_WINE")
    if override:
        return _path("PEROVAGENTS_SCAPS_WINE", override)
    candidates = [
        # CrossOver runtime bundled with Age of Empires II (verified working).
        os.path.expanduser(
            "~/Applications/Age of Empires 2 Definitive Ed.app/Contents/"
            "SharedSupport/wine/bin/wine64"),
        "/Applications/Age of Empires II.app/Contents/SharedSupport/wine/bin/wine64",
    ]
    # Whisky/wswine bundles shipped inside game apps (self-contained wine64;
    # verified running SCAPS to convergence). NOTE: some apps also drop a
    # `wine` shell launcher on PATH that hardcodes THEIR prefix - never use
    # the launcher, always the bundle's bin/wine64 with our own WINEPREFIX.
    for app in ("F1 22 AnkerGames.app",):
        for base in (os.path.expanduser("~/Applications"), "/Applications"):
            candidates.append(os.path.join(
                base, app, "Contents", "Frameworks", "wswine.bundle", "bin", "wine64"))
    # Any other Steam/CrossOver install that ships the same runtime layout.
    for app in ("CrossOver", "Wine Crossover"):
        candidates.append(f"/Applications/{app}.app/Contents/SharedSupport/wine/bin/wine64")
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return shutil.which("wine64") or shutil.which("wine") or ""


SCAPS_WINE = _find_wine()
SCAPS_PREFIX = _path("PEROVAGENTS_SCAPS_PREFIX", os.path.join(SCAPS_DIR, "prefix"))


def _runtime_lib_dirs(wine: str) -> list:
    """Library dirs a bundled wine64 needs on DYLD_FALLBACK_LIBRARY_PATH.

    A Sikarugir/Whisky `wswine.bundle` is normally started by its app's
    launcher, which exports the app's Frameworks dir. Calling bin/wine64
    directly skips that, wine cannot dlopen FreeType ("Wine cannot find the
    FreeType font library") and scaps3309.exe then spins at ~100% CPU forever
    without writing an IV file. Reproduced and fixed with exactly this
    variable (see README, "SCAPS under wine").
    """
    if not wine:
        return []
    bundle = os.path.dirname(os.path.dirname(os.path.realpath(wine)))   # .../wswine.bundle
    if os.path.basename(bundle) != "wswine.bundle":
        return []
    frameworks = os.path.dirname(bundle)
    return [d for d in (frameworks, os.path.join(bundle, "lib")) if os.path.isdir(d)]


def scaps_env(base=None) -> dict:
    """Environment for every wine/SCAPS launch (single source of truth).

    PEROVAGENTS_SCAPS_DYLD (colon-separated) replaces the auto-derived
    library dirs, for runtimes whose layout is not a wswine.bundle.
    """
    env = dict(os.environ if base is None else base, WINEPREFIX=SCAPS_PREFIX)
    override = os.environ.get("PEROVAGENTS_SCAPS_DYLD")
    dirs = [d for d in override.split(":") if d] if override else _runtime_lib_dirs(SCAPS_WINE)
    if dirs:
        env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
            dirs + [p for p in env.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":") if p])
    return env
# SCAPS is a GUI process under wine: it keeps running after the calculation
# finishes, and two copies sharing one script/result path corrupt each other.
SCAPS_TIMEOUT = int(os.environ.get("PEROVAGENTS_SCAPS_TIMEOUT", "600"))


# ---------------------------------------------------------------- LLM
LLM_MODEL = os.environ.get("PEROVAGENTS_LLM_MODEL", "gemini-3.8-flash")
LLM_TEMPERATURE = 0.0
CHATBOT_ENV_FILE = os.path.join(REPO_ROOT, "chatbot", ".env")
KEYS_MD_FILE = os.path.expanduser("~/keys.md")

# opencode `serve` fallback (used when the Gemini key pool is exhausted).
OPENCODE_URL = os.environ.get("PEROVAGENTS_OPENCODE_URL", "http://127.0.0.1:4096")
OPENCODE_MODEL = os.environ.get("PEROVAGENTS_OPENCODE_MODEL",
                                "opencode/muse-spark-1.3-contributor-free")
try:
    OPENCODE_TIMEOUT_S = float(os.environ.get("PEROVAGENTS_OPENCODE_TIMEOUT_S", "300"))
except ValueError:
    OPENCODE_TIMEOUT_S = 300.0
OPENCODE_FALLBACK = os.environ.get("PEROVAGENTS_OPENCODE_FALLBACK", "1") == "1"
OPENCODE_USERNAME = os.environ.get("OPENCODE_SERVER_USERNAME", "opencode")
OPENCODE_PASSWORD = os.environ.get("OPENCODE_SERVER_PASSWORD", "")

# ---------------------------------------------------------------- agent loop


def _num(env, default, cast=float):
    try:
        return cast(os.environ.get(env, default))
    except ValueError:
        return cast(default)


AGENT_MAX_STEPS = _num("PEROVAGENTS_MAX_STEPS", 15, int)         # tool calls per subagent
AGENT_MAX_REJECTIONS = _num("PEROVAGENTS_MAX_REJECTIONS", 3, int)  # gate rejections
AGENT_TIMEOUT_S = _num("PEROVAGENTS_AGENT_TIMEOUT_S", 600.0)     # wall clock per subagent
MAX_PARALLEL = _num("PEROVAGENTS_MAX_PARALLEL", 4, int)          # concurrent subagents
SQL_ROW_LIMIT = _num("PEROVAGENTS_SQL_ROW_LIMIT", 200, int)      # rows returned per query
SQL_TIMEOUT_S = _num("PEROVAGENTS_SQL_TIMEOUT_S", 20.0)
TRACE_DIR = _path("PEROVAGENTS_TRACE_DIR", os.path.join(RESULTS_DIR, "traces"))
SESSION_DIR = _path("PEROVAGENTS_SESSION_DIR", os.path.join(RESULTS_DIR, "sessions"))


def ensure_results(sub=None):
    """Return (and create) a results subdirectory."""
    path = RESULTS_DIR if sub is None else os.path.join(RESULTS_DIR, sub)
    os.makedirs(path, exist_ok=True)
    return path
