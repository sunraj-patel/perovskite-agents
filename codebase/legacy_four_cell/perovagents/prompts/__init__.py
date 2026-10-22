"""Prompt package: versioned agent prompts (S6). `load(name)` reads
`prompts/<name>.md`; callers keep inline fallbacks so missing files never
break a run."""
import os

_DIR = os.path.dirname(os.path.abspath(__file__))


def load(name: str, fallback: str = "") -> str:
    """Load a versioned prompt by stem (e.g. "planner")."""
    path = os.path.join(_DIR, name + ".md")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
            return text if text else fallback
    except OSError:
        return fallback
