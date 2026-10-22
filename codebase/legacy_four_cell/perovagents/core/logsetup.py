"""Logging bootstrap — one call makes the whole agentic flow observable.

Without this, the `logging.getLogger(__name__)` calls scattered through the
package have no handler attached and every diagnostic (planner fallback,
tool failure, LLM key rotation) is silently dropped. Entry points call
`setup_logging()` once at startup; the rest of the codebase just logs.

Two sinks:
  * stderr (console) - so a terminal/`streamlit run` session shows the flow
  * `results/logs/agent.log` - rolling file for post-hoc inspection

Set the level with `PEROVAGENTS_LOG` (default INFO). Use `PEROVAGENTS_LOG`
=DEBUG to include prompt/response detail from the LLM and executor.

Example:
    from perovagents.core.logsetup import setup_logging
    setup_logging()                      # INFO to console + file
    log.info("about to run planner")     # module-level `log`
"""
import logging
import os
import sys

from .. import config

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
DATE_FORMAT = "%H:%M:%S"
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"

_configured = False


def setup_logging(level=None, log_dir=None, quiet_file=False):
    """Attach console (+ rolling file) handlers to the `perovagents` logger.

    Idempotent: calling twice does not duplicate handlers. Never raises - a
    logging problem must not stop a query from running.
    """
    global _configured
    if _configured:
        return logging.getLogger("perovagents")

    lvl = _resolve_level(level)
    root = logging.getLogger("perovagents")
    root.setLevel(lvl)
    # Don't duplicate records through the interpreter root logger.
    root.propagate = False

    fmt = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    try:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(fmt)
        root.addHandler(console)
    except Exception:
        pass

    if not quiet_file:
        try:
            path = log_dir or os.path.join(config.ensure_results("logs"),
                                           "agent.log")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(logging.Formatter(FILE_FORMAT))
            root.addHandler(fh)
        except Exception:
            pass  # file logging is best-effort

    _configured = True
    return root


def _resolve_level(level):
    if level is not None:
        if isinstance(level, str):
            return getattr(logging, level.upper(), logging.INFO)
        return level
    env = os.environ.get("PEROVAGENTS_LOG", "INFO").upper()
    return getattr(logging, env, logging.INFO)


def get_logger(name):
    """A `perovagents.*` child logger (auto-configured on first use)."""
    if not name.startswith("perovagents"):
        name = f"perovagents.{name}"
    return logging.getLogger(name)
