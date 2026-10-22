"""Per-question JSONL trace: every tool call, result, and report on disk.

`config.TRACE_DIR` was reserved (and gitignored) but nothing ever wrote to it.
This fills that in so a run can be audited after the fact:

    results/traces/trace_20260930_143012_ab12.jsonl
      {"ts": ..., "event": "question",  "text": "..."}
      {"ts": ..., "event": "ask",       "agent": "pdb", "task": "..."}
      {"ts": ..., "event": "tool_call", "agent": "pdb", "tool": "sql", "args": {...}}
      {"ts": ..., "event": "tool_result", "agent": "pdb", "result": "..."}
      {"ts": ..., "event": "report",    "agent": "pdb", "text": "..."}
      {"ts": ..., "event": "answer",    "text": "..."}

Writes are append-only and lock-guarded because the orchestrator runs agents on
separate threads. Truncation keeps a runaway tool result from filling the disk.
"""
import json
import os
import threading
import time

from . import config

MAX_FIELD_CHARS = 2000


def _shrink(value, limit=MAX_FIELD_CHARS):
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    return s if len(s) <= limit else s[:limit] + f"... [truncated {len(s) - limit} chars]"


class Trace:
    """Append-only event log for one question. A no-op when disabled."""

    def __init__(self, question=None, trace_dir=None, enabled=True):
        self.enabled = enabled
        self.path = None
        self._q_logged = False
        self._lock = threading.Lock()
        if not enabled:
            return
        d = trace_dir or config.TRACE_DIR
        try:
            os.makedirs(d, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.path = os.path.join(
                d, f"trace_{stamp}_{os.getpid() % 0xFFFF:04x}.jsonl")
        except OSError:
            self.enabled = False      # never let tracing break a query
        if question:
            self.ensure_question(question)

    def ensure_question(self, question):
        """Log the opening question once, no matter who asks first."""
        if self._q_logged:
            return
        self._q_logged = True
        self.event("question", text=question)

    def event(self, event, **fields):
        if not self.enabled or not self.path:
            return
        row = {"ts": round(time.time(), 3), "event": event}
        for k, v in fields.items():
            if v is not None:
                row[k] = _shrink(v)
        line = json.dumps(row, default=str)
        with self._lock:
            try:
                with open(self.path, "a") as f:
                    f.write(line + "\n")
            except OSError:
                self.enabled = False

    def close(self):
        self.enabled = False


def read_trace(path):
    """Load a trace file back as a list of dicts (skips malformed lines)."""
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        pass
    return rows
