"""Provenance store — records every step the agent took.

Every tool invocation is appended as a `ProvenanceRecord`:
  - trace id (one per user question)
  - the tool name, params, status
  - a hash of the returned result (for integrity / diffing)
  - any source identifiers the tool reported (DOIs, model files, CSVs)

Records are held in memory for the session and can be flushed to JSONL in the
results directory so an answer can be audited end to end.
"""
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from .. import config


@dataclass
class ProvenanceRecord:
    trace_id: str
    step_index: int
    tool: str
    params: Dict[str, Any]
    status: str
    elapsed_seconds: float
    result_hash: str = ""
    sources: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


def _hash(obj):
    blob = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class ProvenanceStore:
    """Thread-safe-ish append-only store with optional JSONL persistence."""

    def __init__(self, persist_dir: Optional[str] = None):
        self._records: List[ProvenanceRecord] = []
        self.persist_dir = persist_dir or os.path.join(config.RESULTS_DIR, "provenance")

    # ---- writes ----------------------------------------------------------
    def new_trace_id(self) -> str:
        """Fresh trace id for a new user question."""
        return uuid.uuid4().hex[:12]

    def record(self, trace_id: str, step_index: int, tool: str,
               params: Dict[str, Any], status: str,
               elapsed_seconds: float = 0.0,
               result: Any = None,
               sources: Optional[List[str]] = None) -> ProvenanceRecord:
        rec = ProvenanceRecord(
            trace_id=trace_id, step_index=step_index, tool=tool, params=params,
            status=status, elapsed_seconds=elapsed_seconds,
            result_hash=_hash(result) if result is not None else "",
            sources=list(sources or []),
        )
        self._records.append(rec)
        self._flush([rec])
        return rec

    # ---- reads -----------------------------------------------------------
    def all(self) -> List[ProvenanceRecord]:
        """Every record appended so far (in order)."""
        return list(self._records)

    def for_trace(self, trace_id: str) -> List[ProvenanceRecord]:
        """Records belonging to one trace (one user question)."""
        return [r for r in self._records if r.trace_id == trace_id]

    def sources_used(self, trace_id: str) -> List[str]:
        """All source identifiers (DOIs, files, model paths) collected for a trace."""
        out = []
        for r in self.for_trace(trace_id):
            out.extend(r.sources)
        # de-dup preserving order
        seen, uniq = set(), []
        for s in out:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq

    # ---- persistence -----------------------------------------------------
    def _flush(self, records: List[ProvenanceRecord]):
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            path = os.path.join(self.persist_dir, "provenance.jsonl")
            with open(path, "a") as f:
                for r in records:
                    f.write(json.dumps(r.to_dict()) + "\n")
        except Exception:
            pass  # persistence must never crash a query

    def snapshot(self, trace_id: str) -> Dict[str, Any]:
        """Auditable export of a full trace (records + collected sources)."""
        return {
            "trace_id": trace_id,
            "n_steps": len(self.for_trace(trace_id)),
            "sources": self.sources_used(trace_id),
            "records": [r.to_dict() for r in self.for_trace(trace_id)],
        }


# convenience singleton
_STORE = None


def default_store() -> ProvenanceStore:
    """Process-wide singleton so all agents share one provenance log."""
    global _STORE
    if _STORE is None:
        _STORE = ProvenanceStore()
    return _STORE