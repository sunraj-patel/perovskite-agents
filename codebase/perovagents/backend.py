"""Minimal base class for the domain backends (ML predictor, materials, literature, SCAPS).

Backends do the science and return plain dicts. The agent-facing wrappers (tool
specifications, argument checking, error messages) live in `perovagents/agents/`.
"""
from typing import Any, Dict


class Backend:
    name = "backend"
    description = ""

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError
