"""Core abstractions shared by every PerovAgents tool.

A `Tool` is the unit of scientific capability. Tools are *self-describing*:
each one knows its own name, description, layer, version, and the actions it
supports (declared in the registry). `execute()` is the canonical entry point —
it wraps `call()` with timing, error capture, provenance, and a structured
validation block, so every invocation returns the same envelope:

    {
      "status": "success|error",
      "tool":  "...",
      "action": "...",
      "result": {...},          # the domain payload (deterministic)
      "params": {...},
      "error": "...",
      "elapsed_seconds": ...,
      "provenance": {...},
      "validation": {"schema_valid": bool,
                     "units_valid": bool,
                     "domain_valid": bool,
                     "errors": [...], "warnings": [...]},
      "trace_id": "..."
    }
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ValidationBlock:
    """Layer-2 (schema) + layer-3 (scientific) validation of a result."""
    schema_valid: bool = True
    units_valid: bool = True
    domain_valid: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.schema_valid and self.units_valid and self.domain_valid

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_valid": self.schema_valid, "units_valid": self.units_valid,
                "domain_valid": self.domain_valid, "errors": list(self.errors),
                "warnings": list(self.warnings)}


@dataclass
class ExecutionResult:
    """The canonical structured envelope returned by every tool invocation."""
    tool: str
    action: str = "query"
    status: str = "success"               # success | error
    result: Optional[Dict[str, Any]] = None
    params: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    elapsed_seconds: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)
    validation: ValidationBlock = field(default_factory=ValidationBlock)
    trace_id: str = ""

    def to_dict(self):
        return {
            "tool": self.tool,
            "action": self.action,
            "status": self.status,
            "result": self.result,
            "params": self.params,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
            "provenance": self.provenance,
            "validation": self.validation.to_dict(),
            "trace_id": self.trace_id,
        }


class Tool(ABC):
    """Base class for all PerovAgents tools.

    Subclasses must set at least `name` and `description`, and implement
    `call(params) -> dict`. Tools should be *lazy*: import heavy libraries
    (catboost, sentence-transformers, scipy) inside methods, never at module
    import time.
    """
    name = "base"
    description = ""
    layer = "core"
    version = "0.1.0"
    # capabilities: set of tags used by the registry's tool KG for routing,
    # e.g. {"semantic_search", "citations"} or {"regression", "shap"}
    capabilities = frozenset()

    @abstractmethod
    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool and return a structured result dict."""

    def execute(self, params: Dict[str, Any], action: Optional[str] = None,
                trace_id: Optional[str] = None) -> ExecutionResult:
        """Wrapped entry point: timing + error handling + provenance +
        scientific validation. `action` names the invoked registry action
        (defaults to the value of params['action'], or 'query')."""
        action = action or params.get("action") or "query"
        t0 = time.time()
        try:
            result = self.call(params)
        except Exception as exc:
            return ExecutionResult(
                tool=self.name, action=action, status="error",
                error=str(exc)[:400], params=params,
                elapsed_seconds=time.time() - t0, trace_id=trace_id or "")
        if not isinstance(result, dict):
            result = {"result": result, "result_type": "raw"}
        return ExecutionResult(
            tool=self.name, action=action, status="success", result=result,
            params=params, elapsed_seconds=time.time() - t0,
            provenance=result.get("provenance") if isinstance(result, dict) else {},
            validation=self._validate_result(result),
            trace_id=trace_id or "")

    # ---- validation -------------------------------------------------------
    def _validate_result(self, result: Dict[str, Any]) -> ValidationBlock:
        """Scientific validation of the payload (units + domain bounds).
        Layer-2/3 checks: raise status if physics bounds or structural
        expectations are violated; non-blocking by design (values are still
        returned, but flagged so the answer can say 'flagged by sanity').
        """
        from .validation import validate_scientific
        ok, errors, warnings = validate_scientific(result)
        return ValidationBlock(schema_valid=isinstance(result, dict),
                               units_valid=ok, domain_valid=ok,
                               errors=errors, warnings=warnings)

    # ---- metadata helpers ------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "layer": self.layer,
            "version": self.version,
            "capabilities": sorted(self.capabilities),
        }

    def __repr__(self):
        return f"<Tool {self.name} v{self.version} [layer={self.layer}]>"