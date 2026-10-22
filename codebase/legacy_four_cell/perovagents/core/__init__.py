"""Core package: shared abstractions for PerovAgents."""
from .tool import Tool, ExecutionResult, ValidationBlock
from .registry import (ToolRegistry, ToolKnowledgeGraph, ToolSpec, ActionSpec,
                       default_registry)
from .provenance import ProvenanceStore, ProvenanceRecord, default_store
from .flowlog import FlowLog, read_flow
from .sanity import SanityChecker, SanityReport, default_checker
from .validation import validate_scientific
from .contracts import match_output, match_outputs, OUTPUT_MATCHERS

__all__ = [
    "Tool", "ExecutionResult", "ValidationBlock",
    "ToolRegistry", "ToolKnowledgeGraph", "ToolSpec", "ActionSpec",
    "default_registry",
    "ProvenanceStore", "ProvenanceRecord", "default_store",
    "FlowLog", "read_flow",
    "SanityChecker", "SanityReport", "default_checker",
    "validate_scientific", "match_output", "match_outputs", "OUTPUT_MATCHERS",
]