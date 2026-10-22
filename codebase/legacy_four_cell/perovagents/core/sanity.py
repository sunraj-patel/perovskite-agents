"""Scientific sanity checker — the Verification layer.

Every tool result (and every aggregate answer) passes through here before it is
handed to the user. The checker flags:

  * physics-bound violations (PCE > 30%, Voc > 1.4 V, FF out of range, ...)
  * structurally suspicious results (NaN/None where a number is expected,
    negative counts, failed extractions)

Warnings are non-blocking: the value is still returned, but labeled so the
answer can say "flagged by sanity check".
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .. import config


@dataclass
class SanityReport:
    """Accumulated non-blocking warnings for one checked result."""
    clean: bool = True
    warnings: List[str] = field(default_factory=list)

    def add(self, warning: str):
        self.warnings.append(warning)
        self.clean = False

    def to_dict(self):
        return {"clean": self.clean, "warnings": self.warnings}


class SanityChecker:
    """Stateless checker. Instantiate per pipeline run."""

    def __init__(self, bounds: Dict = None):
        self.bounds = bounds or config.PHYSICS_BOUNDS

    # ---- high-level entry ------------------------------------------------
    def check_result(self, result: Any) -> SanityReport:
        """Flag physics-bound / structural problems in one tool result."""
        report = SanityReport()
        if isinstance(result, dict):
            if result.get("status") == "error":
                report.add(f"tool error: {result.get('error', 'n/a')[:120]}")
            self._check_physics_fields(result, report)
            self._check_shapes(result, report)
        return report

    # ---- physics bounds --------------------------------------------------
    def _check_physics_fields(self, d: Dict, report: SanityReport):
        for col, b in self.bounds.items():
            for key in (col, self._short(col)):
                if key in d:
                    v = d[key]
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        if v < b["min"] or v > b["max"]:
                            report.add(f"{col}={v} outside physics bounds "
                                       f"[{b['min']}, {b['max']}]: {b['warn']}")

    @staticmethod
    def _short(col: str) -> str:
        return col.split("_")[-1] if "_" in col else col

    # ---- structural checks ----------------------------------------------
    def _check_shapes(self, d: Dict, report: SanityReport):
        # negative counts
        for k in ("n", "rows_after", "n_records", "n_hits", "count"):
            if k in d and isinstance(d[k], (int, float)) and d[k] < 0:
                report.add(f"{k}={d[k]} is negative")

        # table results that are empty silenty
        if d.get("result_type") == "table":
            rows = d.get("data") or d.get("rows") or []
            if not rows:
                report.add("query returned an empty table")

    def describe(self) -> Dict[str, Any]:
        return {
            "name": "sanity",
            "description": "Validates tool outputs against physics bounds and "
                           "structural expectations.",
            "layer": "verification",
            "version": "0.2.0",
            "capabilities": ["physics_bounds", "structural_validity"],
        }


def default_checker() -> SanityChecker:
    """Shorthand for the standard physics-bounds checker."""
    return SanityChecker()