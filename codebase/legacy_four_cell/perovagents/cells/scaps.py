"""Cell (d): SCAPS-1D simulation (S5).

Agent: takes the sweep name (or live-run spec) from task params;
LLM-authored device stacks plug into the agent slot at S6.
Tool: `SCAPSTool` (precomputed sweeps; live runs pass through when the
environment provides wine). Gate: `verify_physics` per sweep row, merged
into one report. Precomputed rows carry no solver evidence, so the
`converged` check honestly fails on them (AUDIT C3, DECISIONS.md) — the
cell then returns a FailedResult naming convergence, exactly as spec
section 7 requires for unvalidated paths.
"""
from typing import Any, Dict, List

from ..cells.schemas import VerificationReport
from ..physics.scaps import SCAPSTool
from ..verify.base import check, report
from ..verify.physics_checks import verify_physics
from .base import DomainCell
from .schemas import Task


class SCAPSCell(DomainCell):
    """SCAPS-1D simulation domain cell."""
    cell = "scaps"

    def __init__(self, max_attempts: int = 3, llm=None):
        self.scaps = SCAPSTool()
        self.llm = llm
        super().__init__(agent_fn=self._agent, tool_fn=self._tool,
                         gate_fn=self._gate, max_attempts=max_attempts)

    # -- slots ---------------------------------------------------------
    def _agent(self, task: Task, history: List) -> Dict[str, Any]:
        fallback = {"sweep": task.params.get("sweep_name")
                    or task.params.get("parameter") or "pvk_Eg",
                    "band_gap_ev": task.params.get("band_gap_ev"),
                    "requested": task.params.get("requested"),
                    "live": bool(task.params.get("live", False)),
                    "values": task.params.get("values", [])}
        if history and self.llm is not None:
            from .llm_agent import llm_tool_call
            return llm_tool_call(self.llm, "scaps_agent", task, history,
                                 fallback)
        return fallback

    def _tool(self, call: Dict[str, Any]) -> Dict[str, Any]:
        if call["live"]:
            out = self.scaps.run_sweep_live(call["sweep"], call["values"])
        else:
            out = self.scaps.get_sweep(call["sweep"])
        out = dict(out)
        out["band_gap_ev"] = call["band_gap_ev"]
        out["requested"] = call["requested"]
        # Normalize: the precomputed CSV stores FF in percent (0-100);
        # the gate contract is FF as a fraction. Record the conversion.
        scaled = False
        for row in out.get("data", []):
            try:
                if float(row.get("FF", 0)) > 1.5:
                    row["FF"] = float(row["FF"]) / 100.0
                    scaled = True
            except (TypeError, ValueError):
                pass
        if scaled:
            prov = dict(out.get("provenance", {}))
            prov["ff_scale"] = "percent-to-fraction"
            out["provenance"] = prov
        return out

    def _gate(self, raw: Dict[str, Any],
              call: Dict[str, Any]) -> VerificationReport:
        if not raw.get("found"):
            return report([check(
                "sweep_found", False,
                raw.get("error", f"unknown sweep {call.get('sweep')}"),
                call.get("sweep"))])
        checks = []
        for i, row in enumerate(raw.get("data", [])):
            # Per-row band gap when the sweep itself varies it; an explicit
            # call-level value wins (e.g. fixed-gap thickness sweep).
            eg = call.get("band_gap_ev")
            if eg is None and "gap" in str(row.get("parameter", "")).lower():
                eg = row.get("value")
            rep = verify_physics(dict(row), requested=call.get("requested"),
                                 band_gap_ev=eg)
            for c in rep.checks:
                c = check(c.name, c.passed,
                          f"row {i}: {c.message}", c.offending)
                checks.append(c)
        if not checks:
            return report([check("sweep_found", False, "sweep has no rows",
                                 call.get("sweep"))])
        return report(checks)
