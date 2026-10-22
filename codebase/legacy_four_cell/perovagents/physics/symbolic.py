"""Symbolic math / equation fitting tool (Physics layer).

Fits published empirical models (e.g., band-gap-to-wavelength, thickness trends,
log-linear fits) to data the agent already collected via PDB/SCAPS. Uses
`scipy.optimize` lazily. This is intentionally light — it exposes parameter /
trend fitting, not a general theorem prover.
"""
from typing import Any, Dict

import numpy as np

from ..core.tool import Tool


class EquationFitTool(Tool):
    name = "symbolic"
    description = (
        "Fit simple empirical/semilog curves to x/y data (linear, power-law, "
        "exponential, band-gap<->wavelength). Returns fit parameters, R^2 and "
        "a human-readable equation. Use to quantify trends found by other tools "
        "(SCAPS sweeps, PDB distributions)."
    )
    layer = "physics"
    version = "0.1.0"
    capabilities = {"equation_fitting", "trend_fitting"}

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        x = np.asarray(params.get("x"), dtype=float)
        y = np.asarray(params.get("y"), dtype=float)
        if x.size < 2 or y.size != x.size:
            return {"error": "x and y arrays required with >=2 points"}
        kind = params.get("kind", "linear")
        out = self._fit(x, y, kind)
        out.setdefault("provenance", {"kind": "equation_fit", "model": kind})
        return out

    def _fit(self, x, y, kind):
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if x.size < 2:
            return {"error": "not enough finite points"}
        if kind == "linear":
            a, b = np.polyfit(x, y, 1)
            pred = a * x + b
            return {"result_type": "table", "model": "y = a*x + b",
                    "parameters": {"a": round(float(a), 5), "b": round(float(b), 5)},
                    "r2": self._r2(y, pred), "kind": "linear"}
        if kind == "power":
            pos = (x > 0) & (y > 0)
            if pos.sum() < 2:
                return {"error": "power fit needs positive x,y"}
            a, b = np.polyfit(np.log(x[pos]), np.log(y[pos]), 1)
            pred = np.exp(a) * x ** b
            return {"result_type": "table", "model": "y = A * x^b, A=exp(a)",
                    "parameters": {"A": round(float(np.exp(a)), 5), "b": round(float(b), 5)},
                    "r2": self._r2(y, pred), "kind": "power"}
        if kind == "exponential":
            pos = y > 0
            if pos.sum() < 2:
                return {"error": "exponential fit needs y>0"}
            a, b = np.polyfit(x[pos], np.log(y[pos]), 1)
            pred = np.exp(b) * np.exp(a * x)
            return {"result_type": "table", "model": "y = A * exp(b*x), A=exp(B)",
                    "parameters": {"A": round(float(np.exp(b)), 5), "b": round(float(a), 5)},
                    "r2": self._r2(y, pred), "kind": "exponential"}
        # band-gap <-> wavelength (eV vs nm): y ~ 1239.84/x fits straight line in (1/x)
        if kind == "eg_nm":
            if np.any(x <= 0):
                return {"error": "eg_nm needs positive x"}
            a, b = np.polyfit(1 / x, y, 1)
            pred = a / x + b
            return {"result_type": "table", "model": "E = A / lambda + B",
                    "parameters": {"A": round(float(a), 5), "B": round(float(b), 5)},
                    "r2": self._r2(y, pred), "kind": "eg_nm"}
        return {"error": f"unknown kind {kind}; options: linear, power, exponential, eg_nm"}

    @staticmethod
    def _r2(y, pred) -> float:
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return round(float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"), 4)