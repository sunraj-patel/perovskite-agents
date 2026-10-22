"""Calculator & unit-conversion tool (Orchestration layer).

Small, dependency-free tool for the arithmetic the planner frequently needs:
unit prefixes, energy<->wavelength (eV<->nm), fill-factor math (PCE from
Voc-Jsc-FF), area/volume conversions, and safe symbolic arbitrary expressions
via a whitelisted evaluation.
"""
import math
import re
from typing import Any, Dict, List

from .tool import Tool

# eV <-> nm: E(eV) = 1239.84193 / lambda(nm)
EV_NM = 1239.84193
Q_EV = 1.602176634e-19        # elementary charge in C


def _is_number(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Perform unit conversions and device-performance arithmetic for "
        "perovskite solar cells: energy<->wavelength (eV<->nm), PCE from "
        "Voc/Jsc/FF, power/area conversions, and safe arbitrary math "
        "expressions. Returns numeric results with units."
    )
    layer = "orchestration"
    version = "0.1.0"
    capabilities = {"arithmetic", "unit_conversion", "calculator"}

    # ---- public call -----------------------------------------------------
    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "convert")
        handler = {
            "convert": self.convert,
            "pce": self.pce,
            "eval": self.eval_expr,
            "help": self.help_,
        }.get(action)
        if handler is None:
            return {"error": f"unknown action {action!r}; use convert|pce|eval"}
        return handler(params)

    # ---- conversions -----------------------------------------------------
    def convert(self, params):
        """Unit conversions: eV<->nm, area scaling, mW/cm2<->W/m2."""
        value = params.get("value")
        from_unit = str(params.get("from", "")).strip()
        to_unit = str(params.get("to", "")).strip()
        if not _is_number(value) or from_unit == to_unit:
            return {"error": "value/units required and must differ"}

        # eV <-> nm family
        ev_nm = {"ev": "eV", "electronvolt": "eV", "nm": "nm", "nm_wavelength": "nm"}
        fu, tu = ev_nm.get(from_unit.lower(), from_unit.lower()), ev_nm.get(to_unit.lower(), to_unit.lower())
        if {fu, tu} == {"eV", "nm"}:
            v = float(value)
            out = EV_NM / v if fu == "eV" else EV_NM / v
            return {"from": from_unit, "to": to_unit, "value": float(value),
                    "result": round(out, 6), "unit": tu,
                    "formula": "E*lambda = 1239.84193 eV*nm"}

        # area
        area_map = {"cm2": 1.0, "m2": 1e-4, "mm2": 100.0}
        if from_unit.lower() in area_map and to_unit.lower() in area_map:
            v = float(value)
            return {"from": from_unit, "to": to_unit, "value": v,
                    "result": v * area_map[from_unit] / area_map[to_unit],
                    "unit": to_unit, "formula": "area scaling"}

        # power density / intensity (mW/cm2 <-> W/m2): 100 mW/cm2 = 1000 W/m2
        if from_unit.lower() in {"mw/cm2", "w/m2"} and to_unit.lower() in {"mw/cm2", "w/m2"}:
            v = float(value)
            factor = 10.0 if from_unit.lower() == "mw/cm2" else 0.1
            return {"from": from_unit, "to": to_unit, "value": v,
                    "result": v * factor, "unit": to_unit,
                    "formula": "1 mW/cm2 = 10 W/m2"}

        return {"error": f"no conversion path {from_unit} -> {to_unit}"}

    # ---- pce -------------------------------------------------------------
    def pce(self, params):
        """PCE (%) from Voc (V), Jsc (mA/cm2), FF (fraction)."""
        voc = float(params.get("voc"))
        jsc = float(params.get("jsc"))
        ff = float(params.get("ff"))          # fraction, e.g. 0.78
        if not (voc and jsc and ff):
            return {"error": "voc, jsc, ff required"}
        if voc < 0 or jsc < 0 or not (0 < ff <= 1):
            return {"error": "physically implausible inputs"}
        pce_pct = voc * jsc * ff * 100.0 / 100.0  # Voc[V]*Jsc[mA/cm2]*FF -> mW/cm2, then %
        return {"result_type": "scalar", "pce_percent": round(pce_pct, 3),
                "formula": "PCE = Voc * Jsc * FF / 100 (in %)"}

    # ---- safe expression eval --------------------------------------------
    _SAFE_GLOBALS = {
        "abs": abs, "min": min, "max": max, "round": round, "sum": sum,
        "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "pi": math.pi, "e": math.e,
        "EV_NM": EV_NM, "Q_EV": Q_EV,
    }

    def eval_expr(self, params):
        """Evaluate a whitelisted symbolic expression (no builtins access)."""
        expr = str(params.get("expression", "")).strip()
        if not expr:
            return {"error": "expression required"}
        if len(expr) > 500:
            return {"error": "expression too long"}
        # block attribute access (protects against getattr tricks)
        if re.search(r"(__\w+__|\bimport\b|^\s*[A-Za-z_]\w*\s*\())", expr):
            return {"error": "expression contains forbidden constructs"}
        try:
            out = eval(expr, {"__builtins__": {}}, self._SAFE_GLOBALS)   # noqa: S307
        except Exception as exc:
            return {"error": f"eval failed: {exc}"}
        return {"result_type": "scalar", "expression": expr,
                "result": (round(out, 8) if isinstance(out, float) else out)}

    # ---- help ------------------------------------------------------------
    def help_(self, params):
        return {
            "actions": {
                "convert": {"value": "number", "from": "eV|nm|cm2|m2|mW/cm2|W/m2",
                            "to": "target unit"},
                "pce": {"voc": "V", "jsc": "mA/cm2", "ff": "fraction 0..1"},
                "eval": {"expression": "safe math using exp/log/sqrt/sin/pi ..."},
            }
        }