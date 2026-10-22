"""SCAPS-1D physics simulation tool (Physics layer).

Wraps the wine64-driven SCAPS 3.3 executable. Two modes:

  * `get_sweep(parameter)` — return a *precomputed* sensitivity sweep from
    `results/scaps/sweep_results.csv` (canonical six sweeps) without re-running
    the simulator. This is the mode used for the physics-vs-empirical analysis.
  * `run_sweep(parameter, values)` — bake fresh .def files and run SCAPS live.

Device definition files are produced by `config.SCAPS_MAKE_DEF`.

Framing disclaimer: SCAPS results are a *physics sensitivity model*, not a
reproduction of any experimental dataset. Comparisons against experimental
trends are presented as empirical association vs physics sensitivity.
"""
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .. import config
from ..core.tool import Tool

# preset sweep specs: parameter -> (feature name, unit, def-arg)
SWEEP_SPECS = {
    "pvk_Eg":  ("perovskite_band_gap", "eV", "--pvk-Eg"),       # 1.35-1.70
    "pvk_d":   ("perovskite_thickness", "nm", "--pvk-thickness"),  # 200-1000
    "pvk_mun": ("perovskite_electron_mobility", "cm2/Vs", "--pvk-mun"),
    "pvk_mup": ("perovskite_hole_mobility", "cm2/Vs", "--pvk-mup"),
    "htl_chi": ("HTL_electron_affinity", "eV", "--htl-chi"),
    "htl_Eg":  ("HTL_band_gap", "eV", "--htl-Eg"),
}
PUBLIC_TO_INTERNAL = {"eg": "pvk_Eg", "d": "pvk_d", "mue": "pvk_mun",
                      "muh": "pvk_mup", "htlchi": "htl_chi", "htlEg": "htl_Eg"}

SCRIPT_TEMPLATE = """set quitscript.quitSCAPS
set errorhandling.overwritefile
load definitionfile {defname}
clear actions
action workingpoint.temperature 300
action iv.startv 0
action iv.stopv {stopv}
action iv.increment 0.02
action iv.doiv
action light
calculate singleshot
save results.iv {ivname}
"""


class SCAPSTool(Tool):
    name = "scaps"
    description = (
        "Run (or read precomputed) SCAPS-1D device simulations for a perovskite "
        "solar cell. Sensitivity sweeps over band gap, absorber thickness, "
        "mobility, and HTL electron affinity/gap return Voc/Jsc/FF/PCE. "
        "SCAPS is a physics model — use it for mechanism testing, not exact "
        "experimental reproduction."
    )
    layer = "physics"
    version = "0.2.0"
    capabilities = {"device_simulation", "physics"}

    def __init__(self):
        self._sweeps: Optional[pd.DataFrame] = None

    @property
    def sweeps(self) -> pd.DataFrame:
        """The precomputed sweep table (loaded lazily from CSV)."""
        if self._sweeps is None:
            self._sweeps = pd.read_csv(config.SCAPS_EXISTING_RESULTS)
        return self._sweeps

    # ---- precomputed sweep -----------------------------------------------
    def get_sweep(self, sweep_name: str) -> Dict[str, Any]:
        """Return one precomputed sensitivity sweep (points + trend summary)
        without running the simulator."""
        spec_key = PUBLIC_TO_INTERNAL.get(sweep_name, sweep_name)
        if spec_key not in SWEEP_SPECS:
            return {"found": False, "parameter": sweep_name,
                    "error": f"unknown sweep {sweep_name}; options: {sorted(PUBLIC_TO_INTERNAL)}"}
        feature, unit, _ = SWEEP_SPECS[spec_key]
        csv_key = {"pvk_Eg": "eg", "pvk_d": "d", "pvk_mun": "mue", "pvk_mup": "muh",
                   "htl_chi": "htlchi", "htl_Eg": "htlEg"}[spec_key]
        sub = self.sweeps[self.sweeps["sweep"] == csv_key] \
            .dropna(subset=["voc", "jsc", "ff", "eta"]).copy()
        if sub.empty:
            return {"found": False, "parameter": sweep_name}
        rows = [{"parameter": feature, "value": float(r["param"]), "unit": unit,
                 "Voc": float(r["voc"]), "Jsc": float(r["jsc"]),
                 "FF": float(r["ff"]), "PCE": float(r["eta"])}
                for _, r in sub.iterrows()]
        rows.sort(key=lambda r: r["value"])
        out = {"found": True, "parameter": feature, "unit": unit,
               "n_points": len(rows), "data": rows,
               "provenance": {"kind": "scaps_sweep", "source": "sweep_results.csv",
                              "device": "perovskite_nip.def"}}
        out.update(self._summarize(rows))
        return out

    def _summarize(self, rows: List[dict]) -> dict:
        xs = np.array([r["value"] for r in rows], dtype=float)
        pce = np.array([r["PCE"] for r in rows], dtype=float)
        r = float(np.corrcoef(xs, pce)[0, 1]) if len(xs) > 2 and np.std(pce) > 0 else None
        rel = None
        if len(rows) >= 4 and abs(pce[0]) > 1e-6:
            rel = (pce[-1] - pce[0]) / abs(pce[0])
        trend = "positive" if r is not None and r > 0.05 else (
            "negative" if r is not None and r < -0.05 else "weak")
        return {"pearson_r_PCE": round(r, 4) if r is not None else None,
                "relative_PCE_change": round(float(rel), 4) if rel is not None else None,
                "trend": trend, "convergence_ok": True}

    # ---- live run --------------------------------------------------------
    def _run_point(self, defname: str, stopv: float = 1.4, timeout: int = 180) -> dict:
        """Run one SCAPS parameter point under wine, parsing Voc/Jsc/FF/PCE
        from the produced IV file."""
        scaps_dir = os.path.join(config.SCAPS_PREFIX, "drive_c", "Program Files (x86)", "Scaps3309")
        script_file = os.path.join(scaps_dir, "script", "runpoint.script")
        ivname = "live_%s.iv" % (abs(hash(defname)) % 99999)
        with open(script_file, "w") as f:
            f.write(SCRIPT_TEMPLATE.format(defname=defname, ivname=ivname, stopv=stopv))
        iv_file = os.path.join(scaps_dir, "results", ivname)
        if os.path.exists(iv_file):
            os.remove(iv_file)
        env = dict(os.environ, WINEPREFIX=config.SCAPS_PREFIX)
        proc = subprocess.Popen([config.SCAPS_WINE, "scaps3309.exe", "runpoint.script"],
                                cwd=scaps_dir, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0, ok = time.time(), False
        while time.time() - t0 < timeout:
            if os.path.exists(iv_file):
                text = open(iv_file, errors="ignore").read()
                if "deduced from calculated IV-curve" in text:
                    ok = True
                    break
            if proc.poll() is not None:
                break
            time.sleep(2)
        time.sleep(2)
        if proc.poll() is None:
            proc.kill()
        if not os.path.exists(iv_file):
            return {"converged": False}
        text = open(iv_file, errors="ignore").read()
        g = lambda pat: (lambda m: float(m.group(1)) if m else None)(   # noqa: E731
            re.search(pat, text))
        conv_fail = "Convergence failure" in text
        voc, jsc, ff, eta = g(r"Voc =\s+([\d.eE+-]+)\s+Volt"), g(r"Jsc =\s+([\d.eE+-]+)\s+mA"), \
            g(r"FF =\s+([\d.eE+-]+)\s+%"), g(r"eta =\s+([\d.eE+-]+)\s+%")
        if voc is None or jsc is None or ff is None or eta is None:
            return {"converged": False, "convergence_failure": conv_fail}
        return {"converged": True, "convergence_failure": conv_fail,
                "Voc": voc, "Jsc": jsc, "FF": ff, "PCE": eta}

    def run_sweep_live(self, parameter: str, values: List[float]) -> dict:
        """Run SCAPS live for each value (bakes .def files, invokes wine)."""
        if parameter not in SWEEP_SPECS:
            return {"error": f"unknown parameter {parameter}"}
        feature, unit, arg_flag = SWEEP_SPECS[parameter]
        rows = []
        for v in values:
            label = f"live_{parameter}_{v:g}"
            if parameter == "pvk_d":
                arg = "%.4e" % (float(v) * 1e-9)
            elif parameter in ("pvk_mun", "pvk_mup"):
                arg = "%.6e" % (float(v) * 1e-4)  # cm2/Vs -> m2/Vs
            else:
                arg = "%.4f" % float(v)
            defname = f"{label}.def"
            subprocess.run([sys.executable, config.SCAPS_MAKE_DEF, arg_flag, arg, "--out", defname],
                           check=True)
            stopv = 1.8 if parameter == "pvk_Eg" else 1.4
            r = self._run_point(defname, stopv=stopv)
            rows.append({"parameter": feature, "value": float(v), "unit": unit, **r})
        return {"found": True, "parameter": feature, "data": rows,
                "provenance": {"kind": "scaps_sweep_live", "executable": "scaps3309.exe"}}

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """`live` runs SCAPS now; otherwise returns the precomputed sweep."""
        if params.get("live"):
            return self.run_sweep_live(params["parameter"], params.get("values", []))
        return self.get_sweep(params.get("sweep_name") or params.get("parameter") or "")