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
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .. import config
from ..backend import Backend

# preset sweep specs: parameter -> (feature name, unit, def-arg)
SWEEP_SPECS = {
    "pvk_Eg":  ("perovskite_band_gap", "eV", "--pvk-Eg"),       # 1.35-1.70
    "pvk_d":   ("perovskite_thickness", "nm", "--pvk-thickness"),  # 200-1000
    "pvk_mun": ("perovskite_electron_mobility", "cm2/Vs", "--pvk-mun"),
    "pvk_mup": ("perovskite_hole_mobility", "cm2/Vs", "--pvk-mup"),
    "htl_chi": ("HTL_electron_affinity", "eV", "--htl-chi"),
    "htl_Eg":  ("HTL_band_gap", "eV", "--htl-Eg"),
}
PUBLIC_TO_INTERNAL = {"eg": "pvk_Eg", "d": "pvk_d", "mue": "pvk_mun", "muh": "pvk_mup",
                      "htlchi": "htl_chi", "htlEg": "htl_Eg"}

# ---------------------------------------------------------------------------
# Full parameter namespace for arbitrary device experiments.
#
# The six SWEEP_SPECS above are only what the *precomputed* CSV happens to
# cover. The .def builder can set any field of any of the three layers, so the
# agent is not limited to those six: this registry is the public -> SI mapping
# for everything, including the SRH defect density Nt that used to be hardcoded.
#
#   public name -> (layer, field, public unit, scale to the SI unit in the .def)
# ---------------------------------------------------------------------------
_LAYERS = {"pvk": "Perovskite", "htl": "HTL", "etl": "ETL"}
_FIELDS = {
    # field:    (public unit, scale to SI, description)
    "d":      ("nm", 1e-9, "layer thickness"),
    "Eg":     ("eV", 1.0, "band gap"),
    "chi":    ("eV", 1.0, "electron affinity"),
    "eps":    ("-", 1.0, "relative permittivity"),
    "Nc":     ("1/cm^3", 1e6, "conduction-band DOS"),
    "Nv":     ("1/cm^3", 1e6, "valence-band DOS"),
    "vthn":   ("cm/s", 1e-2, "electron thermal velocity"),
    "vthp":   ("cm/s", 1e-2, "hole thermal velocity"),
    "mun":    ("cm^2/Vs", 1e-4, "electron mobility"),
    "mup":    ("cm^2/Vs", 1e-4, "hole mobility"),
    "Na":     ("1/cm^3", 1e6, "acceptor doping"),
    "Nd":     ("1/cm^3", 1e6, "donor doping"),
    "Nt":     ("1/cm^3", 1e6, "SRH defect density (the dominant recombination knob)"),
    "Et":     ("eV", 1.0, "defect energy level"),
    "sigma_n": ("cm^2", 1e-4, "electron capture cross-section"),
    "sigma_p": ("cm^2", 1e-4, "hole capture cross-section"),
}

PARAM_SPECS: Dict[str, dict] = {}
for _lay, _fld, _unit, _scale, _desc in [
        (l, f, v[0], v[1], v[2]) for l in _LAYERS for f, v in _FIELDS.items()]:
    PARAM_SPECS[f"{_lay}_{_fld}"] = {
        "layer": _lay, "layer_name": _LAYERS[_lay], "field": _fld,
        "unit": _unit, "scale": _scale, "description": _desc,
        "label": f"{_LAYERS[_lay]} {_desc}",
    }
# Friendlier aliases the agent will naturally try.
for _alias, _canon in {
    "thickness": "pvk_d", "htl_thickness": "htl_d", "etl_thickness": "etl_d",
    "perovskite_thickness": "pvk_d", "band_gap": "pvk_Eg", "htl_band_gap": "htl_Eg",
    "etl_band_gap": "etl_Eg", "electron_mobility": "pvk_mun",
    "hole_mobility": "pvk_mup", "defect_density": "pvk_Nt", "htl_defect_density": "htl_Nt",
    "etl_defect_density": "etl_Nt", "htl_chi": "htl_chi", "htl_Eg": "htl_Eg",
}.items():
    if _canon in PARAM_SPECS:
        PARAM_SPECS.setdefault(_alias, dict(PARAM_SPECS[_canon], alias_of=_canon))


def canonical_names() -> List[str]:
    """Every canonical `<layer>_<field>` name, grouped for display."""
    return [f"{lay}_{fld}" for lay in _LAYERS for fld in sorted(_FIELDS)]


def resolve_param(name: str) -> dict:
    """Public parameter name -> spec.

    On a bad name the error lists every valid name grouped by layer, because
    the agent reads this message and retries with it - a truncated list would
    hide the parameter it actually wanted.
    """
    spec = PARAM_SPECS.get(name) or PARAM_SPECS.get((name or "").strip())
    if spec is None:
        raise KeyError(f"unknown parameter {name!r}. Use a <layer>_<field> name from "
                       f"this list: {', '.join(canonical_names())}")
    return spec


def to_si(public_name: str, value: float) -> tuple:
    """(public_name, value) -> (build_def kwarg name, SI value)."""
    spec = resolve_param(public_name)
    return (f"{spec['layer']}_{spec['field']}", float(value) * spec["scale"])


# The reference n-i-p cell exactly as make_perovskite_def.build_def() writes it,
# in public units. Lets the agent state what a simulation actually used instead
# of guessing, and gives `run_point` a documented baseline.
REFERENCE_CELL = {
    "pvk_d": (400.0, "nm"), "pvk_Eg": (1.55, "eV"), "pvk_chi": (3.93, "eV"),
    "pvk_eps": (10.0, "-"), "pvk_Nc": (2.2e18, "1/cm^3"), "pvk_Nv": (2.2e18, "1/cm^3"),
    "pvk_mun": (1.0, "cm^2/Vs"), "pvk_mup": (1.0, "cm^2/Vs"), "pvk_Nt": (1.0e14, "1/cm^3"),
    "htl_d": (200.0, "nm"), "htl_Eg": (3.0, "eV"), "htl_chi": (2.2, "eV"),
    "htl_eps": (3.0, "-"), "htl_mun": (2e-3, "cm^2/Vs"), "htl_mup": (2e-3, "cm^2/Vs"),
    "htl_Na": (1.0e18, "1/cm^3"), "htl_Nt": (1.0e14, "1/cm^3"),
    "etl_d": (50.0, "nm"), "etl_Eg": (3.2, "eV"), "etl_chi": (4.0, "eV"),
    "etl_eps": (9.0, "-"), "etl_mun": (1.0, "cm^2/Vs"), "etl_mup": (1.0, "cm^2/Vs"),
    "etl_Nd": (1.0e18, "1/cm^3"), "etl_Nt": (1.0e14, "1/cm^3"),
}

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


class SCAPSTool(Backend):
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
        self._sweeps = None
        self._live_ok = None          # None = not probed yet
        self._live_error = None

    def live_status(self) -> Dict[str, Any]:
        """Can SCAPS actually run on this machine? Probed once, then cached.

        `os.path.exists(wine)` is not enough: the wine prefix can be present
        while scaps3309.exe still fails to start (a missing Windows DLL makes
        it exit with 0xc0000135). Offering live tools in that state only
        wastes the agent's turns, so probe a real run once and remember it.
        """
        if self._live_ok is not None:
            return {"available": self._live_ok, "error": self._live_error,
                    "wine": config.SCAPS_WINE}
        if not config.SCAPS_WINE or not os.path.exists(config.SCAPS_WINE):
            self._live_ok, self._live_error = False, "wine not found"
            return {"available": False, "error": self._live_error, "wine": config.SCAPS_WINE}
        scaps_dir = os.path.join(config.SCAPS_PREFIX, "drive_c", "Program Files (x86)",
                                 "Scaps3309")
        probe = "perovagents_probe"
        try:
            self._bake({}, probe)
        except Exception as exc:
            self._live_ok, self._live_error = False, f"cannot write a device file: {exc}"
            return {"available": False, "error": self._live_error, "wine": config.SCAPS_WINE}
        r = self._run_point(f"{probe}.def", timeout=300)
        if r.get("converged"):
            self._live_ok, self._live_error = True, None
        else:
            # Distinguish "SCAPS did not start" from "SCAPS ran and failed to
            # converge" - the first is an environment problem worth reporting.
            self._live_ok, self._live_error = False, (
                "scaps3309.exe did not produce an IV curve (the Windows runtime "
                "under wine is incomplete, e.g. a missing DLL); live simulation "
                "is unavailable on this machine")
        return {"available": self._live_ok, "error": self._live_error,
                "wine": config.SCAPS_WINE}

    @property
    def def_dir(self) -> str:
        """Where SCAPS looks for / writes device definition files."""
        return os.path.join(config.SCAPS_PREFIX, "drive_c", "Program Files (x86)",
                            "Scaps3309", "def")

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
        # The CSV stores SI values: thickness in METRES, so convert to the
        # public unit (nm) that SWEEP_SPECS advertises and run_sweep_live takes.
        scale = 1e9 if spec_key == "pvk_d" else 1.0
        rows = [{"parameter": feature, "value": float(r["param"]) * scale,
                 "unit": unit,
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
                "trend": trend}

    # ---- live run --------------------------------------------------------
    def _run_point(self, defname: str, stopv: float = 1.4, timeout: int = 420) -> dict:
        """Run one SCAPS parameter point under wine, parsing Voc/Jsc/FF/PCE
        from the produced IV file."""
        scaps_dir = os.path.join(config.SCAPS_PREFIX, "drive_c", "Program Files (x86)", "Scaps3309")
        # Unique script per run: concurrent runs (or a stale process) must
        # never overwrite each other's runpoint.script mid-flight, or a run
        # computes the wrong IV file and its caller times out waiting.
        ivname = "live_%s.iv" % (abs(hash(defname)) % 99999)
        script_file = os.path.join(scaps_dir, "script", f"runpoint_{ivname[:-3]}.script")
        with open(script_file, "w") as f:
            f.write(SCRIPT_TEMPLATE.format(defname=defname, ivname=ivname, stopv=stopv))
        iv_file = os.path.join(scaps_dir, "results", ivname)
        if os.path.exists(iv_file):
            os.remove(iv_file)
        env = config.scaps_env()
        # New process group: on timeout killpg() takes down the wine loader
        # AND its reparented children (start.exe, scaps3309.exe). Plain
        # proc.kill() leaves orphans spinning at 100% CPU that slow down
        # every later run and contend for prefix files.
        proc = subprocess.Popen([config.SCAPS_WINE, "scaps3309.exe",
                                 os.path.basename(script_file)],
                                cwd=scaps_dir, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
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
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
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

    # ---- arbitrary device experiments ---------------------------------
    def _bake(self, params: Dict[str, float], label: str = "custom") -> str:
        """Write a .def for `params` (public units) and return its file name."""
        sets = []
        for name, value in params.items():
            if value is None:
                continue
            kwarg, si = to_si(name, value)
            sets.append(f"--set={kwarg}={si:.10g}")
        defname = f"{label}.def"
        cmd = [sys.executable, config.SCAPS_MAKE_DEF, "--out", defname] + sets
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ValueError(f"could not build the device definition: "
                             f"{(proc.stderr or proc.stdout).strip()[:300]}")
        return defname

    def read_def(self, defname: str = "perovskite_nip.def") -> Dict[str, Any]:
        """Read a generated .def back and report every layer's parameters.

        Use this to state exactly which values a simulation used, instead of
        assuming the defaults. Returns thickness in nm, mobilities in cm^2/Vs
        and densities in 1/cm^3, i.e. the public units.
        """
        path = os.path.join(self.def_dir, defname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"no such device file {defname!r} in {self.def_dir}")
        text = open(path, errors="ignore").read()
        blocks = dict(re.findall(r"name : (\w+)\n(.*?)(?=\nlayer|\nfront contact)",
                                 text, re.S))
        fields = {"d": r"d : (\S+)", "Eg": r"Eg :\s+(\S+)", "chi": r"chi :\s+(\S+)",
                  "eps": r"eps :\s+(\S+)", "Nc": r"Nc :\s+(\S+)", "Nv": r"Nv :\s+(\S+)",
                  "mun": r"mu_n :\s+(\S+)", "mup": r"mu_p :\s+(\S+)",
                  "Na": r"Na\(y\) :\s+(\S+)", "Nd": r"Nd\(y\) :\s+(\S+)",
                  "Nt": r"Nt\(y\) :\s+(\S+)", "Et": r"Et :\s+(\S+)"}
        layers = {}
        for block_name, body in blocks.items():
            prefix = next((p for p, n in _LAYERS.items() if n == block_name), None)
            if not prefix:
                continue
            out = {}
            for fld, pat in fields.items():
                mo = re.search(pat, body)
                if not mo:
                    continue
                spec = PARAM_SPECS[f"{prefix}_{fld}"]
                # The .def stores 6 significant figures in SI, so dividing back
                # out leaves float noise (200 nm -> 199.99999999999997). Round
                # to the precision the file actually carries.
                val = float(mo.group(1)) / spec["scale"]
                out[f"{prefix}_{fld}"] = {
                    "value": float(f"{val:.6g}"), "unit": spec["unit"]}
            layers[block_name] = out
        return {"found": True, "def_file": defname, "stack": list(blocks),
                "layers": layers,
                "provenance": {"kind": "scaps_def", "source": defname}}

    def run_point(self, params: Dict[str, float] = None, label: str = "custom",
                  timeout: int = 420) -> Dict[str, Any]:
        """Simulate ONE device with ANY combination of parameters set at once.

        `params` uses the public parameter names and units, e.g.
        {"pvk_Nt": 1e15, "etl_Nt": 1e17, "htl_d": 150}. Everything left out
        keeps the reference-cell default. Returns Voc, Jsc, FF, PCE and the
        .def file that was used, so the inputs can be verified with read_def.
        """
        params = dict(params or {})
        for name in params:
            resolve_param(name)                 # validate before baking
        defname = self._bake(params, label or "custom")
        stopv = 1.8 if float(params.get("pvk_Eg", 0) or 0) > 1.6 else 1.4
        r = self._run_point(defname, stopv=stopv, timeout=timeout)
        r.update({"def_file": defname, "inputs": params,
                  "provenance": {"kind": "scaps_point_live",
                                 "executable": "scaps3309.exe"}})
        return r

    def run_sweep_live(self, parameter: str, values: List[float],
                       label: str = None) -> dict:
        """Run SCAPS live for each value of ANY registered parameter.

        Works for the six precomputed sweeps and for everything else in
        PARAM_SPECS (defect densities, layer thicknesses, doping, affinities,
        capture cross-sections, ...). Values are in the public unit for that
        parameter; one parameter is varied while the rest stay at defaults.
        """
        spec = resolve_param(parameter)         # raises KeyError w/ valid names
        feature, unit = spec["label"], spec["unit"]
        rows = []
        for v in values:
            point_label = f"{label or 'sweep'}_{spec['layer']}_{spec['field']}_{v:g}"
            r = self.run_point({parameter: float(v)}, point_label)
            rows.append({"parameter": feature, "parameter_name": parameter,
                         "value": float(v), "unit": unit, **r})
        out = {"found": True, "parameter": feature, "unit": unit,
               "parameter_name": parameter, "data": rows,
               "provenance": {"kind": "scaps_sweep_live",
                              "executable": "scaps3309.exe"}}
        out.update(self._summarize(rows))
        return out

    def describe_parameters(self, layer: str = None) -> Dict[str, Any]:
        """Every settable parameter (optionally for one layer) with its unit."""
        items = []
        for name, spec in sorted(PARAM_SPECS.items()):
            if layer and spec["layer"] != layer:
                continue
            # An alias reports the default of the parameter it points at.
            default = REFERENCE_CELL.get(name) or REFERENCE_CELL.get(spec.get("alias_of"))
            items.append({"name": name, "layer": spec["layer_name"],
                          "field": spec["description"], "unit": spec["unit"],
                          "default": default[0] if default else None,
                          "alias_of": spec.get("alias_of")})
        return {"found": True, "n_parameters": len(items), "parameters": items}

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """`live` runs SCAPS now; otherwise returns the precomputed sweep."""
        if params.get("live"):
            return self.run_sweep_live(params["parameter"], params.get("values", []))
        return self.get_sweep(params.get("sweep_name") or params.get("parameter") or "")