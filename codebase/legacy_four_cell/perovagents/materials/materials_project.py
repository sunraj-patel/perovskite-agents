"""Materials Project client tool (Materials layer).

Fetches computed material properties (band gap, formation energy, density,
stability, crystal structure) from the Materials Project REST API. Results are
cached on disk under `results/materials_project/cache.json`.

Behavior:
  * Offline / no `MP_API_KEY`  -> `{"offline": True, "error": ...}` (framework
    stays usable without external credentials).
  * Multiple phases for a formula -> returns polymorph options so the agent can
    disambiguate interactively.
  * Single stable phase -> auto-selected with a note.
"""
import json
import os
from typing import Any, Dict, Optional

from .. import config
from ..core.tool import Tool

CACHE_PATH = os.path.join(config.ensure_results("materials_project"), "cache.json")

# optional pymatgen/matminer helpers (Materials layer "analysers")
try:
    from pymatgen.core import Composition  # noqa: F401
    HAS_PYMATGEN = True
except Exception:
    HAS_PYMATGEN = False


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        try:
            return json.load(open(CACHE_PATH))
        except Exception:
            return {}
    return {}


class MaterialsProjectTool(Tool):
    """Computed-property lookup backed by the Materials Project REST API,
    with a disk cache and interactive polymorph disambiguation."""
    name = "materials_project"
    description = (
        "Fetch computed properties (band gap, formation energy, density, "
        "stability, structure) for an inorganic compound from the Materials "
        "Project. Returns polymorph options when several phases exist, with "
        "mp-ids. Requires MP_API_KEY; falls back to offline gracefully."
    )
    layer = "materials"
    version = "0.1.0"
    capabilities = {"computed_properties", "materials_project"}

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("MP_API_KEY", "")
        self._use_new_client = self._try_mp_api()

    def _try_mp_api(self) -> bool:
        """Prefer the official mp-api client when present."""
        try:
            import mp_api  # noqa: F401
            return bool(self.api_key)
        except Exception:
            return False

    # ---- core lookup -----------------------------------------------------
    def get_properties(self, formula: str) -> Dict[str, Any]:
        """Cached formula lookup; returns offline/error records on failure."""
        cache = _load_cache()
        if formula in cache:
            return cache[formula]
        if not self.api_key:
            return {"offline": True, "error": "MP_API_KEY not set; offline mode",
                    "formula": formula}
        try:
            rec = self._fetch(formula)
        except Exception as exc:
            return {"offline": True, "error": f"Materials Project error: {exc}",
                    "formula": formula}
        cache[formula] = rec
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=1)
        return rec

    def _fetch(self, formula: str) -> Dict[str, Any]:
        if self._use_new_client:
            from mp_api.client import MPClient
            with MPClient(api_key=self.api_key) as mpr:
                docs = mpr.materials.summary.search(
                    formula=formula, fields=["material_id", "formula_pretty",
                                             "band_gap", "formation_energy_per_atom",
                                             "density", "is_stable", "structure"])
                phases = []
                for d in docs:
                    phases.append({"mp_id": d.material_id,
                                   "formula": d.formula_pretty,
                                   "band_gap": d.band_gap,
                                   "formation_energy_per_atom": d.formation_energy_per_atom,
                                   "density": d.density, "is_stable": d.is_stable})
        else:
            import requests
            r = requests.get("https://api.materialsproject.org/materials/summary/",
                             params={"formula": formula, "_fields": "material_id,formula_pretty,band_gap,formation_energy_per_atom,density,is_stable"},
                             headers={"X-Api-Key": self.api_key}, timeout=30)
            r.raise_for_status()
            phases = []
            for d in r.json().get("data", []):
                phases.append({"mp_id": d.get("material_id"),
                               "formula": d.get("formula_pretty"),
                               "band_gap": d.get("band_gap"),
                               "formation_energy_per_atom": d.get("formation_energy_per_atom"),
                               "density": d.get("density"),
                               "is_stable": d.get("is_stable")})
        return self._postprocess(formula, phases)

    @staticmethod
    def _postprocess(formula: str, phases: list) -> dict:
        if not phases:
            return {"found": False, "formula": formula, "n_phases": 0}
        stable = [p for p in phases if p["is_stable"]]
        if len(phases) == 1:
            phases[0]["selected"] = True
            return {"found": True, "formula": formula, "n_phases": 1,
                    "phase": phases[0], "needs_user_choice": False}
        if len(stable) == 1:
            for p in phases:
                p["selected"] = p is stable[0]
            return {"found": True, "formula": formula, "n_phases": len(phases),
                    "phase": stable[0], "needs_user_choice": False,
                    "note": "auto-selected the single stable phase"}
        # multiple phases, no clear winner -> interactive choice
        for p in phases:
            p["selected"] = False
        return {"found": True, "formula": formula, "n_phases": len(phases),
                "phases": phases, "needs_user_choice": True,
                "note": "several phases; choose from options"}

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        formula = params.get("formula") or params.get("material") or params.get("q")
        if not formula:
            return {"error": "formula required"}
        rec = self.get_properties(str(formula).strip())
        rec.setdefault("provenance", {"kind": "materials_project_lookup",
                                      "source": "Materials Project",
                                      "formula": str(formula).strip()})
        return rec


def formula_from_descriptor(name: str) -> Optional[str]:
    """Best-effort: map a PSC layer material name to a formula for MP lookup."""
    table = {"TiO2": "TiO2", "SnO2": "SnO2", "ZnO": "ZnO", "NiO": "NiO",
             "PCBM": None, "C60": None, "Spiro-MeOTAD": None, "PTAA": None,
             "MAPbI3": "MAPbI3", "FAPbI3": "FAPbI3", "CsPbI3": "CsPbI3",
             "CsMAFA": "Cs0.05(FA0.79MA0.16)Pb(I0.83Br0.17)3"}
    return table.get(name)