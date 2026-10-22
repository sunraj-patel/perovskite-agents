"""Material property lookup tool (Materials layer).

Returns Material / Property / Value / Unit / Source tuples for ETLs, HTLs,
perovskite absorbers and contacts used in the PSC ecosystem. Two backends are
merged (literature tables win, PDB-derived medians fill gaps):

  1. Literature tables: `sandbox/materials_structured.csv` (+ `_properties.csv`)
     — hand-curated values with explicit DOIs.
  2. Derived from the experimental enriched DB: median of the property column
     per material identity (labeled "Perovskite Database (median)").
"""
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from .. import config
from ..core.tool import Tool

PROPERTY_UNITS = {
    "band_gap": "eV", "LUMO": "eV", "HOMO": "eV",
    "electron_mobility": "cm2/Vs", "hole_mobility": "cm2/Vs",
    "dielectric_constant": "-", "min_atomic_mass": "amu", "max_atomic_mass": "amu",
    "avg_atomic_mass": "amu", "min_radius": "pm", "max_radius": "pm", "avg_radius": "pm",
    "en_min": "-", "en_max": "-", "en_avg": "-",
    "tolerance_factor": "-", "octahedral_factor": "-", "tau_descriptor": "-",
    "cation_radius_ratio": "-",
}

# role -> {property: column in enriched DB}
# (dict-of-dicts; keys are just labels, we iterate over the inner mapping)
ROLE_COLUMN_PROPS = {
    "ETL": {"band_gap": "ETL_band_gap", "LUMO": "ETL_LUMO", "HOMO": "ETL_HOMO",
            "electron_mobility": "ETL_electron_mobility",
            "hole_mobility": "ETL_hole_mobility",
            "dielectric_constant": "ETL_dielectric_constant",
            "min_radius": "ETL_min_radius", "max_radius": "ETL_max_radius",
            "avg_radius": "ETL_avg_radius"},
    "HTL": {"band_gap": "HTL_band_gap", "LUMO": "HTL_LUMO", "HOMO": "HTL_HOMO",
            "electron_mobility": "HTL_electron_mobility",
            "hole_mobility": "HTL_hole_mobility",
            "dielectric_constant": "HTL_dielectric_constant",
            "min_radius": "HTL_min_radius", "max_radius": "HTL_max_radius",
            "avg_radius": "HTL_avg_radius"},
    "Perovskite": {"band_gap": "Perovskite_band_gap", "LUMO": "Perovskite_LUMO",
                   "HOMO": "Perovskite_HOMO",
                   "electron_mobility": "Perovskite_electron_mobility",
                   "hole_mobility": "Perovskite_hole_mobility",
                   "dielectric_constant": "Perovskite_dielectric_constant",
                   "tolerance_factor": "perovskite_tolerance_factor",
                   "octahedral_factor": "perovskite_octahedral_factor",
                   "tau_descriptor": "perovskite_tau_descriptor"},
}

SYNONYMS = {
    "spiro": "Spiro-MeOTAD", "spiro-ometad": "Spiro-MeOTAD", "spiroometad": "Spiro-MeOTAD",
    "ptaa": "PTAA", "pedot": "PEDOT:PSS", "pedot:pss": "PEDOT:PSS",
    "nio": "NiO", "nio(x)": "NiO",
    "tio2": "TiO2", "sno2": "SnO2", "zno": "ZnO", "pcbm": "PCBM", "c60": "C60",
    "me-4pac": "Me-4PACz", "me-4pacz": "Me-4PACz",
    "mapbi3": "MAPbI3", "fapbi3": "FAPbI3", "mampbi3": "MAPbI3",
    "csfama": "CsMAFA", "csmafa": "CsMAFA",
}


class MaterialPropertyTool(Tool):
    name = "material"
    description = (
        "Look up computed material properties (band gap, HOMO/LUMO, mobility, "
        "dielectric constant, atomic/ionic size) for ETLs, HTLs, perovskite "
        "absorbers and contacts. Returns value + unit + source (literature DOI "
        "or median across the experimental database)."
    )
    layer = "materials"
    version = "0.2.0"
    capabilities = {"material_properties", "lookup"}

    def __init__(self):
        self._db: Optional[pd.DataFrame] = None

    # ---- DB construction -------------------------------------------------
    def build_db(self) -> pd.DataFrame:
        """Merge literature property tables with PDB-derived medians.
        Literature rows win on duplicates; non-physical mobility values are
        dropped, then the merged table is cached in `self._db`."""
        rows = []
        for path in (config.LIT_PROPERTIES_CSV_1, config.LIT_PROPERTIES_CSV_2):
            if os.path.exists(path):
                rows.extend(self._lit_rows(pd.read_csv(path), os.path.basename(path)))
        lit = pd.DataFrame(rows)
        derived = self._derive_from_enriched()
        db = pd.concat([lit, derived], ignore_index=True) if derived is not None else lit
        db = db.drop_duplicates(subset=["material", "property"], keep="first")  # lit-wins
        db["value"] = pd.to_numeric(db["value"], errors="coerce")
        # drop non-physical / zero mobility rows (PDB "unknown" artifacts)
        db = db.loc[~((db["property"].isin(["electron_mobility", "hole_mobility"])) &
                      (db["value"] <= 0))]
        self._db = db.reset_index(drop=True)
        return self._db

    def _lit_rows(self, df: pd.DataFrame, src: str) -> List[dict]:
        keys = ["band_gap", "LUMO", "HOMO", "electron_mobility", "hole_mobility",
                "dielectric_constant", "min_radius", "max_radius", "avg_radius",
                "min_atomic_mass", "max_atomic_mass", "avg_atomic_mass",
                "en_min", "en_max", "en_avg"]
        rows = []
        for _, r in df.iterrows():
            name = str(r.get("common_name")) if str(r.get("common_name")) not in ("nan", "") \
                else str(r.get("formula"))
            for k in keys:
                v = r.get(k)
                if pd.notna(v):
                    rows.append(dict(material=name, property=k, value=float(v),
                                     unit=PROPERTY_UNITS.get(k, ""),
                                     source=f"{src} DOI={r.get('source','')}", n=1))
        return rows

    def _derive_from_enriched(self) -> Optional[pd.DataFrame]:
        if not os.path.exists(config.ENRICHED_CSV):
            return None
        df = pd.read_csv(config.ENRICHED_CSV, low_memory=False)
        out = []
        for role, mapping in ROLE_COLUMN_PROPS.items():
            name_col = role
            if name_col not in df.columns:
                continue
            for prop, col in mapping.items():
                if col not in df.columns:
                    continue
                vals = pd.to_numeric(df[col], errors="coerce")
                good = vals.notna()
                if not good.any():
                    continue
                for name, group in df.loc[good].groupby(df.loc[good, name_col].astype(str)):
                    if name in ("nan", "Unknown", ""):
                        continue
                    out.append(dict(material=name, property=prop,
                                    value=float(pd.to_numeric(group[col], errors="coerce").median()),
                                    unit=PROPERTY_UNITS.get(prop, ""),
                                    source="Perovskite Database (median)", n=len(group)))
        return pd.DataFrame(out) if out else None

    # ---- public lookups --------------------------------------------------
    @property
    def db(self) -> pd.DataFrame:
        """The merged material-property table (built lazily)."""
        if self._db is None:
            self.build_db()
        return self._db

    def resolve_material(self, name: str) -> Optional[str]:
        """Map a user/LLM material name to the canonical DB name: exact ->
        synonym -> unique substring -> shortest substring match."""
        name = str(name).strip()
        if not name:
            return None
        lower = name.lower()
        candidates = self.db["material"].astype(str).unique().tolist()
        for c in candidates:
            if c.lower() == lower:
                return c
        if lower in SYNONYMS and SYNONYMS[lower] in candidates:
            return SYNONYMS[lower]
        cont = [c for c in candidates if lower in c.lower()]
        if len(cont) == 1:
            return cont[0]
        if len(cont) > 1:
            return sorted(cont, key=len)[0]
        return None

    def lookup_property(self, material: str, prop: str) -> dict:
        """Single (material, property) lookup with value/unit/source."""
        m = self.resolve_material(material)
        if m is None:
            return {"found": False, "material": str(material), "property": prop,
                    "reason": "material not in DB"}
        rows = self.db[(self.db["material"] == m) & (self.db["property"] == prop)]
        if rows.empty:
            return {"found": False, "material": m, "property": prop,
                    "reason": "property not in DB"}
        best = rows.iloc[0]
        return {"found": True, "material": m, "property": prop,
                "value": best["value"], "unit": best["unit"],
                "source": best["source"], "n": int(best["n"])}

    def lookup_all(self, material: str) -> dict:
        """All recorded properties for one resolved material."""
        m = self.resolve_material(material)
        if m is None:
            return {"found": False, "material": str(material)}
        rows = self.db[self.db["material"] == m]
        return {"found": True, "material": m,
                "properties": [{"property": r["property"], "value": r["value"],
                                "unit": r["unit"], "source": r["source"]}
                               for r in rows.to_dict("records")]}

    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        q = params.get("material") or params.get("query")
        if not q:
            return {"error": "material required"}
        if params.get("property"):
            prop = str(params["property"]).lower()
            prop = {v.lower(): v for v in PROPERTY_UNITS}.get(prop, prop)
            return self.lookup_property(q, prop)
        if params.get("all"):
            return self.lookup_all(q)
        m = self.resolve_material(q)
        if m is None:
            return {"found": False, "material": str(q)}
        rows = self.db[self.db["material"] == m]
        return {"found": True, "material": m,
                "n_properties": int(rows.shape[0]),
                "data": rows[["property", "value", "unit", "source"]].to_dict("records"),
                "provenance": {"kind": "material_lookup", "resolved_name": m}}