"""ML feature pipeline (ML layer).

Two responsibilities:

  1. `prepare_features(df)` — reproduce the exact pre-fabrication feature
     recipe used to train the deployed model (engineered band-alignment and
     log-thickness features; missingness indicators + median fill; categorical
     columns kept as strings). PCE-leakage columns are auto-excluded.

  2. `DeployPipeline` — turn a *natural-language device description* into the
     numeric feature vector a saved model expects, resolving materials through
     the Material tool and filling the rest with training medians.
"""
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .. import config
from ..materials.properties import MaterialPropertyTool

LEAKAGE_COLUMNS = {
    "Ref_ID", "Ref_DOI_number", "final_pce", "Perovskite_composition_short_form",
    "doi_loo_mean", "doi_loo_std", "doi_count", "doi_sigma",
    "Ref_journal", "Ref_publication_date", "publication_year",
    "Stability_measured", "EQE_measured", "JV_hysteresis_index",
    "Cell_area_measured_cm2", "Stabilised_performance_measured",
}
PCE_LEAKAGE = {"JV_default_Voc", "JV_default_Jsc", "JV_default_FF"}

CAT_EXTRA = [
    "ETL", "HTL", "Perovskite",
    "Substrate_stack_sequence", "Backcontact_stack_sequence",
    "Perovskite_deposition_solvents", "HTL_deposition_solvents",
    "ETL_deposition_solvents",
    "Perovskite_deposition_synthesis_atmosphere",
    "Perovskite_deposition_thermal_annealing_atmosphere",
    "HTL_additives_compounds", "ETL_additives_compounds",
    "Backcontact_deposition_procedure",
]

CAT_COLS = ["Cell_architecture", "ETL_deposition_method", "HTL_deposition_method",
            "Perovskite_deposition_method"] + CAT_EXTRA

MATERIAL_KEYWORDS = {
    "ETL": {"sno2", "tio2", "zno", "pcbm", "c60", "sno2", "znox"},
    "HTL": {"spiro", "ptaa", "pedot", "nio", "p3ht", "cu2o", "cui",
            "me-4pacz", "npm"},
    "Perovskite": {"csmafa", "fama", "mapbi3", "fapbi3", "csfapbi3"},
}

# mapping candidate keyword -> canonical material name
MATERIAL_CANON = {
    "sno2": "SnO2", "tio2": "TiO2", "zno": "ZnO", "pcbm": "PCBM", "c60": "C60",
    "spiro": "Spiro-MeOTAD", "ptaa": "PTAA", "pedot": "PEDOT:PSS",
    "nio": "NiO", "p3ht": "P3HT", "me-4pacz": "Me-4PACz",
    "csmafa": "CsMAFA", "fama": "FA_MA", "mapbi3": "MAPbI3", "fapbi3": "FAPbI3",
}


class FeaturePipeline:
    """Clean wrapper around `prepare_features` + `build_manifest`."""

    @staticmethod
    def drop_columns(df: pd.DataFrame, predict_pce: bool = True) -> pd.DataFrame:
        cols = set(LEAKAGE_COLUMNS)
        if predict_pce:
            cols |= PCE_LEAKAGE
        return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore").copy()

    @staticmethod
    def numeric(df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(lambda s: pd.to_numeric(s, errors="coerce"))


def prepare_features(df: pd.DataFrame, predict_pce: bool = True) -> Tuple[pd.DataFrame, list]:
    """Match the flagship 'physics + materials' training recipe."""
    df = FeaturePipeline.drop_columns(df, predict_pce=predict_pce).copy()
    num = lambda s: pd.to_numeric(s, errors="coerce")   # noqa: E731
    df["HTL_hole_barrier"] = (num(df.get("HTL_HOMO")) - num(df.get("Perovskite_HOMO"))).abs()
    df["ETL_electron_barrier"] = (num(df.get("ETL_LUMO")) - num(df.get("Perovskite_LUMO"))).abs()
    df["HTL_hole_barrier_signed"] = num(df.get("HTL_HOMO")) - num(df.get("Perovskite_HOMO"))
    df["ETL_electron_barrier_signed"] = num(df.get("ETL_LUMO")) - num(df.get("Perovskite_LUMO"))
    if "ETL_band_gap" in df.columns and "Perovskite_band_gap" in df.columns:
        df["gap_diff_ETL_Pvk"] = num(df["ETL_band_gap"]) - num(df["Perovskite_band_gap"])
    if "HTL_band_gap" in df.columns and "Perovskite_band_gap" in df.columns:
        df["gap_diff_HTL_Pvk"] = num(df["HTL_band_gap"]) - num(df["Perovskite_band_gap"])
    df["log_abs_hole_barrier"] = np.log1p(df["HTL_hole_barrier"])
    df["log_abs_electron_barrier"] = np.log1p(df["ETL_electron_barrier"])
    if "Perovskite_deposition_quenching_induced_crystallisation" in df.columns:
        df = df.rename(columns={
            "Perovskite_deposition_quenching_induced_crystallisation":
            "Perov_quenching_crystallisation"})
    for c in list(df.columns):
        if df[c].dtype == "object" and c not in CAT_COLS:
            df = df.drop(columns=[c])
    df = df.drop(columns=[c for c in df.columns if c in df.columns and
                          df[c].isnull().mean() > 0.8])
    for c in df.columns:
        if c not in CAT_COLS and df[c].isnull().any():
            df[f"{c}_na"] = df[c].isnull().astype(float)
            df[c] = df[c].fillna(df[c].median())
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("unknown").astype(str)
        else:
            df[c] = "unknown"
    return df, CAT_COLS


def build_manifest(df: pd.DataFrame, cat_cols: list, out_dir: str) -> dict:
    """Persist the deploy-time schema (feature names, medians, quantiles)."""
    medians, qs = {}, {}
    for c in df.columns:
        if c not in cat_cols and pd.api.types.is_numeric_dtype(df[c]):
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s):
                medians[c] = float(s.median())
                qs[c] = [float(s.quantile(0.02)), float(s.quantile(0.98))]
            else:
                medians[c] = 0.0
                qs[c] = [0.0, 0.0]
    manifest = {"feature_names": df.columns.tolist(), "cat_features": cat_cols,
                "medians": medians, "quantiles": qs}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


class DeployPipeline:
    """Assemble a model-ready feature vector from a device description."""

    def __init__(self, manifest_path=None, material_tool: Optional[MaterialPropertyTool] = None,
                 feature_names: Optional[list] = None,
                 cat_features: Optional[list] = None):
        self.mat = material_tool
        self.manifest = None
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path) as f:
                self.manifest = json.load(f)
        # Authoritative schema: the model's own feature list takes priority over
        # the (possibly stale) manifest.
        if feature_names:
            if self.manifest:
                self.manifest = dict(self.manifest)
                self.manifest["feature_names"] = feature_names
                if cat_features is not None:
                    self.manifest["cat_features"] = cat_features
            else:
                self.manifest = {"feature_names": feature_names,
                                 "cat_features": cat_features or [],
                                 "medians": {}, "quantiles": {}}

    # ------------------------------------------------------------------
    # description -> raw features
    # ------------------------------------------------------------------
    _ELEM = r"(?:Cs|Rb|FA|MA|Pb|Sn|Bi|Br|Cl|I)"
    _FORMULA = re.compile(rf"(?:{_ELEM}(?:\d+(?:\.\d+)?)?)+")

    def parse_composition(self, comp: str) -> dict:
        """Cation / metal / halide fractions from the *original-case* description.

        Reads perovskite formulas such as ``MAPbI3``, ``FAPbI3`` or
        ``Cs0.05FA0.79MA0.16PbBr0.51I2.49`` (coefficients are normalised within
        each site: A = Cs/Rb/FA/MA, B = Pb/Sn/Bi, X = I/Br/Cl) and the spelled-out
        names (methylammonium, formamidinium, lead, tin, iodide, bromide, chloride).
        Only species that were actually found are returned, so callers can tell
        "stated" from "not stated" and apply their own defaults.
        """
        text = str(comp)
        found = {"A": {}, "B": {}, "X": {}}
        site = {"Cs": "A", "Rb": "A", "FA": "A", "MA": "A",
                "Pb": "B", "Sn": "B", "Bi": "B", "I": "X", "Br": "X", "Cl": "X"}
        triple = re.search(r"(?i)\bcs-?(?:ma-?fa|fa-?ma)\b", text)
        if triple:                       # triple-cation shorthand, e.g. CsFAMA / CsMAFA
            found["A"] = {"Cs": 0.05, "MA": 0.16, "FA": 0.79}
        for word in re.findall(r"[A-Za-z0-9.]+", text):
            m = self._FORMULA.fullmatch(word)
            if not m or len(re.findall(self._ELEM, word)) < 2:
                continue
            if triple and re.fullmatch(r"(?i)cs(?:ma|fa){2}", word):
                continue
            for el, coef in re.findall(rf"({self._ELEM})(\d+(?:\.\d+)?)?", word):
                found[site[el]][el] = found[site[el]].get(el, 0.0) + (float(coef) if coef else 1.0)
        low = text.lower()
        for el, names in {"MA": ("methylammonium", "methylamine"), "FA": ("formamidinium",),
                          "Pb": ("lead",), "Sn": ("tin ", " tin"), "I": ("iodide",),
                          "Br": ("bromide",), "Cl": ("chloride",)}.items():
            if el not in found[site[el]] and any(n in low for n in names):
                found[site[el]][el] = 1.0
        fr = {}
        for grp in found.values():
            total = sum(grp.values())
            if total > 0:
                fr.update({el: round(v / total, 4) for el, v in grp.items()})
        # A-site names that were found but resolved to zero stay absent; absent = "not stated"
        m = re.search(r"FAPbI3\)\s*([\d.]+)", text)
        if m:
            try:
                if 0 <= float(m.group(1)) <= 1:
                    fr["FA"] = float(m.group(1))
            except ValueError:
                pass
        return fr

    def _find_material(self, text: str, role: str) -> str:
        t = text.replace(" ", "")
        # longest keyword first: 'csfapbi3' must win over 'fapbi3' (sets are unordered)
        for kw in sorted(MATERIAL_KEYWORDS[role], key=lambda k: (-len(k), k)):
            if kw in t:
                return MATERIAL_CANON.get(kw, kw.title())
        return "unknown"

    def description_to_features(self, description: str):
        original = str(description)
        d = original.lower()
        f = {}

        # architecture: whole-word p-i-n / n-i-p only ("spin-coated" is not p-i-n)
        is_pin = re.search(r"\bp[\s-]?i[\s-]?n\b|\binverted\b", d)
        is_nip = re.search(r"\bn[\s-]?i[\s-]?p\b|\bregular\b|\bconventional\b", d)
        arch = "pin" if (is_pin and not is_nip) else "nip"
        f["Cell_architecture"] = arch

        # materials per role
        mat_names = {r: self._find_material(d, r)
                     for r in ("ETL", "HTL", "Perovskite")}
        for role, mname in mat_names.items():
            f[role] = mname

        # cation/halide/stoich fractions + band-gap hints
        pf = self.parse_composition(original)
        f.update({"a_FA_fraction": pf.get("FA", 0.0), "a_MA_fraction": pf.get("MA", 0.0),
                  "a_Cs_fraction": pf.get("Cs", 0.0), "a_Rb_fraction": pf.get("Rb", 0.0),
                  "b_Pb_fraction": pf.get("Pb", 1.0), "b_Sn_fraction": pf.get("Sn", 0.0),
                  "b_Bi_fraction": pf.get("Bi", 0.0),
                  "halide_I_fraction": pf.get("I", 1.0), "halide_Br_fraction": pf.get("Br", 0.0),
                  "halide_Cl_fraction": pf.get("Cl", 0.0)})

        # thicknesses / annealing (nm, degC, min)
        def _num(pattern, default=np.nan):
            m = re.search(pattern, d)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return default
            return default

        pvk = _num(r"perovskite(?:\s+\w+){0,2}\s*(?:of|~|:|\b)\s*(\d+)\s*nm")
        if np.isnan(pvk):
            pvk = _num(r"(\d{3,4})\s*nm", np.nan)
        f["Perovskite_thickness_nm"] = pvk
        f["ETL_thickness_nm"] = _num(r"etl(?:\s+\w+){0,2}\s*(?:of|~)\s*(\d+)\s*nm", np.nan)
        f["HTL_thickness_nm"] = _num(r"htl(?:\s+\w+){0,2}\s*(?:of|~)\s*(\d+)\s*nm", np.nan)
        f["Perov_anneal_temp_C"] = _num(r"anneal\w*\s*(?:at|of|:)?\s*(\d+)\s*c\b", np.nan)
        f["Perov_anneal_time_min"] = _num(r"anneal\w*\s{0,3}(?:\w+\s){0,3}?(\d+)\s*min", np.nan)

        if any(k in d for k in ("spin", "spin-coat")):
            f["Perovskite_deposition_method"] = "spin coating"
        elif any(k in d for k in ("vapor", "evaporat", "thermal evaporation")):
            f["Perovskite_deposition_method"] = "vapor"
        if any(k in d for k in ("quench", "antisolvent", "anti-solvent")):
            f["Perov_quenching_crystallisation"] = 1.0
        return f, mat_names

    # ------------------------------------------------------------------
    # raw features -> model vector
    # ------------------------------------------------------------------
    def assemble(self, features: dict) -> np.ndarray:
        """Convert the raw feature dict into the model's fixed schema order:
        start from training medians / 'unknown' cats, fill measured material
        properties via the Material tool, apply direct overrides, then rebuild
        derived band-alignment features (barriers, gap diffs, logs)."""
        if self.manifest is None:
            raise RuntimeError("manifest not loaded")
        names = self.manifest["feature_names"]
        cat = set(self.manifest["cat_features"])
        medians = self.manifest["medians"]

        row = {c: (0.0 if c.endswith("_na") else
                   ("unknown" if c in cat else medians.get(c, 0.0)))
               for c in names}

        # fill measured material properties via the Material tool
        for role, pref in (("ETL", "ETL_"), ("HTL", "HTL_"), ("Perovskite", "Perovskite_")):
            mname = features.get(role, "unknown")
            if mname == "unknown" or self.mat is None:
                continue
            props = self.mat.lookup_all(mname).get("properties", [])
            pmap = {p["property"]: p["value"] for p in props}
            colmap = {"band_gap": "band_gap", "LUMO": "LUMO", "HOMO": "HOMO",
                      "electron_mobility": "electron_mobility",
                      "hole_mobility": "hole_mobility",
                      "dielectric_constant": "dielectric_constant",
                      "min_radius": "min_radius", "max_radius": "max_radius",
                      "avg_radius": "avg_radius",
                      "tolerance_factor": "tolerance_factor",
                      "octahedral_factor": "octahedral_factor",
                      "tau_descriptor": "tau_descriptor"}
            for prop, col in colmap.items():
                cname = pref + col
                if cname in row and prop in pmap:
                    row[cname] = pmap[prop]

        # direct feature overrides
        for k, v in features.items():
            if k in row and not (isinstance(v, float) and np.isnan(v)):
                row[k] = v
            elif k in row and k in cat and isinstance(v, str):
                row[k] = v

        # rebuild derived band-alignment features
        def g(c):
            v = row.get(c)
            return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan
        def na_if_not(c, val):
            if c in row:
                ok = np.isfinite(val)
                row[c] = val if ok else 0.0
                row[f"{c}_na"] = 0.0 if ok else 1.0
        if "HTL_hole_barrier" in row:
            na_if_not("HTL_hole_barrier",
                      abs(g("HTL_HOMO") - g("Perovskite_HOMO")))
            na_if_not("HTL_hole_barrier_signed", g("HTL_HOMO") - g("Perovskite_HOMO"))
        if "ETL_electron_barrier" in row:
            na_if_not("ETL_electron_barrier",
                      abs(g("ETL_LUMO") - g("Perovskite_LUMO")))
            na_if_not("ETL_electron_barrier_signed", g("ETL_LUMO") - g("Perovskite_LUMO"))
        if "gap_diff_ETL_Pvk" in row:
            row["gap_diff_ETL_Pvk"] = (g("ETL_band_gap") - g("Perovskite_band_gap")
                                       if np.isfinite(g("ETL_band_gap")) and np.isfinite(g("Perovskite_band_gap")) else 0.0)
        if "gap_diff_HTL_Pvk" in row:
            row["gap_diff_HTL_Pvk"] = (g("HTL_band_gap") - g("Perovskite_band_gap")
                                       if np.isfinite(g("HTL_band_gap")) and np.isfinite(g("Perovskite_band_gap")) else 0.0)
        if "log_abs_hole_barrier" in row:
            row["log_abs_hole_barrier"] = np.log1p(row["HTL_hole_barrier"])
        if "log_abs_electron_barrier" in row:
            row["log_abs_electron_barrier"] = np.log1p(row["ETL_electron_barrier"])
        for c in ("ETL_thickness_nm", "HTL_thickness_nm"):
            if "log_" + c in row and c in row and np.isfinite(row[c]):
                row["log_" + c] = np.log1p(row[c])

        vec = []
        for c in names:
            if c in cat:
                vec.append(str(row[c]))
            else:
                v = row[c]
                vec.append(float(v) if v is not None and np.isfinite(v) else 0.0)
        return np.array(vec, dtype=object)