"""ML predictor tool (ML layer) — discrete action interface.

Unlike the old flat interface (``ml`` + ``action=predict`` + ``explain=true``),
the tool exposes five real actions, each with its own input schema, output
contract and preconditions:

    predict         PCE + 95% interval + OOD for a device description
    explain_global  mean |SHAP| across a deterministic design grid
    explain_local   SHAP for a single described device
    what_if         sweep ONE controllable input feature over values
    optimize        rank controllable features by predicted improvement

Routing is by ``params["action"]``; there is no way to smuggle one operation
into another via flags. what_if/optimize reject non-controllable (i.e.
measurement-derived, metadata, missingness) parameters.

Models live in ``config.MODEL_DIR`` (see config.py) and are lazily loaded.
Uncertainty comes from a bagged seed ensemble (spread across seeds), reported
as a 95% prediction interval — a prediction interval, not a confidence interval
on the mean.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import config
from ..core.tool import Tool
from ..materials.properties import MaterialPropertyTool
from .controllability import (classify_feature, describe_feature,
                              is_binary_structural, is_controllable,
                              valid_range)
from .features import DeployPipeline

MODEL_DIR = os.environ.get("PEROVAGENTS_ML_MODEL_DIR",
                           os.path.join(config.RESULTS_DIR, "ml"))

ARCH_ALIAS = {"overall": None, "pin": "PIN", "nip": "NIP", "p-i-n": "PIN", "n-i-p": "NIP"}

# categorical material choices for the global-design grid
GLOBAL_ETL = ["SnO2", "TiO2", "PCBM"]
GLOBAL_HTL = ["Spiro-MeOTAD", "PTAA", "PEDOT:PSS", "NiO"]
GLOBAL_PVK = ["CsMAFA", "MAPbI3", "FAPbI3", "FA_MA"]
GLOBAL_ARCHS = ["pin", "nip"]
GLOBAL_SUBSTRATE = ["ITO", "FTO", "glass"]
GLOBAL_BACKCONTACT = ["Au", "Ag", "Cu", "Al", "carbon"]

# controllable material-choice toggles for optimize (bounded alternative sets)
MATERIAL_TOGGLES = {
    "ETL": GLOBAL_ETL,
    "HTL": GLOBAL_HTL,
    "Perovskite": GLOBAL_PVK,
    "Substrate_stack_sequence": GLOBAL_SUBSTRATE,
    "Backcontact_stack_sequence": GLOBAL_BACKCONTACT,
}


class MLPredictorTool(Tool):
    name = "ml"
    description = (
        "Predict the PCE (%) of a perovskite solar cell from a device "
        "description using CatBoost models trained on the experimental "
        "database, explain which features the model associates with PCE "
        "(global or local), sweep a controllable input, or rank controllable "
        "changes. Every numeric result carries a 95% prediction interval and "
        "an out-of-applicability warning."
    )
    layer = "ml"
    version = "0.3.0"
    capabilities = {"regression", "prediction", "shap", "uncertainty",
                    "sensitivity", "interpretation"}

    def __init__(self, model_dir: Optional[str] = None):
        self.dir = str(model_dir) if model_dir else str(MODEL_DIR)
        self._models: Dict[str, Any] = {}
        self._metas: Dict[str, Any] = {}
        self._bags: Dict[str, list] = {}
        self._pipes: Dict[str, Any] = {}

    # ---- model plumbing --------------------------------------------------
    @staticmethod
    def _key(arch: str) -> str:
        return "Overall" if arch not in ("PIN", "NIP") else arch

    def _meta(self, arch):
        key = self._key(arch)
        if key not in self._metas:
            path = os.path.join(self.dir, f"features_{key}.json")
            if not os.path.exists(path):
                return None
            with open(path) as f:
                self._metas[key] = json.load(f)
        return self._metas[key]

    def _model(self, arch):
        key = self._key(arch)
        if key not in self._models:
            from catboost import CatBoostRegressor
            m = CatBoostRegressor()
            m.load_model(os.path.join(self.dir, f"model_{key}.cbm"))
            self._models[key] = m
        return self._models[key]

    def _load_bags(self, arch):
        key = self._key(arch)
        if key not in self._bags:
            outs = []
            for seed in (42, 7, 123):
                p = os.path.join(self.dir, f"bag_{key}_seed{seed}.cbm")
                if os.path.exists(p):
                    from catboost import CatBoostRegressor
                    m = CatBoostRegressor()
                    m.load_model(p)
                    outs.append(m)
            self._bags[key] = outs
        return self._bags[key]

    def _pipeline(self, arch="Overall"):
        if arch not in self._pipes:
            meta = self._meta(arch) or {}
            manifest = os.path.join(self.dir, "manifest.json")
            self._pipes[arch] = DeployPipeline(
                manifest,
                material_tool=MaterialPropertyTool(),
                feature_names=meta.get("feature_names"),
                cat_features=meta.get("cat_features"))
        return self._pipes[arch]

    # ---- feature assembly ------------------------------------------------
    def _vector(self, params, arch):
        desc = params.get("description")
        if desc:
            features, mats = self._pipeline(arch).description_to_features(desc)
            features.update(params.get("features") or {})
        elif params.get("features"):
            features, mats = dict(params["features"]), {}
        else:
            raise ValueError("description or features required")
        vec = self._pipeline(arch).assemble(features)
        meta = self._meta(arch) or {}
        n_model = len(meta.get("feature_names") or [])
        if n_model and len(vec) != n_model:
            raise ValueError(
                f"feature vector length {len(vec)} != model features {n_model}; "
                "model artifacts are out of sync")
        return vec, features, mats

    def _assemble_from(self, features: dict, arch: str):
        """Assemble a vector from raw feature overrides (recomputes derived)."""
        return self._pipeline(arch).assemble(features)

    def _rowpos(self, arch: str) -> Tuple[list, set, list]:
        meta = self._meta(arch or "Overall") or {}
        return (meta.get("feature_names") or [],
                set(meta.get("cat_features") or []),
                meta.get("feature_names") or [])

    def _valid_parameter(self, arch: str, parameter: str) -> str:
        """Resolve a (possibly fuzzy) parameter name to a model input feature,
        then run the what_if/optimize preconditions. Returns the canonical name."""
        names, cats, _ = self._rowpos(arch)
        name = self._resolve_parameter(names, cats, parameter)
        if name is None:
            raise ValueError(
                f"parameter '{parameter}' is not a known model input "
                f"(controllable numeric inputs include: "
                f"{[n for n in names if n not in cats and is_controllable(n)][:8]} …)")
        if name in cats:
            raise ValueError(
                f"parameter '{name}' is categorical; what_if needs a numeric "
                f"input feature")
        if name.endswith("_na"):
            raise ValueError(f"'{name}' is a missingness indicator, "
                             f"not a controllable input")
        rec = describe_feature(name, set(cats))
        if rec["controllability"] == "non_controllable":
            raise ValueError(
                f"parameter '{name}' is {rec['category']} "
                f"(not controllable); what_if/optimize only sweep experimental "
                f"inputs, not measurement-derived or metadata columns")
        if rec["type"] != "continuous":
            raise ValueError(
                f"parameter '{name}' is {rec['type']} (not a continuous input); "
                f"what_if needs a sweepable numeric knob like thickness, "
                f"annealing temperature/time")
        if is_binary_structural(name):
            raise ValueError(
                f"parameter '{name}' is a structural/binary flag, not a "
                f"continuous input; what_if needs a sweepable numeric knob "
                f"(e.g. thickness, annealing temperature/time)")
        return name

    _PARAM_ALIASES = {
        "perovskite thickness": "Perovskite_thickness_nm",
        "absorber thickness": "Perovskite_thickness_nm",
        "pvk thickness": "Perovskite_thickness_nm",
        "etl thickness": "ETL_thickness_nm",
        "htl thickness": "HTL_thickness_nm",
        "annealing temperature": "Perov_anneal_temp_C",
        "anneal temperature": "Perov_anneal_temp_C",
        "anneal temp": "Perov_anneal_temp_C",
        "annealing time": "Perov_anneal_time_min",
        "anneal time": "Perov_anneal_time_min",
        "hole mobility": "ETL_hole_mobility",
        "perovskite band gap": "Perovskite_band_gap",
        "band gap": "Perovskite_band_gap",
    }

    def _resolve_parameter(self, names, cats, parameter):
        if not parameter:
            return None
        p = str(parameter).lower().strip()
        p2 = re.sub(r"\s*\(nm\)|\s*nm\b|\s*temperature\b|\s*thickness\b", "", p).strip()
        for alias, canon in self._PARAM_ALIASES.items():
            if alias in p:
                return canon if canon in names else None
        if p in names or p2 in names:
            return p if p in names else p2
        candidates = []
        for n in names:
            if n in cats or n.endswith("_na") or not is_controllable(n):
                continue
            toks = set(re.findall(r"[a-z0-9]+", n.lower()))
            score = len(toks.intersection(set(re.findall(r"[a-z0-9]+", p))))
            if score:
                candidates.append((score, n))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return candidates[0][1]

    # ---- scoring ----------------------------------------------------------
    def _score(self, vec, arch: str) -> Tuple[float, float, float, int, float]:
        """One prediction: (point, lo, hi, n_bags, sigma)."""
        m = self._model(arch or "Overall")
        x = np.asarray(vec, dtype=object).reshape(1, -1)
        pred = float(m.predict(x)[0])
        bags = self._load_bags(arch or "Overall")
        if bags:
            preds = [float(b.predict(x)[0]) for b in bags]
            sigma = float(np.std(preds))
            lo, hi = max(0.0, pred - 1.96 * sigma), pred + 1.96 * sigma
        else:
            sigma, lo, hi = 0.7, max(0.0, pred - 1.96 * 0.7), pred + 1.96 * 0.7
        return pred, max(0.0, lo), hi, len(bags), sigma

    def _shap_of(self, vec, arch: str) -> List[Dict[str, Any]]:
        import shap
        m = self._model(arch or "Overall")
        x = np.asarray(vec, dtype=object).reshape(1, -1)
        explainer = shap.TreeExplainer(m)
        sv = explainer.shap_values(x)[0]
        names = self._meta(arch or "Overall")["feature_names"]
        out = []
        for i, s in zip(np.argsort(-np.abs(sv)), sv):
            v = float(s)
            out.append({"name": names[i], "importance": round(abs(v), 4),
                        "direction": "positive" if v >= 0 else "negative",
                        "shap": round(v, 4)})
        return out

    # ---- API: actions -----------------------------------------------------
    def predict(self, params):
        """Predict PCE for a described (or feature-supplied) device, with a
        95% prediction interval and an out-of-envelope flag."""
        arch = ARCH_ALIAS.get(str(params.get("architecture", "overall")).lower(), None)
        meta = self._meta(arch or "Overall")
        if meta is None:
            return {"status": "error",
                    "error": f"no model artifacts found in {self.dir}; train the model first"}
        vec, features, mats = self._vector(params, arch or "Overall")
        pred, lo, hi, n_bags, sigma = self._score(vec, arch or "Overall")
        return {"result_type": "prediction",
                "architecture": arch or "Overall",
                "predicted_pce": round(pred, 2),
                "pce_95_interval": [round(lo, 2), round(hi, 2)],
                "interval_spread_percent": round(hi - lo, 2),
                "n_bags": n_bags,
                "hood": self.ood(vec, arch or "Overall"),
                "resolved_materials": mats,
                "provenance": {"kind": "ml_prediction", "model": f"model_{(arch or 'Overall')}.cbm",
                               "target": "PCE"}}

    def ood(self, vec, arch="Overall", threshold=0.1):
        """Count features outside the training 2-98% envelope (+50% band)."""
        meta = self._meta(arch)
        if meta is None:
            return {"applied": False}
        quantiles = (self._pipeline(arch).manifest or {}).get("quantiles", {})
        names, cats = meta["feature_names"], set(meta["cat_features"])
        n_out, details = 0, []
        for i, c in enumerate(names):
            if c in cats:
                continue
            q = quantiles.get(c)
            if not q or len(q) != 2:
                continue
            v = vec[i]
            if v is None or not np.isfinite(v):
                continue
            lo, hi = q[0], q[1]
            margin = (hi - lo) * 0.5
            if v < lo - margin or v > hi + margin:
                n_out += 1
                if len(details) < 8:
                    details.append({"feature": c, "value": float(v), "range": [lo, hi]})
        total = sum(1 for c in names if c not in cats and c in quantiles)
        frac = n_out / max(total, 1)
        return {"applied": True, "n_out_of_envelope": n_out,
                "n_features_checked": total, "fraction_out": round(frac, 4),
                "ood": frac > threshold or n_out > 12, "details": details}

    # ---- explanations ---------------------------------------------------
    def explain_local(self, params):
        """Per-sample SHAP explanation for the described device (top-k
        features with direction; controllability tags)."""
        arch = ARCH_ALIAS.get(str(params.get("architecture", "overall")).lower(), None) or "Overall"
        meta = self._meta(arch)
        if meta is None:
            return {"status": "error", "error": "no model artifacts"}
        vec, _, mats = self._vector(params, arch)
        try:
            feats = self._shap_of(vec, arch)[: params.get("top_k", 15)]
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        return {"result_type": "explanation", "scope": "local",
                "method": "shap", "architecture": arch,
                "features": feats,
                "controllable": {f["name"]: is_controllable(f["name"])
                                 for f in feats},
                "resolved_materials": mats,
                "provenance": {"kind": "shap_explanation", "model": f"model_{arch}.cbm",
                               "scope": "local"}}

    def explain_global(self, params):
        """Aggregate |SHAP| over a deterministic design grid of architecture ×
        ETL × HTL × perovskite choices (medians for continuous features)."""
        arch = ARCH_ALIAS.get(str(params.get("architecture", "overall")).lower(), None) or "Overall"
        meta = self._meta(arch)
        if meta is None:
            return {"status": "error", "error": "no model artifacts"}
        pipe = self._pipeline(arch)
        acc, count = {}, 0
        for a, etl, htl, pvk in self._design_grid():
            desc = f"{a} device with {etl} ETL, {htl} HTL, {pvk} perovskite"
            try:
                feats, _ = pipe.description_to_features(desc)
                vec = pipe.assemble(feats)
            except Exception:
                continue
            try:
                sv = self._shap_of(vec, arch)
            except Exception:
                continue
            for r in sv:
                s = acc.setdefault(r["name"], np.zeros(4))  # [sum|s|, sum s, n, _]
                s[0] += r["importance"]
                s[1] += r["shap"]
                s[2] += 1
            count += 1
        if count == 0:
            return {"status": "error", "error": "design grid produced no valid samples"}
        feats = [{"name": k,
                  "importance": round(float(v[0]) / v[2], 4),
                  "direction": "positive" if v[1] >= 0 else "negative",
                  "shap": round(float(v[1]) / v[2], 4)}
                 for k, v in acc.items()]
        feats.sort(key=lambda f: f["importance"], reverse=True)
        top = feats[: params.get("top_k", 15)]
        return {"result_type": "explanation", "scope": "global",
                "method": "shap", "architecture": arch,
                "n_design_samples": count,
                "features": top,
                "controllable": {f["name"]: is_controllable(f["name"])
                                 for f in top},
                "provenance": {"kind": "shap_explanation",
                               "model": f"model_{arch}.cbm", "scope": "global"}}

    def _design_grid(self):
        for a in GLOBAL_ARCHS:
            for etl in GLOBAL_ETL:
                for htl in GLOBAL_HTL:
                    for pvk in GLOBAL_PVK:
                        yield a, etl, htl, pvk

    # ---- intervention ----------------------------------------------------
    def what_if(self, params):
        """Sweep a single controllable parameter over `values` and return the
        predicted PCE (with intervals) at each point, clamped to the declared
        valid range when one exists."""
        arch = ARCH_ALIAS.get(str(params.get("architecture", "overall")).lower(), None)
        arch = arch or "Overall"
        meta = self._meta(arch)
        if meta is None:
            return {"status": "error", "error": "no model artifacts"}
        parameter = self._valid_parameter(arch, params.get("parameter"))
        values = params.get("values")
        if not values:
            raise ValueError("what_if requires a non-empty 'values' list")

        pipe = self._pipeline(arch)
        desc = params.get("description")
        if desc:
            base_features, mats = pipe.description_to_features(desc)
        else:
            base_features, mats = dict(params.get("features") or {}), {}
        base_features = dict(base_features)
        baseline = base_features.get(parameter)

        vr = valid_range(parameter)
        clamped_any, rows, hoods = False, [], []
        for v in values:
            feats = dict(base_features)
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"what_if values must be numeric, got {v!r}")
            if vr is not None:
                if not (vr[0] <= v <= vr[1]):
                    clamped_any = True
                v = min(max(v, vr[0]), vr[1])
            feats[parameter] = v
            vec = pipe.assemble(feats)
            pred, lo, hi, n_bags, sigma = self._score(vec, arch)
            hood = self.ood(vec, arch)
            rows.append({"parameter_value": v,
                         "predicted_pce": round(pred, 2),
                         "interval_lo": round(lo, 2),
                         "interval_hi": round(hi, 2),
                         "hood": hood.get("ood")})
            if rows[-1]:
                hoods.append(hood)
        return {"result_type": "what_if",
                "architecture": arch,
                "parameter": parameter,
                "category": classify_feature(parameter),
                "controllable": True,
                "valid_range": vr,
                "clamped_to_valid_range": clamped_any,
                "base_value": None if baseline is None or not np.isfinite(float(baseline)) else float(baseline),
                "data": rows,
                "hood": hoods[0] if hoods else {"applied": False},
                "provenance": {"kind": "ml_what_if", "model": f"model_{arch}.cbm",
                               "parameter": parameter, "target": "PCE"}}

    def optimize(self, params):
        """Rank controllable changes by predicted PCE improvement.

        Two kinds of change are considered, both model-derived (SHAP-driven,
        not causal):
          1. numeric controllable inputs already set in the described device
             are swept over a ±20% grid around their baseline; influential
             controllable knobs the description leaves unset are swept around
             their training-median default (flagged source=\"default_stack\");
          2. influential material choices (ETL/HTL/perovskite/substrate/back
             contact) are toggled over their bounded alternative set; material
             slots the description leaves as \"unknown\" are evaluated against
             the median-default stack rather than skipped.
        Recommendations never derive from measurement-derived, metadata or
        missingness columns."""
        arch = ARCH_ALIAS.get(str(params.get("architecture", "overall")).lower(), None)
        arch = arch or "Overall"
        meta = self._meta(arch)
        if meta is None:
            return {"status": "error", "error": "no model artifacts"}
        desc = params.get("description")
        pipe = self._pipeline(arch)
        if desc:
            base_features, mats = pipe.description_to_features(desc)
        else:
            base_features, mats = dict(params.get("features") or {}), {}
        base_features = dict(base_features)
        names, cats, _ = self._rowpos(arch)

        local = self.explain_local({"description": desc or "",
                                    "architecture": arch.lower(),
                                    "features": params.get("features"),
                                    "top_k": 25})
        influential = {f["name"]: f for f in local["features"]}
        if not influential:
            raise ValueError("explanation produced no features; nothing to optimize")

        def _score_raw(feats):
            vec = pipe.assemble(dict(feats))
            pred, lo, hi, *_ = self._score(vec, arch)
            return pred, lo, hi

        recs = []
        medians = (pipe.manifest or {}).get("medians", {})
        # Numeric sweep. Two passes:
        #  a. features already set in the description (SHAP-influential gate
        #     does not apply — the user explicitly gave them);
        #  b. declared-range controllable knobs the description left unset,
        #     seeded from their training-median default so an underspecified
        #     stack still yields actionable guidance (flagged default_stack).
        sweep_set = set()
        for fname in base_features:
            if fname in names and fname not in cats and not fname.endswith("_na"):
                if is_controllable(fname) and not is_binary_structural(fname):
                    sweep_set.add(fname)
        for fname in names:
            if fname in cats or fname.endswith("_na") or not is_controllable(fname):
                continue
            if is_binary_structural(fname):
                continue
            if valid_range(fname) is not None:
                sweep_set.add(fname)
        for fname in sorted(sweep_set):
            raw = base_features.get(fname)
            try:
                base = float(raw)
            except (TypeError, ValueError):
                base = float("nan")
            source = "described"
            if not np.isfinite(base) or base <= 0:
                median = float(medians.get(fname, float("nan")))
                if not np.isfinite(median) or median <= 0:
                    continue
                base = median
                source = "default_stack"
            if not np.isfinite(base) or base <= 0:
                continue
            grid = [round(base + base * d, 2) for d in (-0.2, -0.1, 0.0, 0.1, 0.2)]
            vr = valid_range(fname)
            if vr is not None:
                grid = [min(max(v, vr[0]), vr[1]) for v in grid]
                grid = sorted(set(grid))
            rows = []
            for v in grid:
                feats = dict(base_features)
                feats[fname] = v
                pred, lo, hi = _score_raw(feats)
                rows.append({"parameter_value": v, "predicted_pce": round(pred, 2),
                             "interval_lo": round(lo, 2), "interval_hi": round(hi, 2)})
            best = max(rows, key=lambda r: r["predicted_pce"])
            base_pred = rows[2]["predicted_pce"] if len(rows) > 2 else rows[0]["predicted_pce"]
            if round(best["predicted_pce"] - base_pred, 2) <= 0.0:
                continue
            recs.append({"parameter": fname,
                         "kind": "numeric",
                         "category": classify_feature(fname),
                         "base_value": base,
                         "source": source,
                         "direction": "increase" if best["parameter_value"] > base
                                      else ("decrease" if best["parameter_value"] < base
                                            else "unchanged"),
                         "recommended_value": best["parameter_value"],
                         "base_predicted_pce": base_pred,
                         "best_predicted_pce": best["predicted_pce"],
                         "delta_predicted_pce": round(best["predicted_pce"] - base_pred, 2),
                         "values": rows})

        for slot, alts in MATERIAL_TOGGLES.items():
            if slot not in influential:
                continue
            cur = base_features.get(slot)
            s = str(cur or "").strip()
            base_pred = _score_raw(base_features)[0]
            best_alt, best_pred, best_lo, best_hi = None, base_pred, None, None
            rows = []
            for alt in alts:
                if s and alt == cur:
                    continue
                feats = dict(base_features)
                feats[slot] = alt
                pred, lo, hi = _score_raw(feats)
                rows.append({"material": alt, "predicted_pce": round(pred, 2),
                             "interval_lo": round(lo, 2), "interval_hi": round(hi, 2)})
                if pred > best_pred:
                    best_pred, best_alt, best_lo, best_hi = pred, alt, lo, hi
            if best_alt is None or round(best_pred - base_pred, 2) <= 0.0:
                continue
            specified = bool(s and s.lower() not in ("nan", "unknown", "none"))
            recs.append({"parameter": slot,
                         "kind": "material_choice",
                         "category": "material_choice",
                         "base_value": cur if specified else None,
                         "source": "described" if specified else "default_stack",
                         "direction": (f"switch {slot} {cur} → {best_alt}"
                                       if specified else f"set {slot} to {best_alt}"),
                         "recommended_value": best_alt,
                         "base_predicted_pce": round(base_pred, 2),
                         "best_predicted_pce": round(best_pred, 2),
                         "delta_predicted_pce": round(best_pred - base_pred, 2),
                         "values": rows})

        recs.sort(key=lambda r: r["delta_predicted_pce"], reverse=True)
        recs = recs[: params.get("top_k", 3)]
        if not recs:
            note = ("No controllable change among the inputs the model "
                    "deemed influential improved the predicted PCE above the "
                    "baseline. The described stack may already be near the "
                    "model optimum. Try describing a different stack or adding "
                    "a thickness / annealing input to sweep.")
            pred, lo, hi, *_ = self._score(pipe.assemble(dict(base_features)), arch)
            return {"result_type": "optimization",
                    "optimization_status": "none_found",
                    "architecture": arch,
                    "recommendations": [],
                    "prediction": round(pred, 2),
                    "predicted_pce": round(pred, 2),
                    "base_predicted_pce": round(pred, 2),
                    "interval": [round(lo, 2), round(hi, 2)],
                    "pce_95_interval": [round(lo, 2), round(hi, 2)],
                    "baseline_description": desc or "",
                    "note": note,
                    "provenance": {"kind": "ml_optimize",
                                   "model": f"model_{arch}.cbm",
                                   "requires_experimental_validation": True}}
        from_default = [r for r in recs if r.get("source") == "default_stack"]
        note = ("Recommendations are model-derived sensitivities over "
                "controllable inputs, not causal or experimentally "
                "verified claims")
        if from_default:
            note += ("; recommendations marked source=default_stack are "
                     "evaluated against the training-median default stack "
                     "because the description left that input unspecified "
                     "(e.g. no thickness or annealing given).")
        return {"result_type": "optimization",
                "architecture": arch,
                "recommendations": recs,
                "note": note,
                "provenance": {"kind": "ml_optimize", "model": f"model_{arch}.cbm",
                               "requires_experimental_validation": True}}

    # ---- dispatcher -------------------------------------------------------
    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch to predict / explain_global / explain_local / what_if /
        optimize based on the `action` param."""
        action = params.get("action", "predict")
        if action == "predict":
            return self.predict(params)
        if action == "explain_global":
            return self.explain_global(params)
        if action == "explain_local":
            return self.explain_local(params)
        if action == "what_if":
            return self.what_if(params)
        if action == "optimize":
            return self.optimize(params)
        raise ValueError(
            f"unknown ml action '{action}'. Use predict, explain_global, "
            f"explain_local, what_if, or optimize — parameters such as "
            f"'explain' are not actions")