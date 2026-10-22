"""Perovskite Database (PDB) query tool (Materials layer).

Executes structured Pandas plans against the experimental PSC device database
(40k+ records). The planner produces a JSON plan; this tool validates it
against the schema and executes:

    {
      "operation": "count | filter | top_k | groupby_aggregate | global_aggregate |
                    unique | distribution | sample | describe_columns",
      "filters":  [{"column": "...", "op": "contains|eq|gt|gte|lt|lte|in|...", "value": ...}],
      "group_by": "...", "aggregation": {"column": "...", "agg": "mean|median|..."},
      "sort_by": "...", "ascending": bool, "k": int, "limit": int,
      "select_columns": [...]
    }

Aliases map user-facing names (PCE, Voc, HTL, ...) to real column names, and
every device-level query auto-attaches DOIs for provenance.
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .. import config
from ..core.tool import Tool

OPS = {"eq", "ne", "contains", "not contains", "gt", "gte", "lt", "lte", "in", "regex"}

ALIAS_COLUMNS = {
    "PCE": "JV_default_PCE", "efficiency": "JV_default_PCE",
    "power conversion efficiency": "JV_default_PCE", "pv_pce": "JV_default_PCE",
    "pce_percent": "JV_default_PCE", "pce_%": "JV_default_PCE",
    "avg_pce": "JV_default_PCE", "mean_pce": "JV_default_PCE",
    "Voc": "JV_default_Voc", "open_circuit_voltage": "JV_default_Voc",
    "Jsc": "JV_default_Jsc", "short_circuit_current": "JV_default_Jsc",
    "FF": "JV_default_FF", "fill_factor": "JV_default_FF",
    "architecture": "Cell_architecture", "cell_architecture": "Cell_architecture",
    "HTL": "HTL_stack_sequence", "htl": "HTL_stack_sequence",
    "hole_transport_layer": "HTL_stack_sequence",
    "htl_material": "HTL_stack_sequence", "htl_stack": "HTL_stack_sequence",
    "ETL": "ETL_stack_sequence", "etl": "ETL_stack_sequence",
    "electron_transport_layer": "ETL_stack_sequence",
    "etl_material": "ETL_stack_sequence", "etl_stack": "ETL_stack_sequence",
    "Perovskite": "Perovskite_stack_sequence", "absorber": "Perovskite_stack_sequence",
    "composition": "Perovskite_composition_short_form",
    "Substrate": "Substrate_stack_sequence", "Backcontact": "Backcontact_stack_sequence",
    "DOI": "Ref_DOI_number", "doi": "Ref_DOI_number",
}

MATERIAL_COLUMN_MAP = {
    "Spiro-MeOTAD": ["HTL_stack_sequence", "Cell_stack_sequence"],
    "PTAA": ["HTL_stack_sequence", "Cell_stack_sequence"],
    "PEDOT:PSS": ["HTL_stack_sequence", "Cell_stack_sequence"],
    "P3HT": ["HTL_stack_sequence", "Cell_stack_sequence"],
    "NiO": ["HTL_stack_sequence", "Cell_stack_sequence"],
    "CuI": ["HTL_stack_sequence", "Cell_stack_sequence"],
    "TiO2": ["ETL_stack_sequence", "Cell_stack_sequence"],
    "SnO2": ["ETL_stack_sequence", "Cell_stack_sequence"],
    "PCBM": ["ETL_stack_sequence", "Cell_stack_sequence"],
    "C60": ["ETL_stack_sequence", "Cell_stack_sequence"],
    "ZnO": ["ETL_stack_sequence", "Cell_stack_sequence"],
}

LAYER_COLUMNS = ["HTL_stack_sequence", "ETL_stack_sequence",
                 "Perovskite_stack_sequence", "Perovskite_composition_short_form",
                 "Cell_stack_sequence", "Backcontact_stack_sequence",
                 "Substrate_stack_sequence"]

DEFAULT_SELECT = ["Cell_architecture", "HTL_stack_sequence", "ETL_stack_sequence",
                  "Perovskite_stack_sequence", "JV_default_PCE"]


class PDBTool(Tool):
    name = "pdb"
    description = (
        "Query the experimental Perovskite Solar Cell database (43k device "
        "records, 410 columns). Operations: count, filter, top_k, "
        "groupby_aggregate, global_aggregate, unique, distribution, sample. "
        "Use for factual/historical device questions: average PCE, device "
        "counts, material popularity, architecture comparisons."
    )
    layer = "materials"
    version = "0.2.0"
    capabilities = {"structured_query", "database", "aggregation"}

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path or config.PDB_CSV
        self._df: Optional[pd.DataFrame] = None

    # ---- data ------------------------------------------------------------
    @property
    def df(self) -> pd.DataFrame:
        """Lazily-loaded full table plus two helper columns (`arch_clean`
        normalized architecture, `_doi` string DOI) used by filters."""
        if self._df is None:
            df = pd.read_csv(self.csv_path, low_memory=False)
            df["arch_clean"] = df["Cell_architecture"].astype(str).str.lower()
            df["_doi"] = df["Ref_DOI_number"].astype(str)
            self._df = df
        return self._df

    def resolve_column(self, raw: str) -> Optional[str]:
        """Map a user/LLM column name to a real schema column.

        Resolution order: alias table -> exact match -> case-insensitive ->
        substring / underscore-stripped match. Returns None when unknown.
        """
        if raw is None:
            return None
        if raw in ALIAS_COLUMNS:
            return ALIAS_COLUMNS[raw]
        if raw in self.df.columns:
            return raw
        lowered = str(raw).strip().lower().replace(" ", "_")
        for c in self.df.columns:
            if c.lower() == lowered:
                return c
        for c in self.df.columns:
            if lowered in c.lower() or c.lower().replace("_", "") == lowered.replace("_", ""):
                return c
        return None

    def _resolve_any_column(self, value: str) -> Optional[str]:
        v = str(value)
        for c in LAYER_COLUMNS:
            if self.df[c].astype(str).str.contains(v, case=False).any():
                return c
        return None

    # ---- filters ---------------------------------------------------------
    def apply_filters(self, df: pd.DataFrame, filters: List[dict]) -> pd.DataFrame:
        """Apply a list of filter dicts sequentially (ops: eq/ne/contains/
        gt/gte/lt/lte/in/regex/isnull). Architecture filters are normalized
        to `arch_clean` so 'p-i-n'/'n-i-p' spellings match."""
        for f in filters or []:
            col = self.resolve_column(f.get("column"))
            if col is None:
                col = self._resolve_any_column(f.get("value"))
                if col is None:
                    raise ValueError(f"unknown filter column: {f.get('column')}")
            op = str(f.get("op", "contains")).lower()
            val = f.get("value")
            # normalize architecture values ('p-i-n' -> 'pin', 'n-i-p' -> 'nip')
            if col == "Cell_architecture" or col == "arch_clean":
                col = "arch_clean"
                val = str(val).lower().replace("p-i-n", "pin").replace("n-i-p", "nip")
                if val in ("nip", "pin", "planar", "meso"):
                    df = df[df["arch_clean"].isin({"nip", "pin", "planar", "meso"}.intersection({val, "nip", "pin"}))] if val in ("nip","pin") else df[df["arch_clean"] == val]
                    continue
            if f.get("flag") == "isnull":
                df = df[df[col].isna()] if op == "eq" else df[df[col].notna()]
                continue
            if col in {"Ref_DOI_number", "_doi"}:
                col = "_doi"
            if op not in OPS:
                raise ValueError(f"bad op: {op}")
            target = df[col]
            if op in {"gt", "gte", "lt", "lte"}:
                target = pd.to_numeric(target, errors="coerce")
                df = df[target.notna()]
                target = df[col].astype(float)
                if op == "gt": df = df[target > float(val)]
                elif op == "gte": df = df[target >= float(val)]
                elif op == "lt": df = df[target < float(val)]
                else: df = df[target <= float(val)]
            elif op == "eq":
                df = df[target.astype(str).str.lower() == str(val).lower()]
            elif op == "ne":
                df = df[target.astype(str).str.lower() != str(val).lower()]
            elif op == "in":
                vals = [str(v).lower() for v in (val if isinstance(val, list) else [val])]
                if vals:
                    df = df[target.astype(str).str.lower().isin(vals)]
            elif op == "contains":
                df = df[target.astype(str).str.contains(str(val), case=False, na=False)]
            elif op == "not contains":
                df = df[~target.astype(str).str.contains(str(val), case=False, na=False)]
            elif op == "regex":
                df = df[target.astype(str).str.contains(str(val), case=False, na=False, regex=True)]
        return df

    def _select_columns(self, plan: dict) -> List[str]:
        """Resolve requested columns, always appending the DOI for provenance."""
        sel = plan.get("select_columns") or DEFAULT_SELECT
        resolved = []
        for c in sel:
            rc = self.resolve_column(c)
            if rc and rc not in resolved:
                resolved.append(rc)
        return resolved + ([] if "Ref_DOI_number" in resolved else ["Ref_DOI_number"])

    # ---- main execution --------------------------------------------------
    def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        plan = dict(params)

        # --- tolerate LLM alternate key naming -----------------------------
        # The LLM planner sometimes emits operator/column/value or an
        # aggregate_column; normalize to the canonical plan schema.
        norm_filters = []
        for f in plan.get("filters") or []:
            f = dict(f)
            f["op"] = f.get("op") or ("eq" if f.get("operator") == "==" else
                                      f.get("operator", "contains"))
            if f.get("operator") in (">", ">=", "<", "<="):
                f["op"] = {">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}[f["operator"]]
            if "column" not in f and f.get("field"):
                f["column"] = f["field"]
            if "value" not in f and "values" in f:
                f["value"] = f["values"]
            norm_filters.append(f)
        plan["filters"] = norm_filters
        if "aggregation" not in plan and plan.get("aggregate_column"):
            plan["aggregation"] = {"column": plan["aggregate_column"],
                                   "agg": plan.get("aggregate_function", "mean")}
        if plan.get("group") and not plan.get("group_by"):
            plan["group_by"] = plan["group"]
        if plan.get("metric") and not plan.get("aggregation"):
            plan["aggregation"] = {"column": plan["metric"], "agg": "mean"}
        if plan.get("sort") and not plan.get("sort_by"):
            plan["sort_by"] = plan["sort"]

        op = plan.get("operation")
        if op is None:
            raise ValueError("missing operation")
        df = self.df
        filters = plan.get("filters") or []
        mask = self.apply_filters(df, filters)
        rows_before, rows_after = len(df), len(mask)
        out: Dict[str, Any] = {"rows_before": rows_before, "rows_after": rows_after,
                               "n_records": rows_after,
                               "n_papers": int(mask["_doi"].nunique()) if rows_after else 0,
                               "warnings": []}

        if op == "count":
            out.update({"result_type": "count", "value": rows_after})
        elif op == "filter":
            select = self._select_columns(plan)
            out.update({"result_type": "table",
                        "data": mask.head(plan.get("limit", 20))[select].to_dict("records")})
        elif op == "top_k":
            select = self._select_columns(plan)
            sort_by = self.resolve_column(plan.get("sort_by"))
            sub = mask
            if sort_by:
                sub = sub.assign(_srt=pd.to_numeric(sub[sort_by], errors="coerce")).dropna(subset=["_srt"])
                sub = sub.sort_values("_srt", ascending=bool(plan.get("ascending", False)))
            out.update({"result_type": "table", "k": plan.get("k", 10),
                        "sort_by": sort_by,
                        "data": sub.head(plan.get("k", 10))[select].to_dict("records")})
        elif op == "groupby_aggregate":
            out.update(self._groupby_aggregate(mask, plan))
        elif op == "global_aggregate":
            out.update(self._global_aggregate(mask, plan))
        elif op == "unique":
            col = self.resolve_column(plan.get("column") or plan.get("group_by"))
            if col is None:
                raise ValueError("column not found")
            vals = mask[col].astype(str).value_counts()
            out.update({"result_type": "table", "column": col,
                        "data": [{"value": str(k), "count": int(v)} for k, v in vals.items()]})
        elif op == "distribution":
            col = self.resolve_column(plan.get("column"))
            if col is None:
                raise ValueError("column not found")
            numeric = pd.to_numeric(mask[col], errors="coerce").dropna()
            out.update({"result_type": "distribution", "column": col,
                        "n": int(len(numeric)),
                        "min": float(numeric.min()) if len(numeric) else None,
                        "max": float(numeric.max()) if len(numeric) else None,
                        "mean": float(numeric.mean()) if len(numeric) else None,
                        "median": float(numeric.median()) if len(numeric) else None})
        elif op == "sample":
            n = plan.get("k", 5)
            select = self._select_columns(plan)
            out.update({"result_type": "table",
                        "data": mask.sample(min(n, len(mask)), random_state=42)[select].to_dict("records")})
        elif op == "describe_columns":
            out.update(self._describe_columns(mask, plan))
        else:
            raise ValueError(f"bad operation: {op}")

        out["provenance"] = {"kind": "pdb_query", "source_file": os.path.basename(self.csv_path),
                             "rows_before": rows_before, "rows_after": rows_after,
                             "n_papers": out.get("n_papers", 0),
                             "dois": sorted(set(mask["_doi"].tolist()))[:25]}
        return out

    def _groupby_aggregate(self, mask: pd.DataFrame, plan: dict) -> dict:
        gb = self.resolve_column(plan.get("group_by"))
        if gb is None:
            raise ValueError("group_by column not found")
        agg = plan.get("aggregation") or {}
        agg_col = self.resolve_column(agg.get("column") or plan.get("metric") or "JV_default_PCE")
        agg_fn = agg.get("agg", agg.get("function", "mean"))
        numeric = pd.to_numeric(mask[agg_col], errors="coerce")
        valid = numeric.notna()
        sub, numeric = mask[valid], numeric[valid]
        g = sub.groupby(gb, dropna=False)
        funcs = {"mean": "mean", "median": "median", "min": "min",
                 "max": "max", "std": "std", "count": "count"}
        if agg_fn not in funcs:
            raise ValueError(f"bad agg: {agg_fn}")
        res = []
        for key, idx in g.groups.items():
            vals = pd.to_numeric(mask.loc[idx, agg_col], errors="coerce").dropna()
            if agg_fn == "count":
                value = len(idx)
            elif len(vals):
                value = getattr(vals, funcs[agg_fn])()
            else:
                value = None
            res.append({"group": str(key), "value": None if value is None else float(value)})
        return {"result_type": "table", "group_by": gb, "aggregation": agg_fn,
                "target": agg_col, "data": res, "n_grouped": len(res),
                "n_values": int(numeric.notna().sum())}

    def _global_aggregate(self, mask: pd.DataFrame, plan: dict) -> dict:
        agg = plan.get("aggregation") or {}
        agg_col = self.resolve_column(agg.get("column") or plan.get("metric") or "JV_default_PCE")
        agg_fn = agg.get("agg", "mean")
        numeric = pd.to_numeric(mask[agg_col], errors="coerce").dropna()
        funcs = {"mean": np.mean, "median": np.median, "min": np.min, "max": np.max,
                 "std": np.std, "count": len}
        if agg_fn not in funcs:
            raise ValueError(f"bad agg: {agg_fn}")
        value = funcs[agg_fn](numeric) if agg_fn != "count" else len(numeric)
        out = {"result_type": "scalar", "column": agg_col, "agg": agg_fn,
               "value": None if value is None or (isinstance(value, float) and not np.isfinite(value)) else float(value),
               "n_values": int(len(numeric))}
        bounds = config.PHYSICS_BOUNDS.get(agg_col)
        if bounds and out["value"] is not None and out["value"] > bounds["max"]:
            out.setdefault("warnings", []).append(bounds["warn"])
        return out

    def _describe_columns(self, mask: pd.DataFrame, plan: dict) -> dict:
        cols = plan.get("select_columns") or ["JV_default_PCE", "JV_default_Voc", "Cell_architecture"]
        info = []
        for c in cols:
            rc = self.resolve_column(c) or c
            if rc not in mask.columns:
                info.append({"column": c, "missing": True})
                continue
            s = mask[rc]
            info.append({"column": rc, "n": int(s.notna().sum()),
                         "n_unique": int(s.nunique()), "dtype": str(s.dtype)})
        return {"result_type": "columns", "data": info}


# convenience accessor used by the ML predictor's feature pipeline
def load_pdb(csv_path=None) -> pd.DataFrame:
    """Load the enriched PDB table as a DataFrame via a throwaway tool."""
    return PDBTool(csv_path=csv_path).df