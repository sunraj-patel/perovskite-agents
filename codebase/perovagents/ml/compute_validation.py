"""Group-aware (DOI) validation of the PCE recipe (reviewer-facing metrics).

Rebuilds the exact inference feature matrix with `prepare_features`,
groups rows by `Ref_DOI_number` (the same DOI grouping behind the
leakage fix: PCE-leakage columns and DOI-LOO aggregates are excluded),
and scores the recipe with GroupKFold (default 5 folds, no shuffling —
group order from the CSV is deterministic).

IMPORTANT (read before citing): these numbers validate the modeling
*recipe + data* under a group-aware split. The deployed artifacts
(`model_{arch}.cbm`, trained on all data with the same recipe) share the
recipe; their in-sample fit is necessarily more optimistic. Both facts
are recorded in the output file.

Usage:
    python -m perovagents.ml.compute_validation --arch Overall
    python -m perovagents.ml.compute_validation --arch all --out results/ml/validation.json
"""
import argparse
import json
import os
import time

RECIPE = {"iterations": 800, "depth": 6, "learning_rate": 0.03,
          "l2_leaf_reg": 3.0, "random_seed": 42, "verbose": False,
          "allow_writing_files": False}
N_SPLITS = 5
TARGET = "final_pce"
GROUP_COL = "Ref_DOI_number"


def arch_mask(df, arch: str):
    """Row filter matching the deployed per-arch models (ARCH_ALIAS)."""
    import pandas as pd  # noqa: F401 (ensures pandas present)
    a = df["Cell_architecture"].astype(str).str.lower()
    if arch == "NIP":
        return a == "nip"
    if arch == "PIN":
        return a == "pin"
    return pd.Series(True, index=df.index)


def evaluate_arch(arch: str, n_splits: int = N_SPLITS,
                  recipe: dict = None) -> dict:
    """GroupKFold-by-DOI evaluation for one architecture. Returns the
    metrics record (also usable by tests)."""
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRegressor, Pool
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupKFold

    from .. import config
    from ..ml.features import prepare_features
    from ..ml.predictor import MLPredictorTool

    recipe = recipe or dict(RECIPE)
    df = pd.read_csv(config.ENRICHED_CSV, low_memory=False)
    df = df[arch_mask(df, arch)].reset_index(drop=True)
    meta = MLPredictorTool()._meta(arch)
    names = list(meta["feature_names"])
    cats = [c for c in meta.get("cat_features", []) if c in names]
    X, _ = prepare_features(df)
    X = X[names]
    y = pd.to_numeric(df[TARGET], errors="coerce")
    groups = df[GROUP_COL].astype(str)
    keep = y.notna() & groups.notna() & (groups != "") & (groups != "nan")
    X, y, groups = (X.loc[keep].reset_index(drop=True),
                    y.loc[keep].reset_index(drop=True),
                    groups.loc[keep].reset_index(drop=True))
    cat_idx = [names.index(c) for c in cats]

    oof = np.full(len(X), np.nan)
    fold_r2 = []
    t0 = time.time()
    for fold, (tr, te) in enumerate(
            GroupKFold(n_splits=n_splits).split(X, y, groups)):
        m = CatBoostRegressor(**recipe)
        m.fit(Pool(X.iloc[tr], y.iloc[tr], cat_features=cat_idx))
        pred = m.predict(Pool(X.iloc[te], cat_features=cat_idx))
        oof[te] = pred
        fold_r2.append(float(r2_score(y.iloc[te], pred)))
    rmse = float(mean_squared_error(y, oof) ** 0.5)
    return {
        "arch": arch,
        "metric": "rmse",
        "score": round(rmse, 4),
        "mae": round(float(mean_absolute_error(y, oof)), 4),
        "r2": round(float(r2_score(y, oof)), 4),
        "r2_std": round(float(__import__("numpy").std(fold_r2)), 4),
        "folds_r2": [round(v, 4) for v in fold_r2],
        "split": f"group-doi-{n_splits}fold",
        "group_col": GROUP_COL,
        "n": int(len(X)),
        "n_groups": int(groups.nunique()),
        "n_features": len(names),
        "model": f"model_{arch}.cbm",
        "recipe": {k: recipe[k] for k in
                   ("iterations", "depth", "learning_rate", "random_seed")},
        "elapsed_s": round(time.time() - t0, 1),
        "note": ("recipe+data validation under a group-aware split; "
                 "deployed artifact shares the recipe (see module docstring)"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all",
                    help="Overall | NIP | PIN | all")
    ap.add_argument("--folds", type=int, default=N_SPLITS)
    ap.add_argument("--out", default=None,
                    help="write validation.json here (default: results/ml/)")
    args = ap.parse_args(argv)
    from .. import config
    archs = ["Overall", "NIP", "PIN"] if args.arch == "all" else [args.arch]
    out = {}
    for arch in archs:
        rec = evaluate_arch(arch, n_splits=args.folds)
        out[arch] = rec
        print(f"{arch}: rmse={rec['score']} mae={rec['mae']} r2={rec['r2']} "
              f"n={rec['n']} groups={rec['n_groups']} "
              f"({rec['elapsed_s']}s)")
    path = args.out or os.path.join(config.ensure_results("ml"),
                                    "validation.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)
    return out


if __name__ == "__main__":
    main()
