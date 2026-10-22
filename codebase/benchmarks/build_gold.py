"""Build benchmark ground truth with TRUSTED TOOLS ONLY. No LLM in the loop.

    python -m benchmarks.build_gold            # PDB + ML + literature candidates (offline, fast)
    python -m benchmarks.build_gold --live     # + SCAPS live runs (needs wine, ~minutes)

Writes benchmarks/cross_llm_gold.json. Literature gold values must still be
manually verified against the corpus passages before the benchmark runs:
the script saves candidate evidence; a human marks it verified.

Fairness: gold comes from the exact tool backends the agents call, so MRS
and the direct-LLM baselines are scored against identical tool behavior.
"""
import json
import os
import sqlite3
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOLD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cross_llm_gold.json")


def pdb_connect():
    from perovagents import config
    return sqlite3.connect(f"file:{config.PDB_SQLITE}?mode=ro", uri=True)


def gold_pdb():
    """Q01-Q04 from the current Perovskite Database (read-only)."""
    con = pdb_connect()
    gold = {}
    try:
        q = ("SELECT COUNT(*), AVG(JV_default_PCE) FROM devices "
             "WHERE Cell_architecture = 'nip' AND ETL_stack_sequence LIKE '%SnO2%' "
             "AND JV_default_PCE IS NOT NULL")
        n, mean = con.execute(q).fetchone()
        gold["Q01"] = {"count": n, "mean_pce": round(mean, 3),
                       "sql": q}
        q = ("SELECT HTL_stack_sequence, COUNT(*) AS c FROM devices "
             "WHERE Cell_architecture = 'nip' AND JV_default_PCE > 20 "
             "GROUP BY 1 ORDER BY 2 DESC LIMIT 10")
        gold["Q02"] = {"ranking": [{"htl": r[0], "count": r[1]}
                                   for r in con.execute(q).fetchall()],
                       "sql": q}
        q = ("SELECT Cell_architecture, JV_default_PCE FROM devices "
             "WHERE Cell_architecture IN ('nip', 'pin') "
             "AND JV_default_PCE IS NOT NULL ORDER BY 1, 2")
        groups = {}
        for arch, pce in con.execute(q).fetchall():
            groups.setdefault(arch, []).append(pce)
        gold["Q03"] = {
            arch: {"count": len(v), "mean_pce": round(sum(v) / len(v), 3),
                   "median_pce": round(statistics.median(v), 3)}
            for arch, v in sorted(groups.items())}
        gold["Q03"]["sql"] = q
        q = ("SELECT Ref_DOI_number, HTL_stack_sequence, ETL_stack_sequence, "
             "Perovskite_composition_long_form, JV_default_PCE, JV_default_Voc, "
             "JV_default_Jsc, JV_default_FF FROM devices "
             "WHERE Cell_architecture = 'nip' AND ETL_stack_sequence LIKE '%SnO2%' "
             "AND JV_default_PCE IS NOT NULL "
             "ORDER BY JV_default_PCE DESC LIMIT 5")
        cols = ["doi", "htl", "etl", "absorber", "pce", "voc", "jsc", "ff"]
        gold["Q04"] = {"top_devices": [dict(zip(cols, r))
                                       for r in con.execute(q).fetchall()],
                       "sql": q}
    finally:
        con.close()
    return gold


def gold_ml():
    """Q09-Q12 from the frozen CatBoost artifacts (no retraining)."""
    from perovagents.ml.predictor import MLPredictorTool
    model = MLPredictorTool()
    desc = ("n-i-p cell, SnO2 ETL, Spiro-MeOTAD HTL, "
            "MAPbI3 perovskite 500 nm")
    gold = {}
    gold["Q09"] = model.call({"action": "predict", "architecture": "nip",
                              "description": desc})
    gold["Q10"] = model.call({"action": "explain_global", "architecture": "nip",
                              "top_k": 15})
    gold["Q11"] = {
        "predict": model.call({"action": "predict", "architecture": "nip",
                               "description": desc}),
        "explain_local": model.call({"action": "explain_local",
                                     "architecture": "nip",
                                     "description": desc, "top_k": 10}),
    }
    from perovagents.ml.controllability import DECLARED_RANGES
    param = ("perovskite thickness" if "perovskite thickness" in DECLARED_RANGES
             else sorted(DECLARED_RANGES)[0])
    lo, hi = DECLARED_RANGES[param]
    vals = [round(lo + (hi - lo) * f, 3) for f in (0.25, 0.5, 0.75)]
    gold["Q12"] = {
        "parameter": param,
        "declared_range": [lo, hi],
        "what_if": model.call({"action": "what_if", "architecture": "nip",
                               "description": desc, "parameter": param,
                               "values": vals}),
        "optimize": model.call({"action": "optimize", "architecture": "nip",
                                "description": desc, "top_k": 3}),
    }
    gold["_ml_meta"] = {"device_description": desc, "architecture": "nip",
                        "artifact_dir": model.dir}
    return gold


def gold_literature_candidates():
    """Q05-Q08: candidate evidence for HUMAN verification (not gold yet)."""
    from perovagents.literature.corpus import Corpus
    corpus = Corpus()
    queries = {
        "Q05": ["SnO2 n-i-p perovskite strategies improve performance",
                "SnO2 electron transport layer n-i-p mechanism"],
        "Q06": ["SnO2 n-i-p perovskite solar cell efficiency improvement method",
                "SnO2-based n-i-p device experimental"],
        "Q07": ["SnO2 n-i-p perovskite PCE improvement quantitative champion",
                "SnO2 n-i-p efficiency 21% 22% 23%"],
        "Q08": ["SnO2 n-i-p perovskite improvement numerical value stability",
                "SnO2 interface passivation n-i-p quantitative"],
    }
    out = {}
    for qid, qs in queries.items():
        out[qid] = {"verified": False,
                    "candidates": [corpus.search(q, k=6) for q in qs]}
    return out


def gold_scaps_live():
    """Q13-Q16: execute the frozen benchmark configuration (needs wine)."""
    from perovagents.physics.design import simulate_stack
    from perovagents.physics.scaps import SCAPSTool
    gold = {}
    run = simulate_stack(
        job="gold_q13_pin", architecture="p-i-n", absorber="FAPbI3",
        htl="PTAA", etl="SnO2", absorber_thickness_nm=600.0,
        htl_thickness_nm=200.0, etl_thickness_nm=50.0, sublayers=4)
    gold["Q13"] = run
    scaps = SCAPSTool()
    sweep = scaps.run_sweep_live("pvk_d", [300.0, 400.0, 500.0, 600.0, 700.0])
    gold["Q14"] = sweep
    gold["Q16"] = {"derived_from": "Q14",
                   "note": "sensitive parameters read off the Q14 sweep trend"}
    return gold


def main():
    live = "--live" in sys.argv
    # Preserve human-verified sections across rebuilds: regeneration must
    # never silently un-verify literature gold or drop a live SCAPS section.
    prev = {}
    if os.path.exists(GOLD_PATH):
        try:
            prev = json.load(open(GOLD_PATH))
        except Exception:
            prev = {}
    gold = {"version": 1, "pdb": gold_pdb(), "ml": gold_ml(),
            "literature": gold_literature_candidates()}
    for qid, entry in (prev.get("literature") or {}).items():
        if isinstance(entry, dict) and entry.get("verified"):
            gold["literature"][qid] = entry
    if live:
        gold["scaps"] = gold_scaps_live()
    else:
        gold["scaps"] = prev.get("scaps", {"pending": "run with --live (needs SCAPS/wine)"})
    with open(GOLD_PATH, "w") as f:
        json.dump(gold, f, indent=1, default=str)
    print(f"wrote {GOLD_PATH}")
    for qid in ("Q01", "Q02", "Q03"):
        print(qid, json.dumps(gold["pdb"][qid], default=str)[:220])
    print("Q09 predict:",
          json.dumps(gold["ml"]["Q09"], default=str)[:220])


if __name__ == "__main__":
    main()
