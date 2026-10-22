"""Build the reference values for every frozen question. No LLM is involved.

Sources (all inspectable, all read-only):
  sqlite     the Perovskite Database mirror (SQL given per entry)
  ml         the frozen CatBoost artefacts under results/ml (same tool calls an
             agent can make; determinism is checked by calling twice)
  corpus     the 22-paper literature collection (numbers are asserted to occur
             in the paper text at build time)
  scaps_csv  the precomputed SCAPS sweeps (sweep_results.csv)
  gold_v1    benchmarks/cross_llm_gold.json (literature evidence for Q05-Q08,
             verified by hand against corpus passages when it was created)

Output: reference/reference_values.json (values + provenance + source hashes)
        reference/known_dois.json (DOIs of the corpus and of the database, used
        to detect fabricated citations)

    python -m benchmarks.agentic_paper.build_reference
"""
import hashlib
import json
import re
import sqlite3
import statistics
import time
from pathlib import Path

from .paths import REFERENCE_DIR, REPO, setup_env

setup_env()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(x, nd=4):
    return round(float(x), nd)


def pdb_refs(con):
    r = {}

    def one(sql):
        return con.execute(sql).fetchall()

    # ---- original Q01-Q04 (same SQL as benchmarks/build_gold.py) -------------
    sql = ("SELECT COUNT(*), AVG(JV_default_PCE) FROM devices WHERE Cell_architecture='nip' "
           "AND ETL_stack_sequence LIKE '%SnO2%' AND JV_default_PCE IS NOT NULL")
    n, m = one(sql)[0]
    r["Q01"] = {"count": n, "mean_pce": fnum(m, 3), "sql": sql}
    sql = ("SELECT HTL_stack_sequence, COUNT(*) FROM devices WHERE Cell_architecture='nip' "
           "AND JV_default_PCE > 20 GROUP BY 1 ORDER BY 2 DESC LIMIT 10")
    r["Q02"] = {"ranking": [{"htl": a, "count": b} for a, b in one(sql)], "sql": sql}
    sql = ("SELECT Cell_architecture, JV_default_PCE FROM devices WHERE Cell_architecture "
           "IN ('nip','pin') AND JV_default_PCE IS NOT NULL")
    groups = {}
    for a, p in one(sql):
        groups.setdefault(a, []).append(p)
    r["Q03"] = {a: {"count": len(v), "mean_pce": fnum(sum(v) / len(v), 3),
                    "median_pce": fnum(statistics.median(v), 3)} for a, v in groups.items()}
    r["Q03"]["sql"] = sql
    sql = ("SELECT Ref_DOI_number, HTL_stack_sequence, ETL_stack_sequence, "
           "Perovskite_composition_long_form, JV_default_PCE, JV_default_Voc, JV_default_Jsc, "
           "JV_default_FF FROM devices WHERE Cell_architecture='nip' AND "
           "ETL_stack_sequence LIKE '%SnO2%' AND JV_default_PCE IS NOT NULL "
           "ORDER BY JV_default_PCE DESC LIMIT 5")
    cols = ["doi", "htl", "etl", "absorber", "pce", "voc", "jsc", "ff"]
    r["Q04"] = {"top_devices": [dict(zip(cols, row)) for row in one(sql)], "sql": sql}

    # ---- added Q21-Q24 --------------------------------------------------------
    base = ("FROM devices WHERE Cell_architecture='nip' AND ETL_stack_sequence='TiO2-c | TiO2-mp' "
            "AND HTL_stack_sequence")
    sql = f"SELECT COUNT(*), AVG(JV_default_PCE) {base} LIKE '%Spiro-MeOTAD%' AND JV_default_PCE IS NOT NULL"
    n_like, mean_like = one(sql)[0]
    sql_eq = f"SELECT COUNT(*), AVG(JV_default_PCE) {base} = 'Spiro-MeOTAD' AND JV_default_PCE IS NOT NULL"
    n_eq, mean_eq = one(sql_eq)[0]
    sql_all = f"SELECT COUNT(*) {base} = 'Spiro-MeOTAD'"
    n_all = one(sql_all)[0][0]
    r["Q21"] = {"count": n_like, "count_alts": sorted({n_eq, n_all, n_like}),
                "mean_pce": fnum(mean_like, 3), "mean_pce_eq": fnum(mean_eq, 3),
                "sql": sql, "sql_equal_variant": sql_eq}
    sql = ("SELECT Perovskite_composition_short_form, COUNT(*) c FROM devices WHERE "
           "Cell_architecture='nip' AND JV_default_PCE > 22 GROUP BY 1 ORDER BY 2 DESC LIMIT 5")
    rows = one(sql)
    r["Q22"] = {"ranking": [{"composition": a, "count": b} for a, b in rows],
                "top3": [a for a, _ in rows[:3]], "sql": sql}
    sql = ("SELECT Ref_DOI_number, JV_default_PCE, ETL_stack_sequence FROM devices WHERE "
           "Cell_architecture='pin' AND HTL_stack_sequence='PTAA' AND JV_default_PCE IS NOT NULL "
           "ORDER BY JV_default_PCE DESC LIMIT 3")
    rows = one(sql)
    r["Q23"] = {"max_pce": rows[0][1], "doi": rows[0][0], "runner_up_pce": rows[1][1], "sql": sql}
    assert rows[0][1] > rows[1][1], "Q23 maximum must be unique"
    sql = ("SELECT COUNT(*) FROM devices WHERE Cell_architecture='nip' AND "
           "ETL_stack_sequence LIKE '%LaFeO3%'")
    r["Q24"] = {"count": one(sql)[0][0], "sql": sql}
    assert r["Q24"]["count"] == 0, "Q24 must have no matching records"

    # ---- added multi-tool parts -------------------------------------------------
    sql = ("SELECT COUNT(*) FROM devices WHERE ETL_stack_sequence LIKE '%CdS%'")
    n_cds = one(sql)[0][0]
    sql2 = ("SELECT COUNT(*) FROM devices WHERE ETL_stack_sequence LIKE '%CdS%' "
            "AND ETL_stack_sequence NOT LIKE '%CdSe%'")
    r["Q42"] = {"cds_like": n_cds, "cds_excluding_cdse": one(sql2)[0][0],
                "sql": sql, "sql_excluding_cdse": sql2}
    r["known_dois_db"] = sorted(x[0] for x in one(
        "SELECT DISTINCT Ref_DOI_number FROM devices WHERE Ref_DOI_number IS NOT NULL"))
    return r


def ml_refs():
    from benchmarks.tool_layer import collect_tools
    from .episode import disable_live_scaps
    disable_live_scaps()
    tools, _ = collect_tools()
    nip_desc = "n-i-p cell, SnO2 ETL, Spiro-MeOTAD HTL, MAPbI3 perovskite 500 nm"
    pin_desc = "p-i-n cell, PTAA HTL, PCBM ETL, MAPbI3 perovskite 450 nm"
    out = {"descriptions": {"nip": nip_desc, "pin": pin_desc}}

    def twice(fn, *a):
        a1, a2 = fn(*a), fn(*a)
        same = json.dumps(a1, default=str, sort_keys=True) == json.dumps(a2, default=str, sort_keys=True)
        return a1, same

    p, same = twice(tools["predict"], nip_desc, "nip")
    out["Q09"] = {"predicted_pce": p["predicted_pce"], "interval": p["pce_95_interval"], "deterministic": same}
    g, same = twice(tools["explain_global"], "nip")
    out["Q10"] = {"top_features": [f["name"] for f in g["features"][:8]], "deterministic": same}
    loc, same = twice(tools["explain_local"], nip_desc, "nip")
    feats = sorted(loc["features"], key=lambda f: -abs(f["importance"]))
    out["Q11"] = {"predicted_pce": p["predicted_pce"],
                  "top_local_features": [f["name"] for f in feats[:3]],
                  "local_directions": {f["name"]: f["direction"] for f in feats[:5]},
                  "deterministic": same}
    opt, same = twice(tools["optimize"], nip_desc, "nip")
    rec0 = opt["recommendations"][0]
    out["Q12"] = {"baseline_pce": rec0["base_predicted_pce"], "top_recommended_value": rec0["recommended_value"],
                  "top_recommended_parameter": rec0["parameter"], "top_best_pce": rec0["best_predicted_pce"],
                  "candidate_terms": sorted({str(r["recommended_value"]) for r in opt["recommendations"][:4]}
                                            | {str(v["material"]) for v in rec0.get("values", [])}
                                            | {str(r["parameter"]) for r in opt["recommendations"][:4]}),
                  "candidate_pces": sorted({round(float(r["best_predicted_pce"]), 2)
                                            for r in opt["recommendations"][:4]}),
                  "deterministic": same}
    p2, same = twice(tools["predict"], pin_desc, "pin")
    out["Q29"] = {"predicted_pce": p2["predicted_pce"], "interval": p2["pce_95_interval"], "deterministic": same}
    g2, same = twice(tools["explain_global"], "pin")
    out["Q30"] = {"top_features": [f["name"] for f in g2["features"][:5]], "deterministic": same}
    w, same = twice(tools["what_if"], nip_desc, "perovskite thickness", [300, 500], "nip")
    pts = {int(d["parameter_value"]): d["predicted_pce"] for d in w["data"]}
    out["Q31"] = {"pce_300": pts[300], "pce_500": pts[500], "deterministic": same}
    return out


def corpus_refs():
    idx = json.load(open(REPO / "results/literature/index.json"))
    texts = {}
    from perovagents import config
    for doi, d in idx["docs"].items():
        texts[doi] = Path(d["path"]).read_text(errors="ignore") if Path(d["path"]).exists() else ""
    gold = json.load(open(REPO / "benchmarks/cross_llm_gold.json"))["literature"]
    out = {"known_dois_corpus": sorted(idx["docs"]), "titles": {k: v["title"] for k, v in idx["docs"].items()},
           "Q05": gold["Q05"]["gold_evidence"], "Q06": gold["Q06"]["gold_evidence"],
           "Q07": gold["Q07"]["gold_evidence"], "Q08": gold["Q08"]["gold_evidence"]}
    # added questions: numbers must literally occur in the paper text
    def need(doi, patterns):
        t = texts[doi]
        for p in patterns:
            assert re.search(p, t), f"{doi}: pattern {p!r} not found in paper text"

    out["Q25"] = {"doi": "10.1117/12.2506443", "control_pce": 12.11, "modified_pce": 14.63}
    need("10.1117/12.2506443", [r"12\.11\s?%", r"14\.63\s?%", r"poly-TPD"])
    out["Q26"] = {"doi": "10.1246/cl.180616", "t_cds_pce": 15.0, "t_cds_jsc": 18.7}
    need("10.1246/cl.180616", [r"PCE of 15\.0%", r"N-methyl thiourea|\*N\*-methyl thiourea"])
    out["Q27"] = {"doi": "10.1016/j.joule.2018.06.009"}
    need("10.1016/j.joule.2018.06.009", [r"[Gg]as quenching"])
    absent = {}
    for term in ("NiOx", "T80", "ISOS", "inkjet"):
        absent[term] = [d for d, t in texts.items() if re.search(term, t)]
        assert not absent[term], f"Q28 premise broken: {term} occurs in {absent[term]}"
    out["Q28"] = {"absent_terms_checked": list(absent)}
    return out


def scaps_refs():
    import pandas as pd
    from perovagents import config
    df = pd.read_csv(config.SCAPS_EXISTING_RESULTS)
    out = {}
    s = {k: g.reset_index(drop=True) for k, g in df.groupby("sweep")}
    d = s["d"]
    out["Q33"] = {"pce_200nm": d.eta.iloc[0], "pce_1000nm": d.eta.iloc[-1],
                  "thickness_nm": [round(x * 1e9) for x in d.param]}
    e = s["eg"]
    out["Q34"] = {"pce_1.35": e.eta.iloc[0], "pce_1.70": e.eta.iloc[-1], "voc_1.35": e.voc.iloc[0],
                  "voc_1.70": e.voc.iloc[-1]}
    out["Q35"] = {"span_mue": s["mue"].eta.max() - s["mue"].eta.min(),
                  "span_muh": s["muh"].eta.max() - s["muh"].eta.min(),
                  "span_eg": e.eta.max() - e.eta.min()}
    m = s["muh"]
    out["Q36"] = {"ff_1": float(m.ff.iloc[2]), "ff_0.01": float(m.ff.iloc[0]),
                  "pce_1": float(m.eta.iloc[2]), "pce_0.01": float(m.eta.iloc[0])}
    h = s["htlEg"]
    out["Q37"] = {"pce_2.0": float(h.eta.iloc[0]), "ff_2.0": float(h.ff.iloc[0]),
                  "pce_3.0": float(h.eta.iloc[2])}
    c = s["htlchi"]
    out["Q38"] = {"collapse_chi": float(c.param.iloc[3]), "collapse_pce": float(c.eta.iloc[3]),
                  "nan_chi": [float(x) for x in c.param[c.eta.isna()]]}
    out["Q39"] = {"reference_cell_pvk_d_nm": 400.0, "pvk_Eg_eV": 1.55, "htl": "Spiro-MeOTAD",
                  "etl": "TiO2", "architecture": "n-i-p"}
    out["Q43"] = {"scaps_pce_300nm": float(d.eta.iloc[1]), "scaps_pce_500nm": float(d.eta.iloc[3])}
    out = json.loads(json.dumps(out, default=lambda x: float(x)))
    # round for readability, keep full precision in the CSV
    return {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
            for k, v in out.items()}


def main():
    from perovagents import config
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{config.PDB_SQLITE}?mode=ro", uri=True)
    pdb = pdb_refs(con)
    known_db = pdb.pop("known_dois_db")
    con.close()
    ml = ml_refs()
    lit = corpus_refs()
    scaps = scaps_refs()
    known = {"corpus": lit.pop("known_dois_corpus"), "database": known_db}
    titles = lit.pop("titles")
    # derived multi-tool references
    multi = {
        "Q41": {"db_mean_pce": pdb["Q01"]["mean_pce"], "db_count": pdb["Q01"]["count"],
                "ml_predicted_pce": ml["Q09"]["predicted_pce"]},
        "Q43": {**scaps["Q43"], "ml_pce_300": ml["Q31"]["pce_300"], "ml_pce_500": ml["Q31"]["pce_500"]},
        "Q44": {"doi": lit["Q25"]["doi"], "control_pce": lit["Q25"]["control_pce"],
                "modified_pce": lit["Q25"]["modified_pce"], "ml_predicted_pce": ml["Q29"]["predicted_pce"]},
        "Q45": {"db_pin_mean_pce": pdb["Q03"]["pin"]["mean_pce"]},
    }
    ref = {
        "version": 2,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": {
            "sqlite": {"path": "<PEROVAGENTS_PDB_SQLITE>", "sha256": sha256(config.PDB_SQLITE)},
            "scaps_sweep_csv": {"sha256": sha256(config.SCAPS_EXISTING_RESULTS)},
            "ml_manifest": {"sha256": sha256(Path(config.RESULTS_DIR) / "ml" / "manifest.json")},
            "ml_models": {p.name: sha256(p) for p in sorted((Path(config.RESULTS_DIR) / "ml").glob("*.cbm"))},
            "literature_index": {"sha256": sha256(REPO / "results/literature/index.json")},
            "gold_v1": {"sha256": sha256(REPO / "benchmarks/cross_llm_gold.json")},
            "corpus_files": {Path(d).name: sha256(Path(config.CORPUS_DIR) / Path(d).name)
                             for d in [f"{k.replace('/', '_')}.md" for k in known["corpus"]]
                             if (Path(config.CORPUS_DIR) / Path(d).name).exists()},
        },
        "pdb": pdb, "ml": ml, "literature": lit, "scaps": scaps, "multi_tool": multi,
        "corpus_titles": titles,
    }
    with open(REFERENCE_DIR / "reference_values.json", "w") as f:
        json.dump(ref, f, indent=1, default=str)
    with open(REFERENCE_DIR / "known_dois.json", "w") as f:
        json.dump(known, f)
    print("wrote", REFERENCE_DIR / "reference_values.json")
    print(json.dumps({k: ref[k] for k in ("ml",)}, default=str)[:1200])
    print("ML deterministic:", {k: v.get("deterministic") for k, v in ml.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
