"""Build questions_v4.json: the v3 bank unchanged + six NEW, previously unseen questions
(3 database, 3 literature) and their reference values. No LLM involved.

Database references are computed from the SQLite mirror here (SQL stored with the value);
literature references are asserted to occur in the paper text of the corpus. Authored from the
data BEFORE any model saw them; they are the confirmatory set for database/literature (v2/v3
exposed every older database/literature question during development).

    python -m benchmarks.agentic_paper.build_questions_v4
"""
import json
import sqlite3
import time

from .paths import QUESTIONS_DIR, REFERENCE_DIR, setup_env

setup_env()
import os  # noqa: E402

DB = os.environ["PEROVAGENTS_PDB_SQLITE"]
CORPUS = os.environ["PEROVAGENTS_CORPUS_DIR"]


def num(cid, ref, vals, unit, **tol):
    return {"id": cid, "type": "number", "ref": ref, "values": vals, "unit": unit, "mandatory": True, **tol}


def doi(ref, d):
    return {"id": "doi", "type": "doi_evidence", "dois": [d], "ref": ref, "mandatory": True}


def mk(qid, cat, question, outputs, plan, checks, prov, notes=""):
    return {"id": qid, "category": cat, "question": question, "origin": "v4", "set": "main",
            "difficulty": "normal", "expected_behaviour": "answer", "evaluation_mode": "deterministic",
            "required_outputs": outputs, "reference_plans": [plan], "order_constraints": [],
            "dependencies": "", "checks": checks, "provenance": prov, "recovery_required": False,
            "notes": notes, "tolerance_note": ""}


def paper_has(doi_id, *needles):
    text = open(f"{CORPUS}/{doi_id.replace('/', '_')}.md", encoding="utf-8").read()
    for n in needles:
        assert n in text, f"{doi_id}: {n!r} not in paper text"


def main():
    con = sqlite3.connect(DB)
    one = lambda s: con.execute(s).fetchall()          # noqa: E731
    add = {"pdb": {}, "literature": {}}
    base = "FROM devices WHERE Cell_architecture='pin' AND HTL_stack_sequence='NiO-c'"
    sql = f"SELECT COUNT(*), AVG(JV_default_PCE) {base} AND JV_default_PCE IS NOT NULL"
    n, m = one(sql)[0]
    n_all = one(f"SELECT COUNT(*) {base}")[0][0]
    add["pdb"]["Q46"] = {"count_alts": sorted({n, n_all}), "mean_pce": round(m, 3), "sql": sql}
    base = "FROM devices WHERE Cell_architecture='pin' AND HTL_stack_sequence='PTAA'"
    sql = f"SELECT COUNT(*), AVG(JV_default_Voc) {base} AND JV_default_Voc IS NOT NULL"
    n2, v = one(sql)[0]
    n2_all = one(f"SELECT COUNT(*) {base}")[0][0]
    add["pdb"]["Q47"] = {"count_alts": sorted({n2, n2_all}), "mean_voc": round(v, 3), "sql": sql}
    sql = ("SELECT Perovskite_composition_short_form, COUNT(*) c FROM devices WHERE "
           "Cell_architecture='pin' AND JV_default_PCE > 20 GROUP BY 1 ORDER BY 2 DESC LIMIT 5")
    rows = one(sql)
    assert rows[2][1] > rows[3][1], "third place must not tie"
    add["pdb"]["Q48"] = {"ranking": [{"composition": a, "count": b} for a, b in rows],
                         "top3": [a for a, _ in rows[:3]], "sql": sql}
    L = [("10.1002/adfm.201604733", ["1.46", "8.5%"]), ("10.1007/s10118-017-1891-z", ["12.68", "9.49"]),
         ("10.1021/jp5126624", ["Sn vacanc"])]
    for d, needles in L:
        paper_has(d, *needles)
    add["literature"] = {"Q49": {"doi": L[0][0], "voc_htm": 1.46}, "Q50": {"doi": L[1][0], "pce_new": 12.68,
                         "pce_conventional": 9.49}, "Q51": {"doi": L[2][0]}}
    json.dump({"version": 4, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **add},
              open(REFERENCE_DIR / "reference_values_v4_additions.json", "w"), indent=1)

    p = add["pdb"]
    dbq = {"required": ["db.query"], "optional": []}
    litq = {"required": ["lit.search"], "optional": ["lit.read"]}
    q46 = mk("Q46", "pdb", "How many p-i-n devices in the database use compact NiO (NiO-c) as the hole-transport "
             "layer, and what is their mean reported PCE?", ["count", "mean_pce"], dbq,
             [num("count", "pdb.Q46.count_alts", p["Q46"]["count_alts"], "devices", rel_tol=0.02),
              num("mean_pce", "pdb.Q46.mean_pce", [p["Q46"]["mean_pce"]], "%", rel_tol=0.03)],
             [{"source": "sqlite", "sql": p["Q46"]["sql"]}])
    q47 = mk("Q47", "pdb", "What is the mean reported open-circuit voltage (Voc, in V) of p-i-n devices that use "
             "PTAA as the hole-transport layer, and how many devices does it average over?",
             ["mean_voc", "count"], dbq,
             [num("mean_voc", "pdb.Q47.mean_voc", [p["Q47"]["mean_voc"]], "V", rel_tol=0.03),
              num("count", "pdb.Q47.count_alts", p["Q47"]["count_alts"], "devices", rel_tol=0.02)],
             [{"source": "sqlite", "sql": p["Q47"]["sql"]}])
    t3 = [[a.lower()] for a in p["Q48"]["top3"]]
    q48 = mk("Q48", "pdb", "Which three perovskite compositions (short-form label) are most frequently reported in "
             "p-i-n devices with PCE above 20%? Give the labels and their record counts.",
             ["ranked_labels", "counts"], dbq,
             [{"id": "top3", "type": "set_recall", "items": t3, "min_fraction": 1.0, "ref": "pdb.Q48.top3",
               "mandatory": True},
              num("top_count", "pdb.Q48.ranking[0].count", [p["Q48"]["ranking"][0]["count"]], "devices",
                  abs_tol=0.5)],
             [{"source": "sqlite", "sql": p["Q48"]["sql"]}])
    q49 = mk("Q49", "literature", "Find the paper in the collection on quasi-2D bromide perovskite solar cells of "
             "the type (PEA)2(MA)n-1PbnBr3n+1 with high open-circuit voltage. Give its DOI and the Voc "
             "demonstrated when a hole-transport material is used.", ["doi", "voc"], litq,
             [doi("literature.Q49.doi", L[0][0]), num("voc", "literature.Q49.voc_htm", [1.46], "V", abs_tol=0.005)],
             [{"source": "corpus", "paper": L[0][0]}])
    q50 = mk("Q50", "literature", "Find the paper that grows the perovskite film with the assistance of a PC71BM "
             "layer spin-coated on the precursor film. Give its DOI and the PCE of the devices made by this "
             "simplified method and of those made by the conventional one-step method.",
             ["doi", "pce_new", "pce_conventional"], litq,
             [doi("literature.Q50.doi", L[1][0]),
              num("pce_new", "literature.Q50.pce_new", [12.68], "%", abs_tol=0.05),
              num("pce_conventional", "literature.Q50.pce_conventional", [9.49], "%", abs_tol=0.05)],
             [{"source": "corpus", "paper": L[1][0]}])
    q51 = mk("Q51", "literature", "Find the paper on bromine substitution in lead-free CsSnI3 perovskite solar "
             "cells. Give its DOI and the reason the paper gives for the improved open-circuit voltage.",
             ["doi", "mechanism"], litq,
             [doi("literature.Q51.doi", L[2][0]),
              {"id": "mechanism", "type": "terms", "any_of": ["vacanc"], "all_of": [], "mandatory": True}],
             [{"source": "corpus", "paper": L[2][0]}])
    bank = json.load(open(QUESTIONS_DIR / "questions_v3.json"))
    ids = {q["id"] for q in bank["questions"]}
    assert not ids & {"Q46", "Q47", "Q48", "Q49", "Q50", "Q51"}
    bank["version"] = 4
    bank["questions"] += [q46, q47, q48, q49, q50, q51]
    json.dump(bank, open(QUESTIONS_DIR / "questions_v4.json", "w"), indent=1)
    print("wrote questions_v4.json:", len(bank["questions"]), "questions;", json.dumps(add)[:600])


if __name__ == "__main__":
    main()
