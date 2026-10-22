"""Build the frozen question bank (questions/questions_v2.json).

* Q01-Q20 keep their IDs, text, categories and required outputs from the
  original 20-question benchmark (benchmarks/cross_llm_questions.py, v1).
  Their deterministic check specifications are (re)written here.
* Q21-Q45 are added so that every category has EIGHT questions in the main set.
* Questions that need the LIVE SCAPS solver (Q13, Q14, Q15, Q16, Q20) are kept
  and frozen but flagged ``set: "live_scaps_case_study"``: they are not part of
  the main comparison (no repeated live simulations; the solver environment is
  not reliably available). The 40-question main set is what the figures use.

Reference numbers are read from reference/reference_values.json (built from
the data by build_reference.py) and embedded in each check together with a
``ref`` pointer to their provenance.

    python -m benchmarks.agentic_paper.build_questions
"""
import json

from .paths import QUESTIONS_DIR, REFERENCE_DIR, setup_env

setup_env()
from benchmarks.cross_llm_questions import QUESTIONS as V1  # noqa: E402

REF = json.load(open(REFERENCE_DIR / "reference_values.json"))


def R(path):
    cur = REF
    for part in path.split("."):
        cur = cur[part]
    return cur


# ----------------------------------------------------------------- check helpers
def num(cid, ref, values, abs_tol=None, rel_tol=None, unit="", scale_alts=None, mandatory=True,
        near=None, note=""):
    """Number must appear in the final answer within tolerance of ANY of `values`."""
    vals = values if isinstance(values, list) else [values]
    c = {"id": cid, "type": "number", "ref": ref, "values": vals, "unit": unit, "mandatory": mandatory}
    if abs_tol is not None:
        c["abs_tol"] = abs_tol
    if rel_tol is not None:
        c["rel_tol"] = rel_tol
    if scale_alts:
        c["scale_alts"] = scale_alts
    if near:
        c["near"] = near
    if note:
        c["note"] = note
    return c


def terms(cid, any_of=None, all_of=None, mandatory=True):
    return {"id": cid, "type": "terms", "any_of": any_of or [], "all_of": all_of or [],
            "mandatory": mandatory}


def groups(cid, grps, mandatory=True):
    return {"id": cid, "type": "term_groups", "groups": grps, "mandatory": mandatory}


def recall(cid, items, min_fraction, mandatory=True, ref=""):
    return {"id": cid, "type": "set_recall", "items": items, "min_fraction": min_fraction,
            "ref": ref, "mandatory": mandatory}


def doi(cid, dois, mandatory=True, ref=""):
    return {"id": cid, "type": "doi_evidence", "dois": dois, "ref": ref, "mandatory": mandatory}


def known_doi(cid, min_count=1, mandatory=True):
    return {"id": cid, "type": "doi_known", "min_count": min_count, "mandatory": mandatory}


def rng(cid, lo, hi, mandatory=True):
    return {"id": cid, "type": "number_in_range", "lo": lo, "hi": hi, "mandatory": mandatory}


def decline(cid, phrases, forbidden=()):
    return {"id": cid, "type": "decline", "phrases": phrases, "forbidden_regex": list(forbidden),
            "mandatory": True}


def nref(names):
    """Feature-name matching items: underscores are matched as spaces."""
    return [[n] for n in names]


DECLINE_GENERIC = ["no record", "no matching", "no devices", "not found", "no data", "none ",
                   "zero", "0 devices", "no entries", "does not contain", "no such", "not contain",
                   "no passage", "no paper", "no evidence", "no relevant", "not covered",
                   "no mention", "could not find", "cannot", "can't", "unable", "not available",
                   "no precomputed", "does not include", "not part", "only predicts", "not predict",
                   "does not predict", "not supported", "outside", "no information", "not report",
                   "no sweep", "not established", "could not be", "not possible", "isn't", "is not"]

# canonical action vocabulary for tool-selection scoring (see scorer.py)
PLAN = lambda req, opt=(): {"required": list(req), "optional": list(opt)}  # noqa: E731

TOOL_ACTION = {
    "sql": "db.query", "find_columns": "db.query", "column_values": "db.query",
    "search": "lit.search", "read_paper": "lit.read",
    "predict": "ml.predict", "explain_global": "ml.explain_global",
    "explain_local": "ml.explain_local", "what_if": "ml.what_if", "optimize": "ml.optimize",
    "get_sweep": "scaps.sweep", "reference_cell": "scaps.describe",
    "list_parameters": "scaps.describe", "read_def": "scaps.describe",
    "run_point": "scaps.live", "sweep": "scaps.live",
    "design_stack": "design.layout", "write_execution_script": "design.layout",
    "simulate_stack": "design.simulate", "simulate_defect_profile": "design.simulate",
    "_calc": "calc",
}


def spec(qid, category, text, plans, checks, *, origin, qset="main", difficulty="normal",
         behaviour="answer", mode="deterministic", required_outputs=None, order=None,
         dependencies="", provenance=None, notes="", recovery=False, tolerance_note=""):
    return {"id": qid, "category": category, "question": text, "origin": origin, "set": qset,
            "difficulty": difficulty, "expected_behaviour": behaviour, "evaluation_mode": mode,
            "required_outputs": required_outputs or [], "reference_plans": plans,
            "order_constraints": order or [], "dependencies": dependencies,
            "checks": checks, "provenance": provenance or [], "recovery_required": recovery,
            "notes": notes, "tolerance_note": tolerance_note}


def build():
    v1 = {q["id"]: q for q in V1}
    out = []

    def orig(qid, plans, checks, **kw):
        q = v1[qid]
        out.append(spec(qid, q["category"], q["question"], plans, checks, origin="v1",
                        difficulty=q["difficulty"], required_outputs=q["required_outputs"],
                        recovery=q["recovery_required"], **kw))

    # ============================================================ PDB (original)
    g = R("pdb")
    orig("Q01", [PLAN(["db.query"])], [
        num("count", "pdb.Q01.count", g["Q01"]["count"], rel_tol=0.05, unit="devices"),
        num("mean_pce", "pdb.Q01.mean_pce", g["Q01"]["mean_pce"], rel_tol=0.05, unit="%")],
        provenance=[{"source": "sqlite", "sql": g["Q01"]["sql"]}],
        tolerance_note="count within 5% (NULL-PCE filter variants); mean within 5%")
    top = g["Q02"]["ranking"]
    orig("Q02", [PLAN(["db.query"])], [
        recall("top_htl", [["spiro"], ["ptaa"], ["peai"]], 1.0, ref="pdb.Q02.ranking[:3]"),
        num("top_count", "pdb.Q02.ranking[0].count", top[0]["count"], rel_tol=0.05, unit="devices")],
        provenance=[{"source": "sqlite", "sql": g["Q02"]["sql"]}])
    q3 = g["Q03"]
    orig("Q03", [PLAN(["db.query"])], [
        num("nip_mean", "pdb.Q03.nip.mean_pce", q3["nip"]["mean_pce"], rel_tol=0.03, unit="%",
            near={"term": "n-?i-?p", "after": 120, "before": 10}),
        num("pin_mean", "pdb.Q03.pin.mean_pce", q3["pin"]["mean_pce"], rel_tol=0.03, unit="%",
            near={"term": "p-?i-?n", "after": 120, "before": 10}),
        num("nip_median", "pdb.Q03.nip.median_pce", q3["nip"]["median_pce"], rel_tol=0.03, unit="%",
            near={"term": "n-?i-?p", "after": 120, "before": 10}),
        num("pin_median", "pdb.Q03.pin.median_pce", q3["pin"]["median_pce"], rel_tol=0.03, unit="%",
            near={"term": "p-?i-?n", "after": 120, "before": 10}),
        num("counts", "pdb.Q03.*.count", [q3["nip"]["count"], q3["pin"]["count"]], rel_tol=0.05,
            unit="devices")],
        provenance=[{"source": "sqlite", "sql": q3["sql"]}],
        notes="Values must be attached to the right architecture (window match).")
    td = g["Q04"]["top_devices"][0]
    orig("Q04", [PLAN(["db.query"])], [
        num("top_pce", "pdb.Q04.top_devices[0].pce", td["pce"], rel_tol=0.05, unit="%"),
        terms("fields", all_of=["voc", "jsc", "ff"]), terms("htl", any_of=["spiro"])],
        provenance=[{"source": "sqlite", "sql": g["Q04"]["sql"]}],
        notes="The maximum is a database entry with implausible Jsc (data-quality issue); "
              "the reference is what the database reports.")
    # ============================================================ Literature (original)
    ev = R("literature.Q05")
    dois5 = [ev["doi"]]
    orig("Q05", [PLAN(["lit.search"], ["lit.read"])], [
        doi("gold_doi", dois5, ref="literature.Q05.doi"),
        terms("strategy", any_of=["chloride", "facl", "adahcl", "passivat"]),
        terms("mechanism", any_of=["crystalli", "recombination", "morpholog", "passivat"])],
        provenance=[{"source": "corpus", "note": "gold evidence verified against passages (v1)"}],
        notes="Only one corpus paper was verified as gold; other SnO2 papers are not credited.")
    orig("Q06", [PLAN(["lit.search", "lit.read"])], [
        doi("gold_doi", dois5, ref="literature.Q06.doi"),
        terms("architecture", any_of=["n-i-p", "nip", "n–i–p"])],
        order=[["lit.search", "lit.read"]], dependencies="read the paper after finding it",
        provenance=[{"source": "corpus"}])
    orig("Q07", [PLAN(["lit.search"], ["lit.read"])], [
        doi("gold_doi", dois5, ref="literature.Q07.doi"),
        num("baseline", "literature.Q07.baseline_pce", ev["baseline_pce"], abs_tol=0.05, unit="%"),
        num("result", "literature.Q07.champion_pce", [ev["facl_pce"], ev["champion_pce"]],
            abs_tol=0.05, unit="%"),
        terms("intervention", any_of=["chloride", "facl", "adahcl"])],
        provenance=[{"source": "corpus"}])
    orig("Q08", [PLAN(["lit.search", "lit.read"])], [
        doi("gold_doi", dois5, ref="literature.Q08.doi"),
        num("numeric_evidence", "literature.Q08", [ev["baseline_pce"], ev["facl_pce"], ev["champion_pce"]],
            abs_tol=0.05, unit="%")],
        order=[["lit.search", "lit.read"]], provenance=[{"source": "corpus"}])
    # ============================================================ ML (original)
    m = R("ml")
    q9, q10, q11, q12 = m["Q09"], m["Q10"], m["Q11"], m["Q12"]
    orig("Q09", [PLAN(["ml.predict"])], [
        num("predicted_pce", "ml.Q09.predicted_pce", q9["predicted_pce"], rel_tol=0.10, unit="%"),
        num("interval_lo", "ml.Q09.interval[0]", q9["interval"][0], abs_tol=0.8, unit="%"),
        num("interval_hi", "ml.Q09.interval[1]", q9["interval"][1], abs_tol=0.8, unit="%")],
        provenance=[{"source": "ml", "tool": "predict(nip description)"}],
        tolerance_note="10% on the point estimate (descriptions parse slightly differently)")
    orig("Q10", [PLAN(["ml.explain_global"])], [
        recall("top_features", nref(q10["top_features"][:5]), 0.6, ref="ml.Q10.top_features[:5]")],
        provenance=[{"source": "ml", "tool": "explain_global(nip)"}])
    orig("Q11", [PLAN(["ml.predict", "ml.explain_local"])], [
        num("predicted_pce", "ml.Q11.predicted_pce", q11["predicted_pce"], rel_tol=0.10, unit="%"),
        recall("features", nref(q11["top_local_features"]), 0.66, ref="ml.Q11.top_local_features"),
        terms("direction", any_of=["positive", "negative", "increas", "decreas", "raise", "lower",
                                   "reduc", "push", "improv"])],
        order=[["ml.predict", "ml.explain_local"]],
        provenance=[{"source": "ml", "tool": "predict + explain_local(nip)"}])
    orig("Q12", [PLAN(["ml.optimize"], ["ml.predict", "ml.what_if"]),
                 PLAN(["ml.predict", "ml.what_if"], ["ml.optimize"])], [
        num("baseline", "ml.Q12.baseline_pce", q12["baseline_pce"], rel_tol=0.08, unit="%"),
        terms("candidate", any_of=[t.lower() for t in q12["candidate_terms"]]),
        num("predicted_pce", "ml.Q12.candidate_pces", q12["candidate_pces"], abs_tol=0.3, unit="%")],
        provenance=[{"source": "ml", "tool": "optimize(nip description)"}])
    # ============================================================ SCAPS original (live; case studies)
    for qid, plan in (("Q13", PLAN(["design.layout", "design.simulate"])),
                      ("Q14", PLAN(["scaps.live"])), ("Q15", PLAN(["design.simulate"])),
                      ("Q16", PLAN(["scaps.live"]))):
        orig(qid, [plan], [], qset="live_scaps_case_study", mode="not_scored_in_main",
             notes="Requires live SCAPS. Preserved and frozen; see case_studies/ for stored runs.")
    # ============================================================ multi-tool original
    orig("Q17", [PLAN(["ml.optimize", "lit.search"], ["ml.explain_global", "lit.read", "ml.predict", "db.query"])], [
        known_doi("doi_valid"),
        groups("provenance_split", [["literature", "paper", "study", "studies", "reported"],
                                    ["model", "ml ", "shap", "predicted", "optimi"]]),
        {"id": "numbers", "type": "min_numbers", "count": 2, "mandatory": True}],
        mode="structural", provenance=[{"source": "corpus+ml"}],
        notes="Open-ended; no numeric gold. Structural checks only -> flagged for manual review.")
    orig("Q18", [PLAN(["db.query", "ml.predict", "ml.explain_local"])], [
        rng("db_part_pce", 18, 40),
        num("predicted_pce", "ml.Q11.predicted_pce", q11["predicted_pce"], rel_tol=0.10, unit="%"),
        recall("key_params", nref(q11["top_local_features"]), 0.33, ref="ml.Q11.top_local_features")],
        order=[["db.query", "ml.predict"], ["ml.predict", "ml.explain_local"]],
        dependencies="question states the sequence: database -> prediction -> explanation",
        provenance=[{"source": "sqlite+ml"}])
    orig("Q19", [PLAN(["lit.search", "ml.predict"], ["ml.explain_local", "db.query", "lit.read"])], [
        known_doi("doi_valid"),
        rng("predicted_pce", 3, 25),
        terms("material_properties", any_of=["band gap", "bandgap", "ev", "mobility", "homo", "lumo",
                                             "electron affinity", "work function"]),
        terms("key_params", any_of=["feature", "parameter", "shap", "thickness", "deposition"])],
        mode="structural", notes="Underspecified query ('a candidate ETL and HTL'); structural checks "
                                 "only -> flagged for manual review.", provenance=[{"source": "corpus+ml"}])
    orig("Q20", [PLAN(["design.layout", "design.simulate", "ml.explain_global", "lit.search"])], [],
         qset="live_scaps_case_study", mode="not_scored_in_main",
         notes="Requires live SCAPS. Preserved and frozen.")

    # ============================================================ ADDED: PDB
    q21 = g["Q21"]
    add = lambda *a, **k: out.append(spec(*a, origin="v2", **k))  # noqa: E731
    add("Q21", "pdb", "How many n-i-p devices in the database use a compact + mesoporous TiO2 ETL "
        "(TiO2-c | TiO2-mp) together with Spiro-MeOTAD as the HTL, and what is their mean PCE?",
        [PLAN(["db.query"])], [
            num("count", "pdb.Q21.count_alts", q21["count_alts"], rel_tol=0.02, unit="devices"),
            num("mean_pce", "pdb.Q21.mean_pce", [q21["mean_pce"], q21["mean_pce_eq"]], rel_tol=0.03,
                unit="%")],
        required_outputs=["count", "mean_pce"],
        provenance=[{"source": "sqlite", "sql": q21["sql"], "variant": q21["sql_equal_variant"]}],
        tolerance_note="count accepts LIKE / equality / incl.-NULL-PCE variants (within 2%)")
    q22 = g["Q22"]
    add("Q22", "pdb", "Which three perovskite compositions (short-form label) are most frequently reported "
        "in n-i-p devices with PCE above 22%? Give the labels and their record counts.",
        [PLAN(["db.query"])], [
            recall("top3", [[c.lower()] for c in q22["top3"]], 1.0, ref="pdb.Q22.top3"),
            num("top_count", "pdb.Q22.ranking[0].count", q22["ranking"][0]["count"], abs_tol=0.5,
                unit="devices")],
        required_outputs=["ranked_labels", "counts"], difficulty="adversarial",
        provenance=[{"source": "sqlite", "sql": q22["sql"]}])
    q23 = g["Q23"]
    add("Q23", "pdb", "What is the highest reported PCE among p-i-n devices that use PTAA as the HTL, "
        "and which paper (DOI) reports it?",
        [PLAN(["db.query"])], [
            num("max_pce", "pdb.Q23.max_pce", q23["max_pce"], abs_tol=0.05, unit="%"),
            doi("doi", [q23["doi"]], ref="pdb.Q23.doi")],
        required_outputs=["max_pce", "doi"], provenance=[{"source": "sqlite", "sql": q23["sql"]}])
    add("Q24", "pdb", "What is the average reported PCE of n-i-p perovskite solar cells that use "
        "LaFeO3 as the electron-transport layer?",
        [PLAN([], ["db.query"])], [
            decline("states_no_data", DECLINE_GENERIC,
                    [r"(mean|average)\s+(reported\s+)?pce[^.\n]{0,40}\b\d{1,2}\.\d+\s*%"])],
        required_outputs=["explicit_statement_that_no_matching_records_exist"],
        behaviour="decline", difficulty="insufficient_evidence",
        provenance=[{"source": "sqlite", "sql": g["Q24"]["sql"], "result_count": g["Q24"]["count"]}],
        notes="Zero matching records: the correct response reports that and gives no mean.")
    # ============================================================ ADDED: Literature
    l25, l26, l27 = R("literature.Q25"), R("literature.Q26"), R("literature.Q27")
    add("Q25", "literature", "Which paper in the literature collection studies poly-TPD doped PTAA as the "
        "hole-transport layer? Report its DOI and the PCE of the control device and of the optimised "
        "device.", [PLAN(["lit.search"], ["lit.read"])], [
            doi("doi", [l25["doi"]], ref="literature.Q25.doi"),
            num("control", "literature.Q25.control_pce", l25["control_pce"], abs_tol=0.05, unit="%"),
            num("modified", "literature.Q25.modified_pce", l25["modified_pce"], abs_tol=0.05, unit="%")],
        required_outputs=["doi", "control_pce", "optimised_pce"],
        provenance=[{"source": "corpus", "paper": l25["doi"], "assertion": "numbers occur in paper text"}])
    add("Q26", "literature", "Find the paper that introduces an ultra-thin CdS electron-transport layer "
        "prepared from N-methyl thiourea for planar perovskite solar cells. Give its DOI and the PCE of the "
        "reference device with the thicker CdS layer (T-CdS).",
        [PLAN(["lit.search"], ["lit.read"])], [
            doi("doi", [l26["doi"]], ref="literature.Q26.doi"),
            num("t_cds_pce", "literature.Q26.t_cds_pce", l26["t_cds_pce"], abs_tol=0.05, unit="%")],
        required_outputs=["doi", "t_cds_pce"],
        provenance=[{"source": "corpus", "paper": l26["doi"]}])
    add("Q27", "literature", "Which paper in the collection reviews gas quenching for perovskite thin-film "
        "deposition? Give its DOI and describe how gas quenching works.",
        [PLAN(["lit.search"], ["lit.read"])], [
            doi("doi", [l27["doi"]], ref="literature.Q27.doi"),
            terms("mechanism", any_of=["nitrogen", "gas flow", "gas-assisted", "evaporation of the "
                                       "precursor solvent", "expedit", "solvent removal", "evaporat"])],
        required_outputs=["doi", "mechanism"], provenance=[{"source": "corpus", "paper": l27["doi"]}])
    add("Q28", "literature", "What do the papers in the literature collection report about the "
        "operational lifetime (T80, in hours) of NiOx hole-transport layers under ISOS stability protocols?",
        [PLAN([], ["lit.search", "lit.read"])], [
            decline("states_no_evidence", DECLINE_GENERIC,
                    [r"t80[^.\n]{0,80}\b\d{2,}\s*(h|hours)\b", r"niox[^.\n]{0,100}\b\d{3,}\s*(h|hours)\b"])],
        required_outputs=["explicit_statement_that_the_collection_has_no_such_evidence"],
        behaviour="decline", difficulty="insufficient_evidence",
        provenance=[{"source": "corpus", "absent_terms_checked": R("literature.Q28")["absent_terms_checked"]}],
        notes="The terms NiOx, T80, ISOS and inkjet occur in none of the 22 papers (asserted at build time).")
    # ============================================================ ADDED: ML
    a29, a30, a31 = m["Q29"], m["Q30"], m["Q31"]
    add("Q29", "ml", "What PCE does the ML model predict for a p-i-n cell with PTAA HTL, PCBM ETL and "
        "MAPbI3 perovskite of 450 nm thickness? Report the prediction with its 95% interval.",
        [PLAN(["ml.predict"])], [
            num("predicted_pce", "ml.Q29.predicted_pce", a29["predicted_pce"], rel_tol=0.10, unit="%"),
            num("interval_lo", "ml.Q29.interval[0]", a29["interval"][0], abs_tol=0.8, unit="%"),
            num("interval_hi", "ml.Q29.interval[1]", a29["interval"][1], abs_tol=0.8, unit="%")],
        required_outputs=["predicted_pce", "interval_95"], provenance=[{"source": "ml", "tool": "predict(pin)"}])
    add("Q30", "ml", "According to the ML model, which input features matter most for the PCE of p-i-n "
        "perovskite solar cells? List the top five.",
        [PLAN(["ml.explain_global"])], [
            recall("top5", nref(a30["top_features"]), 0.6, ref="ml.Q30.top_features")],
        required_outputs=["feature_names"], provenance=[{"source": "ml", "tool": "explain_global(pin)"}])
    add("Q31", "ml", "For an n-i-p cell with SnO2 ETL, Spiro-MeOTAD HTL and MAPbI3 perovskite, what does "
        "the ML model predict for the PCE when the perovskite thickness is 300 nm and when it is 500 nm?",
        [PLAN(["ml.what_if"], ["ml.predict"]), PLAN(["ml.predict"])], [
            num("pce_300", "ml.Q31.pce_300", a31["pce_300"], abs_tol=0.15, unit="%"),
            num("pce_500", "ml.Q31.pce_500", a31["pce_500"], abs_tol=0.15, unit="%")],
        required_outputs=["pce_300", "pce_500"], provenance=[{"source": "ml", "tool": "what_if(thickness)"}],
        notes="The two predictions differ by 0.05 points; a faithful answer reports a near-flat response.")
    add("Q32", "ml", "What operational lifetime (T80, in hours) does the ML model predict for a p-i-n cell "
        "with PTAA HTL, PCBM ETL and MAPbI3 perovskite?",
        [PLAN([], ["ml.predict"])], [
            decline("states_unsupported", DECLINE_GENERIC,
                    [r"t80[^.\n]{0,80}\b\d{2,}\s*(h|hours)\b", r"\b\d{2,}\s*(h|hours)\b[^.\n]{0,40}t80"])],
        required_outputs=["explicit_statement_that_the_model_predicts_PCE_only"],
        behaviour="decline", difficulty="insufficient_evidence",
        provenance=[{"source": "ml", "note": "predictor target is PCE only (manifest)"}])
    # ============================================================ ADDED: SCAPS (precomputed)
    s = R("scaps")
    p = "scaps"
    sweeps_note = "precomputed sweeps of the n-i-p reference cell (TiO2 / Spiro-MeOTAD)"
    add("Q33", "scaps", "In the precomputed SCAPS sweep of absorber thickness, what is the simulated PCE at "
        "200 nm and at 1000 nm, and how does PCE change with thickness?",
        [PLAN(["scaps.sweep"], ["scaps.describe"])], [
            num("pce_200", "scaps.Q33.pce_200nm", s["Q33"]["pce_200nm"], abs_tol=0.1, unit="%"),
            num("pce_1000", "scaps.Q33.pce_1000nm", s["Q33"]["pce_1000nm"], abs_tol=0.1, unit="%"),
            terms("trend", any_of=["increas", "rise", "rises", "higher", "grow", "improv"])],
        required_outputs=["pce_200", "pce_1000", "trend"],
        provenance=[{"source": "scaps_csv", "note": sweeps_note}])
    add("Q34", "scaps", "In the precomputed SCAPS band-gap sweep, what are the simulated PCE and Voc at "
        "1.35 eV and at 1.70 eV?",
        [PLAN(["scaps.sweep"], ["scaps.describe"])], [
            num("pce_135", "scaps.Q34.pce_1.35", s["Q34"]["pce_1.35"], abs_tol=0.1, unit="%"),
            num("pce_170", "scaps.Q34.pce_1.70", s["Q34"]["pce_1.70"], abs_tol=0.1, unit="%"),
            num("voc_135", "scaps.Q34.voc_1.35", s["Q34"]["voc_1.35"], abs_tol=0.02, unit="V"),
            num("voc_170", "scaps.Q34.voc_1.70", s["Q34"]["voc_1.70"], abs_tol=0.02, unit="V")],
        required_outputs=["pce_135", "pce_170", "voc_135", "voc_170"],
        provenance=[{"source": "scaps_csv", "note": sweeps_note}])
    add("Q35", "scaps", "Using the precomputed SCAPS sweeps, compare how strongly the perovskite electron "
        "mobility, the perovskite hole mobility and the perovskite band gap affect the PCE: report the PCE "
        "span (maximum minus minimum) of each sweep.",
        [PLAN(["scaps.sweep"], ["scaps.describe", "calc"])], [
            num("span_mue", "scaps.Q35.span_mue", s["Q35"]["span_mue"], abs_tol=0.03, unit="% points"),
            num("span_muh", "scaps.Q35.span_muh", s["Q35"]["span_muh"], abs_tol=0.03, unit="% points"),
            num("span_eg", "scaps.Q35.span_eg", s["Q35"]["span_eg"], abs_tol=0.05, unit="% points")],
        required_outputs=["span_mue", "span_muh", "span_eg"], dependencies="three separate sweeps are needed",
        provenance=[{"source": "scaps_csv", "note": sweeps_note}])
    add("Q36", "scaps", "In the precomputed SCAPS hole-mobility sweep, what happens to the fill factor and "
        "the PCE when the perovskite hole mobility is reduced from 1 to 0.01 cm2/Vs?",
        [PLAN(["scaps.sweep"], ["scaps.describe"])], [
            num("ff_1", "scaps.Q36.ff_1", s["Q36"]["ff_1"], abs_tol=0.1, unit="%"),
            num("ff_001", "scaps.Q36.ff_0.01", s["Q36"]["ff_0.01"], abs_tol=0.1, unit="%"),
            num("pce_1", "scaps.Q36.pce_1", s["Q36"]["pce_1"], abs_tol=0.1, unit="%"),
            num("pce_001", "scaps.Q36.pce_0.01", s["Q36"]["pce_0.01"], abs_tol=0.1, unit="%")],
        required_outputs=["ff_1", "ff_001", "pce_1", "pce_001"],
        provenance=[{"source": "scaps_csv", "note": sweeps_note}], tolerance_note="FF accepted as % only")
    add("Q37", "scaps", "In the precomputed SCAPS sweep of the HTL band gap, what are the simulated PCE and "
        "fill factor at an HTL band gap of 2.0 eV?",
        [PLAN(["scaps.sweep"], ["scaps.describe"])], [
            num("pce_20", "scaps.Q37.pce_2.0", s["Q37"]["pce_2.0"], abs_tol=0.1, unit="%"),
            num("ff_20", "scaps.Q37.ff_2.0", s["Q37"]["ff_2.0"], abs_tol=0.2, unit="%")],
        required_outputs=["pce", "ff"], provenance=[{"source": "scaps_csv", "note": sweeps_note}])
    add("Q38", "scaps", "In the precomputed SCAPS sweep of the HTL electron affinity, at which values does the "
        "simulation fail to give a usable device result (no result or a collapsed device)? Report the "
        "values and what the sweep returns there.",
        [PLAN(["scaps.sweep"], ["scaps.describe"])], [
            num("collapse_chi", "scaps.Q38.collapse_chi", s["Q38"]["collapse_chi"], abs_tol=0.01, unit="eV"),
            num("nan_1", "scaps.Q38.nan_chi[0]", s["Q38"]["nan_chi"][0], abs_tol=0.01, unit="eV"),
            num("nan_2", "scaps.Q38.nan_chi[1]", s["Q38"]["nan_chi"][1], abs_tol=0.01, unit="eV"),
            num("collapse_pce", "scaps.Q38.collapse_pce", s["Q38"]["collapse_pce"], abs_tol=0.003, unit="%")],
        required_outputs=["failing_values", "returned_values"], difficulty="adversarial",
        provenance=[{"source": "scaps_csv", "note": sweeps_note}],
        notes="3.0 eV collapses (PCE ~0.009 %); 3.4 and 3.8 eV have no result.")
    add("Q39", "scaps", "What are the perovskite thickness, perovskite band gap, ETL, HTL and architecture "
        "of the SCAPS reference device on which the precomputed sweeps are based?",
        [PLAN(["scaps.describe"], ["scaps.sweep"])], [
            num("thickness", "scaps.Q39.reference_cell_pvk_d_nm", 400.0, abs_tol=0.5, unit="nm"),
            num("band_gap", "scaps.Q39.pvk_Eg_eV", 1.55, abs_tol=0.005, unit="eV"),
            terms("htl_etl", all_of=["spiro", "tio2"]), terms("architecture", any_of=["n-i-p", "nip"])],
        required_outputs=["thickness", "band_gap", "htl", "etl", "architecture"],
        provenance=[{"source": "scaps_reference_cell()", "note": sweeps_note}])
    add("Q40", "scaps", "How does the simulated PCE depend on the perovskite bulk defect density according to "
        "the precomputed SCAPS sweeps?",
        [PLAN([], ["scaps.describe", "scaps.sweep"])], [
            decline("states_no_sweep", DECLINE_GENERIC,
                    [r"(defect|\bnt\b)[^.\n]{0,60}(increas|decreas|reduc)[^.\n]{0,80}pce[^.\n]{0,40}\d+\.\d+"])],
        required_outputs=["explicit_statement_that_no_defect_density_sweep_exists"],
        behaviour="decline", difficulty="insufficient_evidence",
        provenance=[{"source": "scaps_csv", "note": "six sweeps only: d, eg, mue, muh, htlchi, htlEg"}],
        notes="Live SCAPS is disabled, so the only correct response is to report that no such sweep exists.")
    # ============================================================ ADDED: multi-tool
    mt = R("multi_tool")
    add("Q41", "multi_tool", "Compare what the experimental database and the ML model say about n-i-p "
        "devices with a SnO2 ETL: report the mean PCE of such devices in the database, and the PCE the ML "
        "model predicts for an n-i-p SnO2 / Spiro-MeOTAD / MAPbI3 (500 nm) device.",
        [PLAN(["db.query", "ml.predict"])], [
            num("db_mean", "multi_tool.Q41.db_mean_pce", mt["Q41"]["db_mean_pce"], rel_tol=0.05, unit="%"),
            num("ml_pred", "multi_tool.Q41.ml_predicted_pce", mt["Q41"]["ml_predicted_pce"], rel_tol=0.10,
                unit="%")],
        required_outputs=["db_mean_pce", "predicted_pce"], provenance=[{"source": "sqlite+ml"}])
    add("Q42", "multi_tool", "Which paper in the literature collection introduces an ultra-thin CdS "
        "electron-transport layer, and how many devices in the experimental database contain CdS in their "
        "ETL stack?",
        [PLAN(["lit.search", "db.query"], ["lit.read"])], [
            doi("doi", [R("literature.Q26.doi")], ref="literature.Q26.doi"),
            num("count", "pdb.Q42", [R("pdb.Q42.cds_like"), R("pdb.Q42.cds_excluding_cdse")], rel_tol=0.02,
                unit="devices")],
        required_outputs=["doi", "device_count"],
        provenance=[{"source": "corpus+sqlite", "sql": R("pdb.Q42.sql")}],
        tolerance_note="count accepts LIKE '%CdS%' (incl. CdSe) and the CdSe-excluded variant")
    add("Q43", "multi_tool", "For the n-i-p perovskite device, compare how the precomputed SCAPS "
        "absorber-thickness sweep and the ML model (n-i-p SnO2 / Spiro-MeOTAD / MAPbI3) each predict the "
        "PCE at 300 nm and at 500 nm. Report all four values.",
        [PLAN(["scaps.sweep", "ml.what_if"], ["ml.predict", "scaps.describe"])], [
            num("scaps_300", "multi_tool.Q43.scaps_pce_300nm", mt["Q43"]["scaps_pce_300nm"], abs_tol=0.15, unit="%"),
            num("scaps_500", "multi_tool.Q43.scaps_pce_500nm", mt["Q43"]["scaps_pce_500nm"], abs_tol=0.15, unit="%"),
            num("ml_300", "multi_tool.Q43.ml_pce_300", mt["Q43"]["ml_pce_300"], abs_tol=0.15, unit="%"),
            num("ml_500", "multi_tool.Q43.ml_pce_500", mt["Q43"]["ml_pce_500"], abs_tol=0.15, unit="%")],
        required_outputs=["scaps_300", "scaps_500", "ml_300", "ml_500"],
        provenance=[{"source": "scaps_csv+ml"}],
        notes="The two sources differ by ~15 points in absolute PCE; the answer must keep them apart.")
    add("Q44", "multi_tool", "Find the paper on poly-TPD doped PTAA hole-transport layers in the literature "
        "collection and report its DOI and the control and optimised PCE. Then use the ML model to predict "
        "the PCE of a p-i-n cell with PTAA HTL, PCBM ETL and MAPbI3 (450 nm).",
        [PLAN(["lit.search", "ml.predict"], ["lit.read"])], [
            doi("doi", [mt["Q44"]["doi"]], ref="multi_tool.Q44.doi"),
            num("control", "multi_tool.Q44.control_pce", mt["Q44"]["control_pce"], abs_tol=0.05, unit="%"),
            num("modified", "multi_tool.Q44.modified_pce", mt["Q44"]["modified_pce"], abs_tol=0.05, unit="%"),
            num("ml_pred", "multi_tool.Q44.ml_predicted_pce", mt["Q44"]["ml_predicted_pce"], rel_tol=0.10, unit="%")],
        required_outputs=["doi", "control_pce", "optimised_pce", "predicted_pce"],
        provenance=[{"source": "corpus+ml"}])
    add("Q45", "multi_tool", "Report the SCAPS-simulated PCE of a p-i-n device with a 600 nm CsPbI3 absorber "
        "and a Cu2O hole-transport layer, and compare it with the mean PCE of p-i-n devices in the "
        "experimental database.",
        [PLAN(["db.query"], ["scaps.describe", "scaps.sweep", "design.layout"])], [
            num("db_pin_mean", "multi_tool.Q45.db_pin_mean_pce", mt["Q45"]["db_pin_mean_pce"], rel_tol=0.05, unit="%"),
            decline("no_simulation", ["cannot", "can't", "unable", "not available", "no precomputed",
                                      "not simulated", "no simulation", "could not", "not run", "not possible",
                                      "was not", "not be established", "disabled", "no sweep", "not able"],
                    [r"(scaps|simulat\w+)[^.\n]{0,60}(cspbi3|cu2o)[^.\n]{0,80}\bpce\b[^.\n]{0,25}\b[1-4]\d\.\d+\s*%"])],
        required_outputs=["db_mean_pce", "statement_that_the_simulation_cannot_be_provided"],
        behaviour="partial_decline", difficulty="insufficient_evidence",
        provenance=[{"source": "sqlite", "note": "no precomputed sweep covers this device; live SCAPS disabled"}],
        notes="Half answerable (database mean), half unsupported (no simulation available).")
    return out


def main():
    qs = build()
    for q in qs:                                    # sanity: ids unique, categories valid
        assert q["category"] in ("pdb", "literature", "ml", "scaps", "multi_tool")
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids))
    main_qs = [q for q in qs if q["set"] == "main"]
    cats = {}
    for q in main_qs:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    assert all(v == 8 for v in cats.values()) and len(main_qs) == 40, cats
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    doc = {"version": 2, "tool_action_taxonomy": TOOL_ACTION, "questions": qs}
    with open(QUESTIONS_DIR / "questions_v2.json", "w") as f:
        json.dump(doc, f, indent=1)
    print(f"{len(qs)} questions ({len(main_qs)} main): {cats}")
    print("case-study (live SCAPS):", [q["id"] for q in qs if q["set"] != "main"])


if __name__ == "__main__":
    main()
