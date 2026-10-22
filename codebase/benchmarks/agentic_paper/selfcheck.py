"""Static / unit checks of the benchmark artefacts (no LLM, no network).

    python -m benchmarks.agentic_paper.selfcheck

1. Bank integrity: unique IDs, 8 main questions per category, original Q01-Q20
   text identical to the v1 bank, case-study set = live-SCAPS questions.
2. Scorer controls, per main question:
     oracle answer (built from the check specs)  -> must be complete AND correct
     empty answer                                -> must be incomplete, incorrect
     numerically wrong answer                    -> must be incorrect
3. Tool-F1 sanity: reference plan -> F1 1.0; no tools on a tool question -> 0.
4. Freeze / reference hashes match the files on disk.
"""
import copy
import json
import re
import sys

from .paths import QUESTIONS_DIR, REFERENCE_DIR, setup_env

setup_env()
from benchmarks.cross_llm_questions import QUESTIONS as V1  # noqa: E402

from .scorer import load_bank, load_known_dois, score_episode  # noqa: E402

TOOL_FOR = {"db.query": "sql", "lit.search": "search", "lit.read": "read_paper", "ml.predict": "predict",
            "ml.explain_global": "explain_global", "ml.explain_local": "explain_local",
            "ml.what_if": "what_if", "ml.optimize": "optimize", "scaps.sweep": "get_sweep",
            "scaps.describe": "reference_cell", "design.layout": "design_stack"}


def oracle_answer(q, shift=0.0):
    parts = []
    for c in q["checks"]:
        t = c["type"]
        if t == "number":
            v = c["values"][0] * (1.6 if shift else 1.0) + shift
            lab = f"{c['near']['term'].replace('?', '').replace('-', '-')[:3]}: " if "near" in c else ""
            if "near" in c:
                lab = "nip " if "n-?i" in c["near"]["term"] else "pin "
            parts.append(f"{lab}{c['id']} = {v}")
        elif t == "terms":
            parts.append(" ".join(c["all_of"] + c["any_of"][:1]))
        elif t == "term_groups":
            parts.append(" ".join(g[0] for g in c["groups"]))
        elif t == "set_recall":
            parts.append(", ".join(a[0] for a in c["items"]))
        elif t == "doi_evidence":
            parts.append(f"Supported by the paper with DOI {c['dois'][0]}.")
        elif t == "doi_known":
            parts.append(f"See DOI {sorted(load_known_dois()[0])[0]}.")
        elif t == "number_in_range":
            parts.append(f"value {(c['lo'] + c['hi']) / 2}")
        elif t == "min_numbers":
            parts.append("values 1.23 and 4.56")
        elif t == "decline":
            parts.append(c["phrases"][0] + " is available for this request.")
    return "\n".join(parts)


def episode(q, answer, tools=None, term="answer"):
    plan = q["reference_plans"][0]
    acts = tools if tools is not None else plan["required"]
    calls = [{"agent": "x", "tool": TOOL_FOR.get(a, a), "args": {}, "result": "", "error": False}
             for a in acts]
    return {"episode_id": f"selfcheck_{q['id']}", "qid": q["id"], "condition": "direct", "backbone": "x",
            "model": "oracle", "rep": 0, "final_answer": answer, "termination": term, "tool_calls": calls,
            "n_model_calls": 1, "seconds": 0, "provider": {}, "repeated_tool_calls": 0, "tool_errors": 0}


def main():
    taxonomy, bank = load_bank()
    problems, notes = [], []
    qs = list(bank.values())
    ids = [q["id"] for q in qs]
    if len(ids) != len(set(ids)):
        problems.append("duplicate IDs")
    main_qs = [q for q in qs if q["set"] == "main"]
    cats = {}
    for q in main_qs:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    from .paths import VERSION
    if VERSION == "v2":
        if not (len(main_qs) == 40 and all(v == 8 for v in cats.values()) and len(cats) == 5):
            problems.append(f"category balance wrong: {cats}")
    elif any(q["category"] == "scaps" or any(a.startswith("scaps") for p in q["reference_plans"]
                                             for a in p["required"] + p.get("optional", []))
             for q in main_qs):
        problems.append("v3 active set still contains a SCAPS-dependent question")
    v1 = {q["id"]: q for q in V1}
    for qid, q in v1.items():
        if bank[qid]["question"] != q["question"] or bank[qid]["category"] != q["category"]:
            problems.append(f"{qid}: original question text/category changed")
    live = sorted(q["id"] for q in qs if q["set"] == "live_scaps_case_study")
    if live != ["Q13", "Q14", "Q15", "Q16", "Q20"]:
        problems.append(f"case-study set unexpected: {live}")
    n_decline = sum(1 for q in main_qs if q["expected_behaviour"] != "answer")
    notes.append(f"{len(qs)} questions, {len(main_qs)} main {cats}, {n_decline} insufficient-evidence (decline/partial) in main")
    notes.append(f"deterministic main questions: {sum(1 for q in main_qs if q['evaluation_mode'] == 'deterministic')}, "
                 f"structural (manual review): {[q['id'] for q in main_qs if q['evaluation_mode'] != 'deterministic']}")
    for q in main_qs:
        s_or = score_episode(episode(q, oracle_answer(q)))
        if not (s_or["completed"] and s_or["correct"]):
            problems.append(f"{q['id']}: ORACLE answer not accepted: "
                            f"{[(c['id'], c['present'], c['pass']) for c in s_or['checks'] if not c['pass']]}")
        if abs(s_or["tool_f1"] - 1.0) > 1e-9:
            problems.append(f"{q['id']}: reference plan F1 = {s_or['tool_f1']}")
        s_em = score_episode(episode(q, "", tools=[]))
        if s_em["completed"] or s_em["correct"]:
            problems.append(f"{q['id']}: empty answer accepted")
        if q["expected_behaviour"] == "answer" and any(c["type"] == "number" for c in q["checks"]):
            wrong = oracle_answer(q, shift=7.7)
            s_w = score_episode(episode(q, wrong))
            if s_w["correct"]:
                problems.append(f"{q['id']}: numerically wrong answer accepted")
        if q["reference_plans"][0]["required"]:
            s_nt = score_episode(episode(q, oracle_answer(q), tools=[]))
            if s_nt["tool_f1"] > 0:
                problems.append(f"{q['id']}: no tools but F1>0")
    # freeze / hashes
    from . import freeze
    ok, msg = freeze.verify()
    notes.append(f"freeze: {msg}")
    ref = json.load(open(REFERENCE_DIR / "reference_values.json"))
    notes.append(f"reference sources hashed: {sorted(ref['sources'])}")
    print("\n".join("NOTE  " + n for n in notes))
    if problems:
        print("\n".join("FAIL  " + p for p in problems))
        sys.exit(1)
    print(f"OK   all static checks passed for {len(main_qs)} main questions")


if __name__ == "__main__":
    main()
