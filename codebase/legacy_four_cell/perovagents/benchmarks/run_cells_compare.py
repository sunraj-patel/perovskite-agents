"""Cells-vs-legacy benchmark (cutover evidence, S9+).

Runs a fixed subset of benchmark questions through `answer_cells`
(pipeline="llm", Gemini key pool) and compares against the legacy
rules-pipeline pilot rows (`results/benchmarks/pilot_rules_g_all.json`) on
identical qids.

Subset: all 15 disambiguation questions (planner discrimination) + one
per major cell category (database_query, prediction, literature_retrieval,
sensitivity_analysis, comparison, multi_tool_investigation).

Usage:
    python -m perovagents.benchmarks.run_cells_compare [--limit N]
"""
import argparse
import json
import os
import sys
import time
import traceback

SUBSET_QIDS = [
    # disambiguation (all 15): filled at runtime from the question set
]
ONE_PER_CELL = ["database_query", "prediction", "literature_retrieval",
                "sensitivity_analysis", "comparison",
                "multi_tool_investigation"]


def pick_subset(all_qs, limit=None):
    dis = [q for q in all_qs if q["category"] == "disambiguation"]
    rest = []
    for cat in ONE_PER_CELL:
        hit = [q for q in all_qs if q["category"] == cat]
        if hit:
            rest.append(hit[0])
    out = dis + rest
    return out[:limit] if limit else out


def cell_of_tool(action: str) -> str:
    tool = action.split(".")[0]
    return {"pdb": "pdb_sql", "literature": "literature",
            "fulltext": "literature", "extraction": "literature",
            "citations": "literature", "doi": "literature",
            "material": "literature", "ml": "ml", "scaps": "scaps",
            "calculator": "utility", "python_exec": "utility",
            "symbolic": "utility"}.get(tool, "utility")


def expected_cells(q) -> set:
    cells = set()
    for seq in q.get("gt", []):
        for a in seq:
            cells.add(cell_of_tool(a))
    return cells - {"utility"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/benchmarks/cells_gemini.json")
    args = ap.parse_args(argv)

    from perovagents.benchmarks.questions import all_questions
    from perovagents.control.app import answer_cells
    from perovagents.llm import get_llm

    questions = pick_subset(all_questions(),
                            limit=args.limit or None)
    print(f"subset: {len(questions)} questions", flush=True)

    llm = get_llm()
    assert llm.available, "no Gemini keys configured; set GOOGLE_API_KEY"

    rows = []
    for i, q in enumerate(questions):
        t0 = time.time()
        try:
            res = answer_cells(q["question"], pipeline="llm", llm=llm)
            planned = {t["cell"] for t in res["plan"]["tasks"]}
            status = res["summary"]["status"]
            validated = {t for t, s in status.items() if s == "validated"}
            want = expected_cells(q)
            rows.append({
                "qid": q["qid"], "category": q["category"],
                "planned_cells": sorted(planned),
                "expected_cells": sorted(want),
                "cell_recall": round(len(planned & want) / max(len(want), 1), 2),
                "status": status,
                "task_success": round(len(validated) / max(len(status), 1), 2),
                "fully_validated": all(s == "validated"
                                       for s in status.values()),
                "cited": bool(res["sources"]),
                "answer_chars": len(res["answer"]),
                "trace": bool(res["trace"]),
                "elapsed_s": round(time.time() - t0, 1),
            })
        except Exception as exc:  # never abort the batch on one question
            rows.append({"qid": q["qid"], "category": q["category"],
                         "error": str(exc)[:200],
                         "traceback": traceback.format_exc()[-500:]})
        done = sum(1 for r in rows if "status" in r or "error" in r)
        print(f"[{done}/{len(questions)}] {q['qid']} "
              f"{rows[-1].get('status', 'ERROR')}", flush=True)

    # legacy comparison on identical qids
    legacy = {}
    try:
        pilot = json.load(open("results/benchmarks/pilot_rules_g_all.json"))
        legacy = {r["qid"]: r for r in pilot.get("rows", [])}
    except OSError:
        pass
    for r in rows:
        leg = legacy.get(r.get("qid"), {})
        r["legacy"] = {k: leg.get(k) for k in
                       ("classification_ok", "gt_hit", "contract_satisfied",
                        "final_verdict", "execution_ok")}

    def mean(key, pred=None):
        vals = [r[key] for r in rows
                if key in r and (pred is None or pred(r))]
        return round(sum(vals) / len(vals), 3) if vals else None

    report = {
        "label": "cells_gemini",
        "n": len(rows),
        "cell_recall_mean": mean("cell_recall"),
        "task_success_mean": mean("task_success"),
        "fully_validated_pct": round(100 * sum(
            1 for r in rows if r.get("fully_validated")) / max(len(rows), 1), 1),
        "cited_pct": round(100 * sum(
            1 for r in rows if r.get("cited")) / max(len(rows), 1), 1),
        "rows": rows,
    }
    leg_rows = [r["legacy"] for r in rows if r.get("legacy")]
    if leg_rows:
        for k in ("classification_ok", "gt_hit", "contract_satisfied",
                  "execution_ok"):
            vals = [x[k] for x in leg_rows if isinstance(x.get(k), bool)]
            report[f"legacy_{k}_pct"] = round(
                100 * sum(vals) / len(vals), 1) if vals else None
        report["legacy_sufficient_pct"] = round(100 * sum(
            1 for x in leg_rows if x.get("final_verdict") == "SUFFICIENT")
            / len(leg_rows), 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in report.items()
                      if k != "rows"}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
