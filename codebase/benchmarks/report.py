"""Results aggregation: TABLES 1-4 (spec section 12) from saved run records.

    python -m benchmarks.report results/cross_llm/<stamp>

Reads Qxx.<system>.json records + cross_llm_gold.json, scores every run
with benchmarks.scoring, writes report.json + tables/*.csv, prints
markdown tables. Pure function of saved files: rerunnable any time.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.cross_llm_questions import QUESTIONS  # noqa: E402
from benchmarks.scoring import paired_stats, score_run  # noqa: E402

META = {q["id"]: q for q in QUESTIONS}


def load_records(rundir):
    recs = []
    for f in sorted(os.listdir(rundir)):
        if f.endswith(".json") and f not in ("report.json",):
            try:
                recs.append(json.load(open(os.path.join(rundir, f))))
            except Exception:
                pass
    return [r for r in recs if r.get("qid") in META and not r.get("dry_run")]


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(scores):
    by_sys = {}
    for s in scores:
        by_sys.setdefault(s["system"], []).append(s)
    rows = []
    for sys, ss in sorted(by_sys.items()):
        rec = [s for s in ss if s["qid"] in
               __import__("benchmarks.cross_llm_questions", fromlist=["RECOVERY_SENSITIVE"]).RECOVERY_SENSITIVE]
        rows.append({
            "system": sys, "n": len(ss),
            "completion": mean([x["A_completion"] for x in ss]),
            "contract": mean([x["B_contract"] for x in ss]),
            "tool_f1": mean([x["C_tool_f1"] for x in ss]),
            "scientific": mean([x["D_scientific"] for x in ss]),
            "recovery_rate": mean([1 if x["F_recovered"] else 0 for x in rec]) if rec else None,
            "repeat_rate": mean([x["G_repeat_rate"] for x in ss]),
            "mean_tool_calls": mean([x["H_tool_calls"] for x in ss]),
            "mean_llm_calls": mean([x["I_llm_calls"] for x in ss]),
            "mean_tokens": mean([(x["J_prompt_tokens"] or 0) + (x["K_completion_tokens"] or 0)
                                 for x in ss]),
            "mean_seconds": mean([x["seconds"] for x in ss]),
        })
    return rows


def by_category(scores):
    rows = []
    cats = sorted(set(META[s["qid"]]["category"] for s in scores))
    for cat in cats:
        for sys in sorted(set(s["system"] for s in scores)):
            ss = [s for s in scores if META[s["qid"]]["category"] == cat and s["system"] == sys]
            if ss:
                rows.append({"category": cat, "system": sys, "n": len(ss),
                             "completion": mean([x["A_completion"] for x in ss]),
                             "tool_f1": mean([x["C_tool_f1"] for x in ss]),
                             "scientific": mean([x["D_scientific"] for x in ss])})
    return rows


def recovery_table(scores):
    from benchmarks.cross_llm_questions import RECOVERY_SENSITIVE
    return [{"qid": s["qid"], "system": s["system"],
             "failure": s["failure"], "recovered": s["F_recovered"],
             "final_correct": s["F_detail"]["final_correct"],
             "extra_tool_calls": s["F_detail"]["extra_tool_calls"]}
            for s in scores if s["qid"] in RECOVERY_SENSITIVE]


def md_table(rows, cols):
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None):
    rundir = (argv or sys.argv[1:])[0]
    gold = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "cross_llm_gold.json")))
    recs = load_records(rundir)
    scores = [score_run(META[r["qid"]], gold, r) for r in recs]
    t1 = summarize(scores)
    t2 = by_category(scores)
    t3 = recovery_table(scores)
    systems = sorted(set(s["system"] for s in scores))
    pairs = {}
    if len(systems) >= 2:
        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                a, b = systems[i], systems[j]
                pairs[f"{a}_vs_{b}"] = {
                    "completion": paired_stats(
                        [s for s in scores if s["system"] == a],
                        [s for s in scores if s["system"] == b], "A_completion"),
                    "scientific": paired_stats(
                        [s for s in scores if s["system"] == a],
                        [s for s in scores if s["system"] == b], "D_scientific"),
                }
    out = {"n_runs": len(scores), "table1_overall": t1,
           "table2_category": t2, "table3_recovery": t3,
           "table4_efficiency": [
               {"system": r["system"], "mean_tool_calls": r["mean_tool_calls"],
                "mean_llm_calls": r["mean_llm_calls"],
                "mean_tokens": r["mean_tokens"],
                "mean_seconds": r["mean_seconds"]} for r in t1],
           "paired": pairs, "scores": scores}
    json.dump(out, open(os.path.join(rundir, "report.json"), "w"), indent=1, default=str)
    os.makedirs(os.path.join(rundir, "tables"), exist_ok=True)
    for name, rows in (("table1_overall", t1), ("table2_category", t2),
                       ("table3_recovery", t3)):
        if rows:
            with open(os.path.join(rundir, "tables", name + ".csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print("TABLE 1 overall\n" + md_table(t1, ["system", "n", "completion", "tool_f1",
                                             "scientific", "recovery_rate", "repeat_rate",
                                             "mean_tool_calls", "mean_tokens"]))
    print("\nTABLE 3 recovery\n" + md_table(t3, ["qid", "system", "failure", "recovered",
                                                "final_correct", "extra_tool_calls"]))
    print(f"\nwrote report.json + tables/ in {rundir}")


if __name__ == "__main__":
    main()
