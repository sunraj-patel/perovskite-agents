"""Regenerate BENCHMARK_REPORT.md from the scored tables (numbers are never typed by hand).

    python -m benchmarks.agentic_paper.refresh && python -m benchmarks.agentic_paper.make_report

The prose that interprets results is generated from the intervals themselves
(it only says an effect is distinguishable from zero when the paired bootstrap
interval excludes zero) and the fixed sections (setup, limitations) are static.
"""
import csv
import json
from collections import Counter, defaultdict

from .paths import HERE, RUNS_DIR, SCORED_DIR, TABLES_DIR, setup_env

setup_env()
CATS = {"pdb": "Experimental database", "literature": "Literature retrieval", "ml": "ML prediction / explanation / what-if",
        "scaps": "SCAPS sensitivity (precomputed)", "multi_tool": "Multi-tool"}
COND = {"direct": "Direct tool use", "perovagents": "PerovAgents"}


def rd(name):
    p = TABLES_DIR / name
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def f3(x):
    return "n/a" if x in ("", None) else f"{float(x):.2f}"


def ci(r, m):
    return f"{f3(r[m])} [{f3(r[m + '_lo'])}, {f3(r[m + '_hi'])}]"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    from .providers import load_models
    models = load_models()
    freeze = json.load(open(HERE / "FREEZE.json"))
    overall, diffs, cats = rd("overall_by_model.csv"), rd("paired_differences.csv"), rd("by_category.csv")
    cov, eff = rd("coverage.csv"), rd("supplementary_efficiency_failures.csv")
    scored = [json.loads(l) for l in open(SCORED_DIR / "episodes_scored.jsonl")] \
        if (SCORED_DIR / "episodes_scored.jsonl").exists() else []
    manifest = [json.loads(l) for l in open(RUNS_DIR / "manifest.jsonl")] \
        if (RUNS_DIR / "manifest.jsonl").exists() else []
    L = []
    A = L.append
    A("# PerovAgents benchmark report\n")
    A("_Generated from `tables/` and `scored/` by `make_report.py`; do not edit numbers by hand. "
      "The manuscript was not touched._\n")
    done = sum(int(r["episodes_done"]) for r in cov)
    planned = 40 * len(models) * 2 * 3
    A("## 1. Status\n")
    A(f"* Frozen questions: **{len(freeze['main_set'])} main** (8 per category) + {len(freeze['case_study_set'])} "
      f"live-SCAPS case-study questions kept out of the comparison (freeze_id {freeze['freeze_id']}).")
    A(f"* Episodes scored: **{done} of {planned}** planned (40 questions x {len(models)} backbones x 2 conditions x 3 repetitions). "
      f"{'Complete.' if done >= planned else 'INCOMPLETE; see coverage below and the resume command.'}")
    A("* Backbones and routes (exact identifiers; no fallbacks):\n")
    A(md_table(["id", "model identifier", "route", "temperature"],
               [[m["id"], f"`{m['model']}`", m["route"], m["temperature"]] for m in models]))
    A("\nBudgets (both conditions, from the original harness): 12 tool calls, 12 model turns, 1200 s per episode; "
      "PerovAgents additionally allows 12 turns per sub-agent and 5 orchestrator rounds.\n")
    A("### Coverage (episodes scored per repetition)\n")
    A(md_table(["backbone", "condition", "questions", "rep 0", "rep 1", "rep 2", "total", "missing", "infra/harness rows"],
               [[r["backbone"], r["condition"], r["main_questions"], r["rep0_done"], r["rep1_done"], r["rep2_done"],
                 r["episodes_done"], r["episodes_missing_to_target_720_basis"], r["infra_or_harness_rows"]] for r in cov]))
    A("\nResume: `python -m benchmarks.agentic_paper.run --stage pass --reps 3` (completed cells are skipped; "
      "infrastructure failures are retried; start `./start_opencode_server.sh` first for the OpenCode backbones).\n")
    A("## 2. Overall results (question-level means, 95% paired question bootstrap)\n")
    rows = []
    for r in overall:
        rows.append([r["model_label"], COND[r["condition"]], r["n_questions_paired"], r["n_episodes"],
                     ci(r, "completed"), ci(r, "correct"), ci(r, "tool_f1"), f3(r["partial"])])
    A(md_table(["backbone", "condition", "paired questions", "episodes", "completion", "correctness", "tool F1", "partial credit"], rows))
    A("\nFigure 1: `figures/fig1_overall_comparison.pdf` (data: `figures/fig1_overall_data.csv`).\n")
    A("### Paired differences (PerovAgents minus direct)\n")
    A(md_table(["backbone", "metric", "questions", "difference", "95% CI", "PerovAgents better / tie / direct better", "CI excludes 0"],
               [[r["model_label"], r["metric"], r["n_questions"], f3(r["diff_perovagents_minus_direct"]),
                 f"[{f3(r['ci95_lo'])}, {f3(r['ci95_hi'])}]",
                 f"{r['questions_perovagents_better']} / {r['ties']} / {r['questions_direct_better']}", r["ci_excludes_zero"]]
                for r in diffs]))
    A("\n## 3. By task category\n")
    A("Each cell has at most 8 questions, so intervals are wide. Figure 2: `figures/fig2_category_performance.pdf` "
      "(data: `figures/fig2_category_data.csv`).\n")
    rows = []
    for r in cats:
        rows.append([r["model_label"], CATS[r["category"]], COND[r["condition"]], r["n_questions_paired"], r["n_episodes"],
                     ci(r, "completed"), ci(r, "correct"), f3(r["tool_f1"])])
    A(md_table(["backbone", "category", "condition", "questions", "episodes", "completion", "correctness", "tool F1"], rows))
    A("\n## 4. Efficiency and infrastructure (supplementary)\n")
    A("Full table: `tables/supplementary_efficiency_failures.csv`. Wall-clock includes provider latency and rate-limit "
      "waits and episodes ran concurrently on one machine; use call counts for comparisons.\n")
    A(md_table(["backbone", "condition", "episodes", "model calls", "tool calls", "median s", "retries", "timeouts",
                "rate-limit events", "provider errors"],
               [[r["backbone"], COND[r["condition"]], r["n_episodes"], r["mean_model_calls"], r["mean_tool_calls"],
                 r["median_seconds"], r["mean_retries"], r["timeouts"], r["rate_limit_events"], r["provider_error_events"]]
                for r in eff]))
    infra = [m for m in manifest if m["status"] != "completed"]
    cnt = Counter((m["backbone"], m["status"]) for m in infra)
    A("\nInfrastructure / harness rows (excluded from agent-failure denominators): "
      + (", ".join(f"{b}: {n} x {s}" for (b, s), n in sorted(cnt.items())) or "none") + ".\n")
    A("## 5. Failure analysis\n")
    lab = defaultdict(Counter)
    for s in scored:
        if s["set"] == "main" and not s["correct"]:
            lab[(s["backbone"], s["condition"])][s["primary_failure"]] += 1
    keys = ["budget_exhaustion", "tool_errors", "repeated_actions", "incorrect_tool_selection", "incomplete_answer",
            "incorrect_citation", "incorrect_value", "unsupported_claim", "infrastructure_failure"]
    A("Primary failure label of each *incorrect* episode (one per episode; multi-label counts in "
      "`tables/failure_labels.csv`):\n")
    A(md_table(["backbone", "condition", "incorrect episodes"] + keys,
               [[b, COND[c], sum(v.values())] + [v.get(k, 0) for k in keys] for (b, c), v in sorted(lab.items())]))
    A("\nTrace-substantiated examples (first episode per label, with the evidence recorded by the scorer):\n")
    seen = set()
    for s in sorted(scored, key=lambda x: x["episode_id"]):
        if s["set"] == "main" and not s["correct"] and s["primary_failure"] not in seen and s["failure_evidence"]:
            seen.add(s["primary_failure"])
            A(f"* `{s['primary_failure']}` - `{s['episode_id']}`: {'; '.join(s['failure_evidence'])[:300]}")
    A("")
    A("## 6. Statistical interpretation\n")
    sents = []
    for r in diffs:
        if r["metric"] in ("completed", "correct", "tool_f1"):
            d = float(r["diff_perovagents_minus_direct"])
            lo, hi = float(r["ci95_lo"]), float(r["ci95_hi"])
            nm = {"completed": "completion", "correct": "correctness", "tool_f1": "tool-selection F1"}[r["metric"]]
            if lo > 0:
                v = "higher than direct tool use, with an interval that excludes zero"
            elif hi < 0:
                v = "lower than direct tool use, with an interval that excludes zero"
            else:
                v = "not distinguishable from direct tool use (interval includes zero)"
            sents.append(f"* {r['model_label']}, {nm}: PerovAgents minus direct = {d:+.2f} [{lo:+.2f}, {hi:+.2f}] over "
                         f"{r['n_questions']} questions: {v}.")
    A("\n".join(sents) or "No paired data yet.")
    A("\nThese are descriptive statements about this question set; the question (not the repetition) is the unit, "
      "so the effective sample size per backbone is the number of paired questions. No superiority claim is made "
      "unless the interval excludes zero, and even then it applies to these tools, data, budgets and prompts.\n")
    A("## 7. Limitations\n")
    A("""* **Budgets.** Totals match (12 tool calls / 12 model turns / 1200 s), but PerovAgents additionally gets 12 turns per sub-agent and up to 5 orchestrator rounds; this structural asymmetry is part of the architecture and is not removed.
* **OpenCode routes are not bare completions.** DeepSeek and MiniMax are reached through `opencode serve`, which wraps the model in OpenCode's agent prompt and may take hidden internal steps with read-only file tools (recorded per call as `opencode_internal` events; reported tokens include that overhead). The server runs in an empty sandbox with deny-by-default tools, but model behaviour (for example prose instead of `ASK` lines) can differ from a bare API. Cross-backbone comparisons are therefore confounded by route; within-backbone paired comparisons are not.
* **Free-tier quotas.** Free quotas decide how many episodes could be run; rate-limited or unavailable calls are recorded as infrastructure failures, not agent failures, and cells are re-queued. See the coverage table for exact sample sizes.
* **References are built from the tools under test.** The benchmark measures faithful, complete use of tool outputs, not scientific validity of the predictions or simulations (for example Q04's maximum is a database entry with implausible Jsc; SCAPS values come from stored sweeps of a reference cell with preset transport parameters).
* **Scoring is rule-based.** Number matching is by value in the final answer (with a window rule for architecture-specific values in Q03), so a coincidentally matching number can pass; DOI checks credit one verified paper for Q05-Q08. Q17 and Q19 are open-ended, have structural checks only and need manual review. The `unsupported_claim` label and grounding fraction are heuristics.
* **Live SCAPS is not exercised.** Live solver questions (Q13-Q16, Q20) are preserved as case studies only; while the tool layer was first being built, the framework's own bounded availability probe (`SCAPSTool.live_status`, one real solver run each) executed about four times before live SCAPS was disabled; those runs are not part of any result and no stray solver process remained.
* **Concurrency.** Episodes run concurrently (4 workers per backbone) on one machine, so wall-clock is not an isolated latency measurement.
* **One model per route family and one stochastic run per cell** unless the coverage table shows three repetitions; run-to-run variability is visible in the per-repetition question-level files.
""")
    A("## 8. Suggested evidence-based wording for the manuscript (for you to integrate manually)\n")
    A("The report generator does not write manuscript text. Use the intervals in Section 2 and the per-category "
      "table in Section 3, keep the sample sizes beside every number, and describe the orchestration effect in "
      "terms of the failure modes in Section 5 (for example format-compliance give-ups of the direct baseline versus "
      "escalations of the orchestrated condition) rather than as a general ranking of systems.\n")
    A("## 9. Files\n")
    A("* Raw episodes + traces: `runs/episodes/<question>__<backbone>__<condition>__r<k>/`; run manifest: `runs/manifest.jsonl`.\n"
      "* Scored episodes: `scored/episodes_scored.jsonl`; tables: `tables/*.csv`; figures: `figures/`.\n"
      "* Frozen inputs and hashes: `questions/questions_v2.json`, `reference/`, `FREEZE.json`.\n"
      "* Case study: `case_studies/Q13_design_and_simulation/`.\n")
    (HERE / "BENCHMARK_REPORT.md").write_text("\n".join(L))
    print("wrote", HERE / "BENCHMARK_REPORT.md")


if __name__ == "__main__":
    main()
