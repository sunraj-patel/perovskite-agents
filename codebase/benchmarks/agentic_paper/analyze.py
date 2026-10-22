"""Score every completed episode and aggregate (question = unit of analysis).

    python -m benchmarks.agentic_paper.analyze

Inputs : runs/manifest.jsonl, runs/episodes/*/episode.json, the frozen bank.
Outputs: scored/episodes_scored.jsonl (+ .csv), tables/*.csv, tables/summary.json

Statistics
  * Repetitions are execution variability, not new questions. Each (backbone,
    condition, question) cell is first averaged over its completed repetitions;
    the question-level means are then averaged with equal weight per question.
  * 95% intervals: paired percentile bootstrap that resamples QUESTION IDs
    (B=10000, seed fixed); the same resampled question sets are used for both
    conditions and for every metric of a given model, so the paired difference
    keeps its covariance. A question enters a paired comparison only when it has
    at least one completed, non-infrastructure episode in BOTH conditions.
  * Infrastructure failures (rate limit, provider outage) are excluded from the
    agent-performance denominators and reported separately.
"""
import csv
import json
from collections import defaultdict

import numpy as np

from .paths import RUNS_DIR, SCORED_DIR, TABLES_DIR, setup_env

setup_env()
B = 10000
SEED = 20261021
METRICS = ("completed", "correct", "tool_f1", "partial")
CATS = ("pdb", "literature", "ml", "scaps", "multi_tool")
CAT_LABEL = {"pdb": "Database", "literature": "Literature", "ml": "ML", "scaps": "SCAPS",
             "multi_tool": "Multi-tool"}
CONDS = ("direct", "perovagents")


def load_manifest():
    p = RUNS_DIR / "manifest.jsonl"
    rows = []
    if p.exists():
        for line in open(p):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect():
    """Latest manifest row per episode, scored. Returns (scored, infra_rows)."""
    from .scorer import score_episode
    latest = {}
    for r in load_manifest():
        latest[r["episode_id"]] = r                  # later rows override earlier attempts
    scored, infra = [], []
    for eid, r in latest.items():
        path = RUNS_DIR.parent / r["episode_path"]
        if r["status"] == "completed" and path.exists():
            ep = json.load(open(path))
            scored.append(score_episode(ep))
        elif r["status"] in ("infra_failed", "harness_error"):
            infra.append(r)
    return scored, infra


def write_scored(scored):
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCORED_DIR / "episodes_scored.jsonl", "w") as f:
        for s in sorted(scored, key=lambda x: x["episode_id"]):
            f.write(json.dumps(s, default=str) + "\n")
    flat = ["episode_id", "qid", "category", "set", "condition", "backbone", "model", "rep",
            "behaviour", "evaluation_mode", "termination", "completed", "correct", "partial",
            "tool_precision", "tool_recall", "tool_f1", "sequencing_satisfied",
            "dependencies_satisfied", "seconds", "n_model_calls", "n_tool_calls", "retries",
            "repeated_tool_calls", "tool_errors", "timeout", "rate_limit_events",
            "provider_error_events", "prompt_tokens", "completion_tokens", "grounded_frac",
            "primary_failure", "manual_review"]
    with open(SCORED_DIR / "episodes_scored.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat, extrasaction="ignore")
        w.writeheader()
        for s in sorted(scored, key=lambda x: x["episode_id"]):
            w.writerow(s)


def question_level(scored, only_main=True):
    """{(backbone, cond, qid): {metric: mean over reps, 'n_reps': k, 'category': c}}"""
    cells = defaultdict(list)
    for s in scored:
        if only_main and s["set"] != "main":
            continue
        cells[(s["backbone"], s["condition"], s["qid"])].append(s)
    out = {}
    for key, eps in cells.items():
        row = {m: float(np.mean([e[m] for e in eps if e.get(m) is not None]))
               for m in METRICS if any(e.get(m) is not None for e in eps)}
        row.update(n_reps=len(eps), category=eps[0]["category"])
        out[key] = row
    return out


def boot(qlevel, backbone, qids, metrics=METRICS, seed=SEED):
    """Paired bootstrap over question IDs for one backbone.

    Returns {metric: {cond: (mean, lo, hi), 'diff': (mean, lo, hi)}} on `qids`
    (questions present in both conditions)."""
    rng = np.random.default_rng(seed)
    qids = list(qids)
    n = len(qids)
    res = {}
    if n == 0:
        return res
    idx = rng.integers(0, n, size=(B, n))
    for m in metrics:
        vals = {}
        for c in CONDS:
            vals[c] = np.array([qlevel.get((backbone, c, q), {}).get(m, np.nan) for q in qids])
        ok = ~(np.isnan(vals["direct"]) | np.isnan(vals["perovagents"]))
        if not ok.any():
            continue
        v = {c: vals[c][ok] for c in CONDS}
        k = len(v["direct"])
        sub = idx if ok.all() else rng.integers(0, k, size=(B, k))   # same sample for both conditions
        entry = {}
        for c in CONDS:
            bs = v[c][sub].mean(axis=1)
            entry[c] = (float(v[c].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        d = (v["perovagents"] - v["direct"])
        bsd = d[sub].mean(axis=1)
        entry["diff"] = (float(d.mean()), float(np.percentile(bsd, 2.5)), float(np.percentile(bsd, 97.5)))
        entry["wins"] = int((d > 0).sum())
        entry["losses"] = int((d < 0).sum())
        entry["ties"] = int((d == 0).sum())
        entry["n_questions"] = int(k)
        res[m] = entry
    return res


def paired_qids(qlevel, backbone, cat=None):
    q_d = {k[2] for k in qlevel if k[0] == backbone and k[1] == "direct"}
    q_p = {k[2] for k in qlevel if k[0] == backbone and k[1] == "perovagents"}
    both = sorted(q_d & q_p)
    if cat:
        both = [q for q in both if qlevel[(backbone, "direct", q)]["category"] == cat]
    return both


def fmt(t, nd=3):
    return "" if t is None else round(t, nd)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def coverage(scored, infra, backbones, main_ids, target_reps=3):
    """Exact accounting of planned vs completed cells."""
    done = defaultdict(int)
    for s in scored:
        if s["set"] == "main":
            done[(s["backbone"], s["condition"], s["rep"])] += 1
    inf = defaultdict(int)
    for r in infra:
        inf[(r["backbone"], r["condition"])] += 1
    rows = []
    for bb in backbones:
        for c in CONDS:
            per_rep = [done[(bb, c, r)] for r in range(target_reps)]
            rows.append([bb, c, len(main_ids)] + per_rep + [sum(per_rep),
                         len(main_ids) * target_reps - sum(per_rep), inf[(bb, c)]])
    return rows


def efficiency_rows(scored, backbones):
    rows = []
    for bb in backbones:
        for c in CONDS:
            eps = [s for s in scored if s["backbone"] == bb and s["condition"] == c and s["set"] == "main"]
            if not eps:
                continue

            def mean(k):
                v = [e[k] for e in eps if e.get(k) is not None]
                return fmt(float(np.mean(v)), 2) if v else ""

            def med(k):
                v = [e[k] for e in eps if e.get(k) is not None]
                return fmt(float(np.median(v)), 1) if v else ""
            fails = defaultdict(int)
            for e in eps:
                if not e["correct"]:
                    fails[e["primary_failure"]] += 1
            rows.append([bb, c, len(eps), mean("n_model_calls"), mean("n_tool_calls"), med("seconds"),
                         mean("seconds"), mean("retries"), mean("repeated_tool_calls"),
                         sum(e["timeout"] for e in eps), sum(e["rate_limit_events"] or 0 for e in eps),
                         sum(e["provider_error_events"] or 0 for e in eps),
                         mean("prompt_tokens"), mean("completion_tokens")]
                        + [fails.get(k, 0) for k in FAIL_KEYS])
    return rows


FAIL_KEYS = ["incorrect_tool_selection", "repeated_actions", "tool_errors", "incomplete_answer",
             "unsupported_claim", "incorrect_value", "incorrect_citation", "budget_exhaustion"]


def main():
    from .providers import load_models
    from .scorer import load_bank
    bbs = [b["id"] for b in load_models()]
    names = {b["id"]: b["label"] for b in load_models()}
    scored, infra = collect()
    write_scored(scored)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    main_ids = [q for q, v in load_bank()[1].items() if v["set"] == "main"]
    ql = question_level(scored)
    summary = {"n_episodes_scored": len(scored), "n_infra_or_harness_rows": len(infra),
               "overall": {}, "category": {}, "model_names": names}
    ov_rows, diff_rows, cat_rows = [], [], []
    for bb in bbs:
        qids = paired_qids(ql, bb)
        res = boot(ql, bb, qids)
        summary["overall"][bb] = res
        n_ep = {c: sum(1 for s in scored if s["backbone"] == bb and s["condition"] == c and s["set"] == "main")
                for c in CONDS}
        for c in CONDS:
            row = [bb, names[bb], c, len(qids), n_ep[c]]
            for m in METRICS:
                t = res.get(m, {}).get(c)
                row += [fmt(t[0]), fmt(t[1]), fmt(t[2])] if t else ["", "", ""]
            ov_rows.append(row)
        for m in METRICS:
            e = res.get(m)
            if e:
                diff_rows.append([bb, names[bb], m, e["n_questions"], fmt(e["diff"][0]), fmt(e["diff"][1]),
                                  fmt(e["diff"][2]), e["wins"], e["ties"], e["losses"],
                                  "yes" if (e["diff"][1] > 0 or e["diff"][2] < 0) else "no"])
        for cat in CATS:
            cq = paired_qids(ql, bb, cat)
            cres = boot(ql, bb, cq)
            summary["category"].setdefault(bb, {})[cat] = cres
            for c in CONDS:
                n_ep_c = sum(1 for s in scored if s["backbone"] == bb and s["condition"] == c
                             and s["category"] == cat and s["set"] == "main")
                row = [bb, names[bb], cat, c, len(cq), n_ep_c]
                for m in METRICS:
                    t = cres.get(m, {}).get(c)
                    row += [fmt(t[0]), fmt(t[1]), fmt(t[2])] if t else ["", "", ""]
                d = cres.get("correct", {}).get("diff")
                row += [fmt(d[0]), fmt(d[1]), fmt(d[2])] if d else ["", "", ""]
                cat_rows.append(row)
    mh = []
    for m in METRICS:
        mh += [m, m + "_lo", m + "_hi"]
    write_csv(TABLES_DIR / "overall_by_model.csv",
              ["backbone", "model_label", "condition", "n_questions_paired", "n_episodes"] + mh, ov_rows)
    write_csv(TABLES_DIR / "paired_differences.csv",
              ["backbone", "model_label", "metric", "n_questions", "diff_perovagents_minus_direct",
               "ci95_lo", "ci95_hi", "questions_perovagents_better", "ties", "questions_direct_better",
               "ci_excludes_zero"], diff_rows)
    write_csv(TABLES_DIR / "by_category.csv",
              ["backbone", "model_label", "category", "condition", "n_questions_paired", "n_episodes"] + mh
              + ["correct_diff", "correct_diff_lo", "correct_diff_hi"], cat_rows)
    write_csv(TABLES_DIR / "coverage.csv",
              ["backbone", "condition", "main_questions", "rep0_done", "rep1_done", "rep2_done",
               "episodes_done", "episodes_missing_to_target_720_basis", "infra_or_harness_rows"],
              coverage(scored, infra, bbs, main_ids))
    write_csv(TABLES_DIR / "supplementary_efficiency_failures.csv",
              ["backbone", "condition", "n_episodes", "mean_model_calls", "mean_tool_calls",
               "median_seconds", "mean_seconds", "mean_retries", "mean_repeated_tool_calls", "timeouts",
               "rate_limit_events", "provider_error_events", "mean_prompt_tokens(real)",
               "mean_completion_tokens(real)"] + [f"n_primary_failure:{k}" for k in FAIL_KEYS],
              efficiency_rows(scored, bbs))
    # per-question table (what the figures can be audited against)
    qrows = []
    for (bb, c, q), v in sorted(ql.items()):
        qrows.append([bb, c, q, v["category"], v["n_reps"]] + [fmt(v.get(m)) for m in METRICS])
    write_csv(TABLES_DIR / "question_level.csv",
              ["backbone", "condition", "qid", "category", "n_reps"] + list(METRICS), qrows)
    # failure-mode table by label (multi-label counts)
    lab = defaultdict(int)
    for s in scored:
        if s["set"] != "main":
            continue
        for l in s["failure_labels"]:
            lab[(s["backbone"], s["condition"], l)] += 1
    write_csv(TABLES_DIR / "failure_labels.csv", ["backbone", "condition", "label", "n_episodes"],
              [[*k, v] for k, v in sorted(lab.items())])
    json.dump(summary, open(TABLES_DIR / "summary.json", "w"), indent=1, default=str)
    print(f"scored {len(scored)} episodes; infra/harness rows: {len(infra)}")
    return summary


if __name__ == "__main__":
    main()
