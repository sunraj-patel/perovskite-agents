"""Evidence audit of the collected benchmark episodes (analysis layer only).

    python -m benchmarks.agentic_paper.audit

Does NOT run any model, and does not touch the frozen scorer, runner, questions or
raw data. It re-reads runs/manifest.jsonl + episode.json files, re-scores every
completed episode with the frozen scorer and writes (all new paths):

    audit/episode_validity.csv     one row per episode, validity class + evidence
    audit/coverage_matrix.csv      planned vs observed counts, backbone x condition x category
    audit/metrics_v2.csv           primary (valid episodes) + sensitivity bounds, paired diff CIs
    audit/hypotheses.csv           trace-derived mechanism counts
    figures_audit/fig1_overall_v2.{pdf,png}, fig2_category_v2.{pdf,png}, *_data.csv

Validity classes (assigned from provider evidence, never from answer quality):
    valid                the model produced evaluable behaviour; scored as an agent outcome
    no_model_output      every model reply was empty (avg < 5 chars/call) and no tool ran:
                         the episode carries no information about reasoning
    provider_limited     hit the wall clock while the provider reported call timeouts
    infra_failed         runner recorded an unrecoverable provider failure (re-queued)
Primary results use `valid` only; sensitivity B counts every non-valid attempted
episode as a failure (0), sensitivity A is the legacy rule (only infra_failed removed).
"""
import csv
import json
from collections import defaultdict

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .paths import AUDIT_DIR, FIGAUDIT_DIR, HERE, RUNS_DIR, VERSION, setup_env  # noqa: E402

setup_env()
from .analyze import load_manifest  # noqa: E402
from .scorer import load_bank, score_episode  # noqa: E402

AUD = AUDIT_DIR
FIGA = FIGAUDIT_DIR
SEED, B, TARGET_REPS = 20261021, 10000, 3
MET = ("completed", "correct", "tool_f1")
CATS = [("pdb", "Database"), ("literature", "Literature"), ("ml", "ML"), ("scaps", "SCAPS"),
        ("multi_tool", "Multi-tool")]
COLOR = {"direct": "#4A78A8", "perovagents": "#E4AA28"}
LABEL = {"direct": "Direct tool use", "perovagents": "PerovAgents"}


QUALITY_OK = ("valid", "valid_empty_completion", "unclassified_empty")   # in quality denominators


def classify(ep, manifest_status):
    """v3 episodes carry an evidence-based `validity` (validity.py); v2 episodes are
    classified from the aggregate provider counters (heuristic, as in the first audit)."""
    if (ep.get("provider") or {}).get("call_log") is not None:        # v3: re-derive from raw evidence
        from .validity import classify_episode
        val, evd = classify_episode(ep["provider"], ep.get("termination"), len(ep.get("tool_calls") or []),
                                    ep.get("final_answer"))
        return {"infra_invalid": "infra_failed"}.get(val, val), evd.get("reason", "")
    pv = ep.get("provider") or {}
    calls = max(pv.get("model_calls") or 0, 1)
    evs = pv.get("provider_events") or []
    n_to = sum(1 for e in evs if e["type"] == "timeout")
    if manifest_status in ("infra_failed", "harness_error") or ep.get("infra_failure"):
        return "infra_failed", f"provider: {(pv.get('infra_reason') or '')[:80]}"
    if (pv.get("completion_chars") or 0) / calls < 5 and not ep.get("tool_calls") \
            and not (ep.get("final_answer") or "").strip():
        return "no_model_output", (f"{pv.get('model_calls')} calls, {pv.get('completion_chars')} reply chars, "
                                   f"{round(ep['seconds'])}s")
    if ep.get("termination") == "timeout" and n_to:
        return "provider_limited", f"wall-clock with {n_to} provider call timeouts"
    return "valid", ""


def load_episodes():
    latest = {}
    for r in load_manifest():
        latest[r["episode_id"]] = r
    rows = []
    for eid, r in latest.items():
        ep = json.load(open(RUNS_DIR.parent / r["episode_path"]))
        v, why = classify(ep, r["status"])
        s = score_episode(ep) if r["status"] == "completed" else None
        extra = ep.get("extra") or {}
        rows.append({"eid": eid, "qid": r["qid"], "category": r["category"], "backbone": r["backbone"],
                     "condition": r["condition"], "rep": r["rep"], "status": r["status"],
                     "termination": ep.get("termination"), "validity": v, "evidence": why,
                     "seconds": round(ep.get("seconds") or 0, 1), "model_calls": ep["provider"].get("model_calls"),
                     "tool_calls": len(ep.get("tool_calls") or []),
                     "n_asks": extra.get("n_asks"), "escalations": extra.get("escalations"),
                     "score": s})
    return rows


def first_attempt_rows():
    """End-to-end operational view: the FIRST attempt of every cell, before any replacement."""
    first = {}
    for r in load_manifest():
        if r["episode_id"] not in first or (r.get("attempt") or 1) < (first[r["episode_id"]].get("attempt") or 1):
            first[r["episode_id"]] = r
    rows = []
    for eid, r in first.items():
        ep = json.load(open(RUNS_DIR.parent / (r.get("attempt_path") or r["episode_path"])))
        v, why = classify(ep, r["status"])
        rows.append({"eid": eid, "qid": r["qid"], "category": r["category"], "backbone": r["backbone"],
                     "condition": r["condition"], "validity": v,
                     "score": score_episode(ep) if r["status"] == "completed" else None})
    return rows


def attempts_table():
    """Every attempt of every cell with its validity and the attempt it replaces."""
    out = []
    for r in load_manifest():
        ep = json.load(open(RUNS_DIR.parent / (r.get("attempt_path") or r["episode_path"])))
        v, why = classify(ep, r["status"])
        out.append([r["episode_id"], r.get("attempt") or 1, r["status"], v, r.get("replaces") or "",
                    r.get("attempt_path") or r["episode_path"], why[:120]])
    return sorted(out)


def qlevel(rows, keep):
    cells = defaultdict(list)
    for r in rows:
        if r["score"] and r["score"]["set"] == "main" and keep(r):
            cells[(r["backbone"], r["condition"], r["qid"])].append(r["score"])
    return {k: {m: float(np.mean([e[m] for e in v])) for m in MET} | {"n_reps": len(v)}
            for k, v in cells.items()}


def worst_case(rows):
    """Sensitivity B: every attempted main episode counts; non-valid ones score 0."""
    cells = defaultdict(list)
    for r in rows:
        if r["category"] is None or r["qid"] not in MAIN:
            continue
        ok = r["validity"] in QUALITY_OK and r["score"]
        cells[(r["backbone"], r["condition"], r["qid"])].append(
            {m: (r["score"][m] if ok else 0.0) for m in MET})
    return {k: {m: float(np.mean([e[m] for e in v])) for m in MET} | {"n_reps": len(v)}
            for k, v in cells.items()}


def ci(vals, rng):
    vals = np.asarray(vals, float)
    if len(vals) < 5:
        return None, None            # too few questions for a meaningful interval
    idx = rng.integers(0, len(vals), (B, len(vals)))
    m = vals[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_stats(ql, bb, cat=None):
    qs = sorted({k[2] for k in ql if k[0] == bb and k[1] == "direct"} &
                {k[2] for k in ql if k[0] == bb and k[1] == "perovagents"})
    if cat:
        qs = [q for q in qs if MAIN[q] == cat]
    out = {"qids": qs}
    rng = np.random.default_rng(SEED)
    for m in MET:
        d = np.array([ql[(bb, "direct", q)][m] for q in qs])
        p = np.array([ql[(bb, "perovagents", q)][m] for q in qs])
        # paired bootstrap resamples question indices jointly for both conditions
        if len(qs) >= 5:
            idx = rng.integers(0, len(qs), (B, len(qs)))
            dd = (p[idx] - d[idx]).mean(1)
            lo, hi = np.percentile(dd, [2.5, 97.5])
            dlo, dhi = float(np.percentile(d[idx].mean(1), 2.5)), float(np.percentile(d[idx].mean(1), 97.5))
            plo, phi = float(np.percentile(p[idx].mean(1), 2.5)), float(np.percentile(p[idx].mean(1), 97.5))
        else:
            lo = hi = dlo = dhi = plo = phi = None
        out[m] = {"direct": float(d.mean()) if len(qs) else None, "direct_ci": (dlo, dhi),
                  "perovagents": float(p.mean()) if len(qs) else None, "perov_ci": (plo, phi),
                  "diff": float((p - d).mean()) if len(qs) else None,
                  "diff_ci": (None if lo is None else float(lo), None if hi is None else float(hi))}
    return out


def fnum(x, nd=3):
    return "" if x is None else round(x, nd)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def coverage(rows, bbs):
    planned_q = defaultdict(int)
    for q, c in MAIN.items():
        planned_q[c] += 1
    out = []
    for b in bbs:
        for cond in ("direct", "perovagents"):
            for cat, _ in CATS + [("ALL", "")]:
                sel = [r for r in rows if r["backbone"] == b["id"] and r["condition"] == cond
                       and r["qid"] in MAIN and (cat == "ALL" or r["category"] == cat)]
                nq = len(MAIN) if cat == "ALL" else planned_q[cat]
                cnt = defaultdict(int)
                for r in sel:
                    cnt[r["validity"]] += 1
                valid = [r for r in sel if r["validity"] in QUALITY_OK]
                agent_fail = sum(1 for r in valid if r["score"] and not r["score"]["correct"])
                out.append([b["id"], cond, cat, nq, nq * TARGET_REPS, len(sel), len({r["qid"] for r in sel}),
                            len(valid), agent_fail, cnt["no_model_output"], cnt["provider_limited"],
                            cnt["infra_failed"], cnt["unclassified_empty"], cnt["valid_empty_completion"],
                            nq * TARGET_REPS - len(sel)])
    return out


def hypotheses(rows):
    out = []
    by = defaultdict(dict)
    for r in rows:
        by[(r["backbone"], r["qid"], r["rep"])][r["condition"]] = r
    for b in sorted({r["backbone"] for r in rows}):
        pe = [r for r in rows if r["backbone"] == b and r["condition"] == "perovagents" and r["validity"] in QUALITY_OK]
        de = [r for r in rows if r["backbone"] == b and r["condition"] == "direct" and r["validity"] in QUALITY_OK]
        pair = [(v["direct"], v["perovagents"]) for k, v in by.items()
                if k[0] == b and len(v) == 2 and all(x["validity"] in QUALITY_OK for x in v.values())]
        mc = lambda xs: float(np.mean([x["model_calls"] for x in xs])) if xs else None
        out.append([b, "valid_pairs", len(pair), "", ""])
        out.append([b, "mean_model_calls_perovagents_valid", fnum(mc(pe), 1), "n", len(pe)])
        out.append([b, "mean_model_calls_direct_valid", fnum(mc(de), 1), "n", len(de)])
        esc = [r for r in pe if (r["escalations"] or 0) > 0]
        out.append([b, "perov_episodes_with_escalation", len(esc), "of", len(pe)])
        out.append([b, "escalated_and_correct", sum(1 for r in esc if r["score"]["correct"]), "of", len(esc)])
        nonesc = [r for r in pe if not (r["escalations"] or 0)]
        out.append([b, "non_escalated_and_correct", sum(1 for r in nonesc if r["score"]["correct"]), "of", len(nonesc)])
        noask = [r for r in pe if (r["n_asks"] or 0) == 0]
        out.append([b, "perov_zero_ask_episodes(orchestrator answered without dispatch)", len(noask), "of", len(pe)])
    return out


# ------------------------------------------------------------------ figures
mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "axes.linewidth": 0.7, "pdf.fonttype": 42, "savefig.dpi": 300})


def despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)


def bars(ax, x, st, wb, m, w=0.36):
    """Primary bars with paired-bootstrap CI (only when n>=5) and worst-case marker."""
    rec = []
    for i, cond in enumerate(("direct", "perovagents")):
        v = st[m][cond]
        if v is None:
            continue
        xx = x + (i - 0.5) * w
        ax.bar(xx, v, w * 0.94, color=COLOR[cond], linewidth=0)
        lo, hi = st[m]["direct_ci" if cond == "direct" else "perov_ci"]
        if lo is not None:
            ax.errorbar(xx, v, yerr=[[v - lo], [hi - v]], color="#333333", lw=0.7, capsize=1.6)
        else:                       # no interval (n<5): print the value so a true 0 is not read as missing
            ax.text(xx, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=5.2, color="#333333")
        wv = wb.get(cond)
        if wv is not None:
            ax.plot(xx, wv, marker="_", ms=7, mew=1.3, color="#222222", ls="none", zorder=5)
        rec.append((cond, v, lo, hi, wv))
    return rec


def draw(rows, bbs, ql, qb):
    FIGA.mkdir(parents=True, exist_ok=True)
    data1, data2 = [], []
    # ---- Fig 1
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), sharey=True)
    titles = {"completed": "Task completion rate", "correct": "Final-answer correctness",
              "tool_f1": "Tool-selection F1"}
    for k, (ax, m) in enumerate(zip(axes, MET)):
        despine(ax)
        for j, b in enumerate(bbs):
            st = paired_stats(ql, b["id"])
            n = len(st["qids"])
            if n == 0:
                ax.text(j, 0.5, "no paired\nvalid data", ha="center", va="center", fontsize=6.5, color="#777777")
                continue
            wb = {c: np.mean([qb[(b["id"], c, q)][m] for q in st["qids"] if (b["id"], c, q) in qb] or [np.nan])
                  for c in ("direct", "perovagents")}
            for cond, v, lo, hi, wv in bars(ax, j, st, wb, m):
                data1.append([m, b["id"], b["model"], cond, n, round(v, 4), fnum(lo, 4), fnum(hi, 4), fnum(wv, 4)])
        ax.set_xticks(range(len(bbs)))
        labs = []
        for b in bbs:
            st = paired_stats(ql, b["id"])
            labs.append(f"{b['label'].replace(' ', chr(10), 1)}\nn={len(st['qids'])} q" if st["qids"]
                        else f"{b['label'].replace(' ', chr(10), 1)}\nn=0")
        ax.set_xticklabels(labs, fontsize=6.5)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.6, len(bbs) - 0.4)
        ax.set_title(titles[m], loc="left", fontsize=8.2)
        ax.text(-0.02, 1.07, f"({'abc'[k]})", transform=ax.transAxes, fontweight="bold", fontsize=9, ha="right")
    axes[0].set_ylabel("Question-level mean")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = [Patch(color=COLOR[c], label=LABEL[c]) for c in COLOR] + \
         [Line2D([], [], marker="_", ms=7, mew=1.3, color="#222222", ls="none",
                 label="worst case (infrastructure/empty-output attempts = 0)")]
    fig.legend(handles=hs, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=7)
    fig.text(0.995, 0.005, "Bars: valid episodes, paired by question (rep 0 only; 95% CI shown only if n>=5 questions)",
             ha="right", va="bottom", fontsize=5.5, color="#777777")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(FIGA / f"fig1_overall_v2.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    write_csv(FIGA / "fig1_overall_v2_data.csv", ["metric", "backbone", "model_id", "condition",
              "n_paired_questions", "value", "ci_lo", "ci_hi", "worst_case"], data1)
    # ---- Fig 2
    nb = len(bbs)
    fig, axes = plt.subplots(2, nb, figsize=(2.5 * nb, 4.2), sharey=True, squeeze=False)
    for r_i, m in enumerate(("completed", "correct")):
        for c_i, b in enumerate(bbs):
            ax = axes[r_i][c_i]
            despine(ax)
            for x, (cat, _) in enumerate(CATS):
                st = paired_stats(ql, b["id"], cat)
                n = len(st["qids"])
                if n == 0:
                    ax.text(x, 0.03, "n/a", ha="center", fontsize=6, color="#888888")
                    continue
                for cond, v, lo, hi, wv in bars(ax, x, st, {}, m, 0.38):
                    data2.append([m, b["id"], cat, cond, n, round(v, 4), fnum(lo, 4), fnum(hi, 4)])
                ax.text(x, 1.02, f"n={n}", ha="center", fontsize=5.5, color="#555555", clip_on=False)
            ax.set_xticks(range(len(CATS)))
            ax.set_xticklabels([l for _, l in CATS], rotation=40, ha="right", fontsize=6.8)
            ax.set_ylim(0, 1.0)
            ax.set_xlim(-0.6, len(CATS) - 0.4)
            if r_i == 0:
                ax.set_title(b["label"], fontsize=8.2, pad=12)
            if c_i == 0:
                ax.set_ylabel({"completed": "Task completion rate", "correct": "Final-answer correctness"}[m])
    fig.legend(handles=[Patch(color=COLOR[c], label=LABEL[c]) for c in COLOR], loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=7)
    fig.text(0.995, 0.005, "n = paired questions with valid episodes in both conditions; n/a = no paired data (not zero)",
             ha="right", va="bottom", fontsize=5.5, color="#777777")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(FIGA / f"fig2_category_v2.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    write_csv(FIGA / "fig2_category_v2_data.csv", ["metric", "backbone", "category", "condition",
              "n_paired_questions", "value", "ci_lo", "ci_hi"], data2)


def main():
    global MAIN
    bank = load_bank()[1]
    MAIN = {q["id"]: q["category"] for q in bank.values() if q["set"] == "main"}
    from .providers import load_models
    bbs = load_models()
    rows = load_episodes()
    AUD.mkdir(parents=True, exist_ok=True)
    write_csv(AUD / "episode_validity.csv",
              ["episode_id", "qid", "category", "backbone", "condition", "rep", "manifest_status", "termination",
               "validity", "evidence", "seconds", "model_calls", "tool_calls", "n_asks", "escalations",
               "completed", "correct", "tool_f1"],
              [[r["eid"], r["qid"], r["category"], r["backbone"], r["condition"], r["rep"], r["status"],
                r["termination"], r["validity"], r["evidence"], r["seconds"], r["model_calls"], r["tool_calls"],
                r["n_asks"], r["escalations"]] + ([fnum(r["score"][m]) for m in MET] if r["score"] else ["", "", ""])
               for r in sorted(rows, key=lambda r: (r["backbone"], r["qid"], r["condition"]))])
    write_csv(AUD / "coverage_matrix.csv",
              ["backbone", "condition", "category", "questions_planned", "episodes_planned", "episodes_attempted",
               "unique_questions_attempted", "quality_denominator_episodes", "of_which_incorrect",
               "no_model_output_v2", "provider_limited_v2",
               "infra_failed", "unclassified_empty", "valid_empty_completion", "episodes_not_yet_run"],
              coverage(rows, bbs))
    ql = qlevel(rows, lambda r: r["validity"] in QUALITY_OK)
    qx = qlevel(rows, lambda r: r["validity"] in ("valid", "valid_empty_completion"))
    qa = qlevel(rows, lambda r: r["status"] == "completed")        # legacy rule (A)
    qb = worst_case(rows)                                           # sensitivity B (final attempt per cell)
    qf = worst_case(first_attempt_rows())                           # end-to-end: first attempt, no replacement
    write_csv(AUD / "attempts.csv", ["episode_id", "attempt", "manifest_status", "validity", "replaces",
                                     "attempt_file", "evidence"], attempts_table())
    mrows = []
    for b in bbs:
        for name, q in (("primary_infra_excluded", ql),
                        ("sens_also_excluding_unclassified_empty", qx),
                        ("A_legacy_only_infra_removed", qa),
                        ("operational_all_attempts_nonvalid_is_failure", qb),
                        ("operational_first_attempt_nonvalid_is_failure", qf)):
            st = paired_stats(q, b["id"])
            for m in MET:
                s = st[m]
                mrows.append([b["id"], name, m, len(st["qids"]), fnum(s["direct"]), fnum(s["perovagents"]),
                              fnum(s["diff"]), fnum(s["diff_ci"][0]), fnum(s["diff_ci"][1])])
    write_csv(AUD / "metrics_v2.csv", ["backbone", "analysis", "metric", "n_paired_questions", "direct",
              "perovagents", "diff_perov_minus_direct", "diff_ci_lo", "diff_ci_hi"], mrows)
    write_csv(AUD / "hypotheses.csv", ["backbone", "quantity", "value", "unit", "denominator"], hypotheses(rows))
    draw(rows, bbs, ql, qb)
    n = defaultdict(int)
    for r in rows:
        n[r["validity"]] += 1
    json.dump({"episodes": len(rows), "validity": dict(n)}, open(AUD / "summary.json", "w"), indent=1)
    print(f"audit written: {len(rows)} episodes, {dict(n)}")


if __name__ == "__main__":
    main()
