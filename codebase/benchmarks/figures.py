"""Figure scripts A-E (spec section 17), generated ONLY from saved results.

    python -m benchmarks.figures results/cross_llm/<stamp>

Reads report.json (produced by benchmarks.report) and writes figures/*.png.
No numbers are typed in: every bar comes from the scored runs.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _bars(ax, labels, series, title, ylabel):
    x = range(len(labels))
    w = 0.8 / max(len(series), 1)
    for i, (name, vals) in enumerate(series):
        vals = [0 if v is None else v for v in vals]
        ax.bar([p + i * w for p in x], vals, width=w, label=name)
    ax.set_xticks([p + w * (len(series) - 1) / 2 for p in x])
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()


def main(argv=None):
    rundir = (argv or sys.argv[1:])[0]
    rep = json.load(open(os.path.join(rundir, "report.json")))
    t1, t2 = rep["table1_overall"], rep["table2_category"]
    figdir = os.path.join(rundir, "figures")
    os.makedirs(figdir, exist_ok=True)
    systems = [r["system"] for r in t1]

    fig, ax = plt.subplots(figsize=(8, 4))
    _bars(ax, ["completion", "contract", "tool-F1", "scientific"],
          [(s, [next(r[k] or 0 for r in t1 if r["system"] == s)
                for k in ("completion", "contract", "tool_f1", "scientific")])
           for s in systems],
          "A. overall correctness by system", "score")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "A_overall.png"), dpi=120)

    cats = sorted(set(r["category"] for r in t2))
    fig, ax = plt.subplots(figsize=(9, 4))
    _bars(ax, cats,
          [(s, [next((r["scientific"] or 0 for r in t2
                      if r["category"] == c and r["system"] == s), 0) for c in cats])
           for s in systems],
          "B. scientific score by category", "score")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "B_category.png"), dpi=120)

    t3 = rep.get("table3_recovery", [])
    rec_rate = {s: (sum(1 for r in t3 if r["system"] == s and r["recovered"])
                    / max(sum(1 for r in t3 if r["system"] == s), 1)) for s in systems}
    fig, ax = plt.subplots(figsize=(7, 4))
    _bars(ax, ["recovery rate"], [(s, [rec_rate[s]]) for s in systems],
          "C. recovery success (recovery-sensitive Qs)", "rate")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "C_recovery.png"), dpi=120)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _bars(axes[0], systems,
          [("tool calls", [next((r["mean_tool_calls"] or 0 for r in t1
                                 if r["system"] == s), 0) for s in systems]),
           ("llm calls", [next((r["mean_llm_calls"] or 0 for r in t1
                                if r["system"] == s), 0) for s in systems])],
          "D1. calls by system", "count")
    _bars(axes[1], systems,
          [("k tokens", [next(((r["mean_tokens"] or 0) / 1000 for r in t1
                               if r["system"] == s), 0) for s in systems])],
          "D2. tokens by system", "1000s")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "D_efficiency.png"), dpi=120)

    from collections import Counter
    fails = Counter(s["failure"] for s in rep.get("scores", []))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(list(fails), list(fails.values()))
    ax.set_title("E. failure taxonomy counts")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "E_failures.png"), dpi=120)
    print(f"wrote 5 figures to {figdir}")


if __name__ == "__main__":
    main()
