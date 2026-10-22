"""The two principal evaluation figures, drawn ONLY from the scored tables.

    python -m benchmarks.agentic_paper.figures            # (re)draw both
    python -m benchmarks.agentic_paper.figures --final    # no interim stamp

Stable outputs (same file names every time, so a manuscript can keep
\\includegraphics-ing them while the benchmark grows):

    figures/fig1_overall_comparison.{pdf,png}      completion / correctness / tool-F1 per backbone
    figures/fig2_category_performance.{pdf,png}    completion + correctness per task category
    figures/fig1_overall_data.csv, fig2_category_data.csv   numbers plotted (+ n, CI)
    figures/figure_status.json                     coverage at draw time

Layout, size, axis ranges and colours are fixed; backbones come from
config/models.json (not from the data), so a backbone with no results yet shows
an explicit "no data yet" panel instead of changing the figure geometry.
Style follows the manuscript's figure scripts (Arial/Helvetica 8 pt, open
top/right spines, light horizontal grid, no legend frame, bold panel labels,
300 dpi, Type-42 fonts in the PDF).
"""
import csv
import json
import sys
import time

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .paths import FIG_DIR, TABLES_DIR, setup_env  # noqa: E402

setup_env()
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "pdf.fonttype": 42, "savefig.dpi": 300,
})
COLOR = {"direct": "#4A78A8", "perovagents": "#E4AA28"}
LABEL = {"direct": "Direct tool use", "perovagents": "PerovAgents"}
CATS = [("pdb", "Database"), ("literature", "Literature"), ("ml", "ML"), ("scaps", "SCAPS"),
        ("multi_tool", "Multi-tool")]
TARGET_REPS = 3


def despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)


def read(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    return float(x) if x not in ("", None) else None


def models():
    from .providers import load_models
    return load_models()


def planned_total(bbs):
    from .scorer import load_bank
    n_main = sum(1 for q in load_bank()[1].values() if q["set"] == "main")
    return n_main * len(bbs) * 2 * TARGET_REPS


def coverage_text(bbs):
    cov = read(TABLES_DIR / "coverage.csv")
    done = sum(int(r["episodes_done"]) for r in cov)
    total = planned_total(bbs)
    return done, total


def stamp(fig, bbs, final):
    done, total = coverage_text(bbs)
    if final or done >= total:
        return done, total
    fig.text(0.995, 0.005, f"interim: {done} of {total} planned episodes scored",
             ha="right", va="bottom", fontsize=6, color="#888888")
    return done, total


def bar_pair(ax, x, rows, metric, width=0.36):
    """Draw direct/perovagents bars at x. Returns plotted records."""
    recs = []
    for i, cond in enumerate(("direct", "perovagents")):
        r = rows.get(cond)
        if not r or fnum(r.get(metric)) is None:
            continue
        v, lo, hi = fnum(r[metric]), fnum(r[metric + "_lo"]), fnum(r[metric + "_hi"])
        xx = x + (i - 0.5) * width
        ax.bar(xx, v, width * 0.94, color=COLOR[cond], linewidth=0,
               label=LABEL[cond] if cond not in ax.get_legend_handles_labels()[1] else None)
        ax.errorbar(xx, v, yerr=[[v - lo], [hi - v]], color="#333333", linewidth=0.7, capsize=1.6,
                    capthick=0.7)
        recs.append((cond, v, lo, hi))
    return recs


def fig1(final):
    bbs = models()
    rows = read(TABLES_DIR / "overall_by_model.csv")
    by = {(r["backbone"], r["condition"]): r for r in rows}
    metrics = [("completed", "Task completion rate"), ("correct", "Final-answer correctness"),
               ("tool_f1", "Tool-selection F1")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    data = []
    for k, (ax, (m, title)) in enumerate(zip(axes, metrics)):
        despine(ax)
        for j, b in enumerate(bbs):
            pair = {c: by.get((b["id"], c)) for c in ("direct", "perovagents")}
            if not any(pair.values()):
                ax.text(j, 0.04, "no data\nyet", ha="center", va="bottom", fontsize=6.5, color="#888888")
                continue
            for cond, v, lo, hi in bar_pair(ax, j, pair, m):
                r = pair[cond]
                data.append([m, b["id"], b["model"], cond, r["n_questions_paired"], r["n_episodes"],
                             round(v, 4), round(lo, 4), round(hi, 4)])
        ax.set_xticks(range(len(bbs)))
        ax.set_xticklabels([b["label"].replace(" ", "\n", 1) for b in bbs], fontsize=6.5)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.6, len(bbs) - 0.4)
        ax.set_title(title, loc="left", fontsize=8.2)
        ax.text(-0.02, 1.07, f"({'abc'[k]})", transform=ax.transAxes, fontweight="bold", fontsize=9,
                va="bottom", ha="right")
    axes[0].set_ylabel("Proportion (question-level mean)")
    h, l = axes[0].get_legend_handles_labels()
    if not h:
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            if h:
                break
    if h:
        fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    done, total = stamp(fig, bbs, final)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig1_overall_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_overall_comparison.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    with open(FIG_DIR / "fig1_overall_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "backbone", "model_id", "condition", "n_questions", "n_episodes",
                    "value", "ci95_lo", "ci95_hi"])
        w.writerows(data)
    return done, total


def fig2(final):
    bbs = models()
    rows = read(TABLES_DIR / "by_category.csv")
    by = {(r["backbone"], r["category"], r["condition"]): r for r in rows}
    nb = len(bbs)
    fig, axes = plt.subplots(2, nb, figsize=(2.45 * nb, 4.0), sharey=True, squeeze=False)
    data = []
    letters = "abcdefghij"
    for r_i, (m, mtitle) in enumerate((("completed", "Task completion rate"),
                                      ("correct", "Final-answer correctness"))):
        for c_i, b in enumerate(bbs):
            ax = axes[r_i][c_i]
            despine(ax)
            any_data = False
            for x, (cat, _lab) in enumerate(CATS):
                pair = {c: by.get((b["id"], cat, c)) for c in ("direct", "perovagents")}
                if not any(pair.values()):
                    continue
                any_data = True
                for cond, v, lo, hi in bar_pair(ax, x, pair, m, width=0.38):
                    r = pair[cond]
                    data.append([m, b["id"], b["model"], cat, cond, r["n_questions_paired"],
                                 r["n_episodes"], round(v, 4), round(lo, 4), round(hi, 4)])
            if not any_data:
                ax.text(2, 0.5, "no data yet", ha="center", fontsize=7, color="#888888")
            ax.set_xticks(range(len(CATS)))
            ax.set_xticklabels([lab for _c, lab in CATS], rotation=40, ha="right", fontsize=6.8)
            ax.set_ylim(0, 1.0)
            ax.set_xlim(-0.6, len(CATS) - 0.4)
            if r_i == 0:
                ax.set_title(b["label"], fontsize=8.2)
            if c_i == 0:
                ax.set_ylabel(mtitle)
            ax.text(-0.02, 1.02, f"({letters[r_i * nb + c_i]})", transform=ax.transAxes,
                    fontweight="bold", fontsize=9, va="bottom", ha="right")
    h, l = [], []
    for ax in axes.flat:
        hh, ll = ax.get_legend_handles_labels()
        if hh:
            h, l = hh, ll
            break
    if h:
        fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    done, total = stamp(fig, bbs, final)
    fig.savefig(FIG_DIR / "fig2_category_performance.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_category_performance.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    with open(FIG_DIR / "fig2_category_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "backbone", "model_id", "category", "condition", "n_questions",
                    "n_episodes", "value", "ci95_lo", "ci95_hi"])
        w.writerows(data)
    return done, total


def main():
    final = "--final" in sys.argv
    d1 = fig1(final)
    d2 = fig2(final)
    done, total = d1
    json.dump({"drawn_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "episodes_scored": done,
               "episodes_planned": total, "interim": (done < total and not final)},
              open(FIG_DIR / "figure_status.json", "w"), indent=1)
    print(f"figures written to {FIG_DIR} ({done}/{total} episodes scored)")


if __name__ == "__main__":
    main()
