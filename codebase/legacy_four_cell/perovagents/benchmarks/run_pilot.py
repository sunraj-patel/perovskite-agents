"""Pilot runner: quick validation of the benchmark machinery on a subset.

Usage:
    python -m benchmarks.run_pilot \
        --pipeline rules --limit 6 --subset disambig
"""
import argparse

from perovagents.benchmarks import runner as bencher
from perovagents.benchmarks import questions
from perovagents.benchmarks.ablation import agent_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="rules")
    ap.add_argument("--mode", default="G")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subset", default="ambiguous",
                    choices=["ambiguous", "disambig", "simple", "all"])
    ap.add_argument("--fast", action="store_true",
                    help="deterministic offline analyzer + synthesis (no LLM)")
    args = ap.parse_args()

    qs = {
        "ambiguous": questions.AMBIGUOUS,
        "disambig": questions.DISAMBIGUATION,
        "simple": questions.SIMPLE,
        "all": questions.all_questions(),
    }[args.subset]
    if args.limit:
        qs = qs[: args.limit]

    label = f"pilot_{args.pipeline}_{args.mode}_{args.subset}"
    rep = bencher.run(qs, agent_for(args.mode, args.pipeline, fast=args.fast),
                      label=label, save=True)
    print(f"== {args.pipeline} pipeline, mode {args.mode}, {len(qs)} q ==")
    print(bencher.summarize(rep))
    hist = {}
    for r in rep["rows"]:
        hist[r["task_type"]] = hist.get(r["task_type"], 0) + 1
    print("analyzer task_type histogram:",
          dict(sorted(hist.items())))
    mism = [r for r in rep["rows"] if not r["gt_hit"]]
    if mism:
        print("gt_miss:")
        for r in mism:
            print("   ", r["qid"], r["task_type"], "->", r["plan_actions"])


if __name__ == "__main__":
    main()