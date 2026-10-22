"""Ablation study harness (layered MRS contributions A-G).

Each mode maps to one step of the architecture build-up; the same benchmark
questions are run under every mode so plan-validity/tool-selection/execution/
completion metrics can be attributed to individual components:

    A raw       raw LLM tool calling (baseline)
    B schemas   + first-class action schemas
    C analyzer  + task analyzer feeding the planner
    D gate      + pre-execution validation gate
    E verifier  + output contract verifier (report only)
    F replan    + iterative re-planning to SUFFICIENT
    G full      + task-analyzer->planner contract + claims/ontology

Usage:
    python -m benchmarks.ablation --modes D,E,F,G --limit 30
"""
import argparse

from ..agent.orchestrator import Agent
from . import runner as bencher


def agent_for(mode: str, pipeline: str = "llm", fast: bool = False) -> Agent:
    """Return an agent factory pre-configured for one ablation mode."""

    def factory():
        return Agent(pipeline=pipeline, mode=mode, fast=fast)
    return factory


def run_mode(mode: str, pipeline: str, questions, limit: int = None,
             label: str = None, save: bool = True):
    """Run one ablation mode over (a subset of) the benchmark questions."""
    subset = questions[:limit] if limit and limit > 0 else questions
    label = label or f"mode_{mode}"
    return bencher.run(subset, agent_for(mode, pipeline),
                       label=f"mode_{mode}", save=save)


def run_all(modes, pipeline, questions, limit=None, save=True):
    """Run (and summarize) every requested ablation mode in sequence."""
    reports = {}
    for m in modes:
        print(f">>> running ablation mode {m} ({pipeline} pipeline)")
        reports[m] = run_mode(m, pipeline, questions, limit=limit, save=save)
        print(bencher.summarize(reports[m]) + "\n")
    return reports


def compare(reports) -> str:
    """Side-by-side ASCII comparison table of the ablation modes."""
    import json
    out = ["mode | class | select | planval | exec | contract | sufficient | recover"]
    for m in sorted(reports):
        a = reports[m]["aggregate"]
        out.append(
            f"{m:4} | "
            f"{a['classification_accuracy_pct']:5} | "
            f"{a['selection_accuracy_pct']:6} | "
            f"{a['parameter_validity_pct']:6} | "
            f"{a['execution_success_pct']:4} | "
            f"{a['contract_satisfied_pct']:8} | "
            f"{a['final_sufficient_pct']:9} | "
            f"{a['recovery_rate_pct']}")
    return "  " + "\n  ".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=str,
                    default="A,B,C,D,E,F,G")
    ap.add_argument("--pipeline", default="llm")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from perovagents.benchmarks.questions import all_questions

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    reps = run_all(modes, args.pipeline, all_questions(),
                   limit=args.limit or None)
    print(compare(reps))