"""Command-line entry point: `python -m perovagents "your question"`.

Examples:
    python -m perovagents "what is the average PCE for devices with SnO2?"
    python -m perovagents --pipeline rules "predict PCE for a p-i-n with PTAA/SnO2"
"""
import argparse
import json
import sys

from .agent import Agent
from .core.logsetup import setup_logging

setup_logging()


def build_parser():
    """CLI argument parser."""
    p = argparse.ArgumentParser(prog="perovagents",
                                description="PerovAgents CLI")
    p.add_argument("question", nargs="?", help="research question")
    p.add_argument("--pipeline", choices=["llm", "rules"], default="llm",
                   help="planner backend (default llm, auto-falls-back offline)")
    p.add_argument("--cells", action="store_true",
                   help="use the 4-cell architecture (planner v2 + executor "
                        "v2 + gated cells) instead of the legacy pipeline")
    p.add_argument("--json", action="store_true", help="dump full JSON result")
    p.add_argument("--list-tools", action="store_true", help="print available tools")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cells:  # new architecture path (S9); legacy default untouched
        if not args.question:
            print("question required (or use --list-tools)", file=sys.stderr)
            return 2
        from .control.app import answer_cells
        result = answer_cells(args.question, pipeline=args.pipeline)
        print("\n" + result["answer"])
        print("\n[sources] " + ", ".join(result["sources"] or ["(none)"]))
        if result.get("flow_log"):
            print(f"[flow log] {result['flow_log']}")
        if result["failed"]:
            print("[failed cells]")
            for tid, reason in result["failed"].items():
                print(f"  - {tid}: {reason[:200]}")
        if result["warnings"]:
            print("[warnings] " + "; ".join(result["warnings"]))
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        return 0
    agent = Agent(pipeline=args.pipeline)
    if args.list_tools:
        for t in agent.tools():
            print(f"  [{t.layer}] {t.name}: {t.description[:80]}")
        return 0
    if not args.question:
        print("question required (or use --list-tools)", file=sys.stderr)
        return 2
    result = agent.answer(args.question)
    print("\n" + result["answer"])
    print("\n[sources] " + ", ".join(result["sources"] or ["(none)"]))
    if result.get("flow_log"):
        print(f"[flow log] {result['flow_log']}")
    if result["warnings"]:
        print("[warnings] " + "; ".join(result["warnings"]))
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())