"""Demo: run PerovAgents on a small research question.

Uses the rule planner (offline-friendly) so it works without an API key. The
LLM planner activates automatically when GOOGLE_API_KEY is configured.

    python examples/demo_agent.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perovagents import Agent


def main():
    agent = Agent(pipeline="rules")   # deterministic, offline
    print("Available tools:")
    for t in agent.tools():
        acts = ", ".join(t.actions.keys()) or "(raw)"
        print(f"  [{t.layer:>12}] {t.name}  ({acts})")
    print("\n--- question: What is the average PCE for devices with SnO2 ETL? ---\n")

    result = agent.answer("What is the average PCE for devices with SnO2 ETL?")
    print("ANSWER\n" + result["answer"])
    print("\nSOURCES\n" + ", ".join(result["sources"] or ["(none)"]))
    print("\nTASK SPEC\n" + json.dumps(result["task_spec"], indent=1))
    print("\nVERDICT\n" + json.dumps(result["verdict"], indent=1))
    print("\nTRACE")
    for s in result["execution"]["steps"]:
        print(f"  step {s['step_index']}: {s['tool']}.{s['action']} -> "
              f"{s['result_summary']} ({s['success']})")


if __name__ == "__main__":
    main()