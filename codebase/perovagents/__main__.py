"""python -m perovagents "question" [-v] [--trace] | --agent NAME "task" | --show-agent NAME
   | --session"""
import argparse

from . import answer
from .llm import get_llm
from .team import build_team

p = argparse.ArgumentParser(prog="perovagents")
p.add_argument("question", nargs="?")
p.add_argument("-v", "--verbose", action="store_true", help="print every tool call and result")
p.add_argument("--trace", action="store_true", help="write a JSONL trace to results/traces/")
p.add_argument("--agent", help="skip the orchestrator: give the task straight to one agent")
p.add_argument("--show-agent", help="print an agent's full prompt (what it knows, its tools)")
p.add_argument("--session", action="store_true",
               help="interactive: ask many questions, every run traced and logged")
p.add_argument("--sequential", action="store_true",
               help="run ASKed agents one at a time with a pacing gap (for tight API quotas)")
a = p.parse_args()

if a.session:
    from .session import run_session
    run_session()
elif a.show_agent:
    team, missing = build_team(None, only=[a.show_agent])
    print(team[a.show_agent].system if team else missing)
elif a.question and a.agent:
    from .trace import Trace
    team, missing = build_team(get_llm(), only=[a.agent])
    if not team:
        print(missing)
    else:
        t = Trace(a.question, enabled=a.trace) if a.trace else None
        out = team[a.agent].run(a.question, a.verbose, trace=t)
        print(out)
        if t and t.path and a.verbose:
            print(f"\n[trace] {t.path}")
elif a.question:
    print(answer(a.question, verbose=a.verbose, trace=a.trace, sequential=a.sequential))
else:
    p.print_usage()
