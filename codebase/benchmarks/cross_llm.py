"""Cross-LLM benchmark runner (spec sections 12/14/15/16).

    python -m benchmarks.cross_llm --list
    python -m benchmarks.cross_llm --system mrs --dry-run
    python -m benchmarks.cross_llm --system gemini --question Q01
    python -m benchmarks.cross_llm --system mrs --category pdb --limit 2
    python -m benchmarks.cross_llm --all --limit 1

One JSON record per question x system under results/cross_llm/<stamp>/.
Cost: token counts are the hard control; USD cost needs a rate the runner
does not invent - set CROSS_LLM_PRICE_PER_1M_USD (input+output blended)
or the record keeps estimated_cost_usd=null with pricing_verified=false.
"""
import argparse
import datetime
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.cross_llm_questions import (  # noqa: E402
    CATEGORIES, PILOT, QUESTIONS, by_category, get_question)
from benchmarks.tool_layer import (  # noqa: E402
    MAX_LLM_TURNS, MAX_TOOL_CALLS, RUN_TIMEOUT_S, collect_tools,
    describe_tool_versions)

BENCHMARK_VERSION = 1
SYSTEMS = ("mrs", "mrs-deepseek", "gemini", "deepseek", "claude", "gpt")


def results_dir():
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results", "cross_llm",
                     datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(d, exist_ok=True)
    return d


def price_rate():
    try:
        return float(os.environ.get("CROSS_LLM_PRICE_PER_1M_USD", ""))
    except ValueError:
        return None


def finish_record(base, result):
    tok = result.get("tokens", {})
    pt = (tok.get("real_prompt_tokens") if tok.get("real_usage")
          else tok.get("prompt_tokens_est", 0)) or 0
    ct = (tok.get("real_completion_tokens") if tok.get("real_usage")
          else tok.get("completion_tokens_est", 0)) or 0
    rate = price_rate()
    base.update({
        "termination": result.get("termination"),
        "final_answer": result.get("answer", ""),
        "n_tool_calls": result.get("n_tool_calls", 0),
        "n_llm_calls": tok.get("llm_calls", 0),
        "tool_calls": result.get("tool_calls", []),
        "trace_path": result.get("trace_path"),
        "seconds": result.get("seconds"),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "real_usage": bool(tok.get("real_usage")),
        "estimated_cost_usd": round((pt + ct) / 1e6 * rate, 4) if rate else None,
        "pricing_verified": bool(rate),
    })
    return base


def run_one(qid, system, outdir, dry=False):
    q = get_question(qid)
    if dry:
        return {"qid": qid, "system": system, "dry_run": True,
                "question": q["question"][:100]}
    base = {"timestamp": datetime.datetime.now().isoformat(),
            "benchmark_version": BENCHMARK_VERSION,
            "qid": qid, "category": q["category"], "system": system,
            "question": q["question"],
            "budgets": {"max_tool_calls": MAX_TOOL_CALLS,
                        "max_llm_turns": MAX_LLM_TURNS,
                        "timeout_s": RUN_TIMEOUT_S},
            "tool_versions": describe_tool_versions()}
    try:
        if system == "mrs":
            from benchmarks.mrs_adapter import run_mrs
            from perovagents import config
            base["model"] = config.LLM_MODEL
            result = run_mrs(q["question"])
        elif system == "mrs-deepseek":
            from benchmarks.mrs_adapter import run_mrs
            from benchmarks.providers import OpenRouterProvider
            provider = OpenRouterProvider()
            base["model"] = provider.model
            result = run_mrs(q["question"], provider=provider)
        elif system in ("gemini", "claude", "gpt"):
            from benchmarks.harness import DirectBaseline
            from benchmarks.providers import get_provider
            provider = get_provider(system)
            base["model"] = provider.model
            tools, _agents = collect_tools()
            result = DirectBaseline(
                provider, tools, timeout_s=RUN_TIMEOUT_S).run(q["question"])
        else:
            raise KeyError(f"unknown system {system!r}")
    except Exception as exc:
        base.update({"termination": "system_unavailable"
                     if "not configured" in str(exc) else "harness_error",
                     "error": str(exc)[:500],
                     "traceback": traceback.format_exc(limit=3)})
        path = os.path.join(outdir, f"{qid}.{system}.json")
        json.dump(base, open(path, "w"), indent=1, default=str)
        return {"qid": qid, "system": system, "termination": base["termination"],
                "path": path}
    rec = finish_record(base, result)
    path = os.path.join(outdir, f"{qid}.{system}.json")
    json.dump(rec, open(path, "w"), indent=1, default=str)
    return {"qid": qid, "system": system, "termination": rec["termination"],
            "total_tokens": rec["total_tokens"], "n_tool_calls": rec["n_tool_calls"],
            "path": path}


def main(argv=None):
    p = argparse.ArgumentParser(prog="benchmarks.cross_llm")
    p.add_argument("--list", action="store_true", help="print the 20 questions and exit")
    p.add_argument("--system", choices=SYSTEMS, help="one system to run")
    p.add_argument("--all", action="store_true", help="all four systems")
    p.add_argument("--question", help="single question ID, e.g. Q06")
    p.add_argument("--category", choices=CATEGORIES, help="one category")
    p.add_argument("--pilot", action="store_true", help="the 7-question pilot set")
    p.add_argument("--limit", type=int, help="cap the number of questions")
    p.add_argument("--dry-run", action="store_true", help="plan only, no LLM calls")
    a = p.parse_args(argv)
    if a.list:
        for q in QUESTIONS:
            print(f"{q['id']} [{q['category']}/{q['difficulty']}] {q['question'][:90]}")
        return
    systems = list(SYSTEMS) if a.all else ([a.system] if a.system else ["mrs"])
    if a.question:
        qs = [get_question(a.question)["id"]]
    elif a.category:
        qs = [q["id"] for q in by_category(a.category)]
    elif a.pilot:
        qs = list(PILOT)
    else:
        qs = [q["id"] for q in QUESTIONS]
    if a.limit:
        qs = qs[:a.limit]
    outdir = results_dir()
    print(f"benchmark v{BENCHMARK_VERSION}: {len(qs)} questions x {systems} -> {outdir}",
          flush=True)
    print(f"budgets: {MAX_TOOL_CALLS} tool calls, {MAX_LLM_TURNS} turns, {RUN_TIMEOUT_S}s timeout",
          flush=True)
    for qid in qs:
        for system in systems:
            if not a.dry_run:
                time.sleep(5)               # pacing between runs (quota)
            summary = run_one(qid, system, outdir, dry=a.dry_run)
            print(json.dumps(summary, default=str)[:300], flush=True)


if __name__ == "__main__":
    main()
