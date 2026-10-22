"""Deterministic tests for the v4 direct-baseline protocol (no network, no model)."""
import json

from conftest import ScriptedLLM
from benchmarks.agentic_paper.baseline_v4 import BASELINE_SYSTEM_V4, ProtocolBaseline
from benchmarks.harness import BASELINE_SYSTEM, DirectBaseline
from benchmarks.tool_layer import DONE

call = lambda tool, **args: json.dumps({"tool": tool, "args": args})          # noqa: E731


def echo(x):
    """echo(x) returns x."""
    return f"echo:{x}"


def test_prompt_is_neutral_and_describes_the_protocol():
    p = BASELINE_SYSTEM_V4
    assert '"tool"' in p and '"args"' in p and DONE in p                       # syntax + how to finish
    for banned in ("ASK ", "orchestrator", "progress", "BLOCKED", "INCOMPLETE", "EVIDENCE GATHERED",
                   "agent", "Keep trying until you are satisfied", "REPEATED_ACTION"):
        assert banned not in p, banned
    assert p != BASELINE_SYSTEM                                                  # the v3 wording is retired


def test_valid_call_then_final_answer():
    llm = ScriptedLLM([call("echo", x="a"), f"{DONE} done: echo:a"])
    out = ProtocolBaseline(llm, {"echo": echo}).run("q?")
    assert out["termination"] == "answer" and out["answer"] == "done: echo:a"
    assert out["n_tool_calls"] == 1 and out["tool_calls"][0]["result"] == "echo:a"
    assert BASELINE_SYSTEM_V4.splitlines()[0] in llm.prompts[0] and "echo(x)" in llm.prompts[0]


def test_loop_and_budgets_match_the_original_baseline():
    script = [call("echo", x="1"), "no json here", call("nope"), call("echo", x="2"), f"{DONE} z"]
    a = ProtocolBaseline(ScriptedLLM(list(script)), {"echo": echo}).run("q")
    b = DirectBaseline(ScriptedLLM(list(script)), {"echo": echo}).run("q")
    for k in ("answer", "termination", "n_tool_calls"):
        assert a[k] == b[k], k
    t = ProtocolBaseline(ScriptedLLM(lambda p: call("echo", x="1")), {"echo": echo}, max_tool_calls=2).run("q")
    assert t["termination"] == "tool_budget" and t["n_tool_calls"] == 2
    n = ProtocolBaseline(ScriptedLLM(lambda p: "prose"), {"echo": echo}, max_turns=3).run("q")
    assert n["termination"] == "turn_budget" and n["answer"] == ""
