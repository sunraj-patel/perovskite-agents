"""MRS adapter: run one benchmark question through the FULL current MRS.

Full mode = orchestrator dispatch over the domain agents (pdb, literature,
ml, scaps, design) with their progress contracts, guard/assess/review,
no-progress stops, repeat caps, orchestrator duplicate-ASK control and
trace provenance. Sequential dispatch (one agent at a time) keeps pool
usage under per-minute quota; orchestration logic is otherwise unchanged.

Timeout uses a DAEMON thread + join(timeout): ThreadPoolExecutor's
context-manager shutdown joins the worker, which parks the main thread
forever when the worker overruns (observed: record never written, process
stuck in ThreadHandle_join). A timed-out worker leaks but cannot block
the record or process exit; timeouts should be rare (1200 s budget).

Returns the same record shape as DirectBaseline plus the trace path, so
the runner and metrics treat both systems uniformly.
"""
import threading
import time

from .harness import TokenRecorder
from .tool_layer import RUN_TIMEOUT_S


def run_mrs(question, trace=True, timeout_s=RUN_TIMEOUT_S, verbose=False,
            provider=None):
    from perovagents.agent import Orchestrator
    from perovagents.llm import get_llm
    from perovagents.team import build_team
    from perovagents.trace import Trace
    t0 = time.time()
    # provider=None -> Gemini key pool (default MRS). A Provider object
    # (e.g. DeepSeek via OpenRouter) puts MRS on that model instead, so
    # baseline-vs-MRS comparisons can hold the model constant.
    recorder = TokenRecorder(provider if provider is not None else get_llm())
    team, missing = build_team(recorder)
    if not team:
        raise RuntimeError(f"MRS adapter: no agent built: {missing}")
    orch = Orchestrator(team, recorder, sequential=True)
    t = Trace(f"[cross_llm/mrs] {question}", enabled=bool(trace)) if trace else None
    termination, answer, box = "answer", "", {}
    th = threading.Thread(target=lambda: box.setdefault(
        "answer", orch.ask(question, verbose, t)), daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        termination = "timeout"
        answer = "(MRS run timed out; partial trace kept)"
    else:
        answer = box.get("answer", "")
    rec = {"answer": answer, "termination": termination,
           "trace_path": t.path if t else None,
           "seconds": round(time.time() - t0, 1),
           "tokens": recorder.summary()}
    if t and t.path:
        try:
            from perovagents.trace import read_trace
            evs = list(read_trace(t.path))
            rec["n_tool_calls"] = sum(1 for e in evs if e.get("event") == "tool_call")
            # Pair calls with results in order: tool_result events carry no
            # tool/args of their own (they live on the preceding tool_call).
            pending, paired = [], []
            for e in evs:
                if e.get("event") == "tool_call":
                    pending.append((e.get("tool"), e.get("args")))
                elif e.get("event") == "tool_result" and pending:
                    tool, args = pending.pop(0)
                    paired.append({"tool": tool, "args": args,
                                   "result": str(e.get("result", ""))[:1000]})
            rec["tool_calls"] = paired
        except Exception:
            rec["n_tool_calls"] = 0
            rec["tool_calls"] = []
    return rec
