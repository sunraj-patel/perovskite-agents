"""Run ONE benchmark episode in one of two conditions on one backbone.

direct       the existing ``DirectBaseline`` (single LLM + tool catalogue, no
             orchestrator / agent prompts / progress contracts / verifier).
perovagents  the existing ``Orchestrator`` over the five prompted agents with
             their progress contracts (``team.build_team``).

Both conditions

* call the SAME Python tool functions (wrapped by ``ToolRecorder`` so raw
  outputs, arguments, timings and errors are captured identically);
* use the SAME budgets (12 tool calls / 12 model turns / 1200 s wall clock; the
  orchestrated condition additionally has its own 12-turn cap per sub-agent (+6 for
  large tasks) and 5 orchestrator rounds (+2 for multi-part questions) - a documented
  structural asymmetry, see README). v3: the provider wrapper is instrumented and every
  episode carries an evidence-based `validity` (validity.py);
* talk to ONE strict backbone object (providers.EpisodeLLM): no fallback;
* get a fresh LLM object, fresh agents and fresh trace: no memory between
  episodes.

Live SCAPS is disabled for the main benchmark in BOTH conditions (the tools
that would launch the solver are simply not offered), so only precomputed
sweeps / stored artefacts are available. The framework source is not modified:
the two availability probes are overridden here, in the benchmark process only.
"""
import functools
import hashlib
import json
import threading
import time
import traceback
from pathlib import Path

from .paths import setup_env

setup_env()

from benchmarks.tool_layer import (DONE, MAX_LLM_TURNS, MAX_TOOL_CALLS,   # noqa: E402
                                   RUN_TIMEOUT_S)

MAX_RESULT_CHARS = 30000
_RUNTIME = {"patched": False, "lock": threading.Lock()}


def disable_live_scaps():
    """Offline-only SCAPS for the whole process (idempotent)."""
    with _RUNTIME["lock"]:
        if _RUNTIME["patched"]:
            return
        from perovagents.physics import scaps as scaps_mod
        from perovagents.physics import scaps_run
        reason = "live SCAPS disabled for the benchmark (precomputed sweeps only)"

        def live_status(self):
            return {"available": False, "error": reason, "wine": None}

        def check_installation():
            return {"available": False, "error": reason, "wine_version": None}

        scaps_mod.SCAPSTool.live_status = live_status
        scaps_run.check_installation = check_installation
        _RUNTIME["patched"] = True


class ToolRecorder:
    """Per-episode capture of every tool call with its full raw output."""

    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()
        self._t0 = time.time()

    def wrap(self, agent, name, fn):
        rec = self

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t = time.time()
            err, out = False, None
            try:
                out = fn(*args, **kwargs)
                return out
            except Exception as exc:
                err, out = True, f"ERROR: {type(exc).__name__}: {exc}"
                raise
            finally:
                text = str(out)
                row = {"agent": agent, "tool": name,
                       "args": kwargs if kwargs else list(args),
                       "result": text[:MAX_RESULT_CHARS],
                       "result_chars": len(text),
                       "seconds": round(time.time() - t, 3),
                       "t_offset_s": round(t - rec._t0, 2),
                       "error": err or text.startswith("ERROR")}
                with rec._lock:
                    rec.calls.append(row)
        return wrapper


def tool_version_info():
    """Hashes/paths of the data the tools read (no secrets)."""
    import os
    from perovagents import config
    info = {"pdb_sqlite": config.PDB_SQLITE, "corpus_dir": config.CORPUS_DIR,
            "scaps_results": config.SCAPS_EXISTING_RESULTS, "live_scaps": False}
    try:
        manifest = Path(config.RESULTS_DIR) / "ml" / "manifest.json"
        info["ml_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()[:16]
    except Exception:
        pass
    return info


def _trace_events(path):
    try:
        from perovagents.trace import read_trace
        return list(read_trace(path))
    except Exception:
        return []


def run_direct(question, llm, rec, timeout_s=RUN_TIMEOUT_S):
    from benchmarks.harness import DirectBaseline
    from benchmarks.tool_layer import collect_tools
    disable_live_scaps()
    tools, _ = collect_tools()
    wrapped = {n: rec.wrap("direct", n, fn) for n, fn in tools.items()}
    out = DirectBaseline(llm, wrapped, timeout_s=timeout_s).run(question)
    return {"answer": out["answer"], "termination": out["termination"],
            "extra": {"transcript_tail": out.get("transcript_tail")}}


def run_perovagents(question, llm, rec, trace_dir, timeout_s=RUN_TIMEOUT_S):
    from perovagents.agent import Orchestrator
    from perovagents.team import build_team
    from perovagents.trace import Trace
    disable_live_scaps()
    team, missing = build_team(llm)
    if not team:
        raise RuntimeError(f"no agent built: {missing}")
    for agent in team.values():
        agent.tools = {n: rec.wrap(agent.name, n, fn) for n, fn in agent.tools.items()}
    orch = Orchestrator(team, llm, sequential=False)
    trace = Trace(f"[agentic_paper] {question}", trace_dir=str(trace_dir), enabled=True)
    box = {}

    def target():
        try:
            box["answer"] = orch.ask(question, False, trace)
        except Exception as exc:                       # surfaced, never swallowed
            box["error"] = f"{type(exc).__name__}: {exc}"
            box["tb"] = traceback.format_exc(limit=4)

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout_s)
    termination, answer = "answer", box.get("answer", "")
    if th.is_alive():
        termination, answer = "timeout", ""
        llm.abort()          # stop the leaked worker threads from spending quota
    elif "error" in box:
        termination = "harness_error"
    evs = _trace_events(trace.path) if trace.path else []
    return {"answer": answer, "termination": termination,
            "extra": {"trace_file": Path(trace.path).name if trace.path else None,
                      "missing_agents": missing, "error": box.get("error"),
                      "traceback": box.get("tb"),
                      "n_llm_retry_events": sum(1 for e in evs if e.get("event") == "llm_retry"),
                      "n_asks": sum(1 for e in evs if e.get("event") == "ask"),
                      "n_format_errors": getattr(orch, "format_errors", 0),
                      "escalations": sum(1 for e in evs if e.get("event") == "report"
                                         and e.get("progress_status") in
                                         ("BLOCKED", "NO_PROGRESS", "INVALID", "INCOMPLETE"))}}


def run_episode(q, backbone, condition, rep, out_dir):
    """Execute and persist one episode. Returns the episode record."""
    from .providers_instrumented import InstrumentedEpisodeLLM
    from .validity import classify_episode
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = ToolRecorder()
    llm = InstrumentedEpisodeLLM(backbone)
    started = time.time()
    record = {"episode_id": out_dir.name, "qid": q["id"], "category": q["category"],
              "condition": condition, "backbone": backbone["id"],
              "model": backbone["model"], "route": llm.backend.route, "rep": rep,
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "config_version": "v3",
              "budgets": {"max_tool_calls": MAX_TOOL_CALLS, "max_model_turns": MAX_LLM_TURNS,
                          "timeout_s": RUN_TIMEOUT_S, "agent_extra_turns_large_task": 6,
                          "orchestrator_extra_rounds_multi_part": 2},
              "question": q["question"]}
    try:
        if condition == "direct":
            res = run_direct(q["question"], llm, rec)
        else:
            res = run_perovagents(q["question"], llm, rec, out_dir)
        err = None
    except Exception as exc:
        res = {"answer": "", "termination": "harness_error", "extra": {}}
        err = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=5)
    seconds = round(time.time() - started, 1)
    prov = llm.summary()
    n_tool = len(rec.calls)
    seen, repeats = set(), 0
    for c in rec.calls:
        key = (c["tool"], json.dumps(c["args"], sort_keys=True, default=str).lower())
        repeats += key in seen
        seen.add(key)
    validity, validity_ev = classify_episode(prov, res["termination"], n_tool, res["answer"])
    infra = validity == "infra_invalid"
    record.update({
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": seconds,
        "termination": res["termination"],
        "final_answer": res["answer"],
        "tool_calls": rec.calls,
        "n_tool_calls": n_tool,
        "n_model_calls": prov["model_calls"],
        "repeated_tool_calls": repeats,
        "tool_errors": sum(1 for c in rec.calls if c["error"]),
        "retries": repeats + res["extra"].get("n_llm_retry_events", 0),
        "provider": prov,
        "infra_failure": infra,
        "validity": validity,
        "validity_evidence": validity_ev,
        "harness_error": err or res["extra"].get("error"),
        "extra": res["extra"],
        "tool_versions": tool_version_info(),
    })
    with open(out_dir / "episode.json", "w") as f:
        json.dump(record, f, indent=1, default=str)
    return record
