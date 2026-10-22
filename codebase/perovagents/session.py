"""Interactive test session: ask many queries, keep every run on disk.

    python -m perovagents --session

Each question produces two artefacts:

  results/traces/trace_<stamp>.jsonl      structured: question, every ASK, every
                                         llm_reply, every tool_call / tool_result,
                                         every report, the final answer
  results/sessions/session_<stamp>.log    the same run as a readable transcript,
                                         with the per-question summary

The REPL keeps one LLM/agent set alive for the whole session, so a second
question does not pay the team-build cost again. Traces are always on: a
session that is not logged is not evidence of anything.

Commands (start a line with /):
  /agent <name>     send the next question straight to one agent
  /agents           list agents, and which is pinned
  /history          show the questions asked so far
  /trace <n>        print the JSONL trace of question n
  /last             paths written by the most recent question
  /help             this list
  /quit             leave
Anything else is a question, run through the orchestrator.
"""
import os
import sys
import time

BANNER = """\
perovagents test session. Type a question and press enter. /help for commands.
Every question is logged; paths are printed when the answer comes back."""


def _session_dir():
    from . import config
    d = getattr(config, "SESSION_DIR", None) or os.path.join(config.TRACE_DIR, "..", "sessions")
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    return d


def _stamp():
    return time.strftime("%Y%m%d_%H%M%S")


class Tee:
    """stdout proxy: show the transcript live and keep a copy for the session log."""

    def __init__(self, real, sink):
        self.real, self.sink, self.buf = real, sink, []

    def write(self, s):
        self.real.write(s)
        if s:
            self.sink.write(s)
            self.buf.append(s)
            if s.endswith("\n"):
                self.flush_buf()

    def flush_buf(self):
        if self.buf:
            self.buf = []

    def flush(self):
        self.real.flush()


def run_session(only_agent=None, llm=None):
    """Read questions from stdin until EOF or /quit. Returns the number asked."""
    from . import config
    from .agent import Orchestrator
    from .llm import get_llm
    from .team import build_team
    from .trace import Trace, read_trace

    if llm is None:
        llm = get_llm()
        if not llm.usable:
            raise RuntimeError("No LLM reachable: set GOOGLE_API_KEY or start `opencode serve`.")
    t0 = time.time()
    print("building agents ...", flush=True)
    team, missing = build_team(llm)
    print(f"ready in {time.time() - t0:.1f}s: {', '.join(sorted(team))}", flush=True)
    if missing:
        print(f"not built: {missing}", flush=True)

    log_path = os.path.join(_session_dir(), f"session_{_stamp()}.log")
    asked, pinned, index = 0, only_agent, []

    with open(log_path, "w") as sink:
        sink.write(f"# perovagents session {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        sink.write(f"# agents: {', '.join(sorted(team))}\n")
        sink.write(f"# trace dir: {config.TRACE_DIR}\n\n")

        def head(question, agent):
            bar = "=" * 78
            for stream in (sys.stdout, sink):
                stream.write(f"\n{bar}\nQ{asked + 1} [{agent or 'orchestrator'}]: {question}\n{bar}\n")
            sys.stdout.flush()

        while True:
            try:
                raw = input("perovagents> ").strip()
            except EOFError:
                print()
                break
            if not raw:
                continue
            if raw.startswith("/"):
                cmd, _, rest = raw[1:].partition(" ")
                cmd, rest = cmd.lower(), rest.strip()
                if cmd in ("quit", "exit", "q"):
                    break
                if cmd == "help":
                    print(__doc__.split("Commands", 1)[1])
                    continue
                if cmd == "agents":
                    print("agents : " + ", ".join(sorted(team)))
                    print("pinned : " + (pinned or "(none, orchestrator routes)"))
                    continue
                if cmd == "agent":
                    if rest and rest not in team:
                        print(f"no agent called {rest!r}. Available: {', '.join(sorted(team))}")
                        continue
                    pinned = rest or None
                    print("pinned : " + (pinned or "(none, orchestrator routes)"))
                    continue
                if cmd == "history":
                    for n, q in index:
                        print(f"  Q{n}: {q}")
                    continue
                if cmd == "last":
                    if index:
                        print(f"Q{index[-1][0]} trace: {index[-1][2]}")
                    print(f"session log : {log_path}")
                    continue
                if cmd == "trace":
                    if not index:
                        print("nothing asked yet")
                        continue
                    row = next((r for r in index if r[0] == asked), index[-1])
                    if rest.isdigit():
                        row = next((r for r in index if r[0] == int(rest)), index[-1])
                    for r in read_trace(row[2]):
                        print("   ", r)
                    continue
                print("unknown command; /help for the list")
                continue

            asked += 1
            head(raw, pinned)
            trace = Trace(raw, enabled=True)
            t = time.time()
            try:
                tee = Tee(sys.stdout, sink)
                old = sys.stdout
                sys.stdout = tee
                try:
                    if pinned:
                        answer = team[pinned].run(raw, verbose=True, trace=trace)
                    else:
                        answer = Orchestrator(team, llm).ask(raw, verbose=True, trace=trace)
                finally:
                    sys.stdout = old
            except Exception as exc:
                trace.event("error", error=f"{type(exc).__name__}: {exc}")
                answer = f"**Error:** {type(exc).__name__}: {exc}"
            took = time.time() - t
            trace.event("timing", seconds=round(took, 1))
            index.append((asked, raw, trace.path))
            sink.write(f"\n--- answer ({took:.1f}s) ---\n{answer}\n")
            print(f"\n{answer}\n")
            print(f"[Q{asked}] {took:.1f}s | trace {trace.path}\n"
                  f"       session {log_path}")
            sink.write(f"[Q{asked}] {took:.1f}s | trace {trace.path}\n")

    print(f"\n{asked} questions. session log: {log_path}")
    print(f"per-question traces: {config.TRACE_DIR}")
    return asked
