"""Streamlit front end for PerovAgents.

Shows the flow as it happens instead of hiding it behind one spinner: the
orchestrator's ASK lines, every tool call and its result, then the final
answer. Everything is also written to `results/traces/` so a run can be
re-read afterwards.

    streamlit run app.py
"""
import os
import queue
import sys
import threading
from contextlib import redirect_stdout

import streamlit as st

from perovagents import answer
from perovagents.config import TRACE_DIR
from perovagents.llm import get_llm
from perovagents.team import build_team

st.set_page_config(page_title="PerovAgents", page_icon="🔬", layout="wide")

_DONE = object()


@st.cache_resource
def _team():
    """One LLM + one team for the whole session (cache_resource: not reloaded
    per rerun). Takes no arguments: Streamlit cannot hash an LLM instance."""
    llm = get_llm()
    team, missing = build_team(llm)
    return llm, sorted(team), missing


llm, agents, missing = _team()


@st.cache_resource
def _scaps_note():
    """Why live SCAPS is or is not available, for the sidebar."""
    from perovagents.physics.scaps_run import check_installation
    return check_installation()


with st.sidebar:
    st.header("PerovAgents")
    st.caption(f"orchestrator → {len(agents)} domain agents → plain Python tools")
    st.markdown("**LLM** — " + ("Gemini" if llm.available else "no keys"))
    if llm.available:
        st.caption(f"key pool: {llm.num_keys} keys, rotating")
    st.markdown("**Agents**")
    for a in agents:
        st.markdown(f"- `{a}`")
    if missing:
        st.error("missing: " + ", ".join(missing))
    st.divider()
    if "scaps" in agents or "design" in agents:
        note = _scaps_note()
        if note["available"]:
            st.success("SCAPS-1D: live simulation ready")
        else:
            st.warning("SCAPS-1D: offline")
            st.caption(str(note.get("error"))[:220])
    st.divider()
    st.caption(f"traces → `{TRACE_DIR}`")


class _Tee:
    """stdout proxy: forward to the real stream and collect the transcript."""

    def __init__(self, real, out):
        self.real, self.out, self.buf = real, out, []

    def write(self, s):
        self.real.write(s)
        if s:
            self.buf.append(s)
            if s.endswith("\n"):
                self.flush_buf()

    def flush_buf(self):
        if self.buf:
            self.out.put("".join(self.buf))
            self.buf = []

    def flush(self):
        self.real.flush()


def _run(question):
    """Yield transcript chunks, then a final ('answer', text) tuple."""
    out = queue.Queue()
    result = {}

    def worker():
        tee = _Tee(sys.stdout, out)
        try:
            with redirect_stdout(tee):
                paths = {}
                result["answer"] = answer(question, llm=llm, verbose=True,
                                          trace=True, trace_out=paths)
                result["trace_path"] = paths.get("path", "")
        except Exception as exc:
            result["answer"] = f"**Error:** {type(exc).__name__}: {exc}"
        finally:
            tee.flush_buf()
            out.put(_DONE)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = out.get()
        if item is _DONE:
            break
        yield item
    yield ("answer", result.get("answer", ""), result.get("trace_path", ""))


def _render_chunk(line):
    """Turn one transcript line into something readable."""
    for raw in line.rstrip().splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("== ASK"):
            st.markdown("**" + s + "**")
        elif s.startswith("->"):
            st.code(s, language=None)
        elif s.startswith("[trace]"):
            st.caption(s)
        else:
            st.code(s, language=None)


if "history" not in st.session_state:
    st.session_state.history = []

for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.markdown(a)

typed = st.chat_input("Ask about perovskite solar cells…")
if typed:
    with st.chat_message("user"):
        st.write(typed)
    with st.chat_message("assistant"):
        with st.status("running agents…", expanded=True) as status:
            final, trace_path = "", ""
            for item in _run(typed):
                if isinstance(item, tuple) and item[0] == "answer":
                    final, trace_path = item[1], item[2]
                    break
                _render_chunk(item)
        status.update(label="done", state="complete", expanded=False)
        if final:
            st.markdown(final)
    # Every run is on disk: the JSONL has each ASK, each llm reply, each tool
    # call and result, each report. Show it so a session can be audited later.
    with st.expander("run log", expanded=False):
        st.code(f"trace  {trace_path}\nsession {os.path.join(TRACE_DIR, '..', 'sessions')}")
        if trace_path and os.path.exists(trace_path):
            st.download_button("download trace", open(trace_path).read(),
                               file_name=os.path.basename(trace_path),
                               key=f"dl_{typed[:40]}")
    st.session_state.history.append((typed, final))
