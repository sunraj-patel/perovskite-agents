"""PerovAgents: an orchestrator and four prompted agents (pdb, literature, ml, scaps)."""
from .agent import Agent, Orchestrator
from .team import build_team


def answer(question, llm=None, verbose=False, trace=False, trace_out=None,
           sequential=False):
    """question -> orchestrator -> agents (loop until FINAL ANSWER) -> answer text.

    `trace=True` writes every tool call and result to `config.TRACE_DIR` and
    prints the file path, so a run can be audited afterwards. Pass a dict as
    `trace_out` to get that path back instead of scraping it from stdout.
    """
    if llm is None:
        from .llm import get_llm
        llm = get_llm()
        if not llm.usable:
            raise RuntimeError("No language model reachable: set GOOGLE_API_KEY or start `opencode serve`.")
    team, missing = build_team(llm)
    if not team:
        raise RuntimeError(f"No agent could be built: {missing}")
    from .trace import Trace
    t = Trace(question, enabled=bool(trace)) if trace else None
    if trace_out is not None:
        trace_out["path"] = t.path if t else ""
    try:
        return Orchestrator(team, llm, sequential=sequential).ask(question, verbose, trace=t)
    finally:
        if t and t.path and verbose:
            print(f"\n[trace] {t.path}")
