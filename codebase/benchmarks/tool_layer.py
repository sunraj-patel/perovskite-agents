"""The shared scientific tool layer both systems use.

Fairness rule: MRS agents and the direct-LLM baselines call the SAME
Python functions against the SAME database, corpus, ML artifacts and
SCAPS installation. The only difference under test is orchestration.

Tools are collected from the real MRS agent builders (built with llm=None,
so no model is touched) - not reimplemented. Descriptions come from the
functions' own signatures/docstrings, i.e. the exact catalogue text an
MRS agent sees.
"""
import inspect

# Fixed budgets, identical for every system (spec section 7/14).
MAX_TOOL_CALLS = 12          # mirrors Agent.max_turns
MAX_LLM_TURNS = 12
RUN_TIMEOUT_S = 20 * 60

DONE = "FINAL ANSWER:"


def collect_tools():
    """{tool_name: callable} across pdb/literature/ml/scaps/design agents.

    Built once per process and reused: the functions are stateless w.r.t.
    the benchmark (read-only DB/corpus, frozen ML artifacts), and every
    agent builder creates fresh closures, so re-collecting would only
    churn model handles.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    from perovagents import team
    tools, agents = {}, {}
    for builder in (team.pdb_agent, team.literature_agent, team.ml_agent,
                    team.scaps_agent, team.design_agent):
        try:
            agent = builder(None)
        except Exception as exc:          # e.g. empty corpus, SCAPS missing
            raise RuntimeError(f"benchmark tool layer: {builder.__name__} "
                               f"failed with llm=None: {exc}")
        agents[agent.name] = agent
        for fname, fn in agent.tools.items():
            if fname in tools and tools[fname] is not fn:
                raise RuntimeError(f"tool name collision: {fname}")
            tools[fname] = fn
    _CACHE = (tools, agents)
    return _CACHE


_CACHE = None


def catalog_text(tools) -> str:
    """One `- name(sig): doc` line per tool (same rendering as Agent)."""
    return "\n".join(
        f"- {name}{inspect.signature(fn)}: {inspect.getdoc(fn)}"
        for name, fn in tools.items())


def describe_tool_versions() -> dict:
    """Versions/paths of everything a tool call can depend on (spec 15)."""
    from perovagents import config
    from perovagents.ml.predictor import MLPredictorTool
    info = {"pdb_sqlite": config.PDB_SQLITE, "corpus_dir": config.CORPUS_DIR}
    try:
        import json, os
        model = MLPredictorTool()
        manifest = os.path.join(model.dir, "manifest.json")
        info["ml_artifact_dir"] = model.dir
        if os.path.exists(manifest):
            info["ml_manifest"] = json.load(open(manifest))
    except Exception as exc:
        info["ml_error"] = str(exc)[:200]
    try:
        from perovagents.physics.scaps import SCAPSTool
        live = SCAPSTool().live_status()
        info["scaps_live"] = bool(live.get("available"))
    except Exception as exc:
        info["scaps_error"] = str(exc)[:200]
    return info
