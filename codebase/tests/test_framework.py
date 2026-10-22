import json
from conftest import ScriptedLLM
from perovagents.agent import DONE, Agent, Orchestrator
from perovagents.team import pdb_agent

Q = "SELECT AVG(JV_default_PCE) AS m, COUNT(*) AS n FROM devices WHERE ETL_stack_sequence LIKE '%SnO2%'"
call = lambda tool, **args: json.dumps({"tool": tool, "args": args})          # noqa: E731


def test_agent_retries_after_an_error_and_stops_on_the_phrase(fake_db):
    llm = ScriptedLLM([call("sql", query="SELECT AVG(PCE) FROM devices"),      # wrong column
                       call("sql", query=Q),
                       f"{DONE} Mean PCE of SnO2 devices is 16.67 % over 3 devices."])
    out = pdb_agent(llm).run("mean PCE with SnO2")
    assert "16.67" in out and len(llm.prompts) == 3
    assert "no such column: PCE" in llm.prompts[1] and "JV_default_PCE" in llm.prompts[1]   # saw the error + hint


def test_agent_prompt_contains_data_dictionary_and_tools(fake_db):
    p = pdb_agent(None).system
    for piece in ("ALL COLUMNS", "JV_default_PCE", "sql(query: str)", "find_columns(keyword: str)", "calc", DONE):
        assert piece in p or piece == "calc" and "_calc" in p


def test_agent_gives_up_after_max_turns(fake_db):
    a = pdb_agent(ScriptedLLM(lambda p: call("sql", query=f"SELECT {len(p)}")))   # new args each turn
    a.max_turns = 3
    assert "stopped after 3 turns" in a.run("x")


def test_identical_repeat_is_blocked_with_repeated_action(fake_db):
    a = pdb_agent(ScriptedLLM([call("sql", query=Q), call("sql", query=Q),
                               call("sql", query=Q), f"{DONE} never reached"]))
    out = a.run("count SnO2 devices")
    assert "REPEATED_ACTION" in out and "same" in out.lower()
    assert "SELECT AVG(JV_default_PCE)" in out        # diagnostic names the loop
    # the identical SQL ran exactly REPEAT_LIMIT times before being blocked
    assert a.max_turns > 3                             # guard fired before the turn cap


def test_second_raw_select_blocked_when_aggregation_due(fake_db):
    # Q02 pilot hole: the first raw SELECT satisfies filter:arch-nip, so the
    # old gain-based counter stayed 0 and raw selects looped forever.
    raw1 = call("sql", query="SELECT HTL_stack_sequence, JV_default_PCE FROM devices "
                             "WHERE Cell_architecture = 'nip'")
    raw2 = call("sql", query="SELECT HTL_stack_sequence, JV_default_PCE FROM devices "
                             "WHERE Cell_architecture = 'nip' AND JV_default_PCE > 10")
    a = pdb_agent(ScriptedLLM([raw1, raw2, f"{DONE} never reached"]))
    out = a.run("top 3 HTL materials in n-i-p cells")
    assert "Aggregate the rows" in out           # guard fired on the 2nd raw fetch
    assert "never reached" not in out


def test_satisfied_near_repeat_nudges_to_final(fake_db):
    # contract met, then a NEAR-repeat aggregate: steer to FINAL, not a stop.
    agg1 = call("sql", query="SELECT HTL_stack_sequence, COUNT(*) FROM devices "
                             "WHERE Cell_architecture = 'nip' GROUP BY 1 ORDER BY 2 DESC LIMIT 10")
    agg2 = call("sql", query="SELECT HTL_stack_sequence, COUNT(*) FROM devices "
                             "WHERE Cell_architecture = 'nip' GROUP BY 1 ORDER BY 2 DESC LIMIT 5")
    a = pdb_agent(ScriptedLLM([agg1, agg2, f"{DONE} Top: Spiro-MeOTAD 2, PTAA 2."]))
    out = a.run("top 3 HTL materials in n-i-p cells")
    assert "Spiro-MeOTAD" in out
    assert "REPEATED_ACTION" not in out and "Aggregate the rows" not in out


def test_repeat_detection_ignores_whitespace_and_case(fake_db):
    q1 = call("sql", query="SELECT  AVG(JV_default_PCE) AS m FROM devices")
    q2 = call("sql", query="select avg(jv_default_pce) as m from devices")
    a = pdb_agent(ScriptedLLM([q1, q2, q2, f"{DONE} x"]))
    out = a.run("mean PCE")
    assert "REPEATED_ACTION" in out


def test_one_retry_after_an_error_does_not_trip_the_guard(fake_db):
    bad = call("sql", query="SELECT nope FROM devices")          # raises -> ERROR result
    a = pdb_agent(ScriptedLLM([bad, call("sql", query=Q), f"{DONE} Mean PCE 16.67 over 3."]))
    out = a.run("mean PCE with SnO2")
    assert "16.67" in out                                        # error retry consumed the allowance
    b = pdb_agent(ScriptedLLM([call("sql", query=Q), call("sql", query=Q),
                               call("sql", query=Q), f"{DONE} x"]))
    assert "REPEATED_ACTION" in b.run("mean PCE")                # but three successes -> blocked


def test_different_queries_are_not_considered_duplicates(fake_db):
    a = pdb_agent(ScriptedLLM([call("sql", query=Q),
                               call("sql", query="SELECT COUNT(*) FROM devices"),
                               f"{DONE} Count is 3."]))
    assert a.run("how many") == "Count is 3."


def test_database_is_read_only(fake_db):
    llm = ScriptedLLM([call("sql", query="DROP TABLE devices"), f"{DONE} done"])
    pdb_agent(llm).run("x")
    assert "readonly" in llm.prompts[1].lower() or "ERROR" in llm.prompts[1]


def test_orchestrator_asks_reads_report_then_finishes(fake_db):
    agent_llm = ScriptedLLM([call("sql", query=Q), f"{DONE} Mean PCE 16.67 over 3 devices."])
    orch_llm = ScriptedLLM([f"ASK pdb: mean PCE of SnO2 devices", f"{DONE} The mean PCE is 16.67 %."])
    out = Orchestrator({"pdb": pdb_agent(agent_llm)}, orch_llm).ask("q")
    assert out == "The mean PCE is 16.67 %." and "Mean PCE 16.67 over 3 devices." in orch_llm.prompts[1]


def test_orchestrator_can_ask_again_and_runs_asks_in_parallel(fake_db):
    class Stub:
        job = "stub"
        def __init__(self, r): self.r = r
        def run(self, task, verbose=False): return self.r
    llm = ScriptedLLM(["ASK a: one\nASK b: two", "ASK a: refined request xyz", f"{DONE} final"])
    orch = Orchestrator({"a": Stub("report A"), "b": Stub("report B")}, llm)
    assert orch.ask("q") == "final"
    assert "report A" in llm.prompts[1] and "report B" in llm.prompts[1] and "refined request xyz" not in llm.prompts[1]


def test_unknown_agent_is_reported_back_not_crashed():
    llm = ScriptedLLM(["ASK wizard: do magic", f"{DONE} could not"])
    assert Orchestrator({}, llm).ask("q") == "could not" and "no agent called" in llm.prompts[1]


def test_orchestrator_is_forced_to_answer_after_max_rounds():
    class Stub:
        job = "s"
        def run(self, task, verbose=False): return "r"
    llm = ScriptedLLM(lambda p: f"{DONE} forced" if "must answer now" in p else "ASK a: again")
    assert Orchestrator({"a": Stub()}, llm, max_rounds=2).ask("q") == "forced"


def test_asks_in_one_turn_really_run_concurrently():
    import time
    class Slow:
        job = "s"
        def run(self, task, verbose=False):
            time.sleep(0.3)
            return "r"
    llm = ScriptedLLM(["ASK a: x\nASK b: y", f"{DONE} ok"])
    t0 = time.time()
    Orchestrator({"a": Slow(), "b": Slow()}, llm).ask("q")
    assert time.time() - t0 < 0.55


def test_scaps_agent_builds_and_reads_a_sweep(tmp_path, monkeypatch):
    import pandas as pd
    from perovagents import config
    from perovagents.team import scaps_agent
    path = tmp_path / "sweep_results.csv"
    pd.DataFrame([("eg", 1.4, 1.05, 24.0, 82.0, 20.6), ("eg", 1.7, 1.25, 19.0, 83.0, 19.7)],
                 columns=["sweep", "param", "voc", "jsc", "ff", "eta"]).to_csv(path, index=False)
    monkeypatch.setattr(config, "SCAPS_EXISTING_RESULTS", str(path))
    llm = ScriptedLLM([call("get_sweep", sweep="pvk_Eg"), f"{DONE} PCE falls from 20.6 to 19.7 %."])
    a = scaps_agent(llm)
    assert "pvk_Eg" in a.system and "2 points, 1.4 to 1.7" in a.system
    assert "20.6" in a.run("band gap effect") and "20.6" in llm.prompts[1]


def test_scaps_thickness_sweep_is_reported_in_nm_not_metres(tmp_path, monkeypatch):
    """sweep_results.csv stores thickness in metres; the tool must advertise nm."""
    import pandas as pd
    from perovagents import config
    from perovagents.physics.scaps import SCAPSTool
    path = tmp_path / "sweep_results.csv"
    pd.DataFrame([("d", 2.0e-7, 1.40, 19.1, 90.6, 24.3),
                  ("d", 1.0e-6, 1.36, 26.1, 90.4, 32.1)],
                 columns=["sweep", "param", "voc", "jsc", "ff", "eta"]).to_csv(path, index=False)
    monkeypatch.setattr(config, "SCAPS_EXISTING_RESULTS", str(path))
    out = SCAPSTool().get_sweep("d")
    assert out["unit"] == "nm"
    assert [r["value"] for r in out["data"]] == [200.0, 1000.0]


def test_literature_agent_searches_a_corpus(tmp_path, monkeypatch):
    from perovagents.literature import corpus as C
    from perovagents.team import literature_agent
    idx = tmp_path / "index.json"
    idx.write_text(json.dumps({"passages": [{"doi": "10.1/p", "title": "T", "section": "R",
                     "text": "A 2D capping layer improved moisture stability."}], "docs": {"10.1/p": {"doi": "10.1/p", "title": "T"}}}))
    orig = C.Corpus.__init__
    monkeypatch.setattr(C.Corpus, "__init__", lambda self: orig(self, index_path=str(idx)))
    llm = ScriptedLLM([call("search", query="capping layer moisture"), f"{DONE} 10.1/p: capping layer improved stability."])
    out = literature_agent(llm).run("does capping help?")
    assert "10.1/p" in out and "capping layer improved moisture" in llm.prompts[1]


# ---- tracing ---------------------------------------------------------------
def test_trace_records_asks_tool_calls_and_answer(tmp_path, monkeypatch):
    from perovagents import config
    from perovagents.agent import Orchestrator
    from perovagents.trace import Trace, read_trace
    monkeypatch.setattr(config, "TRACE_DIR", str(tmp_path))
    agent_llm = ScriptedLLM([call("sql", query=Q), f"{DONE} Mean PCE 16.67 over 3."])
    orch_llm = ScriptedLLM([f"ASK pdb: mean PCE", f"{DONE} The mean PCE is 16.67 %."])
    t = Trace("mean PCE?", enabled=True)
    Orchestrator({"pdb": pdb_agent(agent_llm)}, orch_llm).ask("mean PCE?", trace=t)
    assert t.path and t.path.endswith(".jsonl")
    rows = read_trace(t.path)
    events = [r["event"] for r in rows]
    assert events[0] == "question"
    for e in ("ask", "tool_call", "tool_result", "report", "answer"):
        assert e in events, e
    tool = next(r for r in rows if r["event"] == "tool_call")
    assert tool["tool"] == "sql" and "JV_default_PCE" in tool["args"]
    assert rows[-1]["event"] == "answer" and "16.67" in rows[-1]["text"]


def test_trace_is_off_by_default_and_never_breaks_a_query(tmp_path, monkeypatch):
    from perovagents import config
    from perovagents.trace import Trace
    monkeypatch.setattr(config, "TRACE_DIR", str(tmp_path))
    off = Trace("q", enabled=False)
    off.event("answer", text="x")
    assert off.path is None and not list(tmp_path.iterdir())
    # unwritable directory -> tracing disables itself instead of raising
    bad = Trace("q", trace_dir="/proc/nope/deeper", enabled=True)
    assert bad.enabled is False


def test_trace_logs_the_question_exactly_once(tmp_path, monkeypatch):
    from perovagents import config
    from perovagents.agent import Orchestrator
    from perovagents.trace import Trace, read_trace
    monkeypatch.setattr(config, "TRACE_DIR", str(tmp_path))
    agent_llm = ScriptedLLM([call("sql", query=Q), f"{DONE} 16.67."])
    orch_llm = ScriptedLLM([f"ASK pdb: mean PCE", f"{DONE} 16.67 %."])
    t = Trace("mean PCE?", enabled=True)          # ctor already logs it
    Orchestrator({"pdb": pdb_agent(agent_llm)}, orch_llm).ask("mean PCE?", trace=t)
    rows = read_trace(t.path)
    assert [r["event"] for r in rows].count("question") == 1
