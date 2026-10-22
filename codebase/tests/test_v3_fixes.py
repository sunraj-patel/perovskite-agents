"""Deterministic tests for the benchmark-v3 framework fixes (no network, no model).

1. pdb progress contract: row rankings vs aggregate statistics (Q04 trace).
2. Orchestrator output grammar: prose is not a final answer (MiniMax trace).
3. Evidence carried through escalations; adaptive budgets for large tasks.
"""
import json

from conftest import ScriptedLLM
from perovagents.agent import (DONE, EXTRA_TURNS, MAX_FORMAT_RETRIES, Orchestrator,
                               parse_orchestrator_reply)
from perovagents.progress import _AGG_FAMILIES, wants_row_ranking
from perovagents.team import pdb_agent

call = lambda tool, **args: json.dumps({"tool": tool, "args": args})          # noqa: E731

Q04_TASK = ("Query the Perovskite Database for n-i-p perovskite solar cells using SnO2 as the Electron "
            "Transport Layer (ETL). Identify the highest-PCE reported devices and return their HTL, "
            "absorber/material stack, PCE, Voc, Jsc, and FF.")
RANK_SQL = ("SELECT Ref_DOI_number, HTL_stack_sequence, JV_default_PCE FROM devices WHERE "
            "Cell_architecture = 'nip' AND ETL_stack_sequence LIKE '%SnO2%' "
            "ORDER BY JV_default_PCE DESC LIMIT 3")


# ------------------------------------------------------------------ contract: regex defect
def test_perovskite_does_not_trigger_the_comparison_family():
    assert not _AGG_FAMILIES["comparison"].search("n-i-p perovskite solar cells")
    assert _AGG_FAMILIES["comparison"].search("n-i-p vs p-i-n")
    assert _AGG_FAMILIES["comparison"].search("n-i-p versus p-i-n")


def test_row_ranking_classifier():
    assert wants_row_ranking(Q04_TASK)
    assert wants_row_ranking("top 5 devices by PCE")
    assert wants_row_ranking("the top n-i-p perovskite solar cells using SnO2 ordered by PCE")
    assert not wants_row_ranking("top 3 HTL materials in n-i-p cells")        # categories need GROUP BY
    assert not wants_row_ranking("most common ETL in n-i-p cells")
    assert not wants_row_ranking("average PCE of n-i-p devices")
    assert not wants_row_ranking("highest PCE devices and how many are there")
    assert not wants_row_ranking("compare n-i-p vs p-i-n PCE")


# ------------------------------------------------------------------ contract: row ranking
def test_row_ranking_succeeds_with_an_ordered_limited_select(fake_db):
    llm = ScriptedLLM([call("sql", query=RANK_SQL), f"{DONE} Top device: 20.0 % (PTAA)."])
    out = pdb_agent(llm).run(Q04_TASK)
    assert out == "Top device: 20.0 % (PTAA)."                  # reported, not escalated
    assert "INCOMPLETE" not in out and "BLOCKED" not in out


def test_row_ranking_second_ordered_select_is_not_demanded_aggregation(fake_db):
    a = pdb_agent(ScriptedLLM([call("sql", query=RANK_SQL), f"{DONE} done"]))
    a.run(Q04_TASK)
    assert a.last_state.requirement("ranking").satisfied
    assert a.last_state.requirement("aggregate") is None


def test_unordered_select_does_not_satisfy_a_row_ranking(fake_db):
    raw = "SELECT HTL_stack_sequence, JV_default_PCE FROM devices WHERE Cell_architecture = 'nip'"
    out = pdb_agent(ScriptedLLM([call("sql", query=raw), f"{DONE} done"])).run(Q04_TASK)
    assert out.startswith("INCOMPLETE: a row ranking was required")
    assert "EVIDENCE GATHERED SO FAR" in out and "Spiro-MeOTAD" in out       # rows not discarded


def test_empty_ranking_result_is_not_success(fake_db):
    sql = ("SELECT HTL_stack_sequence FROM devices WHERE Cell_architecture = 'nip' AND "
           "JV_default_PCE > 99 ORDER BY JV_default_PCE DESC LIMIT 3")
    out = pdb_agent(ScriptedLLM([call("sql", query=sql), f"{DONE} done"])).run(Q04_TASK)
    assert out.startswith("INCOMPLETE")


# ------------------------------------------------------------------ contract: aggregation
def test_grouped_aggregation_is_still_required_for_category_rankings(fake_db):
    raw1 = call("sql", query="SELECT HTL_stack_sequence FROM devices WHERE Cell_architecture = 'nip'")
    raw2 = call("sql", query="SELECT HTL_stack_sequence, JV_default_PCE FROM devices "
                             "WHERE Cell_architecture = 'nip'")
    out = pdb_agent(ScriptedLLM([raw1, raw2, f"{DONE} never"])).run("top 3 HTL materials in n-i-p cells")
    assert "Aggregate the rows" in out and "never" not in out


def test_grouped_aggregation_succeeds_with_group_by(fake_db):
    gq = ("SELECT HTL_stack_sequence, AVG(JV_default_PCE) m FROM devices WHERE Cell_architecture='nip' "
          "GROUP BY 1 ORDER BY 2 DESC LIMIT 3")
    a = pdb_agent(ScriptedLLM([call("sql", query=gq), f"{DONE} PTAA leads."]))
    assert a.run("average PCE per HTL in n-i-p cells") == "PTAA leads."
    assert a.last_state.requirement("aggregate").satisfied


def test_true_comparison_requires_an_aggregate(fake_db):
    a = pdb_agent(ScriptedLLM([call("sql", query="SELECT JV_default_PCE FROM devices"), f"{DONE} x"]))
    out = a.run("compare n-i-p vs p-i-n PCE")
    assert out.startswith("INCOMPLETE: aggregation was required")


def test_incomplete_aggregation_report_carries_the_fetched_rows(fake_db):
    out = pdb_agent(ScriptedLLM([call("sql", query="SELECT JV_default_PCE FROM devices"), f"{DONE} x"])
                    ).run("average PCE of n-i-p cells")
    assert "INCOMPLETE" in out and "EVIDENCE GATHERED SO FAR" in out and "JV_default_PCE" in out


# ------------------------------------------------------------------ orchestrator grammar
def test_prose_dispatch_is_malformed_not_an_answer():
    r = parse_orchestrator_reply("Dispatched the pdb agent for the p-i-n vs n-i-p statistics. "
                                 "Waiting for its report.")
    assert r["kind"] == "malformed"
    assert parse_orchestrator_reply("   ")["kind"] == "empty"
    assert parse_orchestrator_reply(f"{DONE}   ")["kind"] == "malformed"          # empty body


def test_valid_dispatch_and_final_forms():
    r = parse_orchestrator_reply("ASK pdb: mean PCE of SnO2\nASK literature: find a paper on SnO2")
    assert r["kind"] == "ask" and [a for a, _ in r["asks"]] == ["pdb", "literature"]
    assert parse_orchestrator_reply("- ASK ml: predict PCE")["asks"] == [("ml", "predict PCE")]
    assert parse_orchestrator_reply(f"{DONE} It is 15.3 %.", have_reports=True) == \
        {"kind": "final", "asks": [], "answer": "It is 15.3 %."}
    assert parse_orchestrator_reply("final answer: lower-case marker", have_reports=True)["kind"] == "final"


def test_ask_wins_over_final_until_a_report_exists():
    both = f"ASK pdb: get the mean\n{DONE} guess 1.0"
    assert parse_orchestrator_reply(both, have_reports=False)["kind"] == "ask"
    assert parse_orchestrator_reply(both, have_reports=True)["kind"] == "final"


class Stub:
    job = "stub"

    def __init__(self, report):
        self.report, self.tasks = report, []

    def run(self, task, verbose=False):
        self.tasks.append(task)
        return self.report


def test_malformed_reply_is_retried_then_dispatches():
    pdb = Stub("mean PCE is 15.3")
    llm = ScriptedLLM(["Dispatched the pdb agent. Waiting for its report.",
                       "ASK pdb: mean PCE of SnO2 devices", f"{DONE} 15.3 %"])
    out = Orchestrator({"pdb": pdb}, llm).ask("q")
    assert out == "15.3 %" and pdb.tasks == ["mean PCE of SnO2 devices"]
    assert "FORMAT ERROR" in llm.prompts[1] and "FORMAT ERROR" not in llm.prompts[2]


def test_malformed_replies_are_bounded_and_reported_clearly():
    pdb = Stub("never run")
    llm = ScriptedLLM(lambda p: "Dispatched the pdb agent for you.")
    orch = Orchestrator({"pdb": pdb}, llm)
    out = orch.ask("q")
    assert out.startswith("[FORMAT ERROR]") and "Dispatched the pdb agent" in out
    assert len(llm.prompts) == MAX_FORMAT_RETRIES + 1 and pdb.tasks == [] and orch.format_errors == 3


def test_valid_multi_agent_dispatch_runs_all_agents():
    a, b = Stub("report A"), Stub("report B")
    llm = ScriptedLLM(["ASK a: task one\nASK b: task two", f"{DONE} merged"])
    assert Orchestrator({"a": a, "b": b}, llm).ask("q") == "merged"
    assert a.tasks == ["task one"] and b.tasks == ["task two"]


def test_orchestrator_prompt_states_the_output_grammar():
    p = Orchestrator({"a": Stub("x")}, ScriptedLLM([])).system
    assert "OUTPUT GRAMMAR" in p and "ASK <agent>:" in p and "EVIDENCE GATHERED SO FAR" in p


# ------------------------------------------------------------------ adaptive budgets
def test_large_tasks_get_extra_turns_small_tasks_do_not(fake_db):
    def runs(task):
        llm = ScriptedLLM(lambda p: "no json here")             # never a valid move -> burns every turn
        pdb_agent(llm).run(task)
        return len(llm.prompts)
    small = runs("what is up")
    large = runs("x" * 650)
    assert large == small + EXTRA_TURNS


def test_multi_agent_question_extends_the_round_budget():
    stubs = {n: Stub("r") for n in "abc"}
    llm = ScriptedLLM(lambda p: f"{DONE} forced" if "must answer now" in p
                      else "ASK a: one\nASK b: two\nASK c: three")
    seen = []
    orig = llm.chat
    llm.chat = lambda p, t=None: (seen.append(p), orig(p, t))[1]
    assert Orchestrator(stubs, llm, max_rounds=1).ask("q") == "forced"
    assert len(seen) == 4        # base max_rounds=1 -> 2 prompts; +2 rounds once 3 agents were asked
    base = ScriptedLLM(lambda p: f"{DONE} forced" if "must answer now" in p else "ASK a: one")
    assert Orchestrator({"a": Stub("r")}, base, max_rounds=1).ask("q") == "forced"
    assert len(base.prompts) == 2                                              # single-agent: no extension


# ------------------------------------------------------------------ literature contract (smoke finding)
def test_doi_tail_or_word_ending_does_not_create_a_property_requirement():
    from perovagents.progress import LiteratureContract
    c = LiteratureContract()
    task = ("Find the papers with DOIs 10.1038/s41586-021-03285-w and 10.1126/science.abf7652 and "
            "report the device, the prev dev results")
    assert c.new_state(task).requirement("property:w") is None
    assert not any(r.rid.startswith("property:") for r in c.new_state(task).requirements)


def test_real_property_requests_are_still_detected():
    from perovagents.progress import LiteratureContract
    st = LiteratureContract().new_state("Find the SnO2 work function in eV and the electron mobility")
    assert any(r.rid.startswith("property:work_function") for r in st.requirements)
