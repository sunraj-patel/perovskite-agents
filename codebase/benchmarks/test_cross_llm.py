"""Unit tests for the cross-LLM benchmark. No LLM calls, no network."""
import json
import os

from benchmarks.cross_llm_questions import (CATEGORIES, PILOT, QUESTIONS,
                                            RECOVERY_SENSITIVE, by_category,
                                            get_question)
from benchmarks.harness import DirectBaseline, TokenRecorder
from benchmarks.providers import (ClaudeProvider, GPTProvider, ProviderError,
                                  get_provider)
from benchmarks.scoring import (completion, declares_failure,
                                doi_cited_as_evidence, failure_class,
                                numeric_hit, paired_stats, recovery_record,
                                repeated_action_rate, score_run, tool_f1)

GOLD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cross_llm_gold.json")


def test_ids_stable_and_complete():
    assert [q["id"] for q in QUESTIONS] == [f"Q{i:02d}" for i in range(1, 21)]
    assert {q["category"] for q in QUESTIONS} == set(CATEGORIES)
    for q in QUESTIONS:
        for k in ("id", "category", "question", "difficulty",
                  "expected_task_type", "expected_tools", "expected_actions",
                  "required_outputs", "forbidden_substitutions",
                  "recovery_required"):
            assert k in q, (q["id"], k)
    assert RECOVERY_SENSITIVE == {"Q02", "Q06", "Q08", "Q15", "Q20"}
    assert PILOT == ["Q01", "Q02", "Q06", "Q10", "Q13", "Q17", "Q20"]
    assert len(by_category("pdb")) == 4
    try:
        get_question("Q99")
        raise AssertionError("should raise")
    except KeyError:
        pass


def test_gold_loads_with_trusted_values():
    g = json.load(open(GOLD_PATH))
    assert g["pdb"]["Q01"]["count"] == 4300
    assert g["scaps"]["Q13"]["converged"] is True
    for qid in ("Q05", "Q06", "Q07", "Q08"):
        assert g["literature"][qid]["verified"] is True
    assert g["ml"]["Q09"]["predicted_pce"] > 0


def test_tool_f1():
    got = tool_f1(["pdb.sql"], [{"tool": "sql", "args": {}}])
    assert got["f1"] == 1.0
    got = tool_f1(["ml.explain_global"], [{"tool": "predict", "args": {}}])
    assert got["f1"] == 0.0 and got["expected"] == ["ml.explain_global"]
    assert got["used"] == ["ml.predict"]


def test_numeric_hit():
    assert numeric_hit("mean PCE 15.3 over 4300 devices", 15.303, rel_tol=0.05)
    assert not numeric_hit("mean PCE 12.0", 15.303, rel_tol=0.05)
    assert numeric_hit("n=4300", 4300, rel_tol=0.02)
    assert numeric_hit("anything", None) is None


def test_repeated_action_rate():
    calls = [{"tool": "sql", "args": {"query": "SELECT 1"}},
             {"tool": "sql", "args": {"query": "select 1"}},
             {"tool": "sql", "args": {"query": "SELECT 2"}}]
    assert repeated_action_rate(calls) == round(1 / 3, 3)
    assert repeated_action_rate([]) == 0.0


def test_failure_taxonomy():
    base = {"termination": "answer", "final_answer": "x", "tool_calls": []}
    assert failure_class(dict(base, termination="timeout"), {"score": 0},
                         {"recall": 1}, 0.0) == "no_progress"
    rec = dict(base, tool_calls=[{"tool": "sql", "args": {}}])
    assert failure_class(rec, {"score": 0}, {"recall": 0}, 0.0) == "wrong_tool_selection"
    rep = dict(base, tool_calls=[{"tool": "sql", "args": {"q": "a"}}] * 3)
    assert failure_class(rep, {"score": 1}, {"recall": 1}, 0.8) == "repeated_action"
    assert failure_class(dict(base, termination="system_unavailable"),
                         {"score": None}, {"recall": 0}, 0.0) == "system_unavailable"


def test_recovery_record():
    meta = {"recovery_required": True}
    rec = {"tool_calls": [{"tool": "s", "args": {},
                           "result": "BLOCKED: reread rejected"}],
           "final_answer": ""}
    r = recovery_record(meta, rec, {"score": 0})
    assert r["initial_failure_seen"] and not r["recovered"]
    r = recovery_record(meta, rec, {"score": 1.0})
    assert r["recovered"] and r["final_correct"]


def test_paired_stats_counts():
    a = [{"qid": "Q01", "A_completion": 1}, {"qid": "Q02", "A_completion": 0}]
    b = [{"qid": "Q01", "A_completion": 0}, {"qid": "Q02", "A_completion": 0}]
    p = paired_stats(a, b)
    assert (p["wins"], p["losses"], p["ties"]) == (1, 0, 1)
    assert p["win_rate_ci95"][0] <= p["win_rate"] <= p["win_rate_ci95"][1]


def test_providers():
    for cls in (ClaudeProvider, GPTProvider):
        assert not cls().healthy()
        try:
            cls().chat("hi")
            raise AssertionError("stub should raise")
        except ProviderError:
            pass
    try:
        get_provider("nope")
        raise AssertionError("should raise")
    except KeyError:
        pass


class FakeProvider:
    name, model = "fake", "fake-1"

    def __init__(self, script):
        self.script = list(script)

    def chat(self, prompt, temperature=None):
        return self.script.pop(0) if self.script else "FINAL ANSWER: done"

    def healthy(self):
        return True


def test_direct_baseline_answers_and_budgets():
    tools = {"echo": lambda x="": f"heard {x}"}
    fp = FakeProvider(['{"tool": "echo", "args": {"x": "hi"}}',
                       "FINAL ANSWER: heard hi"])
    rec = DirectBaseline(fp, tools, max_tool_calls=1).run("q?")
    assert rec["termination"] == "answer" and rec["answer"] == "heard hi"
    assert rec["n_tool_calls"] == 1 and rec["tokens"]["llm_calls"] == 2
    fp2 = FakeProvider(['{"tool": "echo", "args": {}}'] * 5)
    rec2 = DirectBaseline(fp2, tools, max_tool_calls=1).run("q?")
    assert rec2["termination"] == "tool_budget"
    fp3 = FakeProvider(['{"tool": "nope", "args": {}}', "FINAL ANSWER: gave up"])
    rec3 = DirectBaseline(fp3, tools).run("q?")
    assert rec3["termination"] == "answer" and rec3["answer"] == "gave up"


def test_recorder_real_usage():
    class P:
        last_usage = {"prompt_tokens": 10, "completion_tokens": 5,
                      "total_tokens": 15}

        def chat(self, prompt, temperature=None):
            self.last_usage = dict(self.last_usage)
            return "ok"

    r = TokenRecorder(P())
    r.chat("hello world, this is a prompt")
    s = r.summary()
    assert s["real_usage"] and s["real_prompt_tokens"] == 10
    assert s["prompt_tokens_est"] == len("hello world, this is a prompt") // 4


def test_tool_layer_collects_real_tools():
    from benchmarks.tool_layer import catalog_text, collect_tools
    tools, agents = collect_tools()
    # run_point/sweep are live-conditional by design (withheld when the
    # SCAPS probe fails), so only always-available tools are asserted.
    assert {"sql", "search", "read_paper", "predict", "explain_global",
            "design_stack"}.issubset(set(tools))
    assert "Predict PCE" in catalog_text({"predict": tools["predict"]})


def test_failure_declaration_caps_scoring():
    assert declares_failure("The ranking could not be established.")
    assert declares_failure("No qualifying paper can be reported yet.")
    assert not declares_failure("The first run failed, then converged to PCE 31.98.")
    # trailing gaps section does NOT count as a lead declaration
    delivered = ("N = 4299 devices, mean PCE 15.29. " + "x" * 400 +
                 " What could NOT be established: std dev.")
    assert not declares_failure(delivered)
    assert declares_failure(delivered, lead=False)
    g = json.load(open(GOLD_PATH))
    meta = get_question("Q02")
    rec = {"system": "mrs", "termination": "answer",
           "final_answer": ("The requested ranking could not be established. "
                            "Spiro-MeOTAD and PTAA were mentioned but no counts "
                            "for the top 10 over 20 percent."),
           "tool_calls": [{"tool": "sql", "args": {}, "result": "raw rows"}],
           "n_llm_calls": 5, "prompt_tokens": 1, "completion_tokens": 1,
           "estimated_cost_usd": None, "seconds": 1.0}
    s = score_run(meta, g, rec)
    assert s["D_scientific"] == 0.0 and s["A_completion"] == 0
    assert s["B_contract"] == 0 and s["failure"] == "incomplete_synthesis"
    assert not s["F_recovered"] and not s["F_detail"]["final_correct"]


def test_cited_and_rejected_doi_does_not_pass():
    doi = "10.1002/aenm.201803587"
    assert doi_cited_as_evidence(
        "Smith and colleagues report 21.2 percent in " + doi + " with FACl.", doi)
    assert not doi_cited_as_evidence(
        "Candidate DOI " + doi + " was examined but rejected as it did not "
        "fulfil the architecture check.", doi)
    assert not doi_cited_as_evidence("no paper found", doi)
    assert not doi_cited_as_evidence("x", "")


def test_score_reproducible_and_shaped():
    g = json.load(open(GOLD_PATH))
    meta = get_question("Q01")
    rec = {"system": "gemini", "termination": "answer",
           "final_answer": "mean PCE 15.3 over 4300 devices",
           "tool_calls": [{"tool": "sql", "args": {"query": "SELECT COUNT(*), AVG"},
                           "result": "4300, 15.3"}],
           "n_llm_calls": 3, "prompt_tokens": 9000, "completion_tokens": 90,
           "estimated_cost_usd": None, "seconds": 10.0}
    s1, s2 = score_run(meta, g, rec), score_run(meta, g, rec)
    assert s1 == s2
    assert s1["A_completion"] == 1 and s1["D_scientific"] == 1.0
    assert s1["C_tool_f1"] == 1.0 and s1["failure"] == "none"
    assert completion(meta, {"score": 0.0}, rec) == 0
