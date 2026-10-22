"""Deterministic tests for provider instrumentation, episode validity and the retry policy.
No network, no model, no API key.
"""
import json
from types import SimpleNamespace as NS

import pytest

from benchmarks.agentic_paper import run as runner
from benchmarks.agentic_paper.providers_instrumented import (InstrumentedEpisodeLLM, diagnose_empty,
                                                             redact, response_detail)
from benchmarks.agentic_paper.validity import classify_episode, should_retry


def resp(text="", finish="STOP", block=None, parts=None, usage=(10, 0, 0), cands=1):
    um = NS(prompt_token_count=usage[0], candidates_token_count=usage[1], thoughts_token_count=usage[2],
            total_token_count=sum(usage))
    content = NS(parts=parts if parts is not None else ([NS(text=text, thought=None, function_call=None)]
                                                        if text else []))
    c = NS(finish_reason=NS(name=finish), finish_message=None, content=content)
    return NS(response_id="r-1", model_version="m-1", usage_metadata=um, text=text,
              prompt_feedback=NS(block_reason=NS(name=block)) if block else None,
              candidates=[c] * cands)


# ------------------------------------------------------------------ secrets
def test_redact_strips_credentials():
    msg = ("403 key=AIzaSyA1234567890abcdefghijklmnopqrstuv Authorization: Bearer abcdefghijklmnop1234 "
           "api_key: sk-abcdefghijklmnopqrstu")
    out = redact(msg, 500)
    assert "AIza" not in out and "abcdefghijklmnop1234" not in out and "sk-abcdef" not in out
    assert "[REDACTED]" in out


def test_response_detail_has_no_secret_fields_and_captures_metadata():
    d = response_detail(resp("hello", usage=(12, 3, 5)))
    assert d["finish_reason"] == "STOP" and d["response_id"] == "r-1" and d["model_version"] == "m-1"
    assert d["usage"] == {"prompt_tokens": 12, "completion_tokens": 3, "thinking_tokens": 5,
                          "total_tokens": 20}
    assert d["part_kinds"] == ["text"] and d["n_candidates"] == 1 and d["block_reason"] is None
    assert "key" not in json.dumps(d).lower().replace("response_id", "")


# ------------------------------------------------------------------ empty-reply diagnosis
@pytest.mark.parametrize("kwargs,expected", [
    (dict(finish="SAFETY"), "provider_filtered"),
    (dict(block="PROHIBITED_CONTENT", finish="STOP"), "provider_filtered"),
    (dict(finish="MALFORMED_FUNCTION_CALL"), "malformed_function_call"),
    (dict(finish="OTHER"), "provider_anomaly"),
    (dict(finish="MAX_TOKENS", usage=(10, 0, 900)), "token_limit_empty"),
    (dict(finish="STOP", usage=(10, 0, 40)), "thinking_only_stop"),
    (dict(finish="STOP", usage=(10, 0, 0)), "empty_stop"),
    (dict(finish="STOP", cands=0), "provider_no_candidate"),
    (dict(finish="STOP", parts=[NS(text=None, thought=None, function_call={"name": "sql"})]),
     "function_call_only"),
    (dict(finish="WEIRD_NEW_VALUE"), "unclassified_empty"),
])
def test_diagnose_empty_from_evidence(kwargs, expected):
    assert diagnose_empty(response_detail(resp(**kwargs))) == expected


def test_diagnose_empty_without_evidence_is_unclassified_not_infra():
    assert diagnose_empty(None) == "unclassified_empty"
    assert diagnose_empty({"status": "exhausted"}) == "provider_exception"


# ------------------------------------------------------------------ per-call log via the wrapper
class StubBackend:
    route = "stub"

    def __init__(self, replies):
        import threading
        self.replies, self._tl = list(replies), threading.local()

    def complete(self, prompt, temperature, events):
        text, detail = self.replies.pop(0)
        self._tl.detail = detail
        return text, None


def make_llm(replies):
    llm = InstrumentedEpisodeLLM({"id": "x", "kind": "opencode", "model": "opencode/x"})
    llm.backbone = {"kind": "gemini"}
    llm.backend = StubBackend(replies)
    return llm


def test_wrapper_logs_every_call_and_classifies_empties():
    ok = response_detail(resp("fine"))
    ok["text_len"] = 4
    ok["raw_text"] = "fine"
    empty = response_detail(resp(finish="SAFETY"))
    llm = make_llm([("fine", ok), ("", empty), ("", None)])
    assert llm.chat("p1") == "fine" and llm.chat("p2") == "" and llm.chat("p3") == ""
    s = llm.summary()
    assert [c["empty"] for c in s["call_log"]] == [False, True, True]
    assert s["empty_reply_classes"] == {"provider_filtered": 1, "unclassified_empty": 1}
    assert s["call_log"][0]["detail"]["raw_text"] == "fine"            # non-empty reply kept
    assert s["call_log"][1]["detail"]["finish_reason"] == "SAFETY"
    blob = json.dumps(s["call_log"])
    assert "p1" not in blob and "p2" not in blob                       # prompt text is never stored
    assert all("prompt_chars" in c for c in s["call_log"])             # only its length


# ------------------------------------------------------------------ episode validity
def prov(classes, n_calls, infra=False, wait=0):
    return {"call_log": [{}] * n_calls, "empty_reply_classes": classes, "infra_failed": infra,
            "infra_reason": "x" if infra else None, "pool_wait_s": wait, "provider_failure_calls": 0}


def test_validity_classes():
    assert classify_episode(prov({}, 5, infra=True), "answer", 0, "")[0] == "infra_invalid"
    assert classify_episode(prov({}, 5, wait=400), "timeout", 0, "")[0] == "infra_invalid"
    assert classify_episode(prov({"provider_filtered": 9}, 9), "timeout", 0, "")[0] == "infra_invalid"
    assert classify_episode(prov({"unclassified_empty": 9}, 9), "timeout", 0, "")[0] == "unclassified_empty"
    assert classify_episode(prov({"empty_stop": 9}, 9), "timeout", 0, "")[0] == "valid_empty_completion"
    # mixed causes are neither confirmed infrastructure nor purely model behaviour -> conservative label
    mixed = prov({"provider_filtered": 5, "empty_stop": 4}, 9)
    assert classify_episode(mixed, "timeout", 0, "")[0] == "unclassified_empty"
    assert classify_episode(prov({"unclassified_empty": 2, "empty_stop": 7}, 9), "timeout", 0, "")[0] \
        == "unclassified_empty"


def test_ordinary_failures_and_partial_empties_stay_valid():
    assert classify_episode(prov({}, 6), "answer", 2, "wrong number 3.0")[0] == "valid"
    assert classify_episode(prov({"provider_filtered": 1}, 6), "answer", 1, "an answer")[0] == "valid"
    assert classify_episode(prov({"provider_filtered": 9}, 9), "answer", 1, "")[0] == "valid"   # a tool ran
    assert classify_episode(prov({"provider_filtered": 2}, 2), "answer", 0, "")[0] == "valid"   # < MIN_CALLS


# ------------------------------------------------------------------ retry policy
def test_retry_policy():
    assert should_retry("infra_invalid", 1, 3) and should_retry("infra_invalid", 2, 3)
    assert not should_retry("infra_invalid", 3, 3)
    for v in ("valid", "valid_empty_completion"):
        assert not should_retry(v, 1, 3)
    assert not should_retry("unclassified_empty", 1, 3)                  # never blind
    assert should_retry("unclassified_empty", 1, 3, retry_unclassified=True)
    assert not should_retry("unclassified_empty", 2, 3, retry_unclassified=True)   # at most one retry


def test_needs_run_uses_latest_attempt():
    st = {"done": False, "attempts": 1, "last_validity": "infra_invalid"}
    assert runner.needs_run(None, 3, False) and runner.needs_run(st, 3, False)
    assert not runner.needs_run({**st, "attempts": 3}, 3, False)
    assert not runner.needs_run({"done": True, "attempts": 1, "last_validity": "valid"}, 3, False)
    done_unc = {"done": True, "attempts": 1, "last_validity": "unclassified_empty"}
    assert not runner.needs_run(done_unc, 3, False) and runner.needs_run(done_unc, 3, True)


# ------------------------------------------------------------------ manifest linking
def test_replacement_attempts_are_linked_and_originals_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path / "runs")
    outcomes = iter([("infra_invalid", True), ("valid", False)])

    def fake_run_episode(q, backbone, condition, rep, out_dir):
        v, infra = next(outcomes)
        rec = {"episode_id": out_dir.name, "termination": "answer", "seconds": 1.0, "infra_failure": infra,
               "validity": v, "final_answer": f"attempt-{v}"}
        out_dir.mkdir(parents=True, exist_ok=True)
        json.dump(rec, open(out_dir / "episode.json", "w"))
        return rec

    from benchmarks.agentic_paper import episode
    monkeypatch.setattr(episode, "run_episode", fake_run_episode)
    import threading
    lock = threading.Lock()
    bank = {"Q01": {"category": "pdb"}}
    cfg = {"route": "r", "model": "m"}
    cell = ("Q01", "gemini", "direct", 0)
    r1 = runner.run_cell(cell, bank, cfg, "main", lock)
    r2 = runner.run_cell(cell, bank, cfg, "main", lock)
    assert (r1["status"], r1["attempt"], r1["replaces"]) == ("infra_failed", 1, None)
    assert (r2["status"], r2["attempt"], r2["replaces"]) == ("completed", 2, r1["attempt_path"])
    root = tmp_path / "runs" / "episodes" / "Q01__gemini__direct__r0"
    assert json.load(open(root / "episode.attempt1.json"))["final_answer"] == "attempt-infra_invalid"
    assert json.load(open(root / "episode.attempt2.json"))["final_answer"] == "attempt-valid"
    st = runner.cell_state(runner.read_manifest("main"))["Q01__gemini__direct__r0"]
    assert st["done"] and st["attempts"] == 2 and not runner.needs_run(st, 3, False)


def test_replies_are_examined_before_a_later_quota_failure():
    """Smoke finding: 10 malformed (empty) replies, then 429s exhausted the keys."""
    log = [{"provider_failure": False}] * 10 + [{"provider_failure": True}] * 9
    p = {"call_log": log, "empty_reply_classes": {"malformed_function_call": 10}, "infra_failed": True,
         "infra_reason": "429", "pool_wait_s": 300, "provider_failure_calls": 9}
    v, ev = classify_episode(p, "turn_budget", 0, "")
    assert v == "valid_empty_completion" and "malformed_function_call" in ev["reason"]


def test_pure_provider_failure_without_replies_is_still_infra():
    p = {"call_log": [{"provider_failure": True}] * 4, "empty_reply_classes": {}, "infra_failed": True,
         "infra_reason": "429", "pool_wait_s": 0, "provider_failure_calls": 4}
    assert classify_episode(p, "answer", 0, "")[0] == "infra_invalid"
