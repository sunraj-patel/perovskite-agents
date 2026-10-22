"""Key-pool rotation: dedup, per-error cooldowns, circuit breaker.

`LLM.chat()` builds a `genai.Client` per attempt from `google.genai`, so these
tests stub that module and assert which key each attempt used.
"""
import pytest

import perovagents.llm as L
from perovagents import config


class _Resp:
    def __init__(self, text):
        self.text = text


def _pool(monkeypatch, keys, behaviour, disable_opencode=True):
    """LLM with `keys`, where behaviour maps key -> text or Exception."""
    monkeypatch.setattr(L.LLM, "_load_keys", classmethod(lambda cls, **k: list(keys)))
    llm = L.LLM()
    L.reset_pool_state()
    if disable_opencode:
        monkeypatch.setattr(llm, "_try_opencode", lambda p: None)
    import google.genai as genai

    used = []

    def _client(api_key=None, **_):
        used.append(api_key)
        class C:
            def __init__(self):
                self.models = self
            def generate_content(self, **_kw):
                r = behaviour[api_key]
                if isinstance(r, Exception):
                    raise r
                return _Resp(r)
        return C()

    monkeypatch.setattr(genai, "Client", _client)
    return llm, used


# ---- dedup -----------------------------------------------------------------
def test_duplicate_keys_are_dropped_across_all_sources(tmp_path, monkeypatch):
    md = tmp_path / "keys.md"
    md.write_text("AIzaShared\n- AIzaShared\nAIzaOther\n# c\n\nAIzaThird\n")
    env = tmp_path / "chatbot.env"
    env.write_text('GOOGLE_API_KEY="AIzaShared"\n')
    monkeypatch.setattr(config, "CHATBOT_ENV_FILE", str(env))
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaShared")
    keys = L.LLM._load_keys(keys_md_file=str(md))
    assert keys == ["AIzaShared", "AIzaOther", "AIzaThird"]


def test_rotation_cursor_is_unaffected_by_duplicates(monkeypatch):
    """A duplicated key made `list.index` resolve to slot 0, so the cursor
    jumped backwards and the tail of the pool was never reached. `chat()`
    advances the cursor via _rotate() after a success, so mirror that."""
    keys = ["AIzaA", "AIzaB", "AIzaC"]
    llm, _ = _pool(monkeypatch, keys, {k: "ok" for k in keys})
    seen = []
    for _ in range(6):
        k = llm._pick_key()
        seen.append(k)
        llm._rotate(k)          # what chat() does on success
    assert seen == ["AIzaA", "AIzaB", "AIzaC"] * 2


def test_cursor_advances_past_a_duplicated_key(monkeypatch):
    """index()-based rotation snapped back to a duplicate's first slot and
    pinned the cursor, so the tail of the pool was never reached."""
    keys = ["AIzaA", "AIzaDup", "AIzaDup", "AIzaC"]   # pre-fix pool shape
    llm, _ = _pool(monkeypatch, keys, {k: "ok" for k in keys})
    seen = []
    for _ in range(4):
        k = llm._pick_key()
        seen.append(k)
        llm._rotate(k)
    assert set(seen) == {"AIzaA", "AIzaDup", "AIzaC"}   # all reachable


def test_cooling_key_is_skipped_even_when_duplicated(monkeypatch):
    """Cooldowns key on the key string, so a duplicate shares one entry: if
    that key is cooling, every copy is skipped."""
    keys = ["AIzaA", "AIzaDup", "AIzaDup", "AIzaC"]
    llm, _ = _pool(monkeypatch, keys, {k: "ok" for k in keys})
    llm._bad_until["AIzaDup"] = L.time.time() + 300
    seen = []
    for _ in range(2):
        k = llm._pick_key()
        seen.append(k)
        llm._rotate(k)
    assert seen == ["AIzaA", "AIzaC"]


# ---- per-error cooldowns ---------------------------------------------------
@pytest.mark.parametrize("err,expected,label", [
    ("429 RESOURCE_EXHAUSTED: quota", 60, "quota"),
    ("API key not valid: API_KEY_INVALID", 24 * 3600, "invalid"),
    ("503 UNAVAILABLE: model overloaded", 20, "transient"),
    ("500 INTERNAL", 20, "server"),
])
def test_error_class_sets_the_right_cooldown(monkeypatch, err, expected, label):
    llm, _ = _pool(monkeypatch, ["AIzaA"], {"AIzaA": Exception(err)})
    llm._mark_bad("AIzaA", Exception(err))
    left = llm._bad_until["AIzaA"] - L.time.time()
    assert expected * 0.8 < left <= expected, label


def test_non_key_400_does_not_ban_the_key_for_a_day(monkeypatch):
    """A request-level 400 (bad model id, bad args) used to trigger the 24 h
    invalid-key ban and silently disable a healthy key for the whole day."""
    llm, _ = _pool(monkeypatch, ["AIzaA"], {"AIzaA": "ok"})
    llm._mark_bad("AIzaA", Exception("400 INVALID_ARGUMENT: unknown model 'x'"))
    assert "AIzaA" not in llm._bad_until
    assert llm._pick_key() == "AIzaA"


# ---- failover --------------------------------------------------------------
def test_chat_fails_over_to_the_next_working_key(monkeypatch):
    keys = ["AIzaBad", "AIzaQuota", "AIzaGood"]
    llm, used = _pool(monkeypatch, keys, {
        "AIzaBad": Exception("API key not valid: API_KEY_INVALID"),
        "AIzaQuota": Exception("429 RESOURCE_EXHAUSTED"),
        "AIzaGood": "REAL ANSWER",
    })
    assert llm.chat("q") == "REAL ANSWER"
    assert used == ["AIzaBad", "AIzaQuota", "AIzaGood"]


def test_all_keys_cooling_returns_error_not_a_storm(monkeypatch):
    """Every key 429s: one walk, then the breaker short-circuits. A second
    call must not re-attempt the dead pool."""
    keys = ["AIzaA", "AIzaB", "AIzaC"]
    llm, used = _pool(monkeypatch, keys,
                      {k: Exception("429 RESOURCE_EXHAUSTED") for k in keys})
    assert "LLM ERROR" in llm.chat("q")
    assert len(used) == 3
    used.clear()
    assert "cooling" in llm.chat("q") or "dead" in llm.chat("q")
    assert used == []            # no further network attempts
    assert llm._bad_until       # cooldowns preserved, not cleared


def test_circuit_breaker_rearms_after_60s(monkeypatch):
    keys = ["AIzaA"]
    llm, used = _pool(monkeypatch, keys, {"AIzaA": Exception("429 RESOURCE_EXHAUSTED")})
    llm.chat("q")
    used.clear()
    llm.chat("q")
    assert used == []
    later = L.time.time() + 61
    monkeypatch.setattr(L.time, "time", lambda: later)
    llm.chat("q")
    assert used == ["AIzaA"]     # re-probed once the breaker expires


def test_success_opens_the_breaker_again(monkeypatch):
    keys = ["AIzaA", "AIzaB"]
    llm, used = _pool(monkeypatch, keys, {"AIzaA": "ok", "AIzaB": "ok"})
    assert llm.chat("q") == "ok"
    llm._bad_until["AIzaA"] = L.time.time() + 300
    llm._bad_until["AIzaB"] = L.time.time() + 300
    llm.chat("q")               # arms the breaker
    assert L._POOL_DEAD_UNTIL > L.time.time()
    llm._bad_until.clear()      # keys recover
    assert llm.chat("q") == "ok"
    assert L._POOL_DEAD_UNTIL == 0.0


def test_cooldowns_are_shared_across_instances(monkeypatch):
    """A Streamlit rerun builds a fresh LLM; without shared cooldowns it would
    re-walk all ~16 dead keys on every question."""
    keys = ["AIzaA", "AIzaB"]
    first, _ = _pool(monkeypatch, keys, {k: Exception("429 RESOURCE_EXHAUSTED") for k in keys})
    first.chat("q")
    second = L.LLM()            # new instance, same process
    assert second._bad_until is first._bad_until
    assert len(second._bad_until) == 2


def test_concurrent_calls_do_not_reuse_the_same_key(monkeypatch):
    """The orchestrator drives 4 agents on 4 threads through one LLM."""
    import threading
    keys = ["AIzaA", "AIzaB", "AIzaC", "AIzaD"]
    llm, used = _pool(monkeypatch, keys, {k: "ok" for k in keys})
    llm._pick_key = llm._pick_key  # keep the real (locked) implementation
    barrier = threading.Barrier(4)
    lock = threading.Lock()

    def worker():
        barrier.wait()
        llm._pick_key()
        with lock:
            pass

    ts = [threading.Thread(target=worker) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(set(used)) <= len(keys)
