"""Key-pool parsing and simple round-robin rotation (no network).

Gemini-only: `chat()` walks the pool in a circle, each key tried once per
call, first success wins; all-fail returns the `[LLM ERROR]` sentinel.
"""
import os
import tempfile
from unittest import mock

sys_path_inserted = os.path.join(os.path.dirname(__file__), "..")
if sys_path_inserted not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path_inserted)

from perovagents import config
from perovagents.llm import LLM

K0 = "AIzaSyFAKE_TEST_KEY_0000000000000000000"
K1 = "AIzaSyFAKE_TEST_KEY_1111111111111111111"
K2 = "AIzaSyFAKE_TEST_KEY_2222222222222222222"

KEYS_MD = "\n".join([
    "# rotation pool",
    f"GOOGLE_API_KEY={K0}",
    f"- {K1}",
    f"* {K2}",
    K0,                       # duplicate must be dropped
    "  ",
])


def _load_keys_only(md_content):
    with tempfile.NamedTemporaryFile("w", suffix=".md",
                                     delete=False) as f:
        f.write(md_content)
        path = f.name
    try:
        with mock.patch.object(config, "CHATBOT_ENV_FILE", "/nonexistent"), \
             mock.patch.dict(os.environ, {}, clear=True):
            keys = LLM._load_keys(keys_md_file=path)
    finally:
        os.unlink(path)
    return keys


class _Quota(Exception):
    pass


class _FakeGenAI:
    """Stand-in for `genai.Client(api_key=...)` + `.models.generate_content`.

    Class-level `fail_keys` decides which keys raise 429 (RESOURCE_EXHAUSTED);
    every client call is recorded in `attempts`.
    """
    fail_keys = {K0}
    attempts = []

    @classmethod
    def client_factory(cls, api_key=""):
        cls.attempts.append(api_key)
        return cls(key=api_key)

    def __init__(self, key):
        self.key = key

    @property
    def models(self):
        outer = self

        def _gen(unused_self, **kwargs):
            if outer.key in _FakeGenAI.fail_keys:
                raise _Quota("429 RESOURCE_EXHAUSTED")
            return type("R", (), {"text": "ok-text"})()
        return type("M", (), {"generate_content": _gen})()


def test_load_keys_parses_and_dedupes():
    keys = _load_keys_only(KEYS_MD)
    assert keys == [K0, K1, K2]


def _make_llm(keys):
    llm = LLM.__new__(LLM)
    llm.model = "fake-model"
    llm.temperature = 0.0
    llm._keys = keys
    llm._idx = 0
    llm._client, llm._client_key = None, None
    llm.available = bool(keys)
    return llm


def _patched_chat(llm, prompt):
    with mock.patch("google.genai.Client", side_effect=_FakeGenAI.client_factory):
        return llm.chat(prompt)


def test_rotation_skips_dead_key_and_cycles():
    _FakeGenAI.attempts = []
    _FakeGenAI.fail_keys = {K0}
    llm = _make_llm([K0, K1, K2])
    text = _patched_chat(llm, "hello")
    assert text == "ok-text"
    assert _FakeGenAI.attempts[0] == K0          # dead key tried first this walk
    assert K1 in _FakeGenAI.attempts             # rotation then found life
    assert llm._idx == (llm._keys.index(K1) + 1) % 3   # balanced: cursor past K1
    # next call starts where rotation stopped (simple circle)
    _FakeGenAI.attempts = []
    _patched_chat(llm, "again")
    assert _FakeGenAI.attempts[0] == K2


def test_all_dead_keys_reports_sentinel_and_tried_each_once():
    _FakeGenAI.attempts = []
    _FakeGenAI.fail_keys = {K0, K1, K2}
    llm = _make_llm([K0, K1, K2])
    text = _patched_chat(llm, "hello")
    assert text.startswith("[LLM ERROR]")
    assert sorted(_FakeGenAI.attempts) == sorted([K0, K1, K2])
    # next call walks the circle again (no cooldowns, always retried)
    _FakeGenAI.attempts = []
    _patched_chat(llm, "retry")
    assert sorted(_FakeGenAI.attempts) == sorted([K0, K1, K2])


def test_no_keys_returns_error_sentinel():
    llm = _make_llm([])
    text = llm.chat("hi")
    assert text.startswith("[LLM ERROR]")
    assert not llm.available
    assert not llm.usable


def test_usable_tracks_available():
    llm = _make_llm([K0])
    assert llm.usable and llm.available