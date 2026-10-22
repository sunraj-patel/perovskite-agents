"""LLM interface for PerovAgents.

Primary backend: Google Gemini via the `google-genai` SDK. Keys are loaded from
(a) `GOOGLE_API_KEY` env, (b) `chatbot/.env`, (c) `~/keys.md` (one key per
line; used for rate-limit rotation).

Keys auto-rotate on every call (balanced round-robin): after a success the
cursor advances to the next healthy key, and keys that just returned 429 /
RESOURCE_EXHAUSTED get a 60 s cooldown so the pool skips them until they
recover. When every key is cooling down the pool is retried fresh.

When no key is available the module stays usable: `LLM` reports
`available=False` and returns `[LLM ERROR] ...` for chat calls, and a
`DeterministicResponder` provides a rule-based JSON fallback for offline mode
(useful for unit tests and planner fallback).

Fallback order inside `chat()`: Gemini pool -> opencode `serve` HTTP backend
(`opencode_llm.py`, when reachable) -> deterministic responder. `usable`
covers keys-or-opencode; `available` keeps its historical meaning (keys).
"""
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import config

logger = logging.getLogger(__name__)

# Per-attempt cap for one Gemini API call. The SDK call has no reliable
# built-in timeout across versions, and a hung socket parks the whole agent
# loop (observed: 0 events for 100+ min on one ESTABLISHED connection).
# Implemented with a throwaway single-worker thread: version-proof, and a
# timed-out attempt is treated as a transient error (short cooldown, rotate
# to the next key) rather than killing the run.
GEMINI_CALL_TIMEOUT_S = 300

# Process-wide circuit breaker: once a full pool walk fails (every key
# rate-limited / dead), all LLM instances share this state so the next
# `chat()` calls short-circuit instantly instead of re-walking the dead
# pool (≈17 network round-trips) on every question. The breaker re-arms
# every 60 s so healthy keys are picked up as soon as their cooldown ends.
_POOL_LOCK = threading.Lock()
_POOL_DEAD_UNTIL = 0.0
_POOL_DEAD_REASON = None

# Cooldowns and the round-robin cursor are shared by every LLM instance in
# the process. Without this, each new instance (a Streamlit rerun builds a
# fresh one) starts with an empty cooldown map and re-walks all ~16 dead keys
# before reaching the breaker. The Orchestrator also drives 4 agents on 4
# threads through one LLM, so selection has to be locked.
_POOL_BAD_UNTIL: dict = {}
_POOL_IDX = 0


def reset_pool_state():
    """Clear all shared pool state (circuit breaker, cooldowns, cursor).

    Used by tests and by any long-lived process that wants to re-probe the
    whole pool immediately.
    """
    global _POOL_DEAD_UNTIL, _POOL_DEAD_REASON, _POOL_IDX
    with _POOL_LOCK:
        _POOL_BAD_UNTIL.clear()
        _POOL_IDX = 0
        _POOL_DEAD_UNTIL = 0.0
        _POOL_DEAD_REASON = None


def _short_reason(exc):
    if exc is None:
        return "unknown error"
    msg = str(exc).strip()
    return (msg[:140] + "…") if len(msg) > 140 else (msg or "unknown error")


def _generate_with_timeout(client, model, prompt, temperature):
    """One blocking SDK call with a hard timeout (see GEMINI_CALL_TIMEOUT_S)."""
    worker = ThreadPoolExecutor(max_workers=1)
    try:
        future = worker.submit(
            client.models.generate_content,
            model=model,
            contents=prompt,
            config={"temperature": temperature},
        )
        return future.result(timeout=GEMINI_CALL_TIMEOUT_S)
    except TimeoutError:
        raise TimeoutError(
            f"gemini call timed out after {GEMINI_CALL_TIMEOUT_S}s (hung socket?)")
    finally:
        worker.shutdown(wait=False, cancel_futures=True)


class DeterministicResponder:
    """Rule-based fallback LLM used offline (no API key).

    It answers '"offline" + <msg>' for chat and returns a single-tool PDB plan
    JSON for planner prompts. This keeps the framework runnable without network.
    """

    def __init__(self):
        self.available = False

    def chat(self, prompt, temperature=None):
        """Answer as the deterministic fallback (offline single-tool plan or
        a plain '[OFFLINE]' message)."""
        if "plan" in prompt.lower() and "json" in prompt.lower():
            return ('[{"tool": "pdb", "action": "query", '
                    '"params": {"operation": "global_aggregate", '
                    '"aggregation": {"column": "PCE", "agg": "mean"}}, '
                    '"purpose": "offline fallback database query", '
                    '"expected_outputs": ["records", "papers", "aggregate"]}]')
        return "[OFFLINE] API key not configured. deterministic-responder active."


class LLM:
    """Minimal Gemini wrapper: prompt -> text and prompt -> structured JSON.

    Example:
        llm = LLM()
        print(llm.available)          # False without key
        plan = llm.json(prompt)       # dict / list parsed from response
    """

    def __init__(self, model=None, temperature=None):
        self.model = model or config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        self._keys = self._load_keys()
        self._bad_until = _POOL_BAD_UNTIL   # shared: survive across instances
        self._picked_slot = None            # slot _pick_key() chose, for _rotate
        self._client = None
        self._client_key = None
        self.available = bool(self._keys)
        self._fallback = DeterministicResponder()
        self._opencode = None       # lazy OpenCodeServeBackend (no net in ctor)
        self.last_usage: Optional[dict] = None   # real token counts, set per chat() when the backend reports them

    # ---- key management --------------------------------------------------
    @classmethod
    def _load_keys(cls, keys_md_file=None):
        """Deduplicated key pool from (a) `GOOGLE_API_KEY` env, (b)
        `chatbot/.env`, (c) `<keys_md_file>` (default `~/keys.md`).

        The markdown file may use a bare key per line, `KEY=<key>`
        assignments, or markdown bullets (`- ` / `* `); `#` comments and
        blank lines are skipped.

        Duplicates are dropped across *all three* sources, keeping first-seen
        order. This matters because rotation uses `list.index(key)`: a key
        that appears twice would always resolve to its first position, so the
        cursor would jump backwards and the second copy would be counted but
        never reached. (A key in `GOOGLE_API_KEY` is usually also in
        `chatbot/.env` and `~/keys.md`.)
        """
        keys, seen = [], set()

        def add(k):
            k = (k or "").strip().strip('"').strip("'")
            if (k.startswith("AIza") or k.startswith("AQ.")) and k not in seen:
                seen.add(k)
                keys.append(k)

        add(os.environ.get("GOOGLE_API_KEY"))
        if os.path.exists(config.CHATBOT_ENV_FILE):
            with open(config.CHATBOT_ENV_FILE) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GOOGLE_API_KEY="):
                        add(line.split("=", 1)[1])
        md = keys_md_file or config.KEYS_MD_FILE
        if os.path.exists(md):
            with open(md) as f:
                for line in f:
                    k = line.strip()
                    if not k or k.startswith("#"):
                        continue
                    if k.startswith("GOOGLE_API_KEY=") or k.startswith("KEY="):
                        k = k.split("=", 1)[1].strip()
                    add(k.lstrip("-*•").strip())
        return keys

    @property
    def num_keys(self):
        """Number of configured API keys (for diagnostics/logging)."""
        return len(self._keys)

    def _rotate(self, key):
        """Advance the shared round-robin cursor just past the slot we picked.

        Uses the recorded slot rather than `list.index(key)` so a pool that
        somehow contains the same key twice still advances one step per call
        instead of snapping back to the first occurrence (which would pin the
        cursor and leave the rest of the pool unreachable).
        """
        global _POOL_IDX
        if not self._keys:
            return
        with _POOL_LOCK:
            if self._picked_slot is not None and self._keys[self._picked_slot] == key:
                _POOL_IDX = (self._picked_slot + 1) % len(self._keys)
            else:
                try:
                    _POOL_IDX = (self._keys.index(key) + 1) % len(self._keys)
                except ValueError:
                    pass
            self._picked_slot = None

    def _pick_key(self):
        """First healthy key starting after the cursor, skipping keys whose
        cooldown has not yet expired. Returns None if all are cooling."""
        global _POOL_IDX
        if not self._keys:
            return None
        with _POOL_LOCK:
            now = time.time()
            for off in range(len(self._keys)):
                slot = (_POOL_IDX + off) % len(self._keys)
                k = self._keys[slot]
                if self._bad_until.get(k, 0) <= now:
                    if off:
                        _POOL_IDX = slot
                    self._picked_slot = slot
                    return k
        return None

    def _mark_bad(self, key, exc):
        """Put a key on cooldown based on the error class, then rotate.

        * invalid / revoked API key -> 24 h (it will not recover on its own)
        * 429 / quota / rate limit  -> 60 s
        * 5xx / unavailable / timeout -> 20 s (transient service blip)
        * anything else             -> no cooldown, just rotate and retry

        Only genuinely key-specific text bans a key for 24 h: a 400 that the
        gateway attributes to the *request* (bad model id, malformed
        arguments) would otherwise silently disable a perfectly good key for
        a day.
        """
        msg = str(exc).upper()
        n = self._keys.index(key) + 1 if key in self._keys else -1
        if ("API KEY" in msg and "INVALID" in msg) or "API_KEY_INVALID" in msg:
            self._bad_until[key] = time.time() + 24 * 3600
            logger.info("LLM key pool: key %d/%d invalid, skipped for 24 h",
                        n, len(self._keys))
        elif "429" in msg or "RESOURCE_EXHAUSTED" in msg or "QUOTA" in msg:
            self._bad_until[key] = time.time() + 60
            logger.info("LLM key pool: key %d/%d cooling down for 60 s",
                        n, len(self._keys))
        elif any(t in msg for t in ("503", "502", "500", "UNAVAILABLE",
                                    "DEADLINE", "TIMEOUT", "SERVICE")):
            self._bad_until[key] = time.time() + 20
            logger.info("LLM key pool: key %d/%d transient error, cooling 20 s",
                        n, len(self._keys))
        self._rotate(key)

    @property
    def usable(self):
        """LLM text reachable via Gemini keys OR the opencode fallback."""
        if self.available:
            return True
        try:
            return self._opencode_backend().healthy()
        except Exception:
            return False

    def _opencode_backend(self):
        """Lazily constructed backend (keeps the ctor network-free; tolerant
        of test doubles built via `__new__` without `__init__`)."""
        backend = getattr(self, "_opencode", None)
        if backend is None:
            from .opencode_llm import OpenCodeServeBackend
            backend = OpenCodeServeBackend()
            try:
                self._opencode = backend
            except AttributeError:
                pass  # exotic test double without __dict__; use uncached
        return backend

    def _try_opencode(self, prompt):
        """Opencode text or None (never raises; one warning per failure)."""
        try:
            backend = self._opencode_backend()
            if not backend.enabled or not backend.url:
                return None  # fallback disabled: stay silent
            text = backend.chat(prompt)
        except Exception as exc:
            logger.warning("opencode fallback failed (%s)", exc)
            return None
        if text and text.strip():
            logger.info("opencode fallback answered (%d chars)", len(text))
            return text.strip()
        return None

    # ---- public API ------------------------------------------------------
    def chat(self, prompt, temperature=None):
        """Single-turn completion returning the raw text.

        Order: Gemini pool -> opencode serve -> deterministic responder.
        The `[LLM ERROR]` / `[OFFLINE]` sentinel contract is preserved for
        the no-backend case.
        """
        global _POOL_DEAD_UNTIL, _POOL_DEAD_REASON
        last_error = None
        pool_walked = False
        self.last_usage = None      # stale counts must never leak into the next call
        if self._keys:
            now = time.time()
            if not (now < _POOL_DEAD_UNTIL and self._pick_key() is None):
                pool_walked = True
                for _ in range(max(len(self._keys), 1)):
                    key = self._pick_key()
                    if key is None:
                        # Every key is on cooldown. Do NOT clear the cooldowns:
                        # the map is shared process-wide, so wiping it here
                        # would resurrect 24 h invalid-key bans and restart the
                        # full walk that the breaker exists to prevent. Fall
                        # through to the opencode/deterministic backends and
                        # let the breaker re-arm.
                        break
                    try:
                        from google import genai
                        client = genai.Client(api_key=key)
                        resp = _generate_with_timeout(
                            client, self.model, prompt,
                            self.temperature if temperature is None
                            else temperature,
                        )
                        self._client = client  # keep the working client warm
                        self._client_key = key
                        self._rotate(key)      # balanced rotation on success
                        try:
                            um = getattr(resp, "usage_metadata", None)
                            self.last_usage = {
                                "prompt_tokens": getattr(um, "prompt_token_count", None),
                                "completion_tokens": getattr(
                                    um, "candidates_token_count", None),
                                "total_tokens": getattr(um, "total_token_count", None),
                            } if um is not None else None
                        except Exception:
                            self.last_usage = None
                        with _POOL_LOCK:       # pool alive again: open breaker
                            _POOL_DEAD_UNTIL = 0.0
                            _POOL_DEAD_REASON = None
                        return (resp.text or "").strip()
                    except Exception as exc:   # quota / 429 / key error
                        last_error = exc
                        self._mark_bad(key, exc)
                # Every key failed this walk: arm the breaker so the next calls
                # skip the walk until cooldowns expire (then a single retry
                # re-probes it).
                with _POOL_LOCK:
                    _POOL_DEAD_UNTIL = time.time() + 60
                    _POOL_DEAD_REASON = _short_reason(last_error)
                logger.warning("LLM key pool down: all %d keys failed (%s)",
                               len(self._keys), _POOL_DEAD_REASON)
        # Opencode fallback before the deterministic/error sentinels.
        opencode_text = self._try_opencode(prompt)
        if opencode_text is not None:
            return opencode_text
        if not self._keys:
            return self._fallback.chat(prompt, temperature)
        if pool_walked:
            return f"[LLM ERROR] {last_error}"
        left = int(_POOL_DEAD_UNTIL - time.time())
        return (f"[LLM ERROR] all {len(self._keys)} API keys cooling/dead"
                f" ({_POOL_DEAD_REASON or 'rate-limited'})"
                f" — auto-retry in ~{left}s")

    def json(self, prompt, temperature=0.0):
        """Completion then parse the first JSON object/array found."""
        text = self.chat(prompt, temperature=temperature)
        return extract_json(text)

    def available_elsewhere(self, fallback):
        """Compatibility hook: when the external provider is unusable, the
        caller's fallback value is returned unchanged."""
        return fallback


# ----------------------------------------------------------------------------
# JSON parsing helpers
# ----------------------------------------------------------------------------
def extract_json(text):
    """Robustly pull the first JSON object/array from an LLM response.

    Scans every `{`/`[` position with `JSONDecoder.raw_decode`, so leading
    prose, trailing junk, fenced blocks, and doubled payloads like
    `{"tool": ...} raw: {"tool": ...}` (which greedy `\\{.*\\}` matching
    chokes on) all yield the first complete value.
    """
    if not text or (str(text).startswith("[LLM ERROR]") or str(text).startswith("[OFFLINE]")):
        return None
    for m in re.finditer(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL):
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    decoder = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", text):
        try:
            value, _ = decoder.raw_decode(text, m.start())
        except Exception:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None


def get_llm(model=None):
    """Construct an LLM client with the default key pool."""
    return LLM(model=model)


def fmt_json(x):
    """Pretty-print JSON for logs/debugging (str-fallback for non-JSON types)."""
    return json.dumps(x, indent=2, default=str)