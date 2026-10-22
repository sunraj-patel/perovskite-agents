"""Strict, single-backbone LLM providers for the benchmark.

Design rules (see README, "Model access"):

* One episode talks to exactly ONE backbone. There is no fallback to another
  model, another provider, or a deterministic responder. If the backbone cannot
  answer after bounded retries, the episode is flagged ``infra_failed`` and is
  reported separately from agent failures (and re-queued on resume).
* Every call is recorded (latency, real token usage when reported, rate-limit
  and provider-error events). Secrets are never recorded: keys are referred to
  by pool slot number only.
* Gemini route  : google-genai SDK, free-tier key pool read from ~/keys.md
                  (keys are rotated; invalid keys are skipped).
* OpenCode route: ``opencode serve`` HTTP API, an explicit ``opencode/<model>``
                  free model, using the framework's own OpenCodeServeBackend.
"""
import json
import os
import threading
import time
from pathlib import Path

from .paths import CONFIG_DIR, setup_env

setup_env()

RATE_WAIT_MAX_S = 240          # total patience (per call) while rate limited
CALL_TIMEOUT_S = 150           # one provider call


class ProviderError(Exception):
    """The backbone could not produce text within the retry budget."""


def load_models():
    with open(CONFIG_DIR / "models.json") as f:
        return json.load(f)["backbones"]


# --------------------------------------------------------------------- Gemini
class GeminiPool:
    """Process-wide free-tier key pool with per-key cooldowns (thread-safe)."""

    def __init__(self):
        from perovagents.llm import LLM
        self.keys = LLM._load_keys()
        self.bad_until = {}
        self.idx = 0
        self.lock = threading.Lock()

    def pick(self):
        with self.lock:
            now = time.time()
            for off in range(len(self.keys)):
                slot = (self.idx + off) % len(self.keys)
                if self.bad_until.get(slot, 0) <= now:
                    self.idx = (slot + 1) % len(self.keys)
                    return slot
        return None

    def earliest(self):
        with self.lock:
            return min(self.bad_until.values(), default=0) if len(self.bad_until) >= len(self.keys) else 0

    def mark(self, slot, seconds):
        with self.lock:
            self.bad_until[slot] = time.time() + seconds


_POOL = None
_POOL_LOCK = threading.Lock()


def gemini_pool():
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = GeminiPool()
        return _POOL


class GeminiBackend:
    route = "google-genai direct (free-tier key pool)"

    def __init__(self, model, temperature=0.0):
        self.model, self.temperature = model, temperature
        self.pool = gemini_pool()

    @property
    def n_keys(self):
        return len(self.pool.keys)

    def complete(self, prompt, temperature, events):
        """-> (text, usage). Raises ProviderError on exhaustion."""
        from concurrent.futures import ThreadPoolExecutor
        from google import genai
        temp = self.temperature if temperature is None else temperature
        t0 = time.time()
        last = "no keys"
        while True:
            slot = self.pool.pick()
            if slot is None:
                if time.time() - t0 > RATE_WAIT_MAX_S:
                    raise ProviderError(f"all keys cooling/dead: {last}")
                wait = max(2.0, min(30.0, self.pool.earliest() - time.time()))
                events.append({"type": "pool_wait", "seconds": round(wait, 1)})
                time.sleep(wait)
                continue
            worker = ThreadPoolExecutor(max_workers=1)
            try:
                client = genai.Client(api_key=self.pool.keys[slot])
                fut = worker.submit(client.models.generate_content, model=self.model,
                                    contents=prompt, config={"temperature": temp})
                resp = fut.result(timeout=CALL_TIMEOUT_S)
                um = getattr(resp, "usage_metadata", None)
                usage = ({"prompt_tokens": getattr(um, "prompt_token_count", None),
                          "completion_tokens": getattr(um, "candidates_token_count", None),
                          "total_tokens": getattr(um, "total_token_count", None)}
                         if um is not None else None)
                text = (resp.text or "").strip()
                return text, usage
            except Exception as exc:
                msg = str(exc).upper()
                last = msg[:120]
                if "API KEY" in msg and "INVALID" in msg or "API_KEY_INVALID" in msg:
                    self.pool.mark(slot, 24 * 3600)
                    events.append({"type": "invalid_key", "slot": slot})
                elif "429" in msg or "RESOURCE_EXHAUSTED" in msg or "QUOTA" in msg:
                    daily = "PER DAY" in msg or "PERDAY" in msg
                    self.pool.mark(slot, 6 * 3600 if daily else 60)
                    events.append({"type": "rate_limit", "slot": slot, "daily": daily})
                elif any(t in msg for t in ("503", "502", "500", "UNAVAILABLE", "DEADLINE",
                                            "TIMEOUT", "TIMED OUT")):
                    self.pool.mark(slot, 20)
                    events.append({"type": "provider_unavailable", "slot": slot,
                                   "detail": last[:60]})
                else:
                    self.pool.mark(slot, 5)
                    events.append({"type": "provider_error", "slot": slot, "detail": last[:100]})
            finally:
                worker.shutdown(wait=False, cancel_futures=True)


# ------------------------------------------------------------------- OpenCode
_OC_BLOCKED = {}          # model -> epoch until which the free quota is exhausted


class OpenCodeBackend:
    """``opencode serve`` message API with the locked-down ``plan`` agent.

    The call is NOT a bare completion: OpenCode wraps the model in its own agent
    prompt (and may take hidden internal steps with its read-only file tools).
    The backend therefore records, per call, the number of internal assistant
    steps, the OpenCode tool names used, and the token usage OpenCode reports
    (which includes OpenCode's own system prompt)."""
    route = "opencode serve (free model, locked plan agent)"

    def __init__(self, model, temperature=None, url=None):
        import urllib.request  # noqa: F401
        self.model = model
        self.url = (url or os.environ.get("PEROVAGENTS_OPENCODE_URL", "http://127.0.0.1:4096")).rstrip("/")
        self.provider_id, self.model_id = model.split("/", 1)

    def _call(self, method, path, body=None, timeout=CALL_TIMEOUT_S):
        import urllib.error
        import urllib.request
        req = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode() if body is not None else None,
            method=method, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
        return json.loads(raw) if raw.strip() else None

    def _free_tier_status(self, sid):
        """If the serve process is stuck retrying because the free quota is gone, say so."""
        try:
            st = (self._call("GET", "/session/status", timeout=10) or {}).get(sid) or {}
        except Exception:
            return None
        if st.get("type") == "retry" and "free" in str(st.get("message", "")).lower():
            return st
        return None

    def _once(self, prompt, events):
        blocked = _OC_BLOCKED.get(self.model)
        if blocked and time.time() < blocked:
            raise RuntimeError(f"free usage exceeded (quota resets at epoch {int(blocked)})")
        sess = self._call("POST", "/session", {"title": "agentic-paper-bench"}, timeout=20)
        sid = sess["id"]
        try:
            try:
                msg = self._call("POST", f"/session/{sid}/message", {
                    "agent": "plan",
                    "model": {"providerID": self.provider_id, "modelID": self.model_id},
                    "system": "Answer with plain text only. Never call tools.",
                    "parts": [{"type": "text", "text": prompt}]})
            except Exception as exc:
                st = self._free_tier_status(sid) if "timed out" in str(exc).lower() else None
                if st:
                    nxt = (st.get("next") or 0) / 1000.0
                    _OC_BLOCKED[self.model] = nxt or time.time() + 3600
                    try:
                        self._call("POST", f"/session/{sid}/abort", {}, timeout=10)
                    except Exception:
                        pass
                    raise RuntimeError(f"free usage exceeded: {st.get('message')} "
                                       f"(quota resets at epoch {int(nxt)})")
                raise
            info = (msg or {}).get("info", {})
            err = info.get("error")
            if err:
                detail = err.get("data", {})
                detail = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
                raise RuntimeError(f"provider {err.get('name', 'error')}: {detail[:200]}")
            text = "\n".join(p.get("text", "") for p in (msg or {}).get("parts", [])
                             if p.get("type") == "text" and p.get("text")).strip()
            # hidden internal steps + token usage across the whole turn
            steps, tools_used, pt, ct = 0, [], 0, 0
            try:
                for m in self._call("GET", f"/session/{sid}/message", timeout=20) or []:
                    if m["info"].get("role") != "assistant":
                        continue
                    steps += 1
                    t = m["info"].get("tokens") or {}
                    pt += (t.get("input") or 0) + ((t.get("cache") or {}).get("read") or 0)
                    ct += (t.get("output") or 0) + (t.get("reasoning") or 0)
                    tools_used += [p.get("tool") for p in m.get("parts", []) if p.get("type") == "tool"]
            except Exception:
                pass
            events.append({"type": "opencode_internal", "assistant_steps": steps,
                           "opencode_tools_used": tools_used})
            return text, ({"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
                          if pt or ct else None)
        finally:
            try:
                self._call("DELETE", f"/session/{sid}", timeout=10)
            except Exception:
                pass

    def complete(self, prompt, temperature, events):
        t0, attempt = time.time(), 0
        while True:
            attempt += 1
            try:
                text, usage = self._once(prompt, events)
                if not text:
                    raise RuntimeError("empty text in serve response")
                return text, usage
            except Exception as exc:
                msg = str(exc)
                low = msg.lower()
                kind = ("rate_limit" if any(k in low for k in ("rate limit", "429", "too many", "free usage",
                                                               "free_tier_limit", "freetiererror"))
                        else "timeout" if "timed out" in low else "provider_error")
                events.append({"type": kind, "detail": msg[:160]})
                if kind == "rate_limit" or attempt >= 3 or time.time() - t0 > RATE_WAIT_MAX_S:
                    raise ProviderError(f"{kind}: {msg[:160]}")
                time.sleep(min(20.0, 5.0 * attempt))


# ------------------------------------------------------------ per-episode LLM
class EpisodeLLM:
    """What the agents see: ``chat(prompt, temperature=None) -> str``.

    Wraps one backend for one episode. After the first unrecoverable provider
    failure the episode is dead: every further call returns a fixed sentinel
    immediately (no network), so the orchestrator/agents wind down quickly and
    the runner records ``infra_failed`` instead of scoring an agent failure.
    """

    SENTINEL = "[PROVIDER FAILURE] backbone unavailable; episode aborted."

    def __init__(self, backbone):
        self.backbone = backbone
        kind = backbone["kind"]
        self.backend = (GeminiBackend(backbone["model"]) if kind == "gemini"
                        else OpenCodeBackend(backbone["model"]))
        self.model_id = backbone["model"]
        self.calls = 0
        self.prompt_chars = 0
        self.completion_chars = 0
        self.real_prompt_tokens = 0
        self.real_completion_tokens = 0
        self.real_usage_calls = 0
        self.latencies = []
        self.events = []
        self.infra_failed = False
        self.infra_reason = None
        self.aborted = False
        self.last_usage = None
        self._lock = threading.Lock()

    def abort(self):
        """Stop spending quota (episode timed out); later calls return the sentinel."""
        self.aborted = True

    def chat(self, prompt, temperature=None):
        if self.infra_failed or self.aborted:
            return self.SENTINEL
        t0 = time.time()
        ev = []
        try:
            text, usage = self.backend.complete(prompt, temperature, ev)
        except ProviderError as exc:
            with self._lock:
                self.events.extend(ev)
                self.infra_failed, self.infra_reason = True, str(exc)[:200]
                self.events.append({"type": "episode_abort", "detail": str(exc)[:200]})
            return self.SENTINEL
        dt = time.time() - t0
        with self._lock:
            self.calls += 1
            self.prompt_chars += len(prompt or "")
            self.completion_chars += len(text or "")
            self.latencies.append(round(dt, 2))
            self.events.extend(ev)
            if usage and isinstance(usage.get("prompt_tokens"), int) \
                    and isinstance(usage.get("completion_tokens"), int):
                self.real_prompt_tokens += usage["prompt_tokens"]
                self.real_completion_tokens += usage["completion_tokens"]
                self.real_usage_calls += 1
        return text

    def summary(self):
        rl = sum(1 for e in self.events if e["type"] == "rate_limit")
        waits = sum(e.get("seconds", 0) for e in self.events if e["type"] == "pool_wait")
        return {
            "model_calls": self.calls,
            "prompt_chars": self.prompt_chars,
            "completion_chars": self.completion_chars,
            "real_prompt_tokens": self.real_prompt_tokens if self.real_usage_calls else None,
            "real_completion_tokens": self.real_completion_tokens if self.real_usage_calls else None,
            "usage_reported_calls": self.real_usage_calls,
            "rate_limit_events": rl,
            "provider_error_events": sum(1 for e in self.events
                                         if e["type"] in ("provider_error", "provider_unavailable",
                                                          "timeout")),
            "pool_wait_s": round(waits, 1),
            "llm_latency_s": round(sum(self.latencies), 1),
            "infra_failed": self.infra_failed,
            "infra_reason": self.infra_reason,
            "provider_events": self.events[:50],
        }


def health_check(backbone):
    """One trivial call; returns {ok, model, seconds, detail} (no secrets)."""
    llm = EpisodeLLM(backbone)
    t0 = time.time()
    text = llm.chat('Reply with exactly the word: PONG', temperature=0.0)
    ok = (not llm.infra_failed) and bool(text.strip())
    return {"ok": ok, "backbone": backbone["id"], "model": backbone["model"],
            "route": llm.backend.route, "seconds": round(time.time() - t0, 1),
            "reply": text[:60], "detail": llm.infra_reason}
