"""OpenCode `serve` HTTP backend for PerovAgents (LLM fallback).

When the Gemini key pool is exhausted, `LLM.chat()` falls back to a locally
running `opencode serve` instance (default http://127.0.0.1:4096) before
dropping to the deterministic responder. One synchronous `chat()` maps to:
POST /session -> POST /session/:id/message (waits) -> DELETE /session/:id.

Lockdown learned empirically against the free tier (Sep 2026):
  * pass `agent: "plan"` + explicit `model` + a "never call tools" system
    prompt, and NO `tools` field — per-message `tools` overrides are
    rejected by the free-tier gateway (403 FreeTierError).
  * the configured default model may not exist; always send an explicit
    `providerID`/`modelID` pair.
  * provider/model errors arrive in-band as `info.error`, not HTTP errors.

Stdlib only (urllib), no new dependencies. Enable/disable + endpoint via:
  PEROVAGENTS_OPENCODE_URL (default http://127.0.0.1:4096; empty disables)
  PEROVAGENTS_OPENCODE_MODEL (default opencode/muse-spark-1.3-contributor-free)
  PEROVAGENTS_OPENCODE_TIMEOUT_S (default 300)
  PEROVAGENTS_OPENCODE_FALLBACK=0 disables the fallback entirely.
Basic auth follows the server convention: OPENCODE_SERVER_USERNAME
(default "opencode") / OPENCODE_SERVER_PASSWORD (unset = no auth header).
"""
import json
import logging
import threading
import time
import urllib.error
import urllib.request

from . import config

logger = logging.getLogger(__name__)

SYSTEM_NO_TOOLS = "Answer with plain text only. Never call tools."
HEALTH_TTL_S = 30.0
HEALTH_TIMEOUT_S = 3.0

_oc_lock = threading.Lock()
_oc_health = {"ts": 0.0, "ok": False}


class OpenCodeError(Exception):
    """The serve backend could not produce text (unreachable, HTTP error,
    provider error in `info.error`, or empty text parts)."""


def _reset_opencode_cache():
    """Test helper: forget the cached health probe."""
    with _oc_lock:
        _oc_health["ts"] = 0.0
        _oc_health["ok"] = False


def split_model(spec: str):
    """'provider/model' -> (providerID, modelID)."""
    spec = (spec or "").strip()
    if "/" in spec:
        provider, _, model = spec.partition("/")
        if provider and model:
            return provider, model
    raise OpenCodeError(f"bad model spec {spec!r}; want provider/model")


class OpenCodeServeBackend:
    """Synchronous single-turn client for `opencode serve`."""

    def __init__(self, url=None, model=None, timeout_s=None, agent="plan",
                 username=None, password=None, enabled=None):
        self.url = (url if url is not None else config.OPENCODE_URL).rstrip("/")
        self.model = model or config.OPENCODE_MODEL
        self.timeout_s = (timeout_s if timeout_s is not None
                          else config.OPENCODE_TIMEOUT_S)
        self.agent = agent
        self.username = (username if username is not None
                         else config.OPENCODE_USERNAME)
        self.password = (password if password is not None
                         else config.OPENCODE_PASSWORD)
        self.enabled = (enabled if enabled is not None
                        else config.OPENCODE_FALLBACK)

    # ---- plumbing ----------------------------------------------------
    def _headers(self, extra=None):
        headers = {"Content-Type": "application/json"}
        if self.password:
            import base64
            token = base64.b64encode(
                f"{self.username}:{self.password}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        if extra:
            headers.update(extra)
        return headers

    def _call(self, method, path, body=None, timeout=None):
        url = self.url + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers=self._headers())
        try:
            with urllib.request.urlopen(
                    req, timeout=self.timeout_s if timeout is None
                    else timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
                return json.loads(raw) if raw.strip() else True
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                detail = ""
            raise OpenCodeError(f"{method} {path}: HTTP {exc.code} {detail}")
        except Exception as exc:
            raise OpenCodeError(f"{method} {path}: {exc}")

    def healthy(self) -> bool:
        """Cached reachability probe (never raises)."""
        if not self.enabled or not self.url:
            return False
        now = time.time()
        with _oc_lock:
            if now - _oc_health["ts"] < HEALTH_TTL_S:
                return _oc_health["ok"]
        try:
            ok = self._call("GET", "/global/health",
                            timeout=HEALTH_TIMEOUT_S).get("healthy") is True
        except OpenCodeError:
            ok = False
        with _oc_lock:
            _oc_health["ts"] = now
            _oc_health["ok"] = ok
        return ok

    # ---- public API --------------------------------------------------
    def chat(self, prompt, temperature=None) -> str:
        """Single-turn completion. `temperature` accepted for interface
        parity but ignored: the message API has no per-call temperature."""
        if not self.enabled or not self.url:
            raise OpenCodeError("opencode fallback disabled")
        provider, model = split_model(self.model)
        session = self._call("POST", "/session", {"title": "perovagents"})
        sid = (session or {}).get("id") if isinstance(session, dict) else None
        if not sid:
            raise OpenCodeError("session create returned no id")
        try:
            msg = self._call("POST", f"/session/{sid}/message", {
                "agent": self.agent,
                "model": {"providerID": provider, "modelID": model},
                "system": SYSTEM_NO_TOOLS,
                "parts": [{"type": "text", "text": prompt}],
            })
            info = (msg or {}).get("info", {})
            err = info.get("error")
            if err:
                name = err.get("name", "error")
                detail = err.get("data", {})
                if isinstance(detail, dict):
                    detail = detail.get("message", str(detail))[:300]
                raise OpenCodeError(f"provider {name}: {detail}")
            texts = [p.get("text", "") for p in (msg or {}).get("parts", [])
                     if p.get("type") == "text" and p.get("text")]
            text = "\n".join(texts).strip()
            if not text:
                raise OpenCodeError("empty text in serve response")
            return text
        finally:
            try:
                self._call("DELETE", f"/session/{sid}", timeout=10)
            except OpenCodeError:
                pass  # session cleanup is best-effort
