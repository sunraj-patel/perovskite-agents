"""Passive provider-response instrumentation (benchmark v3).

``providers.py`` (the v2 provider) is NOT modified. This module subclasses its
classes and adds observation only: the request that is sent, the retry/rotation
policy and the text returned to the agents are identical to v2.

For every model call it records (never an API key, never an authorisation header):
    index, backbone/model id, status, finish_reason, block_reason, candidate count,
    content-part kinds, response text length, response id, model version, latency,
    attempts/retries inside the call, usage metadata (incl. thinking tokens),
    exception class + redacted message, key-pool slot number (not the key),
    and the raw reply text when it is non-empty (replies are model output; the
    prompt is never stored here).

``diagnose_empty`` classifies an EMPTY reply from that evidence only.
"""
import re
import threading
import time

from .providers import (CALL_TIMEOUT_S, RATE_WAIT_MAX_S, EpisodeLLM, GeminiBackend,
                        OpenCodeBackend, ProviderError)

RAW_TEXT_CAP = 4000
_SECRET = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|sk-[0-9A-Za-z_\-]{16,}|(?i:bearer)\s+[0-9A-Za-z._\-]{12,}|"
                     r"(?i:(?:api[_-]?key|key|token|authorization)\s*[=:]\s*)[^\s,;&\"']{8,})")


def redact(text, limit=300):
    """Strip anything that looks like a credential from a string before it is stored."""
    return _SECRET.sub("[REDACTED]", str(text))[:limit]


def _fr(x):
    """finish_reason enum/str -> short upper-case name."""
    if x is None:
        return None
    return str(getattr(x, "name", x)).split(".")[-1].upper()


def response_detail(resp):
    """Everything useful on a genai response, with no side effects."""
    d = {"status": "ok", "response_id": getattr(resp, "response_id", None),
         "model_version": getattr(resp, "model_version", None)}
    pf = getattr(resp, "prompt_feedback", None)
    d["block_reason"] = _fr(getattr(pf, "block_reason", None)) if pf is not None else None
    cands = getattr(resp, "candidates", None) or []
    d["n_candidates"] = len(cands)
    d["finish_reason"] = None
    d["part_kinds"] = []
    if cands:
        c0 = cands[0]
        d["finish_reason"] = _fr(getattr(c0, "finish_reason", None))
        d["finish_message"] = redact(getattr(c0, "finish_message", "") or "", 200) or None
        content = getattr(c0, "content", None)
        parts = (getattr(content, "parts", None) or []) if content is not None else []
        d["content_is_none"] = content is None
        for p in parts:
            if getattr(p, "function_call", None):
                d["part_kinds"].append("function_call")
            elif getattr(p, "thought", None):
                d["part_kinds"].append("thought")
            elif getattr(p, "text", None):
                d["part_kinds"].append("text")
            else:
                d["part_kinds"].append("other")
    um = getattr(resp, "usage_metadata", None)
    d["usage"] = ({"prompt_tokens": getattr(um, "prompt_token_count", None),
                   "completion_tokens": getattr(um, "candidates_token_count", None),
                   "thinking_tokens": getattr(um, "thoughts_token_count", None),
                   "total_tokens": getattr(um, "total_token_count", None)}
                  if um is not None else None)
    return d


class InstrumentedGeminiBackend(GeminiBackend):
    """Same request/rotation policy as v2, plus a per-call detail record."""

    def __init__(self, model, temperature=0.0, function_calling=None):
        super().__init__(model, temperature)
        self._tl = threading.local()
        # None = v2 request. "NONE" = same request plus tool_config.function_calling_config.mode=NONE
        # (the benchmark's tool protocol is plain text/JSON; native function calling is never wanted).
        self.function_calling = function_calling

    def _config(self, temp):
        cfg = {"temperature": temp}
        if self.function_calling:
            cfg["tool_config"] = {"function_calling_config": {"mode": self.function_calling}}
        return cfg

    def complete(self, prompt, temperature, events):
        from concurrent.futures import ThreadPoolExecutor
        from google import genai
        temp = self.temperature if temperature is None else temperature
        t0, last, attempts, attempt_log = time.time(), "no keys", 0, []
        while True:
            slot = self.pool.pick()
            if slot is None:
                if time.time() - t0 > RATE_WAIT_MAX_S:
                    self._tl.detail = {"status": "exhausted", "attempts": attempts,
                                       "attempt_log": attempt_log[-6:], "error": redact(last)}
                    raise ProviderError(f"all keys cooling/dead: {last}")
                wait = max(2.0, min(30.0, self.pool.earliest() - time.time()))
                events.append({"type": "pool_wait", "seconds": round(wait, 1)})
                time.sleep(wait)
                continue
            attempts += 1
            ta = time.time()
            worker = ThreadPoolExecutor(max_workers=1)
            try:
                client = genai.Client(api_key=self.pool.keys[slot])
                fut = worker.submit(client.models.generate_content, model=self.model,
                                    contents=prompt, config=self._config(temp))
                resp = fut.result(timeout=CALL_TIMEOUT_S)
                detail = response_detail(resp)
                try:
                    text = (resp.text or "").strip()
                except Exception as exc:                     # .text can raise on odd candidates
                    text = ""
                    detail["text_accessor_error"] = redact(f"{type(exc).__name__}: {exc}", 160)
                detail.update({"slot": slot, "attempts": attempts, "attempt_log": attempt_log[-6:],
                               "latency_s": round(time.time() - ta, 2), "text_len": len(text)})
                if text:
                    detail["raw_text"] = text[:RAW_TEXT_CAP]
                self._tl.detail = detail
                um = detail["usage"]
                usage = ({"prompt_tokens": um["prompt_tokens"], "completion_tokens": um["completion_tokens"],
                          "total_tokens": um["total_tokens"]} if um else None)
                return text, usage
            except Exception as exc:
                msg = str(exc).upper()
                last = msg[:120]
                kind = type(exc).__name__
                if "API KEY" in msg and "INVALID" in msg or "API_KEY_INVALID" in msg:
                    self.pool.mark(slot, 24 * 3600)
                    events.append({"type": "invalid_key", "slot": slot})
                    cls = "invalid_key"
                elif "429" in msg or "RESOURCE_EXHAUSTED" in msg or "QUOTA" in msg:
                    daily = "PER DAY" in msg or "PERDAY" in msg
                    self.pool.mark(slot, 6 * 3600 if daily else 60)
                    events.append({"type": "rate_limit", "slot": slot, "daily": daily})
                    cls = "rate_limit"
                elif any(t in msg for t in ("503", "502", "500", "UNAVAILABLE", "DEADLINE",
                                            "TIMEOUT", "TIMED OUT")):
                    self.pool.mark(slot, 20)
                    events.append({"type": "provider_unavailable", "slot": slot, "detail": last[:60]})
                    cls = "unavailable_or_timeout"
                else:
                    self.pool.mark(slot, 5)
                    events.append({"type": "provider_error", "slot": slot, "detail": last[:100]})
                    cls = "provider_error"
                attempt_log.append({"slot": slot, "class": cls, "exception": kind,
                                    "message": redact(str(exc), 160),
                                    "seconds": round(time.time() - ta, 2)})
            finally:
                worker.shutdown(wait=False, cancel_futures=True)


class InstrumentedOpenCodeBackend(OpenCodeBackend):
    """OpenCode route: the v2 backend already records its internal steps; only the
    per-call latency/length is added by the episode wrapper."""


# ---------------------------------------------------------------- classification
# Evidence -> label for ONE empty reply. Only labels the evidence supports are used.
PROVIDER_SIDE = {"provider_filtered", "provider_no_candidate", "provider_anomaly", "provider_exception"}


def diagnose_empty(detail):
    """Classify one empty model reply from its recorded provider detail."""
    if not detail:
        return "unclassified_empty"                     # no metadata captured (e.g. non-Gemini route)
    st = detail.get("status")
    if st in ("exhausted", "error"):
        return "provider_exception"
    if detail.get("block_reason"):
        return "provider_filtered"
    if detail.get("n_candidates", 1) == 0:
        return "provider_no_candidate"
    fr = detail.get("finish_reason")
    usage = detail.get("usage") or {}
    if fr in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY",
              "LANGUAGE"):
        return "provider_filtered"
    if fr in ("MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"):
        # No tool is declared in the request, so the model itself produced a native function-call
        # part that the API refused to parse. Observed on the direct JSON-tool prompt (v3 smoke:
        # 17/17 first moves). Model/prompt-attributable, not infrastructure.
        return "malformed_function_call"
    if fr in ("OTHER", "MISSING_THOUGHT_SIGNATURE"):
        return "provider_anomaly"
    if fr == "MAX_TOKENS":
        return "token_limit_empty"                      # budget (e.g. thinking) used up, no visible text
    if "function_call" in detail.get("part_kinds", []):
        return "function_call_only"                     # model answered with a native tool call part
    if fr == "STOP":
        thinking = usage.get("thinking_tokens") or 0
        comp = usage.get("completion_tokens") or 0
        if thinking and not comp:
            return "thinking_only_stop"                 # stopped after hidden thinking, no visible text
        return "empty_stop"                             # STOP, no content: valid-looking empty completion
    return "unclassified_empty"


class InstrumentedEpisodeLLM(EpisodeLLM):
    """EpisodeLLM + per-call log. The text returned to the agents is unchanged."""

    def __init__(self, backbone):
        super().__init__(backbone)
        if backbone["kind"] == "gemini":
            self.backend = InstrumentedGeminiBackend(backbone["model"],
                                                     function_calling=backbone.get("function_calling_mode"))
        self.call_log = []
        self._t_start = time.time()

    def chat(self, prompt, temperature=None):
        t0 = time.time()
        n_before = self.calls
        text = super().chat(prompt, temperature)
        detail = getattr(getattr(self.backend, "_tl", None), "detail", None)
        failed = self.infra_failed and text == self.SENTINEL
        rec = {"i": None, "t_offset_s": round(t0 - self._t_start, 2),
               "latency_s": round(time.time() - t0, 2), "model": self.model_id,
               "prompt_chars": len(prompt or ""), "text_len": 0 if failed else len(text or ""),
               "empty": (not failed) and not (text or "").strip(), "provider_failure": failed}
        if detail:
            rec["detail"] = {k: v for k, v in detail.items() if k != "raw_text"} | (
                {"raw_text": detail["raw_text"]} if detail.get("raw_text") and len(text or "") <= RAW_TEXT_CAP else {})
            self.backend._tl.detail = None
        if rec["empty"]:
            rec["empty_class"] = diagnose_empty(detail if self.backbone["kind"] == "gemini" else None)
        with self._lock:
            rec["i"] = len(self.call_log) + 1
            self.call_log.append(rec)
        return text

    def summary(self):
        out = super().summary()
        empties = [c for c in self.call_log if c["empty"]]
        classes = {}
        for c in empties:
            classes[c["empty_class"]] = classes.get(c["empty_class"], 0) + 1
        out["call_log"] = self.call_log
        out["empty_reply_calls"] = len(empties)
        out["empty_reply_classes"] = classes
        out["provider_failure_calls"] = sum(1 for c in self.call_log if c["provider_failure"])
        return out
