"""LLM interface for PerovAgents.

Single backend: Google Gemini via the `google-genai` SDK. Keys are loaded from
(a) `GOOGLE_API_KEY` env, (b) `chatbot/.env`, (c) `~/keys.md` (one key per
line; used for rate-limit rotation).

`chat()` walks the key pool in a simple round-robin circle: it starts at the
cursor, tries every key in order, and returns the first successful completion
(the cursor then advances that key so the load is balanced). If every key
fails, or none are configured, the sentinel `[LLM ERROR] ...` is returned so
callers can gate on it — there is no opencode serve backend and no
deterministic responder fallback anymore.
"""
import json
import logging
import os
import re

from . import config

logger = logging.getLogger(__name__)


def _short_reason(exc):
    if exc is None:
        return "unknown error"
    msg = str(exc).strip()
    return (msg[:140] + "…") if len(msg) > 140 else (msg or "unknown error")


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
        self._idx = 0
        self._client = None
        self._client_key = None
        self.available = bool(self._keys)

    # ---- key management --------------------------------------------------
    @classmethod
    def _load_keys(cls, keys_md_file=None):
        """Deduplicated key pool from (a) `GOOGLE_API_KEY` env, (b)
        `chatbot/.env`, (c) `<keys_md_file>` (default `~/keys.md`).

        The markdown file may use a bare key per line, `KEY=<key>`
        assignments, or markdown bullets (`- ` / `* `); `#` comments and
        blank lines are skipped.
        """
        keys = []
        env = os.environ.get("GOOGLE_API_KEY")
        if env:
            keys.append(env.strip())
        if os.path.exists(config.CHATBOT_ENV_FILE):
            with open(config.CHATBOT_ENV_FILE) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GOOGLE_API_KEY="):
                        keys.append(line.split("=", 1)[1].strip()
                                    .strip('"').strip("'"))
        md = keys_md_file or config.KEYS_MD_FILE
        if os.path.exists(md):
            with open(md) as f:
                for line in f:
                    k = line.strip()
                    if not k or k.startswith("#"):
                        continue
                    if k.startswith("GOOGLE_API_KEY=") or k.startswith("KEY="):
                        k = k.split("=", 1)[1].strip()
                    k = k.lstrip("-*•").strip()
                    if k.startswith("AIza") and k not in keys:
                        keys.append(k)
        return keys

    @property
    def num_keys(self):
        """Number of configured API keys (for diagnostics/logging)."""
        return len(self._keys)

    def _rotate(self, key):
        """Advance the round-robin cursor just past `key`."""
        try:
            self._idx = (self._keys.index(key) + 1) % len(self._keys)
        except ValueError:
            pass

    def _pick_key(self):
        """Next key after the cursor, wrapping in a circle."""
        key = self._keys[self._idx % len(self._keys)]
        self._idx = (self._idx + 1) % len(self._keys)
        return key

    @property
    def usable(self):
        """GLM text reachable (== available: keys configured)."""
        return self.available

    # ---- public API ------------------------------------------------------
    def chat(self, prompt, temperature=None):
        """Single-turn completion returning the raw text.

        Simple round-robin over the key pool: each key is tried exactly once
        per call, starting after the cursor; the first success wins. When no
        key is configured, or the whole pool failed, the `[LLM ERROR]`
        sentinel is returned (callers gate on it to fall back to rules).
        """
        if not self._keys:
            logger.warning("no API keys configured (checked env, "
                           "chatbot/.env, %s)", config.KEYS_MD_FILE)
            return "[LLM ERROR] no API keys configured"
        last_error = None
        for _ in range(len(self._keys)):
            key = self._pick_key()
            kidx = self._keys.index(key) + 1
            try:
                from google import genai
                client = genai.Client(api_key=key)
                resp = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": self.temperature if temperature is None
                            else temperature},
                )
                self._client = client  # keep the working client warm
                self._client_key = key
                self._rotate(key)      # balanced rotation on success
                text = (resp.text or "").strip()
                logger.info("key %d/%d ok model=%s chars=%d",
                            kidx, len(self._keys), self.model, len(text))
                return text
            except Exception as exc:
                last_error = exc
                logger.info("key %d/%d failed (%s)",
                            kidx, len(self._keys), _short_reason(exc))
        logger.warning("LLM pool: all %d keys failed (%s)",
                       len(self._keys), _short_reason(last_error))
        return f"[LLM ERROR] {_short_reason(last_error)}"

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
    """Robustly pull the first JSON object/array from an LLM response."""
    if not text or (str(text).startswith("[LLM ERROR]") or str(text).startswith("[OFFLINE]")):
        return None
    for m in re.finditer(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL):
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    stripped = text.strip()
    # Prefer whichever bracket opens first so arrays aren't swallowed by the
    # greedy `{...}` matcher below.
    orders = (r"(\[.*\])", r"(\{.*\})") if stripped.startswith("[") \
        else (r"(\{.*\})", r"(\[.*\])")
    for pat in orders:
        for m in re.finditer(pat, text, re.DOTALL):
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    try:
        return json.loads(text)
    except Exception:
        return None


def get_llm(model=None):
    """Construct an LLM client with the default key pool."""
    return LLM(model=model)


def fmt_json(x):
    """Pretty-print JSON for logs/debugging (str-fallback for non-JSON types)."""
    return json.dumps(x, indent=2, default=str)