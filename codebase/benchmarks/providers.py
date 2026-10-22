"""Provider abstraction for the cross-LLM benchmark.

One interface, several vendors::

    provider.chat(prompt, temperature) -> (text, usage)
    provider.healthy() -> bool

`usage` is {"prompt_tokens", "completion_tokens", "total_tokens"} with None
where the backend does not report real counts (the runner falls back to
chars/4 and marks the record estimated=True).

Rollout is stepwise: only Gemini is wired for now. Claude/GPT adapters
exist as explicit stubs that fail with a clear message until API keys are
configured (ANTHROPIC_API_KEY / OPENAI_API_KEY). The benchmark questions
and runner are provider-independent, so adding a vendor later changes no
question text and no MRS code.
"""
import os
from typing import Dict, Optional, Tuple


class ProviderError(Exception):
    """The provider is not configured or the call failed."""


class Provider:
    """Single-vendor text backend. `chat` returns (text, usage-or-None)."""

    name = "base"

    def chat(self, prompt: str, temperature: Optional[float] = None) -> Tuple[str, Optional[Dict]]:
        raise NotImplementedError

    def healthy(self) -> bool:
        raise NotImplementedError


class GeminiProvider(Provider):
    """Gemini through the MRS key pool (rotation, cooldowns, breaker)."""

    name = "gemini"

    def __init__(self, model: Optional[str] = None):
        from perovagents.llm import get_llm
        self._llm = get_llm(model=model)
        self.model = self._llm.model

    def chat(self, prompt, temperature=None):
        text = self._llm.chat(prompt, temperature=temperature)
        return text, self._llm.last_usage

    def healthy(self) -> bool:
        return self._llm.usable

    @property
    def num_keys(self) -> int:
        return self._llm.num_keys


class ClaudeProvider(Provider):
    """STUB. Wire to ANTHROPIC_API_KEY when a key is provided."""

    name = "claude"

    def __init__(self, model: Optional[str] = None):
        self.key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        self.model = model or os.environ.get("PEROVAGENTS_CLAUDE_MODEL",
                                             "claude-sonnet-4-5")

    def chat(self, prompt, temperature=None):
        raise ProviderError(
            "Claude baseline not configured: set ANTHROPIC_API_KEY, then "
            "implement ClaudeProvider.chat (Anthropic messages API, same "
            "tool-calling convention as the Gemini baseline).")

    def healthy(self) -> bool:
        return bool(self.key)


class OpenRouterProvider(Provider):
    """DeepSeek (or any OpenRouter model) via the OpenAI-compatible API.

    Key: OPENROUTER_API_KEY env. Model: OPENROUTER_MODEL env, default
    deepseek/deepseek-chat. Stdlib urllib, no new dependencies. Returns
    real usage when the API reports it.
    """

    name = "deepseek"
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model: Optional[str] = None, timeout_s: int = 300):
        import urllib.request  # noqa: F401 (kept local: stdlib only, lazy)
        self.key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.model = (model or os.environ.get("OPENROUTER_MODEL")
                      or "deepseek/deepseek-chat")
        self.timeout_s = timeout_s

    def healthy(self) -> bool:
        return bool(self.key)

    def chat(self, prompt, temperature=None):
        import json as _json
        import urllib.error as _ue
        import urllib.request as _ur
        if not self.key:
            raise ProviderError(
                "DeepSeek baseline not configured: set OPENROUTER_API_KEY.")
        body = {"model": self.model,
                "messages": [{"role": "user", "content": prompt}]}
        if temperature is not None:
            body["temperature"] = temperature
        req = _ur.Request(
            self.endpoint, data=_json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"},
            method="POST")
        try:
            with _ur.urlopen(req, timeout=self.timeout_s) as resp:
                data = _json.loads(resp.read().decode("utf-8", "replace"))
        except _ue.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                detail = ""
            raise ProviderError(f"openrouter HTTP {exc.code}: {detail}")
        except Exception as exc:
            raise ProviderError(f"openrouter call failed: {exc}")
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise ProviderError(f"openrouter bad response: {str(data)[:300]}")
        usage, u = None, data.get("usage") or {}
        if isinstance(u.get("prompt_tokens"), int):
            usage = {"prompt_tokens": u.get("prompt_tokens"),
                     "completion_tokens": u.get("completion_tokens"),
                     "total_tokens": u.get("total_tokens")}
        return text.strip(), usage


class GPTProvider(Provider):
    """STUB. Wire to OPENAI_API_KEY when a key is provided."""

    name = "gpt"

    def __init__(self, model: Optional[str] = None):
        self.key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.model = model or os.environ.get("PEROVAGENTS_GPT_MODEL", "gpt-5")

    def chat(self, prompt, temperature=None):
        raise ProviderError(
            "GPT baseline not configured: set OPENAI_API_KEY, then implement "
            "GPTProvider.chat (OpenAI responses/chat API, same tool-calling "
            "convention as the Gemini baseline).")

    def healthy(self) -> bool:
        return bool(self.key)


PROVIDERS = {"gemini": GeminiProvider, "deepseek": OpenRouterProvider,
             "claude": ClaudeProvider, "gpt": GPTProvider}


def get_provider(name: str, model: Optional[str] = None) -> Provider:
    """Provider by system name; KeyError on unknown vendor."""
    try:
        return PROVIDERS[name.lower()](model=model)
    except KeyError:
        raise KeyError(f"unknown provider {name!r}; want one of {sorted(PROVIDERS)}")
