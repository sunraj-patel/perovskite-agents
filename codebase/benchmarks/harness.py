"""Benchmark harness: token recorder + direct-LLM baseline loop.

TokenRecorder wraps any chat backend (MRS pool LLM or Provider) and
accumulates exact prompt/completion characters plus real backend token
counts whenever the backend reports them.

DirectBaseline is the "no orchestration" system: one model, the shared
tool layer, fixed budgets. It deliberately receives NONE of: task
analyzer, planner, progress state, verifier, provenance graph, recovery
logic. Its history truncation matches Agent.run so context budgets are
structurally identical.
"""
import time

from .tool_layer import DONE, MAX_LLM_TURNS, MAX_TOOL_CALLS, catalog_text

BASELINE_SYSTEM = """You are a scientific assistant for perovskite solar cells. \
Answer the question using the tools below. Each turn reply with EITHER a single \
JSON tool call {"tool": "<name>", "args": {...}} OR, when you have everything \
needed, FINAL ANSWER: followed by your answer. Use only tool outputs and say \
what could not be established. Do not invent numbers, DOIs or values."""


class TokenRecorder:
    """Wraps .chat backends; records exact chars + real usage when present."""

    def __init__(self, backend):
        self.backend = backend
        self.calls = 0
        self.prompt_chars = 0
        self.completion_chars = 0
        self.real_prompt_tokens = 0
        self.real_completion_tokens = 0
        self.has_real_usage = False

    def chat(self, prompt, temperature=None):
        self.calls += 1
        self.prompt_chars += len(prompt or "")
        out = self.backend.chat(prompt, temperature)
        usage = getattr(self.backend, "last_usage", None)
        if isinstance(out, tuple) and len(out) == 2:
            text, usage = out
        else:
            text = out
        text = text or ""
        self.completion_chars += len(text)
        if isinstance(usage, dict):
            pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if isinstance(pt, int) and isinstance(ct, int):
                self.real_prompt_tokens += pt
                self.real_completion_tokens += ct
                self.has_real_usage = True
        return text

    def summary(self) -> dict:
        return {
            "llm_calls": self.calls,
            "prompt_chars": self.prompt_chars,
            "completion_chars": self.completion_chars,
            "prompt_tokens_est": self.prompt_chars // 4,
            "completion_tokens_est": self.completion_chars // 4,
            "real_prompt_tokens": self.real_prompt_tokens or None,
            "real_completion_tokens": self.real_completion_tokens or None,
            "real_usage": self.has_real_usage,
        }


class DirectBaseline:
    """One model + shared tools + fixed budget. No MRS machinery."""

    def __init__(self, provider, tools, max_tool_calls=MAX_TOOL_CALLS,
                 max_turns=MAX_LLM_TURNS, timeout_s=None):
        from perovagents.agent import _chat_with_retry
        self._chat = _chat_with_retry
        self.provider = TokenRecorder(provider)
        self.tools = tools
        self.max_tool_calls = max_tool_calls
        self.max_turns = max_turns
        self.timeout_s = timeout_s

    def _history(self, log):
        return [f"> {c}\n{r if i >= len(log) - 3 else r[:300]}"
                for i, (c, r) in enumerate(log)]

    def run(self, question):
        from perovagents.llm import extract_json
        t0 = time.time()
        log, tool_calls, termination = [], [], "answer"
        system = (f"{BASELINE_SYSTEM}\n\nTOOLS:\n{catalog_text(self.tools)}")
        for _ in range(self.max_turns):
            if self.timeout_s and time.time() - t0 > self.timeout_s:
                termination = "timeout"
                break
            prompt = (f"{system}\n\nQUESTION: {question}\n\n"
                      + "\n\n".join(self._history(log)) + "\n\nYour next move:")
            try:
                # Same empty-reply redelivery the MRS agents get (transport
                # hygiene, not orchestration): an empty completion carries
                # no decision and must not burn a turn.
                reply = self._chat(self.provider, prompt)
            except Exception as exc:
                log.append(("(llm call failed)", f"ERROR: {exc}"))
                termination = "provider_error"
                break
            if DONE in reply:
                log.append(("(final)", reply.split(DONE, 1)[1].strip()))
                break
            msg = extract_json(reply)
            tool = msg.get("tool") if isinstance(msg, dict) else None
            args = dict(msg.get("args") or {}) if isinstance(msg, dict) else {}
            if tool not in self.tools or len(tool_calls) >= self.max_tool_calls:
                if tool not in self.tools:
                    log.append(("(no valid tool call)",
                                f"ERROR: unknown tool {tool!r}; reply with JSON or {DONE}"))
                    continue
                termination = "tool_budget"
                break
            call = f"{tool}({args})"
            try:
                out = self.tools[tool](**args)
                result = str(out)[:3000]
            except Exception as exc:
                result = f"ERROR: {type(exc).__name__}: {exc}"
            tool_calls.append({"tool": tool, "args": args, "result": result[:1000]})
            log.append((call, result))
        else:
            termination = "turn_budget"
        answer = ""
        for call, result in reversed(log):
            if call == "(final)":
                answer = result
                break
        return {"answer": answer, "termination": termination,
                "tool_calls": tool_calls,
                "n_tool_calls": len(tool_calls),
                "transcript_tail": log[-4:],
                "seconds": round(time.time() - t0, 1),
                "tokens": self.provider.summary()}
