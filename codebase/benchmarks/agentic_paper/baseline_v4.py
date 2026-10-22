"""Direct-baseline protocol v4: a neutral, protocol-compatible tool-use prompt.

The v3 baseline wording ("You are a scientific assistant ... Each turn reply with EITHER ...")
made gemini-3.5-flash-lite emit native function-call parts the API refused
(finish_reason=MALFORMED_FUNCTION_CALL, see V3_REPORT.md section 2). v4 keeps the baseline a
single model over the common tool catalogue, with nothing from PerovAgents (no agent prompts,
orchestrator, progress contracts, escalation, evidence carry-over, decomposition), and only states
the calling protocol: tool catalogue, argument names, one JSON call per turn, how to finish.

Wording chosen from first-move probes of gemini-3.5-flash-lite on DEVELOPMENT questions only
(variants: v3 wording 0/6 non-empty; 'How to call a tool' wording 0/6; this wording 6/6).

`benchmarks/harness.py` is unchanged; the loop below is the same loop (same budgets, same history
truncation, same empty-reply redelivery) with the system prompt as the only difference.
"""
import time

from benchmarks.harness import DirectBaseline, TokenRecorder
from benchmarks.tool_layer import DONE, catalog_text

PROTOCOL_ID = "direct_v4_neutral_json"

BASELINE_SYSTEM_V4 = f"""\
You answer questions about perovskite solar cells using the tools listed below.

Each turn, reply with EITHER
  a tool call, as JSON only:   {{"tool": "<name>", "args": {{...}}}}
  OR, when you have what you need, the phrase {DONE} followed by your answer.
Use the exact tool and argument names from the list. Make at most one call per reply, and write
the JSON as ordinary text in your reply. You will then be shown the tool's result.
Base the answer on tool results only; state what you could not establish. Do not invent numbers,
DOIs or values."""


class ProtocolBaseline(DirectBaseline):
    """DirectBaseline with the v4 protocol prompt; otherwise the identical loop and budgets."""
    system_prompt = BASELINE_SYSTEM_V4

    def run(self, question):
        from perovagents.llm import extract_json
        t0 = time.time()
        log, tool_calls, termination = [], [], "answer"
        system = f"{self.system_prompt}\n\nTOOLS:\n{catalog_text(self.tools)}"
        for _ in range(self.max_turns):
            if self.timeout_s and time.time() - t0 > self.timeout_s:
                termination = "timeout"
                break
            prompt = (f"{system}\n\nQUESTION: {question}\n\n"
                      + "\n\n".join(self._history(log)) + "\n\nYour next move:")
            try:
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
                result = str(self.tools[tool](**args))[:3000]
            except Exception as exc:
                result = f"ERROR: {type(exc).__name__}: {exc}"
            tool_calls.append({"tool": tool, "args": args, "result": result[:1000]})
            log.append((call, result))
        else:
            termination = "turn_budget"
        answer = next((r for c, r in reversed(log) if c == "(final)"), "")
        return {"answer": answer, "termination": termination, "tool_calls": tool_calls,
                "n_tool_calls": len(tool_calls), "transcript_tail": log[-4:],
                "seconds": round(time.time() - t0, 1), "tokens": self.provider.summary()}
