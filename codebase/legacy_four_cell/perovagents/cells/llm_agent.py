"""LLM revision helper for cell agents (spec section 3, "No" branch).

On retry attempts the cell's agent must revise its call from the gate's
critique instead of replaying the failed call. `llm_tool_call` builds that
revision prompt from the versioned cell prompt (`prompts/<name>.md`), the
task, and the compacted attempt history, and parses the reply as the next
tool call (merged over the deterministic fallback so required keys
survive a partial reply). Unparsable replies fall back deterministically —
retries never crash on LLM output.
"""
from typing import Any, Dict, List

from ..cells.schemas import Attempt, Task
from ..llm import extract_json
from ..prompts import load as load_prompt


def render_history(history: List[Attempt], max_chars: int = 1500) -> str:
    """Compact attempt history: tool calls plus failed checks/messages."""
    lines = []
    for att in history:
        failed = [f"{c.name}: {c.message}"
                  for c in att.report.checks if not c.passed]
        lines.append(f"attempt {att.n}: call={att.tool_call} "
                     f"failed=[{'; '.join(failed)}]")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[-max_chars:]


def llm_tool_call(llm, prompt_name: str, task: Task,
                  history: List[Attempt], fallback: Dict[str, Any]
                  ) -> Dict[str, Any]:
    """Ask the LLM agent for a revised tool call. Returns the fallback
    when there is no history (first attempt), no LLM, or no parse."""
    if not history or llm is None:
        return dict(fallback)
    prompt = (load_prompt(prompt_name) +
              f"\n\nTASK: {task.request}\n"
              f"PREVIOUS ATTEMPTS (do not repeat a failed call):\n"
              f"{render_history(history)}\n"
              "Reply with a JSON object for the next tool call only.")
    try:
        parsed = extract_json(llm.chat(prompt, temperature=0.0))
    except Exception:
        parsed = None
    if not isinstance(parsed, dict) or not parsed:
        return dict(fallback)
    merged = dict(fallback)
    merged.update(parsed)
    return merged
