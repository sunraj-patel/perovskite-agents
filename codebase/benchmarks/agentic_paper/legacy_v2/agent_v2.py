"""The whole framework in one file.

    Agent         a prompt + tools. It loops (call a tool, read the result) until it is
                  satisfied, then says the phrase  FINAL ANSWER:  and its report goes
                  back to the orchestrator.
    Orchestrator  reads the question, sends tasks to agents (ASK lines), reads their
                  reports, and decides: ask again, ask someone else, or FINAL ANSWER.
"""
import inspect
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from .llm import extract_json
from .progress import (BLOCKED, INCOMPLETE, INVALID, MAX_NO_PROGRESS, NO_PROGRESS,
                       ESCALATIONS, Contract, SubtaskState, norm_text)


def norm_args(args):
    """Canonical string for a tool-call argument dict, so that calls differing only in
    whitespace, quoting or key order collide (e.g. the same SQL typed slightly differently).
    """
    if args is None:
        return ""
    try:
        s = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        s = str(args)
    s = s.lower()
    return re.sub(r"\s+", " ", s).strip()


def fingerprint(tool, args):
    """Stable identity of a tool call, for repeated-call detection."""
    return f"{tool}|{norm_args(args)}"


DONE = "FINAL ANSWER:"          # the phrase that ends an agent's work (and the orchestrator's)

# v3 budgets for LARGE tasks (long task text or >= 3 contract requirements). Small
# tasks keep the original budgets; the extension is recorded in every trace
# ("turn_budget" event) and is bounded, so wall-clock/tool-call caps still apply.
EXTRA_TURNS = 6
LARGE_TASK_CHARS = 600
LARGE_REQUIREMENTS = 3
EXTRA_ROUNDS = 2                 # orchestrator rounds added for multi-part questions
LARGE_QUESTION_CHARS = 400
MULTI_AGENT_ROUNDS = 3           # distinct agents asked before the extension applies
MAX_FORMAT_RETRIES = 2           # bounded re-prompts for a malformed orchestrator reply
EVIDENCE_CHARS = 1500

_ASK_RE = re.compile(r"^[\s>*_`#-]*ASK\s+(\w+)\s*:\s*(.+)$", re.M | re.I)
_FINAL_RE = re.compile(r"FINAL\s+ANSWER\s*:", re.I)


def parse_orchestrator_reply(reply, have_reports=False):
    """Validate the orchestrator's output grammar.

    Returns {"kind": "empty"|"ask"|"final"|"malformed", "asks": [(agent, task)],
             "answer": str}. A reply is a valid move only if it contains
    `ASK <agent>: <task>` line(s) or the `FINAL ANSWER:` marker with a non-empty body.
    Prose such as "Dispatched the pdb agent ..." is "malformed" - it is NOT an answer.
    ASK lines win over a FINAL marker until at least one report exists (nothing
    can be finalised before an agent has reported).
    """
    text = reply or ""
    if not text.strip():
        return {"kind": "empty", "asks": [], "answer": ""}
    asks = [(a, t.strip()) for a, t in _ASK_RE.findall(text) if t.strip()]
    m = _FINAL_RE.search(text)
    body = text[m.end():].strip() if m else ""
    if asks and (not m or not have_reports):
        return {"kind": "ask", "asks": asks, "answer": ""}
    if m and body:
        return {"kind": "final", "asks": [], "answer": body}
    return {"kind": "malformed", "asks": [], "answer": ""}


def _evidence(log, n=2):
    """The last tool results that actually returned data, to travel with an
    escalation so the orchestrator is not left with a bare diagnostic (v3)."""
    good = [(c, r) for c, r in log
            if not r.startswith(("ERROR", "BLOCKED")) and not c.startswith("(")]
    if not good:
        return ""
    return ("\nEVIDENCE GATHERED SO FAR (verbatim tool results; usable for the answer):\n"
            + "\n".join(f"> {c}\n{r[:EVIDENCE_CHARS]}" for c, r in good[-n:]))


def norm_task(task: str) -> str:
    """Normalise a task for duplicate detection at orchestrator level."""
    return norm_text(task)


def _is_escalation(report: str) -> bool:
    """True when a report is a progress-escalation diagnostic, not a normal report."""
    r = report or ""
    for token in ESCALATIONS:
        if token in r:
            return True
    return "NOT_RERUN" in r or "stopped after" in r


def _dup_blocked(agent: str, task: str) -> str:
    return (f"NOT_RERUN: an identical task to {agent!r} was already sent in a previous round and "
            f"came back as an escalation (loop/no progress), so it was NOT run again. Change the "
            f"task wording, split it, ask a different agent, or answer with what you have.")

# A tool call whose (tool, normalized args) fingerprint has already been used this often.
# The same normalized call may appear up to REPEAT_LIMIT times, because one retry after an
# ERROR: result is legitimate. A further identical call means the cell is looping, not
# progressing, so the cell is stopped and REPEATED_ACTION is reported to the orchestrator.
REPEAT_LIMIT = 2

AGENT_RULES = f"""\
Each turn, reply with EITHER
  a tool call, as JSON only:   {{"tool": "<name>", "args": {{...}}}}
  OR, once you are satisfied that the task is answered, the phrase {DONE} followed by your report.
Keep trying until you are satisfied: read every result, fix errors, refine queries, check
spelling and values. An empty or surprising result is a reason to look closer, not to stop.
Only use numbers that appeared in tool results (use the calc tool to compute new ones).
Your report must say what you found, how (filters, sample sizes) and what you could NOT find.
Do not invent data, columns or tools.

Subtask completion: each tool call is meant to finish one step of the task (retrieve, then
aggregate, then interpret). When the previous result already gave you the data you asked for,
DO NOT fetch it again. Interpret or aggregate what you have instead.
NO REPEATED CALLS: never call the same tool with identical or semantically equivalent
arguments twice in a row after a successful result, and do not re-issue a sub-query you
already answered. The only allowed retry with the same arguments is immediately after an
ERROR: result. If the evidence is insufficient, change the query (different terms, columns,
filters, aggregation, paper), not the call count. Identical repeated calls are blocked
automatically and the cell stops with REPEATED_ACTION."""


def format_examples(examples):
    """Render worked examples (multi-shot prompting) for the system prompt.

    Each example is a dict with `task` and `turns`, where turns alternate
    between what the agent sent (`reply`) and what the tool returned
    (`result`). Showing the whole loop - including an error being read and
    corrected - teaches the retry behaviour far better than the rules alone.
    """
    blocks = ["\n\nWORKED EXAMPLES (real tasks, real tool calls - copy the pattern, "
              "not the numbers):"]
    for i, ex in enumerate(examples, 1):
        blocks.append(f"\n--- Example {i}: {ex['task']}")
        for turn in ex["turns"]:
            reply = turn.get("reply", "")
            label = DONE if DONE in reply else "tool call"
            blocks.append(f"  you: {label}\n    {reply}")
            if turn.get("result") is not None:
                res = str(turn["result"])
                if len(res) > 600:
                    res = res[:600] + " ..."
                blocks.append(f"  tool: {res}")
    return "\n".join(blocks)


def _chat_with_retry(llm, prompt, trace=None, agent=None, turn=None):
    """One LLM turn with immediate retries on EMPTY replies only.

    An empty completion carries no decision, so it must not consume a turn
    (agent loop), pollute history/contract state, or become an empty final
    answer (orchestrator). `[LLM ERROR]` sentinels are returned as-is: under
    a real outage the turn-burn + max_turns bound is the correct behavior.
    """
    reply = llm.chat(prompt)
    tries = 0
    while not (reply or "").strip() and tries < 2:
        tries += 1
        if trace:
            trace.event("llm_retry", agent=agent, turn=turn,
                        attempt=tries, reason="empty reply")
        time.sleep(2)
        reply = llm.chat(prompt)
    return reply


class Agent:
    def __init__(self, name, job, knowledge, tools, llm, max_turns=12, examples=None,
                 contract=None, block_on=True, repeat_cap=True):
        self.name, self.job, self.llm, self.max_turns = name, job, llm, max_turns
        self.tools = {f.__name__: f for f in tools}
        self.contract = contract                    # progress contract (or None)
        self.block_on = block_on                    # run guards / no-progress stops
        self.repeat_cap = repeat_cap                # identical-call REPEAT_LIMIT
        self.last_state = None                      # last SubtaskState (for tests/audit)
        catalog = "\n".join(f"- {f.__name__}{inspect.signature(f)}: {inspect.getdoc(f)}" for f in tools)
        self.system = (f"You are the {name} agent of a perovskite solar-cell research team. {job}\n\n"
                       f"{AGENT_RULES}\n\nWHAT YOU KNOW:\n{knowledge}\n\nYOUR TOOLS (all of them):\n{catalog}")
        if examples:
            self.system += format_examples(examples)

    def _call(self, reply):
        msg = extract_json(reply)
        name = msg.get("tool") if isinstance(msg, dict) else None
        if name not in self.tools:
            return f"(no valid tool call)", (f"ERROR: reply with a tool call as JSON, or {DONE} <report>. "
                                              f"Tools: {list(self.tools)}")
        args = msg.get("args") or {}
        try:
            out = self.tools[name](**args)
        except Exception as exc:                                  # the agent reads the error and retries
            out = f"ERROR: {type(exc).__name__}: {exc}"
        return f"{name}({args})", str(out)[:3000]

    def _chat(self, prompt, trace=None, turn=None):
        return _chat_with_retry(self.llm, prompt, trace=trace,
                                agent=self.name, turn=turn)

    def run(self, task, verbose=False, trace=None):
        log = []                                                  # (call, result) pairs
        counts = {}                                               # fingerprint -> occurrences
        contract = self.contract                  # track progress when a contract exists
        state = contract.new_state(task) if contract is not None else None
        invalid_streak = 0
        satisfied_nudges = 0            # "report what you have" redirections
        large = len(task) >= LARGE_TASK_CHARS or (
            state is not None and len(state.requirements) >= LARGE_REQUIREMENTS)
        turns = self.max_turns + (EXTRA_TURNS if large else 0)
        if trace:
            trace.event("turn_budget", agent=self.name, turns=turns, large_task=large)
        for _ in range(turns):
            history = [f"> {c}\n{r if i >= len(log) - 3 else r[:300]}" for i, (c, r) in enumerate(log)]
            reply = self._chat(f"{self.system}\n\nTASK: {task}\n\n" + "\n\n".join(history)
                               + "\n\nYour next move:",
                               trace=trace, turn=len(log) + 1)
            if trace:
                trace.event("llm_reply", agent=self.name, turn=len(log) + 1, text=reply)
            if DONE in reply:
                report = reply.split(DONE, 1)[1].strip()
                if state is not None:
                    esc = contract.review(state)
                    if esc is not None:
                        esc += _evidence(log)
                        if verbose:
                            print(f"  [{self.name}] {DONE} blocked by contract -> {esc[:300]}")
                        if trace:
                            trace.event("report", agent=self.name, text=esc,
                                        progress_status=INCOMPLETE, escalation_reason=esc[:400])
                        return esc
                if verbose:
                    print(f"  [{self.name}] {DONE} {report[:300]}")
                if trace:
                    trace.event("report", agent=self.name, text=report)
                return report
            msg = extract_json(reply)
            tool = msg.get("tool") if isinstance(msg, dict) else None
            args = dict(msg.get("args") or {}) if isinstance(msg, dict) else {}
            fp = fingerprint(tool, args) if tool in self.tools else None
            if tool in self.tools:
                counts[fp] = counts.get(fp, 0) + 1
                if state is not None and self.block_on:
                    g = contract.guard(state, tool, args, fp)
                    if g is not None:
                        # Contract met but the model re-verifies instead of
                        # reporting (Q02 rerun: ranking rows fetched, then a
                        # repeat GROUP BY hit the guard): steer to FINAL once
                        # or twice rather than stopping with a diagnostic.
                        # Exact identical repeats still stop hard (the loop
                        # signature); only near-redundant calls get the nudge.
                        identical = any(
                            h.fp == fp and not h.gained
                            and h.status not in ("ERROR_RETRY", INVALID)
                            for h in state.history) if fp is not None else False
                        if not identical and not state.missing() and satisfied_nudges < 2:
                            satisfied_nudges += 1
                            nudge = (f"NOTE: every requirement is already met "
                                     f"({state.satisfied()}). Do NOT call another tool - "
                                     f"write {DONE} now with the rows and numbers you "
                                     f"already fetched.")
                            log.append((f"{tool}({args})",
                                        f"BLOCKED: {g} {nudge}"))
                            if verbose:
                                print(f"  [{self.name}] {tool}({args})\n      -> {nudge[:200]}")
                            if trace:
                                trace.event("tool_call", agent=self.name, tool=tool,
                                            args=args, signature=fp)
                                trace.event("tool_result", agent=self.name,
                                            result=f"BLOCKED: {g} {nudge}",
                                            progress_status=BLOCKED)
                            continue
                        diag = f"{g} (occurrence {counts[fp]})" + _evidence(log)
                        if verbose:
                            print(f"  [{self.name}] {tool}({args})\n      -> BLOCKED: {diag[:200]}")
                        if trace:
                            trace.event("tool_call", agent=self.name, tool=tool, args=args,
                                        signature=fp)
                            trace.event("tool_result", agent=self.name, result=f"BLOCKED: {diag}",
                                        progress_status=BLOCKED, escalation_reason=diag[:400])
                            trace.event("report", agent=self.name, text=diag,
                                        progress_status=BLOCKED, escalation_reason=diag[:400])
                        return diag
                if self.repeat_cap and counts[fp] > REPEAT_LIMIT:
                    call = f"{tool}({args})"
                    diag = (f"REPEATED_ACTION: {call} was about to run again with the same "
                            f"arguments (occurrence {counts[fp]}, allowed {REPEAT_LIMIT}). The "
                            f"earlier identical call already returned a result, so repeating it "
                            f"cannot make the task progress. Stopping this agent with a "
                            f"diagnostic so the orchestrator can replan - for a data question, "
                            f"aggregate the rows you already have (GROUP BY/COUNT) or ask for "
                            f"different columns; for literature, read a different paper or search "
                            f"with different terms.") + _evidence(log)
                    if verbose:
                        print(f"  [{self.name}] {call}\n      -> {diag[:200]}")
                    if trace:
                        trace.event("tool_call", agent=self.name, tool=tool, args=args)
                        trace.event("tool_result", agent=self.name, result=f"BLOCKED: {diag}")
                        trace.event("report", agent=self.name, text=diag)
                    return diag
            call, result = self._call(reply)
            error = result.startswith("ERROR")
            assessment = None
            status = None
            if state is not None:
                assessment = contract.assess(state, tool, args, result, error, fp)
                status = "ERROR_RETRY" if error else assessment["status"]
                state.apply(tool, args, fp, status, assessment.get("newly_satisfied", []),
                            assessment.get("rejected", []), assessment.get("note", ""))
                self.last_state = state
                if self.block_on:
                    if status == NO_PROGRESS and state.no_progress_streak() >= MAX_NO_PROGRESS:
                        diag = (f"{NO_PROGRESS}: {state.no_progress_streak()} consecutive tool "
                                f"calls added nothing. Established: {state.satisfied()}; still "
                                f"missing: {state.missing()}. Stop and return to the orchestrator "
                                f"so it can replan with a different strategy.") + _evidence(log)
                        if verbose:
                            print(f"  [{self.name}] {call}\n      -> {diag[:200]}")
                        if trace:
                            trace.event("tool_result", agent=self.name, result=result,
                                        progress_status=NO_PROGRESS, escalation_reason=diag[:400])
                            trace.event("report", agent=self.name, text=diag,
                                        progress_status=NO_PROGRESS, escalation_reason=diag[:400])
                        return diag
                    if status == INVALID:
                        invalid_streak += 1
                        if invalid_streak >= MAX_NO_PROGRESS:
                            diag = (f"{INVALID}: two consecutive tool results were invalid "
                                    f"(e.g. solver did not converge). Stop and return to the "
                                    f"orchestrator; do not keep retrying.") + _evidence(log)
                            if trace:
                                trace.event("report", agent=self.name, text=diag,
                                            progress_status=INVALID, escalation_reason=diag[:300])
                            return diag
                    else:
                        invalid_streak = 0
            if verbose:
                print(f"  [{self.name}] {call}\n      -> {result[:200]}")
            if trace:
                msg2 = extract_json(reply)
                tfields = dict(agent=self.name,
                               tool=(msg2 or {}).get("tool") if isinstance(msg2, dict) else None,
                               args=(msg2 or {}).get("args") if isinstance(msg2, dict) else None)
                if state is not None:
                    tfields["signature"] = fp
                trace.event("tool_call", **tfields)
                rfields = dict(agent=self.name, result=result)
                if state is not None:
                    rfields.update(progress_status=status,
                                   newly_satisfied=assessment.get("newly_satisfied") or [],
                                   still_missing=[r[0] for r in state.missing()],
                                   rejected=[f"{t}:{re}" for t, re, _ri
                                             in assessment.get("rejected", [])],
                                   retry_reason=assessment.get("note") or None)
                trace.event("tool_result", **rfields)
            log.append((call, result))
        last = log[-1][1][:300] if log else "nothing"
        msg = f"(agent stopped after {turns} turns without a final answer; last result: {last})" + _evidence(log)
        if state is not None and state.rejected:
            msg += f" Rejected evidence: {[t for t, _re, _ri in state.rejected]}"
        if trace:
            trace.event("report", agent=self.name, text=msg)
        return msg


class Orchestrator:
    def __init__(self, agents, llm, max_rounds=5, sequential=False):
        self.agents, self.llm, self.max_rounds = agents, llm, max_rounds
        self.sequential = sequential                    # one agent at a time + pacing
        self._asked = {}                              # (agent, norm task) -> reports w/ status
        team = "\n".join(f"- {n}: {a.job}" for n, a in agents.items())
        self.system = f"""You lead a team of specialist agents that answer questions about perovskite solar cells.

TEAM
{team}

Each turn, reply with EITHER
  one or more lines  ASK <agent>: <a complete, self-contained task>   (lines run in parallel;
  an agent sees only its own task text and knows its own data and tools),
  OR, when the reports are enough, {DONE} followed by the answer for the user.
DISPATCH COMPLETENESS: a question that asks for several kinds of evidence must ASK the matching
agents IN THE SAME ROUND, each with its own self-contained subtask - e.g. database statistics
go to pdb, papers to literature, PCE prediction/shap/what-if to ml, device design + simulation
to design. Do not satisfy a multi-part question with reports that cover only one part.
If a report is missing, unclear or looks wrong, ask again with a sharper task or another agent.
A report that says REPEATED_ACTION, NO_PROGRESS, BLOCKED, INVALID_RESULT or INCOMPLETE (an
"escalation") means that agent could not finish the subtask as given and returned a diagnostic
with: what was established, what is still missing, and a recommended next action. Use that
FAILED STRATEGY context when you replan: do NOT re-send that agent the same task verbatim -
split the task, change the agent, change what it is being asked to produce, or answer with
what you have and say what could not be established.
Do not answer {DONE} while an explicitly requested part has NO report and was never ASKed;
dispatch the missing part first. When every requested part has a report, reconcile them:
if they contradict, say so; if a part was not established, say that too.
Use only what the agents reported and say what could not be established. Do not invent numbers.
An ESCALATION may end with "EVIDENCE GATHERED SO FAR": those are verbatim tool results; if they
contain what the question needs, answer from them (quote the numbers) instead of re-asking.
For a ranking of individual devices ("highest-PCE devices") one ORDER BY ... LIMIT query is enough.
OUTPUT GRAMMAR: dispatching happens ONLY through lines that start with `ASK <agent>:`. Never
describe or announce a dispatch in prose ("Dispatched the pdb agent ..."): that sends nothing and
is rejected. Every reply is either ASK line(s) or {DONE} + the answer, nothing else."""

    def ask(self, question, verbose=False, trace=None):
        notes = []                                                # (agent, task, report)
        if trace:
            trace.ensure_question(question)
        self.format_errors = 0
        max_rounds = self.max_rounds + (EXTRA_ROUNDS if len(question) >= LARGE_QUESTION_CHARS else 0)
        extended = len(question) >= LARGE_QUESTION_CHARS
        format_notice, format_retries = "", 0
        round_no = 0
        while round_no <= max_rounds:
            if (not extended and len({a.lower() for (a, _t, _r) in notes}) >= MULTI_AGENT_ROUNDS):
                max_rounds += EXTRA_ROUNDS                    # multi-part question: allow the extra rounds once
                extended = True
                if trace:
                    trace.event("round_budget", rounds=max_rounds, reason="multi-agent question")
            escalations = sum(1 for (_a, _t, r) in notes if _is_escalation(r))
            reports = []
            for (a, t, r) in notes:
                tag = "ESCALATION" if _is_escalation(r) else "REPORT"
                reports.append(f"[{tag}] [{a}] task: {t}\nreport: {r}")
            reports.append(f"# escalation count so far: {escalations}")
            reports_txt = "\n\n".join(reports) or "(none yet)"
            last = round_no == max_rounds
            pressure = " You must answer now with DONE and what you have.\n" if last else ""
            if escalations >= 3 and not last:
                pressure += (" Several agents escalated; if you already used their diagnostics "
                             "and a second attempt was blocked, answer now with what you have "
                             "and say what is still missing.\n")
            reply = _chat_with_retry(
                self.llm,
                f"{self.system}\n\nQUESTION: {question}\n\nREPORTS SO FAR:\n"
                                  + reports_txt + "\n\n" + pressure + format_notice + "Your next move:")
            if trace:
                trace.event("llm_reply", agent="orchestrator", turn=round_no + 1, text=reply)
            move = parse_orchestrator_reply(reply, have_reports=bool(notes))
            if move["kind"] == "empty":
                round_no += 1
                continue                                # triple-empty: no-op round, do not answer ""
            if move["kind"] == "malformed":
                self.format_errors += 1
                if trace:
                    trace.event("format_error", agent="orchestrator", attempt=format_retries + 1,
                                text=(reply or "")[:300])
                if format_retries < MAX_FORMAT_RETRIES:
                    format_retries += 1                 # bounded re-prompt; does not consume a round
                    format_notice = (
                        "FORMAT ERROR: your previous reply had neither an `ASK <agent>: <task>` line "
                        f"nor `{DONE}` followed by an answer, so nothing was dispatched. Prose such as "
                        "'Dispatched the agent' does not dispatch anything. Reply now with one or more "
                        f"lines `ASK <agent>: <complete task>` (agents: {', '.join(self.agents)}) or "
                        f"`{DONE} <answer>`.\n")
                    continue
                msg = (f"[FORMAT ERROR] the orchestrator produced no valid `ASK` dispatch or `{DONE}` "
                       f"after {format_retries + 1} attempts. Last reply: {(reply or '').strip()[:300]}")
                if trace:
                    trace.event("answer", text=msg)
                return msg
            format_notice = ""
            if move["kind"] == "final":
                answer = move["answer"]
                if trace:
                    trace.event("answer", text=answer)
                return answer
            asks = move["asks"]
            if verbose:
                print("\n".join(f"\n== ASK {a}: {t}" for a, t in asks))
            if trace:
                for a, t in asks:
                    trace.event("ask", agent=a, task=t)
            # ---- global anti-loop: never re-run an identical task that escalated -------
            to_run, seen = [], set()
            for a, t in asks:
                key = (a.lower(), norm_task(t))
                if key in seen:
                    notes.append((a, t, _dup_blocked(a, t)))
                    continue
                seen.add(key)
                prior = self._asked.get(key)
                if prior and any(_is_escalation(r) for r in prior[-2:]):
                    notes.append((a, t, _dup_blocked(a, t)))
                    continue
                to_run.append((a, t))
            with ThreadPoolExecutor(max_workers=1 if self.sequential else (len(to_run) or 1)) as pool:
                if self.sequential:
                    reports_now = [self._run_one(a, t, verbose, trace) for a, t in to_run]
                    time.sleep(10)                    # pacing gap: stay under per-minute quota
                else:
                    reports_now = pool.map(
                        lambda at: self._run_one(at[0], at[1], verbose, trace), to_run)
                for (a, t), r in zip(to_run, reports_now):
                    notes.append((a, t, r))
                    key = (a.lower(), norm_task(t))
                    self._asked.setdefault(key, []).append(r)
            round_no += 1
        msg = "No answer could be produced."
        if trace:
            trace.event("answer", text=msg)
        return msg

    def _run_one(self, name, task, verbose, trace=None):
        agent = self.agents.get(name.lower())
        if agent is None:
            return f"There is no agent called {name!r}. Available: {list(self.agents)}."
        # Duck-typed agents may implement just run(task, verbose). Decide from
        # the signature rather than catching TypeError, which would also
        # swallow a real error and re-run (and re-bill) the whole agent.
        if trace is not None and self._accepts_trace(agent):
            return agent.run(task, verbose, trace=trace)
        return agent.run(task, verbose)

    @staticmethod
    def _accepts_trace(agent):
        """True when `agent.run` takes a `trace` keyword or **kwargs."""
        try:
            params = inspect.signature(agent.run).parameters
        except (TypeError, ValueError):
            return False
        if "trace" in params:
            return True
        return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
