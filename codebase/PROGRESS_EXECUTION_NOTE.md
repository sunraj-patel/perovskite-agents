# Progress-aware execution and anti-loop control — implementation note

## 1. Current execution flow (audit)

The framework is intentionally small: one file of control flow (`perovagents/agent.py`,
~250 lines) plus five prompted agents assembled in `perovagents/team.py`.

```
question -> Orchestrator.ask (max_rounds=5)
             ├─ LLM picks: 1+ "ASK <agent>: <free-text task>" lines
             ├─ agents run in parallel threads (ThreadPoolExecutor)
             │    Agent.run (max_turns=12): loop { LLM -> tool JSON -> tool call -> result }
             │    until the LLM writes "FINAL ANSWER: <report>"
             └─ reports come back; orchestrator decides ASK again / ask another / final answer
```

There is **no** separate `orchestrator.py`, `planner.py`, `TaskSpec`, `core/tool.py`,
verifier, or output validator. The pieces the task description maps onto are:

| concept in task brief          | actual location                                    |
|--------------------------------|----------------------------------------------------|
| orchestrator                   | `Orchestrator` in `perovagents/agent.py` (lines 168-250) |
| executor / agent execution loop| `Agent.run` in `perovagents/agent.py` (112-165)     |
| tool call generation           | LLM emits JSON `{"tool": ..., "args": ...}`; `extract_json` parses |
| tool result interpretation     | result string fed back verbatim as next prompt      |
| sufficiency decision           | *only* the LLM's own `FINAL ANSWER` judgement       |
| retries                        | free LLM choice; one hard cap (12 turns)            |
| replanning                     | orchestrator re-ASKs, also free LLM choice          |
| verifier / output validator    | none                                                |
| provenance / claims            | `Trace` (JSONL) logs question/ask/tool_call/tool_result/report/answer |
| benchmarks                     | `perovagents/benchmarks/questions.py` (question sets only, no runner) |
| ablation modes A-G             | referenced in a docstring; **not implemented**      |
| planners, cells runner         | existed in the deleted `chatbot_codex/` (git log S1-S9); gone |

## 2. Identified gaps (the failure mode)

Every domain agent (`pdb`, `literature`, `ml`, `scaps`) shares ONE generic loop with no
task-level state: no model of *required outputs*, *what is already established*, *what is
still missing*, and *whether the next call could possibly resolve it*. Consequences
observed on the Case 1 design query (trace `results/case_studies/case1_*.trace.jsonl`):

- `literature` re-searched the same terms and re-read the same DOI that it had already
  judged p-i-n / lacking numerical properties (17× SQL-equivalent repeats for pdb,
  6× same-paper reads for literature in the earlier baseline trace).
- `pdb` re-executed raw row-selection SQL instead of transitioning to
  aggregate - count - rank.
- The orchestrator restarted the same subtask with the same wording.
- The only existing guard is a *lexical* repeated-fingerprint block
  (`REPEAT_LIMIT = 2` in `agent.py`) - it cannot see "equivalent but re-typed" calls, and
  it knows nothing about whether the task was actually answered.

## 3. New progress-aware layer

A generic, opt-in contract attached to each domain agent. It sits between tool-call
generation and execution, and between result and the next step:

```
Agent.run
  build   SubtaskState(task)                  (per-domain requirement extraction)
  loop:
    GUARD(state, tool, args)   -> BLOCK_*      (before executing - stops loops)
    call tool
    ASSESS(state, call, result) -> ProgressAssessment
    no_progress counting                        (repeated no-gain calls)
    FINAL?  REVIEW(state)       -> accept or escalate (structured INCOMPLETE)
  returns report or structured diagnostic
```

Deterministic rules are preferred; the LLM is consulted only for what it already does
(choosing the next call). No extra LLM calls are added.

Per-domain contracts (all lightweight, keyword/result-shape based):

| agent      | requirements derived from task text | progress evidence from tool results |
|------------|--------------------------------------|--------------------------------------|
| pdb        | filter + aggregate categories (topN, mean, count, distribution, compare) | SQL text: GROUP BY / aggregate fn / COUNT / ORDER BY LIMIT; column filters present |
| literature | architecture (n-i-p / p-i-n), materials, property type, DOI quote | passages from read_paper contain the needed tokens; wrong-architecture paper -> REJECTED(doi) |
| ml         | acceptable tool set for the intent (predict / explain_global / explain_local / what_if / optimize) | which ml.* tools succeeded |
| scaps      | parameter + sweep vs single-point vs verify | get_sweep/run_sweep_live/sweep/run_point returned parsed data; converged; sane PCE |

Anti-loop is enforced at three levels:
1. within a turn-sequence (Agent.run): guard blocks identical **and** near-identical
   re-tooling that already produced a no-gain result; rejected DOIs cannot be re-read;
   raw-row SQL cannot repeat once aggregation is due;
2. no-progress counter: 2 consecutive no-gain-but-repetitive calls stop the subtask with a
   structured diagnostic (completed / missing / attempts / recommended next action);
3. global orchestrator: duplicate (agent,task) re-ASKs are detected and either blocked
   (synthetic "already tried, change the task" report) or forced to change, and escalation
   cycling is capped.

Provenance: every guard/assessment is written to the JSONL trace as
`signature`, `progress_status`, `newly_satisfied`, `still_missing`, `rejected`,
`retry_reason`, `escalation_reason` on existing events plus a `state` event on escalation.

## 4. Where the code lives

| file | change |
|------|--------|
| `perovagents/progress.py` (new) | `SubtaskState`, `ProgressAssessment`, `Contract` base, `PdbContract`/`LiteratureContract`/`MLContract`/`ScapsContract`, `MAX_NO_PROGRESS` |
| `perovagents/agent.py` | `Agent.run` wires guard/assess/review when a `contract` is passed; orchestrator duplicate-ASK detection |
| `perovagents/team.py` | each agent constructed with its contract |
| `perovagents/trace.py` | provenance fields |
| `perovagents/benchmarks/progress.py` (new) | Progress/Recovery benchmark (scripted, metric-reporting) |
| `perovagents/benchmarks/ablation.py` (new) | baseline / progress / progress+anti-loop / full comparison |
| `tests/test_progress.py` (new) | unit + scenario tests |