# Benchmark v3: corrections, instrumentation and bounded smoke test

v2 (first batch, 67 attempts) is closed and preserved unchanged (`runs/`, `scored/`, `tables/`, `figures/`,
`audit/`, `audit/snapshot_pre_diagnosis/`, `legacy_v2/`). v3 writes only under `v3/` and has its own freeze
(`FREEZE_v3.json`, freeze_id 2). **v2 and v3 results must not be pooled.** No full benchmark was run.
Nothing outside `codebase/` was modified; the manuscript was not touched.

## Scope change requested during this task
Every question that needs SCAPS is out of the active set for now: the whole SCAPS category (Q33-Q40) plus
Q43 and Q45 (they ask for SCAPS values). They stay in `questions/questions_v3.json` with `set: deferred_scaps`;
`questions_v2.json` is unchanged. Active set: **30 questions** (database 8, literature 8, ML 8, multi-tool 6),
planned total 30 x 3 backbones x 2 conditions x 3 reps = **540 episodes** (was 720).

## 1. Confirmed causes and fixes (framework, `perovagents/`)

| # | Confirmed cause (evidence) | Fix | Tests |
|---|---|---|---|
| 1 | **Contract regex bug.** `vs\.?` had no word boundary and matched inside "per*ovs*kite", so *every* pdb task demanded an aggregate statistic; a correct `ORDER BY ... LIMIT` ranking was rejected as "raw rows" (Q04 trace). | Word-bounded patterns. New *row-ranking* requirement (`highest-PCE devices`, `top 5 devices`) satisfied by an ordered, limited SELECT that returns rows; aggregation still required for statistics/category rankings (`mean`, `count`, `compare ... vs ...`, `top 3 HTL materials`). | 10 |
| 2 | **Prose treated as a final answer.** MiniMax replied "Dispatched the pdb agent ..." with no `ASK`; the orchestrator returned it as the answer. | `parse_orchestrator_reply` validates the grammar: valid = `ASK <agent>:` line(s) or `FINAL ANSWER:` with a body. Malformed -> up to 2 bounded re-prompts with a FORMAT ERROR notice (no round consumed), then a clear `[FORMAT ERROR]` diagnostic. | 7 |
| 3 | **Information loss on escalation.** Q04: five correct rows were fetched, then discarded by a BLOCKED/INCOMPLETE diagnostic. | Escalation reports carry the last two non-error tool results ("EVIDENCE GATHERED SO FAR"); orchestrator prompt says to answer from them. | 2 |
| 4 | **Second contract defect found in the smoke test.** `property:w` was created from the DOI tail `...03285-w` (and `eV\b` matched words ending in "ev"). | DOIs removed before property detection; bare "W" dropped; `\beV\b`. | 2 |
| 5 | **Large tasks.** (requested) | Agents: +6 turns when the task is long (>=600 chars) or has >=3 requirements. Orchestrator: +2 rounds for long questions (>=400 chars) or once >=3 distinct agents were asked. Prompt: output-grammar rule, ORDER BY...LIMIT is enough for device rankings. Budgets are written to every trace/episode. | 2 |

Documented asymmetry (unchanged in kind, slightly larger): PerovAgents has per-agent and per-round budgets in
addition to the shared 12 tool calls / 12 turns / 1200 s wall clock. The direct baseline got no extension.

## 2. Gemini empty direct replies: diagnosis

Instrumentation (`providers_instrumented.py`, benchmark-side subclass; `providers.py` is untouched) records per
call: model id, status, finish reason, block reason, candidates, content-part kinds, text length, response id,
model version, latency, retries/attempts inside the call, usage incl. thinking tokens, redacted exceptions, key-pool
slot number (never a key). Non-empty replies are stored; prompts are not. Verified by test and by a live call that no key appears.

Evidence (all from recorded metadata):
* All 7-30 empty replies per direct episode have `finish_reason = MALFORMED_FUNCTION_CALL`, one non-text part,
  no text, no completion tokens. **No tool is declared in the request**, so Gemini 3.5 Flash-Lite itself tried to emit a
  native function call that the API refused to parse.
* It is deterministic at temperature 0 (8/8 identical probe calls) and **not** fixed by
  `tool_config.function_calling_config.mode = NONE` (4/4 still malformed; option kept but off by default).
* **Trigger = the baseline's system wording.** With the same 17-tool catalogue: baseline wording -> malformed; baseline
  wording + catalogue without signatures -> malformed; agent-style wording ("reply with a tool call as JSON only ...")
  -> `STOP` and a valid JSON call (2/2). PerovAgents' agent prompts mostly avoid it (3/12 and 2/13 calls malformed,
  recovered).
* The `infra_failed` status on the four smoke direct episodes was a *consequence*: identical deterministic retries
  (3 per turn) produced per-minute 429s (30-36 events, `daily=false`) until the keys cooled.
* Classification (validity.py): replies are examined first. `provider_filtered/anomaly/no_candidate` ->
  infra-invalid; `malformed_function_call/empty_stop/thinking_only/token_limit` -> **valid empty completion**
  (ordinary failure, kept in denominators, not re-queued); anything else -> **unclassified empty** (kept, reported
  separately, retried only with `--retry-unclassified`). Pure provider failures (no usable replies) -> infra-invalid,
  replaced up to `--max-attempts` with every attempt kept (`episode.attempt<n>.json`) and linked by `attempt`/`replaces`.

**Open decision (not made unilaterally):** the direct baseline wording is incompatible with this model. Options:
(a) keep the baseline and report Gemini direct as "protocol-incompatible" (all-malformed) with its count;
(b) adopt agent-style tool-call wording for the baseline (same grammar as the framework's agents; baseline arguably
stronger, but it changes the DirectBaseline definition); (c) drop Gemini 3.5 Flash-Lite as a backbone for the direct
condition. Until chosen, Gemini direct results will be mostly valid-empty-completion failures and must not be read as
a framework win.

## 3. Smoke test (Gemini 3.5 Flash-Lite, free key pool, 8 episodes, `v3/audit/smoke_summary.csv`)

| question | direct | PerovAgents |
|---|---|---|
| Q01 | valid empty completion (10/10 replies malformed), 0 tools | correct, 2 tools, 5 calls |
| Q04 | valid empty completion (30/30 malformed) | completed, 4 tools, **ranking satisfied by one ORDER BY...LIMIT, no BLOCKED**; top-PCE check failed |
| Q05 | valid empty completion (7/7) | correct, 4 tools |
| Q18 (multi-tool) | valid empty completion (25/25) | correct, 6 tools, F1 0.86 |

(The direct rows were labelled `infra_failed` in the smoke manifest because that run used the first classifier; the
audit re-derives validity from the stored call log, which gives the labels above.)
Q04: the agent retrieved the 35.9 % row, flagged it as a probable data artifact (Jsc ~0.1 mA/cm2) and reported the
realistic devices; the reference expects 35.9 as the top value. This is a **question/reference design issue**, not a
framework defect; references were not changed. Verified: raw traces and per-call logs present, model id
`gemini-3.5-flash-lite` on every call, no fallback, denominators in `v3/audit/smoke_summary.csv`.
Not exercised: DeepSeek (OpenCode free quota resets 2026-10-22 00:00 UTC) and the MiniMax orchestrator fix against the
live model (covered by deterministic tests only). Smoke results are development evidence, not benchmark results.

## 4. Verification
Tests: 120 passed, 7 skipped, 1 failed (42 of the passing tests are new). The failure (`test_scaps_design::test_generated_def_matches_reference_device_numerically`,
`HTL.Nt` 1e20 vs 1e23) is in SCAPS design code this task did not touch. New tests: `tests/test_v3_fixes.py`,
`tests/test_instrumentation.py` (42 tests: contract, grammar, evidence carry-over, budgets, redaction, empty-reply
diagnosis, validity classes, retry policy, manifest linking). Selfcheck: OK for 30 active questions. Freeze v3 verified.

## 5. Splits and reporting rules
`config/split_v3.json`: development = the 25 questions run/inspected in v2 (these exposed the defects); **held-out =
Q30, Q31, Q32, Q41, Q42, Q44** (never run, never inspected; ML and multi-tool only, because all database and
literature questions were exposed in v2, so their v3 results are development-set results). Quality metrics use
infra-excluded denominators and always print them; operational completion counts every attempt
(`audit` -> `metrics_v2.csv`: `primary_infra_excluded`, `sens_also_excluding_unclassified_empty`,
`operational_all_attempts_nonvalid_is_failure`).

## 6. Next batch (exact)
Decide section 2 first (any change to the direct wording requires re-freezing: `python -m benchmarks.agentic_paper.freeze`).
Then, from `codebase/`, with the anaconda Python:
```bash
# held-out confirmatory batch, 1 rep, 12 episodes, Gemini only (keep workers low: free-tier RPM)
BENCH_VERSION=v3 /opt/anaconda3/bin/python -m benchmarks.agentic_paper.run --stage pass --reps 1 \
    --backbones gemini --questions Q30,Q31,Q32,Q41,Q42,Q44 --workers 2 --max-attempts 3
BENCH_VERSION=v3 /opt/anaconda3/bin/python -m benchmarks.agentic_paper.audit
# DeepSeek after 2026-10-22 00:00 UTC:  ./benchmarks/agentic_paper/start_opencode_server.sh &
#   ... run --backbones deepseek (same flags)
```
Do not start the full 540-episode pass until the held-out batch and the section-2 decision are reviewed.
