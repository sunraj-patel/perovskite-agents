# Benchmark v4: neutral direct-baseline protocol, new held-out questions, bounded Gemini/DeepSeek runs

v2 and v3 are closed and preserved (`runs/`, `v3/`, `legacy_v2/`, `legacy_v3/`; the legacy_v3 copies hash-match
`FREEZE_v3.json`). v4 writes only under `v4/` with its own freeze (`FREEZE_v4.json`). **Never pool v2, v3 and v4.**
Nothing outside `codebase/` was touched; no manuscript file was read for editing, compiled or regenerated; no SCAPS run.

## 1. Direct-baseline protocol (the blocker)
`baseline_v4.py` (`benchmarks/harness.py` unchanged). Same loop, budgets (12 tool calls / 12 turns / 1200 s), history
truncation and empty-reply redelivery; only the system prompt differs. It states the tool catalogue (with signatures),
the JSON call syntax, "one call per reply, JSON as ordinary text", and `FINAL ANSWER:`. It contains nothing from
PerovAgents (no agent prompts, orchestrator, contracts, escalation, evidence carry-over, decomposition); a test asserts this.
Applied to **every** backbone in v4.

Wording selection used first-move probes of gemini-3.5-flash-lite on development questions only (Q01 etc. style, never
a held-out item), 6 calls per variant: v3 wording 0/6 non-empty; "How to call a tool" wording 0/6; "Each turn, reply with
EITHER ... / write the JSON as ordinary text" 6/6. **Residual:** this model still returns `MALFORMED_FUNCTION_CALL` on some
turns (smoke: 6 of 38 direct calls; redelivery recovers them). They are recorded per call (`empty_reply_classes`), the
episodes are labelled by evidence, and none was re-queued as infrastructure.
The old v3 Gemini empty completions are diagnostics of the previous protocol and are not outcomes in v4.

## 2. Preserved framework fixes
Q04 ranking/aggregation contract, literature DOI/property fix, escalation evidence carry-over, orchestrator grammar and
budgets: unchanged (`framework_*` hashes identical to v3 freeze). Tests: **123 passed, 7 skipped, 1 failed**
(3 new: `tests/test_baseline_v4.py`). The failure, `test_scaps_design::test_generated_def_matches_reference_device_numerically`
(`HTL.Nt` 1e20 vs 1e23), concerns SCAPS `.def` generation (`perovagents/physics/scaps_design.py`); nothing in the
benchmark imports it (grep of `benchmarks/`), live SCAPS is disabled and SCAPS questions are deferred. Not modified.
Run tests with `PEROVAGENTS_PDB_SQLITE`, `PEROVAGENTS_SCAPS_DIR`, `PEROVAGENTS_CORPUS_DIR` exported (as in `paths.py`).

## 3. Questions and split (`config/split_v4.json`, `questions/questions_v4.json`)
v3 active set (30) + six new unseen questions **Q46-Q51**, authored from the database and corpus before any model saw them
(`build_questions_v4.py`; references in `reference/reference_values_v4_additions.json`; SQL stored; literature numbers asserted
to occur in the paper text; selfcheck oracle/wrong-answer/no-tool checks pass for all 36 questions).
Q46 count+mean PCE (p-i-n NiO-c), Q47 mean Voc (p-i-n PTAA), Q48 top-3 compositions (p-i-n, PCE>20), Q49-Q51 papers
10.1002/adfm.201604733 (Voc 1.46 V), 10.1007/s10118-017-1891-z (12.68 vs 9.49 %), 10.1021/jp5126624 (Sn vacancies).
Confirmatory held-out = Q30,Q31,Q32,Q41,Q42,Q44 (ML, multi-tool) + Q46-Q51 (database, literature) = 12 questions.
Everything else is development. Protocol smoke used development questions Q01, Q05, Q18 only. Held-out answers were not
inspected to change prompts, contracts or scoring. SCAPS questions stay deferred (Q33-Q40, Q43, Q45).
Caveat: 3 held-out questions per category; intervals are wide and shown only for n>=5 paired questions.

## 4. Freeze
`FREEZE_v4.json` freeze_id **2** (id 1, kept as `legacy_v3/FREEZE_v4_id1_before_audit_change.json`, was written before the
smoke; id 2 differs only in `analysis_audit`: attempts table + first-attempt operational analysis added, no data or scoring
change). Hashes cover questions, v4 reference additions, scorer, models, episode runner, run policy, providers, instrumentation,
validity, baseline prompt, `harness.py`, split and framework agent/progress/team. `freeze --verify` -> ok.

## 5. Smoke results (development questions, `v4/audit/smoke_summary.csv`)
| backend | direct | PerovAgents |
|---|---|---|
| Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite` on every call) | 6/6 episodes valid; Q01, Q05 correct; Q18 turn-budget (12 tool calls, no final answer: genuine failure) | 3/3 valid and correct |
| DeepSeek V4.1 Flash (`opencode/deepseek-v4.1-flash-free`) | 3 valid, Q05 correct; Q01 wrong; Q18 answered with 0 tool calls (model declined tools) | 3/3 valid and correct |
| MiniMax | **not tested**: shares the OpenCode free quota, which is exhausted until 00:00 UTC | - |
Provider metadata verified in raw call logs: model id, finish reasons, usage and response ids recorded; no key in any log.

## 6. Held-out results so far (1 repetition; `v4/audit/`)
Gemini: 24/24 cells have a valid final attempt. Three direct episodes (Q32, Q41, Q42) first hit per-minute 429s
(`daily=false`, pool exhausted); they were classified `infra_invalid`, kept as `episode.attempt1.json`, and replaced
by linked attempt 2 (resumed with 1 worker). All 12 + 12 episodes valid.
| Gemini (12 paired questions) | direct | PerovAgents | diff (95 % CI, paired bootstrap over questions) |
|---|---|---|---|
| correct, infra excluded (primary) | 0.667 | 0.750 | +0.083 (-0.333, +0.500) |
| completed | 0.750 | 0.917 | +0.167 (-0.167, +0.500) |
| tool F1 | 0.856 | 0.956 | +0.100 (-0.055, +0.306) |
| correct, end-to-end first attempt (infra = 0) | 0.583 | 0.750 | +0.167 (-0.333, +0.583) |
The Gemini difference is not distinguishable from zero at n=12. No claim of a framework win is supported.
DeepSeek: Q30, Q31, Q32, Q41 valid in both conditions; Q42 and Q44 (4 episodes) are `infra_invalid`
("Free usage exceeded", reset 2026-10-22 00:00 UTC). Quality denominator 4 questions per condition (direct 0.75, PerovAgents
1.00 correct): far too few for any inference; operational view counts the 4 as failures (0.50 vs 0.667). Stage 2 for DeepSeek
(Q46-Q51) has NOT run. No fallback to another provider was used.

## 7. Coverage and validity
`v4/audit/coverage_matrix.csv`, `episode_validity.csv` (latest attempt per cell), `attempts.csv` (every attempt, validity,
`replaces`), `metrics_v2.csv` (five analyses incl. `operational_first_attempt_nonvalid_is_failure`), `summary.json`:
36 cells, 32 valid, 4 infra (DeepSeek). Raw traces: `v4/runs/episodes/*`, manifest `v4/runs/manifest.jsonl`.
Planned full v4 pass is 36 questions x 3 backbones x 2 x 3 reps = 648 episodes; not launched. Gemini free-tier budget
observed: ~12 episodes per stage at 2 workers triggered per-minute 429s on direct episodes, so use 1-2 workers.

## 8. Resume (after 2026-10-22 00:00 UTC, from `codebase/`)
```bash
export PYTHONPATH=. BENCH_VERSION=v4
# (server already running on :4096; otherwise ./benchmarks/agentic_paper/start_opencode_server.sh &)
/opt/anaconda3/bin/python -m benchmarks.agentic_paper.run --stage pass --reps 1 --backbones deepseek \
   --questions Q30,Q31,Q32,Q41,Q42,Q44 --workers 2 --max-attempts 3      # retries Q42,Q44 only
/opt/anaconda3/bin/python -m benchmarks.agentic_paper.run --stage pass --reps 1 --backbones deepseek \
   --questions Q46,Q47,Q48,Q49,Q50,Q51 --workers 2 --max-attempts 3
/opt/anaconda3/bin/python -m benchmarks.agentic_paper.audit
# then a bounded MiniMax smoke only if it responds without payment:
/opt/anaconda3/bin/python -m benchmarks.agentic_paper.run --stage smoke --backbones minimax --workers 1 --max-attempts 1
```
Repetitions 2-3 (`--reps 3`) only after the held-out stage is reviewed and the quota is known; the runner is resumable
and stops a backbone on repeated infra failures. Do not change prompts, contracts or scoring after seeing held-out results
without a new freeze id and a new version.
