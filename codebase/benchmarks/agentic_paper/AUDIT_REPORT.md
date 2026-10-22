# Diagnosis of the first benchmark batch (no new inference)

Snapshot taken at 2026-10-21, after stopping the two background jobs (gemini/minimax run and the
sleeping deepseek job). Raw data, manifest, logs and the earlier plots are preserved unchanged in
`audit/snapshot_pre_diagnosis/`. Everything here is reproducible with
`python -m benchmarks.agentic_paper.audit` (reads files only; the frozen scorer, runner and questions are untouched).

## 1. What the 67 attempts actually are

* **Not a smoke test.** The smoke test (`runs/smoke_*`) is separate and is not in the scored set. These are the
  first episodes of the main pass (`--stage pass --reps 3`), run rep-major in question order, **repetition 0 only**.
* Manifest: 67 attempt rows = 47 valid + 12 infra-failed + 6 no-model-output + 2 provider-limited.
  The scored file held 53 (and the first plots 48) because `refresh` had last run before more episodes finished.
* 23 unique questions attempted (Q01–Q12, Q17–Q19, Q21–Q28). **No SCAPS question (Q33–Q40) and no
  Q29–Q32/Q41–Q45 were reached**, and **DeepSeek has zero episodes** (OpenCode free quota exhausted until the
  2026-10-22 00:00 UTC reset). Missing bars are therefore missing data, not zeros.
* Plan: 40 questions x 3 backbones x 2 conditions x 3 reps = 720. Observed attempts 67 (9.3 %), so 653 remain.
  Per-cell coverage: `audit/coverage_matrix.csv`; per-episode validity: `audit/episode_validity.csv`.

## 2. Defects found

| # | Finding | Effect | Action |
|---|---|---|---|
| D1 | **Empty model output scored as agent failure.** 6 gemini-direct episodes (Q01,Q03–Q06,Q08) have 0 reply characters over 18–27 calls and 0 tool calls (2 more, Q02/Q07, got one tool call then only empties). They ended `timeout` and were scored completion=0, tool-F1=0. | 8 of 12 gemini direct scored episodes are not evidence about reasoning; gemini direct was shown at 0.33 completion. | New validity class `no_model_output`; reported separately, primary metrics exclude them, sensitivity keeps them as failures. |
| D2 | **Conditional exclusion of infra failures is condition-dependent.** Minimax PerovAgents: 11 of 19 attempts infra-failed (150 s provider call timeouts); minimax direct 1 of 23. Excluding them leaves only fast episodes. | 8 minimax pairs under the legacy rule are a survivor subset. | Sensitivity B (non-valid = failure) reported next to the primary result. |
| D3 | Wall-clock `timeout` with provider call timeouts counted as agent timeout (minimax PerovAgents Q11, Q12). | Provider latency blamed on the framework. | New class `provider_limited`. |
| D4 | Stale scored/table files vs manifest; figures showed legend duplicates, empty groups, no n, no distinction of failure types. | Presentation. | `audit.py` recomputes from raw episodes; new figures in `figures_audit/` (old ones kept). |
| D5 | Raw replies and finish reasons are not stored. | Cause of D1 cannot be established from the data. | Not changed (frozen provider). Needs a logging addition before re-running (see next steps). |

Checked and **not** defective: question-ID joins (episode id = `qid__backbone__condition__rep`, rep 0 only, paired by
question); tool-name normalisation (both conditions go through the same `ToolRecorder` and canonical action taxonomy;
unknown tool names are counted); correctness is a subset of completion; partial credit is separate; the three failures I
read in full (below) are genuine answers, not parser rejections; model, route and settings are identical across the two
conditions of a backbone (strict single backbone, temperature 0, no fallback; same `collect_tools()` catalogue; same
12-call/12-turn/1200 s budgets).

## 3. Fairness of the matched comparison

* Same tools and data, same backend per backbone, no fallback: **yes**.
* Known, documented asymmetry: PerovAgents also allows 12 turns per sub-agent and 5 orchestrator rounds, and its
  call count is exposed to provider timeouts many more times per episode (D2). OpenCode wraps the model in its own
  agent prompt, which can interact with the orchestrator's `ASK` protocol (see M2).
* Paired comparisons are valid **only on the valid pairs**: 6 questions for gemini and 6 for minimax (rep 0). That is
  too few for conclusions; intervals are shown only for n >= 5.

## 4. Traced episodes (raw query -> trace -> answer -> reference -> metric -> plot)

1. **Success** `Q01__gemini__perovagents__r0` (Database): orchestrator asked pdb; one `sql` call; answer "15.31 % (4,299
   devices)". Checks `count` (ref 4300, 5 % tol) pass, `mean_pce` (ref 15.303) pass -> completed=1, correct=1, F1=1.0.
   Enters Fig 1 via the paired question-level mean (gemini PerovAgents, valid pairs only).
2. **Contract false positive** `Q04__gemini__perovagents__r0`: pdb obtained the 5 correct ranked rows (`ORDER BY ... LIMIT 5`,
   `n_rows: 5`), but the progress contract demanded `aggregate` (GROUP BY) for the ranking, marked the call
   `BLOCKED`, then `INCOMPLETE` twice; the orchestrator answered "could not be established". Scored incomplete (checks
   `top_pce`, `htl` fail). Genuine framework behaviour, not a scoring defect: the data was in hand and discarded.
3. **Information loss / escalation** `Q05`/`Q08__gemini__perovagents__r0`: literature agent reported inability, orchestrator
   declined ("no verified literature quotes"); `doi_evidence`, `strategy`, `mechanism` checks all absent. Genuine
   decline, scored as incomplete for an answerable Q05 and (correctly) incomplete for Q08.
4. **No dispatch** `Q03`/`Q05__minimax__perovagents__r0`: orchestrator replied "Dispatched the pdb agent ..." in prose
   with no `ASK pdb:` line; that reply is treated as the final answer, 0 tool calls, F1 = 0. Prompt-following (and
   possibly wrapper) failure, correctly scored as a failure.
5. **Unscorable-as-reasoning** `Q01__gemini__direct__r0`: 21 calls, 0 reply characters, 1381 s, empty answer ->
   `no_model_output` (D1). The legacy scorer reported completion 0, correct 0, F1 0.

## 5. Observed differences (rep 0; paired, valid episodes; full table `audit/metrics_v2.csv`)

| backbone | n pairs | completion D -> P | correctness D -> P | tool F1 D -> P |
|---|---|---|---|---|
| gemini-3.5-flash-lite | 6 | 0.67 -> 0.83 (diff +0.17, CI 0.0 to 0.5) | 0.67 -> 0.67 (0.0, -0.5 to 0.5) | 0.83 -> 0.93 (+0.10, 0.0 to 0.29) |
| minimax-m3.1-flash | 6 | 0.83 -> 0.33 (-0.50, -0.83 to -0.17) | 0.67 -> 0.33 (-0.33, -0.83 to 0.33) | 1.00 -> 0.33 (-0.67, -1.0 to -0.33) |
| deepseek-v4.1-flash | 0 | no data | no data | no data |

D = direct tool use, P = PerovAgents. Sensitivity (non-valid attempts count as failures, 19 minimax pairs):
correctness 0.42 -> 0.11 (diff -0.32, CI -0.58 to -0.05). The sign is the same under every rule for minimax; for gemini
the sign of the correctness difference is not established (CI contains 0 under all rules). **The two backbones disagree,
and 6 pairs per backbone cannot separate "PerovAgents is worse" from "PerovAgents is not different".**

## 6. Mechanisms supported by the traces (hypotheses tested)

* **Over-strict progress contract / escalation (supported, gemini):** 6 of 13 valid PerovAgents episodes escalated
  (BLOCKED/INCOMPLETE); **0 of 6 escalated episodes were correct vs 5 of 7 non-escalated** (`audit/hypotheses.csv`).
  Q04 shows the contract rejecting a correct top-k ranking.
* **Orchestrator without dispatch (supported, minimax):** 4 of 6 valid minimax PerovAgents episodes made no `ASK` at all.
* **Provider instability (supported):** 11/19 minimax PerovAgents attempts hit 150 s call timeouts; gemini direct returned
  empty replies on 8 of 12 episodes while PerovAgents on the same key pool at the same time received normal replies.
* **Excess calls (not supported):** valid gemini PerovAgents used 10.4 model calls vs 12.8 for direct; minimax 3.0 vs 6.5.
* **Scoring/parsing defects (not found)** in the episodes read; see section 2.
* **Tool availability mismatch (not found):** identical catalogue.

## 7. Decision on resuming

**Not resumed.** The first batch is valid only in part (D1–D3), and the cause of the gemini-direct empty replies is not
observable because raw replies/finish reasons are not logged. Re-running under the unchanged frozen config would keep
adding episodes that are dominated by these confounds. The existing 67 attempts, the manifest and the frozen
definitions are intact, and resuming (skipping every completed cell) is one command once the points below are decided.

Next steps required (need your decision because they touch the frozen provider file):
1. Add logging of raw replies and Gemini `finish_reason` (diagnostic only, no behavioural change), then re-freeze
   (freeze_id 3). Existing episodes stay as they are and are reported under the freeze they were produced with.
2. Decide the policy for provider failures: keep `infra_failed` re-queueing (current), and treat repeatedly empty
   direct episodes the same way (re-queue instead of scoring).
3. Resume: `./benchmarks/agentic_paper/start_opencode_server.sh &` then
   `python -m benchmarks.agentic_paper.run --stage pass --reps 3 --backbones gemini minimax` (deepseek after the
   2026-10-22 00:00 UTC reset). Minimax per-call timeout (150 s) is the main source of infra failures.
4. Then `python -m benchmarks.agentic_paper.audit` for corrected metrics and figures.
