# FINAL REPORT — 4-cell agentic framework refactor (S1–S9)

For: reviewer / manager. Covers what was built, how it maps to the approved
architecture, test evidence, decisions and assumptions, honest findings, and
recommended next steps. Work is committed as one component per commit (see
`git log`); every step left the repo runnable with the old entry point
working.

## 1. Executive summary

The PerovAgents codebase (`perovagents/`) was refactored from a flat
tool-calling loop into the approved 4-cell architecture: **Planner →
Executor → four gated domain cells (PDB/SQL, Literature, ML, SCAPS-1D) →
validated-only synthesis**. Delivered in 9 incremental steps (S1–S9) plus a
Step-0 audit, each committed separately with tests. Final state: **134/134
tests pass** (40 pre-existing, 94 new) in ~3 min. The legacy path
(`Agent.answer`, CLI, frontend, benchmarks) is untouched and still default;
the new path runs via `python -m perovagents --cells`. Two specification
mandates required new fail-closed behaviour, both verified by tests ( §6).

## 2. Objective and scope

Implement the architecture in the approved spec (Planner Agent, Executor /
Orchestrator, four domain cells with Agent→Tool→Verification-gate loops,
typed data contracts, config/prompts/trace cross-cutting requirements, §9
acceptance tests) while: keeping the repo runnable at every step, keeping
the old entry point working, inventing no scientific behaviour, and
recording every judgement call in `DECISIONS.md` (D1–D15).

## 3. What was delivered (per step)

| Step | Content | New tests |
|---|---|---|
| S0 (docs) | `AUDIT.md`: component mapping, missing/duplicated/conflicts, migration plan | — |
| S1 | `cells/schemas.py`: Task (cell/request/params/depends_on), cell-aware Plan, Attempt, Critique, ValidatedResult, FailedResult, RunState, VerificationReport/CheckResult, task_type→cell routing (all 14 types) | 11 contract tests |
| S2 | `verify/`: named deterministic gate checks — `sql_checks` (select-only, known tables/columns, units incl. PCE-fraction rejection, shape), `evidence_checks` (citation resolvability, number-in-source, injectable support/context judges), `ml_checks` (model+validation attached, in-domain, controllability), `physics_checks` (S-Q bounds, JV consistency, IV monotonicity, completeness, explicit convergence) | 34 pass/fail tests incl. all six §9 gate examples |
| S3 | `cells/base.py`: generic DomainCell loop (agent/tool/gate slots, critique retry, `max_attempts=3`, FailedResult; exceptions become critiques) | 4 (§9 skeleton + retry cap) |
| S4 | `tools/sql_tool.py`: SQLite mirror of the 43,412-row PDB (inferred REAL/TEXT), read-only URI, SELECT-only, fetch-cap row limit, progress-handler timeout; deliberately unregistered so the old planner catalog is unchanged | 9 |
| S5 | Four adapters (`cells/pdb_sql|literature|ml|scaps.py`): deterministic agents + S2 gates; FF percent→fraction normalization; per-row band-gap for sweeps | 5 live e2e |
| S6 | `control/planner.py`: cell-aware tasks, subset selection (first cell only; fan-out for comparison/multi-tool), `depends_on` (rules: independent; LLM: parsed+validated), task-type/action alignment fix; `prompts/*.md` (6 versioned prompts + loader with inline fallback) | 14 |
| S7 | `control/executor.py`: ThreadPool dispatch, dependency payload passing, FailedResult surfacing + dependent-skipping, per-task timeout, per-run JSONL trace; `config.yaml` + loader over `config.py` defaults | 10 (§9 parallel + chain) |
| S8 | `control/synthesis.py`: deterministic validated-only answers, per-statement source ids, failed-cell naming, no-claims-on-empty | 4 |
| S9 | `control/app.py` (`answer_cells`), `--cells` CLI flag (additive), `cli.py` console-script fix, `test_acceptance.py` (multi-cell cited answer + trace replay) | 3 |

Total diff S1–S9: 56 files, +3369 lines. Old code modified in exactly two
places, both additive: `--cells` flag in `__main__.py`, new `cli.py` shim.

## 4. Architecture mapping (spec → implementation)

- §2.1 Planner → `control/planner.py` (`plan_rules`/`plan_llm`); pure
  planning, subset selection enforced and tested.
- §2.2 Executor → `control/executor.py` (`ExecutorV2`); RunState statuses
  pending/running/validated/failed/skipped; cells never communicate.
- §3 cell contract → `cells/base.py` (`DomainCell`); history-fed retries;
  `VerificationReport` per attempt.
- §4a–d cells → `cells/pdb_sql|literature|ml|scaps.py` + `verify/*_checks.py`.
- §5 contracts → `cells/schemas.py` (dataclasses, repo-consistent).
- §6 config → `config.yaml` + `control/config.py`; seeds logged in trace
  (`plan` event); per-run JSONL trace (`results/traces/run_*.jsonl`);
  exceptions→attempts; SCAPS precomputed path reads local CSV (live runs
  unchanged); prompts in `prompts/*.md`.
- §7 synthesis → `control/synthesis.py` (deterministic templates).
- §8 layout → adapted: `cells/`, `verify/`, `control/`, `tools/`,
  `prompts/` inside the existing package (no `src/` move).

## 5. Test evidence (134/134, ~190 s)

| Suite | n | What it proves |
|---|---|---|
| test_smoke / llm_rotation / intent_recovery / frontend_smoke (pre-existing) | 40 | old pipeline unregressed |
| test_cell_contracts | 11 | §5 model shapes, report logic, dependency readiness, routing coverage |
| test_verify_gates | 34 | each named check passes + fails; all six §9 gate examples |
| test_cell_loop | 4 | §9 skeleton (fail-once→pass, 2 attempts) + retry cap (3 attempts → FailedResult) + exception-as-critique |
| test_sql_tool | 9 | mirror build, SELECT, unknown column, INSERT rejected, truncation, timeout, gate wiring |
| test_cells_e2e | 5 | live PDB/literature validate; ML + SCAPS honestly fail closed (see §6) |
| test_planner_v2 | 14 | subset selection, fan-out, depends_on, LLM fallback, prompt files |
| test_executor_v2 | 10 | thread-overlap concurrency proof, real PDB+Literature parallel, PDB→ML→SCAPS chain order, fail→skip, timeout, utility, trace replay, config |
| test_synthesis | 4 | validated-only, citations, failed naming, determinism |
| test_acceptance | 3 | multi-cell cited answer + complete trace; entry points intact |

§9 mapping: skeleton→S3, retry cap→S3, parallel→S7, dependency chain→S7,
gate units→S2/S4, per-cell e2e→S5, multi-cell cited answer + trace→S9.

## 6. Honest findings (gates catching real issues)

1. **ML predictions ship with no recorded validation performance.** No
   metrics artifact exists anywhere (`results/ml/` has models/intervals/
   manifests only; `catboost_info/` is empty). Per spec §4c the gate fails
   closed: every live ML run returns FailedResult naming `model_attached`
   until group-aware metrics are recorded. No scores were invented.
2. **Precomputed SCAPS sweeps carry no solver evidence.** `_summarize`
   hard-codes `convergence_ok: True`; the gate rejects assumed convergence,
   so precomputed sweeps fail naming `converged`. FF percent→fraction
   normalization was required and is now recorded in provenance.
3. **S-Q cap table is approximate** (embedded, upper-bound use only,
   boundary values untested) — replace with a citable tabulation before
   publication (DECISIONS.md D8).
4. **Bare "what if" fell through to `predict`** while the task said
   sensitivity_analysis — fixed by task-type/action alignment (S6).
5. **`perovagents.cli:main` never existed** (only `python -m` worked) —
   fixed with a thin `cli.py` shim.
6. **Key pool recovered mid-project** (8/17 alive on re-probe); LLM-judgement
   tests can run live, fixtures remain for offline CI.

## 7. Decisions and assumptions (§10 answers + conflicts)

Full log in `DECISIONS.md` (D1–D15). Headlines: dataclasses over pydantic
(C7); `"utility"` pseudo-cell for helper tools, no 5th cell (C4);
stdlib-sqlite mirror (C1); failed tasks surface + dependents skip, no
re-planning, results to Executor only (C8/§10 as specified); deterministic
gates with injectable LLM judges for literature support/context (§10);
timed-out workers abandoned not killed; deterministic template synthesis
(no LLM narration); cutover flip needs explicit approval.

## 8. How to run

- Legacy (default, unchanged): `python -m perovagents "question"`,
  `frontend/app.py`, `benchmarks/*`, all 40 old tests.
- New path: `python -m perovagents --cells "question"` (rules planner;
  `--pipeline llm` when keys work) → plan → parallel gated cells →
  cited answer + `results/traces/run_*.jsonl`.
- Tests: `pytest tests/ -q` (134 pass, ~3 min).

## 9. Recommended next steps

1. Record group-aware ML validation metrics → `results/ml/validation.json`
   (unblocks cell c; honest computation, not invention).
2. Replace the approximate S-Q table with a citable tabulation.
3. Optional LLM narration layer over deterministic synthesis.
4. Approve the cutover flip (new path as default) and retire MODES A–F.
5. Live SCAPS runs where wine exists (precomputed path stays for CI).
