# AUDIT: current repo vs 4-cell agent architecture spec

Date: 2026-09-29. Scope: `perovagents/` package (Step 0 of the refactor).
No code was changed for this audit. Method: full read of `agent/`, `core/`,
`layers/`, `materials/`, `ml/`, `physics/`, `literature/`, `benchmarks/`,
`config.py`, `llm.py`, `__main__.py`, `tests/`, `frontend/app.py`.

## 0. Domain reading (confirmed)

- **PDB = Perovskite Database** (experimental device records), NOT the Protein
  Data Bank. Evidence: `config.PDB_CSV` → `data/perovskite_db.csv` (~43k device
  rows, 410 cols); `PDBTool` docstring "experimental Perovskite Solar Cell
  database (43k device records)"; `MATERIAL_COLUMN_MAP` (Spiro-MeOTAD, SnO2…).
- **PCE = power conversion efficiency** (%, `JV_default_PCE`). Evidence:
  `calculator.pce` ("PCE from Voc/Jsc/FF"), `PHYSICS_BOUNDS` PCE 0–30,
  `predicted_pce` in ML outputs.

## 1. Component mapping (spec → current code)

| Spec component | Current code | Verdict |
|---|---|---|
| Planner Agent (§2.1): Plan = Task{cell, request, params, depends_on}, subset of cells, pure planning | `agent/task.py:analyze_task` → TaskSpec; `agent/planner.py:LLMPlanner` (+`SYSTEM_TEMPLATE`); `agent/rules.py:RulePlanner` | **PARTIAL.** No cell concept, no `depends_on`, no dependency graph. Plans are flat `PlanStep(tool, action, params, purpose, expected_outputs)` lists (`agent/models.py`). No explicit "which cells" decision. Pure-planning holds for LLMPlanner (only `llm.chat`), RulePlanner embeds domain defaults but calls no tools. |
| Executor / Orchestrator (§2.2): RunState, concurrent dispatch, dependency passing, collect-validated-only, FailedResult surfacing | `agent/orchestrator.py:Agent.answer`, `agent/executor.py:Executor.execute_plan` | **PARTIAL.** Flat sequential step loop, no task statuses, no attempt counts (only plan-level `max_replans=2`), no parallel dispatch (zero concurrency infra outside `llm.py` lock), steps never receive prior results as input, failures recorded but execution continues and synthesis proceeds regardless (can silently drop). |
| Final synthesis from validated results + provenance (§7) | `orchestrator._synthesize`, `_answer_contract`, `_claims`, `_provenance_graph`, `store.snapshot` | **PARTIAL.** Rich provenance exists, but validated-only is prompt-level (`CLAIM_LANGUAGE`), not enforced; failed cells are not named (no FailedResult concept). |
| DomainCell base + Agent→Tool→Gate loop, Critique, max_attempts, history (§3) | — | **MISSING.** No cell abstraction. Retry exists only as plan-level replan with a feedback string, not a per-cell attempt loop with `Critique` objects. |
| Typed contracts Task/Plan/Attempt/Critique/ValidatedResult/FailedResult/RunState/VerificationReport (§5) | `Plan/PlanStep`, `StepResult/PlanResult`, `Verdict`, `ExecutionResult/ValidationBlock`, `ProvenanceRecord` | **PARTIAL.** Related but different semantics; the six spec types are missing. `pydantic 2.12.5` is installed but the codebase uses dataclasses throughout. |
| (a) PDB/SQL cell: SQL Tool, read-only, row/time limits, schema/units/shape gate (§4a) | `materials/pdb.py:PDBTool` (pandas) | **CONFLICT + PARTIAL.** No SQL anywhere in the repo (grep: zero hits for sqlite/sqlalchemy). Gate fragments exist scattered (`validation.py` bounds, `sanity.py` empty-table, `pdb._global_aggregate` bounds warn) but no schema validation, no SELECT-only enforcement, no limits (full CSV load). |
| (b) Literature cell chain + Evidence/Citation gate (§4b) | `literature/search.py`, `fulltext.py`, `extraction.py`, `doi.py`, `citations.py:CitationVerifier` | **PARTIAL.** Chain exists (search→corpus passages; fulltext; table/caption extraction; DOI resolve). Gate is crude: token-overlap support score only. No value-vs-source string check, no context (architecture/composition/setting) check, locator is DOI-only (hits do carry section/snippet — usable). |
| (c) ML cell ops + Model/OOD/Controllability gate (§4c) | `ml/predictor.py` (predict/explain_global/explain_local/what_if/optimize), `ml/controllability.py`, OOD in predictor | **PARTIAL.** Ops, OOD flag, controllability rejection and range clamping all exist and are good. Missing: attaching recorded validation performance to results (only model filename + intervals emitted; no metrics artifact in `results/ml/`), and a unified gate object. |
| (d) SCAPS cell + Convergence/Physical gate (§4d) | `physics/scaps.py` (`get_sweep` precomputed, `run_sweep_live` via wine) | **PARTIAL + CONFLICT.** `converged` detection exists for live runs, but `_summarize` hard-codes `convergence_ok: True` for precomputed sweeps. No S-Q bounds check, no FF/Jsc/Voc/PCE consistency check, no completeness check. |
| Config in one file, no hardcoded paths (§6) | `config.py` + env overrides | **PARTIAL.** Single module, mostly clean — except `SCAPS_WINE` hard-codes a personal path (`.../Age of Empires 2 Definitive Ed.app/.../wine64`). No `config.yaml`. Secrets (keys) handled via env/files, not logged. |
| Determinism: seeded + logged (§6) | seeds `42/7/123` (ML bags), `random_state=42` (pdb.sample) | **PARTIAL.** Seeded, but seeds are scattered literals, not centrally logged per run. |
| One structured run log, replayable (§6) | `ProvenanceStore` JSONL, `step_results`, claims chains | **PARTIAL.** No per-run log file unifying plan + dispatches + attempts + verdicts + critiques (attempts/critiques don't exist yet). |
| Errors as failed attempts, never crash (§6) | `Tool.execute` envelope (`status: error`), executor continues | **PARTIAL.** Envelope exists and runs survive, but nothing counts attempts or builds critiques. |
| Safety: read-only SQL, sandboxed SCAPS, no secrets in logs (§6) | wine subprocess (`cwd=scaps_dir`), provenance JSONL | **PARTIAL.** SCAPS runs in a fixed dir, not a per-run sandbox dir. Provenance logs full params (no secrets today, but no scrubbing rule). |
| Per-task timeout + parallel cells (§6, §9) | — | **MISSING.** |
| Prompts as versioned files (§6) | `SYNTHESIS_PROMPT` (orchestrator:77), `SYSTEM_TEMPLATE`/`LEGACY_TEMPLATE` (planner:31/114), `ANALYZER_PROMPT` (task:41), `CLAIM_LANGUAGE` (claims:58) | **MISSING as files.** All inline strings; zero prompt files in repo. |
| Per-cell unit tests + e2e (§9) | `tests/` (40 tests), `benchmarks/runner.py` (plan-level metrics) | **PARTIAL.** Good unit coverage of current semantics; nothing for cells/retry/parallel/dependencies. Benchmark harness measures old pipeline (action sequences), reusable in shape but not in substance. |
| Old entry point keeps working | `python -m perovagents` (`__main__.py`), `frontend/app.py`, `benchmarks/*` all call `Agent.answer` | **MUST PRESERVE.** Note pre-existing bug: `pyproject.toml` console script `perovagents.cli:main` points at a non-existent module (only `python -m` works). |

Tools with **no home in the 4-cell diagram** (spec §10 decision required):
`calculator`, `python_exec`, `symbolic` (equation fit), `material` (+`materials_project`),
`fulltext`, `extraction`, `doi`. Proposed home (see DECISIONS.md at build time):
cell-internal read-only helpers (e.g. ML agent resolves materials via
`material.lookup`; literature cell uses `fulltext`/`extraction`/`doi` as its
chain already does) plus executor-level utilities (`calculator`, `python_exec`,
`symbolic`). The gate still applies to each cell's main result.

## 2. What's missing (build list)

1. `DomainCell` base + per-cell subclasses; per-cell Agent→Tool→Gate→Critique→retry loop (`max_attempts=3`), attempt history fed back to agent.
2. Spec §5 models: `Task{cell,request,params,depends_on}`, cell-aware `Plan`,
   `Attempt`, `Critique`, `ValidatedResult`, `FailedResult`,
   `VerificationReport` (named checks), `RunState` (statuses, attempts, results, errors).
3. Parallel dispatch with dependency resolution + validated-output passing; per-task timeout.
4. SQL layer for PDB (read-only connection, row/time limits) + SQL schema gate.
5. Named gate checks per cell as pure, unit-testable functions:
   `verify/{sql_checks,evidence_checks,ml_checks,physics_checks}.py`.
6. ML: attach recorded validation performance to predictions (needs a metrics artifact; none exists in `results/ml/`).
7. SCAPS: S-Q bounds, output-consistency, completeness, real convergence confirmation (replace hard-coded `convergence_ok: True`).
8. Literature: value-vs-source and context checks (support/context may use LLM judgement per §10).
9. `config.yaml` (single file; `config.py` stays as defaults shim so old imports keep working).
10. `prompts/*.md` + loader (inline strings remain as fallback).
11. Per-run structured trace log (plan, dispatches, attempts, verdicts, critiques) + seed logging.
12. Synthesis v2: validated-only + per-statement provenance + failed-cell naming.
13. §9 acceptance tests: skeleton loop-once, retry-cap, parallel PDB+Literature, PDB→ML→SCAPS chain, gate pass/fail units, e2e per cell + multi-cell with trace log.

## 3. What's duplicated (consolidate during migration)

- **Physics bounds in 3 places**: `core/validation.py::validate_scientific`
  (errors+warnings) vs `core/sanity.py::SanityChecker` (warnings only) vs
  `materials/pdb.py::_global_aggregate` (warnings list). Same `PHYSICS_BOUNDS`,
  three semantics. → One named-check implementation in `verify/`, others delegate.
- **Material name resolution in 4 places**: `agent/rules.py::extract_material`,
  `agent/task.py::extract_materials`, `materials/properties.py::resolve_material`,
  `ml/features.py::MATERIAL_CANON` (+ `SYNONYMS` in properties.py). Tables can drift. → Single resolver.
- **Column aliasing**: `materials/pdb.py::ALIAS_COLUMNS` (+ `MATERIAL_COLUMN_MAP`)
  vs `config.PERF_ALIAS` — overlapping alias tables. → Single alias map owned by the SQL layer.
- **Claim-type knowledge in 2 maps**: `core/claims.py::TOOL_ACTION_CLAIM` and
  `agent/verification.py::TASK_ALLOWED_CLAIMS` must agree by hand. → Derive the
  verifier's allowed-set from the canonical map.
- **Provenance hashing twice**: `core/provenance.py::_hash` and
  `agent/orchestrator.py::_hash`. → Import one.
- **Three fallback layers**: `DeterministicResponder` (llm.py) → `RulePlanner`
  → `_keyword_spec` (task.py). Keep (defense in depth) but document the order.

## 4. What conflicts (needs a decision, default proposed)

| # | Conflict | Proposed resolution (confirm or override) |
|---|---|---|
| C1 | PDB is pandas; spec mandates SQL read-only. | Load `perovskite_db.csv` into SQLite (stdlib `sqlite3`, no new dep) at build/first-run; `sql_tool` enforces read-only URI, row + time limits. Keep `PDBTool` untouched until cell (a) passes acceptance. |
| C2 | Executor defaults to continue-after-failure; spec forbids silent drops. | New executor path collects `FailedResult`, surfaces it, marks dependents `skipped`. Old `Agent.answer` keeps current default until cutover. |
| C3 | `convergence_ok: True` hard-coded for precomputed SCAPS sweeps. | Precomputed sweeps get `convergence: "assumed-precomputed"` + provenance, never `True`; live runs keep real detection. Gate requires explicit confirmation. |
| C4 | 7 extra tools have no cell in the diagram. | Cell-internal helpers + executor utilities (see §1 table note). No 5th cell. |
| C5 | MODES A–G ablation + `benchmarks/` measure the old pipeline. | Keep running (regression net for old path); add new acceptance suite beside it; retire A–F only after cutover. |
| C6 | `TaskSpec` has 14 task_types; spec has 4 cells. | Routing table task_type → cell (many→one); `comparison`/`multi_tool_investigation` fan out to multiple cells. |
| C7 | pydantic installed but codebase is dataclasses. | Use dataclasses for new contracts (repo-consistent, zero new dep); revisit only if validation logic outgrows hand-rolled checks. |
| C8 | Spec §10: no re-planning after failures; results to Executor only. | Accept as stated: failed task → FailedResult → dependents `skipped`, run continues, synthesis names failures. Re-planning stays out unless you ask. |

## 5. Migration plan (ordered; repo runnable + tested after every step)

- **S1 — Contracts (no behavior change).** New `src`-equivalent module inside the
  package (proposed `perovagents/cells/schemas.py`; layout in §8 adapted — no
  `src/` move): Task/Plan(cell-aware)/Attempt/Critique/ValidatedResult/
  FailedResult/VerificationReport/RunState (+ task_type→cell routing table for
  C6). Unit tests for models only. Old models untouched.
- **S2 — Named gate checks (pure functions, unwired).** New `perovagents/verify/`
  (`sql_checks`, `evidence_checks`, `ml_checks`, `physics_checks`) operating on
  existing tool outputs; dedupe §3 items behind them. §9 gate pass/fail unit
  tests (unknown SQL column, PCE-as-fraction, number-not-in-source, OOD input,
  uncontrollable what-if var, PCE-above-SQ). Nothing calls them yet.
- **S3 — DomainCell base + skeleton.** Generic loop with injectable agent/tool/
  gate functions; dummy-cell test (§9: fail-once-then-pass → ValidatedResult;
  always-fail → FailedResult after `max_attempts`). No LLM agent yet.
- **S4 — SQL tool.** SQLite mirror + `sql_tool` (read-only, row/time limits);
  `PDBTool` keeps serving the old path. Tests per §9 gate list.
- **S5 — Four cell adapters** (`cells/pdb_sql.py`, `literature.py`, `ml.py`,
  `scaps.py`) wrapping existing tools + S2 gates, behind a flag (default off).
  Old path default. Per-cell e2e on one real query each.
- **S6 — Planner v2.** Cell-aware tasks + `depends_on` + subset selection
  (RulePlanner first; LLM prompt after; prompts move to `prompts/*.md` with
  inline fallback). Old planner default.
- **S7 — Executor v2.** RunState + ThreadPool dispatch + dependency passing +
  FailedResult surfacing + per-run JSONL trace + `config.yaml` (config.py as
  shim). Multi-cell e2e + trace-log test.
- **S8 — Synthesis v2.** Validated-only answers, per-statement provenance,
  failed-cell naming. Acceptance: one cited answer per cell + one multi-cell.
- **S9 — Cutover.** New entry (`python -m perovagents --cells` / v2 module) becomes
  default only after §9 acceptance passes; old `Agent.answer`, CLI, frontend,
  benchmarks keep working throughout; fix `perovagents.cli:main` console script
  as a drive-by.
- Throughout: `DECISIONS.md` updated per step; each step = one commit + tests.

## 6. Acceptance mapping (§9 → where it lands)

Skeleton → S3. Retry cap → S3. Parallel PDB+Literature → S7. PDB→ML→SCAPS chain
→ S7. Gate units → S2 (+S4 for SQL). E2E per cell → S5. Multi-cell cited answer
+ trace → S7/S8.

## 7. Open questions for you (blocking S1 details only)

1. Confirm C1–C8 defaults (especially C4 tool homes, C7 dataclasses, C8 no re-planning).
2. SQLite via stdlib for the PDB mirror — or do you want DuckDB/postgres?
3. New code location: `perovagents/cells/`, `perovagents/verify/`,
   `perovagents/control/`, `perovagents/prompts/`, `config.yaml` at
   `perovagents/` root — or a different layout?
4. LLM-gated judgements (lit support/context) may need API quota; ok to run those
   tests with recorded fixtures when keys are dead (current pool state: all
   17 keys cooling/invalid)?
