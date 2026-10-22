# DECISIONS.md — choices where the spec is silent (or conflicts)

Recorded per the refactor brief: simplest option wins; each entry lists the
decision, the reason, and where it is implemented/tested.

## D1 — Dataclasses, not pydantic, for cell contracts (S1)
Spec §5 allows "pydantic or dataclasses". The repo is dataclasses throughout
(`agent/models.py`, `core/tool.py`); pydantic is installed but unused.
Decision: stdlib dataclasses with `to_dict()` (repo convention).
Where: `perovagents/cells/schemas.py`. Tests: `tests/test_cell_contracts.py`.

## D2 — "utility" pseudo-cell for helper tools (S1)
Spec diagram has exactly 4 cells, but 7 tools have no home (calculator,
python_exec, symbolic, material, materials_project, fulltext, extraction,
doi). Decision: `Task.cell` accepts `"utility"` meaning "executor runs a
helper tool directly, no verification gate". `literature` cell uses
fulltext/extraction/doi internally as its chain already does; ML agent may
resolve materials via `material.lookup` inside its step. No 5th cell.
Where: `CELLS` / `CELL_AND_UTILITY`, routing table. Tests: routing coverage.

## D3 — task_type → cell routing table (S1)
14 TaskSpec types → 4 cells: `comparison` and `multi_tool_investigation` fan
out to all four; `sensitivity_analysis` prefers `ml`, `scaps` second;
`calculation` → `utility`; `verification` → `literature` (citations.verify).
Where: `TASK_CELL_ROUTES`, `cells_for()`. Tests: every `TASK_TYPES` entry
routes somewhere.

## D4 — SQLite via stdlib for the PDB mirror (S4, pre-approved)
No SQL exists in the repo (pandas-only). Decision: build a SQLite mirror of
`perovskite_db.csv` with stdlib `sqlite3` (no new dependency); read-only URI
plus row/time limits in the new `sql_tool`. `PDBTool` stays until cell (a)
passes acceptance.

## D5 — No re-planning after failures (S7, per spec §10)
Failed task → `FailedResult` surfaced, dependents marked `skipped`, run
continues. Results flow to the Executor only. Re-planning stays out unless
explicitly requested later.

## D6 — Key pool recovered (2026-09-29 probe: 8/17 alive)
LLM-judgement gate tests (literature support/context) run live when keys
work; recorded fixtures remain as fallback for offline CI.

## D7 — config.yaml later (S7), config.py as shim
`config.py` keeps working (old imports, CLI, frontend). `config.yaml` becomes
the single file; a loader overlays it on `config.py` defaults.

## D8 — S2 gate simplifications- SQL: single-statement SELECT/WITH only; stacked statements and DDL/DML
  keywords rejected; identifier resolution via keyword stoplist + AS-alias
  tracking (no full grammar).
- PCE in (0,1] fails as suspected fraction-instead-of-percent (conservative;
  genuine sub-1% values need an explicit unit note).
- Literature support/context judgements injectable: `judge` callable defaulting
  to the deterministic token-overlap scorer (PASS>0.6/WARN>0.3 bands reused);
  LLM judge plugs in at S5. Missing claim-side context fails (must be explicit
  to be verifiable).
- OOD result fails the ML gate (spec allows flag-or-reject; rejection chosen).
- `model_attached` fails until S5 attaches recorded group-aware validation
  metrics (by design — drives the adapter work).
- S-Q caps are an approximate embedded table for upper-bound flagging only;
  replace with a citable tabulation before publication. `iv_monotone` reports
  an explicit skip (not a silent pass) without IV data. Precomputed
  `convergence_ok` never counts as solver confirmation (AUDIT C3).

## D9 — S3 cell loop wiring- Slots are plain callables (`agent_fn(task, history)`, `tool_fn(call)`,
  `gate_fn(raw, call)`); LLM agents plug in at S5 without changing the loop.
- Tool exceptions become a `tool_ok` check failure inside an Attempt (message
  truncated to 200 chars); the run never crashes.
- `Critique.suggested_change` is empty at base level; the agent revises from
  the failed-check messages carried in history. Cell subclasses may enrich it.

## D10 — S4 SQLite mirror details- Mirror at `results/sqlite/perovskite.db`, rebuilt by `build_sqlite`;
  full mirror infers REAL (all-numeric column) vs TEXT; sampled test mirrors
  stay TEXT (fast) — queries use CAST for numeric filters there.
- Row cap via fetch-many (`truncated` flag), never by rewriting SQL.
- Timeout via SQLite progress handler. Read-only via `mode=ro` URI plus
  SELECT-only rejection in both tool and gate (defense in depth).
- Double-quoted identifiers are stripped as literals by the gate parser, so
  quoted names skip deep column checking (fail-open); generated SQL uses
  bare safe identifiers. CAST type names are stoplisted, not identifiers.
- `SQLTool` subclasses core `Tool` for the envelope but is NOT registered in
  `layers/` until cell (a) passes acceptance (old planner catalog unchanged).

## D11 — S5 cell adapters
- Adapters are deterministic (params-carried specs + small builders); LLM
  authoring plugs into the agent slots at S6 with `prompts/` files.
- ML gate fails closed without a metrics file (no `results/ml/validation.json`
  exists anywhere; catboost_info/ is empty). No scores invented; the e2e
  asserts FailedResult naming `model_attached`, plus a clearly-labeled
  fixture file proving pass wiring only.
- SCAPS adapter normalizes FF percent→fraction before gating (CSV stores
  0–100; gate contract is fraction), recorded in row provenance. Per-row Eg
  is used for band-gap sweeps. Precomputed rows fail `converged` honestly
  (no solver evidence) → FailedResult naming it.

## D12 — S6 planner v2
- Rules backend emits no `depends_on` (tasks independent); chains are built
  by the LLM backend or by hand. Executor honors the field regardless.
- Rules ML action aligns with the analyzed task type when the keyword rules
  abstain (bare "what if" → what_if): fixes a gap where task said
  sensitivity_analysis but the action fell back to predict.
- Prompts live in `prompts/*.md` with inline fallbacks; cells' LLM authoring
  plugs in later without changing the loop.

## D13 — S7 executor v2
- ThreadPool dispatch, one worker per registered cell; per-task timeout
  abandons (never kills) slow worker threads — a timed-out cell keeps
  running detached while the run moves on.
- Dependency payloads arrive as `task.params["dep_results"]` (ignored by
  tools that don't need them). Unknown dependency ids fail the task fast
  (planner bug, surfaced, not a cell fault).
- Agent exceptions, like tool exceptions, become attempt critiques
  (`agent_ok` check); worker-level bugs become task failures, never
  lost futures.
- `config.yaml` overlays `config.py` defaults; missing file = defaults.

## D14 — S8 deterministic synthesis
- Final answers are templated from validated payloads (no LLM narration):
  every number carries a source id; failed cells named with reasons.
- LLM narration is a future enhancement, not a silent default — the brief
  forbids inventing behaviour, and templates are auditable.

## D15 — S9 entry and cutover posture
- New path: `control/app.answer_cells()` + `python -m perovagents --cells`;
  legacy `Agent.answer`/CLI/frontend/benchmarks untouched and default.
- Utility runner covers `calculator` only today; other helpers error loudly
  (fail-fast, never silently skipped).
- Cutover flip (new path as default) waits for explicit approval after this
  acceptance run; MODES A–G ablation stays as the old-path regression net.

## D16 — Gemini-only LLM, simple round-robin key loop (post-S9)
- The opencode `serve` fallback and the deterministic responder were both
  removed. `chat()` now walks the key pool in a simple circle (each key tried
  once per call, first success wins, cursor advances for balance); all-fail
  or no-key returns the `[LLM ERROR]` sentinel, and callers gate on it to
  fall back to the rule planner. `usable` is now just `available` (keys).
- `PEROVAGENTS_LLM_MODEL` / `PEROVAGENTS_LLM_TEMPERATURE` remain the only LLM
  knobs; opencode config keys are gone.

## D17 — literature judges: default measured, LLM optional
- Default gate judge is token overlap (`default_judge`): 4/8 on a labeled
  battery of real corpus pairs (good on verbatim/paraphrase/unrelated;
  blind to contradiction, negation, wrong-material, entailment).
- LLM YES/NO judge scores 6/6 on the same battery (live Gemini probe).
- `LiteratureCell(judge=...)` accepts a callable or `"llm"`; default stays
  deterministic (offline-safe). Support/context remain the only
  LLM-appropriate judgements per spec §10.

## D18 — group-aware ML metrics + live SCAPS (reviewer batch)
- `ml/compute_validation.py`: GroupKFold-by-DOI (7,478 groups Overall),
  repo `prepare_features`, fixed CatBoost recipe. Recorded:
  Overall rmse 3.88/r2 0.45, NIP 3.89/0.47, PIN 3.98/0.38 — same level as
  the training-time 5-fold numbers (0.44/0.47/0.36), i.e. consistent, not
  contradictory. Metrics describe recipe+data; deployed artifacts share
  the recipe (stated in file + script docstring).
- Live SCAPS path had a real bug (`arg_flag[2:]` stripped `--`, argparse
  died) plus FF-percent rows bypassing normalization on the live branch —
  both fixed. One live wine point (pvk_d 500nm) passes all five gate
  checks; live e2e is opt-in (`PEROVAGENTS_LIVE_SCAPS=1`, ~35s).
  Precomputed rows still fail `converged` honestly — no claim dropped,
  no claim invented.
