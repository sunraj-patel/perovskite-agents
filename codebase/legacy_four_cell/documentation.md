# PerovAgents — Technical Documentation

This document mirrors the code in `perovagents/` and is the reference for its
architecture, runtime flow, and extension points. The README is the short
overview; this is the map you follow while reading or modifying the code.

---

## 1. What PerovAgents is

An agentic framework for perovskite solar cell (PSC) research. Given a
natural-language research question it:

1. **analyzes intent** into a structured `TaskSpec` (what evidence is needed),
2. **plans** the minimum valid sequence of scientific tool *actions*,
3. **validates** the plan against per-action schemas before touching any data,
4. **executes** the actions through a lazy tool registry,
5. **verifies** that the executed results really carry what the question required,
6. **synthesizes** an answer whose *epistemic register* matches each result,
   grounded in explicit provenance (DB rows, DOIs, model files, verdicts).

Every number the answer contains can be traced back to the tool which produced
it via the provenance store.

---

## 2. Directory map

```
perovagents/
├── documentation.md            # this file
├── README.md                   # short overview
├── pyproject.toml
├── examples/demo_agent.py      # offline, key-free demo
├── frontend/app.py             # Streamlit chat + live trace UI
├── results/                    # artifacts (ml models, benchmarks, provenance)
├── tests/
│   ├── test_smoke.py           # 26 integration tests
│   ├── test_frontend_smoke.py  # 4 headless frontend tests
│   ├── test_llm_rotation.py    # 4 key-pool rotation/circuit-breaker tests
│   └── test_intent_recovery.py # 6 analyzer/verifier intent tests
└── perovagents/
    ├── __init__.py             # public exports
    ├── __main__.py             # `python -m perovagents` CLI
    ├── config.py               # every path, key, physics bound
    ├── llm.py                  # Gemini client: key pool, rotation, breaker
    ├── agent/                  # the cognitive loop (see §4)
    │   ├── orchestrator.py     #   Agent: answer/run/synthesize
    │   ├── task.py             #   TaskSpec + task analyzer (LLM/keyword)
    │   ├── planner.py          #   LLM planner (+ prompt, coerce, semantic gate)
    │   ├── rules.py            #   deterministic rule planner (baseline)
    │   ├── executor.py         #   runs plans, StepResult, render_for_synthesis
    │   ├── verification.py     #   Verdict: SUFFICIENT/RECOVERABLE/UNRESOLVED
    │   └── models.py           #   Plan / PlanStep / PlannerError
    ├── core/                   # shared abstractions (see §3)
    │   ├── tool.py             #   Tool base, ExecutionResult, ValidationBlock
    │   ├── registry.py         #   ToolRegistry, ActionSpec, ToolKnowledgeGraph
    │   ├── contracts.py        #   output tokens + deterministic matchers
    │   ├── claims.py           #   claim_type taxonomy + CLAIM_LANGUAGE
    │   ├── provenance.py       #   ProvenanceStore (JSONL)
    │   ├── sanity.py           #   SanityChecker scientific bounds
    │   ├── validation.py       #   physics-bounds validator
    │   ├── calculator.py       #   unit conversion / PCE arithmetic
    │   └── python_exec.py      #   sandboxed exec for computations
    ├── layers/__init__.py      # register_all(): the 13 tools + all actions
    ├── ml/                     # predictor, features, controllability ontology
    ├── materials/              # pdb, properties, materials_project
    ├── physics/                # scaps, symbolic
    ├── literature/             # search, fulltext, extraction, citations, doi
    └── benchmarks/             # questions, runner, ablation, run_pilot
```

---

## 3. Core abstractions (`perovagents/core/`)

### 3.1 `Tool`, `ExecutionResult`, `ValidationBlock` (`core/tool.py`)

A `Tool` is **the unit of scientific capability**. Every tool sets `name`,
`description`, `layer`, `version`, `capabilities`, and implements
`call(params) -> dict`. Subclasses must be *lazy*: heavy imports (catboost,
sentence-transformers, scipy) go inside methods, never at module import time —
this keeps `import perovagents` cheap.

`Tool.execute(params, action=None, trace_id=None)` is the canonical entry
point. It wraps `call()` and always returns the same **envelope**:

```json
{
  "status": "success|error",
  "tool": "...", "action": "...",
  "result": { "...": "domain payload, deterministic" },
  "params": { "...": "as invoked" },
  "error": "...",
  "elapsed_seconds": 0.42,
  "provenance": { "...": "DOIs / source files / model files" },
  "validation": {
    "schema_valid": true, "units_valid": true, "domain_valid": true,
    "errors": [], "warnings": []
  },
  "trace_id": "..."
}
```

- `ValidationBlock` = layer-2/3 scientific validation (units + domain bounds);
  non-blocking by design — values are still returned, but flagged.
- `result` is auto-wrapped into `{"result": ..., "result_type": "raw"}` when a
  tool returns a non-dict.

### 3.2 Tool registry and the **action** model (`core/registry.py`, `layers/`)

The registry is the agent's "address book of capabilities". Tools are
registered as *lazy factories*:

```python
registry.register(LiteratureSearchTool, actions=ACTIONS["literature"])
```

Instances are created on first `get()`, so the planner can see the whole
catalog without loading heavy models.

**Actions are first-class.** A tool is not one flat callable; it exposes named
actions, each with its own `ActionSpec`:

| field            | meaning                                        |
|------------------|------------------------------------------------|
| `task_types`     | scientific task classes this action satisfies  |
| `params`         | input schema per parameter (type/enum/required)|
| `outputs`        | output **tokens** the result will carry        |
| `produces`       | subset of tokens confirmed produced            |
| `compatible_with`| downstream actions it can feed                 |
| `requires`       | preconditions (e.g. `trained_model`)           |
| `constraints`    | scientific honesty notes for the planner       |

Passing a flag to *simulate* another action is rejected by schema validation —
the old silent failure mode ("planner emits `explain`, predict drops it") is
structurally impossible.

The registry validates before execution:

- `validate_action(tool, action)` — is the action declared?
- `validate_params(tool, action, params)` → `(ok, errors, warnings)`
- `action_supports_task(tool, action, task_type)` — used by the planner's
  semantic contract check.
- `ToolKnowledgeGraph.related(tool)` — curated co-use graph for the planner.

### 3.3 Output tokens and deterministic matching (`core/contracts.py`)

The whole verification story hangs on a small **vocabulary of output tokens**:

```
prediction | interval | uncertainty | ood_status | architecture
feature_name | importance | direction
parameter_value | predicted_pce
count | aggregate | table | distribution | columns
records | papers
passages | material_property | properties | doi_metadata | citation_support
sweep | equation | calculation | exec_output
```

An action *declares* which tokens its result will carry (`outputs`); the task
analyzer *requires* tokens; the verifier **matches** both by applying a
deterministic matcher function to the actual result dict (`match_output`,
`match_outputs`). No LLM is involved in verification — a token is satisfied
only if the result demonstrably carries it (e.g. `aggregate` requires
`result_type == "scalar"` **and** a numeric `value`).

### 3.4 Claim typing (`core/claims.py`)

Every executed step is tagged with a `claim_type` (deterministic per
`(tool, action)`) so the answer generator is forced into the right **epistemic
register**:

| claim_type                | produced by                                 | register rule                       |
|---------------------------|---------------------------------------------|-------------------------------------|
| `model_prediction`        | `ml.predict`                                | "the model predicts … 95% interval" |
| `model_association`       | `ml.explain_global/local`                   | SHAP = association, **not causal**  |
| `model_sensitivity`       | `ml.what_if`, `ml.optimize`, `scaps`        | "under the model, changing X moves …"|
| `database_evidence`       | `pdb.query`                                 | "across N records (DB), stat"       |
| `material_property_evidence` | `material.lookup`, `materials_project`    | "computed … from MP"                |
| `literature_evidence`     | `literature.*`, `doi`, `extraction`         | "per {doi} …"                        |
| `verification_evidence`   | `citations.verify`                          | report PASS/WARN/FAIL vs a source   |
| `computation`             | `calculator.*`, `python_exec`               | exact arithmetic, with units        |
| `empirical_fit`           | `symbolic.fit`                              | valid only over the fitted range    |

`CLAIM_LANGUAGE` maps each class to the required narrative phrasing; the
synthesizer injects only the registers actually produced.

### 3.5 Provenance store (`core/provenance.py`)

One `trace_id` per question. Every tool invocation is recorded:

- trace id, step index, tool, params, status, elapsed,
- a **hash** of the result (integrity / diffing),
- the source identifiers the tool reported (DOIs, CSVs, model files).

Records are held in memory and appended to
`results/provenance/provenance.jsonl`. `snapshot(trace_id)` exposes the full
linear record for auditability.

---

## 4. The agentic loop (`perovagents/agent/`)

### 4.1 Entry point: `Agent.answer(question)` (`orchestrator.py`)

```
question
  ├─ ① analyze_task()                 -> TaskSpec            (agent/task.py)
  ├─ ② plan_question()                -> Plan                (planner.py | rules.py)
  ├─ ③ validate_plan() [schema gate]  -> reject, never coerce
  ├─ ④ Executor.execute_plan()        -> StepResults         (agent/executor.py)
  │     └ per step: tool.execute() + sanity check + provenance record
  ├─ ⑤ verify_task()                  -> Verdict             (agent/verification.py)
  │     SUFFICIENT   -> break
  │     RECOVERABLE  -> re-plan with missing-token feedback (<= max_replans=2)
  │     UNRESOLVED   -> break (report evidence gathered)
  ├─ ⑥ _synthesize()                  -> answer text         (constrained by contract)
  └─ returns answer dict (§4.7)
```

`Agent` supports ablation modes A–G via `mode=` (see §8). The controllable
flags are `use_analyzer`, `structured_actions`, `schema_gate`, `use_verifier`,
`use_replan`, `enforce_task` (the `MODES` table in `orchestrator.py`).

`fast=True` forces the deterministic (non-LLM) analyzer and synthesis — zero
quota cost, used by offline CI and the rules benchmark.

### 4.2 Task analyzer (`agent/task.py`)

Converts a question into a `TaskSpec` **before any tool is chosen**:

```json
{
  "task_type": "feature_importance",
  "architecture": "n-i-p",
  "objective": "...",
  "requested_output": ["feature_name", "importance", "direction"],
  "requires_prediction": false,
  "requires_explanation": true,
  "requires_intervention": false,
  "requires_sensitivity": false,
  "requires_optimization": false,
  "requires_evidence": true,
  "materials": ["SnO2"]
}
```

Task types (ordered registry, `TASK_TYPES`):
`factual_lookup`, `literature_retrieval`, `literature_extraction`,
`database_query`, `prediction`, `interpretation`, `feature_importance`,
`sensitivity_analysis`, `optimization`, `comparison`, `verification`,
`calculation`, `simulation`, `multi_tool_investigation`.

The analyzer is **LLM-assisted when a key is present** (`ANALYZER_PROMPT`)
and falls back to a deterministic keyword analyzer (`_RULES`, `_keyword_spec`).
The keyword rules deliberately distinguish the interpretation *families*
(Case A–D):

- "most important/relevant parameters", "which features matter", "what drives
  PCE" → `feature_importance` (no need to say "SHAP").
- "this device …" (local) → `interpretation`.
- "what happens if / how sensitive to X / change with …" → `sensitivity_analysis`.
- "what should I change / maximize / optimize / should I use…" → `optimization`.

`_TASK_OUTPUTS` maps each task type to required output tokens;
`_REQ_FLAGS` maps them to the `requires_*` booleans. `extract_materials()`
pulls a small canonical material vocabulary from the text.

### 4.3 Planner (`agent/planner.py`, `agent/rules.py`)

Two planners produce the same `Plan`/`PlanStep` structure
(`agent/models.py`):

```
PlanStep { tool, action, params, reason, purpose,
           expected_outputs, required_for_final_answer }
```

**LLM planner** (`LLMPlanner`, `planner.py`). The system prompt is assembled
from the registry's per-action schemas (`{TOOLS_JSON}`) and the TaskSpec
(`{TASK_JSON}`). Key contract rules in the prompt:

- the TaskSpec is **authoritative** (task_type, architecture, requested outputs);
- `ml` is a *family* of operations — select the correct **action**
  (predict / explain_global / explain_local / what_if / optimize);
- never pick `ml.predict` when the task is feature importance; never use a DB
  query just because the question says "device";
- SHAP = association, not causality; `what_if` = model sensitivity;
  `optimize` may only touch ontology-declared controllable features.

`plan()` calls `_coerce_and_validate()`: unknown tool/action, invalid params,
missing params, or undeclared `expected_outputs` → rejected with corrective
feedback (a `PlannerError`); the orchestrator can retry or fall back. When the
LLM is unavailable (`[LLM ERROR]` from a dead key pool) the planner raises a
truthful `PlannerError("LLM unavailable …")` instead of a misleading
parse error.

`semantic_errors()` implements the **analyzer→planner contract**: the
final-answer steps of a plan must *serve* the analyzed `task_type`
(`action_supports_task`). This stops the planner quietly substituting one
answer class for another after the analyzer decides. Rule plans are exempt
(their intent is coded).

**Rule planner** (`RulePlanner`, `rules.py`): deterministic baseline + offline
fallback. `RULES` maps regex → ordered tool names; `ML_ACTION_RULES` then picks
the ml **action**; `_infer_arch()` picks `pin/nip/overall`; `_whatif_spec()`
returns a real parameter + values grid so `ml.what_if` always executes; the
remaining tools get executable defaults from `DEFAULT_PARAMS`. The fallthrough
case ("no rule matched") returns the default factual `pdb.query` — which is
what the verifier's claim gate is designed to catch when the analyzer demanded
a different evidence class.

### 4.4 Schema gate (`validate_plan`, layer 1)

Before execution every step is re-validated against the registry
(`validate_action` + `validate_params`); with `enforce_task` enabled it also
runs `semantic_errors`. Failures are **rejected with feedback and re-planned**,
never silently coerced.

### 4.5 Executor (`agent/executor.py`)

Runs each step with timeouts, records provenance, and attaches the sanity
warnings. `StepResult` carries, in addition to the tool envelope:

- `purpose`, `expected_outputs`, `required_for_final_answer` (planner intent),
- `claim_type` (`claim_type_for(tool, action)`),
- `result_summary` (`summarize()` — one-line trace summary)
- and the full result dict when `include_result=True`.

`render_for_synthesis(sd, max_chars=4500)` is the **rich**, LLM-facing
rendering used to build the synthesis prompt — it includes the actual values
(feature importances/directions, predicted PCEs, what-if rows, optimization
recommendations, table heads, passages), not just summaries. This is what lets
the synthesizer quote real numbers instead of saying "not provided".

By default the executor *continues* after a failed step (partial evidence is
often useful); `stop_on_error=True` gives SQL-like behavior.

### 4.6 Verifier (`agent/verification.py`, layer 4)

The hardest question: **did the executed steps actually return what the
question requires?** The verdict:

| verdict       | meaning                                                       |
|---------------|---------------------------------------------------------------|
| `SUFFICIENT`  | every required output token present in some successful step   |
| `RECOVERABLE` | some required tokens missing; re-plan with them as feedback   |
| `UNRESOLVED`  | all steps failed / no usable evidence                         |

Required tokens come from three places (deduplicated): the TaskSpec's
`requested_output`, each required step's `expected_outputs`, and the semantic
`requires_*` flag groups (prediction/explanation/intervention token sets).

**Claim-type epistemic gate** (`TASK_ALLOWED_CLAIMS`). The final-answer carrier
must have a `claim_type` *compatible* with the task:

```
prediction            -> {model_prediction}
feature_importance    -> {model_association}
interpretation        -> {model_association, model_prediction}
sensitivity_analysis  -> {model_sensitivity}
optimization          -> {model_prediction, model_sensitivity}
database_query        -> {database_evidence}
comparison            -> {database_evidence}
factual_lookup        -> {material_property_evidence}
literature_*          -> {literature_evidence}
verification          -> {verification_evidence}
calculation           -> {computation}
simulation            -> {computation, model_sensitivity}
multi_tool_investigation -> open
```

If every successful step produced the **wrong evidence class** (e.g. a PCE mean
table for a feature-importance question), the answer is `RECOVERABLE`, *never*
SUFFICIENT — even when the number "looks plausible". Missing
`compatible_claim_level` is appended to the requirements so the replanner knows
to switch action families (`ml.explain_global`, `ml.what_if`, …). The gate fails
closed when `claim_type` is absent. An honest `ml.optimize` that reports
`optimization_status: "none_found"` is exempted from the `parameter_value`
requirement (by construction it produced no sweep).

### 4.7 Answer dict

`Agent.answer()` returns a self-contained dict:

```json
{
  "question", "pipeline", "trace_id",
  "task_spec": {...},
  "plan": {...}, "plans": [...],
  "execution": {"steps": [{tool, action, params, success, result, ...}], ...},
  "verdict": {...},
  "plan_iterations": [{iteration, planner, verdict, note}],
  "answer_contract": {...},
  "answer": "...", "claims": [...], "sources": [...], "warnings": [...],
  "provenance": {...}, "provenance_graph": {"nodes": [...], "edges": [...]}
}
```

Surfaced in the UI as per-step cards (`PENDING → RUNNING → OK/FAILED`), the
verdict badge, a replanning log, and provenance under the answer.

---

## 5. The tools

`layers/__init__.py` → `register_all()` registers 13 tools. Actions are
documented *in code* as `ActionSpec`s; this table is the map.

| tool | layer | actions | one-liner |
|------|-------|---------|-----------|
| `calculator` | orchestration | `eval` | unit conversion / PCE arithmetic |
| `python_exec` | orchestration | `exec` | sandboxed computation |
| `literature` | literature | `search` | semantic search over the corpus → passages |
| `fulltext` | literature | `fetch` | full text by DOI (corpus, then network) |
| `extraction` | literature | `extract` | markdown tables / figure captions |
| `citations` | literature | `verify` | PASS/WARN/FAIL of a claim vs a cited paper |
| `doi` | literature | `resolve` | Crossref metadata |
| `pdb` | materials | `query` | structured Pandas queries over 43k records |
| `material` | materials | `lookup` | curated property tables (lit. wins) |
| `materials_project` | materials | `lookup` | Materials Project REST + polymorph handling |
| `ml` | ml | `predict`, `explain_global`, `explain_local`, `what_if`, `optimize` | CatBoost PCE prediction + SHAP + sweeps |
| `scaps` | physics | `get_sweep` | precomputed (or wine64 live) SCAPS sweeps |
| `symbolic` | physics | `fit` | empirical curve fitting |

### 5.1 `ml` in detail (`ml/predictor.py`)

Models are per-architecture CatBoost regressors (`Overall`, `PIN`, `NIP`)
with a bagged-seed ensemble for 95% prediction intervals. Uncertainty =
`pred ± 1.96·σ` (σ from the seed ensemble). Each action returns a specific
`result_type`:

- `predict` → `prediction`: `predicted_pce`, `pce_95_interval`,
  `interval_spread_percent`, `hood` (out-of-distribution envelope check),
  `resolved_materials`, `provenance`.
- `explain_global` / `explain_local` → `explanation`: `scope`,
  `features: [{name, importance, direction, shap}]` (mean |SHAP| over a
  deterministic design grid for global), `controllable: {name: bool}`.
- `what_if` → `what_if`: `parameter`, `valid_range`, `clamped_to_valid_range`,
  `base_value`, `data: [{parameter_value, predicted_pce, interval_lo,
  interval_hi}]`. The parameter must be ontology-controllable (§5.3).
- `optimize` → `optimization`: ranked `recommendations`
  (`{parameter, kind, direction, recommended_value, base/best/delta PCE}`)
  from a base × {−0.2, −0.1, 0, +0.1, +0.2} grid, plus material-choice
  toggles. **When nothing beats the baseline** it returns
  `optimization_status: "none_found"` with the baseline prediction rather than
  inventing a recommendation.

### 5.2 `pdb` in detail (`materials/pdb.py`)

`operation` enum: `count | filter | top_k | groupby_aggregate |
global_aggregate | unique | distribution | sample | describe_columns`.
Filters use `{column, op, value}` with ops `eq/ne/contains/not contains/gt/gte/
lt/lte/in/regex`. Every result carries `rows_before`, `rows_after`,
`n_records`, `n_papers`, `warnings`, and `provenance` (with ≤25 DOIs);
`Ref_DOI_number` is always auto-selected for provenance. Column aliases live
in `ALIAS_COLUMNS` (e.g. `PCE` → `JV_default_PCE`).

### 5.3 Feature ontology (`ml/controllability.py`)

Deterministic, non-LLM classification of every model feature into a 4-axis
ontology — the gate that decides what `what_if`/`optimize` may sweep:

- `controllability`: `experimentally_controllable | indirectly_controllable
  | non_controllable`;
- `type`: `derived | binary | continuous | categorical`;
- `role`: `material | measurement | metadata | process | device`.

Measurement-derived, metadata, missingness, and binary-structural features are
**not sweepable**. Only five knobs have declared valid ranges
(`DECLARED_RANGES`: thickness×3, anneal temp/time); everything else is
unbounded-but-honest (`valid_range=null`).

### 5.4 Materials Project (`materials/materials_project.py`)

`get_properties(formula)` fetches band gap, formation energy, density,
stability via the MP REST API with a JSON disk cache
(`results/materials_project/cache.json`). Polymorph handling returns
`phase`/`phases` + `needs_user_choice`:

- 0 phases → `{found: false, n_phases: 0}`;
- 1 phase → auto-select, `needs_user_choice: false`;
- many phases, exactly one stable → auto-select the stable one;
- many phases, ambiguous → `needs_user_choice: true` + options (the UI renders
  interactive buttons). Offline (no `MP_API_KEY`) → honest `{offline: true}`.

### 5.5 SCAPS (`physics/scaps.py`)

Default mode reads precomputed `sweep_results.csv`
(`sweep_name` ∈ `eg|d|mue|muh|htlchi|htlEg`), returning per-point
`Voc/Jsc/FF/PCE`, `trend` (positive/negative/weak from Pearson r), and
`relative_PCE_change`. `live=True` runs the SCAPS 3.3 executable via
`wine64` + a generated script. Constraints: SCAPS results are *model
sensitivities, not experimental reproductions*.

---

## 6. LLM client and key management (`llm.py`)

A thin Gemini wrapper (`gemini-2.5-flash` by default; `LLM_MODEL` config):
`chat(prompt)` → text, `json(prompt)` → parsed JSON. There is **no** offline
fallback: with no keys, `chat()` returns the `[LLM ERROR]` sentinel and
callers gate on it to fall back to the rule planner.

**Key pool.** `LLM._load_keys()` merges, in order:

1. `GOOGLE_API_KEY` env var,
2. `chatbot/.env` (`GOOGLE_API_KEY=` line),
3. `~/keys.md` (`config.KEYS_MD_FILE`) — one key per line, `KEY=`/`GOOGLE_API_KEY=`
   assignments, markdown bullets `-`/`*`, `#` comments; deduplicated.

**Simple round-robin loop.** `chat()` walks the key pool in a circle: it
starts after the cursor, tries each key exactly once per call, and returns the
first successful completion (the cursor then advances so the load is
balanced). No cooldowns, no circuit breaker — every call retries the whole
pool fresh, so a quota-exhausted key is picked up as soon as it recovers. When
every key fails, the last error is returned as `[LLM ERROR] <reason>`.

`available` and `usable` are both `True` whenever keys exist. Planners gate on
`usable` and use `text.startswith("[LLM ERROR]")` to decide fallback.

---

## 7. CLI, demo, frontend

```bash
cd perovagents
python -m perovagents "what is the average PCE for devices with SnO2?"
python -m perovagents --pipeline rules --json "predict PCE for p-i-n PTAA/SnO2"
python -m perovagents --list-tools

python examples/demo_agent.py          # offline demo (rule pipeline)

streamlit run frontend/app.py          # chat + live trace UI
```

The CLI (`__main__.py`) prints the answer, sources, and (optional) the full
JSON. Exit codes: `0` ok, `2` missing question.

Note: the only working entry point is `python -m perovagents`; the console
script `perovagents = "perovagents.cli:main"` in `pyproject.toml` points at a
module that does not exist (do not rely on the `perovagents` binary).

---

## 8. Benchmark suite (`perovagents/benchmarks/`)

### Questions (`questions.py`)

- `SIMPLE` — 84 questions across 10 categories (`prediction` 10,
  `global_interpretation` 10, `local_interpretation` 8, `sensitivity_analysis`
  9, `optimization` 8, `comparison` 8, `literature_retrieval` 8,
  `database_query` 12, `verification` 6, `multi_tool_investigation` 5).
- `AMBIGUOUS` — 6 ambiguous phrasings.
- `DISAMBIGUATION` — 15 paired near-identical variants: 5 topics × 3 questions
  (`predict-vs-global`, `global-vs-local`, `sensitivity-vs-optimization`,
  `parameter-meaning`, `verify-vs-query`).
- `all_questions()` = 105 total.

Each record is `{qid, category, question, gt}` where `gt` is a list of
acceptable `["tool.action", …]` sequences (multiple = science-equivalent
alternatives).

### Runner (`runner.py`)

Measures, per question and per layer:

| metric | definition |
|--------|------------|
| `classification_accuracy` | analyzer `task_type` ∈ expected set |
| `selection_accuracy`     | `gt` sequence appears in-order among final-answer steps |
| `parameter_validity`     | step params schema-valid |
| `plan_validity`          | plan passed the schema gate (`_plan_schema_valid`) |
| `execution_success`      | all executed steps succeeded |
| `contract_satisfied`     | output verifier reached SUFFICIENT at some iteration |
| `replanner`              | which planner produced iteration 0 |
| `recovery_rate`          | RECOVERABLE/REJECTED failures later resolved / total such failures |
| `final_sufficient`       | final verdict was SUFFICIENT |

Aggregated overall and per category into
`results/benchmarks/<label>.json`.

```bash
# fast, quota-independent baseline (deterministic analyzer + synthesis)
python -m perovagents.benchmarks.run_pilot --subset all --pipeline rules --fast

# a quicker check on the disambiguation set
python -m perovagents.benchmarks.run_pilot --subset disambig --pipeline rules --fast
```

### Ablation (`ablation.py`)

Layered A–G build-up of the stack so each component's contribution to the
metrics is attributable:

```
A raw       raw LLM tool calling (baseline)
B schemas   + first-class action schemas
C analyzer  + task analyzer feeding the planner
D gate      + pre-execution validation gate
E verifier  + output contract verifier (report only)
F replan    + iterative re-planning
G full      + analyzer→planner contract + claims + ontology
```

```bash
# live-LLM ablation (requires healthy Gemini keys)
python -m perovagents.benchmarks.ablation --modes A,B,C,D,E,F,G --limit 0
```

For the live LLM tests, quota status matters: when the key pool is exhausted,
the framework short-circuits to the rules fallback and logs it
(`LLM key pool down: all N keys failed …`), and the ablation numbers for LLM
modes are only meaningful once a key recovers.

---

## 9. Configuration (`config.py`)

Single source of truth for paths, keys, and physics bounds. Everything is
overridable via env vars:

| config var | default | env var |
|------------|---------|---------|
| `RESULTS_DIR` | `<project>/results` | `PEROVAGENTS_RESULTS_DIR` |
| `PDB_CSV` | `<repo>/opencode_sandbox/perovskite_db.csv` | `PEROVAGENTS_PDB_CSV` |
| `ENRICHED_CSV` | `opencode_sandbox/dataset_json_enriched.csv` | `PEROVAGENTS_ENRICHED_CSV` |
| `CORPUS_DIR` | `<repo>/sandbox/markdown` | `PEROVAGENTS_CORPUS_DIR` |
| `LLM_MODEL` | `gemini-2.5-flash` | `PEROVAGENTS_LLM_MODEL` |
| `KEYS_MD_FILE` | `~/keys.md` | — |
| `CHATBOT_ENV_FILE` | `<repo>/chatbot/.env` | — |
| `SCAPS_*` | under `opencode_sandbox/scaps` | — |
| `PHYSICS_BOUNDS` | PCE 0–30, Voc 0–1.4, Jsc 0–32, FF 0.2–0.95 | — |
| `PERF_ALIAS` | `PCE→JV_default_PCE`, etc. | — |
| `ARCH_ALIAS` | `n-i-p→nip`, `p-i-n→pin`, … | — |

Keys read outside `config.py`: `GOOGLE_API_KEY` (`llm.py`),
`MP_API_KEY` (`materials/materials_project.py`),
`PEROVAGENTS_ML_MODEL_DIR` (`ml/predictor.py`, default `results/ml`).

---

## 10. Extending the framework

To add a capability:

1. Create a `Tool` subclass (set `name`, `description`, `layer`, `version`,
   implement `call()`; keep heavy imports lazy).
2. Define its `ActionSpec`s: `task_types`, `params` schema, `outputs`,
   `available_task_types` compat, `requires`/`constraints`.
3. Register in `layers/__init__.py` (`ACTIONS[name]` + `registry.register`).
4. Add any new output tokens to `core/contracts.py` (`OUTPUT_MATCHERS`) and
   (if it is a new evidence family) a `claim_type` + `CLAIM_LANGUAGE` entry in
   `core/claims.py`, plus its allowed task types in
   `agent/verification.py::TASK_ALLOWED_CLAIMS`.
5. Add expected-task metadata for benchmarks (`runner.py::EXPECTED_TASK` /
   `ACTION_DEFAULT_TASK`) and a couple of benchmark questions.

The layer validator is strict because the planner, verifier, and claim gate all
read the same declarations: a missing schema, token, or claim_type is a silent
quality failure upstream.

---

## 11. Known behaviors / caveats

- **Offline mode**: no API key → analyzer/rules/synthesis all use deterministic
  paths; everything still completes.
- **Circuit breaker**: a quota-exhausted key pool is probed at most once per
  60 s window per process.
- **Embedding fallback**: `literature/search.py` falls back to first-N columns
  if sentence-transformers is unavailable.
- **`citations.verify`** computes `supporting_evidence` but currently hard-wires
  it to `[]`; verdicts still use `match_score` + DOI overlap.
- **`symbolic.fit`** supports an undocumented `kind="eg_nm"` beyond the
  registry enum (`linear|power|exponential`).
- **`ml` OOD** requires a manifest with `quantiles`; the on-disk
  `results/ml/manifest.json` is stale (246 vs 248 features, no `quantiles`), so
  `hood.ood` is effectively always `False` until it is regenerated via
  `ml/features.py::build_manifest`.
- **Repair loop**: the planner retries once with corrective feedback on
  validation failure; the orchestrator re-plans on RECOVERABLE verdicts up to
  `max_replans=2`.
- Tools lazy-load heavy dependencies, so `import perovagents` is cheap.

---

## 12. Tests

```bash
cd perovagents
python -m pytest tests/test_smoke.py tests/test_llm_rotation.py \
                tests/test_intent_recovery.py -q          # 36 fast tests
python -m pytest tests/test_frontend_smoke.py -q          # 4 UI tests
```

- `test_smoke.py` — end-to-end agent runs, verifier gates, sanity, provenance.
- `test_llm_rotation.py` — key parsing/dedup, dead-key skip + cycling,
  all-dead error, and the circuit-breaker short-circuit (network-free).
- `test_intent_recovery.py` — analyzer classification, rules routing, and the
  verifier's wrong-evidence-class rejection.
- `test_frontend_smoke.py` — headless Streamlit health check.