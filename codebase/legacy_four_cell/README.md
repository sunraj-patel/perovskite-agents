# PerovAgents

An agentic framework for perovskite solar cell (PSC) research. Given a
natural-language research question, `PerovAgents` builds an ordered plan of
scientific tool calls, executes them, verifies every number, and returns an
answer grounded in explicit provenance (database rows, paper DOIs, model files).

The framework is organized as **capability layers**. Each layer contains one or
more `Tool` implementations that plug into a shared registry.

| Layer | Tools |
|-------|-------|
| Orchestration | `registry` (tool KG), `python_exec`, `calculator` |
| Literature | `literature` (semantic search), `fulltext` (PDF reader), `extraction` (tables/figures), `citations` (verifier), `doi` (resolver) |
| Materials | `pdb` (Perovskite Database), `material` (property DB), `materials_project` |
| ML | `ml` (PCE predictor) + SHAP + OOD/applicability domain |
| Physics | `scaps` (SCAPS-1D), `symbolic` (equation fitting) |
| Verification | `sanity` (scientific sanity checker) |
| Provenance | `store` (artifact/provenance store) |

## Quick start

```bash
cd perovagents
# optional: pip install -e .  (or rely on your existing anaconda env)
python examples/demo_agent.py

# interactive chat + live pipeline trace UI
streamlit run frontend/app.py
```

## Architecture

```
User Query
   → Agent.answer(question)
       → TaskSpec (task analyzer: task type + required output tokens)  (agent/task.py)
       → Planner (LLM or rules) produces PlanSteps, each declaring
         tool/action/params + purpose + expected_outputs + required_for_final_answer
                                           (agent/planner.py, rules.py)
       → schema gate: validate_plan against the action registry
         (reject on bad action/params — never silently coerce)          (core/registry.py)
   → Executor runs each step through the tool registry   (agent/executor.py, core/registry.py)
            → every result passes the scientific sanity checker  (core/sanity.py)
            → every step is recorded in the provenance store     (core/provenance.py)
   → Verifier grades the executed steps against the TaskSpec:
         SUFFICIENT / RECOVERABLE (re-plan with feedback) / UNRESOLVED  (agent/verification.py)
   → Answer synthesized + citations + warnings (+ verdict, replan log, provenance graph)
```

### Redesign highlights (SQL-verify, not silent-fail)

- **Actions are first-class.** Every tool exposes a small set of named actions
  with their own input schema and output contract — e.g. the old `ml.tool` with
  an `explain: True` flag is gone. `ml` now has `predict`, `explain_global`,
  `explain_local`, `what_if`, `optimize` (see `perovagents/ml/predictor.py`).
  Passing a flag to *simulate* another action is **rejected** by schema
  validation (`core/registry.py` → `validate_params`), so the old silent
  failure mode ("planner emits `explain` on `ml.predict`, predict drops it")
  is structurally impossible.
- **Output contracts.** Each action declares the *output tokens* it produces;
  the task analyzer declares what a question requires
  (`core/contracts.py`). The verifier matches tokens to actual results — no
  LLM involved in verification.
- **4-layer validation:** 1 registry schema gate → 2 step status/timeouts →
  3 `validate_scientific` sanity check → 4 `verify_task` sufficiency verdict.
- **Replan loop.** If a verdict is RECOVERABLE, the planner is re-invoked with
  the list of missing output tokens as feedback (up to `MAX_REPLANS`). The
  whole trajectory (plans + verdicts per iteration) is exposed to the UI.
- **Intervention actions are honest.** `what_if` sweeps ONE controllable
  numeric input; *measurement-derived / metadata / missingness / binary
  structural flags are rejected as preconditions*. `optimize` returns ranked
  model-derived sensitivities over controllable inputs (numeric sweeps +
  material-choice toggles) and, when nothing beats the baseline, says so with
  the baseline prediction instead of inventing a recommendation. Explanations
  are phrased as model associations ("SHAP identifies / the model associates"),
  never as causal claims.

## Frontend (chat + pipeline trace)

`frontend/app.py` is a Streamlit UI with two parts:

1. **Chat** — a persistent conversation. `st.session_state` keeps every
   message; each assistant reply remembers its own full processing trace.
2. **Live pipeline trace** — while the agent works it renders, in real time:
   - the **plan** (numbered steps with the planner's *purpose*/reason and
     expected output tokens for each step),
   - every **step card** as it transitions `PENDING → RUNNING → OK/FAILED`,
     showing the tool, action, layer, JSON parameters, elapsed time, sanity
     warnings, and a human-friendly rendering of the tool's output
     (metrics, tables, distributions, prediction intervals, SHAP feature
     tables, what-if sweeps, ranked optimization, passages, ...),
   - the **current step** highlighted while it executes,
   - a **verdict badge** (`SUFFICIENT`/`RECOVERABLE`/`UNRESOLVED`) plus a
     **replanning log** when the agent had to re-plan,
   - **provenance** (sources/DOIs) under the final answer.

The `answer` dict also exposes `result["task_spec"]`, `result["verdict"]`,
`result["plans"]` (every plan attempted) and `result["plan_iterations"]`
(iteration → planner → per-iteration verdict) for auditing.

Each past assistant message has a "View full processing trace" expander that
replays that conversation turn's plan and step outputs.

The live updates are driven by optional callbacks on
[`Agent.answer`](perovagents/agent/orchestrator.py), and the returned `answer`
dict's `execution.steps` embed each tool's **full result dict**
(`include_result=True`):

```python
agent.answer(question,
             on_plan=...,        # plan dict (planned tools + reasons)
             on_step_start=...,  # (i, step dict) before step i
             on_step_done=...)   # (i, step dict) after step i (ok/error)
# result["execution"]["steps"][i]["result"] == full tool output dict
```

`Agent.run(plan, question, ...).synthesize(question, execution)` expose the
same stepwise control programmatically (no UI required).

## Configuration

See `perovagents/config.py`. Sensible defaults point at the repo's data
(`opencode_sandbox/perovskite_db.csv`, `sandbox/markdown` corpus, etc.). All
paths and API keys can be overridden with environment variables:

| Env var | Purpose |
|---------|---------|
| `PEROVAGENTS_PDB_CSV` | path to perovskite DB CSV |
| `PEROVAGENTS_CORPUS_DIR` | literature corpus (markdown/PDF dir) |
| `PEROVAGENTS_LLM_MODEL` | Gemini model name |
| `GOOGLE_API_KEY` / `MP_API_KEY` | LLM / Materials Project keys |

## Evaluation / benchmark suite

`perovagents/benchmarks/` implements the planner-failure and ablation study:

- **`questions.py`** — 105 ground-truth questions: 10 intent categories
  (`prediction`, `global_interpretation`, `local_interpretation`,
  `sensitivity_analysis`, `optimization`, `comparison`, `literature_retrieval`,
  `database_query`, `verification`, `multi_tool_investigation`), 6 ambiguous
  phrasings ("What improves n-i-p performance?"), and 15 paired
  intent-disambiguation variants (5 topics x 3 minimally-differing questions).
  Each question carries acceptable action sequences (e.g. `["ml.what_if"]`).
- **`runner.py`** — per-question metrics at every layer: task-classification
  accuracy, tool/action-selection accuracy (in-order GT sequence match),
  parameter validity, plan schema validity, execution success,
  output-contract satisfaction, replan-resolved, and **Recovery Rate**
  (recoverable failures later resolved / recoverable failures encountered);
  aggregated per category and overall into JSON/CSV under
  `results/benchmarks/`. A shared analyzer cache makes multi-mode runs cheap.
- **`ablation.py`** — layered A-G modes (raw LLM tool calling -> +action
  schemas -> +task analyzer -> +schema gate -> +output verifier -> +replanning
  -> +task-analyzer-to-planner contract/claims/ontology), each a knob set on
  `Agent(mode=...)`, so component contributions are attributable.

```bash
# fast, quota-independent run (deterministic analyzer + synthesis)
python -m perovagents.benchmarks.run_pilot --subset all --pipeline rules --fast

# live-LLM ablation (A-G) on the full question set
python -m perovagents.benchmarks.ablation --modes A,B,C,D,E,F,G --limit 0
```

## Notes

- Tools lazily load heavy dependencies (`catboost`, `sentence-transformers`,
  `scipy`) only when first invoked, so importing the framework is cheap.
- This repo is the framework-only deliverable; the empirical benchmarks are
  updated separately.
- `perovagents/results/` is the default artifact output dir.