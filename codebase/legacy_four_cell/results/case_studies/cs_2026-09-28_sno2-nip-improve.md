# Case Study Log — "Improve SnO2 n-i-p device PCE" (legacy pipeline)

> Interactive run through the Streamlit frontend (legacy `Agent`, pipeline
> `llm`). This document logs the run as a from-the-UX transcript for a later
> case-study figure: agents invoked, plan iterations, tools called, inputs and
> outputs, verdicts, and the delivered answer.

- **Run date**: 2026-09-28
- **Channel**: Streamlit `frontend/app.py` (chat), legacy agent path
- **Pipeline**: `llm` (planner: LLM; execution: legacy ToolRegistry)
- **Question** (verbatim):

  > I am trying to develop an SnO2 n-i-p device, what can i do to improve the pce? What does literature have about improving these devices?

---

## 1. Plan iterations (agent-level)

| iter | agent | verdict |
|------|-------|---------|
| 1 | llm (planner → execute) | — (replanned) |
| 2 | llm (planner → execute → verify) | **RECOVERABLE** |

First iteration produced an executable plan; the output validator judged the
assembled result **RECOVERABLE** (task not yet SUFFICIENT) and the agent
re-planned, appending additional verification + refinement steps (the
`PENDING` block below).

Missing-requirement driver per the validator logic: the question asked for
both (a) performance-improvement guidance **and** (b) literature-backed
strategies; the citation-support output requirement (`citation_support`) was
not yet satisfied, so verification asked for a claim check against a broad
review paper → triggered the `citations.verify` step.

---

## 2. Tools invoked

### 2.1 `ml` → `optimize` (SUCCESS, `ok` — flagged for fix)

- **Layer**: ml · **tool**: `ml` · **action**: `optimize`
- **Purpose**: identify controllable parameters/optimal values to raise PCE
  for an SnO2 n-i-p stack, via predictive modeling.
- **Result** (as observed in the run):
  - Predicted PCE: **16.3 %** (95% interval **[14.59, 18.01]**)
  - Verbatim: "No controllable change among the inputs described improved the
    predicted PCE above the baseline; the described stack may already be near
    the model optimum. Try describing a different stack or adding a
    thickness / annealing input to sweep."
  - Provenance artifact: `model_NIP.cbm`
- **Elapsed**: 0.36 s
- **Bug then fixed**: the query described the stack as *"SnO2 n-i-p perovskite
  solar cell"* — no thickness, annealing, HTL, or perovskite composition. The
  optimizer only swept inputs that were (a) already set in the description and
  (b) in the SHAP top-25 for the **Overall** model (not the NIP model it was
  scoring), so every real knob (anneal time, perovskite/ETL thickness) was
  skipped and it reported a *flat optimum*. That "no improvement found" read as
  a misunderstanding of *"what can I do to improve PCE?"*.
- **Corrected behavior** (same input, after the fix in
  `perovagents/ml/predictor.py`):
  - underspecified-but-controllable knobs are now swept around their
    training-median baseline and flagged `source=default_stack`;
  - material slots left as `unknown` (HTL/perovskite) are evaluated against
    the median-default stack and recommended via `set <slot> to <alt>`;
  - SHAP now comes from the same architecture being scored.
  - Recommended (top-3): **decrease Perov_anneal_time_min** (Δ +0.21 pp),
    **decrease Perovskite_thickness_nm** (Δ +0.13 pp),
    **decrease ETL_thickness_nm** (Δ +0.12 pp).

### 2.2 `literature` → `search` (SUCCESS, `ok`)

- **Layer**: literature · **tool**: `literature` · **action**: `search`
- **Purpose**: retrieve relevant passages + paper metadata on methods to
  enhance PCE in SnO2 n-i-p devices.
- **Top hits** (semantic score → title):

  | score | hit |
  |-------|-----|
  | 0.74 | Ultra-thin Cadmium Sulfide Electron-transporting Layer for Planar Perovskite Solar Cell |
  | 0.72 | Impact of Anionic Br⁻ Substitution on Open Circuit Voltage in Lead Free Perovskite (CsSnI₃₋ₓBrₓ) Solar Cells |
  | 0.72 | Electrochimica Acta |
  | 0.71 | Journal of Materials Chemistry A |
  | 0.71 | Efficient Perovskite Solar Cells by Doping Poly-TPD into PTAA as Hole Transporting Layer |

  A second search wave re-surfaced the CdS ETL paper (0.73), the
  Electrochimica Acta record (0.73, DOI `10.1016/j.electacta.2017.12.108`,
  Sn-doped TiO₂ low-temperature ETL), an 0.71 ScienceDirect record, and
  "Improved Photovoltaic Performance from High Quality Perovskite Thin Film
  Grown with the Assistance of PC71BM" (0.71).

- **Elapsed**: 6.55 s (first wave) / 7.00 s (second wave)

### 2.3 `citations` → `verify` (FAIL)

- **Layer**: literature · **tool**: `citations` · **action**: `verify`
- **Purpose**: verify a general claim about improving SnO2 n-i-p devices
  against a broad review paper (to satisfy `citation_support`).
- **Input** (claim + target paper):
  - claim: "Strategies exist to improve the power conversion efficiency of
    SnO2 n-i-p perovskite solar cells."
  - doi: `10.1038/s41560-019-0442-7`
- **Verdict**: **FAIL — paper not retrievable** (full text / passage for that
  DOI unavailable in the local corpus).
- **Elapsed**: 0.77 s

---

## 3. Pending (replanned) steps

Replan after the RECOVERABLE verdict queued these steps (shown as PENDING in
the UI; the run was read at this point in this transcript):

| step | tool → action | purpose |
|------|---------------|---------|
| 1 | `literature` → `search` | passages on strategies to improve PCE in SnO2 n-i-p perovskite solar cells |
| 2 | `citations` → `verify` | verify a specific claim from retrieved literature (`citation_support`) |
| 3 | `ml` → `optimize` | identify controllable parameters to improve predicted PCE |
| 4 | `literature` → `search` | methods to enhance PCE in SnO2 n-i-p devices |
| 5 | `citations` → `verify` | verify a general claim against a broad review paper |

---

## 4. Sources cited

`model_NIP.cbm`

| DOI |
|-----|
| 10.1038/ncomms13503 |
| 10.1016/j.nanoen.2018.10.011 |
| 10.1007/s10118-017-1891-z |
| /Users/sunraj/Desktop/College/S6/UGP/sandbox/markdown |
| 10.1021/jp5126624 |
| 10.1246/cl.180616 |
| 10.1016/j.electacta.2017.06.032 |
| 10.1002/aenm.201803587 |
| 10.1038/ncomms15330 |
| 10.1016/j.electacta.2017.12.108 |
| 10.1039/c9ta01893j |
| … 1 more |

---

## 5. Delivered answer (verbatim structure)

1. **ML outcome**: ml.optimize predicts **16.3 %** PCE (95% interval
   [14.59, 18.01]); no controllable input change improved beyond baseline —
   the described stack is near the model optimum; suggest a different stack
   or adding thickness/annealing to sweep. *(Post-fix: the same query instead
   returns rankable knobs — decrease anneal time / perovskite thickness / ETL
   thickness, each flagged `default_stack` — rather than a misleading flat
   optimum, because unset controllable inputs were previously never swept.)*
2. **Literature strategies**:
   - Passivation of the perovskite surface with 1-adamantylamine
     hydrochloride (ADAHCl) — mentioned as a route to further improvements
     [10.1002/aenm.201803587].
   - SnF₂ (20 mol %) addition → short-circuit photocurrent density Jsc of
     27.67 mA/cm² under AM 1.5G [10.1021/jp5126624].
   - Remaining passages are general perovskite properties (efficiency,
     stability), not specific SnO2 n-i-p improvement strategies
     [10.1038/ncomms13503], [10.1038/ncomms15330],
     [10.1016/j.nanoen.2018.10.011], [10.1246/cl.180616],
     [10.1016/j.electacta.2017.06.032], [10.1039/c9ta01893j],
     [10.1117/12.2506443].
3. **Caveat**: attempts to verify specific literature claims returned FAIL
   (`paper not retrievable`); full-text access is needed to extract/verify
   detailed strategies.

---

## 6. Failure / recovery notes (for the figure)

- **ML optimize** presented a *flat optimum* ("no controllable change improved
  PCE", 16.3 % baseline) — **root cause fixed**: the description left all
  controllable inputs unspecified and the old sweep only considered inputs
  already set in the description, so nothing was ever tested. The optimizer
  now sweeps underspecified declared-range knobs around their training-median
  baseline (`source=default_stack`) and evaluates unset material slots, and
  uses the same architecture for SHAP and scoring.
- **citation verification** failed because the review-paper DOI chosen by the
  planner is not in the local full-text corpus → verdict FAIL → contributes
  to RECOVERABLE.
- **Verdict-driven replan** worked: the agent appended targeted verify/search
  steps instead of looping the same plan.

## 7. Suggested figure layout (case study)

Timeline (x) vs agents/tools (y bands):
1. planner (iter 1) → llm badge
2. ml.optimize tick (0.36 s) → green
3. literature.search tick (6.55 s) → blue
4. literature.search second wave (7.00 s) → blue
5. citations.verify tick (0.77 s) → red FAIL
6. validator → RECOVERABLE badge (orange)
7. planner (iter 2) → replan arrow, 5 PENDING steps (dashed)
8. answer card (16.3 % ± interval, 11 DOIs + 1)