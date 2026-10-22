# FINAL REPORT — live SCAPS-1D device design pipeline

**Scope of this document.** This is the four-agent *simple* framework in
`claude_code_perovagents/`. A separate four-cell refactor of `perovagents/` has its own report at
`../perovagents/FINAL_REPORT.md`. The two codebases do not share the files described below.

---

## 1. Executive summary

`--agent design` no longer describes perovskite cells; it builds them and solves them. The agent
calls four plain Python functions: one validates and lays out a device, one writes the SCAPS
`.def`/`.script` pair, one runs SCAPS-1D and returns Voc/Jsc/FF/PCE, and one runs a graded profile
against a uniform one to answer "does depth matter, and by how much" with two real solves.

Every performance number the agent can report therefore comes from the solver. The agent has no
way to produce a plausible-looking device table that was never simulated, which was the failure
mode that made a design agent unusable for a reviewer.

**Headline result**, from the exact query used for acceptance:

```
Design a p-i-n PSC with SnO2 on one side and tell me the defect density profile.
```

| run | stack | Voc (V) | Jsc (mA/cm²) | FF (%) | PCE (%) | time |
|-----|-------|---------|--------------|--------|---------|------|
| graded | ITO/PTAA 150 \| MAPbI3 500 \| SnO2 50/ITO, exp-tail 1e17→1e14 over 120 nm, 4 sub-layers | 1.2353 | 24.187 | 87.76 | 26.22 | 57 s |
| uniform | same stack, uniform 1e15 1/cm³ | 1.3776 | 24.194 | 90.67 | 30.22 | 43 s |

Grading Nt toward the illuminated surface costs ~4 PCE points, almost all of it fill factor.

## 2. What was delivered

| file | role |
|------|------|
| `physics/scaps_def.py` | stack model and `.def` text builder; n-i-p and p-i-n; sub-layer expansion for graded `Nt` |
| `physics/scaps_run.py` | the standard runner: install `.def`, write action script, serialised Wine execution, IV parsing |
| `physics/design.py` | the four agent-facing tools, in engineering units |
| `team.py::design_agent` | the agent: job, physics briefing, tool list, two worked examples |
| `config.py` | Wine/prefix/timeout resolution for SCAPS |
| `tests/test_scaps_design.py` | 29 tests, incl. field-by-field equivalence with a SCAPS-authored reference device |
| `README.md` | the agent contract and the three layout rules |

`physics/devsim_pv.py` (an earlier DevSim attempt) was deleted; the user specified SCAPS-1D.

## 3. The three rules that were wrong before

**Layer order never inverts.** Both architectures are written `HTL | absorber | ETL`, which is what
SCAPS reads. `n-i-p` is illuminated from the right (ETL side), `p-i-n` from the left (HTL side).
"SnO₂ on one side" means the ETL in both cases; for p-i-n it is the far side from the light. An
earlier version of the code and of this document claimed p-i-n reverses the layer order, which is
wrong and produced an unsolvable device.

**Native `Nt(y)` profiles are ignored by this SCAPS build.** `profile : linear` with `Nleft`/`Nright`
was measured to have no effect on the result. Graded profiles are therefore implemented as
uniform-`Nt` sub-layers, which SCAPS does honour. Depth is measured from the illuminated surface,
and the block order is reversed for n-i-p so the profile still refers to the lit side.

**A banner is not a convergence.** SCAPS prints its completion banner as soon as the window opens
and can then stop the sweep around 0.76 V. A result is only reported when `Voc`, `Jsc`, `FF` and
`eta` are all present; otherwise `sweep_completed` is false and the metrics are absent by design.
The default range is 0–1.6 V.

## 4. Test evidence

```
78 passed, 1 skipped          # full suite, ~61 s
74 passed, 1 skipped, 1 deselected   # pytest -m "not slow", ~18 s
```

The slow marker covers one live end-to-end run (agent-authored `.def` → solver → numbers parsed
back). The strongest test is numeric, not textual: the generated n-i-p `.def` is parsed field by
field and compared against `make_perovskite_def.py`, a SCAPS-authored reference, for every layer,
every contact and every working-point setting.

`test_scaps_agent_hides_live_tools_and_says_why` was failing before this work. It was a real bug:
the "simulation is off" block was a plain string, so the agent was told SCAPS was unavailable but
never told why, because the placeholder was never interpolated. Fixed at `team.py:301`.

## 5. Honest limitations

* **Transport parameters dominate.** Mobilities, DOS and permittivity are literature-typical
  presets, not measurements. The 4-point graded-vs-uniform gap above is larger than the
  sensitivity to the assumed SnO₂-vs-TiO₂ mobility, but it is still a model comparison and not a
  claim about a real device.
* **Solve time scales with sub-layer count.** 2 and 4 blocks finish in under a minute; 8 exceeded
  600 s and was abandoned. The agent is told this in its briefing so it does not casually raise it.
* **Only 1D, only one device per run.** No parameter sweeps, per the user's instruction. The
  `simulate_defect_profile` tool runs two points because the question it answers is a comparison.
* **Live SCAPS is machine-local.** It needs a Wine prefix with SCAPS 3.3.09. When that is missing,
  `design` still lays out devices and writes valid files, but `simulate_*` tools are withheld from
  it and its prompt says no number can be given.
* **`claude_code_perovagents/` is untracked** in the parent git repository. Nothing here is
  committed.

## 6. How to run

```bash
cd claude_code_perovagents
export GOOGLE_API_KEY=...
python -m perovagents --agent design \
  "Design a p-i-n PSC with SnO2 on one side and tell me the defect density profile." -v --trace
python -m perovagents --show-agent design      # the full briefing the agent sees
pytest -m slow                                # the live solver contract
streamlit run app.py                          # http://localhost:8502
```

## 7. Recommended next steps

1. Move `scaps_def.py`/`scaps_run.py`/`design.py` into the four-cell `perovagents/` project and
   register the design cell there; the SCAPS execution layer should not be forked per project.
2. Add an `Nt(y)`-capable path once a SCAPS build that honours `profile : linear` is available, so
   sub-layer cost stops scaling with profile resolution.
3. Replace the transport presets with values traceable to a DOI, and record the citation per layer
   in the `.def` provenance so a run can be audited after the fact.