# Case studies (not part of the main comparison)

## Q13 - design and run a p-i-n cell (live SCAPS)

`Q13_design_and_simulation/` holds the two historical pilot records
(`Q13.gemini.json` = direct tool use, `Q13.mrs.json` = PerovAgents, both on
Gemini, run on 2026-10-08) exactly as produced by the original harness. They are
preserved unchanged and are NOT re-run, re-scored or mixed into the main
statistics.

Reference (same solver, identical input files, one live run on 2026-10-08):
Voc 1.3056 V, Jsc 27.13 mA/cm2, FF 90.28 %, PCE 31.98 % (see
`benchmarks/cross_llm_gold.json` -> `scaps.Q13`).

Both systems reported these numbers. As in the manuscript, this is evidence that
the design -> solve -> parse path ran end to end once; it is not evidence that
the simulated efficiency is physically realistic (the transport parameters are
code presets). Note also that the reference is itself a live solver run, so
"matching the reference" tests faithful reporting only.

Other questions that need the live solver (Q14, Q15, Q16, Q20) are frozen in the
question bank under `set: "live_scaps_case_study"` and are excluded from the main
set: the benchmark does not launch repeated live SCAPS simulations, and the
solver environment is not reliably available (see `docs/scaps_wine_diagnosis`).
