# Benchmark version 2 (preserved, closed)

Verbatim copies of the files that defined benchmark v2 (freeze_id 2): `episode_v2.py`,
`providers_v2.py`, `agent_v2.py` and `progress_v2.py` (the framework copies before the v3
fixes), and `FREEZE_v2.json`. v2 results stay in `runs/`, `scored/`, `tables/`, `figures/`,
`audit/` (default names). v3 writes to `v3/` and must not be pooled with v2.
To re-analyse v2: `BENCH_VERSION=v2 python -m benchmarks.agentic_paper.audit`.
