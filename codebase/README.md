# PerovAgents (simple version)

```
question -> Orchestrator --ASK pdb: ...--------> Agent: loop { call tool, read result }
              ^   \--ASK ml: ... (in parallel)         until it writes "FINAL ANSWER: <report>"
              |                                              |
              +----------- reports come back <---------------+
   decides: ASK again / ASK someone else / "FINAL ANSWER: <answer for the user>"
```

* `perovagents/agent.py` (~110 lines): `Agent` and `Orchestrator`. That is the whole framework.
* `perovagents/team.py`: the five agents (pdb, literature, ml, scaps, design). Each is a job
  description, a "what you know" text (data dictionary, units, pitfalls) and a list of tools =
  plain Python functions. The agent's prompt shows each tool's name, arguments and docstring
  automatically.
* Everything else (`ml/`, `physics/`, `literature/`, `materials/`, `llm.py`, `config.py`) is the
  science backends and LLM access from before.

```bash
pip install -e ".[all]"
export GOOGLE_API_KEY=...                         # or `opencode serve`
python -m perovagents "Average PCE of devices with a SnO2 ETL?" -v
python -m perovagents "..." -v --trace            # also write results/traces/*.jsonl
python -m perovagents --agent ml "Predict a p-i-n MAPbI3/PTAA/SnO2 cell" -v
python -m perovagents --agent design "Design a p-i-n PSC with SnO2 and give me Voc/Jsc/FF/PCE" -v
python -m perovagents --show-agent design         # exactly what an agent knows
pytest                                            # offline tests
pytest -m slow                                    # the one live SCAPS run
```

## Live device design with SCAPS-1D

`--agent design` does not describe a device, it builds one and simulates it. The agent calls four
tools, and every number it reports has to come out of `simulate_stack`:

| tool | what it does |
|------|--------------|
| `design_stack` | validates the request and returns the whole device in engineering units; runs nothing |
| `write_execution_script` | writes the real `.def` and `.script` files without running them |
| `simulate_stack` | writes the files, runs SCAPS-1D, returns Voc/Jsc/FF/PCE plus both file paths |
| `simulate_defect_profile` | simulates a graded profile against a uniform one and reports the difference |

```bash
python -m perovagents --agent design \
  "Design a p-i-n PSC with SnO2 on one side and tell me the defect density profile." -v --trace
```

Three things to know about how the device is laid out:

* **Layer order never inverts.** Both architectures are written `HTL | absorber | ETL`, because
  that is what SCAPS reads. `n-i-p` is illuminated from the right (ETL side), `p-i-n` from the
  left (HTL side). "SnO2 on one side" means the ETL in either case.
* **Depth profiles are real sub-layers.** `defect_profile` accepts a number, `exp-tail`,
  `linear`, `step` or explicit `points`; a graded profile is cut into `sublayers` uniform-`Nt`
  blocks (default 4). Depth is measured from the illuminated surface. Native `profile : linear`
  with `Nleft`/`Nright` is silently ignored by this SCAPS build, which is why sub-layers are used.
* **A sweep is not a success.** SCAPS prints its banner as soon as the window opens and can then
  stop around 0.76 V. A result is only reported when `Voc`, `Jsc`, `FF` and `eta` are all
  present; otherwise you get `sweep_completed: false` and the numbers are missing on purpose.

Wine, the prefix and the timeout are resolved in `config.py` (`PEROVAGENTS_SCAPS_WINE`,
`PEROVAGENTS_SCAPS_PREFIX`, `PEROVAGENTS_SCAPS_TIMEOUT`). Jobs are serialised with a lock, because
SCAPS is a GUI process that stays alive after a run. If SCAPS is missing, `design` still lays out
devices and writes files, but the agent is told that no number was produced.

### SCAPS under wine: required configuration

| Variable | Meaning |
|---|---|
| `PEROVAGENTS_SCAPS_WINE` | a **`bin/wine64`** (or equivalent) that starts `scaps3309.exe`; set it explicitly, auto-detection is only a fallback |
| `PEROVAGENTS_SCAPS_PREFIX` | wine prefix that contains `drive_c/Program Files (x86)/Scaps3309` |
| `PEROVAGENTS_SCAPS_DYLD` | optional, colon-separated library dirs for `DYLD_FALLBACK_LIBRARY_PATH` |
| `PEROVAGENTS_SCAPS_TIMEOUT` | seconds before a run is killed (default 600; a healthy run takes ~50 s) |

Verified runtime: the Sikarugir `wswine.bundle` inside a game app (wine-10.0), e.g.
`~/Applications/F1 22 AnkerGames.app/Contents/Frameworks/wswine.bundle/bin/wine64`.
**The bundle's `Frameworks` directory must be on `DYLD_FALLBACK_LIBRARY_PATH`** (FreeType lives
there). `config.scaps_env()` does this automatically for the `wswine.bundle` layout and both live
paths use it. Without it wine prints "Wine cannot find the FreeType font library" and
`scaps3309.exe` spins at ~100% CPU without ever writing an IV file, until the timeout.
For any other runtime layout set `PEROVAGENTS_SCAPS_DYLD` yourself. CrossOver's own `bin/wine`
is a Perl wrapper that insists on a registered bottle, so it cannot be pointed at a plain prefix;
use a wrapper that execs its `bin/wineloader` (see `docs/scaps_wine_diagnosis/cx_wine64.sh`).

```bash
export PEROVAGENTS_SCAPS_WINE="$HOME/Applications/F1 22 AnkerGames.app/Contents/Frameworks/wswine.bundle/bin/wine64"
export PEROVAGENTS_SCAPS_PREFIX=/path/to/opencode_sandbox/scaps/prefix
python -m perovagents.physics.scaps_run path/to/job.script      # one live run
```

Never experiment on the production prefix: clone it (`cp -Rc prefix prefix_test`, APFS
copy-on-write) and point `PEROVAGENTS_SCAPS_PREFIX` at the clone.

Data lives in `results/` (`sqlite/`, `ml/`, `literature/`). If you keep those artifacts in
another checkout, symlink them in or point `PEROVAGENTS_RESULTS_DIR` at it:

```bash
ln -s /path/to/perovagents/results/sqlite results/sqlite   # and ml/, literature/
export PEROVAGENTS_RESULTS_DIR=/path/to/perovagents/results
```

Keys rotate across `GOOGLE_API_KEY`, `chatbot/.env` and `~/keys.md` (deduplicated, in that
order). A key that reports an invalid key is skipped for 24 h, a 429/quota key for 60 s, and
a transient 5xx for 20 s. If every key is cooling, the process-wide breaker short-circuits
instead of re-walking the dead pool, and `opencode serve` (then a deterministic responder)
answers instead. `perovagents.llm.reset_pool_state()` clears all of that.

Add a tool: write a function with a docstring, add it to that agent's list in `team.py`.
Change behaviour: edit the agent's job / knowledge text or `AGENT_RULES` in `agent.py`.
