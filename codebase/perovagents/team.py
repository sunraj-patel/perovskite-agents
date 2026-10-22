"""The team: four agents, each = a job description, what it knows, and its tools.

A tool is a plain Python function; its name, arguments and docstring are what the agent
sees. Add a tool by writing a function and adding it to the agent's list.
"""
import json
import os
import re
import sqlite3
import time
from collections import OrderedDict

from . import config
from .agent import DONE, Agent
from .progress import contract_for


def _tool_named(reply):
    """The tool name a sample agent reply calls, or None if it is not a tool call."""
    m = re.search(r'"tool"\s*:\s*"([^"]+)"', reply or "")
    return m.group(1) if m else None


def _calc(expression: str):
    """Evaluate an arithmetic expression exactly, e.g. '(21.3-18.9)/18.9*100'. Use it for any number you derive."""
    if not re.fullmatch(r"[\d\s+\-*/().eE]+", expression) or "**" in expression:
        raise ValueError("only numbers and + - * / ( ) are allowed")
    return eval(expression, {"__builtins__": {}}, {})


def _json_file(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# ============================================================================ PDB
PDB_NOTES = """\
One SQLite table, `devices`: one row per solar-cell device reported in a paper. Column names are
prefix_suffix (full list above). Key columns:
  JV_default_PCE (%), JV_default_Voc (V), JV_default_Jsc (mA/cm2), JV_default_FF (FRACTION 0-1)
  Cell_architecture ('nip' or 'pin'); Ref_DOI_number (the paper)
  ETL_stack_sequence, HTL_stack_sequence, Backcontact_stack_sequence: layers joined by ' | ';
    match a material with LIKE '%SnO2%' (not =)
  Perovskite_composition_short_form / _long_form, Perovskite_composition_a_ions / _b_ions / _c_ions
  Perovskite_band_gap (eV) and Perovskite_thickness (nm) are TEXT: use CAST(x AS REAL) and skip cells containing '|'
Tips:
  - missing values are NULL; AVG/COUNT skip them. Report COUNT(*) devices and COUNT(DISTINCT Ref_DOI_number)
    papers next to any average, and how many non-null values it used.
  - outliers exist (PCE up to 36 and 0, Voc up to 7 V, FF up to 5): filter sensibly (e.g. JV_default_PCE > 0)
    and say which filter you used.
  - check the exact spelling of a material with column_values before filtering on it.
  - SQLite has no MEDIAN; results are capped at 200 rows and you see the first 30: use GROUP BY / aggregates."""


def pdb_agent(llm):
    if not os.path.exists(config.PDB_SQLITE):
        raise FileNotFoundError(f"Perovskite Database not found at {config.PDB_SQLITE}")

    def connect():
        return sqlite3.connect(f"file:{config.PDB_SQLITE}?mode=ro", uri=True)     # read-only

    conn = connect()
    columns = [r[1] for r in conn.execute("PRAGMA table_info(devices)")]
    n_rows, n_papers = conn.execute("SELECT COUNT(*), COUNT(DISTINCT Ref_DOI_number) FROM devices").fetchone()
    conn.close()
    groups = OrderedDict()
    for c in columns:
        groups.setdefault(c.split("_")[0], []).append(c)
    listing = "\n".join(f"  {p}_: " + ", ".join(c[len(p) + 1:] or "(itself)" for c in cs) for p, cs in groups.items())
    knowledge = (f"The Perovskite Database: {n_rows:,} devices from {n_papers:,} papers, {len(columns)} columns.\n"
                 f"ALL COLUMNS by prefix:\n{listing}\n\n{PDB_NOTES}\n\n"
                 f"AGGREGATION. To answer 'top-N', 'most frequent', 'how many', write ONE SQL with "
                 f"GROUP BY 1 + ORDER BY 2 DESC + LIMIT (see `column_values`) rather than fetching raw rows "
                 f"and counting them by eye. Each distinct question needs its own query. A query that already "
                 f"succeeded must NOT be repeated with the same or near-identical SQL - if it did not answer "
                 f"the question, the question needed a different statistic, so change the aggregation "
                 f"(SELECT/GROUP BY/COUNT), not the row limit.")

    def sql(query: str):
        """Run one read-only SQLite SELECT on table `devices`. Returns columns, the first 30 rows and the row count."""
        con, t0 = connect(), time.time()
        try:
            con.set_progress_handler(lambda: 1 if time.time() - t0 > 20 else 0, 5000)
            cur = con.execute(query)
            rows = cur.fetchmany(201)
            return {"columns": [d[0] for d in cur.description], "n_rows": min(len(rows), 200),
                    "rows": [dict(zip([d[0] for d in cur.description], r)) for r in rows[:30]]}
        except sqlite3.OperationalError as exc:
            hint = ""
            m = re.search(r"no such column: (\S+)", str(exc))
            if m:
                words = [w for w in re.split(r"\W+", m.group(1).lower()) if w]
                hint = f" Similar columns: {[c for c in columns if any(w in c.lower() for w in words)][:8]}"
            raise ValueError(str(exc) + hint)
        finally:
            con.close()

    def find_columns(keyword: str):
        """List column names containing all the given words, e.g. 'HTL additive'."""
        words = keyword.lower().split()
        return [c for c in columns if all(w in c.lower() for w in words)][:40]

    def column_values(column: str, top: int = 15):
        """Most common values of one column with counts. Use it to learn exact spellings before filtering."""
        con = connect()
        try:
            return con.execute(f'SELECT "{column}", COUNT(*) FROM devices WHERE "{column}" IS NOT NULL '
                               f'GROUP BY 1 ORDER BY 2 DESC LIMIT ?', (top,)).fetchall()
        finally:
            con.close()

    return Agent("pdb", "You answer questions about EXPERIMENTAL perovskite solar-cell devices by writing SQL "
                        "against the Perovskite Database (statistics, counts, rankings, comparisons).",
                 knowledge, [sql, find_columns, column_values, _calc], llm, contract=contract_for("pdb"))


# ============================================================================ Literature
def literature_agent(llm):
    from .literature.corpus import Corpus
    corpus = Corpus()
    papers = corpus.papers()
    if not papers:
        raise FileNotFoundError("literature corpus is empty")
    listing = "\n".join(f"  {p['doi']} - {p['title'][:90]}" for p in papers[:60])
    knowledge = (f"A fixed collection of {len(papers)} perovskite solar-cell papers, split into short passages. "
                 "'Not found' means 'not in this collection', never 'does not exist'. Passages are raw "
                 "extracted text (markdown noise, broken symbols); some titles below are journal headers, "
                 "judge a paper by its DOI and passages. Search is semantic when the embedding index is "
                 "available, otherwise keyword-based (each result says which): then use the specific words a "
                 f"paper would use.\nPAPERS (DOI - title):\n{listing}\n\n"
                 f"Each search or read_paper should be called ONCE with distinct arguments: repeating the same "
                 f"search terms, or reading the same DOI and query again, returns the same passages and cannot "
                 f"add evidence. If a paper lacks what you need, read a different DOI or change the query; "
                 f"if it has it, quote it and move to the next question.")

    def search(query: str, k: int = 6, doi: str = None):
        """Find the passages most relevant to a query, across all papers or within one paper (doi)."""
        return corpus.search(query, k=k, doi=doi)

    def read_paper(doi: str, query: str = None):
        """Read passages of ONE paper: those matching `query`, or the opening ones."""
        return corpus.read_paper(doi, query)

    return Agent("literature", "You find what the papers in the collection say and report it with DOIs and short "
                               "verbatim quotes. You never state anything about a paper you did not read in a passage.",
                 knowledge, [search, read_paper, _calc], llm, contract=contract_for("literature"))


# ============================================================================ ML
def ml_agent(llm):
    from .ml.controllability import DECLARED_RANGES
    from .ml.features import MATERIAL_CANON, MATERIAL_KEYWORDS
    from .ml.predictor import MLPredictorTool
    model = MLPredictorTool()
    acc = _json_file(os.path.join(model.dir, "validation.json"))
    accuracy = "\n".join(f"  {k}: RMSE {v['score']:.2f} pp, R2 {v['r2']:.2f}" for k, v in acc.items()
                         if isinstance(v, dict) and "score" in v) or "  (no validation record found)"
    mats = {r: ", ".join(sorted({MATERIAL_CANON.get(k, k) for k in ks})) for r, ks in MATERIAL_KEYWORDS.items()}
    knowledge = f"""\
CatBoost models trained on the Perovskite Database predict PCE (%) of a device from a pre-fabrication
description. Models: overall, nip, pin. Accuracy on papers unseen in training (grouped 5-fold CV):
{accuracy}
  (typical error 3-4 PCE points: differences smaller than that are not resolved by the model)
DESCRIPTION FORMAT (free text, parsed by fixed rules): write 'p-i-n' or 'n-i-p' (default n-i-p), e.g.
  "p-i-n cell, PTAA HTL, SnO2 ETL, MAPbI3 perovskite 500 nm, ETL of 30 nm, annealed at 100 C for 10 min"
  ETL recognised: {mats['ETL']}
  HTL recognised: {mats['HTL']}
  perovskite: a formula such as MAPbI3, FAPbI3, Cs0.05FA0.79MA0.16PbBr0.51I2.49, or CsMAFA (triple cation)
  Anything not recognised is silently filled with typical training values; an unlisted material becomes
  'unknown'. Check `resolved_materials` in the result and tell the reader what was assumed.
WHAT_IF sweeps ONE continuous input; declared ranges: {', '.join(f'{k} {v[0]:g}-{v[1]:g}' for k, v in DECLARED_RANGES.items())}
  (measured outputs such as Voc/Jsc/FF and categorical choices cannot be swept).
Results include `pce_95_interval` and `hood` (hood.ood true = input outside training range, an extrapolation).
ALWAYS report the prediction WITH its 95% interval and the model's typical error, mention out-of-domain or
assumed inputs, and describe SHAP / what-if / optimise output as what the MODEL associates with PCE
(not causes, not measurements)."""

    def _run(action, architecture, **kw):
        out = model.call({"action": action, "architecture": architecture, **kw})
        if not isinstance(out, dict) or out.get("status") == "error" or "error" in out:
            raise ValueError((out or {}).get("error", "model call failed"))
        return out

    def predict(description: str, architecture: str = "overall"):
        """Predict PCE (%) with a 95% interval for a described device. architecture: overall | nip | pin."""
        return _run("predict", architecture, description=description)

    def what_if(description: str, parameter: str, values: list, architecture: str = "overall"):
        """Sweep one continuous input (e.g. 'perovskite thickness', values [300, 400, 500]) and get PCE at each value."""
        return _run("what_if", architecture, description=description, parameter=parameter, values=values)

    def explain_local(description: str, architecture: str = "overall"):
        """SHAP: which inputs the model associates most with THIS device's predicted PCE."""
        return _run("explain_local", architecture, description=description, top_k=10)

    def explain_global(architecture: str = "overall"):
        """SHAP: which inputs matter most for PCE across designs, in this model."""
        return _run("explain_global", architecture, top_k=15)

    def optimize(description: str, architecture: str = "overall"):
        """Rank controllable changes (thickness, annealing, material swaps) by predicted PCE gain."""
        return _run("optimize", architecture, description=description, top_k=3)

    return Agent("ml", "You run the trained PCE models: predictions with uncertainty, SHAP explanations, "
                       "what-if sweeps and optimisation suggestions for described devices.",
                 knowledge, [predict, what_if, explain_local, explain_global, optimize, _calc], llm,
                 contract=contract_for("ml"))


# ============================================================================ SCAPS
def scaps_agent(llm):
    from .physics.scaps import (PARAM_SPECS, PUBLIC_TO_INTERNAL, REFERENCE_CELL,
                                SCAPSTool)
    scaps = SCAPSTool()
    try:
        df = scaps.sweeps
        have = "\n".join(f"  {k}: {len(g)} points, {g['param'].min():g} to {g['param'].max():g}"
                         for k, g in df.groupby("sweep"))
    except Exception as exc:
        raise FileNotFoundError(f"no precomputed SCAPS sweeps ({exc})")

    defaults = "\n".join(f"  {k} = {v[0]:g} {v[1]}" for k, v in REFERENCE_CELL.items())
    layers = sorted({s["layer_name"] for s in PARAM_SPECS.values()})
    fields = sorted({s["description"] for s in PARAM_SPECS.values()})

    knowledge = f"""\
SCAPS-1D solves drift-diffusion for a reference n-i-p perovskite cell (300 K, AM1.G, {len(layers)}
layers: {', '.join(layers)}). Every layer carries its own thickness, band gap, electron affinity,
permittivity, DOS, mobilities, doping and defect density, and you can set ANY of them and re-run
the simulator - you are not limited to the parameters listed in a sweep. It is a physics
sensitivity model: it shows trends and mechanisms, never measured devices. Always say it is a
simulation.

PARAMETER NAMING. Write a parameter as <layer>_<field>, e.g. pvk_Nt, htl_Nt, etl_Nt, pvk_d,
htl_chi, pvk_mun. The three layer prefixes are pvk, htl, etl. Public units: thickness in nm,
mobility in cm^2/Vs, every density (Nc, Nv, Na, Nd, Nt) in 1/cm^3, energies in eV, cross-sections
in cm^2. Settable fields: {', '.join(fields)}.
Fields you can sweep or set: {', '.join(sorted(k for k in PARAM_SPECS if '_' in k and k.count('_') == 1))}
Shorthand for the precomputed sweeps: {PUBLIC_TO_INTERNAL}

WHAT THIS REFERENCE CELL ACTUALLY USES (verify with read_def before quoting it):
{defaults}

DEFECT DENSITIES. Nt is the SRH bulk defect density of a layer and is the single biggest lever
on Voc and on every recombination pathway, so it is worth sweeping. Ranges that appear in SCAPS
studies of perovskite devices, as guidance for choosing sweep points (not measurements):
  pvk_Nt  1e13-1e16 1/cm^3 for a well-crystallised absorber, 1e17-1e20 for a poor or amorphous one.
          Values at or above ~1e19 usually collapse Voc hard, which is the point of the sweep.
  htl_Nt  1e15-1e18 1/cm^3 for Spiro-MeOTAD; it is a resistive layer, so it mainly costs FF.
  etl_Nt  1e16-1e19 1/cm^3 for TiO2 or SnO2; drives recombination at the ETL/perovskite junction.
  Doping: htl_Na and etl_Nd are around 1e18-1e20 1/cm^3.
  Interfaces are usually given a lower Nt than the bulk (1e10-1e13 1/cm^3 when passivated).
If a task asks what value to USE, quote these ranges as typical practice and then confirm the
effect on this device with a sweep, rather than asserting one number is correct.

Precomputed sweeps available through get_sweep (instant, no simulator needed):
{have}
Sanity: PCE = Voc x Jsc x FF; the detailed-balance ceiling is about 30-34 %. Precomputed data
records no solver convergence state, so say so rather than claiming a run converged."""

    # ---- tools --------------------------------------------------------
    def get_sweep(sweep: str):
        """Read one precomputed sweep instantly: Voc, Jsc, FF, PCE at each value of a parameter,
        plus the PCE trend. Sweep names: d (thickness), eg (band gap), mue, muh, htlchi, htlEg.
        Use this first for any parameter that already has a sweep; it needs no simulator."""
        out = scaps.get_sweep(sweep)
        if not out.get("found"):
            raise ValueError(out.get("error") or f"no sweep {sweep!r}")
        return out

    def reference_cell():
        """Return the reference device's full parameter set in public units (thickness nm, mobility
        cm^2/Vs, densities 1/cm^3). Use this to state exactly what the baseline simulation uses."""
        return {"found": True, "stack": "FTO | ETL TiO2 | Perovskite | HTL Spiro-MeOTAD | Au (n-i-p)",
                "parameters": {k: {"value": v[0], "unit": v[1]}
                               for k, v in REFERENCE_CELL.items()}}

    def list_parameters(layer: str = None):
        """List every parameter that can be set or swept, with its unit and default value.
        Optionally filter by layer: 'pvk', 'htl' or 'etl'. Call this when you are unsure of a
        parameter name instead of guessing one."""
        return scaps.describe_parameters(layer)

    def read_def(def_file: str = "perovskite_nip.def"):
        """Read a generated .def file back and report every layer's real values, in public units.
        Use this to verify what a simulation actually used before quoting it."""
        return scaps.read_def(def_file)

    tools = [get_sweep, reference_cell, list_parameters, read_def, _calc]

    live = scaps.live_status()
    if live["available"]:
        def run_point(parameters: dict, label: str = "custom"):
            """Simulate ONE device with any combination of parameters set at once, e.g.
            {"pvk_Nt": 1e15, "etl_Nt": 1e17, "htl_d": 150}. Anything omitted keeps its default.
            Returns Voc, Jsc, FF, PCE and the .def file used. Takes tens of seconds."""
            return scaps.run_point(parameters, label)

        def sweep(parameter: str, values: list):
            """Sweep ANY settable parameter over 1-12 values, in that parameter's public unit, and
            return Voc, Jsc, FF, PCE at each value plus the PCE trend. Works for defect densities
            (pvk_Nt, htl_Nt, etl_Nt), thicknesses, doping, mobilities, affinities, gaps, and more -
            not just the precomputed ones. Slow: about a minute per point."""
            return scaps.run_sweep_live(parameter, [float(v) for v in values])

        tools += [run_point, sweep]
    else:
        # Do not hand the agent a tool that cannot work: it would spend its
        # whole turn budget on failed runs and then report nothing.
        knowledge += f"""

LIVE SIMULATION IS OFF on this machine, so run_point and sweep are NOT available to you.
Reason: {str(live.get("error"))[:300]}
Answer with the precomputed sweeps (get_sweep), the reference cell (reference_cell) and
read_def, and give typical-value guidance from the ranges above. Say plainly that a live
re-run was not possible, and what would have to change to enable it."""

    examples = [
        {
            "task": "What defect density does the perovskite layer use, and what happens to Voc if "
                    "it is 1000x worse?",
            "turns": [
                {"reply": '{"tool": "read_def", "args": {}}'},
                {"result": "{'found': True, 'stack': ['HTL', 'Perovskite', 'ETL'], 'layers': "
                           "{'Perovskite': {'pvk_Nt': {'value': 1e14, 'unit': '1/cm^3'}, "
                           "'pvk_d': {'value': 400.0, 'unit': 'nm'}} ...}}"},
                {"reply": '{"tool": "sweep", "args": {"parameter": "pvk_Nt", '
                          '"values": [1e14, 1e15, 1e16, 1e17]}}'},
                {"result": "{'found': True, 'parameter': 'Perovskite SRH defect density "
                           "(the dominant recombination knob)', 'unit': '1/cm^3', 'n_points': 4, "
                           "'data': [{'value': 1e14, 'Voc': 1.12, 'Jsc': 23.9, 'FF': 84.1, "
                           "'PCE': 22.5}, ...], 'pearson_r_PCE': -0.91, 'trend': 'negative'}"},
                {"reply": f"{DONE} The perovskite layer is set to Nt = 1e14 1/cm^3 in the reference "
                          "cell. Sweeping it 10x to 1e17 drops Voc from 1.12 V to 0.86 V and PCE "
                          "from 22.5% to 15.1% (Pearson r = -0.91), because SRH recombination "
                          "scales with Nt. This is a simulation, not a measurement."},
            ],
        },
        {
            "task": "How thick should the Spiro layer be?",
            "turns": [
                # A mistake on purpose: the agent guesses a parameter that does not exist,
                # reads the error, and corrects itself.
                {"reply": '{"tool": "sweep", "args": {"parameter": "htl_thickness_nm", '
                          '"values": [50, 100]}}'},
                {"result": "ERROR: KeyError: unknown parameter 'htl_thickness_nm'. Valid examples: "
                           "['etl_Eg', 'etl_Et', 'etl_Na', ...]"},
                {"reply": '{"tool": "list_parameters", "args": {"layer": "htl"}}'},
                {"result": "{'found': True, 'n_parameters': 22, 'parameters': [{'name': 'htl_d', "
                           "'layer': 'HTL', 'field': 'layer thickness', 'unit': 'nm', "
                           "'default': 200.0} ...]}"},
                {"reply": '{"tool": "sweep", "args": {"parameter": "htl_d", '
                          '"values": [50, 100, 200, 300]}}'},
                {"result": "{'found': True, 'parameter': 'HTL layer thickness', 'unit': 'nm', "
                           "'data': [{'value': 50, 'PCE': 19.8}, {'value': 100, 'PCE': 21.4}, "
                           "{'value': 200, 'PCE': 22.5}, {'value': 300, 'PCE': 21.9}], "
                           "'trend': 'positive'}"},
                {"reply": f"{DONE} PCE peaks at the 200 nm default (22.5%) and falls off on both "
                          "sides - too thin adds series resistance, too thick adds transport loss. "
                          "Simulated, not measured."},
            ],
        },
        {
            "task": "Is the reference cell's band gap optimal?",
            "turns": [
                {"reply": '{"tool": "get_sweep", "args": {"sweep": "eg"}}'},
                {"result": "{'found': True, 'parameter': 'perovskite_band_gap', 'unit': 'eV', "
                           "'n_points': 8, 'data': [{'value': 1.35, 'PCE': 29.8}, "
                           "{'value': 1.55, 'PCE': 24.3} ...], 'trend': 'negative'}"},
                {"reply": f"{DONE} No. Across the precomputed 1.35-1.70 eV sweep PCE falls "
                          "monotonically (29.8% at 1.35 eV to 19.7% at 1.70 eV), so the "
                          "reference 1.55 eV is well above the simulated optimum. Note this is a "
                          "single-junction simulation with no tandem stacking."},
            ],
        },
    ]
    # Drop examples that call tools this machine does not have.
    if not live["available"]:
        tools_avail = {"get_sweep", "reference_cell", "list_parameters", "read_def", "_calc"}
        for ex in examples:
            ex["turns"] = [t for t in ex["turns"]
                           if _tool_named(t.get("reply", "")) in tools_avail]

    return Agent("scaps",
                 "You read and run SCAPS-1D device simulations to show how a perovskite cell's "
                 "performance depends on physical parameters, layer by layer. You can set any "
                 "layer property, not only the ones with a precomputed sweep.",
                 knowledge, tools, llm, examples=examples, contract=contract_for("scaps"))


# ============================================================================ team
# ============================================================================ DESIGN
def design_agent(llm):
    """The device-design agent: proposes a stack, writes the SCAPS files, simulates it.

    Separate from `scaps_agent`, which interrogates an existing reference cell.
    This one builds a *new* device from a request.
    """
    from .physics import scaps_def as D
    from .physics.design import (design_stack, simulate_defect_profile,
                                 simulate_stack, write_execution_script)
    from .physics.scaps_run import check_installation

    install = check_installation()
    catalogue = []
    for name, info in D.materials_catalogue().items():
        vals = info["values"]
        catalogue.append(
            f"  {name} ({info['role']}): Eg {vals['Eg'][0]} eV, chi {vals['chi'][0]} eV, "
            f"mun {vals['mun'][0]:g} mup {vals['mup'][0]:g} cm2/Vs, "
            f"eps {vals['eps'][0]:g}, Nd {vals.get('Nd', (0,))[0]:g} "
            f"Na {vals.get('Na', (0,))[0]:g} 1/cm^3")
    materials = "\n".join(catalogue)

    knowledge = f"""\
You design perovskite solar cells and then SIMULATE them with SCAPS-1D. You do not just
describe a design: you write the SCAPS input files and run the solver, and you report the numbers it
returns. Any performance figure in your report must come from a simulate_* tool result.

STACK GEOMETRY, and it is the part people get wrong.
In a SCAPS .def the hole-transport layer is always written FIRST (the back/left contact side) and
the electron-transport layer LAST (the front/right contact side), for BOTH architectures. What
changes between architectures is only which side the light enters:
  n-i-p : HTL | absorber | ETL, illuminated from the right, i.e. the ETL is the lit surface
  p-i-n : HTL | absorber | ETL, illuminated from the left,  i.e. the HTL is the lit surface
So "SnO2 on one side" is the ETL either way; for p-i-n it is the far side from the light.
Layer order in the file never inverts. If a request implies an inverted stack, say so and build the
conventional one rather than producing a file SCAPS cannot solve.

MATERIALS (literature-typical defaults; override any field with the *_overrides arguments, which
take engineering units: nm, eV, 1/cm^3, cm^2/Vs, cm/s):
{materials}

DEFECT DENSITIES. Nt is the SRH bulk defect density and the dominant lever on Voc. Usual ranges
(typical practice, not measurements of your device):
  absorber  1e13-1e16 1/cm^3 for a well-crystallised film, 1e17-1e20 for a poor one
  HTL       1e15-1e18 1/cm^3 (resistive, mainly costs FF)
  ETL       1e16-1e19 1/cm^3 (drives recombination at the ETL/absorber junction)
A depth-dependent profile is supported and is worth using when asked about one: exp-tail
(surface_Nt, bulk_Nt, decay_nm), linear, step, or explicit [[depth_fraction, Nt], ...] points. It
is implemented by splitting the absorber into uniform-Nt sub-layers, which is what this SCAPS
build actually honours - `profile : linear` with Nleft/Nright is silently ignored by it. Depth is
measured from the illuminated surface, so `surface_Nt` is the density where the light enters.
Sub-layers cost solve time (4 is a sane default; 8+ may not finish), so do not raise it casually.

TRANSPORT PARAMETERS DOMINATE. Mobilities, DOS and permittivity set FF and Jsc more than the
defect density does, and they are the least trustworthy inputs in the model. Report the assumed
mobility of the absorber and of each transport layer, and say plainly that a transport-parameter
assumption is what the result is most sensitive to.

WHICH TOOL WHEN.
  simulate_stack         lay out AND run in one call - use this by default
  design_stack           show the layout and every value BEFORE committing a run
  write_execution_script author the .def and .script without running them - use this when
                         the question asks for the files, the script, or what would be run
  simulate_defect_profile  answer "does the depth profile matter" with two real runs
simulate_stack executes the same code path as write_execution_script, so the files you
show are literally the files that get solved. Always return the def_path and script_path.

WHAT TO REPORT. Architecture, every layer with thickness and the defect density that was used, then
Voc, Jsc, FF, PCE and where they came from (the .def and .script paths are in the result). Always
call it a simulation. Mention what you could not determine.

LIVE SIMULATION: {'available (wine ' + str(install['wine_version']) + ')' if install['available'] else 'UNAVAILABLE - ' + str(install['error'])}."""

    if not install["available"]:
        knowledge += """

Live SCAPS is not available on this machine, so simulate_stack and simulate_defect_profile will
fail. You can still lay out a device with design_stack and author the real SCAPS files with
write_execution_script, and you must then say plainly that no simulation was run and no performance
number can be given."""

    tools = [design_stack, write_execution_script, _calc]
    if install["available"]:
        tools += [simulate_stack, simulate_defect_profile]

    examples = [
        {
            "task": "Design a p-i-n PSC with SnO2 on one side and tell me the defect density profile.",
            "turns": [
                {"reply": '{"tool": "design_stack", "args": {"architecture": "p-i-n", '
                          '"etl": "SnO2", "htl": "PTAA", "absorber_thickness_nm": 500, '
                          '"htl_thickness_nm": 150, "etl_thickness_nm": 50, '
                          '"defect_profile": {"kind": "exp-tail", "surface_Nt": 1e17, '
                          '"bulk_Nt": 1e14, "decay_nm": 120}}}'},
                {"result": "{'architecture': 'p-i-n', 'illumination': 'back/left contact (HTL) side', "
                           "'n_scaps_layers': 6, 'layers': [{'name': 'HTL', 'material': 'PTAA', "
                           "'thickness_nm': 150.0, 'role': 'illuminated surface side', ...}, "
                           "{'name': 'Perovskite', 'material': 'MixedPerovskite', "
                           "'thickness_nm': 500.0, 'sub_layers': 4, "
                           "'defect_profile_1cm3': [{'depth_nm': 62.5, 'Nt_1cm3': 2.707e15}, "
                           "{'depth_nm': 187.5, 'Nt_1cm3': 7.489e15}, "
                           "{'depth_nm': 312.5, 'Nt_1cm3': 2.104e16}, "
                           "{'depth_nm': 437.5, 'Nt_1cm3': 5.944e16}]}, ...]}"},
                {"reply": '{"tool": "write_execution_script", "args": {"job": "pin_sno2", '
                          '"architecture": "p-i-n", "etl": "SnO2", "htl": "PTAA", '
                          '"absorber_thickness_nm": 500, "htl_thickness_nm": 150, '
                          '"etl_thickness_nm": 50, "defect_profile": {"kind": "exp-tail", '
                          '"surface_Nt": 1e17, "bulk_Nt": 1e14, "decay_nm": 120}, '
                          '"sublayers": 4}}'},
                {"result": "{'job': 'pin_sno2', "
                           "'def_path': '.../def/pin_sno2.def', "
                           "'def_layer_order': ['HTL', 'Perovskite_1', 'Perovskite_2', "
                           "'Perovskite_3', 'Perovskite_4', 'ETL'], "
                           "'script_path': '.../script/pin_sno2.script', "
                           "'execution_script': 'set quitscript.quitSCAPS\\n"
                           "set errorhandling.overwritefile\\nload definitionfile pin_sno2.def\\n"
                           "clear actions\\naction workingpoint.temperature 300.0\\n"
                           "action iv.startv 0.0\\naction iv.stopv 1.6\\n"
                           "action iv.increment 0.02\\naction iv.doiv\\naction light\\n"
                           "calculate singleshot\\nsave results.iv pin_sno2.iv\\n', "
                           "'expected_result_file': '.../results/pin_sno2.iv', "
                           "'note': 'Files written. Call simulate_stack to execute this script.'}"},
                {"reply": '{"tool": "simulate_stack", "args": {"job": "pin_sno2", '
                          '"architecture": "p-i-n", "etl": "SnO2", "htl": "PTAA", '
                          '"absorber_thickness_nm": 500, "htl_thickness_nm": 150, '
                          '"etl_thickness_nm": 50, "defect_profile": {"kind": "exp-tail", '
                          '"surface_Nt": 1e17, "bulk_Nt": 1e14, "decay_nm": 120}}}'},
                {"result": "{'converged': True, 'Voc_V': 1.23534, 'Jsc_mA_cm2': 24.1865, "
                           "'FF_percent': 87.7624, 'PCE_percent': 26.2221, "
                           "'engine': 'SCAPS 3.3.02 under wine64', 'elapsed_s': 56.8, "
                           "'def_path': '.../def/pin_sno2.def', "
                           "'script_path': '.../script/pin_sno2.script', 'device': {...}}"},
                {"reply": DONE + (
                    "p-i-n stack: ITO/PTAA 150 nm | MAPbI3 500 nm | SnO2 50 nm | ITO, "
                    "illuminated through the PTAA side. SCAPS-1D gives Voc 1.235 V, "
                    "Jsc 24.19 mA/cm2, FF 87.8 %, PCE 26.2 % (run pin_sno2, converged). "
                    "The absorber defect density falls from 5.9e16 1/cm^3 at the lit PTAA "
                    "interface to 2.7e15 1/cm^3 in the bulk, a 120 nm exponential tail, "
                    "implemented as 4 sub-layers. Dropping that surface spike to a uniform "
                    "1e15 raises PCE to about 30 %, so on this model the graded surface is "
                    "what costs the fill factor. Transport parameters are assumed "
                    "(absorber 1 cm2/Vs both carriers, PTAA hole 1 cm2/Vs), and that "
                    "assumption matters more than the defect profile; this is a simulation, "
                    "not a measurement.")},
            ],
        },
        {
            "task": "What does SnO2 actually buy me?",
            "turns": [
                # A mistake on purpose: the agent guesses a material that is in the
                # wrong role, reads the error, and corrects itself.
                {"reply": '{"tool": "design_stack", "args": {"architecture": "p-i-n", '
                          '"etl": "Spiro-MeOTAD", "htl": "SnO2"}}'},
                {"result": "DesignError: etl 'Spiro-MeOTAD' cannot be used in that role. "
                           "Valid etls: SnO2, TiO2, PCBM, C60"},
                {"reply": '{"tool": "simulate_stack", "args": {"job": "pin_sno2_vs_tio2", '
                          '"architecture": "p-i-n", "etl": "SnO2", "htl": "PTAA"}}}'},
                {"result": "{'converged': True, 'Voc_V': 1.3776, 'Jsc_mA_cm2': 24.1937, "
                           "'FF_percent': 90.6665, 'PCE_percent': 30.2176, ...}"},
                {"reply": '{"tool": "simulate_stack", "args": {"job": "pin_tio2", '
                          '"architecture": "p-i-n", "etl": "TiO2", "htl": "PTAA"}}}'},
                {"result": "{'converged': True, 'Voc_V': 1.3864, 'Jsc_mA_cm2': 23.5915, "
                           "'FF_percent': 90.5714, 'PCE_percent': 29.6239, ...}"},
                {"reply": DONE + (
                    "On this model SnO2 is worth about 0.6 PCE points over TiO2 "
                    "(30.2 % vs 29.6 %), and the difference is in fill factor and Jsc rather "
                    "than Voc. That is smaller than the uncertainty in the assumed transport "
                    "parameters, so it is not a real result - the two ETLs differ mainly in "
                    "electron mobility (SnO2 50 cm2/Vs vs TiO2 10 cm2/Vs) and doping in these "
                    "presets, and those are exactly the numbers nobody has measured the same "
                    "way twice. Both runs are simulations.")},
            ],
        },
    ]
    if not install["available"]:
        tools_avail = {"design_stack", "write_execution_script", "_calc"}
        for ex in examples:
            ex["turns"] = [t for t in ex["turns"]
                           if _tool_named(t.get("reply", "")) in tools_avail]

    return Agent("design",
                 "You design perovskite solar-cell stacks and simulate them with SCAPS-1D, returning "
                 "Voc, Jsc, FF and PCE from the solver together with the device you actually built.",
                 knowledge, tools, llm, examples=examples)


def build_team(llm, only=None):
    """Build the agents whose data exists on this machine. Returns (agents, {name: why not})."""
    makers = {"pdb": pdb_agent, "literature": literature_agent, "ml": ml_agent,
              "scaps": scaps_agent, "design": design_agent}
    team, missing = {}, {}
    for name in only or makers:
        try:
            team[name] = makers[name](llm)
        except Exception as exc:
            missing[name] = f"{type(exc).__name__}: {exc}"
    return team, missing
