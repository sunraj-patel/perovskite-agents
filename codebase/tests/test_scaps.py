"""SCAPS agent: full parameter namespace, .def round-trip, live-tool gating,
and multi-shot prompting.
"""
import os
import sys

import pytest

from perovagents.agent import DONE, format_examples
from perovagents.physics import scaps as S
from perovagents.team import scaps_agent

MAKE_DEF_DIR = os.path.join(os.path.dirname(os.path.dirname(S.__file__)),
                            "..", "..", "opencode_sandbox", "scaps")
MAKE_DEF_DIR = os.path.abspath(MAKE_DEF_DIR)


# ---- the .def builder ------------------------------------------------------
@pytest.fixture(scope="module")
def mk():
    if not os.path.exists(os.path.join(MAKE_DEF_DIR, "make_perovskite_def.py")):
        pytest.skip("make_perovskite_def.py not on this machine")
    sys.path.insert(0, MAKE_DEF_DIR)
    import make_perovskite_def
    return make_perovskite_def


def _nt_of(def_text, block):
    import re
    body = re.split(r"name : " + block + r"\n", def_text)[1].split("\nlayer")[0]
    return float(re.search(r"Nt\(y\) :\s+(\S+)", body).group(1))


def test_defect_density_is_settable_for_every_layer(mk):
    """Nt used to be hardcoded to 1e20, so the agent could not answer any
    question about defect density at all."""
    d = mk.build_def(pvk_Nt=1e15, htl_Nt=1e17, etl_Nt=1e19)
    assert _nt_of(d, "Perovskite") == pytest.approx(1e15)
    assert _nt_of(d, "HTL") == pytest.approx(1e17)
    assert _nt_of(d, "ETL") == pytest.approx(1e19)


def test_set_is_si_and_set_eng_takes_engineering_units(mk):
    """The density ranges overlap, so guessing the unit would corrupt values."""
    assert mk.parse_set(["pvk_Nt=1e21"]) == {"pvk_Nt": 1e21}
    assert mk.parse_set(["pvk_Nt=1e15"], engineering=True) == {"pvk_Nt": 1e15 * 1e6}
    assert mk.parse_set(["pvk_d=500"], engineering=True)["pvk_d"] == pytest.approx(500e-9)
    assert mk.parse_set(["pvk_mun=1.0"], engineering=True)["pvk_mun"] == pytest.approx(1e-4)


@pytest.mark.parametrize("bad,msg", [
    ("zzz_Nt=1", "unknown layer"),
    ("pvk_Zz=1", "unknown field"),
    ("nounderscore=1", "<layer>_<field>"),
    ("pvk_Nt", "layer.field=value"),
])
def test_bad_set_names_raise_a_readable_error(mk, bad, msg):
    with pytest.raises(ValueError) as e:
        mk.parse_set([bad])
    assert msg in str(e.value)


# ---- the public parameter namespace ---------------------------------------
def test_defect_density_is_in_the_public_registry():
    for name in ("pvk_Nt", "htl_Nt", "etl_Nt"):
        assert name in S.PARAM_SPECS
        assert S.PARAM_SPECS[name]["unit"] == "1/cm^3"
    assert S.PARAM_SPECS["pvk_Nt"]["scale"] == 1e6


@pytest.mark.parametrize("public,si,field", [
    ("pvk_d", 500e-9, "d"),           # nm -> m
    ("pvk_mun", 1e-4, "mun"),         # cm2/Vs -> m2/Vs
    ("pvk_Nt", 1e21, "Nt"),           # 1/cm3 -> 1/m3
    ("htl_Na", 1e24, "Na"),
    ("etl_d", 5e-8, "d"),
])
def test_public_units_convert_to_the_si_the_def_expects(public, si, field):
    assert S.to_si(public, 1 if field == "d" else 1) is not None
    name, value = S.to_si(public, si / S.PARAM_SPECS[public]["scale"])
    assert name == public and value == pytest.approx(si)


def test_unknown_parameter_raises_with_valid_names():
    with pytest.raises(KeyError) as e:
        S.resolve_param("htl_thickness_nm")
    assert "unknown parameter" in str(e.value)
    # the error must list copyable full names for every layer, or the agent
    # cannot find the parameter it actually wanted
    for lay in ("pvk_", "htl_", "etl_"):
        assert lay in str(e.value)
    assert "pvk_Nt" in str(e.value) and "htl_Nt" in str(e.value)


def test_aliases_resolve_to_canonical_parameters():
    assert S.to_si("defect_density", 1e15) == S.to_si("pvk_Nt", 1e15)
    assert S.to_si("thickness", 500) == S.to_si("pvk_d", 500)


def test_reference_cell_matches_the_builder(mk):
    """The agent quotes these numbers, so they must match the real defaults."""
    import re
    d = mk.build_def()
    blocks = dict(re.findall(r"name : (\w+)\n(.*?)(?=\nlayer|\nfront contact)", d, re.S))
    for public, (value, unit) in S.REFERENCE_CELL.items():
        spec = S.PARAM_SPECS[public]
        si = value * spec["scale"]
        assert si > 0
        assert public in S.PARAM_SPECS


# ---- live-tool gating ------------------------------------------------------
def test_scaps_agent_hides_live_tools_and_says_why(monkeypatch):
    real = S.SCAPSTool

    class DeadWine(real):
        def live_status(self):
            return {"available": False, "error": "scaps3309.exe did not start",
                    "wine": "/opt/homebrew/bin/wine"}

    monkeypatch.setattr(S, "SCAPSTool", DeadWine)
    a = scaps_agent(None)
    assert "run_point" not in a.tools and "sweep" not in a.tools
    assert "get_sweep" in a.tools and "read_def" in a.tools
    assert "LIVE SIMULATION IS OFF" in a.system
    assert "scaps3309.exe did not start" in a.system
    # no worked example may call a tool that is not on the list
    for name in ("run_point", "sweep"):
        assert f'"tool": "{name}"' not in a.system


def test_scaps_agent_offers_live_tools_when_available(monkeypatch):
    real = S.SCAPSTool

    class LiveWine(real):
        def live_status(self):
            return {"available": True, "error": None, "wine": "x"}

    monkeypatch.setattr(S, "SCAPSTool", LiveWine)
    a = scaps_agent(None)
    assert "run_point" in a.tools and "sweep" in a.tools
    assert "LIVE SIMULATION IS OFF" not in a.system
    assert '"tool": "sweep"' in a.system


# ---- multi-shot prompting --------------------------------------------------
def test_examples_are_rendered_into_the_system_prompt():
    out = format_examples([{"task": "t", "turns": [
        {"reply": '{"tool": "get_sweep", "args": {"sweep": "eg"}}', "result": "{'found': True}"},
        {"reply": f"{DONE} done"}]}])
    assert "WORKED EXAMPLES" in out
    assert "--- Example 1: t" in out
    assert '{"tool": "get_sweep"' in out
    assert "{'found': True}" in out
    assert DONE in out


def test_long_tool_results_are_truncated_in_examples():
    out = format_examples([{"task": "t", "turns": [
        {"reply": '{"tool": "x", "args": {}}', "result": "y" * 5000}]}])
    assert "[truncated" in out or "..." in out
    assert len(out) < 2000


def test_agent_prompt_documents_defect_density_and_names_it():
    a = scaps_agent(None)
    for piece in ("pvk_Nt", "htl_Nt", "etl_Nt", "1/cm^3", "SRH", "read_def",
                  "WORKED EXAMPLES"):
        assert piece in a.system, piece


def test_scaps_env_puts_wswine_frameworks_on_dylib_path(tmp_path, monkeypatch):
    """Direct bin/wine64 calls need the app's Frameworks dir (FreeType), else
    scaps3309.exe spins at 100% CPU and never writes an IV file."""
    from perovagents import config
    frameworks = tmp_path / "App.app" / "Contents" / "Frameworks"
    bundle = frameworks / "wswine.bundle"
    (bundle / "bin").mkdir(parents=True)
    (bundle / "lib").mkdir()
    wine = bundle / "bin" / "wine64"
    wine.write_text("")
    monkeypatch.delenv("PEROVAGENTS_SCAPS_DYLD", raising=False)
    monkeypatch.setattr(config, "SCAPS_WINE", str(wine))
    monkeypatch.setattr(config, "SCAPS_PREFIX", "/p")
    env = config.scaps_env({"DYLD_FALLBACK_LIBRARY_PATH": "/usr/lib"})
    parts = env["DYLD_FALLBACK_LIBRARY_PATH"].split(":")
    assert env["WINEPREFIX"] == "/p"
    assert parts[:2] == [os.path.realpath(str(frameworks)), os.path.realpath(str(bundle / "lib"))]
    assert parts[-1] == "/usr/lib"
    monkeypatch.setenv("PEROVAGENTS_SCAPS_DYLD", "/x:/y")
    assert config.scaps_env({})["DYLD_FALLBACK_LIBRARY_PATH"] == "/x:/y"
    monkeypatch.delenv("PEROVAGENTS_SCAPS_DYLD")
    monkeypatch.setattr(config, "SCAPS_WINE", "/usr/bin/wine64")   # not a wswine bundle
    assert "DYLD_FALLBACK_LIBRARY_PATH" not in config.scaps_env({})
