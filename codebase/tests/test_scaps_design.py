"""SCAPS design pipeline: .def correctness, defect profiles, and the standard
execution runner.

These tests are about the contract between the agent and SCAPS-1D:

  * the generated `.def` must be numerically identical to a SCAPS-authored
    device file (checked field by field, not by text diff, because whitespace
    is irrelevant to SCAPS);
  * architecture must change the illumination side, and nothing else;
  * a graded defect profile must reach SCAPS as real sub-layers;
  * the runner must not report success for a sweep that never produced Voc.
"""
import math
import os
import re

import pytest

from perovagents.physics import scaps_def as D
from perovagents.physics import scaps_run as R

SCAPS_DEF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "..", "opencode_sandbox", "scaps")
SCAPS_DEF_DIR = os.path.abspath(SCAPS_DEF_DIR)


# ---- helpers ---------------------------------------------------------------
_ROW_FIELDS = ("Eg", "chi", "eps", "Nc", "Nv", "mu_n", "mu_p", "Na", "Nd",
               "Nt", "Et", "sigma_n", "sigma_p")
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_def(text):
    """A SCAPS .def -> (layers, contacts). Only the numbers SCAPS reads."""
    text = text.replace("\r", "")
    layers, contacts, cur, block = {}, {}, None, None
    for raw in text.splitlines():
        line = raw.replace("\t", " ").strip()
        if line in ("layer", "front contact", "back contact", "working point",
                    "convergence"):
            block = line
            continue
        m = re.match(r"^name : (\S+)$", line)
        if m and block == "layer":
            cur = m.group(1)
            layers[cur] = {}
            continue
        m = re.match(r"^d : (\S+) ", line)
        if m and cur:
            layers[cur]["d"] = [float(m.group(1))]
            continue
        m = re.match(r"^(Sn|Sp) : (\S+)", line)
        if m and block in ("front contact", "back contact"):
            contacts.setdefault(block, {})[m.group(1)] = float(m.group(2))
            continue
        m = re.match(r"^(illuminated from) : (\w+)$", line)
        if m and block == "working point":
            contacts.setdefault("working", {})[m.group(1)] = m.group(2)
            continue
        for f in _ROW_FIELDS:
            # Na, Nd and Nt are written as `Na(y) :`, `Nt(y) :`, ...
            if re.match(rf"^{f}(\(y\))? :", line):
                vals = re.findall(_NUM, line)
                if len(vals) >= 7 and cur:
                    layers[cur][f] = [float(v) for v in vals[:7]]
    return layers, contacts


def row(text, layer, field):
    body = re.split(r"name : " + layer + r"\n", text)[1].split("\nlayer")[0]
    line = next(t for t in body.splitlines()
                if re.match(rf"^{field}(\(y\))? :", t.replace("\t", " ").strip()))
    return [float(v) for v in re.findall(_NUM, line)[:7]]


def _reference_device():
    """The .def written by the pre-existing builder, used as ground truth."""
    path = os.path.join(SCAPS_DEF_DIR, "make_perovskite_def.py")
    if not os.path.exists(path):
        pytest.skip("make_perovskite_def.py not on this machine")
    import sys
    sys.path.insert(0, SCAPS_DEF_DIR)
    import make_perovskite_def
    return make_perovskite_def.build_def()


def _reference_equivalent_stack():
    return D.quick_stack(
        "n-i-p", etl="TiO2", htl="Spiro-MeOTAD", absorber="MixedPerovskite",
        thickness_nm=400, etl_thickness_nm=50, htl_thickness_nm=200,
        absorber_params={"Eg": 1.55, "chi": 3.93, "eps": 10.0,
                         "Nc": 2.2e18, "Nv": 2.2e18, "mun": 1.0, "mup": 1.0,
                         "Nd": 2e16},
        defect_profile=1e14)


# ---- the generated .def must equal a SCAPS-authored device ----------------
def test_generated_def_matches_reference_device_numerically():
    """Text diffs are misleading (tabs/blank lines); compare the numbers."""
    ref_layers, ref_contacts = parse_def(_reference_device())
    mine_layers, mine_contacts = parse_def(D.build_def_text(_reference_equivalent_stack()))

    assert list(ref_layers) == list(mine_layers)
    for name in ref_layers:
        for f in ("d",) + _ROW_FIELDS:
            a, b = ref_layers[name].get(f), mine_layers[name].get(f)
            assert (a is None) == (b is None), f"{name}.{f} present in only one file"
            if a is None:
                continue
            for k, (x, y) in enumerate(zip(a, b)):
                assert x == pytest.approx(y, rel=1e-9), f"{name}.{f}[{k}] {x} != {y}"
    assert mine_contacts == ref_contacts


def test_every_scaps_required_block_is_present_exactly_once():
    text = D.build_def_text(_reference_equivalent_stack())
    assert text.count("\nback contact") == 1
    assert text.count("\nfront contact") == 1
    assert text.count("\nworking point") == 1
    # one interface between each pair of layers
    assert text.count("interface properties") == 2
    for block in ("srhrecombination", "illuminated from", "cell reference system",
                  "absorptionmodel pure A material", "composition grading"):
        assert block in text


# ---- architecture ----------------------------------------------------------
def test_architecture_flips_only_the_illumination_side():
    nip = D.build_def_text(D.quick_stack("n-i-p", etl="TiO2", htl="Spiro-MeOTAD"))
    pin = D.build_def_text(D.quick_stack("p-i-n", etl="TiO2", htl="Spiro-MeOTAD"))
    assert "illuminated from : right" in nip
    assert "illuminated from : left" in pin
    # SCAPS always writes the p-type layer first, so the layer order must match
    # SCAPS' own a-SiH pin.def convention: HTL, absorber, ETL.
    for text in (nip, pin):
        names = re.findall(r"^name : (\S+)", text, re.M)
        assert names == ["HTL", "Perovskite", "ETL"]
    # ... and the contacts must be identical: holes left, electrons right.
    # (Only the working point's illumination side may differ.)
    nip_c = {k: v for k, v in parse_def(nip)[1].items() if k != "working"}
    pin_c = {k: v for k, v in parse_def(pin)[1].items() if k != "working"}
    assert nip_c == pin_c
    assert parse_def(nip)[1]["working"] == {"illuminated from": "right"}
    assert parse_def(pin)[1]["working"] == {"illuminated from": "left"}


def test_layers_must_be_htl_then_absorber_then_etl():
    bad = D.Stack(architecture="n-i-p", layers=[
        D.LayerSpec("ETL", "TiO2", 50), D.LayerSpec("X", "MAPbI3", 400),
        D.LayerSpec("HTL", "Spiro-MeOTAD", 200)], absorber_index=1)
    with pytest.raises(ValueError, match="hole transport layer"):
        D.build_def_text(bad)
    bad2 = D.Stack(architecture="p-i-n", layers=[
        D.LayerSpec("HTL", "Spiro-MeOTAD", 200), D.LayerSpec("X", "MAPbI3", 400),
        D.LayerSpec("HTL2", "PTAA", 50)], absorber_index=1)
    with pytest.raises(ValueError, match="electron transport layer"):
        D.build_def_text(bad2)


def test_unknown_architecture_is_rejected_with_valid_names():
    with pytest.raises(ValueError, match="architecture must be one of"):
        D.quick_stack("p-i-n-n", etl="SnO2", htl="PTAA")


# ---- unit handling ---------------------------------------------------------
@pytest.mark.parametrize("field,public,si", [
    ("d", 500, 5e-7), ("mun", 10, 1e-3), ("Nt", 1e15, 1e21),
    ("Na", 1e18, 1e24), ("Et", 0.25, 0.25), ("eps", 26, 26),
])
def test_public_units_convert_to_what_the_def_writes(field, public, si):
    assert D.to_si(field, public, "public") == pytest.approx(si, rel=1e-12)
    assert D.to_si(field, si, "si") == pytest.approx(si, rel=1e-12)


def test_thickness_and_defect_density_reach_the_file_in_the_right_units():
    stack = D.quick_stack("p-i-n", etl="SnO2", htl="PTAA", absorber="MixedPerovskite",
                          thickness_nm=500, etl_thickness_nm=50, htl_thickness_nm=150)
    text = D.build_def_text(stack)
    layers, _ = parse_def(text)
    assert layers["Perovskite"]["d"][0] == pytest.approx(5e-7, rel=1e-9)
    assert layers["Perovskite"]["Eg"][0] == pytest.approx(1.55)
    assert layers["ETL"]["d"][0] == pytest.approx(5e-8, rel=1e-9)
    assert layers["ETL"]["Eg"][0] == pytest.approx(D.MATERIALS["SnO2"]["Eg"])


def test_bad_field_name_is_rejected_readably():
    with pytest.raises(KeyError, match="unknown layer field"):
        D.to_si("bandgap", 1.5)
    lay = D.LayerSpec("X", "MAPbI3", 400, {"bandgap": 1.5})
    with pytest.raises(KeyError, match="unknown field"):
        lay.si_params()


# ---- defect profiles -------------------------------------------------------
def test_uniform_defect_density_is_one_layer():
    stack = D.quick_stack("n-i-p", etl="TiO2", htl="Spiro-MeOTAD", defect_profile=1e15)
    assert len(D.expand_stack(stack)) == 3, "a uniform profile must not sub-layer"
    assert row(D.build_def_text(stack), "Perovskite", "Nt")[0] == pytest.approx(1e21, rel=1e-9)


def test_exp_tail_profile_decays_from_the_illuminated_surface():
    spec = {"kind": "exp-tail", "surface_Nt": 1e17, "bulk_Nt": 1e14, "decay_nm": 100}
    import math
    nts = D.profile_densities(spec, 500, 8)
    assert nts[0] > nts[-1]
    assert nts[0] == pytest.approx(1e14 + (1e17 - 1e14) * math.exp(-(500 / 16) / 100), rel=1e-9)
    assert nts[-1] == pytest.approx(1e14 + (1e17 - 1e14) * math.exp(-(500 * 7.5 / 8) / 100), rel=1e-9)
    # 8 samples over 500 nm put the deepest one at 468.75 nm, i.e. 4.7 decay
    # lengths, so it is already within an order of magnitude of the bulk floor.
    assert nts[-1] < 1e14 * 11
    assert all(a > b for a, b in zip(nts, nts[1:])), "a tail profile must decay"


def test_linear_profile_interpolates_endpoints():
    nts = D.profile_densities({"kind": "linear", "front_Nt": 1e14, "back_Nt": 1e17}, 400, 2)
    assert nts[0] < nts[-1]
    # n=2 samples the quarter and three-quarter depths of a 400 nm layer
    assert nts[0] == pytest.approx(1e14 + 0.25 * (1e17 - 1e14), rel=1e-9)
    assert nts[-1] == pytest.approx(1e14 + 0.75 * (1e17 - 1e14), rel=1e-9)


def test_step_and_explicit_profiles_respect_their_levels():
    steps = D.profile_densities({"kind": "step", "levels": [[0.5, 1e17], [1.0, 1e15]]}, 400, 4)
    assert steps[0] == pytest.approx(1e17) and steps[1] == pytest.approx(1e17)
    assert steps[2] == pytest.approx(1e15) and steps[-1] == pytest.approx(1e15)
    explicit = D.profile_densities([[0.0, 1e17], [1.0, 1e14]], 400, 4)
    assert explicit[0] > explicit[-1], "explicit points must interpolate downward"
    assert explicit[0] == pytest.approx(1e17 - 0.125 * (1e17 - 1e14), rel=1e-9)
    assert explicit[-1] == pytest.approx(1e17 - 0.875 * (1e17 - 1e14), rel=1e-9)


def test_unknown_profile_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown defect profile kind"):
        D.profile_densities({"kind": "magic"}, 400, 4)


def test_graded_profile_becomes_real_sublayers_in_the_file():
    stack = D.quick_stack("n-i-p", etl="TiO2", htl="Spiro-MeOTAD", thickness_nm=400,
                          defect_profile={"kind": "exp-tail", "surface_Nt": 1e17,
                                          "bulk_Nt": 1e14, "decay_nm": 100},
                          sublayers=4)
    text = D.build_def_text(stack)
    names = re.findall(r"^name : (\S+)", text, re.M)
    assert names == ["HTL", "Perovskite_1", "Perovskite_2", "Perovskite_3",
                     "Perovskite_4", "ETL"]
    assert text.count("interface properties") == 5, "one interface per layer boundary"
    layers, _ = parse_def(text)
    # sub-layers must tile the absorber exactly
    total = sum(layers[f"Perovskite_{i}"]["d"][0] for i in range(1, 5))
    assert total == pytest.approx(4e-7, rel=1e-9)
    # n-i-p is lit from the right, so the high-defect end is the LAST block.
    densities = [layers[f"Perovskite_{i}"]["Nt"][0] for i in range(1, 5)]
    assert densities[-1] > densities[0], "n-i-p: surface defects sit on the ETL side"
    assert all(a > b for a, b in zip(reversed(densities), list(reversed(densities))[1:]))
    # the block nearest the lit (right/ETL) surface is the deepest-written one,
    # whose midpoint is 400 / (2 * 4) = 50 nm below the surface
    assert densities[-1] / 1e6 == pytest.approx(
        1e14 + (1e17 - 1e14) * math.exp(-50 / 100), rel=1e-6)


def test_graded_profile_puts_surface_defects_next_to_the_light_in_pin():
    stack = D.quick_stack("p-i-n", etl="SnO2", htl="PTAA", thickness_nm=500,
                          defect_profile={"kind": "exp-tail", "surface_Nt": 1e17,
                                          "bulk_Nt": 1e14, "decay_nm": 120},
                          sublayers=4)
    layers, _ = parse_def(D.build_def_text(stack))
    densities = [layers[f"Perovskite_{i}"]["Nt"][0] for i in range(1, 5)]
    # p-i-n is lit from the left, so the FIRST block written is the one next to
    # the light and must carry the high defect density.
    assert densities[0] > densities[-1]
    assert densities[0] / 1e6 == pytest.approx(
        1e14 + (1e17 - 1e14) * math.exp(-(500 / 8) / 120), rel=1e-6)
    summary = D.stack_summary(stack)
    assert summary["illumination"].startswith("back/left")
    pts = summary["layers"][1]["defect_profile_1cm3"]
    assert pts[0]["Nt_1cm3"] > pts[-1]["Nt_1cm3"], "reported depths run from the light"
    assert pts[0]["Nt_1cm3"] == pytest.approx(
        1e14 + (1e17 - 1e14) * math.exp(-(500 / 8) / 120), rel=1e-6)
    assert pts[0]["depth_nm"] < pts[-1]["depth_nm"]


def test_stack_summary_reports_the_device_the_agent_will_claim():
    stack = D.quick_stack("p-i-n", etl="SnO2", htl="PTAA", absorber="MixedPerovskite",
                          thickness_nm=500, etl_thickness_nm=50, htl_thickness_nm=150)
    s = D.stack_summary(stack)
    assert s["architecture"] == "p-i-n"
    # SCAPS file order is always HTL, absorber, ETL; p-i-n differs only in the
    # illumination side, which is what the summary must report.
    assert [x["material"] for x in s["layers"]] == ["PTAA", "MixedPerovskite", "SnO2"]
    assert [x["thickness_nm"] for x in s["layers"]] == [150, 500, 50]
    for x in s["layers"]:
        assert x["defect_density_1e14cm3"] is not None
        assert len(x["defect_profile_1cm3"]) == 2


# ---- the execution runner --------------------------------------------------
def test_installation_probe_reports_what_is_needed():
    status = R.check_installation()
    assert set(status) >= {"available", "wine64", "scaps_executable", "error"}
    if not status["available"]:
        assert status["error"] and status["missing"]


def test_execution_script_is_valid_plain_scaps_action_list():
    script = R.write_execution_script("job.def", job="t_job", stopv=1.5, step=0.02)
    text = script["script_text"]
    for line in ("set quitscript.quitSCAPS", "load definitionfile job.def",
                 "clear actions", "action workingpoint.temperature 300",
                 "action iv.stopv 1.5", "action iv.increment 0.02",
                 "action iv.doiv", "action light", "calculate singleshot",
                 "save results.iv t_job.iv"):
        assert line in text
    assert os.path.basename(script["script_path"]) == "t_job.script"
    assert script["iv_name"] == "t_job.iv"


def test_parse_rejects_a_sweep_that_never_reached_voc():
    """SCAPS prints the 'parameters deduced' banner even for a truncated sweep."""
    partial = ("   0.300000\t   2.41934875e+01\t   4.16529498e-01\n"
               "   0.760000\t   2.41934875e+01\t   4.15252150e-01\n"
               "solar cell parameters deduced from calculated IV-curve:\n"
               "Jsc =\t   24.19371769\tmA/cm2\n")
    parsed = _parse_text(partial)
    assert parsed["sweep_completed"] is True
    assert parsed["converged"] is False
    assert parsed["Voc_V"] is None
    assert parsed["V_max_reached"] == pytest.approx(0.76)
    assert "aborted at V=0.76" in parsed["incomplete_reason"]


def test_parse_accepts_a_complete_iv_curve():
    good = ("   1.000000\t  -2.36887700e+01\t   1.83848911e-03\n"
            "solar cell parameters deduced from calculated IV-curve:\n"
            "Voc =\t   1.38652800\tVolt\n"
            "Jsc =\t   23.68894771\tmA/cm2\n"
            "FF =\t   90.57720000\t%\n"
            "eta =\t   29.75040000\t%\n")
    p = _parse_text(good)
    assert p["converged"] is True
    assert p["Voc_V"] == pytest.approx(1.386528)
    assert p["Jsc_mA_cm2"] == pytest.approx(23.68894771)
    assert p["FF_percent"] == pytest.approx(90.5772)
    assert p["PCE_percent"] == pytest.approx(29.7504)


def _parse_text(text):
    """parse_iv_file without touching disk."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".iv", delete=False) as fh:
        fh.write(text)
        path = fh.name
    try:
        return R.parse_iv_file(path)
    finally:
        os.unlink(path)


@pytest.mark.skipif(not R.check_installation()["available"],
                    reason="SCAPS 1D is not runnable on this machine")
@pytest.mark.slow
def test_agent_written_def_runs_and_returns_numbers_to_the_agent():
    """The end-to-end contract: agent authors the files, SCAPS answers."""
    stack = D.quick_stack("p-i-n", etl="SnO2", htl="PTAA",
                          absorber="MixedPerovskite", thickness_nm=500,
                          etl_thickness_nm=50, htl_thickness_nm=150)
    result = R.simulate(D.build_def_text(stack), job="pytest_pin_sno2")
    assert result["converged"]
    assert 0.5 < result["Voc_V"] < 2.0
    assert 1 < result["Jsc_mA_cm2"] < 40
    assert 0 < result["FF_percent"] <= 100
    assert 0 < result["PCE_percent"] < 45
    assert result["engine"].startswith("SCAPS")
    assert os.path.exists(result["def_path"])
    assert os.path.exists(result["script_path"])
    assert "load definitionfile" in result["execution_script"]


# ---- agent wiring ----------------------------------------------------------
def test_design_agent_exposes_the_full_design_toolchain():
    """The agent can lay out a stack, author the files, and run SCAPS."""
    from perovagents.team import build_team
    team, missing = build_team(None, only=["design"])
    assert "design" not in missing
    a = team["design"]
    for name in ("design_stack", "write_execution_script", "simulate_stack",
                 "simulate_defect_profile"):
        assert name in a.tools, f"{name} missing from the design agent"
    assert "not just" in a.system and "SIMULATE" in a.system
    # the geometry rule that is wrong most often has to be stated explicitly
    assert "HTL | absorber | ETL" in a.system
    assert "illuminated from the right" in a.system
    assert "illuminated from the left" in a.system


def test_design_agent_drops_live_tools_and_says_why_when_scaps_is_gone(monkeypatch):
    """Offline the agent must not get a tool that cannot run, and must be told why."""
    import perovagents.team as T
    from perovagents.physics import design as G

    monkeypatch.setattr(G.R, "check_installation", lambda: {
        "available": False, "error": "scaps3309.exe did not start",
        "wine": "/opt/homebrew/bin/wine", "wine_version": None,
        "scaps_dir": "", "scaps_executable": "", "missing": ["scaps3309.exe"]})
    monkeypatch.setattr(T, "design_agent", T.design_agent)
    a = T.design_agent(None)
    assert "simulate_stack" not in a.tools
    assert "simulate_defect_profile" not in a.tools
    assert "design_stack" in a.tools and "write_execution_script" in a.tools
    assert "UNAVAILABLE" in a.system
    assert "scaps3309.exe did not start" in a.system
    assert "no simulation was run" in a.system


def test_sublayers_reaches_the_sub_layers_actually_written():
    """`sublayers` must change the generated stack, not be silently ignored."""
    from perovagents.physics import design as G
    spec = {"kind": "exp-tail", "surface_Nt": 1e17, "bulk_Nt": 1e14, "decay_nm": 120}
    counts = {}
    for n in (2, 6):
        out = G.design_stack(architecture="p-i-n", etl="SnO2", htl="PTAA",
                             absorber_thickness_nm=500, defect_profile=spec,
                             sublayers=n)
        counts[n] = (out["layers"][1]["sub_layers"], out["n_scaps_layers"])
        assert len(out["layers"][1]["defect_profile_1cm3"]) == n
    assert counts[2] == (2, 4) and counts[6] == (6, 8)
