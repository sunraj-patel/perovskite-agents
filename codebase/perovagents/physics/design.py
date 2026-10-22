"""Design agent: proposes a device stack, writes the SCAPS files, simulates.

The division of labour is deliberate and is the reason this agent is
trustworthy:

  1. `design_stack` turns the request into a concrete, fully-specified stack and
     reports every value in engineering units, so the agent can state exactly
     what it intends to simulate before anything runs;
  2. `write_execution_script` emits the SCAPS action list -- the file SCAPS
     itself reads, inspectable and re-runnable by hand;
  3. `simulate` hands that script to the standard runner and returns Voc, Jsc,
     FF, PCE plus the provenance of the run.

Nothing here invents a result: every number in an answer has to come out of
one of these three tools.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from . import scaps_def as D
from . import scaps_run as R


class DesignError(RuntimeError):
    """The requested design cannot be built as specified."""


def _material_catalog_markdown() -> str:
    cat = D.materials_catalogue()
    lines = []
    for role in ("absorber", "htl", "etl"):
        names = sorted(n for n, v in cat.items() if v["role"] == role)
        if not names:
            continue
        lines.append(f"{role.upper()} options: {', '.join(names)}")
    return "\n".join(lines)


def _build(architecture, absorber, htl, etl, absorber_thickness_nm,
           htl_thickness_nm, etl_thickness_nm, absorber_overrides,
           htl_overrides, etl_overrides, defect_profile, sublayers=4):
    """Assemble a validated Stack from the flat argument list the tools expose.

    `sublayers` only matters for a graded `defect_profile`: it is how many uniform
    blocks the absorber is cut into. Each extra block costs solve time.
    """
    for label, name, pool in (("absorber", absorber, D.ABSORBER_MATERIALS),
                              ("htl", htl, D.HTL_MATERIALS),
                              ("etl", etl, D.ETL_MATERIALS)):
        if name not in pool:
            raise DesignError(f"{label} {name!r} cannot be used in that role. "
                              f"Valid {label}s: {', '.join(pool)}")
    stack = D.quick_stack(
        architecture=architecture, absorber=absorber, htl=htl, etl=etl,
        thickness_nm=absorber_thickness_nm, htl_thickness_nm=htl_thickness_nm,
        etl_thickness_nm=etl_thickness_nm, absorber_params=absorber_overrides,
        defect_profile=defect_profile, sublayers=sublayers)
    if htl_overrides:
        stack.layers[0].params.update(htl_overrides)
    if etl_overrides:
        stack.layers[-1].params.update(etl_overrides)
    stack.validate()
    return stack


def design_stack(architecture: str = "n-i-p", absorber: str = "MixedPerovskite",
                 htl: str = "Spiro-MeOTAD", etl: str = "TiO2",
                 absorber_thickness_nm: float = 500.0, htl_thickness_nm: float = 200.0,
                 etl_thickness_nm: float = 50.0,
                 absorber_overrides: Optional[Dict[str, float]] = None,
                 htl_overrides: Optional[Dict[str, float]] = None,
                 etl_overrides: Optional[Dict[str, float]] = None,
                 defect_profile: Any = None, sublayers: int = 4) -> Dict[str, Any]:
    """Lay out a complete device and report every value in engineering units.

    Nothing is simulated here; this is the step where you state what the device
    IS, so that a wrong assumption is visible before it costs a simulation.
    Pass `defect_profile` to give the absorber a spatial SRH profile:
      * a number                -> uniform, e.g. 1e15 (1/cm^3)
      * {"kind": "exp-tail", "surface_Nt": 1e17, "bulk_Nt": 1e14, "decay_nm": 120}
      * {"kind": "linear", "front_Nt": 1e14, "back_Nt": 1e17}
      * {"kind": "step", "levels": [[0.5, 1e17], [1.0, 1e15]]}
      * [[0.0, 1e17], [1.0, 1e14]]   explicit points, depth from the lit surface
    """
    stack = _build(architecture, absorber, htl, etl, absorber_thickness_nm,
                   htl_thickness_nm, etl_thickness_nm, absorber_overrides,
                   htl_overrides, etl_overrides, defect_profile, sublayers)
    out = D.stack_summary(stack)
    out["contact_order"] = ("back/left contact is the hole side, front/right is the "
                            "electron side, for both architectures")
    out["note"] = ("Not yet simulated. Call simulate_stack with this layout to get "
                   "Voc/Jsc/FF/PCE from SCAPS-1D.")
    return out


def write_execution_script(architecture: str = "n-i-p", job: str = "design",
                           absorber: str = "MixedPerovskite",
                           htl: str = "Spiro-MeOTAD", etl: str = "TiO2",
                           absorber_thickness_nm: float = 500.0,
                           htl_thickness_nm: float = 200.0,
                           etl_thickness_nm: float = 50.0,
                           absorber_overrides: Optional[Dict[str, float]] = None,
                           htl_overrides: Optional[Dict[str, float]] = None,
                           etl_overrides: Optional[Dict[str, float]] = None,
                           defect_profile: Any = None, sublayers: int = 4,
                           temperature_k: float = 300.0, stopv: float = 1.6,
                           step: float = 0.02) -> Dict[str, Any]:
    """Author the SCAPS .def and action list, without running the simulator.

    Use this when you want to show or check the exact files SCAPS will read.
    `simulate_stack` calls the same code path, so what you inspect here is
    literally what gets executed.
    """
    stack = _build(architecture, absorber, htl, etl, absorber_thickness_nm,
                   htl_thickness_nm, etl_thickness_nm, absorber_overrides,
                   htl_overrides, etl_overrides, defect_profile, sublayers)
    def_text = D.build_def_text(stack)
    def_path = R.install_device_file(def_text, f"{job}.def")
    script = R.write_execution_script(os.path.basename(def_path), job=job,
                                      temperature=temperature_k, stopv=stopv,
                                      step=step)
    return {
        "job": job,
        "def_path": def_path,
        "def_layer_order": [n for n, _ in D.expand_stack(stack)],
        "device": D.stack_summary(stack),
        "script_path": script["script_path"],
        "execution_script": script["script_text"],
        "expected_result_file": script["iv_path"],
        "note": "Files written. Call simulate_stack to execute this script.",
    }


def simulate_stack(job: str = "design", architecture: str = "n-i-p",
                   absorber: str = "MixedPerovskite", htl: str = "Spiro-MeOTAD",
                   etl: str = "TiO2", absorber_thickness_nm: float = 500.0,
                   htl_thickness_nm: float = 200.0, etl_thickness_nm: float = 50.0,
                   absorber_overrides: Optional[Dict[str, float]] = None,
                   htl_overrides: Optional[Dict[str, float]] = None,
                   etl_overrides: Optional[Dict[str, float]] = None,
                   defect_profile: Any = None, sublayers: int = 4,
                   temperature_k: float = 300.0, stopv: float = 1.6, step: float = 0.02,
                   timeout_s: Optional[int] = None) -> Dict[str, Any]:
    """Run SCAPS-1D on a designed stack and return Voc, Jsc, FF and PCE.

    Takes roughly a minute. `defect_profile` accepts the same forms as in
    design_stack; a graded profile is realised as uniform-Nt sub-layers, which
    costs extra solve time, so keep the sub-layer count low (4 is the default).
    """
    status = R.check_installation()
    if not status["available"]:
        raise DesignError(
            f"SCAPS-1D cannot run on this machine: {status['error']} You can still lay "
            f"out the device with design_stack and show the exact files with "
            f"write_execution_script.")
    stack = _build(architecture, absorber, htl, etl, absorber_thickness_nm,
                   htl_thickness_nm, etl_thickness_nm, absorber_overrides,
                   htl_overrides, etl_overrides, defect_profile, sublayers)
    result = R.simulate(D.build_def_text(stack), job=job, temperature=temperature_k,
                        stopv=stopv, step=step, timeout=timeout_s)
    result["device"] = D.stack_summary(stack)
    result["caveat"] = ("SCAPS-1D drift-diffusion simulation with literature-typical "
                        "material parameters; it is a physics model, not a measurement, "
                        "and the transport parameters dominate the result.")
    return result


def simulate_defect_profile(job: str = "profile", architecture: str = "p-i-n",
                            absorber: str = "MixedPerovskite", htl: str = "PTAA",
                            etl: str = "SnO2", absorber_thickness_nm: float = 500.0,
                            surface_Nt: float = 1e17, bulk_Nt: float = 1e14,
                            decay_nm: float = 120, sublayers: int = 4,
                            uniform_Nt: float = 1e15) -> Dict[str, Any]:
    """Compare a graded absorber defect profile against a uniform one.

    Answers "does the depth profile matter, and by how much" with two real
    SCAPS runs instead of an assertion. Roughly two minutes.
    """
    graded = simulate_stack(
        job=f"{job}_graded", architecture=architecture, absorber=absorber, htl=htl,
        etl=etl, absorber_thickness_nm=absorber_thickness_nm,
        defect_profile={"kind": "exp-tail", "surface_Nt": surface_Nt,
                        "bulk_Nt": bulk_Nt, "decay_nm": decay_nm},
        sublayers=sublayers)
    uniform = simulate_stack(
        job=f"{job}_uniform", architecture=architecture, absorber=absorber, htl=htl,
        etl=etl, absorber_thickness_nm=absorber_thickness_nm,
        defect_profile=uniform_Nt)
    keys = ("Voc_V", "Jsc_mA_cm2", "FF_percent", "PCE_percent")
    delta = {k: (graded[k] - uniform[k]) if graded.get(k) is not None and
             uniform.get(k) is not None else None for k in keys}
    return {
        "graded": {k: graded.get(k) for k in keys},
        "uniform": {k: uniform.get(k) for k in keys},
        "difference_graded_minus_uniform": delta,
        "profile": graded["device"]["layers"][1]["defect_profile_1cm3"],
        "profile_units": "depth from the illuminated surface, Nt in 1/cm^3",
        "conclusion": (
            "Grading the defect density toward the illuminated surface costs "
            f"{abs(delta['PCE_percent']):.2f} percentage points of PCE versus a uniform "
            f"{uniform_Nt:g} 1/cm^3 absorber at the same mean thickness."
            if delta["PCE_percent"] is not None else "See the two runs above."),
        "jobs": [graded.get("job"), uniform.get("job")],
        "elapsed_s": [graded.get("elapsed_s"), uniform.get("elapsed_s")],
    }
