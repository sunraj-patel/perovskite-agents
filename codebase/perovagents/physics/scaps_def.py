"""SCAPS-1D device definition builder for the design agent.

Turns a declarative stack (architecture + layers + contacts + defect profile)
into the exact `.def` text SCAPS 3.3 expects.

File layout SCAPS requires, always:

    back contact | layer_1 | interface | layer_2 | ... | front contact | working point

`working point / illuminated from` selects the end the light enters. SCAPS
always writes the hole-transport layer first (left, `back contact`) and the
electron-transport layer last (right, `front contact`), for both architectures;
only the illumination side differs:

    n-i-p : HTL | absorber | ETL,  illuminated from the right (ETL side)
    p-i-n : HTL | absorber | ETL,  illuminated from the left  (HTL side)

All values written here are SI, because the `.def` file is an SI file:
lengths in m, densities in 1/m^3, mobilities in m^2/Vs, velocities in m/s,
capture cross-sections in m^2. `perovagents.physics.scaps_tool` converts the
agent-facing engineering units (nm, 1/cm^3, cm^2/Vs) on the way in.

Defect profiles
----------------
SCAPS 3.3 writes `profile : homogenous` in every example shipped with it, and
`profile : linear` together with `Nleft`/`Nright` was measured to have **no
effect** on the IV curve of this build (identical Voc/Jsc/FF/PCE to 6 s.f.).
So a spatial defect profile is realised the way SCAPS actually supports it:
the layer is split into sub-layers carrying uniform `Nt`, and the recombination
at each sub-layer boundary is negligible because every layer has the same
material properties. That makes any profile -- uniform, exponential tail,
linear ramp, explicit piecewise list -- expressible and *verifiable*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Every per-layer field the writer understands, with its SI unit.
LAYER_FIELDS: Dict[str, str] = {
    "d": "m", "Eg": "eV", "chi": "eV", "eps": "-", "Nc": "1/m^3", "Nv": "1/m^3",
    "vthn": "m/s", "vthp": "m/s", "mun": "m^2/Vs", "mup": "m^2/Vs",
    "Na": "1/m^3", "Nd": "1/m^3", "Nt": "1/m^3", "Et": "eV",
    "sigma_n": "m^2", "sigma_p": "m^2",
    "A": "1/(m*sqrt(eV))", "B": "sqrt(eV)/m",
}

# Public (agent-facing) engineering unit per field, and the factor that turns it
# into the SI unit above. `None` means the value is already SI.
PUBLIC_UNITS: Dict[str, Tuple[str, float]] = {
    "d": ("nm", 1e-9),
    "Eg": ("eV", 1.0),
    "chi": ("eV", 1.0),
    "eps": ("-", 1.0),
    "Nc": ("1/cm^3", 1e6),
    "Nv": ("1/cm^3", 1e6),
    "vthn": ("cm/s", 1e-2),
    "vthp": ("cm/s", 1e-2),
    "mun": ("cm^2/Vs", 1e-4),
    "mup": ("cm^2/Vs", 1e-4),
    "Na": ("1/cm^3", 1e6),
    "Nd": ("1/cm^3", 1e6),
    "Nt": ("1/cm^3", 1e6),
    "Et": ("eV", 1.0),
    "sigma_n": ("cm^2", 1e-4),
    "sigma_p": ("cm^2", 1e-4),
    "A": ("1/(m*sqrt(eV))", 1.0),
    "B": ("sqrt(eV)/m", 1.0),
}


def to_si(field_name: str, value: float, units: str = "public") -> float:
    """Convert one layer field into the SI value that goes into the .def file.

    `units="public"` (default) interprets the value in engineering units, which
    is what the agent reasons in: nm, 1/cm^3, cm^2/Vs. `units="si"` passes it
    straight through, for callers that already converted.
    """
    if field_name not in LAYER_FIELDS:
        raise KeyError(f"unknown layer field {field_name!r}; "
                       f"valid fields: {sorted(LAYER_FIELDS)}")
    if units == "si":
        return float(value)
    if units != "public":
        raise ValueError(f"units must be 'public' or 'si', got {units!r}")
    return float(value) * PUBLIC_UNITS[field_name][1]


# ---------------------------------------------------------------------------
# Material presets.
#
# These are literature-typical room-temperature parameters used as *starting
# points*, not measurements of a specific film. Every entry is overridable per
# layer, and the agent is expected to ground the values it actually uses in the
# PDB / literature tools and state them in the answer.
#
# chi and Eg fix the band offsets; the transport numbers (mobility, DOS,
# thermal velocity) dominate the IV output and are the ones worth sourcing.
# ---------------------------------------------------------------------------
MATERIALS: Dict[str, Dict[str, float]] = {
    # ---- perovskite absorbers (Eg set per composition) -------------------
    "MAPbI3": {
        "Eg": 1.55, "chi": 3.93, "eps": 26.0, "Nc": 2.5e24, "Nv": 2.5e24,
        "mun": 1.0e-4, "mup": 1.0e-4, "vthn": 1e5, "vthp": 1e5,
        "Na": 1.0e22, "Nt": 1.0e20, "Et": 0.0, "sigma_n": 1.0e-19, "sigma_p": 1.0e-19,
        "A": 1.0e7, "B": 0.0,
    },
    "FAPbI3": {
        "Eg": 1.48, "chi": 4.00, "eps": 26.0, "Nc": 2.5e24, "Nv": 2.5e24,
        "mun": 1.0e-4, "mup": 1.0e-4, "vthn": 1e5, "vthp": 1e5,
        "Na": 1.0e22, "Nt": 1.0e20, "Et": 0.0, "sigma_n": 1.0e-19, "sigma_p": 1.0e-19,
        "A": 1.0e7, "B": 0.0,
    },
    "MixedPerovskite": {
        "Eg": 1.55, "chi": 3.93, "eps": 26.0, "Nc": 2.5e24, "Nv": 2.5e24,
        "mun": 1.0e-4, "mup": 1.0e-4, "vthn": 1e5, "vthp": 1e5,
        "Na": 1.0e22, "Nt": 1.0e20, "Et": 0.0, "sigma_n": 1.0e-19, "sigma_p": 1.0e-19,
        "A": 1.0e7, "B": 0.0,
    },
    # ---- electron transport layers --------------------------------------
    "TiO2": {
        "Eg": 3.20, "chi": 4.00, "eps": 9.0, "Nc": 2.2e24, "Nv": 2.2e24,
        "mun": 1.0e-4, "mup": 1.0e-4, "vthn": 1e5, "vthp": 1e5,
        "Nd": 1.0e24, "Nt": 1.0e23, "Et": 0.0, "sigma_n": 1.0e-18, "sigma_p": 1.0e-18,
        "A": 1.0e7, "B": 0.0,
    },
    "SnO2": {
        "Eg": 3.60, "chi": 4.05, "eps": 9.0, "Nc": 2.2e24, "Nv": 2.2e24,
        "mun": 5.0e-4, "mup": 5.0e-5, "vthn": 1e5, "vthp": 1e5,
        "Nd": 1.0e25, "Nt": 1.0e22, "Et": 0.0, "sigma_n": 1.0e-18, "sigma_p": 1.0e-18,
        "A": 1.0e7, "B": 0.0,
    },
    "PCBM": {
        "Eg": 3.10, "chi": 4.00, "eps": 4.0, "Nc": 2.2e24, "Nv": 2.2e24,
        "mun": 1.0e-3, "mup": 1.0e-3, "vthn": 1e5, "vthp": 1e5,
        "Nd": 1.0e23, "Nt": 1.0e22, "Et": 0.0, "sigma_n": 1.0e-17, "sigma_p": 1.0e-17,
        "A": 1.0e7, "B": 0.0,
    },
    "C60": {
        "Eg": 3.10, "chi": 4.10, "eps": 4.5, "Nc": 2.2e24, "Nv": 2.2e24,
        "mun": 1.0e-3, "mup": 1.0e-3, "vthn": 1e5, "vthp": 1e5,
        "Nd": 1.0e23, "Nt": 1.0e22, "Et": 0.0, "sigma_n": 1.0e-17, "sigma_p": 1.0e-17,
        "A": 1.0e7, "B": 0.0,
    },
    # ---- hole transport layers -------------------------------------------
    "Spiro-MeOTAD": {
        "Eg": 3.00, "chi": 2.20, "eps": 3.0, "Nc": 2.2e24, "Nv": 2.2e24,
        "mun": 2.0e-7, "mup": 2.0e-7, "vthn": 1e5, "vthp": 1e5,
        "Na": 1.0e24, "Nt": 1.0e23, "Et": 0.0, "sigma_n": 1.0e-19, "sigma_p": 1.0e-19,
        "A": 1.0e7, "B": 0.0,
    },
    "PTAA": {
        "Eg": 3.20, "chi": 2.20, "eps": 3.0, "Nc": 2.2e24, "Nv": 2.2e24,
        "mun": 1.0e-6, "mup": 1.0e-4, "vthn": 1e5, "vthp": 1e5,
        "Na": 1.0e24, "Nt": 1.0e22, "Et": 0.0, "sigma_n": 1.0e-19, "sigma_p": 1.0e-19,
        "A": 1.0e7, "B": 0.0,
    },
    "NiOx": {
        "Eg": 3.30, "chi": 2.30, "eps": 9.0, "Nc": 2.2e24, "Nv": 2.2e24,
        "mun": 1.0e-4, "mup": 1.0e-2, "vthn": 1e5, "vthp": 1e5,
        "Na": 1.0e25, "Nt": 1.0e22, "Et": 0.0, "sigma_n": 1.0e-17, "sigma_p": 1.0e-17,
        "A": 1.0e7, "B": 0.0,
    },
    "MoOx": {
        "Eg": 3.60, "chi": 2.20, "eps": 9.0, "Nc": 2.2e24, "Nv": 2.2e24,
        "mun": 1.0e-4, "mup": 1.0e-2, "vthn": 1e5, "vthp": 1e5,
        "Na": 1.0e25, "Nt": 1.0e22, "Et": 0.0, "sigma_n": 1.0e-17, "sigma_p": 1.0e-17,
        "A": 1.0e7, "B": 0.0,
    },
}

# Materials that work as an electron-transport layer (right / front side in n-i-p).
ETL_MATERIALS = ("TiO2", "SnO2", "PCBM", "C60")
# Materials that work as a hole-transport layer (left / back side in n-i-p).
HTL_MATERIALS = ("Spiro-MeOTAD", "PTAA", "NiOx", "MoOx")
ABSORBER_MATERIALS = ("MAPbI3", "FAPbI3", "MixedPerovskite")


def materials_catalogue() -> Dict[str, Any]:
    """Everything the agent needs to pick a material, with public units."""
    out: Dict[str, Any] = {}
    for name, p in MATERIALS.items():
        out[name] = {
            "role": ("absorber" if name in ABSORBER_MATERIALS else
                     "etl" if name in ETL_MATERIALS else "htl"),
            "values": {
                fld: (round(p[fld], 6), PUBLIC_UNITS[fld][0])
                for fld in ("Eg", "chi", "d", "mun", "mup", "Nc", "Nv",
                            "Na", "Nd", "Nt", "eps")
                if fld in p
            },
        }
    return out


# ---------------------------------------------------------------------------
# Defect profiles -> sub-layer expansion
# ---------------------------------------------------------------------------
def _exp_tail(depth_nm: float, spec: dict) -> float:
    bulk = float(spec.get("bulk_Nt", 1e14))
    surf = float(spec.get("surface_Nt", 1e17))
    decay = max(float(spec.get("decay_nm", 100.0)), 1e-9)
    return bulk + (surf - bulk) * math.exp(-depth_nm / decay)


def _linear(depth_nm: float, spec: dict, thickness_nm: float) -> float:
    front = float(spec.get("front_Nt", 1e14))
    back = float(spec.get("back_Nt", 1e17))
    frac = 0.0 if thickness_nm <= 0 else min(max(depth_nm / thickness_nm, 0.0), 1.0)
    return front + (back - front) * frac


def profile_densities(spec: Any, thickness_nm: float, n: int) -> List[float]:
    """`n` representative bulk defect densities (1/cm^3) across the layer.

    `spec` may be:
      * a number            -> uniform
      * None                -> uniform at 1e14 1/cm^3
      * {"kind": "exp-tail", "surface_Nt": .., "bulk_Nt": .., "decay_nm": ..}
      * {"kind": "linear", "front_Nt": .., "back_Nt": ..}
      * {"kind": "step", "levels": [[frac, Nt], ...]}
      * [(frac_or_thickness, Nt), ...]  explicit piecewise list

    Depth 0 is the *illuminated* surface of the absorber, which is where surface
    and interface recombination dominate; depth = thickness is the far surface.
    `expand_stack` flips the ordering for p-i-n so this stays true there.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if spec is None:
        return [1e14] * n
    if isinstance(spec, (int, float)):
        return [float(spec)] * n
    if isinstance(spec, (list, tuple)):
        pts = list(spec)
        if len(pts) < 2:
            raise ValueError("an explicit profile needs at least 2 [fraction, Nt] points")
        out = []
        for i in range(n):
            frac = (i + 0.5) / n
            for j in range(len(pts) - 1):
                f0, n0 = float(pts[j][0]), float(pts[j][1])
                f1, n1 = float(pts[j + 1][0]), float(pts[j + 1][1])
                if frac <= f1 or j == len(pts) - 2:
                    span = max(f1 - f0, 1e-12)
                    out.append(n0 + (n1 - n0) * min(max((frac - f0) / span, 0.0), 1.0))
                    break
        return out
    kind = str(spec.get("kind", "uniform"))
    if kind == "uniform":
        return [float(spec.get("bulk_Nt", 1e14))] * n
    if kind == "exp-tail":
        return [_exp_tail(thickness_nm * (i + 0.5) / n, spec) for i in range(n)]
    if kind == "linear":
        return [_linear(thickness_nm * (i + 0.5) / n, spec, thickness_nm) for i in range(n)]
    if kind == "step":
        levels = spec.get("levels") or []
        if not levels:
            raise ValueError("a step profile needs levels=[[fraction, Nt], ...]")
        out = []
        for i in range(n):
            frac = (i + 0.5) / n
            val = float(levels[-1][1])
            for lf, ln in levels:
                if frac <= float(lf):
                    val = float(ln)
                    break
            out.append(val)
        return out
    raise ValueError(f"unknown defect profile kind {kind!r}; use uniform, exp-tail, "
                     f"linear, step, or an explicit [[fraction, Nt], ...] list")


def is_graded(spec: Any) -> bool:
    """Does this profile need sub-layer expansion?"""
    if spec is None or isinstance(spec, (int, float)):
        return False
    if isinstance(spec, (list, tuple)):
        return True
    return str(spec.get("kind", "uniform")) not in ("uniform",)


# ---------------------------------------------------------------------------
# Layer / stack model
# ---------------------------------------------------------------------------
@dataclass
class LayerSpec:
    """One physical layer of the stack.

    `params` are given in *public* engineering units (nm, 1/cm^3, cm^2/Vs) and
    override the material preset. `thickness_nm` is kept separate because the
    defect-profile expansion needs it before conversion.
    """
    name: str
    material: str = "custom"
    thickness_nm: float = 100.0
    params: Dict[str, float] = field(default_factory=dict)
    defect_profile: Any = None
    # Each sub-layer adds a full interface + mesh, and SCAPS runtime grows
    # steeply with interface count (measured: 2 sub-layers 54 s, 8 sub-layers
    # >600 s). 4 is a good default; raise it only with a longer SCAPS_TIMEOUT.
    sublayers: int = 4

    def si_params(self) -> Dict[str, float]:
        base = dict(MATERIALS.get(self.material, {}))
        bad = set(self.params) - set(LAYER_FIELDS)
        if bad:
            raise KeyError(f"layer {self.name!r}: unknown field(s) {sorted(bad)}; "
                           f"valid fields: {sorted(LAYER_FIELDS)}")
        for fld, val in self.params.items():
            if val is None:
                continue
            base[fld] = to_si(fld, val, "public")
        base["d"] = self.thickness_nm * 1e-9
        for req in ("Eg", "chi", "eps", "Nc", "Nv", "vthn", "vthp", "mun", "mup"):
            if base.get(req) is None:
                raise KeyError(f"layer {self.name!r} is missing required field {req!r} "
                               f"(set it in params, or use a known material: "
                               f"{sorted(MATERIALS)})")
        base.setdefault("Na", 0.0)
        base.setdefault("Nd", 0.0)
        base.setdefault("A", 1e7)
        base.setdefault("B", 0.0)
        base.setdefault("Nt", 1e14 * 1e6)
        base.setdefault("Et", 0.0)
        base.setdefault("sigma_n", 1e-19)
        base.setdefault("sigma_p", 1e-19)
        return base


ARCHITECTURES = {
    "n-i-p": "electron transport layer on the illuminated (front/right) side",
    "p-i-n": "hole transport layer on the illuminated (back/left) side",
}


@dataclass
class Stack:
    """A complete device. Layers are listed back/left -> front/right."""
    architecture: str
    layers: List[LayerSpec]
    absorber_index: int = 1
    temperature_k: float = 300.0
    voltage_stop: float = 2.0
    voltage_step: float = 0.02
    notes: str = ""

    def validate(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {sorted(ARCHITECTURES)}, "
                             f"got {self.architecture!r}")
        if len(self.layers) < 2:
            raise ValueError("a device needs at least two layers")
        if not 0 <= self.absorber_index < len(self.layers):
            raise ValueError(f"absorber_index {self.absorber_index} out of range for "
                             f"{len(self.layers)} layers")
        # SCAPS reads back/left first, so the hole layer must be the first
        # block and the electron layer the last, for either architecture.
        back, front = self.layers[0], self.layers[-1]
        if back.material not in HTL_MATERIALS:
            raise ValueError(
                f"the first layer is the back/left contact side and must be a hole "
                f"transport layer {HTL_MATERIALS}; got {back.material!r}")
        if front.material not in ETL_MATERIALS:
            raise ValueError(
                f"the last layer is the front/right contact side and must be an "
                f"electron transport layer {ETL_MATERIALS}; got {front.material!r}")


# ---------------------------------------------------------------------------
# .def writers (formats taken from the .def files SCAPS itself saved)
# ---------------------------------------------------------------------------
def _contact(block: str, sn: float, sp: float) -> str:
    return f"""{block}

flatband : 1 (1: flatband contact imposed; 0: fixed Phi_m value)
recalculate : 1 (1 = do or 0 = do not recalculate phi_m to set flatband in each batch step)
Sn : {sn:12.6e} [m/s]
Sp : {sp:12.6e} [m/s]
Tunneling to contact: 0 (0: no tunneling, 1: with tunneling)
Relative electron mass :  1.00e+00
Relative hole mass :  1.00e+00
"""


def _interface() -> str:
    return ("interface properties\n\ninterfacename : \n"
            "Intraband tunneling : 0 (0: no tunneling, 1: with tunneling)\n"
            "Relative electron mass :  1.00e+00\nRelative hole mass :  1.00e+00\n")


def _layer_block(name: str, p: Dict[str, float]) -> str:
    def row(label: str, val: str) -> str:
        return f"{label} :\t{val}\t{val}\t{val}\t  1.000000e+00\t  1.000000e+00\t{val}\t{val}\t  0\t  0"
    return f"""layer

name : {name}
d : {p['d']:12.6e} [m]
composition grading :\t  0.00000\t  0.00000\t  0.00000\t  0.00000\t  0.00000\t  0.00000\t  0.00000\t 0\t  1\t[-]
v_th_n :\t 1.000e+05\t 1.000e+05\t 1.000e+01\t 1.000e+01\t 1.000e+01\t 1.000e+05\t 1.000e+05\t 0\t  0\t[m/s]
v_th_p :\t 1.000e+05\t 1.000e+05\t 1.000e+01\t 1.000e+01\t 1.000e+01\t 1.000e+05\t 1.000e+05\t 0\t  0\t[m/s]
eps :\t{p['eps']:12.6f}\t{p['eps']:12.6f}\t{p['eps']:12.6f}\t     1.000000\t     1.000000\t{p['eps']:12.6f}\t{p['eps']:12.6f}\t 0\t  0\t[-]
chi :\t{p['chi']:12.6f}\t{p['chi']:12.6f}\t{p['chi']:12.6f}\t  1.000000\t  1.000000\t{p['chi']:12.6f}\t{p['chi']:12.6f}\t 0\t  0\t[eV]
Eg :\t{p['Eg']:12.6f}\t{p['Eg']:12.6f}\t{p['Eg']:12.6f}\t  1.000000\t  1.000000\t{p['Eg']:12.6f}\t{p['Eg']:12.6f}\t 0\t  0\t[eV]
Nc :\t{p['Nc']:14.6e}\t{p['Nc']:14.6e}\t{p['Nc']:14.6e}\t  1.000000e+00\t  1.000000e+00\t{p['Nc']:14.6e}\t{p['Nc']:14.6e}\t 0\t  0\t[/m^3]
Nv :\t{p['Nv']:14.6e}\t{p['Nv']:14.6e}\t{p['Nv']:14.6e}\t  1.000000e+00\t  1.000000e+00\t{p['Nv']:14.6e}\t{p['Nv']:14.6e}\t 0\t  0\t[/m^3]
mu_n :\t{p['mun']:14.6e}\t{p['mun']:14.6e}\t{p['mun']:14.6e}\t  1.000000e+00\t  1.000000e+00\t{p['mun']:14.6e}\t{p['mun']:14.6e}\t 0\t  0\t[m^2/Vs]
mu_p :\t{p['mup']:14.6e}\t{p['mup']:14.6e}\t{p['mup']:14.6e}\t  1.000000e+00\t  1.000000e+00\t{p['mup']:14.6e}\t{p['mup']:14.6e}\t 0\t  0\t[m^2/Vs]
Tunneling in layer: 0 (0: no tunneling, 1: with tunneling)
Relative electron mass :\t 1.000000e+00\t 1.000000e+00\t 1.000000e+00\t 1.000000e+00\t 1.000000e+00\t 1.000000e+00\t 1.000000e+00\t 0\t  0\t[-]
Relative hole mass :\t 1.000000e+00\t 1.000000e+00\t 1.000000e+00\t 1.000000e+00\t 1.000000e+00\t 1.000000e+00\t 1.000000e+00\t 0\t  0\t[-]
K_rad :\t 0.000000e+00\t 0.000000e+00\t 1.000000e+01\t 1.000000e+01\t 1.000000e+01\t 0.000000e+00\t 0.000000e+00\t 0\t  0\t[m^3/s]
c_n_auger :\t 0.000000e+00\t 0.000000e+00\t 1.000000e+01\t 1.000000e+01\t 1.000000e+01\t 0.000000e+00\t 0.000000e+00\t 0\t  0\t[m^6/s]
c_p_auger :\t 0.000000e+00\t 0.000000e+00\t 1.000000e+01\t 1.000000e+01\t 1.000000e+01\t 0.000000e+00\t 0.000000e+00\t 0\t  0\t[m^6/s]
Na(y) :\t{p['Na']:14.6e}\t{p['Na']:14.6e}\t{p['Na']:14.6e}\t  1.000000e+00\t  1.000000e+00\t{p['Na']:14.6e}\t{p['Na']:14.6e}\t 0\t  0\t[/m^3]
Nd(y) :\t{p['Nd']:14.6e}\t{p['Nd']:14.6e}\t{p['Nd']:14.6e}\t  1.000000e+00\t  1.000000e+00\t{p['Nd']:14.6e}\t{p['Nd']:14.6e}\t 0\t  0\t[/m^3]
absorption grading :\t 1033.31\t 1033.31\t  250.00\t  250.00\t    0.00\t    1.20\t    1.20\t 0\t  0\t[nm]
absorptionmodel pure A material (y=0) : coefficients A and B
A : {p['A']:12.6e} [/(m*sqrt(eV))]
B : {p['B']:12.6e} [sqrt(eV)/m]
absorption pure B material (y=1), model : coefficients A and B
pure B material (y=1), A : {p['A']:12.6e} [/(m*sqrt(eV))]
pure B material (y=1), B : {p['B']:12.6e} [sqrt(eV)/m]

srhrecombination

type : neutral ( 1 level(s) )
energy distribution : single
Reference for defect energy :  0 ; 0 = above EV (SCAPS<2.7); 1 = above Ei; 2 = below EC
Ekar :  0.1000 [eV]
Nt(y) :\t{p['Nt']:14.6e}\t{p['Nt']:14.6e}\t{p['Nt']:14.6e}\t  1.000000e+00\t  1.000000e+00\t{p['Nt']:14.6e}\t{p['Nt']:14.6e}\t 0\t  0\t[/m^3]
profile : homogenous
Lkar : 1.000e+00 [m]
Nleft : {p['Nt']:14.6e} [/m^3]
Nright : {p['Nt']:14.6e} [/m^3]
leveltype : neutral\t
sigma_n : {p['sigma_n']:12.6e}\t[m^2]
sigma_p : {p['sigma_p']:12.6e}\t[m^2]
Et :   {p['Et']:.4f}\t[eV]

"""


def expand_stack(stack: Stack) -> List[Tuple[str, Dict[str, float]]]:
    """Flatten a Stack into the layer blocks SCAPS will read, in file order.

    A graded defect profile is realised as uniform-Nt sub-layers of the same
    material, which is what this SCAPS build actually honours.
    """
    stack.validate()
    # profile_densities() returns densities in order of increasing depth from
    # the illuminated surface, so its first entry is the high-defect end.
    # Sub-layers are written left -> right, and which end is first depends on
    # where the light enters: n-i-p is lit from the right, p-i-n from the left.
    lit_from_right = stack.architecture == "n-i-p"
    blocks: List[Tuple[str, Dict[str, float]]] = []
    for i, lay in enumerate(stack.layers):
        p = lay.si_params()
        spec = lay.defect_profile
        if i == stack.absorber_index and is_graded(spec):
            nts = profile_densities(spec, lay.thickness_nm, lay.sublayers)
            if lit_from_right:
                nts = nts[::-1]
            sub_d = lay.thickness_nm / lay.sublayers
            for k, nt in enumerate(nts):
                q = dict(p)
                q["d"] = sub_d * 1e-9
                q["Nt"] = to_si("Nt", nt, "public")
                blocks.append((f"{lay.name}_{k + 1}", q))
        else:
            if spec is not None and not is_graded(spec):
                p["Nt"] = to_si("Nt", profile_densities(spec, lay.thickness_nm, 1)[0], "public")
            blocks.append((lay.name, p))
    return blocks


def build_def_text(stack: Stack) -> str:
    """Render a Stack into the full `.def` text SCAPS 3.3 reads."""
    stack.validate()
    blocks = expand_stack(stack)
    # In a SCAPS .def the `back contact` block is always on the left and
    # `front contact` always on the right, so the p side is the left/back end
    # and the n side the right/front end for *both* architectures. What
    # distinguishes n-i-p from p-i-n is only which end the light enters:
    #   n-i-p -> light enters the ETL, the front/right end
    #   p-i-n -> light enters the HTL, the back/left end
    # Both device files SCAPS ships with agree: `perovskite_nip.def` is
    # illuminated from the right, `a-SiH pin.def` from the left, and both put
    # the p-type layer first in the file with identical contact blocks.
    front = _contact("front contact", 1.0e5, 1.0e-2)
    back = _contact("back contact", 1.0e-2, 1.0e5)
    illuminated = "right" if stack.architecture == "n-i-p" else "left"
    parts = [
        "> SCAPS 3.3.02 ELIS-UGent: Version scaps3302.exe\n"
        "> generated by perovagents design agent\n>\n>\n",
        "convergence\n\nrecalculate mesh :  1 (0: fixed mesh, 1: adaptive mesh)\n",
        back,
    ]
    for i, (name, p) in enumerate(blocks):
        if i:
            parts.append(_interface())
        parts.append(_layer_block(name, p))
    parts.append(front)
    parts.append(
        "working point\n\n"
        f"illuminated from : {illuminated}\n"
        f"internal Rfront :   0.00000000\n"
        f"internal Tfront :  1.0000e+00\n"
        f"voltage V applied to : -1 (+1: right contact; -1: left contact)\n"
        f"cell reference system :  1 (1: as a consumer (consumed power = +VJ); "
        f"0: as a generator (generated power = +VJ) )\n")
    return "\n".join(parts)


def _surface_role(i: int, stack: "Stack") -> str:
    """Where layer `i` sits relative to the incident light."""
    if i == stack.absorber_index:
        return "absorber"
    if i not in (0, len(stack.layers) - 1):
        return "interlayer"
    outermost = i == len(stack.layers) - 1
    lit = outermost if stack.architecture == "n-i-p" else not outermost
    return "illuminated surface side" if lit else "far surface side"


def stack_summary(stack: Stack) -> Dict[str, Any]:
    """Human/agent-readable record of exactly what will be simulated."""
    stack.validate()
    blocks = expand_stack(stack)
    layers_out = []
    for i, lay in enumerate(stack.layers):
        spec = lay.defect_profile
        graded = i == stack.absorber_index and is_graded(spec)
        entry: Dict[str, Any] = {
            "name": lay.name, "material": lay.material,
            "thickness_nm": round(lay.thickness_nm, 3),
            "role": _surface_role(i, stack),
            "defect_density_1e14cm3": None,
        }
        p = lay.si_params()
        entry["band_gap_eV"] = round(p["Eg"], 4)
        entry["electron_affinity_eV"] = round(p["chi"], 4)
        if graded:
            nts = profile_densities(spec, lay.thickness_nm, lay.sublayers)
            entry["sub_layers"] = lay.sublayers
            # Depth always runs away from the light, which is how the profile
            # was generated, so it is reported in that order for both
            # architectures regardless of how the blocks are written to the file.
            shown = nts
            entry["defect_profile_1cm3"] = [
                {"depth_nm": round(lay.thickness_nm * (k + 0.5) / lay.sublayers, 2),
                 "Nt_1cm3": round(v, 3)} for k, v in enumerate(shown)]
            entry["defect_density_range_1cm3"] = [round(min(shown), 3), round(max(shown), 3)]
        else:
            nt_cm3 = p["Nt"] / 1e6
            entry["defect_density_1e14cm3"] = round(nt_cm3 / 1e14, 4)
            flat = [{"depth_nm": 0.0, "Nt_1cm3": float(f"{nt_cm3:.3g}")},
                    {"depth_nm": round(lay.thickness_nm, 2),
                     "Nt_1cm3": float(f"{nt_cm3:.3g}")}]
            entry["defect_profile_1cm3"] = flat
        layers_out.append(entry)
    return {
        "architecture": stack.architecture,
        "architecture_meaning": ARCHITECTURES[stack.architecture],
        "illumination": ("front/right contact (ETL) side" if stack.architecture == "n-i-p"
                         else "back/left contact (HTL) side"),
        "n_scaps_layers": len(blocks),
        "layers": layers_out,
        "units": "public/engineering (nm, eV, 1/cm^3, cm^2/Vs)",
    }


def quick_stack(architecture: str = "n-i-p", etl: str = "TiO2", htl: str = "Spiro-MeOTAD",
                absorber: str = "MixedPerovskite", thickness_nm: float = 400.0,
                absorber_params: Optional[Dict[str, float]] = None,
                etl_thickness_nm: float = 50.0, htl_thickness_nm: float = 200.0,
                defect_profile: Any = None, sublayers: int = 4) -> Stack:
    """Build the conventional three-layer stack for either architecture.

    Both architectures are written as [HTL, absorber, ETL] back/left to
    front/right; n-i-p is illuminated from the front/right, p-i-n from the
    back/left.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {sorted(ARCHITECTURES)}")
    absorber_layer = LayerSpec("Perovskite", absorber, thickness_nm,
                               dict(absorber_params or {}), defect_profile, sublayers)
    # SCAPS always writes HTL first (left/back) and ETL last (right/front);
    # the architecture decides which end is illuminated.
    layers = [LayerSpec("HTL", htl, htl_thickness_nm),
              absorber_layer,
              LayerSpec("ETL", etl, etl_thickness_nm)]
    return Stack(architecture=architecture, layers=layers, absorber_index=1)
