"""Standard SCAPS-1D execution runner.

Division of labour, which is the whole point of this module:

  * the *agent* writes a device file (`job.def`) and an execution script
    (`job.script`) using `write_execution_script()`;
  * this runner knows nothing about perovskites, band gaps or layer stacks. It
    only knows how to hand a script to the wine64-hosted SCAPS executable, wait
    for the IV file it produces, and hand the parsed numbers back.

So all the wine/Windows/DLL environment handling stays in one place, and the
agent's own output is inspectable text that can be read, diffed and re-run by
hand in SCAPS.

Why the locking and the naming discipline
-----------------------------------------
SCAPS is a GUI program driven by an action list. It does not exit on its own
after `calculate`, so the runner kills it once the result file is complete. Two
copies sharing one script path or one result name corrupt each other, and
wine's prefix is itself a shared mutable directory, so every run gets a unique
job name and all runs are serialised through a lock file.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import subprocess
import time
from typing import Any, Dict, Optional

from .. import config

# The action list. This is the contract with the SCAPS GUI: load a .def, set up
# the IV sweep, run a single shot under AM1.5G, save the IV file.
EXECUTION_SCRIPT_TEMPLATE = """set quitscript.quitSCAPS
set errorhandling.overwritefile
load definitionfile {defname}
clear actions
action workingpoint.temperature {temperature}
action iv.startv {startv}
action iv.stopv {stopv}
action iv.increment {step}
action iv.doiv
action light
calculate singleshot
save results.iv {ivname}
"""

RESULT_SENTINEL = "deduced from calculated IV-curve"


class ScapsExecutionError(RuntimeError):
    """SCAPS could not be started or produced no usable result."""


def scaps_dir() -> str:
    return os.path.join(config.SCAPS_PREFIX, "drive_c", "Program Files (x86)", "Scaps3309")


def def_dir() -> str:
    return os.path.join(scaps_dir(), "def")


def script_dir() -> str:
    return os.path.join(scaps_dir(), "script")


def results_dir() -> str:
    return os.path.join(scaps_dir(), "results")


def executable() -> str:
    return os.path.join(scaps_dir(), "scaps3309.exe")


def check_installation() -> Dict[str, Any]:
    """Is a wine64 + SCAPS installation present and complete?"""
    exe, wine = executable(), config.SCAPS_WINE
    missing = [p for p, label in ((wine, "wine64"), (exe, "scaps3309.exe")) if not p
               or not os.path.exists(p)]
    return {
        "available": not missing,
        "wine64": wine,
        "scaps_executable": exe,
        "wine_version": _wine_version(wine) if wine and os.path.exists(wine) else None,
        "missing": missing,
        "error": None if not missing else (
            f"SCAPS 1D cannot run: missing {', '.join(missing)}. Set "
            f"PEROVAGENTS_SCAPS_WINE / PEROVAGENTS_SCAPS_PREFIX."),
    }


def _wine_version(wine: str) -> Optional[str]:
    try:
        out = subprocess.run([wine, "--version"], capture_output=True, text=True, timeout=30)
        return (out.stdout or out.stderr).strip().splitlines()[0] if (
            out.stdout or out.stderr) else None
    except Exception:
        return None


def write_execution_script(defname: str, job: str, temperature: float = 300.0,
                           startv: float = 0.0, stopv: float = 1.6, step: float = 0.02,
                           ivname: Optional[str] = None) -> Dict[str, str]:
    """Write the SCAPS action list for one job. Returns the file paths.

    This is the artefact the agent produces: a plain-text action list, exactly
    what SCAPS itself reads, so it stays valid even if the runner is replaced.
    """
    ivname = ivname or f"{job}.iv"
    text = EXECUTION_SCRIPT_TEMPLATE.format(
        defname=defname, ivname=ivname, temperature=temperature,
        startv=startv, stopv=stopv, step=step)
    os.makedirs(script_dir(), exist_ok=True)
    path = os.path.join(script_dir(), f"{job}.script")
    with open(path, "w") as fh:
        fh.write(text)
    return {"script_path": path, "script_text": text,
            "iv_name": ivname, "iv_path": os.path.join(results_dir(), ivname)}


def parse_iv_file(path: str) -> Dict[str, Any]:
    """Pull Voc/Jsc/FF/eta (and convergence state) out of a SCAPS .iv file.

    The "solar cell parameters deduced from calculated IV-curve" banner is NOT a
    success signal on its own: SCAPS writes that header even when the voltage
    sweep aborted early (deep reverse bias, or a mesh that stops converging),
    in which case it only manages to print Jsc. A usable result therefore
    requires all four parameters to be present, and we also report how far the
    sweep actually got so the failure is diagnosable.
    """
    with open(path, errors="ignore") as fh:
        text = fh.read()

    def grab(pattern: str) -> Optional[float]:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    voc = grab(r"Voc =\s+([\d.eE+-]+)\s+Volt")
    jsc = grab(r"Jsc =\s+([\d.eE+-]+)\s+mA")
    ff = grab(r"FF =\s+([\d.eE+-]+)\s+[%]")
    eta = grab(r"eta =\s+([\d.eE+-]+)\s+[%]")
    volts = [float(v) for v in re.findall(r"^\s*([-+]?\d+\.\d+)\s", text, re.M)]
    v_max = max(volts) if volts else None

    out: Dict[str, Any] = {
        "converged": (RESULT_SENTINEL in text) and None not in (voc, jsc, ff, eta),
        "sweep_completed": RESULT_SENTINEL in text,
        "convergence_failure": "Convergence failure" in text,
        "V_max_reached": v_max,
        "Voc_V": voc,
        "Jsc_mA_cm2": jsc,
        "FF_percent": ff,
        "PCE_percent": eta,
    }
    if not out["converged"]:
        # Surface SCAPS's own complaint instead of a bare "no result".
        notes = re.findall(r"^.*(?:Error|Warning|fail|not converge).*$", text,
                           flags=re.MULTILINE | re.IGNORECASE)
        out["scaps_messages"] = [n.strip() for n in notes[:5]]
        if out["sweep_completed"] and v_max is not None:
            out["incomplete_reason"] = (
                f"the IV sweep aborted at V={v_max:.2f} V before reaching Voc; "
                f"only {', '.join(k for k, v in (('Voc', voc), ('Jsc', jsc), ('FF', ff), ('eta', eta)) if v is None)} "
                f"was reported. Lower iv.stopv (deep reverse bias breaks the mesh) "
                f"or raise iv.increment.")
    return out


def _kill_scaps(proc: subprocess.Popen) -> None:
    """SCAPS never exits by itself; make sure no stale copy holds the prefix."""
    # The run owns its own session (start_new_session), so killpg takes down
    # wine's loader and the reparented start.exe / scaps3309.exe with it.
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        if proc.poll() is None:
            proc.kill()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    subprocess.run(["pkill", "-f", "scaps3309"], capture_output=True)


def execute(script_path: str, timeout: Optional[int] = None,
            keep_artifacts: bool = True) -> Dict[str, Any]:
    """Run an agent-written SCAPS action list and return the parsed IV result.

    `script_path` is the file returned by `write_execution_script()`; its
    directory tells us the job name. The result carries the provenance an
    answer needs: which .def, which script, which wine, how long it took.
    """
    status = check_installation()
    if not status["available"]:
        raise ScapsExecutionError(status["error"])

    timeout = timeout or config.SCAPS_TIMEOUT
    job = os.path.splitext(os.path.basename(script_path))[0]
    script_text = open(script_path, errors="ignore").read()
    m = re.search(r"save results\.iv\s+(\S+)", script_text)
    ivname = m.group(1) if m else f"{job}.iv"
    iv_path = os.path.join(results_dir(), ivname)
    os.makedirs(results_dir(), exist_ok=True)
    if os.path.exists(iv_path):
        os.remove(iv_path)                       # never read a stale curve

    env = config.scaps_env()
    lock_path = os.path.join(config.SCAPS_DIR, ".perovagents_scaps.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    started = time.time()
    with open(lock_path, "w") as lock:
        # Serialise: one SCAPS per wine prefix, and a shared prefix plus a
        # shared script path is a recipe for a half-written action list.
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            argv = [config.SCAPS_WINE, os.path.basename(executable()),
                    os.path.basename(script_path)]
            proc = subprocess.Popen(argv, cwd=scaps_dir(), env=env,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    start_new_session=True)
            while time.time() - started < timeout:
                if os.path.exists(iv_path):
                    with open(iv_path, errors="ignore") as fh:
                        if RESULT_SENTINEL in fh.read():
                            break
                if proc.poll() is not None:
                    break
                time.sleep(1.5)
            time.sleep(1.0)                        # let the file finish flushing
            _kill_scaps(proc)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

    elapsed = time.time() - started
    if not os.path.exists(iv_path):
        raise ScapsExecutionError(
            f"SCAPS produced no result file within {timeout:.0f}s "
            f"(expected {ivname}). The device definition may not have converged "
            f"or the executable may have failed to start under wine.")
    result = parse_iv_file(iv_path)
    result.update({
        "job": job,
        "script_path": script_path,
        "iv_path": iv_path,
        "elapsed_s": round(elapsed, 1),
        "timed_out": not result["converged"],
        "engine": "SCAPS 3.3.02 under wine64",
        "wine64": status["wine64"],
        "wine_version": status["wine_version"],
    })
    if not keep_artifacts and os.path.exists(iv_path):
        os.remove(iv_path)
    if not result["converged"]:
        detail = result.get("incomplete_reason") or "; ".join(
            result.get("scaps_messages") or []) or "no IV curve written"
        raise ScapsExecutionError(f"SCAPS did not converge for job {job!r}: {detail}")
    return result


def install_device_file(text: str, defname: str) -> str:
    """Copy agent-authored .def text into the directory SCAPS reads from."""
    if not defname.endswith(".def"):
        defname += ".def"
    os.makedirs(def_dir(), exist_ok=True)
    path = os.path.join(def_dir(), defname)
    with open(path, "w") as fh:
        fh.write(text)
    return path


def simulate(def_text: str, job: str, temperature: float = 300.0, startv: float = 0.0,
             stopv: float = 1.6, step: float = 0.02,
             timeout: Optional[int] = None) -> Dict[str, Any]:
    """Convenience path: .def text in, parsed IV numbers out.

    Still writes both files, so the agent's actual SCAPS inputs remain on disk
    and reproducible.
    """
    def_path = install_device_file(def_text, f"{job}.def")
    script = write_execution_script(os.path.basename(def_path), job=job,
                                    temperature=temperature, startv=startv,
                                    stopv=stopv, step=step)
    result = execute(script["script_path"], timeout=timeout)
    result["def_path"] = def_path
    result["def_text"] = def_text
    result["execution_script"] = script["script_text"]
    return result


def _main(argv=None) -> int:
    """CLI entry point: `python -m perovagents.physics.scaps_run <script>`."""
    import argparse
    ap = argparse.ArgumentParser(description="Execute a SCAPS-1D action list.")
    ap.add_argument("script", help="path to the .script written for SCAPS")
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    a = ap.parse_args(argv)
    try:
        res = execute(a.script, timeout=a.timeout)
    except ScapsExecutionError as exc:
        print(f"SCAPS execution failed: {exc}")
        return 1
    if a.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"Voc = {res['Voc_V']:.4f} V | Jsc = {res['Jsc_mA_cm2']:.3f} mA/cm2 | "
              f"FF = {res['FF_percent']:.2f} % | PCE = {res['PCE_percent']:.2f} % "
              f"({res['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
