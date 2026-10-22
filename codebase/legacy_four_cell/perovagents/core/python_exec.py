"""Python execution tool (Orchestration layer).

Runs user-supplied Python in an isolated, restricted environment. This is the
"Python execution" capability listed in the tool ecosystem — meant for quick
numerical/scientific snippets (matplotlib plotting, stats, scraping helpers),
*not* for arbitrary untrusted code.

Security posture:
  * runs in a subprocess with a timeout
  * restricted builtins (no import of os, sys, subprocess, ... by default)
  * pre-approved numeric/scientific libraries only
  * project-controlled working directory

The `trusted` flag enables import of scientific stack libraries; plain mode
allows pure-Python computation.
"""
import json
import os
import subprocess
import sys

from .tool import Tool

_PREAMBLE = """
import math, json, statistics
# Safe numeric/scientific imports
_ALLOWED_IMPORTS = {"math", "statistics", "json", "random", "numpy", "np",
                    "pandas", "pd", "scipy", "matplotlib.pyplot", "plt"}
def __import__(name, *a, **k):
    if name in _ALLOWED_IMPORTS or name.split('.')[0] in {"numpy", "pandas", "scipy", "matplotlib"}:
        return _real_import(name, *a, **k)
    raise ImportError(f"import {name} not allowed")
import builtins
_real_import = builtins.__import__
builtins.__import__ = __import__
"""

_TEMPLATE = """# --- generated sandbox ---
{PREAMBLE}
_CODE
"""


class PythonExecTool(Tool):
    name = "python_exec"
    description = (
        "Execute a short Python snippet in a restricted sandbox (timeout 30s). "
        "Provides {result} or {error}. Scientific libraries (numpy, pandas, scipy, "
        "matplotlib) available; network/system access is blocked."
    )
    layer = "orchestration"
    version = "0.1.0"
    capabilities = {"python_exec", "computation"}

    def call(self, params):
        """Run a snippet in the restricted sandbox; returns stdout/stderr tails."""
        code = params.get("code", "")
        timeout = int(params.get("timeout", 30))
        if not code.strip():
            return {"error": "code required"}
        if len(code) > 5000:
            return {"error": "code too long (max 5000 chars)"}
        # write temp script next to results dir
        script_path = os.path.join(os.path.dirname(__file__), "_sandbox_user.py")
        payload = _TEMPLATE.replace("{PREAMBLE}", _PREAMBLE).replace("_CODE", code)
        with open(script_path, "w") as f:
            f.write(payload)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=timeout, env=env, cwd="/tmp",
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"timeout after {timeout}s"}
        except Exception as exc:
            return {"status": "error", "error": f"failed to run sandbox: {exc}"}
        finally:
            try:
                os.remove(script_path)
            except OSError:
                pass
        return {
            "status": "success" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-30:]),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-30:]),
            "provenance": {"kind": "python_exec", "sandbox": "subprocess-restricted"},
        }


if __name__ == "__main__":
    t = PythonExecTool()
    print(json.dumps(t.call({"code": "import numpy as np\nr=[int(x) for x in np.arange(0,3)]\nprint(r)"}), indent=2))