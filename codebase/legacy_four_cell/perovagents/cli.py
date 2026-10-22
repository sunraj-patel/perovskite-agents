"""Console-script target (fixes `perovagents.cli:main`, AUDIT C-clash).

 Installed as `perovagents = "perovagents.cli:main"` in pyproject.toml;
previously this module did not exist and only `python -m perovagents`
worked. Thin shim over the real CLI.
"""
from .__main__ import main
