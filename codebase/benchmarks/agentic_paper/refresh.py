"""One command to bring every derived artefact up to date with the runs so far.

    python -m benchmarks.agentic_paper.refresh            # score, tables, figures
    python -m benchmarks.agentic_paper.refresh --final    # same, without the interim stamp
    python -m benchmarks.agentic_paper.refresh --export-to /path/to/manuscript/figures

The figures keep the same file names and geometry on every refresh, so a
manuscript that includes them picks up new results simply by re-copying (or
re-pointing at) ``figures/``. ``--export-to`` copies the four figure files into
the given directory; it is never run automatically.
"""
import shutil
import sys
from pathlib import Path

from . import analyze, figures
from .paths import FIG_DIR

NAMES = ["fig1_overall_comparison.pdf", "fig1_overall_comparison.png",
         "fig2_category_performance.pdf", "fig2_category_performance.png"]


def main():
    analyze.main()
    figures.main()
    if "--export-to" in sys.argv:
        dest = Path(sys.argv[sys.argv.index("--export-to") + 1])
        dest.mkdir(parents=True, exist_ok=True)
        for n in NAMES:
            shutil.copy2(FIG_DIR / n, dest / n)
        print("exported", NAMES, "->", dest)


if __name__ == "__main__":
    main()
