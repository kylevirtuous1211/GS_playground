"""Every path in this repository resolves from here.

No module and no script may contain an absolute path. The root is derived from
this file's own location; each anchor can be redirected with an environment
variable so data and run outputs can live on another volume without editing
code:

    GS_PLAYGROUND_ROOT      repository root
    GS_PLAYGROUND_DATA      datasets and checkpoints (read-mostly, large)
    GS_PLAYGROUND_OUTPUTS   run directories (write-heavy, large)
"""

from __future__ import annotations

import os
from pathlib import Path

# src/gs_playground/paths.py -> src/gs_playground -> src -> <root>
ROOT = Path(os.environ.get("GS_PLAYGROUND_ROOT", Path(__file__).resolve().parents[2]))

DATA = Path(os.environ.get("GS_PLAYGROUND_DATA", ROOT / "data"))
OUTPUTS = Path(os.environ.get("GS_PLAYGROUND_OUTPUTS", ROOT / "outputs"))

EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
THIRD_PARTY = ROOT / "third_party"
CLONES = THIRD_PARTY / "clones"

#: Upstream clones, by the key used in third_party/PINS.tsv.
PUFFIN = CLONES / "puffin"
PUFFIN_WORLD = PUFFIN / "Puffin-World"


def run_dir(study: str, arm: str, *, create: bool = False) -> Path:
    """Semantic run directory: outputs/<study>/<arm>/.

    Never a uuid and never a bare timestamp -- a comparison table has to be
    buildable from these names alone.
    """
    path = OUTPUTS / study / arm
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    for name in ("ROOT", "DATA", "OUTPUTS", "PUFFIN_WORLD"):
        value = globals()[name]
        print(f"{name:16} {value}  {'ok' if value.exists() else 'MISSING'}")
