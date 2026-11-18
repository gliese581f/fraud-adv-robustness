"""Small helpers for the notebooks in notebooks/: set up the environment, show results inline, and check the files a
notebook wrote against a reference copy (a folder holding a previous build of reports/).

    from fraudadv import notebook as nb        # after the setup cell of a notebook
    nb.show_csv("reports/table1_clean.csv")
    nb.show_figure("fig_transfer_constrained") # reports/<name>.png
    nb.check(["reports/table1_clean.csv"])     # compares with $FRAUDADV_REFERENCE/<same path>, if it is set
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Set before TensorFlow or matplotlib load: the study's Keras models need the legacy Keras API, and a fixed
# SOURCE_DATE_EPOCH makes matplotlib write the same PDF bytes on every run (it is the PDF's creation date).
ENVIRONMENT = {"TF_USE_LEGACY_KERAS": "1", "TF_CPP_MIN_LOG_LEVEL": "3", "SOURCE_DATE_EPOCH": "0",
               "HF_HUB_DISABLE_XET": "1"}


def setup():
    """Work from the folder root (so relative paths such as reports/ resolve) with the study's environment.
    Returns the folder root."""
    for k, v in ENVIRONMENT.items():
        os.environ.setdefault(k, v)
    os.chdir(REPO)
    from fraudadv.paths import REPORTS

    for folder in (REPORTS, REPORTS / "tables"):   # the report steps write here
        folder.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.set_option("display.max_columns", 60, "display.width", 200, "display.max_colwidth", 60)
    return REPO


def show_csv(path, rows=None, **read_csv):
    """Display a CSV file (relative to the folder root) as a table; `rows` limits how many rows are shown."""
    import pandas as pd
    from IPython.display import display

    df = pd.read_csv(REPO / path, **read_csv)
    display(df if rows is None else df.head(rows))
    return df


def show_figure(name, width=900):
    """Display figure <name> (reports/<name>.png) inline."""
    from IPython.display import Image, display

    p = REPO / "reports" / f"{name}.png"
    if p.exists():
        display(Image(filename=str(p), width=width))
    else:
        print(f"{name}.png not found in reports")


def _content(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data    # gzip stores a time stamp: compare content


def check(files, reference=None):
    """Compare each file (relative to the folder root) with the file at the same relative path under `reference`
    (default: the FRAUDADV_REFERENCE environment variable, a folder holding a previous build of reports/).
    Prints one line per file and returns True when every file is identical. Without a reference
    it only lists the files."""
    reference = reference or os.environ.get("FRAUDADV_REFERENCE")
    files = [str(f) for f in files]
    if not reference:
        print("no reference folder set (FRAUDADV_REFERENCE): wrote", len(files), "files")
        for f in files:
            print("  ", f, "" if (REPO / f).exists() else "(missing)")
        return None
    ok = True
    for f in files:
        mine, ref = REPO / f, Path(reference) / f
        if not mine.exists() or not ref.exists():
            status = "missing here" if not mine.exists() else "not in the reference"
            ok = False
        elif _content(mine) == _content(ref):
            status = "identical"
        else:
            status = "DIFFERENT"
            ok = False
        print(f"{status:>20}  {f}")
    print("all identical to the reference" if ok else "some files differ from the reference")
    return ok
