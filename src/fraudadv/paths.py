"""Where the study reads and writes. Defaults are folders next to this code; environment variables override them.

    FRAUDADV_RAW      raw source files, one folder per dataset                  (default data/raw)
    FRAUDADV_PROC     processed arrays read by fraudadv.data.load               (default data/processed)
    FRAUDADV_RUNS     trained models and every attack output                    (default runs)
    FRAUDADV_RUNS_CONSISTENT  outputs of the consistent attacker                (default runs_consistent)
    FRAUDADV_REPORTS  summary CSVs and figures made from runs (make_report)     (default reports)
    FRAUDADV_LOGS     logs                                                      (default logs)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if not (REPO / "pyproject.toml").exists():    # installed as a package, not run from the folder: use the cwd
    REPO = Path.cwd()

RAW = Path(os.environ.get("FRAUDADV_RAW", REPO / "data" / "raw"))
PROC = Path(os.environ.get("FRAUDADV_PROC", REPO / "data" / "processed"))
RUNS = Path(os.environ.get("FRAUDADV_RUNS", REPO / "runs"))
RUNS_CONSISTENT = Path(os.environ.get("FRAUDADV_RUNS_CONSISTENT", REPO / "runs_consistent"))
REPORTS = Path(os.environ.get("FRAUDADV_REPORTS", REPO / "reports"))
LOGS = Path(os.environ.get("FRAUDADV_LOGS", REPO / "logs"))


def attack_dir(dataset, consistent=False):
    """Folder of a dataset's attack outputs. The consistent attacker's outputs live apart, in
    runs_consistent/<dataset>/attacks/, so that the unconstrained/constrained tables never mix in a third attacker."""
    return (RUNS_CONSISTENT if consistent else RUNS) / dataset / "attacks"
