"""Clear a dataset's attack outputs for a few retrained model families, so that only those families are attacked again.

The second hyperparameter pass for the history-aware models (fraudadv.hpo.topup_context) replaces some families'
models. Their old attack results no longer describe the deployed models, and the other families' results are still
valid. This script removes the replaced families' rows from every per-model output of runs/<dataset>/attacks/:
  CSV files    results*.csv, thickness*.csv, robust_auc*.csv, history.csv, budget_check.csv: rows whose `model` is one
               of the families are dropped
  npz files    per_example*.npz: arrays whose key starts with "<family>|" are dropped
The resumable steps (thickness, robust AUC, history injection) then recompute exactly the missing families, and
run_attacks writes their new grid rows with --tag _ctx2 (results_ctx2.csv, per_example_ctx2.npz,
thickness_ctx2.csv), which the report reads. The shared fraud sample, the thresholds of the other families and their
outputs are left alone.

Before changing anything, the first call copies every file it may change, plus transfer_*.npz, to
runs/<dataset>/archive_ctx1/. Later calls keep that first copy.

No paper table reads this script's output directly. It is a one-time step of scripts/pipeline.sh (second tuning
pass of the history-aware models), kept so that the order of that pass can be followed.

usage: python -m fraudadv.attacks.rerun_prep <dataset> fam1,fam2,...
"""
from __future__ import annotations

import shutil
import sys

import numpy as np
import pandas as pd

from fraudadv.hpo.train_all import RUNS

# per-model CSV outputs of the attack steps; every one has a `model` column
CSVS = ["results*.csv", "thickness*.csv", "robust_auc*.csv", "history.csv", "budget_check.csv"]


def main(dataset, fams):
    out = RUNS / dataset / "attacks"
    arch = RUNS / dataset / "archive_ctx1"
    first = not arch.exists()                    # only the first call archives: later calls would copy pruned files
    arch.mkdir(exist_ok=True)
    for p in sorted({q for pat in CSVS for q in out.glob(pat)}):
        name = p.name
        if name.startswith("partial") or p.stat().st_size <= 1:      # checkpoints and empty placeholders
            continue
        if first:
            shutil.copy2(p, arch / name)
        d = pd.read_csv(p)
        keep = d[~d["model"].isin(fams)]
        if len(keep) < len(d):
            keep.to_csv(p, index=False)
            print(dataset, name, "dropped", len(d) - len(keep), "rows", flush=True)
    for p in sorted(out.glob("per_example*.npz")):
        if first:
            shutil.copy2(p, arch / p.name)
        z = np.load(p)
        kept = {k: z[k] for k in z.files if k.split("|")[0] not in fams}     # keys are "<family>|..."
        if len(kept) < len(z.files):
            np.savez_compressed(p, **kept)
            print(dataset, p.name, "dropped", len(z.files) - len(kept), "arrays", flush=True)
    if first:
        for p in out.glob("transfer_*.npz"):     # archived only, never pruned
            shutil.copy2(p, arch / p.name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
