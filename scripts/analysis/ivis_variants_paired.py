"""Holm-adjusted paired comparisons of the ivis variants (Table S-XXVI): ivis retrained to use its similarity
learning, and a nearest-neighbour vote on its embedding, each against the MLP and the study's ivis.

Input: runs_ivisvar/report/paired.csv (written by `python scripts/analysis/ivis_variants.py report`): per dataset,
attacker, variant and comparator, the worst-case ASR at eps 0.25 of both on the frauds both detect and McNemar's
exact p-value. Output: runs_ivisvar/report/paired_holm.csv, the comparisons against the MLP and the study's ivis
(consistent attacker only where it differs from the constrained one), with the difference in ASR and the p-value
adjusted by Holm's step-down correction over all of them.

The other ivis-variant results are CSVs that ivis_variants.py writes itself: Table S-XXV is
runs_ivisvar/report/clean.csv (clean AUPRC) and worst_case.csv (worst-case ASR at eps 0.25 under C, K and U),
Table S-XXVII is geometry.csv (embedding geometry of each variant).

  python scripts/analysis/ivis_variants_paired.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "runs_ivisvar" / "report"
KDS = {"sparkov", "ieee_cis"}            # datasets with a separate consistent attacker (K); elsewhere K equals C


def modes(d):
    """Attackers compared on dataset d: K only where it differs from C."""
    return ("constrained", "consistent", "unconstrained") if d in KDS else ("constrained", "unconstrained")


def holm(p):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def main():
    P = pd.read_csv(SRC / "paired.csv")
    P = P[P.vs.isin(["mlp", "ivis"])].copy()
    P = P[[m in modes(d) for d, m in zip(P.dataset, P["mode"])]].copy()      # consistent only where it differs from C
    P["asr_difference"] = P.asr_model - P.asr_vs
    P["p_holm"] = holm(P.p.values)
    P.to_csv(SRC / "paired_holm.csv", index=False)
    print(P.sort_values("p_holm")[["dataset", "mode", "model", "vs", "asr_model", "asr_vs", "p_holm"]].head(40).round(3).to_string())


if __name__ == "__main__":
    main()
