"""Attack-strength check: would a much larger attack budget evade more of the frauds the grid failed to evade?

A model can look robust only because the attacks gave up too early (Carlini et al. 2019). This script reruns the
grid's attacks with five times their budget and reports how many more frauds evade.

For every model of a dataset, in both threat models (unconstrained, constrained), at eps = EPS_C = 0.25, on the
frauds the grid's worst case was measured on (the detected frauds of the black-box subset of the shared fraud sample):
  * APGD with 5x the grid's iterations per restart (500; 100 for foundation models), the grid's number of restarts
    (5; 2 for foundation models) and other random starts (seed 100 instead of the grid's 0);
  * Square with 5x the grid's queries (1,500; 500 for foundation models), run twice (seeds 200 and 201).
Tree ensembles get the strong Square only: their white-box attack in the grid is already the exact MILP.
Foundation models are checked on a fixed random FM_N = 60 of those frauds (all of them when there are fewer),
because one of their gradients costs about 0.2 s per fraud.

A fraud counts as evaded under the strong attack when the grid's worst case evaded it or any strong attack does
(the union). Per model and threat model, the row reports the grid's attack success rate (ASR), the union's ASR, both
also weighted by the fraud amount, and the number of new evasions.

Inputs (runs/<dataset>/attacks/): tau.json (deployed thresholds), per_example*.npz from the grid (key
"<family>|<mode>|0.25": one row per sampled fraud with [test index, detected, evaded by the grid's worst case,
amount]), and the trained models.
Output: runs/<dataset>/attacks/budget_check<tag>.csv, one row per (model, threat model), saved after every row and
resumable (rows already in the file are skipped). With --tag, two processes on one dataset write separate files
instead of overwriting each other's rows; concatenate them into budget_check.csv, the file the report reads.

Results: Table S-XXI (supplement, five-times budget check) and the budget-check numbers of sections VI-B/VI-C
("7 of 92 cells", printed by scripts/check_claims.py).

usage: python -m fraudadv.attacks.budget_check <dataset> fam1,fam2,... [--tag _x]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

EPS_C = 0.25       # the budget checked: the paper's headline budget
FM_N = 60          # frauds of the grid's worst-case subset checked for each foundation model


def _grid_per_example(out):
    """Every per-example array of the grid, from all per_example*.npz files of the dataset (later files win)."""
    pe = {}
    for f in sorted(out.glob("per_example*.npz")):
        z = np.load(f)
        pe.update({k: z[k] for k in z.files})
    return pe


def main(dataset, families, tag=""):
    """Run the check for `families` of `dataset` and write budget_check<tag>.csv."""
    if any(f in ("chronos2", "timesfm") for f in families):
        import chronos  # noqa: F401  before TensorFlow (ivis)
    from fraudadv.attacks.blackwhite import APGD_BUDGET, apgd, square
    from fraudadv.attacks.constraints import Constraints
    from fraudadv.attacks.run_attacks import FM, RUNS, TREES, context_ids, load_model
    from fraudadv.data.load import load

    a, meta = load(dataset)
    out = RUNS / dataset / "attacks"
    tau = json.loads((out / "tau.json").read_text())
    keep = context_ids(dataset)
    pe = _grid_per_example(out)
    p = out / f"budget_check{tag}.csv"
    rows = pd.read_csv(p).to_dict("records") if p.exists() else []
    have = {(r["model"], r["mode"]) for r in rows}
    for fam in families:
        if fam not in tau:
            continue
        m = None                                          # loaded on first use: a finished model is not loaded
        for mode in ("unconstrained", "constrained"):
            key = f"{fam}|{mode}|{EPS_C}"
            if (fam, mode) in have or key not in pe:
                continue
            m = m or load_model(dataset, fam, 0)
            t0 = time.time()
            P = pe[key]                                   # rows: [test index, detected, evaded (grid worst case), amount]
            if fam in FM and len(P) > FM_N:               # foundation models: a fixed random subset
                P = P[np.sort(np.random.default_rng(3).permutation(len(P))[:FM_N])]
            det = P[:, 1] == 1
            idx = P[det, 0].astype(int)
            ev_grid = P[det, 2] == 1
            amt = P[det, 3]
            X0, c0 = a["X_test"][idx], keep["test"][idx]
            cons = Constraints(meta, a["X_train"], mode)
            t = float(tau[fam])
            fm = fam in FM
            best = m.logit(X0, c0)                        # each fraud's lowest logit over the strong attacks
            if fam not in TREES:
                it_, rs_ = APGD_BUDGET["fm" if fm else "default"]       # the grid's APGD budget, iterations x 5
                _, l = apgd(m, X0, t, EPS_C, cons, n_iter=5 * it_, restarts=rs_, seed=100, ctx=c0)
                best = np.minimum(best, l)
            for r in range(2):
                _, l = square(m, X0, t, EPS_C, cons, n_queries=500 if fm else 1500, seed=200 + r, ctx=c0)
                best = np.minimum(best, l)
            ev_strong = ev_grid | (best < t)
            n = int(det.sum())
            rows.append(dict(dataset=dataset, model=fam, mode=mode, eps=EPS_C, n_detected=n,
                             asr_grid=float(ev_grid.mean()) if n else np.nan, asr_strong=float(ev_strong.mean()) if n else np.nan,
                             asr_usd_grid=float(amt[ev_grid].sum() / max(amt.sum(), 1e-9)),
                             asr_usd_strong=float(amt[ev_strong].sum() / max(amt.sum(), 1e-9)),
                             new_evasions=int((ev_strong & ~ev_grid).sum()), seconds=round(time.time() - t0)))
            print(dataset, fam, mode, "grid", round(rows[-1]["asr_grid"], 3), "strong", round(rows[-1]["asr_strong"], 3),
                  "in", rows[-1]["seconds"], "s", flush=True)
            pd.DataFrame(rows).to_csv(p, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Five-times attack budget check (Table S-XXI).")
    ap.add_argument("dataset")
    ap.add_argument("families", help="comma-separated model families")
    ap.add_argument("--tag", default="", help="suffix of the output file, budget_check<tag>.csv (a second process "
                                              "on the same dataset)")
    args = ap.parse_args()
    main(args.dataset, args.families.split(","), tag=args.tag)
