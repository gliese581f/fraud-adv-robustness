"""Certify the axis-aligned threshold search (AATS) against the exact MILP (Kantchelian et al. 2016).

What it does
  For each tree model (XGBoost, CatBoost) of a dataset it takes up to n test frauds the model detects at its
  1%-FPR threshold, attacks them with AATS (attacks/blackwhite.aats), and then solves the exact MILP
  (attacks/milp.py) inside the radius r = min(AATS cost, EPS_MAX) + 1e-4. Restricting the MILP to r fixes every
  predicate that costs more than r and shrinks the program by orders of magnitude. It loses nothing, because any
  strictly cheaper evasion lies inside r. Each fraud gets one outcome:
    optimal       the MILP solved: exact minimal distance d*, and gap = AATS / d* - 1
    aats_optimal  the MILP is infeasible inside the AATS radius: AATS is optimal (certified)
    robust        both fail inside EPS_MAX and the MILP is infeasible: certified robust at EPS_MAX
    timeout       the time limit was hit: the dual bound is a certified lower bound

Threat models (--mode)
  constrained, unconstrained  the grid's domain-valid threat models, with discrete domains and the affine
                              log-ratio recomputation inside the MILP, so that the certificate is for the problem
                              the grid solves. It is exact except on PaySim, whose balance recomputation is
                              nonlinear (column exact_domain = False).
  continuous                  every feature continuous and mutable within the training range (the first
                              certification).

Inputs: data/processed/<dataset>/ and the models in runs/<dataset>/<family>/.
Output: runs/<dataset>/attacks/certify_<family>.csv (continuous) or certify_<family>_<mode>.csv, one row per fraud.
A family whose file already exists is skipped.

Results: Table S-XXII (constrained and unconstrained), Table S-XXIII (continuous), through
make_report.certify_table (reports/table8_aats_vs_milp.csv).

usage: python -m fraudadv.attacks.certify <dataset> [--n 100] [--time-limit 60] [--families xgboost,catboost]
       [--mode continuous|unconstrained|constrained]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from fraudadv.attacks.blackwhite import aats
from fraudadv.attacks.constraints import Constraints
from fraudadv.attacks.milp import KantchelianMILP
from fraudadv.attacks.run_attacks import RUNS, load_model
from fraudadv.data.load import load
from fraudadv.eval.metrics import threshold_at_fpr

EPS_MAX = 1.0


def outcome(status, verified, aats_cost):
    """The certification outcome of one fraud, from the MILP's status (0 optimal, 2 infeasible), whether its
    point evades on the real model, and the AATS cost (inf when AATS failed)."""
    if status == 0 and verified:
        return "optimal"
    if status == 2:
        return "aats_optimal" if np.isfinite(aats_cost) and aats_cost <= EPS_MAX else "robust"
    return "timeout"


def optimality_gap(aats_cost, milp_dist, result):
    """AATS distance / MILP distance - 1; 0 when AATS is certified optimal, nan when it is undefined."""
    if np.isfinite(aats_cost) and np.isfinite(milp_dist) and milp_dist > 0:
        return aats_cost / milp_dist - 1
    return 0.0 if result == "aats_optimal" else np.nan


def run(dataset, families=("xgboost", "catboost"), n=100, time_limit=60.0, mode="continuous"):
    a, meta = load(dataset)
    cons = Constraints(meta, a["X_train"], mode)
    suffix = "" if mode == "continuous" else f"_{mode}"
    frauds = np.flatnonzero(a["y_test"] == 1)
    rng = np.random.default_rng(0)            # one generator for all families, drawn in family order
    for fam in families:
        out_csv = RUNS / dataset / "attacks" / f"certify_{fam}{suffix}.csv"
        if out_csv.exists():
            continue
        m = load_model(dataset, fam, 0)
        tau = threshold_at_fpr(a["y_val"], m.logit(a["X_val"]), a["w_val"], 0.01)
        detected = frauds[m.logit(a["X_test"][frauds]) > tau]
        idx = np.sort(rng.choice(detected, size=min(n, len(detected)), replace=False))
        X0 = a["X_test"][idx]

        t0 = time.time()
        _, _, res = aats(m, X0, tau, EPS_MAX, cons)
        aats_seconds = (time.time() - t0) / len(X0)

        M = KantchelianMILP(m, fam, X0.shape[1], cons.lower, cons.upper, cons.mutable, time_limit=time_limit,
                            allowed=cons.allowed, cons=cons)
        rows = []
        for k, x0 in enumerate(X0):
            aats_cost = float(res.cost[k]) if res.success[k] else np.inf
            radius = min(aats_cost, EPS_MAX) + 1e-4
            t1 = time.time()
            x, _, lower_bound, status = M.solve_one(x0, tau, eps_max=radius)
            verified = x is not None and float(M.margin_fn(x[None])[0]) <= tau
            milp_dist = float(np.max(np.abs(x - x0)[M.mutable])) if verified else np.inf
            result = outcome(status, verified, aats_cost)
            rows.append(dict(dataset=dataset, model=fam, mode=mode, exact_domain=bool(M.exact_domain), row=int(idx[k]),
                             aats_cost=aats_cost, milp_dist=milp_dist, milp_lower_bound=lower_bound,
                             status=result, milp_seconds=time.time() - t1, aats_seconds=aats_seconds,
                             gap=optimality_gap(aats_cost, milp_dist, result)))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--families", default="xgboost,catboost")
    ap.add_argument("--mode", default="continuous", choices=["continuous", "unconstrained", "constrained"])
    args = ap.parse_args()
    run(args.dataset, args.families.split(","), args.n, args.time_limit, args.mode)
