"""Class-pair boundary thickness for every model on one dataset (Yang et al., NeurIPS 2020, Sec. 3).

Why class pairs
  The attack grid measures thickness along each fraud's own high-confidence adversarial direction. That direction
  only exists for frauds the attack flips within eps = 0.5, which on the most robust models leaves a handful of
  pairs. Yang et al. also define thickness on pairs drawn from different classes. Here every detected fraud in the
  shared attack sample is paired with a random test legit the model passes. The candidate partners are the same
  for every model, so thickness is measured on comparable segments for all models.

How a pair is formed, per threat setting
  unconstrained   the segment runs from the fraud to its legit partner.
  constrained     only attacker-controllable features may move: the partner is the fraud with its mutable features
                  set to the legit partner's values (then projected onto the constraints). Such a partner may still
                  be flagged; pairs that do not cross the threshold are dropped (n counts the kept ones, n_pairs all).

Score and band: threshold-centred as in the grid, g = 2 sigmoid(logit - tau) - 1, band alpha = 0, beta = 0.75.
Exact on tree ensembles, 128-point midpoint rule otherwise (fraudadv.boundary.thickness). Distances are L2 in
standardized feature units.

Inputs: runs/<d>/attacks/adv_unconstrained_<family>.npz (the fraud sample S and its clean rows X0),
        runs/<d>/attacks/tau.json (each model's 1%-FPR threshold), the models and the processed test set.
Output: runs/<d>/attacks/thickness_pairs.csv, one row per model and threat setting (resumable: rows already in the
        file are kept and skipped). Feeds Table S-XXIV (make_report.thickness_analysis, reports/table4_thickness_corr.csv).

usage: python -m fraudadv.boundary.run_thickness <dataset> fam1,fam2,...
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

MODES = ("unconstrained", "constrained")
N_CANDIDATES = 8          # candidate legit partners drawn per fraud; the first one the model passes is used


def make_pairs(m, X_test, ctx_test, X0, ctx0, detected, candidates, tau, cons, mode):
    """For each detected fraud i (positions in the sample), find its partner point.

    The partner is the first of the fraud's candidate legit rows that the model scores below tau (each candidate
    in its own context). Constrained: the fraud with only its mutable features copied from that legit row.
    Frauds whose candidates are all flagged get no pair. Returns lists (frauds, partners, fraud contexts)."""
    frauds, partners, contexts = [], [], []
    for i in detected:
        C = X_test[candidates[i]]
        lc = m.logit(C, ctx_test[candidates[i]])
        passed = np.flatnonzero(lc < tau)
        if not len(passed):
            continue
        partner = C[passed[0]].copy()
        if mode == "constrained":
            y = X0[i].copy()
            y[cons.mutable] = partner[cons.mutable]
            partner = cons.project(y[None], X0[i][None])[0]
        frauds.append(X0[i])
        partners.append(partner)
        contexts.append(ctx0[i])
    return frauds, partners, contexts


def pair_thickness(m, fam, Xr, Xs, cr, tau, n_steps):
    """Thickness of each (fraud, partner) segment, centred on the model's threshold tau."""
    from fraudadv.attacks.run_attacks import TREES
    from fraudadv.attacks.trees import TreeEnsemble
    from fraudadv.boundary.thickness import thickness_exact_trees, thickness_sampled

    if fam in TREES:
        ens = m.tree_ensemble()
        shifted = TreeEnsemble(ens.thresholds, lambda Z, f=ens.margin_fn: f(Z) - tau, ens.n_features, ens.rule)
        return thickness_exact_trees(shifted, Xr, Xs)
    return thickness_sampled(lambda Z, c: 1 / (1 + np.exp(-np.clip(m.logit(Z, c) - tau, -60, 60))), Xr, Xs,
                             n_steps=n_steps, ctx=cr)


def run(dataset, families, n_steps=128, n_fm=60, seed=0):
    """Compute and append the class-pair thickness rows of `families` (foundation models: first n_fm detected frauds)."""
    if any(f in ("chronos2", "timesfm") for f in families):
        import chronos  # noqa: F401  must be imported before TensorFlow (ivis): the reverse order aborts on an LLVM option clash
    from fraudadv.attacks.constraints import Constraints
    from fraudadv.attacks.run_attacks import FM, RUNS, context_ids, load_model
    from fraudadv.data.load import load

    a, meta = load(dataset)
    out = RUNS / dataset / "attacks"
    tau = json.loads((out / "tau.json").read_text())
    ctx_ids = context_ids(dataset)
    legit = np.flatnonzero(a["y_test"] == 0)
    csv = out / "thickness_pairs.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["model"], r["mode"]) for r in rows}
    for fam in families:
        adv_file = out / f"adv_unconstrained_{fam}.npz"
        if not adv_file.exists() or fam not in tau:
            print("skip", fam, "(no attack outputs)", flush=True)
            continue
        if all((fam, mode) in done for mode in MODES):
            continue
        z = np.load(adv_file)
        S, X0 = z["S"], z["X0"]
        m = load_model(dataset, fam, seed)
        t = float(tau[fam])
        ctx0 = ctx_ids["test"][S]
        detected = np.flatnonzero(m.logit(X0, ctx0) > t)
        if fam in FM:
            detected = detected[:n_fm]
        # common random partners: one candidate list per fraud row, the same for every model
        candidates = np.random.default_rng(seed + 1).choice(legit, size=(len(S), N_CANDIDATES),
                                                            replace=False if len(legit) > N_CANDIDATES * len(S) else True)
        for mode in MODES:
            if (fam, mode) in done:
                continue
            cons = Constraints(meta, a["X_train"], mode)
            Xr, Xs, cr = make_pairs(m, a["X_test"], ctx_ids["test"], X0, ctx0, detected, candidates, t, cons, mode)
            if not Xr:
                continue
            Xr, Xs, cr = np.array(Xr), np.array(Xs), np.array(cr)
            crosses = m.logit(Xs, cr) < t                     # constrained partners may still be flagged
            th = pair_thickness(m, fam, Xr[crosses], Xs[crosses], cr[crosses], t, n_steps)
            dist = np.linalg.norm(Xr[crosses] - Xs[crosses], axis=1)
            rows.append(dict(dataset=dataset, model=fam, mode=mode, variant="class-pair", n=int(crosses.sum()), n_pairs=len(Xr),
                             thickness_mean=float(th.mean()) if len(th) else np.nan,
                             thickness_median=float(np.median(th)) if len(th) else np.nan,
                             thickness_rel=float(np.mean(th / np.maximum(dist, 1e-9))) if len(th) else np.nan,
                             seg_l2=float(dist.mean()) if len(th) else np.nan))
            print("thickness", rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(csv, index=False)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Class-pair boundary thickness (writes runs/<d>/attacks/thickness_pairs.csv)")
    ap.add_argument("dataset")
    ap.add_argument("families", help="comma-separated model families")
    args = ap.parse_args()
    run(args.dataset, args.families.split(","))
