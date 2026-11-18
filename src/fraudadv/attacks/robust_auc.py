"""Ranking metrics (AUC, KS) under an attacker who pushes every fraud's score as low as the budget allows.

Why a separate attacker: the attack grid uses min-effort evasions. Each fraud is moved just past the deployed
threshold, which is all a fraudster needs, so ranking metrics such as AUC and KS barely move even when most
detected frauds evade. Ranking metrics are read here against a score-minimising attacker instead.

What it does, per model, threat model (unconstrained, constrained) and budget eps in EPS_R = 0.1, 0.25, 0.5:
  1. Take a fixed random subset of the shared fraud sample (all frauds, detected or not): N_FRAUDS = 200 frauds,
     60 for foundation models, drawn with seed 7.
  2. Move each fraud to the lowest score it can reach within eps:
       differentiable models  APGD without early stopping (100 iterations x 5 restarts; 20 x 2 for foundation
                              models, APGD_BUDGET in blackwhite.py)
       tree ensembles         the exact margin-minimising MILP (Kantchelian et al. 2016, discrete domains,
                              KantchelianMILP.min_margin), 10 s per fraud, the incumbent on timeout; its point is
                              projected onto the threat model and kept only if it stays within eps
       every model            Square Attack without early stopping (300 queries; 100 for foundation models)
     and keep, per fraud, the lowest of its clean, white-box and Square scores.
  3. Legit transactions are not perturbed. AUC and KS of the attacked frauds against all test legits (foundation
     models: their fixed legit subset, fraudadv.hpo.train_context.fm_subset) are computed unit-weighted and
     dollar-weighted (frauds weighted by their amount, legits by weight x amount), as in the grid, together with
     recall at the deployed threshold.

Inputs (runs/<dataset>/attacks/): tau.json (deployed thresholds), sample.npz (the shared fraud sample S), the
trained models.
Outputs (in runs_consistent/<dataset>/attacks/ instead for the consistent attacker, which runs on its own):
  runs/<dataset>/attacks/robust_auc<tag>.csv     one row per (model, threat model, eps): clean_* and robust_* auc,
                                                 ks, auc_usd, ks_usd, recall, recall_usd
  runs/<dataset>/attacks/robust_scores<tag>.npz  the frauds' clean and attacked scores (keys <model>|rows,
                                                 <model>|clean, <model>|<mode>|<eps>), so that the metrics can be
                                                 recomputed without repeating the attacks
Both files are saved after every (model, threat model, eps), and a rerun skips the cells already in the CSV.

Results: Table S-XXX and Table S-XXXI (supplement), the ranking-metric numbers of section VI-K, and
Table S-XVIII for the consistent attacker (--modes consistent --eps 0.25: the paper reports it at eps 0.25 only).

usage: python -m fraudadv.attacks.robust_auc <dataset> fam1,fam2,... [--tag _fm] [--modes unconstrained,constrained]
                                         [--eps 0.1,0.25,0.5]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

N_FRAUDS = {"default": 200, "fm": 60}       # frauds attacked per model (foundation models are slow)
EPS_R = [0.1, 0.25, 0.5]                    # budgets
MILP_SECONDS = 10.0                         # time limit of one margin-minimising MILP solve (trees)


def run(dataset, families, tag="", modes=("unconstrained", "constrained"), eps=None):
    """Attack `families` of `dataset` with the score-minimising attacker in every threat model of `modes` and at
    every budget of `eps` (default EPS_R); returns the rows written to robust_auc<tag>.csv."""
    budgets = EPS_R if eps is None else list(eps)
    consistent = "consistent" in modes
    if consistent and len(modes) > 1:
        raise ValueError("run the consistent attacker on its own: its outputs go to runs_consistent/")
    if any(f.startswith(("chronos2", "timesfm")) for f in families):
        import chronos  # noqa: F401  before TensorFlow (ivis)
    from fraudadv.attacks.blackwhite import APGD_BUDGET, apgd, square
    from fraudadv.attacks.constraints import Constraints
    from fraudadv.attacks.milp import KantchelianMILP
    from fraudadv.attacks.run_attacks import FM, TREES, context_ids, load_model
    from fraudadv.data.load import load
    from fraudadv.eval.metrics import auc, ks
    from fraudadv.hpo.train_context import fm_subset
    from fraudadv.paths import attack_dir

    a, meta = load(dataset)
    grid = attack_dir(dataset)                           # the grid's thresholds and fraud sample
    tau = json.loads((grid / "tau.json").read_text())
    S = np.load(grid / "sample.npz")["S"]
    out = attack_dir(dataset, consistent)
    out.mkdir(parents=True, exist_ok=True)
    keep = context_ids(dataset)
    p = out / f"robust_auc{tag}.csv"
    rows = pd.read_csv(p).to_dict("records") if p.exists() else []
    have = {(r["model"], r["mode"], round(float(r["eps"]), 4)) for r in rows}         # cells already done
    sp = out / f"robust_scores{tag}.npz"
    scores = dict(np.load(sp)) if sp.exists() else {}
    # legit transactions the frauds are ranked against: all test legits, or the foundation models' fixed subset
    lg_all = np.flatnonzero(a["y_test"] == 0)
    it, wt = fm_subset(a, "test")
    wfull = np.zeros(len(a["y_test"]))
    wfull[it] = wt
    lg_fm = it[a["y_test"][it] == 0]
    for fam in families:
        if fam not in tau or all((fam, md, round(e, 4)) in have for md in modes for e in budgets):
            continue
        t0 = time.time()
        m = load_model(dataset, fam, 0)
        fm = fam in FM
        n = N_FRAUDS["fm" if fm else "default"]
        R = np.sort(np.random.default_rng(7).choice(S, size=min(n, len(S)), replace=False))     # attacked frauds
        X0, c0, amt = a["X_test"][R], keep["test"][R], a["amt_test"][R].astype(float)
        lg = lg_fm if fm else lg_all
        wl = wfull[lg] if fm else a["w_test"][lg]                     # legit weights
        al = a["amt_test"][lg].astype(float)                          # legit amounts
        ll = m.logit(a["X_test"][lg], keep["test"][lg])               # legit logits (never perturbed)
        l0 = m.logit(X0, c0)                                          # clean fraud logits

        def metrics(lf):
            """Ranking metrics and recall for fraud logits lf against the fixed legits."""
            y = np.r_[np.ones(len(lf)), np.zeros(len(ll))]
            s = np.r_[lf, ll]
            w = np.r_[np.ones(len(lf)), wl]                           # unit weights
            wd = np.r_[amt, wl * al]                                  # dollar weights
            return dict(auc=auc(y, s, w), ks=ks(y, s, w), auc_usd=auc(y, s, wd), ks_usd=ks(y, s, wd),
                        recall=float(np.mean(lf > tau[fam])), recall_usd=float(amt[lf > tau[fam]].sum() / amt.sum()))

        clean = metrics(l0)
        scores[f"{fam}|rows"], scores[f"{fam}|clean"] = R, l0
        for mode in modes:
            if all((fam, mode, round(e, 4)) in have for e in budgets):
                continue
            cons = Constraints(meta, a["X_train"], mode)
            M = None
            if fam in TREES:
                M = KantchelianMILP(m, fam, X0.shape[1], cons.lower, cons.upper, cons.mutable, time_limit=MILP_SECONDS,
                                    allowed=cons.allowed, cons=cons)
            for e in budgets:
                if (fam, mode, round(e, 4)) in have:
                    continue
                # white-box: lowest logit per fraud within e
                if fam in TREES:
                    lw = _milp_lowest_logits(m, M, cons, X0, l0, e)
                else:
                    _, lw = apgd(m, X0, tau[fam], e, cons, *APGD_BUDGET["fm" if fm else "default"], early_stop=False, ctx=c0)
                # black-box: Square without early stopping
                _, lsq = square(m, X0, tau[fam], e, cons, n_queries=100 if fm else 300, early_stop=False, ctx=c0)
                lf = np.minimum(np.minimum(lw, lsq), l0)
                scores[f"{fam}|{mode}|{e}"] = lf
                r = metrics(lf)
                rows.append(dict(dataset=dataset, model=fam, mode=mode, eps=e, n_frauds=len(R),
                                 **{f"clean_{k}": v for k, v in clean.items()}, **{f"robust_{k}": v for k, v in r.items()}))
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()
                       if k in ("model", "mode", "eps", "clean_auc", "robust_auc", "clean_ks", "robust_ks", "robust_recall")}, flush=True)
                pd.DataFrame(rows).to_csv(p, index=False)        # checkpoint: a rerun resumes after this (mode, eps)
                np.savez_compressed(sp, **scores)
        print(dataset, fam, "robust AUC done in", round(time.time() - t0), "s", flush=True)
    return rows


def _milp_lowest_logits(m, M, cons, X0, l0, e):
    """Tree ensembles: per fraud, the lowest logit of the margin-minimising MILP's point within e, after projecting
    it onto the threat model; the clean logit l0 where the MILP finds no point or the projection leaves the budget."""
    lw = l0.copy()
    for i in range(len(X0)):
        x, _, _, _ = M.min_margin(X0[i], e)            # (point, margin, dual bound, status)
        if x is not None:
            xp = cons.project(x[None], X0[i])
            if cons.linf(xp, X0[i][None])[0] <= e + 1e-5:
                lw[i] = min(lw[i], float(m.logit(xp)[0]))
    return lw


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ranking metrics under a score-minimising attacker (Table S-XXX).")
    ap.add_argument("dataset")
    ap.add_argument("families", help="comma-separated model families")
    ap.add_argument("--tag", default="", help="suffix of the output files")
    ap.add_argument("--modes", default="unconstrained,constrained", help="comma-separated threat models")
    ap.add_argument("--eps", default=",".join(str(e) for e in EPS_R), help="comma-separated L-inf budgets")
    args = ap.parse_args()
    run(args.dataset, args.families.split(","), args.tag, tuple(args.modes.split(",")), [float(e) for e in args.eps.split(",")])
