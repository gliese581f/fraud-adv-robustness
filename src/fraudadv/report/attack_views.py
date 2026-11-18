"""Attack views: what an evasion looks like in feature space, how adversarial frauds differ
from other frauds, and how close they sit to the decision boundary. Tuned XGBoost and MLP bases (seed 0) on the
four datasets of the causal ablation (fraudadv.causal.ablate), both threat settings.

compute   python -m fraudadv.report.attack_views compute <dataset>
  reads the ablation outputs of the two bases (runs/<ds>/causal/out_{xgb,mlp}_seed0_<mode>.npz: the attacked
  frauds, their min-effort adversarial examples per attack and the attack costs) and adds
  - reverse HopSkipJump: the smallest L-inf change (same threat model, same algorithm) that makes a MISSED fraud
    detected and a passed LEGIT transaction flagged, so every group's distance to the boundary is measured alike
  - k-NN fraud share: share of frauds among the 50 nearest training transactions (standardized features) of every
    point, and a k-NN detector set to flag 1% of legit test traffic
  - one 2-D UMAP of legit test transactions, the attacked frauds and both models' adversarial examples
  - decision-function slices: for one fraud, each model's logit on a grid of two attacker-controllable features
    (all other features fixed), and the smallest evasion inside that slice
  writes runs/<ds>/causal/views.pkl
figures   python -m fraudadv.report.attack_views figures
  reports/fig_attack_views.{pdf,png} (Figure 2), fig_attack_map, fig_attack_slices, fig_attack_profile,
  fig_attack_distance (Figures S15 to S18), and CSVs in reports/attack_views/ (Table S-XXXVII in
  attack_examples*.csv, the numbers quoted with Figure 2 in attack_view_numbers.csv); the drawing is in
  fraudadv.report.attack_figures
slice     python -m fraudadv.report.attack_views slice [<dataset> ...]
  recomputes only the slice of stored views (the rest uses unseeded HopSkipJump, so it is not recomputed)

ATTACKS, BASES, MODES and VDS are imported by attack_figures.
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd

from fraudadv.attacks.run_attacks import RUNS, load_model

VDS = ["sparkov", "tabformer", "ieee_cis", "baf"]
BASES = {"xgb": "xgboost", "mlp": "mlp"}
MODES = ["constrained", "unconstrained"]
EPS_V = 0.25                                   # budget of the adversarial examples shown
ATTACKS = {"wb": "white-box", "sq": "Square", "hsj": "HopSkipJump"}
N_REV = 60                                     # missed frauds / legit transactions per reverse HopSkipJump


class Neg:
    """The model with its score negated: evading it means being flagged by the original."""

    def __init__(self, m):
        self.m = m

    def logit(self, X, ctx=None):
        return -self.m.logit(X, ctx)


def adv_points(z, att, eps=EPS_V):
    """(rows into the fraud subset, adversarial points) of the successful eps-budget evasions of detected frauds."""
    tau, det = float(z["tau"]), z["det"]
    if att == "wb":
        pgd_ = np.isnan(z["wb_cost"]).all()                    # APGD stores its eps-0.25 points; AATS its cheapest one
        ok = det & ((z[f"wb_l_{eps}"] < tau) if pgd_ else (z["wb_cost"] <= eps + 1e-6))
        return np.flatnonzero(ok), z["wb_X"][ok]
    if att == "sq":
        ok = det & (z[f"sq_l_{eps}"] < tau)
        return np.flatnonzero(ok), z["sq_X"][ok]
    H = z["hsj_rows"]
    ok = det[H] & (z["hsj_cost"] <= eps + 1e-6)
    return H[ok], z["hsj_X"][ok]


def _slice(m, cons, x0, j, k, tau, lo, hi, n=121):
    """Logit of m on the (j, k) grid through x0 (other features fixed, threat-model projection applied)."""
    gj, gk = np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)
    G = np.repeat(x0[None], n * n, 0)
    G[:, j] = np.repeat(gj, n)
    G[:, k] = np.tile(gk, n)
    G = cons.project(G, x0)
    L = m.logit(G).reshape(n, n) - tau
    # the smallest in-slice evasion (L-inf in standardized units, over the two features)
    D = np.maximum(np.abs(G[:, j] - x0[j]), np.abs(G[:, k] - x0[k])).reshape(n, n)
    D = np.where(L < 0, D, np.inf)
    i = np.unravel_index(np.argmin(D), D.shape)
    best = (G.reshape(n, n, -1)[i][[j, k]], float(D[i])) if np.isfinite(D[i]) else (None, np.inf)
    return dict(gj=G.reshape(n, n, -1)[:, 0, j], gk=G.reshape(n, n, -1)[0, :, k], L=L.astype(np.float32), best=best)


def slice_view(R, models, c, feats):
    """Decision-function slice through one fraud both models detect and both white-box attacks evade (constrained c)."""
    X0 = R["X0"]
    ok_x = set(R["xgb_constrained_wb_rows"].tolist())
    both = [i for i in R["mlp_constrained_wb_rows"].tolist() if i in ok_x]
    cand = both or R["xgb_constrained_wb_rows"].tolist() or list(range(len(X0)))
    # prefer a fraud whose clean score is well above both thresholds (a confidently detected fraud)
    depth = {i: min(R["xgb_l0"][i] - R["xgb_tau"], R["mlp_l0"][i] - R["mlp_tau"]) for i in cand}
    i0 = max(cand, key=lambda i: depth[i])
    dX = np.zeros(len(feats))
    for v in BASES:
        rr, XX = R[f"{v}_constrained_wb_rows"], R[f"{v}_constrained_wb_X"]
        if i0 in rr:
            dX += np.abs(XX[list(rr).index(i0)] - X0[i0])
    mut = [j for j in c.mutable if j not in c.derived]
    cont = [j for j in mut if j not in c.allowed or len(c.allowed[j]) > 12]      # continuous or many levels
    order = sorted(cont, key=lambda j: -dX[j])
    # amount and hour where the attacker controls them (the plane readers know), else the most-changed features
    pref = lambda names: [feats.index(f) for f in names if f in feats and feats.index(f) in cont]
    j = (pref(("log_amt",)) or order)[0]
    k = next(q for q in pref(("hour",)) + order + cont if q != j)
    x0 = X0[i0]
    full = lambda q: q in c.allowed or (c.upper[q] - c.lower[q]) <= 5.0      # discrete or short range: whole range
    lo = np.array([c.lower[q] if full(q) else max(c.lower[q], x0[q] - 2.0) for q in (j, k)])
    hi = np.array([c.upper[q] if full(q) else min(c.upper[q], x0[q] + 2.0) for q in (j, k)])
    return dict(i0=i0, j=j, k=k, x0=x0, lo=lo, hi=hi,
                **{v: _slice(m, c, x0, j, k, R[f"{v}_tau"], lo, hi) for v, m in models.items()})


def compute(ds):
    """Compute every view of dataset ds (see the module docstring) and write runs/<ds>/causal/views.pkl."""
    from sklearn.neighbors import NearestNeighbors

    from fraudadv.attacks.blackwhite import hsj
    from fraudadv.attacks.constraints import Constraints
    from fraudadv.data.load import load

    a, meta = load(ds)
    feats = list(meta["features"])
    out = RUNS / ds / "causal"
    rng = np.random.default_rng(5)
    Z = {(v, md): dict(np.load(out / f"out_{v}_seed0_{md}.npz")) for v in BASES for md in MODES}
    cons = {md: Constraints(meta, a["X_train"], md) for md in MODES}
    models = {v: load_model(ds, f, 0) for v, f in BASES.items()}
    X0 = Z[("xgb", "constrained")]["X0"]
    lg_rows = rng.choice(np.flatnonzero(a["y_test"] == 0), size=3000, replace=False)
    XL = a["X_test"][lg_rows].astype(np.float32)
    R = dict(dataset=ds, features=feats, mutable=list(meta["mutable"]), mu=np.array(meta["mu"]), sd=np.array(meta["sd"]),
             categorical=list(meta["categorical"]), X0=X0, S=Z[("xgb", "constrained")]["S"], legit_rows=lg_rows)
    # ---- margins: legit-quantile of every clean score, per model
    for v, m in models.items():
        z = Z[(v, "constrained")]
        tau = float(z["tau"])
        ll = np.sort(m.logit(a["X_test"][a["y_test"] == 0]))
        R[f"{v}_tau"], R[f"{v}_l0"], R[f"{v}_lL"] = tau, z["l0"], m.logit(XL)
        R[f"{v}_legit_logit_quantiles"] = np.quantile(ll, np.linspace(0, 1, 2001))
    # ---- reverse HopSkipJump: distance for missed frauds to be caught and for passed legit to be flagged
    for v, m in models.items():
        tau = R[f"{v}_tau"]
        miss = np.flatnonzero(~Z[(v, "constrained")]["det"])[:N_REV]
        passed = np.flatnonzero(R[f"{v}_lL"] <= tau)[:N_REV]
        for md in MODES:
            c = cons[md]
            for g, rows, Xg in (("missed", miss, X0[miss]), ("legit", passed, XL[passed])):
                if not len(rows):
                    R[f"{v}_{md}_rev_{g}"] = (rows, np.zeros(0), np.zeros((0, X0.shape[1]), np.float32))
                    continue
                Xh, lh = hsj(Neg(m), Xg, -tau, c)
                d = np.where(m.logit(Xh) > tau, c.linf(Xh, Xg), np.inf)        # success: the original flags it
                R[f"{v}_{md}_rev_{g}"] = (rows, d.astype(np.float32), Xh.astype(np.float32))
                print(ds, v, md, g, "reverse HSJ: median distance", float(np.median(d)), flush=True)
    # ---- k-NN fraud share in the training data
    nn_ = NearestNeighbors(n_neighbors=50, n_jobs=-1).fit(a["X_train"].astype(np.float32))
    ytr = a["y_train"]

    def share(X):
        if not len(X):
            return np.zeros(0, np.float32)
        _, I = nn_.kneighbors(np.asarray(X, np.float32))
        return ytr[I].mean(1).astype(np.float32)

    R["knn_legit"] = share(XL)
    R["knn_fraud"] = share(X0)
    R["knn_thr"] = float(np.quantile(R["knn_legit"], 0.99))
    for v in BASES:
        for md in MODES:
            for att in ATTACKS:
                rows, Xa = adv_points(Z[(v, md)], att)
                R[f"{v}_{md}_{att}_rows"], R[f"{v}_{md}_{att}_X"] = rows, Xa
                R[f"{v}_{md}_{att}_knn"] = share(Xa)
    print(ds, "kNN shares: legit", R["knn_legit"].mean(), "frauds", R["knn_fraud"].mean(), flush=True)
    # ---- UMAP of legit, frauds and white-box adversarial examples of both models (both settings)
    import umap

    blocks = [("legit", XL[:2000]), ("fraud", X0)]
    for v in BASES:
        for md in MODES:
            blocks.append((f"adv_{v}_{md}", R[f"{v}_{md}_wb_X"]))
    Xu = np.vstack([b for _, b in blocks])
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(Xu)
    pos, U = 0, {}
    for name, b in blocks:
        U[name] = emb[pos:pos + len(b)]
        pos += len(b)
    R["umap"] = U
    R["slice"] = slice_view(R, models, cons["constrained"], feats)
    (out / "views.pkl").write_bytes(pickle.dumps(R))
    sl = R["slice"]
    print(ds, "views written; slice fraud", sl["i0"], "features", feats[sl["j"]], feats[sl["k"]], flush=True)



def redo_slice(ds):
    """Recompute only the decision-function slice in runs/<ds>/causal/views.pkl."""
    from fraudadv.attacks.constraints import Constraints
    from fraudadv.data.load import load

    a, meta = load(ds)
    p = RUNS / ds / "causal" / "views.pkl"
    R = pickle.loads(p.read_bytes())
    models = {v: load_model(ds, f, 0) for v, f in BASES.items()}
    R["slice"] = slice_view(R, models, Constraints(meta, a["X_train"], "constrained"), list(meta["features"]))
    p.write_bytes(pickle.dumps(R))
    sl = R["slice"]
    print(ds, "slice fraud", sl["i0"], "features", R["features"][sl["j"]], R["features"][sl["k"]], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "compute":
        for d in sys.argv[2:] or VDS:
            compute(d)
    elif sys.argv[1] == "slice":      # recompute only the slice of stored views (the rest uses unseeded HopSkipJump)
        for d in sys.argv[2:] or VDS:
            redo_slice(d)
    else:
        from fraudadv.report.attack_figures import main

        main()
