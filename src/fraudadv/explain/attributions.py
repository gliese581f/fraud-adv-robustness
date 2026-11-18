"""Feature attributions: what drives each model, how an attack changes that, and whether models agree.

Attribution method per model
  trees (XGBoost, CatBoost)   exact TreeSHAP (XGBoost pred_contribs, CatBoost ShapValues)
  every other model           Integrated Gradients (Sundararajan et al. 2017): 32-step midpoint Riemann sum (16 for
                              the foundation models), baseline = the training mean, which is 0 in standardized
                              space. Covers MLP, FT-Transformer, ivis, both graph models and both foundation models
                              through model.grad.

Rows explained (seed-0 models)
  global importance   1,000 test transactions drawn with seed 0: up to 500 frauds, listed first, then legits.
                      Foundation models use the first 200 of them (frauds).
  attack shift        per threat setting, the first 200 detected frauds (40 for foundation models) whose
                      high-confidence adversarial example from the grid (eps = 0.5, adv_<mode>_<family>.npz) evades.

Outputs (runs/<dataset>/explain/)
  global_importance.csv   mean |attribution| per feature (rows) and model (columns)
  attack_shift.csv        per model, threat setting and feature: mean |delta x| and mean |delta attribution| between
                          each fraud and its adversarial version, mean clean and adversarial attribution, n
  agreement.csv           Kendall tau between the models' global importance vectors

Used by: report/make_report.py: explain_summary (reports/table5_explain_*.csv, Section VI-H, RQ4), Table S-XXXII
(Table S-XXXII), Table S-XXXIII, Figure S12. explain/shap_values.py reuses `attribute` and
the same rows for Figures S8 to S11.

usage: python -m fraudadv.explain.attributions <dataset> [fam1,fam2,...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from fraudadv.attacks.run_attacks import FM, RUNS, context_ids, load_model
from fraudadv.data.load import load

TREES = {"xgboost", "catboost"}
MODES = ("unconstrained", "constrained")


def tree_shap(m, X):
    """Exact TreeSHAP values, shape (n, features); the bias column the libraries append is dropped."""
    if m.family == "XGBoost":
        import xgboost as xgb

        b = m.model.get_booster()
        c = b.predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=(0, m.model.best_iteration + 1))
        return c[:, :-1]
    from catboost import Pool

    return m.model.get_feature_importance(Pool(X), type="ShapValues")[:, :-1]


def integrated_gradients(m, X, ctx, steps=32, batch=2048):
    """Integrated Gradients from the all-zeros baseline: (x - 0) * mean of the logit gradient at `steps` midpoints
    of the straight path from the baseline to x. Gradients are taken in batches of `batch` rows."""
    X = X.astype(np.float32)
    base = np.zeros_like(X)
    total = np.zeros_like(X)
    alphas = (np.arange(steps) + 0.5) / steps
    for alpha in alphas:
        Z = base + alpha * (X - base)
        g = np.concatenate([m.grad(Z[i:i + batch], None if ctx is None else ctx[i:i + batch]) for i in range(0, len(Z), batch)])
        total += g
    return (X - base) * total / steps


def attribute(m, fam, X, ctx):
    """Attributions of model m (family fam) on rows X with context ids ctx: TreeSHAP for trees, else IG."""
    return tree_shap(m, X) if fam in TREES else integrated_gradients(m, X, ctx, steps=16 if fam in FM else 32)


def explained_rows(y, n=1000):
    """Test positions explained for global importance: up to n // 2 random frauds, then random legits up to n (seed 0)."""
    rng = np.random.default_rng(0)
    fr, lg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_fraud = min(n // 2, len(fr))
    return np.r_[rng.choice(fr, size=n_fraud, replace=False), rng.choice(lg, size=n - n_fraud, replace=False)]


def attack_shift_rows(dataset, m, fam, feats, ctx_test):
    """attack_shift.csv rows of one model: how attributions move from each evading fraud to its adversarial version."""
    rows = []
    for mode in MODES:
        p = RUNS / dataset / "attacks" / f"adv_{mode}_{fam}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        sel = np.flatnonzero(z["ok"] & z["det"])[: (40 if fam in FM else 200)]   # detected and evaded at eps 0.5
        if not len(sel):
            continue
        ctx = ctx_test[z["S"][sel]]
        A0 = attribute(m, fam, z["X0"][sel], ctx)
        A1 = attribute(m, fam, z["Xadv"][sel], ctx)
        dx = np.abs(z["Xadv"][sel] - z["X0"][sel]).mean(0)
        da = np.abs(A1 - A0).mean(0)
        for j, f in enumerate(feats):
            rows.append(dict(model=fam, mode=mode, feature=f, mean_abs_dx=float(dx[j]), mean_abs_dattr=float(da[j]),
                             clean_attr=float(A0[:, j].mean()), adv_attr=float(A1[:, j].mean()), n=len(sel)))
    return rows


def run(dataset, families, n=1000):
    """Write global_importance.csv, attack_shift.csv and agreement.csv for `families` on `dataset`."""
    if any(f in ("chronos2", "timesfm") for f in families):
        import chronos  # noqa: F401  must be imported before TensorFlow (ivis): the reverse order aborts on an LLVM option clash
    a, meta = load(dataset)
    feats = meta["features"]
    ctx_ids = context_ids(dataset)
    out = RUNS / dataset / "explain"
    out.mkdir(parents=True, exist_ok=True)
    idx = explained_rows(a["y_test"], n)
    importance, shifts = {}, []
    for fam in families:
        m = load_model(dataset, fam, 0)
        rows = idx[:200] if fam in FM else idx
        A = attribute(m, fam, a["X_test"][rows], ctx_ids["test"][rows])
        importance[fam] = np.abs(A).mean(0)
        shifts += attack_shift_rows(dataset, m, fam, feats, ctx_ids["test"])
        print(dataset, fam, "explained", flush=True)
    G = pd.DataFrame(importance, index=feats)
    G.to_csv(out / "global_importance.csv")
    pd.DataFrame(shifts).to_csv(out / "attack_shift.csv", index=False)
    fams = list(importance)
    agree = pd.DataFrame(np.eye(len(fams)), index=fams, columns=fams)
    for i in fams:
        for j in fams:
            agree.loc[i, j] = kendalltau(G[i], G[j]).statistic
    agree.to_csv(out / "agreement.csv")


if __name__ == "__main__":
    ds = sys.argv[1]
    fams = sys.argv[2].split(",") if len(sys.argv) > 2 else ["xgboost", "catboost", "mlp", "ft_transformer", "ivis"]
    run(ds, fams)
