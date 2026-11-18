"""Per-transaction attributions behind the SHAP figures.

explain/attributions.py keeps only averages (global_importance.csv, attack_shift.csv). This module stores the
per-transaction values behind them, for the families whose attributions are cheap, on exactly the same rows:
  test rows     the 1,000 test transactions of attributions.run (up to 500 frauds first, then legits; seed 0)
  attack pairs  per threat setting, the first 200 detected frauds whose high-confidence adversarial example from
                the grid (eps = 0.5) evades the model, with that adversarial example

Methods: XGBoost and CatBoost exact TreeSHAP, with the base value (expected margin); MLP Integrated Gradients (the
same 32-step path as attributions.py), whose "base value" is the logit at the all-zeros baseline.

Outputs
  runs/<dataset>/explain/shap_<family>.npz
      idx, X, y, phi, base, features, mutable, method, and per threat setting
      sel_<mode>, X0_<mode>, Xadv_<mode>, phi0_<mode>, phi1_<mode>
  reports/shap/<dataset>_<family>_test.csv              test_row, is_fraud, shap_<feature>...
  reports/shap/<dataset>_<family>_attack_<mode>.csv     sample_row, shap_clean_<feature>, shap_adv_<feature>...

Used by report/shap_figures.py: Figures S8 to S11 (Figure S8, Figure S9, Figure S10, Figure S11).
Reads runs/<dataset>/explain/global_importance.csv (from attributions.py) only to print a consistency check.

usage: python -m fraudadv.explain.shap_values [dataset ...]      (default: every dataset of the study)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from fraudadv.attacks.run_attacks import RUNS, context_ids, load_model
from fraudadv.data.load import load
from fraudadv.explain.attributions import TREES, attribute, explained_rows
from fraudadv.report.make_report import DS, REP

FAMS = ["xgboost", "catboost", "mlp"]
MODES = ("unconstrained", "constrained")


def rows(y, n=1000):
    """The rows of attributions.run, drawn the same way."""
    return explained_rows(y, n)


def base_value(m, fam, X):
    """The attribution base value: the trees' expected margin, or the MLP's logit at the all-zeros baseline."""
    if fam == "xgboost":
        import xgboost as xgb

        b = m.model.get_booster()
        return float(b.predict(xgb.DMatrix(X[:1]), pred_contribs=True, iteration_range=(0, m.model.best_iteration + 1))[0, -1])
    if fam == "catboost":
        from catboost import Pool

        return float(m.model.get_feature_importance(Pool(X[:1]), type="ShapValues")[0, -1])
    return float(m.logit(np.zeros((1, X.shape[1]), np.float32), None)[0])


def check_against_global(dataset, out, fam, feats, phi):
    """Print the correlation of mean |phi| with the stored global importance (same rows, same method: about 1)."""
    G = pd.read_csv(out / "global_importance.csv", index_col=0)
    if fam in G:
        r = np.corrcoef(np.abs(phi).mean(0), G[fam].reindex(feats).to_numpy())[0, 1]
        print(dataset, fam, f"mean |phi| vs global_importance.csv: r = {r:.4f}", flush=True)


def attack_pairs(dataset, m, fam, feats, ctx_test, mode):
    """Attributions of the evading frauds of one threat setting and of their adversarial versions.
    Returns (arrays for the npz, the per-pair table) or None when there is nothing to explain."""
    p = RUNS / dataset / "attacks" / f"adv_{mode}_{fam}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    sel = np.flatnonzero(z["ok"] & z["det"])[:200]
    if not len(sel):
        return None
    ctx = ctx_test[z["S"][sel]]
    X0, X1 = z["X0"][sel].astype(np.float32), z["Xadv"][sel].astype(np.float32)
    p0, p1 = attribute(m, fam, X0, ctx), attribute(m, fam, X1, ctx)
    arrays = {f"sel_{mode}": z["S"][sel], f"X0_{mode}": X0, f"Xadv_{mode}": X1,
              f"phi0_{mode}": p0.astype(np.float32), f"phi1_{mode}": p1.astype(np.float32)}
    E = pd.DataFrame({"sample_row": z["S"][sel]})
    for j, f in enumerate(feats):
        E[f"shap_clean_{f}"], E[f"shap_adv_{f}"] = p0[:, j], p1[:, j]
    return arrays, E


def run(dataset):
    """Write shap_<family>.npz and the reports/shap CSVs of every family in FAMS that has a seed-0 model."""
    a, meta = load(dataset)
    feats, mut = list(meta["features"]), set(meta["mutable"])
    ctx_ids = context_ids(dataset)
    y = a["y_test"]
    idx = rows(y)
    X = a["X_test"][idx].astype(np.float32)
    out = RUNS / dataset / "explain"
    csv = REP / "shap"
    csv.mkdir(parents=True, exist_ok=True)
    for fam in FAMS:
        if not (RUNS / dataset / fam / "seed0.pkl").exists():
            continue
        m = load_model(dataset, fam, 0)
        phi = attribute(m, fam, X, ctx_ids["test"][idx])
        rec = dict(idx=idx, X=X, y=y[idx], phi=phi.astype(np.float32), base=base_value(m, fam, X),
                   features=np.array(feats), mutable=np.array([f in mut for f in feats]),
                   method=np.array("TreeSHAP" if fam in TREES else "Integrated Gradients"))
        check_against_global(dataset, out, fam, feats, phi)
        D = pd.DataFrame(phi, columns=[f"shap_{f}" for f in feats])
        D.insert(0, "is_fraud", y[idx].astype(int))
        D.insert(0, "test_row", idx)
        D.to_csv(csv / f"{dataset}_{fam}_test.csv", index=False, float_format="%.5g")
        for mode in MODES:
            pairs = attack_pairs(dataset, m, fam, feats, ctx_ids["test"], mode)
            if pairs is None:
                continue
            arrays, E = pairs
            rec.update(arrays)
            E.to_csv(csv / f"{dataset}_{fam}_attack_{mode}.csv", index=False, float_format="%.5g")
            print(dataset, fam, mode, "attack pairs:", len(E), flush=True)
        np.savez_compressed(out / f"shap_{fam}.npz", **rec)
        print(dataset, fam, "done", flush=True)


if __name__ == "__main__":
    for d in sys.argv[1:] or DS:
        run(d)
