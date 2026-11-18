"""The exact MILP (attacks/milp.py) with an affine derived feature.

On a small synthetic XGBoost model whose features are log_amt, log_amt_mean_prev, log_ratio (recomputed as
log_amt - log_amt_mean_prev) and one other feature, the MILP that models the recomputation (cons=...) finds an
evasion at least as close as brute-force enumeration of threshold crossings, and the MILP without it (derived
features frozen) misses some evasions that go through the derived feature."""
import itertools

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from fraudadv.attacks.constraints import Constraints
from fraudadv.attacks.milp import KantchelianMILP


class _M:
    """The two things KantchelianMILP needs from a study model: .model and .tree_ensemble()."""

    def __init__(self, model):
        self.model = model

    def tree_ensemble(self):
        from fraudadv.attacks.trees import TreeEnsemble
        return TreeEnsemble.from_xgboost(self.model, iteration_range=(0, self.model.best_iteration + 1))


def _setup(seed=0, n=4000):
    """A model, its constrained threat model (log_amt and other mutable, log_ratio recomputed) and the data."""
    rng = np.random.default_rng(seed)
    amt_u = rng.normal(4, 1.2, n)
    mean_u = amt_u - rng.normal(0, 0.8, n)
    ratio_u = amt_u - mean_u
    other = rng.normal(0, 1, n)
    raw = np.c_[amt_u, mean_u, ratio_u, other]
    mu, sd = raw.mean(0), raw.std(0)
    X = ((raw - mu) / sd).astype(np.float32)
    y = ((ratio_u > 0.6) & (other > -0.3) | (amt_u > 6.0)).astype(int)
    y = np.where(rng.random(n) < 0.03, 1 - y, y)
    m = xgb.XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.3, early_stopping_rounds=5,
                          eval_metric="logloss")
    m.fit(X[:3000], y[:3000], eval_set=[(X[3000:], y[3000:])], verbose=False)
    meta = {"features": ["log_amt", "log_amt_mean_prev", "log_ratio", "other"], "mu": mu.tolist(), "sd": sd.tolist(),
            "lower": X.min(0).tolist(), "upper": X.max(0).tolist(), "recompute": "log_ratio",
            "mutable": ["log_amt", "log_ratio", "other"], "categorical": []}
    cons = Constraints(meta, X, "constrained")
    return m, cons, X


def _brute(margin, cons, ens, x0, tau, r_max=1.0):
    """Exact minimal L-inf evasion over the two mutable features by enumeration. Candidate values are x0's own
    value and the values just past each threshold of the feature itself, plus, for log_amt, the values just past
    each log_ratio threshold mapped back through the recomputation."""
    ja, jo, jr = cons.idx["log_amt"], cons.idx["other"], cons.idx["log_ratio"]
    c = cons.sd[ja] / cons.sd[jr]
    cand = {ja: [x0[ja]], jo: [x0[jo]]}
    for j in (ja, jo):
        for t in ens.thresholds[j]:
            cand[j] += [np.float32(t), np.nextafter(np.float32(t), np.float32(-np.inf))]
    for t in ens.thresholds[jr]:
        a = x0[ja] + (float(t) - float(x0[jr])) / c
        for v in np.float32(a) + np.float32(1e-4) * np.arange(-3, 4):
            cand[ja].append(np.float32(v))
    best = np.inf
    for va, vo in itertools.product(cand[ja], cand[jo]):
        d = max(abs(float(va) - float(x0[ja])), abs(float(vo) - float(x0[jo])))
        if d >= best or d > r_max:
            continue
        x = x0.copy()
        x[ja], x[jo] = va, vo
        x = cons.project(x[None], x0)[0]
        if margin(x[None])[0] < tau:
            best = d
    return best


def test_milp_matches_bruteforce_with_derived_feature():
    m, cons, X = _setup()
    mm = _M(m)
    ens = mm.tree_ensemble()
    tau = 0.0
    M = KantchelianMILP(mm, "xgboost", 4, cons.lower, cons.upper, cons.mutable, time_limit=30, cons=cons)
    M0 = KantchelianMILP(mm, "xgboost", 4, cons.lower, cons.upper, cons.mutable, time_limit=30)
    assert M.exact_domain and M.linear_derived
    det = np.flatnonzero(ens.margin_fn(X[3000:]) > tau)[:25] + 3000
    n_better = 0
    for i in det:
        x0 = X[i]
        bf = _brute(ens.margin_fn, cons, ens, x0, tau)
        x, b, lb, st = M.solve_one(x0, tau, eps_max=1.0)
        d = np.inf if x is None else float(np.abs(x - x0)[cons.mutable].max())
        if x is not None:
            assert ens.margin_fn(x[None])[0] < tau                      # verified evasion after recomputation
        if np.isfinite(bf):
            assert np.isfinite(d) and d <= bf + 2e-4, (i, d, bf)          # never worse than enumeration
        else:
            assert st == 2 or x is None                                  # certified robust within 1
        x_, *_ = M0.solve_one(x0, tau, eps_max=1.0)
        d0 = np.inf
        if x_ is not None:
            xp = cons.project(x_[None], x0)[0]
            if ens.margin_fn(xp[None])[0] < tau:
                d0 = float(np.abs(xp - x0)[cons.mutable].max())
        n_better += d < d0 - 1e-6
    assert n_better > 0                                                  # the coupling finds evasions the frozen MILP misses
