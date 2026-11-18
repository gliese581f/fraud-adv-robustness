"""Threat models: which features an attacker may change, and how, applied as a projection after every attack step.

    Constraints(meta, X_train, mode).project(X, X0, eps) maps candidate rows X onto the threat model around the
    original rows X0 (and inside the L-inf budget eps, if given).

mode="unconstrained"  every feature may move inside the data domain: its training range (widened to the row's own
                      value), and a discrete feature snaps to a valid value (a training level or the row's own value;
                      within eps of it when eps is given). Discrete means: every categorical (target-encoded, so its
                      training levels are its categories), every feature whose raw values are integers (codes,
                      counts, hour, flags), and every feature with at most DISCRETE_MAX_LEVELS training values.
                      Without snapping, a tree threshold sitting exactly on a discrete value would let a 1-ulp change
                      flip a split, which no real input can do. Derived features are recomputed from their sources.
mode="constrained"    as unconstrained, but only the attacker-controllable features listed in meta["mutable"] move
                      (the per-dataset schema, listed in the feature-schema appendix).
mode="consistent"     as constrained, then repaired so the row is one a real transaction could produce
                      (fraudadv.attacks.consistent). Same features and budget, so the consistent set lies inside the
                      constrained one.
mode="continuous"     every feature may move anywhere in its training range (plain feature-space L-inf); used only to
                      check AATS against the exact MILP, which is exact in this case (Table S-XXIII).

snap="legacy32" reproduces the study's first attack grid (compared with the final one in the rerun appendix), which snapped only
features with at most 32 distinct values.
"""
from __future__ import annotations

import numpy as np

DERIVED = {
    "log_ratio": {"log_ratio": ("log_amt", "log_amt_mean_prev")},
    "paysim": {"dest_log_ratio": ("log_amt", "dest_log_amt_mean_prev")},
}
PAYSIM_DERIVED = ["log_newbalanceOrig", "err_orig", "err_dest", "dest_log_ratio"]
DISCRETE_MAX_LEVELS = 512          # a feature with this few distinct training values is discrete


def discrete_features(meta: dict, X_train: np.ndarray, cols, snap: str = "all", max_levels: int = 32) -> dict:
    """{feature index: sorted valid values (float32)} for the discrete features among cols."""
    f = meta["features"]
    cats = set(meta.get("categorical", []))
    mu, sd = np.asarray(meta["mu"], np.float64), np.asarray(meta["sd"], np.float64)
    out = {}
    for j in cols:
        j = int(j)
        u = np.unique(X_train[:, j])
        if snap == "legacy32":
            disc = len(u) <= max_levels
        elif snap == "all":
            raw = u.astype(np.float64) * sd[j] + mu[j]
            integer = bool(np.all(np.abs(raw - np.round(raw)) <= np.maximum(2e-3, 1e-6 * np.abs(raw))))
            disc = f[j] in cats or integer or len(u) <= DISCRETE_MAX_LEVELS
        else:
            raise ValueError(snap)
        if disc:
            out[j] = u.astype(np.float32)
    return out


def _slog(e):
    return np.sign(e) * np.log1p(np.abs(e))


def _inv_slog(v):
    return np.sign(v) * np.expm1(np.abs(v))


class Constraints:
    def __init__(self, meta: dict, X_train: np.ndarray, mode: str = "constrained", max_levels: int = 32,
                 snap: str = "all"):
        if snap not in ("all", "legacy32"):
            raise ValueError(f"snap must be 'all' or 'legacy32', not {snap!r}")
        self.meta, self.mode, self.snap = meta, mode, snap
        # the consistent attacker is the constrained one plus a repair step at the end of project()
        self.repair = None
        if mode == "consistent":
            mode = "constrained"
        f = meta["features"]
        self.idx = {n: i for i, n in enumerate(f)}
        self.d = len(f)
        self.mu, self.sd = np.array(meta["mu"]), np.array(meta["sd"])
        self.lower = np.array(meta["lower"], np.float32)
        self.upper = np.array(meta["upper"], np.float32)
        self.hook = meta.get("recompute")
        derived = set(DERIVED.get(self.hook, {}) if self.hook else [])
        if self.hook == "paysim":
            derived |= set(PAYSIM_DERIVED)
        self.derived = [self.idx[n] for n in derived if n in self.idx]
        if mode == "continuous":
            self.mutable = np.arange(self.d)
        elif mode == "unconstrained":                    # derived features follow their sources (recomputed)
            self.mutable = np.array([j for j in range(self.d) if j not in self.derived])
        elif mode == "constrained":
            self.mutable = np.array([self.idx[n] for n in meta["mutable"] if n in self.idx and self.idx[n] not in self.derived])
        else:
            raise ValueError(mode)
        self.allowed = {} if mode == "continuous" else discrete_features(meta, X_train, self.mutable, snap, max_levels)
        self.mask = np.zeros(self.d, bool)
        self.mask[self.mutable] = True
        if self.mode == "consistent":
            from fraudadv.attacks.consistent import ConsistentRepair

            self.repair = ConsistentRepair(meta["name"], self, X_train)

    # unscaled <-> scaled for one column
    def _u(self, X, name):
        j = self.idx[name]
        return X[:, j] * self.sd[j] + self.mu[j]

    def _set(self, X, name, u):
        j = self.idx[name]
        X[:, j] = (u - self.mu[j]) / self.sd[j]

    def project(self, X: np.ndarray, X0: np.ndarray, eps: float | None = None) -> np.ndarray:
        """Map candidate points onto the threat model around X0. project(X0, X0) == X0 exactly.

        Bounds and discrete levels always admit the example's own value (an out-of-time test example
        can lie outside the training range or hold a level unseen in training). Derived features move
        by the change of their formula between X0 and X (delta form), so an unperturbed record keeps
        its recorded values (fills for missing history, PaySim's inconsistent balances) and a perturbed
        one changes exactly as its sources do.
        With eps (L-inf attacks), a discrete feature snaps only to levels within eps of its own value,
        so rounding never carries a point outside the budget.
        """
        X = np.array(X, dtype=np.float32, copy=True)
        X0 = np.broadcast_to(np.asarray(X0, np.float32), X.shape)
        lo, hi = np.minimum(self.lower, X0), np.maximum(self.upper, X0)
        if self.mode == "continuous":
            return np.clip(X, lo, hi)
        X[:, ~self.mask] = X0[:, ~self.mask]
        mu_ = self.mutable
        X[:, mu_] = np.clip(X[:, mu_], lo[:, mu_], hi[:, mu_])
        for j, a in self.allowed.items():
            x, x0 = X[:, j], X0[:, j]
            if len(a) > 1:
                k = np.clip(np.searchsorted(a, x), 1, len(a) - 1)
                cand = np.stack([x0, a[k - 1], a[k]], 1)        # x0 first: kept on ties
            else:
                cand = np.stack([x0, np.full(len(X), a[0], np.float32)], 1)
            dist = np.abs(cand - x[:, None])
            if eps is not None:
                dist = np.where(np.abs(cand - x0[:, None]) <= eps + 1e-6, dist, np.inf)
            X[:, j] = cand[np.arange(len(X)), np.argmin(dist, 1)]
        if self.hook in DERIVED:
            for out, (amt, mean) in DERIVED[self.hook].items():
                if out not in self.idx:
                    continue
                d = (self._u(X, amt) - self._u(X0, amt)) - (self._u(X, mean) - self._u(X0, mean))
                v = self._u(X0, out) + d
                if "isna_" + mean in self.idx:                    # no history: the ratio stays at its fill
                    v = np.where(self._u(X0, "isna_" + mean) > 0.5, self._u(X0, out), v)
                self._set_rows(X, out, v, d != 0)
        if self.hook == "paysim":
            amt, amt0 = np.expm1(self._u(X, "log_amt")), np.expm1(self._u(X0, "log_amt"))
            oo, oo0 = np.expm1(self._u(X, "log_oldbalanceOrg")), np.expm1(self._u(X0, "log_oldbalanceOrg"))
            od, od0 = np.expm1(self._u(X, "log_oldbalanceDest")), np.expm1(self._u(X0, "log_oldbalanceDest"))
            nd, nd0 = np.expm1(self._u(X, "log_newbalanceDest")), np.expm1(self._u(X0, "log_newbalanceDest"))
            no0 = np.expm1(self._u(X0, "log_newbalanceOrig"))
            no = np.maximum(no0 + np.maximum(oo - amt, 0.0) - np.maximum(oo0 - amt0, 0.0), 0.0)
            chg = (amt != amt0) | (oo != oo0) | (od != od0) | (nd != nd0)
            self._set_rows(X, "log_newbalanceOrig", np.log1p(no), chg)
            e = _inv_slog(self._u(X0, "err_orig")) + (no - no0) + (amt - amt0) - (oo - oo0)
            self._set_rows(X, "err_orig", _slog(e), chg)
            e = _inv_slog(self._u(X0, "err_dest")) + (od - od0) + (amt - amt0) - (nd - nd0)
            self._set_rows(X, "err_dest", _slog(e), chg)
        if self.repair is not None:
            X = self.repair(X, X0, eps)
        return X

    def _set_rows(self, X, name, u, rows):
        j = self.idx[name]
        X[:, j] = np.where(rows, ((u - self.mu[j]) / self.sd[j]).astype(np.float32), X[:, j])

    def linf(self, X, X0):
        return np.abs(X - X0)[:, self.mutable].max(1)
