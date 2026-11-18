"""Which feature rows a real transaction could produce: the validator behind the consistent attacker.

The constrained attacker moves each controllable feature on its own. Where the bank derives several features from
one record, that can give a row no transaction could produce. violations(ds, X) flags such rows (standardized feature
space, through the scaling in meta.json):

  every dataset   missing flag isna_<f> = 1 but <f> differs from the value filled in for missing rows
  PaySim          payee history: no earlier transfer but a time since the last one, a 24-hour count or a 24-hour
                  inflow; last transfer over 24 h ago but a 24-hour count or inflow; more transfers in 24 h than
                  ever; a 24-hour inflow without a 24-hour transfer
  Sparkov         dist differs from its definition hypot(lat - merch_lat, (long - merch_long) cos lat)
                  (fraudadv.data.prepare): the merchant moved but the distance did not, or the reverse; or the
                  merchant lies more than 1 degree from the cardholder (the simulator never does this)
  TabFormer       merch_seen = 1 for a card with no earlier transaction

StructuralZeros adds the combinations the hand rules miss: a pair of discrete values that never occurs in training
although independence predicts at least EMIN rows (on IEEE-CIS, dist1 exists only for product W, which never
carries device data). violations_all(ds, X) applies both.

Results: Table S-XIII counts the constrained evasions this validator flags, and the consistent
attacker's repair (fraudadv.attacks.consistent) uses StructuralZeros. Shared with the project's other papers;
identical in behaviour to decoydef/consistency.py (sha256 prefix 5996da32d6e1).
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from fraudadv.paths import PROC

SPARKOV_BOX = 1.0      # largest |merchant - cardholder| offset in degrees in the Sparkov generator
EMIN = 200             # a pair is a structural zero if independence predicts at least this many training rows


@lru_cache(maxsize=None)
def _meta(ds):
    m = json.loads((PROC / ds / "meta.json").read_text())
    return m, {f: i for i, f in enumerate(m["features"])}, np.array(m["mu"], float), np.array(m["sd"], float)


@lru_cache(maxsize=None)
def fills(ds):
    """Fill value of each feature among its rows flagged missing (legit training rows), and its spread there."""
    from fraudadv.data.load import load
    m, ix, mu, sd = _meta(ds)
    a, _ = load(ds)
    X = a["X_train"][a["y_train"] == 0]
    out = {}
    for f in m["features"]:
        if f.startswith("isna_") and f[5:] in ix:
            na = X[:, ix[f]] * sd[ix[f]] + mu[ix[f]] > 0.5
            if na.any():
                v = X[na, ix[f[5:]]]
                out[f[5:]] = (float(np.median(v)), float(np.max(np.abs(v - np.median(v)))))
    return out


def violations(ds, X, detail=False):
    """Rows breaking a hand rule (bool array); with detail, also the share of rows breaking each rule."""
    X = np.asarray(X, np.float32)
    m, ix, mu, sd = _meta(ds)
    raw = lambda f: X[:, ix[f]] * sd[ix[f]] + mu[ix[f]]          # noqa: E731  unstandardized column
    rules = {}
    for b, (fv, spread) in fills(ds).items():
        tol = max(1e-3, 2 * spread) * max(1.0, abs(fv))
        rules[f"isna_{b}"] = (raw("isna_" + b) > 0.5) & (np.abs(X[:, ix[b]] - fv) > tol)
    if ds == "paysim":
        cnt = np.expm1(raw("dest_cnt_prev"))
        dt_h = np.expm1(raw("dest_log_dt_prev")) / 3600.0
        c24 = np.expm1(raw("dest_cnt_24h"))
        s24 = raw("dest_logamt_sum_24h")
        fill_dt = float(np.expm1(fills(ds).get("dest_log_dt_prev", (np.nan, 0))[0] * sd[ix["dest_log_dt_prev"]]
                                 + mu[ix["dest_log_dt_prev"]]) / 3600.0) if "dest_log_dt_prev" in fills(ds) else 11.0
        none = cnt < 0.5
        rules["no_history_but_recency"] = none & (np.abs(dt_h - fill_dt) > 0.05)
        rules["no_history_but_24h"] = none & ((c24 > 0.5) | (s24 > 1e-3))
        rules["stale_but_24h"] = ~none & (dt_h > 24.01) & ((c24 > 0.5) | (s24 > 1e-3))
        rules["more_in_24h_than_ever"] = c24 > cnt + 0.5
        rules["inflow_without_transfer"] = (c24 < 0.5) & (s24 > 1e-3)
    if ds == "sparkov":
        d = np.hypot(raw("lat") - raw("merch_lat"), (raw("long") - raw("merch_long")) * np.cos(np.radians(raw("lat"))))
        rules["dist_not_from_coordinates"] = np.abs(raw("dist") - d) > 1e-3 + 1e-3 * np.abs(d)
        rules["merchant_outside_home_box"] = ((np.abs(raw("merch_lat") - raw("lat")) > SPARKOV_BOX + 1e-3)
                                              | (np.abs(raw("merch_long") - raw("long")) > SPARKOV_BOX + 1e-3))
    if ds == "tabformer":
        rules["merch_seen_without_history"] = (raw("cnt_prev") < 1e-3) & (raw("merch_seen") > 0.5)
    v = np.zeros(len(X), bool)
    for r in rules.values():
        v |= r
    return (v, {k: float(r.mean()) for k, r in rules.items()}) if detail else v


class StructuralZeros:
    """Pairs of discrete values (categoricals, flags, features with few levels) that never occur together in training
    although independence predicts at least emin rows. Only pairs involving an attacker-movable feature (or its
    missing flag) are kept."""

    def __init__(self, ds, X_train, emin=EMIN, max_levels=64):
        m, ix, _, _ = _meta(ds)
        f = m["features"]
        mut = set(m["mutable"]) if isinstance(m["mutable"], list) else set()
        movable = {i for i, n in enumerate(f) if n in mut or (n.startswith("isna_") and n[5:] in mut)}
        X = np.asarray(X_train, np.float32)
        self.disc, self.lev = [], {}
        for j in range(len(f)):
            u = np.unique(X[:, j])
            if len(u) <= max_levels:
                self.disc.append(j)
                self.lev[j] = u
        N = len(X)
        codes = {j: np.searchsorted(self.lev[j], X[:, j]) for j in self.disc}
        cnt = {j: np.bincount(codes[j], minlength=len(self.lev[j])) for j in self.disc}
        self.tabs = []                        # (feature i, feature j, bool table of forbidden level pairs)
        for a_ in range(len(self.disc)):
            for b_ in range(a_ + 1, len(self.disc)):
                i, j = self.disc[a_], self.disc[b_]
                if i not in movable and j not in movable:
                    continue
                ni, nj = len(self.lev[i]), len(self.lev[j])
                joint = np.bincount(codes[i] * nj + codes[j], minlength=ni * nj).reshape(ni, nj)
                exp = np.outer(cnt[i], cnt[j]) / N
                Z = (joint == 0) & (exp >= emin)
                if Z.any():
                    self.tabs.append((i, j, Z))
        self.names = f

    def _codes(self, X, j):
        u = self.lev[j]
        k = np.clip(np.searchsorted(u, X[:, j]), 0, len(u) - 1)
        return k, u[k] == X[:, j]             # level code, and whether the value is a training level

    def pairs_hit(self, X):
        """(rows, tables) bool: which structural-zero tables each row hits."""
        X = np.asarray(X, np.float32)
        out = np.zeros((len(X), len(self.tabs)), bool)
        cache = {}
        for t, (i, j, Z) in enumerate(self.tabs):
            for k in (i, j):
                if k not in cache:
                    cache[k] = self._codes(X, k)
            (ci, oi), (cj, oj) = cache[i], cache[j]
            out[:, t] = Z[ci, cj] & oi & oj
        return out

    def violations(self, X):
        return self.pairs_hit(X).any(1) if self.tabs else np.zeros(len(X), bool)

    def describe(self, X, top=8):
        """The pairs most often hit by the rows of X: (feature, feature, share of rows)."""
        H = self.pairs_hit(X)
        s = H.mean(0)
        o = np.argsort(-s)[:top]
        return [(self.names[self.tabs[t][0]], self.names[self.tabs[t][1]], float(s[t])) for t in o if s[t] > 0]


@lru_cache(maxsize=None)
def structural(ds):
    from fraudadv.data.load import load
    a, _ = load(ds)
    return StructuralZeros(ds, a["X_train"])


def violations_all(ds, X):
    """Hand rules plus structural zeros."""
    return violations(ds, X) | structural(ds).violations(X)
