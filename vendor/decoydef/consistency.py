"""Feature-consistency rules: combinations real transactions cannot produce (2026-11-17).

Most tripwire catches on PaySim and IEEE-CIS turned out to be queries that break the data's own bookkeeping: a value
written into a feature whose missing flag stays on, or a payee history whose parts contradict each other. Real
traffic never does this, because the bank derives these features from one record. A validator that rejects such
queries costs no legitimate transactions, so it is the control that tripwires must beat, and a consistent attacker
(who only sends combinations that could exist) is the threat model they must survive.

violations(ds, X) -> bool array; rules are checked in standardized feature space via meta mu/sd.
  every dataset   missing flag isna_<f> = 1 but <f> differs from the fill value used for missing rows
  paysim          payee history: no earlier transfer but a last-transfer time, a 24-h count or a 24-h inflow;
                  last transfer over 24 h ago but a 24-h count or inflow; more transfers in 24 h than ever;
                  24-h inflow without a 24-h transfer
  sparkov         dist differs from its definition hypot(lat - merch_lat, (long - merch_long) cos(lat))
                  (fraudadv/data/prepare.py), i.e. the merchant moved but the distance did not, or the reverse
  tabformer       merch_seen = 1 for a card with no earlier transaction
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

PROC = Path(os.environ.get("FRAUDADV_PROC", "/home/user/study/data/processed"))
SPARKOV_BOX = 1.0                                # max |merchant - cardholder| offset in degrees (Sparkov generator)


@lru_cache(maxsize=None)
def _meta(ds):
    m = json.loads((PROC / ds / "meta.json").read_text())
    return m, {f: i for i, f in enumerate(m["features"])}, np.array(m["mu"], float), np.array(m["sd"], float)


@lru_cache(maxsize=None)
def fills(ds):
    """Fill value of each base feature among rows flagged missing (legit training rows; checked constant)."""
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
    X = np.asarray(X, np.float32)
    m, ix, mu, sd = _meta(ds)
    raw = lambda f: X[:, ix[f]] * sd[ix[f]] + mu[ix[f]]          # noqa: E731
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
        # the simulator places every merchant within 1 degree of the cardholder (training: |offset| <= 1.000 exactly)
        rules["merchant_outside_home_box"] = ((np.abs(raw("merch_lat") - raw("lat")) > SPARKOV_BOX + 1e-3)
                                              | (np.abs(raw("merch_long") - raw("long")) > SPARKOV_BOX + 1e-3))
    if ds == "tabformer":
        rules["merch_seen_without_history"] = (raw("cnt_prev") < 1e-3) & (raw("merch_seen") > 0.5)
    v = np.zeros(len(X), bool)
    for r in rules.values():
        v |= r
    return (v, {k: float(r.mean()) for k, r in rules.items()}) if detail else v


# ---- structural zeros (2026-11-17) -----------------------------------------------------------------------------
# Hand rules miss combinations the bank's pipeline never generates, e.g. on IEEE-CIS dist1 exists only for product W,
# which never carries device information (0 of 116,000 rows with a known device type have dist1). A pair of discrete
# values (categoricals, flags, small-level features) is a structural zero when it never occurs in training although
# independence would predict at least EMIN rows. Only pairs that involve an attacker-movable feature (or the missing
# flag of one) are kept. A query holding a structural zero is impossible in the same sense as the hand rules.
EMIN = 200                                       # legit val 0.002%, test 0.13% on IEEE-CIS; 0 elsewhere


class StructuralZeros:
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
        self.tabs = []
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
        return k, u[k] == X[:, j]                                       # code, value is a training level

    def pairs_hit(self, X):
        """(n, n_tabs) bool: which structural-zero tables each row hits."""
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
