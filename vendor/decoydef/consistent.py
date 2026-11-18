"""Consistent attacker: every query is a combination a real transaction could produce (2026-11-17).

The study's constrained threat model lets the attacker move each mutable feature on its own. On PaySim the payee-
history features (earlier transfers, their mean amount, time since the last one, 24-h count and inflow, and their
missing flags) then contradict each other, and on IEEE-CIS a value is written into a feature whose missing flag stays
on. Real traffic cannot do either: the bank derives these features from one record. This wrapper adds a repair step
after the study's projection.

NESTED (2026-11-17 20:45Z): the consistent attacker is the constrained attacker restricted to realizable records, so
every repaired query stays inside the constrained set (same mutable features, same budget eps, immutable features and
missing flags untouched). Hence unconstrained >= constrained >= consistent for any fixed search. The first version let
PaySim's snap copy a real payee's missing flags (a no-history payee could become one with history, 2.6 sd, unbudgeted;
at eps 1 consistent beat constrained by up to 24 points, found by the RL thread) and let IEEE-CIS flags turn off when
a value changed; both levers are gone. Switching a missing flag is left to the unconstrained attacker.

  PaySim    the payee block snaps to the nearest payee history that occurs in the training data (the attacker picks
            a real mule account at a real moment), among the K nearest whose missing flags equal the original's and
            whose five values lie within the budget eps of the original block (L-inf, standardized); otherwise the
            original payee is kept. dest_log_ratio (derived) is then recomputed (log_amt - mean; its fill when the
            payee has no history).
  IEEE-CIS  a field whose missing flag is on keeps its original value (a missing field cannot be filled without
            switching the flag, which the constrained set forbids).
  Sparkov   the merchant stays within 1 degree of the cardholder (the simulator's box; every training row), and dist
            is recomputed from its definition hypot(lat - merch_lat, (long - merch_long) cos lat); when the new
            distance leaves the budget eps around the original distance, the merchant's move is shrunk toward the
            original until it fits, and likewise when it leaves the study's domain for dist (largest feasible share of
            the move on a 1/32 grid).
  TabFormer merch_seen stays 0 for a card with no earlier transactions.
  all       structural zeros (decoydef.consistency.StructuralZeros): a pair of discrete values that never occurs in
            training although independence predicts at least 200 rows (e.g. IEEE-CIS dist1 present with a known
            device type: dist1 exists only for product W, which never carries device data). The features of a hit
            pair go back to their original values (with their missing flag or value, and on PaySim the whole payee
            block), repeated until no pair is hit.

usage: ConsistentConstraints(ds, base_constraints, X_train); a drop-in for Constraints in the RL env (project, mask,
mutable, allowed, linf ... are delegated).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

PROC = Path(os.environ.get("FRAUDADV_PROC", "/home/user/study/data/processed"))
PAYEE = ["dest_cnt_prev", "dest_log_amt_mean_prev", "dest_log_dt_prev", "dest_cnt_24h", "dest_logamt_sum_24h",
         "isna_dest_log_amt_mean_prev", "isna_dest_log_ratio", "isna_dest_log_dt_prev"]
K = 64


class ConsistentConstraints:
    def __init__(self, ds, base, X_train):
        self.ds, self.base = ds, base
        m = json.loads((PROC / ds / "meta.json").read_text())
        self.ix = {f: i for i, f in enumerate(m["features"])}
        self.mu, self.sd = np.array(m["mu"], float), np.array(m["sd"], float)
        from decoydef.consistency import structural
        self.sz = structural(ds)
        f = m["features"]
        self.link = {}
        for k, n in enumerate(f):                                        # value <-> its missing flag
            if n.startswith("isna_") and n[5:] in self.ix:
                self.link.setdefault(k, set()).add(self.ix[n[5:]])
                self.link.setdefault(self.ix[n[5:]], set()).add(k)
        if ds == "paysim":                                               # the payee block moves as one
            blk = {self.ix[n] for n in PAYEE} | {self.ix["dest_log_ratio"]}
            for k in blk:
                self.link.setdefault(k, set()).update(blk - {k})
        if ds == "paysim":
            from sklearn.neighbors import KDTree
            self.blk = np.array([self.ix[f] for f in PAYEE])
            L = np.unique(np.asarray(X_train, np.float32)[:, self.blk], axis=0)
            self.lib = L
            self.tree = KDTree(L[:, :5].astype(np.float64), leaf_size=40)
            j = self.ix["dest_log_ratio"]
            na = (X_train[:, self.ix["isna_dest_log_ratio"]] * self.sd[self.ix["isna_dest_log_ratio"]]
                  + self.mu[self.ix["isna_dest_log_ratio"]]) > 0.5
            self.ratio_fill = float(np.median(X_train[na, j]))          # standardized fill of the ratio

    def __getattr__(self, k):                                            # delegate everything else
        return getattr(self.base, k)

    def _raw(self, X, f):
        j = self.ix[f]
        return X[:, j] * self.sd[j] + self.mu[j]

    def project(self, X, X0, eps=None):
        X = self.base.project(X, X0, eps)
        X0 = np.broadcast_to(np.asarray(X0, np.float32), X.shape)
        if self.ds == "paysim":
            X = self._paysim(X, X0, eps)
        elif self.ds == "ieee_cis":
            for b in ("addr1", "addr2", "dist1"):
                if b in self.ix and "isna_" + b in self.ix:
                    j, jf = self.ix[b], self.ix["isna_" + b]
                    miss = (X0[:, jf] * self.sd[jf] + self.mu[jf]) > 0.5        # flag on: the field stays missing
                    X[:, j] = np.where(miss, X0[:, j], X[:, j])
                    X[:, jf] = X0[:, jf]
        elif self.ds == "sparkov":                                        # distance follows the coordinates
            X = self._sparkov(X, X0, eps)
        elif self.ds == "tabformer":
            j, jc = self.ix["merch_seen"], self.ix["cnt_prev"]
            none = self._raw(X, "cnt_prev") < 1e-3
            zero = np.float32((0.0 - self.mu[j]) / self.sd[j])
            X[:, j] = np.where(none, zero, X[:, j])
        return self._structural(X, X0)

    def _structural(self, X, X0):
        if not self.sz.tabs:
            return X
        for _ in range(6):
            H = self.sz.pairs_hit(X)
            bad = H.any(1) & self.sz.pairs_hit(X0).sum(1).__lt__(H.sum(1))   # rows worse than their original
            rows = np.flatnonzero(bad)
            if not len(rows):
                break
            for t in np.flatnonzero(H[rows].any(0)):
                i, j, _ = self.sz.tabs[t]
                r = rows[H[rows, t]]
                cols = {i, j} | self.link.get(i, set()) | self.link.get(j, set())
                for k in cols:
                    X[r, k] = X0[r, k]
        return X

    def _sparkov(self, X, X0, eps):
        from decoydef.consistency import SPARKOV_BOX
        jd, jla, jlo = self.ix["dist"], self.ix["merch_lat"], self.ix["merch_long"]
        rows = np.flatnonzero(np.any(X[:, [jla, jlo, jd]] != X0[:, [jla, jlo, jd]], axis=1))
        if not len(rows):
            return X
        lat, lon = self._raw(X[rows], "lat"), self._raw(X[rows], "long")
        m0 = np.stack([self._raw(X0[rows], "merch_lat"), self._raw(X0[rows], "merch_long")], 1)
        m1 = np.stack([np.clip(self._raw(X[rows], "merch_lat"), lat - SPARKOV_BOX, lat + SPARKOV_BOX),
                       np.clip(self._raw(X[rows], "merch_long"), lon - SPARKOV_BOX, lon + SPARKOV_BOX)], 1)
        d0 = X0[rows, jd]

        def dist_std(m):
            d = np.hypot(lat - m[:, 0], (lon - m[:, 1]) * np.cos(np.radians(lat)))
            return ((d - self.mu[jd]) / self.sd[jd]).astype(np.float32)

        lo = np.minimum(self.base.lower[jd], d0)                         # the study's domain for dist (admits X0)
        hi = np.maximum(self.base.upper[jd], d0)
        t = np.zeros(len(rows))
        for s_ in np.linspace(1.0, 0.0, 33)[:-1][::-1]:                  # ascending; keep the largest feasible share
            d_ = dist_std(m0 + s_ * (m1 - m0))
            ok = (d_ >= lo - 1e-6) & (d_ <= hi + 1e-6)                   # stay in the constrained set's domain
            if eps is not None:                                          # and inside the budget
                ok &= np.abs(d_ - d0) <= eps + 1e-6
            t = np.where(ok, s_, t)
        m = m0 + t[:, None] * (m1 - m0)
        d = np.where(t > 0, dist_std(m), d0)
        X[rows, jla] = np.where(t > 0, ((m[:, 0] - self.mu[jla]) / self.sd[jla]).astype(np.float32), X0[rows, jla])
        X[rows, jlo] = np.where(t > 0, ((m[:, 1] - self.mu[jlo]) / self.sd[jlo]).astype(np.float32), X0[rows, jlo])
        X[rows, jd] = d
        return X

    def _paysim(self, X, X0, eps):
        B, B0 = X[:, self.blk], X0[:, self.blk]
        ch = np.any(B != B0, axis=1)
        if not ch.any():
            return X
        rows = np.flatnonzero(ch)
        k = min(K, len(self.lib))
        _, nn = self.tree.query(B[rows, :5].astype(np.float64), k=k)
        C = self.lib[nn]                                                 # (r, k, 8) real payee blocks
        ok = np.ones(C.shape[:2], bool)
        ok = np.all(C[:, :, 5:] == B0[rows, None, 5:], axis=2)          # missing flags are not attacker-movable
        if eps is not None:
            ok &= np.max(np.abs(C[:, :, :5] - B0[rows, None, :5]), axis=2) <= eps + 1e-6
        first = np.where(ok.any(1), np.argmax(ok, 1), -1)
        newB = np.where((first >= 0)[:, None], C[np.arange(len(rows)), np.maximum(first, 0)], B0[rows])
        X[np.ix_(rows, self.blk)] = newB
        # dest_log_ratio from its definition
        j = self.ix["dest_log_ratio"]
        na = self._raw(X[rows], "isna_dest_log_ratio") > 0.5
        r = self._raw(X[rows], "log_amt") - self._raw(X[rows], "dest_log_amt_mean_prev")
        X[rows, j] = np.where(na, self.ratio_fill, ((r - self.mu[j]) / self.sd[j])).astype(np.float32)
        return X
