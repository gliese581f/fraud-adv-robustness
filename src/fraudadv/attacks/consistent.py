"""The consistent attacker's repair: after the constrained projection, make the row one a real transaction could
produce (Constraints(meta, X_train, "consistent") applies it after every attack step).

The consistent attacker is the constrained attacker restricted to realizable rows. The repair never leaves the
constrained set: same controllable features, same budget eps, immutable features and missing flags untouched, so
for a fixed search unconstrained >= constrained >= consistent. Per dataset:

  PaySim     the payee block (earlier transfers, their mean amount, time since the last one, 24-hour count and inflow,
             and their missing flags) snaps to the nearest block that occurs in the training data, among the K nearest
             whose missing flags equal the original's and whose five values lie within eps of the original block
             (L-inf, standardized); otherwise the original payee is kept. The attacker picks a real payee account at a
             real moment. dest_log_ratio is then recomputed (log_amt - payee mean; its fill when the payee has no
             history).
  IEEE-CIS   a field whose missing flag is on keeps its value (filling it would need the flag switched, which the
             constrained set forbids).
  Sparkov    the merchant stays within 1 degree of the cardholder (the simulator's range) and dist is recomputed from
             its definition hypot(lat - merch_lat, (long - merch_long) cos lat); if the new distance leaves the budget
             around the original distance, or the data domain, the merchant's move is shrunk toward the original
             until it fits (largest feasible share of the move on a 1/32 grid).
  TabFormer  merch_seen stays 0 for a card with no earlier transactions.
  all        structural zeros (fraudadv.attacks.realizable.StructuralZeros): the features of a forbidden pair go back
             to their original values (with their missing flag or value; on PaySim the whole payee block), repeated
             until no pair is hit.

ULB, TabFormer and BAF had no unrealizable evasions, so there the consistent attacker equals the constrained one.
Results: Section III (the threat-model section), the consistent-attacker appendix and every consistent (K) number.
Shared with the project's other papers; identical in behaviour to decoydef/consistent.py (sha256 prefix 602f7eb3f2b6).
"""
from __future__ import annotations

import json

import numpy as np

from fraudadv.paths import PROC

PAYEE = ["dest_cnt_prev", "dest_log_amt_mean_prev", "dest_log_dt_prev", "dest_cnt_24h", "dest_logamt_sum_24h",
         "isna_dest_log_amt_mean_prev", "isna_dest_log_ratio", "isna_dest_log_dt_prev"]
K = 64                    # candidate real payee blocks per query (PaySim)


class ConsistentRepair:
    """repair(X, X0, eps) maps rows already projected onto the constrained set to realizable rows."""

    def __init__(self, ds, base, X_train):
        """base: the constrained Constraints of the dataset (bounds of the data domain)."""
        self.ds, self.base = ds, base
        m = json.loads((PROC / ds / "meta.json").read_text())
        self.ix = {f: i for i, f in enumerate(m["features"])}
        self.mu, self.sd = np.array(m["mu"], float), np.array(m["sd"], float)
        from fraudadv.attacks.realizable import structural
        self.sz = structural(ds)
        f = m["features"]
        self.link = {}                       # features that must go back together (a value and its missing flag)
        for k, n in enumerate(f):
            if n.startswith("isna_") and n[5:] in self.ix:
                self.link.setdefault(k, set()).add(self.ix[n[5:]])
                self.link.setdefault(self.ix[n[5:]], set()).add(k)
        if ds == "paysim":                   # the payee block moves as one
            blk = {self.ix[n] for n in PAYEE} | {self.ix["dest_log_ratio"]}
            for k in blk:
                self.link.setdefault(k, set()).update(blk - {k})
            from sklearn.neighbors import KDTree
            self.blk = np.array([self.ix[f] for f in PAYEE])
            L = np.unique(np.asarray(X_train, np.float32)[:, self.blk], axis=0)
            self.lib = L                     # every payee block of the training data
            self.tree = KDTree(L[:, :5].astype(np.float64), leaf_size=40)
            j = self.ix["dest_log_ratio"]
            na = (X_train[:, self.ix["isna_dest_log_ratio"]] * self.sd[self.ix["isna_dest_log_ratio"]]
                  + self.mu[self.ix["isna_dest_log_ratio"]]) > 0.5
            self.ratio_fill = float(np.median(X_train[na, j]))          # standardized fill of the ratio

    def _raw(self, X, f):
        j = self.ix[f]
        return X[:, j] * self.sd[j] + self.mu[j]

    def __call__(self, X, X0, eps=None):
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
        elif self.ds == "sparkov":
            X = self._sparkov(X, X0, eps)
        elif self.ds == "tabformer":
            j = self.ix["merch_seen"]
            none = self._raw(X, "cnt_prev") < 1e-3
            zero = np.float32((0.0 - self.mu[j]) / self.sd[j])
            X[:, j] = np.where(none, zero, X[:, j])
        return self._structural(X, X0)

    def _structural(self, X, X0):
        if not self.sz.tabs:
            return X
        for _ in range(6):
            H = self.sz.pairs_hit(X)
            bad = H.any(1) & (self.sz.pairs_hit(X0).sum(1) < H.sum(1))   # rows hitting more pairs than the original
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
        from fraudadv.attacks.realizable import SPARKOV_BOX
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

        lo = np.minimum(self.base.lower[jd], d0)                         # the data domain for dist (admits X0)
        hi = np.maximum(self.base.upper[jd], d0)
        t = np.zeros(len(rows))
        for s_ in np.linspace(1.0, 0.0, 33)[:-1][::-1]:                  # ascending: keep the largest feasible share
            d_ = dist_std(m0 + s_ * (m1 - m0))
            ok = (d_ >= lo - 1e-6) & (d_ <= hi + 1e-6)
            if eps is not None:
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
        C = self.lib[nn]                                                 # (rows, k, 8) real payee blocks
        ok = np.all(C[:, :, 5:] == B0[rows, None, 5:], axis=2)          # missing flags are not attacker-movable
        if eps is not None:
            ok &= np.max(np.abs(C[:, :, :5] - B0[rows, None, :5]), axis=2) <= eps + 1e-6
        first = np.where(ok.any(1), np.argmax(ok, 1), -1)
        newB = np.where((first >= 0)[:, None], C[np.arange(len(rows)), np.maximum(first, 0)], B0[rows])
        X[np.ix_(rows, self.blk)] = newB
        j = self.ix["dest_log_ratio"]                                    # recomputed from its definition
        na = self._raw(X[rows], "isna_dest_log_ratio") > 0.5
        r = self._raw(X[rows], "log_amt") - self._raw(X[rows], "dest_log_amt_mean_prev")
        X[rows, j] = np.where(na, self.ratio_fill, ((r - self.mu[j]) / self.sd[j])).astype(np.float32)
        return X
