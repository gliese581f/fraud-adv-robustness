"""Attacks on tree ensembles (XGBoost, CatBoost) by moving features across split thresholds.

A tree ensemble's score is piecewise constant: it changes only when a feature crosses one of the ensemble's split
thresholds. Gradient attacks therefore do not apply. This file holds:

  TreeEnsemble  the view of a fitted XGBoost or CatBoost model the attacks need: the sorted split thresholds of
                every feature and a vectorised margin (logit) function.
  AATS          Axis-Aligned Threshold Search: a beam search over single-feature threshold crossings, scored by
                remaining margin plus a cost penalty, whose successes are then pruned and pulled back toward the
                original point.
  box_search    the L-inf box phase that blackwhite.aats runs after AATS: moving many features at once to the
                edge of a shrinking box.

Inputs: a fitted model (through TreeEnsemble) and frauds as standardized feature rows. Output: AATSResult with the
attacked points, success flags, L-inf costs and the number of model queries per fraud.

The exact MILP of Kantchelian et al. (2016), in milp.py, is the trees' white-box attack in the grid and certifies
how close AATS comes to the minimal distance (Table S-XXII, Table S-XXIII). AATS and box_search (through
blackwhite.aats) are the grid's AATS attack on trees (Table III, Table IV, Table S-XI, Tables S-V to S-X), and
TreeEnsemble also serves the exact boundary thickness (Table S-XXIV, boundary/thickness.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass
class TreeEnsemble:
    """Split thresholds per feature plus a vectorised margin (logit) function.

    rule = "lt": a split sends x left iff x < t (XGBoost); "le": left iff x <= t (CatBoost).
    """

    thresholds: list[np.ndarray]           # sorted float32 thresholds, one array per feature
    margin_fn: Callable[[np.ndarray], np.ndarray]
    n_features: int
    rule: str = "lt"

    def up_value(self, t):
        """Smallest value that lands on the right side of threshold t."""
        t = np.float32(t)
        return t if self.rule == "lt" else np.nextafter(t, np.float32(np.inf))

    def up_values(self, t: np.ndarray) -> np.ndarray:
        """up_value for an array of thresholds."""
        t = t.astype(np.float32)
        return t if self.rule == "lt" else np.nextafter(t, np.float32(np.inf))

    def down_values(self, t: np.ndarray) -> np.ndarray:
        """down_value for an array of thresholds."""
        t = t.astype(np.float32)
        return np.nextafter(t, np.float32(-np.inf)) if self.rule == "lt" else t

    def down_value(self, t):
        """Largest value that lands on the left side of threshold t."""
        t = np.float32(t)
        return np.nextafter(t, np.float32(-np.inf)) if self.rule == "lt" else t

    @classmethod
    def from_xgboost(cls, model, iteration_range=None) -> "TreeEnsemble":
        """Thresholds from the JSON dump of every tree (or of the trees in iteration_range); the margin is the
        booster's raw prediction over the same trees."""
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        n_features = booster.num_features()
        per_feat: list[set[float]] = [set() for _ in range(n_features)]

        def walk(node):
            if "split" in node:
                f = node["split"]
                idx = int(f[1:]) if isinstance(f, str) and f.startswith("f") and f[1:].isdigit() else booster.feature_names.index(f)
                per_feat[idx].add(float(np.float32(node["split_condition"])))
                for child in node.get("children", []):
                    walk(child)

        dumps = booster.get_dump(dump_format="json")
        if iteration_range is not None:
            dumps = dumps[iteration_range[0]:iteration_range[1]]
        for dump in dumps:
            walk(json.loads(dump))
        kw = {} if iteration_range is None else {"iteration_range": iteration_range}

        def margin_fn(X: np.ndarray) -> np.ndarray:
            return booster.inplace_predict(np.asarray(X, dtype=np.float32), predict_type="margin", **kw)

        return cls([np.array(sorted(s), dtype=np.float32) for s in per_feat], margin_fn, n_features, "lt")

    @classmethod
    def from_catboost(cls, model, n_features: int) -> "TreeEnsemble":
        """CatBoost oblivious trees: a float split is (x > border), i.e. left iff x <= border."""
        mj = json.loads(_catboost_json(model))
        per_feat: list[set[float]] = [set() for _ in range(n_features)]
        ff = mj["features_info"]["float_features"]           # float_feature_index -> position in this list
        for tree in mj["oblivious_trees"]:
            for sp in tree["splits"]:
                if sp.get("split_type", "FloatFeature") == "FloatFeature":
                    j = ff[sp["float_feature_index"]]["flat_feature_index"]
                    per_feat[j].add(float(np.float32(sp["border"])))

        def margin_fn(X: np.ndarray) -> np.ndarray:
            return model.predict(np.asarray(X, dtype=np.float32), prediction_type="RawFormulaVal")

        return cls([np.array(sorted(s), dtype=np.float32) for s in per_feat], margin_fn, n_features, "le")


def _catboost_json(model) -> str:
    """The CatBoost model as JSON text (CatBoost writes JSON only to a file)."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.json")
        model.save_model(path, format="json")
        return open(path).read()


@dataclass
class AATSResult:
    x_adv: np.ndarray           # attacked point per fraud (the lowest-margin point found when the attack failed)
    success: np.ndarray         # margin below the target
    cost: np.ndarray            # cost of x_adv in the attack's norm
    n_queries: np.ndarray       # model evaluations spent per fraud


@dataclass
class AATS:
    """Axis-Aligned Threshold Search.

    Each beam step expands every state in the beam by single-feature moves just past nearby thresholds
    (_candidates), drops states costlier than the budget or the best success so far, and scores the rest with the
    model. The cheapest few successes are refined (_prune, then _tighten). The search continues `patience` steps
    after the first success to look for cheaper ones; the non-winning states with the lowest
    margin + cost_weight * cost form the next beam.

    Parameters
    ----------
    ensemble: TreeEnsemble to attack.
    mutable: indices of attacker-controllable features (default: all).
    scale: per-feature scale; cost is measured in units of ``scale``.
    norm: "linf", "l1" or "l0".
    eps: maximum allowed cost (np.inf = minimal-perturbation mode).
    target_margin: success when margin < target_margin (0 = p_fraud < 0.5).
    beam: beam width; k_nearest: thresholds per side considered per feature.
    max_steps: beam steps at most.
    cost_weight: trade-off between remaining margin and cost when ranking the beam.
    patience: extra beam steps after the first success, to look for cheaper ones.
    refine_top: winners per step that get the prune + tighten refinement.
    lower/upper: per-feature bounds.
    allowed: {feature index: sorted array of valid values} for discrete features
             (categoricals, hour, flags); moves snap to a valid value past the threshold.
    project: optional f(X, x0) -> X applied to every candidate (recomputes derived
             features, enforces domain constraints). Cost is measured on ``mutable`` only.
    rng: not used (the search is deterministic); kept so that the constructor's arguments stay the same.
    """

    ensemble: TreeEnsemble
    mutable: Sequence[int] | None = None
    scale: np.ndarray | None = None
    norm: str = "linf"
    eps: float = np.inf
    target_margin: float = 0.0
    beam: int = 8
    k_nearest: int = 2
    max_steps: int = 50
    cost_weight: float = 0.5
    patience: int = 3
    refine_top: int = 3
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    allowed: dict | None = None
    project: Callable | None = None
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def _cost(self, X: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """Cost of each row of X relative to x0, on the attacker's features, in units of scale."""
        d = np.abs(X - x0)[:, self._mutable] / self._scale[self._mutable]
        if self.norm == "linf":
            return d.max(axis=1)
        if self.norm == "l1":
            return d.sum(axis=1)
        if self.norm == "l0":
            return (d > 0).sum(axis=1).astype(float)
        raise ValueError(self.norm)

    def _is_adversarial(self, x: np.ndarray) -> bool:
        """One model query: is the margin of x below the target?"""
        return self.ensemble.margin_fn(x[None])[0] < self.target_margin

    def _jump_idx(self, n: int) -> np.ndarray:
        """Which of n thresholds (sorted from the current value outwards) a move may cross to: the k_nearest
        nearest, the 16th, 32nd, 64th, ... and the farthest."""
        idx = list(range(min(self.k_nearest, n)))
        step = 16
        while step < n:
            idx.append(step - 1)
            step *= 2
        if n > self.k_nearest:
            idx.append(n - 1)
        return np.unique(np.array(idx, dtype=int))

    def _snap(self, j: int, val, v, up: bool):
        """Snap a crossing value of a discrete feature to the nearest valid value past it (going up or down from
        the current value v); v itself when there is none. Continuous features are returned unchanged."""
        a = self._allowed.get(j)
        if a is None:
            return val
        if up:
            k = np.searchsorted(a, val, side="left")
            return a[k] if k < len(a) else v
        k = np.searchsorted(a, val, side="right") - 1
        return a[k] if k >= 0 else v

    def _proj(self, C: np.ndarray, x0: np.ndarray) -> np.ndarray:
        return C if self.project is None else self.project(C, x0)

    def _candidates(self, x: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """Single-feature moves from x to just past the k nearest thresholds on each side, plus geometric jumps
        (16th, 32nd, ... threshold away) so one step can cross many splits when a feature has hundreds of
        thresholds. Moves are snapped to valid values, clipped to the bounds and projected."""
        moves = []
        ens = self.ensemble
        for j in self._mutable:
            thr = ens.thresholds[j]
            if thr.size == 0:
                continue
            v = np.float32(x[j])
            if ens.rule == "lt":                     # left iff x < t
                up_all, down_all = thr[thr > v], thr[thr <= v][::-1]
            else:                                    # left iff x <= t
                up_all, down_all = thr[thr >= v], thr[thr < v][::-1]
            vals = [(ens.up_value(t), True) for t in up_all[self._jump_idx(up_all.size)]]
            vals += [(ens.down_value(t), False) for t in down_all[self._jump_idx(down_all.size)]]
            for val, up in vals:
                val = self._snap(j, val, v, up)
                val = np.clip(val, self._lower[j], self._upper[j])
                if val == v:
                    continue
                c = x.copy()
                c[j] = val
                moves.append(c)
        if not moves:
            return np.empty((0, x.size), np.float32)
        return self._proj(np.array(moves, np.float32), x0)

    def _changed_by_cost(self, x_adv: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """Features where x_adv differs from x0, largest scaled change first."""
        changed = np.flatnonzero(x_adv != x0)
        return changed[np.argsort(-np.abs(x_adv - x0)[changed] / self._scale[changed])]

    def _prune(self, x_adv: np.ndarray, x0: np.ndarray) -> tuple[np.ndarray, int]:
        """Revert changed coordinates one at a time (largest change first) while the point stays adversarial.
        Returns (point, queries)."""
        q = 0
        for j in self._changed_by_cost(x_adv, x0):
            trial = x_adv.copy()
            trial[j] = x0[j]
            trial = self._proj(trial[None], x0)[0]
            q += 1
            if self._is_adversarial(trial):
                x_adv = trial
        return x_adv, q

    def _crossing_values(self, j: int, x0j, xj) -> np.ndarray:
        """The values strictly between x0j and xj at which feature j is just past one of its thresholds, snapped to
        valid values, inside the bounds, without repeats, nearest to x0j first."""
        ens = self.ensemble
        thr = ens.thresholds[j]
        if xj > x0j:
            vals = ens.up_values(thr)
            vals = vals[(vals > x0j) & (vals < xj)]
            vals = np.array([self._snap(j, u, xj, True) for u in vals], np.float32)
        else:
            vals = ens.down_values(thr)[::-1]
            vals = vals[(vals < x0j) & (vals > xj)]
            vals = np.array([self._snap(j, u, xj, False) for u in vals], np.float32)
        vals = vals[(vals >= self._lower[j]) & (vals <= self._upper[j])]
        if len(vals):
            _, first = np.unique(vals, return_index=True)
            vals = vals[np.sort(first)]
        return vals

    def _first_adversarial(self, x_adv: np.ndarray, x0: np.ndarray, j: int, vals: np.ndarray) -> tuple[int, int]:
        """Bisection for the first of vals (ordered from x0 outwards) at which x_adv, with feature j set to it, is
        still adversarial; len(vals) when the search finds none. Returns (index, queries)."""
        q = 0
        a, b = 0, len(vals)
        while a < b:
            mid = (a + b) // 2
            trial = x_adv.copy()
            trial[j] = vals[mid]
            trial = self._proj(trial[None], x0)[0]
            q += 1
            if self._is_adversarial(trial):
                b = mid
            else:
                a = mid + 1
        return a, q

    def _tighten(self, x_adv: np.ndarray, x0: np.ndarray, passes: int = 2) -> tuple[np.ndarray, int]:
        """Pull each changed coordinate back toward x0, to the closest crossing point that keeps the example
        adversarial (bisection over the crossing values between x0_j and x_adv_j, largest-cost coordinate
        first). Returns (point, queries)."""
        q = 0
        for _ in range(passes):
            for j in self._changed_by_cost(x_adv, x0):
                vals = self._crossing_values(j, x0[j], x_adv[j])
                a, qj = self._first_adversarial(x_adv, x0, j, vals)
                q += qj
                if a < len(vals):
                    x_adv = x_adv.copy()
                    x_adv[j] = vals[a]
                    x_adv = self._proj(x_adv[None], x0)[0]
        return x_adv, q

    def _refine(self, x_adv: np.ndarray, x0: np.ndarray) -> tuple[np.ndarray, int]:
        """Prune unneeded coordinates of a success, then tighten the rest. Returns (point, queries)."""
        x, q1 = self._prune(x_adv, x0)
        x, q2 = self._tighten(x, x0)
        return x, q1 + q2

    def attack_one(self, x0: np.ndarray) -> tuple[np.ndarray, bool, float, int]:
        """The beam search for one fraud. Returns (point, success, cost, queries); on failure the point is the
        lowest-margin state seen."""
        x0 = x0.astype(np.float32)
        beam = [x0]
        queries = 0
        best, best_cost, steps_since_win = None, np.inf, None
        lowest, lowest_m = x0, np.inf
        for _ in range(self.max_steps):
            cands = [c for b in beam for c in self._candidates(b, x0)]
            if not cands:
                break
            C = np.unique(np.array(cands), axis=0)
            cost = self._cost(C, x0)
            keep = cost <= min(self.eps, best_cost)          # never expand states already costlier than the best
            C, cost = C[keep], cost[keep]
            if len(C) == 0:
                break
            m = self.ensemble.margin_fn(C)
            queries += len(C)
            k = int(np.argmin(m))
            if m[k] < lowest_m:                               # lowest-margin state, returned on failure
                lowest, lowest_m = C[k], m[k]
            win = m < self.target_margin
            for i in np.flatnonzero(win)[np.argsort(cost[win])][: self.refine_top]:   # the cheapest few winners
                x, q = self._refine(C[i], x0)
                queries += q
                c = self._cost(x[None], x0)[0]
                if c < best_cost:
                    best, best_cost = x, c
            if win.any() and steps_since_win is None:
                steps_since_win = 0
            if steps_since_win is not None:
                steps_since_win += 1
                if steps_since_win > self.patience:
                    break
            # Rank non-winning states by remaining margin plus a cost penalty. Pure "drop per unit cost" stalls on
            # plateaus where single moves leave the margin unchanged; the additive form keeps moving.
            C, m, cost = C[~win], m[~win], cost[~win]
            if len(C) == 0:
                break
            top = np.argsort(m + self.cost_weight * cost)[: self.beam]
            beam = [C[i] for i in top]
        if best is None:
            return lowest, False, float(self._cost(lowest[None], x0)[0]), queries
        return best, True, float(best_cost), queries

    def generate(self, X: np.ndarray) -> AATSResult:
        """Attack every row of X, one at a time."""
        n, d = X.shape
        self._mutable = list(range(d)) if self.mutable is None else list(self.mutable)
        self._scale = np.ones(d) if self.scale is None else np.asarray(self.scale, float)
        self._lower = np.full(d, -np.inf) if self.lower is None else np.asarray(self.lower)
        self._upper = np.full(d, np.inf) if self.upper is None else np.asarray(self.upper)
        self._allowed = {} if self.allowed is None else {int(k): np.asarray(v, np.float32) for k, v in self.allowed.items()}
        out = np.empty_like(X, dtype=np.float32)
        succ = np.zeros(n, bool)
        cost = np.zeros(n)
        nq = np.zeros(n, int)
        for i in range(n):
            out[i], succ[i], cost[i], nq[i] = self.attack_one(X[i])
        return AATSResult(out, succ, cost, nq)


def box_search(margin_fn, X0, target, eps, mutable, project=None, n_grid=7, passes=2, bs_steps=7, rng=None):
    """L-inf box phase of AATS. Returns (X_adv, success, linf_cost).

    For a radius r, `descend(r)` runs coordinate descent over the attacker's features (in a random order, all
    frauds batched): each feature in turn tries n_grid values evenly spaced in [x0 - r, x0 + r] and keeps the one
    with the lowest margin. A binary search (bs_steps halvings) then looks for the smallest r in (0, eps] at which
    the fraud still evades. Frauds that do not evade at r = eps return the point found at eps, with its cost.

    Axis-aligned trees reward moving many features at once to the edge of the box, which single-feature beam
    moves find slowly; this phase finds those corners directly.
    """
    rng = rng or np.random.default_rng(0)
    X0 = X0.astype(np.float32)
    n = len(X0)
    proj = (lambda C, x0: C) if project is None else project
    grid = np.linspace(-1, 1, n_grid).astype(np.float32)
    X0r = np.repeat(X0, n_grid, 0)

    def descend(r):
        """Coordinate descent inside the box of radius r (one radius per fraud). Returns (points, margins)."""
        X = X0.copy()
        m = margin_fn(X)
        for _ in range(passes):
            for j in rng.permutation(mutable):
                C = np.repeat(X, n_grid, 0)
                C[:, j] = np.repeat(X0[:, j], n_grid) + np.tile(grid, n) * np.repeat(r, n_grid)
                C = proj(C, X0r)
                mc = margin_fn(C).reshape(n, n_grid)
                k = mc.argmin(1)
                better = mc[np.arange(n), k] < m
                X[better] = C.reshape(n, n_grid, -1)[np.flatnonzero(better), k[better]]
                m = np.minimum(m, mc[np.arange(n), k])
        return X, m

    lo = np.zeros(n, np.float32)
    hi = np.full(n, eps, np.float32)
    X_eps, m_eps = descend(hi)
    ok = m_eps < target
    best = np.where(ok[:, None], X_eps, X0)
    for _ in range(bs_steps):
        if not ok.any():
            break
        mid = (lo + hi) / 2
        Xm, mm = descend(mid)
        win = (mm < target) & ok
        best[win] = Xm[win]
        hi = np.where(win, mid, hi)
        lo = np.where(ok & ~win, mid, lo)
    cost = np.abs(best - X0)[:, mutable].max(1)
    out = np.where(ok[:, None], best, X_eps)
    return out, ok, np.where(ok, cost, np.abs(X_eps - X0)[:, mutable].max(1))
