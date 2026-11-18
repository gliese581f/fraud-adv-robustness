"""Decision-boundary thickness (Yang et al., NeurIPS 2020), adapted to binary fraud models.

For a pair of points x_r (a detected fraud) and x_s (a point the model passes), walk the straight segment
x(t) = t x_r + (1 - t) x_s, t in [0, 1], and measure how much of it lies in the "uncertain band" of the model:

    thickness = ||x_r - x_s|| * (share of t in [0, 1] with alpha < g(x(t)) < beta),
    g = p_fraud - p_legit = 2 p_fraud - 1.

A thin boundary means the score jumps from "legit" to "fraud" over a short stretch, so a small push can cross it.

Two ways to compute the share of the segment:
  thickness_sampled       any model: evaluate g at n_steps midpoints of the segment (midpoint rule).
  thickness_exact_trees   tree ensembles: the score is constant between the points where the segment crosses a
                          split threshold, so the share is an exact sum over those intervals.

Inputs: paired arrays X_r, X_s (one pair per row) and the model's score function. Output: one thickness per pair.
Distances are in standardized feature units, optionally divided per feature by `scale`.

Used by attacks/run_attacks.py (grid thickness, runs/<d>/attacks/thickness*.csv) and boundary/run_thickness.py
(class-pair thickness); these feed Table S-XXIV, Figure S5 and Section VI-G (RQ3).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from fraudadv.attacks.trees import TreeEnsemble


def _g_from_margin(m: np.ndarray) -> np.ndarray:
    """g = 2 sigmoid(margin) - 1, i.e. p_fraud - p_legit."""
    p = 1.0 / (1.0 + np.exp(-m))
    return 2.0 * p - 1.0


def thickness_sampled(
    proba_fn: Callable[[np.ndarray], np.ndarray],
    X_r: np.ndarray,
    X_s: np.ndarray,
    alpha: float = 0.0,
    beta: float = 0.75,
    n_steps: int = 256,
    scale: np.ndarray | None = None,
    ctx: np.ndarray | None = None,
) -> np.ndarray:
    """Thickness of each pair (X_r[i], X_s[i]) by sampling n_steps midpoints of the segment.

    proba_fn(X) returns P(fraud), shape (n,). For models that need a context id per row (graph and foundation
    models), pass ctx (one id per pair); proba_fn is then called as proba_fn(X, ctx_ids)."""
    scale = np.ones(X_r.shape[1]) if scale is None else scale
    ts = (np.arange(n_steps) + 0.5) / n_steps          # midpoint rule
    out = np.empty(len(X_r))
    for i, (xr, xs) in enumerate(zip(X_r, X_s)):
        points = ts[:, None] * xr + (1 - ts[:, None]) * xs
        g = 2.0 * (proba_fn(points) if ctx is None else proba_fn(points, np.repeat(ctx[i], len(points)))) - 1.0
        share_in_band = np.mean((g > alpha) & (g < beta))
        out[i] = np.linalg.norm((xr - xs) / scale) * share_in_band
    return out


def thickness_exact_trees(
    ens: TreeEnsemble,
    X_r: np.ndarray,
    X_s: np.ndarray,
    alpha: float = 0.0,
    beta: float = 0.75,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    """Exact thickness of each pair for a tree ensemble (no sampling error).

    ens.margin_fn gives the ensemble's margin (logit) and ens.thresholds[j] the split thresholds on feature j."""
    scale = np.ones(X_r.shape[1]) if scale is None else scale
    out = np.empty(len(X_r))
    for i, (xr, xs) in enumerate(zip(X_r.astype(np.float64), X_s.astype(np.float64))):
        delta = xr - xs
        # every t in (0, 1) at which the segment crosses a split threshold of a feature that changes along it
        cuts = [0.0, 1.0]
        for j in np.flatnonzero(delta != 0):
            t = (ens.thresholds[j].astype(np.float64) - xs[j]) / delta[j]
            cuts.extend(t[(t > 0) & (t < 1)])
        cuts = np.unique(cuts)
        # the score is constant on each interval between cuts: evaluate it once, at the interval's midpoint
        mids = (cuts[:-1] + cuts[1:]) / 2
        points = mids[:, None] * xr + (1 - mids[:, None]) * xs
        g = _g_from_margin(ens.margin_fn(points))
        lengths = np.diff(cuts)
        share_in_band = lengths[(g > alpha) & (g < beta)].sum()
        out[i] = np.linalg.norm(delta / scale) * share_in_band
    return out
