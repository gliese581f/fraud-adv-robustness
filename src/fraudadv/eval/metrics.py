"""The study's performance measures: AUC and KS, unit- and dollar-weighted, plus AUPRC, the operating
threshold and recall at that threshold.

Inputs: labels y (1 = fraud), scores s (higher = more fraudulent), weights w and amounts amt, as numpy arrays.
Unit weights are the sampling weights of fraudadv.data.prepare (legitimate rows were subsampled, so each kept
legitimate row has w = 1 / sampling rate), so every metric estimates its value on the whole population.
Dollar weights are w * amount.
Outputs: floats; `all_metrics` returns them together in a dict.

Used by:
  * clean performance (Table S-IV, section VI-A): `all_metrics` in hpo/train_all.py and hpo/train_context.py,
    written to runs/<dataset>/<family>/metrics*.json
  * every attack success rate: the model's decision threshold is `threshold_at_fpr(..., 0.01)`, the score at a
    1% weighted false-positive rate on the validation split (attacks/run_attacks.py, byom.py)
  * ranking metrics under attack (Table S-XXX, Table S-XXXI, Table S-XVIII): attacks/robust_auc.py
  * defenses (Table VII): defenses/run_defenses.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


from fraudadv.paths import LOGS  # noqa: E402


def ks(y, s, w=None, two_sided=False, log_dir=None):
    """Weighted Kolmogorov-Smirnov statistic of a score whose high values mean fraud:
    max over thresholds of TPR - FPR = max_s [F_legit(s) - F_fraud(s)], floored at 0.

    For a correctly oriented score this equals the usual KS, max |F_fraud - F_legit|. The two differ when part
    of the fraud distribution has been pushed below the legitimate one, which is what an attacker does: the
    two-sided statistic then grows with the damage (frauds ranked below every legitimate transaction give
    AUC 0 and a two-sided KS of 1), although a deployed detector cannot flip its decision rule. The paper
    reports the one-sided value; two_sided=True returns the two-sided one.

    Whenever the two values differ, one line is appended to <log_dir>/ks_diff.log (log_dir defaults to the
    study's logs folder, fraudadv.paths.LOGS; nothing is written when the folder does not exist). That log
    lists the calls where the choice of definition changes the reported number."""
    if log_dir is None:
        log_dir = str(LOGS)
    y = np.asarray(y)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    order = np.argsort(s, kind="mergesort")
    s, y, w = np.asarray(s)[order], y[order], w[order]
    w_fraud = np.where(y == 1, w, 0.0)
    w_legit = np.where(y == 0, w, 0.0)
    cdf_fraud, cdf_legit = np.cumsum(w_fraud) / w_fraud.sum(), np.cumsum(w_legit) / w_legit.sum()
    last = np.r_[s[1:] != s[:-1], True]              # evaluate only at distinct score values
    gap = cdf_legit[last] - cdf_fraud[last]
    one, two = float(max(np.max(gap), 0.0)), float(np.max(np.abs(gap)))
    if two - one > 1e-9 and os.path.isdir(log_dir):
        with open(os.path.join(log_dir, "ks_diff.log"), "a") as f:
            f.write(f"{one:.6f} {two:.6f} n={len(y)} {' '.join(sys.argv[:4])}\n")
    return two if two_sided else one


def auc(y, s, w=None):
    """Weighted area under the ROC curve."""
    return float(roc_auc_score(y, s, sample_weight=w))


def auprc(y, s, w=None):
    """Weighted area under the precision-recall curve (average precision); the tuning objective."""
    return float(average_precision_score(y, s, sample_weight=w))


def threshold_at_fpr(y, s, w, fpr=0.01):
    """Score threshold whose weighted false-positive rate on (y, s, w) is `fpr`.

    Legitimate rows are sorted from the highest score down, and the threshold is the score of the first row at
    which their cumulative weight share passes `fpr`. A row counts as flagged when its score is above it."""
    neg = y == 0
    order = np.argsort(-s[neg])
    cum_share = np.cumsum(w[neg][order]) / w[neg].sum()
    k = np.searchsorted(cum_share, fpr, side="right")
    return float(s[neg][order][min(k, len(order) - 1)])


def recall_at(y, s, w, thr):
    """Weighted share of frauds whose score is above the threshold `thr`."""
    pos = y == 1
    return float(np.sum(w[pos] * (s[pos] > thr)) / np.sum(w[pos]))


def all_metrics(y, s, w, amt, thr):
    """AUC, KS, AUPRC and recall at threshold `thr`, each unit-weighted (w) and dollar-weighted (w * amt)."""
    w_usd = w * amt
    return {
        "auc": auc(y, s, w), "auc_usd": auc(y, s, w_usd),
        "ks": ks(y, s, w), "ks_usd": ks(y, s, w_usd),
        "auprc": auprc(y, s, w), "auprc_usd": auprc(y, s, w_usd),
        "recall@1%fpr": recall_at(y, s, w, thr), "recall_usd@1%fpr": recall_at(y, s, w_usd, thr),
    }
