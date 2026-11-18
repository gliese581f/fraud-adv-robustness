"""Statistical tests behind the paper's pooled claims.

What it does
  Small, self-contained test functions. They take arrays or data frames that the report layer has already built
  from the attack grid and return p-values, intervals, ranks or coefficient tables. Nothing here reads or writes
  files.

Functions and where the paper uses them (all called from report/make_report.py: stats_tables, unless noted)
  wilson_ci          95% Wilson interval of an attack success rate (ASR = evaded / detected frauds).
                     reports/table6_worstcase_ci.csv; also the intervals of other make_report tables.
  friedman_nemenyi   Friedman test over models, with mean ranks and the Nemenyi critical difference.
                     reports/table6_friedman.csv -> Table S-XXXIX, Section VI-L.
  pairwise_mcnemar   exact McNemar test for every pair of models on the frauds both detect, Holm-adjusted.
                     reports/table6_mcnemar.csv -> Section VI-F and VI-L.
  mixed_model        Bayesian binomial mixed model of per-fraud evasion.
                     reports/table6_mixed_model*.csv -> Table S-XXXVIII, Section VI-L.
  mcnemar, holm      the exact McNemar p-value and the Holm correction used by pairwise_mcnemar.

The per-example arrays that pairwise_mcnemar reads are the grid's per_example*.npz entries, keyed
"<family>|<threat model>|<eps>", one row per attacked fraud with columns [test row, detected, evaded, amount].
"""
from __future__ import annotations

import numpy as np


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray):
    """Exact McNemar test on paired 0/1 outcomes of two models over the same items.

    Only the discordant pairs count: a_only = items where only model A is correct, b_only = only model B. Under
    the null hypothesis each discordant pair is a fair coin, so the p-value is a two-sided binomial test.
    Returns 1.0 when there are no discordant pairs."""
    from scipy.stats import binomtest

    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    return binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k / n (here: evaded frauds over detected frauds).
    Returns (low, high), clipped to [0, 1]; (nan, nan) when n is 0."""
    if n == 0:
        return (np.nan, np.nan)
    p_hat = k / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half_width = z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def holm(pvals):
    """Holm step-down adjusted p-values, returned in the order the p-values were given.

    Sort the p-values ascending; the r-th smallest (r counted from 0) is multiplied by (m - r), where m is the
    number of tests, and the adjusted values are made non-decreasing along the sorted order and capped at 1."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running_max)
    return adjusted


# Critical values q_0.05 / sqrt(2) of the Nemenyi test, by number of compared models (Demsar 2006, Table 5)
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def friedman_nemenyi(M):
    """Friedman test with Nemenyi critical difference.

    M has one row per block (a dataset and threat setting) and one column per treatment (a model); values are
    attack success rates, so lower means more robust and rank 1 is the most robust model of that block.
    Returns (Friedman p-value, mean rank of each model, Nemenyi critical difference). Two models whose mean ranks
    differ by more than the critical difference differ significantly at the 5% level. The p-value is nan when
    there is only one block or fewer than three models."""
    from scipy.stats import friedmanchisquare, rankdata

    M = np.asarray(M, float)
    ranks = np.vstack([rankdata(row) for row in M])
    n_models, n_blocks = M.shape[1], M.shape[0]
    p = friedmanchisquare(*M.T).pvalue if n_blocks > 1 and n_models > 2 else np.nan
    critical_difference = _Q05.get(n_models, np.nan) * np.sqrt(n_models * (n_models + 1) / (6.0 * n_blocks))
    return p, ranks.mean(0), critical_difference


def pairwise_mcnemar(per_example: dict, families, mode, eps):
    """Compare the worst-case evasion of every pair of models on the same frauds.

    per_example maps "<family>|<mode>|<eps>" to the grid's per-example array (columns: test row, detected,
    evaded, amount). For each pair (a, b) the frauds are matched by test row and only frauds that BOTH models
    detect when clean are kept. Returns one dict per pair: n (frauds kept), asr_a, asr_b, a_only and b_only (frauds
    only that model lets through), the exact McNemar p and its Holm-adjusted value over all pairs."""
    rows = []
    for i, a in enumerate(families):
        for b in families[i + 1:]:
            key_a, key_b = f"{a}|{mode}|{eps}", f"{b}|{mode}|{eps}"
            if key_a not in per_example or key_b not in per_example:
                continue
            A, B = per_example[key_a], per_example[key_b]
            _, rows_a, rows_b = np.intersect1d(A[:, 0], B[:, 0], return_indices=True)   # same frauds, matched by test row
            both_detect = (A[rows_a, 1] == 1) & (B[rows_b, 1] == 1)
            evaded_a, evaded_b = A[rows_a, 2][both_detect] == 1, B[rows_b, 2][both_detect] == 1
            rows.append(dict(model_a=a, model_b=b, n=int(both_detect.sum()),
                             asr_a=float(evaded_a.mean()) if both_detect.any() else np.nan,
                             asr_b=float(evaded_b.mean()) if both_detect.any() else np.nan,
                             a_only=int((evaded_a & ~evaded_b).sum()), b_only=int((~evaded_a & evaded_b).sum()),
                             p=mcnemar(~evaded_a, ~evaded_b)))
    if rows:
        adjusted = holm([r["p"] for r in rows])
        for r, q in zip(rows, adjusted):
            r["p_holm"] = float(q)
    return rows


def mixed_model(df):
    """Bayesian binomial mixed model of per-fraud evasion (detected frauds only):

        evaded ~ model + threat setting + (1 | dataset)

    fitted by variational Bayes (statsmodels BinomialBayesMixedGLM). Reference levels: XGBoost and the constrained
    setting, so each model coefficient is its log-odds of evasion relative to XGBoost.

    df columns: evaded (0/1), model, mode, dataset. Returns a table with one row per fixed-effect term: posterior
    mean (log_odds), posterior sd and the 95% credible interval (mean +- 1.96 sd). Returns an empty table when
    df holds a single dataset (the random effect cannot be fitted)."""
    import pandas as pd
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    if df["dataset"].nunique() < 2:
        return pd.DataFrame()
    model = BinomialBayesMixedGLM.from_formula("evaded ~ C(model, Treatment('xgboost')) + C(mode, Treatment('constrained'))",
                                               {"dataset": "0 + C(dataset)"}, df)
    fit = model.fit_vb()
    names = model.exog_names
    k = len(names)
    mean, sd = fit.fe_mean[:k], fit.fe_sd[:k]
    return pd.DataFrame({"term": names, "log_odds": mean, "sd": sd, "lo95": mean - 1.96 * sd, "hi95": mean + 1.96 * sd})
