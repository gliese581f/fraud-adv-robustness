"""Causal analysis of the architecture ablations (fraudadv.causal.ablate): Table VI and Tables S-XXXIV to S-XXXVI,
Figures S13 and S14.

Inputs: runs/<dataset>/causal/out_<variant>_seed<s>_<mode>.npz (per-fraud outcomes of every ablation variant) and
clean_<variant>_seed<s>.json (clean metrics) on Sparkov, TabFormer, IEEE-CIS and BAF.
Outputs: the summary CSVs in reports/causal/ (per_fraud_outcomes.csv.gz, contrasts.csv, factorial_lpm.csv,
clean_metrics.csv; Table VI and Table S-XXXV are the "pooled" rows of contrasts.csv, Table S-XXXVI is
factorial_lpm.csv), then fraudadv.causal.report.write_all adds variants_clean.csv (Table S-XXXIV),
factorial_effects.csv and Figures S13 and S14.

Estimands, per intervention (treated variant vs its control, same seed, same frauds, same attacks and budgets):
  d_asr     change in attack success on the frauds BOTH models detect (common support), in percentage points;
            conditioning on detection by both avoids comparing different sets of frauds
  d_rr      change in robust recall (share of all attacked frauds still flagged after the attack), which also
            carries the change in clean recall; no conditioning on a post-treatment variable
  d_recall  change in clean recall at 1% FPR on the attacked frauds
  d_cost    change in the median minimal L-inf evasion distance found by HopSkipJump (common support, frauds the
            attack does not evade within eps 1 count as 1.0, a censored value)
Each estimate is the mean over seeds of the paired per-fraud differences; pooled = mean of the four dataset
estimates. 95% intervals: percentile bootstrap over frauds (stratified by dataset, seeds kept paired), 2,000 draws.

Factorial model (the 2 x 2 representation x learned-function design, plus attack type and threat setting): a linear
probability model of evasion on representation (neural), learned function (neural-learned), their interaction,
attack type and setting and their interactions with both factors, dataset fixed effects, on frauds that all four
models of a seed detect, with standard errors clustered by fraud. Coefficients are in percentage points.

usage: python -m fraudadv.causal.analyze      writes reports/causal/*.csv and the figures
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fraudadv.attacks.run_attacks import RUNS
from fraudadv.causal.ablate import VARIANTS, is_base
from fraudadv.report.make_report import DS_NAME, REP

CDS = ["sparkov", "tabformer", "ieee_cis", "baf"]                 # datasets of the ablation
ATT = {"wb": "white-box", "sq": "Square (score)", "hsj": "HopSkipJump (decision)"}
EPS_H = 0.25                                                         # budget of every estimate
# (key, label, treated variant, control variant, group); group: "class" = the 2 x 2 distillation design,
# "trees" / "neural" = one factor changed within XGBoost / within the MLP
CONTRASTS = [
    ("total", "MLP vs XGBoost, both tuned (total effect)", "mlp", "xgb", "class"),
    ("rep_treefn", "Representation tree $\\to$ neural, tree-learned function", "mlp_from_xgb", "xgb", "class"),
    ("rep_nnfn", "Representation tree $\\to$ neural, neural-learned function", "mlp", "xgb_from_mlp", "class"),
    ("fn_tree", "Learned function tree $\\to$ neural, tree representation", "xgb_from_mlp", "xgb", "class"),
    ("fn_nn", "Learned function tree $\\to$ neural, neural representation", "mlp", "mlp_from_xgb", "class"),
    ("xgb_d2", "XGBoost: depth 2", "xgb_d2", "xgb", "trees"),
    ("xgb_d6", "XGBoost: depth 6", "xgb_d6", "xgb", "trees"),
    ("xgb_d10", "XGBoost: depth 10", "xgb_d10", "xgb", "trees"),
    ("xgb_mono", "XGBoost: monotone constraints", "xgb_mono", "xgb", "trees"),
    ("mlp_l1", "MLP: 1 hidden layer", "mlp_l1", "mlp", "neural"),
    ("mlp_l4", "MLP: 4 hidden layers", "mlp_l4", "mlp", "neural"),
    ("mlp_relu", "MLP: ReLU instead of SiLU", "mlp_relu", "mlp", "neural"),
    ("mlp_sn", "MLP: spectral normalisation", "mlp_sn", "mlp", "neural"),
    ("mlp_wd", "MLP: weight decay 0.1", "mlp_wd", "mlp", "neural"),
]


# depth / layer contrasts whose level can equal the tuned base on some dataset; there they are no contrast
LEVEL_CONTRASTS = ("xgb_d10", "mlp_l1", "mlp_l4", "xgb_d6")


def _stored_variant(ds, v):
    """Name under which variant v's files are stored: a variant equal to the tuned base is the base (xgb or mlp)."""
    return ("xgb" if VARIANTS[v][0] == "xgboost" else "mlp") if is_base(ds, v) else v


def _out(ds, v, s, md):
    """Per-fraud outputs of variant v, seed s, setting md (None if missing); a variant equal to the tuned base
    reads the base's outputs."""
    p = RUNS / ds / "causal" / f"out_{_stored_variant(ds, v)}_seed{s}_{md}.npz"
    return dict(np.load(p)) if p.exists() else None


def outcomes(z, att, eps=EPS_H):
    """Per-fraud arrays (rows, detected, evaded, cost) for one attack."""
    tau, det = float(z["tau"]), z["det"].astype(bool)
    if att == "hsj":
        H = z["hsj_rows"]
        l = z[f"hsj_l_{eps}"]
        return H, det[H], det[H] & (l < tau), np.minimum(z["hsj_cost"], 1.0)
    l = z[f"{att}_l_{eps}"]
    cost = np.minimum(z["wb_cost"], 1.0) if att == "wb" else np.full(len(det), np.nan)
    return np.arange(len(det)), det, det & (l < tau), cost


def per_fraud_table(seeds=(0, 1, 2), eps=EPS_H):
    """One row per (dataset, variant, seed, setting, attack, fraud): detected, evaded at eps, L-inf cost, clean
    score quantile q0 and amount. Saved as reports/causal/per_fraud_outcomes.csv.gz."""
    rows = []
    for ds in CDS:
        for v in VARIANTS:
            for s in seeds:
                for md in ("constrained", "unconstrained"):
                    z = _out(ds, v, s, md)
                    if z is None:
                        continue
                    for att in ATT:
                        r, d, e, c = outcomes(z, att, eps)
                        rows.append(pd.DataFrame(dict(dataset=ds, variant=v, seed=s, mode=md, attack=att, fraud=r, det=d,
                                                      evaded=e, cost=c, q0=z["q0"][r], amt=z["amt"][r])))
    return pd.concat(rows, ignore_index=True)


def _mean(x):
    """Mean of the finite values of x (NaN if none)."""
    x = np.asarray(x, float)
    return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan


def _paired(T, ds, tr, ct, md, att):
    """Aligned (seeds x frauds) arrays for treated and control."""
    a = T[(T.dataset == ds) & (T["mode"] == md) & (T.attack == att)]
    A, B = a[a.variant == tr], a[a.variant == ct]
    seeds = sorted(set(A.seed) & set(B.seed))
    if not seeds:
        return None
    get = lambda X, col: np.stack([X[X.seed == s].sort_values("fraud")[col].to_numpy(float) for s in seeds])
    return {k: (get(A, k), get(B, k)) for k in ("det", "evaded", "cost")}, len(seeds)


def _stats(P, idx):
    """Effects for one bootstrap draw of fraud indices (seeds x frauds arrays)."""
    (dt, dc), (et, ec), (ct, cc) = P["det"], P["evaded"], P["cost"]
    dt, dc, et, ec, ct, cc = dt[:, idx], dc[:, idx], et[:, idx], ec[:, idx], ct[:, idx], cc[:, idx]
    both = (dt > 0) & (dc > 0)
    n = both.sum(1)
    asr = np.where(n > 0, ((et - ec) * both).sum(1) / np.maximum(n, 1), np.nan)
    rr = ((dt - et) - (dc - ec)).mean(1)
    rec = (dt - dc).mean(1)
    out = dict(d_asr=np.nanmean(asr) * 100, d_rr=np.mean(rr) * 100, d_recall=np.mean(rec) * 100, n_cs=float(n.mean()))
    if np.isfinite(ct).any() and np.isfinite(cc).any():
        med = [np.nanmedian(np.where(b, x, np.nan)) - np.nanmedian(np.where(b, y, np.nan)) for x, y, b in zip(ct, cc, both) if b.any()]
        out["d_cost"] = float(np.mean(med)) if med else np.nan
    return out


def _boot(P, I):
    """Vectorised bootstrap of (d_asr, d_rr, d_recall): I is (draws x frauds) indices; returns (draws x 3)."""
    (dt, dc), (et, ec) = P["det"], P["evaded"]
    dt, dc, et, ec = dt[:, I], dc[:, I], et[:, I], ec[:, I]          # seeds x draws x frauds
    both = (dt > 0) & (dc > 0)
    n = both.sum(2)
    with np.errstate(invalid="ignore"):
        asr = np.nanmean(np.where(n > 0, ((et - ec) * both).sum(2) / np.maximum(n, 1), np.nan), 0)
    rr = ((dt - et) - (dc - ec)).mean(2).mean(0)
    rec = (dt - dc).mean(2).mean(0)
    return np.stack([asr, rr, rec], 1) * 100


def contrasts(T, n_boot=2000, eps=EPS_H):
    """Every contrast of CONTRASTS per setting and attack: one row per dataset and a pooled row (mean of the
    dataset estimates), with 95% percentile-bootstrap intervals over frauds. Saved as reports/causal/contrasts.csv."""
    rng = np.random.default_rng(0)
    res = []
    for key, label, tr, ct, grp in CONTRASTS:
        for md in ("constrained", "unconstrained"):
            for att in ATT:
                per_ds, boots = {}, {}
                for ds in CDS:
                    if key in LEVEL_CONTRASTS and is_base(ds, key):
                        continue                                     # the level equals the tuned base: no contrast
                    pr = _paired(T, ds, tr, ct, md, att)
                    if pr is None:
                        continue
                    P, ns = pr
                    nf = P["det"][0].shape[1]                        # frauds attacked on this dataset
                    est = _stats(P, np.arange(nf))
                    per_ds[ds] = est
                    I = rng.integers(0, nf, size=(n_boot, nf))       # bootstrap draws of fraud indices
                    boots[ds] = pd.DataFrame(_boot(P, I), columns=["d_asr", "d_rr", "d_recall"])
                    res.append(dict(contrast=key, label=label, group=grp, mode=md, attack=att, dataset=ds, n_seeds=ns, **est,
                                    **{f"{k}_lo": boots[ds][k].quantile(0.025) for k in ("d_asr", "d_rr", "d_recall")},
                                    **{f"{k}_hi": boots[ds][k].quantile(0.975) for k in ("d_asr", "d_rr", "d_recall")}))
                if per_ds:                                           # pooled: mean of the dataset estimates and draws
                    pooled = {k: _mean([e.get(k, np.nan) for e in per_ds.values()]) for k in ("d_asr", "d_rr", "d_recall", "d_cost", "n_cs")}
                    B = sum(b[["d_asr", "d_rr", "d_recall"]].to_numpy() for b in boots.values()) / len(boots)
                    res.append(dict(contrast=key, label=label, group=grp, mode=md, attack=att, dataset="pooled",
                                    n_datasets=len(per_ds), **pooled,
                                    **{f"{k}_lo": float(np.nanquantile(B[:, i], 0.025)) for i, k in enumerate(("d_asr", "d_rr", "d_recall"))},
                                    **{f"{k}_hi": float(np.nanquantile(B[:, i], 0.975)) for i, k in enumerate(("d_asr", "d_rr", "d_recall"))}))
        print("contrast", key, "done", flush=True)
    return pd.DataFrame(res)


def factorial(T):
    """Linear probability model on the 2 x 2 representation x function design, common support of all four models,
    without and with the clean margin. Returns (coefficients, the frame it was fitted on); the coefficients are
    saved as reports/causal/factorial_lpm.csv."""
    import statsmodels.formula.api as smf

    four = ["xgb", "mlp", "xgb_from_mlp", "mlp_from_xgb"]
    D = T[T.variant.isin(four)].copy()
    key = ["dataset", "seed", "mode", "attack", "fraud"]
    cs = D.groupby(key)["det"].agg(["sum", "count"]).reset_index()
    cs = cs[(cs["sum"] == 4) & (cs["count"] == 4)][key]
    D = D.merge(cs, on=key)
    D["neural_rep"] = D.variant.isin(["mlp", "mlp_from_xgb"]).astype(int)
    D["neural_fn"] = D.variant.isin(["mlp", "xgb_from_mlp"]).astype(int)
    D["unconstrained"] = (D["mode"] == "unconstrained").astype(int)
    D["ev"] = D.evaded.astype(float) * 100
    D["cluster"] = D.dataset + ":" + D.fraud.astype(str)
    D["attack"] = pd.Categorical(D.attack, ["wb", "sq", "hsj"])
    D["margin"] = -np.log10(np.clip(1 - D.q0, 1e-6, 1))              # depth of the clean score in the legit tail
    f = ("ev ~ neural_rep * C(attack) + neural_fn * C(attack) + neural_rep:neural_fn + neural_rep:unconstrained"
         " + neural_fn:unconstrained + unconstrained + C(attack):unconstrained" + (" + C(dataset)" if D.dataset.nunique() > 1 else ""))
    fits = {}
    codes = pd.factorize(D.cluster)[0]
    fits["main"] = smf.ols(f, D).fit(cov_type="cluster", cov_kwds={"groups": codes})
    fits["margin"] = smf.ols(f + " + margin", D).fit(cov_type="cluster", cov_kwds={"groups": codes})
    rows = []
    for name, m in fits.items():
        ci = m.conf_int()
        for k in m.params.index:
            if k.startswith("C(dataset)") or k == "Intercept":
                continue
            rows.append(dict(model=name, term=k, coef=m.params[k], lo=ci.loc[k, 0], hi=ci.loc[k, 1], p=m.pvalues[k], n=int(m.nobs)))
    return pd.DataFrame(rows), D


def clean_table():
    """Clean metrics of every variant and seed (clean_<variant>_seed<s>.json), saved as reports/causal/clean_metrics.csv."""
    rows = []
    for ds in CDS:
        for v in VARIANTS:
            for s in (0, 1, 2):
                p = RUNS / ds / "causal" / f"clean_{_stored_variant(ds, v)}_seed{s}.json"
                if p.exists():
                    r = json.loads(p.read_text())
                    r["variant"] = v
                    rows.append(r)
    return pd.DataFrame(rows)


def main():
    """Write the summary CSVs of reports/causal/, print the pooled constrained effects and the factorial model,
    then write the tables and figures (fraudadv.causal.report)."""
    out = REP / "causal"
    out.mkdir(parents=True, exist_ok=True)
    T = per_fraud_table()
    T.to_csv(out / "per_fraud_outcomes.csv.gz", index=False)
    C = contrasts(T)
    C.to_csv(out / "contrasts.csv", index=False)
    F, D = factorial(T)
    F.to_csv(out / "factorial_lpm.csv", index=False)
    K = clean_table()
    K.to_csv(out / "clean_metrics.csv", index=False)
    print(C[(C.dataset == "pooled") & (C["mode"] == "constrained")][["contrast", "attack", "d_asr", "d_asr_lo", "d_asr_hi", "d_rr", "d_recall"]].round(1).to_string())
    print(F.round(2).to_string())
    from fraudadv.causal.report import write_all

    write_all(T, C, F, K)


if __name__ == "__main__":
    main()
