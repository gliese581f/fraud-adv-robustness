"""Result tables that need a computation beyond the summary CSVs of fraudadv.report.make_report, written as CSV
files to reports/tables/. Every other result table is a make_report CSV (see RESULTS.md).

Inputs: data/processed/<d>/meta.json, runs/<d>/ (best_params.json, hpo_v2.json, metrics.json, ablation_noemb.json,
attacks/sample.npz, attacks/budget_check.csv, explain/global_importance.csv, explain/attack_shift.csv), the attack
grid (make_report.load_attacks) and the make_report CSVs table9_robust_auc.csv, table10_history.csv and
table11_halves.csv.

Outputs, one function each (reports/tables/<file>):
  datasets              datasets.csv            Table II: rows, frauds, features, controllable features, fraud sample
  hpo_trials            hpo_trials.csv          Table S-II: finished hyperparameter trials per dataset and family
  grid_per_dataset      grid_per_dataset.csv    Tables S-V to S-X: ASR by dataset, setting, attack, model and budget,
                                                with robust AUC and KS
  robust_auc_mean       robust_auc_mean.csv     Table S-XXX: change in AUC and KS under attack, constrained, mean over
                                                datasets
  halves_by_dataset     halves_by_dataset.csv   Table S-XL (top rows): the two halves of each fraud sample, mean over
                                                the models of each dataset
  drivers               drivers.csv             Table S-XXXIII: each model's three largest features and the three whose
                                                attribution the attack changes most
  budget_check          budget_check.csv        Table S-XXI: the grid against five-times stronger APGD and Square
  foundation_models     foundation_models.csv   Table V: the foundation models beside the MLP and the HGNN

Rates are shares in [0, 1]; AUC and KS are on their usual scale.

usage: python -m fraudadv.report.tables
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fraudadv.paths import PROC
from fraudadv.report.make_report import DS, DS_NAME, FAM, FAM_NAME, REP, RUNS, load_attacks

OUT = REP / "tables"
EPS = 0.25                                            # the main budget
WORST = "worst case (all attacks)"
SETTINGS = ("unconstrained", "constrained")
HISTORY_ENTITY = {"ulb": "", "baf": "", "ieee_cis": "client id", "sparkov": "card", "tabformer": "card",
                  "paysim": "destination account"}
GRAPH_RELATIONS = {"ulb": "", "baf": "", "ieee_cis": "client, card1", "sparkov": "card, merchant",
                   "tabformer": "card, merchant", "paysim": "account"}


def _write(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    return df


def _family_of_name():
    """Display name (FAM_NAME value, as in the make_report CSVs) -> family key."""
    return {v: k for k, v in FAM_NAME.items()}


def _worst_case_at_eps(A):
    """The worst case over all attacks at the main budget, one row per dataset, model and setting."""
    return A[(A["attack"] == WORST) & (A["eps"] == EPS)]


def datasets():
    """Table II: population rows and frauds over the three out-of-time splits, test frauds, features, attacker-
    controllable features (constrained threat model), the history entity and graph relations, and the size of the
    shared fraud sample every model is attacked on."""
    rows = []
    for d in DS:
        meta_path = PROC / d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        sizes = meta["sizes"]
        n_rows = sum(sizes[split]["pop_rows"] for split in sizes)
        n_frauds = sum(sizes[split]["fraud"] for split in sizes)
        sample_path = RUNS / d / "attacks" / "sample.npz"
        rows.append(dict(dataset=d, rows=n_rows, frauds=n_frauds, fraud_rate=n_frauds / n_rows,
                         test_frauds=sizes["test"]["fraud"], features=len(meta["features"]),
                         controllable=len(meta["mutable"]), history_entity=HISTORY_ENTITY[d],
                         graph_relations=GRAPH_RELATIONS[d],
                         fraud_sample=int(len(np.load(sample_path)["S"])) if sample_path.exists() else None))
    return _write(pd.DataFrame(rows), "datasets.csv")


def hpo_trials():
    """Table S-II: finished hyperparameter trials (complete or pruned) of the search that produced each final model;
    a second tuning pass (hpo_v2.json), where there was one, gives the count."""
    rows = []
    for d in DS:
        for f in FAM:
            params_path = RUNS / d / f / "best_params.json"
            if not params_path.exists():
                continue
            n = json.loads(params_path.read_text()).get("n_trials")
            second_pass = RUNS / d / f / "hpo_v2.json"
            if second_pass.exists():
                n = json.loads(second_pass.read_text()).get("n_trials") or n
            rows.append(dict(dataset=d, model=f, n_trials=n))
    return _write(pd.DataFrame(rows), "hpo_trials.csv")


def grid_per_dataset(A):
    """Tables S-V to S-X: every dataset, setting, attack, model and budget of the grid, with robust AUC and KS
    under the minimum-effort attack."""
    cols = ["dataset", "mode", "attack", "model", "eps", "asr", "asr_usd", "robust_auc", "robust_ks"]
    g = A[A["eps"].isin([0.05, 0.1, 0.25, 0.5, 1.0])]
    g = g[[c for c in cols if c in g.columns]].sort_values(["dataset", "mode", "attack", "model", "eps"])
    return _write(g, "grid_per_dataset.csv")


def robust_auc_mean():
    """Table S-XXX: change in AUC and KS under the minimum-effort and the score-minimizing attacker, constrained,
    eps 0.25, mean over datasets per model, from reports/table9_robust_auc.csv (per dataset: that file itself)."""
    p = REP / "table9_robust_auc.csv"
    if not p.exists():
        return None
    T = pd.read_csv(p)
    T = T[T["Setting"] == "constrained"].copy()
    T["model"] = T["Model"].map(_family_of_name())
    cols = ["ASR (min-effort)", "ΔAUC min-effort", "ΔAUC score-min", "ΔAUC ($) score-min", "ΔKS min-effort",
            "ΔKS score-min", "ΔKS ($) score-min"]
    M = T.groupby("model")[cols].mean()
    M = M.loc[[f for f in FAM if f in M.index]].reset_index()
    return _write(M, "robust_auc_mean.csv")


def halves_by_dataset():
    """Table S-XL, top rows: share of the sampled frauds detected (constrained models) and worst-case ASR at
    eps 0.25 on the first and second half of each entity-ordered fraud sample, mean over the models of each
    dataset, from reports/table11_halves.csv (per model, as in the IEEE-CIS rows: that file)."""
    p = REP / "table11_halves.csv"
    if not p.exists():
        return None
    T = pd.read_csv(p)
    T["det_first"], T["det_second"] = T["detected_first"] / T["n_first"], T["detected_second"] / T["n_second"]
    P = T.pivot_table(index=["dataset", "model"], columns="mode", values=["asr_first", "asr_second"])
    P.columns = [f"{a}_{m}" for a, m in P.columns]
    D = T[T["mode"] == "constrained"].set_index(["dataset", "model"])[["det_first", "det_second"]]
    P = P.join(D).reset_index()
    cols = ["det_first", "det_second", "asr_first_constrained", "asr_second_constrained", "asr_first_unconstrained",
            "asr_second_unconstrained"]
    M = P.groupby("dataset")[cols].mean()
    M = M.loc[[d for d in DS if d in M.index]].reset_index()
    return _write(M, "halves_by_dataset.csv")


def drivers():
    """Table S-XXXIII: per dataset and model, the three features with the largest mean absolute attribution over
    1,000 test transactions, and per setting the three whose attribution changes most between each detected fraud
    and its high-confidence adversarial example at eps 0.5 (empty: no successful attack to explain)."""
    rows = []
    for d in DS:
        g, q = RUNS / d / "explain" / "global_importance.csv", RUNS / d / "explain" / "attack_shift.csv"
        if not g.exists():
            continue
        G = pd.read_csv(g, index_col=0)
        Q = pd.read_csv(q) if q.exists() else pd.DataFrame(columns=["model", "mode", "feature", "mean_abs_dattr", "n"])
        for f in FAM:
            if f not in G.columns:
                continue
            row = dict(dataset=d, model=f, drivers=", ".join(str(x) for x in G[f].abs().sort_values(ascending=False).index[:3]))
            for mode in ("constrained", "unconstrained"):
                h = Q[(Q["model"] == f) & (Q["mode"] == mode)].dropna(subset=["mean_abs_dattr"])
                h = h[h["mean_abs_dattr"] > 0]
                row[f"shift_{mode}"] = ", ".join(str(x) for x in h.sort_values("mean_abs_dattr", ascending=False)["feature"][:3])
            rows.append(row)
    return _write(pd.DataFrame(rows), "drivers.csv")


def budget_check():
    """Table S-XXI: worst-case ASR at eps 0.25 of the grid and of the grid plus APGD with five times the iterations
    from new random starts and Square with five times the queries and two seeds, on the same detected frauds, from
    runs/<d>/attacks/budget_check.csv (fraudadv.attacks.budget_check)."""
    fr = [pd.read_csv(p) for p in sorted(RUNS.glob("*/attacks/budget_check.csv")) if p.stat().st_size > 1]
    if not fr:
        return None
    B = pd.concat(fr, ignore_index=True)
    B["stronger_evades_more"] = (np.round(100 * B["asr_strong"]) != np.round(100 * B["asr_grid"])) | (
        B.get("new_evasions", pd.Series(0, index=B.index)) > 0)
    return _write(B, "budget_check.csv")


def _mean_clean_metrics(d, f, name="metrics.json"):
    """Mean test AUC and AUPRC over seeds from runs/<d>/<f>/<name>, or None if the file is missing."""
    q = RUNS / d / f / name
    if not q.exists():
        return None
    m = json.loads(q.read_text())
    return {k: float(np.mean([r[k] for r in m])) for k in ("auc", "auprc")}


def foundation_models(A):
    """Table V: the foundation models beside the MLP (same tabular features, no history encoder) and the HGNN
    (history as a graph): mean test AUC and AUPRC over seeds, AUPRC of the same head with the encoding set to zero
    (runs/<d>/<fm>/ablation_noemb.json, fraudadv.hpo.fm_ablation), worst-case ASR at eps 0.25 in both settings, and
    history injection with eight transactions, best of four amounts (reports/table10_history.csv)."""
    W = _worst_case_at_eps(A) if not A.empty else A
    p = REP / "table10_history.csv"
    H = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if not H.empty:
        H = H.assign(fam=H["Model"].map(_family_of_name()))
    rows = []
    for d in [d for d in DS if (RUNS / d / "chronos2" / "metrics.json").exists()]:
        for f in [f for f in ("mlp", "hgnn", "chronos2", "timesfm") if (RUNS / d / f / "metrics.json").exists()]:
            c = _mean_clean_metrics(d, f)
            no_enc = _mean_clean_metrics(d, f, "ablation_noemb.json") if f in ("chronos2", "timesfm") else None
            row = dict(dataset=d, model=f, auc=c["auc"], auprc=c["auprc"], auprc_no_encoding=no_enc["auprc"] if no_enc else None)
            for m in SETTINGS:
                v = W[(W["dataset"] == d) & (W["mode"] == m) & (W["model"] == f)]["asr"] if not W.empty else []
                row[f"asr_{m}"] = float(v.iloc[0]) if len(v) else None
            h = None
            if not H.empty:
                g = H[(H["Dataset"] == DS_NAME[d]) & (H["fam"] == f) & (H["attack"].str.startswith("best"))]
                h = float(g["asr@k=8"].iloc[0]) if len(g) else None
            row["history_asr_k8"] = h
            rows.append(row)
    return _write(pd.DataFrame(rows), "foundation_models.csv")


def main():
    A = load_attacks()
    datasets()
    hpo_trials()
    if not A.empty:
        grid_per_dataset(A)
    robust_auc_mean()
    halves_by_dataset()
    drivers()
    budget_check()
    foundation_models(A)
    print("wrote", sorted(p.name for p in OUT.glob("*.csv")))


if __name__ == "__main__":
    main()
