"""Tables that put the three attackers side by side: unconstrained (U), constrained (C) and consistent (K), as CSV
files in reports/tables/.

The consistent attacker is the constrained one with every query repaired into a record a real transaction could
produce (fraudadv.attacks.consistent). It has its own runs on PaySim, IEEE-CIS and Sparkov (runs_consistent/<d>/).
On ULB, TabFormer and BAF no constrained query is an impossible record (reports/analysis/consistent/realizable.csv),
so there K equals C and the K numbers are the C ones.

Inputs
  - U and C: the study's attack grid (fraudadv.report.make_report.load_attacks), history injection
    (runs/<d>/attacks/history*.csv), robust AUC (runs/<d>/attacks/robust_auc*.csv), defenses (runs/<d>/defenses).
  - K: runs_consistent/<d>/attacks/results_consistent_*.csv, history_consistent*.csv, robust_auc_consistent*.csv,
    transfer_consistent.npz and runs_consistent/<d>/defenses/results_consistent*.csv.
  - reports/analysis/consistent/realizable.csv (fraudadv.report.realizable_share).

Outputs (reports/tables/ unless noted)
  worst_case_three_attackers.csv  Table III: worst case at eps 0.25 under U, C and K, per dataset and model
  worst_case_rank.csv             Table III, last column: each model's mean normalized rank over the K columns
  consistent_summary.csv          C beside K per attack (mean over models), history injection, best defense
  consistent_grid.csv             every attack, model and budget on the K datasets, C beside K
  consistent_robust_auc.csv       score-minimizing attacker: change in AUC and KS, C beside K
  consistent_history.csv          history injection with the study's and with repaired warm-up records
  consistent_defenses.csv         defenses of the tabular models under C and K
  defenses_context.csv            Table S-XXIX: defenses of the graph and foundation models under C and K
  realizable_summary.csv          share of evasions that are impossible records, range over models
  reports/fig_transfer_consistent.{pdf,png}   the transfer figure for K

Rates are shares in [0, 1].

  python -m fraudadv.report.consistent_tables
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from fraudadv.paths import REPORTS, RUNS, RUNS_CONSISTENT
from fraudadv.report.make_report import DS, FAM, load_attacks

K_DATASETS = ["ieee_cis", "sparkov", "paysim"]       # the datasets with a separate consistent arm
EPS = 0.25                                           # the budget of the main tables
EPS_GRID = [0.05, 0.1, 0.25, 0.5, 1.0]
WORST = "worst case (all attacks)"
LETTER = {"unconstrained": "U", "constrained": "C", "consistent": "K"}
REALIZABLE_CSV = REPORTS / "analysis" / "consistent" / "realizable.csv"
OUT = REPORTS / "tables"
ATTACK_ORDER = ["white-box (PGD)", "white-box (AATS)", "white-box (AATS + exact MILP)", "score black-box (Square)",
                "decision black-box (HopSkipJump)", WORST]
DEFENSES = ["none", "pgd_at", "trades", "adv_aug", "adv_boost", "monotone", "squeeze"]
TABULAR_FAMS = ("xgboost", "catboost", "mlp", "ft_transformer", "ivis")


def _write(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    return df


# ---------------------------------------------------------------- reading the results

def read_all(paths):
    """Concatenate the non-empty CSV files among `paths` (None if there are none)."""
    frames = [pd.read_csv(p) for p in paths if p.stat().st_size > 1]
    return pd.concat(frames, ignore_index=True) if frames else None


def grid_consistent():
    """The consistent attacker's grid rows (same columns as the study's results.csv)."""
    K = read_all(p for d in K_DATASETS for p in sorted((RUNS_CONSISTENT / d / "attacks").glob("results_consistent_*.csv")))
    return K if K is not None else pd.DataFrame(columns=["dataset", "model", "mode", "attack", "eps", "asr"])


def grid_three_attackers():
    """All three attackers in one frame. On ULB, TabFormer and BAF the K rows are copies of the C rows."""
    A = load_attacks()
    A = A[A["mode"].isin(["unconstrained", "constrained"])]
    same_as_c = A[(A["mode"] == "constrained") & ~A["dataset"].isin(K_DATASETS)].assign(mode="consistent")
    return pd.concat([A, grid_consistent(), same_as_c], ignore_index=True)


def attackers_of(dataset):
    """The attacker columns shown for a dataset: K only where it differs from C."""
    return ("unconstrained", "constrained", "consistent") if dataset in K_DATASETS else ("unconstrained", "constrained")


def history_consistent():
    """History injection with repaired warm-up records (fraudadv.attacks.history_consistent)."""
    H = read_all(p for d in K_DATASETS for p in sorted((RUNS_CONSISTENT / d / "attacks").glob("history_consistent*.csv")))
    return H if H is not None else pd.DataFrame(columns=["dataset", "model", "attack", "k", "asr"])


def history_study():
    """The study's history injection (fraudadv.attacks.history), for the K datasets."""
    names = ("history.csv", "history_fm_chronos2.csv", "history_fm_timesfm.csv")
    paths = [RUNS / d / "attacks" / n for d in K_DATASETS for n in names]
    H = read_all(p for p in paths if p.exists())
    return H if H is not None else pd.DataFrame()


def defenses_consistent():
    """Worst case at eps 0.25 of each defended model under the consistent attacker."""
    D = read_all(p for d in K_DATASETS for p in sorted((RUNS_CONSISTENT / d / "defenses").glob("results_consistent*.csv")))
    if D is None:
        return pd.DataFrame(columns=["dataset", "model", "defense", "asr", "clean_auc"])
    return D[(D["attack"] == WORST) & np.isclose(D["eps"], EPS)].drop_duplicates(["dataset", "model", "defense"])


def defenses_study():
    """Worst case at eps 0.25 of each defended model under the study's constrained attacker."""
    D = read_all(p for d in K_DATASETS for p in sorted((RUNS / d / "defenses").glob("results*.csv")))
    if D is None:
        return pd.DataFrame()
    D = D[(D["attack"] == WORST) & np.isclose(D["eps"], EPS) & (D["mode"] == "constrained")]
    return D.drop_duplicates(["dataset", "model", "defense"], keep="last")


def robust_auc_c_and_k():
    """Score-minimising attacker rows at eps 0.25: the study's constrained ones and the consistent ones."""
    paths = [p for d in K_DATASETS for p in sorted((RUNS / d / "attacks").glob("robust_auc*.csv"))]
    paths += [p for d in K_DATASETS for p in sorted((RUNS_CONSISTENT / d / "attacks").glob("robust_auc_consistent*.csv"))]
    R = read_all(paths)
    if R is None or "consistent" not in set(R["mode"]):
        return pd.DataFrame()
    R = R[np.isclose(R["eps"], EPS) & R["mode"].isin(["constrained", "consistent"])]
    return R.drop_duplicates(["dataset", "model", "mode"], keep="last")


# ---------------------------------------------------------------- Table III

def worst_case(G):
    """Table III: worst-case ASR at eps 0.25 for every dataset, model and attacker (K only on the datasets where it
    differs from C), and each model's mean normalized rank (rank / number of models, 1 = least robust) over the six
    consistent columns (K, or C where they coincide)."""
    W = G[(G["attack"] == WORST) & np.isclose(G["eps"], EPS)]
    keep = [(d in K_DATASETS) or (m != "consistent") for d, m in zip(W["dataset"], W["mode"])]
    T = W[keep][["dataset", "model", "mode", "asr", "asr_usd"]].rename(columns={"mode": "attacker"})
    T = T.assign(attacker=T["attacker"].map(LETTER)).sort_values(["dataset", "model", "attacker"])
    _write(T, "worst_case_three_attackers.csv")
    ranks = {}
    for d in [d for d in DS if d in set(W["dataset"])]:
        g = W[(W["dataset"] == d) & (W["mode"] == "consistent")]
        for fam, r in g.set_index("model")["asr"].rank(method="average").items():
            ranks.setdefault(fam, []).append(r / len(g))
    R = pd.DataFrame([dict(model=f, mean_normalized_rank=float(np.mean(ranks[f])), columns=len(ranks[f]))
                      for f in FAM if f in ranks])
    _write(R, "worst_case_rank.csv")
    return T


# ---------------------------------------------------------------- the consistent attacker in detail

def summary(G):
    """C beside K at eps 0.25 on the K datasets: ASR per attack, mean over the models it applies to; history
    injection at k = 8 (best of four amounts), mean over models; each model's best (lowest) defended worst case,
    mean over models."""
    both = ("constrained", "consistent")
    A = G[G["dataset"].isin(K_DATASETS) & np.isclose(G["eps"], EPS) & G["mode"].isin(both)]
    rows = []
    for atk in ATTACK_ORDER:
        for d in K_DATASETS:
            for m in both:
                x = A[(A["dataset"] == d) & (A["mode"] == m) & (A["attack"] == atk)].drop_duplicates("model", keep="last")["asr"]
                rows.append(dict(result=atk, dataset=d, attacker=LETTER[m], mean_asr=x.mean() if len(x) else None, models=len(x)))
    hist_c, hist_k = history_study(), history_consistent()
    for d in K_DATASETS:
        for m, X in (("constrained", hist_c), ("consistent", hist_k)):
            x = X[(X["dataset"] == d) & X["attack"].str.startswith("best") & (X["k"] == 8)] if len(X) else X
            x = x.drop_duplicates("model", keep="last")["asr"] if len(x) else pd.Series(dtype=float)
            rows.append(dict(result="history injection, k=8", dataset=d, attacker=LETTER[m], mean_asr=x.mean() if len(x) else None,
                             models=len(x)))
    def_c, def_k = defenses_study(), defenses_consistent()
    for d in K_DATASETS:
        for m, X in (("constrained", def_c), ("consistent", def_k)):
            x = X[(X["dataset"] == d) & (X["defense"] != "none")] if len(X) else X
            x = x.groupby("model")["asr"].min() if len(x) else pd.Series(dtype=float)
            rows.append(dict(result="best defense", dataset=d, attacker=LETTER[m], mean_asr=x.mean() if len(x) else None, models=len(x)))
    return _write(pd.DataFrame(rows), "consistent_summary.csv")


def grid(G):
    """Every attack, model and budget on the K datasets, constrained (C) and consistent (K), with robust AUC and KS
    under the minimum-effort attack. K can exceed C only through the search (a different path, or HopSkipJump's
    unseeded start), since the consistent set lies inside the constrained one."""
    g = G[G["dataset"].isin(K_DATASETS) & G["eps"].isin(EPS_GRID) & G["mode"].isin(["constrained", "consistent"])]
    have_k = set(zip(g[g["mode"] == "consistent"]["dataset"], g[g["mode"] == "consistent"]["model"],
                     g[g["mode"] == "consistent"]["attack"]))
    g = g[[k in have_k for k in zip(g["dataset"], g["model"], g["attack"])]]
    cols = ["dataset", "attack", "model", "mode", "eps", "asr", "robust_auc", "robust_ks"]
    g = g[[c for c in cols if c in g.columns]].rename(columns={"mode": "attacker"})
    g = g.assign(attacker=g["attacker"].map(LETTER)).sort_values(["dataset", "attack", "model", "attacker", "eps"])
    return _write(g, "consistent_grid.csv")


def robust_auc():
    """Change in AUC and KS (unit- and dollar-weighted) under the score-minimising attacker at eps 0.25,
    constrained (C) and consistent (K), on the same fraud subsets."""
    R = robust_auc_c_and_k()
    if R.empty:
        return R
    R = R[[(d, f) in set(zip(R[R["mode"] == "consistent"]["dataset"], R[R["mode"] == "consistent"]["model"]))
           for d, f in zip(R["dataset"], R["model"])]]
    out = R[["dataset", "model"]].assign(attacker=R["mode"].map(LETTER))
    for attacked, clean, name in (("robust_auc", "clean_auc", "d_auc"), ("robust_auc_usd", "clean_auc_usd", "d_auc_usd"),
                                  ("robust_ks", "clean_ks", "d_ks"), ("robust_ks_usd", "clean_ks_usd", "d_ks_usd")):
        out[name] = R[attacked] - R[clean]
    return _write(out.sort_values(["dataset", "model", "attacker"]), "consistent_robust_auc.csv")


def history():
    """History injection, best of four amounts, after k = 1, 2, 4, 8 injected records: the study's records (S)
    beside the repaired ones (K: missing flags follow the values, and a record whose hour and category never occur
    together is built on another of the card's own transactions), and how many frauds needed a repair."""
    H, S = history_consistent(), history_study()
    if H.empty:
        return None
    rows = []
    for d in K_DATASETS:
        h = H[(H["dataset"] == d) & H["attack"].str.startswith("best")]
        s = S[(S["dataset"] == d) & S["attack"].str.startswith("best")] if not S.empty else S
        for fam in [f for f in FAM if f in set(h["model"])]:
            hk = h[h["model"] == fam].set_index("k")
            sk = s[s["model"] == fam].set_index("k") if not s.empty else pd.DataFrame()
            for k in (1, 2, 4, 8):
                rows.append(dict(dataset=d, model=fam, k=k,
                                 asr_study=sk["asr"].get(k) if len(sk) and k in sk.index else None,
                                 asr_consistent=hk["asr"].get(k) if k in hk.index else None,
                                 repaired=hk["repaired"].get(k) if "repaired" in hk and k in hk.index else None))
    return _write(pd.DataFrame(rows), "consistent_history.csv")


def defenses():
    """Defended worst case at eps 0.25 of the tabular models, the study's constrained attacker (C) beside the
    consistent one (K), per defense. The defended models are rebuilt with the study's recipe and seeds and trained
    against the constrained attacker; only the evaluation changes."""
    K, C = defenses_consistent(), defenses_study()
    if K.empty:
        return None
    rows = []
    for d in K_DATASETS:
        g = K[K["dataset"] == d]
        for fam in [f for f in TABULAR_FAMS if f in set(g["model"])]:
            for x in DEFENSES:
                c = C[(C["dataset"] == d) & (C["model"] == fam) & (C["defense"] == x)]["asr"] if not C.empty else pd.Series(dtype=float)
                k = g[(g["model"] == fam) & (g["defense"] == x)]
                if c.empty and k.empty:
                    continue
                rows.append(dict(dataset=d, model=fam, defense=x, asr_constrained=float(c.iloc[0]) if len(c) else None,
                                 asr_consistent=float(k["asr"].iloc[0]) if len(k) else None,
                                 clean_auc_consistent_run=float(k["clean_auc"].iloc[0]) if len(k) else None))
    return _write(pd.DataFrame(rows), "consistent_defenses.csv")


CONTEXT_FAMS = ("hgnn", "hgt", "chronos2", "timesfm")
CONTEXT_DATASETS = ["ieee_cis", "sparkov", "tabformer", "paysim"]     # the datasets with card or account histories


def defenses_context():
    """Table S-XXIX: the defenses of the graph and foundation models: worst-case ASR at eps 0.25 under the
    constrained attacker (C) and, where it differs, the consistent one (K), and clean AUC, per dataset. C and AUC
    from reports/table7_defenses_by_dataset.csv (make_report.defense_tables), K from runs_consistent. Graph models
    perturb the transaction's own features with its history fixed; the foundation models train their head with the
    frozen encoder's clean embedding and are attacked, as in the grid, on 240 random frauds of the sample."""
    p = REPORTS / "table7_defenses_by_dataset.csv"
    if not p.exists():
        return None
    C = pd.read_csv(p)
    C = C[C["model"].isin(CONTEXT_FAMS)]
    if C.empty:
        return None
    K = defenses_consistent()
    rows = []
    for fam in CONTEXT_FAMS:
        g = C[C["model"] == fam]
        for x in [x for x in DEFENSES if x in set(g["defense"])]:
            for d in [d for d in CONTEXT_DATASETS if d in set(C["dataset"])]:
                r = g[(g["dataset"] == d) & (g["defense"] == x)]
                k = K[(K["dataset"] == d) & (K["model"] == fam) & (K["defense"] == x)]["asr"] if not K.empty else pd.Series(dtype=float)
                if r.empty:
                    continue
                rows.append(dict(model=fam, defense=x, dataset=d, asr_constrained=float(r["asr"].iloc[0]),
                                 asr_consistent=(float(k.iloc[0]) if len(k) else None) if d in K_DATASETS else None,
                                 clean_auc=float(r["auc"].iloc[0])))
    return _write(pd.DataFrame(rows), "defenses_context.csv")


def realizable():
    """Share of successful high-confidence white-box evasions (eps 0.5, the transfer sources) that no real
    transaction could produce, range over models, per attacker (K equals C outside the K datasets); and the
    validator's rate on clean test records (its false-positive cost)."""
    if not REALIZABLE_CSV.exists():
        return None
    R = pd.read_csv(REALIZABLE_CSV)
    rows = []
    for d in DS:
        g = R[R["dataset"] == d]
        if g.empty:
            continue
        for m in [m for m in ("unconstrained", "constrained", "consistent") if m in set(R["mode"])]:
            x = g[g["mode"] == ("constrained" if d not in K_DATASETS and m == "consistent" else m)]["share"]
            rows.append(dict(dataset=d, attacker=LETTER[m], share_min=x.min() if len(x) else None,
                             share_max=x.max() if len(x) else None))
        t = g[g["mode"] == "real"]
        rows.append(dict(dataset=d, attacker="clean test records", share_min=float(t["share"].iloc[0]) if len(t) else None,
                         share_max=float(t["share"].iloc[0]) if len(t) else None,
                         share_legit=float(t["share_legit"].iloc[0]) if len(t) and "share_legit" in t else None))
    return _write(pd.DataFrame(rows), "realizable_summary.csv")


def figure_transfer():
    """The transfer figure drawn for the consistent attacker on the three K datasets, with the study's own figure
    code pointed at runs_consistent/ and a scratch folder; written to reports/fig_transfer_consistent.{pdf,png}."""
    import fraudadv.report.make_report as MR

    if not any((RUNS_CONSISTENT / d / "attacks" / "transfer_consistent.npz").exists() for d in K_DATASETS):
        return
    saved = (MR.RUNS, MR.REP, MR.DS)
    scratch = Path(tempfile.mkdtemp())
    try:
        MR.RUNS, MR.REP, MR.DS = RUNS_CONSISTENT, scratch, [d for d in MR.DS if d in K_DATASETS]
        MR.fig_transfer("consistent")
    finally:
        MR.RUNS, MR.REP, MR.DS = saved
    for ext in ("pdf", "png"):
        f = scratch / f"fig_transfer_consistent.{ext}"
        if f.exists():
            shutil.copy2(f, REPORTS / f.name)


def main():
    G = grid_three_attackers()
    worst_case(G)
    summary(G)
    grid(G)
    robust_auc()
    history()
    defenses()
    defenses_context()
    realizable()
    figure_transfer()
    print("wrote", sorted(p.name for p in OUT.glob("*.csv")))


if __name__ == "__main__":
    main()
