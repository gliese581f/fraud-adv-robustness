"""How the foundation model is adapted, and what that does to robustness: Table S-XXVIII.

Three ways to turn Chronos-2 and TimesFM 2.5 into fraud detectors on the two datasets with full card
histories (Sparkov, TabFormer), attacked with the same protocol (fraudadv.models.foundation_adapt):
  frozen + head   the study's models: frozen encoder, MLP head on [tabular features, encoding]
  LoRA + head     rank-8 LoRA adapters on the encoder's attention, trained with the head
  zero-shot       no training at all: forecast surprise of the transaction's log-amount and log-gap

One row per (dataset, model): clean metrics (test means over the seeds), worst-case ASR at eps = 0.25
(both threat models), history injection at k = 8, the RL attackers at eps = 0.25 and robust recall
(recall x (1 - worst-case constrained ASR)), and Pareto points for fraudadv.report.pareto.

Inputs: the frozen models in --runs (runs/<dataset>/<family>/metrics.json and the grid's attacks/ files), the
LoRA / zero-shot models and their attacks in --adapt-runs (study-artifacts/fm_adapt/runs_v2), and the RL attackers'
results (FRAUDADV_RL_RESULTS). make_results.sh runs this before make_report, which reads pareto_extra_fm_adapt.csv.

usage: python -m fraudadv.report.fm_adapt [--runs DIR] [--adapt-runs DIR] [--rl-adapt DIR] [--rep DIR]
writes <rep>/table17_fm_adapt.csv (Table S-XXVIII) and <rep>/pareto_extra_fm_adapt.csv (points of the Pareto figure)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 0.25
DSETS = ["sparkov", "tabformer"]
DS_NAME = {"sparkov": "Sparkov", "tabformer": "TabFormer"}
BASES = {"chronos2": "Chronos-2", "timesfm": "TimesFM 2.5"}
VARIANTS = [("", "frozen + head"), ("_lora", "LoRA + head"), ("_zs", "zero-shot")]   # (family suffix, name)
GROUP = {"_lora": "lora", "_zs": "zs"}            # file-name tag of the adapted variants' attack outputs
RL_ROOT = Path(os.environ.get("FRAUDADV_RL_RESULTS", "fraud-rl-attack"))   # results of the RL attackers (a separate study)
WORST = "worst case (all attacks)"


def _read(paths):
    """The non-empty CSVs among paths in one frame (empty frame if none)."""
    fr = [pd.read_csv(p) for p in paths if Path(p).exists() and Path(p).stat().st_size > 1]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def clean(runs, ds, fam):
    """Clean test metrics of family fam, means over the seeds of metrics.json (None if not trained)."""
    p = runs / ds / fam / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    out = {k: float(np.mean([r[k] for r in m])) for k in ("auc", "auprc", "recall@1%fpr", "ks")}
    out["auprc_sd"] = float(np.std([r["auprc"] for r in m])) if len(m) > 1 else np.nan
    out["seeds"] = len(m)
    return out


def grid(runs, ds, fam, suf):
    """The attack grid's result rows of family fam (suffix suf: "" frozen, "_lora", "_zs")."""
    att = runs / ds / "attacks"
    if suf:
        return _read(sorted(att.glob(f"results_full_{GROUP[suf]}*.csv"))).query("model == @fam")   # adapted models: one file per part
    return _read([att / f"results_full_{fam}_c.csv", att / f"results_full_{fam}_u.csv"]).query("model == @fam")


def robust(runs, ds, fam, suf):
    """Ranking-metric rows (fraudadv.attacks.robust_auc) of family fam at EPS."""
    att = runs / ds / "attacks"
    R = _read(sorted(att.glob(f"robust_auc_{GROUP[suf]}*.csv" if suf else "robust_auc_fm*.csv")))   # adapted models: one file per family
    return R[(R.model == fam) & (R.eps.round(4) == EPS)] if len(R) else R


def history(runs, ds, fam, suf):
    """ASR of history injection with k = 8 transactions (best of the injection attacks), None if not run."""
    att = runs / ds / "attacks"
    H = _read([att / (f"history_{GROUP[suf]}_{fam}.csv" if suf else f"history_fm_{fam}.csv")])
    if not len(H):
        return None
    g = H[(H.model == fam) & H.attack.str.startswith("best") & (H.k == 8)]
    return float(g.asr.iloc[0]) if len(g) else None


def rl(rl_adapt, ds, fam, suf):
    """Result rows of the RL attackers on family fam: the adapted models' runs in rl_adapt, the frozen models'
    in fraud-rl-attack (consolidated results on Sparkov, the extension runs elsewhere)."""
    if suf:
        p = [rl_adapt / fam / ds / f"results_{ds}.csv"]
    elif ds == "sparkov":
        p = [RL_ROOT / "results" / ds / f"results_{ds}_consol.csv"]
    else:
        p = [RL_ROOT / "results_ext" / fam / ds / f"results_{ds}.csv"]
    R = _read(p)
    return R[R.model == fam] if len(R) else R


def _setting_cells(G, RA, R, mode):
    """The cells of one threat setting (mode), with keys ending in _u or _c: worst-case ASR (with the number of
    frauds attacked and detected), best white-box and Square ASR from the grid rows G, the ranking metrics RA and
    the RL attackers R; for the constrained setting also the RL attacker without a budget (rl_free)."""
    m = mode[0]
    cells = {}
    g = G[(G["mode"] == mode) & (G.eps.round(4) == EPS)]
    w = g[g.attack == WORST]
    cells[f"n_{m}"] = int(w.n.iloc[0]) if len(w) else np.nan
    cells[f"det_{m}"] = int(w.n_detected.iloc[0]) if len(w) else np.nan
    cells[f"asr_{m}"] = float(w.asr.iloc[0]) if len(w) else np.nan
    cells[f"asr_usd_{m}"] = float(w.asr_usd.iloc[0]) if len(w) else np.nan
    wb = g[g.attack.str.startswith("white-box")]
    sq = g[g.attack.str.startswith("score black-box")]
    cells[f"wb_{m}"] = float(wb.asr.max()) if len(wb) else np.nan
    cells[f"square_{m}"] = float(sq.asr.max()) if len(sq) else np.nan
    ra = RA[RA["mode"] == mode] if len(RA) else RA
    cells[f"robust_auc_{m}"] = float(ra.robust_auc.iloc[0]) if len(ra) else np.nan
    cells[f"robust_auc_clean_{m}"] = float(ra.clean_auc.iloc[0]) if len(ra) else np.nan
    q = R[(R["mode"] == mode) & (R.eps.round(4) == EPS)] if len(R) else R
    for a in ("rl_bandit", "rl_mem"):
        x = q[q.attacker == a] if len(q) else q
        cells[f"{a}_{m}"] = float(x.asr.iloc[0]) if len(x) else np.nan
    if len(q):
        cells[f"rl_det_{m}"] = int(q.n_detected.iloc[0])
    if m == "c" and len(R):
        free = R[R.attacker == "fraud_rla"]
        cells["rl_free"] = float(free.asr.iloc[0]) if len(free) else np.nan
    return cells


def rows(runs, rl_adapt, adapt_runs=None):
    """One row per (dataset, foundation model, adaptation) that was trained: clean metrics, the cells of both threat
    settings, history injection at k = 8 and robust recall. runs: the study's runs (frozen models); adapt_runs: where
    the LoRA / zero-shot models and their attacks are (default: the same folder)."""
    adapt_runs = adapt_runs or runs
    out = []
    for ds in DSETS:
        for base, bname in BASES.items():
            for suf, vname in VARIANTS:
                fam = base + suf
                fam_runs = adapt_runs if suf else runs
                c = clean(fam_runs, ds, fam)
                if c is None:
                    continue
                r = dict(dataset=ds, base=base, variant=vname, model=fam, seeds=c["seeds"], auc=c["auc"], auprc=c["auprc"],
                         auprc_sd=c["auprc_sd"], recall=c["recall@1%fpr"], ks=c["ks"])
                G, RA, R = grid(fam_runs, ds, fam, suf), robust(fam_runs, ds, fam, suf), rl(rl_adapt, ds, fam, suf)
                for mode in ("unconstrained", "constrained"):
                    r.update(_setting_cells(G, RA, R, mode))
                r["hist8"] = history(fam_runs, ds, fam, suf)
                r["robust_recall"] = r["recall"] * (1 - r["asr_c"]) if not np.isnan(r.get("asr_c", np.nan)) else np.nan
                out.append(r)
    return pd.DataFrame(out)


def pareto_extra(T):
    """Pareto points of the LoRA and zero-shot models for fraudadv.report.pareto (constrained worst-case ASR)."""
    E = T[T.variant != "frozen + head"]
    return pd.DataFrame(dict(dataset=E.dataset, model=E.model, defense="none", recall=E.recall, asr=E.asr_c, auc=E.auc))


def main(runs, rl_adapt, rep, adapt_runs=None):
    """Write the summary CSV and the Pareto points; print the table."""
    T = rows(runs, rl_adapt, adapt_runs)
    rep.mkdir(parents=True, exist_ok=True)
    T.round(4).to_csv(rep / "table17_fm_adapt.csv", index=False)
    pareto_extra(T).round(4).to_csv(rep / "pareto_extra_fm_adapt.csv", index=False)
    cols = ["dataset", "model", "seeds", "auc", "auprc", "recall", "det_c", "asr_u", "asr_c", "rl_mem_c", "rl_free", "hist8", "robust_recall"]
    print(T[cols].round(3).to_string(index=False))
    return T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.environ.get("FRAUDADV_RUNS", "runs"))
    ap.add_argument("--adapt-runs", default=None, help="LoRA / zero-shot runs (default: --runs)")
    ap.add_argument("--rl-adapt", default=str(RL_ROOT / "results_adapt"))
    ap.add_argument("--rep", default="reports")
    a = ap.parse_args()
    main(Path(a.runs), Path(a.rl_adapt), Path(a.rep), Path(a.adapt_runs) if a.adapt_runs else None)
