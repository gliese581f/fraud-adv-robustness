"""Figures and summaries of the causal ablation analysis, from the frames fraudadv.causal.analyze computes
(per-fraud outcomes T, contrasts C, factorial model F, clean metrics K). Called by analyze.main through write_all.

  reports/fig_causal_effects.{pdf,png}   forest plot of the pooled interventional effects (Figure S13)
  reports/fig_causal_dose.{pdf,png}      depth dose-response per dataset (Figure S14), data in
                                         reports/causal/dose_response.csv
  reports/causal/variants_clean.csv      clean performance of every variant, distillation fidelity (Table S-XXXIV)
  reports/causal/factorial_effects.csv   effects derived from the factorial model
The pooled effects of Table VI and Table S-XXXV are the rows of reports/causal/contrasts.csv with dataset
"pooled"; the factorial model of Table S-XXXVI is reports/causal/factorial_lpm.csv (both written by analyze).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraudadv.causal.ablate import VARIANTS
from fraudadv.report.make_report import DS_NAME, REP, _style

INK, INK2, MUTED, GRID = "#2b2a28", "#52514e", "#8a8984", "#e6e5e0"
BLUE, RED = "#2a78d6", "#e34948"
ATT_NAME = {"wb": "white-box", "sq": "Square (score-based)", "hsj": "HopSkipJump (decision-based)"}
# names of the contrasts, in the order of the figure
SHORT = {
    "total": "MLP vs XGBoost (both tuned)",
    "rep_treefn": "tree → network, tree function",
    "rep_nnfn": "tree → network, network function",
    "fn_tree": "function swap, as trees",
    "fn_nn": "function swap, as network",
    "xgb_d2": "XGBoost depth 2", "xgb_d6": "XGBoost depth 6", "xgb_d10": "XGBoost depth 10",
    "xgb_mono": "XGBoost monotone", "mlp_l1": "MLP 1 layer", "mlp_l4": "MLP 4 layers", "mlp_relu": "MLP ReLU",
    "mlp_sn": "MLP spectral norm", "mlp_wd": "MLP weight decay",
}
PLAIN = dict(SHORT)                                                    # the names in the figure
GROUP_NAME = {"class": "Model class (2 × 2 distillation design)", "trees": "Within XGBoost", "neural": "Within the MLP"}


def fig_effects(C, field="d_asr", name="fig_causal_effects"):
    """Forest plot of the pooled effects: one panel per attack (field, by default d_asr) and a last panel of the
    change in clean recall, one row per contrast: reports/<name>.{png,pdf} (Figure S13)."""
    import matplotlib.pyplot as plt

    _style(plt)
    P = C[C.dataset == "pooled"]
    keys = [k for k in SHORT if k in set(P.contrast)]
    groups = [g for g in GROUP_NAME if g in set(P.group)]
    # y position of each contrast, a gap of 0.9 between groups
    ylab, ypos, y = [], {}, 0
    for g in groups:
        for k in [k for k in keys if P[P.contrast == k].group.iloc[0] == g]:
            ypos[k] = y
            ylab.append(PLAIN[k])
            y += 1
        y += 0.9
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 0.55 + 0.165 * y), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 0.75]))
    for ax, att in zip(axes, ["wb", "sq", "hsj", "recall"]):
        for md, off, face in (("constrained", -0.14, None), ("unconstrained", 0.14, "white")):
            for k, yy in ypos.items():
                r = P[(P.contrast == k) & (P["mode"] == md) & (P.attack == ("sq" if att == "recall" else att))]
                if r.empty:
                    continue
                r = r.iloc[0]
                f = "d_recall" if att == "recall" else field
                if att == "recall" and md == "unconstrained":
                    continue                                  # clean recall does not depend on the setting
                col = BLUE if att != "recall" else INK2
                ax.plot([r[f + "_lo"], r[f + "_hi"]], [yy + (0 if att == "recall" else off)] * 2, color=col, linewidth=1.1,
                        solid_capstyle="round", zorder=2)
                ax.scatter([r[f]], [yy + (0 if att == "recall" else off)], s=16, color=col if face is None else "white",
                           edgecolors=col, linewidths=1.1, zorder=3)
        ax.axvline(0, color=MUTED, linewidth=0.7, zorder=1)
        ax.set_title("clean recall" if att == "recall" else ATT_NAME[att].split(" (")[0], fontsize=8, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(list(ypos.values()), ylab, fontsize=7)
    axes[0].set_ylim(y - 0.4, -0.7)
    gy = 0
    for g in groups:
        n = sum(1 for k in ypos if P[P.contrast == k].group.iloc[0] == g)
        axes[0].text(-0.02, gy - 0.7, GROUP_NAME[g], transform=axes[0].get_yaxis_transform(), ha="right", fontsize=6.5,
                     color=MUTED, style="italic")
        gy += n + 0.9
    fig.text(0.5, 0.01, "change in attack success on frauds both models detect, percentage points "
             "(filled: constrained, hollow: unconstrained; eps = 0.25; 95% bootstrap interval)   |   last panel: change in clean recall",
             ha="center", fontsize=6.5, color=INK2)
    fig.subplots_adjust(left=0.27, right=0.99, top=1 - 0.28 / (0.55 + 0.165 * y), bottom=0.36 / (0.55 + 0.165 * y), wspace=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_dose(T):
    """ASR (detected frauds) against XGBoost depth and MLP layers, per dataset, constrained, eps 0.25:
    reports/fig_causal_dose.{png,pdf} (Figure S14) and its data, reports/causal/dose_response.csv."""
    import matplotlib.pyplot as plt

    from fraudadv.causal.ablate import _params, is_base
    from fraudadv.causal.analyze import CDS

    _style(plt)
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.4), sharey="row")
    rows = []
    for r, (fam, levels) in enumerate((("xgb", ["xgb_d2", "xgb_d6", "xgb_d10", "xgb"]), ("mlp", ["mlp_l1", "mlp", "mlp_l4"]))):
        for ax, ds in zip(axes[r], CDS):
            p = _params(ds, "xgboost" if fam == "xgb" else "mlp")
            for att, col, mk in (("wb", INK, "o"), ("sq", BLUE, "s"), ("hsj", RED, "^")):
                xs, ys = [], []
                for v in levels:
                    if v != fam and is_base(ds, v):
                        continue
                    # the level: tuned value for the base, else the number in the variant name
                    lvl = (p["max_depth"] if v == "xgb" else int(v.split("_d")[1])) if fam == "xgb" else \
                          (p["n_layers"] if v == "mlp" else int(v.split("_l")[1]))
                    D = T[(T.dataset == ds) & (T.variant == v) & (T["mode"] == "constrained") & (T.attack == att) & T.det]
                    if D.empty:
                        continue
                    xs.append(lvl)
                    ys.append(100 * D.evaded.mean())
                    rows.append(dict(dataset=ds, family=fam, level=lvl, attack=att, asr=ys[-1], n=len(D)))
                o = np.argsort(xs)
                ax.plot(np.array(xs)[o], np.array(ys)[o], color=col, marker=mk, markersize=3.5, linewidth=1.1,
                        label=ATT_NAME[att].split(" (")[0])
            ax.set_title(DS_NAME[ds], fontsize=8, loc="left", color=INK)
            ax.grid(color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("max depth" if fam == "xgb" else "hidden layers", fontsize=7)
        axes[r, 0].set_ylabel(("XGBoost" if fam == "xgb" else "MLP") + "\nASR (%)", fontsize=7.5)
    axes[0, 0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.12, hspace=0.6, wspace=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"fig_causal_dose.{ext}", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(REP / "causal" / "dose_response.csv", index=False)


def variants_clean(K):
    """Clean AUC and recall at 1% FPR of every ablation variant per dataset (mean of the seeds), and for the
    distilled variants the Spearman correlation of the student's and the teacher's test scores (fidelity):
    reports/causal/variants_clean.csv (Table S-XXXIV)."""
    agg = dict(auc=("auc", "mean"), recall=("recall@1%fpr", "mean"))
    if "fid_spearman" in K:
        agg["fidelity_spearman"] = ("fid_spearman", "mean")
    G = K.groupby(["variant", "dataset"]).agg(**agg).reset_index()
    order = {v: i for i, v in enumerate(VARIANTS)}
    G = G.sort_values(["variant", "dataset"], key=lambda c: c.map(order) if c.name == "variant" else c)
    if "fidelity_spearman" in G:
        G.loc[~G["variant"].str.endswith(("_from_mlp", "_from_xgb")), "fidelity_spearman"] = np.nan
    G.to_csv(REP / "causal" / "variants_clean.csv", index=False)
    return G


def factorial_effects(F):
    """Effects derived from the main factorial model, in percentage points: the neural representation's effect
    under each attack (its main effect plus the attack interaction) and the neural-learned function's effect in a
    tree and in a network representation: reports/causal/factorial_effects.csv."""
    M = F[F.model == "main"].set_index("term")["coef"]
    g = lambda t: float(M.get(t, 0.0))
    D = {"representation, white-box": g("neural_rep"),
         "representation, Square": g("neural_rep") + g("neural_rep:C(attack)[T.sq]"),
         "representation, HopSkipJump": g("neural_rep") + g("neural_rep:C(attack)[T.hsj]"),
         "function, as trees": g("neural_fn"), "function, as network": g("neural_fn") + g("neural_rep:neural_fn")}
    out = pd.DataFrame([dict(effect=k, coef=v) for k, v in D.items()])
    out.to_csv(REP / "causal" / "factorial_effects.csv", index=False)
    return out


def write_all(T, C, F, K):
    """Write every figure and summary CSV of this module."""
    (REP / "causal").mkdir(exist_ok=True)
    fig_effects(C)
    fig_dose(T)
    variants_clean(K)
    factorial_effects(F)
    print("causal figures and summaries written", flush=True)
