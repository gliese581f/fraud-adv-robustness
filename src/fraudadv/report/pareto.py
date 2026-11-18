"""Clean detection against robust detection: the Pareto frontier of every model and defense (Figure S7).

One point per dataset x model (undefended families, the defended tabular models of RQ5 and, when their
results exist, the LoRA-adapted and zero-shot foundation models): x = recall at the 1%-FPR operating
point on the clean test set, y = robust recall at eps = 0.25 = recall x (1 - worst-case ASR), the share of
all frauds still caught when every attacker spends the budget. Threat model per dataset as in RQ5
(constrained; unconstrained on ULB). A point is on the frontier when no other point of the same dataset is
at least as good on both axes and better on one.

Inputs: the summary CSVs of fraudadv.report.make_report (table7_defenses_by_dataset.csv, table1_clean.csv,
table6_worstcase_ci.csv) and optional extra points (pareto_extra*.csv: the LoRA / zero-shot foundation models
of fraudadv.report.fm_adapt). make_report.main calls main() once those CSVs exist.

usage: python -m fraudadv.report.pareto [--extra runs/<ds>/attacks/pareto_extra.csv ...]
writes reports/fig_pareto.{png,pdf} (Figure S7) and reports/table16_pareto.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraudadv.report.make_report import CLASS_COLOR, DS, DS_NAME, FAM, FAM_CLASS, FAM_NAME, REP, RUNS, _style, style_of

EPS = 0.25
MODE = {"ulb": "unconstrained"}                                       # threat setting per dataset; default constrained
ORDER = ["baf", "sparkov", "tabformer", "ieee_cis", "ulb", "paysim"]  # panel order
DEF_NAME = {"pgd_at": "PGD-AT", "trades": "TRADES", "adv_aug": "adv. augmentation", "adv_boost": "adv. boosting", "monotone": "monotone",
            "squeeze": "squeezing"}
# foundation-model variants: same class colour; marker = adaptation (LoRA D, zero-shot X); the frozen ones keep the
# member marker (Chronos-2 o, TimesFM s)
ADAPT = {"chronos2_lora": ("chronos2", "LoRA"), "timesfm_lora": ("timesfm", "LoRA"),
         "chronos2_zs": ("chronos2", "zero-shot"), "timesfm_zs": ("timesfm", "zero-shot")}


def front(x, y):
    """Boolean mask of the non-dominated points (maximise both)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    on = np.ones(len(x), bool)
    for i in range(len(x)):
        dom = (x >= x[i]) & (y >= y[i]) & ((x > x[i]) | (y > y[i]))
        on[i] = not dom.any()
    return on


def points(extra=()):
    """One row per dataset x model x defense: clean recall, worst-case ASR, AUC, robust recall and whether the point
    is on its dataset's frontier. Defended models and their baselines come from table7_defenses_by_dataset.csv;
    every other undefended model from table1_clean.csv and table6_worstcase_ci.csv; then the rows of the extra files."""
    rows = []
    T = pd.read_csv(REP / "table7_defenses_by_dataset.csv")
    for _, r in T.iterrows():
        rows.append(dict(dataset=r["dataset"], model=r["model"], defense=r["defense"], recall=r["recall"], asr=r["asr"], auc=r["auc"]))
    C = pd.read_csv(REP / "table1_clean.csv")
    inv = {v: k for k, v in DS_NAME.items()}
    name2fam = {v: k for k, v in FAM_NAME.items()}
    W = pd.read_csv(REP / "table6_worstcase_ci.csv")
    have = {(r["dataset"], r["model"]) for r in rows}
    for _, c in C.iterrows():
        d, f = inv.get(c["Dataset"]), name2fam.get(c["Model"])
        if d is None or f is None or (d, f) in have:
            continue
        w = W[(W.dataset == d) & (W.model == f) & (W["mode"] == MODE.get(d, "constrained"))]
        if len(w):
            rows.append(dict(dataset=d, model=f, defense="none", recall=c["_recall@1%fpr"], asr=float(w["asr"].iloc[0]), auc=c["_auc"]))
    for p in extra:
        E = pd.read_csv(p)
        rows += E[["dataset", "model", "defense", "recall", "asr", "auc"]].to_dict("records")
    P = pd.DataFrame(rows)
    P["robust_recall"] = P["recall"] * (1 - P["asr"])
    P["on_front"] = False
    for d, g in P.groupby("dataset"):
        P.loc[g.index, "on_front"] = front(g["recall"], g["robust_recall"])
    return P


def label(r):
    """Short point label, e.g. "XGBoost + adv. boosting" or "Chronos-2 LoRA"."""
    base, var = ADAPT.get(r["model"], (r["model"], None))
    s = FAM_NAME[base].replace("Graph Transformer (HGT)", "HGT").replace("ivis (triplet)", "ivis")
    if var:
        s += f" {var}"
    if r["defense"] != "none":
        s += " + " + DEF_NAME.get(r["defense"], r["defense"])
    return s


INK, MUTED, GRID = "#2b2a28", "#8a8984", "#e6e5e0"


def _is_zero_shot(model):
    """True for a zero-shot foundation model."""
    return ADAPT.get(model, (model, None))[1] == "zero-shot"


def _off_scale_rule(g, zero_shot):
    """Test for a point of panel g lying far below the points that are not zero-shot (by more than a quarter of
    their range). Zero-shot detectors that far below would squash the panel, so they go in a corner note instead."""
    core = g[~zero_shot] if (~zero_shot).any() else g
    cx0, cy0 = core["recall"].min(), core["robust_recall"].min()
    cx1, cy1 = core["recall"].max(), core["robust_recall"].max()

    def off_scale(r):
        return (r["recall"] < cx0 - 0.25 * max(cx1 - cx0, 0.05)) or (r["robust_recall"] < cy0 - 0.25 * max(cy1 - cy0, 0.05))

    return off_scale


def _staircase(front_points):
    """x and y of the frontier drawn as a staircase: each front point dominates the region below and to its left."""
    xs, ys = [], []
    for i, (_, r) in enumerate(front_points.iterrows()):
        if i:                                         # down from the previous point, then across to this one
            xs.append(xs[-1])
            ys.append(r["robust_recall"])
        xs.append(r["recall"])
        ys.append(r["robust_recall"])
    return xs, ys


def _draw_point(ax, r):
    """One point: colour and marker of its family (LoRA D, zero-shot X); hollow when defended."""
    base, var = ADAPT.get(r["model"], (r["model"], None))
    st = style_of(base)
    defended = r["defense"] != "none"
    face = "white" if defended else st["color"]
    marker = {"LoRA": "D", "zero-shot": "X"}.get(var, st["marker"])
    ax.scatter(r["recall"], r["robust_recall"], s=30 if not defended else 22, marker=marker, facecolor=face,
               edgecolor=st["color"] if face == "white" else "white", linewidth=1.1 if face == "white" else 0.7, zorder=3)


def _off_scale_note(ax, offs):
    """Corner note listing the off-scale zero-shot points offs with their coordinates."""
    ax.scatter([0.035], [0.05], marker="X", s=26, transform=ax.transAxes, color=CLASS_COLOR["Foundation (frozen)"],
               edgecolor="white", linewidth=0.7, zorder=4, clip_on=False)
    ax.annotate("zero-shot, off scale:\n" + "\n".join(f"{label(r).replace(' zero-shot', '')} ({r['recall']:.2f}, {r['robust_recall']:.2f})"
                                                      for _, r in offs.iterrows()),
                (0.035, 0.05), xycoords="axes fraction", xytext=(6, -2), textcoords="offset points", ha="left", va="bottom",
                fontsize=5.6, color=MUTED, linespacing=1.1)


def _label_front(ax, tied):
    """Label the front points, highest first; tied points share one label. Labels sit above their point (the
    staircase never passes there); the lowest front point is labelled below-left, inside the dominated region."""
    tied = sorted(tied, key=lambda kv: -kv[0][1])
    for i, ((x, y), t) in enumerate(tied):
        last = i == len(tied) - 1 and i > 0
        ax.annotate("\n".join(label(r) for _, r in t.iterrows()), (x, y), xytext=(-5, -3) if last else (4, 4),
                    textcoords="offset points", ha="right", va="top" if last else "bottom", fontsize=5.8, color=INK,
                    zorder=4, linespacing=1.1)


def _panel(ax, g, d):
    """Dataset d's panel from its points g: the frontier, the points, the off-scale note, the axis limits (with room
    above for the labels and below for the note) and the frontier labels."""
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    zero_shot = np.array([_is_zero_shot(m) for m in g.model])
    off_scale = _off_scale_rule(g, zero_shot)
    front_points = g[g.on_front].sort_values("recall")
    xs, ys = _staircase(front_points)
    ax.plot(xs, ys, color=INK, linewidth=1.0, zorder=2)
    for _, r in g.iterrows():
        if _is_zero_shot(r["model"]) and off_scale(r):
            continue
        _draw_point(ax, r)
    hidden = [z and off_scale(r) for z, (_, r) in zip(zero_shot, g.iterrows())]
    shown, offs = g[[not h for h in hidden]], g[hidden]
    lo_x, hi_x = shown["recall"].min(), shown["recall"].max()
    lo_y, hi_y = shown["robust_recall"].min(), shown["robust_recall"].max()
    if len(offs):
        _off_scale_note(ax, offs)
    px, py = 0.06 * max(hi_x - lo_x, 0.05), 0.06 * max(hi_y - lo_y, 0.05)
    ax.set_xlim(lo_x - px, min(hi_x + px, 1.0015))
    tied = list(front_points.groupby([front_points["recall"].round(4), front_points["robust_recall"].round(4)], sort=False))
    nl = max((len(t) for _, t in tied), default=1)    # most labels stacked on one point
    y0 = lo_y - py - (0.42 * (hi_y - lo_y + py * (1.6 + 1.5 * nl)) if len(offs) else 0)   # room for the off-scale note
    ax.set_ylim(max(0, y0) if not len(offs) else y0, hi_y + py * (1.6 + 1.5 * nl))
    if len(offs):                                     # no ticks in the strip that holds the note, none below 0
        yl = ax.get_ylim()
        ax.set_yticks([t for t in ax.get_yticks() if max(0, lo_y - py) <= t <= yl[1]])
        ax.set_ylim(yl)
    _label_front(ax, tied)
    ax.set_title(DS_NAME[d] + (" (unconstrained)" if MODE.get(d) == "unconstrained" else ""), loc="left", fontsize=8.5)
    ax.tick_params(labelsize=7)


def _legend_handles(P):
    """Legend entries: the model classes present, defended, the member markers, the adapted variants, the frontier."""
    from matplotlib.lines import Line2D

    classes = [c for c in CLASS_COLOR if any(FAM_CLASS[ADAPT.get(m, (m,))[0]] == c for m in P.model)]
    h = [Line2D([], [], marker="o", ls="", color=CLASS_COLOR[c], markeredgecolor="white", ms=5.5, label=c.replace(" (frozen)", " model"))
         for c in classes]
    h += [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=MUTED, ms=5, label="defended (RQ5)")]
    h += [Line2D([], [], marker=m, ls="", color=MUTED, markeredgecolor="white", ms=5, label=t) for m, t in
          (("o", "XGBoost, MLP, Hetero GNN, Chronos-2"), ("s", "CatBoost, FT-Transformer, HGT, TimesFM"), ("^", "ivis"))]
    variants = {ADAPT[m][1] for m in P.model if m in ADAPT}
    if "LoRA" in variants:
        h += [Line2D([], [], marker="D", ls="", color=CLASS_COLOR["Foundation (frozen)"], markeredgecolor="white", ms=5, label="foundation, LoRA")]
    if "zero-shot" in variants:
        h += [Line2D([], [], marker="X", ls="", color=CLASS_COLOR["Foundation (frozen)"], markeredgecolor="white", ms=6, label="foundation, zero-shot")]
    h += [Line2D([], [], color=INK, lw=1.0, label="Pareto frontier")]
    return h


def fig(P, out=REP):
    """One panel per dataset: every point, the frontier as a staircase and labels on the frontier points:
    <out>/fig_pareto.{png,pdf}."""
    import matplotlib.pyplot as plt

    _style(plt)
    dss = [d for d in ORDER if d in set(P.dataset)]
    ncol = 3
    nrow = int(np.ceil(len(dss) / ncol))
    figure, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.55 * nrow + 0.55))
    axes = np.atleast_1d(axes).ravel()
    for ax, d in zip(axes, dss):
        _panel(ax, P[P.dataset == d].copy(), d)
    for ax in axes[len(dss):]:
        ax.set_visible(False)
    for ax in axes[(nrow - 1) * ncol:]:
        ax.set_xlabel("clean recall at 1% FPR", fontsize=7.5)
    for ax in axes[::ncol]:
        ax.set_ylabel(f"robust recall, ε={EPS}", fontsize=7.5)
    h = _legend_handles(P)
    figure.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 0.0),
                  handletextpad=0.3, columnspacing=1.2)
    figure.tight_layout(rect=(0, 0.04 + 0.032 * int(np.ceil(len(h) / 4)), 1, 1))
    for ext in ("png", "pdf"):
        figure.savefig(out / f"fig_pareto.{ext}", bbox_inches="tight")
    plt.close(figure)


def main(extra=()):
    """Write reports/table16_pareto.csv and the figure, print each dataset's frontier, return the points."""
    P = points(extra)
    P["label"] = [label(r) for _, r in P.iterrows()]
    P.round(4).to_csv(REP / "table16_pareto.csv", index=False)
    fig(P)
    for d, g in P[P.on_front].groupby("dataset"):
        print(d, "front:", "; ".join(f"{r['label']} ({r['recall']:.3f}, {r['robust_recall']:.3f})" for _, r in g.sort_values("recall").iterrows()))
    return P


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    main(a.extra)
