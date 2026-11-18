"""SHAP and attribution figures for RQ4 (supplement Figures S8 to S12), written to reports/ as .pdf and .png.

Inputs: runs/<dataset>/explain/shap_<family>.npz (fraudadv.explain.shap_values: attributions phi of test transactions
X, and of each evaded fraud before (phi0_<mode>) and after (phi1_<mode>) the attack) and
runs/<dataset>/explain/global_importance.csv (fraudadv.explain.attributions).

  fig_shap_<family>.pdf     Figure S8 (XGBoost), Figure S9, Figure S11: summary (beeswarm) plot per
                            dataset: one dot per test transaction, x = its attribution to the model's logit,
                            colour = the feature's value (rank within the panel)
  fig_shap_xgboost_three    the XGBoost plot on three datasets only, for a shorter main text
  fig_shap_shift.pdf        Figure S10. XGBoost, constrained: mean TreeSHAP value of the features the attack moves
                            most, on each evaded fraud before and after the attack
  fig_attr_heatmap.pdf      Figure S12. Every family: share of the model's global attribution per feature (top features)

The numbers behind the last two go to reports/shap/. Feature labels ending in " *" are attacker-controllable.
Run by make_results.sh after make_report; can also run alone.

usage: python -m fraudadv.report.shap_figures
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraudadv.report.make_report import DS, DS_NAME, FAM, FAM_NAME, REP, RUNS, _style

INK, INK2, MUTED, GRID = "#2b2a28", "#52514e", "#8a8984", "#e6e5e0"
BLUE, RED, MID = "#2a78d6", "#e34948", "#bdbcb6"
BLUES = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
METHOD = {"xgboost": "TreeSHAP", "catboost": "TreeSHAP", "mlp": "Integrated Gradients"}


def _short(f, n=18):
    """A feature name cut to n characters."""
    return f if len(f) <= n else f[: n - 1] + "…"


def _label(f, mutable):
    """Axis label of a feature; ' *' marks one the attacker controls."""
    return _short(f) + (" *" if mutable else "")


def _load(d, fam):
    """The SHAP arrays of family fam on dataset d (runs/<d>/explain/shap_<fam>.npz), or None if not computed."""
    p = RUNS / d / "explain" / f"shap_{fam}.npz"
    return dict(np.load(p, allow_pickle=True)) if p.exists() else None


def _swarm_offsets(x, width=0.38, bins=60, rng=None):
    """Vertical offsets that spread dots by local density, as in a beeswarm."""
    rng = rng or np.random.default_rng(0)
    if len(x) < 2 or np.ptp(x) == 0:
        return rng.uniform(-width / 3, width / 3, len(x))
    edges = np.linspace(x.min(), x.max(), bins + 1)
    b = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    cnt = np.bincount(b, minlength=bins)
    dens = np.sqrt(cnt[b] / cnt.max())            # row width follows the local density (sina plot)
    return rng.uniform(-1, 1, len(x)) * width * np.maximum(dens, 0.08)


def _feature_rows_axis(ax, labels, top, title):
    """The common look of a panel with one row per feature (labels from the bottom row up): zero line, x grid."""
    ax.axvline(0, color=MUTED, linewidth=0.6, zorder=0)
    ax.set_yticks(range(top), labels, fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, top - 0.3)
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_beeswarm(fam, top=8, only=None, name=None, row_h=2.0):
    """reports/fig_shap_<name or fam>.{png,pdf}: per dataset (all, or those in only), the top features of family fam
    by mean |attribution|, one dot per test transaction coloured by the feature's value."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    _style(plt)
    cmap = LinearSegmentedColormap.from_list("lowhigh", [BLUE, MID, RED])
    ds = [d for d in (only or DS) if _load(d, fam) is not None]
    if not ds:
        return
    nr = int(np.ceil(len(ds) / 3))
    fig, axes = plt.subplots(nr, 3, figsize=(7.16, 0.6 + row_h * nr), squeeze=False)
    for ax, d in zip(axes.flat, ds):
        z = _load(d, fam)
        phi, X, feats, mut = z["phi"], z["X"], list(z["features"]), z["mutable"]
        order = np.argsort(-np.abs(phi).mean(0))[:top]
        for r, j in enumerate(order):
            x = phi[:, j]
            v = X[:, j]
            lo, hi = np.nanpercentile(v, [5, 95])
            c = np.clip((v - lo) / (hi - lo), 0, 1) if hi > lo else np.full(len(v), 0.5)
            yy = top - 1 - r + _swarm_offsets(x)
            s = np.argsort(np.abs(x))      # large attributions on top
            ax.scatter(x[s], yy[s], c=c[s], cmap=cmap, vmin=0, vmax=1, s=3.2, linewidths=0, rasterized=True)
        _feature_rows_axis(ax, [_label(feats[j], mut[j]) for j in order[::-1]], top, DS_NAME[d])
    for ax in axes.flat[len(ds):]:
        ax.axis("off")
    h = 0.6 + row_h * nr                        # figure height in inches; x label and colour bar share a 0.6-inch strip
    fig.subplots_adjust(left=0.13, right=0.99, top=1 - 0.25 / h, bottom=0.62 / h, wspace=0.95, hspace=0.42)
    fig.text(0.13, 0.12 / h, f"horizontal axis: {METHOD[fam]} value (contribution to the logit)", ha="left", fontsize=7.5, color=INK)
    cax = fig.add_axes([0.76, 0.16 / h, 0.2, 0.07 / h])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([])
    cax.text(-0.03, 0.5, "feature value: low", transform=cax.transAxes, ha="right", va="center", fontsize=7, color=INK2)
    cax.text(1.03, 0.5, "high", transform=cax.transAxes, ha="left", va="center", fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.outline.set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"fig_shap_{name or fam}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_shift(fam="xgboost", mode="constrained", top=6, min_n=5):
    """reports/fig_shap_shift.{png,pdf} (Figure S10) and reports/shap/shap_shift_xgboost_constrained.csv: mean
    attribution before and after the attack, per feature, on the frauds the attack evades (datasets with at least
    min_n of them)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _style(plt)
    ds = [d for d in DS if (z := _load(d, fam)) is not None and f"phi0_{mode}" in z and len(z[f"phi0_{mode}"]) >= min_n]
    if not ds:
        return
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.6))
    rows = []
    for ax, d in zip(axes.flat, ds):
        z = _load(d, fam)
        p0, p1, feats, mut = z[f"phi0_{mode}"], z[f"phi1_{mode}"], list(z["features"]), z["mutable"]
        m0, m1 = p0.mean(0), p1.mean(0)
        order = np.argsort(-np.abs(p1 - p0).mean(0))[:top]
        for r, j in enumerate(order):
            y = top - 1 - r
            ax.plot([m0[j], m1[j]], [y, y], color=MID, linewidth=2, solid_capstyle="round", zorder=1)
            ax.scatter([m0[j]], [y], s=26, color=INK2, zorder=2, linewidths=0)
            ax.scatter([m1[j]], [y], s=34, color=BLUE, zorder=3, edgecolors="white", linewidths=1.2)
            rows.append(dict(dataset=d, model=fam, mode=mode, feature=feats[j], controllable=bool(mut[j]),
                             mean_shap_clean=float(m0[j]), mean_shap_adv=float(m1[j]),
                             mean_abs_change=float(np.abs(p1[:, j] - p0[:, j]).mean()), n=len(p0)))
        _feature_rows_axis(ax, [_label(feats[j], mut[j]) for j in order[::-1]], top, f"{DS_NAME[d]} ($n={len(p0)}$)")
    for ax in axes.flat[len(ds):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("mean TreeSHAP value", fontsize=7.5)
    h = [Line2D([], [], marker="o", linestyle="", color=INK2, markersize=5),
         Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=5, markeredgecolor="white")]
    fig.legend(h, ["detected fraud (clean)", "its adversarial example"], loc="lower center", ncol=2, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.13, right=0.99, top=0.94, bottom=0.16, wspace=0.95, hspace=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"fig_shap_shift.{ext}", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(REP / "shap" / "shap_shift_xgboost_constrained.csv", index=False)


def fig_heatmap(top=10):
    """reports/fig_attr_heatmap.{png,pdf} (Figure S12) and reports/shap/attribution_share_top_features.csv: per
    dataset, each family's share of its global attribution on the top features (by mean share over families)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from fraudadv.data.load import load

    _style(plt)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    ds = [d for d in DS if (RUNS / d / "explain" / "global_importance.csv").exists()]
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 8.6))
    rows = []
    for ax, d in zip(axes.flat, ds):
        G = pd.read_csv(RUNS / d / "explain" / "global_importance.csv", index_col=0).abs()
        fams = [f for f in FAM if f in G]
        S = G[fams] / G[fams].sum(0)
        mut = set(load(d)[1]["mutable"])
        feats = S.mean(axis=1).sort_values(ascending=False).index[:top]
        M = S.loc[feats].to_numpy()
        ax.imshow(M, cmap=cmap, vmin=0, vmax=0.5, aspect="auto")
        ax.set_xticks(range(len(fams)), [FAM_NAME[f].replace(" (triplet)", "").replace("Graph Transformer (HGT)", "HGT")
                                         for f in fams], rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(feats)), [_label(f, f in mut) for f in feats], fontsize=6.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(len(fams) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(feats) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        ax.set_title(DS_NAME[d], fontsize=8.5, loc="left", color=INK)
        for f in feats:
            for fam in fams:
                rows.append(dict(dataset=d, feature=f, controllable=f in mut, model=fam, share=float(S.loc[f, fam])))
    for ax in axes.flat[len(ds):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.16, right=0.99, top=0.97, bottom=0.12, wspace=0.55, hspace=0.62)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(0, 0.5)
    cax = fig.add_axes([0.3, 0.025, 0.45, 0.01])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("share of the model's global attribution (each column sums to 1 over all features; scale capped at 0.5)",
                 fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.outline.set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"fig_attr_heatmap.{ext}", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(REP / "shap" / "attribution_share_top_features.csv", index=False)


def main():
    """Draw every SHAP figure."""
    (REP / "shap").mkdir(exist_ok=True)
    for fam in ("xgboost", "catboost", "mlp"):
        fig_beeswarm(fam)
    fig_beeswarm("xgboost", top=6, only=["ulb", "ieee_cis", "sparkov"], name="xgboost_three", row_h=1.55)   # journal-length main text
    fig_shift()
    fig_heatmap()


if __name__ == "__main__":
    main()
