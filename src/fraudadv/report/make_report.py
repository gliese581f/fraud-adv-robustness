"""Turn the raw results in runs/ into summary CSVs and figures in reports/.

Inputs are the files the run steps write under runs/<dataset>/ (models' metrics.json, the attack grid's
attacks/results*.csv and per_example*.npz, transfer, thickness, certification, ranking-metric, history,
defense and explanation outputs). Every output goes to reports/. fraudadv.report.tables and
fraudadv.report.consistent_tables add the result tables that need more than these CSVs (reports/tables/).

Outputs (function, then the results that use it; see RESULTS.md):
  table1_clean.{csv,md}              table_clean            Table S-IV, clean-performance numbers, Pareto figure
  table2_robust.{csv,md},            table_robust           Tables IV and S-XI (mean over datasets)
    table2b_robust_<dataset>.md
  table3_attacks.md                  table_attacks          reports only
  fig_transfer_<mode>.{png,pdf}      fig_transfer           Figure 1 (constrained), Figure S3 (unconstrained);
                                                            report/consistent_tables.py reuses it for Figure S4
  fig_robust_curves_<mode>.{png,pdf} fig_curves             Figure S1 (constrained), Figure S2 (unconstrained)
  table4_thickness*.csv,             thickness_analysis     Table S-XXIV, Figure S5, RQ3 numbers in the text
    fig_thickness_vs_asr.{png,pdf}
  table5_explain.md,                 explain_tables,        Table S-XXXII, RQ4 numbers in the text
    table5_explain_*.csv             explain_summary
  table6_*.csv                       stats_tables           Table S-XXXVIII, Table S-XXXIX, McNemar pairs in the text, Figure S7
  table7_defenses*.{csv,md},         defense_tables,        Table VII, Figure S6, Figure S7
    table7_defense_tradeoff.csv,     fig_defense_tradeoff
    fig_defense_tradeoff.{png,pdf}
  fig_pareto.{png,pdf},              pareto_figure          Figure S7 (drawn by fraudadv.report.pareto)
    table16_pareto.csv
  table8_aats_vs_milp.{csv,md}       certify_table          Table S-XXII, Table S-XXIII
  table9_robust_auc.{csv,md}         robust_auc_table       Table S-XXX, Table S-XXXI
  table10_history.{csv,md}           history_table          Table S-XII
  table11_halves.csv                 halves_table           Table S-XL
  table12_groups.csv                 groups_table           Table S-XLIII, history-injection numbers in the text
  table13_rerun_agreement.csv        rerun_table            reports only
  table14_rerun_by_attack*.csv       rerun_attack_table     Table S-XLI
  table15_hsj_starts.csv             hsj_starts_table       Table S-XLII

The module-level names (DS, FAM, RUNS, REP, the colours, style_of, _style, load_attacks, fig_transfer, ...) are
shared by the other report modules. report/consistent_tables.py swaps RUNS, REP and DS and calls
fig_transfer("consistent"), so every function reads them when it runs.

usage: python -m fraudadv.report.make_report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUNS, REP = ROOT / "runs", ROOT / "reports"
DS = ["ulb", "ieee_cis", "sparkov", "tabformer", "paysim", "baf"]
DS_NAME = {"ulb": "ULB", "ieee_cis": "IEEE-CIS", "sparkov": "Sparkov", "tabformer": "TabFormer", "paysim": "PaySim", "baf": "BAF"}
FAM = ["xgboost", "catboost", "mlp", "ft_transformer", "ivis", "hgnn", "hgt", "chronos2", "timesfm"]
FAM_NAME = {"xgboost": "XGBoost", "catboost": "CatBoost", "mlp": "MLP", "ft_transformer": "FT-Transformer", "ivis": "ivis (triplet)",
            "hgnn": "Hetero GNN", "hgt": "Graph Transformer (HGT)", "chronos2": "Chronos-2", "timesfm": "TimesFM 2.5"}
FAM_CLASS = {"xgboost": "Tree ensemble", "catboost": "Tree ensemble", "mlp": "Deep tabular", "ft_transformer": "Deep tabular",
             "ivis": "Deep tabular", "hgnn": "Graph", "hgt": "Graph", "chronos2": "Foundation (frozen)", "timesfm": "Foundation (frozen)"}
# Nine families exceed the rule of at most eight categorical hues, so lines use composite encoding:
# hue = model class (slots 1-4 of the validated reference palette, fixed order), marker + dash = member within class.
CLASS_COLOR = {"Tree ensemble": "#2a78d6", "Deep tabular": "#eb6834", "Graph": "#1baf7a", "Foundation (frozen)": "#eda100"}
_MEMBER = {"xgboost": 0, "catboost": 1, "mlp": 0, "ft_transformer": 1, "ivis": 2, "hgnn": 0, "hgt": 1, "chronos2": 0, "timesfm": 1}
MARKERS = ["o", "s", "^"]
DASHES = ["-", "--", ":"]

WORST = "worst case (all attacks)"     # the attack name of the worst case over all attacks in the grid


def style_of(f):
    """Plot style of family f: colour of its model class, marker and line dash of its place within the class."""
    k = _MEMBER[f]
    return dict(color=CLASS_COLOR[FAM_CLASS[f]], marker=MARKERS[k], linestyle=DASHES[k])


# sequential blue scale of the transfer heatmaps, light to dark
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def pm(x):
    """'mean ± sd' over seeds with three decimals ('mean' alone for a single seed)."""
    x = np.asarray(x, float)
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def _merged_npz(paths, skip=()):
    """All arrays of the .npz files in paths, in one dict (a later file overrides an earlier one), leaving out
    the names in skip."""
    arrays = {}
    for path in paths:
        z = np.load(path)
        arrays.update({k: z[k] for k in z.files if k not in skip})
    return arrays


# (key in metrics.json, column label) of the clean-performance table, in column order
CLEAN_METRICS = [("auc", "AUC"), ("auc_usd", "AUC ($)"), ("ks", "KS"), ("ks_usd", "KS ($)"),
                 ("auprc", "AUPRC"), ("recall@1%fpr", "Recall@1%FPR")]


def table_clean():
    """Clean detection of every model on every dataset, mean ± sd over seeds: reports/table1_clean.{csv,md}.
    The CSV (with the numeric means in the "_<metric>" columns) feeds Table S-IV, the text and Figure S7."""
    rows = []
    for d in DS:
        for f in FAM:
            metrics_path = RUNS / d / f / "metrics.json"
            if not metrics_path.exists():
                continue
            per_seed = json.loads(metrics_path.read_text())
            row = {"Dataset": DS_NAME[d], "Class": FAM_CLASS[f], "Model": FAM_NAME[f]}
            for key, label in CLEAN_METRICS:
                row[label] = pm([m[key] for m in per_seed])
                row["_" + key] = np.mean([m[key] for m in per_seed])
            best_params = json.loads((RUNS / d / f / "best_params.json").read_text())
            row["HPO trials"] = best_params["n_trials"]
            row["Eval rows"] = "subset" if best_params.get("fm_subset") else "full"
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    shown_columns = [c for c in df.columns if not c.startswith("_")]
    # bold the best mean per dataset for each metric in the markdown version
    show = df[shown_columns].copy()
    for key, label in CLEAN_METRICS:
        is_best = df["_" + key] == df.groupby("Dataset")["_" + key].transform("max")
        show.loc[is_best, label] = "**" + show.loc[is_best, label] + "**"
    df.to_csv(REP / "table1_clean.csv", index=False)
    (REP / "table1_clean.md").write_text(show.to_markdown(index=False))
    return df


def load_attacks():
    """Every row of the attack grid: runs/<dataset>/attacks/results*.csv of all datasets in one frame."""
    frames = [pd.read_csv(p) for p in sorted(RUNS.glob("*/attacks/results*.csv")) if p.stat().st_size > 1]
    frames = [f for f in frames if len(f)]          # a file emptied by a merge would turn numeric columns into objects
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def table_robust(A: pd.DataFrame, eps=(0.1, 0.25, 0.5)):
    """ASR and dollar-weighted ASR of every model x attack x setting at each budget in eps, mean over datasets:
    reports/table2_robust.{csv,md}; the same per dataset with the ranking metrics: table2b_robust_<dataset>.md."""
    if A.empty:
        return
    A = A[A["eps"].isin(eps)].copy()
    A["Model"] = A["model"].map(FAM_NAME)
    A["Setting"] = A["mode"].str.capitalize()
    mean_asr = A.pivot_table(index=["Setting", "attack", "Model"], columns="eps", values=["asr", "asr_usd"], aggfunc="mean")
    mean_asr.columns = [f"{'ASR' if m == 'asr' else 'ASR ($)'} @ε={e}" for m, e in mean_asr.columns]
    mean_asr = mean_asr.reset_index().rename(columns={"attack": "Attack"})
    n_datasets = A.groupby(["mode", "model"])["dataset"].nunique().max()
    mean_asr.to_csv(REP / "table2_robust.csv", index=False)
    (REP / "table2_robust.md").write_text(f"Mean over {n_datasets} datasets.\n\n" + mean_asr.round(3).to_markdown(index=False))
    for d, g in A.groupby("dataset"):
        per_dataset = g.pivot_table(index=["Setting", "attack", "Model"], columns="eps",
                                    values=["asr", "asr_usd", "robust_auc", "robust_ks"], aggfunc="mean")
        per_dataset.columns = [f"{m}@{e}" for m, e in per_dataset.columns]
        (REP / f"table2b_robust_{d}.md").write_text(per_dataset.round(3).reset_index().to_markdown(index=False))


def table_attacks():
    """Catalog of the attacks (knowledge, norm, query budget, models, reference): reports/table3_attacks.md."""
    rows = [
        ("APGD (L∞, step-size schedule, momentum, random restarts)", "White-box, gradients", "L∞", "100 iterations x 5 restarts (FMs 20 x 2)", "MLP, FT-Transformer, ivis, graph, FMs", "Croce & Hein 2020"),
        ("AATS (axis-aligned threshold search, this work)", "White-box, tree structure", "L∞ (minimal)", "≈10³–10⁴ model calls", "XGBoost, CatBoost", "this work"),
        ("AATS + exact MILP inside the AATS radius (discrete-domain extension, this work)", "White-box, tree structure", "L∞ (exact minimal where solved)", "10 s/example, 20 min/model", "XGBoost, CatBoost", "Kantchelian et al. 2016"),
        ("Square Attack", "Score-based black-box", "L∞", "300 queries", "all", "Andriushchenko et al. 2020"),
        ("HopSkipJump", "Decision-based black-box", "L∞ (minimal)", "≈4,000 queries", "all", "Chen et al. 2020"),
        ("Transfer (high-confidence white-box from a surrogate)", "Transfer", "L∞, ε=0.5", "0 queries to target", "all", "Papernot et al. 2016"),
        ("Worst case over all of the above", "Ensemble", "L∞", "sum of the above", "all", "Croce & Hein 2020 (AutoAttack protocol)"),
    ]
    df = pd.DataFrame(rows, columns=["Attack", "Knowledge", "Norm", "Query budget", "Models", "Reference"])
    (REP / "table3_attacks.md").write_text(df.to_markdown(index=False))


def _legend(fig, axes, bottom):
    """One legend for all panels: union of series across panels, in fixed family order."""
    handle_of = {}
    for ax in np.ravel(axes):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handle_of.setdefault(label, handle)
    order = [FAM_NAME[f] for f in FAM if FAM_NAME[f] in handle_of]
    fig.legend([handle_of[l] for l in order], order, loc="lower center", ncol=min(5, len(order)), frameon=False, bbox_to_anchor=(0.5, bottom))


def _style(plt):
    """The house style of every report figure (font size, muted axes, no top/right spines, 200 dpi)."""
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2a28", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200})


MIN_TRANSFER_N = 5      # a transfer rate needs at least this many source evasions to replay; fewer are hatched


def _transfer_matrices(mode):
    """Transfer rates of every dataset in DS from runs/<d>/attacks/transfer_<mode>.npz.

    Returns (fams, rates, counts), or None when no dataset has the file. fams: the families of any dataset, in
    FAM order. rates[d]: source x target transfer rates on fams, NaN where a model is missing or fewer than
    MIN_TRANSFER_N evasions were replayed. counts[d]: the number of replayed evasions behind each rate (0 at NaN)."""
    loaded = {}
    for d in DS:
        path = RUNS / d / "attacks" / f"transfer_{mode}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=True)
            n_replayed = z["N"] if "N" in z else np.full(z["T"].shape, 99.0)   # a file without counts keeps every cell
            loaded[d] = (z["T"], n_replayed, list(z["families"]))
    if not loaded:
        return None
    present = set().union(*[set(file_fams) for _, _, file_fams in loaded.values()])
    fams = [f for f in FAM if f in present]
    rates, counts = {}, {}
    for d, (file_rates, n_replayed, file_fams) in loaded.items():
        M = np.full((len(fams), len(fams)), np.nan)
        C = np.zeros((len(fams), len(fams)))
        for i, source in enumerate(fams):
            for j, target in enumerate(fams):
                if source in file_fams and target in file_fams:
                    s, t = file_fams.index(source), file_fams.index(target)
                    if n_replayed[s, t] >= MIN_TRANSFER_N:
                        M[i, j] = file_rates[s, t]
                        C[i, j] = n_replayed[s, t]
        rates[d] = M
        counts[d] = C
    return fams, rates, counts


def _draw_transfer_panel(ax, M, cmap, labels, title, plt):
    """One heatmap of source x target transfer rates M: the rate in each cell, hatching where it is NaN."""
    n = len(labels)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6 if n > 6 else 7,
                        color="white" if M[i, j] > 0.55 else "#0b0b0b")
            else:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", edgecolor="#c9c8c2", linewidth=0))
    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n), labels, fontsize=7)
    ax.set_title(title, fontsize=9, loc="left")
    ax.spines[:].set_visible(False)


def fig_transfer(mode):
    """Transfer rate of white-box evasions at eps 0.5 between every pair of models, one heatmap per dataset plus
    the rate pooled over datasets: reports/fig_transfer_<mode>.{png,pdf} (Figure 1, Figure S3, and
    Figure S4 through report/consistent_tables.py)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    _style(plt)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    loaded = _transfer_matrices(mode)
    if loaded is None:
        return
    fams, rates, counts = loaded
    # pooled panel: every dataset's rates weighted by its number of replayed evasions
    weighted = [np.nan_to_num(rates[d]) * counts[d] for d in rates]
    n_datasets = len(weighted)
    panels = list(rates) + ["mean"]
    with np.errstate(all="ignore"):
        total = np.sum(list(counts.values()), 0)
        rates["mean"] = np.where(total > 0, np.sum(weighted, 0) / np.maximum(total, 1), np.nan)
    n = len(panels)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    size = 3.1 if len(fams) <= 5 else 3.9
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    labels = [FAM_NAME[f].replace(" (triplet)", "") for f in fams]
    pooled_title = f"Pooled over {n_datasets} dataset{'s' if n_datasets != 1 else ''}"
    for k, d in enumerate(panels):
        ax = axes[k // cols][k % cols]
        _draw_transfer_panel(ax, rates[d], cmap, labels, DS_NAME.get(d, pooled_title), plt)
        if k % cols == 0:
            ax.set_ylabel("source (attack crafted on)")
        if k // cols == rows - 1:
            ax.set_xlabel("target (attack replayed on)")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"Transfer rate of white-box evasions, ε = 0.5, {mode} setting "
                 f"(hatched: fewer than {MIN_TRANSFER_N} source evasions to replay, or model not trained)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"fig_transfer_{mode}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_curves(A: pd.DataFrame, mode, attack=WORST):
    """Worst-case ASR against the budget eps, one panel per dataset, one line per model:
    reports/fig_robust_curves_<mode>.{png,pdf} (Figure S1 for constrained, Figure S2 for unconstrained)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, NullLocator

    if A.empty:
        return
    _style(plt)
    g = A[(A["mode"] == mode) & (A["attack"] == attack)]
    ds = [d for d in DS if d in set(g["dataset"])]
    if not ds:
        return
    cols = min(3, len(ds))
    rows = int(np.ceil(len(ds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.6 * rows), squeeze=False, sharey=True)
    for k, d in enumerate(ds):
        ax = axes[k // cols][k % cols]
        for f in FAM:
            curve = g[(g["dataset"] == d) & (g["model"] == f)].sort_values("eps")
            if len(curve):
                ax.plot(curve["eps"], curve["asr"], markersize=5, linewidth=2, label=FAM_NAME[f], **style_of(f))
        ax.set_xscale("log")
        ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1.0], ["0.05", "0.1", "0.25", "0.5", "1"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.set_title(DS_NAME[d], fontsize=9, loc="left")
        ax.set_xlabel("ε (L∞, feature standard deviations)")
        if k % cols == 0:
            ax.set_ylabel("attack success rate")
    for k in range(len(ds), rows * cols):
        axes[k // cols][k % cols].axis("off")
    _legend(fig, axes, -0.12)
    fig.suptitle(f"Worst-case attack success rate vs budget, {mode} setting", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"fig_robust_curves_{mode}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _within_dataset_rho(g, x, y, n_perm=2000, seed=0):
    """Mean over datasets of the Spearman rank correlation between x and y across models, with a
    permutation p-value that shuffles x within each dataset (thickness scales differ across datasets,
    so only within-dataset ranks are comparable). Returns (rho, p, number of datasets used)."""
    from scipy.stats import spearmanr

    # models whose outputs are being recomputed have missing values, which also turn the columns into
    # object dtype: keep complete numeric pairs only
    g = g.assign(**{x: pd.to_numeric(g[x], errors="coerce"), y: pd.to_numeric(g[y], errors="coerce")}).dropna(subset=[x, y])
    groups = [h for _, h in g.groupby("dataset") if len(h) >= 3 and h[x].nunique() > 1 and h[y].nunique() > 1]
    if not groups:
        return np.nan, np.nan, 0

    def mean_rho(pairs):
        return np.mean([spearmanr(xs, ys).statistic for xs, ys in pairs])

    observed = mean_rho([(h[x].to_numpy(), h[y].to_numpy()) for h in groups])
    rng = np.random.default_rng(seed)
    null = np.array([mean_rho([(rng.permutation(h[x].to_numpy()), h[y].to_numpy()) for h in groups]) for _ in range(n_perm)])
    return observed, float((np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (n_perm + 1)), len(groups)


def _load_thickness(min_pairs):
    """Every runs/<d>/attacks/thickness*.csv (not the partial ones) in one frame, cells with at least min_pairs
    segments only; None when there is no file."""
    frames = []
    for path in sorted(RUNS.glob("*/attacks/thickness*.csv")):
        if path.stat().st_size > 1 and "partial" not in path.name:
            t = pd.read_csv(path)
            if "variant" not in t:
                t["variant"] = "adversarial"         # files without the column hold the adversarial-direction variant
            frames.append(t)
    if not frames:
        return None
    T = pd.concat(frames, ignore_index=True)
    T["variant"] = T["variant"].fillna("adversarial")
    return T[(T["n"] >= min_pairs) & T["thickness_mean"].notna()]


def _fig_thickness(G, correlations, eps_fig):
    """Scatter of boundary thickness against worst-case ASR at eps_fig, one panel per setting, titled with the
    within-dataset rho: reports/fig_thickness_vs_asr.{png,pdf} (Figure S5)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    col = f"asr@{eps_fig:g}"
    _style(plt)
    modes = [m for m in ("unconstrained", "constrained") if m in set(G["mode"])]
    fig, axes = plt.subplots(1, len(modes), figsize=(4.2 * len(modes), 3.4), squeeze=False, sharey=True)
    for k, mode in enumerate(modes):
        ax = axes[0][k]
        g = G[G["mode"] == mode]
        for f in FAM:
            h = g[g["model"] == f]
            if len(h):
                st = style_of(f)
                ax.scatter(h["thickness_mean"], h[col], s=64, color=st["color"], marker=st["marker"], edgecolor="white",
                           linewidth=1.5, label=FAM_NAME[f], zorder=3)
        match = [o for o in correlations if o["Setting"] == mode and o["eps"] == eps_fig and o["Variant"] == g["variant"].iloc[0]]
        rho = match[0]["mean_within_dataset_rho"] if match else np.nan
        nd = match[0]["datasets"] if match else 0
        ax.set_title(f"{mode.capitalize()}: within-dataset ρ = {rho:.2f} ({nd} dataset{'s' if nd != 1 else ''})", fontsize=9, loc="left")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("class-pair boundary thickness (L2, standardized)")
        ax.grid(color="#e6e5e0", linewidth=0.6)
        if k == 0:
            ax.set_ylabel(f"worst-case ASR at ε = {eps_fig:g}")
    _legend(fig, axes, -0.02)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"fig_thickness_vs_asr.{ext}", bbox_inches="tight")
    plt.close(fig)


def thickness_analysis(A: pd.DataFrame, min_pairs=10, eps_fig=0.5):
    """RQ3: does boundary thickness predict robustness? Writes reports/table4_thickness_corr.csv (Table S-XXIV,
    the rank correlations in the text), reports/table4_thickness.csv (every cell) and fig_thickness_vs_asr (Figure S5).

    Two thickness variants: along each fraud's own adversarial direction (attack grid; only frauds flipped at
    eps 0.5) and class-pair (each detected fraud to a legit the model passes; every detected fraud). Cells with
    fewer than min_pairs segments are dropped. Statistic: mean within-dataset Spearman rho between thickness and
    worst-case ASR across models, stratified permutation p-value."""
    T = _load_thickness(min_pairs)
    if T is None or A.empty:
        return
    # worst-case ASR per (dataset, model, mode), one column asr@<eps> per budget
    W = A[A["attack"] == WORST].pivot_table(index=["dataset", "model", "mode"], columns="eps", values="asr").reset_index()
    W.columns = [c if isinstance(c, str) else f"asr@{c:g}" for c in W.columns]
    J = T.merge(W, on=["dataset", "model", "mode"])
    if J.empty:
        return
    correlations = []
    for (variant, mode), g in J.groupby(["variant", "mode"]):
        for e in (0.25, 0.5, 1.0):
            col = f"asr@{e:g}"
            if col not in g:
                continue
            rho, p_value, n_datasets = _within_dataset_rho(g, "thickness_mean", col)
            correlations.append(dict(Variant=variant, Setting=mode, eps=e, datasets=n_datasets, cells=len(g),
                                     mean_within_dataset_rho=rho, p_perm=p_value))
    pd.DataFrame(correlations).round(4).to_csv(REP / "table4_thickness_corr.csv", index=False)
    J.assign(Model=J["model"].map(FAM_NAME), Dataset=J["dataset"].map(DS_NAME)).round(4).to_csv(REP / "table4_thickness.csv", index=False)
    G = J[J["variant"] == "class-pair"]        # the figure shows the class-pair variant when it exists
    if G.empty:
        G = J
    _fig_thickness(G, correlations, eps_fig)


def explain_tables():
    """Top global drivers per model, the features attacks move most (attribution shift) and the agreement of the
    models' rankings, per dataset: reports/table5_explain.md; then the numbers of explain_summary."""
    sections = []
    for d in DS:
        importance_path = RUNS / d / "explain" / "global_importance.csv"
        if not importance_path.exists():
            continue
        G = pd.read_csv(importance_path, index_col=0)
        top = {FAM_NAME.get(f, f): ", ".join(G[f].sort_values(ascending=False).index[:5]) for f in G.columns}
        sections.append(f"### {DS_NAME[d]}\n\n" + pd.DataFrame({"Top-5 drivers (mean |attribution|)": top}).to_markdown())
        shift_path = RUNS / d / "explain" / "attack_shift.csv"
        if shift_path.exists() and shift_path.stat().st_size > 1:
            S = pd.read_csv(shift_path)
            if len(S):
                t = (S.sort_values("mean_abs_dattr", ascending=False).groupby(["model", "mode"]).head(3)
                     .groupby(["model", "mode"])["feature"].apply(", ".join).unstack())
                t.index = [FAM_NAME.get(i, i) for i in t.index]
                sections.append("Largest attribution shift under attack (top 3 features):\n\n" + t.to_markdown())
        agreement_path = RUNS / d / "explain" / "agreement.csv"
        if agreement_path.exists():
            Ag = pd.read_csv(agreement_path, index_col=0)
            Ag.index = [FAM_NAME.get(i, i) for i in Ag.index]
            Ag.columns = [FAM_NAME.get(i, i) for i in Ag.columns]
            sections.append("Kendall τ between models' global importance rankings:\n\n" + Ag.round(2).to_markdown())
    if sections:
        (REP / "table5_explain.md").write_text("\n\n".join(sections))
    explain_summary()


HIST_FEATS = ("cnt_prev", "log_amt_mean_prev", "log_ratio", "log_dt_prev", "cnt_24h", "logamt_sum_24h")
TREE_F, NEURAL_F = ("xgboost", "catboost"), ("mlp", "ft_transformer", "ivis")


def explain_summary():
    """Table 5 in numbers: reports/table5_explain_summary.csv and table5_explain_agreement.csv (Table S-XXXII, RQ4
    numbers in the text). Per dataset and model: share of the global attribution mass (mean |attribution|)
    on attacker-controllable features and on history features, share of the three largest features, the top
    five features, and what successful constrained attacks move (largest mean |dx| and largest attribution
    shift). Per dataset: Kendall tau between models' global rankings, within and across model classes."""
    from fraudadv.data.load import load

    rows, agreement = [], []
    for d in DS:
        importance_path = RUNS / d / "explain" / "global_importance.csv"
        if not importance_path.exists():
            continue
        _, meta = load(d)
        mutable = set(meta["mutable"])
        G = pd.read_csv(importance_path, index_col=0)
        shift_path = RUNS / d / "explain" / "attack_shift.csv"
        S = pd.read_csv(shift_path) if shift_path.exists() and shift_path.stat().st_size > 1 else pd.DataFrame(columns=["model", "mode"])
        for f in G.columns:
            g = G[f].dropna().abs()
            if g.sum() <= 0:
                continue
            share = (g / g.sum()).sort_values(ascending=False)
            history = [c for c in share.index if c.split("dest_")[-1] in HIST_FEATS or c.startswith("isna_")]
            r = dict(dataset=d, model=f, n_features=len(share), n_controllable=int(sum(c in mutable for c in share.index)),
                     share_controllable=float(share[[c for c in share.index if c in mutable]].sum()),
                     share_history=float(share[history].sum()), share_top3=float(share.iloc[:3].sum()),
                     top5=", ".join(share.index[:5]))
            c = S[(S["model"] == f) & (S["mode"] == "constrained")]
            if len(c):
                r.update(n_evasions=int(c["n"].iloc[0]),
                         moved_top3=", ".join(c.sort_values("mean_abs_dx", ascending=False)["feature"].head(3)),
                         shift_top3=", ".join(c.sort_values("mean_abs_dattr", ascending=False)["feature"].head(3)))
            rows.append(r)
        agreement_path = RUNS / d / "explain" / "agreement.csv"
        if agreement_path.exists():
            Ag = pd.read_csv(agreement_path, index_col=0)
            fams = list(Ag.index)
            model_class = {f: ("tree" if f in TREE_F else "neural" if f in NEURAL_F else "history-aware") for f in fams}
            within, across = [], []
            for i, x in enumerate(fams):
                for y in fams[i + 1:]:
                    (within if model_class[x] == model_class[y] else across).append(float(Ag.loc[x, y]))
            every_pair = within + across
            agreement.append(dict(dataset=d, models=len(fams), tau_min=min(every_pair), tau_median=float(np.median(every_pair)),
                                  tau_max=max(every_pair),
                                  tau_within_class=float(np.mean(within)) if within else np.nan,
                                  tau_across_class=float(np.mean(across)) if across else np.nan,
                                  tau_trees=float(Ag.loc["xgboost", "catboost"]) if {"xgboost", "catboost"} <= set(fams) else np.nan))
    if rows:
        pd.DataFrame(rows).round(4).to_csv(REP / "table5_explain_summary.csv", index=False)
    if agreement:
        pd.DataFrame(agreement).round(3).to_csv(REP / "table5_explain_agreement.csv", index=False)


def stats_tables(A: pd.DataFrame, eps=0.25):
    """Inference at eps: reports/table6_worstcase_ci.csv (Wilson CIs on the worst-case ASR; also read by Figure S7),
    table6_friedman.csv (Friedman + Nemenyi ranks across datasets and settings, Table S-XXXIX), table6_mcnemar.csv
    (paired McNemar between models on the same frauds) and table6_mixed_model*.csv (Bayesian mixed model of
    per-example evasion, Table S-XXXVIII)."""
    from fraudadv.stats.inference import friedman_nemenyi, mixed_model, pairwise_mcnemar, wilson_ci

    if A.empty:
        return
    W = A[(A["attack"] == WORST) & (A["eps"] == eps)].copy()
    ci = [wilson_ci(round(asr * n_det), n_det) for asr, n_det in zip(W["asr"], W["n_detected"].clip(lower=0))]
    W["asr_lo95"], W["asr_hi95"] = [c[0] for c in ci], [c[1] for c in ci]
    W[["dataset", "mode", "model", "n_detected", "asr", "asr_lo95", "asr_hi95", "asr_usd"]].round(4).to_csv(
        REP / "table6_worstcase_ci.csv", index=False)

    # Friedman test over blocks (dataset, setting) of the worst-case ASR. PaySim's evasions come from a simulator
    # leak (a tiny overdraw flips the balance-error feature), so every pooled test is also reported without it.
    friedman = []
    for name, fams, drop in [("tabular families, all datasets", FAM[:5], ()), ("all families with history models", FAM[:7], ()),
                             ("tabular families, without PaySim", FAM[:5], ("paysim",)),
                             ("all families with history models, without PaySim", FAM[:7], ("paysim",))]:
        blocks = W[~W["dataset"].isin(drop)].pivot_table(index=["dataset", "mode"], columns="model", values="asr")
        cols = [f for f in fams if f in blocks.columns]
        blocks = blocks[cols].dropna()
        if len(blocks) < 2 or len(cols) < 3:
            continue
        p, ranks, cd = friedman_nemenyi(blocks.values)
        friedman.append(dict(comparison=name, blocks=len(blocks), models=len(cols), friedman_p=p, nemenyi_cd=cd,
                             **{f"rank_{FAM_NAME[c]}": r for c, r in zip(cols, ranks)}))
    if friedman:
        pd.DataFrame(friedman).to_csv(REP / "table6_friedman.csv", index=False)

    # per-example outcomes: McNemar pairs, and one row per detected fraud for the mixed model
    mcnemar, evasions = [], []
    for d in DS:
        files = sorted((RUNS / d / "attacks").glob("per_example*.npz"))
        if not files:
            continue
        per_example = _merged_npz(files)
        for mode in ("unconstrained", "constrained"):
            fams = [f for f in FAM if f"{f}|{mode}|{eps}" in per_example]
            for r in pairwise_mcnemar(per_example, fams, mode, eps):
                mcnemar.append(dict(dataset=d, mode=mode, **r))
            for f in fams:
                X = per_example[f"{f}|{mode}|{eps}"]       # columns: sample row, detected, evaded
                X = X[X[:, 1] == 1]
                evasions += [dict(dataset=d, mode=mode, model=f, evaded=int(e)) for e in X[:, 2]]
    if mcnemar:
        pd.DataFrame(mcnemar).round(5).to_csv(REP / "table6_mcnemar.csv", index=False)
    if evasions:
        R = pd.DataFrame(evasions)
        for name, df in [("table6_mixed_model", R), ("table6_mixed_model_nopaysim", R[R["dataset"] != "paysim"])]:
            try:
                mixed_model(df).round(4).to_csv(REP / f"{name}.csv", index=False)
            except Exception as e:  # report generation must not fail on a degenerate fit
                (REP / f"{name}.err").write_text(repr(e))


# defenses take slots 1-5 of the validated reference palette in this fixed order; the marker names the model
DEFENSE_COLOR = {"pgd_at": "#2a78d6", "trades": "#eb6834", "adv_aug": "#7b5bd6", "adv_boost": "#1baf7a", "monotone": "#eda100",
                 "squeeze": "#e87ba4"}
DEFENSE_NAME = {"pgd_at": "PGD adversarial training", "trades": "TRADES", "adv_aug": "adversarial augmentation",
                "adv_boost": "adversarial boosting",
                "monotone": "monotone constraints", "squeeze": "feature squeezing"}
MODEL_MARKER = {"xgboost": "o", "catboost": "s", "mlp": "^", "ft_transformer": "D", "ivis": "v", "hgnn": "P", "hgt": "X",
                "chronos2": "*", "timesfm": "h"}
DEFENSE_ORDER = ["none", "pgd_at", "trades", "adv_aug", "adv_boost", "monotone", "squeeze"]     # row order of table7_defenses


def fig_defense_tradeoff(T: pd.DataFrame, eps):
    """RQ5 figure: what each defense buys and costs: reports/fig_defense_tradeoff.{png,pdf} (Figure S6) and its
    data, reports/table7_defense_tradeoff.csv. One point per dataset x model x defense, as the change
    from the undefended model on the same dataset: worst-case ASR under the adaptive attack (y) against clean
    AUC and clean recall at 1% FPR (x, one panel each). Points below zero are more robust; points left of
    zero paid for it in clean performance."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    base = T[T["defense"] == "none"].set_index(["dataset", "model"])
    D = T[T["defense"] != "none"].copy()
    D = D[[k in base.index for k in zip(D["dataset"], D["model"])]]
    if D.empty:
        return
    for c in ("auc", "recall", "asr"):
        D["d_" + c] = [v - base.loc[(d, f), c] for d, f, v in zip(D["dataset"], D["model"], D[c])]
    D.round(4).to_csv(REP / "table7_defense_tradeoff.csv", index=False)
    _style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, (c, xlabel) in zip(axes, [("d_auc", "change in clean AUC"), ("d_recall", "change in clean recall at 1% FPR")]):
        ax.axhline(0, color="#8a8984", linewidth=0.8, zorder=1)
        ax.axvline(0, color="#8a8984", linewidth=0.8, zorder=1)
        for (defense, f), g in D.groupby(["defense", "model"]):
            ax.scatter(g[c], 100 * g["d_asr"], s=42, color=DEFENSE_COLOR.get(defense, "#52514e"), marker=MODEL_MARKER.get(f, "o"),
                       edgecolor="white", linewidth=0.8, alpha=0.9, zorder=3)
        ax.grid(color="#e6e5e0", linewidth=0.6, zorder=0)
        ax.set_xlabel(xlabel)
        lo, hi = float(D[c].min()), float(D[c].max())
        pad = 0.08 * max(hi - lo, 0.02)
        ax.set_xlim(lo - pad, max(hi, 0.25 * abs(lo)) + pad)      # most defenses cost clean performance: keep the room on that side
        ax.xaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_ylabel(f"change in worst-case ASR at ε={eps}\n(percentage points)")
    axes[0].annotate("more robust, at a clean cost", xy=(0.02, 0.03), xycoords="axes fraction", ha="left", va="bottom", fontsize=7.5, color="#52514e")
    axes[0].annotate("less robust", xy=(0.02, 0.97), xycoords="axes fraction", ha="left", va="top", fontsize=7.5, color="#52514e")
    defense_handles = [Line2D([], [], marker="o", linestyle="", markersize=7, color=DEFENSE_COLOR[k], markeredgecolor="white",
                              label=DEFENSE_NAME[k]) for k in DEFENSE_COLOR if k in set(D["defense"])]
    model_handles = [Line2D([], [], marker=MODEL_MARKER[f], linestyle="", markersize=6.5, color="#52514e", markeredgecolor="white",
                            label=FAM_NAME[f]) for f in MODEL_MARKER if f in set(D["model"])]
    fig.legend(handles=defense_handles, loc="lower center", ncol=len(defense_handles), frameon=False, bbox_to_anchor=(0.5, -0.07),
               fontsize=7.5, handletextpad=0.2, columnspacing=1.0)
    fig.legend(handles=model_handles, loc="lower center", ncol=len(model_handles), frameon=False, bbox_to_anchor=(0.5, -0.14),
               fontsize=7.5, handletextpad=0.2, columnspacing=1.0)
    fig.suptitle("Defenses under adaptive attack: change from the undefended model, one point per dataset and model", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"fig_defense_tradeoff.{ext}", bbox_inches="tight")
    plt.close(fig)


def _defense_rows(D):
    """One row per (dataset, model, defense) of the worst-case defense results D: clean metrics and ASR."""
    rows = []
    for (d, f, defense), g in D.groupby(["dataset", "model", "defense"]):
        r = g.iloc[0]
        rows.append(dict(dataset=d, model=f, defense=defense, auc=r["clean_auc"], auc_usd=r["clean_auc_usd"], ks=r["clean_ks"],
                         ks_usd=r["clean_ks_usd"], recall=r["clean_recall@1%fpr"], asr=r["asr"], asr_usd=r["asr_usd"]))
    return rows


def _grid_baseline_rows(D, A, eps, defense_mode):
    """Undefended baselines that run_defenses did not evaluate itself: clean metrics of seed 0 (metrics.json)
    and the grid's worst-case ASR at eps in the dataset's defense threat model."""
    rows = []
    for d in D["dataset"].unique():
        for f in D[D["dataset"] == d]["model"].unique():
            p = RUNS / d / f / "metrics.json"
            if not p.exists() or A.empty or ((D["dataset"] == d) & (D["model"] == f) & (D["defense"] == "none")).any():
                continue                    # baseline evaluated by run_defenses itself takes precedence
            m = json.loads(p.read_text())[0]
            w = A[(A["dataset"] == d) & (A["model"] == f) & (A["mode"] == defense_mode(d))
                  & (A["attack"] == WORST) & (A["eps"] == eps)]
            if len(w):
                rows.append(dict(dataset=d, model=f, defense="none", auc=m["auc"], auc_usd=m["auc_usd"], ks=m["ks"], ks_usd=m["ks_usd"],
                                 recall=m["recall@1%fpr"], asr=float(w["asr"].iloc[0]), asr_usd=float(w["asr_usd"].iloc[0])))
    return rows


def defense_tables(A: pd.DataFrame, eps=0.25):
    """Table 7: every defense vs its undefended baseline: clean AUC / KS (unit and $), recall at 1% FPR,
    and worst-case ASR at eps in each dataset's defense threat model (adaptive attacks). Writes
    reports/table7_defenses_by_dataset.csv (Table VII, Figure S7), the mean over datasets in
    reports/table7_defenses.{md,csv}, and Figure S6 through fig_defense_tradeoff."""
    # results.csv: the tabular defenses; results_<tag>.csv: later runs (graph and foundation models, ivis augmentation)
    frames = [pd.read_csv(p) for p in sorted(RUNS.glob("*/defenses/results*.csv")) if p.stat().st_size > 1]
    if not frames:
        return
    from fraudadv.defenses.run_defenses import DEFENSE_MODE

    def defense_mode(d):
        return DEFENSE_MODE.get(d, "constrained")

    D = pd.concat(frames, ignore_index=True)
    if "mode" not in D:
        D["mode"] = "constrained"
    D["mode"] = D["mode"].fillna("constrained")
    D = D[(D["attack"] == WORST) & (D["eps"] == eps) & (D["mode"] == D["dataset"].map(defense_mode))]
    T = pd.DataFrame(_defense_rows(D) + _grid_baseline_rows(D, A, eps, defense_mode))
    if T.empty or not (T["defense"] != "none").any():
        return
    T.round(4).to_csv(REP / "table7_defenses_by_dataset.csv", index=False)
    fig_defense_tradeoff(T, eps)
    M = T.groupby(["model", "defense"]).agg(datasets=("dataset", "nunique"), AUC=("auc", "mean"), AUC_usd=("auc_usd", "mean"), KS=("ks", "mean"),
                                            KS_usd=("ks_usd", "mean"), Recall=("recall", "mean"), ASR=("asr", "mean"), ASR_usd=("asr_usd", "mean")).reset_index()
    # rows in family order, then baseline first and the defenses in DEFENSE_ORDER
    M = M.assign(_f=M["model"].map({f: i for i, f in enumerate(FAM)}), _d=M["defense"].map({k: i for i, k in enumerate(DEFENSE_ORDER)}))
    M = M.sort_values(["_f", "_d"]).drop(columns=["_f", "_d"])
    M["defense"] = M["defense"].map({"none": "none (baseline)", **DEFENSE_NAME})
    M["model"] = M["model"].map(FAM_NAME)
    M = M.rename(columns={"AUC_usd": "AUC ($)", "KS_usd": "KS ($)", "Recall": "Recall@1%FPR", "ASR": f"worst ASR @ε={eps}", "ASR_usd": f"worst ASR ($) @ε={eps}"})
    (REP / "table7_defenses.md").write_text("Constrained setting (unconstrained on ULB, where only amount and hour are attacker-controllable), adaptive attacks; mean over datasets.\n\n" + M.round(3).to_markdown(index=False))
    M.round(3).to_csv(REP / "table7_defenses.csv", index=False)


def pareto_figure():
    """Clean-robust Pareto frontier of every model and defense: reports/fig_pareto.{png,pdf} and table16_pareto.csv
    (Figure S7), drawn by fraudadv.report.pareto from the defense, clean and worst-case CSVs written above, plus the
    LoRA / zero-shot points of fraudadv.report.fm_adapt when present. A failure is written to fig_pareto.err."""
    try:
        from fraudadv.report import pareto

        fm_adapt_points = REP / "pareto_extra_fm_adapt.csv"
        pareto.main(sorted(RUNS.glob("*/attacks/pareto_extra.csv")) + ([fm_adapt_points] if fm_adapt_points.exists() else []))
    except Exception as e:                     # report generation must not fail on it
        (REP / "fig_pareto.err").write_text(repr(e))


def certify_table():
    """Table 8: how close AATS gets to the exact MILP optimum on tree ensembles: reports/table8_aats_vs_milp.{csv,md}
    (Table S-XXII, Table S-XXIII)."""
    frames = [pd.read_csv(p) for p in sorted(RUNS.glob("*/attacks/certify_*.csv")) if p.stat().st_size > 1]
    if not frames:
        return
    C = pd.concat(frames, ignore_index=True)
    if "mode" not in C:
        C["mode"] = np.nan
    C["mode"] = C["mode"].fillna("continuous")      # first run (certify_<family>.csv): continuous model only
    # relative gaps are meaningless when the optimum is a float32 rounding step (a threshold sitting exactly
    # on the clean value): count those as "evadable at ~0" and use the absolute gap for them
    tiny = C["milp_dist"] < 1e-3
    C["gap_rel"] = np.where(tiny, np.nan, C["gap"])
    C["gap_abs"] = np.where(np.isfinite(C["aats_cost"]) & np.isfinite(C["milp_dist"]), C["aats_cost"] - C["milp_dist"], np.nan)
    C["near_zero"] = tiny & (C["status"] == "optimal")
    g = C.groupby(["dataset", "model", "mode"])

    def rel(x):
        return x["gap_rel"].dropna()

    T = pd.DataFrame({
        "n": g.size(),
        "exact or AATS certified optimal": g.apply(lambda x: np.mean(x["status"].isin(["optimal", "aats_optimal"]))),
        "certified robust at ε=1": g.apply(lambda x: np.mean(x["status"] == "robust")),
        "timeouts": g.apply(lambda x: np.mean(x["status"] == "timeout")),
        "optimum < 0.001": g["near_zero"].mean(),
        "median gap (AATS/opt - 1)": g.apply(lambda x: np.median(rel(x)) if len(rel(x)) else np.nan),
        "share within 5% of optimum": g.apply(lambda x: np.mean(rel(x) <= 0.05) if len(rel(x)) else np.nan),
        "median abs gap": g["gap_abs"].median(),
        "AATS s/example": g["aats_seconds"].mean(),
        "MILP s/example": g["milp_seconds"].mean(),
    }).reset_index()
    T["model"] = T["model"].map(FAM_NAME)
    T["dataset"] = T["dataset"].map(DS_NAME)
    T.round(3).to_csv(REP / "table8_aats_vs_milp.csv", index=False)
    (REP / "table8_aats_vs_milp.md").write_text(
        "AATS against the exact MILP. mode: constrained / unconstrained = the grid's threat models (discrete domains, "
        "derived features); continuous = first run, all features continuous and mutable. Relative gaps only where the "
        "optimum is at least 0.001.\n\n" + T.round(3).to_markdown(index=False))


def robust_auc_table(A: pd.DataFrame, eps=0.25):
    """Table 9: ranking metrics under the two attacker objectives: reports/table9_robust_auc.{csv,md} (Table S-XXX,
    Table S-XXXI). Min-effort (the grid: each fraud moved just past the threshold) vs score-minimising (each
    fraud pushed as low as eps allows)."""
    frames = [pd.read_csv(p) for p in sorted(RUNS.glob("*/attacks/robust_auc*.csv")) if p.stat().st_size > 1]
    if not frames or A.empty:
        return
    R = pd.concat(frames, ignore_index=True)
    R = R[R["eps"] == eps]
    W = A[(A["attack"] == WORST) & (A["eps"] == eps)]
    W = W.assign(dauc_me=W["robust_auc"] - W["clean_auc"], dks_me=W["robust_ks"] - W["clean_ks"])[
        ["dataset", "model", "mode", "asr", "dauc_me", "dks_me"]].rename(columns={"asr": "ASR"})
    J = R.merge(W, on=["dataset", "model", "mode"], how="left")
    # each attacker's change is measured against clean scores of its own fraud subset
    T = pd.DataFrame({"Dataset": J["dataset"].map(DS_NAME), "Setting": J["mode"], "Model": J["model"].map(FAM_NAME),
                      "ASR (min-effort)": J["ASR"], "AUC clean": J["clean_auc"], "ΔAUC min-effort": J["dauc_me"],
                      "ΔAUC score-min": J["robust_auc"] - J["clean_auc"], "ΔAUC ($) score-min": J["robust_auc_usd"] - J["clean_auc_usd"],
                      "KS clean": J["clean_ks"], "ΔKS min-effort": J["dks_me"], "ΔKS score-min": J["robust_ks"] - J["clean_ks"],
                      "ΔKS ($) score-min": J["robust_ks_usd"] - J["clean_ks_usd"]})
    T = T.sort_values(["Dataset", "Setting", "Model"])
    T.round(3).to_csv(REP / "table9_robust_auc.csv", index=False)
    (REP / "table9_robust_auc.md").write_text(
        f"ε = {eps}. Min-effort: each fraud moved just past the 1%-FPR threshold (the attack grid). Score-min: each fraud "
        "pushed to the lowest score reachable within ε (APGD or exact MILP, plus Square, no early stop), on a fixed "
        "random subset of the attack sample. Δ = attacked minus clean on the same fraud subset.\n\n" + T.round(3).to_markdown(index=False))


def history_table():
    """Table 10: history injection (camouflage) ASR by number of injected transactions k:
    reports/table10_history.{csv,md} (Table S-XII)."""
    # history.csv: tabular and graph families; history_fm.csv: the two foundation models (run separately)
    files = sorted(RUNS.glob("*/attacks/history.csv")) + sorted(RUNS.glob("*/attacks/history_fm*.csv"))
    frames = [pd.read_csv(p) for p in files if p.stat().st_size > 1]
    if not frames:
        return
    H = pd.concat(frames, ignore_index=True)
    H["Dataset"], H["Model"] = H["dataset"].map(DS_NAME), H["model"].map(FAM_NAME)
    P = H.pivot_table(index=["Dataset", "attack", "Model"], columns="k", values=["asr", "asr_usd"])
    P.columns = [f"{m}@k={k}" for m, k in P.columns]
    P = P.reset_index()
    P.round(3).to_csv(REP / "table10_history.csv", index=False)
    (REP / "table10_history.md").write_text(
        "History injection: k innocuous same-card transactions, 10 minutes apart, just before the fraud. ASR among "
        "detected frauds; an attack fails if any injected transaction is itself flagged.\n\n" + P.round(3).to_markdown(index=False))


def _same_sample(P, S):
    """True when the per-example outcomes P cover exactly the attack sample S, row for row."""
    return P is not None and len(P) == len(S) and np.array_equal(P[:, 0].astype(int), S)


def halves_table(eps=0.25):
    """Table 11: reports/table11_halves.csv (Table S-XL). The fraud sample is stored in entity order, and a first
    version of the grid ran the black-box attacks on its first half only. Per dataset, model and setting: share of
    the sample each model detects and worst-case ASR on the first half (the former black-box subset, sample.npz
    n_bb) and on the rest. Needs the full-sample grid (per-example outcomes for every row of the sample)."""
    from fraudadv.stats.inference import wilson_ci

    rows = []
    for d in DS:
        sample_path = RUNS / d / "attacks" / "sample.npz"
        if not sample_path.exists():
            continue
        z = np.load(sample_path)
        S, n_first = z["S"], int(z["n_bb"])
        per_example = _merged_npz(sorted((RUNS / d / "attacks").glob("per_example*.npz")))
        for f in FAM:
            for mode in ("constrained", "unconstrained"):
                P = per_example.get(f"{f}|{mode}|{eps}")
                if not _same_sample(P, S):
                    continue
                r = dict(dataset=d, model=f, mode=mode, eps=eps)
                for half, rows_of_half in (("first", slice(0, n_first)), ("second", slice(n_first, None))):
                    detected, evaded = P[rows_of_half, 1] == 1, P[rows_of_half, 2] == 1
                    n, k = int(detected.sum()), int((detected & evaded).sum())
                    lo, hi = wilson_ci(k, n)
                    r.update({f"n_{half}": int(detected.size), f"detected_{half}": n, f"asr_{half}": k / n if n else np.nan,
                              f"asr_{half}_lo95": lo, f"asr_{half}_hi95": hi})
                rows.append(r)
    if rows:
        pd.DataFrame(rows).round(4).to_csv(REP / "table11_halves.csv", index=False)


def groups_table(eps=0.25):
    """Table 12: reports/table12_groups.csv (Table S-XLIII, history-injection numbers in the text). Datasets where
    only part of the rows have an entity (IEEE-CIS: transactions with no billing address get no client id,
    entity -1, and are pooled under one pseudo-client by the preparation). Per model and group: clean AUC and
    recall at the deployed threshold on the whole test set, share of the fraud sample detected, worst-case ASR
    per setting (needs the full-sample grid) and history injection (best of 4, k=8; needs
    history_per_example_pe.npz)."""
    from sklearn.metrics import roc_auc_score

    from fraudadv.data.load import load
    from fraudadv.stats.inference import wilson_ci

    rows = []
    for d in DS:
        out = RUNS / d / "attacks"
        if not (out / "sample.npz").exists() or not (out / "tau.json").exists():
            continue
        arrays, _ = load(d)
        entity, y = arrays["ent_test"], arrays["y_test"]
        no_id = entity == -1
        if not 0 < no_id.mean() < 1:
            continue
        S = np.load(out / "sample.npz")["S"]
        tau = json.loads((out / "tau.json").read_text())
        per_example = _merged_npz(sorted(out.glob("per_example*.npz")))
        history_path = out / "history_per_example_pe.npz"
        H = np.load(history_path) if history_path.exists() else None
        for f in FAM:
            scores_path = RUNS / d / f / "test_scores_seed0.npy"
            if f not in tau or not scores_path.exists():
                continue
            s = np.load(scores_path)
            if len(s) != len(y):
                continue
            thr = float(tau[f])                     # deployed threshold (1% FPR on validation), logit scale
            for name, g in (("no client id", no_id), ("client id", ~no_id)):
                r = dict(dataset=d, model=f, group=name, rows=int(g.sum()), frauds=int(y[g].sum()),
                         clean_auc=roc_auc_score(y[g], s[g]), recall_at_tau=float((s[g & (y == 1)] > thr).mean()),
                         fpr_at_tau=float((s[g & (y == 0)] > thr).mean()), sample=int(g[S].sum()))
                for mode in ("constrained", "unconstrained"):
                    P = per_example.get(f"{f}|{mode}|{eps}")
                    if not _same_sample(P, S):
                        continue
                    detected, evaded = (P[:, 1] == 1) & g[S], (P[:, 2] == 1) & g[S]
                    n, k = int(detected.sum()), int((detected & evaded).sum())
                    lo, hi = wilson_ci(k, n)
                    r.update({"detected": n, f"asr_{mode}": k / n if n else np.nan, f"asr_{mode}_lo95": lo,
                              f"asr_{mode}_hi95": hi})
                if H is not None and f"{f}|detected" in H.files and np.array_equal(H["S"], S):
                    detected = H[f"{f}|detected"] & g[S]
                    evaded = H[f"{f}|best of 4 (4 queries)|8"] & g[S]
                    r.update(hist_detected=int(detected.sum()), hist_asr_k8=float(evaded.sum() / max(detected.sum(), 1)))
                rows.append(r)
    if rows:
        pd.DataFrame(rows).round(4).to_csv(REP / "table12_groups.csv", index=False)


def _first_grid_rerun(d):
    """Folder holding the whole-sample rerun of the first grid (single-start PGD): runs/<d>/attacks/ at first; once
    the final grid (APGD, snapping) replaced it there, it was archived to archive_attacks_20261114/attacks/."""
    archived = RUNS / d / "archive_attacks_20261114" / "attacks"
    return archived if archived.is_dir() else RUNS / d / "attacks"


def rerun_table():
    """Table 13: reports/table13_rerun_agreement.csv. The full-sample grid reran every attack, so on the examples
    both runs attacked (the first half of the sample) the two runs can be compared outcome by outcome. An outcome
    is one detected fraud in one cell (model, setting, budget). Per dataset and model: outcomes compared, evaded in
    the first run only, in the rerun only, and the largest number of differing examples in one cell."""
    rows = []
    for d in DS:
        first = _merged_npz(sorted((RUNS / d / "archive_half").glob("per_example*.npz")))
        rerun = _merged_npz(sorted(_first_grid_rerun(d).glob("per_example_full_*.npz")))
        by_model = {}
        for cell, o in first.items():
            if cell not in rerun:
                continue
            n = rerun[cell]
            position = {int(s): i for i, s in enumerate(n[:, 0])}
            ix = np.array([position.get(int(s), -1) for s in o[:, 0]])     # row of each first-run example in the rerun
            ok = (ix >= 0) & (o[:, 1] == 1)                                  # in both runs, detected
            first_only = int(((o[ok, 2] == 1) & (n[ix[ok], 2] != 1)).sum())
            rerun_only = int(((o[ok, 2] != 1) & (n[ix[ok], 2] == 1)).sum())
            model = cell.split("|")[0]
            r = by_model.setdefault(model, dict(dataset=d, model=model, cells=0, outcomes=0,
                                                first_run_only=0, rerun_only=0, cells_differing=0, max_in_cell=0))
            r["cells"] += 1
            r["outcomes"] += int(ok.sum())
            r["first_run_only"] += first_only
            r["rerun_only"] += rerun_only
            r["cells_differing"] += int(first_only + rerun_only > 0)
            r["max_in_cell"] = max(r["max_in_cell"], first_only + rerun_only)
        rows += list(by_model.values())
    if rows:
        pd.DataFrame(rows).to_csv(REP / "table13_rerun_agreement.csv", index=False)


def _rerun_summary(g):
    """Summary of a group of rerun_attack_table cells: how many differ, in which direction, and by how much."""
    return dict(cells=len(g), outcomes=int(g.detected.sum()), cells_differing=int((g["diff"] != 0).sum()),
                rerun_higher=int((g["diff"] > 0).sum()), rerun_lower=int((g["diff"] < 0).sum()),
                max_count=int(g["diff"].abs().max()), median_points=float(g.points.median()),
                p90_points=float(g.points.quantile(0.9)), max_points=float(g.points.max()))


def rerun_attack_table():
    """Table 14: reports/table14_rerun_by_attack.csv (Table S-XLI) and its cells, table14_rerun_by_attack_cells.csv.
    The same comparison of the first grid with the whole-sample rerun, by attack. The first grid kept
    per-example outcomes only for the worst case, so single attacks are compared by their number of successes on
    the same frauds: the first grid's result rows against the rerun's stored scores on the same leading rows of
    the sample. PGD and the heuristic tree attack are deterministic. Square is seeded but vectorised, so its
    random draws change with the number of examples in the batch. The exact tree attack runs under a time limit.
    HopSkipJump (ART) drew its starting points from an unseeded generator in both runs (see attacks/hsjguard.py),
    so its differences measure how much two runs of that attack differ. Foundation models are left out: their
    rerun used another sample. One row per (dataset, attack) and per attack over all datasets; differences in
    percentage points of the detected frauds."""
    cells = []
    for d in DS:
        first_files = sorted((RUNS / d / "archive_half").glob("results*.csv"))
        if not first_files:
            continue
        first = pd.concat([pd.read_csv(f) for f in first_files], ignore_index=True)
        scores = _merged_npz(sorted(_first_grid_rerun(d).glob("attack_scores_full_*.npz")), skip=("S",))
        for r in first.itertuples():
            key = f"{r.model}|{r.mode}|{r.attack}|{float(r.eps)}"
            if r.model in ("chronos2", "timesfm") or key not in scores:
                continue
            tau, n = float(scores[f"{r.model}|tau"]), int(r.n)
            detected = scores[f"{r.model}|clean"][:n] > tau
            rerun_successes = int(((scores[key][:n] < tau) & detected).sum())
            first_successes = int(round(r.asr * r.n_detected))
            if int(detected.sum()) != int(r.n_detected):
                continue                                  # not the same model or sample: no comparison
            cells.append(dict(dataset=d, model=r.model, mode=r.mode, attack=r.attack, eps=r.eps, detected=int(detected.sum()),
                              first_run=first_successes, rerun=rerun_successes))
    if not cells:
        return
    C = pd.DataFrame(cells)
    C["diff"] = C["rerun"] - C["first_run"]
    C["points"] = 100 * C["diff"].abs() / C["detected"].clip(lower=1)
    C.to_csv(REP / "table14_rerun_by_attack_cells.csv", index=False)
    rows = [dict(dataset="all", attack=a, **_rerun_summary(g)) for a, g in C.groupby("attack", sort=False)]
    rows += [dict(dataset=d, attack=a, **_rerun_summary(g)) for (d, a), g in C.groupby(["dataset", "attack"], sort=False)]
    pd.DataFrame(rows).round(2).to_csv(REP / "table14_rerun_by_attack.csv", index=False)


def hsj_starts_table():
    """Table 15: reports/table15_hsj_starts.csv, every runs/<d>/attacks/hsj_starts.csv in one file (Table S-XLII). HopSkipJump from five seeded starts (attacks/hsj_starts.py), every model
    except the foundation models, all budgets: success rate of single starts (min, mean, max), of their union, the
    grid's HopSkipJump, Square and worst case on the same detected frauds, and the evasions the starts add to the
    grid's worst case."""
    frames = [pd.read_csv(p) for p in sorted(RUNS.glob("*/attacks/hsj_starts.csv")) if p.stat().st_size > 1]
    if frames:
        pd.concat(frames, ignore_index=True).round(4).to_csv(REP / "table15_hsj_starts.csv", index=False)


# The steps of main(), in the order they run. pareto_figure reads table1_clean, table6_worstcase_ci and
# table7_defenses_by_dataset, so it runs after the steps that write them.

def grid_summaries(A):
    """Summaries of the attack grid A: ASR tables, attack catalog, transfer heatmaps and ASR-budget curves."""
    table_robust(A)
    table_attacks()
    for mode in ("unconstrained", "constrained"):
        fig_transfer(mode)
        fig_curves(A, mode)


def analysis_tables(A):
    """RQ3 thickness, RQ4 explanations and the statistical tests."""
    thickness_analysis(A)
    explain_tables()
    stats_tables(A)


def defense_summaries(A):
    """RQ5 defenses, then the Pareto frontier of every model and defense."""
    defense_tables(A)
    pareto_figure()


def attack_check_tables(A):
    """Checks of the attacks and the sample: certification, ranking metrics, history injection, the two halves of
    the sample, clients with and without an id, first grid against the rerun, five-start HopSkipJump."""
    certify_table()
    robust_auc_table(A)
    history_table()
    halves_table()
    groups_table()
    rerun_table()
    rerun_attack_table()
    hsj_starts_table()


def main():
    """Write every summary of reports/ (see the module docstring)."""
    REP.mkdir(exist_ok=True)
    table_clean()
    A = load_attacks()
    grid_summaries(A)
    analysis_tables(A)
    defense_summaries(A)
    attack_check_tables(A)


if __name__ == "__main__":
    main()
