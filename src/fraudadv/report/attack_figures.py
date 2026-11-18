"""Figures and tables of the attack views, drawn from runs/<dataset>/causal/views.pkl (written by
`python -m fraudadv.report.attack_views compute`) and the ablation outputs out_{xgb,mlp}_seed0_<mode>.npz beside it.
Run through `python -m fraudadv.report.attack_views figures`. Figures go to reports/ as .pdf and .png, their
numbers to reports/attack_views/*.csv.

  fig_attack_views     Figure 2, main text: (a, b) the same Sparkov fraud as XGBoost and the MLP
                       see it: logit minus the threshold on a grid of two attacker-controllable features, with the
                       smallest evasion in that slice; (c) distance to the decision boundary of detected frauds, missed
                       frauds and passed legit transactions; (d) k-NN fraud share of adversarial frauds against legit
                       and real frauds
  fig_attack_slices    Figure S15: the slices for all four datasets (supplement)
  fig_attack_map       Figure S17: UMAP of legit transactions, frauds and white-box adversarial examples (supplement)
  fig_attack_profile   Figure S18: mean absolute change per feature, per attack and model (supplement)
  fig_attack_distance  Figure S16: distance-to-boundary distributions per dataset (supplement)
  attack_views/attack_examples*.csv  Table S-XXXVII: before / after transactions in raw units
  attack_views/attack_view_numbers.csv   the per-dataset numbers quoted with Figure 2
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from fraudadv.report.attack_views import ATTACKS, BASES, MODES, VDS
from fraudadv.report.make_report import DS_NAME, REP, RUNS, _style

INK, INK2, MUTED, GRID = "#2b2a28", "#52514e", "#8a8984", "#e6e5e0"
BLUE, RED, ORANGE, GREEN = "#2a78d6", "#e34948", "#e08a1e", "#2e9e6a"
MODEL_COL = {"xgb": BLUE, "mlp": RED}
MODEL_NAME = {"xgb": "XGBoost", "mlp": "MLP"}
OUT = REP / "attack_views"

PRETTY = {"log_amt": "amount (\\$)", "hour": "hour of day", "dow": "day of week", "log_ratio": "amount / past mean (log)",
          "merch_lat": "merchant latitude", "merch_long": "merchant longitude", "dist": "distance to merchant (deg)",
          "category": "merchant category", "mcc": "merchant category (MCC)", "use_chip": "chip / swipe / online",
          "online": "online", "merch_seen": "merchant seen before", "addr1": "billing region code", "addr2": "billing country code",
          "dist1": "distance", "P_emaildomain": "purchaser e-mail domain", "R_emaildomain": "recipient e-mail domain",
          "ProductCD": "product code", "DeviceType": "device type", "browser": "browser",
          "current_address_months_count": "months at current address", "prev_address_months_count": "months at previous address",
          "intended_balcon_amount": "intended balance amount", "name_email_similarity": "name/e-mail similarity",
          "session_length_in_minutes": "session length (min)", "days_since_request": "days since request",
          "velocity_6h": "applications in 6 h", "velocity_24h": "applications in 24 h", "velocity_4w": "applications in 4 weeks",
          "zip_count_4w": "applications from the ZIP code in 4 weeks", "proposed_credit_limit": "proposed credit limit",
          "customer_age": "age", "income": "income (decile)", "housing_status": "housing status",
          "payment_type": "payment type", "employment_status": "employment status", "device_os": "device OS",
          "email_is_free": "free e-mail provider", "phone_home_valid": "home phone valid", "phone_mobile_valid": "mobile phone valid",
          "has_other_cards": "has other cards", "keep_alive_session": "keep-alive session", "source": "source",
          "credit_risk_score": "credit risk score", "bank_months_count": "months with bank",
          "foreign_request": "foreign request", "device_distinct_emails_8w": "distinct e-mails on device (8 weeks)",
          "date_of_birth_distinct_emails_4w": "e-mails per birth date (4 weeks)", "bank_branch_count_8w": "applications at branch (8 weeks)"}


def _views(d):
    """The stored attack views of dataset d (runs/<d>/causal/views.pkl), or None if not computed."""
    p = RUNS / d / "causal" / "views.pkl"
    return pickle.loads(p.read_bytes()) if p.exists() else None


def _available_views():
    """dataset -> its attack views, for the datasets of VDS (in that order) whose views are computed."""
    views = {d: _views(d) for d in VDS}
    return {d: r for d, r in views.items() if r is not None}


def raw_value(R, j, x):
    """A standardized value in raw units (amounts in dollars, categories as their training fraud rate)."""
    f = R["features"][j]
    v = x * R["sd"][j] + R["mu"][j]
    if f in R["categorical"]:
        return v                                           # smoothed target encoding = the category's fraud rate
    if f in ("log_amt", "log_amt_mean_prev", "log_city_pop", "cnt_prev", "cnt_24h", "log_dt_prev"):
        return np.expm1(v)
    return v


def _label(R, j):
    """Readable name of feature j."""
    f = R["features"][j]
    return PRETTY.get(f, f.replace("_", " "))


# ------------------------------------------------------------------ slices
def _draw_slice(ax, R, v, cb_ax=None, title=None, legend=False):
    """Draw model v's decision-function slice of R on ax: logit minus threshold, the threshold line, the clean fraud
    (star) and the smallest evasion in the slice (circle). Returns the colour mesh, for a colour bar."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    s = R["slice"]
    j, k, x0 = s["j"], s["k"], s["x0"]
    q = s[v]
    gx, gy = raw_value(R, j, q["gj"]), raw_value(R, k, q["gk"])
    cmap = LinearSegmentedColormap.from_list("pf", ["#1c5cab", "#9ec5f4", "#f4f3ef", "#f2a3a2", "#b8282a"])
    L = np.clip(q["L"].T, -8, 8)
    im = ax.pcolormesh(gx, gy, L, cmap=cmap, norm=TwoSlopeNorm(0, -8, 8), shading="nearest", rasterized=True)
    ax.contour(gx, gy, q["L"].T, levels=[0], colors=[INK], linewidths=0.9)
    px, py = raw_value(R, j, x0[j]), raw_value(R, k, x0[k])
    if q["best"][0] is not None:
        bx, by = raw_value(R, j, q["best"][0][0]), raw_value(R, k, q["best"][0][1])
        ax.annotate("", xy=(bx, by), xytext=(px, py), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, shrinkA=3, shrinkB=2))
        ax.scatter([bx], [by], s=22, color="white", edgecolors=INK, linewidths=1.0, zorder=5, label="smallest evasion")
    ax.scatter([px], [py], s=55, marker="*", color="white", edgecolors=INK, linewidths=0.9, zorder=6, label="fraud (clean)")
    if R["features"][j] in ("log_amt",):                  # amounts on a log axis in dollars
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}" if x >= 1 else f"{x:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        lo, hi = float(np.min(gx)), float(np.max(gx))
        if hi / max(lo, 1e-9) < 300:  # under 2.5 decades: a few round dollar values, no minor labels
            ticks = np.array([lo, hi])
            for subs in ((1, 2, 5), (1, 3), (1,)):
                cand = np.sort((np.array(subs) * 10.0 ** np.arange(-1, 7)[:, None]).ravel())
                t = cand[(cand >= lo) & (cand <= hi)]
                if 2 <= len(t) <= 4:
                    ticks = t
                    break
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(_label(R, j), fontsize=7)
    ax.set_ylabel(_label(R, k), fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_title(title or MODEL_NAME[v], fontsize=8, loc="left", color=INK)
    if legend:
        ax.legend(fontsize=6, frameon=True, framealpha=0.85, loc="lower left", handletextpad=0.3, borderpad=0.3)
    return im


# ------------------------------------------------------------------ groups for distance and k-NN panels
def distance_groups(R, md="constrained"):
    """Minimal L-inf distance to the threshold per group and model (HopSkipJump in both directions)."""
    g = {}
    for v in BASES:
        z = np.load(RUNS / R["dataset"] / "causal" / f"out_{v}_seed0_{md}.npz")
        det = z["det"][z["hsj_rows"]]
        g[(v, "detected fraud: to evade")] = z["hsj_cost"][det]
        g[(v, "missed fraud: to be caught")] = R[f"{v}_{md}_rev_missed"][1]
        g[(v, "passed legit: to be flagged")] = R[f"{v}_{md}_rev_legit"][1]
    return g


def knn_groups(R, md="constrained", att="wb"):
    """k-NN fraud share per group; 'before' is the same frauds as 'after', before the attack (paired)."""
    det_x, det_m = R["xgb_l0"] > R["xgb_tau"], R["mlp_l0"] > R["mlp_tau"]
    g = {"legit": R["knn_legit"], "fraud missed by both": R["knn_fraud"][~det_x & ~det_m],
         "fraud detected by both": R["knn_fraud"][det_x & det_m]}
    for v in BASES:
        g[f"evaded {v}: before"] = R["knn_fraud"][R[f"{v}_{md}_{att}_rows"]]
        g[f"evaded {v}: after"] = R[f"{v}_{md}_{att}_knn"]
    return g


def _rows_panel(ax, groups, xlog, xlab, xlim=None):
    """groups: list of (label, {colour: (pooled values, per-dataset medians)}); one row per label, colours offset."""
    for i, (lab, series) in enumerate(groups):
        offs = np.linspace(-0.16, 0.16, len(series)) if len(series) > 1 else [0.0]
        for (c, (vals, meds)), o in zip(series.items(), offs):
            v = np.asarray(vals, float)
            v = v[~np.isnan(v)]
            if not len(v):
                continue
            lo, md_, hi = np.percentile(v, [25, 50, 75])
            ax.plot([lo, hi], [i + o] * 2, color=c, linewidth=2.4, solid_capstyle="round", alpha=0.35, zorder=2)
            ax.scatter(meds, [i + o] * len(meds), s=9, marker="|", color=c, linewidths=0.9, zorder=3)
            ax.scatter([md_], [i + o], s=22, color=c, edgecolors="white", linewidths=0.7, zorder=4)
    ax.set_yticks(range(len(groups)), [g[0] for g in groups], fontsize=6.5)
    ax.set_ylim(len(groups) - 0.5, -0.5)
    if xlog:
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.25, 0.5, 1, 2, 4]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(xlab, fontsize=6.8, color=INK2)


def fig_main(slice_ds="sparkov", md="constrained"):
    """Figure 2, reports/fig_attack_views.{png,pdf} (Figure 2): slices of one slice_ds fraud for XGBoost and
    the MLP, distance to the threshold, and k-NN fraud share (setting md)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _style(plt)
    Rs = _available_views()
    if slice_ds not in Rs:
        return
    W, H = 7.16, 2.55
    fig = plt.figure(figsize=(W, H))
    ax_ = lambda x, w: fig.add_axes([x / W, 0.86 / H, w / W, 1.4 / H])
    R = Rs[slice_ds]
    a0, a1 = ax_(0.42, 1.2), ax_(1.72, 1.2)
    im = _draw_slice(a0, R, "xgb", title="(a) XGBoost")
    _draw_slice(a1, R, "mlp", title="(b) MLP")
    a1.set_ylabel("")
    a1.tick_params(labelleft=False)
    cax = fig.add_axes([0.42 / W, 0.06 / H, 1.2 / W, 0.05 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5.5, length=0, pad=1)
    cb.outline.set_visible(False)
    cb.ax.set_title("logit $-$ threshold", fontsize=6.3, color=INK2, pad=2)
    h = [Line2D([], [], marker="*", linestyle="", markersize=8, markerfacecolor="white", markeredgecolor=INK),
         Line2D([], [], marker="o", linestyle="", markersize=4.5, markerfacecolor="white", markeredgecolor=INK),
         Line2D([], [], color=INK, linewidth=0.9)]
    fig.legend(h, ["fraud", "smallest evasion", "threshold"], loc="lower left", bbox_to_anchor=(1.75 / W, -0.01),
               fontsize=6.2, frameon=False, ncol=1, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.1)
    # (c) distance to the boundary (HopSkipJump in both directions), pooled; ticks: per-dataset medians
    a2 = ax_(3.85, 1.25)
    groups = []
    for gname, lab in (("detected fraud: to evade", "detected fraud\nto evade"), ("missed fraud: to be caught", "missed fraud\nto be caught"),
                       ("passed legit: to be flagged", "legit\nto be flagged")):
        ser = {}
        for v in BASES:
            per = [np.minimum(distance_groups(r, md)[(v, gname)], 4.0) for r in Rs.values()]
            ser[MODEL_COL[v]] = (np.concatenate(per), [float(np.median(p)) for p in per if len(p)])
        groups.append((lab, ser))
    _rows_panel(a2, groups, True, "smallest L$_\\infty$ change (sd)", xlim=(0.03, 4.5))
    a2.set_title("(c) distance to the threshold", fontsize=8, loc="left", color=INK)
    # (d) k-NN fraud share
    a3 = ax_(5.85, 1.25)
    groups = []
    for gname, lab, cols in (("legit", "legit", {MUTED: "legit"}), ("fraud missed by both", "missed fraud", {ORANGE: 0}),
                             ("fraud detected by both", "detected fraud", {INK: 0}),
                             ("before", "evaded fraud,\nbefore attack", {MODEL_COL[v]: v for v in BASES}),
                             ("after", "evaded fraud,\nafter attack", {MODEL_COL[v]: v for v in BASES})):
        ser = {}
        for c, v in cols.items():
            key = f"evaded {v}: {gname}" if gname in ("before", "after") else gname
            per = [100 * knn_groups(r, md)[key] for r in Rs.values()]
            ser[c] = (np.concatenate(per), [float(np.median(p)) for p in per if len(p)])
        groups.append((lab, ser))
    _rows_panel(a3, groups, False, "frauds among 50 nearest\ntraining transactions (%)")
    a3.set_title("(d) neighbourhood", fontsize=8, loc="left", color=INK)
    h = [Line2D([], [], marker="o", linestyle="", color=MODEL_COL[v], markersize=4.5) for v in BASES]
    fig.legend(h, [MODEL_NAME[v] for v in BASES], loc="lower left", bbox_to_anchor=(3.85 / W, -0.01), fontsize=6.5, frameon=False,
               ncol=2, handletextpad=0.2, columnspacing=0.8)
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"fig_attack_views.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_slices():
    """reports/fig_attack_slices.{png,pdf} (Figure S15): the decision-function slices of every dataset and model."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _style(plt)
    Rs = list(_available_views().items())
    fig, axes = plt.subplots(2, len(Rs), figsize=(7.16, 4.1), squeeze=False)
    for c, (d, R) in enumerate(Rs):
        for r, v in enumerate(BASES):
            im = _draw_slice(axes[r, c], R, v, title=f"{DS_NAME[d]} · {MODEL_NAME[v]}")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.22, wspace=0.45, hspace=0.62)
    h = [Line2D([], [], marker="*", linestyle="", markersize=8, markerfacecolor="white", markeredgecolor=INK),
         Line2D([], [], marker="o", linestyle="", markersize=4.5, markerfacecolor="white", markeredgecolor=INK)]
    fig.legend(h, ["fraud (clean)", "smallest evasion in the slice"], loc="lower left", bbox_to_anchor=(0.02, 0.0),
               fontsize=6.8, frameon=False, ncol=1)
    cax = fig.add_axes([0.3, 0.035, 0.4, 0.015])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=6, length=0)
    cb.outline.set_visible(False)
    cb.set_label("logit $-$ threshold (blue: passes, red: flagged; black line: the threshold)", fontsize=7, color=INK2)
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"fig_attack_slices.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_map(md="constrained"):
    """reports/fig_attack_map.{png,pdf} (Figure S17): UMAP of legit transactions, frauds and both models'
    white-box adversarial examples, a line from each fraud to its adversarial example."""
    import matplotlib.pyplot as plt

    _style(plt)
    Rs = list(_available_views().items())
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.2), squeeze=False)
    for ax, (d, R) in zip(axes.flat, Rs):
        U = R["umap"]
        det_x, det_m = R["xgb_l0"] > R["xgb_tau"], R["mlp_l0"] > R["mlp_tau"]
        ax.scatter(*U["legit"].T, s=2, color="#cfcec8", linewidths=0, rasterized=True, label="legit")
        F = U["fraud"]
        ax.scatter(*F[det_x | det_m].T, s=7, color=INK, linewidths=0, label="fraud, detected", zorder=3)
        ax.scatter(*F[~det_x & ~det_m].T, s=9, facecolors="none", edgecolors=ORANGE, linewidths=0.8, label="fraud, missed by both", zorder=3)
        for v in BASES:
            A = U[f"adv_{v}_{md}"]
            rows = R[f"{v}_{md}_wb_rows"]
            for (x1, y1), i in zip(A, rows):
                ax.plot([F[i, 0], x1], [F[i, 1], y1], color=MODEL_COL[v], linewidth=0.35, alpha=0.55, zorder=2)
            ax.scatter(*A.T, s=6, color=MODEL_COL[v], linewidths=0, label=f"adversarial ({MODEL_NAME[v]})", zorder=4)
        ax.set_title(DS_NAME[d], fontsize=8.5, loc="left", color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7, markerscale=2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.06, wspace=0.05, hspace=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"fig_attack_map.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_profile(md="constrained", top=8):
    """reports/fig_attack_profile.{png,pdf} (Figure S18) and reports/attack_views/perturbation_profile.csv:
    mean absolute change per feature (the top features) for every model and attack."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    _style(plt)
    BLUES = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    Rs = list(_available_views().items())
    nc = 2 if len(Rs) > 2 else len(Rs)
    nr = (len(Rs) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(7.16, 2.55 * nr + 0.4), squeeze=False)
    rows_csv = []
    for ax, (d, R) in zip(axes.ravel(), Rs):
        X0 = R["X0"]
        cols, M, L0 = [], [], []
        for v in BASES:
            for att in ATTACKS:
                rows, Xa = R[f"{v}_{md}_{att}_rows"], R[f"{v}_{md}_{att}_X"]
                D = np.abs(Xa - X0[rows]) if len(rows) else np.zeros((1, X0.shape[1]))
                M.append(D.mean(0))
                L0.append(float((D > 1e-4).sum(1).mean()) if len(rows) else np.nan)
                cols.append(f"{MODEL_NAME[v]} {ATTACKS[att].replace('white-box', 'WB').replace('HopSkipJump', 'HSJ').replace('Square', 'Sq')}")
                for j, f in enumerate(R["features"]):
                    rows_csv.append(dict(dataset=d, model=v, mode=md, attack=att, feature=f, mean_abs_change=float(D.mean(0)[j]),
                                         n=len(rows)))
        M = np.array(M).T
        keep = np.argsort(-M.max(1))[:top]
        im = ax.imshow(M[keep], cmap=cmap, aspect="auto", vmin=0, vmax=0.25)
        ax.set_yticks(range(len(keep)), [PRETTY.get(R["features"][j], R["features"][j]).replace("\\$", "$") for j in keep], fontsize=6.5)
        ax.set_xticks(range(len(cols)), [f"{c} ({l:.1f})" for c, l in zip(cols, L0)], rotation=45, ha="right", fontsize=6.2)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(DS_NAME[d], fontsize=8, loc="left", color=INK)
    for ax in axes.ravel()[len(Rs):]:
        ax.set_visible(False)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.95, bottom=0.2, wspace=0.75, hspace=0.85)
    cax = fig.add_axes([0.35, 0.025, 0.3, 0.015])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max")
    cb.ax.tick_params(labelsize=6, length=0)
    cb.outline.set_visible(False)
    cb.set_label("mean |change| over the successful adversarial examples (sd units; eps = 0.25, constrained);"
                 " in brackets: mean number of features changed", fontsize=6.3, color=INK2)
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"fig_attack_profile.{ext}", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows_csv).to_csv(OUT / "perturbation_profile.csv", index=False)


def fig_distance():
    """reports/fig_attack_distance.{png,pdf} (Figure S16) and reports/attack_views/distance_to_boundary.csv:
    cumulative distributions of the distance to the threshold per dataset, setting, model and group."""
    import matplotlib.pyplot as plt

    _style(plt)
    Rs = list(_available_views().items())
    fig, axes = plt.subplots(2, len(Rs), figsize=(7.16, 3.6), squeeze=False, sharey=True)
    rows_csv = []
    styles = {"detected fraud: to evade": "-", "missed fraud: to be caught": "--", "passed legit: to be flagged": ":"}
    for c, (d, R) in enumerate(Rs):
        for r, md in enumerate(MODES):
            ax = axes[r, c]
            G = distance_groups(R, md)
            for (v, g), vals in G.items():
                x = np.sort(np.minimum(vals, 4.0))
                if not len(x):
                    continue
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=MODEL_COL[v], linestyle=styles[g], linewidth=1.0,
                        label=f"{MODEL_NAME[v]}, {g}")
                rows_csv.append(dict(dataset=d, mode=md, model=v, group=g, n=len(x), median=float(np.median(x)),
                                     q25=float(np.quantile(x, 0.25)), q75=float(np.quantile(x, 0.75))))
            ax.set_xscale("log")
            ax.set_xlim(0.01, 4.2)
            ax.grid(color=GRID, linewidth=0.5)
            ax.tick_params(labelsize=6.5)
            ax.set_title(f"{DS_NAME[d]}, {md}", fontsize=7.5, loc="left", color=INK)
            if r == 1:
                ax.set_xlabel("smallest L$_\\infty$ change (sd units)", fontsize=7)
        axes[0, 0].set_ylabel("share of points", fontsize=7)
        axes[1, 0].set_ylabel("share of points", fontsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.5)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.25, wspace=0.12, hspace=0.45)
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"fig_attack_distance.{ext}", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows_csv).to_csv(OUT / "distance_to_boundary.csv", index=False)


def knn_summary():
    """reports/attack_views/knn_fraud_share.csv: median k-NN fraud share and k-NN flag rate per group; returns it."""
    rows = []
    for d in VDS:
        R = _views(d)
        if R is None:
            continue
        thr = R["knn_thr"]
        for md in MODES:
            G = knn_groups(R, md)
            for g, vals in G.items():
                rows.append(dict(dataset=d, mode=md, group=g, n=len(vals), median_share=float(np.median(vals)) if len(vals) else np.nan,
                                 knn_flag_rate=float(np.mean(vals > thr)) if len(vals) else np.nan))
            for v in BASES:
                for att in ATTACKS:
                    vals = R[f"{v}_{md}_{att}_knn"]
                    rows.append(dict(dataset=d, mode=md, group=f"adversarial {v} {att}", n=len(vals),
                                     median_share=float(np.median(vals)) if len(vals) else np.nan,
                                     knn_flag_rate=float(np.mean(vals > thr)) if len(vals) else np.nan))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "knn_fraud_share.csv", index=False)
    return D


def tab_examples(md="constrained", max_feats=6):
    """Table S-XXXVII, reports/attack_views/attack_examples.csv and attack_examples_scores.csv: one detected fraud per
    dataset (the slice fraud) before and after the white-box attack on XGBoost (AATS) and on the MLP (APGD), in raw
    units: the features the attacks change (at most six, largest change first; NaN: that model was not evaded within
    the budget), and each model's score as a percentile among legit test transactions before and after (the
    threshold is the 99th percentile)."""
    from fraudadv.attacks.run_attacks import load_model

    rows_csv, rows_scores = [], []
    for d in VDS:
        R = _views(d)
        if R is None:
            continue
        i0 = R["slice"]["i0"]
        x0 = R["X0"][i0]
        adv = {}
        for v in BASES:
            rr = list(R[f"{v}_{md}_wb_rows"])
            adv[v] = R[f"{v}_{md}_wb_X"][rr.index(i0)] if i0 in rr else None
        ch = np.zeros(len(x0))
        for v, xa in adv.items():
            if xa is not None:
                ch = np.maximum(ch, np.abs(xa - x0))
        for j in [j for j in np.argsort(-ch) if ch[j] > 1e-4][:max_feats]:
            rows_csv.append(dict(dataset=d, feature=R["features"][j], original=float(raw_value(R, j, x0[j])),
                                 **{f"{v}_adv": (float(raw_value(R, j, adv[v][j])) if adv[v] is not None else np.nan) for v in BASES}))
        for v, f in BASES.items():
            m = load_model(d, f, 0)
            Q = R[f"{v}_legit_logit_quantiles"]
            before = float(np.interp(m.logit(x0[None])[0], Q, np.linspace(0, 100, len(Q))))
            after = None if adv[v] is None else float(np.interp(m.logit(adv[v][None])[0], Q, np.linspace(0, 100, len(Q))))
            rows_scores.append(dict(dataset=d, model=f, score_percentile_before=before, score_percentile_after=after))
    pd.DataFrame(rows_csv).to_csv(OUT / "attack_examples.csv", index=False)
    pd.DataFrame(rows_scores).to_csv(OUT / "attack_examples_scores.csv", index=False)


def numbers(md="constrained"):
    """The attack-view numbers quoted with Figure 2, reports/attack_views/attack_view_numbers.csv: per dataset,
    constrained, the median distance to the decision boundary (capped at 4 sd, as in the figures) of detected frauds
    (to evade), missed frauds (to be caught) and passed legit transactions (to be flagged) per base model; the median
    k-NN fraud share of evaded frauds before and after the attack and of frauds both models miss, and the share of
    them a k-NN detector at 1% legit flag rate would flag; and the dollar change of the Sparkov slice fraud's
    smallest XGBoost evasion. The text quotes ranges over datasets of these per-dataset values (IEEE-CIS apart for
    the distances, since the constrained attacker controls little there)."""
    Rs = _available_views()
    if not Rs:
        return
    rows = []
    R = Rs.get("sparkov") or next(iter(Rs.values()))
    sl = R["slice"]
    j = sl["j"]
    best = sl["xgb"]["best"][0]
    rows.append(dict(dataset="sparkov" if "sparkov" in Rs else next(iter(Rs)), quantity="slice fraud: amount change of the smallest XGBoost evasion ($)",
                     model="xgboost", value=np.nan if best is None else float(raw_value(R, j, sl["x0"][j]) - raw_value(R, j, best[0]))))

    def med(a):                                       # median distance, capped at 4 sd like the figures
        return float(np.median(np.minimum(a, 4.0))) if len(a) else np.nan

    for d, r in Rs.items():
        g = distance_groups(r, md)
        for v in BASES:
            for group in ("detected fraud: to evade", "missed fraud: to be caught", "passed legit: to be flagged"):
                rows.append(dict(dataset=d, quantity=f"median distance to the boundary, {group}", model=BASES[v], value=med(g[(v, group)])))
        k = knn_groups(r, md)
        for v in BASES:
            for when in ("before", "after"):
                x = k[f"evaded {v}: {when}"]
                rows.append(dict(dataset=d, quantity=f"median k-NN fraud share, evaded frauds {when} the attack", model=BASES[v],
                                 value=float(np.median(x)) if len(x) else np.nan))
            x = k[f"evaded {v}: after"]
            rows.append(dict(dataset=d, quantity="k-NN detector flag rate, evaded frauds after the attack", model=BASES[v],
                             value=float(np.mean(x > r["knn_thr"])) if len(x) else np.nan))
        x = k["fraud missed by both"]
        rows.append(dict(dataset=d, quantity="median k-NN fraud share, frauds both models miss", model="",
                         value=float(np.median(x)) if len(x) else np.nan))
        rows.append(dict(dataset=d, quantity="k-NN detector flag rate, frauds both models miss (at least 10)", model="",
                         value=float(np.mean(x > r["knn_thr"])) if len(x) >= 10 else np.nan))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "attack_view_numbers.csv", index=False)
    print(D.round(3).to_string(index=False))


def main():
    """Write every attack-view figure and CSV (python -m fraudadv.report.attack_views figures)."""
    OUT.mkdir(parents=True, exist_ok=True)
    numbers()
    fig_main()
    fig_slices()
    fig_map()
    fig_profile()
    fig_distance()
    print(knn_summary().round(3).to_string())
    tab_examples()


if __name__ == "__main__":
    main()
