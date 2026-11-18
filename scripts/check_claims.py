"""Numbers behind the qualitative statements of the results section (RQ1, RQ2), printed from the run files.
Rerun after every merge and compare the statements with the output.  usage: PYTHONPATH=src python3 scripts/check_claims.py"""
import numpy as np
import pandas as pd

from fraudadv.report.make_report import DS, FAM, load_attacks

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
WORST = "worst case (all attacks)"
A = load_attacks()
A["eps"] = A["eps"].round(4)
A = A.drop_duplicates(["dataset", "model", "mode", "attack", "eps"], keep="last")
print("attacks:", sorted(A["attack"].unique()))
W = A[A["attack"] == WORST]

print("\n== A. worst-case ASR (%), constrained, by budget")
P = W[W["mode"] == "constrained"].pivot_table(index=["dataset", "model"], columns="eps", values="asr") * 100
print(P.reindex([(d, f) for d in DS for f in FAM if (d, f) in P.index]).round(1).to_string())

print("\n== B. worst case at eps 0.25: unconstrained, constrained, ratio C/U, n detected")
Q = W[W["eps"] == 0.25].pivot_table(index=["dataset", "model"], columns="mode", values="asr") * 100
Q["C/U"] = Q["constrained"] / Q["unconstrained"]
Q["n"] = W[(W["eps"] == 0.25) & (W["mode"] == "constrained")].set_index(["dataset", "model"])["n_detected"]
Q = Q.reindex([(d, f) for d in DS for f in FAM if (d, f) in Q.index])
print(Q.round(2).to_string())
Q2 = Q.reset_index()
Q2["class"] = Q2["model"].map(lambda f: "tree" if f in ("xgboost", "catboost") else "neural" if f in ("mlp", "ft_transformer", "ivis")
                              else "graph" if f in ("hgnn", "hgt") else "foundation")
print("\nmedian C/U by class, outside PaySim and ULB:")
print(Q2[~Q2["dataset"].isin(["paysim", "ulb"])].groupby("class")["C/U"].agg(["median", "min", "max", "count"]).round(2).to_string())
print("\nmedian C/U by dataset:")
print(Q2.groupby("dataset")["C/U"].agg(["median", "min", "max"]).round(2).to_string())

print("\n== C. by attack at eps 0.25 (ASR %, same model and setting)")
X = A[A["eps"] == 0.25].copy()
X["kind"] = X["attack"].map(lambda a: "white" if a.startswith("white-box") else "square" if "Square" in a else "hsj" if "HopSkipJump" in a
                            else "worst" if a == WORST else "other")
print("attack kinds:", X.groupby("kind")["attack"].unique().to_dict())
# the trees have two white-box rows (AATS alone, AATS + exact MILP); the white-box attack is the stronger of the two
R = X[X["kind"] != "other"].pivot_table(index=["dataset", "mode", "model"], columns="kind", values="asr", aggfunc="max") * 100
R["square-white"] = R["square"] - R["white"]
R["hsj/square"] = R["hsj"] / R["square"]
R = R.reindex([(d, m, f) for d in DS for m in ("unconstrained", "constrained") for f in FAM if (d, m, f) in R.index])
print(R.round(1).to_string())
T = R.reset_index()
tr = T[T["model"].isin(["xgboost", "catboost"])]
print("\ntrees: cells where white-box >= Square:", int((tr["white"] >= tr["square"] - 1e-9).sum()), "of", len(tr),
      "| >= HopSkipJump:", int((tr["white"] >= tr["hsj"] - 1e-9).sum()), "of", len(tr))
print(tr[(tr["white"] < tr["square"] - 1e-9) | (tr["white"] < tr["hsj"] - 1e-9)].round(1).to_string())
nn = T[~T["model"].isin(["xgboost", "catboost"])]
print("\ndifferentiable models: cells where Square > PGD:", int((nn["square-white"] > 1e-9).sum()), "of", len(nn),
      "| median difference", round(float(nn["square-white"].median()), 1), "| largest:")
print(nn.sort_values("square-white", ascending=False).head(8).round(1).to_string())
print("\nHopSkipJump / Square, cells with Square >= 5%: quartiles",
      np.round(T[T["square"] >= 5]["hsj/square"].quantile([0.25, 0.5, 0.75]).values, 2))
gain = T["worst"] - T[["white", "square", "hsj"]].max(axis=1)
print("\nworst case minus the best single attack (points): quartiles and max", np.round(gain.quantile([0.25, 0.5, 0.75, 1.0]).values, 1),
      "| cells with a gain:", int((gain > 0.05).sum()), "of", len(T))
print(T[gain > 0.05].assign(gain=gain[gain > 0.05]).round(1).to_string())
d = T[~T["model"].isin(["xgboost", "catboost"])]
print("\ndifferentiable models: Square >= PGD in", int((d["square"] >= d["white"] - 1e-9).sum()), "of", len(d),
      "| PGD ahead by more than 2 points in", int((d["white"] - d["square"] > 2).sum()))
for sel, name in [(~d["model"].isin(["chronos2", "timesfm"]), "without foundation models"), (d["model"].isin(["chronos2", "timesfm"]), "foundation models")]:
    g = d[sel]
    print("  ", name, ": Square >= PGD in", int((g["square"] >= g["white"] - 1e-9).sum()), "of", len(g), "| Square > PGD in", int((g["square"] > g["white"] + 1e-9).sum()),
          "| PGD / worst case min", round(float((g["white"] / g["worst"]).min()), 2))
print("median Square - PGD by dataset (differentiable):", d.groupby("dataset")["square-white"].median().round(1).to_dict())
tr2 = T[T["model"].isin(["xgboost", "catboost"])]
for m in ("constrained", "unconstrained"):
    g = tr2[tr2["mode"] == m]
    print("trees", m, ": white-box >= Square in", int((g["white"] >= g["square"] - 1e-9).sum()), "of", len(g), "| largest Square lead", round(float((g["square"] - g["white"]).max()), 1))
hs = T[T["square"] >= 5].assign(r=lambda x: x["hsj"] / x["square"])
for name, sel in [("trees", hs["model"].isin(["xgboost", "catboost"])), ("neural tabular", hs["model"].isin(["mlp", "ft_transformer", "ivis"])), ("graph", hs["model"].isin(["hgnn", "hgt"]))]:
    print("HopSkipJump / Square", name, "quartiles", np.round(hs[sel]["r"].quantile([0.25, 0.5, 0.75]).values, 2), "n", int(sel.sum()))
print(hs[hs["model"].isin(["hgnn", "hgt"])][["dataset", "mode", "model", "hsj", "square", "r"]].round(2).to_string())
