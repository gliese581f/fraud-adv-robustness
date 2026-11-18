"""The study's defenses attacked by the consistent attacker (Table S-XX, the consistent-attacker appendix).

Each defended model is rebuilt with the study's own training call (fraudadv.defenses.run_defenses.train_defended:
same recipe, hyperparameters, seed 0 and threat model as the constrained defense results), because the defended
models were not stored. Only the attacker changes: fraudadv.defenses.run_defenses.evaluate runs with the consistent
threat model (Constraints(..., "consistent"): every query is repaired into a record a real transaction could
produce). The squeezing detector keeps the projection it was calibrated with.

At the end the clean metrics of every rebuilt model are printed next to the stored rows of the constrained run
(runs/<d>/defenses/results*.csv), as a check that the rebuild is the same model.

  python -m fraudadv.defenses.run_defenses_consistent <dataset> [--families ...] [--defenses a,b] [--eps 0.25] [--tag _x]
                                                      [--both]

--both (run_both) is how the defenses added later ran (graph and foundation models, ivis's adversarial augmentation):
each defended model is trained once and attacked by both the study's attacker and the consistent one.

Output: runs_consistent/<dataset>/defenses/results_consistent<tag>.csv, resumable per (family, defense).
Budget: eps 0.25 only by default, the budget of Table VII.
"""
from __future__ import annotations

import argparse

import pandas as pd

import fraudadv.defenses.run_defenses as D
from fraudadv.attacks.constraints import Constraints
from fraudadv.data.load import load
from fraudadv.paths import RUNS, RUNS_CONSISTENT

FAMILIES = "xgboost,catboost,mlp,ft_transformer,ivis"


def rebuild_check(stored_dir, rows):
    """Print |clean metric of the rebuilt model - clean metric stored by the constrained run| per model and defense."""
    stored = [pd.read_csv(p) for p in sorted(stored_dir.glob("results*.csv"))
              if "consistent" not in p.name and p.stat().st_size > 1]
    if not stored or not rows:
        return
    S = pd.concat(stored).drop_duplicates(["model", "defense"])
    R = pd.DataFrame(rows).drop_duplicates(["model", "defense"])
    J = R.merge(S, on=["model", "defense"], suffixes=("", "_stored"))
    for c in ("clean_auc", "clean_ks", "clean_recall@1%fpr"):
        if c in J and c + "_stored" in J:
            J[c + "_diff"] = (J[c] - J[c + "_stored"]).abs()
    diffs = [c for c in J if c.endswith("_diff")]
    print("rebuild check (|clean metric - stored|):\n", J[["model", "defense"] + diffs].round(5).to_string(), flush=True)


def run(dataset, families, tag="", only=None, eps=(0.25,)):
    """Rebuild each defended model of `families` (or only the defenses named in `only`) and attack it with the
    consistent attacker at the budgets `eps`. Appends to the results file after each (family, defense)."""
    a, meta = load(dataset)
    cons_train = Constraints(meta, a["X_train"], D.DEFENSE_MODE.get(dataset, "constrained"))   # as trained
    cons_attack = Constraints(meta, a["X_train"], "consistent")
    out = RUNS_CONSISTENT / dataset / "defenses"
    out.mkdir(parents=True, exist_ok=True)
    res = out / f"results_consistent{tag}.csv"
    rows = pd.read_csv(res).to_dict("records") if res.exists() else []
    done = {(r["model"], r["defense"]) for r in rows}
    for fam in families:
        for defense in D.DEFENSES.get(fam, []):
            if (fam, defense) in done or (only and defense not in only):
                continue
            try:
                m = D.train_defended(dataset, fam, defense, a, meta, cons_train)
                rows += D.evaluate(dataset, fam, defense, m, a, cons_attack, mode="consistent", eps=list(eps))
            except Exception as e:                                  # one failed defense must not stop the rest
                print("FAILED", dataset, fam, defense, repr(e), flush=True)
            pd.DataFrame(rows).to_csv(res, index=False)
    rebuild_check(RUNS / dataset / "defenses", rows)


K_DATASETS = ("ieee_cis", "sparkov", "paysim")   # where the consistent attacker differs from the constrained one


def _rows(path):
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def run_both(dataset, families, tag="", only=None, eps=(0.25,)):
    """The defenses added later (graph and foundation models, ivis's adversarial augmentation): train each defended
    model once and attack it with the study's threat model (constrained; unconstrained on ULB), appending to
    runs/<d>/defenses/results<tag>.csv as run_defenses does, and, on the datasets of K_DATASETS, with the consistent
    attacker too, appending to runs_consistent/<d>/defenses/results_consistent<tag>.csv. Resumable per (family,
    defense) in each file; a file is written after each evaluation."""
    a, meta = load(dataset)
    mode = D.DEFENSE_MODE.get(dataset, "constrained")
    cons_train = Constraints(meta, a["X_train"], mode)
    consistent = dataset in K_DATASETS
    cons_k = Constraints(meta, a["X_train"], "consistent") if consistent else None
    res_c = RUNS / dataset / "defenses" / f"results{tag}.csv"
    res_k = RUNS_CONSISTENT / dataset / "defenses" / f"results_consistent{tag}.csv"
    res_c.parent.mkdir(parents=True, exist_ok=True)
    if consistent:
        res_k.parent.mkdir(parents=True, exist_ok=True)
    rows_c, rows_k = _rows(res_c), _rows(res_k)
    for fam in families:
        for defense in D.DEFENSES.get(fam, []):
            if only and defense not in only:
                continue
            need_c = (fam, defense) not in {(r["model"], r["defense"]) for r in rows_c}
            need_k = consistent and (fam, defense) not in {(r["model"], r["defense"]) for r in rows_k}
            if not (need_c or need_k):
                continue
            try:
                m = D.train_defended(dataset, fam, defense, a, meta, cons_train)
                if need_c:
                    rows_c += D.evaluate(dataset, fam, defense, m, a, cons_train, mode=mode, eps=list(eps))
                    pd.DataFrame(rows_c).to_csv(res_c, index=False)
                if need_k:
                    rows_k += D.evaluate(dataset, fam, defense, m, a, cons_k, mode="consistent", eps=list(eps))
                    pd.DataFrame(rows_k).to_csv(res_k, index=False)
            except Exception as e:                                  # one failed defense must not stop the rest
                print("FAILED", dataset, fam, defense, repr(e), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--families", default=FAMILIES)
    ap.add_argument("--defenses", default="", help="subset of each family's defenses (default: all)")
    ap.add_argument("--eps", default="0.25", help="comma-separated L-inf budgets (default: 0.25)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--both", action="store_true",
                    help="train once, attack with the study's threat model and the consistent one (run_both)")
    args = ap.parse_args()
    if {"chronos2", "timesfm"} & set(args.families.split(",")):
        import chronos  # noqa: F401  before TensorFlow (ivis): the reverse order aborts on an LLVM option clash
    (run_both if args.both else run)(args.dataset, args.families.split(","), args.tag, set(args.defenses.split(",")) - {""},
                                     [float(e) for e in args.eps.split(",")])
