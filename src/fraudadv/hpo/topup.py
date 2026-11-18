"""Second hyperparameter search for tabular families whose first search (fraudadv.hpo.train_all) finished
too few trials.

The first search gave each family a fixed CPU time. On the large datasets one deep-model trial can use all
of it (FT-Transformer finished 1 trial on BAF and Sparkov). The second search applies one rule to every
dataset, so the model families are compared at a similar tuning effort:

  * a new Optuna study (multivariate TPE plus Hyperband pruner, the first search's space), whose first
    trial is the first search's best configuration
  * trials train on a fixed sample of the training split (at most HPO_ROWS rows, every fraud kept) and are
    scored by weighted AUPRC on the full validation split
  * it runs to MIN_TRIALS finished trials (pruned trials count, because Hyperband stops them on purpose) or
    TIME_CAP seconds
  * if it finished more trials than the first search, the best configuration replaces the first one and is
    retrained with three seeds on the usual training rows (fraudadv.hpo.train_all.run); otherwise the
    first-search models are kept
Families whose first search already reached MIN_TRIALS keep their models without a second search.

Inputs:  data/processed/<dataset>/arrays.npz; runs/<dataset>/<family>/ from train_all (trials.csv, study.db).
Outputs: runs/<dataset>/hpo_v2/<family>/ (the second study while it runs); for a replaced family, the first
         search moved to runs/<dataset>/archive_hpo1/<family>/ and new files in runs/<dataset>/<family>/
         (trials.csv, study_v2.db, best_params.json with "protocol": "v2", models and metrics). Every family
         it has decided on carries hpo_v2.json (kept_first_pass, trial counts).

Results: the final tabular models of every result; Table S-II and Table S-IV.

usage: python -m fraudadv.hpo.topup <dataset> [--families ...] [--catboost-gpu]
Prints "RETRAINED <dataset> <families>" when any family got a new model; scripts/pipeline.sh then redoes that
dataset's attacks, explanations, certification and defenses.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np
import optuna
import pandas as pd

from fraudadv.data.load import load
from fraudadv.eval.metrics import auprc
from fraudadv.hpo import train_all
from fraudadv.hpo.train_all import RUNS, space, subsample
from fraudadv.models.zoo import FAMILIES

# per family: finished trials to reach, training rows per trial, search seconds
MIN_TRIALS = {"xgboost": 20, "catboost": 20, "mlp": 15, "ft_transformer": 12, "ivis": 10}
HPO_ROWS = {"xgboost": 200_000, "catboost": 200_000, "mlp": 200_000, "ft_transformer": 60_000, "ivis": 100_000}
TIME_CAP = {"xgboost": 1800, "catboost": 1800, "mlp": 1800, "ft_transformer": 2700, "ivis": 2400}
TAB = ["xgboost", "catboost", "mlp", "ft_transformer", "ivis"]


def finished_trials(d):
    """Number of finished trials (complete or pruned) of the first search, from <d>/trials.csv."""
    p = d / "trials.csv"
    if not p.exists():
        return 0
    t = pd.read_csv(p)
    return int(t["state"].isin(["COMPLETE", "PRUNED"]).sum()) if "state" in t else len(t)


def old_best_raw(d, dataset, fam):
    """First-pass best configuration as raw Optuna parameters (for enqueue_trial)."""
    db = d / "study.db"
    if not db.exists():
        return None
    try:
        return optuna.load_study(study_name=f"{dataset}-{fam}", storage=f"sqlite:///{db}").best_params
    except Exception:
        return None


def run(dataset, families=TAB, catboost_gpu=False):
    """Run the second search for each family of `families` that needs it (rules in the module docstring) and
    return the families that got new models. catboost_gpu is passed to train_all (CatBoost on the GPU)."""
    a, _ = load(dataset)
    Xtr, ytr = a["X_train"], a["y_train"]
    Xva, yva, wva = a["X_val"], a["y_val"], a["w_val"]
    retrained = []
    for fam in families:
        d = RUNS / dataset / fam
        if (d / "hpo_v2.json").exists() or not (d / "metrics.json").exists():
            continue
        n1 = finished_trials(d)
        if n1 >= MIN_TRIALS[fam]:
            (d / "hpo_v2.json").write_text(json.dumps({"kept_first_pass": True, "first_pass_trials": n1}))
            print(dataset, fam, "kept: first pass finished", n1, "trials", flush=True)
            continue
        warm = old_best_raw(d, dataset, fam)
        v2 = RUNS / dataset / "hpo_v2" / fam                # the search lives here until it is complete
        v2.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(direction="maximize", study_name=f"{dataset}-{fam}-v2", load_if_exists=True,
                                    storage=f"sqlite:///{v2 / 'study.db'}",
                                    sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=1, n_startup_trials=5),
                                    pruner=optuna.pruners.HyperbandPruner(min_resource=2, max_resource=25, reduction_factor=3))
        if warm and not study.trials:
            study.enqueue_trial(warm)
        Xh, yh = subsample(Xtr, ytr, HPO_ROWS[fam])
        cls = FAMILIES[fam]
        fit_options = {"gpu": True} if fam == "catboost" and catboost_gpu else {}

        def objective(trial):
            m = cls().fit(Xh, yh, Xva, yva, wva, space(fam, trial), trial=trial, seed=0, **fit_options)
            return auprc(yva, m.logit(Xva), wva)

        done = sum(t.state.is_finished() for t in study.trials)
        spent = sum((t.duration.total_seconds() if t.duration else 0.0) for t in study.trials)
        if done < MIN_TRIALS[fam] and spent < TIME_CAP[fam]:
            t0 = time.time()
            study.optimize(objective, n_trials=MIN_TRIALS[fam] - done, timeout=max(120.0, TIME_CAP[fam] - spent),
                           catch=(Exception,))
            spent += time.time() - t0
        study.trials_dataframe().to_csv(v2 / "trials.csv", index=False)
        n2 = sum(t.state.is_finished() for t in study.trials)
        if n2 <= n1:                                        # the second search was not more thorough: keep the first
            (d / "hpo_v2.json").write_text(json.dumps({"kept_first_pass": True, "first_pass_trials": n1, "v2_trials": n2}))
            print(dataset, fam, "kept: v2 search finished only", n2, "trials (first pass", n1, ")", flush=True)
            continue
        best = space(fam, optuna.trial.FixedTrial(study.best_params))
        # replace: archive the first search, install the second, retrain three seeds with its best configuration
        arch = RUNS / dataset / "archive_hpo1" / fam
        arch.parent.mkdir(parents=True, exist_ok=True)
        if arch.exists():
            shutil.rmtree(arch)
        shutil.move(str(d), str(arch))
        d.mkdir(parents=True)
        shutil.copy(v2 / "trials.csv", d / "trials.csv")
        shutil.copy(v2 / "study.db", d / "study_v2.db")
        (d / "best_params.json").write_text(json.dumps({"params": best, "val_auprc": study.best_value, "n_trials": n2,
                                                        "hpo_seconds": spent, "protocol": "v2", "hpo_rows": HPO_ROWS[fam],
                                                        "first_pass_trials": n1}, indent=1))
        print(dataset, fam, "v2 search:", n2, "trials, best val AUPRC", round(study.best_value, 4), "; retraining", flush=True)
        train_all.run(dataset, [fam], catboost_gpu=catboost_gpu)
        (d / "hpo_v2.json").write_text(json.dumps({"kept_first_pass": False, "first_pass_trials": n1, "v2_trials": n2}))
        retrained.append(fam)
    # report every family the second search replaced, including those replaced before an interruption
    for fam in families:
        f = RUNS / dataset / fam / "hpo_v2.json"
        if fam not in retrained and f.exists() and not json.loads(f.read_text()).get("kept_first_pass", True):
            retrained.append(fam)
    if retrained:
        print("RETRAINED", dataset, ",".join(retrained), flush=True)
    return retrained


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--families", default=",".join(TAB))
    ap.add_argument("--catboost-gpu", action="store_true",
                    help="train CatBoost on the GPU when one is selected (off: CPU, as for every paper model)")
    args = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run(args.dataset, args.families.split(","), catboost_gpu=args.catboost_gpu)
