"""Tune and train the five tabular model families: XGBoost, CatBoost, MLP, FT-Transformer and ivis.

For each family:
  1. Hyperparameter search with Optuna (multivariate TPE sampler plus a Hyperband pruner). Each trial trains
     one model with seed 0 and is scored by weighted AUPRC on the validation split. The search stops after
     MAX_TRIALS trials or BUDGET_S seconds, whichever comes first (both multiplied by --scale). Only the MLP
     and FT-Transformer report a score per epoch, so only their trials can be pruned early.
  2. The best configuration is retrained with each seed in SEEDS (0, 1, 2) on the same training rows.
  3. Each seed's decision threshold is the score with a 1% weighted false-positive rate on validation, and
     the test-split metrics of fraudadv.eval.metrics.all_metrics are computed at it.

Inputs:  data/processed/<dataset>/arrays.npz (X_train, y_train, X_val, y_val, w_val, X_test, y_test, w_test,
         amt_test), via fraudadv.data.load.
Outputs: runs/<dataset>/<family>/
           study.db, trials.csv              the Optuna search (an interrupted search resumes from study.db)
           best_params.json                  best configuration, its validation AUPRC, trials, search time
           seed<k>.pkl (ivis: folder seed<k>/) the trained model
           test_scores_seed<k>.npy           its logits on the test split
           metrics_seed<k>.json, metrics.json test metrics per seed
         A family that already has metrics.json is skipped, so the script can be rerun after an interruption.

Results: the trained models are attacked for every grid result. Directly: Table S-II (
from best_params.json), Table S-IV and Section VI-A (from metrics*.json). The search spaces in
space() are Table S-III. load_model() is how the attacks load these models.

usage: python -m fraudadv.hpo.train_all <dataset> [--families xgboost,catboost,...] [--scale 1.0]
                                        [--seeds 0,1,2] [--catboost-gpu]
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import optuna

from fraudadv.data.load import load
from fraudadv.eval.metrics import all_metrics, auprc, threshold_at_fpr
from fraudadv.models.zoo import FAMILIES

from fraudadv.paths import RUNS  # noqa: E402  (FRAUDADV_RUNS overrides)

SEEDS = [0, 1, 2]                                # seeds of the retrained models (--seeds replaces them)
BUDGET_S = {"xgboost": 420, "catboost": 420, "mlp": 420, "ft_transformer": 600, "ivis": 360}   # search seconds
MAX_TRIALS = {"xgboost": 40, "catboost": 25, "mlp": 25, "ft_transformer": 12, "ivis": 8}
SUBSAMPLE = {"ft_transformer": 60_000}          # max training rows (all frauds kept), for search and retrain


def space(fam, t: optuna.Trial):
    """The hyperparameters of family `fam` for Optuna trial `t` (the search spaces of Table S-III).

    Values that are not searched (number of rounds, epochs, batch size, patience) are fixed here too, so the
    returned dict is the complete configuration passed to the model's fit(). Called with
    optuna.trial.FixedTrial(best_params) it rebuilds the best configuration."""
    if fam == "xgboost":
        return dict(n_estimators=2000, learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    max_depth=t.suggest_int("max_depth", 3, 10), min_child_weight=t.suggest_float("min_child_weight", 1, 50, log=True),
                    subsample=t.suggest_float("subsample", 0.5, 1.0), colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30, log=True), reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
                    scale_pos_weight=t.suggest_float("scale_pos_weight", 1, 20, log=True))
    if fam == "catboost":
        return dict(iterations=1500, depth=t.suggest_int("depth", 4, 9), learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                    l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 30, log=True), random_strength=t.suggest_float("random_strength", 0.1, 10, log=True),
                    bagging_temperature=t.suggest_float("bagging_temperature", 0.0, 1.0))
    if fam == "mlp":
        return dict(n_layers=t.suggest_int("n_layers", 1, 4), width=t.suggest_categorical("width", [64, 128, 256, 512]),
                    dropout=t.suggest_float("dropout", 0.0, 0.4), lr=t.suggest_float("lr", 1e-4, 3e-3, log=True),
                    weight_decay=t.suggest_float("weight_decay", 1e-6, 1e-2, log=True), pos_weight=t.suggest_float("pos_weight", 1, 20, log=True),
                    batch_size=1024, epochs=25, patience=4)
    if fam == "ft_transformer":
        return dict(d_token=t.suggest_categorical("d_token", [32, 64]), n_blocks=t.suggest_int("n_blocks", 1, 3), n_heads=4,
                    dropout=t.suggest_float("dropout", 0.0, 0.3), lr=t.suggest_float("lr", 1e-4, 1e-3, log=True),
                    weight_decay=t.suggest_float("weight_decay", 1e-6, 1e-3, log=True), pos_weight=t.suggest_float("pos_weight", 1, 20, log=True),
                    batch_size=512, epochs=12, patience=3)
    if fam == "ivis":
        return dict(embedding_dims=t.suggest_categorical("embedding_dims", [8, 16, 32]), k=t.suggest_categorical("k", [15, 30, 60]),
                    supervision_weight=t.suggest_float("supervision_weight", 0.1, 0.9),
                    distance=t.suggest_categorical("distance", ["pn", "softmax_ratio_pn", "cosine_pn"]),
                    C=t.suggest_float("C", 0.01, 10, log=True), epochs=20, max_train=50_000)
    raise KeyError(fam)


def subsample(X, y, n, seed=0):
    """At most n rows: every fraud plus a random sample of legitimate rows, in their original order.
    n=None, or a training set no larger than n, returns X and y unchanged."""
    if n is None or len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    idx = np.sort(np.r_[pos, rng.choice(neg, size=n - len(pos), replace=False)])
    return X[idx], y[idx]


def run(dataset, families, scale=1.0, catboost_gpu=False):
    """Search, retrain and evaluate each family of `families` on `dataset` (the steps in the module docstring).

    scale multiplies each family's trial count (at least 2) and search time (at least 60 s).
    catboost_gpu trains CatBoost on the GPU when fraudadv.device selects one. It is off by default because
    CatBoost's GPU mode supports fewer options and grows different trees; every paper model was trained on CPU."""
    a, meta = load(dataset)
    Xtr, ytr = a["X_train"], a["y_train"]
    Xva, yva, wva = a["X_val"], a["y_val"], a["w_val"]
    for fam in families:
        out = RUNS / dataset / fam
        out.mkdir(parents=True, exist_ok=True)
        if (out / "metrics.json").exists():                     # resumable: finished families are skipped
            print(dataset, fam, "already trained", flush=True)
            continue
        cls = FAMILIES[fam]
        Xf, yf = subsample(Xtr, ytr, SUBSAMPLE.get(fam))
        fit_options = {"gpu": True} if fam == "catboost" and catboost_gpu else {}

        def objective(trial):
            m = cls().fit(Xf, yf, Xva, yva, wva, space(fam, trial), trial=trial, seed=0, **fit_options)
            return auprc(yva, m.logit(Xva), wva)

        if (out / "best_params.json").exists():
            best = json.loads((out / "best_params.json").read_text())["params"]
        else:
            # trials persist in SQLite so an interrupted search resumes where it stopped
            study = optuna.create_study(direction="maximize", study_name=f"{dataset}-{fam}", load_if_exists=True,
                                        storage=f"sqlite:///{out / 'study.db'}",
                                        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=0, n_startup_trials=5),
                                        pruner=optuna.pruners.HyperbandPruner(min_resource=2, max_resource=25, reduction_factor=3))
            t0 = time.time()
            # trials and seconds already used by an interrupted run count against the budget
            done = sum(t.state.is_finished() for t in study.trials)
            spent = sum((t.duration.total_seconds() if t.duration else 0.0) for t in study.trials)
            left_n = max(0, max(2, int(MAX_TRIALS[fam] * scale)) - done)
            left_s = max(60.0, BUDGET_S[fam] * scale - spent)
            if left_n:
                study.optimize(objective, n_trials=left_n, timeout=left_s, catch=(Exception,))
            study.trials_dataframe().to_csv(out / "trials.csv", index=False)
            best = space(fam, optuna.trial.FixedTrial(study.best_params))
            (out / "best_params.json").write_text(json.dumps({"params": best, "val_auprc": study.best_value,
                                                              "n_trials": len(study.trials), "hpo_seconds": spent + time.time() - t0}, indent=1))
        mets = []
        for seed in SEEDS:
            ms = out / f"metrics_seed{seed}.json"
            if ms.exists():
                mets.append(json.loads(ms.read_text()))
                continue
            m = cls().fit(Xf, yf, Xva, yva, wva, best, seed=seed, **fit_options)
            sv = m.logit(Xva)
            thr = threshold_at_fpr(yva, sv, wva, 0.01)        # operating point: 1% FPR on validation
            st = m.logit(a["X_test"])
            met = all_metrics(a["y_test"], st, a["w_test"], a["amt_test"], thr)
            met.update(seed=seed, threshold=thr)
            mets.append(met)
            np.save(out / f"test_scores_seed{seed}.npy", st)
            save_model(m, out / f"seed{seed}")
            ms.write_text(json.dumps(met))
            print(dataset, fam, seed, {k: round(v, 4) for k, v in met.items()}, flush=True)
        (out / "metrics.json").write_text(json.dumps(mets, indent=1))


def save_model(m, path: Path):
    """Save a trained model: ivis as the folder <path>/ (its Keras network saved by ivis, its logistic head
    pickled), every other family as one pickle, <path>.pkl."""
    if m.family == "ivis (triplet)":
        path.mkdir(exist_ok=True)
        m.ivis.save_model(str(path / "ivis"), overwrite=True)
        pickle.dump(m.head, open(path / "head.pkl", "wb"))
    else:
        pickle.dump(m, open(str(path) + ".pkl", "wb"))


def load_model(dataset, fam, seed=0):
    """Load the model that run() saved for (dataset, family, seed), ready for logit/proba/grad."""
    path = RUNS / dataset / fam / f"seed{seed}"
    if fam == "ivis":
        import tensorflow as tf
        from ivis import Ivis

        from fraudadv.models.zoo import IvisModel

        m = IvisModel()
        m.ivis = Ivis()
        m.ivis.load_model(str(path / "ivis"))
        m.head = pickle.load(open(path / "head.pkl", "rb"))
        m.encoder = getattr(m.ivis, "encoder_", None) or m.ivis.encoder
        m._w = tf.constant(m.head.coef_.T.astype(np.float32))
        m._b = tf.constant(m.head.intercept_.astype(np.float32))
        return m
    return pickle.load(open(str(path) + ".pkl", "rb"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--catboost-gpu", action="store_true",
                    help="train CatBoost on the GPU when one is selected (off: CPU, as for every paper model)")
    args = ap.parse_args()
    SEEDS[:] = [int(s) for s in args.seeds.split(",")]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run(args.dataset, args.families.split(","), args.scale, catboost_gpu=args.catboost_gpu)
