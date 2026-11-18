"""Tune and train the history-aware model families, which also read each card's earlier transactions:
the heterogeneous graph network (hgnn), the heterogeneous graph transformer (hgt), the Chronos-2 and TimesFM 2.5
foundation models (chronos2, timesfm), and their LoRA and zero-shot versions (chronos2_lora, timesfm_lora,
chronos2_zs, timesfm_zs; models/foundation_adapt.py).

The protocol is that of fraudadv.hpo.train_all: an Optuna search (multivariate TPE plus Hyperband) scored by
weighted validation AUPRC, the best configuration retrained with seeds 0, 1, 2, and a threshold at 1% FPR on
validation. Two differences in the rows used:
  * Foundation models run a 120M-230M parameter network on every transaction, the bottleneck on CPU. They
    use fixed subsets: all frauds plus FM_LEGIT legitimate rows per split, with the legitimate weights scaled
    up so the metrics still estimate population values (fm_subset). The test rows are saved as
    test_subset.npy (row indices into the test split), so the stored test scores can be matched to rows.
  * The graph models train on all frauds plus legitimate rows up to GRAPH_TRAIN_ROWS in total, and are
    evaluated on the full validation and test splits.

Inputs:  data/processed/<dataset>/arrays.npz and ctx.npz, the history context built by
         `fraudadv.data.prepare <d> --context` (each row's index into it: fraudadv.attacks.run_attacks.context_ids).
Outputs: runs/<dataset>/<family>/ with the files of train_all (study.db, trials.csv, best_params.json,
         seed<k>.pkl, test_scores_seed<k>.npy, metrics_seed<k>.json, metrics.json) plus test_subset.npy for
         the foundation models. Models are saved by save() and loaded by load_context_model().

Results: Table V, Section VI-F and the foundation-model and graph rows of every grid table;
Table S-II, Table S-IV; Table S-XXVIII for the LoRA and zero-shot
families. The search spaces in space() are Table S-III.

usage: python -m fraudadv.hpo.train_context <dataset> --families hgnn,hgt,chronos2,timesfm [--seeds 0,1,2]
                                            [--lora-lr 2e-4] [--lora-epochs 12]
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import optuna

from fraudadv.attacks.run_attacks import FM, context_ids
from fraudadv.data.load import load
from fraudadv.eval.metrics import all_metrics, auprc, threshold_at_fpr
from fraudadv.hpo.train_all import RUNS

SEEDS = [0, 1, 2]                    # seeds of the retrained models (--seeds replaces them)
# search budget per family: seconds and trials (the second pass, topup_context, raises both)
BUDGET_S = {"hgnn": 900, "hgt": 1200, "chronos2": 300, "timesfm": 300}
MAX_TRIALS = {"hgnn": 12, "hgt": 10, "chronos2": 15, "timesfm": 15}
FM_LEGIT = {"train": 20_000, "val": 10_000, "test": 20_000}   # legitimate rows per split, foundation models
GRAPH_TRAIN_ROWS = 150_000                                     # training rows, graph models
LORA_LR, LORA_EPOCHS = 2e-4, 12      # adapter learning rate and epochs of the LoRA families (--lora-lr, --lora-epochs)


def make(fam, dataset, meta):
    """A new, untrained model of family `fam` for `dataset`."""
    if fam == "hgnn":
        from fraudadv.models.graph import HeteroGNN

        return HeteroGNN(dataset)
    if fam == "hgt":
        from fraudadv.models.graph import GraphTransformer

        return GraphTransformer(dataset)
    from fraudadv.models.foundation_adapt import ADAPT_FAMS, make_adapt

    if fam in ADAPT_FAMS:
        return make_adapt(fam, dataset, meta)
    from fraudadv.models.foundation import Chronos2Model, TimesFMModel

    cls = Chronos2Model if fam == "chronos2" else TimesFMModel
    return cls(dataset, meta["features"], meta["mu"], meta["sd"])


def space(fam, t):
    """The hyperparameters of family `fam` for Optuna trial `t` (Table S-III), fixed values included. Every
    foundation-model family (frozen, LoRA, zero-shot) shares one space: the MLP head on the encoding."""
    if fam in ("hgnn", "hgt"):
        p = dict(hidden=t.suggest_categorical("hidden", [32, 64]), dropout=t.suggest_float("dropout", 0.0, 0.3),
                 lr=t.suggest_float("lr", 3e-4, 3e-3, log=True), weight_decay=t.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                 pos_weight=t.suggest_float("pos_weight", 1, 20, log=True), K1=t.suggest_categorical("K1", [4, 8, 16]), K2=8,
                 K1b=4, K2b=4, epochs=8, batch_size=512, patience=2)
        if fam == "hgt":
            p["n_heads"] = 4
        return p
    return dict(hidden=t.suggest_categorical("hidden", [64, 128, 256]), proj=t.suggest_categorical("proj", [16, 32, 64]),
                dropout=t.suggest_float("dropout", 0.0, 0.4), lr=t.suggest_float("lr", 1e-4, 3e-3, log=True),
                weight_decay=t.suggest_float("weight_decay", 1e-6, 1e-2, log=True), pos_weight=t.suggest_float("pos_weight", 1, 20, log=True),
                epochs=40, patience=5)


def fm_subset(a, part, seed=0):
    """All frauds + FM_LEGIT[part] legit rows; returns (row indices, rescaled weights)."""
    rng = np.random.default_rng(seed)
    y, w = a[f"y_{part}"], a[f"w_{part}"].astype(float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k = min(FM_LEGIT[part], len(neg))
    pick = np.sort(rng.choice(neg, size=k, replace=False))
    idx = np.sort(np.r_[pos, pick])
    w2 = w.copy()
    w2[neg] = w[neg] * len(neg) / k
    return idx, w2[idx]


def split_rows(a, fam):
    """The rows of each split that family `fam` trains and is evaluated on: training rows, (validation rows,
    weights), (test rows, weights). Foundation models: fm_subset of each split. Graph models: all training frauds
    plus random legitimate rows (seed 0) up to GRAPH_TRAIN_ROWS, and the whole validation and test splits."""
    if fam in FM:
        itr, _ = fm_subset(a, "train")
        return itr, fm_subset(a, "val"), fm_subset(a, "test")
    rng = np.random.default_rng(0)
    ytr = a["y_train"]
    pos, neg = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
    itr = np.sort(np.r_[pos, rng.choice(neg, size=min(len(neg), GRAPH_TRAIN_ROWS - len(pos)), replace=False)])
    return itr, (np.arange(len(a["y_val"])), a["w_val"]), (np.arange(len(a["y_test"])), a["w_test"])


def fit_options(fam, lora_lr, lora_epochs):
    """Extra keyword arguments of fit() for family `fam`: the adapter learning rate and epochs of the LoRA
    families, none for the others."""
    from fraudadv.models.foundation_adapt import LORA_FAMS

    return dict(lora_lr=lora_lr, lora_epochs=lora_epochs) if fam in LORA_FAMS else {}


def run(dataset, families, before_retrain=None, lora_lr=LORA_LR, lora_epochs=LORA_EPOCHS):
    """Search, retrain and evaluate each family of `families` on `dataset` (the protocol of the module docstring).

    before_retrain(out_dir, study_or_None) runs once the best configuration is fixed and before the seed
    retrains (the second HPO pass uses it to drop stale seed models when the best configuration changed).
    lora_lr and lora_epochs set the adapter training of the LoRA families."""
    a, meta = load(dataset)
    keep = context_ids(dataset)
    for fam in families:
        out = RUNS / dataset / fam
        out.mkdir(parents=True, exist_ok=True)
        if before_retrain is None and (out / "hpo_v2_wip.json").exists():
            # the second HPO pass (topup_context) owns this family until it finishes; a first-pass call here
            # would search the same study concurrently and retrain seeds the second pass then discards
            print(dataset, fam, "second HPO pass in progress; left to topup_context", flush=True)
            continue
        if before_retrain is None and (out / "metrics.json").exists():
            print(dataset, fam, "already trained", flush=True)
            continue
        fm = fam in FM
        itr, (iva, wva), (ite, wte) = split_rows(a, fam)
        if fm:
            np.save(out / "test_subset.npy", ite)
        Xtr, ytr, ctr = a["X_train"][itr], a["y_train"][itr], keep["train"][itr]
        Xva, yva, cva = a["X_val"][iva], a["y_val"][iva], keep["val"][iva]
        Xte, yte, cte, ate = a["X_test"][ite], a["y_test"][ite], keep["test"][ite], a["amt_test"][ite]

        def objective(trial):
            m = make(fam, dataset, meta).fit(Xtr, ytr, Xva, yva, wva, space(fam, trial), ctr, cva, trial=trial, seed=0,
                                             **fit_options(fam, lora_lr, lora_epochs))
            return auprc(yva, m.logit(Xva, cva), wva)

        study = None
        if (out / "best_params.json").exists():
            best = json.loads((out / "best_params.json").read_text())["params"]
        else:
            study = optuna.create_study(direction="maximize", study_name=f"{dataset}-{fam}", load_if_exists=True,
                                        storage=f"sqlite:///{out / 'study.db'}",
                                        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=0, n_startup_trials=4),
                                        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=40, reduction_factor=3))
            t0 = time.time()
            # trials and seconds already used by an interrupted run count against the budget
            done = sum(t.state.is_finished() for t in study.trials)
            spent = sum((t.duration.total_seconds() if t.duration else 0.0) for t in study.trials)
            if MAX_TRIALS[fam] - done > 0:
                study.optimize(objective, n_trials=MAX_TRIALS[fam] - done, timeout=max(60.0, BUDGET_S[fam] - spent), catch=(Exception,))
            study.trials_dataframe().to_csv(out / "trials.csv", index=False)
            best = space(fam, optuna.trial.FixedTrial(study.best_params))
            (out / "best_params.json").write_text(json.dumps({"params": best, "val_auprc": study.best_value, "n_trials": len(study.trials),
                                                              "hpo_seconds": spent + time.time() - t0, "fm_subset": fm}, indent=1))
        if before_retrain is not None:
            before_retrain(out, study)
        mets = []
        for seed in SEEDS:
            ms = out / f"metrics_seed{seed}.json"
            if ms.exists():
                mets.append(json.loads(ms.read_text()))
                continue
            m = make(fam, dataset, meta).fit(Xtr, ytr, Xva, yva, wva, best, ctr, cva, seed=seed,
                                             **fit_options(fam, lora_lr, lora_epochs))
            thr = threshold_at_fpr(yva, m.logit(Xva, cva), wva, 0.01)        # operating point: 1% FPR on validation
            st = m.logit(Xte, cte)
            met = all_metrics(yte, st, wte, ate, thr)
            met.update(seed=seed, threshold=thr, eval_rows=int(len(ite)))
            mets.append(met)
            np.save(out / f"test_scores_seed{seed}.npy", st)
            save(m, out / f"seed{seed}.pkl")
            ms.write_text(json.dumps(met))
            print(dataset, fam, seed, {k: round(v, 4) for k, v in met.items()}, flush=True)
        (out / "metrics.json").write_text(json.dumps(mets, indent=1))


def save(m, path):
    """Save a trained history-aware model to `path`. LoRA and zero-shot models use foundation_adapt.save_adapt.
    For the others, the network weights are stored as CPU state dicts next to the model's other attributes,
    so the file loads on any machine; the frozen foundation backbone is not stored (it is loaded again by name)."""
    import torch

    if getattr(m, "family_key", ""):                  # LoRA / zero-shot foundation models
        from fraudadv.models.foundation_adapt import save_adapt

        return save_adapt(m, path)
    st = {k: v for k, v in m.__dict__.items() if k not in ("net", "proj")}
    st["cls"] = type(m).__name__
    cpu = lambda sd: {k: v.detach().cpu() for k, v in sd.items()}    # noqa: E731  saved models load on any machine
    st["net_state"] = cpu(m.net.state_dict())
    if hasattr(m, "proj"):
        st["proj_state"] = cpu(m.proj.state_dict())
    torch.save(st, path)


def load_context_model(dataset, fam, seed=0):
    """Load the model that run() saved for (dataset, family, seed): rebuild the networks from the stored
    shapes and parameters, then load their weights."""
    import torch
    import torch.nn as nn

    from fraudadv.models.foundation_adapt import ADAPT_FAMS, load_adapt

    if fam in ADAPT_FAMS:
        return load_adapt(RUNS / dataset / fam / f"seed{seed}.pkl", dataset, fam, load(dataset)[1])
    st = torch.load(RUNS / dataset / fam / f"seed{seed}.pkl", weights_only=False, map_location="cpu")
    _, meta = load(dataset)
    m = make(fam, dataset, meta)
    for k, v in st.items():
        if k not in ("cls", "net_state", "proj_state"):
            setattr(m, k, v)
    if fam in ("hgnn", "hgt"):
        from fraudadv.models.graph import build_net

        m.net = build_net(m.kind, len(meta["features"]), m.params)
    else:
        # foundation models: the head's widths are read from the stored weights (dropout is off at inference)
        H = st["net_state"]["0.weight"].shape[0]
        proj = st["proj_state"]["1.weight"].shape[0]
        m.proj = nn.Sequential(nn.LayerNorm(m.emb_dim), nn.Linear(m.emb_dim, proj))
        m.net = nn.Sequential(nn.Linear(len(meta["features"]) + proj, H), nn.ReLU(), nn.Dropout(0.0), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 1))
        m.proj.load_state_dict(st["proj_state"])
        m.proj.eval()
    m.net.load_state_dict(st["net_state"])
    m.net.eval()
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--families", default="hgnn,hgt")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lora-lr", type=float, default=LORA_LR, help="adapter learning rate of the LoRA families")
    ap.add_argument("--lora-epochs", type=int, default=LORA_EPOCHS, help="training epochs of the LoRA families")
    args = ap.parse_args()
    SEEDS[:] = [int(s) for s in args.seeds.split(",")]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run(args.dataset, args.families.split(","), lora_lr=args.lora_lr, lora_epochs=args.lora_epochs)
