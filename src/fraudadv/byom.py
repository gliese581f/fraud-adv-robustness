"""Bring your own model: attack any fraud detector with the study's threat models, attacks and metrics.

No result of the paper uses this file. It runs the protocol behind the attack grid (Table IV, Tables S-V to S-X)
on a model you supply, so its numbers compare with the paper's tables.

Inputs: a dataset name, a model (an object with the methods below), and a threat model ("constrained" or
"unconstrained"). Output: a DataFrame with one row per attack and budget eps (columns dataset, model, mode,
attack, eps, n, n_detected, asr, asr_usd, robust_recall, tau), optionally saved as CSV by the command line.

A model is any object with

    logit(X, ctx=None) -> array (n,)      fraud score of the standardized feature rows X (higher = more fraudulent;
                                          a logit or any score monotone in it)

and, to enable the white-box attacks, optionally

    grad(X, ctx=None) -> array (n, d)     gradient of logit with respect to X               -> APGD
    tree_ensemble() -> TreeEnsemble       axis-aligned trees (fraudadv.attacks.trees)       -> AATS

X is the study's feature matrix for the dataset (`fraudadv.data.load.load(dataset)`: target-encoded, imputed and
standardized with training statistics), so a model must be trained on that representation; `X_train`, `y_train`,
`w_train` (sampling weights) and the validation split come with it. Adapters below wrap scikit-learn style
classifiers, PyTorch modules, and XGBoost / CatBoost classifiers (which also get the tree attack).

    from fraudadv import byom
    a, meta = byom.data("sparkov")
    clf = sklearn.ensemble.HistGradientBoostingClassifier().fit(a["X_train"], a["y_train"])
    res = byom.evaluate(byom.from_sklearn(clf), "sparkov", mode="constrained")
    print(res.pivot(index="attack", columns="eps", values="asr"))

`evaluate` does what the study does for each of its own models: threshold at a 1% weighted false-positive rate on
validation legitimate transactions, the study's stored fraud sample (so numbers compare with the paper's tables),
the threat-model projection after every attack step, every attack at every budget, and the per-example worst case.
Command line: python -m fraudadv.byom <dataset> <module:factory> [--mode constrained|unconstrained] [--out file.csv],
where factory(a, meta) returns a fitted model.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPS = (0.05, 0.1, 0.25, 0.5, 1.0)


# ------------------------------------------------------------------------------------------------ adapters
class Model:
    """Adapter from plain functions: logit_fn(X) -> (n,), optional grad_fn(X) -> (n, d) and ensemble_fn() ->
    TreeEnsemble. Methods are only present when their function is given, so `evaluate` picks the attacks."""

    def __init__(self, logit_fn, grad_fn=None, ensemble_fn=None, name="model"):
        self._logit, self.name = logit_fn, name
        if grad_fn is not None:
            self.grad = lambda X, ctx=None: np.asarray(grad_fn(np.asarray(X, np.float32)), np.float32)
        if ensemble_fn is not None:
            self.tree_ensemble = ensemble_fn

    def logit(self, X, ctx=None):
        return np.asarray(self._logit(np.asarray(X, np.float32)), np.float64).ravel()


def _to_logit(p):
    """log(p / (1 - p)), with p clipped away from 0 and 1."""
    p = np.clip(np.asarray(p, np.float64).ravel(), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def from_proba(proba_fn, name="model"):
    """Black-box adapter from a function X -> P(fraud) (score-based and decision-based attacks only)."""
    return Model(lambda X: _to_logit(proba_fn(X)), name=name)


def from_sklearn(clf, name=None):
    """Any classifier with predict_proba (or decision_function); black-box attacks only."""
    if hasattr(clf, "predict_proba"):
        return from_proba(lambda X: clf.predict_proba(X)[:, 1], name or type(clf).__name__)
    return Model(lambda X: clf.decision_function(X), name=name or type(clf).__name__)


def from_torch(module, name=None, device="cpu"):
    """A torch module mapping (n, d) float32 inputs to (n,) or (n, 1) fraud logits; white-box (APGD) enabled."""
    import torch

    module = module.to(device).eval()

    def logit(X):
        with torch.no_grad():
            return module(torch.as_tensor(X, device=device)).reshape(-1).cpu().numpy()

    def grad(X):
        x = torch.as_tensor(X, device=device).requires_grad_(True)
        module(x).reshape(-1).sum().backward()
        return x.grad.detach().cpu().numpy()

    return Model(logit, grad, name=name or type(module).__name__)


def from_xgboost(clf, name="xgboost"):
    """A fitted xgboost.XGBClassifier; white-box via AATS on its trees."""
    from fraudadv.attacks.trees import TreeEnsemble

    best = getattr(clf, "best_iteration", None)
    rng = None if best is None else (0, best + 1)
    booster = clf.get_booster()

    def logit(X):
        return booster.inplace_predict(X, predict_type="margin", **({"iteration_range": rng} if rng else {}))

    return Model(logit, ensemble_fn=lambda: TreeEnsemble.from_xgboost(clf, iteration_range=rng), name=name)


def from_catboost(clf, n_features, name="catboost"):
    """A fitted catboost.CatBoostClassifier on the study's numeric features; white-box via AATS."""
    from fraudadv.attacks.trees import TreeEnsemble

    return Model(lambda X: clf.predict(X, prediction_type="RawFormulaVal"),
                 ensemble_fn=lambda: TreeEnsemble.from_catboost(clf, n_features=n_features), name=name)


# ------------------------------------------------------------------------------------------------ evaluation
def data(dataset):
    """The study's processed arrays and metadata for a dataset (run `python -m fraudadv.data.prepare` first)."""
    from fraudadv.data.load import load

    return load(dataset)


def _sample(dataset, a, s_test, tau, n_detected=200, cap=1000):
    """The study's stored fraud sample when present (runs/<dataset>/attacks/sample.npz), else one drawn the same
    way for this model: sized so that it detects about n_detected frauds."""
    from fraudadv.attacks.run_attacks import RUNS

    p = Path(RUNS) / dataset / "attacks" / "sample.npz"
    if p.exists():
        return np.load(p)["S"], "study"
    fr = np.flatnonzero(a["y_test"] == 1)
    r = max(float(np.mean(s_test[fr] > tau)), 0.05)
    n = int(min(len(fr), np.ceil(n_detected / r), cap))
    return np.sort(np.random.default_rng(0).choice(fr, size=n, replace=False)), "drawn"


def evaluate(model, dataset, mode="constrained", eps=EPS, attacks=("white-box", "square", "hopskipjump"),
             apgd_budget=(100, 5), square_queries=300, hsj_iters=10, sample=None, verbose=True):
    """Attack `model` on `dataset` in threat model `mode`; returns one row per (attack, eps) with the number of
    detected frauds, attack success rate (ASR, share of detected frauds that evade), dollar ASR and robust recall
    (recall on the sample x (1 - ASR)), plus the per-example worst case over the attacks run.
    White-box: AATS if the model has tree_ensemble(), else APGD if it has grad(), else skipped.
    sample: indices into the test split to attack instead of the study's fraud sample."""
    from fraudadv.attacks.blackwhite import aats, apgd, hsj, square
    from fraudadv.attacks.constraints import Constraints
    from fraudadv.eval.metrics import threshold_at_fpr

    a, meta = data(dataset)
    tau = threshold_at_fpr(a["y_val"], model.logit(a["X_val"]), a["w_val"], 0.01)
    if sample is None:
        S, origin = _sample(dataset, a, model.logit(a["X_test"]), tau)
    else:
        S, origin = np.asarray(sample), "given"
    X0, amt = a["X_test"][S].astype(np.float32), a["amt_test"][S].astype(float)
    l0 = model.logit(X0)
    det = l0 > tau
    recall = float(det.mean())
    cons = Constraints(meta, a["X_train"], mode)
    if verbose:
        print(f"{dataset} {mode}: tau={tau:.4f}, {len(S)} frauds ({origin} sample), {int(det.sum())} detected", flush=True)
    rows, per_eps = [], {e: [] for e in eps}

    def add(attack, e, l_adv, extra=None):
        """Record one (attack, eps) row from the adversarial logits l_adv; an evasion is a detected fraud whose
        adversarial logit falls below tau. Successful logits also enter the per-example worst case."""
        ev = det & (l_adv < tau)
        asr = float(ev.sum() / max(det.sum(), 1))
        rows.append(dict(dataset=dataset, model=getattr(model, "name", "model"), mode=mode, attack=attack, eps=e,
                         n=len(S), n_detected=int(det.sum()), asr=asr,
                         asr_usd=float(amt[ev].sum() / max(amt[det].sum(), 1e-9)), robust_recall=recall * (1 - asr),
                         tau=tau, **(extra or {})))
        per_eps[e].append(np.where(l_adv < tau, l_adv, l0))

    if "white-box" in attacks and hasattr(model, "tree_ensemble"):
        _, la, res = aats(model, X0, tau, max(eps), cons)
        dist = np.where(res.success, res.cost, np.inf)
        for e in eps:
            add("white-box (AATS)", e, np.where(dist <= e + 1e-6, la, l0))
    elif "white-box" in attacks and hasattr(model, "grad"):
        for e in eps:
            add("white-box (APGD)", e, apgd(model, X0, tau, e, cons, n_iter=apgd_budget[0], restarts=apgd_budget[1])[1])
    if "square" in attacks:
        for e in eps:
            add("score black-box (Square)", e, square(model, X0, tau, e, cons, n_queries=square_queries)[1])
    if "hopskipjump" in attacks:
        Xh, lh = hsj(model, X0, tau, cons, max_iter=hsj_iters)
        dh = cons.linf(Xh, X0)
        for e in eps:
            add("decision black-box (HopSkipJump)", e, np.where((lh < tau) & (dh <= e + 1e-6), lh, l0))
    for e in eps:
        if per_eps[e]:
            add("worst case (all attacks)", e, np.min(np.stack(per_eps[e]), 0))
    out = pd.DataFrame(rows)
    if verbose:
        print(out.pivot_table(index="attack", columns="eps", values="asr").round(3).to_string(), flush=True)
    return out


def main(argv=None):
    """Command line: build the model with the given factory, attack it, and optionally save the results."""
    ap = argparse.ArgumentParser(description="Attack your own fraud model with the study's grid.")
    ap.add_argument("dataset")
    ap.add_argument("factory", help="module:function; function(a, meta) returns a fitted model (see module docstring)")
    ap.add_argument("--mode", default="constrained", choices=["constrained", "unconstrained"])
    ap.add_argument("--out", help="CSV file for the results")
    args = ap.parse_args(argv)
    mod, fn = args.factory.split(":")
    sys.path.insert(0, ".")
    a, meta = data(args.dataset)
    model = getattr(importlib.import_module(mod), fn)(a, meta)
    res = evaluate(model, args.dataset, mode=args.mode)
    if args.out:
        res.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
