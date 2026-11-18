"""Defenses, each attacked by an attacker who knows it (adaptive evaluation).

What it does
  For every model family, trains the defended variants listed in DEFENSES from the family's tuned
  hyperparameters (no re-tuning, so a comparison isolates the defense), then attacks each one on the main grid's
  fraud sample and writes one row per (defense, attack, eps).

Defenses
  none        the undefended seed-0 model, run through the same evaluation (same attacks, sample and budgets), so
              every defense is compared with a baseline measured the same way
  pgd_at      mlp, ft_transformer, hgnn, hgt, chronos2, timesfm: PGD adversarial training (Madry et al. 2018). The
              graph models perturb the transaction's own features with its history fixed; the foundation models
              train their head with the frozen encoder's clean embedding held fixed (fraudadv.models.foundation)
  trades      the same families: TRADES (Zhang et al. 2019), beta = 6
  adv_aug     ivis: adversarial augmentation. ivis trains through the ivis library's own triplet-loss loop, which
              takes no per-batch loss, so PGD-AT and TRADES cannot be applied; instead, as adversarial boosting,
              two rounds of: attack up to 500 training frauds with APGD, add the evasions as extra frauds, refit.
  adv_boost   xgboost, catboost: adversarial boosting (Kantchelian et al. 2016). Two rounds of: attack up to 500
              training frauds with AATS, add the evasions as extra frauds, refit.
  monotone    xgboost: monotone constraints on the attacker-controllable features, direction = sign of the
              training-set Spearman correlation with the label (Incer et al. 2018)
  squeeze     every family: feature-squeezing detector (Xu et al. 2018). A transaction whose fraud
              probability moves by more than delta when its attacker-controllable features are rounded to a
              0.25-sd grid is sent to review. delta and the score threshold are set together so that the system
              still flags 1% of legitimate validation traffic.
  Adversarial training uses eps_train = 0.25 (L-inf, standardized units), constrained threat model, frauds only.

Evaluation (evaluate)
  Threshold: 1% weighted FPR on validation, for each defended model. Attacks, per eps in run_attacks.EPS:
    trees (not squeeze)          AATS on the robust trees                  -> "white-box (AATS)"
    differentiable (not squeeze) APGD on the defended model               -> "white-box (PGD)"
    every model                  Square, 300 queries                       -> "score black-box (Square)"
  Graph and foundation models are thresholded and evaluated on the rows of the grid (hpo.train_context.split_rows),
  with each transaction's context; the foundation models are attacked on the grid's 240 random frauds of the sample
  with the grid's foundation-model budgets (APGD 20 iterations x 2 restarts, Square 100 queries).
                                 (the squeezing system end to end, on its combined score)
    per fraud, the lowest score of the attacks above                       -> "worst case (all attacks)"
  Threat model: constrained, except ULB (DEFENSE_MODE). ULB's anonymised PCA features leave only amount and hour
  attacker-controllable, so its constrained ASR is about 2% for every model, too low to measure a defense; ULB
  defenses are trained and attacked in the unconstrained setting.

Inputs: processed data, runs/<d>/<family>/best_params.json and seed0.pkl, runs/<d>/attacks/sample.npz (fraud sample).
Output: runs/<d>/defenses/results<tag>.csv (resumable: finished (model, defense, mode) rows are skipped).
        With --tag, families can run as separate processes or machines; the report code reads every results*.csv.
        --defenses runs a subset of each family's defenses, --eps a subset of the budgets (the graph and
        foundation-model defenses and ivis's adversarial augmentation ran at eps 0.25 only, the budget of Table VII).

Used by: Table VII (reports/table7_defenses_by_dataset.csv), Figures S6 and S7, Section VI-J (RQ5,
make_report.defense_tables). defenses/run_defenses_consistent.py reuses DEFENSE_MODE, DEFENSES, train_defended and
evaluate (with eps=[0.25]) for Table S-XX.

usage: python -m fraudadv.defenses.run_defenses <dataset> [--families ...] [--defenses a,b] [--eps 0.25] [--tag _x]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from fraudadv.attacks.blackwhite import APGD_BUDGET, aats, apgd, square
from fraudadv.attacks.constraints import Constraints
from fraudadv.attacks.run_attacks import BUDGET, CONTEXT_FAMS, EPS, FM, RUNS, context_ids, load_model  # EPS: read at call time
from fraudadv.data.load import load
from fraudadv.eval.metrics import all_metrics, threshold_at_fpr
from fraudadv.hpo.train_all import SUBSAMPLE, subsample
from fraudadv.models.zoo import FAMILIES

EPS_TRAIN = 0.25
DEFENSE_MODE = {"ulb": "unconstrained"}          # every other dataset: "constrained"
ADV_TRAINED = ["none", "pgd_at", "trades", "squeeze"]
DEFENSES = {"mlp": ADV_TRAINED, "ft_transformer": ADV_TRAINED,
            "xgboost": ["none", "adv_boost", "monotone", "squeeze"], "catboost": ["none", "adv_boost", "squeeze"],
            "ivis": ["none", "adv_aug", "squeeze"],
            "hgnn": ADV_TRAINED, "hgt": ADV_TRAINED, "chronos2": ADV_TRAINED, "timesfm": ADV_TRAINED}
FM_N = 240                                       # frauds of the sample the foundation models are attacked on (grid)


class Squeezed:
    """A base model plus a feature-squeezing detector, exposed as one score.

    change(x) = |P_fraud(x) - P_fraud(squeeze(x))|, where squeeze rounds the mutable features to a grid.
    The combined logit is max(base logit, tau_base + 50 (change - delta)), so it is >= tau_base exactly when the
    base model flags (logit >= tau_base) or the detector flags (change >= delta). It stays continuous, so a
    score-based attacker (Square) can attack the whole system."""

    differentiable = False

    def __init__(self, base, cons, grid=0.25):
        self.base, self.cons, self.grid = base, cons, grid
        self.family = base.family + " + squeezing"

    def squeeze(self, X):
        """Round the mutable features to the grid, then project back onto the threat model's constraints."""
        Z = np.array(X, np.float32, copy=True)
        m = self.cons.mutable
        Z[:, m] = np.round(Z[:, m] / self.grid) * self.grid
        return self.cons.project(Z, X)

    def _parts(self, X, ctx=None):
        """(base logit, detector change) of each row."""
        logit = self.base.logit(X, ctx)
        logit_squeezed = self.base.logit(self.squeeze(X), ctx)
        return logit, np.abs(1 / (1 + np.exp(-logit)) - 1 / (1 + np.exp(-logit_squeezed)))

    def calibrate(self, Xv, yv, wv, fpr=0.01, det_share=0.5, ctx=None):
        """Set delta and tau_base on validation data (with its context ids `ctx`, for graph and foundation models)
        so that the system flags `fpr` of legitimate traffic.

        delta: the detector alone flags a share det_share of that budget (unweighted quantile on legit rows).
        tau_base: of 400 candidate score thresholds (legit quantiles 0.95 to 0.9999), the one whose combined
        weighted FPR (score flag OR detector flag) is closest to `fpr`."""
        logit, change = self._parts(Xv, ctx)
        legit = yv == 0
        self.delta = float(np.quantile(change[legit], 1 - fpr * det_share))
        detector_flags = change > self.delta
        candidates = np.quantile(logit[legit], np.linspace(0.95, 0.9999, 400))
        w = wv[legit]
        fpr_of = [np.sum(w[(logit[legit] >= c) | detector_flags[legit]]) / w.sum() for c in candidates]
        self.tau_base = float(candidates[int(np.argmin(np.abs(np.array(fpr_of) - fpr)))])
        return self

    def logit(self, X, ctx=None):
        logit, change = self._parts(X, ctx)
        return np.maximum(logit, self.tau_base + 50.0 * (change - self.delta))


def monotone_signs(X, y, mutable):
    """Monotone-constraint direction per feature: the sign of the Spearman correlation with the label on up to
    200,000 random training rows (seed 0), for mutable features with |rho| >= 0.01; 0 (unconstrained) otherwise."""
    from scipy.stats import spearmanr

    s = np.zeros(X.shape[1], int)
    rng = np.random.default_rng(0)
    rows = rng.choice(len(y), size=min(200_000, len(y)), replace=False)
    for j in mutable:
        r = spearmanr(X[rows, j], y[rows]).statistic
        s[j] = 0 if not np.isfinite(r) or abs(r) < 0.01 else int(np.sign(r))
    return s


def adversarial_boosting(dataset, fam, Xtr, ytr, Xva, yva, wva, params, cons, rounds=2, n_attacked=500):
    """Start from the stored seed-0 model; each round attacks up to n_attacked random training frauds (AATS,
    eps_train), adds the evasions as frauds and refits on the grown training set."""
    m = load_model(dataset, fam, 0)
    X, y = Xtr, ytr
    rng = np.random.default_rng(0)
    for _ in range(rounds):
        tau = threshold_at_fpr(yva, m.logit(Xva), wva, 0.01)
        frauds = np.flatnonzero(y == 1)
        pick = rng.choice(frauds, size=min(n_attacked, len(frauds)), replace=False)
        Xa, la, _ = aats(m, X[pick], tau, EPS_TRAIN, cons)
        evaded = la < tau
        X, y = np.vstack([X, Xa[evaded]]), np.r_[y, np.ones(evaded.sum(), y.dtype)]
        print(dataset, fam, "adv_boost round: added", int(evaded.sum()), "evasions", flush=True)
        m = FAMILIES[fam]().fit(X, y, Xva, yva, wva, params, seed=0)
    return m


def adversarial_augmentation(dataset, fam, Xtr, ytr, Xva, yva, wva, params, cons, rounds=2, n_attacked=500):
    """Adversarial training for ivis, whose library training loop takes no per-batch loss: as adversarial_boosting,
    start from the stored seed-0 model; each round attacks up to n_attacked random training frauds with APGD at
    eps_train, adds the evasions as frauds and refits on the grown training set."""
    m = load_model(dataset, fam, 0)
    X, y = Xtr, ytr
    rng = np.random.default_rng(0)
    for _ in range(rounds):
        tau = threshold_at_fpr(yva, m.logit(Xva), wva, 0.01)
        frauds = np.flatnonzero(y == 1)
        pick = rng.choice(frauds, size=min(n_attacked, len(frauds)), replace=False)
        Xa, la = apgd(m, X[pick], tau, EPS_TRAIN, cons, *APGD_BUDGET["default"])
        evaded = la < tau
        X, y = np.vstack([X, Xa[evaded]]), np.r_[y, np.ones(evaded.sum(), y.dtype)]
        print(dataset, fam, "adv_aug round: added", int(evaded.sum()), "evasions", flush=True)
        m = FAMILIES[fam]().fit(X, y, Xva, yva, wva, params, seed=0)
    return m


def adversarial_training_options(defense, cons):
    """The `adv` entry of a model's params for PGD-AT or TRADES (see fraudadv.models.zoo.adversarial_loss)."""
    return dict(method="pgd" if defense == "pgd_at" else "trades", eps=EPS_TRAIN, steps=5, mutable=list(cons.mutable),
                lower=cons.lower, upper=cons.upper, frauds_only=True, beta=6.0)


def split_rows(dataset, fam, a):
    """(validation rows, weights), (test rows, weights) and the context ids of both that family `fam` is thresholded
    and evaluated on: for graph and foundation models those of hpo.train_context.split_rows, for the others the whole
    splits (context None)."""
    if fam not in CONTEXT_FAMS:
        return (slice(None), a["w_val"]), (slice(None), a["w_test"]), None, None
    from fraudadv.hpo.train_context import split_rows as context_split_rows

    keep = context_ids(dataset)
    _, (iva, wva), (ite, wte) = context_split_rows(a, fam)
    return (iva, wva), (ite, wte), keep["val"][iva], keep["test"][ite]


def train_defended_context(dataset, fam, defense, a, meta, cons):
    """train_defended for the graph and foundation models: their training rows and context, the same training
    call as hpo.train_context with the family's tuned hyperparameters, seed 0."""
    from fraudadv.hpo.train_context import make, split_rows as context_split_rows

    if defense == "none":
        return load_model(dataset, fam, 0)
    keep = context_ids(dataset)
    itr, (iva, wva), _ = context_split_rows(a, fam)
    Xva, yva, cva = a["X_val"][iva], a["y_val"][iva], keep["val"][iva]
    if defense == "squeeze":
        return Squeezed(load_model(dataset, fam, 0), cons).calibrate(Xva, yva, wva, ctx=cva)
    if defense in ("pgd_at", "trades"):
        params = json.loads((RUNS / dataset / fam / "best_params.json").read_text())["params"]
        Xtr, ytr, ctr = a["X_train"][itr], a["y_train"][itr], keep["train"][itr]
        return make(fam, dataset, meta).fit(Xtr, ytr, Xva, yva, wva, dict(params, adv=adversarial_training_options(defense, cons)),
                                            ctr, cva, seed=0)
    raise ValueError(defense)


def train_defended(dataset, fam, defense, a, meta, cons):
    """Return the model of `defense` for family `fam` (trained here, or the stored seed-0 model for "none").
    `meta` is used by the graph and foundation models only."""
    if fam in CONTEXT_FAMS:
        return train_defended_context(dataset, fam, defense, a, meta, cons)
    params = json.loads((RUNS / dataset / fam / "best_params.json").read_text())["params"]
    Xtr, ytr = subsample(a["X_train"], a["y_train"], SUBSAMPLE.get(fam))
    Xva, yva, wva = a["X_val"], a["y_val"], a["w_val"]
    if defense == "none":
        return load_model(dataset, fam, 0)
    if defense == "squeeze":
        return Squeezed(load_model(dataset, fam, 0), cons).calibrate(Xva, yva, wva)
    if defense in ("pgd_at", "trades"):
        return FAMILIES[fam]().fit(Xtr, ytr, Xva, yva, wva, dict(params, adv=adversarial_training_options(defense, cons)), seed=0)
    if defense == "monotone":
        s = monotone_signs(Xtr, ytr, cons.mutable)
        p = dict(params, monotone_constraints="(" + ",".join(str(v) for v in s) + ")")
        return FAMILIES[fam]().fit(Xtr, ytr, Xva, yva, wva, p, seed=0)
    if defense == "adv_boost":
        return adversarial_boosting(dataset, fam, Xtr, ytr, Xva, yva, wva, params, cons)
    if defense == "adv_aug":
        return adversarial_augmentation(dataset, fam, Xtr, ytr, Xva, yva, wva, params, cons)
    raise ValueError(defense)


def attack_sample(dataset, a, n):
    """Test positions of the attacked frauds: the main grid's sample if it exists, else n random frauds (seed 0)."""
    sp = RUNS / dataset / "attacks" / "sample.npz"
    if sp.exists():
        return np.load(sp)["S"]
    fr = np.flatnonzero(a["y_test"] == 1)
    return np.sort(np.random.default_rng(0).choice(fr, size=min(n, len(fr)), replace=False))


def evaluate(dataset, fam, defense, m, a, cons, n=200, n_bb=100, mode="constrained", eps=None):
    """Attack model m adaptively at every budget in eps (default: the grid's EPS, read at call time); return the
    result rows.
    n: sample size when the grid's sample is missing. n_bb: unused (the black-box attacks run on the whole sample);
    kept for callers."""
    (iva, wva), (ite, wte), cva, cte = split_rows(dataset, fam, a)
    tau = threshold_at_fpr(a["y_val"][iva], m.logit(a["X_val"][iva], cva), wva, 0.01)
    clean = all_metrics(a["y_test"][ite], m.logit(a["X_test"][ite], cte), wte, a["amt_test"][ite], tau)
    S = attack_sample(dataset, a, n)
    if fam in FM:                     # the grid's fixed random subset of the sample (all of it when it is smaller)
        S = S[np.sort(np.random.default_rng(1).permutation(len(S))[:FM_N])]
    ctx0 = context_ids(dataset)["test"][S] if fam in CONTEXT_FAMS else None
    X0, amt = a["X_test"][S], a["amt_test"][S].astype(float)
    detected = m.logit(X0, ctx0) > tau
    apgd_budget = APGD_BUDGET["fm" if fam in FM else "default"]
    square_queries = BUDGET["fm" if fam in FM else "default"][3]
    rows = []

    def add(attack, e, attacked_logit):
        """One result row: ASR = evaded / detected frauds, and its dollar-weighted version."""
        evaded = detected & (attacked_logit < tau)
        rows.append(dict(dataset=dataset, model=fam, defense=defense, mode=mode, attack=attack, eps=e,
                         n_detected=int(detected.sum()), asr=float(evaded.sum() / max(detected.sum(), 1)),
                         asr_usd=float(amt[evaded].sum() / max(amt[detected].sum(), 1e-9)),
                         **{f"clean_{k}": v for k, v in clean.items()}))

    for e in (EPS if eps is None else eps):
        attacked = []
        if defense != "squeeze" and fam in ("xgboost", "catboost"):
            _, la, _ = aats(m, X0, tau, e, cons)
            add("white-box (AATS)", e, la)
            attacked.append(la)
        elif defense != "squeeze" and getattr(m, "differentiable", False):
            _, la = apgd(m, X0, tau, e, cons, *apgd_budget, ctx=ctx0)
            add("white-box (PGD)", e, la)
            attacked.append(la)
        _, ls = square(m, X0, tau, e, cons, n_queries=square_queries, ctx=ctx0)
        add("score black-box (Square)", e, ls)
        attacked.append(ls)
        add("worst case (all attacks)", e, np.min(np.stack(attacked), 0))
        print(rows[-1]["model"], defense, e, "worst ASR", round(rows[-1]["asr"], 3), flush=True)
    return rows


def run(dataset, families, tag="", only=None, eps=None):
    """Train and evaluate every defense of `families` (or only those named in `only`) on `dataset` at the budgets
    `eps` (default: the grid's EPS), appending to defenses/results<tag>.csv."""
    a, meta = load(dataset)
    mode = DEFENSE_MODE.get(dataset, "constrained")
    cons = Constraints(meta, a["X_train"], mode)
    out = RUNS / dataset / "defenses"
    out.mkdir(parents=True, exist_ok=True)
    res = out / f"results{tag}.csv"
    rows = pd.read_csv(res).to_dict("records") if res.exists() else []
    for r in rows:                                                  # rows from files without a mode column
        if not isinstance(r.get("mode"), str):
            r["mode"] = "constrained"
    done = {(r["model"], r["defense"], r["mode"]) for r in rows}
    for fam in families:
        for defense in DEFENSES.get(fam, []):
            if (fam, defense, mode) in done or (only and defense not in only):     # resumable
                continue
            try:
                m = train_defended(dataset, fam, defense, a, meta, cons)
                rows += evaluate(dataset, fam, defense, m, a, cons, mode=mode, eps=eps)
            except Exception as e:                                  # one failed defense must not stop the rest
                print("FAILED", dataset, fam, defense, repr(e), flush=True)
            pd.DataFrame(rows).to_csv(res, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--families", default="xgboost,catboost,mlp,ft_transformer,ivis")
    ap.add_argument("--defenses", default="", help="subset of each family's defenses (default: all)")
    ap.add_argument("--eps", default="", help="comma-separated L-inf budgets (default: the grid's EPS)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if {"chronos2", "timesfm"} & set(args.families.split(",")):
        import chronos  # noqa: F401  before TensorFlow (ivis): the reverse order aborts on an LLVM option clash
    run(args.dataset, args.families.split(","), args.tag, set(args.defenses.split(",")) - {""},
        [float(e) for e in args.eps.split(",")] if args.eps else None)
