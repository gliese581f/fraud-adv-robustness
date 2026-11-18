"""Controlled architecture ablations: which property of a model makes it easy or hard to evade.

Design
  Each variant changes ONE factor of a family's tuned base model. Everything else is held fixed: data, split,
  training rows, seed, threshold rule (1% FPR on validation), fraud sample, attack algorithms and budgets. Because
  we set the factor and nothing else changes, a robustness difference between a variant and its base is the
  interventional (causal) effect of that factor within the study design. The same frauds are attacked under every
  variant, so contrasts are paired per fraud.

Variants (tuned hyperparameters unless named; the tuned bases reuse the stored seed-0/1/2 models)
  XGBoost   xgb            tuned base
            xgb_d2/d6/d10  max_depth 2 / 6 / 10 (a level equal to the tuned depth is the base itself, and skipped)
            xgb_mono       monotone constraints on attacker-controllable features (training-set Spearman sign)
            xgb_from_mlp   XGBoost regressor fitted to the logits of the MLP base of the same seed: the neural
                           model's learned function in a tree representation (distillation)
  MLP       mlp            tuned base (SiLU)
            mlp_l1/l4      1 / 4 hidden layers
            mlp_relu       ReLU instead of SiLU (piecewise-linear instead of smooth activation)
            mlp_sn         spectral normalisation of every layer, output gain 4: the logit is at most about
                           4-Lipschitz (L2, standardized units); a direct smoothness intervention
            mlp_wd         weight decay 0.1 (AdamW, decoupled), a strong L2 regulariser
            mlp_from_xgb   MLP fitted to the logits of the XGBoost base of the same seed: the tree model's learned
                           function in a neural representation (distillation)
  The two distilled variants and the two bases form a 2 x 2 design that separates the REPRESENTATION (trees vs a
  smooth network) from the FUNCTION that was learned.

Attacks per model and threat setting (constrained / unconstrained), on one fixed random subset of the main grid's
fraud sample (sample.npz, n_fraud frauds, seed 13), the same for every variant and seed:
  white-box   XGBoost: AATS, one run at eps 0.5; a fraud counts as evaded at a smaller eps when its L-inf cost is
              within it. MLP: APGD, 100 iterations x 5 restarts, per eps.
  score       Square, 300 queries, per eps (same algorithm for every model)
  decision    HopSkipJump, minimal L-inf, one run, on the first n_hsj frauds (same algorithm for every model)
  random      non-adversarial sensitivity: the lowest score over 32 random corners of the eps = 0.25 ball
Budgets: EPS_A. Per fraud we store the clean logit, detection, the attacked logit per attack and eps, the L-inf cost
(AATS, HopSkipJump), and the attacked points (used by the attack views).

Outputs (runs/<dataset>/causal/)
  <variant>_seed<s>.pkl, train_<variant>_seed<s>.json   trained variant and its training info
  clean_<variant>_seed<s>.json                          clean metrics, distillation fidelity, training time
  out_<variant>_seed<s>_<mode>.npz                      per-fraud attack outcomes (resumable: existing files skipped)

Used by: causal/analyze.py and causal/report.py for Table VI, Tables S-XXXIV to S-XXXVI
(Table S-XXXIV, Table S-XXXV, Table S-XXXVI), Figures S13 and S14 (Figure S13,
Figure S14) and Section VI-I; report/attack_views.py reads the base outputs (out_{xgb,mlp}_seed0_<mode>.npz)
for Figure 2 and Figures S15 to S18.

usage: python -m fraudadv.causal.ablate <dataset> [--variants a,b] [--seeds 0,1,2] [--modes constrained,unconstrained]
                                        [--n-fraud 200] [--n-hsj 100]
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import torch.nn as nn

from fraudadv.attacks.blackwhite import APGD_BUDGET, aats, apgd, hsj, square
from fraudadv.attacks.constraints import Constraints
from fraudadv.attacks.run_attacks import RUNS, load_model
from fraudadv.data.load import load
from fraudadv.eval.metrics import all_metrics, threshold_at_fpr
from fraudadv.models.zoo import MLP, XGB, N_THREADS

EPS_A = [0.1, 0.25, 0.5]
N_FRAUD = 200       # frauds attacked by the white-box attack and Square (default of --n-fraud)
N_HSJ = 100         # of those, the first N_HSJ are attacked by HopSkipJump (default of --n-hsj)
SN_GAIN = 4.0
VARIANTS = {
    "xgb": ("xgboost", None), "xgb_d2": ("xgboost", {"max_depth": 2}), "xgb_d6": ("xgboost", {"max_depth": 6}),
    "xgb_d10": ("xgboost", {"max_depth": 10}), "xgb_mono": ("xgboost", "monotone"), "xgb_from_mlp": ("xgboost", "distil"),
    "mlp": ("mlp", None), "mlp_l1": ("mlp", {"n_layers": 1}), "mlp_l4": ("mlp", {"n_layers": 4}),
    "mlp_relu": ("mlp", {"activation": "relu"}), "mlp_sn": ("mlp", {"spectral_norm": SN_GAIN}),
    "mlp_wd": ("mlp", {"weight_decay": 0.1}), "mlp_from_xgb": ("mlp", "distil"),
}
TEACHER = {"xgb_from_mlp": "mlp", "mlp_from_xgb": "xgboost"}


# ------------------------------------------------------------------ model variants
# The class names below are part of the pickled models' paths: do not rename them.

class Gain(nn.Module):
    """Fixed output gain after the spectrally normalised layers."""

    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, x):
        return self.g * x


class AblMLP(MLP):
    """The zoo MLP with the ablation switches: activation, spectral normalisation, logit distillation."""

    family = "MLP (ablation)"

    def _build(self, d, p):
        from torch.nn.utils.parametrizations import spectral_norm

        act = {"silu": nn.SiLU, "relu": nn.ReLU}[p.get("activation", "silu")]
        sn = p.get("spectral_norm")
        lin = (lambda i, o: spectral_norm(nn.Linear(i, o))) if sn else nn.Linear
        layers, w = [], d
        for _ in range(p["n_layers"]):
            layers += [lin(w, p["width"]), act(), nn.Dropout(p["dropout"])]
            w = p["width"]
        layers.append(lin(w, 1))
        if sn:
            layers.append(Gain(float(sn)))
        return nn.Sequential(*layers)

    def _bake(self):
        """Replace the spectral-norm parametrisations by their current normalised weights, so the model pickles
        and its weights stay fixed at test time."""
        from torch.nn.utils import parametrize

        for mod in self.net.modules():
            if parametrize.is_parametrized(mod, "weight"):
                parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)

    def fit(self, Xtr, ytr, Xva, yva, wva, params, trial=None, seed=0):
        super().fit(Xtr, ytr, Xva, yva, wva, params, trial, seed)
        self._bake()
        return self

    def fit_distil(self, Xtr, ttr, Xva, tva, params, seed=0):
        """Regress the teacher's logits ttr (MSE, Ba & Caruana 2014), with early stopping on validation MSE.
        Twice the usual epochs (regression converges more slowly than BCE) and patience + 2."""
        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(N_THREADS)
        self.net = self._build(Xtr.shape[1], params)
        dev = self._dev()
        opt = torch.optim.AdamW(self.net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        Xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
        tt = torch.tensor(ttr, dtype=torch.float32, device=dev)
        bs = params.get("batch_size", 1024)
        best, best_state, bad = np.inf, None, 0
        for ep in range(params.get("epochs", 30) * 2):
            self.net.train()
            perm = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(Xt), bs):
                b = perm[i:i + bs]
                loss = torch.mean((self.net(Xt[b]).squeeze(-1) - tt[b]) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
            mse = float(np.mean((self.logit(Xva) - tva) ** 2))
            if mse < best - 1e-4:
                best, bad = mse, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= params.get("patience", 5) + 2:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self


class XGBDistil(XGB):
    """XGBoost regressor on a teacher's logits; its margin is the student's logit, so the tree attacks apply."""

    family = "XGBoost (distilled)"

    def fit_distil(self, Xtr, ttr, Xva, tva, params, seed=0):
        import xgboost as xgb

        p = {k: v for k, v in params.items() if k not in ("scale_pos_weight", "n_estimators")}
        self.model = xgb.XGBRegressor(n_estimators=params.get("n_estimators", 2000), tree_method="hist", n_jobs=N_THREADS,
                                      random_state=seed, objective="reg:squarederror", eval_metric="rmse",
                                      early_stopping_rounds=50, **p)
        self.model.fit(Xtr, ttr, eval_set=[(Xva, tva)], verbose=False)
        return self


# ------------------------------------------------------------------ training
def _params(dataset, fam):
    """The family's tuned hyperparameters."""
    return json.loads((RUNS / dataset / fam / "best_params.json").read_text())["params"]


def is_base(dataset, variant):
    """True when the variant's setting equals the tuned base (e.g. xgb_d10 when the tuned depth is 10)."""
    fam, change = VARIANTS[variant]
    if change is None:
        return True
    if isinstance(change, dict):
        p = _params(dataset, fam)
        return all(p.get(k, {"activation": "silu"}.get(k)) == v for k, v in change.items())
    return False


def build(dataset, variant, seed, a, cons_c):
    """Train one variant, or load it when it is a tuned base or already trained. Returns (model, info).
    cons_c: the constrained threat model (its mutable features are the ones xgb_mono constrains)."""
    fam, change = VARIANTS[variant]
    if is_base(dataset, variant):
        return load_model(dataset, fam, seed), {"trained": False, "same_as_base": variant != ("xgb" if fam == "xgboost" else "mlp")}
    out = RUNS / dataset / "causal"
    model_file = out / f"{variant}_seed{seed}.pkl"
    if model_file.exists():
        return pickle.loads(model_file.read_bytes()), json.loads((out / f"train_{variant}_seed{seed}.json").read_text())
    p = _params(dataset, fam)
    Xtr, ytr, Xva, yva, wva = a["X_train"], a["y_train"], a["X_val"], a["y_val"], a["w_val"]
    t0 = time.time()
    info = {"trained": True}
    if change == "distil":
        teacher = load_model(dataset, TEACHER[variant], seed)
        ttr, tva = np.clip(teacher.logit(Xtr), -20, 20), np.clip(teacher.logit(Xva), -20, 20)
        m = (XGBDistil() if fam == "xgboost" else AblMLP()).fit_distil(Xtr, ttr, Xva, tva, p, seed=seed)
        info["teacher"] = TEACHER[variant]
    elif change == "monotone":
        from fraudadv.defenses.run_defenses import monotone_signs

        s = monotone_signs(Xtr, ytr, cons_c.mutable)
        m = XGB().fit(Xtr, ytr, Xva, yva, wva, dict(p, monotone_constraints="(" + ",".join(str(v) for v in s) + ")"), seed=seed)
        info["monotone_signs"] = s.tolist()
    else:
        m = (XGB() if fam == "xgboost" else AblMLP()).fit(Xtr, ytr, Xva, yva, wva, dict(p, **change), seed=seed)
    info["train_seconds"] = time.time() - t0
    if fam == "xgboost":
        info["n_trees"] = int(m.model.best_iteration + 1)
    out.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(pickle.dumps(m))
    (out / f"train_{variant}_seed{seed}.json").write_text(json.dumps(info, indent=1))
    return m, info


# ------------------------------------------------------------------ evaluation
def fraud_subset(dataset, a, n_fraud=N_FRAUD):
    """Test positions of the attacked frauds: n_fraud of the main grid's fraud sample, in a fixed random order
    (seed 13). `a` is unused; kept for callers."""
    S = np.load(RUNS / dataset / "attacks" / "sample.npz")["S"]
    pick = np.random.default_rng(13).permutation(len(S))[:n_fraud]
    return S[pick]


def clean_metrics(dataset, variant, seed, m, a):
    """Clean test metrics at the 1%-FPR validation threshold, plus, for a distilled student, its fidelity to the
    teacher. Returns (record, threshold, sorted legit test logits)."""
    tau = threshold_at_fpr(a["y_val"], m.logit(a["X_val"]), a["w_val"], 0.01)
    st = m.logit(a["X_test"])
    rec = dict(dataset=dataset, variant=variant, seed=seed, tau=float(tau), **all_metrics(a["y_test"], st, a["w_test"], a["amt_test"], tau))
    if variant in TEACHER:
        from scipy.stats import spearmanr

        t = load_model(dataset, TEACHER[variant], seed)
        tt = t.logit(a["X_test"])
        t_tau = threshold_at_fpr(a["y_val"], t.logit(a["X_val"]), a["w_val"], 0.01)
        fr = a["y_test"] == 1
        ds, dt = st > tau, tt > t_tau                          # student / teacher flags
        rec.update(fid_spearman=float(spearmanr(st, tt).statistic), fid_pearson=float(np.corrcoef(st, tt)[0, 1]),
                   fid_agree=float(np.mean(ds == dt)), fid_agree_fraud=float(np.mean(ds[fr] == dt[fr])),
                   fid_jaccard_detected=float(np.sum(ds[fr] & dt[fr]) / max(np.sum(ds[fr] | dt[fr]), 1)),
                   teacher_recall=float(np.mean(dt[fr])))
    legit_sorted = np.sort(st[a["y_test"] == 0])
    return rec, tau, legit_sorted


def _white_box(m, fam, X0, l0, tau, cons, r):
    """XGBoost: one AATS run at the largest eps, success at a smaller eps judged by its cost. MLP: APGD per eps."""
    if fam == "xgboost":
        Xa, la, res = aats(m, X0, tau, max(EPS_A), cons)
        cost = np.where(res.success, res.cost, np.inf)
        r["wb_cost"], r["wb_X"] = cost.astype(np.float32), Xa.astype(np.float32)
        for e in EPS_A:
            r[f"wb_l_{e}"] = np.where(cost <= e + 1e-6, la, l0).astype(np.float32)
    else:
        for e in EPS_A:
            Xa, la = apgd(m, X0, tau, e, cons, *APGD_BUDGET["default"])
            r[f"wb_l_{e}"] = la.astype(np.float32)
            if e == 0.25:
                r["wb_X"] = Xa.astype(np.float32)
        r["wb_cost"] = np.full(len(X0), np.nan, np.float32)


def _square(m, X0, tau, cons, r):
    for e in EPS_A:
        Xs, ls = square(m, X0, tau, e, cons, n_queries=300)
        r[f"sq_l_{e}"] = ls.astype(np.float32)
        if e == 0.25:
            r["sq_X"] = Xs.astype(np.float32)


def _hopskipjump(m, X0, l0, tau, cons, n_hsj, r):
    """One minimal-L-inf HopSkipJump run on the first n_hsj frauds; evaded at eps when the found point evades and
    its L-inf distance is within eps."""
    H = np.arange(min(n_hsj, len(X0)))
    Xh, lh = hsj(m, X0[H], tau, cons)
    dist = np.where(lh < tau, cons.linf(Xh, X0[H]), np.inf)
    r["hsj_cost"], r["hsj_X"], r["hsj_rows"] = dist.astype(np.float32), Xh.astype(np.float32), H
    for e in EPS_A:
        r[f"hsj_l_{e}"] = np.where(dist <= e + 1e-6, lh, l0[H]).astype(np.float32)


def _random_corners(m, X0, cons, r, n_draws=32, eps=0.25):
    """Lowest logit over n_draws random corners of the eps ball in the mutable features (seed 0)."""
    rng = np.random.default_rng(0)
    lowest = np.full(len(X0), np.inf, np.float32)
    for _ in range(n_draws):
        D = np.zeros_like(X0)
        D[:, cons.mutable] = eps * rng.choice([-1.0, 1.0], size=(len(X0), len(cons.mutable)))
        lowest = np.minimum(lowest, m.logit(cons.project(X0 + D, X0, eps)))
    r["rand_l_0.25"] = lowest


def attack_all(m, fam, X0, tau, cons, n_hsj=N_HSJ):
    """Every attack on the frauds X0. Returns (arrays for out_*.npz, seconds per attack)."""
    r = {}
    l0 = m.logit(X0)
    t = {}
    t0 = time.time()
    _white_box(m, fam, X0, l0, tau, cons, r)
    t["wb"] = time.time() - t0
    t0 = time.time()
    _square(m, X0, tau, cons, r)
    t["sq"] = time.time() - t0
    t0 = time.time()
    _hopskipjump(m, X0, l0, tau, cons, n_hsj, r)
    t["hsj"] = time.time() - t0
    _random_corners(m, X0, cons, r)
    return r, t


def run(dataset, variants, seeds, modes, n_fraud=N_FRAUD, n_hsj=N_HSJ):
    """Train (or load), measure and attack every variant x seed x threat setting not yet done."""
    a, meta = load(dataset)
    out = RUNS / dataset / "causal"
    out.mkdir(parents=True, exist_ok=True)
    S = fraud_subset(dataset, a, n_fraud)
    X0 = a["X_test"][S].astype(np.float32)
    cons = {md: Constraints(meta, a["X_train"], md) for md in modes}
    cons_c = cons.get("constrained") or Constraints(meta, a["X_train"], "constrained")
    for v in variants:
        fam = VARIANTS[v][0]
        for s in seeds:
            if is_base(dataset, v) and v not in ("xgb", "mlp"):
                print(dataset, v, "equals the tuned base: reusing it", flush=True)
                continue
            todo = [md for md in modes if not (out / f"out_{v}_seed{s}_{md}.npz").exists()]
            if not todo:
                continue
            try:
                m, info = build(dataset, v, s, a, cons_c)
                rec, tau, legit_sorted = clean_metrics(dataset, v, s, m, a)
                rec.update({k: info[k] for k in ("train_seconds", "n_trees") if k in info})
                (out / f"clean_{v}_seed{s}.json").write_text(json.dumps(rec, indent=1))
                l0 = m.logit(X0)
                q0 = np.searchsorted(legit_sorted, l0) / len(legit_sorted)     # legit quantile of each fraud's clean score
                print(dataset, v, s, {k: round(rec[k], 4) for k in ("auc", "recall@1%fpr") if k in rec},
                      {k: round(rec[k], 3) for k in rec if k.startswith("fid_")}, flush=True)
                for md in todo:
                    t0 = time.time()
                    r, t = attack_all(m, fam, X0, tau, cons[md], n_hsj)
                    np.savez_compressed(out / f"out_{v}_seed{s}_{md}.npz", S=S, X0=X0, l0=l0.astype(np.float32), q0=q0, tau=tau,
                                        det=l0 > tau, amt=a["amt_test"][S].astype(np.float32), eps=np.array(EPS_A), **r)
                    asr = {e: round(float(np.mean((r[f"sq_l_{e}"] < tau)[l0 > tau])), 3) for e in EPS_A}
                    print(dataset, v, s, md, "Square ASR", asr, "times", {k: round(x) for k, x in t.items()},
                          "total", round(time.time() - t0), "s", flush=True)
            except Exception as e:                                  # one failed variant must not stop the rest
                import traceback

                traceback.print_exc()
                print("FAILED", dataset, v, s, repr(e), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--modes", default="constrained,unconstrained")
    ap.add_argument("--n-fraud", type=int, default=N_FRAUD, help="frauds attacked by the white-box attack and Square")
    ap.add_argument("--n-hsj", type=int, default=N_HSJ, help="of those, the first n are attacked by HopSkipJump")
    args = ap.parse_args()
    run(args.dataset, args.variants.split(","), [int(x) for x in args.seeds.split(",")], args.modes.split(","),
        args.n_fraud, args.n_hsj)
