"""The attack grid: every model of one dataset, under each threat model, attacked at five budgets.

usage: python -m fraudadv.attacks.run_attacks <dataset> [--families ...] [--modes ...] [--full-bb]
                                              [--milp-budget 3600] [--fm-full-n 240] [--tag _name]

What happens, for each threat model (unconstrained, constrained or consistent; fraudadv.attacks.constraints)
and each model:

  1. The model's alert threshold tau is set at a 1% (weighted) false-positive rate on the validation split.
  2. All models are attacked on one shared sample of test frauds (fraud_sample), so outcomes can be paired.
     Attack success rate (ASR) = share of the frauds the model detects that the attack makes it miss.
  3. White-box attack at every budget eps in EPS:
       tree ensembles   AATS (axis-aligned threshold search), then the exact MILP inside the AATS radius
       other models     APGD (100 iterations x 5 restarts; foundation models 20 x 2)
  4. Score-based black-box attack: Square, at every budget.
  5. Decision-based black-box attack: HopSkipJump (minimal distance; success at eps = evaded within eps).
  6. Worst case: per fraud, the best of the attacks above (an AutoAttack-style ensemble).
  7. Transfer examples: a high-confidence white-box evasion at eps 0.5 of every detected fraud, saved for
     fraudadv.attacks.transfer, and the decision-boundary thickness along it (fraudadv.boundary.thickness).

Outputs in runs/<dataset>/attacks/ (runs_consistent/<dataset>/attacks/ for the consistent attacker; tag appended to
each name):
  results.csv          one row per (model, threat model, attack, eps): ASR, dollar ASR, AUC and KS before and
                       after the attack, n detected; the source of almost every robustness number in the paper
  per_example.npz      per fraud: sample row, detected, evaded in the worst case, amount (paired tests)
  attack_scores.npz    the attacked score of every fraud for every row of results.csv (metrics can be
                       recomputed without attacking again)
  milp_status.csv      how many tree attacks the MILP solved exactly, certified, or timed out on
  adv_<mode>_<model>.npz, transfer_<mode>.npz, thickness.csv, tau.json, sample.npz

The paper's runs used --full-bb (black-box attacks on the whole fraud sample) and --milp-budget 3600; the
foundation models ran on a random 240 frauds of the sample (--fm-full-n 240), all of it on Sparkov.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraudadv.attacks.blackwhite import APGD_BUDGET, aats, apgd, hsj, pgd, square
from fraudadv.attacks.constraints import Constraints
from fraudadv.attacks.trees import TreeEnsemble
from fraudadv.boundary.thickness import thickness_exact_trees, thickness_sampled
from fraudadv.data.load import load
from fraudadv.eval.metrics import auc, ks, threshold_at_fpr
from fraudadv.hpo.train_all import load_model as _load_tabular
from fraudadv.paths import RUNS, attack_dir  # noqa: F401  (RUNS: imported from here by other modules)

EPS = [0.05, 0.1, 0.25, 0.5, 1.0]     # L-inf budgets, in standard deviations of each feature
EPS_T = 0.5                           # budget of the transfer examples
KAPPA = 2.0                           # transfer examples push the logit KAPPA below the threshold
MAX_SAMPLE = 1000                     # largest shared fraud sample
MILP_TIME_LIMIT = 10.0                # seconds per fraud for the exact tree attack (HiGHS); on timeout the best
                                      # solution found so far, if any, is used

TREES = {"xgboost", "catboost"}
# LoRA-adapted and zero-shot foundation models (fraudadv.models.foundation_adapt) are treated as foundation
# models everywhere: same evaluation subsets, budgets and attacks as the frozen ones
ADAPT_FAMS = {"chronos2_lora", "timesfm_lora", "chronos2_zs", "timesfm_zs"}
FM = {"chronos2", "timesfm"} | ADAPT_FAMS
CONTEXT_FAMS = {"hgnn", "hgt"} | FM   # models that read the transaction's context (graph or history series)

# CPU budget per family: (white-box examples, PGD steps of the transfer attack, black-box examples,
# Square queries, run HopSkipJump, thickness steps); None = the whole sample
BUDGET = {"default": (None, 40, None, 300, True, 128), "fm": (60, 20, 30, 100, False, 24)}

ATTACK_PGD = "white-box (PGD)"        # result label of the APGD rows (kept: the report code reads it)
ATTACK_AATS = "white-box (AATS)"
ATTACK_MILP = "white-box (AATS + exact MILP)"
ATTACK_SQUARE = "score black-box (Square)"
ATTACK_HSJ = "decision black-box (HopSkipJump)"
ATTACK_WORST = "worst case (all attacks)"


def load_model(dataset, fam, seed=0):
    """A trained model with .logit(X, ctx) (and .grad for differentiable ones)."""
    if fam in CONTEXT_FAMS:
        from fraudadv.hpo.train_context import load_context_model

        return load_context_model(dataset, fam, seed)
    return _load_tabular(dataset, fam, seed)


def context_ids(dataset):
    """Per split, each row's index into the context matrix that graph and sequence models read (ctx.npz);
    plain row numbers for datasets without context."""
    from fraudadv.data.load import PROC

    p = PROC / dataset / "ctx.npz"
    if p.exists():
        z = np.load(p, mmap_mode="r")
        return {k: np.asarray(z[f"keep_{k}"]) for k in ("train", "val", "test")}
    a, _ = load(dataset)
    return {k: np.arange(len(a[f"y_{k}"])) for k in ("train", "val", "test")}


def milp_attack(m, fam, X0, det, tau, cons, d_aats, l_aats, eps_max=None, shuffle=False, budget_s=1200.0):
    """Exact minimal L-inf evasion of each detected fraud with the MILP of Kantchelian et al. (2016), extended to
    discrete domains and derived features (fraudadv.attacks.milp). The search is restricted to the AATS radius
    r = min(AATS distance, largest budget): any cheaper evasion lies inside r, so nothing is lost, and the
    restriction makes hard cases tractable. Outcome per fraud:
      exact      the MILP found the optimum (distance <= AATS)
      certified  no evasion inside r: the AATS evasion is optimal, or the fraud is robust at the largest budget
      timeout    time limit without proof: keep the better of AATS and the MILP's best solution
    Every MILP solution is projected onto the threat model and re-scored on the real model.
    shuffle: visit the frauds in a fixed random order, so that if budget_s (seconds per model and threat model)
    runs out, the frauds left with their AATS result are a random subset, not the last rows of the sample.
    Returns (distance, logit at the evasion, status per fraud; '' = not attacked)."""
    from fraudadv.attacks.milp import KantchelianMILP

    eps_max = max(EPS) if eps_max is None else eps_max
    M = KantchelianMILP(m, fam, X0.shape[1], cons.lower, cons.upper, cons.mutable, time_limit=MILP_TIME_LIMIT,
                        allowed=cons.allowed, cons=cons)
    d, lg = d_aats.astype(float).copy(), l_aats.astype(float).copy()
    status = np.array([""] * len(X0), dtype=object)
    t0 = time.time()
    order = np.flatnonzero(det)
    if shuffle:
        order = np.random.default_rng(0).permutation(order)
    for i in order:
        if time.time() - t0 > budget_s:
            break
        r = min(float(d_aats[i]), eps_max) + 1e-4
        x, b, lb, st = M.solve_one(X0[i], tau, eps_max=r)
        status[i] = "certified" if st == 2 else ("exact" if st == 0 else "timeout")
        if x is None:
            continue
        xp = cons.project(x[None], X0[i])
        l = float(m.logit(xp)[0])
        dd = float(cons.linf(xp, X0[i][None])[0])
        if l < tau and dd < d[i]:
            d[i], lg[i] = dd, l
        elif st == 0 and not (l < tau):
            status[i] = "timeout"                       # the optimum does not survive the projection: no proof
    n = {k: int(np.sum(status == k)) for k in ("exact", "certified", "timeout")}
    print("  MILP", fam, cons.mode, n, "of", int(det.sum()), "detected in", round(time.time() - t0), "s", flush=True)
    return d, lg, status


def fraud_sample(dataset, a, models, tau, keep, n_wb, n_bb, out, reuse=False):
    """The shared fraud sample S: the same test frauds for every model. ASR is measured on the frauds a model
    detects, so S is sized for about n_wb detected frauds on the model with the lowest recall (foundation models
    excluded): |S| = min(#test frauds, n_wb / recall, MAX_SAMPLE). Black-box attacks without --full-bb use the
    first |S| * n_bb / n_wb rows. Saved to sample.npz; a run with a tag reuses the saved sample, so all runs of
    a dataset attack the same frauds."""
    p = out / "sample.npz"
    if reuse and p.exists():
        z = np.load(p)
        return z["S"], int(z["n_bb"])
    fr = np.flatnonzero(a["y_test"] == 1)
    rec = [float(np.mean(m.logit(a["X_test"][fr], keep["test"][fr]) > tau[f])) for f, m in models.items() if f not in FM]
    r = max(min(rec), 0.05) if rec else 1.0
    n_s = int(min(len(fr), np.ceil(n_wb / r), MAX_SAMPLE))
    S = np.sort(np.random.default_rng(0).choice(fr, size=n_s, replace=False))
    n_b = int(np.ceil(n_s * n_bb / n_wb))
    np.savez(p, S=S, n_bb=n_b, min_recall=r)
    print(dataset, "fraud sample", n_s, "of", len(fr), "(lowest recall at 1% FPR", round(r, 3), "); black-box on", n_b, flush=True)
    return S, n_b


@dataclass
class Grid:
    """Everything the attacks share for one dataset: data, models, thresholds and the fraud sample."""
    dataset: str
    a: dict                  # data splits (fraudadv.data.load)
    meta: dict               # feature names, scaling, who controls which feature
    models: dict             # family -> model
    tau: dict                # family -> alert threshold (logit) at 1% FPR on validation
    S: np.ndarray            # test-row indices of the shared fraud sample
    n_bb: int                # black-box sample size without --full-bb
    X0: np.ndarray           # the sample's feature rows
    amt: np.ndarray          # the sample's amounts (for dollar-weighted metrics)
    ctx0: np.ndarray         # the sample's context ids
    legit: dict              # family -> (test legit logits, unit weights, amounts): the negatives of AUC and KS
    l0: dict                 # family -> clean logits of the sample
    det: dict                # family -> detected (clean logit above tau)


def setup(dataset, families, seed, n_wb, n_bb, sample_dir, reuse_sample):
    a, meta = load(dataset)
    models = {f: load_model(dataset, f, seed) for f in families}
    keep = context_ids(dataset)
    # foundation models are scored on fixed evaluation subsets (all frauds and sampled legits, weights rescaled)
    from fraudadv.hpo.train_context import fm_subset

    sub = {"val": fm_subset(a, "val"), "test": fm_subset(a, "test")}
    tau = {}
    for f, m in models.items():
        if f in FM:
            iv, wv = sub["val"]
            tau[f] = threshold_at_fpr(a["y_val"][iv], m.logit(a["X_val"][iv], keep["val"][iv]), wv, 0.01)
        else:
            tau[f] = threshold_at_fpr(a["y_val"], m.logit(a["X_val"], keep["val"]), a["w_val"], 0.01)
    S, n_bb = fraud_sample(dataset, a, models, tau, keep, n_wb, n_bb, sample_dir, reuse=reuse_sample)
    X0, amt = a["X_test"][S], a["amt_test"][S].astype(float)
    ctx0 = keep["test"][S]
    legit_all = np.flatnonzero(a["y_test"] == 0)
    it, wt = sub["test"]
    w_fm = np.zeros(len(a["y_test"]))
    w_fm[it] = wt
    legit_fm = it[a["y_test"][it] == 0]
    legit = {}
    for f, m in models.items():
        lg = legit_fm if f in FM else legit_all
        w = w_fm[lg] if f in FM else a["w_test"][lg]
        legit[f] = (m.logit(a["X_test"][lg], keep["test"][lg]), w, a["amt_test"][lg].astype(float))
    l0 = {f: m.logit(X0, ctx0) for f, m in models.items()}
    det = {f: l0[f] > tau[f] for f in models}
    return Grid(dataset, a, meta, models, tau, S, n_bb, X0, amt, ctx0, legit, l0, det)


class Recorder:
    """Collects one results row (and the attacked scores behind it) per (model, threat model, attack, eps)."""

    def __init__(self, g: Grid):
        self.g = g
        self.rows = []
        self.att = {}        # "<family>|<mode>|<attack>|<eps>" -> attacked scores; "...|rows" -> their rows of S

    def add(self, fam, mode, attack, eps, l_adv, idx=None, extra=None):
        """l_adv: the attacked logit of each fraud in S[idx] (its clean logit where the attack failed). Metrics:
        ASR (and dollar ASR) among the detected frauds, and AUC / KS (unit and dollar) of attacked frauds against
        all test legits, before and after the attack."""
        g = self.g
        idx = np.arange(len(g.S)) if idx is None else idx
        ll, wl, al = g.legit[fam]
        y = np.r_[np.ones(len(idx)), np.zeros(len(ll))]
        succ = l_adv < g.tau[fam]
        d = g.det[fam][idx]
        ev = d & succ
        lf = np.where(succ, l_adv, g.l0[fam][idx])
        s = np.r_[lf, ll]
        w = np.r_[np.ones(len(idx)), wl]
        wd = np.r_[g.amt[idx], wl * al]
        s0 = np.r_[g.l0[fam][idx], ll]
        row = dict(dataset=g.dataset, model=fam, mode=mode, attack=attack, eps=eps, n=len(idx), n_detected=int(d.sum()),
                   asr=float(ev.sum() / max(d.sum(), 1)),
                   asr_usd=float(g.amt[idx][ev].sum() / max(g.amt[idx][d].sum(), 1e-9)),
                   clean_auc=auc(y, s0, w), clean_ks=ks(y, s0, w), clean_auc_usd=auc(y, s0, wd), clean_ks_usd=ks(y, s0, wd),
                   robust_auc=auc(y, s, w), robust_ks=ks(y, s, w), robust_auc_usd=auc(y, s, wd), robust_ks_usd=ks(y, s, wd))
        row.update(extra or {})
        self.rows.append(row)
        self.att[f"{fam}|{mode}|{attack}|{eps}"] = lf.astype(np.float32)
        self.att[f"{fam}|{mode}|{attack}|rows"] = np.asarray(idx, np.int32)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
               if k in ("model", "mode", "attack", "eps", "asr", "asr_usd", "robust_auc", "robust_ks")}, flush=True)


class CellCheckpoint:
    """Saves each finished (attack, eps) cell of one model, so that a run cut off mid-model (a 2 h job limit)
    resumes from the last finished cell. Used only if the threshold and fraud sample are unchanged."""

    def __init__(self, path, tau, S, rec: Recorder, fam, mode):
        self.path, self.tau, self.S, self.rec, self.fam, self.mode = path, float(tau), S, rec, fam, mode
        self.done = {}
        if path.exists():
            c = pickle.loads(path.read_bytes())
            if c["tau"] == self.tau and np.array_equal(c["S"], S):
                self.done = c["done"]
                print(rec.g.dataset, mode, fam, "resuming", len(self.done), "finished cells", flush=True)

    def cell(self, attack, eps, compute):
        """Restore the cell, or run compute() (which records one row and returns the per-example scores)."""
        k = (attack, float(eps))
        if k in self.done:
            row, att, pe = self.done[k]
            self.rec.rows.append(row)
            self.rec.att.update(att)
            return pe
        pe = compute()
        keys = (f"{self.fam}|{self.mode}|{attack}|{eps}", f"{self.fam}|{self.mode}|{attack}|rows")
        self.done[k] = (self.rec.rows[-1], {q: self.rec.att[q] for q in keys}, pe)
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(pickle.dumps(dict(tau=self.tau, S=self.S, done=self.done)))
        os.replace(tmp, self.path)
        return pe


def attack_model(g: Grid, rec: Recorder, fam, mode, cons, out, cells: CellCheckpoint, full_bb, grid_only, fm_full_n,
                 milp_budget):
    """Every attack on one model under one threat model. Returns (transfer examples or None, thickness rows,
    MILP status rows, per-example worst-case outcomes)."""
    m, tau, X0, ctx0, l0, det, S = g.models[fam], g.tau[fam], g.X0, g.ctx0, g.l0[fam], g.det[fam], g.S
    per_ex = defaultdict(list)            # eps -> attacked logits of the black-box rows, one array per attack
    thick_rows, milp_rows, per_example = [], [], {}
    nwb, steps, nbb_f, nq, use_hsj, th_steps = BUDGET["fm" if fam in FM else "default"]
    W = np.arange(len(S) if nwb is None else min(nwb, len(S)))         # white-box rows of S
    B = np.arange(min(g.n_bb if nbb_f is None else nbb_f, len(W)))     # black-box rows of S
    if full_bb and nbb_f is None:
        B = W
    elif full_bb and fm_full_n > 0:
        # foundation models: every attack on one fixed random subset of the sample (all of it if fm_full_n >= |S|)
        # instead of its first rows, which are in entity order and cover only a dozen cards
        W = np.sort(np.random.default_rng(1).permutation(len(S))[:fm_full_n])
        B = W
    n_bbf = len(B)

    # ---- white-box
    if fam in TREES:
        Xa, la, res = aats(m, X0, tau, max(EPS), cons)
        dist = np.where(res.success, res.cost, np.inf)
        dm, lm, mst = milp_attack(m, fam, X0, det, tau, cons, dist, la, shuffle=full_bb, budget_s=milp_budget)
        milp_rows.append(dict(dataset=g.dataset, model=fam, mode=mode, detected=int(det.sum()),
                              **{k: int(np.sum(mst == k)) for k in ("exact", "certified", "timeout")},
                              not_attacked=int(np.sum(det & (mst == "")))))
        for e in EPS:
            ok = dist <= e + 1e-6
            rec.add(fam, mode, ATTACK_AATS, e, np.where(ok, la, l0),
                    extra={"median_min_linf": float(np.median(dist[det])) if det.any() else None})
            okm = dm <= e + 1e-6
            rec.add(fam, mode, ATTACK_MILP, e, np.where(okm, lm, l0),
                    extra={"median_min_linf": float(np.median(dm[det])) if det.any() else None})
            per_ex[e].append(np.where(okm, lm, l0)[:n_bbf])
        if not grid_only:
            Xt, lt, _ = aats(m, X0, tau, EPS_T, cons, kappa=KAPPA)
    else:
        for e in EPS:
            def white_box(e=e):
                n_iter, restarts = APGD_BUDGET["fm" if fam in FM else "default"]
                _, la = apgd(m, X0[W], tau, e, cons, n_iter=n_iter, restarts=restarts, ctx=ctx0[W])
                rec.add(fam, mode, ATTACK_PGD, e, la, idx=W)
                return la[:n_bbf]
            per_ex[e].append(cells.cell(ATTACK_PGD, e, white_box))
        if not grid_only:
            Xt, lt = pgd(m, X0[W], tau, EPS_T, cons, steps=int(steps * 1.5), kappa=KAPPA, ctx=ctx0[W])
            if len(W) < len(S):                     # frauds not attacked keep their clean point
                Xf, lf = X0.copy(), np.full(len(S), np.inf)
                Xf[W], lf[W] = Xt, lt
                Xt, lt = Xf, lf

    # ---- transfer examples and boundary thickness along them
    if grid_only:
        transfer = None
        okt = np.zeros(len(S), bool)
    else:
        transfer = (Xt, lt < tau)
        np.savez_compressed(out / f"adv_{mode}_{fam}.npz", S=S, X0=X0, Xadv=Xt, ok=lt < tau, det=det, tau=tau)
        okt = (lt < tau) & det
    if okt.any():
        # threshold-centred thickness (Yang et al. 2020): g = 2 sigmoid(logit - tau) - 1 on the segment from each
        # detected fraud to its high-confidence adversarial example
        if fam in TREES:
            ens = m.tree_ensemble()
            shifted = TreeEnsemble(ens.thresholds, lambda Z, f=ens.margin_fn, t=tau: f(Z) - t, ens.n_features, ens.rule)
            th = thickness_exact_trees(shifted, X0[okt], Xt[okt])
        else:
            th = thickness_sampled(lambda Z, c: 1 / (1 + np.exp(-(m.logit(Z, c) - tau))), X0[okt], Xt[okt],
                                   n_steps=th_steps, ctx=ctx0[okt])
        seg = np.linalg.norm((X0[okt] - Xt[okt]), axis=1)
        thick_rows.append(dict(dataset=g.dataset, model=fam, mode=mode, n=int(okt.sum()),
                               thickness_mean=float(th.mean()), thickness_median=float(np.median(th)),
                               thickness_rel=float(np.mean(th / np.maximum(seg, 1e-9))), seg_l2=float(seg.mean())))
        print("thickness", thick_rows[-1], flush=True)

    # ---- score-based black-box
    for e in EPS:
        def score_based(e=e):
            _, ls = square(m, X0[B], tau, e, cons, n_queries=nq, ctx=ctx0[B])
            rec.add(fam, mode, ATTACK_SQUARE, e, ls, idx=B)
            return ls
        per_ex[e].append(cells.cell(ATTACK_SQUARE, e, score_based))

    # ---- decision-based black-box (minimal distance, then success at each budget)
    if use_hsj:
        Xh, lh = hsj(m, X0[B], tau, cons, ctx=ctx0[B])
    else:
        Xh, lh = X0[B].copy(), l0[B].copy()
    dh = cons.linf(Xh, X0[B])
    okh = lh < tau
    for e in (EPS if use_hsj else []):
        ok = okh & (dh <= e + 1e-6)
        rec.add(fam, mode, ATTACK_HSJ, e, np.where(ok, lh, l0[B]), idx=B,
                extra={"median_min_linf": float(np.median(np.where(okh, dh, np.inf)[det[B]])) if det[B].any() else None})
        per_ex[e].append(np.where(ok, lh, l0[B]))

    # ---- worst case over all attacks, per fraud
    for e in EPS:
        worst = np.min(np.stack(per_ex[e]), 0)
        rec.add(fam, mode, ATTACK_WORST, e, worst, idx=B)
        per_example[f"{fam}|{mode}|{e}"] = np.stack([S[B], det[B], worst < tau, g.amt[B]], 1)
    return transfer, thick_rows, milp_rows, per_example


def transfer_matrix(g: Grid, transfer, mode):
    """T[i, j]: share of source i's high-confidence evasions (of frauds both i and j detect) that also evade j;
    N[i, j]: how many such frauds. The paper's matrices come from fraudadv.attacks.transfer, which does the same
    for models attacked in separate runs."""
    fams = list(g.models)
    T = np.full((len(fams), len(fams)), np.nan)
    N = np.zeros_like(T)
    for i, s in enumerate(fams):
        Xs, ok_s = transfer[(mode, s)]
        for j, t in enumerate(fams):
            base = ok_s & g.det[s] & g.det[t]
            N[i, j] = base.sum()
            if base.any():
                T[i, j] = float(np.mean(g.models[t].logit(Xs[base], g.ctx0[base]) < g.tau[t]))
    return T, N, fams


def run(dataset, families, n_wb=200, n_bb=100, seed=0, modes=("unconstrained", "constrained"), tag="", full_bb=False,
        grid_only=False, milp_budget=1200.0, fm_full_n=0, runs_dir=None):
    """full_bb: black-box attacks, and so the worst case, on the whole fraud sample instead of its first rows (the
    test rows are ordered by entity, so that first part is not a random subset). grid_only: write only the attack
    grid (results, per-example outcomes, attacked scores, MILP status), leaving the transfer examples, transfer
    matrices and thickness of an earlier run untouched. milp_budget: seconds of exact tree attack per model and
    threat model. fm_full_n: with full_bb, foundation models are attacked on this many random frauds of the sample.
    runs_dir: read the fraud sample from and write every output to runs_dir/<dataset>/attacks/ instead of the
    study's folders (the ivis variants use runs_ivisvar/; there a consistent run's files carry their own tag).
    Each (threat model, model) is checkpointed when it finishes, so an interrupted run resumes after it."""
    if any(f in FM for f in families):
        import chronos  # noqa: F401  must be imported before TensorFlow (ivis): the reverse order aborts
    consistent = "consistent" in modes
    if consistent and len(modes) > 1:
        raise ValueError("run the consistent attacker on its own: its outputs go to runs_consistent/")
    if runs_dir is None:
        out, sample_dir = attack_dir(dataset, consistent), attack_dir(dataset)
    else:
        out = sample_dir = Path(runs_dir) / dataset / "attacks"
    out.mkdir(parents=True, exist_ok=True)
    g = setup(dataset, families, seed, n_wb, n_bb, sample_dir, reuse_sample=bool(tag) or consistent)
    rec = Recorder(g)
    transfer, thick_rows, milp_rows, per_example = {}, [], [], {}
    ck = out / f"ckpt{tag}"
    ck.mkdir(exist_ok=True)
    for mode in modes:
        cons = Constraints(g.meta, g.a["X_train"], mode)
        for fam in g.models:
            cf = ck / f"{mode}_{fam}.pkl"
            if cf.exists():
                c = pickle.loads(cf.read_bytes())
                if c["tau"] == float(g.tau[fam]) and np.array_equal(c["S"], g.S):
                    rec.att.update(c.get("att", {}))
                    rec.rows += c["rows"]
                    thick_rows += c["thick"]
                    milp_rows += c["milp"]
                    per_example.update(c["per_example"])
                    transfer[(mode, fam)] = c["transfer"]
                    print(dataset, mode, fam, "resumed from checkpoint", flush=True)
                    continue
            n_rows = len(rec.rows)
            t0 = time.time()
            cells = CellCheckpoint(ck / f"{mode}_{fam}.cells.pkl", g.tau[fam], g.S, rec, fam, mode)
            tr, th, mi, pe = attack_model(g, rec, fam, mode, cons, out, cells, full_bb, grid_only, fm_full_n, milp_budget)
            transfer[(mode, fam)] = tr
            thick_rows += th
            milp_rows += mi
            per_example.update(pe)
            print(dataset, mode, fam, "done in", round(time.time() - t0), "s", flush=True)
            cf.write_bytes(pickle.dumps(dict(
                tau=float(g.tau[fam]), S=g.S, rows=rec.rows[n_rows:], thick=th, milp=mi, per_example=pe,
                att={k: v for k, v in rec.att.items() if k.startswith(f"{fam}|{mode}|")}, transfer=tr)))
            cells.path.unlink(missing_ok=True)
            pd.DataFrame(rec.rows).to_csv(out / f"partial{tag}_results.csv", index=False)
            if not grid_only:
                pd.DataFrame(thick_rows).to_csv(out / f"partial{tag}_thickness.csv", index=False)
        if grid_only:
            continue
        T, N, fams = transfer_matrix(g, transfer, mode)
        np.savez(out / f"transfer_{mode}{tag}.npz", T=T, N=N, families=np.array(fams))
        print(mode, "transfer\n", np.round(T, 2), flush=True)

    np.savez_compressed(out / f"per_example{tag}.npz", **per_example)
    pd.DataFrame(rec.rows).to_csv(out / f"results{tag}.csv", index=False)
    np.savez_compressed(out / f"attack_scores{tag}.npz", S=g.S, **{f"{f}|clean": g.l0[f].astype(np.float32) for f in g.models},
                        **{f"{f}|tau": np.float32(g.tau[f]) for f in g.models}, **rec.att)
    if not grid_only:
        pd.DataFrame(thick_rows).to_csv(out / f"thickness{tag}.csv", index=False)
    if milp_rows:
        pd.DataFrame(milp_rows).to_csv(out / f"milp_status{tag}.csv", index=False)
    if not grid_only:
        tp = out / "tau.json"
        old = json.loads(tp.read_text()) if tp.exists() else {}
        old.update({k: float(v) for k, v in g.tau.items()})
        tmp = tp.with_suffix(f".{os.getpid()}.tmp")       # write, then rename: other runs may read the file
        tmp.write_text(json.dumps(old))
        os.replace(tmp, tp)
    shutil.rmtree(ck, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--families", default="xgboost,catboost,mlp,ft_transformer,ivis")
    ap.add_argument("--n-wb", type=int, default=200)
    ap.add_argument("--n-bb", type=int, default=100)
    ap.add_argument("--tag", default="", help="suffix for output files when families are attacked in separate runs")
    ap.add_argument("--modes", default="unconstrained,constrained", help="any of unconstrained, constrained, consistent")
    ap.add_argument("--full-bb", action="store_true", help="black-box attacks and worst case on the whole fraud sample")
    ap.add_argument("--grid-only", action="store_true", help="do not rewrite transfer examples, transfer matrices or thickness")
    ap.add_argument("--milp-budget", type=float, default=1200.0, help="seconds of exact tree attack per model and threat model")
    ap.add_argument("--fm-full-n", type=int, default=0, help="with --full-bb: random frauds per foundation model (0: first rows)")
    args = ap.parse_args()
    run(args.dataset, args.families.split(","), args.n_wb, args.n_bb, tag=args.tag, modes=tuple(args.modes.split(",")),
        full_bb=args.full_bb, grid_only=args.grid_only, milp_budget=args.milp_budget, fm_full_n=args.fm_full_n)
