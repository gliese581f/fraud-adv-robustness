"""Why is ivis not more robust? Diagnostic probes per dataset and threat model (seed-0 models, the study's fraud
sample). Prints one JSON line per dataset; no file is written and no paper table reads it (it informed the ivis
variants experiment, scripts/analysis/ivis_variants.py).

1. budget       on the frauds that MLP, FT-Transformer and ivis all detect: the share each model lets through at
                every grid budget (eps at which the grid's worst case first evades)
2. first order  for those frauds, per model: median logit margin (logit - tau), median L1 norm of the logit
                gradient on the attacker-controllable features, and the first-order L-inf margin (their ratio)
3. embedding    how ivis's eps-0.25 white-box (APGD) evasions move in its embedding space:
                - alignment of the movement with the logistic head's weight vector
                - fraud share among the 50 nearest training embeddings before / after the attack
                - a k-NN head on the same embedding (threshold at 1% FPR on test legits): clean recall, and the
                  share of the evasions it still flags

Inputs (read relative to the working directory, which must be the repository root):
  runs/<d>/attacks/tau.json, runs/<d>/attacks/per_example_full_{neural_graph,all,neural}.npz, runs/<d>/ivis/best_params.json

usage: python scripts/analysis/ivis_probe.py <dataset> [<dataset> ...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from fraudadv.data.load import load  # noqa: E402
from fraudadv.hpo.train_all import load_model  # noqa: E402
from fraudadv.attacks.constraints import Constraints  # noqa: E402
from fraudadv.attacks.blackwhite import apgd  # noqa: E402

EPS = [0.05, 0.1, 0.25, 0.5, 1.0]
NN = ["mlp", "ft_transformer", "ivis"]
MODES = ("constrained", "unconstrained")
rng = np.random.default_rng(0)          # one generator for the whole run (shared across datasets, in order)


def knn(Eref, yref, Q, k=50, cosine=False):
    """Fraud share among the k nearest reference embeddings of each query (cosine: on l2-normalised vectors)."""
    from sklearn.neighbors import NearestNeighbors
    if cosine:
        Eref = Eref / np.linalg.norm(Eref, axis=1, keepdims=True)
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    nn = NearestNeighbors(n_neighbors=k).fit(Eref)
    idx = nn.kneighbors(Q, return_distance=False)
    return yref[idx].mean(1)


def load_per_example(d):
    """The grid's per-example outcomes of the neural models, "<model>|<mode>|<eps>" -> [test row, detected, evaded, amount]."""
    pe = {}
    for f in ("neural_graph", "all", "neural"):
        q = f"runs/{d}/attacks/per_example_full_{f}.npz"
        if os.path.exists(q):
            z_ = np.load(q)
            pe.update({k: z_[k] for k in z_.files})
    return pe


def detected_by_all(pe, mode):
    """Frauds of the sample that all three neural models detect (boolean mask over the sample)."""
    det = {m: pe[f"{m}|{mode}|0.25"][:, 1].astype(bool) for m in NN}
    return det["mlp"] & det["ft_transformer"] & det["ivis"]


def budget_probe(pe, mode):
    """Probe 1: share of the commonly detected frauds evaded at each budget, per model."""
    both = detected_by_all(pe, mode)
    S = pe[f"ivis|{mode}|0.25"][:, 0].astype(int)
    r = {"n_all_detect": int(both.sum())}
    for m in NN:
        first = np.full(len(S), np.inf)               # smallest budget at which the worst case evades
        for e in EPS[::-1]:
            ev = pe[f"{m}|{mode}|{e}"][:, 2].astype(bool)
            first[ev] = e
        f = first[both]
        r[m] = {"evaded@" + str(e): round(float((f <= e).mean()), 3) for e in EPS}
    return r


def first_order_probe(models, X0, tau, cons):
    """Probe 2: first-order margins of the commonly detected (constrained) frauds X0 under one threat model."""
    mask = cons.mask
    mm = {}
    for m in NN:
        z = models[m].logit(X0)
        g = models[m].grad(X0)
        l1 = np.abs(g[:, mask]).sum(1)
        mm[m] = {"median_logit_margin": round(float(np.median(z - tau[m])), 3),
                 "median_grad_l1": round(float(np.median(l1)), 3),
                 "median_first_order_linf": round(float(np.median((z - tau[m]) / np.maximum(l1, 1e-12))), 4)}
    return mm


def embedding_movement(iv, enc, Eref, yref, w, thr, cosine, X0a, Xa, ok):
    """Probe 3, per threat model: how the successful APGD evasions moved in the embedding."""
    E0, E1 = enc(X0a[ok]), enc(Xa[ok])
    dE = E1 - E0
    along = np.abs(dE @ w) / (np.linalg.norm(dE, axis=1) * np.linalg.norm(w) + 1e-12)
    cf, cl = Eref[yref == 1].mean(0), Eref[yref == 0].mean(0)            # fraud and legit centroids
    k0, k1 = knn(Eref, yref, E0, cosine=cosine), knn(Eref, yref, E1, cosine=cosine)
    zl0, zl1 = iv.logit(X0a[ok]), iv.logit(Xa[ok])
    return {
        "n_evaded": int(ok.sum()),
        "median_cos_dE_w": round(float(np.median(along)), 3),
        "median_|dE|/|centroid gap|": round(float(np.median(np.linalg.norm(dE, axis=1)) / np.linalg.norm(cf - cl)), 3),
        "median_logit_drop": round(float(np.median(zl0 - zl1)), 3),
        "knn_fraud_share_before": round(float(np.median(k0)), 3), "knn_fraud_share_after": round(float(np.median(k1)), 3),
        "knn_head_flags_before": round(float((k0 > thr).mean()), 3), "knn_head_flags_after": round(float((k1 > thr).mean()), 3),
        "ref_fraud_frac": round(float(yref.mean()), 3)}


def main(d):
    """All three probes on dataset d; returns the result dict."""
    a, meta = load(d)
    tau = json.load(open(f"runs/{d}/attacks/tau.json"))
    pe = load_per_example(d)
    out = {"dataset": d}
    for mode in MODES:
        out[mode] = budget_probe(pe, mode)

    # 2. first-order margins on the frauds all three models detect in the constrained grid, both threat models
    Xte = a["X_test"].astype(np.float32)
    S = pe["ivis|constrained|0.25"][:, 0].astype(int)
    X0 = Xte[S[detected_by_all(pe, "constrained")]]
    models = {m: load_model(d, m, 0) for m in NN}
    for mode in MODES:
        out[mode]["first_order"] = first_order_probe(models, X0, tau, Constraints(meta, a["X_train"], mode))

    # 3. ivis embedding: reference set of up to 5,000 training frauds and 45,000 training legits
    iv = models["ivis"]
    P = json.load(open(f"runs/{d}/ivis/best_params.json"))["params"]
    cosine = "cosine" in P.get("distance", "pn")
    ytr = a["y_train"]
    pos, neg = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
    pos = rng.choice(pos, min(len(pos), 5000), replace=False)
    neg = rng.choice(neg, 45000, replace=False)
    ref = np.r_[pos, neg]
    enc = lambda X: np.concatenate([iv.encoder(X[i:i + 8192].astype(np.float32), training=False).numpy() for i in range(0, len(X), 8192)])  # noqa: E731
    Eref, yref = enc(a["X_train"][ref]), ytr[ref]
    w = iv.head.coef_[0]
    # k-NN head: threshold at the 99th percentile of up to 20,000 test legits' fraud share
    yte = a["y_test"]
    leg = rng.choice(np.flatnonzero(yte == 0), min(20000, int((yte == 0).sum())), replace=False)
    s_leg = knn(Eref, yref, enc(Xte[leg]), cosine=cosine)
    thr = np.quantile(s_leg, 0.99)
    frauds_all = np.flatnonzero(yte == 1)
    s_fr = knn(Eref, yref, enc(Xte[frauds_all]), cosine=cosine)
    z_fr = iv.logit(Xte[frauds_all])
    out["knn_head"] = {"cosine": cosine, "thr": round(float(thr), 3),
                       "recall_knn": round(float((s_fr > thr).mean()), 3), "recall_lr": round(float((z_fr > tau["ivis"]).mean()), 3),
                       "fpr_lr_on_sample": round(float((iv.logit(Xte[leg]) > tau["ivis"]).mean()), 4)}
    for mode in MODES:
        C = Constraints(meta, a["X_train"], mode)
        Sd = pe[f"ivis|{mode}|0.25"]
        dS = Sd[Sd[:, 1] == 1, 0].astype(int)            # frauds ivis detects
        X0a = Xte[dS]
        Xa, la = apgd(iv, X0a, tau["ivis"], 0.25, C, n_iter=100, restarts=5)
        ok = la < tau["ivis"]
        out[mode]["apgd_asr_detected"] = round(float(ok.mean()), 3)
        if ok.sum() < 3:
            out[mode]["embedding"] = {"n_evaded": int(ok.sum())}
            continue
        out[mode]["embedding"] = embedding_movement(iv, enc, Eref, yref, w, thr, cosine, X0a, Xa, ok)
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(main(d)), flush=True)
