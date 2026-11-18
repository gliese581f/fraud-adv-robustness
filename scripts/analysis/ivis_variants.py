"""ivis variants: retrain ivis so that its similarity learning carries the decision, then attack it with the
study's grid (no adversarial training anywhere).

Variants (fraudadv.models.ivis_variants.IvisVariant), each with two heads on the same encoder:
  base     the study's ivis encoders (runs/<d>/ivis/seed*), unchanged          heads: knn (lr = the study's ivis)
  sup      maaten encoder, supervision weight searched in [0.7, 1.0]            heads: lr, knn
  smooth   spectrally normalized encoder (each layer's norm <= cap, cap in {2, 3}, no alpha dropout),
           supervision weight searched in [0.1, 1.0]                            heads: lr, knn
  "lr" is the logistic head on the embedding, "knn" the neighbour vote among the k nearest training embeddings.
  The five attacked models are named ivis_<variant>_<head> (MODELS).

Protocol
  Tuning follows the study's ivis protocol: 10 TPE trials (seed 0, 5 random), at most 2,400 s, on the same
  100,000-row training subsample; embedding size, neighbours k, distance and C searched as before. The objective
  is the neighbour vote's validation AUPRC (the decision the variants are meant to improve). Final models: three
  seeds. Thresholds: 1% FPR on validation, as in the study.
  Attacks: fraudadv.attacks.run_attacks on the study's stored fraud sample (runs/<d>/attacks/sample.npz), whole
  sample, grid only (APGD 100x5, Square 300 queries, HopSkipJump, worst case), so the rates compare with the
  study's per_example_full_* files. On top, `adaptive` runs PGD steered by smooth surrogates of the neighbour vote.
  Threat models: unconstrained and constrained by default; `--modes consistent` re-attacks the saved models under
  the consistent attacker (Constraints(..., "consistent")) into separate files with the suffix _consistent.
Everything is written under runs_ivisvar/<d>/ (and runs_ivisvar/report/), so the study's report scripts never see it.

Commands
  python scripts/analysis/ivis_variants.py tune     <dataset> sup|smooth
  python scripts/analysis/ivis_variants.py train    <dataset> sup|smooth|base
  python scripts/analysis/ivis_variants.py attack   <dataset> [models] [--modes unconstrained,constrained]
                                                    default models: all five
  python scripts/analysis/ivis_variants.py adaptive <dataset> [models] [--modes unconstrained,constrained]
                                                    default models: the three neighbour votes
  python scripts/analysis/ivis_variants.py diag     [datasets]      embedding geometry (runs_ivisvar/report/geometry.csv)
  python scripts/analysis/ivis_variants.py report   [datasets]      worst_case.csv, paired.csv, clean.csv
  default datasets: sparkov, tabformer, baf, ieee_cis

Results: Table S-XXV is runs_ivisvar/report/clean.csv and worst_case.csv, Table S-XXVI is paired.csv with the Holm
correction of scripts/analysis/ivis_variants_paired.py (paired_holm.csv), Table S-XXVII is geometry.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import torch  # noqa: E402,F401  before TensorFlow: the study's MLP (diag) is torch, and the reverse order aborts on an LLVM option clash
import tensorflow as tf  # noqa: E402

if int(os.environ.get("FRAUDADV_THREADS", "0")):            # parallel lanes on one machine share its cores
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["FRAUDADV_THREADS"]))
    tf.config.threading.set_inter_op_parallelism_threads(2)

from fraudadv.data.load import load  # noqa: E402
from fraudadv.eval.metrics import all_metrics, threshold_at_fpr  # noqa: E402
from fraudadv.hpo.train_all import RUNS, subsample  # noqa: E402
from fraudadv.models.ivis_variants import IvisVariant, KNNHead, choose_knn, val_subsample  # noqa: E402
from fraudadv.paths import RUNS_CONSISTENT  # noqa: E402

OUT = REPO / "runs_ivisvar"
SEEDS = (0, 1, 2)
N_TRIALS, TIME_CAP, HPO_ROWS = 10, 2400, 100_000
MODELS = ["ivis_base_knn", "ivis_sup_lr", "ivis_sup_knn", "ivis_smooth_lr", "ivis_smooth_knn"]
DATASETS = ["sparkov", "tabformer", "baf", "ieee_cis"]
MODES = ("unconstrained", "constrained")      # threat models of attack and adaptive (default of --modes)


def mode_suffix(modes):
    """Output-file suffix: "_consistent" for a consistent-only run, so it never overwrites the other files."""
    return "_consistent" if tuple(modes) == ("consistent",) else ""


# ------------------------------------------------------------------ tuning and training
def space(variant, t):
    """Search space of one variant (optuna trial t). The neighbour vote ("knn") is the head being tuned."""
    p = dict(embedding_dims=t.suggest_categorical("embedding_dims", [8, 16, 32]), k=t.suggest_categorical("k", [15, 30, 60]),
             distance=t.suggest_categorical("distance", ["pn", "softmax_ratio_pn", "cosine_pn"]),
             C=t.suggest_float("C", 0.01, 10, log=True), epochs=20, max_train=50_000, head="knn")
    if variant == "sup":
        p.update(supervision_weight=t.suggest_float("supervision_weight", 0.7, 1.0), sn=False, drop=0.1)
    elif variant == "smooth":
        p.update(supervision_weight=t.suggest_float("supervision_weight", 0.1, 1.0), sn=True,
                 cap=float(t.suggest_categorical("cap", [2.0, 3.0])), drop=0.0)
    else:
        raise KeyError(variant)
    return p


def tune(d, variant):
    """TPE search (resumable: the optuna study lives in runs_ivisvar/<d>/<variant>/study.db); writes best_params.json."""
    out = OUT / d / variant
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best_params.json").exists():
        print(d, variant, "already tuned", flush=True)
        return
    a, _ = load(d)
    Xf, yf = subsample(a["X_train"], a["y_train"], HPO_ROWS)
    iv, wv = val_subsample(a["y_val"], a["w_val"])

    def objective(trial):
        m = IvisVariant().fit(Xf, yf, a["X_val"][iv], a["y_val"][iv], wv, space(variant, trial), seed=0)
        return m.knn_val_auprc

    study = optuna.create_study(direction="maximize", study_name=f"{d}-{variant}", load_if_exists=True,
                                storage=f"sqlite:///{out / 'study.db'}",
                                sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=0, n_startup_trials=5))
    t0 = time.time()
    done = sum(t.state.is_finished() for t in study.trials)
    spent = sum((t.duration.total_seconds() if t.duration else 0.0) for t in study.trials)
    if N_TRIALS - done > 0:
        study.optimize(objective, n_trials=N_TRIALS - done, timeout=max(60.0, TIME_CAP - spent), catch=(Exception,))
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    best = space(variant, optuna.trial.FixedTrial(study.best_params))
    (out / "best_params.json").write_text(json.dumps({"params": best, "val_auprc_knn": study.best_value, "n_trials": len(study.trials),
                                                      "hpo_seconds": spent + time.time() - t0}, indent=1))
    print(d, variant, "tuned", study.best_value, best, flush=True)


def base_variant(d, seed):
    """The study's ivis encoder of one seed as an IvisVariant (its LR head, plus a neighbour vote on its own
    training subsample: all frauds and legitimate rows up to max_train, drawn as in models.zoo.IvisModel.fit)."""
    import tensorflow as tf

    from fraudadv.hpo.train_all import load_model

    a, _ = load(d)
    p = json.loads((RUNS / d / "ivis" / "best_params.json").read_text())["params"]
    iv0 = load_model(d, "ivis", seed)
    Xf, yf = a["X_train"], a["y_train"]          # train_all passes ivis the whole training set (no SUBSAMPLE entry)
    n = min(len(Xf), p.get("max_train", 60_000))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(yf == 1), np.flatnonzero(yf == 0)
    idx = np.r_[pos, rng.choice(neg, size=max(n - len(pos), 1), replace=False)]
    m = IvisVariant()
    m.params = dict(p, d=int(Xf.shape[1]), head="knn", base=True)
    m.encoder, m.head, m.cosine = iv0.encoder, iv0.head, "cosine" in p.get("distance", "pn")
    m._Eref, m._yref = m.embed(Xf[idx]), yf[idx]
    m._w = tf.constant(m.head.coef_.T.astype(np.float32))
    m._b = tf.constant(m.head.intercept_.astype(np.float32))
    # the neighbour vote's k and temperature are chosen on a validation subsample, as for the new variants
    ivs, wv = val_subsample(a["y_val"], a["w_val"])
    acc, k, T = choose_knn(m._Eref, m._yref, m.embed(a["X_val"][ivs]), a["y_val"][ivs], wv, m.cosine)
    m.knn, m.knn_val_auprc, m.head_kind = KNNHead(m._Eref, m._yref, k, T, m.cosine), acc, "knn"
    return m


def train(d, variant):
    """Final models of one variant, three seeds (resumable per seed): the model under seed<s>/, clean test metrics of
    both heads at the 1%-FPR validation threshold (metrics_seed<s>.json, metrics.json) and test scores."""
    a, _ = load(d)
    out = OUT / d / variant
    out.mkdir(parents=True, exist_ok=True)
    Xf, yf = a["X_train"], a["y_train"]          # as train_all: the model draws its own 50,000-row subsample
    if variant != "base":
        best = json.loads((out / "best_params.json").read_text())["params"]
    mets = {"lr": [], "knn": []}
    for seed in SEEDS:
        ms = out / f"metrics_seed{seed}.json"
        if ms.exists():
            r = json.loads(ms.read_text())
        else:
            if variant == "base":
                m = base_variant(d, seed)
            else:
                m = IvisVariant().fit(Xf, yf, a["X_val"], a["y_val"], a["w_val"], best, seed=seed)
            m.save(out / f"seed{seed}")
            r = {}
            for h in ("lr", "knn"):
                mm = m.with_head(h)
                sv = mm.logit(a["X_val"])
                thr = threshold_at_fpr(a["y_val"], sv, a["w_val"], 0.01)
                st = mm.logit(a["X_test"])
                met = all_metrics(a["y_test"], st, a["w_test"], a["amt_test"], thr)
                met.update(seed=seed, threshold=float(thr))
                r[h] = met
                np.save(out / f"test_scores_{h}_seed{seed}.npy", st)
            ms.write_text(json.dumps(r))
            print(d, variant, seed, {h: {k: round(v, 4) for k, v in r[h].items() if k in ("auc", "auprc", "recall@1%fpr")} for h in r}, flush=True)
        for h in ("lr", "knn"):
            mets[h].append(r[h])
    (out / "metrics.json").write_text(json.dumps(mets, indent=1))


def load_variant(d, fam, seed=0):
    """Load model "ivis_<variant>_<head>" of one seed."""
    _, variant, head = fam.split("_")
    return IvisVariant.load(OUT / d / variant / f"seed{seed}", head)


# ------------------------------------------------------------------ attacks
def attack(d, models, modes=MODES):
    """The study's attack grid on each model (seed 0), with run_attacks pointed at runs_ivisvar/ and its model
    loader replaced by load_variant. Writes runs_ivisvar/<d>/attacks/*_full_<model><suffix>.*; skips models done."""
    import fraudadv.attacks.run_attacks as R

    modes, suffix = tuple(modes), mode_suffix(modes)
    (OUT / d / "attacks").mkdir(parents=True, exist_ok=True)
    tmp = OUT / d / "attacks" / f"sample.{os.getpid()}.tmp"     # atomic: lanes on one machine copy it at the same time
    shutil.copy2(RUNS / d / "attacks" / "sample.npz", tmp)
    os.replace(tmp, OUT / d / "attacks" / "sample.npz")
    R.load_model = lambda dataset, fam, seed=0: load_variant(dataset, fam, seed)
    for fam in models:
        if (OUT / d / "attacks" / f"results_full_{fam}{suffix}.csv").exists():
            print(d, fam, "already attacked", modes, flush=True)
            continue
        R.run(d, [fam], tag=f"_full_{fam}{suffix}", modes=modes, full_bb=True, grid_only=True, runs_dir=OUT)


def adaptive_pgd(model, sur, X0, tau, eps, cons, n_iter=100, restarts=3, seed=0):
    """Min-effort L-inf PGD that follows the surrogate's gradient and keeps, per fraud, the lowest score of the
    attacked model itself; step 2 eps, halved every quarter of the iterations; random starts as in APGD (the first
    restart at half the radius). Returns the lowest score found per fraud."""
    from fraudadv.attacks.blackwhite import _ball

    rng = np.random.default_rng(seed)
    X0 = X0.astype(np.float32)
    best = model.logit(X0)
    for r in range(restarts):
        a = np.flatnonzero(best >= tau)             # frauds not yet evaded
        if not len(a):
            break
        Xa0 = X0[a]
        X = _ball(Xa0 + rng.uniform(-eps, eps, Xa0.shape).astype(np.float32) * (0.5 if r == 0 else 1.0), Xa0, eps, cons)
        live = np.ones(len(a), bool)
        for k in range(n_iter):
            L = np.flatnonzero(live)
            if not len(L):
                break
            eta = 2.0 * eps * 0.5 ** (4 * k // n_iter)
            X[L] = _ball(X[L] - eta * np.sign(sur.grad(X[L])), Xa0[L], eps, cons)
            lt = model.logit(X[L])
            b = lt < best[a[L]]
            best[a[L[b]]] = lt[b]
            live[L] = best[a[L]] >= tau
    return best


def adaptive(d, models, modes=MODES):
    """Attacks aimed at the neighbour vote, on top of the grid (whose APGD uses the vote's exact gradient, which is
    zero wherever all k neighbours share a label): PGD steered by three surrogates, success judged on the vote.
      lse      class-wise log-sum-exp over the K = k nearest fraud / legitimate references, T as the vote's
      lse_hot  the same with K = 100 and T ten times higher (a smoother, longer-range pull)
      lr       the logistic head on the same encoder (transfer within the model)
    Same fraud sample, threshold rule, budgets and threat models as the grid. Writes per-example outcomes to
    adaptive_<model><suffix>.npz ("<model>|<mode>|<eps>" -> [test row, detected, evaded by any surrogate, amount])
    and per-surrogate rates to adaptive_<model><suffix>.csv."""
    import pandas as pd

    from fraudadv.attacks.constraints import Constraints
    from fraudadv.attacks.run_attacks import EPS
    from fraudadv.models.ivis_variants import ClassLSE

    suffix = mode_suffix(modes)
    a, meta = load(d)
    S = np.load(RUNS / d / "attacks" / "sample.npz")["S"]
    X0 = a["X_test"][S]
    for fam in models:
        f = OUT / d / "attacks" / f"adaptive_{fam}{suffix}.npz"
        if f.exists():
            print(d, fam, "adaptive already done", flush=True)
            continue
        m = load_variant(d, fam)
        tau = threshold_at_fpr(a["y_val"], m.logit(a["X_val"]), a["w_val"], 0.01)
        l0 = m.logit(X0)
        det = l0 > tau
        surs = {"lse": ClassLSE(m), "lse_hot": ClassLSE(m, K=100, T=10 * m.knn.T), "lr": m.with_head("lr")}
        pe, rows = {}, []
        for mode in modes:
            cons = Constraints(meta, a["X_train"], mode)
            D = np.flatnonzero(det)
            for e in EPS:
                t0 = time.time()
                ev = np.zeros(len(S), bool)
                for name, sur in surs.items():
                    lb = adaptive_pgd(m, sur, X0[D], tau, e, cons)
                    ok = lb < tau
                    ev[D[ok]] = True
                    rows.append(dict(dataset=d, model=fam, mode=mode, eps=e, surrogate=name, n_detected=len(D), asr=float(ok.mean())))
                rows.append(dict(dataset=d, model=fam, mode=mode, eps=e, surrogate="any", n_detected=len(D), asr=float(ev[D].mean())))
                pe[f"{fam}|{mode}|{e}"] = np.stack([S, det, ev, a["amt_test"][S]], 1).astype(float)   # as the grid's per_example files
                print(d, fam, mode, e, {r["surrogate"]: round(r["asr"], 3) for r in rows[-4:]}, f"{time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / d / "attacks" / f"adaptive_{fam}{suffix}.csv", index=False)
        np.savez_compressed(f, **pe)


# ------------------------------------------------------------------ analysis
def diag(datasets):
    """Question: are frauds farther from legitimate transactions in the embedding than in feature space,
    and does that buy robustness? Per encoder (seed 0; the study's MLP penultimate layer for comparison), on the
    study's fraud sample S and 20,000 training legits L (both in the space the head uses: l2-normalised for cosine
    ivis variants):
      rel_gap_in / rel_gap_emb  median distance from a fraud to its nearest legit, divided by the median distance
                                from a legit to its nearest other legit (feature space L2 / embedding)
      sigma                     median spectral norm of the encoder's Jacobian at the frauds (all features)
      amp                       sigma * (legit NN scale in feature space) / (legit NN scale in the embedding):
                                local stretch at frauds relative to the encoder's typical scaling
      shortcut                  median over frauds of (embedding gap / sigma) / feature-space gap: the first-order
                                input step that closes the embedding gap, as a share of the real feature-space gap
    Writes runs_ivisvar/report/geometry.csv (Table S-XXVII).
    """
    import pandas as pd
    import torch
    from sklearn.neighbors import NearestNeighbors

    from fraudadv.hpo.train_all import load_model

    rows = []
    for d in datasets:
        a, _ = load(d)
        rng = np.random.default_rng(0)
        L = a["X_train"][rng.choice(np.flatnonzero(a["y_train"] == 0), 20_000, replace=False)].astype(np.float32)
        S = np.load(RUNS / d / "attacks" / "sample.npz")["S"]
        F = a["X_test"][S].astype(np.float32)
        q = np.arange(2000)

        def gaps(Lz, Fz):
            """(distance of each fraud to its nearest legit, median legit-to-nearest-other-legit distance over 2,000 legits)"""
            nn = NearestNeighbors(n_neighbors=2).fit(Lz)
            scale = float(np.median(nn.kneighbors(Lz[q])[0][:, 1]))
            return nn.kneighbors(Fz, n_neighbors=1)[0][:, 0], scale

        g_in, sc_in = gaps(L, F)
        encs = {}
        mlp = load_model(d, "mlp", 0)
        body = mlp.net[:-1].cpu().eval()
        encs["mlp (penultimate)"] = ("torch", body)
        for v in ("base", "sup", "smooth"):
            if (OUT / d / v / "seed0").exists():
                encs[f"ivis_{v}"] = ("tf", IvisVariant.load(OUT / d / v / "seed0", "knn"))
        for name, (kind, e) in encs.items():
            if kind == "torch":
                with torch.no_grad():
                    emb = lambda X: body(torch.tensor(X)).numpy()  # noqa: E731
                    Lz, Fz = emb(L), emb(F)
                x = torch.tensor(F, requires_grad=True)       # rows are independent: one backward pass per output
                z = body(x)                                     # unit gives that row of every fraud's Jacobian
                J = np.stack([torch.autograd.grad(z[:, j].sum(), x, retain_graph=True)[0].numpy() for j in range(z.shape[1])], 1)
                # (torch.func would import triton, whose LLVM clashes with TensorFlow's)
                cos = False
            else:
                cos = e.cosine
                prep = (lambda Z: Z / np.linalg.norm(Z, axis=1, keepdims=True)) if cos else (lambda Z: Z)
                Lz, Fz = prep(e.embed(L)), prep(e.embed(F))
                x = tf.constant(F)
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    z = e.encoder(x, training=False)
                    z = tf.math.l2_normalize(z, axis=1) if cos else z
                J = tape.batch_jacobian(z, x).numpy()
            sig = np.linalg.svd(J, compute_uv=False)[:, 0]
            g_e, sc_e = gaps(Lz, Fz)
            rows.append(dict(dataset=d, model=name, cosine=cos, rel_gap_in=float(np.median(g_in) / sc_in),
                             rel_gap_emb=float(np.median(g_e) / sc_e), sigma=float(np.median(sig)),
                             amp=float(np.median(sig) * sc_in / sc_e), shortcut=float(np.median((g_e / sig) / g_in))))
            print({k: (round(v_, 3) if isinstance(v_, float) else v_) for k, v_ in rows[-1].items()}, flush=True)
    rep = OUT / "report"
    rep.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(rep / "geometry.csv", index=False)


REF = ["mlp", "ft_transformer", "ivis", "hgnn", "hgt"]      # the study's models the variants are compared with


def per_example_outcomes(d):
    """Per-fraud outcomes on dataset d, "<model>|<mode>|<eps>" -> [test row, detected, evaded, amount]:
    the study's reference models (grid, and the consistent grid in runs_consistent/), the variants' grid, and for
    the neighbour votes the grid OR any adaptive surrogate (the grid alone stays available as "<model>@grid")."""
    pe = {}
    for f in sorted((RUNS / d / "attacks").glob("per_example_full_*.npz")):
        z = np.load(f)
        pe.update({k: z[k] for k in z.files if k.split("|")[0] in REF})
    for f in sorted((RUNS_CONSISTENT / d / "attacks").glob("per_example_consistent*.npz")):
        z = np.load(f)
        pe.update({k: z[k] for k in z.files if k.split("|")[0] in REF and k.split("|")[1] == "consistent"})
    for f in sorted((OUT / d / "attacks").glob("per_example_full_*.npz")):
        z = np.load(f)
        pe.update({k: z[k] for k in z.files})
    for f in sorted((OUT / d / "attacks").glob("adaptive_*.npz")):
        z = np.load(f)
        for k in z.files:
            if k in pe:
                grid, adapt = pe[k], z[k]
                assert np.array_equal(grid[:, 0], adapt[:, 0])
                fam, rest = k.split("|", 1)
                pe[f"{fam}@grid|{rest}"] = grid
                combined = grid.copy()
                combined[:, 2] = np.maximum(grid[:, 2], adapt[:, 2])
                pe[k] = combined
    return pe


def worst_case_rows(d, pe):
    """Worst-case ASR (evaded / detected) of every entry of pe."""
    rows = []
    for k, v in pe.items():
        fam, mode, e = k.split("|")
        det, ev = v[:, 1].astype(bool), v[:, 2].astype(bool)
        rows.append(dict(dataset=d, model=fam, mode=mode, eps=float(e), detected=int(det.sum()),
                         asr=float((det & ev).sum() / max(det.sum(), 1))))
    return rows


def paired_rows(d, pe):
    """Exact McNemar of each variant against the study's MLP, ivis and FT-Transformer at eps 0.25, on the frauds
    both models detect."""
    from scipy.stats import binomtest

    rows = []
    for mode in ("constrained", "unconstrained", "consistent"):
        for fam in MODELS:
            for other in ("mlp", "ivis", "ft_transformer"):
                ka, kb = f"{fam}|{mode}|0.25", f"{other}|{mode}|0.25"
                if ka not in pe or kb not in pe:
                    continue
                A, B = pe[ka], pe[kb]
                assert np.array_equal(A[:, 0], B[:, 0])
                both = A[:, 1].astype(bool) & B[:, 1].astype(bool)
                ea, eb = A[both, 2].astype(bool), B[both, 2].astype(bool)
                a_only, b_only = int((ea & ~eb).sum()), int((~ea & eb).sum())
                p = binomtest(a_only, a_only + b_only).pvalue if a_only + b_only else 1.0
                rows.append(dict(dataset=d, mode=mode, model=fam, vs=other, n=int(both.sum()), asr_model=float(ea.mean()),
                                 asr_vs=float(eb.mean()), model_only=a_only, vs_only=b_only, p=p))
    return rows


def clean_rows(d):
    """Mean clean AUC, AUPRC and recall at 1% FPR over seeds, of the variants and the study's reference models."""
    rows = []
    for variant in ("base", "sup", "smooth"):
        f = OUT / d / variant / "metrics.json"
        if f.exists():
            M = json.loads(f.read_text())
            for h, ms in M.items():
                if variant == "base" and h == "lr":            # that is the study's ivis itself
                    continue
                rows.append(dict(dataset=d, model=f"ivis_{variant}_{h}", **{k: float(np.mean([m[k] for m in ms])) for k in ("auc", "auprc", "recall@1%fpr")}))
    for fam in REF:
        f = RUNS / d / fam / "metrics.json"
        if f.exists():
            ms = json.loads(f.read_text())
            rows.append(dict(dataset=d, model=fam, **{k: float(np.mean([m[k] for m in ms])) for k in ("auc", "auprc", "recall@1%fpr")}))
    return rows


def report(datasets):
    """Worst-case ASR at every budget per model and setting, with the study's neural and graph models on the same
    fraud sample, paired McNemar against the study's MLP and ivis on frauds both detect, and clean metrics.
    Writes runs_ivisvar/report/worst_case.csv, paired.csv, clean.csv (Tables S-XXV and S-XXVI)."""
    import pandas as pd

    rows, pairs, clean = [], [], []
    for d in datasets:
        pe = per_example_outcomes(d)
        rows += worst_case_rows(d, pe)
        pairs += paired_rows(d, pe)
        clean += clean_rows(d)
    rep = OUT / "report"
    rep.mkdir(parents=True, exist_ok=True)
    T = pd.DataFrame(rows)
    T.to_csv(rep / "worst_case.csv", index=False)
    P = pd.DataFrame(pairs)
    P.to_csv(rep / "paired.csv", index=False)
    C = pd.DataFrame(clean)
    C.to_csv(rep / "clean.csv", index=False)
    piv = T[T.eps == 0.25].pivot_table(index=["dataset", "mode"], columns="model", values="asr")
    print((piv * 100).round(1).to_string())
    print(C.round(3).to_string())
    if len(P):
        print(P.round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description="ivis variants: tune, train, attack and report (runs_ivisvar/)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("tune", "train"):
        p = sub.add_parser(cmd)
        p.add_argument("dataset")
        p.add_argument("variant", help="sup|smooth (tune), sup|smooth|base (train)")
    for cmd, default in (("attack", "all five models"), ("adaptive", "the three neighbour votes")):
        p = sub.add_parser(cmd)
        p.add_argument("dataset")
        p.add_argument("models", nargs="?", help=f"comma-separated (default: {default})")
        p.add_argument("--modes", default=",".join(MODES), help="threat models; 'consistent' alone writes *_consistent files")
    for cmd in ("diag", "report"):
        p = sub.add_parser(cmd)
        p.add_argument("datasets", nargs="?", help="comma-separated (default: " + ",".join(DATASETS) + ")")
    args = ap.parse_args()
    if args.cmd == "tune":
        tune(args.dataset, args.variant)
    elif args.cmd == "train":
        train(args.dataset, args.variant)
    elif args.cmd == "attack":
        attack(args.dataset, args.models.split(",") if args.models else MODELS, tuple(args.modes.split(",")))
    elif args.cmd == "adaptive":
        models = args.models.split(",") if args.models else [f for f in MODELS if f.endswith("knn")]
        adaptive(args.dataset, models, tuple(args.modes.split(",")))
    else:
        datasets = args.datasets.split(",") if args.datasets else DATASETS
        (diag if args.cmd == "diag" else report)(datasets)


if __name__ == "__main__":
    main()
