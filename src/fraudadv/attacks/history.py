"""History injection (camouflage): a problem-space attack that every model family sees.

The attack. Before the fraudulent transaction the fraudster makes k innocuous transactions with the same card,
SPACING = 600 seconds apart, the last one SPACING seconds before the fraud. Nothing about the fraud itself changes;
what changes is its history, and each model family sees that history its own way:
  tabular models   the past-only per-card aggregates in the fraud's feature vector (count of previous transactions,
                   mean previous log-amount, log-ratio to it, time since the previous transaction, 24-hour count and
                   log-amount sum, missing-history indicators)
  graph models     the same feature vector; in addition the k injected transactions become the most recent same-card
                   neighbours (relation r1) of the fraud node, displacing genuine history. Each injected node has a
                   full feature vector: the card's most recent genuine transaction as a template, with the injected
                   amount, time and aggregates
  foundation       the same feature vector; in addition the k injected transactions become the most recent points of
  models           the fraud's series (log-amount and time gap of the last 31 same-card transactions), and each injected
                   transaction is scored with the series it would itself have had

Timeline. The injected transactions must come after the card's last genuine transaction. When that transaction is
less than (k + 1) * SPACING before the fraud (card-testing bursts: nearly every TabFormer fraud, half of the Sparkov
ones at k = 8), the k injections do not fit in the gap. The whole scenario is then delayed (`timeline`): the first
injection comes SPACING after the last genuine transaction and the fraud SPACING after the last injection. The fraud
keeps its hour and day of week. (Without the delay, injections would sit before the last genuine transaction while
being ranked after it, a zero time gap in the foundation models' series.)

The attacker's choice, per fraud and k: one amount for all injected transactions, from
  mimic   the card's mean previous amount (no queries to the model)
  best    the lowest-scoring of {mean previous, last genuine, a $2 card-testing amount, the fraud's own amount}:
          4 queries (score-based black-box)
Success: the fraud's score falls below the deployed threshold while none of the injected transactions is itself
flagged (a flagged warm-up transaction would give the attack away). k runs over K_LIST = 1, 2, 4, 8.

Datasets: those with per-card history features (Sparkov, TabFormer, IEEE-CIS); others are skipped with a message.
Same fraud sample (sample.npz) and deployed thresholds (tau.json) as the attack grid. A run can attack a random
subset of the sample instead (n_frauds, --n-frauds; 0 = the whole sample, the default). The subset is meant for
the slow foundation models: it is drawn as in the grid (seed 1), so it matches the grid's foundation-model subset of
the same size.

Outputs, per model (resumable: models already in the CSV are skipped):
  runs/<dataset>/attacks/history<tag>.csv               one row per (model, variant, k): n_detected, asr, asr_usd,
                                                        mean_logit_drop, injected_flagged
  runs/<dataset>/attacks/history_per_example<tag>.npz   per fraud of the sample: "<family>|detected",
                                                        "<family>|<variant>|<k>" = evaded, and "S", so results can be
                                                        split by group afterwards (Table S-XLIII)

Results: Table S-XII (supplement), the foundation-model rows of Table V, Table S-XLIII and the history-injection
numbers of section VI-E. attacks/history_consistent.py reuses Injector, timeline, fm_logits and the constants for
the consistent attacker (Table S-XIX).

usage: python -m fraudadv.attacks.history <dataset> fam1,fam2,... [--tag _x] [--n-frauds 0]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

K_LIST = [1, 2, 4, 8]           # numbers of injected transactions
SPACING = 600.0                 # seconds between consecutive injected transactions, and before the fraud
HIST = ["cnt_prev", "log_amt_mean_prev", "log_ratio", "log_dt_prev", "cnt_24h", "logamt_sum_24h"]   # history features
GRAPH = {"hgnn", "hgt"}
FMS = {"chronos2", "timesfm", "chronos2_lora", "timesfm_lora", "chronos2_zs", "timesfm_zs"}
# Default of run(n_frauds=...): attack the whole sample.
HIST_N = 0


class Injector:
    """Reads and rewrites the history features of standardized feature vectors.

    ix: feature name -> column; mu, sd: the standardization of every column; isna: columns of the missing-history
    indicators of the HIST features.
    """

    def __init__(self, meta):
        f = meta["features"]
        self.ix = {n: i for i, n in enumerate(f)}
        self.mu, self.sd = np.asarray(meta["mu"], np.float64), np.asarray(meta["sd"], np.float64)
        self.isna = [self.ix[n] for n in f if n.startswith("isna_") and n[5:] in HIST]

    def u(self, X, n):
        """Feature n of X in its original (unstandardized) units, float64."""
        j = self.ix[n]
        return X[..., j].astype(np.float64) * self.sd[j] + self.mu[j]

    def s(self, X, n, v):
        """Set feature n of X, in place, to the original-unit value v (standardized; ignored if X has no feature n)."""
        j = self.ix.get(n)
        if j is not None:
            X[..., j] = ((v - self.mu[j]) / self.sd[j]).astype(np.float32)

    def history(self, x):
        """(count, mean previous log-amount or nan, 24h count, 24h log-amount sum) of a feature vector."""
        n = int(round(np.expm1(self.u(x, "cnt_prev"))))
        mean = self.u(x, "log_amt_mean_prev") if n > 0 else np.nan
        return n, mean, np.expm1(self.u(x, "cnt_24h")), self.u(x, "logamt_sum_24h")

    def apply(self, x, a_cur, t_cur, n, mean, c24, s24, injected):
        """Feature vector of a transaction at time t_cur with log-amount a_cur, made from x, when the card's history
        (n previous transactions with mean log-amount `mean`, c24 of them in the last 24 hours with log-amount sum s24)
        is followed by the `injected` log-amounts, each SPACING seconds after the one before."""
        x = x.copy()
        k = len(injected)
        tot = (n * mean if n > 0 else 0.0) + float(np.sum(injected))       # sum of all previous log-amounts
        m2 = tot / (n + k) if n + k > 0 else np.nan                        # new mean previous log-amount
        self.s(x, "cnt_prev", np.log1p(n + k))
        if n + k > 0:
            self.s(x, "log_amt_mean_prev", m2)
            self.s(x, "log_ratio", a_cur - m2)
        if k:                                             # the previous transaction is now an injected one
            self.s(x, "log_dt_prev", np.log1p(SPACING))
            for j in self.isna:                           # and the card has a history
                x[j] = (0.0 - self.mu[j]) / self.sd[j]
        self.s(x, "cnt_24h", np.log1p(c24 + k))
        self.s(x, "logamt_sum_24h", s24 + float(np.sum(injected)))
        self.s(x, "log_amt", a_cur)
        if "hour" in self.ix:
            self.s(x, "hour", float((t_cur // 3600) % 24))
        if "dow" in self.ix:
            self.s(x, "dow", float(((t_cur // 86400) + 3) % 7))
        return x


def timeline(dt_prev, k):
    """(delay of the whole scenario, gap from the last genuine transaction to the first injection), in seconds,
    from the time since the card's previous transaction (inf or nan when there is none)."""
    dt = np.where(np.isfinite(dt_prev), dt_prev, np.inf)
    delay = np.where(np.isfinite(dt), np.maximum(0.0, (k + 1) * SPACING - dt), 0.0)
    gap0 = dt + delay - k * SPACING                      # >= SPACING; inf when the card has no history
    return delay, gap0


def scenarios(inj, X0, t0, tmpl, amounts, k, dt_prev=None):
    """Feature vectors of one injection scenario per fraud.

    X0: the frauds' feature vectors; t0: their times; tmpl: per fraud, the card's most recent genuine transaction
    (the template of the injected ones); amounts[i]: the injected log-amount for fraud i; k: number of injections.
    With dt_prev (seconds since the card's previous transaction) the scenario follows `timeline`.
    Returns the frauds' new vectors (n, d) and, per fraud, the k injected transactions' vectors (k, d), oldest first.
    """
    Xt, Xi = [], []
    delay, gap0 = timeline(dt_prev, k) if dt_prev is not None else (np.zeros(len(X0)), np.full(len(X0), np.inf))
    for i in range(len(X0)):
        n, mean, c24, s24 = inj.history(X0[i])
        a_t = inj.u(X0[i], "log_amt")
        seq = [amounts[i]] * k
        Xt.append(inj.apply(X0[i], a_t, t0[i], n, mean, c24, s24, seq))
        if "hour" in inj.ix:                              # the fraud keeps its own hour / dow
            Xt[-1][inj.ix["hour"]] = X0[i][inj.ix["hour"]]
        if "dow" in inj.ix:
            Xt[-1][inj.ix["dow"]] = X0[i][inj.ix["dow"]]
        Xi.append(np.stack([inj.apply(tmpl[i], amounts[i], t0[i] + delay[i] - (k - j) * SPACING, n, mean, c24, s24, seq[:j])
                            for j in range(k)]) if k else np.empty((0, X0.shape[1]), np.float32))
        if k and np.isfinite(gap0[i]):                    # the first injection follows the last genuine transaction
            inj.s(Xi[-1][0], "log_dt_prev", np.log1p(gap0[i]))
    return np.stack(Xt), Xi


def graph_logits(m, dataset, Xt, Xi, ids, k, delay=None):
    """Graph models: logits of the frauds after injection, and of the highest-scoring injected transaction per fraud.

    The graph context is extended, for the duration of the call, by B * k injected nodes (feature vectors Xi, k per
    fraud, at SPACING intervals before the possibly delayed fraud time) and by one clone of each fraud node (feature
    vector Xt). Each injected node's same-card neighbours (r1) are the injections before it and then the card's
    genuine history; the clone's r1 neighbours start with all k injections, displacing the oldest genuine ones. The
    injected nodes take the relation-2 neighbours of the card's most recent genuine transaction (the fraud's own when
    the card has none); the clone keeps the fraud's.
    Returns (fraud logits (B,), max logit over each fraud's injected transactions (B,)).
    """
    import torch

    from fraudadv.models import graph as G

    base = G.context(dataset)
    N = base["Z"].shape[0]
    B = len(ids)
    nb1, nb2, T = base["nbr1"], base["nbr2"], base["time"]
    K1 = nb1.shape[1]
    tgt = torch.as_tensor(ids)
    inj_ids = N + torch.arange(B * k).view(B, k)                          # chronological per fraud
    clone_ids = N + B * k + torch.arange(B)
    Z_new = torch.from_numpy(np.concatenate([np.concatenate(Xi), Xt]).astype(np.float32)) if k else torch.from_numpy(Xt.astype(np.float32))
    t0 = T[tgt] if delay is None else T[tgt] + torch.as_tensor(np.asarray(delay), dtype=T.dtype)   # the delayed fraud
    t_inj = (t0.unsqueeze(1) - SPACING * (k - torch.arange(k, dtype=T.dtype)).unsqueeze(0)) if k else t0[:0].view(B, 0)
    orig1 = nb1[tgt]                                                       # (B, K1), most recent first
    rows1 = []
    for j in range(k):                                                     # injected node j: inj_{j-1..0}, then genuine
        prev = inj_ids[:, :j].flip(1)
        rows1.append(torch.cat([prev, orig1[:, : K1 - j]], 1))
    clone1 = torch.cat([inj_ids.flip(1), orig1[:, : K1 - k]], 1)
    tmpl_nb2 = torch.where((orig1[:, :1] >= 0), nb2[orig1[:, 0].clamp(min=0)], nb2[tgt])
    nb1_new = torch.cat([torch.stack(rows1, 1).view(B * k, K1) if k else nb1[:0], clone1]).to(nb1.dtype)
    nb2_new = torch.cat([tmpl_nb2.repeat_interleave(k, 0) if k else nb2[:0], nb2[tgt]]).to(nb2.dtype)
    aug = {"Z": torch.cat([base["Z"], Z_new]), "nbr1": torch.cat([nb1, nb1_new]), "nbr2": torch.cat([nb2, nb2_new]),
           "time": torch.cat([T, t_inj.reshape(-1).to(T.dtype), t0])}
    G._CTX[dataset] = aug
    try:
        lt = m.logit(Xt, clone_ids.numpy())
        li = m.logit(np.concatenate(Xi), inj_ids.reshape(-1).numpy()).reshape(B, k)
        return lt, li.max(1)
    finally:
        G._CTX[dataset] = base


def fm_logits(m, dataset, Xt, Xi, ids, k, delay=None):
    """Foundation models: score fraud clones whose series starts (most recent first) with the k injected
    transactions, and every injected transaction with the series it would have had. The series are built
    from the CPU context (same-card neighbours, times) and the per-node log-amounts, so both are extended
    by the injected nodes and the clones for the duration of the call, then restored, and the model's cached
    embeddings of the temporary node ids are dropped.
    Returns (fraud logits (B,), max logit over each fraud's injected transactions (B,), -inf when k = 0)."""
    import torch

    from fraudadv.models import foundation as F
    from fraudadv.models import graph as G

    base = G.context(dataset, cpu=True)
    lam = F._log_amt(dataset)
    nb1, T = base["nbr1"], base["time"]
    N, B, K1 = nb1.shape[0], len(ids), nb1.shape[1]
    tgt = torch.as_tensor(ids)
    inj_ids = N + torch.arange(B * k).view(B, k)                          # chronological per fraud
    clone_ids = N + B * k + torch.arange(B)
    t0 = T[tgt] if delay is None else T[tgt] + torch.as_tensor(np.asarray(delay), dtype=T.dtype)   # the delayed fraud
    t_inj = (t0.unsqueeze(1) - SPACING * (k - torch.arange(k, dtype=T.dtype)).unsqueeze(0)) if k else t0[:0].view(B, 0)
    orig1 = nb1[tgt]                                                       # (B, K1), most recent first
    rows1 = [torch.cat([inj_ids[:, :j].flip(1), orig1[:, : K1 - j]], 1) for j in range(k)]
    clone1 = torch.cat([inj_ids.flip(1), orig1[:, : K1 - k]], 1)
    Xi_all = np.concatenate(Xi) if k else np.empty((0, Xt.shape[1]), np.float32)

    def unscaled_log_amt(X):
        return torch.as_tensor(X[:, m.ia].astype(np.float64) * float(m.sd[m.ia]) + float(m.mu[m.ia]), dtype=lam.dtype)

    aug = dict(base)
    aug["nbr1"] = torch.cat([nb1, torch.stack(rows1, 1).view(B * k, K1) if k else nb1[:0], clone1]).to(nb1.dtype)
    aug["time"] = torch.cat([T, t_inj.reshape(-1).to(T.dtype), t0])
    G._CTX[dataset, "cpu"] = aug
    F._LOGAMT[dataset] = torch.cat([lam, unscaled_log_amt(Xi_all), unscaled_log_amt(Xt)])
    try:
        lt = m.logit(Xt, clone_ids.numpy())
        li = m.logit(Xi_all, inj_ids.reshape(-1).numpy()).reshape(B, k).max(1) if k else np.full(B, -np.inf)
        return lt, li
    finally:
        G._CTX[dataset, "cpu"] = base
        F._LOGAMT[dataset] = lam
        cache = m.cache                                   # embeddings are cached per node id: drop the temporary ids
        for key in [q for q in cache if q >= N]:
            del cache[key]


def _attacker_inputs(dataset, a, meta, S, keep):
    """What the attacker knows about each fraud of the sample S, and the candidate injected amounts.

    Returns ids (context node ids), X0 (feature vectors), amt (amounts), t0 (times), inj (Injector), dt_prev
    (seconds since the card's previous transaction, inf without one), tmpl (the card's most recent genuine
    transaction, the fraud itself without one) and cands: {candidate name: injected log-amount per fraud}.
    """
    from fraudadv.models import graph as G

    ids = keep["test"][S]
    X0, amt = a["X_test"][S], a["amt_test"][S].astype(float)
    t0 = a["t_test"][S] if "t_test" in a else np.asarray(G.context(dataset, cpu=True)["time"][ids])
    inj = Injector(meta)
    ctx = G.context(dataset, cpu=True)                    # CPU copy: on a GPU machine the default copy is on CUDA
    last = ctx["nbr1"][ids, 0].numpy()                    # the card's most recent genuine transaction (-1: none)
    Tc = np.asarray(ctx["time"], np.float64)
    dt_prev = np.where(last >= 0, Tc[ids] - Tc[np.maximum(last, 0)], np.inf)     # seconds since the card's previous transaction
    Zc = ctx["Z"]
    tmpl = np.where((last >= 0)[:, None], Zc[np.maximum(last, 0)].numpy(), X0)
    hist = [inj.history(x) for x in X0]
    a_t = np.array([inj.u(x, "log_amt") for x in X0])     # the fraud's own log-amount
    mean_prev = np.array([h[1] if h[0] > 0 else a_t[i] for i, h in enumerate(hist)])
    a_last = np.array([inj.u(tmpl[i], "log_amt") if last[i] >= 0 else a_t[i] for i in range(len(X0))])
    cands = {"mean_prev": mean_prev, "last": a_last, "test_2usd": np.full(len(X0), np.log1p(2.0)), "own": a_t}
    return ids, X0, amt, t0, inj, dt_prev, tmpl, cands


def _scenario_logits(m, fam, dataset, Xt, Xi, ids, k, delay):
    """Logits of the frauds after injection (Xt) and the highest logit among each fraud's injected transactions (Xi),
    scored the way the model family sees history."""
    if fam in GRAPH:
        return graph_logits(m, dataset, Xt, Xi, ids, k, delay)
    if fam in FMS:
        return fm_logits(m, dataset, Xt, Xi, ids, k, delay)
    lt = m.logit(Xt, ids)
    li = m.logit(np.concatenate(Xi), np.repeat(ids, k)).reshape(len(Xt), k).max(1)
    return lt, li


def run(dataset, families, tag="", n_frauds=0):
    """History injection against `families` of `dataset`; n_frauds > 0 attacks a random subset of the sample of
    that size (seed 1, as the grid's foundation-model subset). Returns the rows written to history<tag>.csv."""
    if any(f in FMS for f in families):
        import chronos  # noqa: F401  must be imported before TensorFlow (ivis)
    from fraudadv.attacks.run_attacks import RUNS, context_ids, load_model
    from fraudadv.data.load import load

    a, meta = load(dataset)
    if not all(n in meta["features"] for n in HIST):
        print(dataset, "has no per-card history features; skipped", flush=True)
        return []
    out = RUNS / dataset / "attacks"
    tau = json.loads((out / "tau.json").read_text())
    S = np.load(out / "sample.npz")["S"]
    if n_frauds > 0:                                      # same random subset as the foundation models' attack grid
        S = S[np.sort(np.random.default_rng(1).permutation(len(S))[:n_frauds])]
    keep = context_ids(dataset)
    ids, X0, amt, t0, inj, dt_prev, tmpl, cands = _attacker_inputs(dataset, a, meta, S, keep)
    p = out / f"history{tag}.csv"
    rows = pd.read_csv(p).to_dict("records") if p.exists() else []
    have = {r["model"] for r in rows}
    pp = out / f"history_per_example{tag}.npz"
    PE = dict(np.load(pp)) if pp.exists() else {}
    PE["S"] = S
    for fam in families:
        if fam in have or fam not in tau:
            continue
        m = load_model(dataset, fam, 0)
        t = float(tau[fam])
        l0 = m.logit(X0, ids)
        det = l0 > t
        PE[f"{fam}|detected"] = det
        for k in K_LIST:
            flagged = {}                                  # per candidate amount: detected frauds with a flagged injection
            L = {}                                        # per candidate amount: the fraud's logit (not below l0 if it fails)
            for c, amts in cands.items():
                Xt, Xi = scenarios(inj, X0, t0, tmpl, amts, k, dt_prev)
                delay = timeline(dt_prev, k)[0]
                lt, li = _scenario_logits(m, fam, dataset, Xt, Xi, ids, k, delay)
                # an injected transaction that is itself flagged gives the attack away: the scenario fails
                L[c] = np.where(li < t, lt, np.maximum(lt, l0))
                flagged[c] = flagged.get(c, 0) + int(np.sum(det & (li >= t)))
            best = np.min(np.stack(list(L.values())), 0)  # the best of the 4 amounts, per fraud
            for variant, l in (("mimic (no queries)", L["mean_prev"]), ("best of 4 (4 queries)", best)):
                ev = det & (l < t)
                PE[f"{fam}|{variant}|{k}"] = ev
                rows.append(dict(dataset=dataset, model=fam, attack=variant, k=k, n_detected=int(det.sum()),
                                 asr=float(ev.sum() / max(det.sum(), 1)),
                                 asr_usd=float(amt[ev].sum() / max(amt[det].sum(), 1e-9)),
                                 mean_logit_drop=float(np.mean((l0 - l)[det])) if det.any() else np.nan,
                                 injected_flagged=flagged["mean_prev"] if variant.startswith("mimic") else int(sum(flagged.values()))))
                print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in rows[-1].items()
                       if kk in ("model", "attack", "k", "asr", "asr_usd", "mean_logit_drop")}, flush=True)
        pd.DataFrame(rows).to_csv(p, index=False)
        np.savez_compressed(pp, **PE)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="History injection (Table S-XII).")
    ap.add_argument("dataset")
    ap.add_argument("families", help="comma-separated model families")
    ap.add_argument("--tag", default="", help="suffix of the output files (a rerun that must not touch history.csv)")
    ap.add_argument("--n-frauds", type=int, default=HIST_N,
                    help="attack a random subset of the sample of this size (seed 1, as the grid's foundation-model "
                         "subset); 0 = the whole sample")
    args = ap.parse_args()
    run(args.dataset, args.families.split(","), tag=args.tag, n_frauds=args.n_frauds)
