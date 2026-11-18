"""History injection under the consistent attacker: fraudadv.attacks.history with every injected (warm-up)
transaction repaired into a row a real transaction could produce, checked with fraudadv.attacks.realizable
(hand rules and structural zeros).

The study's injection builds each warm-up transaction from the card's most recent genuine transaction (its
template), with the injected amount, time and history aggregates. Two things can make such a row impossible:
  missing flags    the first warm-up transaction of a card whose template had no history keeps the template's
                   missing-history flags while it gets the card's real history values (IEEE-CIS, 1-8% of the
                   warm-up rows). Repair: the flags follow their values (off once the card has a previous
                   transaction), as the bank's pipeline sets them.
  structural zero  the warm-up time gives the template's merchant category an hour at which that category never
                   occurs in training (Sparkov, hour x category, 8% of the warm-up rows). Repair: the attacker shops
                   where that hour is possible: the template becomes the card's most recent genuine transaction whose
                   category (and merchant, coordinates, distance) gives a possible row at that time; a structural
                   zero the template itself already holds is not counted against the row.
A scenario with a warm-up row that no template of the card makes possible fails, as one with a flagged warm-up
transaction does. The fraud itself is unchanged except for its history aggregates, which the validator accepts.
Graph models get the repaired rows as the injected nodes, each with the relation-2 neighbours of its own template.

usage: python -m fraudadv.attacks.history_consistent <dataset> fam1,fam2,... [--tag _x]
writes runs_consistent/<dataset>/attacks/history_consistent<tag>.csv and history_per_example_consistent<tag>.npz
(as fraudadv.attacks.history, plus per row: frauds with a repaired warm-up row and frauds no template makes possible)
Results: Table S-XIX and the consistent history numbers of the consistent-attacker appendix.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from fraudadv.attacks import history as H
from fraudadv.paths import attack_dir


def flags_follow(inj, v):
    """Missing-history flags off once the record has a previous transaction (their values are then real)."""
    if np.expm1(inj.u(v, "cnt_prev")) >= 0.5:
        for j in inj.isna:
            v[j] = (0.0 - inj.mu[j]) / inj.sd[j]
    return v


class Checker:
    def __init__(self, ds):
        from fraudadv.attacks.realizable import structural, violations

        self.ds, self.sz, self.hand = ds, structural(ds), violations

    def possible(self, v, tm):
        """v (a warm-up record) is possible: no hand rule broken and no structural zero its template lacks."""
        if self.hand(self.ds, v[None])[0]:
            return False
        if not self.sz.tabs:
            return True
        hv, ht = self.sz.pairs_hit(v[None])[0], self.sz.pairs_hit(tm[None])[0]
        return not np.any(hv & ~ht)


def scenarios_consistent(inj, chk, X0, t0, T_cands, T_nodes, amounts, k, dt_prev):
    """As history.scenarios, with each warm-up record built from the first template of T_cands[i] (the card's genuine
    transactions, most recent first; T_cands[i][0] is the study's template) that makes it possible.
    Returns targets, warm-up records, their template node ids (B, k), repaired (B,) and impossible (B,) flags."""
    Xt, Xi = [], []
    tn = np.full((len(X0), k), -1, np.int64)
    repaired, bad = np.zeros(len(X0), bool), np.zeros(len(X0), bool)
    delay, gap0 = H.timeline(dt_prev, k)
    for i in range(len(X0)):
        n, mean, c24, s24 = inj.history(X0[i])
        a_t = inj.u(X0[i], "log_amt")
        seq = [amounts[i]] * k
        xt = inj.apply(X0[i], a_t, t0[i], n, mean, c24, s24, seq)
        for f in ("hour", "dow"):                       # the fraud keeps its own hour / dow
            if f in inj.ix:
                xt[inj.ix[f]] = X0[i][inj.ix[f]]
        Xt.append(xt)
        recs = []
        for j in range(k):
            tj = t0[i] + delay[i] - (k - j) * H.SPACING
            first = None
            for c, tm in enumerate(T_cands[i]):
                v0 = inj.apply(tm, amounts[i], tj, n, mean, c24, s24, seq[:j])
                if j == 0 and np.isfinite(gap0[i]):       # the first injection follows the last genuine transaction
                    inj.s(v0, "log_dt_prev", np.log1p(gap0[i]))
                v = flags_follow(inj, v0.copy())
                if first is None:
                    first = v
                if chk.possible(v, tm):
                    repaired[i] |= (c > 0) or not np.array_equal(v, v0)
                    tn[i, j] = T_nodes[i][c]
                    break
            else:
                bad[i] = True
                v = first
                tn[i, j] = T_nodes[i][0]
            recs.append(v)
        Xi.append(np.stack(recs) if k else np.empty((0, X0.shape[1]), np.float32))
    return np.stack(Xt), Xi, tn, repaired, bad


def graph_logits_c(m, dataset, Xt, Xi, ids, k, delay, tmpl_nodes):
    """history.graph_logits, except that each injected node takes the relation-2 neighbours of its own template."""
    import torch

    from fraudadv.models import graph as G

    base = G.context(dataset)
    N = base["Z"].shape[0]
    B = len(ids)
    nb1, nb2, T = base["nbr1"], base["nbr2"], base["time"]
    K1 = nb1.shape[1]
    tgt = torch.as_tensor(ids)
    inj_ids = N + torch.arange(B * k).view(B, k)
    clone_ids = N + B * k + torch.arange(B)
    Z_new = torch.from_numpy(np.concatenate([np.concatenate(Xi), Xt]).astype(np.float32))
    t0 = T[tgt] + torch.as_tensor(np.asarray(delay), dtype=T.dtype)
    t_inj = t0.unsqueeze(1) - H.SPACING * (k - torch.arange(k, dtype=T.dtype)).unsqueeze(0)
    orig1 = nb1[tgt]
    rows1 = [torch.cat([inj_ids[:, :j].flip(1), orig1[:, : K1 - j]], 1) for j in range(k)]
    clone1 = torch.cat([inj_ids.flip(1), orig1[:, : K1 - k]], 1)
    tnode = torch.as_tensor(tmpl_nodes, dtype=torch.long).to(nb2.device)
    nb2_inj = torch.where((tnode >= 0).unsqueeze(-1), nb2[tnode.clamp(min=0)], nb2[tgt].unsqueeze(1))   # (B, k, K2)
    nb1_new = torch.cat([torch.stack(rows1, 1).view(B * k, K1), clone1]).to(nb1.dtype)
    nb2_new = torch.cat([nb2_inj.reshape(B * k, -1), nb2[tgt]]).to(nb2.dtype)
    aug = {"Z": torch.cat([base["Z"], Z_new.to(base["Z"].device)]), "nbr1": torch.cat([nb1, nb1_new]),
           "nbr2": torch.cat([nb2, nb2_new]), "time": torch.cat([T, t_inj.reshape(-1).to(T.dtype), t0])}
    G._CTX[dataset] = aug
    try:
        lt = m.logit(Xt, clone_ids.numpy())
        li = m.logit(np.concatenate(Xi), inj_ids.reshape(-1).numpy()).reshape(B, k)
        return lt, li.max(1)
    finally:
        G._CTX[dataset] = base


def run(dataset, families, tag="", n_sub=0):
    """n_sub > 0: a random n_sub frauds of the sample (as fraudadv.attacks.history)."""
    if any(f in H.FMS for f in families):
        import chronos  # noqa: F401  before TensorFlow (ivis)
    from fraudadv.attacks.run_attacks import context_ids, load_model
    from fraudadv.data.load import load
    from fraudadv.models import graph as G

    a, meta = load(dataset)
    main = attack_dir(dataset)
    out = attack_dir(dataset, consistent=True)
    out.mkdir(parents=True, exist_ok=True)
    tau = json.loads((main / "tau.json").read_text())
    S = np.load(main / "sample.npz")["S"]
    if n_sub > 0:
        S = S[np.sort(np.random.default_rng(1).permutation(len(S))[:n_sub])]
    keep = context_ids(dataset)
    ids = keep["test"][S]
    X0, amt = a["X_test"][S], a["amt_test"][S].astype(float)
    t0 = a["t_test"][S] if "t_test" in a else np.asarray(G.context(dataset, cpu=True)["time"][ids])
    inj = H.Injector(meta)
    chk = Checker(dataset)
    ctx = G.context(dataset, cpu=True)
    nbr = ctx["nbr1"][ids].numpy()                       # the card's earlier transactions, most recent first
    last = nbr[:, 0]
    Tc = np.asarray(ctx["time"], np.float64)
    dt_prev = np.where(last >= 0, Tc[ids] - Tc[np.maximum(last, 0)], np.inf)
    Zc = ctx["Z"].numpy() if hasattr(ctx["Z"], "numpy") else np.asarray(ctx["Z"])
    tmpl = np.where((last >= 0)[:, None], Zc[np.maximum(last, 0)], X0)
    T_nodes = [[int(q) for q in r if q >= 0] or [-1] for r in nbr]
    T_cands = [Zc[r] if r[0] >= 0 else X0[i][None] for i, r in enumerate(T_nodes)]
    hist = [inj.history(x) for x in X0]
    a_t = np.array([inj.u(x, "log_amt") for x in X0])
    mean_prev = np.array([h[1] if h[0] > 0 else a_t[i] for i, h in enumerate(hist)])
    a_last = np.array([inj.u(tmpl[i], "log_amt") if last[i] >= 0 else a_t[i] for i in range(len(X0))])
    cands = {"mean_prev": mean_prev, "last": a_last, "test_2usd": np.full(len(X0), np.log1p(2.0)), "own": a_t}
    built = {}                                           # the scenarios do not depend on the model
    for kk in H.K_LIST:
        for c, amts in cands.items():
            built[kk, c] = scenarios_consistent(inj, chk, X0, t0, T_cands, T_nodes, amts, kk, dt_prev)
            Xt, Xi, tn, rep, bad = built[kk, c]
            print(dataset, "k", kk, c, "frauds with a repaired warm-up record", int(rep.sum()), "impossible", int(bad.sum()),
                  "of", len(X0), flush=True)
    p = out / f"history_consistent{tag}.csv"
    rows = pd.read_csv(p).to_dict("records") if p.exists() else []
    have = {r["model"] for r in rows}
    pp = out / f"history_per_example_consistent{tag}.npz"
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
        for kk in H.K_LIST:
            flagged, L, nrep, nbad = {}, {}, {}, {}
            for c in cands:
                Xt, Xi, tn, rep, bad = built[kk, c]
                delay = H.timeline(dt_prev, kk)[0]
                if fam in H.GRAPH:
                    lt, li = graph_logits_c(m, dataset, Xt, Xi, ids, kk, delay, tn)
                elif fam in H.FMS:
                    lt, li = H.fm_logits(m, dataset, Xt, Xi, ids, kk, delay)
                else:
                    lt = m.logit(Xt, ids)
                    li = m.logit(np.concatenate(Xi), np.repeat(ids, kk)).reshape(len(X0), kk).max(1)
                L[c] = np.where((li < t) & ~bad, lt, np.maximum(lt, l0))   # flagged or impossible warm-up: fails
                flagged[c] = int(np.sum(det & (li >= t)))
                nrep[c], nbad[c] = int(np.sum(det & rep)), int(np.sum(det & bad))
            best = np.min(np.stack(list(L.values())), 0)
            for variant, l in (("mimic (no queries)", L["mean_prev"]), ("best of 4 (4 queries)", best)):
                ev = det & (l < t)
                PE[f"{fam}|{variant}|{kk}"] = ev
                mim = variant.startswith("mimic")
                rows.append(dict(dataset=dataset, model=fam, mode="consistent", attack=variant, k=kk, n_detected=int(det.sum()),
                                 asr=float(ev.sum() / max(det.sum(), 1)),
                                 asr_usd=float(amt[ev].sum() / max(amt[det].sum(), 1e-9)),
                                 mean_logit_drop=float(np.mean((l0 - l)[det])) if det.any() else np.nan,
                                 injected_flagged=flagged["mean_prev"] if mim else int(sum(flagged.values())),
                                 repaired=nrep["mean_prev"] if mim else max(nrep.values()),
                                 impossible=nbad["mean_prev"] if mim else max(nbad.values())))
                print({q: (round(v, 3) if isinstance(v, float) else v) for q, v in rows[-1].items()
                       if q in ("model", "attack", "k", "asr", "repaired", "impossible")}, flush=True)
        pd.DataFrame(rows).to_csv(p, index=False)
        np.savez_compressed(pp, **PE)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("families")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-sub", type=int, default=0, help="attack a random subset of this many frauds (0: all)")
    args = ap.parse_args()
    run(args.dataset, args.families.split(","), tag=args.tag, n_sub=args.n_sub)
