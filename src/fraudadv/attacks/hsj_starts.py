"""HopSkipJump from several seeded starts: how much does the decision-based result depend on the run?

ART's HopSkipJump searches from a random starting point. The grid ran it once per model and threat model, and the
library drew that start from a generator the attack's seed does not reach (see attacks/hsjguard.py), so the grid's
HopSkipJump rows are one draw that cannot be repeated exactly. For every model of a dataset, on the detected frauds
of the shared fraud sample, this script runs the attack from n_starts seeded starts (seeds 0, 1, ...; with the guard
installed, so each start is repeatable) in each threat model, and reports for each budget eps of the grid:
  * the success rate of every start (how far single runs differ): min, mean and max over the starts;
  * the success rate of their union (a decision-only attacker who restarts) and of their intersection;
  * the grid's own HopSkipJump, Square and worst-case success rates on the same frauds;
  * the evasions the starts add to the grid's worst case (the attack-strength check for this attack; APGD and
    Square have theirs in budget_check.py).
Foundation models are skipped: the grid ran no decision-based attack on them.

Inputs (runs/<dataset>/attacks/): sample.npz (the shared fraud sample S), attack_scores_full_*.npz (the grid's
per-fraud attacked logits, keys "<family>|<mode>|<attack>|<eps>", optional "<family>|<mode>|<attack>|rows",
"<family>|tau", "<family>|clean"), and the trained models. A model whose clean logits no longer give the grid's
detected frauds is skipped with a message.
Outputs: runs/<dataset>/attacks/hsj_starts<tag>.csv (one row per model, threat model and eps) and
hsj_starts_per_example<tag>.npz: under "<family>|<mode>" the minimal L-inf perturbation found by each start
(starts x detected frauds, inf where the start failed), under "<family>|<mode>|rows" the positions of those frauds
in the sample. Both are saved after every (model, threat model) and the run resumes from them. With --tag, a second
process on the same dataset writes its own files.

Result: Table S-XLII (supplement, five-start HopSkipJump).

usage: python -m fraudadv.attacks.hsj_starts <dataset> fam1,fam2,... [n_starts] [--tag _x]
                                             [--modes unconstrained,constrained]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

N_STARTS = 5
# attack names in the grid's outputs
WORST, SQUARE, HSJ = "worst case (all attacks)", "score black-box (Square)", "decision black-box (HopSkipJump)"


def main(dataset, families, n_starts=N_STARTS, tag="", modes=("unconstrained", "constrained")):
    """Run n_starts seeded HopSkipJump starts for `families` of `dataset` in each threat model of `modes`."""
    from fraudadv.attacks import hsjguard
    from fraudadv.attacks.blackwhite import hsj
    from fraudadv.attacks.constraints import Constraints
    from fraudadv.attacks.run_attacks import EPS, FM, RUNS, context_ids, load_model
    from fraudadv.data.load import load

    a, meta = load(dataset)
    out = RUNS / dataset / "attacks"
    keep = context_ids(dataset)
    S = np.load(out / "sample.npz")["S"]
    sc = {}
    for f in sorted(out.glob("attack_scores_full_*.npz")):
        z = np.load(f)
        sc.update({k: z[k] for k in z.files if k != "S"})
    p, pp = out / f"hsj_starts{tag}.csv", out / f"hsj_starts_per_example{tag}.npz"
    rows = pd.read_csv(p).to_dict("records") if p.exists() else []
    pe = dict(np.load(pp)) if pp.exists() else {}
    have = {(r["model"], r["mode"]) for r in rows}
    X0, c0 = a["X_test"][S], keep["test"][S]          # the shared fraud sample and its context ids

    def evaded_in_grid(fam, mode, attack, e):
        """Per sample row: did `attack` evade at budget e in the stored grid (False where it did not run on the row)."""
        v, r = sc[f"{fam}|{mode}|{attack}|{e}"], sc.get(f"{fam}|{mode}|{attack}|rows")
        if r is None:                                     # scores for every sample row
            assert len(v) == len(S)
            return v < float(sc[f"{fam}|tau"])
        s = np.full(len(S), np.inf)                       # scores for some rows only: the others did not evade
        s[r] = v
        return s < float(sc[f"{fam}|tau"])

    for fam in families:
        if fam in FM or f"{fam}|tau" not in sc:
            continue
        m = None
        for mode in modes:
            if (fam, mode) in have or f"{fam}|{mode}|{HSJ}|{EPS[0]}" not in sc:
                continue
            m = m or load_model(dataset, fam, 0)             # loaded on first use: a finished model is not loaded
            t = float(sc[f"{fam}|tau"])
            l0 = m.logit(X0, c0)
            det = l0 > t
            if not np.array_equal(det, sc[f"{fam}|clean"] > t):
                print(dataset, fam, mode, "skipped: the model does not reproduce the grid's detected frauds", flush=True)
                continue
            idx = np.flatnonzero(det)                     # positions of the detected frauds in the sample
            cons = Constraints(meta, a["X_train"], mode)
            t0, stuck0 = time.time(), hsjguard.STATS["stuck"]
            # minimal L-inf perturbation found by each start, per detected fraud (inf: the start did not evade)
            dist = np.full((n_starts, len(idx)), np.inf, np.float32)
            for s in range(n_starts):
                if len(idx):
                    Xh, lh = hsj(m, X0[idx], t, cons, ctx=c0[idx], seed=s)
                    dist[s] = np.where(lh < t, cons.linf(Xh, X0[idx]), np.inf)
            pe[f"{fam}|{mode}"], pe[f"{fam}|{mode}|rows"] = dist, idx.astype(np.int32)
            n = len(idx)
            for e in EPS:                                 # the grid's budgets
                ok = dist <= e + 1e-6                     # evaded within e: starts x detected frauds
                per, union = ok.mean(1) if n else np.full(n_starts, np.nan), ok.any(0)    # per start; any start
                g_h, g_s, g_w = (evaded_in_grid(fam, mode, k, e)[idx] for k in (HSJ, SQUARE, WORST))
                rows.append(dict(dataset=dataset, model=fam, mode=mode, eps=e, n_detected=n, starts=n_starts,
                                 asr_start_min=float(per.min()), asr_start_mean=float(per.mean()), asr_start_max=float(per.max()),
                                 asr_union=float(union.mean()) if n else np.nan, asr_all_starts=float(ok.all(0).mean()) if n else np.nan,
                                 asr_hsj_grid=float(g_h.mean()) if n else np.nan, asr_square_grid=float(g_s.mean()) if n else np.nan,
                                 asr_worst_grid=float(g_w.mean()) if n else np.nan,
                                 asr_worst_with_starts=float((g_w | union).mean()) if n else np.nan,
                                 new_evasions=int((union & ~g_w).sum()), new_evasions_one_start_max=int((ok & ~g_w).sum(1).max()) if n else 0,
                                 stuck=int(hsjguard.STATS["stuck"] - stuck0), seconds=round(time.time() - t0)))
            r = next(r for r in rows[::-1] if r["eps"] == 0.25)
            print(dataset, fam, mode, "detected", n, "| eps 0.25: starts", round(r["asr_start_min"], 3), "to", round(r["asr_start_max"], 3),
                  "union", round(r["asr_union"], 3), "| grid HSJ", round(r["asr_hsj_grid"], 3), "Square", round(r["asr_square_grid"], 3),
                  "worst", round(r["asr_worst_grid"], 3), "new", r["new_evasions"], "| in", r["seconds"], "s", flush=True)
            np.savez_compressed(pp, **pe)
            pd.DataFrame(rows).to_csv(p, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HopSkipJump from several seeded starts (Table S-XLII).")
    ap.add_argument("dataset")
    ap.add_argument("families", help="comma-separated model families")
    ap.add_argument("n_starts", nargs="?", type=int, default=N_STARTS, help="number of seeded starts")
    ap.add_argument("--tag", default="", help="suffix of the output files (a second process on the same dataset)")
    ap.add_argument("--modes", default="unconstrained,constrained", help="comma-separated threat models")
    args = ap.parse_args()
    main(args.dataset, args.families.split(","), args.n_starts, tag=args.tag, modes=tuple(args.modes.split(",")))
