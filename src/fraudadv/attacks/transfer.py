"""Transfer matrices: do one model's evasions also evade the others?

Every saved high-confidence white-box example (adv_<mode>_<family>.npz, written by fraudadv.attacks.run_attacks at
eps 0.5) is replayed on every other trained model at that model's 1%-FPR threshold:

    T[s, t] = among frauds the source s detects and its attack takes below s's threshold (source-successful),
              and that the target t detects on the clean point: the share that also evade t.
    N[s, t] = how many such frauds; T_margin, N_margin: the same with the stricter source condition
              logit < tau - KAPPA.

Models attacked in separate runs (e.g. the foundation models) still get one complete matrix.

usage: python -m fraudadv.attacks.transfer <dataset> [fam1,fam2,...] [--tag _x] [--modes unconstrained,constrained]
writes transfer_<mode><tag>.npz in the mode's attack folder (runs_consistent/ for the consistent attacker)
Results: Figure 1, Figure S3, Figure S4 and the transfer rates quoted in Section VI-D.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from fraudadv.attacks.run_attacks import KAPPA, context_ids, load_model
from fraudadv.paths import attack_dir

ORDER = ["xgboost", "catboost", "mlp", "ft_transformer", "ivis", "hgnn", "hgt", "chronos2", "timesfm"]


def matrix(dataset, mode, fams, tau, seed=0):
    """(T, N, T_margin, N_margin) for the families fams, from adv_<mode>_<family>.npz."""
    out = attack_dir(dataset, mode == "consistent")
    keep = context_ids(dataset)
    Z = {f: np.load(out / f"adv_{mode}_{f}.npz") for f in fams}
    S = Z[fams[0]]["S"]
    assert all(np.array_equal(Z[f]["S"], S) for f in fams), "attack samples differ between runs"
    ctx0 = keep["test"][S]
    models = {f: load_model(dataset, f, seed) for f in fams}
    X0 = Z[fams[0]]["X0"]
    det = {f: models[f].logit(X0, ctx0) > tau[f] for f in fams}
    T = np.full((len(fams), len(fams)), np.nan)
    N = np.zeros_like(T)
    Tm, Nm = T.copy(), N.copy()
    for i, s in enumerate(fams):
        src = np.flatnonzero(det[s] & Z[s]["ok"].astype(bool))      # detected by s, then evading it
        ls = np.full(len(S), np.inf)
        if len(src):
            ls[src] = models[s].logit(Z[s]["Xadv"][src], ctx0[src])
        ok = ls < tau[s]
        okm = ls < tau[s] - KAPPA
        for j, t in enumerate(fams):
            base = ok & det[t]
            N[i, j], Nm[i, j] = base.sum(), (okm & det[t]).sum()
            if base.any():
                ev = np.zeros(len(S), bool)
                ev[base] = models[t].logit(Z[s]["Xadv"][base], ctx0[base]) < tau[t]
                T[i, j] = float(ev[base].mean())
                if Nm[i, j]:
                    Tm[i, j] = float(ev[okm & det[t]].mean())
    return T, N, Tm, Nm


def run(dataset, seed=0, order=None, tag="", modes=("unconstrained", "constrained")):
    """order: families to include (default ORDER, the paper's nine); tag: suffix of the output files, so an
    extended matrix (e.g. with the LoRA and zero-shot foundation models) does not replace the paper's."""
    order = ORDER if order is None else order
    if any(f.startswith(("chronos2", "timesfm")) for f in order):
        import chronos  # noqa: F401  must be imported before TensorFlow (ivis): the reverse order aborts
    tau = json.loads((attack_dir(dataset) / "tau.json").read_text())
    for mode in modes:
        out = attack_dir(dataset, mode == "consistent")
        fams = [f for f in order if f in tau and (out / f"adv_{mode}_{f}.npz").exists()]
        if not fams:
            continue
        T, N, Tm, Nm = matrix(dataset, mode, fams, tau, seed)
        np.savez(out / f"transfer_{mode}{tag}.npz", T=T, N=N, T_margin=Tm, N_margin=Nm, families=np.array(fams))
        print(dataset, mode, fams, "\n", np.round(T, 2), "\nN\n", N.astype(int), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("families", nargs="?", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--modes", default="unconstrained,constrained")
    a = ap.parse_args()
    run(a.dataset, order=a.families.split(",") if a.families else None, tag=a.tag, modes=tuple(a.modes.split(",")))
