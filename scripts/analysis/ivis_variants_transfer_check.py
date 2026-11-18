"""Extra check on the neighbour votes: do transferred attacks find evasions the grid missed?

For each ivis variant (seed 0) and threat setting, replay the study's eps-0.5 high-confidence white-box examples of
MLP, FT-Transformer and ivis (runs/<d>/attacks/adv_<mode>_<source>.npz, kappa 2) against the variant, and count the
detected frauds they evade that the variant's own eps-0.5 worst case (grid plus adaptive attacks) did not.
Prints one line per dataset, model and setting; writes nothing. A robustness check for the ivis variants
(Tables S-XXV to S-XXVII report the grid and adaptive results; this confirms transfer adds little).

usage (from the repository root): PYTHONPATH=src python scripts/analysis/ivis_variants_transfer_check.py [datasets]
       default datasets: sparkov, tabformer, baf, ieee_cis
"""
import sys

sys.path.insert(0, "scripts/analysis")   # run from the repository root with PYTHONPATH=src
import ivis_variants as V  # noqa: E402
import numpy as np  # noqa: E402
from fraudadv.data.load import load  # noqa: E402
from fraudadv.eval.metrics import threshold_at_fpr  # noqa: E402

DATASETS = ["sparkov", "tabformer", "baf", "ieee_cis"]
MODELS = ["ivis_base_knn", "ivis_sup_knn", "ivis_smooth_knn", "ivis_smooth_lr"]
SOURCES = ("mlp", "ft_transformer", "ivis")


def worst_case_with_adaptive(d, fam):
    """The variant's per-example grid outcomes, with 'evaded' also set where an adaptive attack evaded."""
    pe = dict(np.load(V.OUT / d / "attacks" / f"per_example_full_{fam}.npz"))
    ad = V.OUT / d / "attacks" / f"adaptive_{fam}.npz"
    if ad.exists():
        z = np.load(ad)
        for k in z.files:
            pe[k] = pe[k].copy()
            pe[k][:, 2] = np.maximum(pe[k][:, 2], z[k][:, 2])
    return pe


def main(datasets):
    for d in datasets:
        a, _ = load(d)
        for fam in MODELS:
            m = V.load_variant(d, fam)
            tau = threshold_at_fpr(a["y_val"], m.logit(a["X_val"]), a["w_val"], 0.01)
            pe = worst_case_with_adaptive(d, fam)
            for mode in ("constrained", "unconstrained"):
                g = pe[f"{fam}|{mode}|0.5"]
                det, ev = g[:, 1].astype(bool), g[:, 2].astype(bool)
                extra = np.zeros(len(g), bool)            # evaded by at least one transferred example
                for src in SOURCES:
                    z = np.load(V.RUNS / d / "attacks" / f"adv_{mode}_{src}.npz")
                    assert np.array_equal(z["S"], g[:, 0].astype(int))
                    extra |= m.logit(z["Xadv"]) < tau
                print(d, fam, mode, f"grid@0.5 {ev[det].mean():.3f}", f"transfer adds {int((extra & det & ~ev).sum())} of {int(det.sum())}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else DATASETS)
