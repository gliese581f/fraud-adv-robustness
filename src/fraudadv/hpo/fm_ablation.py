"""What does the frozen encoder add? The foundation-model head trained without the encoding.

Chronos-2 and TimesFM 2.5 are used as frozen encoders of the card's amount and time-gap series. The classifier
is an MLP head on [tabular features, projected encoding] (fraudadv.models.foundation), trained on all frauds
and a sample of legitimate rows (train_context.fm_subset). Comparing that model with the MLP trained on the
whole training split mixes two differences, the encoding and the training rows. This ablation isolates the
encoding: the same head, the same tuned hyperparameters, the same training, validation and test rows and the
same three seeds, with the encoding replaced by zeros, so the head sees the tabular features only.

Inputs:  data/processed/<dataset>/ (arrays.npz, ctx.npz); runs/<dataset>/<family>/best_params.json from
         train_context.
Outputs: runs/<dataset>/<family>/ablation_noemb.json (test metrics per seed, as in metrics.json). A family that
         already has it, or has no best_params.json, is skipped.

Results: Section VI-F (foundation models: what the encoding adds over the head alone).

usage: python -m fraudadv.hpo.fm_ablation <dataset> [chronos2,timesfm]
"""
from __future__ import annotations

import json
import sys

import numpy as np

SEEDS = (0, 1, 2)


class _ZeroEncoder:
    """Stands in for the frozen backbone: every series encodes to the same zero vector."""

    def embed(self, amt, gap, mask):
        import torch

        return torch.zeros(amt.shape[0], 8)


def run(dataset, families=("chronos2", "timesfm")):
    """Retrain each family's tuned head on zero encodings with seeds 0, 1, 2 and write ablation_noemb.json."""
    import fraudadv.models.foundation as F
    from fraudadv.attacks.run_attacks import context_ids
    from fraudadv.data.load import load
    from fraudadv.eval.metrics import all_metrics, threshold_at_fpr
    from fraudadv.hpo.train_all import RUNS
    from fraudadv.hpo.train_context import fm_subset, make

    F.backbone = lambda name: _ZeroEncoder()          # every model below gets the zero encoder; no backbone is loaded
    a, meta = load(dataset)
    keep = context_ids(dataset)
    itr, _ = fm_subset(a, "train")
    iva, wva = fm_subset(a, "val")
    ite, wte = fm_subset(a, "test")
    Xtr, ytr, ctr = a["X_train"][itr], a["y_train"][itr], keep["train"][itr]
    Xva, yva, cva = a["X_val"][iva], a["y_val"][iva], keep["val"][iva]
    Xte, yte, cte, ate = a["X_test"][ite], a["y_test"][ite], keep["test"][ite], a["amt_test"][ite]
    for fam in families:
        out = RUNS / dataset / fam
        p = out / "ablation_noemb.json"
        if p.exists() or not (out / "best_params.json").exists():
            continue
        best = json.loads((out / "best_params.json").read_text())["params"]
        mets = []
        for seed in SEEDS:
            F._EMB_CACHE.clear()                       # no encoding cached by an earlier model is reused
            m = make(fam, dataset, meta).fit(Xtr, ytr, Xva, yva, wva, best, ctr, cva, seed=seed)
            thr = threshold_at_fpr(yva, m.logit(Xva, cva), wva, 0.01)
            met = all_metrics(yte, m.logit(Xte, cte), wte, ate, thr)
            met.update(seed=seed, threshold=thr, eval_rows=int(len(ite)), ablation="no encoding")
            mets.append(met)
            print(dataset, fam, "no encoding", seed, {k: round(v, 4) for k, v in met.items() if isinstance(v, float)}, flush=True)
        p.write_text(json.dumps(mets, indent=1))


if __name__ == "__main__":
    run(sys.argv[1], tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("chronos2", "timesfm"))
