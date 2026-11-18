"""Share of the study's adversarial examples that no real transaction could produce (Table S-XIII, the consistent-attacker appendix).

For each dataset it applies the realizability check (fraudadv.attacks.realizable.violations_all: the hand-written
rules plus the structural zeros learned from the training data) to
  - every test record, legitimate and fraud (row "real": the false-positive cost of the check), and
  - the successful adversarial examples stored by the attack grid (adv_<mode>_<family>.npz: the high-confidence
    white-box examples at eps 0.5, margin 2, that serve as transfer sources), for the unconstrained and
    constrained attackers (runs/<d>/attacks) and the consistent attacker (runs_consistent/<d>/attacks).
An example counts as impossible only if it breaks a rule its original record did not.

  python -m fraudadv.report.realizable_share   ->   reports/analysis/consistent/realizable.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraudadv.attacks.realizable import violations_all
from fraudadv.data.load import load
from fraudadv.paths import REPORTS, RUNS, RUNS_CONSISTENT

DATASETS = ("ulb", "ieee_cis", "sparkov", "tabformer", "paysim", "baf")
MODES = (("unconstrained", RUNS), ("constrained", RUNS), ("consistent", RUNS_CONSISTENT))


def test_records_row(dataset, a):
    """Share of real test records the check would call impossible."""
    v = violations_all(dataset, a["X_test"])
    return dict(dataset=dataset, mode="real", model="test records", n=len(v), impossible=int(v.sum()),
                share=float(v.mean()), share_legit=float(v[a["y_test"] == 0].mean()),
                share_fraud=float(v[a["y_test"] == 1].mean()))


def adversarial_rows(dataset, mode, runs_dir):
    """One row per stored adversarial-example file of this attacker: the share of its successful examples that
    break a rule the original record did not break."""
    rows = []
    for f in sorted((runs_dir / dataset / "attacks").glob(f"adv_{mode}_*.npz")):
        z = np.load(f)
        ok = z["ok"].astype(bool) & z["det"].astype(bool)          # detected before, evading after
        if not ok.any():
            continue
        new = violations_all(dataset, z["Xadv"][ok])
        orig = violations_all(dataset, z["X0"][ok])
        rows.append(dict(dataset=dataset, mode=mode, model=f.stem.split(f"adv_{mode}_")[1], n=int(ok.sum()),
                         impossible=int((new & ~orig).sum()), share=float((new & ~orig).mean()),
                         share_original=float(orig.mean())))
    return rows


def main():
    rows = []
    for d in DATASETS:
        a, _ = load(d)
        rows.append(test_records_row(d, a))
        for mode, runs_dir in MODES:
            rows += adversarial_rows(d, mode, runs_dir)
    T = pd.DataFrame(rows)
    out = REPORTS / "analysis" / "consistent"
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "realizable.csv", index=False)
    print(T.round(3).to_string())
    print(T[T["mode"] != "real"].groupby(["dataset", "mode"])["share"].agg(["min", "max", "median"]).round(3).to_string())


if __name__ == "__main__":
    main()
