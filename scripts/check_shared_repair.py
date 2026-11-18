"""Check that the library's consistent attacker is the one shared with the project's other papers.

The repair and the realizability check live in the library (fraudadv.attacks.consistent, fraudadv.attacks.realizable).
The project's other papers use the shared copy in vendor/decoydef (pinned by sha256 prefix below). This script
projects the same random perturbations of test frauds with both and requires identical arrays: the projected rows,
their L-inf distance and the realizability flags, at several budgets.

  python scripts/check_shared_repair.py [datasets]        default: ieee_cis sparkov paysim
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "vendor"
PINNED = {"consistent.py": "602f7eb3f2b6", "consistency.py": "5996da32d6e1"}
sys.path.insert(0, str(VENDOR))

from fraudadv.attacks.constraints import Constraints  # noqa: E402
from fraudadv.attacks.realizable import violations_all  # noqa: E402
from fraudadv.data.load import load  # noqa: E402


def check_pins():
    for name, prefix in PINNED.items():
        got = hashlib.sha256((VENDOR / "decoydef" / name).read_bytes()).hexdigest()[:12]
        if got != prefix:
            raise SystemExit(f"vendor/decoydef/{name} is {got}, pinned {prefix}: the shared repair changed")


def shared_project(dataset, constrained, X_train):
    """The shared repair wrapped around the study's constrained projection, as the other papers use it."""
    from decoydef.consistent import ConsistentConstraints

    return ConsistentConstraints(dataset, constrained, X_train)


def main(datasets):
    check_pins()
    from decoydef.consistency import violations_all as shared_violations

    for d in datasets:
        a, meta = load(d)
        ours = Constraints(meta, a["X_train"], "consistent")
        theirs = shared_project(d, Constraints(meta, a["X_train"], "constrained"), a["X_train"])
        rng = np.random.default_rng(0)
        frauds = np.flatnonzero(a["y_test"] == 1)
        X0 = a["X_test"][np.sort(rng.choice(frauds, size=min(400, len(frauds)), replace=False))].astype(np.float64)
        same = True
        for eps in (0.1, 0.25, 1.0):
            for scale in (0.5, 3.0):
                X = X0 + rng.normal(0, scale * eps, X0.shape)
                P, Q = ours.project(X.copy(), X0, eps), theirs.project(X.copy(), X0, eps)
                same &= np.array_equal(P, Q) and np.array_equal(ours.linf(P, X0), np.abs(Q - X0)[:, ours.mutable].max(1))
                same &= np.array_equal(violations_all(d, P), shared_violations(d, Q))
        same &= np.array_equal(violations_all(d, a["X_test"][:20000]), shared_violations(d, a["X_test"][:20000]))
        print(d, "identical to the shared repair" if same else "DIFFERS from the shared repair", flush=True)
        if not same:
            raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["ieee_cis", "sparkov", "paysim"])
