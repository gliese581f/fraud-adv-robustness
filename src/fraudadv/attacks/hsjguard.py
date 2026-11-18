"""Two repairs to ART's HopSkipJump (art 1.20, hop_skip_jump.py), installed by fraudadv.attacks.blackwhite.hsj.

The hypernetwork-defense study found both problems; it shares this guard with the main study so that the two
run the attack in the same way.

1. A step-size search without an end. After each gradient estimate HopSkipJump halves its step until the
   stepped point is adversarial (`while not success: epsilon /= 2`). The loop has no cap. It relies on the point
   it steps from being adversarial, which holds for a continuous model in exact arithmetic but not always here:
   the point is recomputed in float32 after the binary search and the model sees it through Constraints.project,
   so it can sit a hair on the wrong side of the boundary. If the estimated direction then does not cross the
   boundary either, the step underflows to zero and the attack never returns.

   install() patches the class once per process:
     * every single point HopSkipJump has verified to be adversarial is remembered, and the one closest to the
       original (L-inf, in the attack's own coordinates) is kept;
     * after CAP consecutive failed single-point checks the search for that example is stopped and the closest
       verified point is returned (the original input if none was ever verified).
   The checks themselves are passed through untouched, so a search that is not stopped runs exactly as in the
   library. STATS counts stopped searches ("stuck") and those that still returned an adversarial point
   ("recovered"). CAP: a binary search makes at most log2(range / theta) < 45 consecutive failed checks; after
   100 halvings the step is 1e-30 of the distance to the original, far below anything the model can resolve.

2. An unseeded start. The library draws the random starting points from a fresh np.random.RandomState()
   without a seed, so two runs with the same np.random.seed differ. While HopSkipJump looks for its starting
   point, RandomState() without a seed is now seeded from the global generator, which hsj() seeds.

Results. The paper's attack grid (runs/<dataset>/attacks/results_full_*.csv) was computed before this guard
existed: its HopSkipJump rows used unseeded starts and cannot be repeated bit for bit. No search of that grid was
stuck (a stuck search never returns, and every run finished). The five-start HopSkipJump check (hsj_starts.py,
Table S-XLII) runs with the guard. Table S-XLI (reports/table14_rerun_by_attack.csv) measures how much two
unseeded runs of the grid differ.

If another copy of this guard (hyperdef.hsjguard) has already patched the class in this process, install()
leaves it alone: two guards on top of each other would count every failed check twice.
"""
from __future__ import annotations

import numpy as np

CAP = 100                                  # consecutive failed single-point checks before a search is stopped
STATS = {"stuck": 0, "recovered": 0}
_ORIG = {}


class _Stuck(RuntimeError):
    """Raised inside the library's search to stop it; caught in _perturb."""


def install():
    """Patch ART's HopSkipJump class (three methods) once per process."""
    from art.attacks.evasion import HopSkipJump as H

    if _ORIG or not all(getattr(H, f).__module__ == H.__module__ for f in ("_adversarial_satisfactory", "_perturb", "_init_sample")):
        return                                           # already installed, here or by another copy of this guard
    sat, perturb, init = H._adversarial_satisfactory, H._perturb, H._init_sample
    _ORIG.update(sat=sat, perturb=perturb, init=init, rs=np.random.RandomState)

    def _adversarial_satisfactory(self, samples, target, clip_min, clip_max):
        """The library's check, plus: remember the closest verified point, count consecutive failures."""
        res = sat(self, samples, target, clip_min, clip_max)
        st = getattr(self, "_hd", None)
        if st is None:
            return res
        if len(samples) != 1:                            # the batch of the gradient estimate
            st["fail"] = 0
        elif res[0]:
            st["fail"] = 0
            z = np.clip(samples[0], clip_min, clip_max)
            d = float(np.max(np.abs(z - st["x"])))
            if d < st["d"]:
                st["d"], st["best"] = d, z.astype(st["x"].dtype)
        else:
            st["fail"] += 1
            if st["fail"] >= CAP:
                raise _Stuck()
        return res

    def _perturb(self, x, y, y_p, init_pred, adv_init, mask, clip_min, clip_max):
        """The library's attack on one example; a stopped search returns the closest verified point."""
        self._hd = dict(x=np.array(x), d=np.inf, best=None, fail=0)
        try:
            return perturb(self, x, y, y_p, init_pred, adv_init, mask, clip_min, clip_max)
        except _Stuck:
            best = self._hd["best"]
            STATS["stuck"] += 1
            STATS["recovered"] += best is not None
            return x if best is None else best
        finally:
            self._hd = None

    def _init_sample(self, *a, **k):
        """The library's search for a starting point, with unseeded RandomState() seeded from np.random."""
        rs, seed = np.random.RandomState, int(np.random.randint(2**31 - 1))

        class Seeded(rs):
            def __init__(self, *sa, **sk):
                super().__init__(*sa, **sk) if sa or sk else super().__init__(seed)

        np.random.RandomState = Seeded
        try:
            return init(self, *a, **k)
        finally:
            np.random.RandomState = rs

    H._adversarial_satisfactory, H._perturb, H._init_sample = _adversarial_satisfactory, _perturb, _init_sample


def uninstall():
    """Back to the library's own methods (tests)."""
    from art.attacks.evasion import HopSkipJump as H

    if _ORIG:
        H._adversarial_satisfactory, H._perturb, H._init_sample = _ORIG["sat"], _ORIG["perturb"], _ORIG["init"]
        _ORIG.clear()
