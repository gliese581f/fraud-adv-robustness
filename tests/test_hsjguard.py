"""The guard around ART's HopSkipJump (attacks/hsjguard.py): it stops the uncapped step-size search, makes the
random start repeatable from np.random.seed, and changes nothing else. Also checks that the study's wrapper
(blackwhite.hsj) gives the same points for the same seed."""
import numpy as np
import pytest

pytest.importorskip("art")


def _attack(predict, x, seed=0):
    """ART's HopSkipJump on a black-box classifier over [-3, 3]^d, seeded through np.random.seed."""
    from art.attacks.evasion import HopSkipJump
    from art.estimators.classification import BlackBoxClassifier

    np.random.seed(seed)
    clf = BlackBoxClassifier(predict, input_shape=(x.shape[1],), nb_classes=2, clip_values=(-3.0, 3.0))
    atk = HopSkipJump(clf, targeted=False, norm=np.inf, max_iter=5, max_eval=200, init_eval=20, init_size=50, verbose=False)
    return atk.generate(x, y=np.array([[0, 1]] * len(x)))


def _linear(Z):
    """A linear classifier with hard decisions."""
    p = (np.asarray(Z) @ np.array([1.0, -0.5, 0.25, 0.0]) > 0.3).astype(float)       # class 1 = flagged
    return np.stack([1 - p, p], 1)


def test_guard_leaves_normal_searches_alone_and_makes_them_repeatable(monkeypatch):
    """Without the guard two runs differ; with it the same seed repeats, another seed differs, nothing is stopped,
    and removing the cap gives the same result."""
    from fraudadv.attacks import hsjguard

    x = np.array([[1.0, -1.0, 0.5, 0.2], [2.0, 0.0, 1.0, -1.0]], np.float32)
    hsjguard.uninstall()
    assert not np.array_equal(_attack(_linear, x), _attack(_linear, x))       # the library's start is unseeded
    hsjguard.install()
    n0 = hsjguard.STATS["stuck"]
    out = _attack(_linear, x)
    assert np.array_equal(out, _attack(_linear, x)) and not np.array_equal(out, _attack(_linear, x, seed=1))
    assert hsjguard.STATS["stuck"] == n0 and (_linear(out)[:, 1] == 0).all()
    monkeypatch.setattr(hsjguard, "CAP", 10**9)                                # no cap: same result, the cap was never reached
    assert np.array_equal(out, _attack(_linear, x))
    assert np.random.RandomState is hsjguard._ORIG["rs"]


def test_guard_stops_a_search_that_cannot_succeed():
    """After 150 queries the model flags everything, so no step size is adversarial: the unpatched attack loops forever."""
    from fraudadv.attacks import hsjguard

    calls, seen = [0], []

    def predict(Z):
        calls[0] += len(Z)
        if calls[0] > 150:
            return np.stack([np.zeros(len(Z)), np.ones(len(Z))], 1)
        out = _linear(Z)
        if len(Z) == 1 and out[0, 1] == 0:
            seen.append(np.array(Z[0]))
        return out

    x = np.array([[1.0, -1.0, 0.5, 0.2]], np.float32)
    hsjguard.install()
    n0 = dict(hsjguard.STATS)
    out = _attack(predict, x)
    assert hsjguard.STATS["stuck"] == n0["stuck"] + 1 and hsjguard.STATS["recovered"] == n0["recovered"] + 1
    assert _linear(out)[0, 1] == 0                                   # an adversarial point is still returned
    d = [np.abs(np.clip(z, -3, 3) - x[0]).max() for z in seen]
    assert np.abs(out[0] - x[0]).max() <= min(d) + 1e-6              # and it is the closest one the attack had verified


def test_guard_returns_the_input_when_nothing_was_verified():
    """A stopped search that never verified an adversarial point returns the input."""
    from fraudadv.attacks import hsjguard

    calls = [0]

    def predict(Z):                                                   # one adversarial answer (the random start), then never again
        calls[0] += len(Z)
        p = np.ones(len(Z)) if calls[0] > 2 else np.r_[np.ones(len(Z) - 1), 0.0] if calls[0] == 2 else np.ones(len(Z))
        return np.stack([1 - p, p], 1)

    x = np.array([[1.0, -1.0, 0.5, 0.2]], np.float32)
    hsjguard.install()
    n0 = dict(hsjguard.STATS)
    out = _attack(predict, x)
    assert np.array_equal(out, x)
    assert hsjguard.STATS["recovered"] == n0["recovered"]


def test_install_leaves_a_class_alone_that_another_guard_has_patched(monkeypatch):
    """When another copy of the guard has patched the class, install() does nothing."""
    from art.attacks.evasion import HopSkipJump as H

    from fraudadv.attacks import hsjguard

    hsjguard.uninstall()

    def other(self, *a, **k):                                      # stands for hyperdef.hsjguard's wrapper
        raise AssertionError

    monkeypatch.setattr(H, "_perturb", other)
    hsjguard.install()
    assert not hsjguard._ORIG and H._perturb is other
    monkeypatch.undo()
    hsjguard.install()
    assert hsjguard._ORIG


def test_wrapper_is_repeatable():
    """fraudadv's own wrapper: same seed, same points; another seed, other points."""
    from fraudadv.attacks.blackwhite import hsj

    class Cons:
        mutable = np.arange(4)
        lower, upper = np.full(4, -3.0, np.float32), np.full(4, 3.0, np.float32)

        @staticmethod
        def project(F, x0):
            return np.clip(F, -3.0, 3.0)

    class Model:
        @staticmethod
        def logit(F, ctx=None):
            return 4.0 * (np.asarray(F) @ np.array([1.0, -0.5, 0.25, 0.0]) - 0.3)

    x = np.array([[1.0, -1.0, 0.5, 0.2], [2.0, 0.0, 1.0, -1.0], [0.9, 0.1, 0.0, 0.0]], np.float32)
    a, la = hsj(Model, x, 0.0, Cons, max_iter=5, max_eval=200, init_eval=20)
    b, _ = hsj(Model, x, 0.0, Cons, max_iter=5, max_eval=200, init_eval=20)
    c, _ = hsj(Model, x, 0.0, Cons, max_iter=5, max_eval=200, init_eval=20, seed=1)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and (la < 0).all()
