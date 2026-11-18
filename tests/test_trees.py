"""Tree-ensemble attacks (attacks/trees.py) and exact boundary thickness (boundary/thickness.py) on a small XGBoost
model fitted to synthetic data: the margin view matches XGBoost, AATS evades and leaves immutable features alone,
and the exact thickness of tree ensembles agrees with dense sampling along the same segments."""
import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from fraudadv.attacks.trees import AATS, TreeEnsemble
from fraudadv.boundary.thickness import thickness_exact_trees, thickness_sampled


@pytest.fixture(scope="module")
def fitted():
    """150 trees of depth 4 on 4,000 synthetic rows with 8 features."""
    rng = np.random.default_rng(0)
    n, d = 4000, 8
    X = rng.normal(size=(n, d)).astype(np.float32)
    logit = 2.5 * X[:, 0] - 2 * X[:, 1] + X[:, 2] * X[:, 3] - 4.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    model = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1).fit(X, y)
    return model, X, y


def test_margin_matches_xgboost(fitted):
    """TreeEnsemble.margin_fn gives XGBoost's raw margin."""
    model, X, _ = fitted
    ens = TreeEnsemble.from_xgboost(model)
    np.testing.assert_allclose(ens.margin_fn(X[:50]), model.predict(X[:50], output_margin=True), rtol=1e-5)


def test_aats_evades_flagged_frauds(fitted):
    """Without a budget, AATS evades more than 90% of flagged rows, and every success is below 0.5 on XGBoost."""
    model, X, _ = fitted
    ens = TreeEnsemble.from_xgboost(model)
    flagged = X[model.predict_proba(X)[:, 1] > 0.5][:30]
    res = AATS(ens, norm="linf").generate(flagged)
    assert res.success.mean() > 0.9
    assert (model.predict_proba(res.x_adv[res.success])[:, 1] < 0.5).all()


def test_aats_respects_immutable_features(fitted):
    """Features outside `mutable` keep their values."""
    model, X, _ = fitted
    ens = TreeEnsemble.from_xgboost(model)
    flagged = X[model.predict_proba(X)[:, 1] > 0.5][:20]
    res = AATS(ens, mutable=[1, 2, 3, 4]).generate(flagged)
    np.testing.assert_array_equal(res.x_adv[:, [0, 5, 6, 7]], flagged[:, [0, 5, 6, 7]])


def test_exact_thickness_matches_sampling(fitted):
    """thickness_exact_trees (threshold crossings) agrees with 4,096-point sampling along each segment."""
    model, X, _ = fitted
    ens = TreeEnsemble.from_xgboost(model)
    flagged = X[model.predict_proba(X)[:, 1] > 0.5][:10]
    res = AATS(ens).generate(flagged)
    xr, xs = flagged[res.success], res.x_adv[res.success]
    # push the partner point further out so that the segment spans the whole boundary band
    xs = xr + 3 * (xs - xr)
    exact = thickness_exact_trees(ens, xr, xs)
    sampled = thickness_sampled(lambda Z: model.predict_proba(Z.astype(np.float32))[:, 1], xr, xs, n_steps=4096)
    np.testing.assert_allclose(exact, sampled, atol=0.02 * np.linalg.norm(xr - xs, axis=1).max())
