"""Bring-your-own-model API (fraudadv/byom.py): the adapters on synthetic data, and one end-to-end run of the study's
attack protocol when ULB's processed arrays exist (skipped otherwise)."""
import numpy as np
import pytest

from fraudadv import byom
from fraudadv.paths import PROC


def _toy(n=400, d=6, seed=0):
    """Synthetic rows and labels from a noisy linear rule."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=n) > 0.8).astype(int)
    return X, y


def test_model_exposes_only_given_methods():
    """byom.Model has grad and tree_ensemble only when they are given, so the protocol picks attacks by access."""
    m = byom.Model(lambda X: X[:, 0])
    assert not hasattr(m, "grad") and not hasattr(m, "tree_ensemble")
    g = byom.Model(lambda X: X[:, 0], grad_fn=lambda X: np.eye(X.shape[1])[[0] * len(X)])
    X = np.ones((3, 4), np.float32)
    assert g.logit(X).shape == (3,) and g.grad(X).shape == (3, 4)


def test_sklearn_adapter_returns_logits():
    """from_sklearn returns the decision function (computed from probabilities)."""
    from sklearn.linear_model import LogisticRegression

    X, y = _toy()
    clf = LogisticRegression().fit(X, y)
    np.testing.assert_allclose(byom.from_sklearn(clf).logit(X), clf.decision_function(X), rtol=1e-3, atol=1e-3)  # via probabilities


def test_torch_adapter_gradient():
    """from_torch returns the logit and its gradient (the weights for a linear layer)."""
    torch = pytest.importorskip("torch")
    lin = torch.nn.Linear(6, 1)
    m = byom.from_torch(lin)
    X, _ = _toy(n=5)
    np.testing.assert_allclose(m.grad(X), np.repeat(lin.weight.detach().numpy(), 5, 0), rtol=1e-6)
    np.testing.assert_allclose(m.logit(X), (X @ lin.weight.detach().numpy().T).ravel() + lin.bias.item(), rtol=1e-5)


def test_xgboost_adapter_tree_ensemble_matches_margin():
    """from_xgboost exposes a TreeEnsemble whose margin equals the model's logit."""
    xgb = pytest.importorskip("xgboost")
    X, y = _toy()
    clf = xgb.XGBClassifier(n_estimators=20, max_depth=3, n_jobs=1).fit(X, y)
    m = byom.from_xgboost(clf)
    ens = m.tree_ensemble()
    np.testing.assert_allclose(ens.margin_fn(X), m.logit(X), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not (PROC / "ulb" / "arrays.npz").exists(), reason="ULB not prepared (python -m fraudadv.data.prepare ulb)")
def test_evaluate_end_to_end_on_ulb():
    """byom.evaluate with Square on 40 ULB frauds: the expected rows, rates in [0, 1], and a larger budget evades at
    least as often."""
    from sklearn.linear_model import LogisticRegression

    a, _ = byom.data("ulb")
    clf = LogisticRegression(max_iter=300).fit(a["X_train"][:20000], a["y_train"][:20000])
    frauds = np.flatnonzero(a["y_test"] == 1)[:40]
    res = byom.evaluate(byom.from_sklearn(clf), "ulb", eps=(0.1, 0.5), attacks=("square",), square_queries=30,
                        sample=frauds, verbose=False)
    assert set(res["attack"]) == {"score black-box (Square)", "worst case (all attacks)"}
    assert res["asr"].between(0, 1).all() and (res["n"] == 40).all()
    sq = res[res.attack.str.startswith("score")].set_index("eps")["asr"]
    assert sq[0.5] >= sq[0.1] - 1e-9      # a larger budget evades at least as often
