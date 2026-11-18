"""The evasion attacks of the attack grid: APGD, PGD, Square, HopSkipJump and AATS.

Every attack takes a model, a batch of frauds X0 (standardized features, one row per fraud), the deployed
threshold tau on the fraud logit, an L-inf budget eps and the threat model `cons`
(fraudadv.attacks.constraints.Constraints: which features the attacker controls, their ranges and valid
values, and the recomputation of derived features). It returns the attacked points and their logits. A fraud
evades when its logit falls below tau.

"Min-effort" runs (early_stop=True, the grid's setting) stop working on a fraud as soon as it evades: a fraudster
needs no more than that. "Max-confidence" runs (early_stop=False) keep lowering the score; the ranking-metric
attacker (robust_auc.py) uses them.

    apgd    Auto-PGD (Croce & Hein 2020), white-box: the grid's attack on every differentiable model.
    pgd     L-inf PGD with sign steps and a random start (Madry et al. 2018), white-box: makes the examples,
            pushed a margin kappa below the threshold, that transfer.py sends to the other models.
    square  Square Attack (Andriushchenko et al. 2020), score-based black-box; on tabular data a "square" is
            a random subset of the attacker's features.
    hsj     HopSkipJump (Chen et al. 2020), decision-based black-box, ART's implementation run one fraud at a
            time in the attacker's features (with the two repairs of hsjguard.py).
    aats    Axis-aligned threshold search for tree ensembles (trees.py), followed by an L-inf box search.

Gradients come from model.grad (torch or TensorFlow), scores from model.logit. `ctx` is the optional per-row
context the history-aware models need (node ids for graph and foundation models); it is sliced with the rows.

Used by the attack grid (run_attacks.py), so by every attack-success result: Table III, Table IV,
Table S-XI, Tables S-V to S-X, Table V, Table S-XXIV, Figure S1, Figure S2; by the transfer results
(Figure 1, Figure S3); by the ranking metrics (Table S-XXX, Table S-XXXI), the budget check
(Table S-XXI), the five-start HopSkipJump (Table S-XLII) and the certification (Table S-XXII); and by the
defenses (Table VII) and the ablations (Table VI).
"""
from __future__ import annotations

import numpy as np

from fraudadv.attacks.constraints import Constraints


def _sub(ctx, idx):
    """The rows idx of the context (None stays None)."""
    return None if ctx is None else ctx[idx]


def _ball(X, X0, eps, cons: Constraints):
    """Clip X into the L-inf ball of radius eps around X0, undo changes to features the attacker does not
    control, and apply the threat model's projection (ranges, valid values, recomputed features)."""
    D = np.clip(X - X0, -eps, eps)
    D[:, ~cons.mask] = 0.0
    return cons.project(X0 + D, X0, eps)


def pgd(model, X0, tau, eps, cons: Constraints, steps=50, alpha=None, early_stop=True, kappa=0.0, seed=0, ctx=None):
    """L-inf PGD on the logit: a random start in half the ball, then `steps` sign-gradient steps of size alpha,
    each projected back into the threat model. Keeps the lowest-logit point seen for every fraud. With
    early_stop, a fraud stops once its logit is below tau - kappa (kappa > 0 asks for a margin, as the transfer
    step does). Returns (points, logits)."""
    rng = np.random.default_rng(seed)
    alpha = alpha or 2.5 * eps / steps * 4
    X0 = X0.astype(np.float32)
    X = _ball(X0 + rng.uniform(-eps, eps, X0.shape).astype(np.float32) * 0.5, X0, eps, cons)
    best, best_l = X0.copy(), model.logit(X0, ctx)
    l = model.logit(X, ctx)
    upd = l < best_l
    best[upd], best_l[upd] = X[upd], l[upd]
    active = np.ones(len(X0), bool)
    for _ in range(steps):
        if early_stop:
            active = best_l >= tau - kappa
            if not active.any():
                break
        g = model.grad(X[active], _sub(ctx, active))
        step = -alpha * np.sign(g)
        step[:, ~cons.mask] = 0.0
        X[active] = _ball(X[active] + step, X0[active], eps, cons)
        l = model.logit(X[active], _sub(ctx, active))
        a = np.flatnonzero(active)
        upd = l < best_l[a]
        best[a[upd]], best_l[a[upd]] = X[active][upd], l[upd]
    return best, best_l


# The grid's APGD budget as (iterations per restart, restarts). Foundation models get 20 x 2 = 40 gradients per
# fraud, because one Chronos-2 gradient costs about 0.2 s per fraud on CPU.
APGD_BUDGET = {"default": (100, 5), "fm": (20, 2)}


def _apgd_checkpoints(n_iter):
    """The iterations at which APGD may halve its step (Croce & Hein 2020, Sec. 3.2): fractions 0, 0.22, ... of
    n_iter, each gap 0.03 shorter than the one before and at least 0.06."""
    p = [0.0, 0.22]
    while p[-1] < 1.0:
        p.append(p[-1] + max(p[-1] - p[-2] - 0.03, 0.06))
    return sorted({int(np.ceil(q * n_iter)) for q in p if 0 < q <= 1.0})


def apgd(model, X0, tau, eps, cons: Constraints, n_iter=100, restarts=5, early_stop=True, kappa=0.0, seed=0, ctx=None,
         alpha=0.75, rho=0.75):
    """Auto-PGD (Croce & Hein, ICML 2020) on the logit, L-inf, with the threat-model projection after every step.

    Each restart starts at a random point of the ball (half the ball on the first restart) and runs n_iter
    momentum steps (_apgd_restart). Every fraud keeps the lowest logit over all restarts. With early_stop, a
    fraud that has evaded (logit < tau - kappa) is not attacked further, neither in this restart nor in the next
    ones. Returns (points, logits).

    APGD is the grid's white-box attack on differentiable models, so that the white-box column is a strong
    attack, as the exact MILP is on trees.
    """
    rng = np.random.default_rng(seed)
    X0 = X0.astype(np.float32)
    n = len(X0)
    best, best_l = X0.copy(), model.logit(X0, ctx)
    goal = tau - kappa
    checkpoints = set(_apgd_checkpoints(n_iter))
    for r in range(restarts):
        attacked = np.flatnonzero((best_l >= goal) if early_stop else np.ones(n, bool))
        if not len(attacked):
            break
        start_scale = 0.5 if r == 0 else 1.0
        run_best, run_best_l = _apgd_restart(model, X0[attacked], _sub(ctx, attacked), goal, eps, cons, n_iter,
                                             checkpoints, early_stop, alpha, rho, rng, start_scale)
        upd = run_best_l < best_l[attacked]
        best[attacked[upd]], best_l[attacked[upd]] = run_best[upd], run_best_l[upd]
    return best, best_l


def _apgd_restart(model, X0, ctx, goal, eps, cons, n_iter, checkpoints, early_stop, alpha, rho, rng, start_scale):
    """One APGD restart on the frauds X0. Returns this restart's lowest-logit point and logit for each fraud.

    The step starts at 2 * eps. At each checkpoint, a fraud's step is halved, and the fraud goes back to its best
    point, when fewer than rho of the steps since the last checkpoint lowered its logit, or when neither its step
    nor its best logit changed since then. Each new point mixes the projected gradient step with the previous
    move (momentum alpha). With early_stop, a fraud stops once its best logit is below `goal`.
    """
    X = _ball(X0 + rng.uniform(-eps, eps, X0.shape).astype(np.float32) * start_scale, X0, eps, cons)
    l = model.logit(X, ctx)
    run_best, run_best_l = X.copy(), l.copy()
    step = np.full(len(X0), 2.0 * eps, np.float32)
    X_prev = X.copy()
    n_improved = np.zeros(len(X0))                  # steps that lowered the logit since the last checkpoint
    last_check, step_at_check, best_at_check = 0, step.copy(), run_best_l.copy()
    live = run_best_l >= goal if early_stop else np.ones(len(X0), bool)
    for k in range(1, n_iter + 1):
        if early_stop and not live.any():
            break
        L = np.flatnonzero(live)
        g = model.grad(X[L], _sub(ctx, L))
        Z = _ball(X[L] - step[L, None] * np.sign(g), X0[L], eps, cons)
        X_new = X[L] + alpha * (Z - X[L]) + (1 - alpha) * (X[L] - X_prev[L]) if k > 1 else Z
        X_new = _ball(X_new, X0[L], eps, cons)
        l_new = model.logit(X_new, _sub(ctx, L))
        X_prev[L] = X[L]
        n_improved[L] += l_new < l[L]
        X[L], l[L] = X_new, l_new
        upd = l_new < run_best_l[L]
        run_best[L[upd]], run_best_l[L[upd]] = X_new[upd], l_new[upd]
        if k in checkpoints:
            span = k - last_check
            halve = (n_improved < rho * span) | ((step == step_at_check) & (run_best_l >= best_at_check))
            step_at_check, best_at_check = step.copy(), run_best_l.copy()
            step = np.where(halve, step / 2, step).astype(np.float32)
            X[halve], l[halve] = run_best[halve], run_best_l[halve]          # go back to the best point
            n_improved[:] = 0
            last_check = k
        if early_stop:
            live = run_best_l >= goal
    return run_best, run_best_l


def square(model, X0, tau, eps, cons: Constraints, n_queries=300, p_init=0.3, early_stop=True, seed=0, ctx=None):
    """Square Attack, L-inf, score-based. Returns (points, logits).

    Start: every attacker feature at +eps or -eps from the fraud (random signs), kept only if it lowers the
    logit. Then, for n_queries rounds, each fraud tries a new point that resets a random subset of the attacker's
    features to +eps or -eps from the fraud; the subset is a fraction p of the attacker's features, halved as in
    the paper's schedule; the try is kept if it lowers the logit. On tabular data a "square" is that subset.
    With early_stop, a fraud stops once its logit is below tau.
    """
    rng = np.random.default_rng(seed)
    X0 = X0.astype(np.float32)
    n, d = X0.shape
    feats = cons.mutable
    X = X0.copy()
    X[:, feats] += eps * rng.choice([-1.0, 1.0], size=(n, len(feats))).astype(np.float32)
    X = _ball(X, X0, eps, cons)
    l = model.logit(X, ctx)
    l0 = model.logit(X0, ctx)
    better = l < l0
    X[~better], l[~better] = X0[~better], l0[~better]
    for it in range(n_queries):
        active = (l >= tau) if early_stop else np.ones(n, bool)
        if not active.any():
            break
        p = p_init * (0.5 ** (it / max(n_queries / 8, 1)))         # fraction of features to reset this round
        k = max(1, int(round(p * len(feats))))
        rows = np.flatnonzero(active)
        X_try = X[rows].copy()
        for r, i in enumerate(rows):
            cols = rng.choice(feats, size=k, replace=False)
            X_try[r, cols] = X0[i, cols] + eps * rng.choice([-1.0, 1.0], size=k)
        X_try = _ball(X_try, X0[rows], eps, cons)
        l_try = model.logit(X_try, _sub(ctx, rows))
        upd = l_try < l[rows]
        X[rows[upd]], l[rows[upd]] = X_try[upd], l_try[upd]
    return X, l


def hsj(model, X0, tau, cons: Constraints, max_iter=10, max_eval=400, init_eval=40, seed=0, ctx=None):
    """HopSkipJump (ART), untargeted, L-inf, one fraud at a time. Returns (points, logits).

    The attack sees only the decision (flagged or not) and searches for the smallest L-inf perturbation of the
    attacker's features; it has no budget, so the caller compares the distance it found with eps. The features
    the attacker controls are the attack's coordinates; the others stay at the fraud's values, and every query
    goes through the threat model's projection. hsjguard.install() seeds the library's starting points from
    `seed` and ends a step-size search that cannot succeed.
    """
    from art.attacks.evasion import HopSkipJump
    from art.estimators.classification import BlackBoxClassifier

    from fraudadv.attacks import hsjguard

    hsjguard.install()
    np.random.seed(seed)
    X0 = X0.astype(np.float32)
    m = np.asarray(cons.mutable)
    m = m[cons.upper[m] > cons.lower[m]]                 # ART rejects empty ranges (features constant in training)
    lo, hi = cons.lower[m], cons.upper[m]
    out = X0.copy()
    for i in range(len(X0)):
        x0 = X0[i]

        def predict(Z, x0=x0, c=None if ctx is None else ctx[i]):
            """ART's view of the model: Z holds the attacker's features; class 1 means flagged (logit > tau)."""
            F = np.repeat(x0[None], len(Z), 0)
            F[:, m] = Z
            F = cons.project(F, x0)
            p = 1.0 / (1.0 + np.exp(-(model.logit(F, None if c is None else np.repeat(c, len(F))) - tau)))
            return np.stack([1 - p, p], 1)

        # the search domain is the training range, widened to include the fraud's own value
        clf = BlackBoxClassifier(predict, input_shape=(len(m),), nb_classes=2,
                                 clip_values=(np.minimum(lo, x0[m]), np.maximum(hi, x0[m])))
        atk = HopSkipJump(clf, targeted=False, norm=np.inf, max_iter=max_iter, max_eval=max_eval,
                          init_eval=init_eval, init_size=50, verbose=False)
        try:
            z = atk.generate(x0[None, m], y=np.array([[0, 1]]))[0]
        except Exception:                               # no starting point found: the fraud stays as it is
            z = x0[m]
        F = x0.copy()
        F[m] = z
        out[i] = cons.project(F[None], x0)[0]
    return out, model.logit(out, ctx)


def aats(model, X0, tau, eps, cons: Constraints, kappa=0.0, beam=8, max_steps=40):
    """Attack a tree ensemble: AATS (beam search over threshold crossings, trees.AATS), then the L-inf box search
    (trees.box_search). For each fraud, keep the cheaper success of the two, or, when both fail, the point with
    the lower margin. Returns (points, logits, the AATS result with x_adv, success and cost updated)."""
    from fraudadv.attacks.trees import AATS, box_search

    ens = model.tree_ensemble()
    atk = AATS(ens, mutable=list(cons.mutable), norm="linf", eps=eps, target_margin=tau - kappa, beam=beam,
               max_steps=max_steps, lower=cons.lower, upper=cons.upper, allowed=cons.allowed,
               project=(lambda C, x0: cons.project(C, x0)))
    res = atk.generate(X0.astype(np.float32))
    Xb, okb, cb = box_search(ens.margin_fn, X0, tau - kappa, eps, cons.mutable, project=cons.project)
    mb, ma = ens.margin_fn(Xb), ens.margin_fn(res.x_adv)
    take_b = (okb & (~res.success | (cb < res.cost))) | (~okb & ~res.success & (mb < ma))
    res.x_adv[take_b] = Xb[take_b]
    res.cost[take_b] = cb[take_b]
    res.success = res.success | okb
    return res.x_adv, model.logit(res.x_adv), res
