"""Exact minimal L-inf evasion of a tree ensemble, solved as a mixed-integer linear program (MILP).

What it does
  For one fraud x0 that a tree ensemble (XGBoost or CatBoost) flags, it finds the smallest L-inf change to the
  attacker's features that brings the model's margin (its logit) down to the target:
      min b   subject to   margin(x) <= target,   |x_j - x0_j| <= b for every mutable feature j,
                           x inside the threat model's box and its discrete domains.
  This is the formulation of Kantchelian et al. (ICML 2016), solved by HiGHS through scipy.optimize.milp. When a
  solve stops at its time limit, HiGHS still returns a dual bound, and that bound is a certified lower bound on the
  smallest distance.

Inputs
  A study model (.model is the XGBoost or CatBoost object, .tree_ensemble().margin_fn its margin), the threat
  model's box per feature (lower, upper), its mutable features, its discrete domains (allowed) and, optionally,
  the threat model itself (cons, a fraudadv.attacks.constraints.Constraints), so that derived features are
  recomputed inside the program.

Outputs
  KantchelianMILP.solve_one(x0, target, eps_max) returns (point, distance b, lower bound, HiGHS status). The status
  is 0 (optimal), 1 (time limit) or 2 (infeasible; with eps_max this means no evasion within eps_max, so the fraud
  is certified robust there). min_margin(x0, eps) returns the lowest margin reachable within eps instead (the
  score-minimising attacker).

Where the results use it
  * The tree ensembles' white-box attack in the attack grid (run_attacks.milp_attack, which solves inside the
    AATS radius): Table III, Table IV, Table S-XI, Tables S-V to S-X.
  * The certification of AATS (attacks/certify.py): Table S-XXII, Table S-XXIII.
  * The tree rows of the ranking metrics under the score-minimising attacker (attacks/robust_auc.py, through
    min_margin): Table S-XXX, Table S-XXXI.

The program
  Variables, in this column order:
    p[j,k]  binary, 1 if feature j lies on the "left" of its k-th sorted threshold (XGBoost: x < t; CatBoost:
            x <= t). For a fixed feature, p is monotone in k: left of a threshold means left of every larger one.
    l[i,v]  one indicator per leaf v of tree i, continuous in [0, 1]. Each tree has exactly one active leaf
            (sum_v l = 1), and a leaf can be active only if the predicates on its path agree: for every internal
            node on predicate (j,k), sum(leaves of its left subtree) <= p[j,k] and
            sum(leaves of its right subtree) <= 1 - p[j,k]. With binary predicates these rows force the leaf
            indicators to be integral (Kantchelian et al., Sec. 3.1), so they need not be declared integer.
    b       continuous, the L-inf radius. Flipping a predicate away from its value at x0 costs the distance from
            x0 to that threshold: b >= cost[j,k] * |p[j,k] - p0[j,k]|.
    z[j]    continuous, only for the features tied together by a recomputation (see "Derived features").
  Margin row: base + scale * sum_v value_v * l[i,v] <= target, with base taken from the real model at x0.
  Immutable features keep their clean predicates, and thresholds the attacker's box cannot reach are fixed. The
  returned point is rebuilt from the predicates (the point of the selected cell closest to x0). Callers verify it
  on the real model.

Discrete domains (allowed = {feature: sorted valid values})
  A discrete feature can only take valid values, so (i) crossing threshold t costs the distance from x0 to the
  nearest valid value on the other side of t, not to t itself, (ii) a threshold with no valid value on the other
  side is fixed, and (iii) a cell between consecutive thresholds that holds no valid value is forbidden
  (p_c = p_{c-1}). The point is rebuilt with the valid value nearest x0 inside the selected cell. With these rules
  the MILP is exact for the domain-valid L-inf problem on the mutable features.

Derived features (only when cons is given)
  A derived feature that is an affine function of its sources is modelled exactly. This covers
  log_ratio = log_amt - log_amt_mean_prev (Sparkov, TabFormer, IEEE-CIS). In standardized units it reads
  d = d0 + sum_s c_s (x_s - x0_s), and it stays frozen for rows without history. Each continuous source and the
  derived feature get a variable z, tied to their predicates by big-M rows (p = 1 exactly when z is left of the
  threshold). On the derived feature's side these rows keep a margin of DERIVED_MARGIN, so that float32 rounding
  of the recomputation cannot flip a split. The sources' L-inf cost is taken on z, and the derived feature's
  predicates are free inside the range its sources can reach. The MILP is then exact for the domain-valid L-inf
  problem including the recomputation, up to that margin.
  PaySim's balance recomputation (max, expm1) is not linear. There the derived features stay frozen, exact_domain
  is False, and the MILP is a strong attack whose "certified" status is not a proof. Without cons, derived
  features are frozen too.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

DELTA = 1e-6                   # a strict side (x < t or x > t) cannot be reached at t itself: crossing to it costs DELTA more
DERIVED_MARGIN = 1e-5          # z-side margin on derived-feature thresholds (float32 rounding of the recomputation)


@dataclass
class Tree:
    leaves: list            # leaf values
    paths: list             # per leaf: list of (feature, threshold, go_left)


def xgb_trees(model):
    """The trees of an XGBoost model up to its early-stopping iteration (the ones the study's model predicts
    with), as Tree objects. Returns (trees, "lt"): a split sends x left when x < threshold."""
    booster = model.get_booster()
    dumps = booster.get_dump(dump_format="json")[: model.best_iteration + 1]
    names = booster.feature_names

    def feature_index(f):
        return int(f[1:]) if names is None else names.index(f)

    out = []
    for d in dumps:
        leaves, paths = [], []

        def walk(node, path):
            if "leaf" in node:
                leaves.append(float(node["leaf"]))
                paths.append(list(path))
                return
            j, t = feature_index(node["split"]), float(np.float32(node["split_condition"]))
            children = {c["nodeid"]: c for c in node["children"]}
            walk(children[node["yes"]], path + [(j, t, True)])
            walk(children[node["no"]], path + [(j, t, False)])

        walk(json.loads(d), [])
        out.append(Tree(leaves, paths))
    return out, "lt"


def catboost_trees(model, n_features):
    """The oblivious trees of a CatBoost model as Tree objects, one leaf per combination of the tree's splits.
    Returns (trees, "le", scale, bias): a split sends x left when x <= border, and the model's raw score is
    scale * (sum of leaf values) + bias."""
    from fraudadv.attacks.trees import _catboost_json

    mj = json.loads(_catboost_json(model))
    float_features = mj["features_info"]["float_features"]
    out = []
    for tree in mj["oblivious_trees"]:
        splits = [(float_features[s["float_feature_index"]]["flat_feature_index"], float(np.float32(s["border"])))
                  for s in tree["splits"]]
        values = tree["leaf_values"]
        depth = len(splits)
        leaves, paths = [], []
        for leaf in range(2 ** depth):
            # bit i of the leaf index is (x_{j_i} > border_i); left = not greater
            paths.append([(splits[i][0], splits[i][1], not (leaf >> i) & 1) for i in range(depth)])
            leaves.append(float(values[leaf]))
        out.append(Tree(leaves, paths))
    scale, bias = mj.get("scale_and_bias", [1.0, [0.0]])
    bias = bias[0] if isinstance(bias, list) else bias
    return out, "le", float(scale), float(bias)


class _Rows:
    """The constraint rows of one program, in the order they are added: lo <= sum(vals * x[cols]) <= hi."""

    def __init__(self):
        self.rows, self.cols, self.vals, self.lo, self.hi = [], [], [], [], []

    def add(self, cols, vals, lo, hi):
        r = len(self.lo)
        self.rows.extend([r] * len(cols))
        self.cols.extend(cols)
        self.vals.extend(vals)
        self.lo.append(lo)
        self.hi.append(hi)

    def __len__(self):
        return len(self.lo)

    def move_column(self, old, new):
        """Rows added so far that use column `old` use column `new` instead."""
        self.cols = [new if c == old else c for c in self.cols]


class KantchelianMILP:
    """The MILP of one tree model under one threat model. Build it once, then call solve_one for each fraud.

    model: a study model (.model is the XGBoost or CatBoost object, .tree_ensemble().margin_fn its margin).
    family: "xgboost" or "catboost". n_features: number of features.
    lower, upper: the threat model's box, per feature. mutable: indices of the attacker's features.
    time_limit: seconds per solve. allowed: {feature: sorted valid values} for discrete features.
    cons: the threat model (Constraints). With it, affine derived features are modelled exactly (see the module
    docstring).
    """

    def __init__(self, model, family, n_features, lower, upper, mutable, time_limit=120.0, allowed=None, cons=None):
        if family == "xgboost":
            self.trees, self.rule = xgb_trees(model.model)
            self.scale, self.bias = 1.0, None
        else:
            self.trees, self.rule, self.scale, self.bias = catboost_trees(model.model, n_features)
        self.margin_fn = model.tree_ensemble().margin_fn
        self.d, self.lower, self.upper = n_features, np.asarray(lower, float), np.asarray(upper, float)
        self.mutable = np.zeros(n_features, bool)
        self.mutable[list(mutable)] = True
        self.time_limit = time_limit
        self.allowed = {int(k): np.asarray(v, np.float32) for k, v in (allowed or {}).items()}
        # one predicate variable per (feature, distinct threshold): features in order, thresholds sorted
        thresholds = [set() for _ in range(n_features)]
        for tree in self.trees:
            for path in tree.paths:
                for j, t, _ in path:
                    thresholds[j].add(t)
        self.thr = [np.array(sorted(s)) for s in thresholds]
        self.off = np.cumsum([0] + [len(t) for t in self.thr])           # column of feature j's first predicate
        self.pid = {(j, t): self.off[j] + k for j in range(n_features) for k, t in enumerate(self.thr[j])}
        self.linear_derived, self.exact_domain = self._affine_derived(cons)
        self.cons = cons

    def _affine_derived(self, cons):
        """The derived features this MILP recomputes exactly, and whether it is exact for the threat model.

        Returns (linear_derived, exact_domain). linear_derived lists
        (derived index, [(source index, coefficient)], index of the missing-history flag or None), with
        coefficients in standardized units: d - d0 = sum_s c_s (x_s - x0_s)."""
        linear_derived = []
        if cons is None or not getattr(cons, "hook", None):
            return linear_derived, True
        from fraudadv.attacks.constraints import DERIVED

        if cons.hook != "log_ratio":
            return linear_derived, False                                    # PaySim: nonlinear recomputation
        for out, (amt, mean) in DERIVED["log_ratio"].items():
            if out not in cons.idx:
                continue
            jd = cons.idx[out]
            sources = [(cons.idx[amt], float(cons.sd[cons.idx[amt]] / cons.sd[jd])),
                       (cons.idx[mean], -float(cons.sd[cons.idx[mean]] / cons.sd[jd]))]
            guard = cons.idx.get("isna_" + mean)
            linear_derived.append((jd, sources, guard))
        return linear_derived, True

    def _left(self, x, t):
        return x < t if self.rule == "lt" else x <= t

    def _tree_value(self, x):
        """Sum of the leaf values x reaches (without CatBoost's scale and bias)."""
        s = 0.0
        for tree in self.trees:
            for v, path in zip(tree.leaves, tree.paths):
                if all(self._left(np.float32(x[j]), np.float32(t)) == go_left for j, t, go_left in path):
                    s += v
                    break
        return s

    def min_margin(self, x0, eps):
        """Lowest reachable margin within ||x - x0||_inf <= eps (the max-confidence attack; Kantchelian
        et al.'s second formulation). Returns (point, MILP margin, dual bound, status)."""
        return self.solve_one(x0, None, eps_max=eps, minimize_margin=True)

    def solve_one(self, x0, target, eps_max=None, minimize_margin=False):
        """Minimal L-inf evasion of x0 (margin <= target at the smallest radius b).

        With eps_max, predicates that cost more than eps_max to flip are fixed and the leaves they rule out are
        dropped, so the solve answers "minimal distance, or certified > eps_max" on a far smaller program.
        With minimize_margin (min_margin), the radius is capped at eps_max and the margin is minimised instead.
        Returns (point or None, b or the margin, dual bound, HiGHS status: 0 optimal, 1 time limit,
        2 infeasible, which with eps_max means certified robust at eps_max)."""
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix

        x0 = np.asarray(x0, np.float32)
        n_pred = int(self.off[-1])
        n_leaves = sum(len(tree.leaves) for tree in self.trees)
        # b's column is known only once the unreachable leaves are dropped: rows written before that use a
        # placeholder column, moved to b's real column afterwards
        b_placeholder = n_pred + n_leaves
        rows = _Rows()

        # 1. affine derived features active for this example, and the range each tied feature can reach
        radius = np.inf if eps_max is None else float(eps_max)
        z_range, coupling = self._derived_ranges(x0, radius)
        derived_free = {jd for jd, _ in coupling}

        # 2. predicates: bounds, forbidden empty cells, monotonicity and the radius rows
        p_lb, p_ub = self._predicate_rows(x0, eps_max, z_range, derived_free, rows, b_placeholder)

        # 3. leaves: one active leaf per tree, consistent with the predicates
        leaf_cols, leaf_values, const, n_leaf_end = self._leaf_rows(p_lb, p_ub, rows)
        n = n_leaf_end + 1
        b_col = n_leaf_end

        # 4. margin row: base + scale * sum(v l) <= target, base taken from the real model at x0
        base = float(self.margin_fn(x0[None])[0]) - self.scale * self._tree_value(x0) + self.scale * const
        if not minimize_margin:
            rows.add(leaf_cols, list(self.scale * np.asarray(leaf_values)), -np.inf, target - base - 1e-6)
        rows.move_column(b_placeholder, b_col)

        # 5. continuous z for the tied features (after b): big-M links to predicates, cost on z, affine recompute
        z_col = {j: n + k for k, j in enumerate(sorted(z_range))}
        self._z_rows(x0, z_range, z_col, derived_free, coupling, rows, b_col)

        # 6. variable bounds, objective and the solve
        z_lb = [z_range[j][0] for j in sorted(z_range)]
        z_ub = [z_range[j][1] for j in sorted(z_range)]
        lb = np.r_[p_lb[:n_pred], np.zeros(n - n_pred - 1), 0.0, z_lb]
        ub = np.r_[p_ub[:n_pred], np.ones(n - n_pred - 1), np.inf if eps_max is None else eps_max, z_ub]
        n = n + len(z_col)
        A = coo_matrix((rows.vals, (rows.rows, rows.cols)), shape=(len(rows), n)).tocsr()
        integrality = np.zeros(n)
        integrality[:n_pred] = 1
        objective = np.zeros(n)
        if minimize_margin:
            objective[leaf_cols] = self.scale * np.asarray(leaf_values)
        else:
            objective[b_col] = 1.0
        res = milp(objective, constraints=LinearConstraint(A, rows.lo, rows.hi), integrality=integrality,
                   bounds=Bounds(lb, ub), options={"time_limit": self.time_limit, "disp": False, "mip_rel_gap": 1e-4})

        # 7. the answer
        lower_bound = getattr(res, "mip_dual_bound", None)
        lower_bound = np.nan if lower_bound is None else float(lower_bound)
        if res.status == 2 and eps_max is not None:                         # infeasible: no evasion within eps_max
            return None, np.inf, float(eps_max), res.status
        if res.x is None:
            return None, np.inf, lower_bound, res.status
        x = self._point_from_solution(x0, res.x, n_pred, z_col, derived_free)
        if minimize_margin:
            return x, base + float(res.fun), base + lower_bound, res.status
        return x, float(res.x[b_col]), lower_bound, res.status

    def _derived_ranges(self, x0, radius):
        """Affine derived features that are free for x0, and the range of every tied feature.

        A derived feature is free when the fraud has history (else the ratio stays at its fill value), at least
        one of its sources is mutable and continuous, and it is not itself mutable. Each such source can reach
        [lower, upper] (widened to x0's own value) intersected with x0 +- radius; the derived feature reaches the
        affine image of those ranges. Returns ({feature: (low, high)}, [(derived index, free sources)])."""
        z_range, coupling = {}, []
        for jd, sources, guard in self.linear_derived:
            if guard is not None and float(x0[guard]) * self.cons.sd[guard] + self.cons.mu[guard] > 0.5:
                continue                                                     # no history: the ratio stays at its fill
            sources = [(js, c) for js, c in sources if self.mutable[js] and js not in self.allowed]
            if not sources or self.mutable[jd]:
                continue
            d_lo = d_hi = float(x0[jd])
            for js, c in sources:
                xs = float(x0[js])
                z_lo = max(min(float(self.lower[js]), xs), xs - radius)
                z_hi = min(max(float(self.upper[js]), xs), xs + radius)
                z_range[js] = (z_lo, z_hi)
                d_lo += min(c * (z_lo - xs), c * (z_hi - xs))
                d_hi += max(c * (z_lo - xs), c * (z_hi - xs))
            z_range[jd] = (d_lo, d_hi)
            coupling.append((jd, sources))
        return z_range, coupling

    def _predicate_rows(self, x0, eps_max, z_range, derived_free, rows, b_col):
        """Bounds of the predicate variables, and their rows: forbidden empty cells of discrete features,
        monotonicity in k, and the radius rows b >= cost * |p - p0|. Rows go feature by feature, thresholds in
        sorted order. Returns (lower bounds, upper bounds) of the predicates; equal bounds fix a predicate."""
        n_pred = int(self.off[-1])
        p_lb, p_ub = np.zeros(n_pred), np.ones(n_pred)
        for j in range(self.d):
            T = self.thr[j]
            valid = self.allowed.get(j) if self.mutable[j] else None
            if valid is not None and len(T):
                self._forbid_empty_cells(j, x0, valid, p_lb, p_ub, rows)
            for k, t in enumerate(T):
                i = self.off[j] + k
                p0 = float(self._left(x0[j], np.float32(t)))               # the predicate's value at x0
                has_next = k + 1 < len(T)
                if j in derived_free:                                        # free inside the reachable range
                    fixed = self._fixed_by_range(t, *z_range[j])
                    if fixed is None:
                        if has_next:
                            rows.add([i, i + 1], [1.0, -1.0], -np.inf, 0.0)
                    else:
                        p_lb[i] = p_ub[i] = fixed
                    continue
                if not self.mutable[j]:
                    p_lb[i] = p_ub[i] = p0
                    continue
                fixed = self._fixed_by_box(j, t)                             # the box cannot reach the other side
                if fixed is not None:
                    p_lb[i] = p_ub[i] = fixed
                if has_next:                                                 # monotone: p_k <= p_{k+1}
                    rows.add([i, i + 1], [1.0, -1.0], -np.inf, 0.0)
                cost = self._flip_cost(j, t, x0[j], p0, valid)
                if cost is None:                                             # no valid value across t
                    p_lb[i] = p_ub[i] = p0
                    continue
                if eps_max is not None and cost > eps_max:
                    p_lb[i] = p_ub[i] = p0
                    continue
                if p_lb[i] == p_ub[i]:
                    continue
                if p0 == 1:                                                  # leaving the left side
                    rows.add([b_col, i], [1.0, cost], cost, np.inf)          # b >= c (1 - p)
                else:
                    rows.add([b_col, i], [1.0, -cost], 0.0, np.inf)          # b >= c p
        return p_lb, p_ub

    def _forbid_empty_cells(self, j, x0, valid, p_lb, p_ub, rows):
        """Discrete feature j: forbid every cell between consecutive thresholds that holds no valid value.
        Cell c lies right of T[c-1] and left of T[c]; it is selected when p[c-1] = 0 and p[c] = 1. x0's own
        cell stays allowed."""
        T = self.thr[j]
        inside = np.array([self._left(valid, np.float32(t)) for t in T])          # (K, |valid|)
        own_cell = int(np.sum(~self._left(x0[j], T.astype(np.float32))))
        for c in range(len(T) + 1):
            in_cell = (inside[c] if c < len(T) else np.ones(len(valid), bool)) & (~inside[c - 1] if c > 0 else True)
            if in_cell.any() or c == own_cell:
                continue
            if c == 0:                                                       # left of every threshold: p[0] = 0
                p_ub[self.off[j]] = min(p_ub[self.off[j]], 0.0)
            elif c == len(T):                                                # right of every threshold: p[-1] = 1
                p_lb[self.off[j] + c - 1] = max(p_lb[self.off[j] + c - 1], 1.0)
            else:                                                            # p[c] <= p[c-1]
                rows.add([self.off[j] + c, self.off[j] + c - 1], [1.0, -1.0], -np.inf, 0.0)

    def _fixed_by_range(self, t, z_lo, z_hi):
        """Value of a derived feature's predicate on threshold t when its reachable range [z_lo, z_hi] lies
        entirely on one side of t (with DERIVED_MARGIN to spare), else None (free)."""
        if self.rule == "lt":
            return 1.0 if t > z_hi + DERIVED_MARGIN else (0.0 if t <= z_lo - DERIVED_MARGIN else None)
        return 1.0 if t >= z_hi + DERIVED_MARGIN else (0.0 if t < z_lo - DERIVED_MARGIN else None)

    def _fixed_by_box(self, j, t):
        """Value of feature j's predicate on threshold t when the box [lower, upper] lies on one side of t,
        else None."""
        if self.rule == "lt":
            if self.lower[j] >= t:
                return 0.0
            if self.upper[j] < t:
                return 1.0
            return None
        if self.lower[j] > t:
            return 0.0
        if self.upper[j] <= t:
            return 1.0
        return None

    def _flip_cost(self, j, t, x0j, p0, valid):
        """L-inf cost of moving feature j from x0j to the other side of threshold t. A discrete feature moves to
        the nearest valid value across t (None when there is none); a continuous one moves to t itself, plus
        DELTA when the other side is strict."""
        if valid is not None:
            other = valid[self._left(valid, np.float32(t)) != bool(p0)]
            if not len(other):
                return None
            return float(np.min(np.abs(other.astype(np.float64) - float(x0j))))
        gap = abs(float(t) - float(x0j))
        return gap + (DELTA if (self.rule == "le") == (p0 == 1) else 0.0)

    def _reachable_leaves(self, tree, p_lb, p_ub):
        """(value, path) of the leaves of `tree` that no fixed predicate rules out."""
        reachable = []
        for v, path in zip(tree.leaves, tree.paths):
            ok = True
            for j, t, go_left in path:
                i = self.pid[(j, t)]
                if p_lb[i] == p_ub[i] and (p_lb[i] == 1) != go_left:
                    ok = False
                    break
            if ok:
                reachable.append((v, path))
        return reachable

    def _leaf_rows(self, p_lb, p_ub, rows):
        """Leaf columns and their rows, tree by tree: one active leaf (sum l = 1), then for every free
        predicate on a path, left-subtree leaves <= p and right-subtree leaves <= 1 - p. A tree with a single
        reachable leaf is a constant and gets no columns.
        Returns (leaf columns, their values, sum of the constant trees, first column after the leaves)."""
        col = int(self.off[-1])
        leaf_cols, leaf_values = [], []
        const = 0.0
        for tree in self.trees:
            reachable = self._reachable_leaves(tree, p_lb, p_ub)
            if len(reachable) == 1:
                const += reachable[0][0]
                continue
            ids = list(range(col, col + len(reachable)))
            rows.add(ids, [1.0] * len(ids), 1.0, 1.0)
            left_of, right_of = {}, {}                                      # predicate -> leaves below each side
            for lid, (_, path) in zip(ids, reachable):
                for j, t, go_left in path:
                    i = self.pid[(j, t)]
                    if p_lb[i] != p_ub[i]:
                        (left_of if go_left else right_of).setdefault(i, []).append(lid)
            for i, leaves in left_of.items():
                rows.add(leaves + [i], [1.0] * len(leaves) + [-1.0], -np.inf, 0.0)
            for i, leaves in right_of.items():
                rows.add(leaves + [i], [1.0] * len(leaves) + [1.0], -np.inf, 1.0)
            leaf_cols.extend(ids)
            leaf_values.extend(v for v, _ in reachable)
            col += len(reachable)
        return leaf_cols, leaf_values, const, col

    def _z_rows(self, x0, z_range, z_col, derived_free, coupling, rows, b_col):
        """Rows of the continuous z variables: big-M links between each z and its feature's predicates
        (p = 1 exactly when z is left of the threshold), the L-inf cost of each source on z, and the affine
        recomputation z_d - sum c z_s = d0 - sum c x0_s."""
        for j, iz in z_col.items():
            z_lo, z_hi = z_range[j]
            T = self.thr[j]
            margin = DERIVED_MARGIN if j in derived_free else 0.0
            M = (z_hi - z_lo) + 3.0                                          # > any |z - t| for the t kept below
            for k, t in enumerate(T):
                if not (z_lo - 1.0 <= t <= z_hi + 1.0):
                    continue
                i = self.off[j] + k
                if self.rule == "lt":                                        # p=1 <=> z < t ; p=0 <=> z >= t
                    rows.add([iz, i], [1.0, M], -np.inf, float(t) - max(margin, DERIVED_MARGIN) + M)
                    rows.add([iz, i], [1.0, M], float(t) + margin, np.inf)
                else:                                                        # p=1 <=> z <= t ; p=0 <=> z > t
                    rows.add([iz, i], [1.0, M], -np.inf, float(t) - margin + M)
                    rows.add([iz, i], [1.0, M], float(t) + max(margin, DERIVED_MARGIN), np.inf)
            if j not in derived_free:                                        # L-inf cost of a source, on z
                rows.add([b_col, iz], [1.0, -1.0], -float(x0[j]), np.inf)
                rows.add([b_col, iz], [1.0, 1.0], float(x0[j]), np.inf)
        for jd, sources in coupling:
            rows.add([z_col[jd]] + [z_col[js] for js, _ in sources], [1.0] + [-c for _, c in sources],
                     *(2 * [float(x0[jd]) - sum(c * float(x0[js]) for js, c in sources)]))

    def _point_from_solution(self, x0, solution, n_pred, z_col, derived_free):
        """The point the solution selects: each feature at the point of its cell closest to x0, tied sources at
        their z value, then (with a threat model) projected so that derived features are recomputed."""
        x = self._rebuild(x0, np.round(solution[:n_pred]))
        if z_col:
            for j, iz in z_col.items():
                if j not in derived_free:
                    x[j] = self._into_cell(j, np.float32(solution[iz]), np.round(solution[self.off[j]: self.off[j + 1]]))
        if self.cons is not None:
            x = self.cons.project(x[None], x0)[0]
        return x

    def _into_cell(self, j, v, pj):
        """v moved (by ulps at most, when the solve is consistent) into the cell the predicates pj select."""
        T = self.thr[j]
        left, right = np.flatnonzero(pj == 1), np.flatnonzero(pj == 0)
        hi_t = np.float32(T[left[0]]) if len(left) else np.float32(np.inf)
        lo_t = np.float32(T[right[-1]]) if len(right) else np.float32(-np.inf)
        if self.rule == "lt":                                                # lo_t <= v < hi_t
            v = max(v, lo_t) if np.isfinite(lo_t) else v
            if np.isfinite(hi_t) and v >= hi_t:
                v = np.nextafter(hi_t, np.float32(-np.inf))
        else:                                                                # lo_t < v <= hi_t
            if np.isfinite(lo_t) and v <= lo_t:
                v = np.nextafter(lo_t, np.float32(np.inf))
            v = min(v, hi_t) if np.isfinite(hi_t) else v
        return np.float32(v)

    def _rebuild(self, x0, p):
        """The point closest to x0 in the cell the predicates p select (for a discrete feature: the valid
        value nearest x0 inside that cell). Immutable features keep x0's value."""
        x = x0.astype(np.float32).copy()
        for j in range(self.d):
            T = self.thr[j]
            if not len(T) or not self.mutable[j]:
                continue
            pj = p[self.off[j]: self.off[j + 1]]
            left = np.flatnonzero(pj == 1)
            right = np.flatnonzero(pj == 0)
            hi_t = T[left[0]] if len(left) else np.inf          # must stay left of hi_t
            lo_t = T[right[-1]] if len(right) else -np.inf      # must stay right of lo_t
            v = np.float32(x0[j])
            valid = self.allowed.get(j)
            if valid is not None:                               # valid value in the cell nearest x0
                ok = (~self._left(valid, np.float32(lo_t)) if np.isfinite(lo_t) else np.ones(len(valid), bool)) \
                    & (self._left(valid, np.float32(hi_t)) if np.isfinite(hi_t) else np.ones(len(valid), bool))
                if ok.any():
                    cand = valid[ok]
                    x[j] = cand[np.argmin(np.abs(cand - v))]
                continue
            if self.rule == "lt":                               # lo_t <= x < hi_t
                if v < lo_t:
                    v = np.float32(lo_t)
                elif v >= hi_t:
                    v = np.nextafter(np.float32(hi_t), np.float32(-np.inf))
            else:                                               # lo_t < x <= hi_t
                if v <= lo_t:
                    v = np.nextafter(np.float32(lo_t), np.float32(np.inf))
                elif v > hi_t:
                    v = np.float32(hi_t)
            x[j] = v
        return x

    def generate(self, X0, target, eps_max=None):
        """solve_one for every row of X0. Returns one dict per row: milp_b, milp_dist (L-inf distance of the
        returned point on the mutable features), milp_lower_bound, milp_status, milp_verified (the real model
        scores the point at or below target)."""
        out = []
        for x0 in X0:
            x, b, lbnd, st = self.solve_one(x0, target, eps_max)
            ok = x is not None and float(self.margin_fn(x[None])[0]) <= target
            dist = float(np.max(np.abs(x - x0)[self.mutable])) if x is not None else np.inf
            out.append(dict(milp_b=b, milp_dist=dist, milp_lower_bound=lbnd, milp_status=int(st), milp_verified=bool(ok)))
        return out
