"""The five tabular model families of the study: XGBoost, CatBoost, MLP, FT-Transformer and ivis.

What it does
    Each class wraps one library behind the same small interface, so the attacks, metrics and explainers never
    need to know which family they are working on:

        fit(Xtr, ytr, Xva, yva, wva, params, trial=None, seed=0)   train; early stopping on validation AUPRC
        logit(X)       fraud log-odds, shape (n,); the attacks push this below the decision threshold
        proba(X)       P(fraud) = sigmoid(logit)
        grad(X)        d logit / d X, shape (n, d); differentiable families only (MLP, FT-Transformer, ivis)
        tree_ensemble()  the trees as a fraudadv.attacks.trees.TreeEnsemble; XGBoost and CatBoost only

    `FAMILIES` maps the family keys used on the command line and in runs/<dataset>/<key>/ to the classes.

Inputs and outputs
    X arrays are the standardized feature matrices of data/processed/<dataset>/arrays.npz; y is 0/1 and wva
    holds the validation sampling weights. hpo/train_all.py tunes and trains these classes and pickles them to
    runs/<dataset>/<key>/seed<k>.pkl (ivis: a folder, see train_all.save_model). Pickles store the module path
    and class names, which is why neither may be renamed.

Results that use it
    Every result on the tabular families: the models and their tuned hyperparameters (Table S-II), clean
    performance (Table S-IV), the attack grid and survey (Table III, Table IV, Table S-XI, Tables S-V to S-X),
    transfer (Figure 1), certification (Table S-XXII), and the adversarially trained defenses
    (Table VII, via `adversarial_loss`; `bound_net` applies it to the graph and foundation models too).

Threads: FRAUDADV_THREADS (default 4) sets the number of CPU threads every family trains and predicts with.
"""
from __future__ import annotations

import math
import os

import numpy as np
from sklearn.metrics import average_precision_score

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")      # ivis 2.0.x needs Keras 2 (tf-keras)
N_THREADS = int(os.environ.get("FRAUDADV_THREADS", "4"))


def _gpu():
    """True when fraudadv.device picked a CUDA device (FRAUDADV_DEVICE, or a visible GPU)."""
    from fraudadv.device import use_gpu

    return use_gpu()


def sigmoid(z):
    """Logistic function, with the input clipped so exp() never overflows."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class Base:
    """Shared part of every family: P(fraud) from the logit. `ctx` is ignored by the tabular families; it is
    there so that tabular, graph and foundation models can be called the same way."""

    family = ""
    differentiable = False

    def proba(self, X, ctx=None):
        return sigmoid(self.logit(X, ctx))


# ------------------------------------------------------------------ trees
class XGB(Base):
    """Gradient-boosted trees (XGBoost, histogram method), early-stopped on weighted validation AUPRC."""

    family = "XGBoost"

    def fit(self, Xtr, ytr, Xva, yva, wva, params, trial=None, seed=0):
        import xgboost as xgb

        p = dict(params)
        n_rounds = p.pop("n_estimators", 1000)
        gpu = _gpu()
        self.model = xgb.XGBClassifier(
            n_estimators=n_rounds, tree_method="hist", n_jobs=N_THREADS, random_state=seed,
            eval_metric="aucpr", early_stopping_rounds=50, **({"device": "cuda"} if gpu else {}), **p)
        self.model.fit(Xtr, ytr, eval_set=[(Xva, yva)], sample_weight_eval_set=[wva], verbose=False)
        if gpu:                                   # attacks query small numpy batches: predict on CPU
            self.model.set_params(device="cpu")
            self.model.get_booster().set_param({"device": "cpu"})
        return self

    def logit(self, X, ctx=None):
        # Only the trees up to the early-stopping point count, in scoring and in the tree view below.
        return self.model.get_booster().inplace_predict(np.asarray(X, np.float32), predict_type="margin",
                                                        iteration_range=(0, self.model.best_iteration + 1))

    def tree_ensemble(self):
        from fraudadv.attacks.trees import TreeEnsemble

        return TreeEnsemble.from_xgboost(self.model, iteration_range=(0, self.model.best_iteration + 1))


class CatBoost(Base):
    """Oblivious (symmetric) boosted trees (CatBoost), early-stopped on validation PR-AUC.

    gpu=True trains on the GPU, but only when fraudadv.device also picked a CUDA device. It is off by default
    because CatBoost's GPU mode supports fewer options and grows different trees; every model of the paper was
    trained on CPU. `python -m fraudadv.hpo.train_all --catboost-gpu` turns it on."""

    family = "CatBoost"

    def fit(self, Xtr, ytr, Xva, yva, wva, params, trial=None, seed=0, gpu=False):
        from catboost import CatBoostClassifier, Pool

        self.model = CatBoostClassifier(iterations=params.get("iterations", 1500), eval_metric="PRAUC",
                                        thread_count=N_THREADS, random_seed=seed, verbose=False,
                                        od_type="Iter", od_wait=60, use_best_model=True,
                                        **({"task_type": "GPU"} if _gpu() and gpu else {}),
                                        **{k: v for k, v in params.items() if k != "iterations"})
        self.model.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva, weight=wva))
        self.n_features = Xtr.shape[1]
        return self

    def logit(self, X, ctx=None):
        return self.model.predict(np.asarray(X, np.float32), prediction_type="RawFormulaVal", thread_count=N_THREADS)

    def tree_ensemble(self):
        from fraudadv.attacks.trees import TreeEnsemble

        # Older pickles have no n_features attribute; the model's feature names give the same count.
        return TreeEnsemble.from_catboost(self.model, n_features=getattr(self, "n_features", None) or len(self.model.feature_names_))


# ------------------------------------------------------------------ torch models
class TorchModel(Base):
    """Shared training and scoring for the two PyTorch families (MLP, FT-Transformer).

    Training: AdamW on class-weighted binary cross-entropy (params["pos_weight"]), mini-batches in a random
    order drawn from the seed, one validation AUPRC per epoch, and the weights of the best epoch kept. Training
    stops after params["patience"] epochs without a gain of more than 1e-4. With params["adv"] set, each batch
    uses the adversarial-training loss of `adversarial_loss` instead (the defenses of Table VII).

    Subclasses only say how to build the network (`_build`)."""

    differentiable = True

    def _build(self, d, params):
        raise NotImplementedError

    def _dev(self):
        """The compute device, with the network moved to it (models are stored and loaded on CPU)."""
        from fraudadv.device import torch_device

        dev = torch_device()
        p = next(self.net.parameters(), None)
        if p is not None and p.device != dev:
            self.net.to(dev)
        return dev

    def __getstate__(self):
        st = dict(self.__dict__)
        net = st.get("net")
        if net is not None and any(q.device.type != "cpu" for q in net.parameters()):
            import copy

            st["net"] = copy.deepcopy(net).cpu()  # pickles hold CPU tensors, whatever the training device
        return st

    def fit(self, Xtr, ytr, Xva, yva, wva, params, trial=None, seed=0):
        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(N_THREADS)
        self.net = self._build(Xtr.shape[1], params)
        dev = self._dev()
        opt = torch.optim.AdamW(self.net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        pos_w = torch.tensor(params.get("pos_weight", 1.0), device=dev)
        Xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
        yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
        batch_size = params.get("batch_size", 1024)
        adv = params.get("adv")                   # None: plain training; a dict: adversarial training
        best_auprc, best_state, epochs_since_gain = -1.0, None, 0
        for epoch in range(params.get("epochs", 30)):
            self.net.train()
            order = torch.randperm(len(Xt), device=dev)
            for start in range(0, len(Xt), batch_size):
                rows = order[start:start + batch_size]
                if adv is None:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(self.net(Xt[rows]).squeeze(-1), yt[rows], pos_weight=pos_w)
                else:
                    loss = adversarial_loss(self.net, Xt[rows], yt[rows], pos_w, adv)
                opt.zero_grad()
                loss.backward()
                opt.step()
            val_auprc = average_precision_score(yva, self.logit(Xva), sample_weight=wva)
            if trial is not None:                 # Optuna's Hyperband pruner may stop a poor trial early
                trial.report(val_auprc, epoch)
                if trial.should_prune():
                    import optuna

                    raise optuna.TrialPruned()
            if val_auprc > best_auprc + 1e-4:
                best_auprc, epochs_since_gain = val_auprc, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                epochs_since_gain += 1
                if epochs_since_gain >= params.get("patience", 5):
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def logit(self, X, ctx=None):
        import torch

        self.net.eval()
        dev = self._dev()
        out = []
        with torch.no_grad():
            for start in range(0, len(X), 8192):
                xb = torch.tensor(np.asarray(X[start:start + 8192]), dtype=torch.float32, device=dev)
                out.append(self.net(xb).squeeze(-1).cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)

    def grad(self, X, ctx=None):
        """Exact gradient of the logit of every row with respect to its own features (rows are independent,
        so the gradient of the summed logit gives each row's gradient)."""
        import torch

        self.net.eval()
        x = torch.tensor(np.asarray(X), dtype=torch.float32, device=self._dev(), requires_grad=True)
        self.net(x).sum().backward()
        return x.grad.cpu().numpy()

    def torch_forward(self, x):
        """The logit as a differentiable torch function, for attacks written in torch (APGD)."""
        return self.net(x).squeeze(-1)


def adversarial_loss(net, x, y, pos_w, adv):
    """Adversarial-training loss of one batch, in the attacker's threat model (defenses of Table VII).

    adv = {"method": "pgd" | "trades", "eps", "steps", "mutable": feature ids, "lower", "upper",
           "frauds_only": bool, "beta": TRADES weight}
    pgd     Madry et al. 2018: minimise the loss at an L-inf worst case found by PGD inside the ball
    trades  Zhang et al. 2019: clean loss + beta * KL(p(x) || p(x')), x' maximising the KL
    Only mutable features move; with frauds_only, only frauds are perturbed, since legitimate
    customers do not attack the model (class-conditional threat).

    The inner attack starts from a uniform random point in the ball, takes `steps` signed-gradient steps of
    size 2.5 * eps / steps, and after each step projects back onto the ball and the feature bounds."""
    import torch
    import torch.nn.functional as F

    net.eval()
    eps, steps = adv["eps"], adv.get("steps", 5)
    mask = torch.zeros(x.shape[1], device=x.device)
    mask[adv["mutable"]] = 1.0
    if adv.get("frauds_only", True):
        mask = mask.unsqueeze(0) * y.unsqueeze(1)
    lo = torch.as_tensor(adv["lower"], dtype=x.dtype, device=x.device)
    hi = torch.as_tensor(adv["upper"], dtype=x.dtype, device=x.device)
    alpha = 2.5 * eps / steps
    xa = (x + mask * eps * (2 * torch.rand_like(x) - 1)).clamp(lo, hi).detach()
    with torch.no_grad():
        p_clean = torch.sigmoid(net(x).squeeze(-1))
    for _ in range(steps):
        xa.requires_grad_(True)
        z = net(xa).squeeze(-1)
        if adv["method"] == "trades":
            q = torch.sigmoid(z).clamp(1e-6, 1 - 1e-6)
            obj = (p_clean * torch.log(p_clean.clamp(1e-6) / q) + (1 - p_clean) * torch.log((1 - p_clean).clamp(1e-6) / (1 - q))).sum()
        else:
            obj = F.binary_cross_entropy_with_logits(z, y, pos_weight=pos_w, reduction="sum")
        g, = torch.autograd.grad(obj, xa)
        xa = xa.detach() + alpha * mask * g.sign()
        xa = torch.max(torch.min(xa, x + eps), x - eps).clamp(lo, hi).detach()
    net.train()
    if adv["method"] == "trades":
        z_clean = net(x).squeeze(-1)
        p = torch.sigmoid(z_clean)
        q = torch.sigmoid(net(xa).squeeze(-1)).clamp(1e-6, 1 - 1e-6)
        kl = (p * torch.log(p.clamp(1e-6) / q) + (1 - p) * torch.log((1 - p).clamp(1e-6) / (1 - q))).mean()
        return F.binary_cross_entropy_with_logits(z_clean, y, pos_weight=pos_w) + adv.get("beta", 6.0) * kl
    return F.binary_cross_entropy_with_logits(net(xa).squeeze(-1), y, pos_weight=pos_w)



def bound_net(forward, *modules):
    """A torch module for `adversarial_loss` whose forward(x) is forward(x) as a column and whose parameters are
    those of `modules`. It lets the same adversarial training serve networks whose input x is one of several
    arguments: the graph models (x and the node's context) and the foundation-model heads (x and the embedding)."""
    import torch.nn as nn

    class Bound(nn.Module):
        def __init__(self):
            super().__init__()
            self.parts = nn.ModuleList(modules)    # so that eval() and train() reach every layer (dropout)

        def forward(self, x):
            return forward(x).unsqueeze(-1)

    return Bound()

class MLP(TorchModel):
    """Feed-forward network: n_layers blocks of Linear -> SiLU -> Dropout of equal width, then one logit."""

    family = "MLP"

    def _build(self, d, p):
        import torch.nn as nn

        layers, w = [], d
        for _ in range(p["n_layers"]):
            layers += [nn.Linear(w, p["width"]), nn.SiLU(), nn.Dropout(p["dropout"])]
            w = p["width"]
        layers.append(nn.Linear(w, 1))
        return nn.Sequential(*layers)


_FTNET_CLS = None


def _ftnet_class():
    """FTNet is built lazily so importing the zoo does not import torch. It is published as
    fraudadv.models.zoo._FTNet through the module __getattr__ below, so pickles saved in one
    process load in a fresh one (pickle resolves the class by that name)."""
    global _FTNET_CLS
    if _FTNET_CLS is not None:
        return _FTNET_CLS
    import torch
    import torch.nn as nn

    class FTNet(nn.Module):
        """Each feature j becomes a token x_j * w_j + b_j of width k; a learned [CLS] token is prepended, the
        tokens go through n_blocks pre-norm Transformer layers, and the head reads the [CLS] output."""

        def __init__(self, d, k, n_heads, n_blocks, dropout):
            super().__init__()
            self.w = nn.Parameter(torch.randn(d, k) / math.sqrt(k))
            self.b = nn.Parameter(torch.zeros(d, k))
            self.cls = nn.Parameter(torch.randn(1, 1, k) / math.sqrt(k))
            layer = nn.TransformerEncoderLayer(k, n_heads, dim_feedforward=2 * k, dropout=dropout,
                                               batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, n_blocks, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(k), nn.ReLU(), nn.Linear(k, 1))

        def forward(self, x):
            tok = x.unsqueeze(-1) * self.w + self.b                      # (batch, d, k)
            tok = torch.cat([self.cls.expand(len(x), -1, -1), tok], 1)   # (batch, 1 + d, k)
            return self.head(self.enc(tok)[:, 0])

    FTNet.__name__ = FTNet.__qualname__ = "_FTNet"
    FTNet.__module__ = __name__
    _FTNET_CLS = FTNet
    return FTNet


def __getattr__(name):
    if name == "_FTNet":
        return _ftnet_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _make_ftnet(d, p):
    return _ftnet_class()(d, p["d_token"], p["n_heads"], p["n_blocks"], p["dropout"])


class FTTransformer(TorchModel):
    """Feature Tokenizer + Transformer (Gorishniy et al., NeurIPS 2021), compact CPU-sized version."""

    family = "FT-Transformer"

    def _build(self, d, p):
        return _make_ftnet(d, p)


# ------------------------------------------------------------------ ivis (triplet loss)
class IvisModel(Base):
    """Supervised ivis (Siamese triplet-loss network) embedding + logistic head.

    Training: ivis learns an embedding on all training frauds plus legitimate rows drawn at random (from the
    seed) up to params["max_train"] rows in total; a class-balanced logistic regression on that embedding
    gives the logit. The head is applied on top of ivis's Keras encoder in TensorFlow, so the whole pipeline is
    differentiable and white-box gradients are exact.
    """

    family = "ivis (triplet)"
    differentiable = True

    def fit(self, Xtr, ytr, Xva, yva, wva, params, trial=None, seed=0):
        import tensorflow as tf
        from ivis import Ivis
        from sklearn.linear_model import LogisticRegression

        tf.random.set_seed(seed)
        np.random.seed(seed)
        n_rows = min(len(Xtr), params.get("max_train", 60_000))
        rng = np.random.default_rng(seed)
        frauds = np.flatnonzero(ytr == 1)
        legit = np.flatnonzero(ytr == 0)
        rows = np.r_[frauds, rng.choice(legit, size=max(n_rows - len(frauds), 1), replace=False)]
        self.ivis = Ivis(embedding_dims=params["embedding_dims"], k=params["k"], model=params.get("model", "maaten"),
                         supervision_weight=params["supervision_weight"], distance=params.get("distance", "pn"),
                         epochs=params.get("epochs", 30), n_epochs_without_progress=5, batch_size=512,
                         verbose=0, n_trees=20)
        self.ivis.fit(Xtr[rows].astype(np.float32), ytr[rows])
        Etr = self.ivis.transform(Xtr[rows].astype(np.float32))
        self.head = LogisticRegression(C=params.get("C", 1.0), max_iter=2000, class_weight="balanced").fit(Etr, ytr[rows])
        self.encoder = getattr(self.ivis, "encoder_", None) or self.ivis.encoder
        self._w = tf.constant(self.head.coef_.T.astype(np.float32))
        self._b = tf.constant(self.head.intercept_.astype(np.float32))
        return self

    def logit(self, X, ctx=None):
        import tensorflow as tf

        X = np.asarray(X, np.float32)
        out = [self.encoder(tf.convert_to_tensor(X[i:i + 8192]), training=False).numpy() for i in range(0, len(X), 8192)]
        E = np.concatenate(out) if out else np.empty((0, self.head.coef_.shape[1]))
        return E @ self.head.coef_[0] + self.head.intercept_[0]

    def grad(self, X, ctx=None):
        import tensorflow as tf

        x = tf.convert_to_tensor(np.asarray(X, np.float32))
        with tf.GradientTape() as tape:
            tape.watch(x)
            z = tf.squeeze(tf.matmul(self.encoder(x, training=False), self._w) + self._b, -1)
        return tape.gradient(z, x).numpy()


FAMILIES = {"xgboost": XGB, "catboost": CatBoost, "mlp": MLP, "ft_transformer": FTTransformer, "ivis": IvisModel}
