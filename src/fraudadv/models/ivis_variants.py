"""ivis variants that try to make the similarity learning, not the classifier head, carry the decision.

The study's ivis (models.zoo.IvisModel) is the "maaten" Siamese encoder (Dense 500-500-2000, SELU) with a
logistic regression on its embedding. Its triplet loss shapes the embedding only during training, and nothing
bounds how far the embedding moves when the input moves, so under attack it behaves like an MLP. This module
tests three changes, none of them adversarial training, and their combinations:

  supervision weight   the share of the loss on the labels; the study's tuning searched [0.1, 0.9] and chose
                       0.24-0.40 on four datasets. Variant "sup" searches [0.7, 1.0].
  sn                   spectral normalization on every Dense layer of the encoder, including the embedding layer,
                       so the encoder is Lipschitz (each layer's spectral norm <= 1; SELU's slope is at most
                       1.76): an embedding gap of the triplet margin then needs an input change of a fixed size.
  head                 "lr": logistic regression on the embedding (as in the study);
                       "knn": a soft neighbour vote, s(x) = sum over the k nearest training embeddings of
                       softmax(-d^2 / T) * label, logit log((s + 1e-3) / (1 - s + 1e-3)); k and T are chosen
                       on validation AUPRC. The vote is differentiable, so the white-box attack (APGD) uses its
                       exact gradient; Square and HopSkipJump query it directly.

Contents:
  IvisVariant         the model: fit (ivis encoder, then both heads), logit / grad (the chosen head), save / load
  build_encoder       the encoder network, optionally spectrally normalized
  KNNHead, choose_knn the neighbour vote and the choice of its k and T on validation
  ClassLSE            a smooth surrogate of the neighbour vote, for adaptive attacks
  val_subsample       the validation rows used to choose k and T

Inputs and outputs: the arrays of fraudadv.data.load; a model is saved by IvisVariant.save as
<path>/encoder.weights.h5 (the architecture is rebuilt from the params), <path>/head.pkl (logistic head and
params) and <path>/knn.npz (reference embeddings, labels, k, T).

Results: Tables S-XXV to S-XXVII (Table S-XXV, Table S-XXVI, Table S-XXVII), through
scripts/analysis/ivis_variants.py, which tunes, trains and attacks the variants (sup, smooth, and the study's
own encoder with the neighbour vote) under runs_ivisvar/.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from fraudadv.models.zoo import Base  # noqa: E402

EPS_S = 1e-3                          # keeps the vote's log-odds finite when s is 0 or 1
KNN_K = (10, 30, 100)                 # neighbour counts tried
KNN_C = (0.1, 0.3, 1.0, 3.0)          # T = c * median squared distance to the k-th neighbour (validation)


def build_encoder(d, emb, sn=False, cap=1.0, drop=0.1):
    """The ivis "maaten" encoder (Dense 500-500-2000, SELU, alpha dropout 0.1) plus the embedding layer. With sn,
    every Dense kernel is spectrally normalized and its output scaled by cap, so each layer's spectral norm is at
    most cap and the encoder is Lipschitz (L2) with constant at most (1.76 cap)^3 cap."""
    import tf_keras as keras
    from tf_keras.layers import Activation, AlphaDropout, Dense, Rescaling, SpectralNormalization

    def dense(h, n, act):
        init = "lecun_normal" if act == "selu" else "glorot_uniform"
        if not sn:
            return Dense(n, activation=act, kernel_initializer=init)(h)
        h = Rescaling(cap)(SpectralNormalization(Dense(n, kernel_initializer=init))(h))
        return Activation(act)(h) if act else h

    inp = keras.Input(shape=(d,))
    h = dense(inp, 500, "selu")
    h = AlphaDropout(drop)(h) if drop else h
    h = dense(h, 500, "selu")
    h = AlphaDropout(drop)(h) if drop else h
    h = dense(h, 2000, "selu")
    return keras.Model(inp, dense(h, emb, None))


def auprc_w(y, s, w):
    """Weighted AUPRC (average precision) of score s."""
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s, sample_weight=w))


class KNNHead:
    """Soft k-nearest-neighbour vote over reference embeddings R with labels yR (1 = fraud).

    s(E) = sum over the k nearest rows of R of softmax(-d^2 / T) * label, returned as the log-odds
    log((s + EPS_S) / (1 - s + EPS_S)). With cosine, embeddings are L2-normalized first."""

    def __init__(self, R, yR, k, T, cosine):
        self.R, self.yR, self.k, self.T, self.cosine = R.astype(np.float32), yR.astype(np.float32), int(k), float(T), bool(cosine)

    def _prep(self, E):
        """Embeddings as compared: L2-normalized for a cosine ivis distance, unchanged otherwise."""
        import tensorflow as tf

        return tf.math.l2_normalize(E, axis=1) if self.cosine else E

    def logit_tf(self, E):
        """The vote's log-odds for a batch of embeddings E, as a differentiable TensorFlow tensor."""
        import tensorflow as tf

        R = self._prep(tf.constant(self.R))
        E = self._prep(E)
        d2 = tf.reduce_sum(E * E, 1, keepdims=True) + tf.reduce_sum(R * R, 1)[None, :] - 2.0 * tf.matmul(E, R, transpose_b=True)
        v, i = tf.math.top_k(-d2, k=self.k)          # v = -d^2 of the k nearest references
        w = tf.nn.softmax(v / self.T, axis=1)
        s = tf.reduce_sum(w * tf.gather(tf.constant(self.yR), i), 1)
        return tf.math.log(s + EPS_S) - tf.math.log(1.0 - s + EPS_S)


class ClassLSE:
    """Surrogate for adaptive attacks on the neighbour vote (in the spirit of Sitawarin and Wagner, 2019): with the K
    nearest fraud and the K nearest legitimate reference embeddings found separately,
    z = logsumexp_fraud(-d^2 / T) - logsumexp_legit(-d^2 / T). The vote's own gradient vanishes when all k neighbours
    share a label (s = 0 or 1 exactly), which hides the boundary from APGD; this one never vanishes and points the
    embedding toward the nearest legitimate references."""

    def __init__(self, m, K=None, T=None):
        """m: an IvisVariant with a neighbour vote; K and T default to the vote's own k and T."""
        h = m.knn
        self.m, self.K, self.T = m, int(K or h.k), float(T or h.T)
        self.Rf, self.Rl = h.R[h.yR == 1], h.R[h.yR == 0]

    def _logit_tf(self, x):
        """z for a batch of inputs x (TensorFlow tensor in, tensor out)."""
        import tensorflow as tf

        h = self.m.knn
        E = h._prep(self.m.encoder(x, training=False))
        out = []
        for R in (self.Rf, self.Rl):
            R = h._prep(tf.constant(R))
            d2 = tf.reduce_sum(E * E, 1, keepdims=True) + tf.reduce_sum(R * R, 1)[None, :] - 2.0 * tf.matmul(E, R, transpose_b=True)
            v, _ = tf.math.top_k(-d2, k=min(self.K, int(R.shape[0])))
            out.append(tf.reduce_logsumexp(v / self.T, axis=1))
        return out[0] - out[1]

    def logit(self, X, ctx=None):
        """z for inputs X, in batches of 1,024 rows."""
        import tensorflow as tf

        X = np.asarray(X, np.float32)
        return np.concatenate([self._logit_tf(tf.constant(X[i:i + 1024])).numpy() for i in range(0, len(X), 1024)])

    def grad(self, X, ctx=None):
        """dz/dX for inputs X, in batches of 1,024 rows."""
        import tensorflow as tf

        X = np.asarray(X, np.float32)
        gs = []
        for i in range(0, len(X), 1024):
            x = tf.constant(X[i:i + 1024])
            with tf.GradientTape() as tape:
                tape.watch(x)
                z = self._logit_tf(x)
            gs.append(tape.gradient(z, x).numpy())
        return np.concatenate(gs)


def choose_knn(Eref, yref, Eva, yva, wva, cosine):
    """(validation AUPRC, k, T) of the best neighbour vote, over every (k, T) in KNN_K x KNN_C. The 100 nearest
    reference embeddings of every validation point are found once, and every (k, T) is scored from them.
    Ties keep the first (k, T) in that order."""
    import tensorflow as tf

    h = KNNHead(Eref, yref, max(KNN_K), 1.0, cosine)
    R = h._prep(tf.constant(h.R))
    V, L = [], []
    for i in range(0, len(Eva), 1024):
        E = h._prep(tf.constant(Eva[i:i + 1024].astype(np.float32)))
        d2 = tf.reduce_sum(E * E, 1, keepdims=True) + tf.reduce_sum(R * R, 1)[None, :] - 2.0 * tf.matmul(E, R, transpose_b=True)
        v, ix = tf.math.top_k(-d2, k=max(KNN_K))
        V.append(-v.numpy())
        L.append(h.yR[ix.numpy()])
    # squared distances (rounding errors below 0 set to 0) and labels of the 100 nearest references
    D2, Y = np.maximum(np.concatenate(V), 0.0), np.concatenate(L)
    best = None
    for k in KNN_K:
        med = float(np.median(np.maximum(D2[:, k - 1], 1e-12)))
        for c in KNN_C:
            T = c * med
            a_ = -D2[:, :k] / T
            a_ -= a_.max(1, keepdims=True)                # softmax over the k neighbours, computed stably
            w = np.exp(a_)
            s = (w * Y[:, :k]).sum(1) / w.sum(1)
            z = np.log(s + EPS_S) - np.log(1 - s + EPS_S)
            a = auprc_w(yva, z, wva)
            if best is None or a > best[0]:
                best = (a, k, T)
    return best


def val_subsample(yva, wva, n_legit=20_000, seed=0):
    """Validation rows used to choose k and T: every fraud plus n_legit random legitimate rows, with the
    legitimate weights scaled up so the weighted class balance is unchanged. Returns (row indices, weights);
    a validation split with at most n_legit legitimate rows is used whole."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(yva == 1), np.flatnonzero(yva == 0)
    if len(neg) <= n_legit:
        return np.arange(len(yva)), wva
    keep = rng.choice(neg, n_legit, replace=False)
    idx = np.sort(np.r_[pos, keep])
    w = wva[idx].astype(float).copy()
    w[yva[idx] == 0] *= len(neg) / n_legit          # keep the weighted class balance
    return idx, w


class IvisVariant(Base):
    """ivis encoder with a logistic head and, if params["head"] == "knn", a neighbour vote.

    params: those of the study's ivis (embedding_dims, k, supervision_weight, distance, C, epochs, max_train)
    plus sn and cap (spectral normalization), drop (alpha dropout rate) and head ("lr" or "knn").
    head_kind says which head logit() and grad() use; with_head() gives the same encoder with the other head.
    """

    family = "ivis variant"
    differentiable = True

    def fit(self, Xtr, ytr, Xva, yva, wva, params, trial=None, seed=0):
        """Train the encoder with ivis on all frauds plus random legitimate rows (max_train rows in total), fit
        the logistic head on their embeddings and, for head "knn", choose the vote's k and T on validation.
        The training embeddings are kept as the vote's references. trial is accepted for the common
        interface and not used."""
        import tensorflow as tf
        from ivis import Ivis
        from sklearn.linear_model import LogisticRegression

        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.params = dict(params)
        n = min(len(Xtr), params.get("max_train", 50_000))
        rng = np.random.default_rng(seed)
        pos, neg = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
        idx = np.r_[pos, rng.choice(neg, size=max(n - len(pos), 1), replace=False)]
        X = Xtr[idx].astype(np.float32)
        self.params["d"] = int(X.shape[1])
        enc = build_encoder(X.shape[1], params["embedding_dims"], sn=params.get("sn", False), cap=params.get("cap", 1.0), drop=params.get("drop", 0.1))
        self.ivis = Ivis(embedding_dims=None, k=params["k"], model=enc, supervision_weight=params["supervision_weight"],
                         distance=params.get("distance", "pn"), epochs=params.get("epochs", 20), n_epochs_without_progress=5,
                         batch_size=512, verbose=0, n_trees=20)
        self.ivis.fit(X, ytr[idx])
        self.encoder = self.ivis.encoder_
        E = self.embed(X)
        self.head = LogisticRegression(C=params.get("C", 1.0), max_iter=2000, class_weight="balanced").fit(E, ytr[idx])
        self.cosine = "cosine" in params.get("distance", "pn")
        self.head_kind = params.get("head", "lr")
        self.knn = None
        if self.head_kind == "knn":
            iv, wv = val_subsample(yva, wva)
            a, k, T = choose_knn(E, ytr[idx], self.embed(Xva[iv]), yva[iv], wv, self.cosine)
            self.knn = KNNHead(E, ytr[idx], k, T, self.cosine)
            self.knn_val_auprc = a
        self._Eref, self._yref = E, ytr[idx]
        self._w = tf.constant(self.head.coef_.T.astype(np.float32))
        self._b = tf.constant(self.head.intercept_.astype(np.float32))
        return self

    def with_head(self, kind, Xva=None, yva=None, wva=None):
        """Same encoder, other head ("lr" or "knn"; knn needs the validation set to choose k and T)."""
        import copy

        m = copy.copy(self)
        m.head_kind = kind
        m._fn_cache = {}
        if kind == "knn" and m.knn is None:
            iv, wv = val_subsample(yva, wva)
            a, k, T = choose_knn(self._Eref, self._yref, self.embed(Xva[iv]), yva[iv], wv, self.cosine)
            m.knn = KNNHead(self._Eref, self._yref, k, T, self.cosine)
            m.knn_val_auprc = a
        return m

    def embed(self, X):
        """Encoder embeddings of X (NumPy), in batches of 8,192 rows."""
        X = np.asarray(X, np.float32)
        out = [self.encoder(X[i:i + 8192], training=False).numpy() for i in range(0, len(X), 8192)]
        return np.concatenate(out) if out else np.empty((0, self.params["embedding_dims"]), np.float32)

    def _logit_tf(self, x):
        """Fraud log-odds of the current head for a batch x (TensorFlow tensor in, tensor out)."""
        import tensorflow as tf

        E = self.encoder(x, training=False)
        if self.head_kind == "knn":
            return self.knn.logit_tf(E)
        return tf.squeeze(tf.matmul(E, self._w) + self._b, -1)

    def _fns(self):
        """Graph-compiled score and gradient for the current head (one trace per head; HopSkipJump and Square
        query in small batches, where eager dispatch would dominate)."""
        import tensorflow as tf

        key = self.head_kind
        cache = self.__dict__.setdefault("_fn_cache", {})
        if key not in cache:
            sig = [tf.TensorSpec([None, self.params["d"]], tf.float32)]

            @tf.function(input_signature=sig, reduce_retracing=True)
            def f(x):
                return self._logit_tf(x)

            @tf.function(input_signature=sig, reduce_retracing=True)
            def g(x):
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    z = self._logit_tf(x)
                return tape.gradient(z, x)

            cache[key] = (f, g)
        return cache[key]

    def logit(self, X, ctx=None):
        """Fraud log-odds of X under the current head (batches of 1,024 rows for the vote, 8,192 otherwise)."""
        import tensorflow as tf

        X = np.asarray(X, np.float32)
        f = self._fns()[0]
        bs = 1024 if self.head_kind == "knn" else 8192
        out = [f(tf.constant(X[i:i + bs])).numpy() for i in range(0, len(X), bs)]
        return np.concatenate(out) if out else np.empty(0)

    def grad(self, X, ctx=None):
        """Gradient of logit() with respect to X, in batches of 1,024 rows."""
        import tensorflow as tf

        X = np.asarray(X, np.float32)
        g = self._fns()[1]
        gs = [g(tf.constant(X[i:i + 1024])).numpy() for i in range(0, len(X), 1024)]
        return np.concatenate(gs) if gs else np.empty_like(X)

    # ---------------------------------------------------------------- persistence
    def save(self, path: Path):
        """Write the model to the folder `path` (files listed in the module docstring)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_weights(str(path / "encoder.weights.h5"))
        pickle.dump(dict(head=self.head, params=self.params, cosine=self.cosine), open(path / "head.pkl", "wb"))
        np.savez_compressed(path / "knn.npz", R=self._Eref, yR=self._yref,
                            k=self.knn.k if self.knn else 0, T=self.knn.T if self.knn else 0.0,
                            val_auprc=getattr(self, "knn_val_auprc", np.nan))

    @classmethod
    def load(cls, path: Path, head="lr"):
        """Load a model written by save(), using head `head` ("lr" or "knn")."""
        import tensorflow as tf

        path = Path(path)
        m = cls()
        h = pickle.load(open(path / "head.pkl", "rb"))
        m.head, m.params, m.cosine = h["head"], h["params"], h["cosine"]
        p = m.params
        m.encoder = build_encoder(p["d"], p["embedding_dims"], sn=p.get("sn", False), cap=p.get("cap", 1.0), drop=p.get("drop", 0.1))
        m.encoder(np.zeros((1, p["d"]), np.float32), training=False)   # build the layers before loading weights
        m.encoder.load_weights(str(path / "encoder.weights.h5"))
        z = np.load(path / "knn.npz")
        m._Eref, m._yref = z["R"], z["yR"]
        m.knn = KNNHead(z["R"], z["yR"], int(z["k"]), float(z["T"]), m.cosine) if int(z["k"]) > 0 else None
        m.head_kind = head
        m._w = tf.constant(m.head.coef_.T.astype(np.float32))
        m._b = tf.constant(m.head.intercept_.astype(np.float32))
        return m
