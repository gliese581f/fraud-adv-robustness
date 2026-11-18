"""Time-series foundation models (Chronos-2, TimesFM 2.5) as frozen history encoders.

What it does
    Both models are forecasters, so they are turned into fraud detectors the same way:
      * build the entity's history ending at the scored transaction: two series, log(1 + amount) and
        log(1 + seconds since the previous transaction), last L = 32 points (31 earlier transactions + the
        current one; shorter histories are left-padded and masked)
      * encode both series with the frozen foundation model -> one embedding (mean over tokens and series)
      * classification head = MLP on [tabular features x, projected embedding]; only the head is trained

    The current transaction's amount and time gap enter the series from the input x, so an attacker who changes
    the amount changes what the foundation model sees, and white-box gradients flow through the frozen encoder
    back to x. History points come from the context and are fixed (unless a history attack edits them).
    Embeddings of clean inputs are cached per context row, so training and evaluation encode each transaction
    once. Embeddings are computed in batches of 512 rows: Chronos-2's output depends on which rows share a
    batch, so that batch size is part of the result.

Inputs and outputs
    Interface: fit(Xtr, ytr, Xva, yva, wva, params, ctx_tr, ctx_va, trial=None, seed=0), logit(X, ctx),
    proba(X, ctx), grad(X, ctx), with ctx = row ids into data/processed/<d>/ctx.npz (models/graph.py:
    context); the history amounts come from its log_amt array. The pretrained weights come from Hugging Face
    (amazon/chronos-2, google/timesfm-2.5-200m-pytorch). hpo/train_context.py trains the head and saves it to
    runs/<dataset>/{chronos2,timesfm}/seed<k>.pkl.

Results that use it
    The foundation-model table (Table V), the foundation models in the grid tables (Table III, Table IV,
    Tables S-V to S-X), clean performance (Table S-IV), history injection (Table S-XII), and the no-embedding ablation
    of Section VI-F (hpo/fm_ablation.py). models/foundation_adapt.py builds the LoRA and zero-shot variants
    (Table S-XXVIII) on these classes.

Threads: FRAUDADV_THREADS (default 4). The backbone runs on the device of fraudadv.device; the head on CPU.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.metrics import average_precision_score

from fraudadv.data.load import PROC

from fraudadv.models.graph import context
from fraudadv.models.zoo import adversarial_loss, bound_net

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
N_THREADS = int(os.environ.get("FRAUDADV_THREADS", "4"))
L = 32
_BACKBONES: dict = {}     # one loaded backbone per name, shared by every model in the process


class Chronos2Backbone:
    """Frozen Chronos-2. The two series of a transaction are encoded together as one group (Chronos-2 lets the
    series of a group attend to each other)."""

    name = "Chronos-2"

    def __init__(self):
        from chronos import Chronos2Pipeline

        from fraudadv.device import torch_device

        self.device = torch_device()              # the frozen backbone is the expensive part: it goes on the GPU
        self.pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=str(self.device))
        self.model = self.pipe.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def embed(self, amt, gap, mask):
        """amt, gap: (B, L) float tensors; mask: (B, L) bool, True = observed."""
        import torch

        B, back = amt.shape[0], amt.device
        amt, gap, mask = amt.to(self.device), gap.to(self.device), mask.to(self.device)
        ctx = torch.stack([amt, gap], 1).reshape(2 * B, L)
        m = torch.stack([mask, mask], 1).reshape(2 * B, L)
        ctx = torch.where(m, ctx, torch.full_like(ctx, float("nan")))
        group = torch.arange(B, device=self.device).repeat_interleave(2)
        enc, *_ = self.model.encode(context=ctx, context_mask=m, group_ids=group)
        h = enc[0]                                                       # (2B, T, D)
        return h.mean(1).view(B, 2, -1).mean(1).to(back)                 # the head lives on the caller's device (CPU)


class TimesFM25Backbone:
    """Frozen TimesFM 2.5 (200M). Each series is standardized by its own observed mean and spread and encoded
    separately; the embedding is the output at the last patch, of both series side by side."""

    name = "TimesFM 2.5"

    def __init__(self):
        import timesfm

        from fraudadv.device import torch_device

        self.device = torch_device()
        self.wrap = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        self.model = self.wrap.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _one(self, x, mask):
        """Embedding (B, D) of one series x (B, L) with observed points `mask`."""
        import torch

        pad = ~mask                                                      # TimesFM: True = padded
        n = mask.float().sum(1, keepdim=True).clamp(min=1)
        mu = (x * mask).sum(1, keepdim=True) / n
        sd = (((x - mu) ** 2) * mask).sum(1, keepdim=True).div(n).sqrt().clamp(min=1e-3)
        z = torch.where(pad, torch.zeros_like(x), (x - mu) / sd)
        (_, out, _, _), _ = self.model(z.view(-1, 1, L), pad.view(-1, 1, L))
        return out[:, -1]                                                # (B, D)

    def embed(self, amt, gap, mask):
        import torch

        back = amt.device
        amt, gap, mask = amt.to(self.device), gap.to(self.device), mask.to(self.device)
        return torch.cat([self._one(amt, mask), self._one(gap, mask)], -1).to(back)


def backbone(name):
    """The shared frozen backbone "chronos2" or "timesfm", loaded on first use."""
    if name not in _BACKBONES:
        _BACKBONES[name] = Chronos2Backbone() if name == "chronos2" else TimesFM25Backbone()
    return _BACKBONES[name]


class FoundationModel:
    """Frozen-encoder detector; the subclasses below only pick the backbone.

    `features`, `mu`, `sd` (from meta.json) locate the log-amount and log-gap columns in x and undo their
    standardization, so the series carry the same unscaled values as the history. Training: AdamW on
    class-weighted binary cross-entropy over precomputed embeddings, batches of 1024 rows, one validation
    AUPRC per epoch, the best epoch kept, stop after params["patience"] epochs without a gain of more than
    1e-4. With params["adv"] set, the head trains with fraudadv.models.zoo.adversarial_loss instead (the
    defenses of Table VII)."""

    differentiable = True
    backbone_name = "chronos2"
    family = ""

    def __init__(self, dataset=None, features=None, mu=None, sd=None):
        self.dataset = dataset
        if features is not None:
            self.ia = features.index("log_amt")
            self.ig = next((i for i, f in enumerate(features) if f.endswith("log_dt_prev")), None)
            self.mu, self.sd = np.asarray(mu, np.float32), np.asarray(sd, np.float32)

    @property
    def cache(self):
        # shared across instances (HPO trials, seeds): clean embeddings depend only on the frozen backbone
        return _EMB_CACHE.setdefault((self.backbone_name, self.dataset), {})

    # ---- series construction
    def _series(self, X, ids):
        """(amt, gap, mask), each (B, L), oldest point first and the current transaction last. The history
        is the card's L - 1 most recent earlier transactions (nbr1 of the context); the current point is
        read from X, so it follows any perturbation of x."""
        import torch

        c = context(self.dataset, cpu=True)                              # series are built on CPU, encoded on the device
        nb = c["nbr1"][ids, : L - 1]                                     # most recent first
        hist_ok = nb >= 0
        lam = _log_amt(self.dataset)
        T = c["time"]
        a_hist = lam[nb.clamp(min=0)]
        t_hist = T[nb.clamp(min=0)]
        t_cur = T[ids].unsqueeze(1)
        prev_t = torch.cat([t_hist[:, 1:], t_hist[:, -1:]], 1)
        g_hist = torch.log1p((t_hist - prev_t).clamp(min=0)).float()
        g_hist = torch.where(torch.cat([hist_ok[:, 1:], torch.zeros_like(hist_ok[:, :1])], 1), g_hist, torch.zeros_like(g_hist))
        # current point from x (attackable), unscaled
        x = X
        a_cur = x[:, self.ia] * float(self.sd[self.ia]) + float(self.mu[self.ia])
        if self.ig is not None:
            g_cur = x[:, self.ig] * float(self.sd[self.ig]) + float(self.mu[self.ig])
        else:
            g_cur = torch.log1p((t_cur.squeeze(1) - t_hist[:, 0]).clamp(min=0)).float()
        # oldest -> newest, current last
        amt = torch.cat([a_hist.flip(1), a_cur.unsqueeze(1)], 1)
        gap = torch.cat([g_hist.flip(1), g_cur.unsqueeze(1)], 1)
        mask = torch.cat([hist_ok.flip(1), torch.ones_like(hist_ok[:, :1])], 1)
        return amt.float(), gap.float(), mask

    def _embed(self, X, ids, grad=False):
        """Embeddings of rows (X, ids). grad=True: computed with autograd, no cache. Otherwise a cached
        embedding is reused when the row's amount and gap are the cached ones, and the rest are encoded in
        batches of 512 rows (the cache holds up to 600,000 rows)."""
        import torch

        bb = backbone(self.backbone_name)
        if grad:
            return bb.embed(*self._series(X, ids))
        out = np.zeros((len(ids), self.emb_dim), np.float32)
        Xn = X.detach().numpy()
        idn = ids.numpy()
        todo = []
        for k, (i, xa) in enumerate(zip(idn, Xn[:, self.ia])):
            hit = self.cache.get(int(i))
            if hit is not None and hit[0] == xa and (self.ig is None or hit[1] == Xn[k, self.ig]):
                out[k] = hit[2]
            else:
                todo.append(k)
        for s in range(0, len(todo), 512):
            kk = np.array(todo[s:s + 512])
            with torch.no_grad():
                e = bb.embed(*self._series(X[kk], ids[kk])).numpy()
            out[kk] = e
            if len(self.cache) < 600_000:
                for j, k in enumerate(kk):
                    self.cache[int(idn[k])] = (Xn[k, self.ia], Xn[k, self.ig] if self.ig is not None else None, e[j])
        return torch.from_numpy(out)

    def _head(self, x, e):
        """Logit from tabular features x and embedding e."""
        import torch

        return self.net(torch.cat([x, self.proj(e)], -1)).squeeze(-1)

    def fit(self, Xtr, ytr, Xva, yva, wva, params, ctx_tr, ctx_va, trial=None, seed=0):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        torch.set_num_threads(N_THREADS)
        self.emb_dim = backbone(self.backbone_name).embed(*self._series(torch.tensor(Xtr[:2]), torch.tensor(ctx_tr[:2]))).shape[1]
        Etr = self._embed(torch.tensor(Xtr), torch.tensor(ctx_tr))
        Eva = self._embed(torch.tensor(Xva), torch.tensor(ctx_va))
        H = params["hidden"]
        self.proj = nn.Sequential(nn.LayerNorm(self.emb_dim), nn.Linear(self.emb_dim, params["proj"]))
        self.net = nn.Sequential(nn.Linear(Xtr.shape[1] + params["proj"], H), nn.ReLU(), nn.Dropout(params["dropout"]),
                                 nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 1))
        mods = nn.ModuleList([self.proj, self.net])
        opt = torch.optim.AdamW(mods.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        pos_w = torch.tensor(params.get("pos_weight", 1.0))
        Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)
        Xv = torch.tensor(Xva)
        best, state, bad = -1.0, None, 0
        adv = params.get("adv")      # None: plain training; a dict: adversarial training (fraudadv.models.zoo.adversarial_loss)
        for ep in range(params.get("epochs", 30)):
            mods.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 1024):
                b = perm[i:i + 1024]
                if adv is None:
                    loss = nn.functional.binary_cross_entropy_with_logits(self._head(Xt[b], Etr[b]), yt[b], pos_weight=pos_w)
                else:
                    # the head's tabular input is perturbed with the clean embedding held fixed: the encoder is
                    # frozen, and re-encoding every inner step would multiply the cost of training many times over
                    # (the evaluation attacks do re-encode every perturbed series)
                    e = Etr[b]
                    loss = adversarial_loss(bound_net(lambda x: self._head(x, e), self.proj, self.net), Xt[b], yt[b], pos_w, adv)
                opt.zero_grad()
                loss.backward()
                opt.step()
            mods.eval()
            with torch.no_grad():
                ap = average_precision_score(yva, self._head(Xv, Eva).numpy(), sample_weight=wva)
            if trial is not None:
                trial.report(ap, ep)
                if trial.should_prune():
                    import optuna

                    raise optuna.TrialPruned()
            if ap > best + 1e-4:
                best, bad, state = ap, 0, {k: v.clone() for k, v in mods.state_dict().items()}
            else:
                bad += 1
                if bad >= params.get("patience", 5):
                    break
        mods.load_state_dict(state)
        mods.eval()
        return self

    def logit(self, X, ctx=None):
        import torch

        X = torch.tensor(np.asarray(X, np.float32))
        ids = torch.tensor(np.asarray(ctx))
        with torch.no_grad():
            return self._head(X, self._embed(X, ids)).numpy()

    def proba(self, X, ctx=None):
        return 1 / (1 + np.exp(-self.logit(X, ctx)))

    def grad(self, X, ctx=None):
        import torch

        x = torch.tensor(np.asarray(X, np.float32), requires_grad=True)
        ids = torch.tensor(np.asarray(ctx))
        self._head(x, self._embed(x, ids, grad=True)).sum().backward()
        g = x.grad.numpy()
        # The amount and time-gap features also enter the card's series, and autograd through the backbone's
        # instance scaling is unreliable there: a series with one observation, or with near-constant
        # values, has (near-)zero spread, and the backward pass returns NaN or values of order 1e15 while the
        # score itself is finite and smooth. Those two columns get a central finite difference instead.
        X0 = np.asarray(X, np.float32)
        c0 = np.asarray(ctx)
        h = 1e-2
        for j in [k for k in (self.ia, self.ig) if k is not None]:
            Xp, Xm = X0.copy(), X0.copy()
            Xp[:, j] += h
            Xm[:, j] -= h
            g[:, j] = (self.logit(Xp, c0) - self.logit(Xm, c0)) / (2 * h)
        return np.nan_to_num(g)


_LOGAMT: dict = {}        # dataset -> unscaled log-amount of every context row (history injection appends rows)
_EMB_CACHE: dict = {}     # (backbone, dataset) -> {context row: (amount, gap, embedding)}


def _log_amt(dataset):
    """Unscaled log(1 + amount) of every transaction of the context, as a torch tensor."""
    import torch

    if dataset not in _LOGAMT:
        _LOGAMT[dataset] = torch.from_numpy(np.load(PROC / dataset / "ctx.npz")["log_amt"])
    return _LOGAMT[dataset]


class Chronos2Model(FoundationModel):
    backbone_name = "chronos2"
    family = "Chronos-2"


class TimesFMModel(FoundationModel):
    backbone_name = "timesfm"
    family = "TimesFM 2.5"
