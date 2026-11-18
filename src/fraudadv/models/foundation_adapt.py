"""Two more ways to use Chronos-2 and TimesFM 2.5 as fraud detectors, next to the frozen encoder of
models/foundation.py. They test whether the way a foundation model is adapted matters for its robustness.

What it does
  LoRA     (chronos2_lora, timesfm_lora)  the same detector as the frozen family (same series, same head,
           same tuned head hyperparameters, same training rows), but the encoder is fine-tuned with
           low-rank adapters (Hu et al. 2022) on its attention projections, trained end to end with the
           head on the fraud labels. Rank 8, alpha 16, adapters only on the attention projections
           (Chronos-2: q, k, v, o of every time and group attention block; TimesFM 2.5: qkv_proj and out
           of every transformer layer); the pretrained weights stay frozen.
  zero-shot (chronos2_zs, timesfm_zs)     no fraud labels and no trained parameters at all: the forecaster
           predicts the card's current log-amount and log-gap from its earlier transactions (the same
           31 points the encoders see) and the score is the forecast surprise of the current point,
             s = sum over the two series of |y - q50| / max(q90 - q10, SPREAD_FLOOR),
           with q10, q50, q90 the model's own forecast quantiles. Only the 1%-FPR threshold is set on
           validation, as for every model of the study. A card with no earlier transaction cannot be
           forecast and scores 0. The forecast depends on the history only, so the score is piecewise
           linear in the current amount and gap and its gradient is exact.

The attacker sees the zero-shot score through the same two inputs as everyone else (amount and time gap
of the current transaction, both in x) and, with history injection, through the history itself.

Inputs and outputs
    Same interface and context as models/foundation.py. `make_adapt` builds a model by family key,
    `save_adapt` / `load_adapt` write and read runs/<dataset>/<family>/seed<k>.pkl (LoRA: head weights and
    adapter weights only; the pretrained backbone is loaded again from Hugging Face). The zero-shot models also keep
    their forecasts of the dataset's own transactions in runs/<dataset>/<family>/forecast_cache.npz.

Results that use it
    The LoRA and zero-shot table (Table S-XXVIII) and their points on the clean-robust Pareto frontiers
    (Figure S7). They were trained with `python -m fraudadv.hpo.train_context <d> --families
    chronos2_lora,timesfm_lora,chronos2_zs,timesfm_zs` and attacked by the grid.

LoRA training settings: learning rate of the adapters `lora_lr` (default LORA_LR = 2e-4) and at most
`lora_epochs` epochs (default LORA_EPOCHS = 12), both arguments of LoRAFoundationModel.fit and flags
--lora-lr / --lora-epochs of hpo/train_context.py; early stopping after LORA_PATIENCE epochs without a gain
in validation AUPRC; batches of LORA_BATCH rows.
"""
from __future__ import annotations

import math

import numpy as np

from fraudadv.models.foundation import (L, N_THREADS, Chronos2Backbone, FoundationModel, TimesFM25Backbone,
                                        backbone)

LORA_R, LORA_ALPHA = 8, 16
LORA_LR = 2e-4
LORA_EPOCHS = 12
LORA_PATIENCE = 3
LORA_BATCH = 256
LORA_TARGETS = {"chronos2": ("q", "k", "v", "o"), "timesfm": ("qkv_proj", "query", "key", "value", "out")}   # nn.Linear names that get adapters
SPREAD_FLOOR = 0.1        # smallest forecast spread q90 - q10 in the zero-shot score
BASE = {"chronos2_lora": "chronos2", "timesfm_lora": "timesfm", "chronos2_zs": "chronos2", "timesfm_zs": "timesfm"}
LORA_FAMS = {"chronos2_lora", "timesfm_lora"}
ZS_FAMS = {"chronos2_zs", "timesfm_zs"}
ADAPT_FAMS = LORA_FAMS | ZS_FAMS


# ---------------------------------------------------------------- LoRA
def _lora_linear_cls():
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        """y = W x + b + (alpha / r) B A x, with W, b frozen and B initialised to zero (so training starts
        from the pretrained model)."""

        def __init__(self, base, r=LORA_R, alpha=LORA_ALPHA):
            super().__init__()
            self.base = base
            self.scale = alpha / r
            w = base.weight
            self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=w.dtype, device=w.device))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=w.dtype, device=w.device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        @property
        def weight(self):                       # some blocks read .weight (dtype / device checks)
            return self.base.weight

        @property
        def bias(self):
            return self.base.bias

        @property
        def in_features(self):
            return self.base.in_features

        @property
        def out_features(self):
            return self.base.out_features

        def forward(self, x):
            return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale

    return LoRALinear


def add_lora(model, targets):
    """Wrap every nn.Linear child named in `targets` with a LoRA adapter; returns the wrapped module paths."""
    import torch.nn as nn

    LoRALinear = _lora_linear_cls()
    todo = [(pname, parent, cname, child) for pname, parent in model.named_modules()
            for cname, child in parent.named_children() if isinstance(child, nn.Linear) and cname in targets]
    for pname, parent, cname, child in todo:
        setattr(parent, cname, LoRALinear(child))
    return [f"{p}.{c}" if p else c for p, _, c, _ in todo]


def lora_state(model):
    """CPU copies of the adapter weights (parameters named lora_*) of a wrapped model."""
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if "lora_" in k}


class LoRAFoundationModel(FoundationModel):
    """Frozen-family detector with a LoRA-adapted encoder. Each instance owns its backbone copy and its
    embedding cache (embeddings depend on the adapters, so nothing is shared across seeds or families)."""

    family_key = ""

    @property
    def cache(self):
        if "_cache" not in self.__dict__:
            self._cache = {}
        return self._cache

    def _make_backbone(self):
        """A fresh backbone of this family with adapters added (not the shared frozen one)."""
        bb = Chronos2Backbone() if self.backbone_name == "chronos2" else TimesFM25Backbone()
        self.lora_modules = add_lora(bb.model, LORA_TARGETS[self.backbone_name])
        bb.model.eval()
        return bb

    def _embed(self, X, ids, grad=False):
        """As FoundationModel._embed, with this model's own adapted backbone and cache, in batches of 1024."""
        import torch

        bb = self.bb
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
        for s in range(0, len(todo), 1024):
            kk = np.array(todo[s:s + 1024])
            with torch.no_grad():
                e = bb.embed(*self._series(X[kk], ids[kk])).numpy()
            out[kk] = e
            if len(self.cache) < 600_000:
                for j, k in enumerate(kk):
                    self.cache[int(idn[k])] = (Xn[k, self.ia], Xn[k, self.ig] if self.ig is not None else None, e[j])
        return torch.from_numpy(out)

    def _embed_nocache(self, X, ids, bs=1024):
        """Embeddings without the cache, for validation during training, while the adapters still change."""
        import torch

        with torch.no_grad():
            return torch.cat([self.bb.embed(*self._series(X[i:i + bs], ids[i:i + bs])) for i in range(0, len(X), bs)])

    def fit(self, Xtr, ytr, Xva, yva, wva, params, ctx_tr, ctx_va, trial=None, seed=0,
            lora_lr=LORA_LR, lora_epochs=LORA_EPOCHS):
        """Train adapters and head together: AdamW with the head's tuned learning rate and weight decay for
        the head and `lora_lr` without weight decay for the adapters, gradients clipped to norm 1, one
        validation AUPRC per epoch, best epoch kept (head and adapters). `trial` is not used: the head
        hyperparameters are the frozen family's tuned ones."""
        import time

        import torch
        import torch.nn as nn
        from sklearn.metrics import average_precision_score

        torch.manual_seed(seed)
        torch.set_num_threads(N_THREADS)
        self.bb = self._make_backbone()
        self.lora = dict(r=LORA_R, alpha=LORA_ALPHA, lr=lora_lr, epochs=lora_epochs, batch=LORA_BATCH,
                         targets=list(LORA_TARGETS[self.backbone_name]))
        lora_params = [p for n, p in self.bb.model.named_parameters() if "lora_" in n]
        for p in lora_params:
            p.requires_grad_(True)
        Xt, it = torch.tensor(Xtr), torch.tensor(ctx_tr)
        Xv, iv = torch.tensor(Xva), torch.tensor(ctx_va)
        with torch.no_grad():
            self.emb_dim = self.bb.embed(*self._series(Xt[:2], it[:2])).shape[1]
        H = params["hidden"]
        self.proj = nn.Sequential(nn.LayerNorm(self.emb_dim), nn.Linear(self.emb_dim, params["proj"]))
        self.net = nn.Sequential(nn.Linear(Xtr.shape[1] + params["proj"], H), nn.ReLU(), nn.Dropout(params["dropout"]),
                                 nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 1))
        mods = nn.ModuleList([self.proj, self.net])
        opt = torch.optim.AdamW([{"params": list(mods.parameters()), "lr": params["lr"], "weight_decay": params["weight_decay"]},
                                 {"params": lora_params, "lr": lora_lr, "weight_decay": 0.0}])
        pos_w = torch.tensor(params.get("pos_weight", 1.0))
        yt = torch.tensor(ytr, dtype=torch.float32)
        best, state, bad = -1.0, None, 0
        self.history_ = []
        for ep in range(lora_epochs):
            t0 = time.time()
            mods.train()
            perm = torch.randperm(len(Xt))
            tot = 0.0
            for i in range(0, len(Xt), LORA_BATCH):
                b = perm[i:i + LORA_BATCH]
                e = self.bb.embed(*self._series(Xt[b], it[b]))
                loss = nn.functional.binary_cross_entropy_with_logits(self._head(Xt[b], e), yt[b], pos_weight=pos_w)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(mods.parameters()) + lora_params, 1.0)
                opt.step()
                tot += loss.item() * len(b)
            mods.eval()
            with torch.no_grad():
                ap = average_precision_score(yva, self._head(Xv, self._embed_nocache(Xv, iv)).numpy(), sample_weight=wva)
            self.history_.append(dict(epoch=ep, loss=tot / len(Xt), val_auprc=float(ap), seconds=time.time() - t0))
            print(self.dataset, self.family_key, "seed", seed, "epoch", ep, "loss", round(tot / len(Xt), 4), "val AUPRC",
                  round(float(ap), 4), f"{time.time() - t0:.0f}s", flush=True)
            if ap > best + 1e-4:
                best, bad = ap, 0
                state = ({k: v.clone() for k, v in mods.state_dict().items()}, lora_state(self.bb.model))
            else:
                bad += 1
                if bad >= LORA_PATIENCE:
                    break
        mods.load_state_dict(state[0])
        self._load_lora(state[1])
        for p in lora_params:
            p.requires_grad_(False)
        mods.eval()
        self.bb.model.eval()
        self._cache = {}
        return self

    def _load_lora(self, sd):
        """Copy adapter weights `sd` (from lora_state) into this model's backbone."""
        import torch

        named = dict(self.bb.model.named_parameters())
        with torch.no_grad():
            for k, v in sd.items():
                named[k].copy_(v.to(named[k].device, named[k].dtype))


class Chronos2LoRAModel(LoRAFoundationModel):
    backbone_name = "chronos2"
    family = "Chronos-2 (LoRA)"
    family_key = "chronos2_lora"


class TimesFMLoRAModel(LoRAFoundationModel):
    backbone_name = "timesfm"
    family = "TimesFM 2.5 (LoRA)"
    family_key = "timesfm_lora"


def save_lora(m, path):
    """Save a LoRA model: its attributes, the head weights and the adapter weights (not the backbone)."""
    import torch

    st = {k: v for k, v in m.__dict__.items() if k not in ("net", "proj", "bb", "_cache")}
    st["cls"] = type(m).__name__
    cpu = lambda sd: {k: v.detach().cpu() for k, v in sd.items()}    # noqa: E731
    st["net_state"] = cpu(m.net.state_dict())
    st["proj_state"] = cpu(m.proj.state_dict())
    st["lora_state"] = lora_state(m.bb.model)
    torch.save(st, path)


def load_lora(path, dataset, fam, meta):
    """Load a LoRA model saved by save_lora: a fresh pretrained backbone, its adapters, and the head."""
    import torch
    import torch.nn as nn

    st = torch.load(path, weights_only=False, map_location="cpu")
    m = make_adapt(fam, dataset, meta)
    for k, v in st.items():
        if k not in ("cls", "net_state", "proj_state", "lora_state"):
            setattr(m, k, v)
    m.bb = m._make_backbone()
    m._load_lora(st["lora_state"])
    for p in m.bb.model.parameters():
        p.requires_grad_(False)
    H = st["net_state"]["0.weight"].shape[0]
    proj = st["proj_state"]["1.weight"].shape[0]
    m.proj = nn.Sequential(nn.LayerNorm(m.emb_dim), nn.Linear(m.emb_dim, proj))
    m.net = nn.Sequential(nn.Linear(len(meta["features"]) + proj, H), nn.ReLU(), nn.Dropout(0.0), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 1))
    m.proj.load_state_dict(st["proj_state"])
    m.net.load_state_dict(st["net_state"])
    m.proj.eval()
    m.net.eval()
    m._cache = {}
    return m


# ---------------------------------------------------------------- zero-shot
def _chronos_forecast(bb, amt, gap, mask):
    """One-step joint forecast of both series from their histories (B, H): (B, 2, 3) = q10, q50, q90."""
    import torch

    B, H = amt.shape
    dev = bb.device
    ctx = torch.stack([amt, gap], 1).reshape(2 * B, H).to(dev)
    m = torch.stack([mask, mask], 1).reshape(2 * B, H).to(dev)
    ctx = torch.where(m, ctx, torch.full_like(ctx, float("nan")))
    group = torch.arange(B, device=dev).repeat_interleave(2)
    out = bb.model(context=ctx, context_mask=m, group_ids=group, num_output_patches=1)
    q = out.quantile_preds[:, :, 0]                                        # (2B, Q) first horizon step
    lv = [float(v) for v in bb.model.chronos_config.quantiles]
    sel = [lv.index(min(lv, key=lambda v: abs(v - t))) for t in (0.1, 0.5, 0.9)]
    return q[:, sel].reshape(B, 2, 3).float().cpu()


def _timesfm_forecast(bb, amt, gap, mask):
    """As _chronos_forecast for TimesFM 2.5: each series is forecast on its own, standardized by its observed
    mean and spread, and the quantiles are mapped back to the series' scale."""
    import torch

    B, H = amt.shape
    dev = bb.device
    outs = []
    for x in (amt, gap):
        xp = torch.cat([torch.zeros(B, L - H), x], 1).to(dev)              # left-pad the history to one patch
        mk = torch.cat([torch.zeros(B, L - H, dtype=torch.bool), mask], 1).to(dev)
        pad = ~mk
        n = mk.float().sum(1, keepdim=True).clamp(min=1)
        mu = (xp * mk).sum(1, keepdim=True) / n
        sd = (((xp - mu) ** 2) * mk).sum(1, keepdim=True).div(n).sqrt().clamp(min=1e-3)
        z = torch.where(pad, torch.zeros_like(xp), (xp - mu) / sd)
        (_, _, ts, _), _ = bb.model(z.view(-1, 1, L), pad.view(-1, 1, L))
        ts = ts.reshape(B, 1, bb.model.o, bb.model.q)[:, 0, 0, :]          # first horizon step: mean, q10 ... q90
        outs.append(mu + sd * ts[:, [1, 5, 9]])
    return torch.stack(outs, 1).float().cpu()                               # (B, 2, 3)


class ZeroShotFoundationModel(FoundationModel):
    """Forecast-surprise detector; logit() returns the surprise score (higher = more anomalous)."""

    differentiable = True
    family_key = ""

    @property
    def cache(self):
        if "_cache" not in self.__dict__:
            self._cache = {}
        return self._cache

    def fit(self, Xtr, ytr, Xva, yva, wva, params, ctx_tr, ctx_va, trial=None, seed=0):
        """Nothing to train; records the score's settings."""
        self.zero_shot = dict(spread_floor=SPREAD_FLOOR, quantiles=(0.1, 0.5, 0.9))
        return self                                                          # nothing is trained

    def _disk(self):
        """Forecasts of the dataset's own transactions are kept on disk next to the model (runs/<ds>/<fam>/), so the
        separate steps (training, grid, robust AUC, history, RL) do not forecast the same 30k transactions again. The
        file is tied to the context it was computed on (its node count); ids of injected transactions (history
        attack) are never stored."""
        from fraudadv.hpo.train_all import RUNS

        if "_n_real" not in self.__dict__:
            from fraudadv.data.load import PROC

            self._n_real = int(np.load(PROC / self.dataset / "ctx.npz")["time"].shape[0])
            self._disk_path = RUNS / self.dataset / self.family_key / "forecast_cache.npz"
            self._disk_new = 0
            if self._disk_path.exists():
                z = np.load(self._disk_path)
                if int(z["n_real"]) == self._n_real:
                    for i, f, n in zip(z["ids"], z["F"], z["nh"]):
                        self.cache[int(i)] = (f, int(n))
        return self._disk_path

    def _save_disk(self):
        """Write the forecasts of the dataset's own transactions (ids below the real node count) to disk."""
        ids = np.array([k for k in self.cache if k < self._n_real], np.int64)
        if not len(ids):
            return
        F = np.stack([self.cache[int(i)][0] for i in ids]).astype(np.float32)
        nh = np.array([self.cache[int(i)][1] for i in ids], np.int32)
        self._disk_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._disk_path.with_name("forecast_cache.tmp.npz")
        np.savez(tmp, ids=ids, F=F, nh=nh, n_real=self._n_real)
        tmp.replace(self._disk_path)
        self._disk_new = 0

    def _forecasts(self, X, ids):
        """(n, 2, 3) forecast quantiles and (n,) number of earlier transactions, from the cache when known.
        The history part of the series does not depend on X (only the current point does)."""
        import torch

        self._disk()
        ids_np = np.asarray(ids)
        F = np.zeros((len(ids_np), 2, 3), np.float32)
        nh = np.zeros(len(ids_np), np.int32)
        todo = []
        for k, i in enumerate(ids_np):
            hit = self.cache.get(int(i))
            if hit is None:
                todo.append(k)
            else:
                F[k], nh[k] = hit
        if todo:
            torch.set_num_threads(N_THREADS)
            bb = backbone(self.backbone_name)
            fc = _chronos_forecast if self.backbone_name == "chronos2" else _timesfm_forecast
            todo = np.array(todo)
            Xt = torch.as_tensor(np.asarray(X, np.float32)[todo])
            it = torch.as_tensor(ids_np[todo])
            amt, gap, mask = self._series(Xt, it)
            amt, gap, mask = amt[:, :-1], gap[:, :-1], mask[:, :-1]          # history only
            n = mask.sum(1).numpy()
            have = np.flatnonzero(n > 0)
            Fk = np.zeros((len(todo), 2, 3), np.float32)
            for s in range(0, len(have), 1024):
                h = torch.as_tensor(have[s:s + 1024])
                with torch.no_grad():
                    Fk[have[s:s + 1024]] = fc(bb, amt[h], gap[h], mask[h]).numpy()
            F[todo], nh[todo] = Fk, n
            if len(self.cache) < 2_000_000:
                for j, k in enumerate(todo):
                    self.cache[int(ids_np[k])] = (Fk[j], int(n[j]))
                self._disk_new += int(np.sum(ids_np[todo] < self._n_real))
                if self._disk_new >= 500:
                    self._save_disk()
        return F, nh

    def _current(self, X):
        """Unscaled log-amount and log-gap (None when the gap is not a feature) of the current transactions."""
        X = np.asarray(X, np.float32)
        a = X[:, self.ia].astype(np.float64) * float(self.sd[self.ia]) + float(self.mu[self.ia])
        if self.ig is not None:
            g = X[:, self.ig].astype(np.float64) * float(self.sd[self.ig]) + float(self.mu[self.ig])
        else:
            g = None
        return a, g

    def _parts(self, X, ctx):
        """Residuals r (n, 2) of the current point from the median forecast, spreads w (n, 2), and whether
        the card has any earlier transaction."""
        F, nh = self._forecasts(X, ctx)
        a, g = self._current(X)
        if g is None:                                    # gap not in x: computed from the event times
            import torch

            _, gap, _ = self._series(torch.as_tensor(np.asarray(X, np.float32)), torch.as_tensor(np.asarray(ctx)))
            g = gap[:, -1].numpy().astype(np.float64)
        w = np.maximum(F[:, :, 2] - F[:, :, 0], SPREAD_FLOOR).astype(np.float64)   # (n, 2)
        r = np.stack([a - F[:, 0, 1], g - F[:, 1, 1]], 1)                          # residuals
        return r, w, nh > 0

    def logit(self, X, ctx=None):
        r, w, ok = self._parts(X, ctx)
        return np.where(ok, np.abs(r / w).sum(1), 0.0).astype(np.float32)

    def grad(self, X, ctx=None):
        """Exact gradient of the surprise score: sign(r) / w per series, times the standardization scale."""
        r, w, ok = self._parts(X, ctx)
        g = np.zeros(np.asarray(X).shape, np.float32)
        s = np.sign(r) / w * ok[:, None]
        g[:, self.ia] = s[:, 0] * float(self.sd[self.ia])
        if self.ig is not None:
            g[:, self.ig] = s[:, 1] * float(self.sd[self.ig])
        return g


class Chronos2ZeroShotModel(ZeroShotFoundationModel):
    backbone_name = "chronos2"
    family = "Chronos-2 (zero-shot)"
    family_key = "chronos2_zs"


class TimesFMZeroShotModel(ZeroShotFoundationModel):
    backbone_name = "timesfm"
    family = "TimesFM 2.5 (zero-shot)"
    family_key = "timesfm_zs"


CLASSES = {"chronos2_lora": Chronos2LoRAModel, "timesfm_lora": TimesFMLoRAModel,
           "chronos2_zs": Chronos2ZeroShotModel, "timesfm_zs": TimesFMZeroShotModel}


def make_adapt(fam, dataset, meta):
    """An untrained model of family `fam` for `dataset` (meta from data/processed/<d>/meta.json)."""
    return CLASSES[fam](dataset, meta["features"], meta["mu"], meta["sd"])


def save_adapt(m, path):
    """Save a LoRA or zero-shot model to `path` (in-memory caches are left out)."""
    import torch

    if isinstance(m, LoRAFoundationModel):
        return save_lora(m, path)
    st = {k: v for k, v in m.__dict__.items() if k not in ("_cache", "_n_real", "_disk_path", "_disk_new")}
    st["cls"] = type(m).__name__
    torch.save(st, path)


def load_adapt(path, dataset, fam, meta):
    """Load a model saved by save_adapt."""
    import torch

    if fam in LORA_FAMS:
        return load_lora(path, dataset, fam, meta)
    st = torch.load(path, weights_only=False, map_location="cpu")
    m = make_adapt(fam, dataset, meta)
    for k, v in st.items():
        if k != "cls":
            setattr(m, k, v)
    m._cache = {}
    return m
