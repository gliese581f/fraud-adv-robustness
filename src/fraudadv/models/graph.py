"""Temporal heterogeneous graph models over transaction histories: HGNN and HGT.

What it does
    Graph: every transaction is a node carrying the same standardized features the tabular models see. Two
    typed relations link a transaction to strictly earlier transactions:
      r1  same card / account (the entity's own history)
      r2  same merchant (Sparkov, TabFormer), same card1 issuer group (IEEE-CIS); none for PaySim
    Edges only point to the past, so no future information leaks into a score. Neighborhoods are sampled
    GraphSAGE-style: the K1 most recent r1 and K2 most recent r2 neighbors per node, two hops (targets ->
    neighbors -> neighbors' neighbors; K1b, K2b on the second hop).

    Models
      HeteroGNN         relation-specific mean aggregation (R-GCN / HeteroSAGE), 2 layers
      GraphTransformer  HGT-style multi-head attention over typed neighbors with relation-specific key/value
                        maps and relative temporal encoding (Hu et al., WWW 2020), 2 layers

    The scored transaction's own features come from the input X, so attacks can perturb them; its history
    comes from the context matrix and is fixed unless a history attack changes it.

Inputs and outputs
    Interface: fit(Xtr, ytr, Xva, yva, wva, params, ctx_tr, ctx_va, trial=None, seed=0), logit(X, ctx),
    proba(X, ctx), grad(X, ctx), with ctx = row ids into the dataset's context, data/processed/<d>/ctx.npz
    (written by `python -m fraudadv.data.prepare <d> --context`): Z (node features), nbr1 / nbr2 (neighbor ids,
    most recent first, -1 = none) and time (event time in seconds). hpo/train_context.py trains these models
    and saves their weights to runs/<dataset>/{hgnn,hgt}/seed<k>.pkl.

Results that use it
    The graph families in every grid result (Table III, Table IV, Tables S-V to S-X), clean performance (Table S-IV),
    their hyperparameters (Table S-II), transfer (Figure 1) and history injection (Table S-XII).

Threads: FRAUDADV_THREADS (default 4).
"""
from __future__ import annotations

import math
import os

import numpy as np
from sklearn.metrics import average_precision_score

from fraudadv.data.load import PROC
from fraudadv.models.zoo import adversarial_loss, bound_net

N_THREADS = int(os.environ.get("FRAUDADV_THREADS", "4"))
_CTX: dict = {}       # (dataset, "cpu") -> CPU copy of the context; dataset -> the copy on the compute device


def context(dataset, cpu=False):
    """Graph / sequence context of a dataset as torch tensors. The default copy lives on the compute
    device when it fits there (see fraudadv.device); cpu=True returns the CPU copy, which the foundation
    models use to build their series."""
    import torch

    if (dataset, "cpu") not in _CTX:
        z = np.load(PROC / dataset / "ctx.npz")
        _CTX[dataset, "cpu"] = {"Z": torch.from_numpy(np.ascontiguousarray(z["Z"])), "nbr1": torch.from_numpy(z["nbr1"].astype(np.int64)),
                                "nbr2": torch.from_numpy(z["nbr2"].astype(np.int64)), "time": torch.from_numpy(z["time"])}
    c = _CTX[dataset, "cpu"]
    if cpu:
        return c
    if dataset not in _CTX:
        from fraudadv.device import torch_device

        dev = torch_device()
        if dev.type == "cuda":
            need = sum(t.numel() * t.element_size() for t in c.values())
            free, _ = torch.cuda.mem_get_info(dev)
            if need > 0.6 * free:
                print(f"{dataset}: graph context ({need / 1e9:.1f} GB) does not fit the GPU ({free / 1e9:.1f} GB free); "
                      "graph models stay on CPU", flush=True)
                dev = torch.device("cpu")
        _CTX[dataset] = c if dev.type == "cpu" else {k: v.to(dev) for k, v in c.items()}
    return _CTX[dataset]


def _time_enc(dt, dim=8):
    """Sinusoidal relative-time encoding of log(1 + dt seconds)."""
    import torch

    x = torch.log1p(dt.clamp(min=0)).float().unsqueeze(-1)
    freqs = torch.exp(torch.arange(dim // 2, dtype=x.dtype, device=x.device) * (-math.log(10.0) / (dim // 2)))
    return torch.cat([torch.sin(x * freqs), torch.cos(x * freqs)], -1)


def build_net(kind, d, p):
    """The network of a graph model: kind "hgnn" (SageLayer) or "hgt" (HGTLayer), d input features, p the
    hyperparameters (hidden, dropout, n_heads). Also used by hpo/train_context.py to rebuild a saved model
    before loading its weights."""
    import torch
    import torch.nn as nn

    H, TE = p["hidden"], 8          # hidden width; width of the time encoding

    class SageLayer(nn.Module):
        """out = Dropout(ReLU(LayerNorm(W_self h + sum over relations r of the mean over observed neighbors of
        W_r [h_neighbor, time encoding])))."""

        def __init__(self, din):
            super().__init__()
            self.self_ = nn.Linear(din, H)
            self.rel = nn.ModuleList([nn.Linear(din + TE, H, bias=False) for _ in range(2)])
            self.norm = nn.LayerNorm(H)
            self.drop = nn.Dropout(p["dropout"])

        def forward(self, h, nb, mask, te):
            # h (B, din); nb list over relations of (B, K, din); mask (B, K); te (B, K, TE)
            out = self.self_(h)
            for r in range(2):
                m = mask[r].unsqueeze(-1).float()
                msg = self.rel[r](torch.cat([nb[r], te[r]], -1)) * m
                out = out + msg.sum(1) / m.sum(1).clamp(min=1.0)
            return self.drop(torch.relu(self.norm(out)))

    class HGTLayer(nn.Module):
        """Multi-head attention of the node over its r1 neighbors, its r2 neighbors and itself, with separate
        key and value maps and a learned attention prior per relation; then a residual feed-forward block."""

        def __init__(self, din):
            super().__init__()
            self.nh = p["n_heads"]
            self.dk = H // self.nh
            self.q = nn.Linear(din, H)
            self.k = nn.ModuleList([nn.Linear(din + TE, H) for _ in range(3)])      # r1, r2, self
            self.v = nn.ModuleList([nn.Linear(din + TE, H) for _ in range(3)])
            self.prior = nn.Parameter(torch.zeros(3, self.nh))                     # relation-specific attention prior
            self.o = nn.Linear(H, H)
            self.skip = nn.Linear(din, H)
            self.n1, self.n2 = nn.LayerNorm(H), nn.LayerNorm(H)
            self.ff = nn.Sequential(nn.Linear(H, 2 * H), nn.GELU(), nn.Linear(2 * H, H))
            self.drop = nn.Dropout(p["dropout"])

        def forward(self, h, nb, mask, te):
            B = h.shape[0]
            q = self.q(h).view(B, 1, self.nh, self.dk)
            ks, vs, ms, pr = [], [], [], []
            zero_te = torch.zeros(B, 1, TE, dtype=h.dtype, device=h.device)
            self_mask = torch.ones(B, 1, dtype=torch.bool, device=h.device)
            items = [(nb[0], te[0], mask[0], 0), (nb[1], te[1], mask[1], 1), (h.unsqueeze(1), zero_te, self_mask, 2)]
            for x, t, m, r in items:
                xin = torch.cat([x, t], -1)
                ks.append(self.k[r](xin).view(B, -1, self.nh, self.dk))
                vs.append(self.v[r](xin).view(B, -1, self.nh, self.dk))
                ms.append(m)
                pr.append(self.prior[r].expand(B, x.shape[1], self.nh))
            K, V, M, P = torch.cat(ks, 1), torch.cat(vs, 1), torch.cat(ms, 1), torch.cat(pr, 1)
            att = (q * K).sum(-1) / math.sqrt(self.dk) + P                         # (B, N, nh)
            att = att.masked_fill(~M.unsqueeze(-1), -1e9).softmax(1)
            agg = (att.unsqueeze(-1) * V).sum(1).reshape(B, H)
            z = self.n1(self.skip(h) + self.drop(self.o(agg)))
            return self.n2(z + self.drop(self.ff(z)))

    Layer = SageLayer if kind == "hgnn" else HGTLayer

    class Net(nn.Module):
        """Two graph layers over the two-hop neighborhood, then a head on [layer-2 output, x]."""

        def __init__(self):
            super().__init__()
            self.l1 = Layer(d)
            self.l2 = Layer(H)
            self.head = nn.Sequential(nn.Linear(H + d, H), nn.ReLU(), nn.Linear(H, 1))

        def forward(self, x, ids, ctx, K1, K2, K1b, K2b):
            Z, n1, n2, T = ctx["Z"], ctx["nbr1"], ctx["nbr2"], ctx["time"]
            B = x.shape[0]
            t0 = T[ids]
            # 1-hop neighbors of targets; -1 marks a missing neighbor (masked, index clamped to 0)
            a1, a2 = n1[ids, :K1], n2[ids, :K2]                                 # (B, K)
            m1, m2 = a1 >= 0, a2 >= 0
            a1c, a2c = a1.clamp(min=0), a2.clamp(min=0)
            hop1 = torch.cat([a1c, a2c], 1)                                      # (B, K1+K2)
            hop1_mask = torch.cat([m1, m2], 1)
            # 2-hop: neighbors of each 1-hop node
            flat = hop1.reshape(-1)
            b1, b2 = n1[flat, :K1b], n2[flat, :K2b]
            mb1, mb2 = (b1 >= 0) & hop1_mask.reshape(-1, 1), (b2 >= 0) & hop1_mask.reshape(-1, 1)
            tf = T[flat]
            te_b = [_time_enc(tf.unsqueeze(1) - T[b1.clamp(min=0)]), _time_enc(tf.unsqueeze(1) - T[b2.clamp(min=0)])]
            h_hop1 = self.l1(Z[flat], [Z[b1.clamp(min=0)], Z[b2.clamp(min=0)]], [mb1, mb2], te_b)    # (B*(K1+K2), H)
            te_a = [_time_enc(t0.unsqueeze(1) - T[a1c]), _time_enc(t0.unsqueeze(1) - T[a2c])]
            # The target's own features come from x (attackable), its neighbors' from the context Z.
            h_t1 = self.l1(x, [Z[a1c], Z[a2c]], [m1, m2], te_a)                                       # targets, layer 1
            h_hop1 = h_hop1.view(B, K1 + K2, -1)
            h_t2 = self.l2(h_t1, [h_hop1[:, :K1], h_hop1[:, K1:]], [m1, m2], te_a)                  # targets, layer 2
            return self.head(torch.cat([h_t2, x], -1)).squeeze(-1)

    return Net()


class GraphModel:
    """Training and scoring shared by the two graph families.

    Training: AdamW on class-weighted binary cross-entropy, mini-batches in a random order drawn from the seed
    (optionally only the first params["steps_per_epoch"] batches of each epoch), one validation AUPRC per
    epoch, and the weights of the best epoch kept; stops after params["patience"] epochs without a gain of
    more than 1e-4. With params["adv"] set, each batch uses the adversarial-training loss of
    fraudadv.models.zoo.adversarial_loss instead (the defenses of Table VII)."""

    differentiable = True
    kind = "hgnn"
    family = "Heterogeneous GNN"

    def __init__(self, dataset=None):
        self.dataset = dataset

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items()}

    def _dev(self):
        """The device of the dataset's graph context, with the network moved to it."""
        dev = context(self.dataset)["Z"].device
        p = next(self.net.parameters(), None)
        if p is not None and p.device != dev:
            self.net.to(dev)
        return dev

    def _fwd(self, x, ids):
        p = self.params
        return self.net(x, ids, context(self.dataset), p["K1"], p["K2"], p["K1b"], p["K2b"])

    def fit(self, Xtr, ytr, Xva, yva, wva, params, ctx_tr, ctx_va, trial=None, seed=0):
        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(N_THREADS)
        self.params = dict(params)
        self.net = build_net(self.kind, Xtr.shape[1], self.params)
        dev = self._dev()
        opt = torch.optim.AdamW(self.net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        pos_w = torch.tensor(params.get("pos_weight", 1.0), device=dev)
        Xt, yt, it = torch.tensor(Xtr, device=dev), torch.tensor(ytr, dtype=torch.float32, device=dev), torch.tensor(ctx_tr, device=dev)
        batch_size = params.get("batch_size", 512)
        best_auprc, best_state, epochs_since_gain = -1.0, None, 0
        steps_per_epoch = params.get("steps_per_epoch", None)
        adv = params.get("adv")      # None: plain training; a dict: adversarial training (fraudadv.models.zoo.adversarial_loss)
        for epoch in range(params.get("epochs", 10)):
            self.net.train()
            order = torch.randperm(len(Xt), device=dev)
            if steps_per_epoch:
                order = order[: steps_per_epoch * batch_size]
            for start in range(0, len(order), batch_size):
                rows = order[start:start + batch_size]
                if adv is None:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(self._fwd(Xt[rows], it[rows]), yt[rows], pos_weight=pos_w)
                else:                # the node's own features are perturbed; its history (the context) stays fixed
                    ids = it[rows]
                    loss = adversarial_loss(bound_net(lambda x: self._fwd(x, ids), self.net), Xt[rows], yt[rows], pos_w, adv)
                opt.zero_grad()
                loss.backward()
                opt.step()
            val_auprc = average_precision_score(yva, self.logit(Xva, ctx_va), sample_weight=wva)
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
                if epochs_since_gain >= params.get("patience", 3):
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
            for start in range(0, len(X), 4096):
                xb = torch.tensor(np.asarray(X[start:start + 4096], np.float32), device=dev)
                out.append(self._fwd(xb, torch.tensor(np.asarray(ctx[start:start + 4096]), device=dev)).cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)

    def proba(self, X, ctx=None):
        return 1 / (1 + np.exp(-self.logit(X, ctx)))

    def grad(self, X, ctx=None):
        import torch

        self.net.eval()
        dev = self._dev()
        x = torch.tensor(np.asarray(X, np.float32), device=dev, requires_grad=True)
        self._fwd(x, torch.tensor(np.asarray(ctx), device=dev)).sum().backward()
        return x.grad.cpu().numpy()


class HeteroGNN(GraphModel):
    kind = "hgnn"
    family = "Heterogeneous GNN"


class GraphTransformer(GraphModel):
    kind = "hgt"
    family = "Graph Transformer (HGT)"
