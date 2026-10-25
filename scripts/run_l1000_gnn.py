#!/usr/bin/env python3
"""Approach 2 as a trained model: a drug enters a PPI graph at its targets; a GNN predicts its signature.

Pre-registered 2026-10-03 as G3b in report/10_09_record.md. Same slices, folds (run.make_folds),
cohorts, metrics and row schema as run_l1000_unseen_drug.py, so every row pairs with that
runner's per_condition.parquet. Multi-task over the lines of a slice; the target is the
residual from the runner's `mean` arm (training-compound mean per line, all compounds).

Arms (graph arms and mlp_target/mlp_chem train on KNOWN compounds; mlp_chem_all on all):
  gnn_ppi       STRING physical (score >= 400); decoupled-propagation GNN (SGC/SIGN family): the drug's
                target signal propagated 0..3 hops and propagated learnable gene embeddings
                (the per-drug 2-layer GCN of the original spec ran at ~55 s/epoch on CPU; see record)
  gnn_rewired   same, degree-preserving rewired graph (seed 0, as the kNN control)
  gnn_nograph   same, identity adjacency (no message passing)
  mlp_target    MLP on [multi-hot targets ; chemistry embedding]
  mlp_chem      MLP on Morgan bits, KNOWN training compounds
  mlp_chem_all  MLP on Morgan bits, all training compounds
  mlp_ppi       mlp_target + the PPI-propagated target signal (hops 0..3) at every landmark gene;
                mlp_ppi_rewired / mlp_ppi_nograph swap the graph (added 2026-10-04, exploratory)
  dot_ppi       DecoupledGNN encoder with a dot-product head (d' W_line g + per-gene propagated-signal
                term); dot_rewired / dot_nograph swap the graph (added 2026-10-04, exploratory)
  gcn_ppi       the G3b design as pre-registered (2026-10-03 06:55): the drug enters the graph at its target
                nodes (indicator x [learned vector + chemistry MLP] added to learnable gene embeddings), 2 GCN
                layers per drug, per-line readout over [landmark node state ; target-pooled state + chemistry].
                gcn_rewired / gcn_nograph swap the graph. Run on GPU (added 2026-10-07; not in the default arms)
Fixed a priori: Adam 1e-3, weight decay 1e-5, <= 150 epochs, early stopping patience 15 on the
first inner split (criterion: full Pearson on ACTIVE validation rows), best-epoch weights.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

import run_l1000_unseen_drug as run

HIDDEN = 64
# Output shrinkage chosen on the inner validation split, as every kNN/KRR arm has (alpha / lambda).
# Added 2026-10-04 after the first full runs early-stopped at epoch 1 in every fold (record).
GRID_ALPHA = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
LR, WD, MAX_EPOCHS, PATIENCE = 1e-3, 1e-5, 150, 15
SHRINK = False
BATCH_GNN, BATCH_MLP = 64, 64
N_HOPS = 3
GRAPH_ARMS = ("gnn_ppi", "gnn_rewired", "gnn_nograph")
PROP_ARMS = {"mlp_ppi": "gnn_ppi", "mlp_ppi_rewired": "gnn_rewired", "mlp_ppi_nograph": "gnn_nograph"}
DOT_ARMS = {"dot_ppi": "gnn_ppi", "dot_rewired": "gnn_rewired", "dot_nograph": "gnn_nograph"}
GCN_ARMS = {"gcn_ppi": "gnn_ppi", "gcn_rewired": "gnn_rewired", "gcn_nograph": "gnn_nograph"}
ALL_ARMS = GRAPH_ARMS + ("mlp_target", "mlp_chem", "mlp_chem_all") + tuple(PROP_ARMS) + tuple(DOT_ARMS)
DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- models

def normalized_adjacency(A: sp.csr_matrix | None, n: int) -> torch.Tensor:
    """D^-1/2 (A + I) D^-1/2 as a torch sparse COO tensor; A=None gives the identity."""
    M = sp.identity(n, format="csr") if A is None else (A + sp.identity(n, format="csr")).tocsr()
    deg = np.asarray(M.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    M = (dinv @ M @ dinv).tocoo()
    idx = torch.tensor(np.vstack([M.row, M.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(M.data, dtype=torch.float32), (n, n)).coalesce().to_sparse_csr()


def spmm_batch(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """A (n x n sparse) applied to every drug of X (B x n x h) -> (B x n x h)."""
    B, n, h = X.shape
    out = torch.sparse.mm(A, X.permute(1, 0, 2).reshape(n, B * h))
    return out.reshape(n, B, h).permute(1, 0, 2)


class ChemEncoder(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, HIDDEN))

    def forward(self, x):
        return self.net(x)


class DecoupledGNN(nn.Module):
    """Decoupled-propagation GNN (SGC/SIGN family). Graph enters twice: (i) the drug's target
    signal propagated 0..K hops over the normalized adjacency, read at every landmark gene
    (precomputed, `prop`), and (ii) learnable gene embeddings propagated over the same graph on
    every forward pass. A drug readout pools propagated embeddings at its target nodes and adds
    the chemistry embedding. Swapping the adjacency gives the rewired / no-graph controls."""

    def __init__(self, A: torch.Tensor, n_nodes: int, landmark_nodes: np.ndarray, n_lines: int, n_chem: int,
                 n_hops: int):
        super().__init__()
        self.A = A
        self.emb = nn.Parameter(torch.randn(n_nodes, HIDDEN) * 0.05)
        self.chem = ChemEncoder(n_chem)
        self.register_buffer("lm", torch.tensor(landmark_nodes, dtype=torch.long))
        self.gene_proj = nn.Linear(3 * HIDDEN, HIDDEN)
        self.prop_proj = nn.Linear(n_hops + 1, HIDDEN)
        self.drug_proj = nn.Linear(4 * HIDDEN, HIDDEN)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(0.1), nn.Linear(3 * HIDDEN, HIDDEN), nn.ReLU(),
                                  nn.Linear(HIDDEN, n_lines))
        self.bias = nn.Parameter(torch.zeros(n_lines, len(landmark_nodes)))

    def forward(self, T: torch.Tensor, x: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        """T: (B, n_nodes) target weights; x: (B, n_chem); prop: (B, G, K+1) -> (B, n_lines, G)."""
        E0 = self.emb
        E1 = self.A @ E0
        E2 = self.A @ E1
        Ecat = torch.cat([E0, E1, E2], 1)                              # (n, 3H)
        readout = T @ Ecat                                             # (B, 3H) target-node pooling
        d = self.drug_proj(torch.cat([readout, self.chem(x)], 1))      # (B, H)
        g = self.gene_proj(Ecat[self.lm])                              # (G, H)
        p = self.prop_proj(prop)                                       # (B, G, H)
        B, G = p.shape[0], g.shape[0]
        z = torch.cat([g.unsqueeze(0).expand(B, -1, -1), p, d.unsqueeze(1).expand(-1, G, -1)], 2)
        return self.head(z).permute(0, 2, 1) + self.bias               # (B, L, G)


class DotGNN(DecoupledGNN):
    """DecoupledGNN with the concatenation-MLP head replaced by a dot-product head (added 2026-10-04,
    exploratory, user request): per line l, out[b, l, g] = d_b' W_l g_g + sum_k beta[l, g, k] prop[b, g, k]
    + bias[l, g]. Same encoder, so a difference from DecoupledGNN is attributable to the head."""

    def __init__(self, A: torch.Tensor, n_nodes: int, landmark_nodes: np.ndarray, n_lines: int, n_chem: int,
                 n_hops: int):
        super().__init__(A, n_nodes, landmark_nodes, n_lines, n_chem, n_hops)
        del self.head, self.prop_proj
        self.W = nn.Parameter(torch.randn(n_lines, HIDDEN, HIDDEN) / HIDDEN)
        self.beta = nn.Parameter(torch.zeros(n_lines, len(landmark_nodes), n_hops + 1))

    def forward(self, T: torch.Tensor, x: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        Ecat = torch.cat([self.emb, self.A @ self.emb, self.A @ (self.A @ self.emb)], 1)
        d = self.drug_proj(torch.cat([T @ Ecat, self.chem(x)], 1))                 # (B, H)
        g = self.gene_proj(Ecat[self.lm])                                         # (G, H)
        dot = torch.einsum("bh,lhk,gk->blg", d, self.W, g)
        local = torch.einsum("bgk,lgk->blg", prop, self.beta)
        return dot + local + self.bias                                            # (B, L, G)


class GCN(nn.Module):
    """The pre-registered G3b GNN (record 2026-10-03 06:55): per-drug message passing. Node state
    H0 = E + 1[target] * (v + chem(x)); two GCN layers over the normalized adjacency; per-line readout over
    [landmark node state ; target-pooled state + chemistry]. Unlike DecoupledGNN, every landmark state
    depends on the drug, so the concatenation head can express drug x gene interactions."""

    def __init__(self, A: torch.Tensor, n_nodes: int, landmark_nodes: np.ndarray, n_lines: int, n_chem: int):
        super().__init__()
        self.A = A
        self.emb = nn.Parameter(torch.randn(n_nodes, HIDDEN) * 0.05)
        self.target_vec = nn.Parameter(torch.randn(HIDDEN) * 0.05)
        self.chem = ChemEncoder(n_chem)
        self.register_buffer("lm", torch.tensor(landmark_nodes, dtype=torch.long))
        self.W1 = nn.Linear(HIDDEN, HIDDEN)
        self.W2 = nn.Linear(HIDDEN, HIDDEN)
        self.drug_proj = nn.Linear(2 * HIDDEN, HIDDEN)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(0.1), nn.Linear(2 * HIDDEN, HIDDEN), nn.ReLU(),
                                  nn.Linear(HIDDEN, n_lines))
        self.bias = nn.Parameter(torch.zeros(n_lines, len(landmark_nodes)))

    def forward(self, T: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """T: (B, n_nodes) target weights (1/n_targets); x: (B, n_chem) -> (B, n_lines, G)."""
        c = self.chem(x)                                                           # (B, H)
        H = self.emb.unsqueeze(0) + (T > 0).float().unsqueeze(-1) * (self.target_vec + c).unsqueeze(1)
        H = torch.relu(spmm_batch(self.A, self.W1(H)))                             # (B, n, H)
        H = spmm_batch(self.A, self.W2(H))
        d = self.drug_proj(torch.cat([torch.einsum("bn,bnh->bh", T, H), c], 1))    # (B, H)
        L = H[:, self.lm]                                                          # (B, G, H)
        z = torch.cat([L, d.unsqueeze(1).expand(-1, L.shape[1], -1)], 2)
        return self.head(z).permute(0, 2, 1) + self.bias                           # (B, L, G)


class FlatMLP(nn.Module):
    def __init__(self, n_in: int, n_lines: int, n_genes: int, chem_dim: int | None = None):
        super().__init__()
        self.chem = ChemEncoder(chem_dim) if chem_dim else None
        width = n_in + (HIDDEN if chem_dim else 0)
        self.net = nn.Sequential(nn.Linear(width, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 256), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(256, n_lines * n_genes))
        self.shape = (n_lines, n_genes)

    def forward(self, a: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        z = a if self.chem is None else torch.cat([a, self.chem(x)], 1)
        return self.net(z).reshape(-1, *self.shape)


# --------------------------------------------------------------------------- training

def masked_pearson_full(pred_res, R, M, active, m):
    """Mean row-wise Pearson of (m + pred) vs (m + R) over measured ACTIVE (compound, line) rows."""
    P = (pred_res + m[None]).reshape(-1, R.shape[-1]); Y = (R + m[None]).reshape(-1, R.shape[-1])
    keep = (M & active).reshape(-1)
    if keep.sum() == 0:
        return np.nan
    return float(run.rowwise_pearson(P[keep], Y[keep]).mean())


def train_model(model, inputs_fn, idx_fit, idx_val, R, M, active, m, batch, log, label):
    """Masked-MSE training with early stopping on validation full Pearson; returns best state."""
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    Rt = torch.tensor(R, dtype=torch.float32, device=DEVICE); Mt = torch.tensor(M, dtype=torch.float32, device=DEVICE)
    best, best_state, bad, rng = -np.inf, None, 0, np.random.default_rng(0)
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(idx_fit)
        total = 0.0
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            out = model(*(t.to(DEVICE) for t in inputs_fn(b)))
            w = Mt[b].unsqueeze(-1)
            loss = (((out - Rt[b]) ** 2) * w).sum() / (w.sum() * R.shape[-1]).clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * len(b)
        pv = predict(model, inputs_fn, idx_val, batch)
        alphas = GRID_ALPHA if SHRINK else (1.0,)
        scores = {a: masked_pearson_full(a * pv, R[idx_val], M[idx_val], active[idx_val], m) for a in alphas}
        a_best = max(scores, key=scores.get); score = scores[a_best]
        log.append({"arm": label, "epoch": epoch, "train_mse": total / max(1, len(order)), "val_pearson": score,
                    "alpha": a_best})
        if score > best + 1e-5:
            best, bad, best_alpha = score, 0, a_best
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.alpha = best_alpha
    return epoch + 1, best


@torch.no_grad()
def predict(model, inputs_fn, idx, batch):
    model.eval()
    outs = [model(*(t.to(DEVICE) for t in inputs_fn(idx[s:s + batch]))).cpu().numpy()
            for s in range(0, len(idx), batch)]
    return np.concatenate(outs) if outs else np.zeros((0,))


# --------------------------------------------------------------------------- main

def main(args: argparse.Namespace) -> None:
    started = time.time()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.manual_seed(args.seed)
    print(f"host {platform.node()} python {sys.version.split()[0]} torch {torch.__version__} numpy {np.__version__} "
          f"slice {args.slice} split {args.split} seed {args.seed} init-seed {args.init_seed} device {DEVICE} "
          f"arms {args.arms}", flush=True)
    if DEVICE.type == "cuda":
        print(f"cuda {torch.version.cuda} {torch.cuda.get_device_name(DEVICE)}", flush=True)
    d = args.derived
    sl = np.load(d / f"slice_{args.slice}.npz", allow_pickle=False)
    keys = sl["inchikey"].astype(str); line_rows = sl["line"].astype(str)
    compounds = np.array(sorted(set(keys))); cidx = {k: i for i, k in enumerate(compounds)}
    row_c = np.array([cidx[k] for k in keys]); n_c = len(compounds)
    lines = sorted(set(line_rows)); lidx = {L: i for i, L in enumerate(lines)}
    G = sl["Y"].shape[1]
    Yc = np.zeros((n_c, len(lines), G), np.float64); Mc = np.zeros((n_c, len(lines)), bool)
    Ac = np.zeros((n_c, len(lines)), bool)
    act_rows = sl["active"].astype(bool) if "active" in sl.files else sl["tas"] >= run.ACTIVE_TAS
    for r, (c, L) in enumerate(zip(row_c, line_rows)):
        Yc[c, lidx[L]] = sl["Y"][r]; Mc[c, lidx[L]] = True; Ac[c, lidx[L]] = act_rows[r]
    table = pd.read_parquet(d / "drug_table.parquet")
    table = table[table.source.eq(args.source) & table.inchikey.isin(compounds)]
    first = table.groupby("inchikey").first().reindex(compounds)
    ik14 = first.ik14.astype(str).to_numpy(); scaf = first.generic_scaffold.fillna("").astype(str).to_numpy()
    targets = {k: set() for k in compounds}
    for r in table.itertuples():
        if r.targets:
            targets[r.inchikey].update(r.targets.split(";"))
    test_cf, train_cf, inner_cf, group_c = run.make_folds(args.split, args.seed, ik14, scaf, row_c,
                                                          sl["plates"].astype(str), n_c)
    feats = np.load(d / "drug_features.npz", allow_pickle=False)
    fpos = pd.Series(np.arange(len(feats["inchikey"])), index=feats["inchikey"].astype(str))[compounds].to_numpy()
    X = feats["morgan2"][fpos].astype(np.float32)
    cp_path = d / "cell_painting_features.npz"
    has_morph = np.isin(compounds, np.load(cp_path, allow_pickle=False)["inchikey"].astype(str)) if cp_path.exists() \
        else np.zeros(n_c, bool)

    genes, A_ppi, n_edges = run.load_ppi(args.string_root, args.min_score, None)
    _, A_rew, _ = run.load_ppi(args.string_root, args.min_score, args.rewire_seed)
    gpos = {g: i for i, g in enumerate(genes)}
    mapped = [sorted(t for t in targets[k] if t in gpos) for k in compounds]
    known = np.array([len(t) > 0 for t in mapped])
    lm_symbols = sl["gene_symbol"].astype(str)
    extra = [g for g in lm_symbols if g not in gpos]
    n_nodes = len(genes) + len(extra)
    node_of = {**gpos, **{g: len(genes) + i for i, g in enumerate(extra)}}
    landmark_nodes = np.array([node_of[g] for g in lm_symbols])
    pad = lambda A: sp.block_diag([A, sp.csr_matrix((len(extra), len(extra)))]).tocsr()  # isolated landmark nodes
    adjacency = {"gnn_ppi": normalized_adjacency(pad(A_ppi), n_nodes),
                 "gnn_rewired": normalized_adjacency(pad(A_rew), n_nodes),
                 "gnn_nograph": normalized_adjacency(None, n_nodes)}
    Tw = np.zeros((n_c, n_nodes), np.float32)
    for c, t in enumerate(mapped):
        for g in t:
            Tw[c, gpos[g]] = 1.0 / len(t)
    t_genes = sorted({g for t in mapped for g in t}); tpos = {g: i for i, g in enumerate(t_genes)}
    Tm = np.zeros((n_c, len(t_genes)), np.float32)
    for c, t in enumerate(mapped):
        for g in t:
            Tm[c, tpos[g]] = 1.0
    # propagated target signal at landmark nodes, hops 0..N_HOPS, per graph (scipy, computed once)
    def landmark_propagation(A: sp.csr_matrix | None) -> np.ndarray:
        n = n_nodes
        M = sp.identity(n, format="csr") if A is None else (pad(A) + sp.identity(n, format="csr")).tocsr()
        deg = np.asarray(M.sum(1)).ravel(); dinv = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
        M = (dinv @ M @ dinv).tocsr()
        out = np.zeros((n_c, len(landmark_nodes), N_HOPS + 1), np.float32)
        for start in range(0, n_c, 1024):
            S = sp.csr_matrix(Tw[start:start + 1024].T)               # (n, b)
            for k in range(N_HOPS + 1):
                out[start:start + 1024, :, k] = S[landmark_nodes].toarray().T
                S = M @ S
        # log-scale: hop-k mass decays geometrically and is heavy-tailed at hubs
        return np.log1p(out * 1000.0)
    props = {"gnn_ppi": torch.tensor(landmark_propagation(A_ppi)), "gnn_rewired": torch.tensor(landmark_propagation(A_rew)),
             "gnn_nograph": torch.tensor(landmark_propagation(None))}
    xcell = Mc.sum(1) >= 2
    print(f"{n_c:,} compounds, {len(lines)} lines, KNOWN {known.sum():,}, nodes {n_nodes:,} "
          f"({len(extra)} isolated landmark nodes), edges {n_edges:,} ({time.time() - started:.0f}s)", flush=True)

    Xt, Twt, Tmt = torch.tensor(X), torch.tensor(Tw), torch.tensor(Tm)
    records, log, epochs_used = [], [], []
    for f in args.folds:
        tr_all = np.where(train_cf[:, f])[0]
        te_all = np.where(test_cf[:, f])[0]
        if len(te_all) == 0:
            continue
        # per-line training mean over all training compounds measured in that line (= runner's `mean`)
        m = np.stack([Yc[tr_all[Mc[tr_all, l]], l].mean(0) for l in range(len(lines))])  # float64, as the runner
        R = (Yc - m[None]) * Mc[..., None]
        inner_val, inner_train = inner_cf[f][0]
        preds = {}
        for arm in args.arms:
            pool = np.ones(n_c, bool) if arm == "mlp_chem_all" else known
            fit = np.where(inner_train & train_cf[:, f] & pool)[0]
            val = np.where(inner_val & train_cf[:, f] & pool)[0]
            te = te_all[pool[te_all]]
            torch.manual_seed((args.seed if args.init_seed is None else args.init_seed) * 1000 + f)
            if arm in GCN_ARMS:
                model = GCN(adjacency[GCN_ARMS[arm]].to(DEVICE), n_nodes, landmark_nodes, len(lines), X.shape[1])
                fn, batch = (lambda b: (Twt[b], Xt[b])), BATCH_GNN
            elif arm in DOT_ARMS:
                prop = props[DOT_ARMS[arm]]
                model = DotGNN(adjacency[DOT_ARMS[arm]].to(DEVICE), n_nodes, landmark_nodes, len(lines), X.shape[1], N_HOPS)
                fn, batch = (lambda b, prop=prop: (Twt[b], Xt[b], prop[b])), BATCH_GNN
            elif arm in GRAPH_ARMS:
                model = DecoupledGNN(adjacency[arm].to(DEVICE), n_nodes, landmark_nodes, len(lines), X.shape[1], N_HOPS)
                prop = props[arm]
                fn, batch = (lambda b, prop=prop: (Twt[b], Xt[b], prop[b])), BATCH_GNN
            elif arm in PROP_ARMS:
                # identical to mlp_target plus the PPI-propagated target signal at every landmark (hops 0..K)
                prop = props[PROP_ARMS[arm]]
                model = FlatMLP(Tm.shape[1] + prop.shape[1] * prop.shape[2], len(lines), G, chem_dim=X.shape[1])
                fn, batch = (lambda b, prop=prop: (torch.cat([Tmt[b], prop[b].reshape(len(b), -1)], 1), Xt[b])), BATCH_MLP
            elif arm == "mlp_target":
                model = FlatMLP(Tm.shape[1], len(lines), G, chem_dim=X.shape[1])
                fn, batch = (lambda b: (Tmt[b], Xt[b])), BATCH_MLP
            else:
                model = FlatMLP(X.shape[1], len(lines), G)
                fn, batch = (lambda b: (Xt[b],)), BATCH_MLP
            model.to(DEVICE)
            n_ep, best = train_model(model, fn, fit, val, R, Mc, Ac, m, batch, log, f"{arm}|fold{f}")
            preds[arm] = (te, model.alpha * predict(model, fn, te, batch))
            epochs_used.append({"arm": arm, "fold": f, "epochs": n_ep, "best_val_pearson": best, "alpha": model.alpha,
                                "n_fit": len(fit), "n_val": len(val), "n_test": len(te)})
            print(f"  fold {f} {arm}: fit {len(fit)} val {len(val)} test {len(te)} epochs {n_ep} "
                  f"val {best:.4f} ({time.time() - started:.0f}s)", flush=True)
        preds["mean"] = (te_all, np.zeros((len(te_all), len(lines), G), np.float32))
        for arm, (te, P) in preds.items():
            for j, c in enumerate(te):
                for l in np.where(Mc[c])[0]:
                    pred = (m[l] + P[j, l]).astype(np.float64)[None]; truth = Yc[c, l].astype(np.float64)[None]
                    full = run.rowwise_pearson(pred, truth)[0]
                    resid = run.rowwise_pearson(pred - m[l], truth - m[l])[0]
                    top = len(np.intersect1d(np.argsort(-np.abs(truth[0]))[:100], np.argsort(-np.abs(pred[0]))[:100])) / 100
                    records.append((arm, lines[l], compounds[c], ik14[c], group_c[c], f, bool(Ac[c, l]), bool(known[c]),
                                    bool(xcell[c]), bool(has_morph[c]), full, resid, top,
                                    float(np.linalg.norm(pred - m[l])), float(np.linalg.norm(truth - m[l]))))
    out = args.out / f"{args.slice}_{args.split}_seed{args.seed}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records, columns=["arm", "line", "inchikey", "ik14", "group", "fold", "active", "known",
                                           "xcell", "morph", "pearson_full", "pearson_resid", "top100",
                                           "norm_pred", "norm_true"])
    frame.to_parquet(out / "per_condition.parquet", index=False)
    pd.DataFrame(log).to_csv(out / "training_curves.csv", index=False)
    pd.DataFrame(epochs_used).to_csv(out / "epochs.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "slice": args.slice, "split": args.split, "seed": args.seed, "arms": list(args.arms), "folds": list(args.folds),
        "hidden": HIDDEN, "shrink": SHRINK, "lr_used": LR, "n_hops": N_HOPS, "architecture": "decoupled-propagation GNN (SGC/SIGN family); gcn_* = pre-registered per-drug 2-layer GCN",
        "init_seed": args.init_seed, "device": str(DEVICE), "lr": LR, "weight_decay": WD, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "min_score": args.min_score, "rewire_seed": args.rewire_seed, "n_nodes": n_nodes, "n_known": int(known.sum()),
        "runner_sha256": run.sha256(Path(__file__).resolve()), "slice_sha256": run.sha256(d / f"slice_{args.slice}.npz"),
        "features_sha256": run.sha256(d / "drug_features.npz"), "torch": torch.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "seconds": round(time.time() - started, 1)}, indent=2) + "\n")
    print(frame[frame.active].groupby(["arm"])[["pearson_full", "pearson_resid"]].mean().round(4).to_string(), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base = Path("/nfs-stor/yuqi.lei/Therapeutic")
    p.add_argument("--slice", default="P2-24", choices=["P1-24", "P1-6", "P2-24", "T-0.05", "T-0.5", "T-5"])
    p.add_argument("--source", default="L1000", choices=["L1000", "Tahoe"])
    p.add_argument("--split", default="ik14", choices=["ik14", "scaffold", "platemap"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=list(ALL_ARMS), choices=list(ALL_ARMS) + list(GCN_ARMS))
    p.add_argument("--init-seed", type=int, default=None, help="model-initialisation seed; folds keep --seed")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--folds", nargs="+", type=int, default=list(range(run.N_OUTER)))
    p.add_argument("--min-score", type=int, default=400)
    p.add_argument("--rewire-seed", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--tag", default="")
    p.add_argument("--shrink", action="store_true", help="select output shrinkage alpha on inner validation")
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--derived", type=Path, default=base / "derived/l1000_unseen_drug")
    p.add_argument("--string-root", type=Path, default=base / "raw/string_v12")
    p.add_argument("--out", type=Path, default=base / "results/l1000_gnn")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    MAX_EPOCHS = a.max_epochs
    SHRINK = a.shrink
    LR = a.lr
    DEVICE = torch.device(a.device)
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but torch.cuda.is_available() is false")
    main(a)
