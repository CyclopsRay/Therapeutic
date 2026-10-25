#!/usr/bin/env python3
"""Integrated end-to-end model, phase 1: unseen drug on L1000 (Jun Wen design; record 2026-10-08/09).

One network, arms differ only in the knowledge graph (KG) the drug and gene vectors pass through:
  skel   no KG: h_d = DrugEnc(Morgan2), gene vector = free landmark embedding
  kg_a   PrimeKG RGCN over PPI + drug-disease (indication, contraindication, off-label use) + disease-protein
  kg_b   PrimeKG RGCN over drug-disease + disease-protein (no PPI)
  kg_t   kg_a + drug-protein (targets); defined for later, not in the user's first round
Node inputs: compounds = DrugEnc(Morgan2) (inductive: every slice compound is its own node; a compound matched to a
PrimeKG drug inherits that drug's edges); other PrimeKG drugs and genes = free embeddings; diseases = linear map of
frozen SapBERT(name) + PubMedBERT(name. definition). RGCN (2 layers, mean aggregation, residual, LayerNorm) runs
over the whole graph at every step, end to end with the response loss, plus an optional DistMult link loss.
Decoder (factorized): pred[d, l, g] = <W_l h_d, e_g> / sqrt(D), with e_g = free landmark embedding + RGCN state of
the gene's node. Target: residual from the per-line training mean (the runner's `mean` arm, so mu[l, g] is built in);
masked MSE on Level-5 z (continuous output). Early stopping and output shrinkage on the first inner split, criterion
full Pearson on ACTIVE validation rows. Same slices, folds and row schema as run_l1000_unseen_drug.py.

Round 2 (record 2026-10-11): --anchor krr adds a per-line Tanimoto(Morgan2) KRR prediction as a fixed offset, so the
network fits only what KRR leaves (alpha = 0 recovers KRR exactly). Fit-compound anchors are cross-fitted over the
inner groups 1..N-1 (a compound never sees its own group), validation anchors come from KRR on the fit set, test
anchors from KRR on the whole outer training set; lambda is shared across lines and chosen on inner validation.
The anchor itself is written as arm `anchor`. --rewire SEED replaces every KG relation by a degree-preserving
double-edge swap (per relation; the reverse direction is regenerated from the swapped forward edges), so each
compound keeps its number of disease / target edges; such arms are labelled `<arm>_rw<SEED>`.
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
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_l1000_unseen_drug as run
from netnull import _swap, hgnc_map

KG_RELATIONS = {
    "kg_a": ("protein_protein", "indication", "contraindication", "off-label use", "disease_protein"),
    "kg_b": ("indication", "contraindication", "off-label use", "disease_protein"),
    "kg_t": ("protein_protein", "indication", "contraindication", "off-label use", "disease_protein", "drug_protein"),
}
ARMS = ("skel",) + tuple(KG_RELATIONS)
DRUG_DISEASE = ("indication", "contraindication", "off-label use")
GRID_ALPHA = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- graph

class KG:
    """PrimeKG subgraph for one arm, with every slice compound as its own drug node."""

    def __init__(self, root: Path, relations: tuple[str, ...], compounds: np.ndarray, lm_symbols: np.ndarray,
                 load_text: bool = True, rewire_seed: int | None = None):
        g = np.load(root / "graph.npz", allow_pickle=False)
        nodes = pd.read_parquet(root / "nodes.parquet")
        rel_names = g["rel_names"].astype(str)
        keep_rel = np.array([r.split(":")[0] in relations for r in rel_names])
        e = pd.DataFrame({"s": g["src"], "t": g["dst"], "r": g["rel"]})
        e = e[keep_rel[e.r.to_numpy()]]
        self.rel_names = rel_names[keep_rel]
        e["r"] = pd.Series(np.cumsum(keep_rel) - 1)[e.r.to_numpy()].to_numpy()
        types = nodes.type.to_numpy()
        n_kg = len(nodes)
        # compounds -> PrimeKG drug node (full InChIKey, else connectivity block)
        dm = pd.read_csv(root / "drug_map.tsv", sep="\t")
        by_full = dm.drop_duplicates("inchikey").set_index("inchikey").node
        by_blk = dm.drop_duplicates("ik14").set_index("ik14").node
        matched = np.array([by_full.get(k, by_blk.get(k[:14], -1)) for k in compounds])
        self.compound_node = n_kg + np.arange(len(compounds))
        # move edges of matched PrimeKG drugs onto their compound nodes (one copy per compound)
        cmap = pd.DataFrame({"old": matched[matched >= 0], "new": self.compound_node[matched >= 0]})
        moved = set(cmap.old)
        e_s = e[e.s.isin(moved)].merge(cmap, left_on="s", right_on="old").assign(s=lambda x: x.new)[["s", "t", "r"]]
        e_t = e[e.t.isin(moved)].merge(cmap, left_on="t", right_on="old").assign(t=lambda x: x.new)[["s", "t", "r"]]
        e_both = e_s[e_s.t.isin(moved)]  # drug-drug edges are not kept; guard anyway
        e = pd.concat([e[~e.s.isin(moved) & ~e.t.isin(moved)], e_s[~e_s.t.isin(moved)], e_t[~e_t.s.isin(moved)]])
        assert len(e_both) == 0
        if rewire_seed is not None:
            e = rewire_relations(e, self.rel_names, rewire_seed)
        self.n = n_kg + len(compounds)
        self.edge_index = torch.tensor(np.stack([e.s.to_numpy(), e.t.to_numpy()]), dtype=torch.long)
        self.edge_type = torch.tensor(e.r.to_numpy(), dtype=torch.long)
        self.n_rel = len(self.rel_names)
        # node groups
        self.gene_nodes = np.where(types == "gene/protein")[0]
        self.free_drug_nodes = np.setdiff1d(np.where(types == "drug")[0], list(moved))
        if load_text:
            dis = np.load(root / "disease_text_emb.npz", allow_pickle=False)
            assert np.array_equal(np.sort(dis["node"]), np.where(types == "disease")[0])
            self.disease_nodes = dis["node"]
            self.disease_x = torch.tensor(np.concatenate([dis["sapbert_name"], dis["pubmedbert_def"]], 1))
        hg = hgnc_map()
        gpos = pd.Series(self.gene_nodes, index=nodes.name_hgnc.to_numpy()[self.gene_nodes])
        gpos = gpos[~gpos.index.duplicated()]
        self.landmark_node = np.array([gpos.get(hg.get(s, s), -1) for s in lm_symbols])
        dd_rel = [i for i, r in enumerate(self.rel_names) if r.split(":")[0] in DRUG_DISEASE]
        s_np = self.edge_index[0].numpy()
        has_dd = np.zeros(self.n, bool); has_dd[s_np[np.isin(self.edge_type.numpy(), dd_rel)]] = True
        self.compound_in_kg = has_dd[self.compound_node]
        self.compound_matched = matched >= 0

    def summary(self) -> str:
        return (f"nodes {self.n:,}, edges {self.edge_index.shape[1]:,}, relations {self.n_rel}; compounds matched "
                f"{self.compound_matched.sum():,}, with drug-disease edges {self.compound_in_kg.sum():,}; "
                f"landmarks in KG {(self.landmark_node >= 0).sum()}/{len(self.landmark_node)}")


def rewire_relations(e: pd.DataFrame, rel_names: np.ndarray, seed: int) -> pd.DataFrame:
    """Degree-preserving null per relation. Relations are 'name:X->Y' with both directions listed; a homogeneous
    relation (X == Y) is swapped as undirected, a heterogeneous one as directed X->Y (bipartite, so it stays X->Y),
    and the Y->X relation is rebuilt as the exact reverse."""
    base = [r.split(":")[0] for r in rel_names]
    ends = [tuple(r.split(":", 1)[1].split("->")) for r in rel_names]
    out, done = [], set()
    for i in range(len(rel_names)):
        if i in done:
            continue
        E = e.loc[e.r == i, ["s", "t"]].to_numpy()
        if ends[i][0] == ends[i][1]:
            E = _swap(np.unique(np.sort(E, 1), axis=0), seed, directed=False)
            out.append(pd.DataFrame({"s": np.r_[E[:, 0], E[:, 1]], "t": np.r_[E[:, 1], E[:, 0]], "r": i}))
            done.add(i)
            continue
        j = next(k for k in range(len(rel_names)) if base[k] == base[i] and ends[k] == ends[i][::-1])
        n_rev = int((e.r == j).sum())
        E = _swap(np.unique(E, axis=0), seed, directed=True)
        assert len(E) == n_rev, (rel_names[i], len(E), n_rev)  # the two directions list the same edges
        out.append(pd.DataFrame({"s": E[:, 0], "t": E[:, 1], "r": i}))
        out.append(pd.DataFrame({"s": E[:, 1], "t": E[:, 0], "r": j}))
        done |= {i, j}
    return pd.concat(out).drop_duplicates()


# --------------------------------------------------------------------------- model

class E2E(nn.Module):
    def __init__(self, n_bits: int, n_lines: int, n_genes: int, dim: int, kg: KG | None, layers: int, dropout: float):
        super().__init__()
        self.drug = nn.Sequential(nn.Linear(n_bits, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, dim),
                                  nn.LayerNorm(dim))
        self.gene = nn.Embedding(n_genes, dim); nn.init.normal_(self.gene.weight, std=0.1)
        self.W = nn.Parameter(torch.eye(dim).repeat(n_lines, 1, 1) + 0.01 * torch.randn(n_lines, dim, dim))
        self.norm_g = nn.LayerNorm(dim); self.norm_h = nn.LayerNorm(dim)  # residual RGCN states grow; decode on unit scale
        self.dim, self.kg = dim, kg
        if kg is not None:
            from torch_geometric.nn import RGCNConv
            self.kg_gene = nn.Embedding(len(kg.gene_nodes), dim); nn.init.normal_(self.kg_gene.weight, std=0.1)
            self.kg_drug = nn.Embedding(max(1, len(kg.free_drug_nodes)), dim); nn.init.normal_(self.kg_drug.weight, std=0.1)
            self.dis = nn.Sequential(nn.Linear(kg.disease_x.shape[1], dim), nn.LayerNorm(dim))
            self.convs = nn.ModuleList([RGCNConv(dim, dim, kg.n_rel, aggr="mean") for _ in range(layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
            self.rel_diag = nn.Parameter(torch.ones(kg.n_rel, dim))  # DistMult link head
            self.drop = nn.Dropout(dropout)
            for name in ("edge_index", "edge_type", "disease_x"):
                self.register_buffer(name, getattr(kg, name).clone(), persistent=False)
            for name in ("gene_nodes", "free_drug_nodes", "disease_nodes", "compound_node"):
                self.register_buffer(name, torch.tensor(getattr(kg, name), dtype=torch.long), persistent=False)
            lm = torch.tensor(kg.landmark_node, dtype=torch.long)
            self.register_buffer("lm_node", lm.clamp(min=0), persistent=False)
            self.register_buffer("lm_mask", (lm >= 0).float().unsqueeze(1), persistent=False)

    def graph(self, X_all: torch.Tensor) -> torch.Tensor:
        H = torch.zeros(self.kg.n, self.dim, device=X_all.device)
        H = H.index_copy(0, self.gene_nodes, self.kg_gene.weight)
        if len(self.kg.free_drug_nodes):
            H = H.index_copy(0, self.free_drug_nodes, self.kg_drug.weight)
        H = H.index_copy(0, self.disease_nodes, self.dis(self.disease_x))
        H = H.index_copy(0, self.compound_node, self.drug(X_all))
        for conv, norm in zip(self.convs, self.norms):
            H = H + self.drop(F.relu(norm(conv(H, self.edge_index, self.edge_type))))
        return H

    def decode(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bd,lde,ge->blg", self.norm_h(h), self.W, self.norm_g(g)) / self.dim ** 0.5

    def forward(self, b: torch.Tensor, X_all: torch.Tensor, H: torch.Tensor | None = None) -> torch.Tensor:
        if self.kg is None:
            return self.decode(self.drug(X_all[b]), self.gene.weight)
        H = self.graph(X_all) if H is None else H
        g = self.gene.weight + self.lm_mask * H[self.lm_node]
        return self.decode(H[self.compound_node[b]], g)

    def link_loss(self, H: torch.Tensor, n_edges: int, gen: torch.Generator) -> torch.Tensor:
        E = self.edge_index.shape[1]
        pick = torch.randint(0, E, (n_edges,), device=H.device, generator=gen)
        s, t, r = self.edge_index[0, pick], self.edge_index[1, pick], self.edge_type[pick]
        neg = torch.randint(0, H.shape[0], (n_edges,), device=H.device, generator=gen)
        H = F.layer_norm(H, (self.dim,)); scale = self.dim ** 0.5  # scaled DistMult: logits O(1) at init
        pos_score = (H[s] * self.rel_diag[r] * H[t]).sum(1) / scale; neg_score = (H[s] * self.rel_diag[r] * H[neg]).sum(1) / scale
        return F.binary_cross_entropy_with_logits(torch.cat([pos_score, neg_score]),
                                                  torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]))


# --------------------------------------------------------------------------- training

def masked_pearson_full(pred_res, R, M, active, m):
    P = (pred_res + m[None]).reshape(-1, R.shape[-1]); Y = (R + m[None]).reshape(-1, R.shape[-1])
    keep = (M & active).reshape(-1)
    return float(run.rowwise_pearson(P[keep], Y[keep]).mean()) if keep.any() else np.nan


def krr_lines(K, R, M, train_idx, pred_idx, lambdas):
    """Per-line KRR on residuals: line l is fitted on the training compounds measured in l and predicts the
    prediction compounds measured in l. Returns {lambda: array (len(pred_idx), lines, genes)}."""
    out = {lam: np.zeros((len(pred_idx), R.shape[1], R.shape[2]), np.float32) for lam in lambdas}
    for l in range(R.shape[1]):
        tr = train_idx[M[train_idx, l]]; pr_pos = np.where(M[pred_idx, l])[0]
        if len(tr) < 5 or len(pr_pos) == 0:
            continue
        tr_t = torch.tensor(tr, device=DEVICE); pr_t = torch.tensor(pred_idx[pr_pos], device=DEVICE)
        vals, vecs = torch.linalg.eigh(K[tr_t][:, tr_t] + 1e-6 * torch.eye(len(tr), device=DEVICE, dtype=K.dtype))
        proj = vecs.T @ torch.tensor(R[tr, l], device=DEVICE, dtype=K.dtype)
        K_pr = K[pr_t][:, tr_t]
        for lam in lambdas:
            out[lam][pr_pos, l] = (K_pr @ (vecs @ (proj / (vals.clamp(min=0) + lam)[:, None]))).cpu().numpy()
    return out


def krr_anchor(X_all, R, M, active, m, fit, val, train, test, inner_val_masks):
    """Anchor A (compounds, lines, genes) for fit/val/test compounds and the chosen lambda (see module docstring)."""
    Xb = X_all.double(); inter = Xb @ Xb.T; cnt = Xb.sum(1)
    K = inter / (cnt[:, None] + cnt[None, :] - inter).clamp(min=1e-9)
    A = np.zeros(R.shape, np.float32)
    sel = krr_lines(K, R, M, fit, val, run.GRID_LAMBDA)
    scores = {lam: masked_pearson_full(p, R[val], M[val], active[val], m) for lam, p in sel.items()}
    lam = max(scores, key=scores.get)
    A[val] = sel[lam]
    fit_set = np.zeros(len(R), bool); fit_set[fit] = True
    for v in inner_val_masks:  # inner groups other than the validation group partition the fit set
        hold = np.where(fit_set & v)[0]
        if len(hold):
            A[hold] = krr_lines(K, R, M, np.where(fit_set & ~v)[0], hold, [lam])[lam]
    A[test] = krr_lines(K, R, M, train, test, [lam])[lam]
    del K, inter
    return A, lam, scores


@torch.no_grad()
def predict(model, X_all, idx, chunk=512):
    model.eval()
    H = model.graph(X_all) if model.kg is not None else None
    out = [model(torch.tensor(idx[s:s + chunk], device=DEVICE), X_all, H).cpu().numpy() for s in range(0, len(idx), chunk)]
    return np.concatenate(out) if out else np.zeros((0,))


def train(model, X_all, fit, val, R, M, active, m, a, log, label, A):
    """Fits R - A (A = anchor, zeros without one); validation prediction is A + alpha * network."""
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    Rt = torch.tensor(R - A, dtype=torch.float32, device=DEVICE); Mt = torch.tensor(M, dtype=torch.float32, device=DEVICE)
    rng = np.random.default_rng(0); gen = torch.Generator(device=DEVICE); gen.manual_seed(0)
    best, best_state, best_alpha, bad = -np.inf, None, 1.0, 0
    for epoch in range(a.max_epochs):
        model.train(); t_ep = time.time(); tot, tot_link, steps = 0.0, 0.0, 0
        order = rng.permutation(fit)
        for s in range(0, len(order), a.batch):
            b = torch.tensor(order[s:s + a.batch], device=DEVICE)
            H = model.graph(X_all) if model.kg is not None else None
            out = model(b, X_all, H)
            w = Mt[b].unsqueeze(-1)
            loss = (((out - Rt[b]) ** 2) * w).sum() / (w.sum() * R.shape[-1]).clamp(min=1)
            total = loss
            if model.kg is not None and a.link_weight > 0:
                ll = model.link_loss(H, a.link_edges, gen); total = loss + a.link_weight * ll; tot_link += float(ll)
            opt.zero_grad(); total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss); steps += 1
        pv = predict(model, X_all, val)
        alphas = GRID_ALPHA if a.shrink else (1.0,)
        scores = {al: masked_pearson_full(A[val] + al * pv, R[val], M[val], active[val], m) for al in alphas}
        al = max(scores, key=scores.get); score = scores[al]
        sec = time.time() - t_ep
        log.append({"arm": label, "epoch": epoch, "train_mse": tot / steps, "link_loss": tot_link / steps,
                    "val_pearson": score, "val_pearson_alpha1": scores.get(1.0, score), "alpha": al,
                    "seconds": sec, "sec_per_step": sec / steps})
        if epoch % 10 == 0:
            print(f"    {label} ep {epoch} mse {tot / steps:.4f} link {tot_link / steps:.3f} val {score:.4f} "
                  f"(a={al}) {sec:.1f}s", flush=True)
        if score > best + 1e-5:
            best, best_alpha, bad = score, al, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return epoch + 1, best, best_alpha


# --------------------------------------------------------------------------- main

def main(a: argparse.Namespace) -> None:
    global DEVICE
    DEVICE = torch.device(a.device)
    started = time.time()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    import torch_geometric
    print(f"host {platform.node()} python {sys.version.split()[0]} torch {torch.__version__} pyg "
          f"{torch_geometric.__version__} numpy {np.__version__} slice {a.slice} split {a.split} seed {a.seed} "
          f"device {DEVICE} arms {a.arms} args {vars(a)}", flush=True)
    if DEVICE.type == "cuda":
        print(f"cuda {torch.version.cuda} {torch.cuda.get_device_name(DEVICE)}", flush=True)
    d = a.derived
    sl = np.load(d / f"slice_{a.slice}.npz", allow_pickle=False)
    keys = sl["inchikey"].astype(str); line_rows = sl["line"].astype(str)
    compounds = np.array(sorted(set(keys))); cidx = {k: i for i, k in enumerate(compounds)}
    row_c = np.array([cidx[k] for k in keys]); n_c = len(compounds)
    lines = sorted(set(line_rows)); lidx = {L: i for i, L in enumerate(lines)}
    G = sl["Y"].shape[1]
    Yc = np.zeros((n_c, len(lines), G), np.float64); Mc = np.zeros((n_c, len(lines)), bool)
    Ac = np.zeros((n_c, len(lines)), bool)
    act_rows = sl["active"].astype(bool) if "active" in sl.files else sl["tas"] >= run.ACTIVE_TAS
    Y = sl["Y"]  # read once: NpzFile re-reads the member on every access
    for r, (c, L) in enumerate(zip(row_c, line_rows)):
        Yc[c, lidx[L]] = Y[r]; Mc[c, lidx[L]] = True; Ac[c, lidx[L]] = act_rows[r]
    table = pd.read_parquet(d / "drug_table.parquet")
    table = table[table.source.eq(a.source) & table.inchikey.isin(compounds)]
    first = table.groupby("inchikey").first().reindex(compounds)
    ik14 = first.ik14.astype(str).to_numpy(); scaf = first.generic_scaffold.fillna("").astype(str).to_numpy()
    known = np.zeros(n_c, bool)
    for r in table.itertuples():
        if r.targets:
            known[cidx[r.inchikey]] = True
    test_cf, train_cf, inner_cf, group_c = run.make_folds(a.split, a.seed, ik14, scaf, row_c, sl["plates"].astype(str), n_c)
    feats = np.load(d / "drug_features.npz", allow_pickle=False)
    fpos = pd.Series(np.arange(len(feats["inchikey"])), index=feats["inchikey"].astype(str))[compounds].to_numpy()
    X_all = torch.tensor(feats["morgan2"][fpos].astype(np.float32), device=DEVICE)
    cp_path = d / "cell_painting_features.npz"
    has_morph = np.isin(compounds, np.load(cp_path, allow_pickle=False)["inchikey"].astype(str)) if cp_path.exists() \
        else np.zeros(n_c, bool)
    xcell = Mc.sum(1) >= 2
    lm_symbols = sl["gene_symbol"].astype(str)
    kgs = {arm: KG(a.kg_root, KG_RELATIONS[arm], compounds, lm_symbols, rewire_seed=a.rewire)
           for arm in a.arms if arm != "skel"}
    label = {arm: arm if arm == "skel" or a.rewire is None else f"{arm}_rw{a.rewire}" for arm in a.arms}
    for arm, kg in kgs.items():
        print(f"{arm}: {kg.summary()}; relations {list(kg.rel_names)}", flush=True)
    ref = next(iter(kgs.values())) if kgs else KG(a.kg_root, KG_RELATIONS["kg_b"], compounds, lm_symbols, load_text=False)
    in_kg = ref.compound_in_kg
    print(f"{n_c:,} compounds, {len(lines)} lines, in_kg {in_kg.sum():,} ({time.time() - started:.0f}s)", flush=True)

    records, log, epochs_used = [], [], []
    for f in a.folds:
        tr_all = np.where(train_cf[:, f])[0]; te_all = np.where(test_cf[:, f])[0]
        if len(te_all) == 0:
            continue
        m = np.stack([Yc[tr_all[Mc[tr_all, l]], l].mean(0) for l in range(len(lines))])
        R = (Yc - m[None]) * Mc[..., None]
        inner_val, inner_train = inner_cf[f][0]
        fit = np.where(inner_train & train_cf[:, f])[0]; val = np.where(inner_val & train_cf[:, f])[0]
        preds = {}
        A = np.zeros(R.shape, np.float32)
        if a.anchor == "krr":
            A, lam, lam_scores = krr_anchor(X_all, R, Mc, Ac, m, fit, val, tr_all, te_all,
                                            [v for v, _ in inner_cf[f][1:]])
            preds["anchor"] = A[te_all]
            epochs_used.append({"arm": "anchor", "fold": f, "lambda": lam, "best_val_pearson": lam_scores[lam],
                                "n_fit": len(fit), "n_val": len(val), "n_test": len(te_all)})
            print(f"  fold {f} anchor: lambda {lam} val {lam_scores[lam]:.4f} ({time.time() - started:.0f}s)", flush=True)
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        for arm in a.arms:
            torch.manual_seed(a.seed * 1000 + f)
            model = E2E(X_all.shape[1], len(lines), G, a.dim, kgs.get(arm), a.layers, a.dropout).to(DEVICE)
            n_ep, best, alpha = train(model, X_all, fit, val, R, Mc, Ac, m, a, log, f"{label[arm]}|fold{f}", A)
            preds[label[arm]] = A[te_all] + alpha * predict(model, X_all, te_all)
            epochs_used.append({"arm": label[arm], "fold": f, "epochs": n_ep, "best_val_pearson": best, "alpha": alpha,
                                "n_fit": len(fit), "n_val": len(val), "n_test": len(te_all),
                                "params": sum(p.numel() for p in model.parameters())})
            print(f"  fold {f} {label[arm]}: fit {len(fit)} val {len(val)} test {len(te_all)} epochs {n_ep} val {best:.4f} "
                  f"alpha {alpha} ({time.time() - started:.0f}s)", flush=True)
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        preds["mean"] = np.zeros((len(te_all), len(lines), G), np.float32)
        for arm, P in preds.items():
            for j, c in enumerate(te_all):
                for l in np.where(Mc[c])[0]:
                    pred = (m[l] + P[j, l]).astype(np.float64)[None]; truth = Yc[c, l].astype(np.float64)[None]
                    full = run.rowwise_pearson(pred, truth)[0]
                    resid = run.rowwise_pearson(pred - m[l], truth - m[l])[0]
                    top = len(np.intersect1d(np.argsort(-np.abs(truth[0]))[:100], np.argsort(-np.abs(pred[0]))[:100])) / 100
                    records.append((arm, lines[l], compounds[c], ik14[c], group_c[c], f, bool(Ac[c, l]), bool(known[c]),
                                    bool(xcell[c]), bool(has_morph[c]), bool(in_kg[c]), full, resid, top,
                                    float(np.linalg.norm(pred - m[l])), float(np.linalg.norm(truth - m[l]))))
    out = a.out / f"{a.slice}_{a.split}_seed{a.seed}{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records, columns=["arm", "line", "inchikey", "ik14", "group", "fold", "active", "known", "xcell",
                                           "morph", "in_kg", "pearson_full", "pearson_resid", "top100", "norm_pred",
                                           "norm_true"])
    frame.to_parquet(out / "per_condition.parquet", index=False)
    pd.DataFrame(log).to_csv(out / "training_curves.csv", index=False)
    pd.DataFrame(epochs_used).to_csv(out / "epochs.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(a).items()}, "kg_relations": {k: KG_RELATIONS[k] for k in kgs},
        "kg_summary": {k: kg.summary() for k, kg in kgs.items()}, "n_in_kg": int(in_kg.sum()),
        "runner_sha256": run.sha256(Path(__file__).resolve()), "slice_sha256": run.sha256(d / f"slice_{a.slice}.npz"),
        "graph_sha256": run.sha256(a.kg_root / "graph.npz"), "torch": torch.__version__, "device": str(DEVICE),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "seconds": round(time.time() - started, 1)}, indent=2) + "\n")
    act = frame[frame.active]
    print(act.groupby("arm")[["pearson_full", "pearson_resid"]].mean().round(4).to_string(), flush=True)
    print("in_kg rows:\n" + act[act.in_kg].groupby("arm")[["pearson_full", "pearson_resid"]].mean().round(4).to_string(),
          flush=True)


def parse_args() -> argparse.Namespace:
    base = Path("/nfs-stor/yuqi.lei/Therapeutic")
    p = argparse.ArgumentParser()
    p.add_argument("--slice", default="P1-24", choices=["P1-24", "P1-6", "P2-24", "T-0.05", "T-0.5", "T-5"])
    p.add_argument("--source", default="L1000", choices=["L1000", "Tahoe"])
    p.add_argument("--split", default="ik14", choices=["ik14", "scaffold", "platemap"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=["skel", "kg_a", "kg_b"], choices=list(ARMS))
    p.add_argument("--folds", nargs="+", type=int, default=list(range(run.N_OUTER)))
    p.add_argument("--device", default="cpu")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=128, help="compounds per step")
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--no-shrink", dest="shrink", action="store_false")
    p.add_argument("--link-weight", type=float, default=0.1)
    p.add_argument("--link-edges", type=int, default=8192)
    p.add_argument("--anchor", default="none", choices=["none", "krr"], help="fixed KRR(Morgan2) offset, round 2")
    p.add_argument("--rewire", type=int, default=None, help="degree-preserving rewired KG with this seed (null)")
    p.add_argument("--tag", default="")
    p.add_argument("--derived", type=Path, default=base / "derived/l1000_unseen_drug")
    p.add_argument("--kg-root", type=Path, default=base / "derived/e2e_kg/v1")
    p.add_argument("--out", type=Path, default=base / "results/l1000_e2e")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
