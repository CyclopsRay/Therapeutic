#!/usr/bin/env python3
"""Network sources and null models for the control-ladder study (PLAN.md Phase 6, E0; RECOMB 2027).

A network is a `Network(name, genes, edges, directed)`: gene symbols, an (m, 2) int array of unique
edges without self-loops (u < v when undirected; source -> target when directed).

Sources (`load(name)`; raw files under /nfs-stor/yuqi.lei/Therapeutic/raw, see `slurm/fetch_networks.sbatch`):
  string_physical400   STRING v12.0 physical, combined score >= 400 (the runner's graph)
  string_functional700 STRING v12.0 functional, score >= 700
  biogrid              BioGRID human physical interactions (latest release in raw/networks/biogrid)
  huri                 HuRI = the Luck et al. 2020 (PMID 32296183) subset of BioGRID
  primekg              PrimeKG v2.1 protein_protein edges
  omnipath             OmniPath signaling, directed edges only (consensus direction)
  collectri            CollecTRI TF -> target, directed
  signaling_tf         omnipath + collectri, directed (signaling into transcription)

Null models (`null(net, kind, seed)`), a ladder that removes one property at a time:
  real     unchanged
  rewired  degree-preserving double-edge swaps (undirected: every degree kept; directed: every in-
           and out-degree kept); 5 swaps per edge, as run.rewire
  labels   node-label permutation: same graph, gene names shuffled (keeps topology and the degree
           distribution, breaks which gene is a hub)
  er       the same number of edges placed uniformly at random (keeps sparsity only)
  empty    no edges
Optional HGNC harmonisation maps previous/alias symbols to approved symbols when unambiguous.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

RAW = Path("/nfs-stor/yuqi.lei/Therapeutic/raw")
DERIVED = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/networks")
HURI_PMID = "PUBMED:32296183"
SOURCES = ("string_physical400", "string_functional700", "biogrid", "huri", "primekg", "omnipath", "collectri",
           "signaling_tf")
NULLS = ("real", "rewired", "labels", "er", "empty")


@dataclass
class Network:
    name: str
    genes: np.ndarray
    edges: np.ndarray
    directed: bool

    @property
    def n(self) -> int:
        return len(self.genes)

    def adjacency(self, symmetric: bool = True) -> sp.csr_matrix:
        u, v = self.edges[:, 0], self.edges[:, 1]
        if symmetric or not self.directed:
            u, v = np.r_[u, v], np.r_[v, u]
        A = sp.coo_matrix((np.ones(len(u)), (u, v)), shape=(self.n, self.n)).tocsr()
        A.data[:] = 1.0
        return A

    def degree(self) -> np.ndarray:
        """Undirected degree (in + out for directed networks)."""
        return np.bincount(self.edges.ravel(), minlength=self.n)


# --------------------------------------------------------------------------- construction

def from_pairs(name: str, a, b, directed: bool, harmonize: bool = False) -> Network:
    a = pd.Series(a, dtype=str).str.strip().to_numpy(); b = pd.Series(b, dtype=str).str.strip().to_numpy()
    if harmonize:
        m = hgnc_map(); a = np.array([m.get(x, x) for x in a]); b = np.array([m.get(x, x) for x in b])
    keep = (a != b) & (a != "") & (b != "") & (a != "nan") & (b != "nan")
    a, b = a[keep], b[keep]
    genes = np.array(sorted(set(a) | set(b)))
    idx = {g: i for i, g in enumerate(genes)}
    u = np.array([idx[x] for x in a]); v = np.array([idx[x] for x in b])
    if not directed:
        u, v = np.minimum(u, v), np.maximum(u, v)
    edges = np.unique(np.stack([u, v], 1), axis=0) if len(u) else np.zeros((0, 2), int)
    return Network(name, genes, edges, directed)


@lru_cache(maxsize=1)
def hgnc_map() -> dict:
    """Previous and alias symbols -> approved symbol, only where the mapping is unambiguous."""
    h = pd.read_csv(RAW / "networks/hgnc/hgnc_complete_set.txt", sep="\t", usecols=["symbol", "prev_symbol",
                    "alias_symbol"], dtype=str)
    approved = set(h.symbol)
    cand: dict[str, set] = {}
    for col in ("prev_symbol", "alias_symbol"):
        for sym, other in zip(h.symbol, h[col].fillna("")):
            for o in other.split("|"):
                if o and o not in approved:
                    cand.setdefault(o, set()).add(sym)
    return {k: next(iter(v)) for k, v in cand.items() if len(v) == 1}


def _string(kind: str, min_score: int, harmonize: bool) -> Network:
    root = RAW / "string_v12"
    info = pd.read_csv(root / "9606.protein.info.v12.0.txt.gz", sep="\t")
    name = dict(zip(info["#string_protein_id"], info["preferred_name"]))
    f = "9606.protein.physical.links.v12.0.txt.gz" if kind == "physical" else "9606.protein.links.v12.0.txt.gz"
    links = pd.read_csv(root / f, sep=" ")
    links = links[links.combined_score >= min_score]
    return from_pairs(f"string_{kind}{min_score}", links.protein1.map(name), links.protein2.map(name), False, harmonize)


def _biogrid_frame() -> pd.DataFrame:
    root = RAW / "networks/biogrid"
    files = sorted(root.glob("BIOGRID-ORGANISM-Homo_sapiens*.tab3.txt"))
    if not files:
        with zipfile.ZipFile(root / "BIOGRID-ORGANISM-LATEST.tab3.zip") as z:
            member = next(m for m in z.namelist() if m.startswith("BIOGRID-ORGANISM-Homo_sapiens"))
            z.extract(member, root); files = [root / member]
    cols = ["Official Symbol Interactor A", "Official Symbol Interactor B", "Experimental System Type",
            "Organism ID Interactor A", "Organism ID Interactor B", "Publication Source"]
    df = pd.read_csv(files[0], sep="\t", usecols=cols, dtype=str, low_memory=False)
    df = df[(df["Organism ID Interactor A"] == "9606") & (df["Organism ID Interactor B"] == "9606")]
    return df[df["Experimental System Type"] == "physical"]


def _primekg_pairs() -> pd.DataFrame:
    cache = DERIVED / "primekg_ppi.tsv"
    if not cache.exists():
        parts = []
        for chunk in pd.read_csv(RAW / "primekg_v2.1/kg.csv", usecols=["relation", "x_name", "y_name"], dtype=str,
                                 chunksize=1_000_000, low_memory=False):
            parts.append(chunk[chunk.relation.eq("protein_protein")][["x_name", "y_name"]])
        DERIVED.mkdir(parents=True, exist_ok=True)
        pd.concat(parts).to_csv(cache, sep="\t", index=False)
    return pd.read_csv(cache, sep="\t", dtype=str)


def _omnipath(dataset: str) -> pd.DataFrame:
    f = RAW / ("networks/omnipath/omnipath_interactions.tsv" if dataset == "omnipath"
               else "networks/collectri/collectri_interactions.tsv")
    df = pd.read_csv(f, sep="\t", dtype=str)
    if dataset == "omnipath":
        df = df[df.consensus_direction.astype(str).eq("True")]
    return df


def load(name: str, harmonize: bool = False) -> Network:
    if name == "string_physical400":
        return _string("physical", 400, harmonize)
    if name == "string_functional700":
        return _string("functional", 700, harmonize)
    if name in ("biogrid", "huri"):
        df = _biogrid_frame()
        if name == "huri":
            df = df[df["Publication Source"].eq(HURI_PMID)]
        return from_pairs(name, df["Official Symbol Interactor A"], df["Official Symbol Interactor B"], False, harmonize)
    if name == "primekg":
        df = _primekg_pairs()
        return from_pairs(name, df.x_name, df.y_name, False, harmonize)
    if name in ("omnipath", "collectri"):
        df = _omnipath(name)
        return from_pairs(name, df.source_genesymbol, df.target_genesymbol, True, harmonize)
    if name == "signaling_tf":
        a, b = _omnipath("omnipath"), _omnipath("collectri")
        return from_pairs(name, pd.concat([a.source_genesymbol, b.source_genesymbol]),
                          pd.concat([a.target_genesymbol, b.target_genesymbol]), True, harmonize)
    raise ValueError(name)


# --------------------------------------------------------------------------- null models

def _swap(edges: np.ndarray, seed: int, directed: bool, swaps_per_edge: int = 5) -> np.ndarray:
    """Double-edge swaps. Undirected (a-b, c-d) -> (a-d, c-b) or (a-c, b-d); directed (a->b, c->d) ->
    (a->d, c->b), which keeps every out- and in-degree. Self-loops and duplicates are rejected."""
    rng = np.random.default_rng(seed)
    E = edges.copy(); m = len(E)
    key = (lambda x, y: (x, y)) if directed else (lambda x, y: (min(x, y), max(x, y)))
    present = set(map(tuple, E.tolist()))
    done = tries = 0; target = swaps_per_edge * m
    while done < target and tries < 20 * target:
        i, j = rng.integers(0, m, 2); tries += 1
        if i == j:
            continue
        a, b = E[i]; c, d = E[j]
        if not directed and rng.random() < 0.5:
            c, d = d, c
        if a == d or c == b or (not directed and (a == c or b == d)):
            continue
        e1, e2 = key(a, d), key(c, b)
        if e1 in present or e2 in present:
            continue
        present.discard(key(a, b)); present.discard(key(c, d)); present.add(e1); present.add(e2)
        E[i] = e1; E[j] = e2; done += 1
    return E


def _swap2k(edges: np.ndarray, seed: int, directed: bool, swaps_per_edge: int = 5) -> np.ndarray:
    """Joint-degree-preserving double-edge swaps (2K null). (a-b, c-d) -> (a-d, c-b) only when b and d have the same
    degree (directed: same in-degree for target swaps, same out-degree for source swaps), so every degree and the
    number of edges between each pair of degree classes are kept. Endpoint slots of one class only ever exchange
    nodes of that class, so the per-class slot lists are fixed. Self-loops and duplicates are rejected."""
    rng = np.random.default_rng(seed)
    E = edges.copy(); m = len(E); n = int(E.max()) + 1
    key = (lambda x, y: (x, y)) if directed else (lambda x, y: (min(x, y), max(x, y)))
    present = set(key(x, y) for x, y in E.tolist())
    # slot s = 2 * edge + side; class of a slot = the degree that must match there
    if directed:
        k_out, k_in = np.bincount(E[:, 0], minlength=n), np.bincount(E[:, 1], minlength=n)
        cls = np.stack([k_out[E[:, 0]], k_in[E[:, 1]] + n + 1], 1).ravel()     # disjoint source / target classes
    else:
        k = np.bincount(E.ravel(), minlength=n)
        cls = k[E].ravel()
    order = np.argsort(cls, kind="stable"); sorted_cls = cls[order]
    start = np.searchsorted(sorted_cls, cls, "left"); size = np.searchsorted(sorted_cls, cls, "right") - start
    done = tries = 0; target = swaps_per_edge * m
    while done < target and tries < 20 * target:
        tries += 1
        s = int(rng.integers(0, 2 * m))
        if size[s] < 2:
            continue
        t = int(order[start[s] + rng.integers(0, size[s])])
        i, si = divmod(s, 2); j, sj = divmod(t, 2)
        if i == j or si != sj and directed:
            continue
        b, a = E[i, si], E[i, 1 - si]; d, c = E[j, sj], E[j, 1 - sj]
        if b == d or a == c or a == d or c == b:
            continue
        new_i = (a, d) if si == 1 else (d, a)          # slot si of edge i receives d
        new_j = (c, b) if sj == 1 else (b, c)
        e1, e2 = key(*new_i), key(*new_j)
        if e1 in present or e2 in present:
            continue
        present.discard(key(*E[i])); present.discard(key(*E[j])); present.add(e1); present.add(e2)
        E[i, si] = d; E[j, sj] = b; done += 1
    if not directed:
        E = np.sort(E, 1)
    return E


def null(net: Network, kind: str, seed: int = 0) -> Network:
    rng = np.random.default_rng(seed)
    if kind == "real":
        return net
    if kind == "empty":
        return Network(f"{net.name}|empty", net.genes, np.zeros((0, 2), int), net.directed)
    if kind == "rewired":
        return Network(f"{net.name}|rewired{seed}", net.genes, _swap(net.edges, seed, net.directed), net.directed)
    if kind == "rewired2k":
        return Network(f"{net.name}|rewired2k{seed}", net.genes, _swap2k(net.edges, seed, net.directed), net.directed)
    if kind == "labels":
        perm = rng.permutation(net.n)
        E = perm[net.edges]
        if not net.directed:
            E = np.sort(E, 1)
        return Network(f"{net.name}|labels{seed}", net.genes, E, net.directed)
    if kind == "er":
        m, n = len(net.edges), net.n
        chosen: set = set()
        while len(chosen) < m:
            u = rng.integers(0, n, 2 * (m - len(chosen))); v = rng.integers(0, n, len(u))
            for x, y in zip(u, v):
                if x == y:
                    continue
                e = (x, y) if net.directed else (min(x, y), max(x, y))
                chosen.add(e)
                if len(chosen) == m:
                    break
        return Network(f"{net.name}|er{seed}", net.genes, np.array(sorted(chosen)), net.directed)
    raise ValueError(kind)


def summary(net: Network) -> dict:
    deg = net.degree()
    return {"network": net.name, "genes": net.n, "edges": int(len(net.edges)), "directed": net.directed,
            "median_degree": float(np.median(deg)), "max_degree": int(deg.max()) if len(deg) else 0}


if __name__ == "__main__":
    import sys
    for s in (sys.argv[1:] or SOURCES):
        print(summary(load(s)), flush=True)
