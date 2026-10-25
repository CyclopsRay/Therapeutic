#!/usr/bin/env python3
"""Unseen-drug prediction on L1000: retrieval (approach 1) versus knowledge/PPI similarity (approach 2).

Pre-registered 2026-10-03 in report/10_09_record.md. Secondary task, distinct from the
primary unseen-cell-line benchmark: every line is seen, whole compounds are held out.

For one slice (P1-24, P1-6 or P2-24) and every line separately, compounds are split into
five folds by InChIKey connectivity block (ik14; stereoisomers stay together) or by
generic Bemis-Murcko scaffold (--split scaffold). Each arm predicts the 978-landmark
Level-5 signature of a held-out compound in that line:

  mean                training-compound mean in the line
  knn_<sim>           m + alpha * sum_topk softmax(s/tau) (y_j - m); (k, tau, alpha) by inner
                      4-fold group CV maximizing mean full Pearson on ACTIVE inner rows
  krr_<sim>           kernel ridge on y - m, lambda by the same inner CV
  oracle_knn          ceiling: similarity = true response correlation in the same line
  xcell_knn           probe: similarity = response correlation in the other lines of the slice
  xcell_copy          probe: m + alpha * mean over other lines of (y_d,l' - m_l')
  xcell1_knn/_copy    the same probes from ONE other line per compound (stable hash choice)
  knn_chem_select     the chemistry encoder with the best inner-CV score in each line-fold
  *_xplate            neighbours sharing an L1000 plate with the query are excluded (batch confound)

KNOWN-only arms use only KNOWN training compounds as neighbors (targets that map into the
STRING physical graph). Every kNN arm shares one implementation, so arms differ only in
the similarity matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp

# Grids widened 2026-10-03 after the smoke run hit their edges (record: post-smoke amendment).
GRID_K = (1, 3, 10, 30, 100, 300, 1000)
GRID_TAU = (0.01, 0.03, 0.1, 0.3, 1.0, np.inf)
GRID_ALPHA = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
GRID_LAMBDA = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
GRID_COPY_ALPHA = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
ACTIVE_TAS = 0.2
N_OUTER, N_INNER = 5, 4


# --------------------------------------------------------------------------- utilities

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fold(keys, n: int, salt: str) -> np.ndarray:
    return np.array([int(hashlib.md5(f"{salt}|{k}".encode()).hexdigest(), 16) % n for k in keys])


def rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    out = np.zeros(len(a))
    ok = den > 1e-12
    out[ok] = (a[ok] * b[ok]).sum(-1) / den[ok]
    return out  # zero where either side is constant (e.g. a residual prediction of exactly 0)


def tanimoto(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32); B = B.astype(np.float32)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return (A @ B.T).astype(np.float32)


def corr_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    return cosine(A, B)


# --------------------------------------------------------------------------- kNN / KRR

def topk_order(S: np.ndarray, kmax: int) -> np.ndarray:
    kmax = min(kmax, S.shape[1])
    part = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]
    sub = np.take_along_axis(S, part, 1)
    return np.take_along_axis(part, np.argsort(-sub, axis=1, kind="stable"), 1)


def knn_residual(S: np.ndarray, order: np.ndarray, R_tr: np.ndarray, k: int, tau: float) -> np.ndarray:
    k = min(k, order.shape[1])
    idx = order[:, :k]
    s = np.take_along_axis(S, idx, 1).astype(np.float64)
    valid = np.isfinite(s)  # -inf marks an excluded neighbour (same-plate arms)
    if np.isinf(tau):
        w = valid.astype(np.float64)
    else:
        smax = np.where(valid, s, -np.inf).max(1, keepdims=True)
        w = np.where(valid, np.exp((np.where(valid, s, 0.0) - np.where(np.isfinite(smax), smax, 0.0)) / tau), 0.0)
    total = w.sum(1, keepdims=True)
    w = np.where(total > 0, w / np.maximum(total, 1e-300), 0.0)
    rows = np.repeat(np.arange(len(S)), k)
    W = sp.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(len(S), len(R_tr)))
    return np.asarray(W @ R_tr)


def pearson_over_alpha(m: np.ndarray, agg: np.ndarray, Y: np.ndarray, alphas) -> np.ndarray:
    """Row-wise Pearson of (m + alpha * agg) with Y for every alpha, from inner products.
    Returns (len(alphas), n_rows); identical to rowwise_pearson up to float error."""
    u = np.broadcast_to(m - m.mean(), agg.shape)
    v = agg - agg.mean(1, keepdims=True)
    w = Y - Y.mean(1, keepdims=True)
    uw, vw = (u * w).sum(1), (v * w).sum(1)
    uu, uv, vv, ww = (u * u).sum(1), (u * v).sum(1), (v * v).sum(1), (w * w).sum(1)
    out = []
    for a in alphas:
        den = np.sqrt(np.maximum(uu + 2 * a * uv + a * a * vv, 0) * ww)
        out.append(np.where(den > 1e-12, (uw + a * vw) / np.maximum(den, 1e-12), 0.0))
    return np.array(out)


def select_knn(S_full: np.ndarray, tr: np.ndarray, Y: np.ndarray, active: np.ndarray,
               splits: list, pool: np.ndarray) -> tuple:
    """Inner group CV over (k, tau, alpha). tr: training row indices; pool: bool over tr rows
    usable as neighbors; scoring rows: inner-validation rows that are in pool and ACTIVE.
    The inner mean always uses every inner-training row, as the outer prediction does.
    Returns ((k, tau, alpha), inner score)."""
    sums, counts = {}, {}
    for inner_val, inner_train in splits:
        va_mask = inner_val & pool & active[tr]
        nb_mask = inner_train & pool
        if va_mask.sum() < 5 or nb_mask.sum() < 5:
            continue
        va, nb = tr[va_mask], tr[nb_mask]
        m = Y[tr[inner_train]].mean(0)
        R_nb = Y[nb] - m
        S = S_full[np.ix_(va, nb)]
        order = topk_order(S, max(GRID_K))
        for k in GRID_K:
            for tau in GRID_TAU:
                agg = knn_residual(S, order, R_nb, k, tau)
                r = pearson_over_alpha(m, agg, Y[va], GRID_ALPHA)
                for alpha, row in zip(GRID_ALPHA, r):
                    key = (k, tau, alpha)
                    sums[key] = sums.get(key, 0.0) + row.sum(); counts[key] = counts.get(key, 0) + len(row)
    if not sums:
        return (10, np.inf, 0.5), np.nan
    best = max(sums, key=lambda key: sums[key] / counts[key])
    return best, sums[best] / counts[best]


def krr_fit_predict(K_tr: np.ndarray, K_te: np.ndarray, R_tr: np.ndarray, lambdas) -> dict:
    vals, vecs = np.linalg.eigh(K_tr.astype(np.float64))
    vals = np.clip(vals, 0, None)
    proj = vecs.T @ R_tr
    out = {}
    for lam in lambdas:
        alpha = vecs @ (proj / (vals + lam)[:, None])
        out[lam] = K_te @ alpha
    return out


def select_krr(K_full, tr, Y, active, splits, pool=None, K_cross=None) -> float:
    """Inner group CV for lambda. Kernel rows: pool (default all of tr); the inner mean uses
    every inner-training row, matching the outer prediction m + K (K + lam)^-1 (y - m).
    K_cross (default K_full) supplies the validation-to-training block, so a masked kernel
    (--exclude-same-plate) can be used for prediction while the training block stays PSD."""
    pool = np.ones(len(tr), bool) if pool is None else pool
    K_cross = K_full if K_cross is None else K_cross
    scores = {lam: [0.0, 0] for lam in GRID_LAMBDA}
    for inner_val, inner_train in splits:
        va = tr[inner_val & pool & active[tr]]
        nb = tr[inner_train & pool]
        if len(va) < 5 or len(nb) < 5:
            continue
        m = Y[tr[inner_train]].mean(0)
        preds = krr_fit_predict(K_full[np.ix_(nb, nb)] + 1e-6 * np.eye(len(nb)), K_cross[np.ix_(va, nb)],
                                Y[nb] - m, GRID_LAMBDA)
        for lam, p in preds.items():
            r = rowwise_pearson(m + p, Y[va]); scores[lam][0] += r.sum(); scores[lam][1] += len(r)
    return max(GRID_LAMBDA, key=lambda lam: scores[lam][0] / max(1, scores[lam][1]))


# --------------------------------------------------------------------------- PPI propagation

def load_ppi(string_root: Path, min_score: int, rewire_seed: int | None):
    info = pd.read_csv(string_root / "9606.protein.info.v12.0.txt.gz", sep="\t")
    name = dict(zip(info["#string_protein_id"], info["preferred_name"]))
    links = pd.read_csv(string_root / "9606.protein.physical.links.v12.0.txt.gz", sep=" ")
    links = links[links.combined_score >= min_score]
    a = links.protein1.map(name).to_numpy(); b = links.protein2.map(name).to_numpy()
    genes = np.array(sorted(set(a) | set(b)))
    index = {g: i for i, g in enumerate(genes)}
    u = np.array([index[x] for x in a]); v = np.array([index[x] for x in b])
    keep = u < v
    edges = np.unique(np.stack([u[keep], v[keep]], 1), axis=0)
    if rewire_seed is not None:
        edges = rewire(edges, len(genes), rewire_seed)
    n = len(genes)
    A = sp.coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(n, n)).tocsr()
    return genes, A, len(edges)


def rewire(edges: np.ndarray, n: int, seed: int, swaps_per_edge: int = 5) -> np.ndarray:
    """Degree-preserving double-edge swaps (rejecting self-loops and multi-edges)."""
    rng = np.random.default_rng(seed)
    E = edges.copy()
    present = set(map(tuple, E.tolist()))
    m = len(E); done = 0; tries = 0
    target = swaps_per_edge * m
    while done < target and tries < 20 * target:
        i, j = rng.integers(0, m, 2); tries += 1
        if i == j:
            continue
        a, b = E[i]; c, d = E[j]
        if rng.random() < 0.5:
            c, d = d, c
        if a == d or c == b or a == c or b == d:
            continue
        e1 = (min(a, d), max(a, d)); e2 = (min(c, b), max(c, b))
        if e1 in present or e2 in present:
            continue
        present.discard((min(a, b), max(a, b))); present.discard((min(c, d), max(c, d)))
        present.add(e1); present.add(e2)
        E[i] = e1; E[j] = e2; done += 1
    return E


def propagate(A: sp.csr_matrix, seeds: np.ndarray, restart: float, tol: float = 1e-8) -> np.ndarray:
    """Random walk with restart, column-normalized; seeds (n_genes, n_drugs) columns sum to 1."""
    deg = np.asarray(A.sum(0)).ravel()
    W = A @ sp.diags(1.0 / np.maximum(deg, 1))
    P = seeds.copy()
    for _ in range(200):
        nxt = restart * seeds + (1 - restart) * (W @ P)
        if np.abs(nxt - P).max() < tol:
            return nxt
        P = nxt
    return P


def plate_map_folds(maps_c: list, ik14: np.ndarray, eligible: np.ndarray, n: int, salt: str):
    """Plate-map grouped folds. A compound is held out in fold f only if every plate map it was
    measured on hashes to f; it trains in fold f only if none of its maps hashes to f and no
    member of its ik14 family is held out in f. Returns (test, train) bool arrays (n_c, n),
    both restricted to `eligible`."""
    h = [{int(hashlib.md5(f"{salt}|{m}".encode()).hexdigest(), 16) % n for m in maps} for maps in maps_c]
    test = np.array([[eligible[c] and len(h[c]) == 1 and f in h[c] for f in range(n)] for c in range(len(maps_c))])
    fam_test = pd.DataFrame(test).groupby(ik14).transform("any").to_numpy()
    train = np.array([[eligible[c] and f not in h[c] for f in range(n)] for c in range(len(maps_c))]) & ~fam_test
    return test, train


def make_folds(split: str, seed: int, ik14: np.ndarray, scaf: np.ndarray, row_c: np.ndarray,
               plates: np.ndarray, n_c: int):
    """Folds as compound-level masks: test_cf[c, f] / train_cf[c, f], and per outer fold a list of
    inner (validation, training) masks. ik14 / scaffold reproduce plain hashed group folds;
    platemap guarantees that no training compound shares an L1000 plate map with a test compound.
    Returns (test_cf, train_cf, inner_cf, group_c); group_c is the bootstrap unit."""
    if split in ("ik14", "scaffold"):
        if split == "ik14":
            fold_c = stable_fold(ik14, N_OUTER, f"outer{seed}")
            group_c = ik14
        else:
            group_c = np.where(scaf != "", scaf, ik14)
            fold_c = stable_fold(group_c, N_OUTER, f"scaffold{seed}")
        inner_c = stable_fold(group_c, N_INNER, f"inner{seed}")
        test_cf = np.stack([fold_c == f for f in range(N_OUTER)], 1)
        train_cf = ~test_cf
        inner_cf = [[(inner_c == g, inner_c != g) for g in range(N_INNER)] for f in range(N_OUTER)]
        return test_cf, train_cf, inner_cf, group_c
    maps_c = [set() for _ in range(n_c)]
    for c, row_plates in zip(row_c, plates):
        maps_c[c].update(p.split("_")[0] for p in row_plates.split("|"))
    test_cf, train_cf = plate_map_folds(maps_c, ik14, np.ones(n_c, bool), N_OUTER, f"plate{seed}")
    inner_cf = []
    for f in range(N_OUTER):
        v, t = plate_map_folds(maps_c, ik14, train_cf[:, f], N_INNER, f"plateinner{seed}")
        inner_cf.append([(v[:, g], t[:, g]) for g in range(N_INNER)])
    group_c = np.array([sorted(m)[0] for m in maps_c])
    print(f"platemap split: {test_cf.any(1).sum():,} of {n_c:,} compounds test-eligible; "
          f"training share per fold {train_cf.mean(0).round(3).tolist()}", flush=True)
    return test_cf, train_cf, inner_cf, group_c


# --------------------------------------------------------------------------- main

def main(args: argparse.Namespace) -> None:
    started = time.time()
    print(f"host {platform.node()} python {sys.version.split()[0]} numpy {np.__version__} "
          f"scipy {scipy.__version__} pandas {pd.__version__} split {args.split} seed {args.seed}", flush=True)
    d = args.derived
    sl = np.load(d / f"slice_{args.slice}.npz", allow_pickle=False)
    Y_all = sl["Y"].astype(np.float64); keys_all = sl["inchikey"].astype(str); line_all = sl["line"].astype(str)
    tas_all = sl["tas"]
    # Tahoe slices carry an explicit pre-declared ACTIVE flag (no TAS); L1000 uses TAS >= 0.2
    active_all = sl["active"].astype(bool) if "active" in sl.files else tas_all >= ACTIVE_TAS
    compounds = np.array(sorted(set(keys_all)))
    cidx = {k: i for i, k in enumerate(compounds)}
    row_c = np.array([cidx[k] for k in keys_all])
    n_c = len(compounds)
    print(f"slice {args.slice}: {len(Y_all):,} rows, {n_c:,} compounds", flush=True)

    # annotation per compound (L1000 source rows only)
    table = pd.read_parquet(d / "drug_table.parquet")
    table = table[table.source.eq(args.source) & table.inchikey.isin(compounds)]
    first = table.groupby("inchikey").first()
    ik14 = first.reindex(compounds)["ik14"].astype(str).to_numpy()
    scaf = first.reindex(compounds)["generic_scaffold"].fillna("").astype(str).to_numpy()
    targets = {k: set() for k in compounds}; moas = {k: set() for k in compounds}
    for r in table.itertuples():
        if r.targets:
            targets[r.inchikey].update(r.targets.split(";"))
        if r.moa:
            moas[r.inchikey].update(r.moa.split(";"))
    if args.targets == "union":
        # exploratory: add PrimeKG/DrugBank targets (by full InChIKey, else connectivity block)
        pk = pd.read_csv(d / "primekg_targets.tsv", sep="\t")
        by_key = dict(zip(pk.inchikey, pk.targets)); by_14 = pk.groupby("ik14").targets.agg(";".join).to_dict()
        added = 0
        for k, b14 in zip(compounds, ik14):
            extra = by_key.get(k, by_14.get(b14, ""))
            if extra:
                before = len(targets[k]); targets[k].update(t for t in extra.split(";") if t); added += len(targets[k]) > before
        print(f"--targets union: PrimeKG added targets to {added:,} compounds", flush=True)

    test_cf, train_cf, inner_cf, group_c = make_folds(args.split, args.seed, ik14, scaf, row_c, sl["plates"].astype(str), n_c)

    # chemistry similarity matrices over the slice's compounds
    feats = np.load(d / "drug_features.npz", allow_pickle=False)
    fpos = pd.Series(np.arange(len(feats["inchikey"])), index=feats["inchikey"].astype(str))[compounds].to_numpy()
    sims = {}
    sims["morgan2"] = tanimoto(feats["morgan2"][fpos], feats["morgan2"][fpos])
    sims["maccs"] = tanimoto(feats["maccs"][fpos], feats["maccs"][fpos])
    X = feats["rdkit2d"][fpos].astype(np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    X = np.nan_to_num((X - np.nanmean(X, 0)) / np.maximum(np.nanstd(X, 0), 1e-9), nan=0.0).clip(-10, 10)
    sims["rdkit2d"] = cosine(X, X)
    for lm in ("chemberta", "molformer"):
        path = d / f"smiles_lm_{lm}.npz"
        if path.exists():
            e = np.load(path, allow_pickle=False)
            pos = pd.Series(np.arange(len(e["inchikey"])), index=e["inchikey"].astype(str))[compounds].to_numpy()
            E = e["embedding"][pos].astype(np.float64)
            sims[lm] = cosine(E - E.mean(0), E - E.mean(0))
        elif args.allow_missing_lm:
            print(f"WARNING: {path} missing, arm knn_{lm} skipped", flush=True)
        else:
            raise FileNotFoundError(f"{path} missing; pass --allow-missing-lm only for smoke tests")
    print(f"chemistry similarities built ({time.time() - started:.0f}s)", flush=True)

    # knowledge: targets in the PPI graph
    genes, A, n_edges = load_ppi(args.string_root, args.min_score, None)
    gindex = {g: i for i, g in enumerate(genes)}
    mapped = [sorted(t for t in targets[k] if t in gindex) for k in compounds]
    known = np.array([len(t) > 0 for t in mapped])
    kn = np.where(known)[0]
    print(f"PPI: {len(genes):,} genes, {n_edges:,} edges (score >= {args.min_score}); KNOWN compounds "
          f"{known.sum():,} of {n_c:,} (any target annotated: {sum(bool(t) for t in targets.values()):,})", flush=True)
    T_genes = sorted({g for t in mapped for g in t})
    tpos = {g: i for i, g in enumerate(T_genes)}
    T = np.zeros((n_c, len(T_genes)), dtype=np.uint8)
    for i, t in enumerate(mapped):
        for g in t:
            T[i, tpos[g]] = 1
    rng = np.random.default_rng(args.seed)
    T_perm = T.copy(); T_perm[kn] = T[rng.permutation(kn)]
    sims["target"] = tanimoto(T, T)
    sims["target_perm"] = tanimoto(T_perm, T_perm)
    sims["chem_target"] = 0.5 * (sims["morgan2"] + sims["target"])
    moa_lab = sorted({m for k in compounds for m in moas[k]})
    mpos = {m: i for i, m in enumerate(moa_lab)}
    M = np.zeros((n_c, max(1, len(moa_lab))), dtype=np.uint8)
    for i, k in enumerate(compounds):
        for m in moas[k]:
            M[i, mpos[m]] = 1
    sims["moa"] = (tanimoto(M, M) > 0).astype(np.float32)  # shared-MOA indicator
    for label, seed in (("ppi", None), ("ppi_rewired", args.seed)):
        g2, A2, _ = (genes, A, n_edges) if seed is None else load_ppi(args.string_root, args.min_score, seed)
        gi2 = {g: i for i, g in enumerate(g2)}
        seeds = np.zeros((len(g2), len(kn)))
        for j, c in enumerate(kn):
            for g in mapped[c]:
                seeds[gi2[g], j] = 1.0 / len(mapped[c])
        P = propagate(A2, seeds, args.restart).T  # (n_known, n_genes)
        P = P - P.mean(0)  # remove the hub profile shared by every compound
        S = np.zeros((n_c, n_c), dtype=np.float32)
        S[np.ix_(kn, kn)] = cosine(P, P)
        sims[label] = S
    print(f"knowledge similarities built ({time.time() - started:.0f}s)", flush=True)

    # morphology (LINCS Cell Painting, A549 48 h): a measured phenotype of the compound itself
    has_morph = np.zeros(n_c, bool)
    morph_path = d / "cell_painting_features.npz"
    if morph_path.exists():
        cp = np.load(morph_path, allow_pickle=False)
        cpos_m = pd.Series(np.arange(len(cp["inchikey"])), index=cp["inchikey"].astype(str))
        hit = np.array([k in cpos_m.index for k in compounds])
        has_morph = hit
        mi = np.where(hit)[0]
        for label, key in (("morph", "morph10"), ("morph_doses", "morph_doses")):
            F = cp[key][cpos_m[compounds[mi]].to_numpy()].astype(np.float64)
            F = (F - F.mean(0)) / np.maximum(F.std(0), 1e-9)
            S = np.zeros((n_c, n_c), dtype=np.float32)
            S[np.ix_(mi, mi)] = cosine(F, F)
            sims[label] = S
        sims["chem_morph"] = 0.5 * (sims["morgan2"] + sims["morph"])
        print(f"morphology: {has_morph.sum():,} of {n_c:,} compounds have a 10 uM Cell Painting profile", flush=True)
    elif not args.allow_missing_morph:
        raise FileNotFoundError(f"{morph_path} missing; pass --allow-missing-morph to skip morphology arms")
    morph_arms = ["morgan2", "morph", "morph_doses", "chem_morph"] if has_morph.any() else []

    # TxGNN drug-node embeddings (exploratory, record 2026-10-07 13:40): PrimeKG-covered compounds only,
    # matched by full InChIKey, else connectivity block; control = Morgan2 on the same pool, null =
    # embeddings permuted among covered compounds
    has_tx = np.zeros(n_c, bool)
    if args.txgnn is not None:
        tx = np.load(args.txgnn, allow_pickle=False)
        tik = tx["inchikey"].astype(str); keep = tik != ""
        Etx = tx["embedding"][keep].astype(np.float64); tik = tik[keep]
        by_key = {k: i for i, k in enumerate(tik)}
        by_14 = {}
        for i, k in enumerate(tik):
            by_14.setdefault(k[:14], i)
        tpos = np.array([by_key.get(k, by_14.get(b14, -1)) for k, b14 in zip(compounds, ik14)])
        has_tx = tpos >= 0
        ti = np.where(has_tx)[0]
        F = Etx[tpos[ti]]; F = F - F.mean(0)
        Fp = F[np.random.default_rng(args.seed + 7).permutation(len(F))]
        for label, M_ in (("txgnn", F), ("txgnn_perm", Fp)):
            S = np.zeros((n_c, n_c), dtype=np.float32)
            S[np.ix_(ti, ti)] = cosine(M_, M_)
            sims[label] = S
        sims["chem_tx"] = 0.5 * (sims["morgan2"] + sims["txgnn"])
        print(f"TxGNN: {has_tx.sum():,} of {n_c:,} compounds covered "
              f"({int(sum(k in by_key for k in compounds)):,} by full InChIKey)", flush=True)
    tx_arms = ["morgan2", "txgnn", "txgnn_perm", "chem_tx"] if has_tx.sum() >= 20 else []

    chem_arms = [a for a in ("morgan2", "maccs", "rdkit2d", "chemberta", "molformer") if a in sims]
    known_arms = ["morgan2", "target", "target_perm", "chem_target", "moa", "ppi", "ppi_rewired"]

    # cross-line response similarity for the xcell probe
    lines = sorted(set(line_all))
    resp_rows = {L: np.where(line_all == L)[0] for L in lines}
    measured = np.zeros((n_c, len(lines)), bool)
    for j, L in enumerate(lines):
        measured[row_c[resp_rows[L]], j] = True
    per_line_corr = {}
    for L in lines:
        rows = resp_rows[L]
        C = np.full((n_c, n_c), np.nan, dtype=np.float32)
        C[np.ix_(row_c[rows], row_c[rows])] = corr_matrix(Y_all[rows], Y_all[rows])
        per_line_corr[L] = C
    print(f"per-line response correlations built ({time.time() - started:.0f}s)", flush=True)

    records = []
    selections = []
    plates_all = sl["plates"].astype(str) if "plates" in sl.files else None
    if plates_all is None:
        raise ValueError("slice has no 'plates' field; rebuild it with build_l1000_unseen_drug.py")
    for L in lines:
        rows = resp_rows[L]
        cpos = row_c[rows]
        Y = Y_all[rows]
        active = active_all[rows]
        test_rf = test_cf[cpos]; train_rf = train_cf[cpos]
        others = [x for x in lines if x != L]
        with np.errstate(invalid="ignore"):
            stack = np.stack([per_line_corr[x][np.ix_(cpos, cpos)] for x in others])
            xsim = np.nanmean(stack, 0)
        has_other = measured[cpos][:, [lines.index(x) for x in others]].any(1)
        local = {name: sims[name][np.ix_(cpos, cpos)]
                 for name in set(chem_arms) | set(known_arms) | set(morph_arms) | set(tx_arms)}
        xsim = np.nan_to_num(xsim, nan=-1.0).astype(np.float32)
        osim = per_line_corr[L][np.ix_(cpos, cpos)]
        # same-plate confound: pairs of rows sharing any L1000 plate are excluded (-inf) in *_xplate arms
        plate_lists = [p.split("|") for p in plates_all[rows]]
        plate_ids = {p: i for i, p in enumerate(sorted({p for ps in plate_lists for p in ps}))}
        B = sp.csr_matrix((np.ones(sum(len(ps) for ps in plate_lists)),
                           (np.repeat(np.arange(len(rows)), [len(ps) for ps in plate_lists]),
                            [plate_ids[p] for ps in plate_lists for p in ps])), shape=(len(rows), len(plate_ids)))
        same_plate = np.asarray((B @ B.T).todense()) > 0
        local["morgan2_xplate"] = np.where(same_plate, -np.inf, local["morgan2"]).astype(np.float32)
        local["oracle_xplate"] = np.where(same_plate, -np.inf, osim).astype(np.float32)
        # single-anchor probe: each compound is "measured" in ONE other line, chosen by a stable hash
        anchor = np.full(len(cpos), -1)
        for r, c in enumerate(cpos):
            options = [lines.index(x) for x in others if measured[c, lines.index(x)]]
            if options:
                anchor[r] = options[int(hashlib.md5(f"anchor{args.seed}|{compounds[c]}|{L}".encode()).hexdigest(), 16) % len(options)]
        xsim1 = np.full((len(cpos), len(cpos)), -1.0, dtype=np.float32)
        for j in set(anchor[anchor >= 0].tolist()):
            rr = np.where(anchor == j)[0]
            xsim1[rr] = np.nan_to_num(per_line_corr[lines[j]][np.ix_(cpos[rr], cpos)], nan=-1.0)
        local["oracle"] = osim
        local["xcell"] = xsim
        local["xcell1"] = xsim1
        krr_names = ("morgan2", "target", "ppi", "ppi_rewired", "chem_target")
        kernel = {name: local[name] for name in krr_names}
        kernel_cross = dict(kernel)
        if args.exclude_same_plate:
            # exploratory X2: no arm may use a neighbour that shares an L1000 plate with the query
            for name in list(local):
                local[name] = np.where(same_plate, -np.inf, local[name]).astype(np.float32)
            kernel_cross = {name: np.where(same_plate, 0.0, K).astype(np.float32) for name, K in kernel.items()}
        for f in range(N_OUTER):
            te = np.where(test_rf[:, f])[0]; tr = np.where(train_rf[:, f])[0]
            if len(te) == 0:
                continue
            inner = [(v[cpos[tr]], t[cpos[tr]]) for v, t in inner_cf[f]]
            m = Y[tr].mean(0)
            preds = {"mean": np.repeat(m[None], len(te), 0)}
            sel = {}
            inner_score = {}
            # chemistry, confound arms and ceilings/probes: all (or all measured) training compounds
            for name in chem_arms + ["morgan2_xplate", "oracle", "oracle_xplate", "xcell", "xcell1"]:
                pool = np.ones(len(tr), bool) if not name.startswith("xcell") else has_other[tr]
                (k, tau, alpha), score = select_knn(local[name], tr, Y, active, inner, pool)
                nb = tr[pool]
                S = local[name][np.ix_(te, nb)]
                agg = knn_residual(S, topk_order(S, max(GRID_K)), Y[nb] - m, k, tau)
                label = f"{name}_knn" if name.startswith(("oracle", "xcell")) else f"knn_{name}"
                preds[label] = m + alpha * agg
                sel[name] = (k, tau, alpha); inner_score[name] = score
            # honest "best chemistry": the encoder with the best inner-CV score in this line-fold
            best_enc = max(chem_arms, key=lambda e: inner_score[e])
            preds["knn_chem_select"] = preds[f"knn_{best_enc}"]; sel["chem_select"] = best_enc
            # knowledge arms: KNOWN training compounds only
            kpool = known[cpos[tr]]
            for name in known_arms:
                (k, tau, alpha), _ = select_knn(local[name], tr, Y, active & known[cpos], inner, kpool)
                nb = tr[kpool]
                S = local[name][np.ix_(te, nb)]
                agg = knn_residual(S, topk_order(S, max(GRID_K)), Y[nb] - m, k, tau)
                label = {"morgan2": "knn_chem_known", "moa": "moa_knn"}.get(name, f"knn_{name}")
                preds[label] = m + alpha * agg
                sel[f"known_{name}"] = (k, tau, alpha)
            # morphology arms: training compounds with a Cell Painting profile only
            mpool = has_morph[cpos[tr]]
            if mpool.sum() >= 20:
                for name in morph_arms:
                    (k, tau, alpha), _ = select_knn(local[name], tr, Y, active & has_morph[cpos], inner, mpool)
                    nb = tr[mpool]
                    S = local[name][np.ix_(te, nb)]
                    agg = knn_residual(S, topk_order(S, max(GRID_K)), Y[nb] - m, k, tau)
                    label = "knn_chem_morphpool" if name == "morgan2" else f"knn_{name}"
                    preds[label] = m + alpha * agg
                    sel[f"morph_{name}"] = (k, tau, alpha)
            # TxGNN arms: training compounds with a TxGNN drug node only
            tpool = has_tx[cpos[tr]]
            if tx_arms and tpool.sum() >= 20:
                for name in tx_arms:
                    (k, tau, alpha), _ = select_knn(local[name], tr, Y, active & has_tx[cpos], inner, tpool)
                    nb = tr[tpool]
                    S = local[name][np.ix_(te, nb)]
                    agg = knn_residual(S, topk_order(S, max(GRID_K)), Y[nb] - m, k, tau)
                    label = "knn_chem_txpool" if name == "morgan2" else f"knn_{name}"
                    preds[label] = m + alpha * agg
                    sel[f"tx_{name}"] = (k, tau, alpha)
                # deployable form: TxGNN where the compound is covered, the training mean otherwise
                cov = has_tx[cpos[te]][:, None]
                preds["txgnn_or_mean"] = np.where(cov, preds["knn_txgnn"], preds["mean"])
                preds["chemtx_or_chem"] = np.where(cov, preds["knn_chem_tx"], preds[f"knn_{best_enc}"])
            # kernel ridge: morgan (all training) and every knowledge kernel (KNOWN pool)
            lam = select_krr(kernel["morgan2"], tr, Y, active, inner, K_cross=kernel_cross["morgan2"])
            p = krr_fit_predict(kernel["morgan2"][np.ix_(tr, tr)] + 1e-6 * np.eye(len(tr)),
                                kernel_cross["morgan2"][np.ix_(te, tr)], Y[tr] - m, [lam])[lam]
            preds["krr_morgan2"] = m + p; sel["krr_morgan2"] = lam
            ktr = tr[kpool]
            for name, label in (("morgan2", "krr_chem_known"), ("target", "krr_target"), ("ppi", "krr_ppi"),
                                ("ppi_rewired", "krr_ppi_rewired"), ("chem_target", "krr_chem_target")):
                lam = select_krr(kernel[name], tr, Y, active, inner, pool=kpool, K_cross=kernel_cross[name])
                p = krr_fit_predict(kernel[name][np.ix_(ktr, ktr)] + 1e-6 * np.eye(len(ktr)),
                                    kernel_cross[name][np.ix_(te, ktr)], Y[ktr] - m, [lam])[lam]
                preds[label] = m + p; sel[label] = lam
            # copy probes: the compound's residual in its anchor line (xcell1) or mean over other lines (xcell)
            copy1 = np.zeros((len(te), Y.shape[1])); tr_copy1 = np.zeros((len(tr), Y.shape[1]))
            copy = np.zeros((len(te), Y.shape[1])); ncopy = np.zeros(len(te))
            tr_copy = np.zeros((len(tr), Y.shape[1])); tr_n = np.zeros(len(tr))
            for x in others:
                j = lines.index(x)
                xr = resp_rows[x]; xc = row_c[xr]
                mx = Y_all[xr[train_cf[xc, f]]].mean(0)  # line x's own training mean in this outer fold
                lookup = pd.Series(np.arange(len(xr)), index=xc)
                for idx, acc, cnt, acc1 in ((te, copy, ncopy, copy1), (tr, tr_copy, tr_n, tr_copy1)):
                    hit = np.isin(cpos[idx], xc)
                    if hit.any():
                        resid_x = Y_all[xr[lookup.loc[cpos[idx][hit]].to_numpy()]] - mx
                        acc[hit] += resid_x; cnt[hit] += 1
                        is_anchor = anchor[idx][hit] == j
                        acc1[np.where(hit)[0][is_anchor]] = resid_x[is_anchor]
            copy = copy / np.maximum(ncopy, 1)[:, None]
            tr_copy = tr_copy / np.maximum(tr_n, 1)[:, None]
            ok = active[tr] & (tr_n > 0)
            for label, test_mat, train_mat in (("xcell_copy", copy, tr_copy), ("xcell1_copy", copy1, tr_copy1)):
                # alpha only rescales the copied residual; choose it on training compounds' own copies
                a = max(GRID_COPY_ALPHA, key=lambda a: rowwise_pearson(m + a * train_mat[ok], Y[tr][ok]).mean())
                preds[label] = m + a * test_mat; sel[label] = a

            truth = Y[te]
            for arm, pred in preds.items():
                if not np.isfinite(pred).all():
                    raise ValueError(f"{arm}: non-finite prediction in {L} fold {f}")
                full = rowwise_pearson(pred, truth)
                resid = rowwise_pearson(pred - m, truth - m)
                top_t = np.argsort(-np.abs(truth), 1)[:, :100]
                top_p = np.argsort(-np.abs(pred), 1)[:, :100]
                overlap = np.array([len(np.intersect1d(a, b, assume_unique=True)) / 100 for a, b in zip(top_t, top_p)])
                norm_p = np.linalg.norm(pred - m, axis=1); norm_t = np.linalg.norm(truth - m, axis=1)
                for j, r in enumerate(te):
                    records.append((arm, L, compounds[cpos[r]], ik14[cpos[r]], group_c[cpos[r]], f, bool(active[r]),
                                    bool(known[cpos[r]]), bool(has_other[r]), bool(has_morph[cpos[r]]),
                                    bool(has_tx[cpos[r]]), full[j], resid[j], overlap[j], norm_p[j], norm_t[j]))
            selections.append({"line": L, "fold": f, **{k: str(v) for k, v in sel.items()},
                               **{f"inner_{k}": v for k, v in inner_score.items()}})
            print(f"  {L} fold {f}: {len(te)} test, {int(active[te].sum())} active, chem_select={best_enc} "
                  f"({time.time() - started:.0f}s)", flush=True)

    out = args.out / (f"{args.slice}_{args.split}_seed{args.seed}{'_xplate' if args.exclude_same_plate else ''}"
                      f"{'_tgtunion' if args.targets == 'union' else ''}{'_txgnn' if args.txgnn else ''}")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records, columns=["arm", "line", "inchikey", "ik14", "group", "fold", "active", "known",
                                           "xcell", "morph", "tx", "pearson_full", "pearson_resid", "top100",
                                           "norm_pred", "norm_true"])
    frame.to_parquet(out / "per_condition.parquet", index=False)
    pd.DataFrame(selections).to_csv(out / "selections.csv", index=False)
    summary = summarize(frame, args.n_boot, args.seed)
    for name, table_out in summary.items():
        table_out.to_csv(out / f"{name}.csv", index=False)
    manifest = {
        "slice": args.slice, "split": args.split, "seed": args.seed, "n_rows": len(Y_all),
        "n_compounds": n_c, "n_known_mapped_to_string": int(known.sum()), "ppi_genes": len(genes),
        "ppi_edges": n_edges, "min_score": args.min_score, "restart": args.restart, "grids": {
            "k": GRID_K, "tau": [str(t) for t in GRID_TAU], "alpha": GRID_ALPHA, "lambda": GRID_LAMBDA,
            "copy_alpha": GRID_COPY_ALPHA}, "active_tas": ACTIVE_TAS, "arms": sorted(frame.arm.unique()),
        "chem_arms": chem_arms, "n_boot": args.n_boot,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "slice_sha256": sha256(d / f"slice_{args.slice}.npz"),
        "features_sha256": sha256(d / "drug_features.npz"),
        "lm_sha256": {lm: sha256(d / f"smiles_lm_{lm}.npz") for lm in ("chemberta", "molformer")
                      if (d / f"smiles_lm_{lm}.npz").exists()},
        "drug_table_sha256": sha256(d / "drug_table.parquet"),
        "morph_sha256": sha256(morph_path) if morph_path.exists() else None, "n_morph": int(has_morph.sum()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "seconds": round(time.time() - started, 1),
        "exclude_same_plate": bool(args.exclude_same_plate), "source": args.source, "targets": args.targets,
        "txgnn": str(args.txgnn) if args.txgnn else None,
        "txgnn_sha256": sha256(args.txgnn) if args.txgnn else None, "n_txgnn": int(has_tx.sum()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for name in ("contrasts", "headroom", "magnitude"):
            print(summary[name].to_string(), flush=True)


COHORTS = {
    "ALL": lambda f: np.ones(len(f), bool),
    "ACTIVE": lambda f: f.active.to_numpy(),
    "KNOWN": lambda f: f.known.to_numpy(),
    "KNOWN_ACTIVE": lambda f: (f.known & f.active).to_numpy(),
    "XCELL": lambda f: f.xcell.to_numpy(),
    "XCELL_ACTIVE": lambda f: (f.xcell & f.active).to_numpy(),
    "MORPH": lambda f: f.morph.to_numpy(),
    "MORPH_ACTIVE": lambda f: (f.morph & f.active).to_numpy(),
    "MORPH_XCELL_ACTIVE": lambda f: (f.morph & f.xcell & f.active).to_numpy(),
    "TX": lambda f: f.tx.to_numpy(),
    "TX_ACTIVE": lambda f: (f.tx & f.active).to_numpy(),
    "TX_KNOWN_ACTIVE": lambda f: (f.tx & f.known & f.active).to_numpy(),
}
KNOWN_ONLY = {"knn_chem_known", "knn_target", "knn_target_perm", "knn_chem_target", "moa_knn", "knn_ppi",
              "knn_ppi_rewired", "krr_chem_known", "krr_target", "krr_ppi", "krr_ppi_rewired", "krr_chem_target"}
XCELL_ONLY = {"xcell_knn", "xcell1_knn", "xcell_copy", "xcell1_copy"}
MORPH_ONLY = {"knn_chem_morphpool", "knn_morph", "knn_morph_doses", "knn_chem_morph"}
TX_ONLY = {"knn_chem_txpool", "knn_txgnn", "knn_txgnn_perm", "knn_chem_tx"}
TX_ARMS = TX_ONLY | {"txgnn_or_mean", "chemtx_or_chem"}  # present only in --txgnn runs


def arm_valid(arm: str, cohort: str) -> bool:
    """Knowledge arms only predict KNOWN rows and xcell probes only XCELL rows."""
    if arm in KNOWN_ONLY:
        return cohort.startswith("KNOWN")
    if arm in XCELL_ONLY:
        return cohort.startswith("XCELL") or cohort == "MORPH_XCELL_ACTIVE"
    if arm in MORPH_ONLY:
        return cohort.startswith("MORPH")
    if arm in TX_ONLY:
        return cohort.startswith("TX")
    return True


CONTRASTS = [
    # G1 / G2 (approach 1): chemistry retrieval vs mean, and learned encoders vs fingerprints
    ("ACTIVE", "knn_chem_select", "mean"), ("ACTIVE", "knn_morgan2", "mean"), ("ACTIVE", "knn_maccs", "mean"),
    ("ACTIVE", "knn_rdkit2d", "mean"), ("ACTIVE", "knn_chemberta", "mean"), ("ACTIVE", "knn_molformer", "mean"),
    ("ACTIVE", "krr_morgan2", "mean"), ("ACTIVE", "krr_morgan2", "knn_morgan2"),
    ("ACTIVE", "knn_chemberta", "knn_morgan2"), ("ACTIVE", "knn_molformer", "knn_morgan2"),
    ("ACTIVE", "knn_maccs", "knn_morgan2"), ("ACTIVE", "knn_rdkit2d", "knn_morgan2"),
    ("ACTIVE", "oracle_knn", "mean"), ("ALL", "knn_chem_select", "mean"),
    # plate confound
    ("ACTIVE", "knn_morgan2", "knn_morgan2_xplate"), ("ACTIVE", "knn_morgan2_xplate", "mean"),
    ("ACTIVE", "oracle_knn", "oracle_xplate_knn"),
    # G3 (approach 2): knowledge beyond chemistry, and its controls
    ("KNOWN_ACTIVE", "knn_chem_known", "mean"), ("KNOWN_ACTIVE", "knn_target", "knn_chem_known"),
    ("KNOWN_ACTIVE", "knn_ppi", "knn_chem_known"), ("KNOWN_ACTIVE", "knn_ppi", "knn_ppi_rewired"),
    ("KNOWN_ACTIVE", "knn_target", "knn_target_perm"), ("KNOWN_ACTIVE", "knn_chem_target", "knn_chem_known"),
    ("KNOWN_ACTIVE", "moa_knn", "knn_chem_known"), ("KNOWN_ACTIVE", "knn_target", "mean"),
    ("KNOWN_ACTIVE", "knn_ppi", "mean"), ("KNOWN_ACTIVE", "knn_chem_select", "knn_chem_known"),
    ("KNOWN_ACTIVE", "krr_chem_known", "knn_chem_known"), ("KNOWN_ACTIVE", "krr_target", "krr_chem_known"),
    ("KNOWN_ACTIVE", "krr_ppi", "krr_chem_known"), ("KNOWN_ACTIVE", "krr_ppi", "krr_ppi_rewired"),
    ("KNOWN_ACTIVE", "krr_chem_target", "krr_chem_known"),
    # G4: one measured context vs chemistry
    ("XCELL_ACTIVE", "xcell_knn", "knn_chem_select"), ("XCELL_ACTIVE", "xcell_copy", "knn_chem_select"),
    ("XCELL_ACTIVE", "xcell1_knn", "knn_chem_select"), ("XCELL_ACTIVE", "xcell1_copy", "knn_chem_select"),
    ("XCELL_ACTIVE", "xcell1_knn", "mean"), ("XCELL_ACTIVE", "xcell_knn", "xcell1_knn"),
    ("XCELL_ACTIVE", "knn_chem_select", "mean"),
    # G5 (added before any official run): measured morphology as the new drug-side modality
    ("MORPH_ACTIVE", "knn_morph", "knn_chem_morphpool"), ("MORPH_ACTIVE", "knn_morph_doses", "knn_chem_morphpool"),
    ("MORPH_ACTIVE", "knn_chem_morph", "knn_chem_morphpool"), ("MORPH_ACTIVE", "knn_morph", "mean"),
    ("MORPH_ACTIVE", "knn_chem_morphpool", "mean"), ("MORPH_XCELL_ACTIVE", "xcell1_knn", "knn_morph"),
    ("MORPH_XCELL_ACTIVE", "knn_morph_doses", "knn_morph"),
    # TxGNN (exploratory, record 2026-10-07 13:40; primary = first row, on P1-24 and P2-24)
    ("TX_ACTIVE", "knn_txgnn", "knn_chem_txpool"), ("TX_ACTIVE", "knn_txgnn", "knn_txgnn_perm"),
    ("TX_ACTIVE", "knn_chem_tx", "knn_chem_txpool"), ("TX_ACTIVE", "knn_txgnn", "mean"),
    ("TX_ACTIVE", "knn_chem_txpool", "mean"), ("TX_KNOWN_ACTIVE", "knn_txgnn", "knn_target"),
    ("ACTIVE", "txgnn_or_mean", "mean"), ("ACTIVE", "chemtx_or_chem", "knn_chem_select"),
]


def line_macro_boot(frame: pd.DataFrame, values: np.ndarray, n_boot: int, seed: int):
    """Line-macro mean and bootstrap over split groups (ik14 or scaffold), jointly across lines.
    A line that receives no rows in a resample is left out of that replicate's macro mean."""
    groups, gidx = np.unique(frame.group.to_numpy(), return_inverse=True)
    lines = sorted(frame.line.unique())
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=n_boot).astype(np.float64)
    per_line, boots = {}, []
    for L in lines:
        mask = frame.line.to_numpy() == L
        dsum = np.bincount(gidx[mask], weights=values[mask], minlength=len(groups))
        nsum = np.bincount(gidx[mask], minlength=len(groups)).astype(np.float64)
        per_line[L] = dsum.sum() / max(nsum.sum(), 1)
        den = counts @ nsum
        boots.append(np.where(den > 0, (counts @ dsum) / np.maximum(den, 1e-12), np.nan))
    boot = np.nanmean(np.array(boots), 0)
    return float(np.mean(list(per_line.values()))), np.quantile(boot, [0.025, 0.975]), per_line


def summarize(frame: pd.DataFrame, n_boot: int, seed: int) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    rows = []
    for cohort, fn in COHORTS.items():
        for arm, sub in frame.groupby("arm"):
            if not arm_valid(arm, cohort):
                continue
            sub = sub[fn(sub)]
            if len(sub) == 0:
                continue
            for metric in ("pearson_full", "pearson_resid", "top100"):
                per_line = sub.groupby("line")[metric].mean()
                rows.append({"cohort": cohort, "arm": arm, "metric": metric, "line_macro": per_line.mean(),
                             "n_rows": len(sub), **{f"line_{k}": v for k, v in per_line.items()}})
    cohorts = pd.DataFrame(rows)
    contrasts = []
    for cohort, a, b in CONTRASTS:
        fa = frame[frame.arm == a]; fb = frame[frame.arm == b]
        if fa.empty or fb.empty:
            if {a, b} & MORPH_ONLY:
                print(f"WARNING: contrast {a} - {b} skipped (morphology arm absent)", flush=True)
                continue
            if {a, b} & TX_ARMS:
                continue  # TxGNN arms exist only in --txgnn runs
            raise ValueError(f"contrast {a} - {b}: arm missing from the run")
        merged = fa.merge(fb[["line", "inchikey", "pearson_full", "pearson_resid"]], on=["line", "inchikey"],
                          suffixes=("_a", "_b"))
        merged = merged[COHORTS[cohort](merged)]
        for metric in ("pearson_full", "pearson_resid"):
            diff = (merged[f"{metric}_a"] - merged[f"{metric}_b"]).to_numpy()
            est, ci, per_line = line_macro_boot(merged, diff, n_boot, seed)
            contrasts.append({"cohort": cohort, "arm": a, "minus": b, "metric": metric, "diff": est,
                              "boot_low": ci[0], "boot_high": ci[1], "n_rows": len(merged),
                              "lines_positive": int(sum(v > 0 for v in per_line.values())),
                              "n_lines": len(per_line)})
    # share of the oracle headroom captured (ACTIVE, point estimates)
    head = []
    act = cohorts[cohorts.cohort.eq("ACTIVE")].set_index(["arm", "metric"]).line_macro
    for metric in ("pearson_full", "pearson_resid"):
        base, top = act.get(("mean", metric), np.nan), act.get(("oracle_knn", metric), np.nan)
        for arm in sorted({a for a, mtr in act.index if mtr == metric}):
            head.append({"metric": metric, "arm": arm, "score": act[(arm, metric)],
                         "share_of_oracle_headroom": (act[(arm, metric)] - base) / (top - base)})
    mags = []
    for cohort in ("ALL", "KNOWN", "XCELL", "MORPH", "TX"):
        for arm, sub in frame.groupby("arm"):
            if arm == "mean" or not arm_valid(arm, cohort):
                continue
            sub = sub[COHORTS[cohort](sub)]
            for L, s in sub.groupby("line"):
                rho = spearmanr(s.norm_pred, s.norm_true).statistic if s.norm_pred.std() > 0 else np.nan
                auc = roc_auc_score(s.active, s.norm_pred) if s.active.nunique() == 2 and s.norm_pred.std() > 0 else np.nan
                mags.append({"cohort": cohort, "arm": arm, "line": L, "spearman_norm": rho, "auroc_active": auc})
    mag = pd.DataFrame(mags).groupby(["cohort", "arm"])[["spearman_norm", "auroc_active"]].mean().reset_index()
    return {"summary_cohorts": cohorts, "contrasts": pd.DataFrame(contrasts), "headroom": pd.DataFrame(head),
            "magnitude": mag}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base = Path("/nfs-stor/yuqi.lei/Therapeutic")
    p.add_argument("--slice", default="P1-24", choices=["P1-24", "P1-6", "P2-24", "T-0.05", "T-0.5", "T-5"])
    p.add_argument("--source", default="L1000", choices=["L1000", "Tahoe"],
                   help="drug_table rows used for ik14/scaffold/targets/MOA")
    p.add_argument("--split", default="ik14", choices=["ik14", "scaffold", "platemap"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--min-score", type=int, default=400)
    p.add_argument("--restart", type=float, default=0.5)
    p.add_argument("--allow-missing-lm", action="store_true")
    p.add_argument("--allow-missing-morph", action="store_true")
    p.add_argument("--targets", default="compoundinfo", choices=["compoundinfo", "union"],
                   help="union adds PrimeKG/DrugBank targets (exploratory)")
    p.add_argument("--txgnn", type=Path, default=None,
                   help="TxGNN drug-embedding npz (scripts/txgnn_drug_embeddings.py); adds TX arms (exploratory)")
    p.add_argument("--exclude-same-plate", action="store_true",
                   help="exploratory X2: drop same-plate neighbours from every retrieval arm")
    p.add_argument("--derived", type=Path, default=base / "derived/l1000_unseen_drug")
    p.add_argument("--string-root", type=Path, default=base / "raw/string_v12")
    p.add_argument("--out", type=Path, default=base / "results/l1000_unseen_drug")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
