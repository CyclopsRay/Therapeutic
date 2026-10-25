#!/usr/bin/env python3
"""Integrated L1000 unseen-drug model, step 1: multiple-kernel KRR (PLAN.md Phase 7; Jun Wen's
integrated-model direction; exploratory, 2026-10-05).

One kernel per information view over compounds (zero rows/columns where a compound lacks the view,
which keeps every kernel PSD):
  chem   Morgan-2 Tanimoto (all compounds)
  tgt    Tanimoto of target multi-hot (KNOWN)            | null: targets permuted among KNOWN
  moa    Tanimoto of MOA multi-hot (compounds with a MOA)
  deg    RBF on [mean, max log(1 + STRING degree) of targets, log #targets] (KNOWN)
  ppi1   cosine of D^-1/2 (A + I) D^-1/2 t, one-hop target expansion (KNOWN) | null: rewired graph
  morph  cosine of z-scored 10 uM Cell Painting (MORPH)
  reg    cosine of the drug's downstream influence over OmniPath signaling + CollecTRI (TF -> target):
         x = t A_out + 0.5 t A_out^2 with row-normalised directed adjacency (KNOWN in that network)
         | null: in/out-degree-preserving rewired network (added 2026-10-06)
  xresp  mean response correlation in the OTHER lines of the slice (the drug measured elsewhere;
         a different deployment tier, never the target line)
Weights: centred kernel-target alignment on the outer-training compounds of each (line, fold),
w_v proportional to max(alignment, 0); lambda by inner group CV (run.GRID_LAMBDA). Prediction:
line training mean + KRR on residuals, as krr_* arms of run_l1000_unseen_drug.py, so rows pair with
that runner's per_condition.parquet (same folds via run.make_folds).
Tiers: T1 chem; T2 + tgt, moa, deg, ppi1; T3 + morph; T4 + xresp; plus T2null / T3null with the
null versions of tgt and ppi1, and every single-view KRR.
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

import netnull
import run_l1000_unseen_drug as run

VIEWS_T1 = ["chem"]
VIEWS_T2 = VIEWS_T1 + ["tgt", "moa", "deg", "ppi1"]
TIERS = {"T1": VIEWS_T1, "T2": VIEWS_T2, "T3": VIEWS_T2 + ["morph"], "T4": VIEWS_T2 + ["morph", "xresp"],
         "T2null": ["chem", "tgt_null", "moa", "deg", "ppi1_null"],
         "T3null": ["chem", "tgt_null", "moa", "deg", "ppi1_null", "morph"],
         # added 2026-10-06 (E1 follow-up): directed signaling -> TF -> target view and its null
         "T2reg": VIEWS_T2 + ["reg"], "T2regnull": VIEWS_T2 + ["reg_null"],
         "T3reg": VIEWS_T2 + ["reg", "morph"], "T3regnull": VIEWS_T2 + ["reg_null", "morph"]}


def tanimoto_masked(F: np.ndarray, has: np.ndarray) -> np.ndarray:
    K = run.tanimoto(F, F).astype(np.float64)
    K[~has] = 0; K[:, ~has] = 0
    return K


def cosine_masked(F: np.ndarray, has: np.ndarray) -> np.ndarray:
    K = np.zeros((len(F), len(F)))
    i = np.where(has)[0]
    K[np.ix_(i, i)] = run.cosine(F[i], F[i])
    return K


def one_hop(net: netnull.Network, mapped: list[list[str]]) -> np.ndarray:
    gpos = {g: i for i, g in enumerate(net.genes)}
    M = (net.adjacency(True) + sp.identity(net.n, format="csr")).tocsr()
    d = np.asarray(M.sum(1)).ravel(); Dm = sp.diags(1 / np.sqrt(np.maximum(d, 1)))
    M = Dm @ M @ Dm
    T = np.zeros((len(mapped), net.n))
    for c, t in enumerate(mapped):
        for g in t:
            if g in gpos:
                T[c, gpos[g]] = 1.0
    return np.asarray((M @ T.T).T)


def downstream(net: netnull.Network, targets: list[set]) -> tuple[np.ndarray, np.ndarray]:
    """1- and 2-hop downstream mass from each drug's targets along directed edges (row-normalised)."""
    gpos = {g: i for i, g in enumerate(net.genes)}
    A = net.adjacency(symmetric=False).astype(float)
    out = np.asarray(A.sum(1)).ravel(); A = sp.diags(1 / np.maximum(out, 1)) @ A
    T = np.zeros((len(targets), net.n)); has = np.zeros(len(targets), bool)
    for c, t in enumerate(targets):
        idx = [gpos[g] for g in t if g in gpos]
        if idx:
            T[c, idx] = 1.0 / len(idx); has[c] = True
    X1 = np.asarray((A.T @ T.T).T); X2 = np.asarray((A.T @ X1.T).T)
    return X1 + 0.5 * X2, has


def centered_alignment(K: np.ndarray, R: np.ndarray) -> float:
    n = len(K); H = np.eye(n) - 1.0 / n
    Kc = H @ K @ H; Y = H @ (R @ R.T) @ H
    den = np.linalg.norm(Kc) * np.linalg.norm(Y)
    return float((Kc * Y).sum() / den) if den > 0 else 0.0


def main(args: argparse.Namespace) -> None:
    started = time.time()
    print(f"host {platform.node()} python {sys.version.split()[0]} numpy {np.__version__} pandas {pd.__version__} "
          f"slice {args.slice} split {args.split} seed {args.seed}", flush=True)
    d = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/l1000_unseen_drug")
    sl = np.load(d / f"slice_{args.slice}.npz", allow_pickle=False)
    keys = sl["inchikey"].astype(str); line_rows = sl["line"].astype(str); Y_all = sl["Y"].astype(np.float64)
    active_all = sl["tas"] >= run.ACTIVE_TAS
    compounds = np.array(sorted(set(keys))); cidx = {k: i for i, k in enumerate(compounds)}
    row_c = np.array([cidx[k] for k in keys]); n_c = len(compounds)
    table = pd.read_parquet(d / "drug_table.parquet")
    table = table[table.source.eq("L1000") & table.inchikey.isin(compounds)]
    first = table.groupby("inchikey").first().reindex(compounds)
    ik14 = first.ik14.astype(str).to_numpy(); scaf = first.generic_scaffold.fillna("").astype(str).to_numpy()
    targets = {k: set() for k in compounds}; moas = {k: set() for k in compounds}
    for r in table.itertuples():
        if r.targets:
            targets[r.inchikey].update(r.targets.split(";"))
        if r.moa:
            moas[r.inchikey].update(r.moa.split(";"))  # as the runner
    test_cf, train_cf, inner_cf, group_c = run.make_folds(args.split, args.seed, ik14, scaf, row_c,
                                                          sl["plates"].astype(str), n_c)
    feats = np.load(d / "drug_features.npz", allow_pickle=False)
    fpos = pd.Series(np.arange(len(feats["inchikey"])), index=feats["inchikey"].astype(str))[compounds].to_numpy()
    K = {"chem": run.tanimoto(feats["morgan2"][fpos], feats["morgan2"][fpos]).astype(np.float64)}
    # knowledge views on the runner's STRING graph
    net = netnull.load("string_physical400"); rew = netnull.null(net, "rewired", args.seed)
    gset = set(net.genes)
    mapped = [sorted(t & gset) for t in (targets[k] for k in compounds)]
    known = np.array([len(t) > 0 for t in mapped]); kn = np.where(known)[0]
    tg = sorted({g for t in mapped for g in t}); tpos = {g: i for i, g in enumerate(tg)}
    Tm = np.zeros((n_c, len(tg)), np.uint8)
    for c, t in enumerate(mapped):
        Tm[c, [tpos[g] for g in t]] = 1
    rng = np.random.default_rng(args.seed)
    Tp = Tm.copy(); Tp[kn] = Tm[rng.permutation(kn)]
    K["tgt"] = tanimoto_masked(Tm, known); K["tgt_null"] = tanimoto_masked(Tp, known)
    ml = sorted({m for k in compounds for m in moas[k]}); mp = {m: i for i, m in enumerate(ml)}
    Mm = np.zeros((n_c, max(1, len(ml))), np.uint8)
    for c, k in enumerate(compounds):
        Mm[c, [mp[m] for m in moas[k]]] = 1
    K["moa"] = tanimoto_masked(Mm, Mm.any(1))
    deg = dict(zip(net.genes, net.degree()))
    F = np.zeros((n_c, 3))
    for c, t in enumerate(mapped):
        if t:
            ld = np.log1p([deg[g] for g in t]); F[c] = [ld.mean(), ld.max(), np.log1p(len(t))]
    Fz = (F - F[kn].mean(0)) / np.maximum(F[kn].std(0), 1e-9)
    D2 = ((Fz[:, None, :] - Fz[None, :, :]) ** 2).sum(-1)
    bw = np.median(D2[np.ix_(kn, kn)]) or 1.0
    K["deg"] = np.exp(-D2 / bw); K["deg"][~known] = 0; K["deg"][:, ~known] = 0
    K["ppi1"] = cosine_masked(one_hop(net, mapped), known)
    K["ppi1_null"] = cosine_masked(one_hop(rew, mapped), known)
    reg = netnull.load("signaling_tf"); reg_rew = netnull.null(reg, "rewired", args.seed)
    tsets = [targets[k] for k in compounds]
    Xr, has_reg = downstream(reg, tsets); Xn, _ = downstream(reg_rew, tsets)
    K["reg"] = cosine_masked(Xr, has_reg); K["reg_null"] = cosine_masked(Xn, has_reg)
    print(f"signaling_tf: {netnull.summary(reg)}; compounds with a target in it: {has_reg.sum()}", flush=True)
    cp = np.load(d / "cell_painting_features.npz", allow_pickle=False)
    cpos = pd.Series(np.arange(len(cp["inchikey"])), index=cp["inchikey"].astype(str))
    has_morph = np.array([k in cpos.index for k in compounds])
    Fm = np.zeros((n_c, cp["morph10"].shape[1]))
    Fm[has_morph] = cp["morph10"][cpos[compounds[has_morph]].to_numpy()]
    Fm[has_morph] = (Fm[has_morph] - Fm[has_morph].mean(0)) / np.maximum(Fm[has_morph].std(0), 1e-9)
    K["morph"] = cosine_masked(Fm, has_morph)
    lines = sorted(set(line_rows))
    Yc = np.full((n_c, len(lines), Y_all.shape[1]), np.nan)
    for r, (c, L) in enumerate(zip(row_c, line_rows)):
        Yc[c, lines.index(L)] = Y_all[r]
    print(f"kernels built: KNOWN {known.sum()}, MORPH {has_morph.sum()} ({time.time() - started:.0f}s)", flush=True)

    records, weights_log = [], []
    for li, L in enumerate(lines):
        if args.lines and L not in args.lines:
            continue
        rows = np.where(line_rows == L)[0]; cl = row_c[rows]; Y = Y_all[rows]; act = active_all[rows]
        # xresp: mean correlation of the two compounds' residual signatures over the OTHER lines
        others = [j for j in range(len(lines)) if j != li]
        acc = np.zeros((n_c, n_c)); cnt = np.zeros((n_c, n_c))
        for j in others:
            m_j = ~np.isnan(Yc[:, j, 0]); idx = np.where(m_j)[0]
            Z = Yc[idx, j] - np.nanmean(Yc[idx, j], 0)
            C = run.corr_matrix(Z, Z).astype(np.float64)
            acc[np.ix_(idx, idx)] += C; cnt[np.ix_(idx, idx)] += 1
        Kx = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
        np.fill_diagonal(Kx, np.where(np.diag(cnt) > 0, 1.0, 0.0))
        has_x = np.diag(cnt) > 0
        local = {v: M[np.ix_(cl, cl)] for v, M in K.items()}
        local["xresp"] = Kx[np.ix_(cl, cl)]
        for f in (args.folds or range(run.N_OUTER)):
            tr = np.where(train_cf[cl, f])[0]; te = np.where(test_cf[cl, f])[0]
            if len(te) == 0 or len(tr) < 20:
                continue
            m = Y[tr].mean(0); R = Y[tr] - m
            align = {v: centered_alignment(local[v][np.ix_(tr, tr)], R) for v in local}
            # inner splits as row masks over this line's training rows
            splits = [(iv[cl][tr], it[cl][tr]) for iv, it in inner_cf[f]]
            preds = {}
            arms = {f"krr_{v}": [v] for v in local} | {f"mkl_{t}": vs for t, vs in TIERS.items()}
            for arm, vs in arms.items():
                w = np.array([max(align[v], 0.0) for v in vs])
                if w.sum() <= 0:
                    w = np.ones(len(vs))
                w = w / w.sum()
                Kc = sum(wi * local[v] for wi, v in zip(w, vs))
                lam = run.select_krr(Kc, tr, Y, act, splits)
                p = run.krr_fit_predict(Kc[np.ix_(tr, tr)] + 1e-6 * np.eye(len(tr)), Kc[np.ix_(te, tr)], R, [lam])[lam]
                preds[arm] = m + p
                if arm.startswith("mkl_"):
                    weights_log.append({"line": L, "fold": f, "arm": arm, "lambda": lam,
                                        **{f"w_{v}": float(wi) for wi, v in zip(w, vs)}})
            preds["mean"] = np.tile(m, (len(te), 1))
            for arm, P in preds.items():
                full = run.rowwise_pearson(P, Y[te]); resid = run.rowwise_pearson(P - m, Y[te] - m)
                for j, r_ in enumerate(te):
                    c = cl[r_]
                    records.append((arm, L, compounds[c], ik14[c], group_c[c], f, bool(act[r_]), bool(known[c]),
                                    bool(has_morph[c]), bool(has_x[c]), float(full[j]), float(resid[j])))
        print(f"  {L} done ({time.time() - started:.0f}s)", flush=True)
    out = Path(args.out) / f"{args.slice}_{args.split}_seed{args.seed}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records, columns=["arm", "line", "inchikey", "ik14", "group", "fold", "active", "known",
                                           "morph", "xcell", "pearson_full", "pearson_resid"])
    frame.to_parquet(out / "per_condition.parquet", index=False)
    pd.DataFrame(weights_log).to_csv(out / "weights.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "slice": args.slice, "split": args.split, "seed": args.seed, "tiers": TIERS, "network": "string_physical400",
        "script_sha256": run.sha256(Path(__file__).resolve()), "slice_sha256": run.sha256(d / f"slice_{args.slice}.npz"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "seconds": round(time.time() - started, 1)}, indent=2) + "\n")
    a = frame[frame.active]
    print(a.groupby("arm")[["pearson_full", "pearson_resid"]].mean().round(4).to_string(), flush=True)
    for coh, mask in (("KNOWN", a.known), ("MORPH", a.morph), ("XCELL", a.xcell)):
        print(coh, a[mask].groupby("arm").pearson_full.mean().round(4).to_dict(), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slice", default="P2-24", choices=["P1-24", "P1-6", "P2-24"])
    p.add_argument("--split", default="platemap", choices=["ik14", "scaffold", "platemap"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--lines", nargs="*", default=None, help="restrict to these lines (smoke tests)")
    p.add_argument("--folds", nargs="*", type=int, default=None)
    p.add_argument("--out", default="/nfs-stor/yuqi.lei/Therapeutic/results/l1000_mkl")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
