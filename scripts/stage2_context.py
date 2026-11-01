#!/usr/bin/env python3
"""Second stage on the UnDetect cache (`cache_undetect.py`): context-gated compound term (C-lite) and exact-analog
term (A). Closed form; settings chosen on valid (PCC_DEG, as every UnDetect arm); test once.

u = g·G + h·H is UnDetect's compound term (pool_kta = B + u). On training rows G and H are leave-one-out values.
C-lite, per gene j, ridge on training rows towards the current model:
    r_ij − u_ij ≈ u_ij · (δ_j + β_j·x̃_ij + γ_j·z_i),   r = y − B,
x̃ = control expression z-scored on training rows, z = top-K PCs of the training control rows (standardised).
Arms: ctx_scale (δ only: per-gene rescaling, the control), ctx_gene (δ, β), ctx_pc (δ, γ), ctx_both; with --v2
also ctx_pc_ext (K up to 80) and ctx_line (per line × gene rescaling: line identity instead of control state). Ridge strength
c·mean_j Σ_i u_ij² with c in CS; c = inf returns pool_kta, so every arm nests the current model.
A: n_i = mean over training compounds with Morgan Tanimoto ≥ τ to the test compound of their residual in the same
line (dup_line; their compound mean when not measured in that line) or of their compound mean (dup_mean); pool_kta +
c·n with c in SCALES (0 allowed), τ in TAUS. Writes results/sdst_hybrid/<tag>/fold_<f>/{summary.json, pred_test.npz}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sdst_hybrid import D, OUT, SCALES, drug_kernels  # noqa: E402

CS = (0.001, 0.01, 0.1, 1.0, 10.0, float("inf"))
KS = (5, 10, 20)
KS_EXT = (5, 10, 20, 40, 80)  # v2 (record 2026-10-31 20:45): validation chose K = 20, the v1 grid edge
TAUS = (1.0, 0.9, 0.8)


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def gate_fit(U: np.ndarray, T_: np.ndarray, F: np.ndarray, lam: float) -> np.ndarray:
    """Per gene j: ridge of T_[:, j] on U[:, j]·F (F: n × p context features incl. a constant). Returns p × genes."""
    p = F.shape[1]; F2 = (F[:, :, None] * F[:, None, :]).reshape(len(F), p * p)
    Gm = ((U * U).T @ F2).reshape(-1, p, p) + lam * np.eye(p)[None]
    b = (U * T_).T @ F
    return np.linalg.solve(Gm, b[:, :, None])[:, :, 0].T


def gate_apply(U: np.ndarray, F: np.ndarray, W: np.ndarray) -> np.ndarray:
    return U * (F @ W)


def gate_fit_gene(U, T_, X, lam):
    """Per gene j: ridge of T_[:, j] on [U_j, U_j·X_j] (gene-local gating). Returns 2 × genes."""
    a = (U * U).sum(0) + lam; b = (U * U * X).sum(0); d = (U * U * X * X).sum(0) + lam
    r1 = (U * T_).sum(0); r2 = (U * X * T_).sum(0); det = a * d - b * b
    return np.stack([(d * r1 - b * r2) / det, (a * r2 - b * r1) / det])


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--cache", default="cache_v1"); p.add_argument("--tag", default="stage2_v1")
    p.add_argument("--v2", action="store_true", help="add ctx_pc_ext and ctx_line (record 2026-10-31 20:45)"); args = p.parse_args()
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, pidx, cell = z["ctl"].astype(np.float64), z["delta"].astype(np.float64), z["group"], z["pert_idx"], z["cell_idx"]
    table = pd.read_csv(D / "drugs.tsv", sep="\t", keep_default_na=False)
    rowof = {d: i for i, d in enumerate(table.pert_idx)}; drow = np.array([rowof[d] for d in pidx])
    K, _ = drug_kernels(table); Kc = K["chem"]; del K
    for f in args.folds:
        te = g == f; va = g == (f + 1) % 5; tr = ~te & ~va
        cz = np.load(OUT / args.cache / f"fold_{f}" / "cache.npz"); S = json.loads((OUT / args.cache / f"fold_{f}" / "summary.json").read_text())
        B, G, H = (cz[k].astype(np.float64) for k in ("B", "G", "H"))
        U = S["g"] * G + S["h"] * H; R = Y - B; P0 = B + U
        sc = lambda P, m: float(pcc(Y[m], P[m]).mean())
        res = {"fold": f, "cache": S, "arms": {}}; preds = {"pool_kta": P0[te]}
        res["arms"]["pool_kta"] = {"valid": sc(P0, va), "test": sc(P0, te)}
        print(f"fold {f}: pool_kta valid {res['arms']['pool_kta']['valid']:.4f} test {res['arms']['pool_kta']['test']:.4f} "
              f"(cache {S['test']['pool_kta']:.4f})", flush=True)
        # ---------- C-lite
        mu, sd = ctl[tr].mean(0), ctl[tr].std(0) + 1e-9; X = (ctl - mu) / sd
        Xc = ctl[tr] - mu; _, _, Vt = np.linalg.svd(Xc[np.unique(Xc.round(5), axis=0, return_index=True)[1]], full_matrices=False)
        Zall = (ctl - mu) @ Vt[:max(KS_EXT)].T; Zall /= Zall[tr].std(0) + 1e-9
        scale = float((U[tr] ** 2).sum(0).mean()); T_ = R[tr] - U[tr]
        best = {}
        for cs in CS:
            lam = cs * scale
            cand = {}
            if np.isinf(cs):
                for arm in ("ctx_scale", "ctx_gene", "ctx_pc", "ctx_both") + (("ctx_pc_ext", "ctx_line") if args.v2 else ()):
                    cand[(arm, None)] = P0
            else:
                one = np.ones((tr.sum(), 1))
                Wd = gate_fit(U[tr], T_, one, lam); cand[("ctx_scale", None)] = P0 + gate_apply(U, np.ones((len(Y), 1)), Wd)
                Wg = gate_fit_gene(U[tr], T_, X[tr], lam); cand[("ctx_gene", None)] = P0 + U * (Wg[0][None] + Wg[1][None] * X)
                for k in KS:
                    Ftr = np.hstack([one, Zall[tr, :k]]); Fall = np.hstack([np.ones((len(Y), 1)), Zall[:, :k]])
                    Wp = gate_fit(U[tr], T_, Ftr, lam); cand[("ctx_pc", k)] = P0 + gate_apply(U, Fall, Wp)
                    # both: gene-local term fitted on the residual of the PC gate (two-step, closed form)
                    T2 = T_ - gate_apply(U[tr], Ftr, Wp); Wb = gate_fit_gene(U[tr], T2, X[tr], lam)
                    cand[("ctx_both", k)] = cand[("ctx_pc", k)] + U * (Wb[0][None] + Wb[1][None] * X)
                if args.v2:
                    for k in KS_EXT:
                        Ftr = np.hstack([one, Zall[tr, :k]]); Fall = np.hstack([np.ones((len(Y), 1)), Zall[:, :k]])
                        cand[("ctx_pc_ext", k)] = P0 + gate_apply(U, Fall, gate_fit(U[tr], T_, Ftr, lam))
                    # per (line, gene) rescaling by line identity: delta_lj = sum u r / (sum u^2 + lam_line)
                    lam_l = lam / max(len(np.unique(cell[tr])), 1)  # same relative shrinkage as the per-gene ridge
                    ct = cell[tr]; Ut = U[tr]
                    num = np.zeros((cell.max() + 1, Y.shape[1])); den = np.zeros_like(num)
                    np.add.at(num, ct, Ut * T_); np.add.at(den, ct, Ut * Ut); Dl = num / (den + lam_l)
                    cand[("ctx_line", None)] = P0 + U * Dl[cell]
            for (arm, k), P in cand.items():
                v = sc(P, va)
                if arm not in best or v > best[arm][0]:
                    best[arm] = (v, cs, k, P)
        for arm, (v, cs, k, P) in best.items():
            res["arms"][arm] = {"valid": v, "test": sc(P, te), "c": cs, "K": k}; preds[arm] = P[te]
            print(f"  {arm}: c {cs} K {k} valid {v:.4f} test {res['arms'][arm]['test']:.4f}", flush=True)
        # ---------- A: exact-analog term
        trd = np.unique(drow[tr]); tinv = np.searchsorted(trd, drow[tr])
        C = np.zeros((len(trd), Y.shape[1])); np.add.at(C, tinv, R[tr]); C /= np.bincount(tinv)[:, None]
        pair = {}
        for j, (d_, l_) in enumerate(zip(drow[tr], cell[tr])):
            pair.setdefault((int(d_), int(l_)), j)
        Rtr = R[tr]
        best_a = {}
        for tau in TAUS:
            nl = np.zeros_like(Y); nm = np.zeros_like(Y); has = np.zeros(len(Y), bool)
            for m in (va, te):
                rows = np.flatnonzero(m); uq = np.unique(drow[rows])
                sims = Kc[np.ix_(uq, trd)]; an = {d: np.flatnonzero(sims[i] >= tau - 1e-9) for i, d in enumerate(uq)}
                for i in rows:
                    a_ = an[drow[i]]
                    if not len(a_):
                        continue
                    has[i] = True; nm[i] = C[a_].mean(0)
                    vals = [Rtr[pair[(int(trd[j]), int(cell[i]))]] if (int(trd[j]), int(cell[i])) in pair else C[j] for j in a_]
                    nl[i] = np.mean(vals, 0)
            for arm, Nn in (("dup_line", nl), ("dup_mean", nm)):
                for c in SCALES:
                    P = P0 + c * Nn; v = sc(P, va)
                    if arm not in best_a or v > best_a[arm][0]:
                        best_a[arm] = (v, tau, c, P, float(has[te].mean()))
        for arm, (v, tau, c, P, cov) in best_a.items():
            res["arms"][arm] = {"valid": v, "test": sc(P, te), "tau": tau, "c": c, "test_coverage": cov}; preds[arm] = P[te]
            print(f"  {arm}: tau {tau} c {c} (test rows with an analog {cov:.3f}) valid {v:.4f} test {res['arms'][arm]['test']:.4f}", flush=True)
        out = OUT / args.tag / f"fold_{f}"; out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "pred_test.npz", y=Y[te].astype(np.float16), ctl=ctl[te].astype(np.float16),
                 **{k: v.astype(np.float16) for k, v in preds.items()})
        (out / "summary.json").write_text(json.dumps(res, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
