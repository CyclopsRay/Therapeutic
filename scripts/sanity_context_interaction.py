#!/usr/bin/env python3
"""Is there room for a context-aware model beyond UnDetect's backbone + compound term? (Bai et al. 2026 drug-blind
benchmark; sanity checks only, no network training.)

UnDetect = f(x) + a g(d) + b h_l(d): the control profile x enters only the drug-free backbone, and the compound term
knows the cell line only as an identity. Rhaister-O instead fits drug-specific slopes on the DMSO baseline, and the
course work weighted reference lines by DMSO similarity, w = exp(-d / (h s)) with d = 1 - Pearson of reference-centred
baseline profiles and s = the median line-line distance (`run_context_slopes.py`). Three questions:

Part 1, descriptive (all samples, no split): E_i = plate-centred response (y_i minus the mean of the other samples on
  its control row, as in `analyze_sdst_compounds.py`). For two samples of the same compound in different lines, does
  Pearson(E_i, E_j) rise as the two lines' control centroids get closer? Null per pair: E_i against a random sample of a
  different compound in line l_j. Line centroids use control profiles only (model inputs), never responses.
Part 2, oracle headroom (v3 test predictions; uses test responses, so an upper bound, not a model): predict test
  sample i of compound d by the backbone plus the backbone residuals of d's *other* test samples (other lines):
  uniform mean (perfect context-free compound representation), line-similarity weighted mean (the course kernel), k
  nearest lines, and per-gene leave-one-out ridge slopes on the sample's own control (Rhaister-style, n_d >= 8). If the
  weighted mean beats the uniform mean, a compound x context term has headroom; the gap between UnDetect and the
  uniform oracle is the headroom left for the compound representation itself. h and k are cross-fitted: chosen on
  fold (f+1) % 5's test predictions (= fold f's validation compounds), scored on fold f.
Part 3, backbone: drug-free alternatives fitted on train rows, chosen on valid, scored on test: full and diagonal
  (per-gene, Rhaister-O's form) ridge from control to delta, against the v3 MLP; and a context ceiling, the
  leave-self-out mean of every sample (all folds) sharing the test sample's control row.
Metrics as in `analyze_sdst_centred.py`: PCC_DEG = mean per-sample Pearson over 978 genes; PCC_plate = centred on
m = mean train delta of the sample's control group (test samples whose control row occurs in train).
Writes results/sdst_hybrid/context_sanity/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
OUT = RES / "context_sanity"
HS = (0.1, 0.25, 0.5, 1.0, 2.0)   # line-kernel bandwidths (course default 0.5)
KS = (1, 3, 5, 10)                # nearest other lines
SLOPE_LAMS = (1.0, 3.0, 10.0, 30.0)  # ridge on per-gene slopes, in units of the compound's sample count
SLOPE_MIN_N = 8
RIDGE_LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)  # x mean per-feature sum of squares
UNDETECT = ("nnk_multi_norm", "pool_multi_norm", "krr_multi_norm")
SEED = 0


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def zrows(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(1, keepdims=True)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def boot_diff(d: np.ndarray, drug: np.ndarray, rng, n: int = 2000) -> tuple[float, float, float]:
    """Mean paired difference with a bootstrap over compounds (cells within a compound are not independent)."""
    u, gi = np.unique(drug, return_inverse=True); nd = np.bincount(gi).astype(float); s = np.bincount(gi, weights=d)
    W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=n); b = (W @ s) / (W @ nd)
    return float(d.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def oracles(r: np.ndarray, line: np.ndarray, X: np.ndarray, drug: np.ndarray, Dl: np.ndarray, s: float) -> dict:
    """Leave-one-out predictions of each row's residual from the other rows of the same compound."""
    n_rows = len(r); out = {"uniform": np.full(r.shape, np.nan)}
    for h in HS:
        out[f"w{h}"] = np.full(r.shape, np.nan)
    for k in KS:
        out[f"k{k}"] = np.full(r.shape, np.nan)
    for lam in SLOPE_LAMS:
        out[f"slope{lam}"] = np.full(r.shape, np.nan)
    nother = np.zeros(n_rows, int)
    for d in np.unique(drug):
        idx = np.flatnonzero(drug == d); n = len(idx)
        if n < 2:
            continue
        nother[idx] = n - 1
        R = r[idx]; tot = R.sum(0)
        out["uniform"][idx] = (tot[None, :] - R) / (n - 1)
        dist = Dl[np.ix_(line[idx], line[idx])]
        for h in HS:
            W = np.exp(-dist / (h * s)); np.fill_diagonal(W, 0.0)
            out[f"w{h}"][idx] = (W @ R) / np.maximum(W.sum(1, keepdims=True), 1e-300)
        dd = dist.copy(); np.fill_diagonal(dd, np.inf); order = np.argsort(dd, 1)
        for k in KS:
            kk = min(k, n - 1); nb = order[:, :kk]
            out[f"k{k}"][idx] = R[nb].mean(1)
        if n >= SLOPE_MIN_N:
            # per-gene ridge of residual on the sample's own (globally standardised) control, leave one out
            x = X[idx]; Sx, Sy, Sxx, Sxy = x.sum(0), R.sum(0), (x * x).sum(0), (x * R).sum(0)
            m = n - 1
            sx, sy = Sx[None] - x, Sy[None] - R
            sxx, sxy = Sxx[None] - x * x, Sxy[None] - x * R
            xb, yb = sx / m, sy / m
            cxx, cxy = sxx - m * xb * xb, sxy - m * xb * yb
            for lam in SLOPE_LAMS:
                b = cxy / (cxx + lam * m)
                out[f"slope{lam}"][idx] = yb + b * (x - xb)
    out["_nother"] = nother
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, pidx, cell = z["ctl"].astype(np.float64), z["delta"].astype(np.float64), z["group"], z["pert_idx"], z["cell_idx"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in z["ctl"]]), return_inverse=True)[1]
    nk = key.max() + 1
    res = {"hs": HS, "ks": KS, "slope_lams": SLOPE_LAMS, "seed": SEED}

    # line centroids from distinct control rows (inputs only), centred over lines; course distance and bandwidth
    first = np.unique(key, return_index=True)[1]; kline = cell[first]
    nl = cell.max() + 1; cnt = np.bincount(kline, minlength=nl)
    Lc = np.zeros((nl, ctl.shape[1])); np.add.at(Lc, kline, ctl[first]); Lc /= np.maximum(cnt, 1)[:, None]
    have = cnt > 0; Lc -= Lc[have].mean(0)
    Zl = zrows(Lc); Dl = 1.0 - Zl @ Zl.T
    s = float(np.median(Dl[np.ix_(have, have)][np.triu_indices(have.sum(), 1)]))
    res["line_distance_median_s"] = s
    print(f"{have.sum()} lines, {nk} control rows; median line distance s = {s:.4f}", flush=True)

    # ---------------- Part 1: does a compound respond more alike in lines with similar controls?
    ng = np.bincount(key); S = np.zeros((nk, Y.shape[1])); np.add.at(S, key, Y)
    ok = np.flatnonzero(ng[key] >= 2)
    E = np.zeros_like(Y); E[ok] = Y[ok] - (S[key[ok]] - Y[ok]) / (ng[key[ok]] - 1)[:, None]
    Ez = zrows(E[ok])
    dok, lok = pidx[ok], cell[ok]
    byline = {l: np.flatnonzero(lok == l) for l in np.unique(lok)}
    rows = []
    order = np.argsort(dok, kind="stable"); bounds = np.flatnonzero(np.r_[True, dok[order][1:] != dok[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        if len(idx) < 2:
            continue
        C = Ez[idx] @ Ez[idx].T; iu = np.triu_indices(len(idx), 1); li, lj = lok[idx][iu[0]], lok[idx][iu[1]]
        keep = li != lj
        if not keep.any():
            continue
        pi, pj = idx[iu[0]][keep], idx[iu[1]][keep]
        null = np.array([rng.choice(byline[l]) for l in lok[pj]])
        bad = dok[null] == dok[pi]
        null[bad] = np.array([rng.choice(byline[l]) for l in lok[pj][bad]])  # one redraw; rare
        rows.append(pd.DataFrame({"drug": dok[pi], "dist": Dl[lok[pi], lok[pj]], "resp": C[iu][keep],
                                  "null": (Ez[pi] * Ez[null]).sum(1), "null_same_drug": dok[null] == dok[pi]}))
    P1 = pd.concat(rows, ignore_index=True); P1 = P1[~P1.null_same_drug]
    P1["dbin"] = pd.qcut(P1.dist, 10)
    t1 = P1.groupby("dbin", observed=True).agg(pairs=("resp", "size"), dist_mid=("dist", "median"),
                                               same_compound=("resp", "mean"), other_compound=("null", "mean")).reset_index()
    t1["excess"] = t1.same_compound - t1.other_compound
    print("\nPart 1: same-compound response similarity by line-control distance decile")
    print(t1.drop(columns="dbin").round(4).to_string(index=False), flush=True)
    # within-compound: does the closer line pair respond more alike? (removes compound-level confounds)
    wc = []
    for d, grp in P1.groupby("drug"):
        if len(grp) >= 10 and grp.dist.nunique() > 2:
            wc.append((grp.dist.rank().corr(grp.resp.rank()), grp.dist.rank().corr(grp.null.rank())))
    wc = np.array(wc)
    res["part1"] = {"pairs": int(len(P1)), "deciles": t1.astype({"dbin": str}).to_dict("records"),
                    "within_compound_spearman_dist_vs_resp_mean": float(np.nanmean(wc[:, 0])),
                    "within_compound_frac_negative": float(np.nanmean(wc[:, 0] < 0)),
                    "within_compound_spearman_dist_vs_null_mean": float(np.nanmean(wc[:, 1])),
                    "within_compound_n": int(len(wc)),
                    "spearman_dist_resp_all_pairs": float(P1.dist.rank().corr(P1.resp.rank()))}
    print(f"within-compound Spearman(distance, response similarity): mean {np.nanmean(wc[:, 0]):.4f} "
          f"(null {np.nanmean(wc[:, 1]):.4f}; {len(wc)} compounds with >= 10 pairs; "
          f"{np.nanmean(wc[:, 0] < 0):.2f} negative)", flush=True)
    del E, Ez

    # ---------------- Part 2: oracle headroom on v3 test predictions
    Xz = (ctl - ctl.mean(0)) / np.maximum(ctl.std(0), 1e-9)
    fold_scores = {}
    for f in range(5):
        Pz = np.load(RES / "v3" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        te = g == f; tr = (g != f) & (g != (f + 1) % 5)
        Yt = Y[te]; assert np.allclose(Pz["y"].astype(np.float64), Yt, atol=1e-2)
        B = Pz["mlp"].astype(np.float64); r = Yt - B
        O = oracles(r, cell[te], Xz[te], pidx[te], Dl, s); nother = O.pop("_nother")
        Sm = np.zeros((nk, Y.shape[1])); n_tr = np.bincount(key[tr], minlength=nk); np.add.at(Sm, key[tr], Y[tr])
        m = Sm[key[te]] / np.maximum(n_tr[key[te]], 1)[:, None]; sh = n_tr[key[te]] > 0
        preds = {"mlp": B, **{a: Pz[a].astype(np.float64) for a in UNDETECT}}
        for name, o in O.items():
            preds[f"or_{name}"] = B + np.nan_to_num(o)
        sc = {}
        for a, Pa in preds.items():
            sc[a] = {"deg": pcc(Yt, Pa), "plate": np.where(sh, pcc(Yt - m, Pa - m), np.nan)}
        fold_scores[f] = {"sc": sc, "nother": nother, "sh": sh, "drug": pidx[te]}
        print(f"fold {f}: {te.sum()} test samples, {(nother >= 1).sum()} with >= 1 other line, "
              f"{(nother >= SLOPE_MIN_N - 1).sum()} with n_d >= {SLOPE_MIN_N}", flush=True)
    # cross-fitted bandwidth / k / slope lambda: chosen on fold (f+1)%5's PCC_plate over rows with >= 1 other line
    fams = {"w": [f"or_w{h}" for h in HS], "k": [f"or_k{k}" for k in KS], "slope": [f"or_slope{l}" for l in SLOPE_LAMS]}
    tab, con = [], []
    for f in range(5):
        F = fold_scores[f]; V = fold_scores[(f + 1) % 5]
        base = F["nother"] >= 1; base_v = V["nother"] >= 1
        slope_rows = F["nother"] >= SLOPE_MIN_N - 1; slope_rows_v = V["nother"] >= SLOPE_MIN_N - 1
        pick = {}
        for fam, arms in fams.items():
            rv = slope_rows_v if fam == "slope" else base_v
            pick[fam] = max(arms, key=lambda a: np.nanmean(V["sc"][a]["plate"][rv & V["sh"]]))
        for subset, rows_ in (("n_d>=2", base), (f"n_d>={SLOPE_MIN_N}", slope_rows)):
            for a, v in F["sc"].items():
                if subset == "n_d>=2" and a.startswith("or_slope"):
                    continue  # undefined below n_d = 8 (falls back to the backbone)
                tab.append({"fold": f, "subset": subset, "arm": a, "n": int(rows_.sum()),
                            "pcc_deg": float(v["deg"][rows_].mean()), "pcc_plate": float(np.nanmean(v["plate"][rows_ & F["sh"]]))})
            for fam, a in pick.items():
                if subset == "n_d>=2" and fam == "slope":
                    continue
                v = F["sc"][a]
                tab.append({"fold": f, "subset": subset, "arm": f"or_{fam}*", "chosen": a, "n": int(rows_.sum()),
                            "pcc_deg": float(v["deg"][rows_].mean()), "pcc_plate": float(np.nanmean(v["plate"][rows_ & F["sh"]]))})
            contrasts = [("or_uniform", "nnk_multi_norm"), ("nnk_multi_norm", "mlp"), (pick["w"], "or_uniform"),
                         (pick["k"], "or_uniform")]
            if subset != "n_d>=2":
                contrasts.append((pick["slope"], "or_uniform"))
            for a, b in contrasts:
                for met in ("deg", "plate"):
                    rr = rows_ & (F["sh"] if met == "plate" else True)
                    mu, lo, hi = boot_diff(F["sc"][a][met][rr] - F["sc"][b][met][rr], F["drug"][rr], rng)
                    con.append({"fold": f, "subset": subset, "metric": met, "a": a, "b": b, "diff": mu, "lo": lo, "hi": hi})
        # by number of other lines (uniform oracle and course-kernel gain)
        for lo_, hi_ in ((1, 1), (2, 4), (5, 9), (10, 10**6)):
            rr = (F["nother"] >= lo_) & (F["nother"] <= hi_) & F["sh"]
            if rr.any():
                tab.append({"fold": f, "subset": f"others {lo_}-{hi_}", "arm": "or_uniform", "n": int(rr.sum()),
                            "pcc_plate": float(np.nanmean(F["sc"]["or_uniform"]["plate"][rr])),
                            "w_minus_uniform": float(np.nanmean(F["sc"][pick["w"]]["plate"][rr] - F["sc"]["or_uniform"]["plate"][rr])),
                            "undetect": float(np.nanmean(F["sc"]["nnk_multi_norm"]["plate"][rr])),
                            "mlp": float(np.nanmean(F["sc"]["mlp"]["plate"][rr]))})
    T2 = pd.DataFrame(tab); C2 = pd.DataFrame(con)
    pd.set_option("display.width", 220)
    for subset in ("n_d>=2", f"n_d>={SLOPE_MIN_N}"):
        t = T2[T2.subset == subset]
        print(f"\nPart 2 ({subset}): PCC_plate (shared rows) and PCC_DEG, 5-fold mean")
        print(t.groupby("arm")[["n", "pcc_plate", "pcc_deg"]].mean().sort_values("pcc_plate").round(4).to_string())
        print(T2[(T2.subset == subset) & T2.arm.str.endswith("*")][["fold", "arm", "chosen"]].to_string(index=False))
    print("\nPart 2 contrasts (bootstrap over compounds):")
    print(C2.groupby(["subset", "metric", "a", "b"]).agg(diff=("diff", "mean"), folds_pos=("diff", lambda x: int((x > 0).sum())),
                                                         ci_pos=("lo", lambda x: int((x > 0).sum())),
                                                         ci_neg=("hi", lambda x: int((x < 0).sum()))).round(4).to_string())
    tb = T2[T2.subset.str.startswith("others")]
    print("\nPart 2 by number of other lines of the compound (PCC_plate, shared rows):")
    print(tb.groupby("subset")[["n", "mlp", "undetect", "pcc_plate", "w_minus_uniform"]].mean().round(4).to_string())
    T2.to_csv(OUT / "part2_scores.csv", index=False); C2.to_csv(OUT / "part2_contrasts.csv", index=False)

    # ---------------- Part 3: drug-free backbones
    ng_all = np.bincount(key, minlength=nk); S_all = np.zeros((nk, Y.shape[1])); np.add.at(S_all, key, Y)
    t3 = []
    for f in range(5):
        te = g == f; va = g == (f + 1) % 5; tr = ~te & ~va
        xm, ym = ctl[tr].mean(0), Y[tr].mean(0)
        Xtr, Ytr = ctl[tr] - xm, Y[tr] - ym
        G, XY = Xtr.T @ Xtr, Xtr.T @ Ytr; sxx, sxy = (Xtr * Xtr).sum(0), (Xtr * Ytr).sum(0)
        scale = np.trace(G) / G.shape[0]
        best = {}
        for name in ("ridge_full", "ridge_diag"):
            cand = []
            for lam in RIDGE_LAMS:
                if name == "ridge_full":
                    W = np.linalg.solve(G + lam * scale * np.eye(G.shape[0]), XY)
                    fit = lambda X, W=W: (X - xm) @ W + ym
                else:
                    b = sxy / (sxx + lam * scale)
                    fit = lambda X, b=b: (X - xm) * b + ym
                cand.append((float(pcc(Y[va], fit(ctl[va])).mean()), lam, fit))
            v, lam, fit = max(cand, key=lambda c: c[0]); best[name] = (v, lam, fit)
        Pz = np.load(RES / "v3" / f"fold_{f}" / "pred_test.npz", allow_pickle=False); B = Pz["mlp"].astype(np.float64)
        Sm = np.zeros((nk, Y.shape[1])); n_tr = np.bincount(key[tr], minlength=nk); np.add.at(Sm, key[tr], Y[tr])
        m = Sm[key[te]] / np.maximum(n_tr[key[te]], 1)[:, None]; sh = n_tr[key[te]] > 0
        kt = key[te]; loso_ok = ng_all[kt] >= 2
        ctx = (S_all[kt] - Y[te]) / np.maximum(ng_all[kt] - 1, 1)[:, None]
        row = {"fold": f, "mlp": float(pcc(Y[te], B).mean()), "mlp_plate": float(pcc(Y[te][sh] - m[sh], B[sh] - m[sh]).mean())}
        for name, (v, lam, fit) in best.items():
            Pt = fit(ctl[te])
            row.update({name: float(pcc(Y[te], Pt).mean()), f"{name}_lam": lam, f"{name}_valid": v,
                        f"{name}_plate": float(pcc(Y[te][sh] - m[sh], Pt[sh] - m[sh]).mean())})
        row.update({"ctx_ceiling_rows": int(loso_ok.sum()), "ctx_ceiling": float(pcc(Y[te][loso_ok], ctx[loso_ok]).mean()),
                    "mlp_same_rows": float(pcc(Y[te][loso_ok], B[loso_ok]).mean())})
        t3.append(row); print(f"Part 3 fold {f}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    T3 = pd.DataFrame(t3)
    print("\nPart 3: drug-free backbones, test PCC_DEG / PCC_plate (5-fold mean)")
    print(T3.drop(columns="fold").mean().round(4).to_string())
    T3.to_csv(OUT / "part3_backbones.csv", index=False)
    res["part2_mean"] = T2[~T2.subset.str.startswith("others")].groupby(["subset", "arm"])[["pcc_plate", "pcc_deg"]].mean().reset_index().to_dict("records")
    res["part3_mean"] = T3.drop(columns="fold").mean().to_dict()
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
