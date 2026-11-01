#!/usr/bin/env python3
"""Cross-line multimodal kernel model for unseen drugs on L1000 (exploratory sanity check, 2026-10-28).

The MKL model (run_l1000_mkl.py) fits one KRR per cell line, so a training compound measured only
in other lines never informs line L. Here a drug-level consensus model is added:
  c(d)   = mean over the lines that measured d of (Y[d, l] - mu_l)      (training compounds only)
  g      = KRR of c on a drug kernel over ALL training compounds of the fold
  pred   = mu_L + a_L * g(d*) + b_L * f_L(d*)
where f_L is the unchanged per-line KRR and (a_L, b_L) are chosen per (line, fold) by the same inner
group CV that selects lambda. Kernels are built exactly as in run_l1000_mkl.py (T1 = chem; T3 = chem,
tgt, moa, deg, ppi1, morph with centred-alignment weights), so the per-line arms krr_chem and
mkl_T3 must reproduce that script's per_condition.parquet bit for bit when its outputs exist.
Test compounds are held out of every line in a fold (compound-level folds), so g never sees them.
--gate K (added 2026-11-01, report/11_06_record.md): the drug × cell-line gate of the drug-blind benchmark
(`stage2_context.py`) on top of pool_K. With u = a_L g + b_L (f_L - mu_L) the chosen compound term,
  pred = mu_L + u * (1 + delta),  delta_j = sum_i u_ij t_ij / (sum_i u_ij^2 + lam),  t = y - mu_L - u,
fitted on the inner-CV predictions of the training rows (the rows that choose (a, b)), lam = c * mean_j sum_i u_ij^2,
c in GATE_CS chosen by cross-fitting over the inner folds (c = inf returns pool_K). gate_line_K: one delta per
(line, gene), the line-identity gate; gate_scale_K: one delta per gene shared by all lines (control).
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

import netnull
import run_l1000_mkl as mkl
import run_l1000_unseen_drug as run

KERNELS = {"T1": ["chem"], "T3": ["chem", "tgt", "moa", "deg", "ppi1", "morph"],
           # T3n (added 05:45): T3 with availability normalisation K_ij / sqrt(K_ii K_jj). Every view kernel has unit
           # diagonal where the drug has the view, so K_ii = total weight of the drug's views; D^-1/2 K D^-1/2 stays PSD
           # and gives a drug with fewer views the same self-similarity as a fully annotated one.
           "T3n": ["chem", "tgt", "moa", "deg", "ppi1", "morph"],
           # D (added 2026-10-28 23:10): the paper's DECANT views only (no degree / STRING views), normalised.
           "D": ["chem", "tgt", "moa", "morph"],
           # Dml / Dl1000 (added 2026-10-31, report/11_06_record.md): D with other view weights for the consensus g
           # only; the per-line f_L keeps D's per-line alignment weights, so mkl_Dml = mkl_Dl1000 = mkl_D.
           # Dml: weights, amplitude and noise by GP marginal likelihood on the fold's training compounds
           # (as run_undetect_v2.ml_weights). Dl1000: fixed weights learned on the L1000 drug-blind benchmark.
           "Dml": ["chem", "tgt", "moa", "morph"],
           "Dl1000": ["chem", "tgt", "moa", "morph"]}
NORMALISED = {"T3n", "D", "Dml", "Dl1000"}
# 5-fold mean of the marginal-likelihood weights in results/sdst_hybrid/v7_ext_cpu/fold_*/summary.json (L1000
# responses only; no Tahoe data).
G_FIXED = {"Dl1000": {"chem": 0.2650, "tgt": 0.2279, "moa": 0.2006, "morph": 0.3064}}
G_ML = {"Dml"}
ML_STEPS = 250
GRID_A = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
GRID_B = (0.0, 0.25, 0.5, 0.75, 1.0)
GATE_CS = (0.001, 0.01, 0.1, 1.0, 10.0, float("inf"))  # as stage2_context.CS


def build_kernels(d: Path, compounds: np.ndarray, targets: dict, moas: dict, seed: int):
    """chem, tgt, moa, deg, ppi1, morph exactly as run_l1000_mkl.main builds them."""
    n_c = len(compounds)
    feats = np.load(d / "drug_features.npz", allow_pickle=False)
    fpos = pd.Series(np.arange(len(feats["inchikey"])), index=feats["inchikey"].astype(str))[compounds].to_numpy()
    K = {"chem": run.tanimoto(feats["morgan2"][fpos], feats["morgan2"][fpos]).astype(np.float64)}
    net = netnull.load("string_physical400")
    gset = set(net.genes)
    mapped = [sorted(t & gset) for t in (targets[k] for k in compounds)]
    known = np.array([len(t) > 0 for t in mapped]); kn = np.where(known)[0]
    tg = sorted({g for t in mapped for g in t}); tpos = {g: i for i, g in enumerate(tg)}
    Tm = np.zeros((n_c, len(tg)), np.uint8)
    for c, t in enumerate(mapped):
        Tm[c, [tpos[g] for g in t]] = 1
    K["tgt"] = mkl.tanimoto_masked(Tm, known)
    ml = sorted({m for k in compounds for m in moas[k]}); mp = {m: i for i, m in enumerate(ml)}
    Mm = np.zeros((n_c, max(1, len(ml))), np.uint8)
    for c, k in enumerate(compounds):
        Mm[c, [mp[m] for m in moas[k]]] = 1
    K["moa"] = mkl.tanimoto_masked(Mm, Mm.any(1))
    deg = dict(zip(net.genes, net.degree()))
    F = np.zeros((n_c, 3))
    for c, t in enumerate(mapped):
        if t:
            ld = np.log1p([deg[g] for g in t]); F[c] = [ld.mean(), ld.max(), np.log1p(len(t))]
    Fz = (F - F[kn].mean(0)) / np.maximum(F[kn].std(0), 1e-9)
    D2 = ((Fz[:, None, :] - Fz[None, :, :]) ** 2).sum(-1)
    bw = np.median(D2[np.ix_(kn, kn)]) or 1.0
    K["deg"] = np.exp(-D2 / bw); K["deg"][~known] = 0; K["deg"][:, ~known] = 0
    K["ppi1"] = mkl.cosine_masked(mkl.one_hop(net, mapped), known)
    cp = np.load(d / "cell_painting_features.npz", allow_pickle=False)
    cpos = pd.Series(np.arange(len(cp["inchikey"])), index=cp["inchikey"].astype(str))
    has_morph = np.array([k in cpos.index for k in compounds])
    Fm = np.zeros((n_c, cp["morph10"].shape[1]))
    Fm[has_morph] = cp["morph10"][cpos[compounds[has_morph]].to_numpy()]
    Fm[has_morph] = (Fm[has_morph] - Fm[has_morph].mean(0)) / np.maximum(Fm[has_morph].std(0), 1e-9)
    K["morph"] = mkl.cosine_masked(Fm, has_morph)
    return K, known, has_morph


def fast_alignment(K: np.ndarray, R: np.ndarray) -> float:
    """centred kernel-target alignment <HKH, H R R'H> / (|HKH| |H R R'H|) without n x n x n products."""
    Kc = K - K.mean(0)[None, :] - K.mean(1)[:, None] + K.mean()
    Rc = R - R.mean(0)
    num = float(np.einsum("ij,ij->", Rc.T @ Kc, Rc.T))
    den = np.linalg.norm(Kc) * np.linalg.norm(Rc.T @ Rc)
    return num / den if den > 0 else 0.0


def normalise(K: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    return K / d[:, None] / d[None, :]


def combine(K: dict, views: list, align: dict, norm: bool = False) -> np.ndarray:
    w = np.array([max(align[v], 0.0) for v in views])
    if w.sum() <= 0:
        w = np.ones(len(views))
    w = w / w.sum()
    Kc = sum(wi * K[v] for wi, v in zip(w, views))
    return normalise(Kc) if norm else Kc


def ml_view_weights(K: dict, views: list, idx: np.ndarray, C: np.ndarray, nd: np.ndarray, steps: int = ML_STEPS) -> dict:
    """run_undetect_v2.ml_weights on the CPU: maximise the GP marginal likelihood of z-scored mean residuals C over
    view logits (softmax), amplitude, a constant noise and a 1/n_d noise; fused kernel diagonal-normalised."""
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    Kt = [torch.as_tensor(K[v][np.ix_(idx, idx)], dtype=torch.float64) for v in views]
    Cz = torch.as_tensor(C, dtype=torch.float64); Cz = (Cz - Cz.mean(0)) / Cz.std(0).clamp_min(1e-9); G = Cz.shape[1]
    inv_nd = 1.0 / torch.as_tensor(nd, dtype=torch.float64)
    th = torch.zeros(len(views) + 3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([th], lr=0.05); trace = []
    for it in range(steps):
        w = torch.softmax(th[:len(views)], 0)
        Kf = sum(w[i] * Kt[i] for i in range(len(views)))
        dg = torch.sqrt(torch.diagonal(Kf).clamp_min(1e-12)); Kn = Kf / dg[:, None] / dg[None, :]
        S = torch.exp(th[-3]) * Kn + torch.diag(torch.exp(th[-2]) + torch.exp(th[-1]) * inv_nd)
        L = torch.linalg.cholesky(S)
        nll = 0.5 * (Cz * torch.cholesky_solve(Cz, L)).sum() / G + torch.log(torch.diagonal(L)).sum()
        opt.zero_grad(); nll.backward(); opt.step()
        if it % 50 == 0 or it == steps - 1:
            trace.append((it, round(float(nll.detach()), 3)))
    th = th.detach(); w = torch.softmax(th[:len(views)], 0).numpy()
    return {"w": dict(zip(views, map(float, w))), "amp": float(torch.exp(th[-3])), "noise_const": float(torch.exp(th[-2])),
            "noise_per_sample": float(torch.exp(th[-1])), "nll_trace": trace}


def krr_inner(Kc: np.ndarray, tr: np.ndarray, Y: np.ndarray, splits, lam: float) -> np.ndarray:
    """Inner-CV predictions m + KRR at one lambda for every inner-validation row of tr (NaN elsewhere)."""
    out = np.full((len(tr), Y.shape[1]), np.nan)
    for inner_val, inner_train in splits:
        va = np.where(inner_val)[0]; nb = np.where(inner_train)[0]
        if len(va) == 0 or len(nb) < 5:
            continue
        m = Y[tr[nb]].mean(0)
        p = run.krr_fit_predict(Kc[np.ix_(tr[nb], tr[nb])] + 1e-6 * np.eye(len(nb)), Kc[np.ix_(tr[va], tr[nb])],
                                Y[tr[nb]] - m, [lam])[lam]
        out[va] = m + p
    return out


def gate_delta(U: np.ndarray, T: np.ndarray, c: float) -> np.ndarray:
    """Per-gene ridge of T on U (one coefficient per gene), lam = c * mean_j sum_i U_ij^2; zero at c = inf."""
    if np.isinf(c):
        return np.zeros(U.shape[1])
    uu = (U * U).sum(0)
    return (U * T).sum(0) / (uu + c * uu.mean())


def gate_select(U: np.ndarray, T: np.ndarray, M: np.ndarray, Y: np.ndarray, fid: np.ndarray) -> tuple[float, np.ndarray]:
    """c by cross-fitting over the inner folds fid (mean row Pearson of M + U (1 + delta) vs Y; ties keep c = inf);
    returns c and delta refitted on all rows."""
    best_c, best_s = float("inf"), float(run.rowwise_pearson(M + U, Y).mean())
    for c in GATE_CS[:-1]:
        P = M + U
        for k in np.unique(fid):
            fit = fid != k
            P[~fit] = M[~fit] + U[~fit] * (1 + gate_delta(U[fit], T[fit], c))
        s = float(run.rowwise_pearson(P, Y).mean())
        if s > best_s:
            best_c, best_s = c, s
    return best_c, gate_delta(U, T, best_c)


def main(args: argparse.Namespace) -> None:
    started = time.time()
    print(f"host {platform.node()} python {sys.version.split()[0]} numpy {np.__version__} pandas {pd.__version__} "
          f"slice {args.slice} split {args.split} seed {args.seed}", flush=True)
    d = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/l1000_unseen_drug")
    sl = np.load(d / f"slice_{args.slice}.npz", allow_pickle=False)
    keys = sl["inchikey"].astype(str); line_rows = sl["line"].astype(str); Y_all = sl["Y"].astype(np.float64)
    active_all = sl["active"].astype(bool) if "active" in sl.files else sl["tas"] >= run.ACTIVE_TAS
    compounds = np.array(sorted(set(keys))); cidx = {k: i for i, k in enumerate(compounds)}
    row_c = np.array([cidx[k] for k in keys]); n_c = len(compounds)
    table = pd.read_parquet(d / "drug_table.parquet")
    table = table[table.source.eq(args.source) & table.inchikey.isin(compounds)]
    first = table.groupby("inchikey").first().reindex(compounds)
    ik14 = first.ik14.astype(str).to_numpy(); scaf = first.generic_scaffold.fillna("").astype(str).to_numpy()
    targets = {k: set() for k in compounds}; moas = {k: set() for k in compounds}
    for r in table.itertuples():
        if r.targets:
            targets[r.inchikey].update(r.targets.split(";"))
        if r.moa:
            moas[r.inchikey].update(r.moa.split(";"))
    test_cf, train_cf, inner_cf, group_c = run.make_folds(args.split, args.seed, ik14, scaf, row_c,
                                                          sl["plates"].astype(str), n_c)
    K, known, has_morph = build_kernels(d, compounds, targets, moas, args.seed)
    lines = sorted(set(line_rows))
    print(f"{len(Y_all):,} rows, {n_c:,} compounds, {len(lines)} lines; KNOWN {known.sum()}, MORPH {has_morph.sum()} "
          f"({time.time() - started:.0f}s)", flush=True)

    records, log = [], []
    for f in (args.folds or range(run.N_OUTER)):
        # ---- drug-level consensus model g over all training compounds of the fold
        mu = {L: Y_all[(line_rows == L) & train_cf[row_c, f]].mean(0) for L in lines}
        resid = Y_all - np.stack([mu[L] for L in line_rows])
        ctr = np.where(train_cf[:, f])[0]
        tr_rows = train_cf[row_c, f]
        csum = np.zeros((n_c, Y_all.shape[1])); ccnt = np.zeros(n_c)
        np.add.at(csum, row_c[tr_rows], resid[tr_rows]); np.add.at(ccnt, row_c[tr_rows], 1)
        ctr = ctr[ccnt[ctr] > 0]
        Cc = csum[ctr] / ccnt[ctr][:, None]
        act_c = np.zeros(n_c, bool); act_c[row_c[active_all]] = True
        g_splits = [(iv[ctr], it[ctr]) for iv, it in inner_cf[f]]
        te_c = np.where(test_cf[:, f])[0]
        g_pred, g_inner = {}, {}
        for kname, views in KERNELS.items():
            if kname not in args.kernels:
                continue
            align = {v: fast_alignment(K[v][np.ix_(ctr, ctr)], Cc) for v in views}
            extra = {}
            if kname in G_ML:
                fit = ml_view_weights(K, views, ctr, Cc, ccnt[ctr]); gw = fit["w"]
                extra = {"ml_amp": fit["amp"], "ml_noise_const": fit["noise_const"],
                         "ml_noise_per_sample": fit["noise_per_sample"], "ml_nll_trace": json.dumps(fit["nll_trace"])}
            else:
                gw = G_FIXED.get(kname, align)
            Kg = combine(K, views, gw, kname in NORMALISED)
            if kname in G_ML or kname in G_FIXED:
                extra.update({f"gw_{v}": float(gw[v]) for v in views})
            Kloc = Kg[np.ix_(ctr, ctr)]
            lam = run.select_krr(Kloc, np.arange(len(ctr)), Cc, act_c[ctr], g_splits)
            mc = Cc.mean(0)
            g_pred[kname] = np.zeros((n_c, Y_all.shape[1]))
            g_pred[kname][te_c] = mc + run.krr_fit_predict(Kloc + 1e-6 * np.eye(len(ctr)), Kg[np.ix_(te_c, ctr)],
                                                            Cc - mc, [lam])[lam]
            gi = np.full((n_c, Y_all.shape[1]), np.nan)
            gi[ctr] = krr_inner(Kloc, np.arange(len(ctr)), Cc, g_splits, lam)
            g_inner[kname] = gi
            log.append({"fold": f, "model": f"g_{kname}", "lambda": lam, "n_train_compounds": len(ctr),
                        **{f"w_{v}": align[v] for v in views}, **extra})
            print(f"  fold {f} g_{kname}: {len(ctr)} compounds, lambda {lam} ({time.time() - started:.0f}s)", flush=True)

        # ---- per line: the unchanged per-line KRR plus the consensus term
        pooled = {k: [] for k in args.gate}  # gate_scale: per line (inner U, T, M, Y, fid; test u, m, Y, cl, act)

        def emit(arm, L, P, Yte, m, cl_te, act_te):
            full = run.rowwise_pearson(P, Yte); res = run.rowwise_pearson(P - m, Yte - m)
            for j, c in enumerate(cl_te):
                records.append((arm, L, compounds[c], ik14[c], group_c[c], f, bool(act_te[j]), bool(known[c]),
                                bool(has_morph[c]), int(ccnt[c]), float(full[j]), float(res[j])))

        for L in lines:
            if args.lines and L not in args.lines:
                continue
            rows = np.where(line_rows == L)[0]; cl = row_c[rows]; Y = Y_all[rows]; act = active_all[rows]
            tr = np.where(train_cf[cl, f])[0]; te = np.where(test_cf[cl, f])[0]
            if len(te) == 0 or len(tr) < 20:
                continue
            m = Y[tr].mean(0); R = Y[tr] - m
            assert np.allclose(m, mu[L])
            local = {v: M[np.ix_(cl, cl)] for v, M in K.items()}
            splits = [(iv[cl][tr], it[cl][tr]) for iv, it in inner_cf[f]]
            preds = {"mean": np.tile(m, (len(te), 1))}
            for kname, views in KERNELS.items():
                if kname not in args.kernels:
                    continue
                align = {v: mkl.centered_alignment(local[v][np.ix_(tr, tr)], R) for v in views}
                w = np.array([max(align[v], 0.0) for v in views])
                if w.sum() <= 0:
                    w = np.ones(len(views))
                w = w / w.sum()
                Kc = sum(wi * local[v] for wi, v in zip(w, views))  # same order of operations as run_l1000_mkl
                if kname in NORMALISED:
                    Kc = normalise(Kc)
                lam = run.select_krr(Kc, tr, Y, act, splits)
                f_te = m + run.krr_fit_predict(Kc[np.ix_(tr, tr)] + 1e-6 * np.eye(len(tr)), Kc[np.ix_(te, tr)],
                                               R, [lam])[lam]
                preds["krr_chem" if kname == "T1" else f"mkl_{kname}"] = f_te
                # inner predictions of f_L and g for choosing (a, b); validation rows = active, both available
                f_in = krr_inner(Kc, tr, Y, splits, lam)
                g_in = g_inner[kname][cl[tr]]
                mu_in = np.full_like(f_in, np.nan)
                for inner_val, inner_train in splits:
                    mu_in[inner_val] = Y[tr[inner_train]].mean(0)
                ok = act[tr] & ~np.isnan(f_in[:, 0]) & ~np.isnan(g_in[:, 0])
                Yv, Fv, Gv, Mv = Y[tr][ok], f_in[ok] - mu_in[ok], g_in[ok], mu_in[ok]
                best = {}
                for name, grid_b in ((f"pool_{kname}", GRID_B), (f"gonly_{kname}", (0.0,))):
                    sc = {(a, b): run.rowwise_pearson(Mv + a * Gv + b * Fv, Yv).mean()
                          for a in GRID_A for b in grid_b}
                    best[name] = max(sc, key=sc.get)
                    a, b = best[name]
                    preds[name] = m + a * g_pred[kname][cl[te]] + b * (f_te - m)
                gate = {}
                if kname in args.gate:
                    fid = np.full(len(tr), -1)
                    for k, (inner_val, _) in enumerate(splits):
                        fid[inner_val] = k
                    a, b = best[f"pool_{kname}"]
                    U_in = a * Gv + b * Fv; T_in = Yv - Mv - U_in; u_te = preds[f"pool_{kname}"] - m
                    c_l, d_l = gate_select(U_in, T_in, Mv, Yv, fid[ok])
                    preds[f"gate_line_{kname}"] = m + u_te * (1 + d_l)
                    gate = {"gate_line_c": c_l, "gate_line_mean_abs_delta": float(np.abs(d_l).mean())}
                    pooled[kname].append((L, U_in, T_in, Mv, Yv, fid[ok], u_te, m, Y[te], cl[te], act[te]))
                log.append({"fold": f, "line": L, "model": kname, "lambda": lam,
                            **{f"ab_{k}": v for k, v in best.items()}, **{f"w_{v}": float(wi) for wi, v in zip(w, views)},
                            **gate})
            for arm, P in preds.items():
                emit(arm, L, P, Y[te], m, cl[te], act[te])
            print(f"  fold {f} {L} done ({time.time() - started:.0f}s)", flush=True)
        for kname, parts in pooled.items():  # gate_scale: one delta per gene over all lines of the fold
            if not parts:
                continue
            cat = [np.concatenate([p[i] for p in parts]) for i in range(1, 6)]
            c_s, d_s = gate_select(*cat)
            for L, *_, u_te, m, Yte, cl_te, act_te in parts:
                emit(f"gate_scale_{kname}", L, m + u_te * (1 + d_s), Yte, m, cl_te, act_te)
            log.append({"fold": f, "model": f"gate_scale_{kname}", "gate_scale_c": c_s,
                        "gate_scale_mean_abs_delta": float(np.abs(d_s).mean())})
            print(f"  fold {f} gate_scale_{kname}: c {c_s} ({time.time() - started:.0f}s)", flush=True)

    out = Path(args.out) / f"{args.slice}_{args.split}_seed{args.seed}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records, columns=["arm", "line", "inchikey", "ik14", "group", "fold", "active", "known",
                                           "morph", "n_train_lines", "pearson_full", "pearson_resid"])
    frame.to_parquet(out / "per_condition.parquet", index=False)
    pd.DataFrame(log).to_csv(out / "selection.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "slice": args.slice, "split": args.split, "seed": args.seed, "kernels": KERNELS,
        "grid_a": GRID_A, "grid_b": GRID_B, "script_sha256": run.sha256(Path(__file__).resolve()),
        "slice_sha256": run.sha256(d / f"slice_{args.slice}.npz"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "seconds": round(time.time() - started, 1)}, indent=2) + "\n")
    a = frame[frame.active]
    per_line = a.groupby(["arm", "line"]).pearson_full.mean().unstack()
    print(per_line.round(4).to_string(), flush=True)
    print("line-macro ACTIVE full:", per_line.mean(1).round(4).to_dict(), flush=True)
    if not (a.arm == "krr_chem").any():  # --kernels without T1 (e.g. the T3n-only runs): pair with the full run instead
        return
    base = a[a.arm == "krr_chem"].set_index(["line", "inchikey", "fold"]).pearson_full
    for arm in sorted(a.arm.unique()):
        x = a[a.arm == arm].set_index(["line", "inchikey", "fold"]).pearson_full.reindex(base.index)
        sub = a[a.arm == "krr_chem"].copy()
        mean, ci, _ = run.line_macro_boot(sub, (x - base).to_numpy(), 2000, 0)
        print(f"  {arm} - krr_chem: {mean:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slice", default="P2-24", choices=["P1-24", "P1-6", "P2-24", "T-0.05", "T-0.5", "T-5"])
    p.add_argument("--source", default="L1000", choices=["L1000", "Tahoe"])
    p.add_argument("--split", default="ik14", choices=["ik14", "scaffold", "platemap"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--lines", nargs="*", default=None)
    p.add_argument("--kernels", nargs="*", default=list(KERNELS), choices=list(KERNELS))
    p.add_argument("--folds", nargs="*", type=int, default=None)
    p.add_argument("--gate", nargs="*", default=[], choices=list(KERNELS), help="add gate_line_K / gate_scale_K on pool_K")
    p.add_argument("--out", default="/nfs-stor/yuqi.lei/Therapeutic/results/l1000_xline")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
