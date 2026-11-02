#!/usr/bin/env python3
"""Interaction (ctx_pc) on fully cross-fitted training inputs, drug-blind benchmark, `cache_v1` backbone
(report/11_06_record.md 2026-11-02 E4; answers whether the leave-one-out inputs are out-of-sample w.r.t. the view weights).

`cache_undetect.run_fold` gives the training rows leave-one-out transfer terms G, H for a FIXED fused kernel whose KTA
weights were computed from all training compounds, so a left-out compound still entered its own weights. Here the
training-row G and H are replaced by Q-fold cross-fitted values over the fold's training compounds (seed --xfit-seed):
for each part q the compound mean residuals of the other parts give the KTA weights (`xfit`; fixed at the cache weights
in `xfit_fixw`), the normalised fused kernel, the shared KRR (G) and the per-line KRR (H) at the cache lambda, which
then predict part q. Valid/test terms, lambda and (g, h) are unchanged; the interaction is `sdst_view_ablation.interaction`
(c, K on valid PCC_DEG). Arms: loo (the paper's pipeline; must reproduce views_final multi_norm), xfit, xfit_fixw; each
with transfer (= B + U, identical across arms on test) and undetect (+ interaction). The backbone B stays in-sample on
training rows. Writes results/sdst_hybrid/<tag>/fold_<f>/{scores.npz, summary.json}; --summarize: per-fold
undetect − transfer with a 2,000-replicate bootstrap over test compounds, into <tag>/{contrasts.csv, summary.json}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_undetect import run_fold  # noqa: E402
from run_sdst_hybrid import D, OUT, VIEWS, alignment, drug_kernels  # noqa: E402
from run_undetect_v2 import T  # noqa: E402
from sdst_view_ablation import context_pcs, interaction  # noqa: E402
from stage2_context import pcc  # noqa: E402

ARMS = ("loo", "xfit", "xfit_fixw")


def crossfit_terms(Knp: dict, R: np.ndarray, di: np.ndarray, cell: np.ndarray, tr: np.ndarray, lam: float,
                   w_fixed: dict | None, Q: int, seed: int) -> tuple[np.ndarray, np.ndarray, list]:
    """Cross-fitted G, H on training rows (zeros elsewhere). w_fixed None: KTA weights refitted per part."""
    tdr = np.unique(di[tr]); grp = np.random.default_rng(seed).permutation(len(tdr)) % Q
    gpos = np.full(di.max() + 1, -1); gpos[tdr] = grp
    trl = np.flatnonzero(tr); G = np.zeros_like(R); H = np.zeros_like(R); weights = []
    for q in range(Q):
        fit_r = trl[gpos[di[trl]] != q]; out_r = trl[gpos[di[trl]] == q]
        u, inv = np.unique(di[fit_r], return_inverse=True)
        C = np.zeros((len(u), R.shape[1])); np.add.at(C, inv, R[fit_r]); C /= np.bincount(inv)[:, None]
        if w_fixed is None:
            a = {v: alignment(Knp[v][np.ix_(u, u)], C) for v in VIEWS}
            w = np.array([max(a[v], 0) for v in VIEWS]); w = w / w.sum()
        else:
            w = np.array([w_fixed[v] for v in VIEWS])
        weights.append(dict(zip(VIEWS, map(float, w))))
        Kf = sum(wi * Knp[v][np.ix_(tdr, tdr)] for wi, v in zip(w, VIEWS))
        dg = np.sqrt(np.clip(np.diag(Kf), 1e-12, None)); Kf = Kf / dg[:, None] / dg[None, :]
        loc = np.searchsorted(tdr, u); out_loc = np.searchsorted(tdr, di[out_r])
        alpha = np.linalg.solve(Kf[np.ix_(loc, loc)] + lam * np.eye(len(loc)), C)
        G[out_r] = Kf[np.ix_(out_loc, loc)] @ alpha
        for c in np.unique(cell[out_r]):
            rc = fit_r[cell[fit_r] == c]; oc = out_r[cell[out_r] == c]
            if not len(rc):
                continue
            dl, li = np.unique(di[rc], return_inverse=True)
            Cl = np.zeros((len(dl), R.shape[1])); np.add.at(Cl, li, R[rc]); Cl /= np.bincount(li)[:, None]
            ll = np.searchsorted(tdr, dl)
            al = np.linalg.solve(Kf[np.ix_(ll, ll)] + lam * np.eye(len(ll)), Cl)
            H[oc] = Kf[np.ix_(np.searchsorted(tdr, di[oc]), ll)] @ al
    return G, H, weights


def run(args) -> None:
    t0 = time.time(); log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    z = np.load(D / "sdst.npz", allow_pickle=False)
    A = {k: z[k] for k in ("ctl", "delta", "pert_idx", "cell_idx", "group")}
    A["delta"] = A["delta"].astype(np.float64)
    table = pd.read_csv(D / "drugs.tsv", sep="\t", keep_default_na=False)
    rowof = {d: i for i, d in enumerate(table.pert_idx)}
    A["drow"] = np.array([rowof[d] for d in A["pert_idx"]])
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in A["ctl"]]), return_inverse=True)[1]
    Knp, _ = drug_kernels(table); Kv = [T(Knp[v]) for v in VIEWS]
    ctl, Y, g, di, cell = A["ctl"].astype(np.float64), A["delta"], A["group"], A["drow"], A["cell_idx"]
    log("data and kernels ready")
    for f in args.folds:
        te = g == f; va = g == (f + 1) % 5; tr = ~te & ~va
        B = np.load(OUT / args.backbone / f"fold_{f}" / "cache.npz")["B"].astype(np.float64)
        n = np.bincount(key[tr], minlength=key.max() + 1); Sm = np.zeros((key.max() + 1, Y.shape[1]))
        np.add.at(Sm, key[tr], Y[tr]); m = (Sm / np.maximum(n, 1)[:, None])[key[te]]; sh = n[key[te]] > 0
        Yt = Y[te]; Z = context_pcs(ctl, tr)
        score = lambda P: {"deg": pcc(Yt, P), "plate": pcc(Yt[sh] - m[sh], P[sh] - m[sh])}
        res, _, G0, H0 = run_fold(f, A, Kv, Knp, args, views=VIEWS, B=B, save=False)
        out = {f"none|backbone|{k}": v for k, v in score(B[te]).items()}
        summ = {"fold": f, "lam": res["lam"], "g": res["g"], "h": res["h"], "weights_kta": res["weights_kta"], "arms": {}}
        for arm in ARMS:
            G, H = G0.copy(), H0.copy()
            if arm != "loo":
                Gx, Hx, ws = crossfit_terms(Knp, Y - B, di, cell, tr, res["lam"],
                                            res["weights_kta"] if arm == "xfit_fixw" else None, args.q, args.xfit_seed)
                G[tr], H[tr] = Gx[tr], Hx[tr]
                summ["arms"][arm] = {"part_weights": ws, "train_corr_G_loo": float(pcc(G0[tr], Gx[tr]).mean())}
            U = res["g"] * G + res["h"] * H
            Pi, ch = interaction(Y, B, U, Z, tr, va, te)
            for a2, P in (("transfer", B[te] + U[te]), ("undetect", Pi)):
                out.update({f"{arm}|{a2}|{k}": v for k, v in score(P).items()})
            summ["arms"].setdefault(arm, {}).update({"interaction": ch, "test": {
                a2: {k: float(out[f"{arm}|{a2}|{k}"].mean()) for k in ("deg", "plate")} for a2 in ("transfer", "undetect")}})
            log(f"fold {f} {arm}: " + json.dumps(summ["arms"][arm]["test"]) + f" c {ch['c']} K {ch['K']}")
        o = OUT / args.tag / f"fold_{f}"; o.mkdir(parents=True, exist_ok=True)
        np.savez(o / "scores.npz", d=di[te], sh=sh, **{k: v.astype(np.float32) for k, v in out.items()})
        (o / "summary.json").write_text(json.dumps(summ, indent=2, default=float) + "\n")


def summarize(args) -> None:
    rng = np.random.default_rng(0); cons = []
    for f in range(5):
        S = np.load(OUT / args.tag / f"fold_{f}" / "scores.npz", allow_pickle=False); d, sh = S["d"], S["sh"]
        for mt in ("deg", "plate"):
            drugs, gi = np.unique(d if mt == "deg" else d[sh], return_inverse=True); nd = np.bincount(gi).astype(float)
            W = rng.multinomial(len(drugs), np.full(len(drugs), 1 / len(drugs)), size=2000)
            for arm in ARMS:
                for ref in ("transfer", "backbone"):
                    rk = f"{arm}|transfer|{mt}" if ref == "transfer" else f"none|backbone|{mt}"
                    dd = S[f"{arm}|undetect|{mt}"].astype(np.float64) - S[rk]
                    boot = (W @ np.bincount(gi, weights=dd)) / (W @ nd)
                    cons.append({"fold": f, "arm": arm, "vs": ref, "metric": mt, "diff": float(dd.mean()),
                                 "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)),
                                 "undetect": float(S[f"{arm}|undetect|{mt}"].mean())})
    C = pd.DataFrame(cons); C.to_csv(OUT / args.tag / "contrasts.csv", index=False)
    agg = C.groupby(["arm", "vs", "metric"]).agg(undetect=("undetect", "mean"), diff=("diff", "mean"),
                                                 folds_pos=("diff", lambda v: int((v > 0).sum())),
                                                 ci_pos=("lo", lambda v: int((v > 0).sum())),
                                                 ci_neg=("hi", lambda v: int((v < 0).sum()))).reset_index()
    pd.set_option("display.width", 200)
    print(C.round(4).to_string(index=False)); print("\nfive-fold:\n" + agg.round(4).to_string(index=False))
    (OUT / args.tag / "summary.json").write_text(json.dumps(agg.to_dict(orient="records"), indent=2, default=float) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--backbone", default="cache_v1")
    p.add_argument("--q", type=int, default=5)
    p.add_argument("--xfit-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=1729, help="unused (no backbone is trained); kept for run_fold")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--tag", default="xfit_interaction")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    print(vars(args), flush=True)
    summarize(args) if args.summarize else run(args)


if __name__ == "__main__":
    main()
