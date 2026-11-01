#!/usr/bin/env python3
"""View ablation of UnDetect on the drug-blind benchmark, on the final backbone (`cache_v1`; user 2026-11-01).

Per fold and view set: the transfer terms exactly as `cache_undetect.py` (KTA weights over the set's views, normalised
fused kernel, leave-one-out terms on training rows), on the cached backbone B instead of a new one; then the
interaction as `stage2_context.py` arm ctx_pc (c in CS, K in KS, chosen on valid PCC_DEG in the same order and with the
same strict comparison; c = inf returns the transfer terms). View sets as `run_sdst_hybrid.py --viewsets`: multi_norm
(all four views; must reproduce stage2_v2), only_<v>, minus_<v> and perm_<v> (v's compound labels permuted among the
compounds that have v, seed --perm-seed). Stores per-sample test scores (PCC_DEG on all test rows, PCC_plate on rows
with a training plate-mate, plate mean as `finalize_sdst.py`), not predictions.
Writes results/sdst_hybrid/<tag>/fold_<f>/{scores.npz, summary.json}. `--summarize` pools the folds into
<tag>/{table.csv, summary.json}: five-fold means and paired contrasts vs multi_norm (same arm) with a 2,000-replicate
bootstrap over test compounds per fold.
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
from run_sdst_hybrid import D, OUT, VIEWS, drug_kernels, permute_view, viewset  # noqa: E402
from run_undetect_v2 import T  # noqa: E402
from stage2_context import CS, KS, pcc  # noqa: E402

SETS = ["multi_norm"] + [f"{k}_{v}" for k in ("only", "minus", "perm") for v in VIEWS]
ARMS = ("transfer", "undetect")


def context_pcs(ctl: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """z of stage2_context.py: scores on the top max(KS) PCs of the distinct training control rows, standardised."""
    mu = ctl[tr].mean(0); Xc = ctl[tr] - mu
    _, _, Vt = np.linalg.svd(Xc[np.unique(Xc.round(5), axis=0, return_index=True)[1]], full_matrices=False)
    Z = (ctl - mu) @ Vt[:max(KS)].T
    return Z / (Z[tr].std(0) + 1e-9)


def interaction(Y, B, U, Z, tr, va, te) -> tuple[np.ndarray, dict]:
    """ctx_pc: per gene, ridge of (r - u) on u * (1, z_1..z_K) over training rows. Returns test predictions + choice."""
    R = Y - B; P0 = B + U; Ut = U[tr]; Tt = R[tr] - Ut; scale = float((Ut ** 2).sum(0).mean())
    sc = lambda P: float(pcc(Y[va], P).mean())
    sys_k = {}
    for k in KS:
        F = np.hstack([np.ones((tr.sum(), 1)), Z[tr, :k]]); p = k + 1
        F2 = (F[:, :, None] * F[:, None, :]).reshape(len(F), p * p)
        sys_k[k] = (((Ut * Ut).T @ F2).reshape(-1, p, p), (Ut * Tt).T @ F)
    Fz = {s: np.hstack([np.ones((m.sum(), 1)), Z[m]]) for s, m in (("va", va), ("te", te))}
    best = None
    for cs in CS:
        for k in (KS if np.isfinite(cs) else (None,)):
            if k is None:
                v, W = sc(P0[va]), None
            else:
                Gm, b = sys_k[k]
                W = np.linalg.solve(Gm + cs * scale * np.eye(k + 1)[None], b[:, :, None])[:, :, 0].T
                v = sc(P0[va] + U[va] * (Fz["va"][:, :k + 1] @ W))
            if best is None or v > best[0]:
                best = (v, cs, k, W)
    v, cs, k, W = best
    Pte = P0[te] if W is None else P0[te] + U[te] * (Fz["te"][:, :k + 1] @ W)
    return Pte, {"valid": v, "c": cs, "K": k, "valid_transfer": sc(P0[va])}


def run(args) -> None:
    t0 = time.time(); log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    z = np.load(D / "sdst.npz", allow_pickle=False)
    A = {k: z[k] for k in ("ctl", "delta", "pert_idx", "cell_idx", "group")}
    A["delta"] = A["delta"].astype(np.float64)
    table = pd.read_csv(D / "drugs.tsv", sep="\t", keep_default_na=False)
    rowof = {d: i for i, d in enumerate(table.pert_idx)}
    A["drow"] = np.array([rowof[d] for d in A["pert_idx"]])
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in A["ctl"]]), return_inverse=True)[1]
    Knp, feats = drug_kernels(table)
    for v in sorted({s.split("_", 1)[1] for s in args.sets if s.startswith("perm_")}):
        permute_view(Knp, feats["has"], v, args.perm_seed)
    ctl, Y, g = A["ctl"].astype(np.float64), A["delta"], A["group"]
    log("data and kernels ready")
    for f in args.folds:
        te = g == f; va = g == (f + 1) % 5; tr = ~te & ~va
        B = np.load(OUT / args.backbone / f"fold_{f}" / "cache.npz")["B"].astype(np.float64)
        n = np.bincount(key[tr], minlength=key.max() + 1); Sm = np.zeros((key.max() + 1, Y.shape[1]))
        np.add.at(Sm, key[tr], Y[tr]); m = (Sm / np.maximum(n, 1)[:, None])[key[te]]; sh = n[key[te]] > 0
        Yt = Y[te]; Z = context_pcs(ctl, tr)
        score = lambda P: {"deg": pcc(Yt, P), "plate": pcc(Yt[sh] - m[sh], P[sh] - m[sh])}
        out = {f"none|backbone|{k}": v for k, v in score(B[te]).items()}
        summ = {"fold": f, "backbone": args.backbone, "perm_seed": args.perm_seed, "sets": {}}
        for s in args.sets:
            views, norm = viewset(s); assert norm, s
            res, _, G, H = run_fold(f, A, [T(Knp[v]) for v in views], Knp, args, views=views, B=B, save=False)
            U = res["g"] * G + res["h"] * H
            Pi, ch = interaction(Y, B, U, Z, tr, va, te)
            for arm, P in (("transfer", B[te] + U[te]), ("undetect", Pi)):
                out.update({f"{s}|{arm}|{k}": v for k, v in score(P).items()})
            summ["sets"][s] = {k: res[k] for k in ("lam", "a", "g", "h", "pool_valid", "weights_kta")} | {"interaction": ch} | {
                "test": {arm: {k: float(out[f"{s}|{arm}|{k}"].mean()) for k in ("deg", "plate")} for arm in ARMS}}
            log(f"fold {f} {s}: " + json.dumps(summ["sets"][s]["test"]) + f" c {ch['c']} K {ch['K']}")
        o = OUT / args.tag / f"fold_{f}"; o.mkdir(parents=True, exist_ok=True)
        np.savez(o / "scores.npz", d=A["drow"][te], sh=sh, **{k: v.astype(np.float32) for k, v in out.items()})
        (o / "summary.json").write_text(json.dumps(summ, indent=2, default=float) + "\n")


def summarize(args) -> None:
    rng = np.random.default_rng(0); rows, cons = [], []
    for f in range(5):
        S = np.load(OUT / args.tag / f"fold_{f}" / "scores.npz", allow_pickle=False)
        d, sh = S["d"], S["sh"]; keys = [k for k in S.files if "|" in k]
        for k in keys:
            s, arm, mt = k.split("|"); rows.append({"fold": f, "set": s, "arm": arm, "metric": mt, "score": float(S[k].mean())})
        for mt in ("deg", "plate"):
            drugs, gi = np.unique(d if mt == "deg" else d[sh], return_inverse=True); nd = np.bincount(gi).astype(float)
            W = rng.multinomial(len(drugs), np.full(len(drugs), 1 / len(drugs)), size=2000)
            for k in keys:
                s, arm, m_ = k.split("|")
                if m_ != mt or s == "multi_norm":
                    continue
                for ref_arm in ((arm,) if arm in ARMS else ARMS):
                    dd = S[k].astype(np.float64) - S[f"multi_norm|{ref_arm}|{mt}"]
                    sm = np.bincount(gi, weights=dd); boot = (W @ sm) / (W @ nd)
                    cons.append({"fold": f, "set": s, "arm": arm, "ref_arm": ref_arm, "metric": mt, "diff": float(dd.mean()),
                                 "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975))})
    Tm = pd.DataFrame(rows).pivot_table(index=["set", "arm"], columns="metric", values="score", aggfunc="mean")
    C = pd.DataFrame(cons).groupby(["set", "arm", "ref_arm", "metric"]).agg(
        diff=("diff", "mean"), folds_neg=("diff", lambda v: int((v < 0).sum())), ci_neg=("hi", lambda v: int((v < 0).sum())),
        ci_pos=("lo", lambda v: int((v > 0).sum()))).reset_index()
    Tm.to_csv(OUT / args.tag / "means.csv"); C.to_csv(OUT / args.tag / "contrasts.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print("five-fold means:\n" + Tm.round(4).to_string())
    print("\npaired vs multi_norm (folds < 0; folds with CI < 0; folds with CI > 0):\n" + C.round(4).to_string(index=False))
    (OUT / args.tag / "summary.json").write_text(json.dumps(
        {"means": {f"{s}|{a}": r for (s, a), r in Tm.to_dict(orient="index").items()},
         "contrasts": C.to_dict(orient="records")}, indent=2, default=float) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--sets", nargs="+", default=SETS)
    p.add_argument("--backbone", default="cache_v1")
    p.add_argument("--perm-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=1729, help="unused (no backbone is trained); kept for run_fold")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--tag", default="views_final")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    print(vars(args), flush=True)
    summarize(args) if args.summarize else run(args)


if __name__ == "__main__":
    main()
