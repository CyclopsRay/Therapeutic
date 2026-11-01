#!/usr/bin/env python3
"""Cache of UnDetect's parts on the drug-blind benchmark for second-stage experiments (no new model here).

Per fold, with the backbone seed and code path of `run_undetect_v2.py` `pool_kta` (KTA weights, normalised fused
kernel, lambda and a chosen on valid, then (g, h) on valid): backbone B, shared compound term G and per-line term H for
ALL rows. On training rows G is the closed-form leave-one-compound-out KRR prediction and H the leave-one-out
prediction within the line (the compound's own response never enters its own value), so a second stage fitted on
training rows sees out-of-sample compound terms. On valid/test rows both are the ordinary predictions. Also stores
lambda, a, (g, h) and the KTA weights. Writes results/sdst_hybrid/<tag>/fold_<f>/{cache.npz, summary.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sdst_hybrid import D, DEV, LAMS, OUT, SCALES, VIEWS, alignment, drug_kernels, train_mlp  # noqa: E402
from run_undetect_v2 import F64, T, fuse, score  # noqa: E402


def eigh_cpu(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """LAPACK eigh, results back on DEV: a cusolver failure poisons the CUDA context (`run_undetect_v2.eigh`), and these
    kernels are small. Identical to `eigh` on a CPU run. syevd can fail to converge on a kernel with many zero rows
    (a single sparse view, e.g. Cell Painting alone in one line: job 359224); then scipy's MRRR driver (evr)."""
    try:
        return tuple(x.to(DEV) for x in torch.linalg.eigh(M.cpu()))
    except torch.linalg.LinAlgError:
        import scipy.linalg
        print("  eigh: syevd failed, retry with scipy evr", flush=True)
        w, V = scipy.linalg.eigh(M.cpu().numpy(), driver="evr")
        return T(w), T(V)


def run_fold(f: int, A: dict, Kv: list, Knp: dict, args, views: list = VIEWS, B: np.ndarray | None = None,
             save: bool = True) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Kv: kernels of `views` on DEV. With B given (a cached backbone) no backbone is trained; save=False writes nothing.
    Returns (summary, B, G, H)."""
    t0 = time.time()
    g = A["group"]; te = g == f; va = g == (f + 1) % 5; tr = ~te & ~va
    ctl, Y, di, cell = A["ctl"], A["delta"], A["drow"], A["cell_idx"]
    if B is None:
        pred_mlp, vbest, vep, _ = train_mlp(ctl[tr], Y[tr], ctl[va], Y[va], args.seed, args.epochs, args.patience)
        B = pred_mlp(ctl).astype(np.float64)
        print(f"fold {f}: mlp valid {vbest:.4f} at epoch {vep}; {time.time() - t0:.0f}s", flush=True)
    else:
        vbest, vep = float("nan"), -1
    R = Y - B
    tdr, tinv = np.unique(di[tr], return_inverse=True); nd = np.bincount(tinv).astype(float)
    C = np.zeros((len(tdr), Y.shape[1])); np.add.at(C, tinv, R[tr]); C /= nd[:, None]
    align = {v: alignment(Knp[v][np.ix_(tdr, tdr)], C) for v in views}
    w = np.array([max(align[v], 0) for v in views]); w = w / w.sum()
    Kf = fuse(Kv, w); tdr_t = T(tdr, torch.long); Ct = T(C)
    vals, vecs = eigh_cpu(Kf[tdr_t][:, tdr_t]); vals = vals.clamp_min(0); proj = vecs.T @ Ct
    rows = {"valid": np.flatnonzero(va), "test": np.flatnonzero(te)}
    Ys = {s: T(Y[r]) for s, r in rows.items()}; Bs = {s: T(B[r]) for s, r in rows.items()}

    def shared_rows(r, lam):
        u, inv = np.unique(di[r], return_inverse=True)
        alpha = vecs @ (proj / (vals + lam)[:, None])
        return (Kf[T(u, torch.long)][:, tdr_t] @ alpha)[T(inv, torch.long)]

    Gv = {lam: shared_rows(rows["valid"], lam) for lam in LAMS}
    vk, lam, a = max((float(score(Ys["valid"], Bs["valid"] + a * Gv[lam]).mean()), lam, a) for lam in LAMS for a in SCALES)
    G = np.zeros_like(Y); H = np.zeros_like(Y)
    for s, r in rows.items():
        G[r] = (Gv[lam] if s == "valid" else shared_rows(r, lam)).cpu().numpy()
    filt = vals / (vals + lam); hd = (vecs ** 2) @ filt
    Ghat = vecs @ (filt[:, None] * proj)
    G[tr] = ((Ghat - hd[:, None] * Ct) / (1 - hd)[:, None]).cpu().numpy()[tinv]
    # per-line KRR at the same lambda (solve as run_undetect_v2.line_models) + its leave-one-out on training rows
    trl = np.flatnonzero(tr)
    for c in np.unique(cell[tr]):
        rc = trl[cell[trl] == c]; dl, inv = np.unique(di[rc], return_inverse=True)
        Cl = np.zeros((len(dl), Y.shape[1])); np.add.at(Cl, inv, R[rc]); Cl /= np.bincount(inv)[:, None]
        dl_t = T(dl, torch.long); Kl = Kf[dl_t][:, dl_t]; Clt = T(Cl)
        al = torch.linalg.solve(Kl + lam * torch.eye(len(dl), dtype=F64, device=DEV), Clt)
        lv, lV = eigh_cpu(Kl); lf = lv.clamp_min(0) / (lv.clamp_min(0) + lam); hl = (lV ** 2) @ lf
        Hhat = lV @ (lf[:, None] * (lV.T @ Clt))
        H[rc] = ((Hhat - hl[:, None] * Clt) / (1 - hl)[:, None]).cpu().numpy()[inv]
        for s, r in rows.items():
            sel = r[cell[r] == c]
            if len(sel):
                H[sel] = (Kf[T(di[sel], torch.long)][:, dl_t] @ al).cpu().numpy()
    Hv = T(H[rows["valid"]]); Gvt = T(G[rows["valid"]])
    best = (-2.0, None)
    for gs, hs in product(SCALES, SCALES):
        v = float(score(Ys["valid"], Bs["valid"] + gs * Gvt + hs * Hv).mean())
        if v > best[0]:
            best = (v, (gs, hs))
    vb, (gs, hs) = best
    Pt = B[rows["test"]] + gs * G[rows["test"]] + hs * H[rows["test"]]
    res = {"fold": f, "mlp_valid": float(vbest), "mlp_epoch": int(vep), "lam": lam, "a": a, "krr_valid": vk,
           "g": gs, "h": hs, "pool_valid": vb, "weights_kta": dict(zip(views, map(float, w))),
           "test": {"mlp": float(score(Ys["test"], Bs["test"]).mean()),
                    "krr_kta": float(score(Ys["test"], Bs["test"] + a * T(G[rows["test"]])).mean()),
                    "pool_kta": float(score(Ys["test"], T(Pt)).mean())},
           "loo_train_mean_abs": {"G": float(np.abs(G[tr]).mean()), "H": float(np.abs(H[tr]).mean())},
           "seconds": round(time.time() - t0, 1)}
    print(f"  {json.dumps(res)}", flush=True)
    if save:
        out = OUT / args.tag / f"fold_{f}"; out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "cache.npz", B=B.astype(np.float32), G=G.astype(np.float32), H=H.astype(np.float32))
        (out / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    return res, B, G, H


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--tag", default="cache_v1")
    args = p.parse_args()
    print(vars(args), "device", DEV, flush=True)
    z = np.load(D / "sdst.npz", allow_pickle=False)
    A = {k: z[k] for k in ("ctl", "delta", "pert_idx", "cell_idx", "group")}
    A["delta"] = A["delta"].astype(np.float64)
    table = pd.read_csv(D / "drugs.tsv", sep="\t", keep_default_na=False)
    rowof = {d: i for i, d in enumerate(table.pert_idx)}
    A["drow"] = np.array([rowof[d] for d in A["pert_idx"]])
    Knp, _ = drug_kernels(table)
    Kv = [T(Knp[v]) for v in VIEWS]
    for f in args.folds:
        run_fold(f, A, Kv, Knp, args)


if __name__ == "__main__":
    main()
