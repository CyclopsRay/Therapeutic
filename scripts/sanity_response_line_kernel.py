#!/usr/bin/env python3
"""Follow-up to `sanity_context_interaction.py` Part 2: is a line similarity learned from *responses* a better context
than the control profile? (Bai et al. 2026 drug-blind benchmark; no network training.)

Molina & Zhang (2026) find that the perturbation x cell-line interaction is not predictable from control expression but
is learnable from measured responses of other perturbations in the target line. In the drug-blind split every line is
seen, so its response geometry is available from training compounds. Per fold, from train rows only: E = plate-centred
response among train rows; for lines l, l' the similarity S = sum over training compounds measured in both of
Pearson(E_dl, E_dl') / (shared compounds + KAPPA) (shrunk towards 0 when few compounds are shared). The same oracle as
Part 2 (backbone + weighted mean of the test compound's other-line residuals) is then run with distance 1 - S and with
the control distance, bandwidths cross-fitted on the neighbouring fold. Uses test responses: an upper bound, not a model.
Writes results/sdst_hybrid/context_sanity/response_kernel_* (run 353168 used h 0.1-2 and is overwritten).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sanity_context_interaction import D, OUT, RES, boot_diff, pcc, zrows  # noqa: E402

KAPPA = 5.0
HS = (0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)  # extended below 0.1 after run 353168 chose h = 0.1 at the grid edge
SEED = 0


def weighted(r: np.ndarray, line: np.ndarray, drug: np.ndarray, Dl: np.ndarray, s: float) -> dict:
    out = {"uniform": np.full(r.shape, np.nan), **{f"w{h}": np.full(r.shape, np.nan) for h in HS}}
    for d in np.unique(drug):
        idx = np.flatnonzero(drug == d); n = len(idx)
        if n < 2:
            continue
        R = r[idx]; out["uniform"][idx] = (R.sum(0)[None] - R) / (n - 1)
        dist = Dl[np.ix_(line[idx], line[idx])]
        for h in HS:
            W = np.exp(-dist / (h * s)); np.fill_diagonal(W, 0.0)
            out[f"w{h}"][idx] = (W @ R) / np.maximum(W.sum(1, keepdims=True), 1e-300)
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, pidx, cell = z["ctl"].astype(np.float64), z["delta"].astype(np.float64), z["group"], z["pert_idx"], z["cell_idx"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in z["ctl"]]), return_inverse=True)[1]
    nk, nl = key.max() + 1, cell.max() + 1
    # control distance, as in sanity_context_interaction.py
    first = np.unique(key, return_index=True)[1]; kline = cell[first]; cnt = np.bincount(kline, minlength=nl)
    Lc = np.zeros((nl, ctl.shape[1])); np.add.at(Lc, kline, ctl[first]); Lc /= np.maximum(cnt, 1)[:, None]
    Lc -= Lc[cnt > 0].mean(0); Zl = zrows(Lc); Dctl = 1.0 - Zl @ Zl.T
    kern = {}
    for f in range(5):
        tr = (g != f) & (g != (f + 1) % 5)
        n_tr = np.bincount(key[tr], minlength=nk); S = np.zeros((nk, Y.shape[1])); np.add.at(S, key[tr], Y[tr])
        ok = np.flatnonzero(tr & (n_tr[key] >= 2))
        Ez = zrows(Y[ok] - (S[key[ok]] - Y[ok]) / (n_tr[key[ok]] - 1)[:, None])
        dok, lok = pidx[ok], cell[ok]
        num = np.zeros((nl, nl)); den = np.zeros((nl, nl))
        order = np.argsort(dok, kind="stable"); b = np.flatnonzero(np.r_[True, dok[order][1:] != dok[order][:-1], True])
        for a0, a1 in zip(b[:-1], b[1:]):
            idx = order[a0:a1]
            if len(idx) < 2:
                continue
            C = Ez[idx] @ Ez[idx].T; li = lok[idx]
            np.add.at(num, (li[:, None], li[None, :]), C); np.add.at(den, (li[:, None], li[None, :]), 1.0)
        Sr = num / (den + KAPPA); np.fill_diagonal(Sr, 1.0)
        kern[f] = {"resp": 1.0 - Sr, "ctl": Dctl, "shared_median": float(np.median(den[np.triu_indices(nl, 1)]))}
        iu = np.triu_indices(nl, 1); m = den[iu] >= 10
        print(f"fold {f}: line-pair response similarity, pairs with >= 10 shared train compounds: {m.sum()}; "
              f"mean {Sr[iu][m].mean():.4f}; Spearman vs control distance "
              f"{pd.Series(Sr[iu][m]).rank().corr(pd.Series(-Dctl[iu][m]).rank()):.3f}", flush=True)
    sc = {}
    for f in range(5):
        te = g == f; tr = (g != f) & (g != (f + 1) % 5)
        P = np.load(RES / "v3" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        Yt = Y[te]; B = P["mlp"].astype(np.float64); r = Yt - B
        n_tr = np.bincount(key[tr], minlength=nk); Sm = np.zeros((nk, Y.shape[1])); np.add.at(Sm, key[tr], Y[tr])
        m = Sm[key[te]] / np.maximum(n_tr[key[te]], 1)[:, None]; sh = n_tr[key[te]] > 0
        nd = np.bincount(pidx[te], minlength=pidx.max() + 1)[pidx[te]]; rows = sh & (nd >= 2)
        sc[f] = {"rows": rows, "drug": pidx[te],
                 "undetect": pcc(Yt - m, P["nnk_multi_norm"].astype(np.float64) - m), "mlp": pcc(Yt - m, B - m)}
        for name in ("resp", "ctl"):
            Dl = kern[f][name]; s = float(np.median(Dl[np.triu_indices(nl, 1)]))
            for arm, o in weighted(r, cell[te], pidx[te], Dl, s).items():
                sc[f][f"{name}_{arm}"] = pcc(Yt - m, B + np.nan_to_num(o) - m)
    tab, con = [], []
    for f in range(5):
        F, V = sc[f], sc[(f + 1) % 5]
        pick = {name: max([f"{name}_w{h}" for h in HS], key=lambda a: V[a][V["rows"]].mean()) for name in ("resp", "ctl")}
        rr = F["rows"]
        for a in [k for k in F if k not in ("rows", "drug")]:
            tab.append({"fold": f, "arm": a, "pcc_plate": float(F[a][rr].mean())})
        for name, a in pick.items():
            tab.append({"fold": f, "arm": f"{name}_w*", "chosen": a, "pcc_plate": float(F[a][rr].mean())})
        for a, b_ in ((pick["resp"], "resp_uniform"), (pick["ctl"], "resp_uniform"), (pick["resp"], pick["ctl"])):
            mu, lo, hi = boot_diff(F[a][rr] - F[b_][rr], F["drug"][rr], rng)
            con.append({"fold": f, "a": a.split("_")[0] + "_w*", "b": b_ if b_ == "resp_uniform" else "ctl_w*",
                        "diff": mu, "lo": lo, "hi": hi})
    T, C = pd.DataFrame(tab), pd.DataFrame(con)
    print("\nPCC_plate, shared rows with >= 1 other line (5-fold mean):")
    print(T.groupby("arm").pcc_plate.mean().sort_values().round(4).to_string())
    print(T[T.arm.str.endswith("*")][["fold", "arm", "chosen"]].to_string(index=False))
    print("\ncontrasts (bootstrap over compounds):")
    print(C.groupby(["a", "b"]).agg(diff=("diff", "mean"), folds_pos=("diff", lambda x: int((x > 0).sum())),
                                    ci_pos=("lo", lambda x: int((x > 0).sum()))).round(4).to_string())
    T.to_csv(OUT / "response_kernel_scores.csv", index=False); C.to_csv(OUT / "response_kernel_contrasts.csv", index=False)


if __name__ == "__main__":
    main()
