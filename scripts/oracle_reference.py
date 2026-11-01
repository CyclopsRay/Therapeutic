#!/usr/bin/env python3
"""Reference scale for PCC_plate: how far does UnDetect get towards knowing the compound's own responses elsewhere?
(Bai et al. 2026 drug-blind benchmark; v3 saved predictions; no training.)

Oracle = backbone + a * (mean backbone residual of the same test compound's other test samples, i.e. its measured
responses in other lines); the single-measurement version uses only the compound's most control-similar other line.
`sanity_context_interaction.py` used a = 1; here a is chosen on the neighbouring fold's test predictions (= fold f's
validation compounds) per metric, so the reference is shrunk like the models are. Uses test responses: a reference
point, not a model. Rows: shared-control test rows of compounds with >= 2 test samples (PCC_plate as in
`analyze_sdst_centred.py`). Writes results/sdst_hybrid/context_sanity/oracle_reference.csv.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
AS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
ARMS = ("mlp", "pool_multi_norm", "nnk_multi_norm")


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def main() -> None:
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, pidx, cell = z["ctl"], z["delta"].astype(np.float64), z["group"], z["pert_idx"], z["cell_idx"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]; nk = key.max() + 1
    first = np.unique(key, return_index=True)[1]; kl = cell[first]; nl = cell.max() + 1; cnt = np.bincount(kl, minlength=nl)
    Lc = np.zeros((nl, ctl.shape[1])); np.add.at(Lc, kl, ctl[first].astype(np.float64)); Lc /= np.maximum(cnt, 1)[:, None]
    Lc -= Lc[cnt > 0].mean(0); Lc -= Lc.mean(1, keepdims=True); Lc /= np.maximum(np.linalg.norm(Lc, axis=1, keepdims=True), 1e-12)
    Dl = 1.0 - Lc @ Lc.T
    S = {}
    for f in range(5):
        te = g == f; tr = (g != f) & (g != (f + 1) % 5)
        P = np.load(RES / "v3" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        Yt = Y[te]; B = P["mlp"].astype(np.float64); r = Yt - B; dt, lt = pidx[te], cell[te]
        U = np.full(r.shape, np.nan); N1 = np.full(r.shape, np.nan)
        for d in np.unique(dt):
            idx = np.flatnonzero(dt == d)
            if len(idx) < 2:
                continue
            R = r[idx]; U[idx] = (R.sum(0)[None] - R) / (len(idx) - 1)
            dd = Dl[np.ix_(lt[idx], lt[idx])].copy(); np.fill_diagonal(dd, np.inf); N1[idx] = R[dd.argmin(1)]
        n_tr = np.bincount(key[tr], minlength=nk); Sm = np.zeros((nk, Y.shape[1])); np.add.at(Sm, key[tr], Y[tr])
        m = Sm[key[te]] / np.maximum(n_tr[key[te]], 1)[:, None]
        rows = (n_tr[key[te]] > 0) & ~np.isnan(U[:, 0])
        sc = {a: {"plate": pcc(Yt[rows] - m[rows], P[a][rows].astype(np.float64) - m[rows]),
                  "deg": pcc(Yt[rows], P[a][rows].astype(np.float64))} for a in ARMS}
        for name, O in (("oracle_mean", U), ("oracle_one_line", N1)):
            for a in AS:
                Pa = B[rows] + a * O[rows]
                sc[f"{name}_a{a}"] = {"plate": pcc(Yt[rows] - m[rows], Pa - m[rows]), "deg": pcc(Yt[rows], Pa)}
        S[f] = {k: {mt: float(v.mean()) for mt, v in d.items()} for k, d in sc.items()}
        S[f]["_n"] = int(rows.sum())
    out = []
    for f in range(5):
        V = S[(f + 1) % 5]; row = {"fold": f, "n": S[f]["_n"]}
        for a in ARMS:
            row.update({f"{a}_plate": S[f][a]["plate"], f"{a}_deg": S[f][a]["deg"]})
        for name in ("oracle_mean", "oracle_one_line"):
            for mt in ("plate", "deg"):
                a = max(AS, key=lambda x: V[f"{name}_a{x}"][mt])
                row[f"{name}_{mt}"] = S[f][f"{name}_a{a}"][mt]; row[f"{name}_{mt}_a"] = a
            row[f"{name}_plate_a1"] = S[f][f"{name}_a1.0"]["plate"]
        out.append(row)
    T = pd.DataFrame(out)
    T["undetect_share_of_mean_oracle"] = (T.nnk_multi_norm_plate - T.mlp_plate) / (T.oracle_mean_plate - T.mlp_plate)
    pd.set_option("display.width", 250)
    print(T.round(4).T.to_string())
    print("\n5-fold mean:\n" + T.drop(columns="fold").mean().round(4).to_string())
    T.to_csv(RES / "context_sanity" / "oracle_reference.csv", index=False)


if __name__ == "__main__":
    main()
