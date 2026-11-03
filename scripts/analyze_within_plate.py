#!/usr/bin/env python3
"""Within-plate contrast metric PCC_within on the drug-blind benchmark (no training; design and reading rule in
report/11_06_record.md 2026-11-03 07:00).

PCC_plate centres truth and prediction on the same plate-mean estimate m_i, whose error is shared by both sides, so a
compound-blind prediction scores above zero. PCC_within removes the shared reference: for each plate (control row)
with >= 2 test compounds, truth and predictions are averaged per (plate, compound), the truth is centred on its own
plate mean and each prediction on its own; the score is the Pearson over genes per (plate, compound), averaged over
units. A prediction that is constant within a plate scores exactly 0. Bootstrap: 2,000 replicates over test compounds
per fold. Writes results/sdst_hybrid/within_plate/{units.csv.gz, folds.csv, contrasts.csv, summary.json}.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
OUT = RES / "within_plate"
ARMS = ("mlp", "mlp_bai", "pool_plate", "pool_chem", "mlp_input", "nnf_multi", "pool_kta", "ctx_pc", "xpert",
        "xpert_zero", "transigen", "deepce", "ctx_pc+xpert")
PAIRS = (("ctx_pc", "mlp"), ("ctx_pc", "xpert"), ("ctx_pc+xpert", "ctx_pc"), ("ctx_pc+xpert", "xpert"),
         ("xpert", "xpert_zero"), ("ctx_pc", "pool_kta"), ("ctx_pc", "nnf_multi"), ("ctx_pc", "mlp_input"),
         ("pool_chem", "pool_plate"))
CONST_TOL = 1e-3  # a centred prediction below this max-abs is treated as constant within the plate
SEED = 0


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1.0), 0.0)


def group_mean(x: np.ndarray, inv: np.ndarray, k: int) -> np.ndarray:
    s = np.zeros((k, x.shape[1])); np.add.at(s, inv, x)
    return s / np.bincount(inv, minlength=k)[:, None]


def main() -> None:
    print("host", platform.node(), "python", sys.version.split()[0], "numpy", np.__version__, "pandas", pd.__version__,
          "seed", SEED, flush=True)
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, drug = z["ctl"], z["delta"].astype(np.float64), z["group"], z["pert_idx"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    units, cov, spread = [], [], {}
    for f in range(5):
        te = np.flatnonzero(g == f)
        P = np.load(RES / "final_cpu" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        assert np.allclose(P["y"].astype(np.float64), Y[te], rtol=2e-3, atol=1e-2), "row order differs from sdst.npz"
        u = pd.DataFrame({"plate": key[te], "drug": drug[te]})
        uid, inv = np.unique(u.to_numpy(), axis=0, return_inverse=True); inv = inv.ravel()
        up, ud = uid[:, 0], uid[:, 1]
        ncomp = pd.Series(up).map(pd.Series(up).value_counts()).to_numpy()
        keep = ncomp >= 2
        pid = np.unique(up[keep], return_inverse=True)[1].ravel(); npl = pid.max() + 1
        Yu = group_mean(Y[te], inv, len(uid))[keep]
        Yc = Yu - group_mean(Yu, pid, npl)[pid]
        fr = pd.DataFrame({"fold": f, "plate": up[keep], "drug": ud[keep], "n_comp": ncomp[keep]})
        for a in ARMS:
            Pu = group_mean(P[a].astype(np.float64), inv, len(uid))[keep]
            Pc = Pu - group_mean(Pu, pid, npl)[pid]
            mx = np.abs(Pc).max(1)
            spread[f"{a}_f{f}"] = float(mx.max())
            Pc[mx < CONST_TOL] = 0.0
            fr[a] = pcc(Yc, Pc)
        units.append(fr)
        rows_cov = np.isin(inv, np.flatnonzero(keep)).sum()
        cov.append({"fold": f, "test_rows": len(te), "rows_covered": int(rows_cov), "share_rows": rows_cov / len(te),
                    "plates_kept": int(npl), "units": int(keep.sum()), "duplicate_rows_merged": int(len(te) - len(uid))})
        print(cov[-1], flush=True)
    U = pd.concat(units, ignore_index=True)
    folds = U.groupby("fold")[list(ARMS)].mean()
    plate_macro = U.groupby(["fold", "plate"])[list(ARMS)].mean().groupby("fold").mean()
    rng = np.random.default_rng(SEED); con = []
    for f, x in U.groupby("fold"):
        drugs, gi = np.unique(x.drug.to_numpy(), return_inverse=True); nd = np.bincount(gi).astype(float)
        W = rng.multinomial(len(drugs), np.full(len(drugs), 1 / len(drugs)), size=2000)
        for a in ARMS:
            v = x[a].to_numpy(); boot = (W @ np.bincount(gi, weights=v)) / (W @ nd)
            con.append({"fold": f, "pair": a, "diff": float(v.mean()), "lo": float(np.quantile(boot, 0.025)),
                        "hi": float(np.quantile(boot, 0.975))})
        for a, b in PAIRS:
            dd = (x[a] - x[b]).to_numpy(); boot = (W @ np.bincount(gi, weights=dd)) / (W @ nd)
            con.append({"fold": f, "pair": f"{a}-{b}", "diff": float(dd.mean()), "lo": float(np.quantile(boot, 0.025)),
                        "hi": float(np.quantile(boot, 0.975))})
    C = pd.DataFrame(con)
    S = C.groupby("pair", sort=False).agg(mean=("diff", "mean"), folds_pos=("diff", lambda s: int((s > 0).sum())),
                                          ci_pos=("lo", lambda s: int((s > 0).sum())),
                                          ci_neg=("hi", lambda s: int((s < 0).sum())))
    OUT.mkdir(parents=True, exist_ok=True)
    U.to_csv(OUT / "units.csv.gz", index=False); folds.to_csv(OUT / "folds.csv"); C.to_csv(OUT / "contrasts.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "coverage": cov, "max_within_plate_spread": spread, "fold_means": folds.reset_index().to_dict(orient="records"),
        "five_fold_mean": folds.mean().to_dict(), "plate_macro_five_fold_mean": plate_macro.mean().to_dict(),
        "contrast_summary": S.reset_index().to_dict(orient="records")}, indent=2) + "\n")
    pd.set_option("display.width", 200)
    print("max within-plate spread of centred predictions (before zeroing):",
          {a: round(max(spread[f"{a}_f{f}"] for f in range(5)), 6) for a in ARMS})
    print("five-fold mean PCC_within:\n", folds.mean().round(4).to_string())
    print("plate-macro five-fold mean:\n", plate_macro.mean().round(4).to_string())
    print("per fold:\n", folds.round(4).to_string())
    print("contrasts (mean over folds; folds > 0; folds CI > 0; folds CI < 0):\n", S.round(4).to_string())


if __name__ == "__main__":
    main()
