"""Scan every held-out compound of the drug-blind benchmark for the Fig. 1 / Fig. 3 case study (no training).

Per pert_idx, in the fold where it is held out: number of test rows and of rows with a training plate-mate, and mean
PCC_plate of the drug-free MLP (Bai), chemistry kernel, XPert, UnDetect and the ensemble, with the plate mean exactly as
in scripts/finalize_sdst.py. Ranks named compounds (cmap_name not a BRD id) with >= MIN_SHARED rows with a plate-mate by
min(UnDetect - XPert, UnDetect - chemistry kernel). Writes results/sdst_hybrid/case_study/scan.csv.
Run (Slurm): slurm/case_study_scan.sbatch
"""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
OUT = RES / "case_study"
ARMS = ("mlp_bai", "pool_chem", "xpert", "ctx_pc", "ctx_pc+xpert")
MIN_SHARED = 30
SEED = 0


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1.0), np.nan)


def main() -> None:
    print("host", platform.node(), "python", sys.version.split()[0], "numpy", np.__version__, "pandas", pd.__version__,
          "seed", SEED, flush=True)
    t = pd.read_csv(D / "drugs.tsv", sep="\t", keep_default_na=False).set_index("pert_idx")
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, pidx = z["ctl"], z["delta"].astype(np.float64), z["group"], z["pert_idx"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    out = []
    for f in range(5):
        te = np.flatnonzero(g == f); tr = (g != f) & (g != (f + 1) % 5)
        n = np.bincount(key[tr], minlength=key.max() + 1); Sm = np.zeros((key.max() + 1, Y.shape[1]))
        np.add.at(Sm, key[tr], Y[tr]); m = (Sm / np.maximum(n, 1)[:, None])[key[te]]; sh = n[key[te]] > 0
        P = np.load(RES / "final_cpu" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        Yt = Y[te]
        sc = {a: pcc(Yt - m, P[a].astype(np.float64) - m) for a in ARMS}
        df = pd.DataFrame({"pert_idx": pidx[te], "sh": sh, **sc})
        agg = df[df.sh].groupby("pert_idx")[list(ARMS)].mean()
        agg["n_shared"] = df[df.sh].groupby("pert_idx").size()
        agg["n_rows"] = df.groupby("pert_idx").size().reindex(agg.index)
        agg["fold"] = f
        out.append(agg.reset_index())
        print(f"fold {f}: {len(agg)} compounds", flush=True)
    S = pd.concat(out)
    S["name"] = S.pert_idx.map(t.cmap_name); S["pert_id"] = S.pert_idx.map(t.pert_id)
    S["target"] = S.pert_idx.map(t.target); S["moa"] = S.pert_idx.map(t.moa)
    S["d_xpert"] = S.ctx_pc - S.xpert; S["d_chem"] = S.ctx_pc - S.pool_chem; S["score"] = S[["d_xpert", "d_chem"]].min(1)
    OUT.mkdir(parents=True, exist_ok=True); S.to_csv(OUT / "scan.csv", index=False)
    top = S[(S.n_shared >= MIN_SHARED) & ~S.name.str.startswith("BRD")].sort_values("score", ascending=False)
    cols = ["name", "pert_id", "fold", "n_shared", "mlp_bai", "pool_chem", "xpert", "ctx_pc", "d_xpert", "d_chem",
            "target", "moa"]
    print(f"named compounds with >= {MIN_SHARED} plate-mate rows: {len(top)}; UnDetect > XPert in "
          f"{(top.d_xpert > 0).mean():.0%}, > chemistry kernel in {(top.d_chem > 0).mean():.0%}", flush=True)
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        print(top[cols].head(30).round(3).to_string(index=False), flush=True)
    print("wrote", OUT / "scan.csv", flush=True)


if __name__ == "__main__":
    main()
