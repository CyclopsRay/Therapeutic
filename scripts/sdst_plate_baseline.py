#!/usr/bin/env python3
"""Model-free plate baselines on the Bai et al. 2026 drug-blind benchmark (same folds as run_sdst_hybrid.py).

In the XPert SDST release every (cell line, plate set) has exactly one control profile X_ctl, shared by all treated
samples on those plates, so X_ctl identifies the plate. Baselines, fitted on train rows only, scored by PCC_DEG:
  cell mean      mean train delta of the sample's cell line
  plate mean     mean train delta of the samples sharing the test sample's control row (fallback: cell mean)
Also reports the share of test samples whose control row occurs among train samples.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
OUT = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/plate_baseline")


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def group_mean(key: np.ndarray, rows: np.ndarray, Y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    S = np.zeros((k, Y.shape[1])); n = np.bincount(key[rows], minlength=k)
    np.add.at(S, key[rows], Y[rows])
    return S / np.maximum(n, 1)[:, None], n


def main() -> None:
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, cell = z["ctl"], z["delta"].astype(np.float64), z["group"], z["cell_idx"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    print(f"{len(key)} samples, {key.max() + 1} distinct control rows, {cell.max() + 1} cell ids", flush=True)
    res = []
    for f in range(5):
        te = g == f; tr = (g != f) & (g != (f + 1) % 5)
        Mc, _ = group_mean(cell, tr, Y, cell.max() + 1)
        Mp, n = group_mean(key, tr, Y, key.max() + 1)
        shared = n[key[te]] > 0
        Pc = Mc[cell[te]]; Pp = np.where(shared[:, None], Mp[key[te]], Pc)
        rc, rp = pcc(Y[te], Pc), pcc(Y[te], Pp)
        r = {"fold": f, "share_test_with_train_control": float(shared.mean()), "cell_mean": float(rc.mean()),
             "plate_mean": float(rp.mean()), "plate_mean_shared": float(rp[shared].mean()),
             "plate_mean_unshared": float(rp[~shared].mean()) if (~shared).any() else None}
        res.append(r); print(json.dumps(r), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    print("mean plate_mean", np.mean([r["plate_mean"] for r in res]), "mean cell_mean", np.mean([r["cell_mean"] for r in res]))


if __name__ == "__main__":
    main()
