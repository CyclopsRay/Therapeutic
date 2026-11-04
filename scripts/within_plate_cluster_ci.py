#!/usr/bin/env python3
"""Clustered bootstrap intervals for PCC_within (reviewer T3, 2026-11-04; no training).

Units are (plate, compound) pairs from `analyze_within_plate.py` (results/sdst_hybrid/within_plate/units.csv.gz).
Compounds on one plate share the plate's centring, so units are dependent within plates as well as within compounds.
Per fold, 2,000 replicates of three bootstraps of the unit mean (and of paired differences):
  compound   resample test compounds (the scheme of analyze_within_plate.py),
  plate      resample plates,
  two-way    resample plates and compounds independently; unit weight = (plate count) x (compound count)
             (pigeonhole bootstrap for crossed clusters).
Writes results/sdst_hybrid/within_plate/cluster_ci.csv and prints a summary.
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/within_plate")
SEED, B = 0, 2000
ARMS = ["ctx_pc", "pool_kta", "pool_chem", "pool_plate", "xpert", "transigen", "deepce", "ctx_pc+xpert"]
PAIRS = [("ctx_pc", "xpert"), ("ctx_pc", "pool_kta"), ("ctx_pc+xpert", "xpert"), ("ctx_pc+xpert", "ctx_pc")]


def boot(x: np.ndarray, p: np.ndarray, d: np.ndarray, kind: str, rng) -> np.ndarray:
    """B bootstrap means of the columns of x (units x k)."""
    up, pi = np.unique(p, return_inverse=True)
    ud, di = np.unique(d, return_inverse=True)
    out = np.empty((B, x.shape[1]))
    for b in range(B):
        if kind == "compound":
            w = np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud))[di]
        elif kind == "plate":
            w = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up))[pi]
        else:
            w = (np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up))[pi]
                 * np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud))[di])
        out[b] = w @ x / w.sum()
    return out


def main() -> None:
    print("host", platform.node(), "python", platform.python_version(), "numpy", np.__version__, "pandas",
          pd.__version__, "seed", SEED, "B", B, flush=True)
    U = pd.read_csv(D / "units.csv.gz")
    rows = []
    for f, g in U.groupby("fold"):
        rng = np.random.default_rng(SEED + int(f))
        cols = {a: g[a].to_numpy(float) for a in ARMS}
        for a, b in PAIRS:
            cols[f"{a} - {b}"] = cols[a] - cols[b]
        names = list(cols)
        x = np.column_stack([cols[n] for n in names])
        for kind in ("compound", "plate", "two-way"):
            bs = boot(x, g["plate"].to_numpy(), g["drug"].to_numpy(), kind, rng)
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            for k, n in enumerate(names):
                rows.append(dict(fold=int(f), quantity=n, scheme=kind, mean=x[:, k].mean(), lo=lo[k], hi=hi[k]))
        print("fold", f, "units", len(g), "plates", g["plate"].nunique(), "compounds", g["drug"].nunique(), flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(D / "cluster_ci.csv", index=False)
    R["above0"] = R.lo > 0
    S = R.groupby(["quantity", "scheme"]).agg(mean=("mean", "mean"), folds_above0=("above0", "sum"),
                                                min_lo=("lo", "min")).reset_index()
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(S.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("wrote", D / "cluster_ci.csv")


if __name__ == "__main__":
    sys.exit(main())
