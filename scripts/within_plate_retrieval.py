#!/usr/bin/env python3
"""Within-plate self-retrieval on the drug-blind benchmark: can a compound's predicted signature pick out its own
measured signature among the other test compounds on the same plate? (connectivity-style screening check; no training;
design recorded in report/11_06_record.md 2026-11-04 before the run).

Units as `analyze_within_plate.py`: per fold, test rows averaged per (plate = control row, compound); plates with >= 2
test compounds. Truth and each prediction are centred on their own plate mean over the plate's test compounds, so
anything shared by the plate (cell line, control, backbone) cancels. For compound d on plate p with n compounds,
s(d, d') = corr(predicted_c[d], measured_c[d']) over genes for all d' on p; the rank of d' = d among the n candidates
(ties averaged) gives: top-1 (rank 1; chance 1/n), reciprocal rank (chance H_n / n), normalised rank (rank - 1)/(n - 1)
(0 = best, chance 0.5). A prediction constant within the plate ties every candidate and scores exactly chance.
Means over units pooled over the five folds, overall and by plate size; paired differences with a 2,000-replicate
bootstrap over test compounds. Writes results/sdst_hybrid/within_retrieval/{units.csv.gz, summary.csv, summary.json}.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
OUT = RES / "within_retrieval"
ARMS = ("mlp", "mlp_bai", "ciger", "multidcp", "deepce", "transigen", "xpert", "xpert_zero", "pool_chem", "pool_kta",
        "ctx_pc", "ctx_pc+xpert")
PAIRS = (("ctx_pc", "xpert"), ("ctx_pc", "transigen"), ("ctx_pc", "pool_chem"), ("ctx_pc+xpert", "xpert"),
         ("ctx_pc+xpert", "ctx_pc"))
SIZES = ((2, 2), (3, 5), (6, 10), (11, 10 ** 6))
CONST_TOL = 1e-3
SEED, B = 0, 2000


def group_mean(x: np.ndarray, inv: np.ndarray, k: int) -> np.ndarray:
    s = np.zeros((k, x.shape[1])); np.add.at(s, inv, x)
    return s / np.bincount(inv, minlength=k)[:, None]


def unit_scale(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(1, keepdims=True); n = np.sqrt((x * x).sum(1, keepdims=True))
    return np.where(n > 0, x / np.where(n > 0, n, 1.0), 0.0)


def main() -> None:
    print("host", platform.node(), "python", platform.python_version(), "numpy", np.__version__, "pandas",
          pd.__version__, "seed", SEED, flush=True)
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, drug = z["ctl"], z["delta"].astype(np.float64), z["group"], z["pert_idx"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    parts = []
    for f in range(5):
        te = np.flatnonzero(g == f)
        P = np.load(RES / "final_cpu" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        assert np.allclose(P["y"].astype(np.float64), Y[te], rtol=2e-3, atol=1e-2), "row order differs from sdst.npz"
        u = pd.DataFrame({"plate": key[te], "drug": drug[te]})
        uid, inv = np.unique(u.to_numpy(), axis=0, return_inverse=True); inv = inv.ravel()
        up, ud = uid[:, 0], uid[:, 1]
        ncomp = pd.Series(up).map(pd.Series(up).value_counts()).to_numpy()
        keep = np.flatnonzero(ncomp >= 2)
        Yu = group_mean(Y[te], inv, len(uid))
        Pu = {a: group_mean(P[a].astype(np.float64), inv, len(uid)) for a in ARMS}
        rec = {a: [] for a in ARMS}
        plates = pd.Series(keep).groupby(up[keep]).apply(np.asarray)
        meta = []
        for p, idx in plates.items():
            n = len(idx); Tc = unit_scale(Yu[idx] - Yu[idx].mean(0))
            meta += [(f, p, ud[i], n) for i in idx]
            for a in ARMS:
                Pc = Pu[a][idx] - Pu[a][idx].mean(0)
                Pc[np.abs(Pc).max(1) < CONST_TOL] = 0.0
                S = unit_scale(Pc) @ Tc.T                                   # rows: predictions, cols: candidates
                r = np.array([rankdata(-S[k], method="average")[k] for k in range(n)])
                rec[a].append(r)
        fr = pd.DataFrame(meta, columns=["fold", "plate", "drug", "n"])
        for a in ARMS:
            fr[a] = np.concatenate(rec[a])
        parts.append(fr)
        print("fold", f, "units", len(fr), "plates", fr.plate.nunique(), flush=True)
    U = pd.concat(parts, ignore_index=True)
    n = U.n.to_numpy(float)
    M = {"top1": lambda r: (r == 1).astype(float), "rr": lambda r: 1.0 / r, "nrank": lambda r: (r - 1) / (n - 1)}
    chance = {"top1": (1 / n).mean(), "rr": np.mean([np.sum(1 / np.arange(1, k + 1)) / k for k in n]), "nrank": 0.5}
    rng = np.random.default_rng(SEED); out = []
    groups = [("all", np.ones(len(U), bool))] + [(f"n {lo}-{hi}" if hi < 10 ** 6 else f"n >= {lo}",
                                                  (U.n >= lo).to_numpy() & (U.n <= hi).to_numpy()) for lo, hi in SIZES]
    for lab, sel in groups:
        x = U[sel]; nn = x.n.to_numpy(float)
        drugs, gi = np.unique(x.drug.to_numpy(), return_inverse=True); nd = np.bincount(gi).astype(float)
        W = rng.multinomial(len(drugs), np.full(len(drugs), 1 / len(drugs)), size=B)
        for mt in M:
            f_ = {"top1": lambda r: (r == 1).astype(float), "rr": lambda r: 1.0 / r,
                  "nrank": lambda r: (r - 1) / (nn - 1)}[mt]
            ch = {"top1": (1 / nn).mean(), "rr": np.mean([np.sum(1 / np.arange(1, k + 1)) / k for k in nn]),
                  "nrank": 0.5}[mt]
            rec = {"group": lab, "metric": mt, "units": int(sel.sum()), "chance": float(ch),
                   **{a: float(f_(x[a].to_numpy()).mean()) for a in ARMS}}
            for a, b in PAIRS:
                dd = f_(x[a].to_numpy()) - f_(x[b].to_numpy()); bt = (W @ np.bincount(gi, weights=dd)) / (W @ nd)
                rec.update({f"{a}-{b}": float(dd.mean()), f"{a}-{b}|lo": float(np.quantile(bt, 0.025)),
                            f"{a}-{b}|hi": float(np.quantile(bt, 0.975))})
            out.append(rec)
    S = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    U.to_csv(OUT / "units.csv.gz", index=False); S.to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"overall_chance": {k: float(v) for k, v in chance.items()},
                                                  "rows": out}, indent=2) + "\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(S[["group", "metric", "units", "chance"] + list(ARMS)].round(4).to_string(index=False))
    print(S[["group", "metric"] + [c for c in S if "-" in c]].round(4).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
