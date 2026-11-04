#!/usr/bin/env python3
"""95% intervals for Fig. 3b: response similarity of chemically unlike pairs by view-similarity bin (no training).

Rebuilds the pairs of `analyze_sdst_compounds.py --final` Part A with the same code path (signatures = mean plate-centred
response per compound, compounds with >= MIN_N samples and a target or Cell Painting view; response similarity =
Pearson of signatures; chemically unlike = Morgan Tanimoto < CHEM_LOW) and checks that every bin mean and count equals
compound_analysis_final/summary.json. Pairs share compounds, so the interval resamples compounds: each replicate draws
the compounds with replacement and weights a pair by the product of its two compounds' draw counts (as the hop contrast
in analyze_sdst_compounds.py). Also the chemical-analogue line (Tanimoto >= CHEM_HIGH).
Writes results/sdst_hybrid/compound_analysis_final/view_bins_ci.json.
Run (Slurm): slurm/view_bins_ci.sbatch
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sdst_compounds import CHEM_HIGH, CHEM_LOW, MIN_N, zrows  # noqa: E402
from run_sdst_hybrid import D, OUT, VIEWS, drug_kernels  # noqa: E402

RES = OUT / "compound_analysis_final"
EDGES = {"tgt": [0, 1e-9, 0.5, 1.0001], "moa": [0, 1e-9, 0.5, 1.0001], "morph": [-1, 0.3, 0.5, 0.7, 1.0001]}
NBOOT, SEED = 1000, 0


def main() -> None:
    print("host", platform.node(), "python", sys.version.split()[0], "numpy", np.__version__, "pandas", pd.__version__,
          "seed", SEED, "boot", NBOOT, flush=True)
    rng = np.random.default_rng(SEED)
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, pidx = z["ctl"], z["delta"].astype(np.float64), z["pert_idx"]
    table = pd.read_csv(D / "drugs.tsv", sep="\t", keep_default_na=False)
    rowof = {d: i for i, d in enumerate(table.pert_idx)}; di = np.array([rowof[d] for d in pidx])
    K, feats = drug_kernels(table); has = feats["has"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    ng = np.bincount(key); S = np.zeros((ng.size, Y.shape[1])); np.add.at(S, key, Y)
    ok = ng[key] >= 2
    E = Y[ok] - (S[key[ok]] - Y[ok]) / (ng[key[ok]] - 1)[:, None]
    dE = di[ok]; nd = np.bincount(dE, minlength=len(table))
    sig = np.zeros((len(table), Y.shape[1])); np.add.at(sig, dE, E); sig /= np.maximum(nd, 1)[:, None]
    sel = np.flatnonzero((nd >= MIN_N) & (has["tgt"] | has["morph"]))
    Zs = zrows(sig[sel]); Rr = Zs @ Zs.T
    iu = np.triu_indices(len(sel), 1)
    P = pd.DataFrame({"i": sel[iu[0]], "j": sel[iu[1]], "resp": Rr[iu], "chem": K["chem"][np.ix_(sel, sel)][iu]})
    for v in VIEWS[1:]:
        pi, pj = P.i.to_numpy(), P.j.to_numpy()
        P[v] = np.where(has[v][pi] & has[v][pj], K[v][pi, pj], np.nan)
    low = P[P.chem < CHEM_LOW]; high = P[P.chem >= CHEM_HIGH]
    print(f"{len(sel)} compounds, {len(P)} pairs, {len(low)} chemically unlike, {len(high)} analogues", flush=True)

    ref = json.loads((RES / "summary.json").read_text())
    groups = {}
    for v, edges in EDGES.items():
        m = low[v].notna(); b = pd.cut(low.loc[m, v], edges, right=False)
        for k, (lab, sub) in enumerate(low[m].groupby(b, observed=True)):
            r0 = ref["view_bins_chem_low"][v][k]
            assert str(lab) == r0["bin"] and len(sub) == r0["n_pairs"] and abs(sub.resp.mean() - r0["resp_mean"]) < 1e-9, \
                (v, str(lab), len(sub), r0)
            groups[(v, str(lab))] = sub
    assert abs(high.resp.mean() - ref["resp_mean"]["chem_high_analogues"]) < 1e-9
    groups[("analogues", f"chem >= {CHEM_HIGH}")] = high
    print("bin means and counts reproduce summary.json", flush=True)

    comp = np.unique(np.r_[low.i, low.j, high.i, high.j])
    arrs = {k: (g.i.to_numpy(), g.j.to_numpy(), g.resp.to_numpy()) for k, g in groups.items()}
    boots = {k: [] for k in groups}
    for b in range(NBOOT):
        cnt = np.bincount(rng.choice(comp, len(comp)), minlength=len(table)).astype(float)
        for k, (i, j, r) in arrs.items():
            w = cnt[i] * cnt[j]
            if w.sum() > 0:
                boots[k].append(float(np.dot(w, r) / w.sum()))
        if b % 100 == 0:
            print("replicate", b, flush=True)
    res = []
    for k, g in groups.items():
        lo, hi = np.quantile(boots[k], [0.025, 0.975])
        res.append({"view": k[0], "bin": k[1], "resp_mean": float(g.resp.mean()), "n_pairs": len(g),
                    "n_compounds": int(np.unique(np.r_[g.i, g.j]).size), "ci95": [float(lo), float(hi)],
                    "n_boot": len(boots[k])})
        print(res[-1], flush=True)
    (RES / "view_bins_ci.json").write_text(json.dumps({"seed": SEED, "n_boot": NBOOT, "bins": res}, indent=1))
    print("wrote", RES / "view_bins_ci.json", flush=True)


if __name__ == "__main__":
    main()
