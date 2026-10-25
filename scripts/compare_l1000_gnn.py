#!/usr/bin/env python3
"""Paired contrasts between the neural arms (run_l1000_gnn.py) and the kNN/KRR arms of the same slice/split.

Rows are paired on (line, inchikey); the bootstrap is run.line_macro_boot over the runner's
grouping column (ik14 or plate map). Cohort: KNOWN∩ACTIVE unless an arm is defined on all rows.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import run_l1000_unseen_drug as run

CONTRASTS = [
    ("gnn_ppi", "gnn_rewired"), ("gnn_ppi", "gnn_nograph"), ("gnn_ppi", "mean"),
    ("mlp_ppi", "mlp_ppi_rewired"), ("mlp_ppi", "mlp_ppi_nograph"), ("mlp_ppi", "mlp_target"),
    ("mlp_target", "mlp_chem"), ("mlp_chem", "knn_chem_known"), ("mlp_chem", "krr_chem_known"),
    ("mlp_target", "krr_chem_target"), ("mlp_ppi", "krr_ppi"), ("mlp_target", "mean"), ("mlp_ppi", "mean"),
    ("mlp_chem_all", "knn_chem_select"), ("mlp_chem_all", "mean"),
    ("dot_ppi", "dot_rewired"), ("dot_ppi", "dot_nograph"), ("dot_ppi", "mean"), ("dot_nograph", "mean"),
    ("dot_ppi", "mlp_ppi"), ("dot_nograph", "mlp_target"), ("dot_ppi", "krr_ppi"), ("dot_ppi", "krr_chem_target"),
    # G3b re-run, pre-registered GCN (record 2026-10-07): the first three are the gate
    ("gcn_ppi", "gcn_rewired"), ("gcn_ppi", "gcn_nograph"), ("gcn_ppi", "krr_chem_known"), ("gcn_ppi", "mean"),
    ("gcn_nograph", "mean"), ("gcn_ppi", "mlp_target"), ("gcn_ppi", "krr_ppi"),
]


def main(a):
    base = Path("/nfs-stor/yuqi.lei/Therapeutic/results")
    g = pd.read_parquet(base / "l1000_gnn" / f"{a.run}{a.tag}" / "per_condition.parquet")
    k = pd.read_parquet(base / "l1000_unseen_drug" / a.run / "per_condition.parquet")
    gm = g[g.arm == "mean"].merge(k[k.arm == "mean"], on=["line", "inchikey"], suffixes=("_g", "_k"))
    print(f"mean-arm agreement: {len(gm)} rows, max |diff| {np.abs(gm.pearson_full_g - gm.pearson_full_k).max():.2e}")
    flags = k[k.arm == "mean"][["line", "inchikey", "group", "active", "known"]]
    allp = pd.concat([g[g.arm != "mean"][["arm", "line", "inchikey", "pearson_full", "pearson_resid"]],
                      k[["arm", "line", "inchikey", "pearson_full", "pearson_resid"]]])
    rows = []
    for arm_a, arm_b in CONTRASTS:
        A = allp[allp.arm == arm_a]; B = allp[allp.arm == arm_b]
        if A.empty or B.empty:
            continue
        m = A.merge(B, on=["line", "inchikey"], suffixes=("_a", "_b")).merge(flags, on=["line", "inchikey"])
        m = m[m.active & (m.known | (arm_a == "mlp_chem_all"))]
        for metric in ("pearson_full", "pearson_resid"):
            est, ci, pl = run.line_macro_boot(m, (m[f"{metric}_a"] - m[f"{metric}_b"]).to_numpy(), 2000, 0)
            rows.append({"arm": arm_a, "minus": arm_b, "metric": metric, "diff": est, "low": ci[0], "high": ci[1],
                         "lines_positive": sum(v > 0 for v in pl.values()), "n_lines": len(pl), "rows": len(m)})
    out = pd.DataFrame(rows)
    out.to_csv(base / "l1000_gnn" / f"{a.run}{a.tag}" / "contrasts_vs_knn.csv", index=False)
    with pd.option_context("display.width", 200):
        print(out[out.metric == "pearson_full"].round(4).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="P2-24_ik14_seed0")
    p.add_argument("--tag", default="_shrink")
    main(p.parse_args())
