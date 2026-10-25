#!/usr/bin/env python3
"""X4 (exploratory, post hoc): chemistry retrieval gain as a function of the nearest training analogue.

report/10_09_record.md, 2026-10-03. For every scored (compound, line) row of an official run, the
maximum Morgan Tanimoto to that line's training compounds in the same outer fold is computed, and
the per-row gain of `knn_morgan2` (and its same-plate-excluded twin) over `mean` is averaged by
similarity bin. Answers: if the library holds a close analogue, how much does approach 1 deliver?
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import run_l1000_unseen_drug as run


def main(a):
    d = a.derived
    sl = np.load(d / f"slice_{a.slice}.npz", allow_pickle=False)
    keys = sl["inchikey"].astype(str); lines = sl["line"].astype(str)
    comp = np.array(sorted(set(keys))); cidx = {k: i for i, k in enumerate(comp)}; row_c = np.array([cidx[k] for k in keys])
    t = pd.read_parquet(d / "drug_table.parquet"); t = t[t.source.eq("L1000") & t.inchikey.isin(comp)].groupby("inchikey").first().reindex(comp)
    ik14 = t.ik14.astype(str).to_numpy(); scaf = t.generic_scaffold.fillna("").astype(str).to_numpy()
    test_cf, train_cf, _, _ = run.make_folds(a.split, 0, ik14, scaf, row_c, sl["plates"].astype(str), len(comp))
    f = np.load(d / "drug_features.npz", allow_pickle=False)
    pos = pd.Series(np.arange(len(f["inchikey"])), index=f["inchikey"].astype(str))[comp].to_numpy()
    T = run.tanimoto(f["morgan2"][pos], f["morgan2"][pos])
    frame = pd.read_parquet(a.results / f"{a.slice}_{a.split}_seed0" / "per_condition.parquet")
    wide = frame.pivot_table(index=["line", "inchikey"], columns="arm", values="pearson_full").reset_index()
    flags = frame[frame.arm == "mean"][["line", "inchikey", "active", "fold"]]
    wide = wide.merge(flags, on=["line", "inchikey"])
    measured = {L: set(row_c[lines == L]) for L in set(lines)}
    maxsim = []
    for r in wide.itertuples():
        c = cidx[r.inchikey]
        tr = np.array([j for j in measured[r.line] if train_cf[j, r.fold]])
        maxsim.append(T[c, tr].max())
    wide["max_tanimoto"] = maxsim
    wide["bin"] = pd.cut(wide.max_tanimoto, [0, .3, .4, .5, .6, .8, 1.0001], right=False)
    act = wide[wide.active]
    out = act.groupby("bin", observed=True).apply(lambda g: pd.Series({
        "rows": len(g), "share": len(g) / len(act),
        "mean_full": g["mean"].mean(), "knn_morgan2_gain": (g.knn_morgan2 - g["mean"]).mean(),
        "knn_morgan2_xplate_gain": (g.knn_morgan2_xplate - g["mean"]).mean(),
        "oracle_gain": (g.oracle_knn - g["mean"]).mean()})).round(4)
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out / f"{a.slice}_{a.split}.csv")
    print(a.slice, a.split); print(out.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    base = Path("/nfs-stor/yuqi.lei/Therapeutic")
    p.add_argument("--slice", default="P1-24"); p.add_argument("--split", default="ik14")
    p.add_argument("--derived", type=Path, default=base / "derived/l1000_unseen_drug")
    p.add_argument("--results", type=Path, default=base / "results/l1000_unseen_drug")
    p.add_argument("--out", type=Path, default=base / "results/l1000_applicability")
    main(p.parse_args())
