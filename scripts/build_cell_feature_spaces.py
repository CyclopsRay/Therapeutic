#!/usr/bin/env python
"""Per-cell-line feature matrices, keyed by Cellosaurus ID, for kernel distances.

Pre-registered F1 (report/10_02_record.md, 2026-09-30). Each output parquet is
indexed by CVCL id; columns are features. These are used ONLY to define the
reference-line weighting w(c*, c) in run_context_slopes.py's kernel arm, so they
add no fitted parameters -- the design principle the project's own results imply
(reweighting has produced gains, added capacity has not).

Continuous spaces are gene-standardised over the lines present. Binary mutation
spaces are left as 0/1 and restricted to genes mutated in 2..n-2 lines, so
invariant and near-invariant columns cannot dominate a correlation distance.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def std(df):
    x = df.astype(float)
    x = x.loc[:, x.notna().all() & (x.std() > 1e-9)]
    return (x - x.mean()) / x.std()

def main(a):
    root = Path(a.depmap_root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = pd.read_csv(root / "tahoe_observed_Model.csv").dropna(subset=["RRID", "ModelID"])
    m2c = dict(zip(model.ModelID.astype(str), model.RRID.astype(str)))
    info = {}
    def emit(name, df):
        df = df.copy(); df.index = [m2c.get(str(i)) for i in df.index]
        df = df[[i is not None for i in df.index]]
        df = df[~df.index.duplicated()]
        df.to_parquet(out / f"{name}.parquet")
        info[name] = {"lines": int(len(df)), "features": int(df.shape[1])}
        print(f"{name:22s} {df.shape}", flush=True)

    emit("depmap_expression", std(pd.read_parquet(root / "tahoe_OmicsExpressionProteinCodingGenesTPMLogp1.parquet")))
    emit("cnv", std(pd.read_parquet(root / "tahoe_OmicsCNGene.parquet")))
    crispr = pd.read_parquet(root / "tahoe_CRISPRGeneEffect.parquet")
    emit("crispr_dependency", std(crispr.loc[:, crispr.notna().all()]))
    for nm, f in [("mutation_damaging", "tahoe_OmicsSomaticMutationsMatrixDamaging"),
                  ("mutation_hotspot", "tahoe_OmicsSomaticMutationsMatrixHotspot")]:
        d = (pd.read_parquet(root / (f + ".parquet")).fillna(0) > 0).astype(float)
        keep = d.sum() >= 2
        keep &= d.sum() <= len(d) - 2
        emit(nm, d.loc[:, keep])
    lin = pd.get_dummies(model.set_index("ModelID").OncotreeLineage.astype(str)).astype(float)
    emit("lineage", lin)
    json.dump(info, open(out / "feature_spaces.json", "w"), indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--depmap-root", default="/nfs-stor/yuqi.lei/Therapeutic/derived/depmap_24q4")
    p.add_argument("--out", default="/nfs-stor/yuqi.lei/Therapeutic/derived/cell_feature_spaces")
    main(p.parse_args())
