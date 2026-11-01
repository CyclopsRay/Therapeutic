#!/usr/bin/env python3
"""Arrays for the Bai et al. 2026 drug-blind L1000 benchmark (XPert L1000 SDST, 78,453 pairs).

Reads the raw h5ad with h5py only (no pickle), and writes float32 control / delta matrices, per-sample drug and cell
indices, the benchmark's drug-blind group of every sample, and a drug table. Groups follow
drug-perturbation-benchmark experiments/main_benchmark/data_prep/create_splits.py: unique pert_idx in order of first
appearance, RandomState(131419).shuffle, np.array_split into 5; fold f tests group f and validates on group (f+1) % 5.
Sample and drug counts per fold are checked against the benchmark's A_split_integrity.csv.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

RAW = Path("/nfs-stor/yuqi.lei/Therapeutic/raw/xpert_sdst/processed_data")
OUT = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
INTEGRITY = Path(__file__).resolve().parents[1] / ("repro/dpb/upstream/data/analysis/robustness_and_diagnostics/"
                                                    "training_seeds/A_split_integrity.csv")
SEED = 131419


def categorical(g: h5py.Group) -> np.ndarray:
    cats = g["categories"][:].astype(str)
    return cats[g["codes"][:]]


def plates(h5: Path) -> None:
    """Sparse (sample, detection plate) incidence for the plate-confound control; det_plate lists ';'-joined plates."""
    with h5py.File(h5, "r") as f:
        pl = categorical(f["obs/det_plate"]); proj = categorical(f["obs/project_code"])
    vocab, rows, cols = {}, [], []
    for i, s in enumerate(pl):
        for x in s.split(";"):
            rows.append(i); cols.append(vocab.setdefault(x, len(vocab)))
    pv, prow, pcol = {}, [], []
    for i, s in enumerate(proj):
        for x in s.split(";"):
            prow.append(i); pcol.append(pv.setdefault(x, len(pv)))
    np.savez(OUT / "plates.npz", plate_row=np.array(rows), plate_col=np.array(cols), n_plates=len(vocab),
             proj_row=np.array(prow), proj_col=np.array(pcol), n_proj=len(pv))
    print(f"plates: {len(vocab)} plates, {len(rows)} incidences; projects: {len(pv)}", flush=True)


def main() -> None:
    h5 = RAW / "l1000_sdst_78453.h5ad"
    if "--plates-only" in __import__("sys").argv:
        return plates(h5)
    with h5py.File(h5, "r") as f:
        pert = f["X"][:].astype(np.float32)
        ctl = f["obsm/X_ctl"][:].astype(np.float32)
        obs = f["obs"]
        pert_idx = obs["pert_idx"][:]
        cell_idx = categorical(obs["cell_idx"]).astype(int)
        cell = categorical(obs["cell_iname"])
        pert_id = categorical(obs["pert_id"])
        genes = f["var/gene_symbol"][:].astype(str) if isinstance(f["var/gene_symbol"], h5py.Dataset) \
            else categorical(f["var/gene_symbol"])
    drugs = pd.unique(pert_idx)
    rng = np.random.RandomState(SEED)
    rng.shuffle(drugs)
    group_of = {d: g for g, grp in enumerate(np.array_split(drugs, 5)) for d in grp}
    group = np.array([group_of[d] for d in pert_idx], np.int8)

    ref = pd.read_csv(INTEGRITY)
    for r in ref.itertuples():
        te = group == r.fold; va = group == (r.fold + 1) % 5; tr = ~te & ~va
        got = (tr.sum(), va.sum(), te.sum(), len(set(pert_idx[tr])), len(set(pert_idx[va])), len(set(pert_idx[te])))
        want = (r.n_train, r.n_validation, r.n_test, r.n_train_drugs, r.n_validation_drugs, r.n_test_drugs)
        assert got == want, (r.fold, got, want)
    print(f"split reproduced: {len(drugs)} drugs; per-fold counts match A_split_integrity.csv", flush=True)

    # The info CSV has one row per (pert_id, target); pert_idx is compound-level (several BRD aliases share one).
    info = pd.read_csv(RAW / "l1000_sdst_drugs_info_8432.csv", keep_default_na=False)
    node = info.groupby("pert_id").drug_node_idx.agg(lambda s: sorted(set(s)))
    assert all(len(v) == 1 for v in node), "pert_id maps to several drug_node_idx"
    sample_node = node.reindex(pert_id).map(lambda v: v[0] if isinstance(v, list) else -1).to_numpy()
    assert (sample_node == pert_idx).all(), f"drug_node_idx != pert_idx for {(sample_node != pert_idx).sum()} samples"
    union = lambda s: ";".join(sorted({x for v in s for x in v.split("|") if x}))  # noqa: E731
    first = lambda s: next((x for x in s if x), "")  # noqa: E731
    table = (info[info.drug_node_idx.isin(drugs)].groupby("drug_node_idx")
             .agg(pert_id=("pert_id", lambda s: ";".join(sorted(set(s)))), cmap_name=("cmap_name", first),
                  target=("target", union), moa=("moa", union), canonical_smiles=("canonical_smiles", first),
                  inchi_key=("inchi_key", first))
             .rename_axis("pert_idx").reset_index())
    assert len(table) == len(drugs)
    print(f"drug table: {len(table)} drugs; smiles {table.canonical_smiles.ne('').sum()}, target "
          f"{table.target.ne('').sum()}, moa {table.moa.ne('').sum()}, aliased {table.pert_id.str.contains(';').sum()}",
          flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "sdst.npz", ctl=ctl, delta=pert - ctl, pert_idx=pert_idx, cell_idx=cell_idx, cell=cell,
             group=group, genes=genes)
    table.to_csv(OUT / "drugs.tsv", sep="\t", index=False)
    manifest = {"source": str(h5), "source_sha256": hashlib.sha256(h5.read_bytes()).hexdigest(),
                "n": int(len(pert_idx)), "drugs": int(len(drugs)), "cells": int(len(set(cell_idx))), "seed": SEED,
                "split_rule": "create_splits.py @ drug-perturbation-benchmark 55edbfe"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
