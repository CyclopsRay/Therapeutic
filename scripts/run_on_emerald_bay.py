#!/usr/bin/env python
"""Run an existing Tahoe method script unchanged on EmeraldBay.

Usage:
    EB_DIR=<sanity output dir> python scripts/run_on_emerald_bay.py \
        scripts/run_rhaister_o_compact.py --output-root ... [script args]

Every method runner obtains its data through three loaders in
run_rhaister_o_compact: build_or_load_compact_cache, load_split and
load_hvg_centroids. This wrapper replaces those three on the imported module and
then executes the target script as __main__, so each script's own estimator,
inner selection, bootstrap and gates run exactly as on Tahoe.

EmeraldBay data (from scripts/emerald_bay_sanity.py):
  delta      mean log1p(count) of treated cells minus the same line's pooled
             DMSO_TF mean log1p(count) on the Tahoe 2K panel -- the Tahoe 2K
             cell_eval convention (log1p of raw X_hvg counts). Single-drug
             conditions only; combinations are excluded as in the paper.
  centroid   per-line pooled DMSO_TF mean raw counts on the panel. The Tahoe
             runner applies log1p to its centroid itself, and the Tahoe HVG
             centroid is a raw-count mean, so the two are like for like.
  plates     EmeraldBay has no plate column; every condition is assigned a
             single pseudo-plate 1, so "plate-matched" reduces to "same line,
             pooled DMSO".
  splits     split s in 5..9 holds out the Tahoe split-s holdout lines that are
             present in EmeraldBay (23 of Tahoe's 25; 4-5 per split). Every
             single-drug condition of a held-out line is a test condition. The
             five EmeraldBay lines outside Tahoe's 50 are always reference.
"""

from __future__ import annotations

import ast
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import run_rhaister_o_compact as roc  # noqa: E402

EB_DIR = Path(os.environ.get("EB_DIR", "/nfs-stor/yuqi.lei/Therapeutic/derived/emerald_bay/sanity_v1"))
TAHOE_SPLIT_ROOT = Path("/home/yuqi.lei/Therapeutic/external/Rhaister/splits/tahoe")
MIN_CELLS = int(os.environ.get("EB_MIN_CELLS", "10"))  # keeps the full 48x44 grid; higher thresholds drop exactly the drug-killed conditions (outcome-dependent selection)
# Lines with almost no surviving cells after 5-day culture (127-199 cells in
# total, 8-14 DMSO cells). Excluded on cell count alone, before any model is
# fit; all four also fail cross-dataset identity purely from sparsity.
LOW_COVERAGE = ("CVCL_1577", "CVCL_1119", "CVCL_1603", "CVCL_1056")
# EB_HALF_PLATES=1: expose the two seeded DMSO cell halves as pseudo-plates
# 101/102 so plate-based line-quality scores (own-plate reproducibility, used by
# run_reference_count_sweep.py) become DMSO split-half reproducibility. With a
# single pseudo-plate that score is undefined (NaN) and quality orderings are
# arbitrary. The query/reference centroid becomes the mean of the two half means
# instead of the pooled mean -- a negligible change, but results from this mode
# are only compared within the mode.
HALF_PLATES = os.environ.get("EB_HALF_PLATES", "0") == "1"
PB_DIR = Path(os.environ.get("EB_PB_DIR", "/nfs-stor/yuqi.lei/Therapeutic/derived/emerald_bay/pseudobulk_v1"))
EXCLUDE_LINES = set(filter(None, os.environ.get("EB_EXCLUDE_LINES", ",".join(LOW_COVERAGE)).split(",")))


def _usable(meta: pd.DataFrame) -> np.ndarray:
    """Single-drug, non-vehicle, enough cells, line not excluded."""
    return (meta.single.to_numpy()
            & ~meta.drug.str.startswith("dmso").to_numpy()
            & (meta.n_cells.to_numpy() >= MIN_CELLS)
            & ~meta.line.isin(EXCLUDE_LINES).to_numpy())


def _single(cond: str) -> bool:
    return len(ast.literal_eval(cond)) == 1


def eb_compact(data_root=None, repo_root=None, cache_root=None, rebuild=False) -> roc.CompactData:
    meta = pd.read_parquet(EB_DIR / "line_condition_meta.parquet")
    delta = np.load(EB_DIR / "delta_log1p.npy")
    genes = list(pd.read_parquet(EB_DIR / "dmso_meancount_by_line.parquet").columns)
    keep = _usable(meta)
    meta = meta[keep].reset_index(drop=True)
    delta = delta[keep]
    order = np.lexsort((meta.cond.to_numpy(), meta.line.to_numpy()))
    meta, delta = meta.iloc[order].reset_index(drop=True), delta[order]
    print(f"[emerald] compact: {len(meta):,} single-drug line-conditions, "
          f"{meta.line.nunique()} lines, {meta.cond.nunique()} conditions, min_cells={MIN_CELLS}, excluded_lines={sorted(EXCLUDE_LINES)}", flush=True)
    return roc.CompactData(
        cells=meta.line.astype(str).to_numpy(),
        treatments=meta.cond.astype(str).to_numpy(),
        plates=[(101, 102) if HALF_PLATES else (1,)] * len(meta),
        delta=delta.astype(np.float32),
        genes=genes,
    )


# EB_NEW_SPLITS=1 (pre-registered C2): hold out the EmeraldBay lines that the
# Tahoe splits never hold out, so the query lines are ones no EmeraldBay run has
# scored before. Five groups by seeded permutation.
NEW_SPLITS = os.environ.get("EB_NEW_SPLITS", "0") == "1"
NEW_SPLIT_SEED = int(os.environ.get("EB_NEW_SPLIT_SEED", "0"))


def eb_split(repo_root, split: int):
    tahoe_holdout, _, split_path = _orig_load_split(repo_root, split)
    meta = pd.read_parquet(EB_DIR / "line_condition_meta.parquet")
    meta = meta[_usable(meta)]
    eb_lines = set(meta.line)
    if NEW_SPLITS:
        ever = set()
        for s in range(5, 10):
            ever |= set(_orig_load_split(repo_root, s)[0])
        never = sorted(eb_lines - ever)
        order = np.random.default_rng(NEW_SPLIT_SEED).permutation(len(never))
        groups = np.array_split(order, 5)
        holdout = sorted(never[j] for j in groups[split - 5])
        print(f"[emerald] new split {split}: {holdout} (of {len(never)} never-held-out lines)", flush=True)
    else:
        holdout = [c for c in tahoe_holdout if c in eb_lines]
    tests = {c: set(meta.cond[meta.line == c]) for c in holdout}
    return holdout, tests, split_path


def _half_dmso_counts(genes) -> dict:
    groups = pd.read_parquet(PB_DIR / "groups.parquet")
    counts = np.load(PB_DIR / "sums_counts.npy", mmap_mode="r")
    pb_genes = pd.read_json(PB_DIR / "genes.json", typ="series").tolist()
    col = [pb_genes.index(g) for g in genes]
    out = {}
    for cell, block in groups[groups.drug == "DMSO_TF"].groupby("cell_line"):
        if cell in EXCLUDE_LINES:
            continue
        s = np.asarray(counts[block.index.to_numpy()]).sum(0)          # (2, P)
        n = block[["n_cells_h0", "n_cells_h1"]].to_numpy().sum(0)
        for h, plate in enumerate((101, 102)):
            out[(cell, plate)] = (s[h][col] / max(n[h], 1)).astype(np.float32)
    return out


def eb_hvg(data_root=None, genes=None):
    if HALF_PLATES:
        genes = genes if genes is not None else list(pd.read_parquet(EB_DIR / "dmso_meancount_by_line.parquet").columns)
        return _half_dmso_counts(list(genes))
    frame = pd.read_parquet(EB_DIR / "dmso_meancount_by_line.parquet")
    frame = frame.loc[[c for c in frame.index if c not in EXCLUDE_LINES]]
    if genes is not None:
        frame = frame[list(genes)]
    return {(cell, 1): frame.loc[cell].to_numpy(dtype=np.float32) for cell in frame.index}


_orig_load_split = roc.load_split
roc.build_or_load_compact_cache = eb_compact
roc.load_split = eb_split
roc.load_hvg_centroids = eb_hvg


if __name__ == "__main__":
    target = sys.argv[1]
    sys.argv = [target, *sys.argv[2:]]
    print(f"[emerald] EB_DIR={EB_DIR} running {target} {' '.join(sys.argv[1:])}", flush=True)
    if Path(target).name == "run_rhaister_o_compact.py":
        # run_path would re-execute the module in a fresh namespace and bypass
        # the patch; call the already-patched module's entry point instead.
        roc.main()
    else:
        runpy.run_path(target, run_name="__main__")
