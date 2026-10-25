#!/usr/bin/env python
"""Stream EmeraldBay single cells into (cell_line, sample, half) pseudobulk sums.

One EmeraldBay `sample` carries exactly one condition (checked on the first
shards), so (cell_line, sample) is the replicate unit. There is no plate
column; DMSO_TF samples are the vehicle controls.

Genes: the fixed Tahoe 2,000-gene panel (`static_2k_genes.json`), mapped via
Tahoe-100M token IDs, which EmeraldBay preserves verbatim for its first 62,710
tokens. Mapping by symbol would lose 27 genes whose EmeraldBay symbols are older
aliases (e.g. CCN2 = CTGF). EmeraldBay has no leading marker token: token IDs
start at 3 and every expression value is a non-negative integer count; both are
asserted per shard.

Per (group, half) three sums are accumulated over cells, where half is a
seeded per-cell coin flip used for split-half reproducibility:

  log1p     sum of log1p(count)                 -- Tahoe 2K cell_eval convention
                                                   (log1p of raw X_hvg counts)
  lognorm   sum of log1p(count * T / library)   -- normalize_total + log1p,
                                                   library over all genes, T = 1872
                                                   (Tahoe full-genome target_sum)
  counts    sum of raw counts                   -- for pseudobulk-level transforms

Outputs (versioned, under --out): groups.parquet, sums_<kind>.npy with shape
(n_groups, 2, 2000), genes.json, provenance.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TARGET_SUM = 1872.0
META_COLS = ["drug", "drugname_drugconc", "cell_line", "sample"]


def load_panel_tokens(panel_path: Path, tahoe_gene_meta: Path, eb_gene_meta: Path):
    panel = json.load(open(panel_path))
    tahoe = pd.read_parquet(tahoe_gene_meta)
    tok = dict(zip(tahoe.gene_symbol, tahoe.token_id))
    missing = [g for g in panel if g not in tok]
    if missing:
        raise SystemExit(f"panel genes missing from Tahoe metadata: {missing}")
    tokens = np.array([tok[g] for g in panel], dtype=np.int64)
    eb = pd.read_parquet(eb_gene_meta)
    eb_tok = set(eb.token_id)
    absent = [g for g, t in zip(panel, tokens) if t not in eb_tok]
    if absent:
        raise SystemExit(f"panel tokens absent from EmeraldBay vocabulary: {absent}")
    lookup = np.full(int(eb.token_id.max()) + 1, -1, dtype=np.int64)
    lookup[tokens] = np.arange(len(panel))
    return panel, lookup


def shard_meta(path: str) -> pd.DataFrame:
    return pq.read_table(path, columns=META_COLS).to_pandas()


def _init_worker():
    # One compute thread per worker: Arrow's default pool sizes itself to the
    # whole node, and 12 workers x node-wide pools thrashed (job 284833).
    pa.set_cpu_count(1)
    pa.set_io_thread_count(2)


def process_shard(args):
    path, group_index, lookup, seed = args
    started = time.time()
    n_panel = int(lookup.max()) + 1
    table = pq.read_table(path, columns=["genes", "expressions", "cell_line", "sample"])
    genes_col = table.column("genes").combine_chunks()
    expr_col = table.column("expressions").combine_chunks()
    offsets = genes_col.offsets.to_numpy()
    gv = genes_col.values.to_numpy()
    ev = expr_col.values.to_numpy()
    assert np.array_equal(offsets, expr_col.offsets.to_numpy()), path
    assert gv.min() >= 3, f"{path}: token < 3 (marker?)"
    assert (ev >= 0).all() and np.array_equal(ev, np.round(ev)), f"{path}: non-count values"

    n_cells = len(offsets) - 1
    keys = pd.Series(
        table.column("cell_line").to_pandas().astype(str) + "||" + table.column("sample").to_pandas().astype(str)
    )
    gid = keys.map(group_index).to_numpy()
    assert not pd.isna(gid).any(), path
    gid = gid.astype(np.int64)

    # Seeded per-cell half assignment, stable across reruns: hash of barcode would
    # be ideal, but shard + row order is fixed by the pinned revision.
    rng = np.random.default_rng([seed, zlib.crc32(os.path.basename(path).encode())])
    half = rng.integers(0, 2, size=n_cells)

    nnz = np.diff(offsets)
    cell_of_value = np.repeat(np.arange(n_cells), nnz)
    library = np.bincount(cell_of_value, weights=ev, minlength=n_cells)
    ngenes = nnz.astype(np.float64)

    local_groups, local_idx = np.unique(gid * 2 + half, return_inverse=True)
    n_local = len(local_groups)

    pidx = lookup[np.minimum(gv, len(lookup) - 1)]
    pidx[gv >= len(lookup)] = -1
    keep = pidx >= 0
    cell_k = cell_of_value[keep]
    gene_k = pidx[keep]
    cnt_k = ev[keep]
    flat = local_idx[cell_k] * n_panel + gene_k
    size = n_local * n_panel
    sums = {
        "log1p": np.bincount(flat, weights=np.log1p(cnt_k), minlength=size),
        "lognorm": np.bincount(flat, weights=np.log1p(cnt_k * TARGET_SUM / library[cell_k]), minlength=size),
        "counts": np.bincount(flat, weights=cnt_k, minlength=size),
    }
    sums = {k: v.reshape(n_local, n_panel) for k, v in sums.items()}
    stats = {
        "n_cells": np.bincount(local_idx, minlength=n_local),
        "library_sum": np.bincount(local_idx, weights=library, minlength=n_local),
        "ngenes_sum": np.bincount(local_idx, weights=ngenes, minlength=n_local),
        "panel_count_sum": sums["counts"].sum(axis=1),
    }
    return path, local_groups, sums, stats, time.time() - started


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--tahoe-gene-meta", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-shards", type=int, default=0, help="smoke test on the first N shards")
    args = ap.parse_args()

    t0 = time.time()
    print("host", platform.node(), "python", sys.version.split()[0], "numpy", np.__version__,
          "pandas", pd.__version__, "pyarrow", pa.__version__, "seed", args.seed, flush=True)
    shards = sorted(glob.glob(str(args.raw / "expression_data" / "*.parquet")))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    print("shards", len(shards), flush=True)

    panel, lookup = load_panel_tokens(args.panel, args.tahoe_gene_meta, args.raw / "metadata" / "gene_metadata.parquet")

    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as ex:
        metas = list(ex.map(shard_meta, shards))
    meta = pd.concat(metas, ignore_index=True)
    print("cells", len(meta), "lines", meta.cell_line.nunique(), "samples", meta["sample"].nunique(),
          "conditions", meta.drugname_drugconc.nunique(), flush=True)
    per_sample = meta.groupby("sample").drugname_drugconc.nunique()
    assert (per_sample == 1).all(), "a sample carries more than one condition"
    groups = (meta.groupby(["cell_line", "sample"], as_index=False)
              .agg(drug=("drug", "first"), condition=("drugname_drugconc", "first"), n_cells=("drug", "size")))
    groups["key"] = groups.cell_line + "||" + groups["sample"]
    group_index = dict(zip(groups.key, range(len(groups))))
    G, P = len(groups), len(panel)
    print("groups", G, "elapsed", round(time.time() - t0, 1), flush=True)

    sums = {k: np.zeros((G * 2, P)) for k in ("log1p", "lognorm", "counts")}
    stats = {k: np.zeros(G * 2) for k in ("n_cells", "library_sum", "ngenes_sum", "panel_count_sum")}
    jobs = [(s, group_index, lookup, args.seed) for s in shards]
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as ex:
        futures = [ex.submit(process_shard, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            path, lg, s, st, took = fut.result()
            for k in sums:
                sums[k][lg] += s[k]
            for k in stats:
                stats[k][lg] += st[k]
            print(f"  {i}/{len(jobs)} {os.path.basename(path)} {took:.1f}s (total {time.time() - t0:.0f}s)", flush=True)

    n_half = stats["n_cells"].reshape(G, 2)
    assert np.array_equal(n_half.sum(1), groups.n_cells.to_numpy()), "cell count mismatch"
    groups["n_cells_h0"], groups["n_cells_h1"] = n_half[:, 0].astype(int), n_half[:, 1].astype(int)
    groups["library_mean"] = stats["library_sum"].reshape(G, 2).sum(1) / groups.n_cells
    groups["ngenes_mean"] = stats["ngenes_sum"].reshape(G, 2).sum(1) / groups.n_cells
    groups["panel_fraction"] = stats["panel_count_sum"].reshape(G, 2).sum(1) / stats["library_sum"].reshape(G, 2).sum(1)

    args.out.mkdir(parents=True, exist_ok=True)
    groups.drop(columns="key").to_parquet(args.out / "groups.parquet", index=False)
    for k, v in sums.items():
        np.save(args.out / f"sums_{k}.npy", v.reshape(G, 2, P))
    json.dump(panel, open(args.out / "genes.json", "w"))
    json.dump({
        "raw": str(args.raw), "n_shards": len(shards), "n_cells": int(len(meta)), "n_groups": G,
        "seed": args.seed, "target_sum": TARGET_SUM, "panel": str(args.panel),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(time.time() - t0, 1),
    }, open(args.out / "provenance.json", "w"), indent=2)
    print("done", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
