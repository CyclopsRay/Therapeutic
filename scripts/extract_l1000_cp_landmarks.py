#!/usr/bin/env python3
"""Extract every compound (trt_cp) Level-5 signature x 978 landmark genes from L1000.

--pert-types extracts other perturbation types in the same decompression pass (2026-10-14: trt_sh.cgs
consensus knockdowns and trt_oe over-expressions, the measured causal gene -> gene prior; record
10_16 06:10). Each type gets its own <accession>_<pert_type>_landmarks.npz/.json; trt_cp output is unchanged.

Input for the unseen-drug branch (report/10_09_record.md). Unlike the earlier
L1000 extractions, which pulled a few Tahoe-matched lines, this keeps every
trt_cp signature at every dose and time, so the cohort can be defined downstream
without decompressing the archive again.

The GCTX archive is decompressed to scratch under /l, streamed in contiguous
blocks along the signature axis (memory stays bounded), and deleted. Output:

  <derived>/<accession>_trt_cp_landmarks.npz
      values    (n_sig, n_landmark) float32, Level-5 moderated z-scores
      sig_id    (n_sig,)  str, in GCTX column order
      gene_id   (n_landmark,) str, Entrez ids, gene_info landmark order
      gene_symbol (n_landmark,) str
  <derived>/<accession>_trt_cp_landmarks.json   provenance manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def decode(values: np.ndarray) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args: argparse.Namespace) -> None:
    started = time.time()
    root = args.raw_root / args.accession
    source = next(root.glob("*Level5*.gctx.gz"))
    args.derived.mkdir(parents=True, exist_ok=True)
    types = list(dict.fromkeys(args.pert_types))
    outs = {t: args.derived / f"{args.accession}_{t}_landmarks.npz" for t in types}

    print(f"host {platform.node()} python {sys.version.split()[0]} numpy {np.__version__} "
          f"h5py {h5py.__version__} pandas {pd.__version__}", flush=True)
    sig_info = pd.read_csv(root / "sig_info.txt.gz", sep="\t", low_memory=False)
    type_of = dict(zip(sig_info["sig_id"].astype(str), sig_info["pert_type"].astype(str)))
    wanted = {t: set(sig_info.loc[sig_info["pert_type"].eq(t), "sig_id"].astype(str)) for t in types}
    gene_info = pd.read_csv(root / "gene_info.txt.gz", sep="\t", low_memory=False)
    landmarks = gene_info[gene_info["pr_is_lm"].astype(int).eq(1)].copy()
    landmarks["pr_gene_id"] = landmarks["pr_gene_id"].astype(str)
    print(f"{args.accession}: " + ", ".join(f"{len(w):,} {t}" for t, w in wanted.items())
          + f" signatures requested, {len(landmarks)} landmarks", flush=True)

    work = args.derived / "work"
    work.mkdir(parents=True, exist_ok=True)
    temporary = work / f"{args.accession}.{os.environ.get('SLURM_JOB_ID', 'local')}.gctx"
    try:
        print(f"decompressing {source} ({source.stat().st_size:,} bytes)", flush=True)
        with temporary.open("wb") as handle:
            subprocess.run(["pigz", "-dc", "-p", str(args.threads), str(source)],
                           stdout=handle, check=True)
        print(f"decompressed {temporary.stat().st_size:,} bytes in "
              f"{time.time() - started:.0f}s", flush=True)
        with h5py.File(temporary, "r") as handle:
            rows = decode(handle["/0/META/ROW/id"][:])
            cols = decode(handle["/0/META/COL/id"][:])
            matrix = handle["/0/DATA/0/matrix"]
            print(f"matrix {matrix.shape} chunks {matrix.chunks} dtype {matrix.dtype}", flush=True)
            row_lookup = {g: i for i, g in enumerate(rows)}
            gene_ids = [g for g in landmarks["pr_gene_id"] if g in row_lookup]
            if len(gene_ids) != len(landmarks):
                raise ValueError(f"{len(landmarks) - len(gene_ids)} landmarks absent from GCTX")
            gene_pos = np.array([row_lookup[g] for g in gene_ids])
            col_type = np.array([type_of.get(c, "") for c in cols])
            keeps = {t: np.array([ct == t and c in wanted[t] for c, ct in zip(cols, col_type)]) for t in types}
            for t, k in keeps.items():
                if k.sum() != len(wanted[t]):
                    raise ValueError(f"{len(wanted[t]) - k.sum()} requested {t} signatures absent")
            keep = np.logical_or.reduce(list(keeps.values()))
            if matrix.shape == (len(cols), len(rows)):
                sig_axis = 0
            elif matrix.shape == (len(rows), len(cols)):
                sig_axis = 1
            else:
                raise ValueError(f"unexpected matrix shape {matrix.shape}")
            n_cols = len(cols)
            values = {t: np.empty((int(k.sum()), len(gene_pos)), dtype=np.float32) for t, k in keeps.items()}
            filled = {t: 0 for t in types}
            for start in range(0, n_cols, args.block):
                stop = min(start + args.block, n_cols)
                mask = keep[start:stop]
                if not mask.any():
                    continue
                if sig_axis == 0:
                    block = np.asarray(matrix[start:stop, :], dtype=np.float32)[:, gene_pos]
                else:
                    block = np.asarray(matrix[:, start:stop], dtype=np.float32)[gene_pos, :].T
                for t, k in keeps.items():
                    sub = block[k[start:stop]]
                    values[t][filled[t]:filled[t] + len(sub)] = sub
                    filled[t] += len(sub)
                print(f"  columns {stop:,}/{n_cols:,}, kept " + ", ".join(f"{t} {n:,}" for t, n in filled.items()),
                      flush=True)
            for t in types:
                assert filled[t] == len(values[t])
            sig_ids = {t: [c for c, k in zip(cols, keeps[t]) if k] for t in types}
            matrix_shape = list(matrix.shape)
        decompressed_bytes = temporary.stat().st_size
    finally:
        if temporary.exists():
            temporary.unlink()

    symbols = landmarks.set_index("pr_gene_id").loc[gene_ids, "pr_gene_symbol"].astype(str)
    for t in types:
        if not np.isfinite(values[t]).all():
            raise ValueError(f"non-finite values in extracted {t} block")
        out = outs[t]
        partial = out.with_name(f".{out.name}.tmp.npz")
        np.savez(partial, values=values[t], sig_id=np.asarray(sig_ids[t], dtype=str),
                 gene_id=np.asarray(gene_ids, dtype=str), gene_symbol=np.asarray(symbols, dtype=str))
        os.replace(partial, out)
        manifest = {
            "accession": args.accession, "pert_type": t, "source": str(source),
            "source_size_bytes": source.stat().st_size, "decompressed_size_bytes": decompressed_bytes,
            "gctx_matrix_shape": matrix_shape, "n_signatures": len(sig_ids[t]), "n_landmarks": len(gene_ids),
            "values_abs_max": float(np.abs(values[t]).max()), "output": str(out),
            "output_sha256": sha256(out), "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()), "seconds": round(time.time() - started, 1),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--accession", default="GSE92742", choices=["GSE92742", "GSE70138"])
    parser.add_argument("--raw-root", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/raw/lincs-l1000"))
    parser.add_argument("--derived", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/derived/l1000_unseen_drug"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--block", type=int, default=20000)
    parser.add_argument("--pert-types", nargs="+", default=["trt_cp"],
                        help="e.g. trt_sh.cgs trt_oe; trt_cp keeps its original output name")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
