#!/usr/bin/env python3
"""Tahoe replication slices for the unseen-drug benchmark (report/10_09_record.md, 2026-10-03).

Same format as build_l1000_unseen_drug.py so run_l1000_unseen_drug.py runs unchanged with
--source Tahoe. One slice per dose (T-0.05, T-0.5, T-5.0). A row is (compound, line) with the
frozen compact Rhaister delta over the fixed 2,000-gene panel; a compound is the standardized-
parent InChIKey of the Tahoe drug (drug names sharing a key are averaged). Tahoe has no TAS:
ACTIVE is pre-declared as a delta L2 norm at or above the median of that line at that dose.
`plates` lists the Tahoe plates of the condition ("plate6|plate14"); same-plate conditions
share a DMSO control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_rhaister_signal import parse_treatment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args: argparse.Namespace) -> None:
    obs = pd.read_parquet(args.processed / "observations.parquet")
    delta = np.load(args.processed / "delta.npy", mmap_mode="r")
    genes = json.loads(args.genes.read_text())
    table = pd.read_parquet(args.derived / "drug_table.parquet")
    key_of = dict(zip(table.loc[table.source.eq("Tahoe"), "source_id"], table.loc[table.source.eq("Tahoe"), "inchikey"]))
    parsed = [parse_treatment(t) for t in obs.treatment]
    obs = obs.assign(drug=[p[0] for p in parsed], dose=[p[1] for p in parsed], unit=[p[2] for p in parsed])
    obs["inchikey"] = obs.drug.map(key_of).fillna("")
    missing = sorted(obs.loc[obs.inchikey.eq(""), "drug"].unique())
    print(f"drugs without a standardized key (dropped): {missing}", flush=True)
    for dose in sorted(obs.dose.unique()):
        sub = obs[obs.dose.eq(dose) & obs.inchikey.ne("")].copy()
        sub["row"] = sub.index.to_numpy()
        groups = sub.groupby(["inchikey", "cell_line"], sort=True)
        keys = list(groups.groups.keys())
        Y = np.empty((len(keys), delta.shape[1]), np.float32)
        plates = []
        for i, (_, frame) in enumerate(groups):
            Y[i] = np.asarray(delta[frame.row.to_numpy()]).mean(0)
            plates.append("|".join(sorted({f"plate{p}" for ps in frame.plates.astype(str) for p in ps.split(",")})))
        lines = np.array([k[1] for k in keys])
        norm = np.linalg.norm(Y, axis=1)
        active = np.zeros(len(keys), bool)
        for L in np.unique(lines):
            m = lines == L
            active[m] = norm[m] >= np.median(norm[m])
        name = f"T-{dose:g}"
        out = args.derived / f"slice_{name}.npz"
        np.savez(out, Y=Y, inchikey=np.asarray([k[0] for k in keys], dtype=str), line=lines.astype(str),
                 tas=norm.astype(np.float64), active=active, cc=np.full(len(keys), np.nan), n_sig=np.ones(len(keys)),
                 plates=np.asarray(plates, dtype=str), gene_symbol=np.asarray(genes, dtype=str))
        meta = {"slice": name, "dose_uM": dose, "n_rows": len(keys), "n_compounds": len({k[0] for k in keys}),
                "n_lines": int(len(np.unique(lines))), "n_active": int(active.sum()), "dropped_drugs": missing,
                "output_sha256": sha256(out), "runner_sha256": sha256(Path(__file__).resolve())}
        (args.derived / f"slice_{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps({k: v for k, v in meta.items() if k != "dropped_drugs"}), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base = Path("/nfs-stor/yuqi.lei/Therapeutic")
    p.add_argument("--processed", type=Path, default=base / "processed/rhaister_o_compact")
    p.add_argument("--genes", type=Path, default=base / "raw/tahoe-de-rhaister/definition/static_2k_genes.json")
    p.add_argument("--derived", type=Path, default=base / "derived/l1000_unseen_drug")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
