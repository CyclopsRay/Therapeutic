#!/usr/bin/env python3
"""Build the pre-registered L1000 unseen-drug slices (report/10_09_record.md, 2026-10-03).

A row is one (compound, line). A compound is the standardized-parent InChIKey from
build_drug_features.py; every Level-5 signature of every pert_id sharing that key in
that line, at the slice's time and dose, is averaged. Quality covariates (TAS,
distil_cc_q75) are averaged the same way.

Output: <derived>/slice_<name>.npz with
  Y (n_rows, 978) float32, inchikey, line, tas, cc, n_sig, gene_symbol, plates ('|'-joined
  sig_id plate prefixes, e.g. CPC006_A549_24H, used by the same-plate confound arm)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SLICES = {
    "P1-24": ("GSE92742", "24 h", ("10 µM",),
              ("PC3", "MCF7", "VCAP", "A549", "HA1E", "A375", "HCC515")),
    "P1-6": ("GSE92742", "6 h", ("10 µM",),
             ("PC3", "MCF7", "VCAP", "A549", "A375", "HT29", "HCC515", "HA1E", "HEPG2")),
    "P2-24": ("GSE70138", "24 h", ("10.0 um",),
              ("HT29", "MCF7", "PC3", "A375", "HA1E", "YAPC", "HELA")),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args: argparse.Namespace) -> None:
    table = pd.read_parquet(args.derived / "drug_table.parquet")
    key_of = dict(zip(table.loc[table.source.eq("L1000"), "source_id"],
                      table.loc[table.source.eq("L1000"), "inchikey"]))
    cache = {}
    for name, (accession, hours, doses, lines) in SLICES.items():
        if accession not in cache:
            with np.load(args.derived / f"{accession}_trt_cp_landmarks.npz", allow_pickle=False) as a:
                cache[accession] = (a["values"], a["sig_id"].astype(str), a["gene_symbol"].astype(str))
        values, sig_ids, genes = cache[accession]
        root = args.raw_root / accession
        info = pd.read_csv(root / "sig_info.txt.gz", sep="\t", low_memory=False)
        metrics = pd.read_csv(root / "sig_metrics.txt.gz", sep="\t", low_memory=False)
        info = info.merge(metrics[["sig_id", "tas", "distil_cc_q75"]], on="sig_id", how="left")
        sel = info[info.pert_type.eq("trt_cp") & info.pert_itime.eq(hours)
                   & info.pert_idose.isin(doses) & info.cell_id.isin(lines)].copy()
        sel["inchikey"] = sel.pert_id.astype(str).map(key_of).fillna("")
        dropped = int((sel.inchikey == "").sum())
        sel = sel[sel.inchikey != ""]
        position = pd.Series(np.arange(len(sig_ids)), index=sig_ids)
        sel["pos"] = position.reindex(sel.sig_id.astype(str)).to_numpy()
        if sel["pos"].isna().any():
            raise ValueError(f"{name}: {int(sel.pos.isna().sum())} signatures missing from extraction")
        groups = sel.groupby(["inchikey", "cell_id"], sort=True)
        keys = list(groups.groups.keys())
        Y = np.empty((len(keys), values.shape[1]), dtype=np.float32)
        tas, cc, n_sig = (np.empty(len(keys)) for _ in range(3))
        plates = []
        for i, (_, frame) in enumerate(groups):
            plates.append("|".join(sorted({sid.split(":")[0] for sid in frame.sig_id.astype(str)})))
            Y[i] = values[frame.pos.to_numpy(dtype=int)].mean(0)
            tas[i] = frame.tas.mean()
            cc[i] = frame.distil_cc_q75.replace(-666, np.nan).mean()
            n_sig[i] = len(frame)
        out = args.derived / f"slice_{name}.npz"
        np.savez(out, Y=Y, inchikey=np.asarray([k[0] for k in keys], dtype=str),
                 line=np.asarray([k[1] for k in keys], dtype=str), tas=tas, cc=cc, n_sig=n_sig,
                 plates=np.asarray(plates, dtype=str),
                 gene_symbol=genes)
        per_line = pd.DataFrame({"line": [k[1] for k in keys], "tas": tas})
        summary = per_line.groupby("line").agg(n=("tas", "size"), active=("tas", lambda x: int((x >= 0.2).sum())))
        print(f"{name}: {len(keys):,} rows, {len({k[0] for k in keys}):,} compounds, "
              f"{dropped} signatures without a standardized key dropped\n{summary.to_string()}", flush=True)
        (args.derived / f"slice_{name}.json").write_text(json.dumps({
            "slice": name, "accession": accession, "time": hours, "doses": doses, "lines": lines,
            "n_rows": len(keys), "n_compounds": len({k[0] for k in keys}),
            "n_signatures_used": int(len(sel)), "n_signatures_dropped_no_key": dropped,
            "output_sha256": sha256(out), "runner_sha256": sha256(Path(__file__).resolve()),
        }, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base = Path("/nfs-stor/yuqi.lei/Therapeutic")
    p.add_argument("--raw-root", type=Path, default=base / "raw/lincs-l1000")
    p.add_argument("--derived", type=Path, default=base / "derived/l1000_unseen_drug")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
