#!/usr/bin/env python3
"""Subset release-pinned DepMap matrices to the observed Tahoe cell lines."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


MATRICES = (
    "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "CRISPRGeneEffect.csv",
    "OmicsCNGene.csv",
    "OmicsSomaticMutationsMatrixHotspot.csv",
    "OmicsSomaticMutationsMatrixDamaging.csv",
)
GENE_COLUMN = re.compile(r"^(?P<symbol>.+) \((?P<entrez_id>\d+)\)$")


def subset_matrix(path: Path, model_ids: set[str], output: Path) -> dict[str, object]:
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, index_col=0, chunksize=16, low_memory=False):
        chunk.index = chunk.index.astype(str)
        match = chunk.loc[chunk.index.isin(model_ids)]
        if not match.empty:
            selected.append(match)
    if not selected:
        raise ValueError(f"No Tahoe ModelIDs found in {path}")
    result = pd.concat(selected, axis=0)
    result = result.loc[~result.index.duplicated(keep="first")].sort_index()
    result.index.name = "ModelID"
    result.to_parquet(output, compression="zstd")
    return {
        "source": path.name,
        "rows": int(len(result)),
        "columns": int(result.shape[1]),
        "missing_model_ids": sorted(model_ids - set(result.index)),
        "output": str(output),
    }


def feature_map(path: Path, gene_sets: dict[str, set[str]]) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0, index_col=0).columns
    records = []
    for raw in columns:
        matched = GENE_COLUMN.match(raw)
        symbol = matched.group("symbol") if matched else raw
        entrez = int(matched.group("entrez_id")) if matched else None
        record: dict[str, object] = {
            "source": path.name,
            "raw_column": raw,
            "gene_symbol": symbol,
            "entrez_id": entrez,
        }
        for name, genes in gene_sets.items():
            record[name] = symbol in genes
        records.append(record)
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic")
    )
    args = parser.parse_args()
    raw = args.data_root / "raw" / "depmap" / "DepMap_Public_24Q4"
    output = args.data_root / "derived" / "depmap_24q4"
    output.mkdir(parents=True, exist_ok=True)

    crosswalk_path = args.data_root / "derived" / "metadata" / "tahoe_observed_cell_lines.csv"
    crosswalk = pd.read_csv(crosswalk_path)
    usable = crosswalk.loc[crosswalk["Cell_ID_DepMap"].notna()].copy()
    model_ids = set(usable["Cell_ID_DepMap"].astype(str))

    model = pd.read_csv(raw / "Model.csv", low_memory=False)
    model_subset = model.loc[model["ModelID"].astype(str).isin(model_ids)].copy()
    model_subset.to_csv(output / "tahoe_observed_Model.csv", index=False)

    rhaister_genes = set(
        json.loads(
            (
                args.data_root
                / "raw/tahoe-de-rhaister/definition/static_2k_genes.json"
            ).read_text()
        )
    )
    l1000 = pd.read_csv(
        args.data_root / "raw/lincs-l1000/GSE92742/gene_info.txt.gz", sep="\t"
    )
    landmark_genes = set(
        l1000.loc[l1000["pr_is_lm"].eq(1), "pr_gene_symbol"].astype(str)
    )
    gene_sets = {
        "in_rhaister_2k": rhaister_genes,
        "in_l1000_landmarks": landmark_genes,
    }

    reports = []
    maps = []
    for filename in MATRICES:
        path = raw / filename
        reports.append(subset_matrix(path, model_ids, output / f"tahoe_{path.stem}.parquet"))
        maps.append(feature_map(path, gene_sets))
    feature_maps = pd.concat(maps, ignore_index=True)
    feature_maps.to_csv(output / "gene_columns.csv", index=False)

    summary = {
        "release": "DepMap Public 24Q4",
        "tahoe_observed_lines": int(len(crosswalk)),
        "tahoe_lines_with_depmap_id": int(len(model_ids)),
        "model_metadata_rows": int(len(model_subset)),
        "missing_from_model_metadata": sorted(model_ids - set(model_subset["ModelID"])),
        "matrices": reports,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

