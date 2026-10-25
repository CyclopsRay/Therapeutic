#!/usr/bin/env python3
"""Prepare lightweight Tahoe/DepMap/LINCS crosswalks and condition metadata."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi


RDLogger.DisableLog("rdApp.error")


def normalized_name(value: object) -> str:
    """Conservative name key; intended for candidate matches, not final IDs."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def clean_identifier(value: object) -> str | None:
    text = str(value).strip()
    if not text or text.lower() == "nan" or text == "-666":
        return None
    return text


def pubchem_key(value: object) -> str | None:
    try:
        return str(int(float(str(value))))
    except (TypeError, ValueError):
        return None


def inchi_key_from_smiles(value: object) -> str | None:
    smiles = clean_identifier(value)
    if smiles is None:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return inchi.MolToInchiKey(molecule)


def parse_tahoe_treatment(value: str) -> tuple[str, float, str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or len(parsed) != 1 or len(parsed[0]) != 3:
        raise ValueError(f"Unexpected Tahoe treatment encoding: {value!r}")
    drug, dose, unit = parsed[0]
    return str(drug).strip(), float(dose), str(unit)


def load_tahoe(data_root: Path, output_root: Path) -> dict[str, object]:
    root = data_root / "raw" / "tahoe-100m"
    metadata = root / "metadata"
    sample_shard = root / "sample" / "train-00000-of-03388.parquet"

    cell_meta = pd.read_parquet(metadata / "cell_line_metadata.parquet")
    cell_meta = cell_meta.drop_duplicates("Cell_ID_Cellosaur")
    observed_ids = set(
        pd.read_parquet(sample_shard, columns=["cell_line_id"])["cell_line_id"]
    )
    observed_cells = (
        pd.DataFrame({"Cell_ID_Cellosaur": sorted(observed_ids)})
        .merge(cell_meta, how="left", on="Cell_ID_Cellosaur", validate="one_to_one")
        .sort_values("cell_name")
    )
    if observed_cells["cell_name"].isna().any():
        raise ValueError("An observed Tahoe cell line is missing from cell metadata")
    observed_cells.to_csv(output_root / "tahoe_observed_cell_lines.csv", index=False)

    samples = pd.read_parquet(metadata / "sample_metadata.parquet")
    parsed = samples["drugname_drugconc"].map(parse_tahoe_treatment)
    parsed_frame = pd.DataFrame(
        parsed.tolist(), columns=["parsed_drug", "dose", "dose_unit"], index=samples.index
    )
    samples = pd.concat([samples, parsed_frame], axis=1)
    if not (samples["drug"].str.strip() == samples["parsed_drug"]).all():
        raise ValueError("Parsed Tahoe treatment names disagree with the drug column")
    samples.to_csv(output_root / "tahoe_sample_conditions.csv", index=False)

    genes = pd.read_parquet(metadata / "gene_metadata.parquet")
    drugs = pd.read_parquet(metadata / "drug_metadata.parquet")
    return {
        "cells": observed_cells,
        "genes": genes,
        "drugs": drugs,
        "samples": samples,
    }


def lincs_overlap(
    accession: str,
    data_root: Path,
    output_root: Path,
    tahoe: dict[str, object],
) -> dict[str, object]:
    root = data_root / "raw" / "lincs-l1000" / accession
    sig = pd.read_csv(root / "sig_info.txt.gz", sep="\t", low_memory=False)
    pert = pd.read_csv(root / "pert_info.txt.gz", sep="\t", low_memory=False)
    genes = pd.read_csv(root / "gene_info.txt.gz", sep="\t", low_memory=False)

    chemical = sig.loc[sig["pert_type"].eq("trt_cp")].copy()
    tahoe_cells = tahoe["cells"].copy()
    tahoe_cells["name_key"] = tahoe_cells["cell_name"].map(normalized_name)
    lincs_cells = pd.DataFrame({"lincs_cell_id": sorted(chemical["cell_id"].astype(str).unique())})
    lincs_cells["name_key"] = lincs_cells["lincs_cell_id"].map(normalized_name)
    cell_matches = tahoe_cells.merge(lincs_cells, how="inner", on="name_key")
    cell_matches.insert(0, "lincs_accession", accession)
    cell_matches.to_csv(output_root / f"{accession}_tahoe_cell_overlap.csv", index=False)

    tahoe_drugs = (
        tahoe["drugs"][["drug", "pubchem_cid", "canonical_smiles"]]
        .drop_duplicates("drug")
        .copy()
    )
    tahoe_drugs["name_key"] = tahoe_drugs["drug"].map(normalized_name)
    tahoe_drugs["pubchem_key"] = tahoe_drugs["pubchem_cid"].map(pubchem_key)
    tahoe_drugs["inchi_key"] = tahoe_drugs["canonical_smiles"].map(
        inchi_key_from_smiles
    )

    pert = pert.loc[pert["pert_type"].eq("trt_cp")].copy()
    pert = pert.loc[pert["pert_id"].astype(str).isin(set(chemical["pert_id"].astype(str)))]
    pert = pert.drop_duplicates("pert_id")
    lincs_drugs = pert[["pert_id", "pert_iname", "canonical_smiles", "inchi_key"]].copy()
    lincs_drugs["pert_id"] = lincs_drugs["pert_id"].astype(str)
    lincs_drugs["name_key"] = lincs_drugs["pert_iname"].map(normalized_name)
    if "pubchem_cid" in pert.columns:
        lincs_drugs["pubchem_key"] = pert["pubchem_cid"].map(pubchem_key)
    else:
        lincs_drugs["pubchem_key"] = None
    supplied_inchi = lincs_drugs["inchi_key"].map(clean_identifier)
    derived_inchi = lincs_drugs["canonical_smiles"].map(inchi_key_from_smiles)
    lincs_drugs["inchi_key"] = supplied_inchi.fillna(derived_inchi)

    match_parts = []
    for method, key in (
        ("name", "name_key"),
        ("pubchem_cid", "pubchem_key"),
        ("inchi_key", "inchi_key"),
    ):
        left = tahoe_drugs.loc[tahoe_drugs[key].notna(), ["drug", key]]
        right = lincs_drugs.loc[lincs_drugs[key].notna(), ["pert_id", key]]
        part = left.merge(right, how="inner", on=key)[["drug", "pert_id"]]
        part["match_method"] = method
        match_parts.append(part)
    pair_methods = pd.concat(match_parts, ignore_index=True).drop_duplicates()
    pair_methods = (
        pair_methods.groupby(["drug", "pert_id"], as_index=False)["match_method"]
        .agg(lambda values: ",".join(sorted(set(values))))
    )
    drug_crosswalk = (
        pair_methods.merge(tahoe_drugs, how="left", on="drug", validate="many_to_one")
        .merge(
            lincs_drugs,
            how="left",
            on="pert_id",
            validate="many_to_one",
            suffixes=("_tahoe", "_lincs"),
        )
        .sort_values(["drug", "pert_id"])
    )
    drug_crosswalk.insert(0, "lincs_accession", accession)
    drug_crosswalk["structure_validated"] = drug_crosswalk["match_method"].str.contains(
        "pubchem_cid|inchi_key", regex=True
    )
    drug_crosswalk.to_csv(
        output_root / f"{accession}_tahoe_drug_crosswalk.csv", index=False
    )
    name_matches = drug_crosswalk.loc[
        drug_crosswalk["match_method"].str.contains("name")
    ].copy()
    name_matches.to_csv(
        output_root / f"{accession}_tahoe_drug_name_overlap.csv", index=False
    )

    landmark_symbols = set(
        genes.loc[genes["pr_is_lm"].eq(1), "pr_gene_symbol"].dropna().astype(str)
    )
    tahoe_genes = tahoe["genes"].copy()
    gene_matches = tahoe_genes.loc[
        tahoe_genes["gene_symbol"].astype(str).isin(landmark_symbols)
    ].drop_duplicates("gene_symbol")
    gene_matches.to_csv(output_root / f"{accession}_tahoe_landmark_genes.csv", index=False)

    return {
        "accession": accession,
        "all_signatures": int(len(sig)),
        "chemical_signatures": int(len(chemical)),
        "chemical_cell_lines": int(chemical["cell_id"].nunique()),
        "chemical_perturbagens": int(chemical["pert_iname"].nunique()),
        "gene_rows": int(len(genes)),
        "landmark_genes": int(len(landmark_symbols)),
        "tahoe_cell_name_matches": int(len(cell_matches)),
        "tahoe_drug_name_matches": int(name_matches["drug"].nunique()),
        "tahoe_drug_structure_matches": int(
            drug_crosswalk.loc[drug_crosswalk["structure_validated"], "drug"].nunique()
        ),
        "tahoe_drug_any_matches": int(drug_crosswalk["drug"].nunique()),
        "tahoe_landmark_gene_matches": int(len(gene_matches)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic")
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output_root = args.output_root or args.data_root / "derived" / "metadata"
    output_root.mkdir(parents=True, exist_ok=True)

    tahoe = load_tahoe(args.data_root, output_root)
    lincs = [
        lincs_overlap(accession, args.data_root, output_root, tahoe)
        for accession in ("GSE92742", "GSE70138")
    ]
    summary = {
        "tahoe": {
            "observed_cell_lines": int(len(tahoe["cells"])),
            "observed_cell_lines_with_depmap_id": int(
                tahoe["cells"]["Cell_ID_DepMap"].notna().sum()
            ),
            "metadata_cell_lines": int(
                pd.read_parquet(
                    args.data_root
                    / "raw/tahoe-100m/metadata/cell_line_metadata.parquet",
                    columns=["Cell_ID_Cellosaur"],
                )["Cell_ID_Cellosaur"].nunique()
            ),
            "drugs": int(tahoe["drugs"]["drug"].nunique()),
            "gene_rows": int(len(tahoe["genes"])),
            "samples": int(len(tahoe["samples"])),
            "plates": int(tahoe["samples"]["plate"].nunique()),
            "dmso_samples": int(tahoe["samples"]["drug"].eq("DMSO_TF").sum()),
            "dose_counts": {
                str(k): int(v)
                for k, v in tahoe["samples"]["dose"].value_counts().sort_index().items()
            },
        },
        "lincs": lincs,
        "notes": [
            "Drug crosswalks report name, PubChem CID, and exact InChIKey evidence; inspect one-to-many mappings and salt forms before modeling.",
            "Observed Tahoe lines are inferred from the downloaded expression shard; every Mosaic well contains the cell village.",
        ],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
