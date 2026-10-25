#!/usr/bin/env python3
"""Standardize every L1000 / Tahoe compound and compute chemistry + annotation features.

Unseen-drug branch, report/10_09_record.md. One table for both datasets, keyed by
the InChIKey of the RDKit-standardized parent molecule (largest organic fragment,
neutralized), so salts and batches of one compound collapse to one key and the
same key joins L1000 to Tahoe.

Outputs under <out>:
  drug_table.parquet   one row per source identifier (L1000 pert_id or Tahoe drug name):
                       source, source_id, name, smiles_std, inchikey, ik14, scaffold,
                       generic_scaffold, targets (';'-joined), moa (';'-joined)
  drug_features.npz    keyed by unique inchikey (sorted): morgan2 (uint8, 2048),
                       maccs (uint8, 167), rdkit2d (float32), rdkit2d_names
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
MISSING = {"-666", "", "nan", "None", "restricted"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def standardize(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    Chem.SanitizeMol(mol)
    return mol


def scaffolds(mol) -> tuple[str, str]:
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        plain = Chem.MolToSmiles(core)
        generic = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core))
    except Exception:  # noqa: BLE001 - rare valence failures on exotic cores
        plain, generic = "", ""
    return plain, generic


def split_list(value, commas: bool = True) -> list[str]:
    """Split an annotation cell. Target cells are comma lists ("EGFR, ERBB2"); MOA labels can
    contain commas ("Indoleamine 2,3-dioxygenase inhibitor"), so MOA splits on '|' / ';' only."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip().strip('"')
    if text in MISSING:
        return []
    text = text.replace("|", ";")
    if commas:
        text = text.replace(",", ";")
    parts = [p.strip() for p in text.split(";")]
    return sorted({p for p in parts if p and p not in MISSING})


def main(args: argparse.Namespace) -> None:
    print(f"host {platform.node()} python {sys.version.split()[0]} rdkit {rdBase.rdkitVersion} "
          f"numpy {np.__version__} pandas {pd.__version__}", flush=True)
    rows = []
    for accession in ("GSE92742", "GSE70138"):
        info = pd.read_csv(args.lincs_root / accession / "pert_info.txt.gz", sep="\t", low_memory=False)
        info = info[info["pert_type"].eq("trt_cp")]
        for r in info.itertuples():
            rows.append(("L1000", str(r.pert_id), str(r.pert_iname), str(r.canonical_smiles)))
    tahoe = pd.read_parquet(args.tahoe_drugs)
    for r in tahoe.itertuples():
        rows.append(("Tahoe", str(r.drug), str(r.drug), str(r.canonical_smiles)))
    table = pd.DataFrame(rows, columns=["source", "source_id", "name", "smiles_raw"])
    table = table.drop_duplicates(["source", "source_id"]).reset_index(drop=True)

    # Annotations: LINCS2020 compoundinfo (one row per pert_id x target) and Tahoe metadata.
    info = pd.read_csv(args.compoundinfo, sep="\t", low_memory=False)
    targets_by_pert: dict[str, set] = {}
    moa_by_pert: dict[str, set] = {}
    for r in info.itertuples():
        targets_by_pert.setdefault(str(r.pert_id), set()).update(split_list(r.target))
        moa_by_pert.setdefault(str(r.pert_id), set()).update(split_list(r.moa, commas=False))
    tahoe_targets = {str(r.drug): set(split_list(r.targets)) for r in tahoe.itertuples()}
    tahoe_moa = {str(d): {str(m)} - {"unclear", "None", "nan"}
                 for d, m in zip(tahoe["drug"], tahoe["moa-fine"])}

    std_smiles, keys, plain_scaf, generic_scaf, mols = [], [], [], [], {}
    for smi in table["smiles_raw"]:
        mol = None if smi in MISSING else standardize(smi)
        if mol is None:
            std_smiles.append(""); keys.append(""); plain_scaf.append(""); generic_scaf.append("")
            continue
        key = Chem.MolToInchiKey(mol)
        std_smiles.append(Chem.MolToSmiles(mol)); keys.append(key)
        p, g = scaffolds(mol)
        plain_scaf.append(p); generic_scaf.append(g)
        mols.setdefault(key, mol)
    table["smiles_std"] = std_smiles
    table["inchikey"] = keys
    table["ik14"] = [k[:14] for k in keys]
    table["scaffold"] = plain_scaf
    table["generic_scaffold"] = generic_scaf
    targets, moas = [], []
    for r in table.itertuples():
        if r.source == "L1000":
            t, m = targets_by_pert.get(r.source_id, set()), moa_by_pert.get(r.source_id, set())
        else:
            t, m = tahoe_targets.get(r.source_id, set()), tahoe_moa.get(r.source_id, set())
        targets.append(";".join(sorted(t))); moas.append(";".join(sorted(m)))
    table["targets"] = targets
    table["moa"] = moas
    print(f"{len(table):,} source rows; {int((table.inchikey == '').sum())} unparseable; "
          f"{table.inchikey[table.inchikey != ''].nunique():,} unique standardized InChIKeys", flush=True)

    unique = sorted(mols)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    names = [n for n, _ in Descriptors.descList]
    morgan = np.zeros((len(unique), 2048), dtype=np.uint8)
    maccs = np.zeros((len(unique), 167), dtype=np.uint8)
    desc = np.zeros((len(unique), len(names)), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, key in enumerate(unique):
            mol = mols[key]
            morgan[i] = morgan_gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
            maccs[i] = np.frombuffer(MACCSkeys.GenMACCSKeys(mol).ToBitString().encode(), dtype=np.uint8) - 48
            values = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan)
            desc[i] = np.array([values[n] for n in names], dtype=np.float64).clip(-1e6, 1e6)
            if i % 5000 == 0:
                print(f"  features {i:,}/{len(unique):,}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out / "drug_table.parquet", index=False)
    np.savez(args.out / "drug_features.npz", inchikey=np.asarray(unique, dtype=str), morgan2=morgan,
             maccs=maccs, rdkit2d=desc, rdkit2d_names=np.asarray(names, dtype=str))
    unique_smiles = (table[table.inchikey != ""].drop_duplicates("inchikey")[["inchikey", "smiles_std"]]
                     .sort_values("inchikey"))
    unique_smiles.to_csv(args.out / "unique_smiles.tsv", sep="\t", index=False)
    manifest = {
        "n_source_rows": len(table), "n_unique_inchikey": len(unique),
        "n_unparseable": int((table.inchikey == "").sum()),
        "n_with_targets": int((table.targets != "").sum()),
        "inputs": {str(p): sha256(p) for p in [args.lincs_root / "GSE92742/pert_info.txt.gz",
                                                args.lincs_root / "GSE70138/pert_info.txt.gz",
                                                args.tahoe_drugs, args.compoundinfo]},
        "runner_sha256": sha256(Path(__file__).resolve()), "rdkit": rdBase.rdkitVersion,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (args.out / "drug_features.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base = Path("/nfs-stor/yuqi.lei/Therapeutic")
    p.add_argument("--lincs-root", type=Path, default=base / "raw/lincs-l1000")
    p.add_argument("--tahoe-drugs", type=Path, default=base / "raw/tahoe-100m/metadata/drug_metadata.parquet")
    p.add_argument("--compoundinfo", type=Path, default=base / "raw/annotations/clue_lincs2020/compoundinfo_beta.txt")
    p.add_argument("--out", type=Path, default=base / "derived/l1000_unseen_drug")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
