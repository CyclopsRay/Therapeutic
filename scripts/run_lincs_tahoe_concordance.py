#!/usr/bin/env python3
"""Small external LINCS L1000 versus Tahoe response-concordance experiment.

This is an external validation analysis, not a Tahoe training-data merge.  It
selects structure-validated compound matches in exact shared cell lines, keeps
24-hour LINCS Level-5 compound signatures, and matches the nearest LINCS dose
to each Tahoe dose only when the ratio is at most two.  It then extracts only
the required signatures and Tahoe/Rhaister genes from the official GCTX files.

The primary panel is the directly measured L1000 landmarks that also occur in
Rhaister's fixed Tahoe 2K panel.  BING and all-distributed overlaps are saved as
clearly labelled secondary panels.  Concordance is tested against a drug-label
permutation within cell line, Tahoe dose, and accession.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from evaluate_rhaister_signal import parse_treatment
from run_depmap_expression_ridge import atomic_csv, sha256, write_json
from run_rhaister_o_compact import rowwise_metrics


PROJECT_ROOT = Path("/home/yuqi.lei/Therapeutic")
BULK_ROOT = Path("/nfs-stor/yuqi.lei/Therapeutic")
RAW_ROOT = BULK_ROOT / "raw/lincs-l1000"
METADATA_ROOT = BULK_ROOT / "derived/metadata"
COMPACT_ROOT = BULK_ROOT / "processed/rhaister_o_compact"
BASELINE_ROOT = BULK_ROOT / "results/rhaister_o_compact"
DERIVED_ROOT = BULK_ROOT / "derived/lincs_tahoe_concordance"
OUTPUT_ROOT = BULK_ROOT / "results/lincs_tahoe_concordance"
ACCESSIONS = ("GSE92742", "GSE70138")
PANELS = ("landmark_measured", "bing", "all_distributed")


def parse_lincs_dose_um(frame: pd.DataFrame) -> np.ndarray:
    """Parse Phase-I numeric or Phase-II textual doses into micromolar."""
    if "pert_dose" in frame.columns:
        dose = pd.to_numeric(frame["pert_dose"], errors="coerce").to_numpy(float)
        units = frame.get("pert_dose_unit", pd.Series("uM", index=frame.index))
    else:
        text = frame["pert_idose"].astype(str)
        dose = pd.to_numeric(
            text.str.extract(r"([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")[0],
            errors="coerce",
        ).to_numpy(float)
        units = text
    normalized = units.astype(str).str.lower().str.replace("μ", "u").str.replace("µ", "u")
    dose[normalized.str.contains(r"\bnm\b", regex=True, na=False).to_numpy()] /= 1000.0
    dose[normalized.str.contains(r"\bmm\b", regex=True, na=False).to_numpy()] *= 1000.0
    return dose


def is_24_hour(frame: pd.DataFrame) -> np.ndarray:
    if "pert_time" in frame.columns:
        time = pd.to_numeric(frame["pert_time"], errors="coerce").to_numpy(float)
        unit = frame.get("pert_time_unit", pd.Series("h", index=frame.index)).astype(str)
        return np.isclose(time, 24.0) & unit.str.lower().str.startswith("h").to_numpy()
    return frame["pert_itime"].astype(str).str.lower().str.match(
        r"^24(?:\.0)?\s*h"
    ).to_numpy()


def build_eligibility(args: argparse.Namespace, accession: str) -> pd.DataFrame:
    observations = pd.read_parquet(
        args.compact_root / "observations.parquet", columns=["cell_line", "treatment"]
    ).drop_duplicates()
    parsed = observations["treatment"].map(parse_treatment)
    observations["drug"] = [item[0] for item in parsed]
    observations["tahoe_dose_um"] = [float(item[1]) for item in parsed]

    signatures = pd.read_csv(
        args.raw_root / accession / "sig_info.txt.gz", sep="\t", low_memory=False
    )
    signatures = signatures[
        signatures["pert_type"].astype(str).eq("trt_cp") & is_24_hour(signatures)
    ].copy()
    signatures["lincs_dose_um"] = parse_lincs_dose_um(signatures)
    signatures = signatures[
        np.isfinite(signatures["lincs_dose_um"]) & (signatures["lincs_dose_um"] > 0)
    ]

    crosswalk = pd.read_csv(
        args.metadata_root / f"{accession}_tahoe_drug_crosswalk.csv"
    )
    validated = crosswalk["structure_validated"].astype(str).str.lower().eq("true")
    crosswalk = crosswalk[validated].drop_duplicates(["drug", "pert_id"])
    cells = pd.read_csv(
        args.metadata_root / f"{accession}_tahoe_cell_overlap.csv"
    )[["Cell_ID_Cellosaur", "cell_name", "lincs_cell_id"]].rename(
        columns={"Cell_ID_Cellosaur": "tahoe_cell", "cell_name": "tahoe_cell_name"}
    )

    eligible = (
        signatures.merge(cells, left_on="cell_id", right_on="lincs_cell_id")
        .merge(
            crosswalk[["drug", "pert_id", "match_method", "structure_validated"]],
            on="pert_id",
        )
        .merge(
            observations,
            left_on=["tahoe_cell", "drug"],
            right_on=["cell_line", "drug"],
        )
    )
    eligible["abs_log2_dose_ratio"] = np.abs(
        np.log2(eligible["lincs_dose_um"] / eligible["tahoe_dose_um"])
    )
    eligible = eligible[
        eligible["abs_log2_dose_ratio"] <= np.log2(args.max_dose_ratio) + 1e-12
    ].copy()
    condition_keys = ["tahoe_cell", "drug", "tahoe_dose_um"]
    nearest = eligible.groupby(condition_keys)["abs_log2_dose_ratio"].transform("min")
    eligible = eligible[
        np.isclose(eligible["abs_log2_dose_ratio"], nearest, rtol=0, atol=1e-10)
    ].drop_duplicates(["sig_id", *condition_keys])
    eligible["dose_ratio"] = np.maximum(
        eligible["lincs_dose_um"] / eligible["tahoe_dose_um"],
        eligible["tahoe_dose_um"] / eligible["lincs_dose_um"],
    )
    eligible.insert(0, "accession", accession)
    columns = [
        "accession",
        "sig_id",
        "pert_id",
        "pert_iname",
        "lincs_cell_id",
        "tahoe_cell",
        "tahoe_cell_name",
        "drug",
        "treatment",
        "tahoe_dose_um",
        "lincs_dose_um",
        "dose_ratio",
        "match_method",
        "structure_validated",
    ]
    return eligible[columns].sort_values(condition_keys + ["sig_id"]).reset_index(drop=True)


def decode_ids(values: np.ndarray) -> list[str]:
    result = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def published_sha512(accession_root: Path, filename: str) -> str:
    with gzip.open(accession_root / "SHA512SUMS.txt.gz", "rt", errors="replace") as handle:
        for line in handle:
            if filename in line:
                return line.split()[0]
    raise ValueError(f"No published SHA-512 entry for {filename}")


def extract_gctx(
    args: argparse.Namespace,
    accession: str,
    eligible: pd.DataFrame,
    static_genes: list[str],
) -> dict:
    accession_root = args.raw_root / accession
    source_path = next(accession_root.glob("*.gctx.gz"))
    output_path = args.derived_root / f"{accession}_selected_signatures.npz"
    manifest_path = args.derived_root / f"{accession}_extraction.json"
    args.derived_root.mkdir(parents=True, exist_ok=True)

    selected_signatures = sorted(eligible["sig_id"].astype(str).unique())
    gene_info = pd.read_csv(accession_root / "gene_info.txt.gz", sep="\t")
    gene_info["pr_gene_symbol"] = gene_info["pr_gene_symbol"].astype(str)
    gene_info["pr_gene_id"] = gene_info["pr_gene_id"].astype(str)
    by_symbol = gene_info.drop_duplicates("pr_gene_symbol").set_index("pr_gene_symbol")
    selected_genes = [gene for gene in static_genes if gene in by_symbol.index]
    selected_gene_info = by_symbol.loc[selected_genes]

    if output_path.exists() and manifest_path.exists():
        with np.load(output_path, allow_pickle=False) as archive:
            cached_signatures = archive["sig_id"].astype(str).tolist()
            cached_genes = archive["gene_symbol"].astype(str).tolist()
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            set(cached_signatures) == set(selected_signatures)
            and cached_genes == selected_genes
            and manifest.get("source_size_bytes") == source_path.stat().st_size
        ):
            print(f"{accession}: using cached extraction {output_path}", flush=True)
            return manifest

    work_root = args.derived_root / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    temporary_gctx = work_root / f"{accession}.{os.environ.get('SLURM_JOB_ID', 'local')}.gctx"
    if temporary_gctx.exists():
        temporary_gctx.unlink()
    try:
        print(
            f"{accession}: decompressing {source_path.name} ({source_path.stat().st_size:,} bytes)",
            flush=True,
        )
        with temporary_gctx.open("wb") as destination:
            subprocess.run(
                ["pigz", "-dc", "-p", str(args.threads), str(source_path)],
                stdout=destination,
                check=True,
            )
        print(
            f"{accession}: decompressed size {temporary_gctx.stat().st_size:,} bytes",
            flush=True,
        )
        with h5py.File(temporary_gctx, "r") as handle:
            row_ids = decode_ids(handle["/0/META/ROW/id"][:])
            column_ids = decode_ids(handle["/0/META/COL/id"][:])
            matrix = handle["/0/DATA/0/matrix"]
            row_lookup = {value: index for index, value in enumerate(row_ids)}
            column_lookup = {value: index for index, value in enumerate(column_ids)}
            missing_genes = [
                gene for gene, gene_id in zip(selected_genes, selected_gene_info["pr_gene_id"])
                if str(gene_id) not in row_lookup
            ]
            missing_signatures = [sig for sig in selected_signatures if sig not in column_lookup]
            if missing_genes or missing_signatures:
                raise ValueError(
                    f"{accession}: GCTX misses {len(missing_genes)} genes and "
                    f"{len(missing_signatures)} signatures"
                )
            gene_positions = np.array([
                row_lookup[str(gene_id)] for gene_id in selected_gene_info["pr_gene_id"]
            ])
            pairs = sorted((column_lookup[sig], sig) for sig in selected_signatures)
            column_positions = np.array([pair[0] for pair in pairs])
            ordered_signatures = [pair[1] for pair in pairs]
            if matrix.shape == (len(row_ids), len(column_ids)):
                block = np.asarray(matrix[:, column_positions], dtype=np.float32).T
            elif matrix.shape == (len(column_ids), len(row_ids)):
                block = np.asarray(matrix[column_positions, :], dtype=np.float32)
            else:
                raise ValueError(
                    f"Unexpected matrix {matrix.shape}, rows={len(row_ids)}, cols={len(column_ids)}"
                )
            values = block[:, gene_positions]
            matrix_shape = list(matrix.shape)

        temporary_output = output_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary_output,
            accession=np.asarray([accession], dtype=str),
            sig_id=np.asarray(ordered_signatures, dtype=str),
            gene_symbol=np.asarray(selected_genes, dtype=str),
            gene_id=np.asarray(selected_gene_info["pr_gene_id"].astype(str), dtype=str),
            is_landmark=selected_gene_info["pr_is_lm"].to_numpy(dtype=bool),
            is_bing=selected_gene_info["pr_is_bing"].to_numpy(dtype=bool),
            values=values,
        )
        os.replace(temporary_output, output_path)
        manifest = {
            "accession": accession,
            "source": str(source_path),
            "source_size_bytes": source_path.stat().st_size,
            "published_sha512": published_sha512(accession_root, source_path.name),
            "decompressed_size_bytes": temporary_gctx.stat().st_size,
            "gctx_matrix_shape": matrix_shape,
            "n_selected_signatures": len(ordered_signatures),
            "n_selected_genes": len(selected_genes),
            "n_landmark_genes": int(selected_gene_info["pr_is_lm"].sum()),
            "n_bing_genes": int(selected_gene_info["pr_is_bing"].sum()),
            "output": str(output_path),
        }
        write_json(manifest, manifest_path)
        print(
            f"{accession}: extracted {len(ordered_signatures)} signatures x "
            f"{len(selected_genes)} genes",
            flush=True,
        )
        return manifest
    finally:
        if temporary_gctx.exists():
            temporary_gctx.unlink()
        if work_root.exists() and not any(work_root.iterdir()):
            work_root.rmdir()


def rank_rows(values: np.ndarray) -> np.ndarray:
    """Fast ranks; Level-5/Tahoe floating values have negligible exact ties."""
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(values, dtype=np.float64)
    rows = np.arange(len(values))[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=np.float64)[None, :]
    return ranks


def cross_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    pearson = rowwise_metrics(truth, prediction)["pearson"]
    spearman = rowwise_metrics(rank_rows(truth), rank_rows(prediction))["pearson"]
    truth64 = np.asarray(truth, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    cosine_denominator = np.linalg.norm(truth64, axis=1) * np.linalg.norm(
        prediction64, axis=1
    )
    cosine = np.divide(
        np.sum(truth64 * prediction64, axis=1),
        cosine_denominator,
        out=np.zeros(len(truth64)),
        where=cosine_denominator > 0,
    )
    valid_sign = (np.abs(truth64) > 1e-12) & (np.abs(prediction64) > 1e-12)
    sign = np.divide(
        np.sum((np.sign(truth64) == np.sign(prediction64)) & valid_sign, axis=1),
        valid_sign.sum(axis=1),
        out=np.zeros(len(truth64)),
        where=valid_sign.sum(axis=1) > 0,
    )
    return {"pearson": pearson, "spearman": spearman, "cosine": cosine, "sign_agreement": sign}


def panel_masks(archive) -> dict[str, np.ndarray]:
    return {
        "landmark_measured": archive["is_landmark"].astype(bool),
        "bing": archive["is_bing"].astype(bool),
        "all_distributed": np.ones(len(archive["gene_symbol"]), dtype=bool),
    }


def aggregate_signatures(
    args: argparse.Namespace,
    accession: str,
    eligible: pd.DataFrame,
    compact_observations: pd.DataFrame,
    compact_delta: np.ndarray,
    static_genes: list[str],
) -> dict:
    extraction_path = args.derived_root / f"{accession}_selected_signatures.npz"
    with np.load(extraction_path, allow_pickle=False) as archive:
        signature_ids = archive["sig_id"].astype(str)
        gene_symbols = archive["gene_symbol"].astype(str)
        lincs_values = archive["values"].astype(np.float32)
        masks = panel_masks(archive)
    signature_lookup = {sig: index for index, sig in enumerate(signature_ids)}
    compact_lookup = {
        (str(row.cell_line), str(row.treatment)): index
        for index, row in enumerate(compact_observations.itertuples(index=False))
    }
    static_lookup = {gene: index for index, gene in enumerate(static_genes)}
    tahoe_gene_positions = np.array([static_lookup[gene] for gene in gene_symbols])

    condition_keys = [
        "accession", "tahoe_cell", "tahoe_cell_name", "drug", "treatment", "tahoe_dose_um"
    ]
    metadata_records = []
    tahoe_vectors = []
    lincs_vectors = []
    for key, group in eligible.groupby(condition_keys, sort=True):
        rows = np.array([signature_lookup[sig] for sig in group["sig_id"].astype(str)])
        lincs_vectors.append(lincs_values[rows].mean(axis=0, dtype=np.float32))
        compact_index = compact_lookup[(str(key[1]), str(key[4]))]
        tahoe_vectors.append(
            np.asarray(compact_delta[compact_index, tahoe_gene_positions], dtype=np.float32)
        )
        metadata_records.append({
            "accession": key[0],
            "tahoe_cell": key[1],
            "tahoe_cell_name": key[2],
            "drug": key[3],
            "treatment": key[4],
            "tahoe_dose_um": key[5],
            "lincs_dose_um": float(np.exp(np.log(group["lincs_dose_um"]).mean())),
            "dose_ratio": float(group["dose_ratio"].min()),
            "n_lincs_signatures": int(len(group)),
        })
    metadata = pd.DataFrame(metadata_records)
    tahoe = np.stack(tahoe_vectors)
    lincs = np.stack(lincs_vectors)

    output_path = args.derived_root / f"{accession}_condition_signatures.npz"
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        accession=np.asarray([accession], dtype=str),
        gene_symbol=np.asarray(gene_symbols, dtype=str),
        is_landmark=masks["landmark_measured"],
        is_bing=masks["bing"],
        tahoe_cell=metadata["tahoe_cell"].to_numpy(dtype=str),
        drug=metadata["drug"].to_numpy(dtype=str),
        treatment=metadata["treatment"].to_numpy(dtype=str),
        tahoe_dose_um=metadata["tahoe_dose_um"].to_numpy(float),
        lincs_dose_um=metadata["lincs_dose_um"].to_numpy(float),
        tahoe=tahoe,
        lincs=lincs,
    )
    os.replace(temporary, output_path)
    return {
        "metadata": metadata,
        "gene_symbols": gene_symbols,
        "masks": masks,
        "tahoe": tahoe,
        "lincs": lincs,
        "output": output_path,
    }


def clustered_interval(
    values: np.ndarray, drugs: np.ndarray, seed: int, draws: int
) -> tuple[float, float, float, float]:
    frame = pd.DataFrame({"drug": drugs, "value": values})
    drug_means = frame.groupby("drug", sort=True)["value"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(drug_means), size=(draws, len(drug_means)))
    sampled = drug_means[indices].mean(axis=1)
    return (
        float(np.mean(values)),
        float(drug_means.mean()),
        float(np.quantile(sampled, 0.025)),
        float(np.quantile(sampled, 0.975)),
    )


def drug_permutation_null(
    truth: np.ndarray,
    lincs: np.ndarray,
    metadata: pd.DataFrame,
    mask: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    groups = [
        np.asarray(indices, dtype=int)
        for indices in metadata.groupby(["tahoe_cell", "tahoe_dose_um"]).indices.values()
    ]
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for draw in range(permutations):
        shuffled = np.arange(len(metadata))
        for indices in groups:
            shuffled[indices] = rng.permutation(indices)
        pearson = rowwise_metrics(truth[:, mask], lincs[shuffled][:, mask])["pearson"]
        frame = pd.DataFrame({"drug": metadata["drug"], "pearson": pearson})
        null[draw] = frame.groupby("drug")["pearson"].mean().mean()
    actual = rowwise_metrics(truth[:, mask], lincs[:, mask])["pearson"]
    actual_drug_macro = (
        pd.DataFrame({"drug": metadata["drug"], "pearson": actual})
        .groupby("drug")["pearson"]
        .mean()
        .mean()
    )
    return null, float(actual_drug_macro)


def evaluate_concordance(
    args: argparse.Namespace, data_by_accession: dict[str, dict]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    condition_records = []
    aggregate_records = []
    permutation_records = []
    permutation_summary = {}
    for accession, data in data_by_accession.items():
        metadata = data["metadata"]
        for panel_index, panel in enumerate(PANELS):
            mask = data["masks"][panel]
            metrics = cross_metrics(data["tahoe"][:, mask], data["lincs"][:, mask])
            for row in range(len(metadata)):
                condition_records.append({
                    **metadata.iloc[row].to_dict(),
                    "panel": panel,
                    "n_genes": int(mask.sum()),
                    **{name: float(values[row]) for name, values in metrics.items()},
                })
            for metric_name, values in metrics.items():
                condition_macro, drug_macro, low, high = clustered_interval(
                    values,
                    metadata["drug"].astype(str).to_numpy(),
                    args.seed + ACCESSIONS.index(accession) * 100 + panel_index * 10 + len(metric_name),
                    args.bootstrap,
                )
                aggregate_records.append({
                    "accession": accession,
                    "panel": panel,
                    "n_genes": int(mask.sum()),
                    "n_conditions": len(metadata),
                    "n_cells": metadata["tahoe_cell"].nunique(),
                    "n_drugs": metadata["drug"].nunique(),
                    "metric": metric_name,
                    "condition_macro": condition_macro,
                    "drug_macro": drug_macro,
                    "drug_bootstrap_low": low,
                    "drug_bootstrap_high": high,
                })
            null, actual = drug_permutation_null(
                data["tahoe"],
                data["lincs"],
                metadata,
                mask,
                args.permutations,
                args.seed + ACCESSIONS.index(accession) * 10_000 + panel_index * 100,
            )
            p_value = float((1 + np.sum(null >= actual)) / (1 + len(null)))
            permutation_summary[f"{accession}:{panel}"] = {
                "actual_drug_macro_pearson": actual,
                "null_mean": float(null.mean()),
                "null_95th_percentile": float(np.quantile(null, 0.95)),
                "empirical_p_value": p_value,
            }
            for permutation_id, value in enumerate(null):
                permutation_records.append({
                    "accession": accession,
                    "panel": panel,
                    "permutation_id": permutation_id,
                    "drug_macro_pearson": value,
                })
    return (
        pd.DataFrame(condition_records),
        pd.DataFrame(aggregate_records),
        pd.DataFrame(permutation_records),
        permutation_summary,
    )


def phase_reproducibility(data_by_accession: dict[str, dict]) -> pd.DataFrame:
    first = data_by_accession[ACCESSIONS[0]]
    second = data_by_accession[ACCESSIONS[1]]
    keys = ["tahoe_cell", "drug", "tahoe_dose_um"]
    first_lookup = {
        tuple(row[key] for key in keys): index
        for index, row in first["metadata"].iterrows()
    }
    records = []
    for second_index, row in second["metadata"].iterrows():
        key = tuple(row[name] for name in keys)
        if key not in first_lookup:
            continue
        first_index = first_lookup[key]
        for panel in PANELS:
            mask = first["masks"][panel]
            metrics = cross_metrics(
                first["lincs"][[first_index]][:, mask],
                second["lincs"][[second_index]][:, mask],
            )
            records.append({
                "tahoe_cell": key[0],
                "drug": key[1],
                "tahoe_dose_um": key[2],
                "panel": panel,
                "n_genes": int(mask.sum()),
                **{name: float(values[0]) for name, values in metrics.items()},
            })
    return pd.DataFrame(records)


def frozen_rhaister_lookup(
    baseline_root: Path, wanted_keys: set[tuple[str, str]]
) -> tuple[dict, list[str]]:
    lookup = {}
    genes = None
    for split in range(5, 10):
        with np.load(baseline_root / f"split_{split}" / "predictions.npz", allow_pickle=False) as archive:
            if genes is None:
                genes = archive["genes"].astype(str).tolist()
            cells = archive["cell_line"].astype(str)
            treatments = archive["treatment"].astype(str)
            selected = [
                row for row, key in enumerate(zip(cells, treatments)) if key in wanted_keys
            ]
            if not selected:
                continue
            # NPZ members are compressed independently.  Load each dense member
            # once per split; indexing archive[member] inside the row loop would
            # decompress the entire member repeatedly.
            truth = archive["truth"]
            drug_mean = archive["perturbation_mean"]
            rhaister = archive["rhaister_o"]
            for row in selected:
                key = (cells[row], treatments[row])
                lookup[key] = {
                    "split": split,
                    "truth": truth[row].astype(np.float32),
                    "drug_mean": drug_mean[row].astype(np.float32),
                    "rhaister_o": rhaister[row].astype(np.float32),
                }
    return lookup, genes or []


def rhaister_aligned_comparison(
    data_by_accession: dict[str, dict], baseline_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wanted_keys = {
        (str(row["tahoe_cell"]), str(row["treatment"]))
        for data in data_by_accession.values()
        for _, row in data["metadata"].iterrows()
    }
    frozen, frozen_genes = frozen_rhaister_lookup(baseline_root, wanted_keys)
    frozen_gene_lookup = {gene: index for index, gene in enumerate(frozen_genes)}
    records = []
    for accession, data in data_by_accession.items():
        gene_positions = np.array([frozen_gene_lookup[gene] for gene in data["gene_symbols"]])
        for row, metadata in data["metadata"].iterrows():
            key = (str(metadata["tahoe_cell"]), str(metadata["treatment"]))
            if key not in frozen:
                continue
            baseline = frozen[key]
            for panel in PANELS:
                mask = data["masks"][panel]
                truth = baseline["truth"][gene_positions][None, mask]
                predictions = {
                    "lincs_level5": data["lincs"][[row]][:, mask],
                    "drug_mean": baseline["drug_mean"][gene_positions][None, mask],
                    "rhaister_o": baseline["rhaister_o"][gene_positions][None, mask],
                }
                for model, prediction in predictions.items():
                    metrics = cross_metrics(truth, prediction)
                    records.append({
                        "accession": accession,
                        "split": baseline["split"],
                        "tahoe_cell": metadata["tahoe_cell"],
                        "tahoe_cell_name": metadata["tahoe_cell_name"],
                        "drug": metadata["drug"],
                        "tahoe_dose_um": metadata["tahoe_dose_um"],
                        "panel": panel,
                        "n_genes": int(mask.sum()),
                        "model": model,
                        **{name: float(values[0]) for name, values in metrics.items()},
                    })
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame, frame
    summary = (
        frame.groupby(["accession", "panel", "n_genes", "model"], as_index=False)
        .agg(
            n_conditions=("drug", "size"),
            n_drugs=("drug", "nunique"),
            n_cells=("tahoe_cell", "nunique"),
            pearson=("pearson", "mean"),
            spearman=("spearman", "mean"),
            cosine=("cosine", "mean"),
            sign_agreement=("sign_agreement", "mean"),
        )
    )
    return frame, summary


def main(args: argparse.Namespace) -> None:
    args.derived_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    static_path = args.project_root / "external/Rhaister/splits/tahoe/static_2k_genes.json"
    with static_path.open(encoding="utf-8") as handle:
        static_genes = json.load(handle)
    compact_observations = pd.read_parquet(args.compact_root / "observations.parquet")
    compact_delta = np.load(args.compact_root / "delta.npy", mmap_mode="r")

    eligibility_by_accession = {
        accession: build_eligibility(args, accession) for accession in ACCESSIONS
    }
    eligibility = pd.concat(eligibility_by_accession.values(), ignore_index=True)
    atomic_csv(eligibility, args.output_root / "eligible_signatures.csv")
    extraction = {
        accession: extract_gctx(args, accession, eligibility_by_accession[accession], static_genes)
        for accession in ACCESSIONS
    }
    data_by_accession = {
        accession: aggregate_signatures(
            args,
            accession,
            eligibility_by_accession[accession],
            compact_observations,
            compact_delta,
            static_genes,
        )
        for accession in ACCESSIONS
    }
    condition_metadata = pd.concat(
        [data["metadata"] for data in data_by_accession.values()], ignore_index=True
    )
    atomic_csv(condition_metadata, args.output_root / "matched_conditions.csv")

    condition_metrics, aggregate, permutation_null, permutation_summary = evaluate_concordance(
        args, data_by_accession
    )
    atomic_csv(condition_metrics, args.output_root / "per_condition_metrics.csv")
    atomic_csv(aggregate, args.output_root / "aggregate_summary.csv")
    atomic_csv(permutation_null, args.output_root / "drug_permutation_null.csv")
    cell_summary = (
        condition_metrics.groupby(
            ["accession", "panel", "n_genes", "tahoe_cell", "tahoe_cell_name"],
            as_index=False,
        )
        .agg(
            n_conditions=("drug", "size"),
            n_drugs=("drug", "nunique"),
            pearson=("pearson", "mean"),
            spearman=("spearman", "mean"),
            cosine=("cosine", "mean"),
            sign_agreement=("sign_agreement", "mean"),
        )
    )
    atomic_csv(cell_summary, args.output_root / "cell_summary.csv")

    phase = phase_reproducibility(data_by_accession)
    atomic_csv(phase, args.output_root / "phase_reproducibility.csv")
    aligned, aligned_summary = rhaister_aligned_comparison(
        data_by_accession, args.baseline_root
    )
    atomic_csv(aligned, args.output_root / "rhaister_aligned_metrics.csv")
    atomic_csv(aligned_summary, args.output_root / "rhaister_aligned_summary.csv")

    phase_summary = []
    if not phase.empty:
        phase_summary = (
            phase.groupby(["panel", "n_genes"], as_index=False)
            .agg(
                n_conditions=("drug", "size"),
                n_cells=("tahoe_cell", "nunique"),
                n_drugs=("drug", "nunique"),
                pearson=("pearson", "mean"),
                spearman=("spearman", "mean"),
                cosine=("cosine", "mean"),
                sign_agreement=("sign_agreement", "mean"),
            )
            .to_dict(orient="records")
        )
    summary = {
        "purpose": "external 24-hour LINCS L1000 versus Tahoe response concordance",
        "task_label": "external validation, not strict-zero-shot training augmentation",
        "accessions": list(ACCESSIONS),
        "matching": {
            "cell": "exact curated Tahoe-LINCS cell-name crosswalk",
            "compound": "structure_validated Tahoe-LINCS crosswalk rows only",
            "time": "24 hours exactly",
            "dose": f"nearest LINCS dose within a {args.max_dose_ratio:g}-fold ratio",
            "signature_aggregation": "mean Level-5 z-score at the selected dose",
        },
        "panels": {
            panel: int(data_by_accession[ACCESSIONS[0]]["masks"][panel].sum())
            for panel in PANELS
        },
        "primary_panel": "landmark_measured",
        "n_bootstrap": args.bootstrap,
        "bootstrap_unit": "drug; all matched cells/doses clustered",
        "n_permutations": args.permutations,
        "permutation": "shuffle L1000 drug assignments within cell and Tahoe-dose strata",
        "seed": args.seed,
        "static_gene_file": str(static_path),
        "static_gene_file_sha256": sha256(static_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "extraction": extraction,
        "eligibility": {
            accession: {
                "n_signature_mappings": len(eligibility_by_accession[accession]),
                "n_signatures": eligibility_by_accession[accession]["sig_id"].nunique(),
                "n_conditions": eligibility_by_accession[accession][
                    ["tahoe_cell", "drug", "tahoe_dose_um"]
                ].drop_duplicates().shape[0],
                "n_cells": eligibility_by_accession[accession]["tahoe_cell"].nunique(),
                "n_drugs": eligibility_by_accession[accession]["drug"].nunique(),
            }
            for accession in ACCESSIONS
        },
        "aggregate": aggregate.to_dict(orient="records"),
        "permutation_summary": permutation_summary,
        "phase_reproducibility": phase_summary,
        "rhaister_aligned": aligned_summary.to_dict(orient="records"),
    }
    write_json(summary, args.output_root / "summary.json")
    print("\nAggregate Tahoe-L1000 concordance:", flush=True)
    print(aggregate.to_string(index=False), flush=True)
    print("\nDrug-label permutation:", flush=True)
    print(json.dumps(permutation_summary, indent=2), flush=True)
    print("\nRhaister-aligned subset:", flush=True)
    print(aligned_summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT)
    parser.add_argument("--compact-root", type=Path, default=COMPACT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--derived-root", type=Path, default=DERIVED_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-dose-ratio", type=float, default=2.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
