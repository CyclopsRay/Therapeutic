#!/usr/bin/env python3
"""All-split Rhaister diagnostics, train-fit PC removal, and DMSO kNN.

This implements experiment-plan steps 1--4 on the five frozen Rhaister-O
outer splits.  It does not fit or evaluate DepMap models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_rhaister_signal import (
    clustered_bootstrap,
    line_dmso_vectors,
    parse_treatment,
)
from run_rhaister_o_compact import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_REVISION,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT as BASELINE_OUTPUT_ROOT,
    DEFAULT_PROJECT_ROOT,
    build_or_load_compact_cache,
    group_mean,
    load_hvg_centroids,
    load_split,
    rowwise_metrics,
)


DEFAULT_OUTPUT_ROOT = Path("/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_all_split_evaluation")
MODELS = ("perturbation_mean", "nearest_dmso_1", "nearest_dmso_3", "rhaister_o")
LEVELS = ("full", "drug_residual", "train_pca1_removed")


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def fit_first_pc(
    values: np.ndarray,
    seed: int = 0,
    iterations: int = 12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a deterministic first right singular vector by power iteration."""
    matrix = np.asarray(values, dtype=np.float32)
    mean = matrix.mean(axis=0, dtype=np.float32)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=matrix.shape[1]).astype(np.float32)
    vector /= np.linalg.norm(vector)
    for _ in range(iterations):
        scores = matrix @ vector - float(mean @ vector)
        updated = matrix.T @ scores - mean * scores.sum(dtype=np.float32)
        norm = float(np.linalg.norm(updated))
        if norm <= np.finfo(np.float32).eps:
            raise ValueError("Cannot fit PC1 to a constant response matrix")
        vector = updated / norm
    scores = matrix @ vector - float(mean @ vector)
    explained = float(np.sum(np.square(scores), dtype=np.float64))
    total = float(
        np.sum(np.square(matrix), dtype=np.float64)
        - len(matrix) * np.sum(np.square(mean), dtype=np.float64)
    )
    ratio = explained / max(total, np.finfo(np.float64).eps)
    return mean, vector.astype(np.float32), ratio


def remove_pc(values: np.ndarray, mean: np.ndarray, component: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float32) - mean
    scores = centered @ component
    return centered - scores[:, None] * component[None, :]


def dmso_knn_predictions(
    compact,
    hvg: dict[tuple[str, int], np.ndarray],
    holdout_cells: list[str],
    test_cells: np.ndarray,
    test_treatments: np.ndarray,
    fallback: np.ndarray,
    neighbors: tuple[int, ...] = (1, 3),
) -> tuple[dict[int, np.ndarray], pd.DataFrame, dict[int, np.ndarray]]:
    reference_cells = sorted(set(compact.cells.tolist()) - set(holdout_cells))
    reference_z, reference_plate_counts = line_dmso_vectors(hvg, reference_cells)
    query_z, query_plate_counts = line_dmso_vectors(hvg, holdout_cells)

    reference_centered = reference_z.astype(np.float64)
    reference_centered -= reference_centered.mean(axis=1, keepdims=True)
    reference_centered /= np.maximum(
        np.linalg.norm(reference_centered, axis=1, keepdims=True),
        np.finfo(np.float64).eps,
    )
    query_centered = query_z.astype(np.float64)
    query_centered -= query_centered.mean(axis=1, keepdims=True)
    query_centered /= np.maximum(
        np.linalg.norm(query_centered, axis=1, keepdims=True),
        np.finfo(np.float64).eps,
    )
    similarity = query_centered @ reference_centered.T
    ranking = np.argsort(-similarity, axis=1)

    neighbor_records = []
    for query_index, query in enumerate(holdout_cells):
        for rank in range(max(neighbors)):
            reference_index = int(ranking[query_index, rank])
            reference = reference_cells[reference_index]
            neighbor_records.append({
                "cell_line": query,
                "rank": rank + 1,
                "reference_cell_line": reference,
                "dmso_pearson": float(similarity[query_index, reference_index]),
                "query_dmso_plates": query_plate_counts[query],
                "reference_dmso_plates": reference_plate_counts[reference],
            })
    neighbor_frame = pd.DataFrame(neighbor_records)

    condition_to_row = {
        (cell, treatment): index
        for index, (cell, treatment) in enumerate(zip(compact.cells, compact.treatments))
    }
    predictions = {
        k: np.asarray(fallback, dtype=np.float32).copy() for k in neighbors
    }
    covered = {k: np.zeros(len(test_cells), dtype=bool) for k in neighbors}
    query_to_index = {cell: index for index, cell in enumerate(holdout_cells)}
    for output_index, (cell, treatment) in enumerate(zip(test_cells, test_treatments)):
        ranked_references = [
            reference_cells[int(i)] for i in ranking[query_to_index[cell]]
        ]
        for k in neighbors:
            source_rows = [
                condition_to_row[(reference, treatment)]
                for reference in ranked_references[:k]
                if (reference, treatment) in condition_to_row
            ]
            if source_rows:
                predictions[k][output_index] = np.mean(
                    np.asarray(compact.delta[source_rows], dtype=np.float32),
                    axis=0,
                    dtype=np.float32,
                )
                covered[k][output_index] = True
    return predictions, neighbor_frame, covered


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def analyze_split(args: argparse.Namespace, split: int, compact, hvg) -> dict:
    output_dir = args.output_root / f"split_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    holdout_cells, test_treatment_sets, split_path = load_split(
        args.project_root / "external" / "Rhaister", split
    )
    holdout = set(holdout_cells)
    test_mask = np.array([
        cell in holdout and treatment in test_treatment_sets.get(cell, set())
        for cell, treatment in zip(compact.cells, compact.treatments)
    ])
    train_mask = np.array([cell not in holdout for cell in compact.cells])
    train_indices = np.flatnonzero(train_mask)
    expected_cells = compact.cells[test_mask]
    expected_treatments = compact.treatments[test_mask]

    baseline_path = args.baseline_root / f"split_{split}" / "predictions.npz"
    with np.load(baseline_path, allow_pickle=False) as archive:
        genes = archive["genes"].copy()
        cells = archive["cell_line"].copy()
        treatments = archive["treatment"].copy()
        truth = archive["truth"].astype(np.float32)
        drug_mean = archive["perturbation_mean"].astype(np.float32)
        rhaister = archive["rhaister_o"].astype(np.float32)
    if not np.array_equal(cells, expected_cells) or not np.array_equal(treatments, expected_treatments):
        raise RuntimeError(f"Split {split}: baseline rows do not align with compact data")

    train_y = np.asarray(compact.delta[train_indices], dtype=np.float32)
    train_treatments = compact.treatments[train_indices]
    unique_train_treatments, train_treatment_index = np.unique(
        train_treatments, return_inverse=True
    )
    train_means = group_mean(
        train_y, train_treatment_index, len(unique_train_treatments)
    )
    train_residual = train_y - train_means[train_treatment_index]
    pc_mean, pc1, pc1_ratio = fit_first_pc(
        train_residual, seed=args.seed + split, iterations=args.pc_iterations
    )
    del train_y, train_means, train_residual

    knn, neighbor_frame, knn_covered = dmso_knn_predictions(
        compact,
        hvg,
        holdout_cells,
        cells,
        treatments,
        drug_mean,
        neighbors=(1, 3),
    )
    models = {
        "perturbation_mean": drug_mean,
        "nearest_dmso_1": knn[1],
        "nearest_dmso_3": knn[3],
        "rhaister_o": rhaister,
    }
    parsed = [parse_treatment(label) for label in treatments]
    condition = pd.DataFrame({
        "split": split,
        "cell_line": cells,
        "treatment": treatments,
        "drug": [item[0] for item in parsed],
        "dose": [item[1] for item in parsed],
        "dose_unit": [item[2] for item in parsed],
        "nearest_dmso_1_covered": knn_covered[1],
        "nearest_dmso_3_covered": knn_covered[3],
    })

    truth_residual = truth - drug_mean
    truth_pc_removed = remove_pc(truth_residual, pc_mean, pc1)
    level_truth = {
        "full": truth,
        "drug_residual": truth_residual,
        "train_pca1_removed": truth_pc_removed,
    }
    level_prediction: dict[tuple[str, str], np.ndarray] = {}
    for model, prediction in models.items():
        residual_prediction = prediction - drug_mean
        level_prediction[(model, "full")] = prediction
        level_prediction[(model, "drug_residual")] = residual_prediction
        level_prediction[(model, "train_pca1_removed")] = remove_pc(
            residual_prediction, pc_mean, pc1
        )
        for level in LEVELS:
            metrics = rowwise_metrics(level_truth[level], level_prediction[(model, level)])
            for metric, values in metrics.items():
                condition[f"{model}__{level}__{metric}"] = values

    line_records = []
    bootstrap_records = []
    for cell_index, cell in enumerate(holdout_cells):
        subset = condition[condition["cell_line"].eq(cell)]
        for level_index, level in enumerate(LEVELS):
            baseline_values = subset[f"perturbation_mean__{level}__pearson"].to_numpy()
            for model_index, model in enumerate(MODELS):
                line_records.append({
                    "split": split,
                    "cell_line": cell,
                    "level": level,
                    "model": model,
                    "n_conditions": len(subset),
                    **{
                        metric: float(subset[f"{model}__{level}__{metric}"].mean())
                        for metric in ("pearson", "rmse", "mae")
                    },
                })
                result = clustered_bootstrap(
                    subset[f"{model}__{level}__pearson"].to_numpy(),
                    baseline_values,
                    subset["drug"].to_numpy(),
                    args.n_bootstrap,
                    args.seed + split * 10000 + cell_index * 100 + level_index * 10 + model_index,
                )
                bootstrap_records.append({
                    "split": split,
                    "cell_line": cell,
                    "level": level,
                    "model": model,
                    "n_conditions": len(subset),
                    **result,
                })

    line_summary = pd.DataFrame(line_records)
    bootstrap = pd.DataFrame(bootstrap_records)
    atomic_csv(condition, output_dir / "per_condition_metrics.csv")
    atomic_csv(line_summary, output_dir / "per_line_summary.csv")
    atomic_csv(bootstrap, output_dir / "drug_cluster_bootstrap.csv")
    atomic_csv(neighbor_frame, output_dir / "dmso_neighbors.csv")

    pc_path = output_dir / "train_residual_pc1.npz"
    pc_temporary = output_dir / "train_residual_pc1.tmp.npz"
    np.savez_compressed(
        pc_temporary,
        genes=np.asarray(genes, dtype=str),
        mean=pc_mean.astype(np.float32),
        component=pc1.astype(np.float32),
        explained_variance_ratio=np.array(pc1_ratio, dtype=np.float64),
    )
    os.replace(pc_temporary, pc_path)

    prediction_path = output_dir / "dmso_knn_predictions.npz"
    prediction_temporary = output_dir / "dmso_knn_predictions.tmp.npz"
    np.savez_compressed(
        prediction_temporary,
        genes=np.asarray(genes, dtype=str),
        cell_line=np.asarray(cells, dtype=str),
        treatment=np.asarray(treatments, dtype=str),
        nearest_dmso_1=knn[1].astype(np.float32),
        nearest_dmso_3=knn[3].astype(np.float32),
        nearest_dmso_1_covered=knn_covered[1],
        nearest_dmso_3_covered=knn_covered[3],
    )
    os.replace(prediction_temporary, prediction_path)

    result = {
        "split": split,
        "split_file": str(split_path),
        "split_file_sha256": sha256(split_path),
        "holdout_cells": holdout_cells,
        "n_test_conditions": len(condition),
        "n_reference_conditions": len(train_indices),
        "n_reference_cells": len(set(compact.cells[train_indices].tolist())),
        "train_residual_pc1_explained_variance_ratio": pc1_ratio,
        "nearest_dmso_1_coverage": float(knn_covered[1].mean()),
        "nearest_dmso_3_coverage": float(knn_covered[3].mean()),
        "per_line": line_records,
    }
    write_json(result, output_dir / "summary.json")
    print(
        f"split {split}: test={len(condition):,}, train-residual PC1={pc1_ratio:.4%}, "
        f"1NN coverage={knn_covered[1].mean():.2%}, 3NN coverage={knn_covered[3].mean():.2%}",
        flush=True,
    )
    return result


def main(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    repo_root = project_root / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    hvg = load_hvg_centroids(args.data_root, compact.genes)
    results = [analyze_split(args, split, compact, hvg) for split in range(5, 10)]

    all_lines = pd.concat([
        pd.read_csv(args.output_root / f"split_{split}" / "per_line_summary.csv")
        for split in range(5, 10)
    ], ignore_index=True)
    all_bootstrap = pd.concat([
        pd.read_csv(args.output_root / f"split_{split}" / "drug_cluster_bootstrap.csv")
        for split in range(5, 10)
    ], ignore_index=True)
    atomic_csv(all_lines, args.output_root / "per_line_summary.csv")
    atomic_csv(all_bootstrap, args.output_root / "drug_cluster_bootstrap.csv")

    aggregate_records = []
    for level in LEVELS:
        for model in MODELS:
            subset = all_lines[(all_lines["level"] == level) & (all_lines["model"] == model)]
            aggregate_records.append({
                "level": level,
                "model": model,
                "n_lines": len(subset),
                "line_macro_pearson": float(subset["pearson"].mean()),
                "line_sd_pearson": float(subset["pearson"].std(ddof=0)),
                "line_macro_rmse": float(subset["rmse"].mean()),
                "line_macro_mae": float(subset["mae"].mean()),
            })
    aggregate_frame = pd.DataFrame(aggregate_records)
    atomic_csv(aggregate_frame, args.output_root / "aggregate_summary.csv")

    baseline_summary = args.baseline_root / "summary.json"
    summary = {
        "purpose": "experiment-plan steps 1--4 on all 25 outer held-out lines",
        "splits": list(range(5, 10)),
        "n_heldout_lines": 25,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_unit": "drug, all doses clustered",
        "pca_definition": "PC1 fit only to reference-line drug-mean residuals",
        "pc_iterations": args.pc_iterations,
        "dmso_knn": [1, 3],
        "dmso_knn_weighting": "uniform among available neighbors",
        "seed": args.seed,
        "data_revision": args.data_revision,
        "baseline_summary": str(baseline_summary),
        "baseline_summary_sha256": sha256(baseline_summary),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "models": list(MODELS),
        "levels": list(LEVELS),
        "aggregate": aggregate_records,
        "per_split": results,
    }
    write_json(summary, args.output_root / "summary.json")
    print("\nAggregate line-macro results:", flush=True)
    print(aggregate_frame.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-revision", default=DEFAULT_DATA_REVISION)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--pc-iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
