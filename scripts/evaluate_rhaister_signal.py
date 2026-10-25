#!/usr/bin/env python3
"""Small split-5 signal pilot on top of the frozen Rhaister-O baseline.

The pilot intentionally avoids a model sweep.  It adds one transparent
nearest-DMSO-line predictor and three evaluation levels:

1. full expression delta;
2. residual after subtracting the reference-line drug-dose mean;
3. that residual after removing each held-out line's mean response template.

Uncertainty is estimated by resampling drugs while keeping all doses of a drug
together.  The target-line treatment outcomes are used only after predictions
are frozen, for evaluation and diagnostic decomposition.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from run_rhaister_o_compact import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_REVISION,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT as BASELINE_OUTPUT_ROOT,
    DEFAULT_PROJECT_ROOT,
    build_or_load_compact_cache,
    load_hvg_centroids,
    load_split,
    rowwise_metrics,
)


DEFAULT_OUTPUT_ROOT = Path("/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_signal_pilot")
MODEL_NAMES = ("perturbation_mean", "nearest_dmso_line", "rhaister_o")
LEVELS = ("full", "drug_residual", "template_removed")


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


def parse_treatment(label: str) -> tuple[str, float, str]:
    """Return a drug cluster label, dose, and unit from a Tahoe treatment."""
    try:
        parsed = ast.literal_eval(label)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Cannot parse Tahoe treatment {label!r}") from error
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"Unexpected Tahoe treatment structure: {label!r}")
    drugs = []
    doses = []
    units = []
    for item in parsed:
        if not isinstance(item, tuple) or len(item) != 3:
            raise ValueError(f"Unexpected Tahoe treatment tuple: {item!r}")
        drug, dose, unit = item
        drugs.append(str(drug))
        doses.append(float(dose))
        units.append(str(unit))
    drug_cluster = " + ".join(drugs)
    dose_value = doses[0] if len(set(doses)) == 1 else float("nan")
    unit_value = units[0] if len(set(units)) == 1 else "mixed"
    return drug_cluster, dose_value, unit_value


def centered_cosine_similarity(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Pearson correlation between query and reference rows across genes."""
    query64 = np.asarray(query, dtype=np.float64)
    reference64 = np.asarray(reference, dtype=np.float64)
    query64 -= query64.mean(axis=1, keepdims=True)
    reference64 -= reference64.mean(axis=1, keepdims=True)
    query_norm = np.linalg.norm(query64, axis=1, keepdims=True)
    reference_norm = np.linalg.norm(reference64, axis=1, keepdims=True)
    query64 /= np.maximum(query_norm, np.finfo(np.float64).eps)
    reference64 /= np.maximum(reference_norm, np.finfo(np.float64).eps)
    return query64 @ reference64.T


def line_dmso_vectors(
    hvg: dict[tuple[str, int], np.ndarray], cells: list[str]
) -> tuple[np.ndarray, dict[str, int]]:
    """Average log1p DMSO centroids over every available plate for each line."""
    vectors = []
    plate_counts: dict[str, int] = {}
    for cell in cells:
        rows = [value for (candidate, _), value in hvg.items() if candidate == cell]
        if not rows:
            raise ValueError(f"No DMSO HVG centroid found for {cell}")
        plate_counts[cell] = len(rows)
        vectors.append(np.log1p(np.stack(rows)).mean(axis=0))
    return np.stack(vectors).astype(np.float32), plate_counts


def nearest_line_predictions(
    compact,
    hvg: dict[tuple[str, int], np.ndarray],
    holdout_cells: list[str],
    test_cells: np.ndarray,
    test_treatments: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    reference_cells = sorted(set(compact.cells) - set(holdout_cells))
    reference_z, reference_plate_counts = line_dmso_vectors(hvg, reference_cells)
    query_z, query_plate_counts = line_dmso_vectors(hvg, holdout_cells)
    similarity = centered_cosine_similarity(query_z, reference_z)
    nearest_index = similarity.argmax(axis=1)
    nearest = {
        cell: reference_cells[int(nearest_index[i])]
        for i, cell in enumerate(holdout_cells)
    }
    neighbor_frame = pd.DataFrame({
        "cell_line": holdout_cells,
        "nearest_reference_line": [nearest[cell] for cell in holdout_cells],
        "dmso_pearson": [float(similarity[i, nearest_index[i]]) for i in range(len(holdout_cells))],
        "query_dmso_plates": [query_plate_counts[cell] for cell in holdout_cells],
        "reference_dmso_plates": [reference_plate_counts[nearest[cell]] for cell in holdout_cells],
    })

    condition_to_row = {
        (cell, treatment): index
        for index, (cell, treatment) in enumerate(zip(compact.cells, compact.treatments))
    }
    prediction = np.asarray(fallback, dtype=np.float32).copy()
    covered = np.zeros(len(test_cells), dtype=bool)
    for index, (cell, treatment) in enumerate(zip(test_cells, test_treatments)):
        source_index = condition_to_row.get((nearest[cell], treatment))
        if source_index is not None:
            prediction[index] = compact.delta[source_index]
            covered[index] = True
    return prediction, neighbor_frame, covered


def remove_line_template(
    values: np.ndarray,
    cells: np.ndarray,
    treatments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove each line's mean residual on the cross-line common treatment set."""
    unique_cells = sorted(set(cells.tolist()))
    treatment_sets = [set(treatments[cells == cell].tolist()) for cell in unique_cells]
    common_treatments = sorted(set.intersection(*treatment_sets))
    lookup = {
        (cell, treatment): index
        for index, (cell, treatment) in enumerate(zip(cells.tolist(), treatments.tolist()))
    }
    ordered_indices = np.array([
        lookup[(cell, treatment)]
        for cell in unique_cells
        for treatment in common_treatments
    ])
    cube = np.asarray(values, dtype=np.float32)[ordered_indices].reshape(
        len(unique_cells), len(common_treatments), values.shape[1]
    )
    cube -= cube.mean(axis=1, keepdims=True, dtype=np.float32)
    return cube.reshape(-1, values.shape[1]), ordered_indices


def clustered_bootstrap(
    values: np.ndarray,
    baseline: np.ndarray,
    drugs: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """Equal-weight drug bootstrap; doses of one drug stay in one cluster."""
    frame = pd.DataFrame({"drug": drugs, "value": values, "baseline": baseline})
    grouped = frame.groupby("drug", sort=True)[["value", "baseline"]].mean()
    scores = grouped["value"].to_numpy(dtype=np.float64)
    differences = (grouped["value"] - grouped["baseline"]).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), size=(n_bootstrap, len(grouped)))
    boot_score = scores[draws].mean(axis=1)
    boot_difference = differences[draws].mean(axis=1)
    return {
        "n_drugs": int(len(grouped)),
        "drug_macro_mean": float(scores.mean()),
        "ci_low": float(np.quantile(boot_score, 0.025)),
        "ci_high": float(np.quantile(boot_score, 0.975)),
        "improvement_over_drug_mean": float(differences.mean()),
        "improvement_ci_low": float(np.quantile(boot_difference, 0.025)),
        "improvement_ci_high": float(np.quantile(boot_difference, 0.975)),
    }


def add_metrics(
    frame: pd.DataFrame,
    level: str,
    model: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    row_indices: np.ndarray | None = None,
) -> None:
    metrics = rowwise_metrics(truth, prediction)
    if row_indices is None:
        row_indices = np.arange(len(frame))
    for metric, values in metrics.items():
        column = f"{model}__{level}__{metric}"
        if column not in frame:
            frame[column] = np.nan
        frame.loc[row_indices, column] = values


def magnitude_quartiles(values: pd.Series) -> pd.Series:
    labels = ["Q1_weak", "Q2", "Q3", "Q4_strong"]
    return pd.qcut(values.rank(method="first"), 4, labels=labels)


def run_pilot(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    repo_root = project_root / "external" / "Rhaister"
    baseline_dir = args.baseline_root / f"split_{args.split}"
    baseline_summary_path = args.baseline_root / "summary.json"
    output_dir = args.output_root / f"split_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    holdout_cells, test_treatment_sets, split_path = load_split(repo_root, args.split)
    test_mask = np.array([
        cell in set(holdout_cells) and treatment in test_treatment_sets.get(cell, set())
        for cell, treatment in zip(compact.cells, compact.treatments)
    ])
    expected_cells = compact.cells[test_mask]
    expected_treatments = compact.treatments[test_mask]

    with np.load(baseline_dir / "predictions.npz", allow_pickle=False) as archive:
        genes = archive["genes"].copy()
        cells = archive["cell_line"].copy()
        treatments = archive["treatment"].copy()
        truth = archive["truth"].astype(np.float32)
        drug_mean = archive["perturbation_mean"].astype(np.float32)
        rhaister = archive["rhaister_o"].astype(np.float32)
    if not np.array_equal(cells, expected_cells) or not np.array_equal(treatments, expected_treatments):
        raise RuntimeError("Frozen prediction rows do not align with the compact split")

    parsed = [parse_treatment(label) for label in treatments]
    frame = pd.DataFrame({
        "cell_line": cells,
        "treatment": treatments,
        "drug": [item[0] for item in parsed],
        "dose": [item[1] for item in parsed],
        "dose_unit": [item[2] for item in parsed],
        "truth_delta_rms": np.sqrt(np.mean(np.square(truth), axis=1)),
    })

    hvg = load_hvg_centroids(args.data_root, compact.genes)
    nearest, neighbors, nearest_covered = nearest_line_predictions(
        compact, hvg, holdout_cells, cells, treatments, drug_mean
    )
    frame["nearest_covered"] = nearest_covered
    models = {
        "perturbation_mean": drug_mean,
        "nearest_dmso_line": nearest,
        "rhaister_o": rhaister,
    }

    for model, prediction in models.items():
        add_metrics(frame, "full", model, truth, prediction)

    truth_residual = truth - drug_mean
    frame["truth_drug_residual_rms"] = np.sqrt(np.mean(np.square(truth_residual), axis=1))
    residual_predictions: dict[str, np.ndarray] = {}
    for model, prediction in models.items():
        residual_prediction = prediction - drug_mean
        residual_predictions[model] = residual_prediction
        add_metrics(frame, "drug_residual", model, truth_residual, residual_prediction)

    truth_template_removed, common_indices = remove_line_template(
        truth_residual, cells, treatments
    )
    frame["truth_template_removed_rms"] = np.nan
    frame.loc[common_indices, "truth_template_removed_rms"] = np.sqrt(
        np.mean(np.square(truth_template_removed), axis=1)
    )
    for model, residual_prediction in residual_predictions.items():
        prediction_template_removed, model_indices = remove_line_template(
            residual_prediction, cells, treatments
        )
        if not np.array_equal(model_indices, common_indices):
            raise RuntimeError("Template-removal condition order changed between models")
        add_metrics(
            frame,
            "template_removed",
            model,
            truth_template_removed,
            prediction_template_removed,
            row_indices=common_indices,
        )

    line_records = []
    bootstrap_records = []
    for cell in holdout_cells:
        cell_mask = frame["cell_line"].eq(cell)
        for level in LEVELS:
            baseline_column = f"perturbation_mean__{level}__pearson"
            valid_baseline = frame.loc[cell_mask, baseline_column].notna()
            for model in MODEL_NAMES:
                subset = frame.loc[cell_mask & frame[f"{model}__{level}__pearson"].notna()]
                metrics = {
                    metric: float(subset[f"{model}__{level}__{metric}"].mean())
                    for metric in ("pearson", "rmse", "mae")
                }
                line_records.append({
                    "cell_line": cell,
                    "level": level,
                    "model": model,
                    "n_conditions": len(subset),
                    **metrics,
                })
                baseline_values = frame.loc[subset.index, baseline_column].to_numpy()
                result = clustered_bootstrap(
                    subset[f"{model}__{level}__pearson"].to_numpy(),
                    baseline_values,
                    subset["drug"].to_numpy(),
                    args.n_bootstrap,
                    args.seed + holdout_cells.index(cell) * 100 + MODEL_NAMES.index(model) * 10 + LEVELS.index(level),
                )
                bootstrap_records.append({
                    "cell_line": cell,
                    "level": level,
                    "model": model,
                    "n_conditions": len(subset),
                    **result,
                })

    line_summary = pd.DataFrame(line_records)
    bootstrap = pd.DataFrame(bootstrap_records)

    dose_records = []
    for level in ("full", "drug_residual", "template_removed"):
        for model in MODEL_NAMES:
            column = f"{model}__{level}__pearson"
            for (dose, unit), subset in frame.dropna(subset=[column]).groupby(["dose", "dose_unit"]):
                dose_records.append({
                    "level": level,
                    "model": model,
                    "dose": float(dose),
                    "dose_unit": unit,
                    "n_conditions": len(subset),
                    "pearson": float(subset[column].mean()),
                })
    dose_summary = pd.DataFrame(dose_records)

    magnitude_records = []
    magnitude_columns = {
        "full": "truth_delta_rms",
        "drug_residual": "truth_drug_residual_rms",
        "template_removed": "truth_template_removed_rms",
    }
    for level, magnitude_column in magnitude_columns.items():
        eligible = frame[magnitude_column].notna()
        quartile = pd.Series(index=frame.index, dtype="object")
        quartile.loc[eligible] = (
            frame.loc[eligible]
            .groupby("cell_line", group_keys=False)[magnitude_column]
            .apply(magnitude_quartiles)
            .astype(str)
        )
        for model in MODEL_NAMES:
            column = f"{model}__{level}__pearson"
            working = frame.loc[eligible, [column]].copy()
            working["quartile"] = quartile.loc[eligible]
            for label, subset in working.groupby("quartile", sort=True):
                magnitude_records.append({
                    "level": level,
                    "model": model,
                    "within_line_magnitude_quartile": label,
                    "n_conditions": len(subset),
                    "pearson": float(subset[column].mean()),
                })
    magnitude_summary = pd.DataFrame(magnitude_records)

    def atomic_csv(dataframe: pd.DataFrame, name: str) -> None:
        path = output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        dataframe.to_csv(temporary, index=False)
        os.replace(temporary, path)

    atomic_csv(frame, "per_condition_metrics.csv")
    atomic_csv(line_summary, "per_line_summary.csv")
    atomic_csv(bootstrap, "drug_cluster_bootstrap.csv")
    atomic_csv(neighbors, "nearest_neighbors.csv")
    atomic_csv(dose_summary, "dose_summary.csv")
    atomic_csv(magnitude_summary, "magnitude_summary.csv")

    prediction_path = output_dir / "nearest_predictions.npz"
    prediction_temporary = output_dir / "nearest_predictions.tmp.npz"
    np.savez_compressed(
        prediction_temporary,
        genes=np.asarray(genes, dtype=str),
        cell_line=np.asarray(cells, dtype=str),
        treatment=np.asarray(treatments, dtype=str),
        covered=nearest_covered,
        nearest_dmso_line=nearest.astype(np.float32),
    )
    os.replace(prediction_temporary, prediction_path)

    summary = {
        "purpose": "small diagnostic signal pilot; not a final benchmark estimate",
        "split": args.split,
        "holdout_cells": holdout_cells,
        "n_test_conditions": len(frame),
        "n_common_template_conditions": len(common_indices),
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_unit": "drug, with all doses kept together",
        "seed": args.seed,
        "nearest_definition": "one reference line maximizing Pearson across all-plate mean log1p DMSO HVG expression",
        "nearest_coverage_count": int(nearest_covered.sum()),
        "nearest_coverage_fraction": float(nearest_covered.mean()),
        "data_revision": args.data_revision,
        "split_file": str(split_path),
        "split_file_sha256": sha256(split_path),
        "baseline_summary": str(baseline_summary_path),
        "baseline_summary_sha256": sha256(baseline_summary_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "models": list(MODEL_NAMES),
        "levels": list(LEVELS),
    }
    write_json(summary, output_dir / "summary.json")
    print(neighbors.to_string(index=False), flush=True)
    print("\nPer-line Pearson:", flush=True)
    print(
        line_summary.pivot_table(index=["cell_line", "level"], columns="model", values="pearson").to_string(),
        flush=True,
    )
    print(f"\nWrote pilot outputs to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=int, default=5, choices=range(5, 10))
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-revision", default=DEFAULT_DATA_REVISION)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run_pilot(parse_args())
