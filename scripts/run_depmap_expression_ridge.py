#!/usr/bin/env python3
"""Leakage-safe DepMap-expression ridge on Tahoe drug-response residuals.

For each of the five Rhaister outer splits, this script:

* excludes every held-out line from DepMap scaling and PCA;
* uses line-level ridge LOOCV on the reference lines to choose a small PCA
  dimension and ridge penalty;
* predicts residuals around the frozen 45-line drug-dose mean;
* evaluates all expression-covered held-out lines against drug mean and
  frozen Rhaister-O;
* repeats hyperparameter selection under cell-label permutations to form a
  null distribution.

It implements experiment-plan steps 5--6 for DepMap expression only.  Other
modalities and nonlinear models are gated on the saved summary decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_rhaister_signal import clustered_bootstrap, parse_treatment
from evaluate_all_rhaister import remove_pc
from run_rhaister_o_compact import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_REVISION,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT as BASELINE_OUTPUT_ROOT,
    DEFAULT_PROJECT_ROOT,
    build_or_load_compact_cache,
    load_split,
    rowwise_metrics,
)


DEFAULT_DEPMAP_ROOT = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/depmap_24q4")
DEFAULT_COMPONENT_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_all_split_evaluation"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/depmap_expression_residual_ridge"
)
LEVELS = ("full", "drug_residual", "train_pca1_removed")


@dataclass
class RidgeCandidate:
    n_components: int
    ridge_lambda: float
    design: np.ndarray
    inverse_gram: np.ndarray
    hat: np.ndarray
    loo_operator: np.ndarray
    loo_rmse: float


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


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_depmap_expression(
    depmap_root: Path,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, dict[str, str]], Path, Path]:
    expression_path = depmap_root / "tahoe_OmicsExpressionProteinCodingGenesTPMLogp1.parquet"
    mapping_path = depmap_root / "tahoe_observed_Model.csv"
    expression = pd.read_parquet(expression_path)
    expression.index = expression.index.astype(str)
    mapping = pd.read_csv(mapping_path)
    mapping = mapping.dropna(subset=["RRID", "ModelID"])
    cvcl_to_model = dict(zip(mapping["RRID"].astype(str), mapping["ModelID"].astype(str)))
    metadata = {
        str(row.RRID): {
            "model_id": str(row.ModelID),
            "cell_line_name": str(row.CellLineName),
            "lineage": str(row.OncotreeLineage),
        }
        for row in mapping.itertuples()
    }
    return expression, cvcl_to_model, metadata, expression_path, mapping_path


def reference_only_pca(
    reference: np.ndarray,
    query: np.ndarray,
    unwhitened: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Standardize genes and fit PCA using reference lines only.

    unwhitened=True (E6, 2026-09-29): centre only -- no per-gene scaling and no
    per-component unit-variance rescaling -- so ridge on the leading components
    is ridge on the raw log-TPM features, penalising low-variance directions more.
    """
    reference64 = np.asarray(reference, dtype=np.float64)
    query64 = np.asarray(query, dtype=np.float64)
    mean = np.nanmean(reference64, axis=0)
    reference_filled = np.where(np.isfinite(reference64), reference64, mean)
    query_filled = np.where(np.isfinite(query64), query64, mean)
    scale = reference_filled.std(axis=0)
    keep = np.isfinite(mean) & np.isfinite(scale) & (scale > 1e-8)
    if unwhitened:
        scale = np.ones_like(scale)
    if int(keep.sum()) < 2:
        raise ValueError("Too few variable DepMap expression genes")
    reference_scaled = (reference_filled[:, keep] - mean[keep]) / scale[keep]
    query_scaled = (query_filled[:, keep] - mean[keep]) / scale[keep]
    u, singular_values, vt = np.linalg.svd(reference_scaled, full_matrices=False)
    scores = u * singular_values[None, :]
    query_scores = query_scaled @ vt.T
    score_scale = scores.std(axis=0)
    usable = score_scale > 1e-10
    divisor = np.ones_like(score_scale) if unwhitened else score_scale
    scores = scores[:, usable] / divisor[usable]
    query_scores = query_scores[:, usable] / divisor[usable]
    return scores, query_scores, {
        "n_input_genes": reference.shape[1],
        "n_variable_genes": int(keep.sum()),
        "n_available_components": scores.shape[1],
    }


def loo_rmse_from_gram(loo_operator: np.ndarray, response_gram: np.ndarray, n_outputs: int) -> float:
    sse = float(np.sum((loo_operator @ response_gram) * loo_operator))
    n_samples = loo_operator.shape[0]
    return float(np.sqrt(max(sse, 0.0) / (n_samples * n_outputs)))


def build_candidates(
    pca_scores: np.ndarray,
    response_gram: np.ndarray,
    n_outputs: int,
    component_grid: tuple[int, ...],
    lambda_grid: tuple[float, ...],
) -> list[RidgeCandidate]:
    candidates: list[RidgeCandidate] = []
    configurations = [(0, 0.0)] + [
        (components, ridge_lambda)
        for components in component_grid
        for ridge_lambda in lambda_grid
        if components <= pca_scores.shape[1]
    ]
    for components, ridge_lambda in configurations:
        if components == 0:
            design = np.ones((len(pca_scores), 1), dtype=np.float64)
        else:
            design = np.column_stack([
                np.ones(len(pca_scores), dtype=np.float64),
                pca_scores[:, :components],
            ])
        penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_lambda
        penalty[0, 0] = 0.0
        inverse_gram = np.linalg.inv(design.T @ design + penalty)
        hat = design @ inverse_gram @ design.T
        denominator = np.maximum(1.0 - np.diag(hat), 1e-8)
        loo_operator = (np.eye(len(design)) - hat) / denominator[:, None]
        candidates.append(RidgeCandidate(
            n_components=components,
            ridge_lambda=ridge_lambda,
            design=design,
            inverse_gram=inverse_gram,
            hat=hat,
            loo_operator=loo_operator,
            loo_rmse=loo_rmse_from_gram(loo_operator, response_gram, n_outputs),
        ))
    return candidates


def target_weights(candidate: RidgeCandidate, query_scores: np.ndarray) -> np.ndarray:
    if candidate.n_components == 0:
        query_design = np.ones((len(query_scores), 1), dtype=np.float64)
    else:
        query_design = np.column_stack([
            np.ones(len(query_scores), dtype=np.float64),
            query_scores[:, :candidate.n_components],
        ])
    return query_design @ candidate.inverse_gram @ candidate.design.T


def response_cube(
    compact,
    reference_cells: list[str],
    candidate_treatments: list[str],
) -> tuple[np.ndarray, list[str]]:
    lookup = {
        (cell, treatment): index
        for index, (cell, treatment) in enumerate(zip(compact.cells, compact.treatments))
    }
    common = [
        treatment for treatment in candidate_treatments
        if all((cell, treatment) in lookup for cell in reference_cells)
    ]
    indices = np.array([
        lookup[(cell, treatment)]
        for cell in reference_cells
        for treatment in common
    ])
    cube = np.asarray(compact.delta[indices], dtype=np.float32).reshape(
        len(reference_cells), len(common), len(compact.genes)
    )
    return cube, common


def map_drug_means(treatments: np.ndarray, means: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for treatment, vector in zip(treatments, means):
        if treatment in result:
            if not np.allclose(result[treatment], vector, rtol=0, atol=1e-7):
                raise ValueError(f"Drug-mean prediction differs across lines for {treatment}")
        else:
            result[treatment] = vector
    return result


def add_model_metrics(
    frame: pd.DataFrame,
    model: str,
    truth: np.ndarray,
    drug_mean: np.ndarray,
    prediction: np.ndarray,
    pc_mean: np.ndarray,
    pc1: np.ndarray,
) -> None:
    truth_residual = truth - drug_mean
    prediction_residual = prediction - drug_mean
    arrays = {
        "full": (truth, prediction),
        "drug_residual": (truth_residual, prediction_residual),
        "train_pca1_removed": (
            remove_pc(truth_residual, pc_mean, pc1),
            remove_pc(prediction_residual, pc_mean, pc1),
        ),
    }
    for level, (level_truth, level_prediction) in arrays.items():
        metrics = rowwise_metrics(level_truth, level_prediction)
        for metric, values in metrics.items():
            frame[f"{model}__{level}__{metric}"] = values


def split_analysis(
    args: argparse.Namespace,
    split: int,
    compact,
    expression: pd.DataFrame,
    cvcl_to_model: dict[str, str],
    metadata: dict[str, dict[str, str]],
) -> dict:
    repo_root = args.project_root / "external" / "Rhaister"
    holdout_cells, test_treatment_sets, split_path = load_split(repo_root, split)
    holdout = set(holdout_cells)
    reference_all = sorted(set(compact.cells.tolist()) - holdout)
    reference_cells = [
        cell for cell in reference_all
        if cvcl_to_model.get(cell) in expression.index
    ]
    available_queries = [
        cell for cell in holdout_cells
        if cvcl_to_model.get(cell) in expression.index
    ]

    baseline_path = args.baseline_root / f"split_{split}" / "predictions.npz"
    with np.load(baseline_path, allow_pickle=False) as archive:
        genes = archive["genes"].copy()
        cells = archive["cell_line"].copy()
        treatments = archive["treatment"].copy()
        truth = archive["truth"].astype(np.float32)
        drug_mean = archive["perturbation_mean"].astype(np.float32)
        rhaister = archive["rhaister_o"].astype(np.float32)
    candidate_treatments = sorted(set(treatments.tolist()))
    drug_mean_map = map_drug_means(treatments, drug_mean)
    response, common_treatments = response_cube(
        compact, reference_cells, candidate_treatments
    )
    common_set = set(common_treatments)
    reference_mean = np.stack([drug_mean_map[t] for t in common_treatments])
    response_residual = response - reference_mean[None, :, :]
    response_flat = response_residual.reshape(len(reference_cells), -1)
    response_gram = (response_flat @ response_flat.T).astype(np.float64)

    reference_x = expression.loc[
        [cvcl_to_model[cell] for cell in reference_cells]
    ].to_numpy(dtype=np.float64)
    query_x = expression.loc[
        [cvcl_to_model[cell] for cell in available_queries]
    ].to_numpy(dtype=np.float64)
    pca_scores, query_scores, pca_info = reference_only_pca(
        reference_x, query_x, unwhitened=getattr(args, "unwhitened", False)
    )
    candidates = build_candidates(
        pca_scores,
        response_gram,
        response_flat.shape[1],
        tuple(args.component_grid),
        tuple(args.lambda_grid),
    )
    selected_index = int(np.argmin([candidate.loo_rmse for candidate in candidates]))
    selected = candidates[selected_index]
    weights_by_candidate = [target_weights(candidate, query_scores) for candidate in candidates]
    predicted_common = weights_by_candidate[selected_index] @ response_flat
    predicted_common = predicted_common.reshape(
        len(available_queries), len(common_treatments), len(compact.genes)
    )

    prediction_residual = np.zeros_like(truth, dtype=np.float32)
    covered = np.zeros(len(cells), dtype=bool)
    query_index = {cell: index for index, cell in enumerate(available_queries)}
    common_index = {treatment: index for index, treatment in enumerate(common_treatments)}
    for row, (cell, treatment) in enumerate(zip(cells, treatments)):
        if cell in query_index and treatment in common_index:
            prediction_residual[row] = predicted_common[
                query_index[cell], common_index[treatment]
            ]
            covered[row] = True
    prediction = drug_mean + prediction_residual

    component_path = args.component_root / f"split_{split}" / "train_residual_pc1.npz"
    with np.load(component_path, allow_pickle=False) as component_archive:
        pc_mean = component_archive["mean"].astype(np.float32)
        pc1 = component_archive["component"].astype(np.float32)

    parsed = [parse_treatment(label) for label in treatments]
    frame = pd.DataFrame({
        "split": split,
        "cell_line": cells,
        "treatment": treatments,
        "drug": [item[0] for item in parsed],
        "dose": [item[1] for item in parsed],
        "dose_unit": [item[2] for item in parsed],
        "depmap_expression_available": [cell in query_index for cell in cells],
        "depmap_ridge_covered": covered,
    })
    add_model_metrics(frame, "perturbation_mean", truth, drug_mean, drug_mean, pc_mean, pc1)
    add_model_metrics(frame, "rhaister_o", truth, drug_mean, rhaister, pc_mean, pc1)
    add_model_metrics(frame, "depmap_expression_ridge", truth, drug_mean, prediction, pc_mean, pc1)

    line_records = []
    bootstrap_records = []
    for cell in holdout_cells:
        subset = frame[frame["cell_line"].eq(cell)]
        available = cell in query_index
        for level in LEVELS:
            for model in ("perturbation_mean", "rhaister_o", "depmap_expression_ridge"):
                line_records.append({
                    "split": split,
                    "cell_line": cell,
                    "cell_line_name": metadata.get(cell, {}).get("cell_line_name"),
                    "lineage": metadata.get(cell, {}).get("lineage"),
                    "depmap_expression_available": available,
                    "level": level,
                    "model": model,
                    "n_conditions": len(subset),
                    "coverage_fraction": float(subset["depmap_ridge_covered"].mean()) if available else 0.0,
                    **{
                        metric: float(subset[f"{model}__{level}__{metric}"].mean())
                        for metric in ("pearson", "rmse", "mae")
                    },
                })
            if available:
                depmap_values = subset[f"depmap_expression_ridge__{level}__pearson"].to_numpy()
                for comparison, comparison_model in (
                    ("drug_mean", "perturbation_mean"),
                    ("rhaister_o", "rhaister_o"),
                ):
                    baseline_values = subset[f"{comparison_model}__{level}__pearson"].to_numpy()
                    result = clustered_bootstrap(
                        depmap_values,
                        baseline_values,
                        subset["drug"].to_numpy(),
                        args.n_bootstrap,
                        args.seed + split * 1000 + holdout_cells.index(cell) * 20 + LEVELS.index(level) * 2 + (comparison == "rhaister_o"),
                    )
                    bootstrap_records.append({
                        "split": split,
                        "cell_line": cell,
                        "level": level,
                        "model": "depmap_expression_ridge",
                        "comparison": comparison,
                        **result,
                    })

    # Permute the reference-line response labels relative to fixed DepMap
    # features.  Hyperparameter selection is repeated inside every null draw.
    permutation_records = []
    rng = np.random.default_rng(args.seed + split * 100_000)
    available_row_lookup = {
        cell: np.flatnonzero(cells == cell) for cell in available_queries
    }
    for permutation_id in range(args.n_permutations):
        permutation = rng.permutation(len(reference_cells))
        permuted_gram = response_gram[np.ix_(permutation, permutation)]
        permuted_scores = [
            loo_rmse_from_gram(candidate.loo_operator, permuted_gram, response_flat.shape[1])
            for candidate in candidates
        ]
        null_index = int(np.argmin(permuted_scores))
        null_common = weights_by_candidate[null_index] @ response_flat[permutation]
        null_common = null_common.reshape(
            len(available_queries), len(common_treatments), len(compact.genes)
        )
        for query_position, cell in enumerate(available_queries):
            rows = available_row_lookup[cell]
            null_residual = np.zeros((len(rows), len(compact.genes)), dtype=np.float32)
            for local_row, global_row in enumerate(rows):
                treatment = treatments[global_row]
                if treatment in common_index:
                    null_residual[local_row] = null_common[
                        query_position, common_index[treatment]
                    ]
            truth_residual = truth[rows] - drug_mean[rows]
            null_levels = {
                "drug_residual": (
                    truth_residual,
                    null_residual,
                ),
                "train_pca1_removed": (
                    remove_pc(truth_residual, pc_mean, pc1),
                    remove_pc(null_residual, pc_mean, pc1),
                ),
            }
            for level, (level_truth, level_prediction) in null_levels.items():
                pearson = rowwise_metrics(level_truth, level_prediction)["pearson"]
                permutation_records.append({
                    "permutation_id": permutation_id,
                    "split": split,
                    "cell_line": cell,
                    "level": level,
                    "pearson": float(pearson.mean()),
                    "selected_components": candidates[null_index].n_components,
                    "selected_lambda": candidates[null_index].ridge_lambda,
                })

    output_dir = args.output_root / f"split_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    line_summary = pd.DataFrame(line_records)
    bootstrap = pd.DataFrame(bootstrap_records)
    permutations = pd.DataFrame(permutation_records)
    candidate_frame = pd.DataFrame([
        {
            "n_components": candidate.n_components,
            "ridge_lambda": candidate.ridge_lambda,
            "loo_rmse": candidate.loo_rmse,
            "selected": index == selected_index,
        }
        for index, candidate in enumerate(candidates)
    ])
    atomic_csv(frame, output_dir / "per_condition_metrics.csv")
    atomic_csv(line_summary, output_dir / "per_line_summary.csv")
    atomic_csv(bootstrap, output_dir / "drug_cluster_bootstrap.csv")
    atomic_csv(permutations, output_dir / "permutation_null.csv")
    atomic_csv(candidate_frame, output_dir / "ridge_selection.csv")

    prediction_path = output_dir / "predictions.npz"
    prediction_temporary = output_dir / "predictions.tmp.npz"
    np.savez_compressed(
        prediction_temporary,
        genes=np.asarray(genes, dtype=str),
        cell_line=np.asarray(cells, dtype=str),
        treatment=np.asarray(treatments, dtype=str),
        depmap_expression_available=frame["depmap_expression_available"].to_numpy(),
        covered=covered,
        depmap_expression_ridge=prediction.astype(np.float32),
    )
    os.replace(prediction_temporary, prediction_path)

    split_summary = {
        "split": split,
        "split_file": str(split_path),
        "split_file_sha256": sha256(split_path),
        "n_reference_lines_total": len(reference_all),
        "n_reference_lines_with_expression": len(reference_cells),
        "reference_lines_missing_expression": sorted(set(reference_all) - set(reference_cells)),
        "heldout_lines": holdout_cells,
        "heldout_lines_with_expression": available_queries,
        "heldout_lines_missing_expression": sorted(holdout - set(available_queries)),
        "n_candidate_treatments": len(candidate_treatments),
        "n_common_treatments": len(common_treatments),
        "coverage_fraction_available_queries": float(
            frame.loc[frame["depmap_expression_available"], "depmap_ridge_covered"].mean()
        ),
        "pca": pca_info,
        "selected_components": selected.n_components,
        "selected_lambda": selected.ridge_lambda,
        "selected_loo_rmse": selected.loo_rmse,
        "per_line": line_records,
    }
    write_json(split_summary, output_dir / "summary.json")
    print(
        f"split {split}: reference expression={len(reference_cells)}/{len(reference_all)}, "
        f"query expression={len(available_queries)}/{len(holdout_cells)}, "
        f"common treatments={len(common_treatments)}/{len(candidate_treatments)}, "
        f"selected PC={selected.n_components}, lambda={selected.ridge_lambda:g}, "
        f"LOO RMSE={selected.loo_rmse:.6f}",
        flush=True,
    )
    return split_summary


def line_bootstrap_interval(values: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    sampled = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def main(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    repo_root = project_root / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    expression, cvcl_to_model, metadata, expression_path, mapping_path = load_depmap_expression(
        args.depmap_root
    )
    split_summaries = [
        split_analysis(
            args, split, compact, expression, cvcl_to_model, metadata
        )
        for split in range(5, 10)
    ]

    line_summary = pd.concat([
        pd.read_csv(args.output_root / f"split_{split}" / "per_line_summary.csv")
        for split in range(5, 10)
    ], ignore_index=True)
    bootstrap = pd.concat([
        pd.read_csv(args.output_root / f"split_{split}" / "drug_cluster_bootstrap.csv")
        for split in range(5, 10)
    ], ignore_index=True)
    permutations = pd.concat([
        pd.read_csv(args.output_root / f"split_{split}" / "permutation_null.csv")
        for split in range(5, 10)
    ], ignore_index=True)
    atomic_csv(line_summary, args.output_root / "per_line_summary.csv")
    atomic_csv(bootstrap, args.output_root / "drug_cluster_bootstrap.csv")
    atomic_csv(permutations, args.output_root / "permutation_null.csv")

    available = line_summary[line_summary["depmap_expression_available"]]
    aggregate_records = []
    gate_details = {}
    for level in LEVELS:
        model_table = available[available["level"].eq(level)].pivot(
            index="cell_line", columns="model", values="pearson"
        )
        for model in model_table.columns:
            mean, low, high = line_bootstrap_interval(
                model_table[model].to_numpy(), args.seed + LEVELS.index(level) * 100 + len(model)
            )
            aggregate_records.append({
                "level": level,
                "comparison": model,
                "n_lines": len(model_table),
                "line_macro_mean": mean,
                "line_bootstrap_low": low,
                "line_bootstrap_high": high,
            })
        for comparison in ("perturbation_mean", "rhaister_o"):
            difference = (
                model_table["depmap_expression_ridge"] - model_table[comparison]
            ).to_numpy()
            mean, low, high = line_bootstrap_interval(
                difference,
                args.seed + LEVELS.index(level) * 1000 + (comparison == "rhaister_o"),
            )
            aggregate_records.append({
                "level": level,
                "comparison": f"depmap_minus_{comparison}",
                "n_lines": len(model_table),
                "line_macro_mean": mean,
                "line_bootstrap_low": low,
                "line_bootstrap_high": high,
            })

        if level != "full":
            actual = float(model_table["depmap_expression_ridge"].mean())
            null = (
                permutations[permutations["level"].eq(level)]
                .groupby("permutation_id")["pearson"]
                .mean()
                .to_numpy()
            )
            p_value = float((1 + np.sum(null >= actual)) / (1 + len(null)))
            gate_details[level] = {
                "actual_line_macro": actual,
                "null_mean": float(null.mean()),
                "null_95th_percentile": float(np.quantile(null, 0.95)),
                "empirical_p_value": p_value,
            }

    aggregate = pd.DataFrame(aggregate_records)
    atomic_csv(aggregate, args.output_root / "aggregate_summary.csv")
    residual_rows = aggregate[aggregate["level"].eq("drug_residual")].set_index("comparison")
    stable_over_drug_mean = bool(
        residual_rows.loc["depmap_minus_perturbation_mean", "line_bootstrap_low"] > 0
    )
    stable_over_rhaister = bool(
        residual_rows.loc["depmap_minus_rhaister_o", "line_bootstrap_low"] > 0
    )
    exceeds_permutation = bool(
        gate_details["drug_residual"]["actual_line_macro"]
        > gate_details["drug_residual"]["null_95th_percentile"]
        and gate_details["drug_residual"]["empirical_p_value"] <= 0.05
    )
    expand_gate = stable_over_drug_mean and stable_over_rhaister and exceeds_permutation

    baseline_summary = args.baseline_root / "summary.json"
    component_summary = args.component_root / "summary.json"
    summary = {
        "purpose": "experiment-plan steps 5--6: DepMap-expression residual ridge and gate",
        "data_revision": args.data_revision,
        "depmap_release": "Public 24Q4",
        "depmap_expression": str(expression_path),
        "depmap_expression_sha256": sha256(expression_path),
        "depmap_mapping": str(mapping_path),
        "depmap_mapping_sha256": sha256(mapping_path),
        "baseline_summary": str(baseline_summary),
        "baseline_summary_sha256": sha256(baseline_summary),
        "component_summary": str(component_summary),
        "component_summary_sha256": sha256(component_summary),
        "component_grid": args.component_grid,
        "lambda_grid": args.lambda_grid,
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "preprocessing": "outer-reference-only gene scaling and PCA",
        "selection": "reference-line analytic ridge LOOCV; repeated under each permutation",
        "runner_sha256": sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "n_heldout_lines_total": 25,
        "n_heldout_lines_with_expression": int(
            available[available["level"].eq("full")]["cell_line"].nunique()
        ),
        "gate": {
            "stable_over_drug_mean": stable_over_drug_mean,
            "stable_over_rhaister_o": stable_over_rhaister,
            "exceeds_permutation_null": exceeds_permutation,
            "expand_to_other_modalities_or_mlp": expand_gate,
            "rule": (
                "Require drug-residual line-bootstrap lower bound >0 versus both drug mean "
                "and Rhaister-O, plus actual line-macro above permutation 95th percentile "
                "with empirical p<=0.05."
            ),
        },
        "permutation": gate_details,
        "aggregate": aggregate_records,
        "per_split": split_summaries,
    }
    write_json(summary, args.output_root / "summary.json")
    print("\nAggregate available-line results:", flush=True)
    print(aggregate.to_string(index=False), flush=True)
    print("\nPermutation gate:", json.dumps(gate_details, indent=2), flush=True)
    print("\nExpansion gate:", json.dumps(summary["gate"], indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_OUTPUT_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--depmap-root", type=Path, default=DEFAULT_DEPMAP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-revision", default=DEFAULT_DATA_REVISION)
    parser.add_argument("--component-grid", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unwhitened", action="store_true",
                        help="E6: ridge on raw centred log-TPM (no gene scaling, no PC whitening)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
