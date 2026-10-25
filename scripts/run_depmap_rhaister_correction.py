#!/usr/bin/env python3
"""Test whether DepMap expression corrects frozen Rhaister-O errors.

The response target for this stage is not the drug-mean residual.  For every
outer reference line we first construct a leave-one-line-out Rhaister-O
prediction and use ``truth - cross-fitted Rhaister-O`` as the correction
target.  DepMap scaling, PCA, ridge selection, and correction fitting all use
outer reference lines only.  The final prediction is the frozen outer-test
Rhaister-O prediction plus the inferred correction.

The expansion gate requires a positive held-out-line bootstrap interval for
the drug-residual Pearson improvement over frozen Rhaister-O and an observed
improvement above a label-permutation null in which ridge selection is
repeated.  This is the final experiment in the predeclared baseline sequence;
it does not run additional DepMap modalities or an MLP.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_all_rhaister import remove_pc
from evaluate_rhaister_signal import clustered_bootstrap, parse_treatment
from run_depmap_expression_ridge import (
    DEFAULT_COMPONENT_ROOT,
    DEFAULT_DEPMAP_ROOT,
    LEVELS,
    add_model_metrics,
    atomic_csv,
    build_candidates,
    line_bootstrap_interval,
    load_depmap_expression,
    loo_rmse_from_gram,
    reference_only_pca,
    response_cube,
    sha256,
    target_weights,
    write_json,
)
from run_rhaister_o_compact import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_REVISION,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT as BASELINE_OUTPUT_ROOT,
    DEFAULT_PROJECT_ROOT,
    build_or_load_compact_cache,
    condition_hvg,
    load_hvg_centroids,
    load_split,
    rowwise_metrics,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/depmap_rhaister_residual_correction"
)
MODEL_NAMES = ("rhaister_o", "rhaister_depmap_correction")


def line_hvg_vectors(compact, hvg: dict, cells: list[str]) -> np.ndarray:
    """Match Rhaister-O's per-line mean log1p DMSO feature construction."""
    vectors = []
    for cell in cells:
        indices = np.flatnonzero(compact.cells == cell)
        values, missing = condition_hvg(
            compact.cells, compact.plates, indices, hvg, len(compact.genes)
        )
        if missing:
            raise ValueError(f"Missing {missing} DMSO feature rows for {cell}")
        vectors.append(np.log1p(values).mean(axis=0, dtype=np.float64))
    return np.stack(vectors)


def loo_rhaister_corrections(
    response: np.ndarray,
    raw_hvg: np.ndarray,
    ridge_lambda: float = 1.0,
    blend: float = 0.95,
) -> np.ndarray:
    """Return truth minus exact balanced-design leave-one-line-out Rhaister.

    ``response`` is line x treatment x gene and every treatment is observed in
    every line.  The closed form is algebraically identical to fitting the
    pinned diagonal model C times, but avoids repeatedly materializing its
    training cube.
    """
    y = np.asarray(response, dtype=np.float64)
    z = np.asarray(raw_hvg, dtype=np.float64)
    if y.ndim != 3 or z.shape != (y.shape[0], y.shape[2]):
        raise ValueError(f"Incompatible response {y.shape} and DMSO {z.shape}")
    n_lines = y.shape[0]
    if n_lines < 3:
        raise ValueError("At least three lines are required for LOO corrections")

    sum_y = y.sum(axis=0)
    sum_z = z.sum(axis=0)
    sum_z2 = np.square(z).sum(axis=0)
    sum_zy = np.einsum("cg,ctg->tg", z, y, optimize=True)
    corrections = np.empty_like(y, dtype=np.float32)
    n_train = n_lines - 1
    for line in range(n_lines):
        train_sum_y = sum_y - y[line]
        mean_y = train_sum_y / n_train
        mean_z = (sum_z - z[line]) / n_train
        target_z = z[line] - mean_z
        numerator = (
            sum_zy
            - z[line][None, :] * y[line]
            - mean_z[None, :] * train_sum_y
        )
        denominator = (
            sum_z2
            - np.square(z[line])
            - n_train * np.square(mean_z)
            + ridge_lambda
        )
        diagonal_prediction = mean_y + (
            blend * numerator / denominator[None, :] * target_z[None, :]
        )
        corrections[line] = (y[line] - diagonal_prediction).astype(np.float32)
    return corrections


def split_analysis(
    args: argparse.Namespace,
    split: int,
    compact,
    expression: pd.DataFrame,
    cvcl_to_model: dict[str, str],
    metadata: dict[str, dict[str, str]],
    hvg: dict,
) -> dict:
    repo_root = args.project_root / "external" / "Rhaister"
    holdout_cells, _, split_path = load_split(repo_root, split)
    holdout = set(holdout_cells)
    reference_all = sorted(set(compact.cells.tolist()) - holdout)
    reference_cells = [
        cell for cell in reference_all if cvcl_to_model.get(cell) in expression.index
    ]
    available_queries = [
        cell for cell in holdout_cells if cvcl_to_model.get(cell) in expression.index
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
    response_all, common_treatments = response_cube(
        compact, reference_all, candidate_treatments
    )
    raw_hvg = line_hvg_vectors(compact, hvg, reference_all)
    correction_all = loo_rhaister_corrections(
        response_all, raw_hvg, args.rhaister_lambda, args.rhaister_blend
    )
    reference_lookup = {cell: index for index, cell in enumerate(reference_all)}
    correction = correction_all[[reference_lookup[cell] for cell in reference_cells]]
    correction_flat = correction.reshape(len(reference_cells), -1)
    response_gram = (correction_flat @ correction_flat.T).astype(np.float64)

    reference_x = expression.loc[
        [cvcl_to_model[cell] for cell in reference_cells]
    ].to_numpy(dtype=np.float64)
    query_x = expression.loc[
        [cvcl_to_model[cell] for cell in available_queries]
    ].to_numpy(dtype=np.float64)
    pca_scores, query_scores, pca_info = reference_only_pca(reference_x, query_x)
    candidates = build_candidates(
        pca_scores,
        response_gram,
        correction_flat.shape[1],
        tuple(args.component_grid),
        tuple(args.lambda_grid),
    )
    selected_index = int(np.argmin([candidate.loo_rmse for candidate in candidates]))
    selected = candidates[selected_index]
    weights_by_candidate = [target_weights(candidate, query_scores) for candidate in candidates]
    predicted_common = weights_by_candidate[selected_index] @ correction_flat
    predicted_common = predicted_common.reshape(
        len(available_queries), len(common_treatments), len(compact.genes)
    )

    predicted_correction = np.zeros_like(truth, dtype=np.float32)
    covered = np.zeros(len(cells), dtype=bool)
    query_index = {cell: index for index, cell in enumerate(available_queries)}
    common_index = {treatment: index for index, treatment in enumerate(common_treatments)}
    for row, (cell, treatment) in enumerate(zip(cells, treatments)):
        if cell in query_index and treatment in common_index:
            predicted_correction[row] = predicted_common[
                query_index[cell], common_index[treatment]
            ]
            covered[row] = True
    corrected = rhaister + predicted_correction

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
        "depmap_correction_covered": covered,
    })
    add_model_metrics(frame, "rhaister_o", truth, drug_mean, rhaister, pc_mean, pc1)
    add_model_metrics(
        frame,
        "rhaister_depmap_correction",
        truth,
        drug_mean,
        corrected,
        pc_mean,
        pc1,
    )

    line_records = []
    bootstrap_records = []
    for cell in holdout_cells:
        subset = frame[frame["cell_line"].eq(cell)]
        available = cell in query_index
        for level in LEVELS:
            for model in MODEL_NAMES:
                line_records.append({
                    "split": split,
                    "cell_line": cell,
                    "cell_line_name": metadata.get(cell, {}).get("cell_line_name"),
                    "lineage": metadata.get(cell, {}).get("lineage"),
                    "depmap_expression_available": available,
                    "level": level,
                    "model": model,
                    "n_conditions": len(subset),
                    "coverage_fraction": (
                        float(subset["depmap_correction_covered"].mean()) if available else 0.0
                    ),
                    **{
                        metric: float(subset[f"{model}__{level}__{metric}"].mean())
                        for metric in ("pearson", "rmse", "mae")
                    },
                })
            if available:
                corrected_values = subset[
                    f"rhaister_depmap_correction__{level}__pearson"
                ].to_numpy()
                rhaister_values = subset[f"rhaister_o__{level}__pearson"].to_numpy()
                result = clustered_bootstrap(
                    corrected_values,
                    rhaister_values,
                    subset["drug"].to_numpy(),
                    args.n_bootstrap,
                    args.seed + split * 1000 + holdout_cells.index(cell) * 10 + LEVELS.index(level),
                )
                bootstrap_records.append({
                    "split": split,
                    "cell_line": cell,
                    "level": level,
                    "model": "rhaister_depmap_correction",
                    "comparison": "rhaister_o",
                    **result,
                })

    permutation_records = []
    rng = np.random.default_rng(args.seed + split * 100_000)
    available_row_lookup = {
        cell: np.flatnonzero(cells == cell) for cell in available_queries
    }
    for permutation_id in range(args.n_permutations):
        permutation = rng.permutation(len(reference_cells))
        permuted_gram = response_gram[np.ix_(permutation, permutation)]
        permuted_scores = [
            loo_rmse_from_gram(
                candidate.loo_operator, permuted_gram, correction_flat.shape[1]
            )
            for candidate in candidates
        ]
        null_index = int(np.argmin(permuted_scores))
        null_common = weights_by_candidate[null_index] @ correction_flat[permutation]
        null_common = null_common.reshape(
            len(available_queries), len(common_treatments), len(compact.genes)
        )
        for query_position, cell in enumerate(available_queries):
            rows = available_row_lookup[cell]
            null_correction = np.zeros((len(rows), len(compact.genes)), dtype=np.float32)
            for local_row, global_row in enumerate(rows):
                treatment = treatments[global_row]
                if treatment in common_index:
                    null_correction[local_row] = null_common[
                        query_position, common_index[treatment]
                    ]
            truth_residual = truth[rows] - drug_mean[rows]
            rhaister_residual = rhaister[rows] - drug_mean[rows]
            null_residual = rhaister_residual + null_correction
            null_levels = {
                "drug_residual": (truth_residual, rhaister_residual, null_residual),
                "train_pca1_removed": (
                    remove_pc(truth_residual, pc_mean, pc1),
                    remove_pc(rhaister_residual, pc_mean, pc1),
                    remove_pc(null_residual, pc_mean, pc1),
                ),
            }
            for level, (level_truth, level_rhaister, level_null) in null_levels.items():
                null_pearson = float(rowwise_metrics(level_truth, level_null)["pearson"].mean())
                rhaister_pearson = float(
                    rowwise_metrics(level_truth, level_rhaister)["pearson"].mean()
                )
                permutation_records.append({
                    "permutation_id": permutation_id,
                    "split": split,
                    "cell_line": cell,
                    "level": level,
                    "pearson": null_pearson,
                    "rhaister_pearson": rhaister_pearson,
                    "improvement": null_pearson - rhaister_pearson,
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
    temporary = output_dir / "predictions.tmp.npz"
    np.savez_compressed(
        temporary,
        genes=np.asarray(genes, dtype=str),
        cell_line=np.asarray(cells, dtype=str),
        treatment=np.asarray(treatments, dtype=str),
        depmap_expression_available=frame["depmap_expression_available"].to_numpy(),
        covered=covered,
        predicted_correction=predicted_correction.astype(np.float32),
        rhaister_depmap_correction=corrected.astype(np.float32),
    )
    os.replace(temporary, prediction_path)

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
            frame.loc[frame["depmap_expression_available"], "depmap_correction_covered"].mean()
        ),
        "pca": pca_info,
        "selected_components": selected.n_components,
        "selected_lambda": selected.ridge_lambda,
        "selected_loo_rmse": selected.loo_rmse,
        "per_line": line_records,
    }
    write_json(split_summary, output_dir / "summary.json")
    print(
        f"split {split}: correction references={len(reference_cells)}/{len(reference_all)}, "
        f"queries={len(available_queries)}/{len(holdout_cells)}, "
        f"common treatments={len(common_treatments)}/{len(candidate_treatments)}, "
        f"selected PC={selected.n_components}, lambda={selected.ridge_lambda:g}, "
        f"LOO correction RMSE={selected.loo_rmse:.6f}",
        flush=True,
    )
    return split_summary


def main(args: argparse.Namespace) -> None:
    repo_root = args.project_root / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    expression, cvcl_to_model, metadata, expression_path, mapping_path = load_depmap_expression(
        args.depmap_root
    )
    hvg = load_hvg_centroids(args.data_root, compact.genes)
    split_summaries = [
        split_analysis(args, split, compact, expression, cvcl_to_model, metadata, hvg)
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
    permutation_summary = {}
    for level in LEVELS:
        table = available[available["level"].eq(level)].pivot(
            index="cell_line", columns="model", values="pearson"
        )
        for model in MODEL_NAMES:
            mean, low, high = line_bootstrap_interval(
                table[model].to_numpy(), args.seed + LEVELS.index(level) * 100 + len(model)
            )
            aggregate_records.append({
                "level": level,
                "comparison": model,
                "n_lines": len(table),
                "line_macro_mean": mean,
                "line_bootstrap_low": low,
                "line_bootstrap_high": high,
            })
        difference = (table["rhaister_depmap_correction"] - table["rhaister_o"]).to_numpy()
        mean, low, high = line_bootstrap_interval(
            difference, args.seed + LEVELS.index(level) * 1000
        )
        aggregate_records.append({
            "level": level,
            "comparison": "correction_minus_rhaister_o",
            "n_lines": len(table),
            "line_macro_mean": mean,
            "line_bootstrap_low": low,
            "line_bootstrap_high": high,
        })
        if level != "full":
            null = (
                permutations[permutations["level"].eq(level)]
                .groupby("permutation_id")["improvement"]
                .mean()
                .to_numpy()
            )
            p_value = float((1 + np.sum(null >= mean)) / (1 + len(null)))
            permutation_summary[level] = {
                "actual_improvement": mean,
                "null_mean": float(null.mean()),
                "null_95th_percentile": float(np.quantile(null, 0.95)),
                "empirical_p_value": p_value,
            }

    aggregate = pd.DataFrame(aggregate_records)
    atomic_csv(aggregate, args.output_root / "aggregate_summary.csv")
    residual_difference = aggregate[
        aggregate["level"].eq("drug_residual")
        & aggregate["comparison"].eq("correction_minus_rhaister_o")
    ].iloc[0]
    stable_improvement = bool(residual_difference["line_bootstrap_low"] > 0)
    residual_permutation = permutation_summary["drug_residual"]
    exceeds_permutation = bool(
        residual_permutation["actual_improvement"]
        > residual_permutation["null_95th_percentile"]
        and residual_permutation["empirical_p_value"] <= 0.05
    )
    expand_gate = stable_improvement and exceeds_permutation

    baseline_summary = args.baseline_root / "summary.json"
    component_summary = args.component_root / "summary.json"
    summary = {
        "purpose": "DepMap-expression correction of cross-fitted/frozen Rhaister-O residuals",
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
        "rhaister_lambda": args.rhaister_lambda,
        "rhaister_blend": args.rhaister_blend,
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "preprocessing": "outer-reference-only gene scaling and PCA",
        "correction_target": "truth minus leave-one-line-out Rhaister-O on outer references",
        "selection": "reference-line analytic ridge LOOCV; repeated under each permutation",
        "runner_sha256": sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "n_heldout_lines_total": 25,
        "n_heldout_lines_with_expression": int(
            available[available["level"].eq("full")]["cell_line"].nunique()
        ),
        "gate": {
            "stable_improvement_over_rhaister_o": stable_improvement,
            "exceeds_permutation_null": exceeds_permutation,
            "expand_to_other_modalities_or_mlp": expand_gate,
            "rule": (
                "Require drug-residual line-bootstrap lower bound >0 for corrected minus "
                "frozen Rhaister-O, plus actual improvement above the permutation 95th "
                "percentile with empirical p<=0.05."
            ),
        },
        "permutation": permutation_summary,
        "aggregate": aggregate_records,
        "per_split": split_summaries,
    }
    write_json(summary, args.output_root / "summary.json")
    print("\nAggregate available-line results:", flush=True)
    print(aggregate.to_string(index=False), flush=True)
    print("\nPermutation gate:", json.dumps(permutation_summary, indent=2), flush=True)
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
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[1, 10, 100])
    parser.add_argument("--rhaister-lambda", type=float, default=1.0)
    parser.add_argument("--rhaister-blend", type=float, default=0.95)
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
