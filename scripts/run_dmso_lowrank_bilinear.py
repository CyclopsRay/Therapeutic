#!/usr/bin/env python3
"""Low-rank cross-gene response modelling on Rhaister-O's own DMSO input.

Rhaister-O is a per-gene *diagonal* map: for treatment ``p`` and gene ``g`` it
regresses the response on the target line's DMSO level of that same gene ``g``.
Gene ``g``'s predicted response therefore cannot depend on any other gene's
baseline, and no strength is shared across treatments.

This script asks whether that diagonal restriction discards recoverable
structure, using the *same* input Rhaister-O already consumes -- the
plate-matched Tahoe DMSO profile.  No new modality is introduced, so the failed
DepMap expansion gate does not apply: this is a modelling question, not a data
question.

The model is bilinear.  With a reference-only low-dimensional embedding
``z(c)`` of the line's DMSO profile and a shared response program ``B``,

    Delta_hat[c, p, g] = mu[p, g] + sum_k z_k(c) * B_k[p, g],

so gene ``g``'s response may depend on the whole baseline profile through a
rank-K bottleneck, and every treatment shares the same embedding.

Two arms are fitted on identical features and splits:

``dmso_lowrank_direct``
    Target is the drug-dose-mean residual.  Asks whether a purely cross-gene
    low-rank model can stand on its own against the diagonal map.
``dmso_lowrank_correction``
    Target is ``truth - leave-one-reference-line-out Rhaister-O``; the inferred
    correction is added to the frozen outer-test Rhaister-O prediction.  Asks
    the sharper question: does cross-gene structure remain *after* the diagonal
    map has taken what it can?

Every DMSO line has a profile in the compact cache, so both arms evaluate all
25 held-out lines rather than the 23 with DepMap expression.
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
    LEVELS,
    add_model_metrics,
    atomic_csv,
    build_candidates,
    line_bootstrap_interval,
    loo_rmse_from_gram,
    map_drug_means,
    reference_only_pca,
    response_cube,
    sha256,
    target_weights,
    write_json,
)
from run_depmap_rhaister_correction import (
    line_hvg_vectors,
    loo_rhaister_corrections,
)
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

DEFAULT_OUTPUT_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/dmso_lowrank_bilinear"
)
ARMS = ("dmso_lowrank_direct", "dmso_lowrank_correction")
MODELS = ("perturbation_mean", "rhaister_o") + ARMS


def fit_bilinear(
    target_flat: np.ndarray,
    scores: np.ndarray,
    query_scores: np.ndarray,
    n_query: int,
    n_common: int,
    n_genes: int,
    component_grid: tuple[int, ...],
    lambda_grid: tuple[float, ...],
    n_permutations: int,
    seed: int,
    on_permutation,
) -> dict:
    """Ridge from a reference-only DMSO embedding to a flattened response.

    ``target_flat`` is lines x (treatments * genes).  The fitted map is the
    rank-K bilinear program: selection is by analytic reference-line LOOCV and
    is repeated inside every permutation draw so the null inherits the same
    selection procedure.
    """
    gram = (target_flat @ target_flat.T).astype(np.float64)
    candidates = build_candidates(
        scores, gram, target_flat.shape[1], component_grid, lambda_grid
    )
    selected_index = int(np.argmin([candidate.loo_rmse for candidate in candidates]))
    selected = candidates[selected_index]
    weights_by_candidate = [target_weights(c, query_scores) for c in candidates]
    predicted = (weights_by_candidate[selected_index] @ target_flat).astype(
        np.float32
    ).reshape(n_query, n_common, n_genes)

    rng = np.random.default_rng(seed)
    n_reference = target_flat.shape[0]
    for permutation_id in range(n_permutations):
        # Whole reference lines are permuted against the response tensor, so
        # drugs and doses stay together and only the line-to-context pairing
        # is destroyed.
        permutation = rng.permutation(n_reference)
        permuted_gram = gram[np.ix_(permutation, permutation)]
        null_scores = [
            loo_rmse_from_gram(c.loo_operator, permuted_gram, target_flat.shape[1])
            for c in candidates
        ]
        null_index = int(np.argmin(null_scores))
        null_prediction = (
            weights_by_candidate[null_index] @ target_flat[permutation]
        ).astype(np.float32).reshape(n_query, n_common, n_genes)
        on_permutation(permutation_id, null_prediction)
        del null_prediction

    return {
        "prediction": predicted,
        "selected_components": selected.n_components,
        "selected_lambda": selected.ridge_lambda,
        "selected_loo_rmse": selected.loo_rmse,
        "candidates": [
            {
                "n_components": c.n_components,
                "ridge_lambda": c.ridge_lambda,
                "loo_rmse": c.loo_rmse,
                "selected": index == selected_index,
            }
            for index, c in enumerate(candidates)
        ],
    }


def split_analysis(
    args: argparse.Namespace,
    split: int,
    compact,
    hvg: dict,
    metadata: dict[str, dict[str, str]],
) -> dict:
    repo_root = args.project_root / "external" / "Rhaister"
    holdout_cells, _, split_path = load_split(repo_root, split)
    holdout = set(holdout_cells)
    reference_cells = sorted(set(compact.cells.tolist()) - holdout)

    baseline_path = args.baseline_root / f"split_{split}" / "predictions.npz"
    with np.load(baseline_path, allow_pickle=False) as archive:
        genes = archive["genes"].copy()
        cells = archive["cell_line"].copy()
        treatments = archive["treatment"].copy()
        truth = archive["truth"].astype(np.float32)
        drug_mean = archive["perturbation_mean"].astype(np.float32)
        rhaister = archive["rhaister_o"].astype(np.float32)

    candidate_treatments = sorted(set(treatments.tolist()))
    response, common_treatments = response_cube(
        compact, reference_cells, candidate_treatments
    )
    drug_mean_map = map_drug_means(treatments, drug_mean)
    reference_mean = np.stack([drug_mean_map[t] for t in common_treatments])

    # Arm 1 target: residual around the conserved drug-dose mean.
    direct_flat = (response - reference_mean[None, :, :]).reshape(
        len(reference_cells), -1
    )
    # Arm 2 target: what the diagonal map still gets wrong, cross-fitted so no
    # reference line contributes to its own Rhaister-O prediction.
    reference_hvg = line_hvg_vectors(compact, hvg, reference_cells)
    correction_flat = loo_rhaister_corrections(
        response, reference_hvg, args.rhaister_lambda, args.rhaister_blend
    ).reshape(len(reference_cells), -1)

    query_hvg = line_hvg_vectors(compact, hvg, holdout_cells)
    scores, query_scores, pca_info = reference_only_pca(reference_hvg, query_hvg)

    query_index = {cell: index for index, cell in enumerate(holdout_cells)}
    common_index = {t: index for index, t in enumerate(common_treatments)}
    row_lookup = {cell: np.flatnonzero(cells == cell) for cell in holdout_cells}

    def scatter(cube: np.ndarray) -> np.ndarray:
        out = np.zeros_like(truth, dtype=np.float32)
        for row, (cell, treatment) in enumerate(zip(cells, treatments)):
            if cell in query_index and treatment in common_index:
                out[row] = cube[query_index[cell], common_index[treatment]]
        return out

    covered = np.array(
        [
            cell in query_index and treatment in common_index
            for cell, treatment in zip(cells, treatments)
        ]
    )

    parsed = [parse_treatment(label) for label in treatments]
    frame = pd.DataFrame({
        "split": split,
        "cell_line": cells,
        "treatment": treatments,
        "drug": [item[0] for item in parsed],
        "dose": [item[1] for item in parsed],
        "covered": covered,
    })
    component_path = args.component_root / f"split_{split}" / "train_residual_pc1.npz"
    with np.load(component_path, allow_pickle=False) as component_archive:
        pc_mean = component_archive["mean"].astype(np.float32)
        pc1 = component_archive["component"].astype(np.float32)
    add_model_metrics(frame, "perturbation_mean", truth, drug_mean, drug_mean, pc_mean, pc1)
    add_model_metrics(frame, "rhaister_o", truth, drug_mean, rhaister, pc_mean, pc1)

    permutation_records: list[dict] = []
    arm_results: dict[str, dict] = {}

    for arm_position, arm in enumerate(ARMS):
        target_flat = direct_flat if arm == "dmso_lowrank_direct" else correction_flat
        # The direct arm predicts a residual around the drug mean; the
        # correction arm predicts an offset from frozen Rhaister-O.
        anchor = drug_mean if arm == "dmso_lowrank_direct" else rhaister

        def record_permutation(permutation_id: int, null_cube: np.ndarray, arm=arm, anchor=anchor) -> None:
            null_prediction = anchor + scatter(null_cube)
            for cell in holdout_cells:
                rows = row_lookup[cell]
                truth_residual = truth[rows] - drug_mean[rows]
                null_residual = null_prediction[rows] - drug_mean[rows]
                for level, (lt, lp) in {
                    "drug_residual": (truth_residual, null_residual),
                    "train_pca1_removed": (
                        remove_pc(truth_residual, pc_mean, pc1),
                        remove_pc(null_residual, pc_mean, pc1),
                    ),
                }.items():
                    permutation_records.append({
                        "permutation_id": permutation_id,
                        "split": split,
                        "arm": arm,
                        "cell_line": cell,
                        "level": level,
                        "pearson": float(rowwise_metrics(lt, lp)["pearson"].mean()),
                    })

        result = fit_bilinear(
            target_flat,
            scores,
            query_scores,
            len(holdout_cells),
            len(common_treatments),
            len(compact.genes),
            tuple(args.component_grid),
            tuple(args.lambda_grid),
            args.n_permutations,
            args.seed + split * 100_000 + arm_position * 7_919,
            record_permutation,
        )
        prediction = anchor + scatter(result["prediction"])
        add_model_metrics(frame, arm, truth, drug_mean, prediction, pc_mean, pc1)
        result.pop("prediction")
        arm_results[arm] = result

    line_records = []
    bootstrap_records = []
    for cell in holdout_cells:
        subset = frame[frame["cell_line"].eq(cell)]
        for level in LEVELS:
            for model in MODELS:
                line_records.append({
                    "split": split,
                    "cell_line": cell,
                    "cell_line_name": metadata.get(cell, {}).get("cell_line_name"),
                    "lineage": metadata.get(cell, {}).get("lineage"),
                    "level": level,
                    "model": model,
                    "n_conditions": len(subset),
                    **{
                        metric: float(subset[f"{model}__{level}__{metric}"].mean())
                        for metric in ("pearson", "rmse", "mae")
                    },
                })
            for arm in ARMS:
                result = clustered_bootstrap(
                    subset[f"{arm}__{level}__pearson"].to_numpy(),
                    subset[f"rhaister_o__{level}__pearson"].to_numpy(),
                    subset["drug"].to_numpy(),
                    args.n_bootstrap,
                    args.seed + split * 613 + holdout_cells.index(cell) * 29 + LEVELS.index(level),
                )
                bootstrap_records.append({
                    "split": split,
                    "cell_line": cell,
                    "level": level,
                    "arm": arm,
                    "versus": "rhaister_o",
                    **result,
                })

    output_dir = args.output_root / f"split_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(frame, output_dir / "per_condition_metrics.csv")
    atomic_csv(pd.DataFrame(line_records), output_dir / "per_line_summary.csv")
    atomic_csv(pd.DataFrame(bootstrap_records), output_dir / "drug_cluster_bootstrap.csv")
    atomic_csv(pd.DataFrame(permutation_records), output_dir / "permutation_null.csv")
    atomic_csv(
        pd.DataFrame([
            {"arm": arm, **candidate}
            for arm in ARMS
            for candidate in arm_results[arm]["candidates"]
        ]),
        output_dir / "ridge_selection.csv",
    )

    summary = {
        "split": split,
        "split_file": str(split_path),
        "split_file_sha256": sha256(split_path),
        "n_reference_lines": len(reference_cells),
        "heldout_lines": holdout_cells,
        "n_candidate_treatments": len(candidate_treatments),
        "n_common_treatments": len(common_treatments),
        "coverage_fraction": float(covered.mean()),
        "dmso_pca": pca_info,
        "arms": {
            arm: {k: v for k, v in arm_results[arm].items() if k != "candidates"}
            for arm in ARMS
        },
        "per_line": line_records,
    }
    write_json(summary, output_dir / "summary.json")
    selection = ", ".join(
        f"{arm}: K={arm_results[arm]['selected_components']}"
        f"/lam={arm_results[arm]['selected_lambda']:g}"
        for arm in ARMS
    )
    print(
        f"split {split}: reference={len(reference_cells)}, query={len(holdout_cells)}, "
        f"treatments={len(common_treatments)}/{len(candidate_treatments)}, "
        f"coverage={covered.mean():.4f} | {selection}",
        flush=True,
    )
    return summary


def main(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    repo_root = project_root / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    hvg = load_hvg_centroids(args.data_root, compact.genes)
    mapping = pd.read_csv(args.model_metadata)
    metadata = {
        str(row.RRID): {
            "cell_line_name": str(row.CellLineName),
            "lineage": str(row.OncotreeLineage),
        }
        for row in mapping.dropna(subset=["RRID"]).itertuples()
    }

    split_summaries = [
        split_analysis(args, split, compact, hvg, metadata)
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

    aggregate_records = []
    permutation_summary: dict[str, dict] = {}
    per_split_improvement: dict[str, dict[int, float]] = {arm: {} for arm in ARMS}
    for level in LEVELS:
        table = line_summary[line_summary["level"].eq(level)].pivot(
            index="cell_line", columns="model", values="pearson"
        )
        for model in table.columns:
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
        for arm in ARMS:
            difference = (table[arm] - table["rhaister_o"]).to_numpy()
            mean, low, high = line_bootstrap_interval(
                difference, args.seed + LEVELS.index(level) * 17 + len(arm)
            )
            aggregate_records.append({
                "level": level,
                "comparison": f"{arm}_minus_rhaister_o",
                "n_lines": len(table),
                "line_macro_mean": mean,
                "line_bootstrap_low": low,
                "line_bootstrap_high": high,
            })

        if level == "drug_residual":
            for arm in ARMS:
                for split in range(5, 10):
                    subset = line_summary[
                        line_summary["split"].eq(split) & line_summary["level"].eq(level)
                    ].pivot(index="cell_line", columns="model", values="pearson")
                    per_split_improvement[arm][split] = float(
                        (subset[arm] - subset["rhaister_o"]).mean()
                    )

        if level != "full":
            level_nulls = permutations[permutations["level"].eq(level)]
            permutation_summary[level] = {}
            for arm in ARMS:
                actual = float(table[arm].mean())
                null = (
                    level_nulls[level_nulls["arm"].eq(arm)]
                    .groupby("permutation_id")["pearson"]
                    .mean()
                    .to_numpy()
                )
                permutation_summary[level][arm] = {
                    "actual_line_macro": actual,
                    "null_mean": float(null.mean()),
                    "null_95th_percentile": float(np.quantile(null, 0.95)),
                    "n_permutations": int(len(null)),
                    "empirical_p_value": float((1 + np.sum(null >= actual)) / (1 + len(null))),
                }

    aggregate = pd.DataFrame(aggregate_records)
    atomic_csv(aggregate, args.output_root / "aggregate_summary.csv")

    residual = aggregate[aggregate["level"].eq("drug_residual")].set_index("comparison")
    pc1_removed = aggregate[aggregate["level"].eq("train_pca1_removed")].set_index("comparison")
    rmse_table = line_summary[line_summary["level"].eq("drug_residual")].pivot(
        index="cell_line", columns="model", values="rmse"
    )

    gates = {}
    for arm in ARMS:
        improvement = float(residual.loc[f"{arm}_minus_rhaister_o", "line_macro_mean"])
        low = float(residual.loc[f"{arm}_minus_rhaister_o", "line_bootstrap_low"])
        pc1_improvement = float(pc1_removed.loc[f"{arm}_minus_rhaister_o", "line_macro_mean"])
        splits_positive = sum(1 for value in per_split_improvement[arm].values() if value > 0)
        rmse_ratio = float(rmse_table[arm].mean() / rmse_table["rhaister_o"].mean())
        above_null = bool(
            permutation_summary["drug_residual"][arm]["actual_line_macro"]
            > permutation_summary["drug_residual"][arm]["null_95th_percentile"]
        )
        gates[arm] = {
            "improvement_drug_residual": improvement,
            "improvement_bootstrap_low": low,
            "improvement_pc1_removed": pc1_improvement,
            "splits_with_positive_improvement": splits_positive,
            "rmse_ratio_versus_rhaister": rmse_ratio,
            "above_permutation_null": above_null,
            "meets_min_improvement": bool(improvement >= args.min_improvement),
            "meets_positive_bootstrap": bool(low > 0),
            "meets_pc1_improvement": bool(pc1_improvement >= args.min_improvement),
            "meets_split_majority": bool(splits_positive >= 4),
            "meets_rmse_tolerance": bool(rmse_ratio <= 1.01),
            "passes": bool(
                improvement >= args.min_improvement
                and low > 0
                and pc1_improvement >= args.min_improvement
                and splits_positive >= 4
                and rmse_ratio <= 1.01
                and above_null
            ),
        }

    summary = {
        "purpose": (
            "Does a low-rank cross-gene bilinear model recover structure that "
            "Rhaister-O's per-gene diagonal map leaves on the table, using the "
            "same plate-matched Tahoe DMSO input?"
        ),
        "model": (
            "Delta_hat[c,p,g] = anchor[c,p,g] + sum_k z_k(c) B_k[p,g] with z a "
            "reference-only PCA embedding of the line's log1p DMSO HVG profile."
        ),
        "arms": {
            "dmso_lowrank_direct": "anchor = drug-dose mean; target = drug-mean residual",
            "dmso_lowrank_correction": (
                "anchor = frozen Rhaister-O; target = truth minus "
                "leave-one-reference-line-out Rhaister-O"
            ),
        },
        "data_revision": args.data_revision,
        "component_grid": args.component_grid,
        "lambda_grid": args.lambda_grid,
        "rhaister_lambda": args.rhaister_lambda,
        "rhaister_blend": args.rhaister_blend,
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "preprocessing": "reference-only DMSO standardization and PCA; selection by reference-line analytic LOOCV",
        "input_boundary": (
            "Only the target line's plate-matched DMSO profile is consumed, the "
            "same input the frozen Rhaister-O baseline uses. No DepMap, L1000, "
            "or target-line treated response enters the model."
        ),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gate_rule": (
            f"Require >= {args.min_improvement} absolute drug-residual line-macro "
            "Pearson improvement over frozen Rhaister-O, a positive paired "
            "line-bootstrap lower bound, the same improvement surviving "
            "reference-only PC1 removal, positive improvement in at least 4 of 5 "
            "outer splits, at most 1% drug-residual RMSE degradation, and a "
            "line-block permutation null below the observed value."
        ),
        "gates": gates,
        "per_split_improvement": per_split_improvement,
        "permutation": permutation_summary,
        "aggregate": aggregate_records,
        "per_split": split_summaries,
    }
    write_json(summary, args.output_root / "summary.json")

    print("\nDrug-residual line-macro Pearson (25 held-out lines):", flush=True)
    print(
        aggregate[aggregate["level"].eq("drug_residual")].to_string(index=False),
        flush=True,
    )
    print("\nPermutation nulls:", json.dumps(permutation_summary, indent=2), flush=True)
    print("\nGates:", json.dumps(gates, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_OUTPUT_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model-metadata",
        type=Path,
        default=Path("/nfs-stor/yuqi.lei/Therapeutic/derived/depmap_24q4/tahoe_observed_Model.csv"),
    )
    parser.add_argument("--data-revision", default=DEFAULT_DATA_REVISION)
    parser.add_argument(
        "--component-grid", type=int, nargs="+",
        default=[1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40],
    )
    parser.add_argument(
        "--lambda-grid", type=float, nargs="+",
        default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
    )
    parser.add_argument("--rhaister-lambda", type=float, default=1.0)
    parser.add_argument("--rhaister-blend", type=float, default=0.95)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
