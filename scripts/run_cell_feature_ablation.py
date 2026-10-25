#!/usr/bin/env python3
"""Cell-representation ablation on the strict Tahoe complete-line holdout.

The question is whether a natural-language description of a cell line carries
information about drug-specific response beyond the structured metadata fields
the description is written from.  Four cell representations are compared under
one fixed response head:

``expression_pca``
    DepMap 24Q4 protein-coding expression.  The anchor arm, so that this run is
    directly comparable with the existing residual-ridge result.
``structured_metadata``
    One-hot / numeric encoding of DepMap ``Model.csv`` context fields.
``template_text``
    Frozen language-model embedding of a sentence generated deterministically
    from *exactly* the ``structured_metadata`` fields.  Comparing this with
    ``structured_metadata`` isolates the representation change: no new
    measured fact is added, only the language model's prior.
``cellosaurus_text``
    Frozen embedding of real Cellosaurus comment text, with identity tokens
    masked and drug-response sentences removed.  This arm may carry curated
    facts absent from the structured fields.

Each arm additionally gets its own line-label permutation null, which is the
control for "the model is not actually using the line-to-feature pairing".

Every arm shares the response head, the outer splits, the component/penalty
grid, and the reference-only fitting rule, so differences are attributable to
the cell representation alone.  Text embeddings are precomputed by
``scripts/embed_cell_text.py`` and are frozen here.
"""

from __future__ import annotations

import argparse
import json
import os
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
from run_depmap_expression_ridge import (
    DEFAULT_COMPONENT_ROOT,
    DEFAULT_DEPMAP_ROOT,
    atomic_csv,
    build_candidates,
    line_bootstrap_interval,
    load_depmap_expression,
    loo_rmse_from_gram,
    map_drug_means,
    reference_only_pca,
    response_cube,
    sha256,
    target_weights,
    write_json,
)

DEFAULT_TEXT_ROOT = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/cellosaurus")
DEFAULT_OUTPUT_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/cell_feature_ablation"
)
LEVELS = ("full", "drug_residual", "train_pca1_removed")
ARMS = ("expression_pca", "structured_metadata", "template_text", "cellosaurus_text")

# Structured context fields.  Identifiers are excluded because they would let a
# model key on line identity, and treatment/response fields are excluded
# because they are outcome information under the declared input boundary.
STRUCTURED_CATEGORICAL = (
    "OncotreeLineage",
    "OncotreePrimaryDisease",
    "OncotreeSubtype",
    "DepmapModelType",
    "Sex",
    "AgeCategory",
    "PatientRace",
    "PrimaryOrMetastasis",
    "SampleCollectionSite",
    "TissueOrigin",
    "GrowthPattern",
    "OnboardedMedia",
    "SerumFreeMedia",
    "PlateCoating",
    "EngineeredModel",
)
STRUCTURED_NUMERIC = ("Age",)
EXCLUDED_LEAKAGE_FIELDS = (
    "ModelTreatment",
    "PatientTreatmentStatus",
    "PatientTreatmentType",
    "PatientTreatmentDetails",
    "PatientTreatmentResponse",
    "CulturedResistanceDrug",
    "PublicComments",
)


def build_structured_features(mapping: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One-hot / numeric encoding of context fields, indexed by RRID."""
    frame = mapping.dropna(subset=["RRID"]).copy()
    frame.index = frame["RRID"].astype(str)
    blocks: list[pd.DataFrame] = []
    used: dict[str, int] = {}
    for column in STRUCTURED_CATEGORICAL:
        if column not in frame.columns:
            continue
        values = frame[column].astype("string").fillna("__missing__")
        dummies = pd.get_dummies(values, prefix=column, dtype=float)
        if dummies.shape[1] <= 1:
            continue
        blocks.append(dummies)
        used[column] = int(dummies.shape[1])
    for column in STRUCTURED_NUMERIC:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() < 2:
            continue
        filled = values.fillna(values.mean())
        blocks.append(filled.to_frame(name=f"{column}__numeric").astype(float))
        blocks.append(
            values.isna().astype(float).to_frame(name=f"{column}__missing")
        )
        used[column] = 2
    features = pd.concat(blocks, axis=1)
    features.index = features.index.astype(str)
    info = {
        "n_features": int(features.shape[1]),
        "fields_used": used,
        "excluded_leakage_fields": list(EXCLUDED_LEAKAGE_FIELDS),
        "rule": (
            "Identifiers and treatment/response fields excluded; categoricals "
            "one-hot encoded with an explicit missing level; Age carried as a "
            "mean-filled numeric plus a missingness indicator."
        ),
    }
    return features, info


def load_text_embeddings(path: Path) -> tuple[pd.DataFrame, dict]:
    with np.load(path, allow_pickle=False) as archive:
        rrids = [str(value) for value in archive["rrid"]]
        embeddings = archive["embedding"].astype(np.float64)
        metadata = {
            "model_name": str(archive["model_name"][0]),
            "pooling": str(archive["pooling"][0]),
            "text_kind": str(archive["text_kind"][0]),
            "embedding_dim": int(embeddings.shape[1]),
            "n_lines": int(embeddings.shape[0]),
        }
    return pd.DataFrame(embeddings, index=rrids), metadata


def arm_features(
    arm: str,
    expression: pd.DataFrame,
    cvcl_to_model: dict[str, str],
    structured: pd.DataFrame,
    template: pd.DataFrame,
    cellosaurus: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Return a per-RRID feature frame and a short provenance string."""
    if arm == "expression_pca":
        rows = {
            rrid: expression.loc[model_id].to_numpy(dtype=np.float64)
            for rrid, model_id in cvcl_to_model.items()
            if model_id in expression.index
        }
        frame = pd.DataFrame.from_dict(rows, orient="index")
        return frame, "DepMap 24Q4 protein-coding expression (log2 TPM+1)"
    if arm == "structured_metadata":
        return structured, "DepMap Model.csv context fields, one-hot/numeric"
    if arm == "template_text":
        return template, "frozen LM embedding of template text from those fields"
    if arm == "cellosaurus_text":
        return cellosaurus, "frozen LM embedding of redacted Cellosaurus comments"
    raise ValueError(f"unknown arm {arm}")


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
        for metric, values in rowwise_metrics(level_truth, level_prediction).items():
            frame[f"{model}__{level}__{metric}"] = values


def fit_arm(
    arm: str,
    features: pd.DataFrame,
    reference_cells: list[str],
    query_cells: list[str],
    response_flat: np.ndarray,
    response_gram: np.ndarray,
    n_genes: int,
    n_common: int,
    component_grid: tuple[int, ...],
    lambda_grid: tuple[float, ...],
    n_permutations: int,
    seed: int,
    on_permutation,
) -> dict:
    reference_x = features.loc[reference_cells].to_numpy(dtype=np.float64)
    query_x = features.loc[query_cells].to_numpy(dtype=np.float64)
    scores, query_scores, pca_info = reference_only_pca(reference_x, query_x)
    candidates = build_candidates(
        scores, response_gram, response_flat.shape[1], component_grid, lambda_grid
    )
    selected_index = int(np.argmin([candidate.loo_rmse for candidate in candidates]))
    selected = candidates[selected_index]
    weights_by_candidate = [target_weights(c, query_scores) for c in candidates]
    predicted = (weights_by_candidate[selected_index] @ response_flat).astype(
        np.float32
    ).reshape(len(query_cells), n_common, n_genes)

    # Each null cube is consumed immediately: holding all permutations would
    # need tens of gigabytes at this tensor size.
    rng = np.random.default_rng(seed)
    for permutation_id in range(n_permutations):
        permutation = rng.permutation(len(reference_cells))
        permuted_gram = response_gram[np.ix_(permutation, permutation)]
        scores_under_null = [
            loo_rmse_from_gram(c.loo_operator, permuted_gram, response_flat.shape[1])
            for c in candidates
        ]
        null_index = int(np.argmin(scores_under_null))
        null_prediction = (
            weights_by_candidate[null_index] @ response_flat[permutation]
        ).astype(np.float32).reshape(len(query_cells), n_common, n_genes)
        on_permutation(permutation_id, null_prediction)
        del null_prediction

    return {
        "arm": arm,
        "prediction": predicted,
        "pca_info": pca_info,
        "selected_components": selected.n_components,
        "selected_lambda": selected.ridge_lambda,
        "selected_loo_rmse": selected.loo_rmse,
        "n_input_features": int(reference_x.shape[1]),
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
    feature_sets: dict[str, pd.DataFrame],
    metadata: dict[str, dict[str, str]],
) -> dict:
    repo_root = args.project_root / "external" / "Rhaister"
    holdout_cells, _, split_path = load_split(repo_root, split)
    holdout = set(holdout_cells)
    reference_all = sorted(set(compact.cells.tolist()) - holdout)

    # Primary cohort: lines for which EVERY arm has features, so that all
    # comparisons are paired on identical lines.
    def covered(cell: str) -> bool:
        return all(cell in frame.index for frame in feature_sets.values())

    reference_cells = [cell for cell in reference_all if covered(cell)]
    query_cells = [cell for cell in holdout_cells if covered(cell)]
    if not query_cells:
        raise ValueError(f"split {split}: no held-out line covered by all arms")

    baseline_path = args.baseline_root / f"split_{split}" / "predictions.npz"
    with np.load(baseline_path, allow_pickle=False) as archive:
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
    reference_mean = np.stack([drug_mean_map[t] for t in common_treatments])
    response_residual = response - reference_mean[None, :, :]
    response_flat = response_residual.reshape(len(reference_cells), -1)
    response_gram = (response_flat @ response_flat.T).astype(np.float64)

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
        "in_primary_cohort": [cell in set(query_cells) for cell in cells],
    })
    add_model_metrics(frame, "perturbation_mean", truth, drug_mean, drug_mean, pc_mean, pc1)
    add_model_metrics(frame, "rhaister_o", truth, drug_mean, rhaister, pc_mean, pc1)

    query_index = {cell: index for index, cell in enumerate(query_cells)}
    common_index = {t: index for index, t in enumerate(common_treatments)}
    row_lookup = {cell: np.flatnonzero(cells == cell) for cell in query_cells}

    def scatter(cube: np.ndarray) -> np.ndarray:
        out = np.zeros_like(truth, dtype=np.float32)
        for row, (cell, treatment) in enumerate(zip(cells, treatments)):
            if cell in query_index and treatment in common_index:
                out[row] = cube[query_index[cell], common_index[treatment]]
        return out

    arm_results: dict[str, dict] = {}
    permutation_records: list[dict] = []
    for arm_position, arm in enumerate(ARMS):
        def record_permutation(permutation_id: int, null_cube: np.ndarray, arm=arm) -> None:
            null_full = scatter(null_cube)
            for cell in query_cells:
                rows = row_lookup[cell]
                truth_residual = truth[rows] - drug_mean[rows]
                null_residual = null_full[rows]
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

        result = fit_arm(
            arm,
            feature_sets[arm],
            reference_cells,
            query_cells,
            response_flat,
            response_gram,
            len(compact.genes),
            len(common_treatments),
            tuple(args.component_grid),
            tuple(args.lambda_grid),
            args.n_permutations,
            args.seed + split * 100_000 + arm_position * 1_000,
            record_permutation,
        )
        prediction = drug_mean + scatter(result["prediction"])
        add_model_metrics(frame, arm, truth, drug_mean, prediction, pc_mean, pc1)
        result.pop("prediction")
        arm_results[arm] = result

    line_records = []
    bootstrap_records = []
    models = ("perturbation_mean", "rhaister_o") + ARMS
    for cell in query_cells:
        subset = frame[frame["cell_line"].eq(cell)]
        for level in LEVELS:
            for model in models:
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
            # Paired drug-cluster bootstrap for each text arm against the
            # structured arm it is supposed to improve on.
            for arm, reference_arm in (
                ("template_text", "structured_metadata"),
                ("cellosaurus_text", "structured_metadata"),
                ("cellosaurus_text", "template_text"),
                ("structured_metadata", "expression_pca"),
            ):
                result = clustered_bootstrap(
                    subset[f"{arm}__{level}__pearson"].to_numpy(),
                    subset[f"{reference_arm}__{level}__pearson"].to_numpy(),
                    subset["drug"].to_numpy(),
                    args.n_bootstrap,
                    args.seed + split * 977 + query_cells.index(cell) * 31 + LEVELS.index(level),
                )
                bootstrap_records.append({
                    "split": split,
                    "cell_line": cell,
                    "level": level,
                    "arm": arm,
                    "versus": reference_arm,
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
        "n_reference_lines_all_arms": len(reference_cells),
        "n_reference_lines_total": len(reference_all),
        "reference_lines_dropped": sorted(set(reference_all) - set(reference_cells)),
        "heldout_lines": holdout_cells,
        "heldout_lines_all_arms": query_cells,
        "heldout_lines_dropped": sorted(holdout - set(query_cells)),
        "n_common_treatments": len(common_treatments),
        "arms": {
            arm: {
                key: value
                for key, value in arm_results[arm].items()
                if key != "candidates"
            }
            for arm in ARMS
        },
        "per_line": line_records,
    }
    write_json(summary, output_dir / "summary.json")
    selection = ", ".join(
        f"{arm}: PC={arm_results[arm]['selected_components']}"
        f"/lam={arm_results[arm]['selected_lambda']:g}"
        for arm in ARMS
    )
    print(
        f"split {split}: reference={len(reference_cells)}/{len(reference_all)}, "
        f"query={len(query_cells)}/{len(holdout_cells)}, "
        f"treatments={len(common_treatments)} | {selection}",
        flush=True,
    )
    return summary


def main(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    repo_root = project_root / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    expression, cvcl_to_model, metadata, expression_path, mapping_path = (
        load_depmap_expression(args.depmap_root)
    )
    mapping = pd.read_csv(mapping_path)
    structured, structured_info = build_structured_features(mapping)
    template, template_info = load_text_embeddings(args.template_embeddings)
    cellosaurus, cellosaurus_info = load_text_embeddings(args.cellosaurus_embeddings)

    feature_sets: dict[str, pd.DataFrame] = {}
    provenance: dict[str, str] = {}
    for arm in ARMS:
        frame, description = arm_features(
            arm, expression, cvcl_to_model, structured, template, cellosaurus
        )
        feature_sets[arm] = frame
        provenance[arm] = description
        print(f"{arm:22s} {frame.shape[0]:3d} lines x {frame.shape[1]:6d} features", flush=True)

    split_summaries = [
        split_analysis(args, split, compact, feature_sets, metadata)
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
    permutation_summary = {}
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
        for arm, reference_arm in (
            ("template_text", "structured_metadata"),
            ("cellosaurus_text", "structured_metadata"),
            ("cellosaurus_text", "template_text"),
            ("structured_metadata", "expression_pca"),
            ("template_text", "expression_pca"),
            ("cellosaurus_text", "expression_pca"),
        ):
            difference = (table[arm] - table[reference_arm]).to_numpy()
            mean, low, high = line_bootstrap_interval(
                difference, args.seed + LEVELS.index(level) * 13 + len(arm + reference_arm)
            )
            aggregate_records.append({
                "level": level,
                "comparison": f"{arm}_minus_{reference_arm}",
                "n_lines": len(table),
                "line_macro_mean": mean,
                "line_bootstrap_low": low,
                "line_bootstrap_high": high,
            })

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
                    "empirical_p_value": float((1 + np.sum(null >= actual)) / (1 + len(null))),
                }

    aggregate = pd.DataFrame(aggregate_records)
    atomic_csv(aggregate, args.output_root / "aggregate_summary.csv")

    residual = aggregate[aggregate["level"].eq("drug_residual")].set_index("comparison")
    text_beats_structured = bool(
        residual.loc["template_text_minus_structured_metadata", "line_bootstrap_low"] > 0
        or residual.loc["cellosaurus_text_minus_structured_metadata", "line_bootstrap_low"] > 0
    )
    text_beats_permutation = bool(
        permutation_summary["drug_residual"]["template_text"]["empirical_p_value"] <= 0.05
        or permutation_summary["drug_residual"]["cellosaurus_text"]["empirical_p_value"] <= 0.05
    )

    summary = {
        "purpose": (
            "Does natural-language description of a cell line add drug-response "
            "information beyond the structured metadata it is written from?"
        ),
        "arms": provenance,
        "structured_features": structured_info,
        "template_embeddings": template_info,
        "cellosaurus_embeddings": cellosaurus_info,
        "data_revision": args.data_revision,
        "depmap_expression": str(expression_path),
        "depmap_expression_sha256": sha256(expression_path),
        "component_grid": args.component_grid,
        "lambda_grid": args.lambda_grid,
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "cohort_rule": (
            "Primary cohort restricted to lines covered by every arm, so all "
            "comparisons are paired on identical held-out lines."
        ),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "decision": {
            "text_beats_structured_metadata": text_beats_structured,
            "text_beats_permutation_null": text_beats_permutation,
            "text_branch_justified": bool(text_beats_structured and text_beats_permutation),
            "rule": (
                "Justify a text branch only if a text arm's paired line-bootstrap "
                "improvement over structured metadata has a lower bound above zero "
                "AND the same arm exceeds its own line-label permutation null."
            ),
        },
        "permutation": permutation_summary,
        "aggregate": aggregate_records,
        "per_split": split_summaries,
    }
    write_json(summary, args.output_root / "summary.json")

    print("\nDrug-residual line-macro Pearson:", flush=True)
    print(
        aggregate[aggregate["level"].eq("drug_residual")].to_string(index=False),
        flush=True,
    )
    print("\nPermutation nulls:", json.dumps(permutation_summary, indent=2), flush=True)
    print("\nDecision:", json.dumps(summary["decision"], indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_OUTPUT_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--depmap-root", type=Path, default=DEFAULT_DEPMAP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--template-embeddings",
        type=Path,
        default=DEFAULT_TEXT_ROOT / "embeddings_template.npz",
    )
    parser.add_argument(
        "--cellosaurus-embeddings",
        type=Path,
        default=DEFAULT_TEXT_ROOT / "embeddings_cellosaurus.npz",
    )
    parser.add_argument("--data-revision", default=DEFAULT_DATA_REVISION)
    parser.add_argument("--component-grid", type=int, nargs="+", default=[2, 5, 10, 15, 20, 25, 30, 35, 40])
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
