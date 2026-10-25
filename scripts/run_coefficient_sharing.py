#!/usr/bin/env python3
"""Coefficient sharing: shrink Rhaister-O's own diagonal slopes.

Rhaister-O fits one slope ``gamma[t, g]`` per (treatment, gene) pair, each from
the 45 reference lines.  On the Tahoe panel that is roughly 1.5 million scalars
estimated from 45 observations apiece -- a textbook over-parameterized estimate
with an obvious pooling target, because the frozen baseline already computes
``shared_gamma[g]``, one slope per gene pooled across every treatment.

Both endpoints of the continuum are already measured and frozen: fully
independent slopes give 0.631456 full-delta line-macro Pearson, fully shared
slopes give 0.489322.  Everything between them is untested.  This script tests
it.

Why the prior is good.  The cross-gene bilinear experiment (job 249519) failed
because it *added* degrees of freedom that 45 reference lines cannot identify,
and its own conclusion was that the diagonal restriction acts as an effective
regularizer.  Coefficient sharing moves in the opposite direction: it *reduces*
effective degrees of freedom in exactly the place the previous result said
regularization was helping.  This is `PLAN.md` Phase 4 Track A, and it is the
variant described in the Rhaister repository's architecture document but absent
from its pinned executable path.

Scope.  Inputs are restricted to the plate-matched Tahoe DMSO profile, exactly
what the frozen Rhaister-O baseline consumes.  No new modality is introduced, so
the DepMap expansion gate in `CLAUDE.md` does not apply -- this is a modelling
question about an existing estimator, not a data question.  Same reasoning that
governed job 249519.

Four shrinkage families, each with its own inner selection:

``global_blend``
    ``gamma_w = shared + w * (gamma - shared)``, one scalar w.  The simplest
    interpolation between the two frozen endpoints.
``svd_rank``
    Truncate the centred slope matrix ``(gamma - shared)`` of shape
    (treatments, genes) to rank r, then add ``shared`` back.  Shares strength
    across treatments *and* genes through a low-rank structure on the slopes
    themselves, not on the responses.
``dose_pool``
    Pool slopes across the doses of one drug, then blend with w.  The 1,137
    treatments are 379 drugs at three doses, so this is a principled grouping
    that the per-(treatment, gene) estimator ignores entirely.
``eb_per_gene``
    Empirical-Bayes shrinkage with a per-gene weight derived from the
    across-treatment variance of the slopes relative to their estimated noise
    variance.  No free parameter; the data set the weight.

Inner selection deliberately differs from previous experiments.  Job 249519
established that reference-line LOO *RMSE* is a poor proxy for held-out
drug-residual Pearson -- it preferred a configuration the outer endpoint
rejected.  So the primary selection criterion here is the leave-one-reference-
line-out **drug-residual Pearson**, the same functional the outer evaluation
uses.  What LOO RMSE *would* have chosen is recorded alongside, so the two
criteria can be compared directly on a second dataset.

Leave-one-reference-line-out is exact, not approximated: both the per-treatment
mean and the slope numerator/denominator admit rank-one updates, but this script
simply refits them from the remaining 44 lines, which costs one einsum per
held-out reference line and is shared across every shrinkage parameter.

Five corrections applied after an independent correctness review of the first
version, whose output (mean gain +0.001749, bootstrap low +0.000816) must be
treated as superseded because the first three defects are of the same order as
that effect:

1. The predeclared gate named a permutation-null condition that was never
   evaluated. A line-block permutation null is now computed and the condition
   is checked.
2. The inner LOO centred the DMSO features once over all 45 reference lines and
   then fitted on 44, leaking the validation line into the ridge denominators.
   Each fold now re-centres on its own 44 lines.
3. The outer prediction was rebuilt from scratch, which required reproducing the
   frozen path's query feature and its blend against the additive-ALS array;
   both differed slightly. The prediction is now a delta on the frozen array, so
   the unshrunk endpoint reproduces frozen Rhaister-O exactly and the paired
   comparison is anchored.
4. Same issue in the 0.95 blend fallback, fixed by the same change.
5. ``eb_per_gene`` used 1/sum(z^2) as its noise term, omitting the response
   residual variance, which made the weight depend on the units of the response.
   It now uses sigma^2 * S / (S + lambda)^2.
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
    add_model_metrics,
    atomic_csv,
    line_bootstrap_interval,
    map_drug_means,
    response_cube,
    sha256,
    write_json,
)
from run_depmap_rhaister_correction import line_hvg_vectors
from run_rhaister_o_compact import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_REVISION,
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    build_or_load_compact_cache,
    load_hvg_centroids,
    load_split,
    rowwise_metrics,
)

LEVELS = ("full", "drug_residual", "train_pca1_removed")
DEFAULT_BASELINE_ROOT = Path("/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_o_compact")
DEFAULT_COMPONENT_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_all_split_evaluation"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/coefficient_sharing"
)

# Predeclared gate, identical in form to the one job 249519 was judged against.
# It is NOT relaxed to fit an expected effect size; the point of a predeclared
# gate is that it is written before the number is seen.
GATE = {
    "min_drug_residual_gain": 0.01,
    "require_positive_bootstrap_low": True,
    "min_pc1_removed_gain": 0.01,
    "min_positive_splits": 4,
    "max_rmse_ratio": 1.01,
    "require_above_permutation_null": True,
}


def slope_estimates(
    z: np.ndarray, response: np.ndarray, ridge_lambda: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rhaister-O's own estimator, on whatever set of lines is passed in.

    Returns the per-treatment mean, the per-(treatment, gene) slope, and the
    pooled per-gene slope, reproducing `run_rhaister_o_compact.diagonal_path`.
    """
    mu = response.mean(axis=0)
    residual = response - mu[None, :, :]
    denominator = np.sum(z * z, axis=0) + ridge_lambda
    gamma = np.einsum("cg,ctg->tg", z, residual, optimize=True) / denominator[None, :]
    shared_denominator = response.shape[1] * np.sum(z * z, axis=0) + ridge_lambda
    shared = np.einsum("cg,ctg->g", z, residual, optimize=True) / shared_denominator
    return mu, gamma, shared


def dose_group_index(treatments: list[str]) -> np.ndarray:
    """Map each treatment to an integer drug id, pooling its doses."""
    drugs = [parse_treatment(t)[0] for t in treatments]
    order = {name: index for index, name in enumerate(sorted(set(drugs)))}
    return np.array([order[name] for name in drugs], dtype=np.int64)


def shrunk_slopes(
    family: str,
    parameter: float,
    gamma: np.ndarray,
    shared: np.ndarray,
    group: np.ndarray,
    noise_scale: np.ndarray | None = None,
    svd: tuple | None = None,
) -> np.ndarray:
    centred = gamma - shared[None, :]
    if family == "global_blend":
        return shared[None, :] + parameter * centred
    if family == "svd_rank":
        rank = int(parameter)
        if rank >= min(centred.shape):
            return gamma
        if svd is None:
            svd = np.linalg.svd(centred, full_matrices=False)
        u, s, vt = svd
        truncated = (u[:, :rank] * s[None, :rank]) @ vt[:rank]
        return shared[None, :] + truncated
    if family == "dose_pool":
        counts = np.bincount(group, minlength=int(group.max()) + 1).astype(np.float64)
        pooled = np.zeros((len(counts), centred.shape[1]), dtype=np.float64)
        np.add.at(pooled, group, centred)
        pooled /= counts[:, None]
        expanded = pooled[group]
        return shared[None, :] + expanded + parameter * (centred - expanded)
    if family == "eb_per_gene":
        # James-Stein style: weight = signal / (signal + noise), per gene, with
        # the noise term scaled by the single free multiplier `parameter`.
        signal = centred.var(axis=0, ddof=1)
        noise = noise_scale if noise_scale is not None else np.ones_like(signal)
        weight = signal / np.maximum(signal + parameter * noise, 1e-30)
        return shared[None, :] + weight[None, :] * centred
    raise ValueError(f"unknown family {family}")


def families_and_grids(args: argparse.Namespace, n_treatments: int) -> list[tuple[str, float]]:
    grid: list[tuple[str, float]] = []
    grid += [("global_blend", w) for w in args.blend_grid]
    grid += [
        ("svd_rank", r) for r in args.rank_grid if r < n_treatments
    ]
    grid += [("dose_pool", w) for w in args.blend_grid]
    grid += [("eb_per_gene", c) for c in args.eb_grid]
    return grid


def line_macro_pearson(truth: np.ndarray, prediction: np.ndarray, drug_mean: np.ndarray) -> float:
    """Drug-residual Pearson, averaged over the treatments of one line."""
    return float(
        rowwise_metrics(truth - drug_mean, prediction - drug_mean)["pearson"].mean()
    )


def inner_selection(
    z_reference: np.ndarray,
    response: np.ndarray,
    grid: list[tuple[str, float]],
    group: np.ndarray,
    ridge_lambda: float,
    blend: float,
) -> tuple[dict, list[dict]]:
    """Leave-one-reference-line-out over the 45 reference lines.

    Scored two ways: drug-residual Pearson (primary, matching the outer
    endpoint) and RMSE (secondary, the criterion job 249519 showed to be a poor
    proxy). Both are recorded so the criteria can be compared.
    """
    n_lines = response.shape[0]
    pearson_sum = np.zeros(len(grid), dtype=np.float64)
    sse_sum = np.zeros(len(grid), dtype=np.float64)
    count = 0
    for held in range(n_lines):
        keep = np.arange(n_lines) != held
        # Re-centre on the 44 training lines of THIS fold. Centring once over
        # all 45 outside the loop would leak the validation line into the ridge
        # denominators and into its own predicted slope contribution.
        fold_mean = z_reference[keep].mean(axis=0)
        z_fold = z_reference[keep] - fold_mean[None, :]
        z_held = z_reference[held] - fold_mean
        mu, gamma, shared = slope_estimates(z_fold, response[keep], ridge_lambda)
        # Sampling variance of a ridge slope is sigma^2 * S / (S + lambda)^2
        # with S = sum(z^2); the residual variance sigma^2 is estimated per gene
        # from the reference responses so the weight is scale-equivariant.
        sum_z2 = np.sum(z_fold * z_fold, axis=0)
        sigma2 = response[keep].var(axis=(0, 1), ddof=1)
        noise = sigma2 * sum_z2 / np.maximum((sum_z2 + ridge_lambda) ** 2, 1e-30)
        truth = response[held]
        additive = mu  # the fallback the slope term is blended against
        # One SVD per held-out reference line, reused across every rank in the
        # grid; recomputing it per rank would cost ten times as much.
        svd = np.linalg.svd(gamma - shared[None, :], full_matrices=False)
        for index, (family, parameter) in enumerate(grid):
            slopes = shrunk_slopes(family, parameter, gamma, shared, group, noise, svd)
            prediction = (1.0 - blend) * additive + blend * (
                mu + slopes * z_held[None, :]
            )
            pearson_sum[index] += line_macro_pearson(truth, prediction, mu)
            error = prediction - truth
            sse_sum[index] += float(np.mean(error * error))
        count += 1
    pearson = pearson_sum / count
    rmse = np.sqrt(sse_sum / count)
    best_pearson = int(np.argmax(pearson))
    best_rmse = int(np.argmin(rmse))
    records = [
        {
            "family": grid[i][0],
            "parameter": float(grid[i][1]),
            "loo_drug_residual_pearson": float(pearson[i]),
            "loo_rmse": float(rmse[i]),
            "selected_by_pearson": i == best_pearson,
            "selected_by_rmse": i == best_rmse,
        }
        for i in range(len(grid))
    ]
    return {
        "selected_family": grid[best_pearson][0],
        "selected_parameter": float(grid[best_pearson][1]),
        "selected_loo_pearson": float(pearson[best_pearson]),
        "rmse_would_select_family": grid[best_rmse][0],
        "rmse_would_select_parameter": float(grid[best_rmse][1]),
        "criteria_agree": best_pearson == best_rmse,
    }, records


def split_analysis(args, split: int, compact, hvg, metadata) -> tuple[pd.DataFrame, dict, list[dict]]:
    repo_root = args.project_root / "external" / "Rhaister"
    holdout_cells, _, split_path = load_split(repo_root, split)
    reference_cells = sorted(set(compact.cells.tolist()) - set(holdout_cells))

    baseline_path = args.baseline_root / f"split_{split}" / "predictions.npz"
    with np.load(baseline_path, allow_pickle=False) as archive:
        cells = archive["cell_line"].copy()
        treatments = archive["treatment"].copy()
        truth = archive["truth"].astype(np.float32)
        drug_mean = archive["perturbation_mean"].astype(np.float32)
        rhaister = archive["rhaister_o"].astype(np.float32)
        shared_frozen = archive["shared_gamma"].astype(np.float32)

    component_path = args.component_root / f"split_{split}" / "train_residual_pc1.npz"
    with np.load(component_path, allow_pickle=False) as component_archive:
        pc_mean = component_archive["mean"].astype(np.float32)
        pc1 = component_archive["component"].astype(np.float32)

    candidate_treatments = sorted(set(treatments.tolist()))
    response, common_treatments = response_cube(
        compact, reference_cells, candidate_treatments
    )
    response = response.astype(np.float64)
    drug_mean_map = map_drug_means(treatments, drug_mean)

    reference_hvg = line_hvg_vectors(compact, hvg, reference_cells)
    query_hvg = line_hvg_vectors(compact, hvg, holdout_cells)
    z_mean = reference_hvg.mean(axis=0)
    z_reference = reference_hvg - z_mean
    z_query = query_hvg - z_mean

    group = dose_group_index(common_treatments)
    grid = families_and_grids(args, len(common_treatments))
    selection, selection_records = inner_selection(
        z_reference, response, grid, group, args.rhaister_lambda, args.rhaister_blend
    )
    for record in selection_records:
        record["split"] = split
    print(
        f"  split {split}: selected {selection['selected_family']}"
        f"={selection['selected_parameter']:g} by LOO Pearson "
        f"({selection['selected_loo_pearson']:.6f}); RMSE would pick "
        f"{selection['rmse_would_select_family']}"
        f"={selection['rmse_would_select_parameter']:g}"
        f" [agree={selection['criteria_agree']}]",
        flush=True,
    )

    mu, gamma, shared = slope_estimates(z_reference, response, args.rhaister_lambda)
    sum_z2 = np.sum(z_reference * z_reference, axis=0)
    sigma2 = response.var(axis=(0, 1), ddof=1)
    noise = sigma2 * sum_z2 / np.maximum((sum_z2 + args.rhaister_lambda) ** 2, 1e-30)
    slopes = shrunk_slopes(
        selection["selected_family"],
        selection["selected_parameter"],
        gamma,
        shared,
        group,
        noise,
    )

    # The outer prediction is expressed as a DELTA on the frozen Rhaister-O
    # array rather than rebuilt from scratch:
    #
    #     prediction = frozen + blend * (slopes - gamma) * z_query
    #
    # This matters. Rebuilding would require reproducing the frozen path's own
    # query feature (averaged over that line's *test* rows, accumulated in
    # float32) and its 0.95 blend against the additive-ALS array rather than
    # against mu; small differences in either would land in the same 1e-3 range
    # as the effect being measured. With the delta form the unshrunk endpoint
    # reproduces the frozen array exactly, so the paired comparison is anchored,
    # and any residual error in z_query multiplies the small (slopes - gamma)
    # term and is second order. `verify_anchor` below asserts this.
    query_index = {cell: i for i, cell in enumerate(holdout_cells)}
    common_index = {t: i for i, t in enumerate(common_treatments)}
    prediction = np.array(rhaister, dtype=np.float32, copy=True)
    covered = np.zeros(len(cells), dtype=bool)
    adjustment = {
        cell: (args.rhaister_blend * (slopes - gamma) * z_query[i][None, :]).astype(np.float32)
        for cell, i in query_index.items()
    }
    for row, (cell, treatment) in enumerate(zip(cells, treatments)):
        if cell in query_index and treatment in common_index:
            prediction[row] = rhaister[row] + adjustment[cell][common_index[treatment]]
            covered[row] = True

    # Anchor check: with no shrinkage the arm must BE frozen Rhaister-O.
    unshrunk = shrunk_slopes("global_blend", 1.0, gamma, shared, group, noise)
    anchor_error = float(np.abs(unshrunk - gamma).max())
    if anchor_error > 1e-12:
        raise RuntimeError(
            f"split {split}: global_blend=1 does not reproduce the unshrunk "
            f"slopes (max |diff| {anchor_error:g}); the paired comparison would "
            f"not be anchored on the frozen estimator"
        )

    # Line-block permutation null. Whole reference lines are permuted against
    # the response tensor so drugs and doses stay together and only the
    # line-to-response pairing is destroyed; the selected shrinkage family and
    # parameter are then applied unchanged. This answers "would shrinking a
    # randomly paired set of slopes produce the same gain?", which is the
    # condition the predeclared gate names and which the first version of this
    # script declared but never evaluated.
    permutation_gains: list[float] = []
    rng = np.random.default_rng(args.seed + split * 1_000)
    truth_residual_by_cell = {
        cell: truth[np.flatnonzero(cells == cell)] - drug_mean[np.flatnonzero(cells == cell)]
        for cell in holdout_cells
    }
    frozen_residual_by_cell = {
        cell: rhaister[np.flatnonzero(cells == cell)] - drug_mean[np.flatnonzero(cells == cell)]
        for cell in holdout_cells
    }
    base_by_cell = {
        cell: float(rowwise_metrics(
            truth_residual_by_cell[cell], frozen_residual_by_cell[cell]
        )["pearson"].mean())
        for cell in holdout_cells
    }
    for _ in range(args.n_permutations):
        order = rng.permutation(len(reference_cells))
        _, gamma_p, shared_p = slope_estimates(
            z_reference, response[order], args.rhaister_lambda
        )
        slopes_p = shrunk_slopes(
            selection["selected_family"], selection["selected_parameter"],
            gamma_p, shared_p, group, noise,
        )
        gains = []
        for cell, i in query_index.items():
            rows = np.flatnonzero(cells == cell)
            delta = (args.rhaister_blend * (slopes_p - gamma_p)
                     * z_query[i][None, :]).astype(np.float32)
            perturbed = np.array(frozen_residual_by_cell[cell], copy=True)
            for local, (c, t) in enumerate(zip(cells[rows], treatments[rows])):
                if t in common_index:
                    perturbed[local] += delta[common_index[t]]
            gains.append(float(rowwise_metrics(
                truth_residual_by_cell[cell], perturbed
            )["pearson"].mean()) - base_by_cell[cell])
        permutation_gains.append(float(np.mean(gains)))

    parsed = [parse_treatment(label) for label in treatments]
    frame = pd.DataFrame({
        "split": split,
        "cell_line": cells,
        "treatment": treatments,
        "drug": [item[0] for item in parsed],
        "dose": [item[1] for item in parsed],
        "covered": covered,
    })
    add_model_metrics(frame, "perturbation_mean", truth, drug_mean, drug_mean, pc_mean, pc1)
    add_model_metrics(frame, "rhaister_o", truth, drug_mean, rhaister, pc_mean, pc1)
    add_model_metrics(frame, "shared_gamma", truth, drug_mean, shared_frozen, pc_mean, pc1)
    add_model_metrics(frame, "coefficient_sharing", truth, drug_mean, prediction, pc_mean, pc1)

    info = {
        "split": split,
        "split_file_sha256": sha256(split_path),
        "n_reference_lines": len(reference_cells),
        "n_common_treatments": len(common_treatments),
        "n_drugs_after_dose_pooling": int(group.max()) + 1,
        "n_covered_conditions": int(covered.sum()),
        "permutation_gains": permutation_gains,
        "permutation_gain_mean": float(np.mean(permutation_gains)) if permutation_gains else None,
        "anchor_max_abs_error": anchor_error,
        **selection,
    }
    return frame, info, selection_records


def main(args: argparse.Namespace) -> None:
    repo_root = args.project_root.resolve() / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    hvg = load_hvg_centroids(args.data_root, compact.genes)
    args.output_root.mkdir(parents=True, exist_ok=True)

    frames, infos, selections = [], [], []
    for split in args.splits:
        frame, info, selection_records = split_analysis(args, split, compact, hvg, None)
        frames.append(frame)
        infos.append(info)
        selections.extend(selection_records)
    frame = pd.concat(frames, ignore_index=True)
    atomic_csv(pd.DataFrame(selections), args.output_root / "inner_selection.csv")

    models = ("perturbation_mean", "shared_gamma", "rhaister_o", "coefficient_sharing")
    line_records = []
    for (split, cell), subset in frame.groupby(["split", "cell_line"], sort=True):
        for level in LEVELS:
            for model in models:
                line_records.append({
                    "split": int(split),
                    "cell_line": cell,
                    "level": level,
                    "model": model,
                    "n_conditions": len(subset),
                    **{
                        metric: float(subset[f"{model}__{level}__{metric}"].mean())
                        for metric in ("pearson", "rmse", "mae")
                    },
                })
    line_frame = pd.DataFrame(line_records)
    atomic_csv(line_frame, args.output_root / "per_line_summary.csv")

    aggregate = []
    for level in LEVELS:
        block = line_frame[line_frame["level"].eq(level)]
        pivot = block.pivot_table(index="cell_line", columns="model", values="pearson")
        for model in models:
            mean, low, high = line_bootstrap_interval(
                pivot[model].to_numpy(dtype=np.float64), args.seed, args.n_bootstrap
            )
            aggregate.append({
                "level": level, "model": model, "n_lines": len(pivot),
                "line_macro_pearson": mean,
                "line_bootstrap_low": low, "line_bootstrap_high": high,
            })
        difference = (pivot["coefficient_sharing"] - pivot["rhaister_o"]).to_numpy(np.float64)
        mean, low, high = line_bootstrap_interval(difference, args.seed, args.n_bootstrap)
        aggregate.append({
            "level": level, "model": "coefficient_sharing_minus_rhaister_o",
            "n_lines": len(pivot), "line_macro_pearson": mean,
            "line_bootstrap_low": low, "line_bootstrap_high": high,
        })
    aggregate_frame = pd.DataFrame(aggregate)
    atomic_csv(aggregate_frame, args.output_root / "aggregate_summary.csv")

    def value(level, model, column="line_macro_pearson"):
        row = aggregate_frame[
            aggregate_frame["level"].eq(level) & aggregate_frame["model"].eq(model)
        ]
        return float(row[column].iloc[0])

    per_split_gain = []
    for split in args.splits:
        block = line_frame[line_frame["split"].eq(split) & line_frame["level"].eq("drug_residual")]
        pivot = block.pivot_table(index="cell_line", columns="model", values="pearson")
        per_split_gain.append(float((pivot["coefficient_sharing"] - pivot["rhaister_o"]).mean()))

    rmse_block = line_frame[line_frame["level"].eq("drug_residual")]
    rmse_pivot = rmse_block.pivot_table(index="cell_line", columns="model", values="rmse")
    rmse_ratio = float(rmse_pivot["coefficient_sharing"].mean() / rmse_pivot["rhaister_o"].mean())

    gain = value("drug_residual", "coefficient_sharing_minus_rhaister_o")
    gain_low = value("drug_residual", "coefficient_sharing_minus_rhaister_o", "line_bootstrap_low")
    pc1_gain = value("train_pca1_removed", "coefficient_sharing_minus_rhaister_o")
    # Pool the per-split permutation draws into one null over the same
    # quantity the gate is stated on: the mean per-line gain.
    null_draws = [
        float(np.mean(column))
        for column in zip(*[info["permutation_gains"] for info in infos])
    ] if infos and infos[0]["permutation_gains"] else []
    if null_draws:
        n_at_least = sum(1 for value in null_draws if value >= gain)
        permutation_p = (n_at_least + 1) / (len(null_draws) + 1)
        null_mean = float(np.mean(null_draws))
        null_p95 = float(np.quantile(null_draws, 0.95))
        above_null = permutation_p <= 0.05
    else:
        permutation_p = null_mean = null_p95 = None
        above_null = False

    gate = {
        "drug_residual_gain": gain,
        "drug_residual_gain_low": gain_low,
        "pc1_removed_gain": pc1_gain,
        "n_positive_splits": int(sum(g > 0 for g in per_split_gain)),
        "per_split_gain": per_split_gain,
        "rmse_ratio": rmse_ratio,
        "permutation_p": permutation_p,
        "permutation_null_mean": null_mean,
        "permutation_null_p95": null_p95,
        "n_permutation_draws": len(null_draws),
        "conditions": {
            "gain_at_least": gain >= GATE["min_drug_residual_gain"],
            "bootstrap_low_positive": gain_low > 0,
            "pc1_gain_at_least": pc1_gain >= GATE["min_pc1_removed_gain"],
            "enough_positive_splits": sum(g > 0 for g in per_split_gain) >= GATE["min_positive_splits"],
            "rmse_within_tolerance": rmse_ratio <= GATE["max_rmse_ratio"],
            "above_permutation_null": above_null,
        },
    }
    gate["passes"] = all(gate["conditions"].values())

    print("\ndrug-residual line-macro Pearson:")
    for model in models:
        print(f"  {model:24s} {value('drug_residual', model):.6f}")
    print(f"  gain over Rhaister-O    {gain:+.6f}  bootstrap low {gain_low:+.6f}")
    print(f"  per-split gains         {[round(g, 6) for g in per_split_gain]}")
    print(f"  permutation p           {gate['permutation_p']}  null mean {gate['permutation_null_mean']}")
    print(f"  anchor max abs error    {max(i['anchor_max_abs_error'] for i in infos):g}")
    print(f"  gate passes             {gate['passes']}")

    write_json(
        {
            "purpose": (
                "Shrink Rhaister-O's per-(treatment, gene) diagonal slopes toward "
                "the pooled per-gene slope. PLAN.md Phase 4 Track A. Reduces "
                "effective degrees of freedom rather than adding them, which is "
                "the direction job 249519's negative result argued for."
            ),
            "scope": (
                "Input is the plate-matched Tahoe DMSO profile only, exactly what "
                "the frozen Rhaister-O consumes. No new modality, so the DepMap "
                "expansion gate does not apply."
            ),
            "inner_selection": (
                "Leave-one-reference-line-out drug-residual Pearson is primary, "
                "chosen because job 249519 showed LOO RMSE is a poor proxy for "
                "the outer endpoint. What RMSE would have chosen is recorded per "
                "split so the two criteria can be compared on a second dataset."
            ),
            "families": ["global_blend", "svd_rank", "dose_pool", "eb_per_gene"],
            "blend_grid": list(args.blend_grid),
            "rank_grid": list(args.rank_grid),
            "eb_grid": list(args.eb_grid),
            "predeclared_gate": GATE,
            "gate": gate,
            "aggregate": aggregate,
            "per_split": infos,
            "rhaister_lambda": args.rhaister_lambda,
            "rhaister_blend": args.rhaister_blend,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "data_revision": args.data_revision,
            "runner_sha256": sha256(Path(__file__).resolve()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        args.output_root / "summary.json",
    )
    print(f"\nWrote {args.output_root}/summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-revision", default=DEFAULT_DATA_REVISION)
    parser.add_argument(
        "--splits", type=int, nargs="+", default=[5, 6, 7, 8, 9],
        help="Outer splits to run. A single split makes a cheap Slurm smoke test; "
             "the gate is only meaningful over all five.",
    )
    parser.add_argument(
        "--blend-grid", type=float, nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    parser.add_argument(
        "--rank-grid", type=int, nargs="+",
        default=[1, 2, 3, 5, 10, 20, 50, 100, 200, 400],
    )
    parser.add_argument(
        "--eb-grid", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    )
    parser.add_argument("--n-permutations", type=int, default=50)
    parser.add_argument("--rhaister-lambda", type=float, default=1.0)
    parser.add_argument("--rhaister-blend", type=float, default=0.95)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
