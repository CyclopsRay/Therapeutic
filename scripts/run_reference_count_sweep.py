#!/usr/bin/env python3
"""Is the strict zero-shot task limited by method, or by the number of contexts?

Every modelling attempt in this project has failed to beat Rhaister-O's diagonal
map: DepMap expression (0.2087), cell-identity representations (0.029--0.093),
cross-gene low-rank conditioning (a negative point estimate with a permutation
p of 0.9901). The natural reading is that the method space is exhausted. There
is a competing reading: that 45 reference cell lines is simply not enough data
for *any* of these models to be identifiable, and the bottleneck is the screen,
not the estimator.

Those two readings imply opposite next steps, and they are distinguishable. If
performance is still climbing as reference lines are added, the task is
data-limited and more cell lines would buy accuracy. If it has plateaued, the
task is saturated at this scale and no amount of additional context data helps,
which makes the negative results a statement about the problem rather than about
our attempts.

The Rhaister paper makes a claim here that we can check independently. For its
few-shot model it reports gains going flat past roughly 20 reference contexts.
For the zero-shot model it reports "a pattern of diminishing returns across
metrics" but that "the metrics do not appear to have plateaued", and concludes
that more reference contexts would likely improve zero-shot performance. That is
their measurement on their pipeline; this is ours, on the frozen compact
derivative, at the drug-residual level they do not report.

Design, mirroring the plate-count sweep (job 270015). For each outer split the
45 reference lines are subsampled to m of them, m on a declared ladder, with
several independent draws per m. The pinned diagonal path is refit from those m
lines only and evaluated on that split's five held-out lines. Everything else is
held fixed: the frozen upstream Cell-Eval targets with no recomputation, the
treatment set determined once from all 45 lines so the prediction target does
not move with m, the pinned lambda and blend, and all 25 held-out lines.

Note what changes with m and what does not. The drug-dose mean is refit from the
m subsampled lines, so the evaluation's drug-residual level is computed against
the *frozen* 45-line drug mean while the model only sees m lines. That is the
honest setting: it asks how well a model trained on m contexts recovers the same
target, not how well it recovers an easier target.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_all_rhaister import remove_pc
from run_depmap_expression_ridge import (
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
    "/nfs-stor/yuqi.lei/Therapeutic/results/reference_count_sweep"
)


def diagonal_predict(
    response: np.ndarray, z_reference: np.ndarray, z_query: np.ndarray,
    ridge_lambda: float, blend: float,
) -> np.ndarray:
    """The pinned Rhaister-O path, fitted on whatever reference set is passed."""
    mu = response.mean(axis=0)
    centred = z_reference - z_reference.mean(axis=0)[None, :]
    residual = response - mu[None, :, :]
    denominator = np.sum(centred * centred, axis=0) + ridge_lambda
    gamma = np.einsum("cg,ctg->tg", centred, residual, optimize=True) / denominator[None, :]
    query_centred = z_query - z_reference.mean(axis=0)[None, :]
    prediction = (
        mu[None, :, :] + blend * gamma[None, :, :] * query_centred[:, None, :]
    )
    return prediction.astype(np.float32)


def draws_for(count: int, total: int, n_draws: int, rng: np.random.Generator) -> list[np.ndarray]:
    if count >= total:
        return [np.arange(total)]
    seen: list[tuple] = []
    attempts = 0
    while len(seen) < n_draws and attempts < 200 * n_draws:
        attempts += 1
        candidate = tuple(sorted(rng.choice(total, size=count, replace=False).tolist()))
        if candidate not in seen:
            seen.append(candidate)
    return [np.array(item) for item in seen]


def line_quality(hvg: dict, cells: list[str], n_genes: int) -> np.ndarray:
    """Internal reproducibility of a line's own DMSO data.

    The mean Pearson among that line's per-plate centroids, on the log1p scale.
    Measured 2026-09-21 to run from 0.654 (NCI-H596) to about 0.99, with a
    cohort median of 0.9755 and a cliff after the two worst lines.
    """
    scores = []
    for cell in cells:
        rows = [np.log1p(v) for (c, _p), v in hvg.items() if c == cell]
        if len(rows) < 2:
            scores.append(np.nan)
            continue
        matrix = np.stack(rows)
        matrix = matrix - matrix.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(matrix, axis=1)
        matrix = matrix / np.maximum(norms, np.finfo(np.float64).eps)[:, None]
        similarity = matrix @ matrix.T
        upper = similarity[np.triu_indices(len(matrix), k=1)]
        scores.append(float(upper.mean()))
    return np.asarray(scores, dtype=np.float64)


def quality_ordered_draws(
    count: int, quality: np.ndarray, mode: str, n_draws: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Subsets chosen by baseline-reproducibility rank.

    ``best_quality`` / ``worst_quality`` take the deterministic top or bottom m.
    Those two answer "is quality predictive", but a reviewer is entitled to ask
    whether the answer is an artifact of one particular set. The two randomised
    modes separate that: ``random_top30`` draws m at random from the 30 best
    lines, and ``drop_bottom5`` draws m at random from the 40 best, i.e. it only
    avoids the worst five rather than selecting the best.
    """
    order = np.argsort(-quality)  # best first
    if mode == "best_quality":
        return [np.sort(order[:count])]
    if mode == "worst_quality":
        return [np.sort(order[::-1][:count])]
    pool = {"random_top30": order[:30], "drop_bottom5": order[:40]}.get(mode)
    if pool is None:
        raise ValueError(f"unknown quality mode {mode}")
    if count >= len(pool):
        return [np.sort(pool)]
    seen: list[tuple] = []
    attempts = 0
    while len(seen) < n_draws and attempts < 200 * n_draws:
        attempts += 1
        candidate = tuple(sorted(rng.choice(pool, size=count, replace=False).tolist()))
        if candidate not in seen:
            seen.append(candidate)
    return [np.array(item) for item in seen]


def split_analysis(args, split: int, compact, hvg, rng) -> tuple[list[dict], dict]:
    repo_root = args.project_root / "external" / "Rhaister"
    holdout_cells, _, split_path = load_split(repo_root, split)
    reference_cells = sorted(set(compact.cells.tolist()) - set(holdout_cells))

    with np.load(
        args.baseline_root / f"split_{split}" / "predictions.npz", allow_pickle=False
    ) as archive:
        cells = archive["cell_line"].copy()
        treatments = archive["treatment"].copy()
        truth = archive["truth"].astype(np.float32)
        drug_mean = archive["perturbation_mean"].astype(np.float32)

    with np.load(
        args.component_root / f"split_{split}" / "train_residual_pc1.npz", allow_pickle=False
    ) as component_archive:
        pc_mean = component_archive["mean"].astype(np.float32)
        pc1 = component_archive["component"].astype(np.float32)

    # The treatment set is fixed once, from all 45 lines, so the prediction
    # target does not move as m shrinks.
    candidate_treatments = sorted(set(treatments.tolist()))
    response, common_treatments = response_cube(
        compact, reference_cells, candidate_treatments
    )
    response = response.astype(np.float64)
    common_index = {t: i for i, t in enumerate(common_treatments)}
    query_index = {c: i for i, c in enumerate(holdout_cells)}
    covered = np.array([
        cell in query_index and treatment in common_index
        for cell, treatment in zip(cells, treatments)
    ])

    reference_z = line_hvg_vectors(compact, hvg, reference_cells)
    query_z = line_hvg_vectors(compact, hvg, holdout_cells)

    quality = line_quality(hvg, reference_cells, len(compact.genes))

    def subsets_for(count: int) -> list[tuple[str, int, np.ndarray]]:
        out: list[tuple[str, int, np.ndarray]] = []
        for draw_id, subset in enumerate(
            draws_for(count, len(reference_cells), args.n_draws, rng)
        ):
            out.append(("random", draw_id, subset))
        if args.quality_modes and count < len(reference_cells):
            for mode in args.quality_modes:
                for draw_id, subset in enumerate(
                    quality_ordered_draws(count, quality, mode, args.n_draws, rng)
                ):
                    out.append((mode, draw_id, subset))
        return out

    records: list[dict] = []
    for count in args.reference_counts:
        for mode, draw_id, subset in subsets_for(count):
            cube = diagonal_predict(
                response[subset], reference_z[subset], query_z,
                args.rhaister_lambda, args.rhaister_blend,
            )
            prediction = np.array(drug_mean, dtype=np.float32, copy=True)
            for row, (cell, treatment) in enumerate(zip(cells, treatments)):
                if cell in query_index and treatment in common_index:
                    prediction[row] = cube[query_index[cell], common_index[treatment]]
            for cell in holdout_cells:
                rows = np.flatnonzero((cells == cell) & covered)
                if not len(rows):
                    continue
                truth_residual = truth[rows] - drug_mean[rows]
                prediction_residual = prediction[rows] - drug_mean[rows]
                values = {
                    "full": rowwise_metrics(truth[rows], prediction[rows]),
                    "drug_residual": rowwise_metrics(truth_residual, prediction_residual),
                    "train_pca1_removed": rowwise_metrics(
                        remove_pc(truth_residual, pc_mean, pc1),
                        remove_pc(prediction_residual, pc_mean, pc1),
                    ),
                }
                for level, metrics in values.items():
                    records.append({
                        "split": split, "n_reference": count, "draw_id": draw_id,
                        "mode": mode,
                        "subset_mean_quality": float(np.nanmean(quality[subset])),
                        "cell_line": cell, "level": level,
                        "n_conditions": int(len(rows)),
                        "pearson": float(metrics["pearson"].mean()),
                        "rmse": float(metrics["rmse"].mean()),
                    })
            print(f"  split {split} m={count:>2} {mode} draw {draw_id} (mean QC {np.nanmean(quality[subset]):.4f})", flush=True)
    info = {
        "split": split,
        "split_file_sha256": sha256(split_path),
        "n_reference_lines": len(reference_cells),
        "n_common_treatments": len(common_treatments),
        "n_covered_conditions": int(covered.sum()),
    }
    return records, info


def main(args: argparse.Namespace) -> None:
    repo_root = args.project_root.resolve() / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    hvg = load_hvg_centroids(args.data_root, compact.genes)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    records, infos = [], []
    for split in args.splits:
        split_records, info = split_analysis(args, split, compact, hvg, rng)
        records.extend(split_records)
        infos.append(info)

    frame = pd.DataFrame(records)
    atomic_csv(frame, args.output_root / "per_line_metrics.csv")

    aggregate = []
    for level in LEVELS:
        subset = frame[frame["level"].eq(level)]
        for mode in sorted(subset["mode"].unique()):
          for count in sorted(subset["n_reference"].unique()):
            block = subset[subset["n_reference"].eq(count) & subset["mode"].eq(mode)]
            if not len(block):
                continue
            per_line = block.groupby("cell_line")["pearson"].mean()
            mean, low, high = line_bootstrap_interval(
                per_line.to_numpy(dtype=np.float64), args.seed, args.n_bootstrap
            )
            aggregate.append({
                "level": level, "mode": mode, "n_reference": int(count),
                "subset_mean_quality": float(block["subset_mean_quality"].mean()),
                "n_draws": int(block["draw_id"].nunique()),
                "n_lines": int(len(per_line)),
                "line_macro_pearson": float(mean),
                "line_bootstrap_low": float(low), "line_bootstrap_high": float(high),
                "across_draw_sd": float(
                    block.groupby("draw_id")["pearson"].mean().std(ddof=1)
                ) if block["draw_id"].nunique() > 1 else 0.0,
            })
    aggregate_frame = pd.DataFrame(aggregate)
    atomic_csv(aggregate_frame, args.output_root / "aggregate_summary.csv")

    drug = aggregate_frame[aggregate_frame["level"].eq("drug_residual")
                           & aggregate_frame["mode"].eq("random")].sort_values("n_reference")
    print("\ndrug-residual line-macro Pearson versus number of reference lines:")
    full_value = float(drug[drug["n_reference"].eq(drug["n_reference"].max())]["line_macro_pearson"].iloc[0])
    previous = None
    marginal = []
    for row in drug.itertuples():
        delta = "" if previous is None else f"  marginal {row.line_macro_pearson - previous:+.6f}"
        if previous is not None:
            marginal.append({
                "from": int(prev_count), "to": int(row.n_reference),
                "gain": float(row.line_macro_pearson - previous),
                "gain_per_line": float(
                    (row.line_macro_pearson - previous) / (row.n_reference - prev_count)
                ),
            })
        print(
            f"  m={row.n_reference:>2}  {row.line_macro_pearson:.6f}  "
            f"CI [{row.line_bootstrap_low:.6f}, {row.line_bootstrap_high:.6f}]  "
            f"draw sd {row.across_draw_sd:.6f}  ({100*row.line_macro_pearson/full_value:.2f}% of full){delta}"
        )
        previous, prev_count = row.line_macro_pearson, row.n_reference

    write_json(
        {
            "purpose": (
                "Distinguish a method-limited task from a data-limited one by "
                "subsampling the reference cell lines and refitting the pinned "
                "diagonal path. Independently checks the Rhaister paper's claim "
                "that zero-shot performance has not plateaued in the number of "
                "reference contexts."
            ),
            "design_note": (
                "The treatment set is fixed once from all 45 reference lines so "
                "the prediction target does not move with m. The drug-residual "
                "level is scored against the frozen 45-line drug mean, so a "
                "model fitted on m lines is asked to recover the same target, "
                "not an easier one."
            ),
            "reference_counts": list(args.reference_counts),
            "n_draws_per_count": args.n_draws,
            "marginal_gains": marginal,
            "full_reference_drug_residual": full_value,
            "rhaister_lambda": args.rhaister_lambda,
            "rhaister_blend": args.rhaister_blend,
            "no_target_recomputation": True,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "data_revision": args.data_revision,
            "runner_sha256": sha256(Path(__file__).resolve()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "aggregate": aggregate,
            "per_split": infos,
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
    parser.add_argument("--splits", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    parser.add_argument(
        "--reference-counts", type=int, nargs="+",
        default=[5, 10, 15, 20, 25, 30, 35, 40, 45],
    )
    parser.add_argument("--n-draws", type=int, default=5)
    parser.add_argument(
        "--quality-modes", nargs="*",
        default=["best_quality", "worst_quality", "random_top30", "drop_bottom5"],
        help="Also run deterministic best-QC and worst-QC subsets at each m, "
             "to separate the value of more reference lines from the value of "
             "better ones. Pass with no values to disable.",
    )
    parser.add_argument("--rhaister-lambda", type=float, default=1.0)
    parser.add_argument("--rhaister-blend", type=float, default=0.95)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
