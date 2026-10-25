#!/usr/bin/env python3
"""How sensitive is Rhaister-O to measurement noise in its DMSO feature?

The shared-control diagnostic (job 258115) matched both of its arms at seven
averaged plates *on purpose*, so the extra noise from using half the data would
cancel in the subtraction.  That makes it blind to the absolute noise level: it
bounds the shared-control channel at about 1% and says nothing about how much
the feature's own measurement noise costs.

This script measures that directly.  The DMSO feature is built from m randomly
chosen plates, for m in a declared ladder, with several independent draws per m,
and everything else is held fixed: the frozen upstream Cell-Eval targets, the
full condition set, the pinned diagonal path (lambda 1.0, blend 0.95), the five
outer splits and all 25 held-out lines.  The output is a curve of drug-residual
line-macro Pearson against m.

Reading it.  A curve already flat by m=2 says the line-level DMSO profile is
robust to measurement noise and the benchmark is not limited by it.  A curve
still climbing at m=14 says the opposite, and that more control depth would buy
accuracy.

Comparability.  Every point uses a plain mean over an explicit plate subset,
which is the functional form of ``plate_subset_feature``.  The pinned Rhaister-O
feature instead averages the plate-matched centroids of each observed condition,
so the m=14 point here is this sweep's own internal reference and is not the
headline 0.5380.  Only the shape of the curve, and differences within this
analysis, are the result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_all_rhaister import remove_pc
from evaluate_rhaister_signal import parse_treatment
from run_depmap_expression_ridge import (
    atomic_csv,
    line_bootstrap_interval,
    map_drug_means,
    response_cube,
    sha256,
    write_json,
)
from run_shared_control_diagnostic import diagonal_predict, plate_subset_feature
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

N_PLATES = 14
LEVELS = ("full", "drug_residual", "train_pca1_removed")
DEFAULT_BASELINE_ROOT = Path("/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_o_compact")
DEFAULT_COMPONENT_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_all_split_evaluation"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/plate_count_sweep"
)


def draws_for(count: int, n_draws: int, rng: np.random.Generator) -> list[set[int]]:
    """Distinct random plate subsets of the given size; m=14 has only one."""
    plates = np.arange(1, N_PLATES + 1)
    if count >= N_PLATES:
        return [set(plates.tolist())]
    seen: list[set[int]] = []
    attempts = 0
    while len(seen) < n_draws and attempts < 200 * n_draws:
        attempts += 1
        candidate = set(rng.choice(plates, size=count, replace=False).tolist())
        if candidate not in seen:
            seen.append(candidate)
    return seen


def split_analysis(
    args: argparse.Namespace, split: int, compact, hvg, rng: np.random.Generator
) -> tuple[list[dict], dict]:
    repo_root = args.project_root / "external" / "Rhaister"
    holdout_cells, _, split_path = load_split(repo_root, split)
    reference_cells = sorted(set(compact.cells.tolist()) - set(holdout_cells))

    baseline_path = args.baseline_root / f"split_{split}" / "predictions.npz"
    with np.load(baseline_path, allow_pickle=False) as archive:
        cells = archive["cell_line"].copy()
        treatments = archive["treatment"].copy()
        truth = archive["truth"].astype(np.float32)
        drug_mean = archive["perturbation_mean"].astype(np.float32)

    component_path = args.component_root / f"split_{split}" / "train_residual_pc1.npz"
    with np.load(component_path, allow_pickle=False) as component_archive:
        pc_mean = component_archive["mean"].astype(np.float32)
        pc1 = component_archive["component"].astype(np.float32)

    candidate_treatments = sorted(set(treatments.tolist()))
    response, common_treatments = response_cube(
        compact, reference_cells, candidate_treatments
    )
    common_index = {t: i for i, t in enumerate(common_treatments)}
    query_index = {c: i for i, c in enumerate(holdout_cells)}
    covered = np.array([
        cell in query_index and treatment in common_index
        for cell, treatment in zip(cells, treatments)
    ])

    def scatter(cube: np.ndarray) -> np.ndarray:
        out = np.array(drug_mean, dtype=np.float32, copy=True)
        for row, (cell, treatment) in enumerate(zip(cells, treatments)):
            if cell in query_index and treatment in common_index:
                out[row] = cube[query_index[cell], common_index[treatment]]
        return out

    all_cells = reference_cells + list(holdout_cells)
    records: list[dict] = []
    for count in args.plate_counts:
        for draw_id, plates in enumerate(draws_for(count, args.n_draws, rng)):
            feature = plate_subset_feature(hvg, all_cells, plates, len(compact.genes))
            prediction_cube, _ = diagonal_predict(
                response,
                feature[: len(reference_cells)],
                feature[len(reference_cells) :],
                args.rhaister_lambda,
                args.rhaister_blend,
            )
            prediction = scatter(prediction_cube)
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
                        "split": split,
                        "n_plates": count,
                        "draw_id": draw_id,
                        "plates": "|".join(str(p) for p in sorted(plates)),
                        "cell_line": cell,
                        "level": level,
                        "n_conditions": int(len(rows)),
                        "pearson": float(metrics["pearson"].mean()),
                        "rmse": float(metrics["rmse"].mean()),
                    })
            print(
                f"  split {split} m={count:>2} draw {draw_id}: plates={sorted(plates)}",
                flush=True,
            )
    info = {
        "split": split,
        "split_file_sha256": sha256(split_path),
        "n_reference_lines": len(reference_cells),
        "n_common_treatments": len(common_treatments),
        "n_covered_conditions": int(covered.sum()),
        "n_conditions": int(len(cells)),
    }
    return records, info


def main(args: argparse.Namespace) -> None:
    repo_root = args.project_root.resolve() / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    hvg = load_hvg_centroids(args.data_root, compact.genes)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    infos: list[dict] = []
    for split in range(5, 10):
        split_records, info = split_analysis(args, split, compact, hvg, rng)
        records.extend(split_records)
        infos.append(info)

    frame = pd.DataFrame(records)
    atomic_csv(frame, args.output_root / "per_line_metrics.csv")

    # Line-macro per (m, draw), then averaged over draws; bootstrap over lines.
    aggregate = []
    for level in LEVELS:
        subset = frame[frame["level"].eq(level)]
        for count in sorted(subset["n_plates"].unique()):
            block = subset[subset["n_plates"].eq(count)]
            per_line = block.groupby("cell_line")["pearson"].mean()
            mean, low, high = line_bootstrap_interval(
                per_line.to_numpy(dtype=np.float64), args.seed, args.n_bootstrap
            )
            aggregate.append({
                "level": level,
                "n_plates": int(count),
                "n_draws": int(block["draw_id"].nunique()),
                "n_lines": int(len(per_line)),
                "line_macro_pearson": float(mean),
                "line_bootstrap_low": float(low),
                "line_bootstrap_high": float(high),
                "across_draw_sd": float(
                    block.groupby(["draw_id"])["pearson"].mean().std(ddof=1)
                ) if block["draw_id"].nunique() > 1 else 0.0,
            })
    aggregate_frame = pd.DataFrame(aggregate)
    atomic_csv(aggregate_frame, args.output_root / "aggregate_summary.csv")

    drug = aggregate_frame[aggregate_frame["level"].eq("drug_residual")].sort_values("n_plates")
    print("\ndrug-residual line-macro Pearson versus averaged plates:")
    for row in drug.itertuples():
        print(
            f"  m={row.n_plates:>2}  {row.line_macro_pearson:.6f}  "
            f"CI [{row.line_bootstrap_low:.6f}, {row.line_bootstrap_high:.6f}]  "
            f"draw sd {row.across_draw_sd:.6f}"
        )
    fourteen = drug[drug["n_plates"].eq(N_PLATES)]["line_macro_pearson"]
    reference_value = float(fourteen.iloc[0]) if len(fourteen) else float("nan")
    write_json(
        {
            "purpose": (
                "Absolute sensitivity of the pinned diagonal path to DMSO "
                "feature measurement noise, by varying how many plates the "
                "feature averages. Complements job 258115, which matched plate "
                "count across arms and is therefore blind to this."
            ),
            "comparability_note": (
                "Every point uses a plain mean over an explicit plate subset. "
                "The pinned Rhaister-O feature averages plate-matched centroids "
                "per observed condition, so the m=14 point is this sweep's own "
                "reference and is not the headline 0.5380."
            ),
            "plate_counts": list(args.plate_counts),
            "n_draws_per_count": args.n_draws,
            "m14_drug_residual_reference": reference_value,
            "fraction_of_m14": {
                str(int(row.n_plates)): (
                    float(row.line_macro_pearson) / reference_value
                    if reference_value == reference_value and reference_value != 0
                    else None
                )
                for row in drug.itertuples()
            },
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
    parser.add_argument(
        "--plate-counts", type=int, nargs="+", default=[1, 2, 3, 4, 7, 10, 14]
    )
    parser.add_argument("--n-draws", type=int, default=5)
    parser.add_argument("--rhaister-lambda", type=float, default=1.0)
    parser.add_argument("--rhaister-blend", type=float, default=0.95)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
