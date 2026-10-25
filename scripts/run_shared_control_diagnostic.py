#!/usr/bin/env python3
"""Is Rhaister-O's diagonal advantage partly shared-control coupling?

The benchmark target is ``Delta = log1p(treated) - log1p(plate-matched DMSO)``
and Rhaister-O's predictor feature is that same line's DMSO profile.  The same
measurement therefore enters the label negatively and the feature positively,
so estimation noise in the DMSO centroid induces a mechanical negative
association between target and feature.  A model regressing Delta on the DMSO
level can exploit that coupling, and a held-out line's Delta carries the same
noise term, so the exploit transfers.  Neither drug-mean removal nor
reference-fitted PC1 removal can strip it: the term is constant across
treatments within a line, and it is fitted on the query line, not the
reference lines.

This script isolates that channel without recomputing any target.  Every one of
the 50 lines has a DMSO centroid on all 14 plates, and most conditions belong
to a single plate, so the plates can be halved:

``same``   conditions drawn from plate half A; feature built from half A.
           The target's control and the feature share plates -- the status quo.
``cross``  conditions drawn from plate half A; feature built from half B.
           The feature is measured on plates disjoint from every control that
           defines the target.

Both features average the same number of plates, so the extra noise from using
half the data is matched and cancels in the comparison.  The ``same`` minus
``cross`` difference is the shared-control contribution.

Interpretation. A small difference means the diagonal map is exploiting a real
baseline-to-response relationship. A large difference means a material part of
the headline 0.538 is the control appearing on both sides of the equation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_rhaister_signal import parse_treatment
from run_depmap_expression_ridge import (
    atomic_csv,
    line_bootstrap_interval,
    sha256,
    write_json,
)
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

DEFAULT_OUTPUT_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/shared_control_diagnostic"
)
ARMS = ("same", "cross")
N_PLATES = 14


def plate_subset_feature(
    hvg: dict[tuple[str, int], np.ndarray],
    cells: list[str],
    plates: set[int],
    n_genes: int,
) -> np.ndarray:
    """Mean log1p DMSO centroid over an explicit plate subset, per line.

    Rhaister-O applies log1p to a centroid and then averages; this keeps that
    order and only changes which plates contribute, so both arms share one
    functional form.
    """
    vectors = []
    for cell in cells:
        rows = [hvg[(cell, plate)] for plate in sorted(plates) if (cell, plate) in hvg]
        if not rows:
            raise ValueError(f"No DMSO centroid for {cell} on plates {sorted(plates)}")
        vectors.append(np.log1p(np.stack(rows)).mean(axis=0, dtype=np.float64))
    return np.stack(vectors)


def diagonal_predict(
    y_reference: np.ndarray,
    z_reference: np.ndarray,
    z_query: np.ndarray,
    ridge_lambda: float,
    blend: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The pinned Rhaister-O path: per (treatment, gene) diagonal ridge.

    Returns the query prediction and the reference drug-dose mean, which is
    also the additive fallback the slope term is blended against.
    """
    y = np.asarray(y_reference, dtype=np.float64)
    z = np.asarray(z_reference, dtype=np.float64)
    mean_y = y.mean(axis=0)
    mean_z = z.mean(axis=0)
    centered = z - mean_z[None, :]
    numerator = np.einsum("cg,ctg->tg", centered, y, optimize=True)
    denominator = np.square(centered).sum(axis=0) + ridge_lambda
    gamma = numerator / denominator[None, :]
    query_centered = np.asarray(z_query, dtype=np.float64) - mean_z[None, :]
    prediction = (
        mean_y[None, :, :] + blend * gamma[None, :, :] * query_centered[:, None, :]
    )
    return prediction.astype(np.float32), mean_y.astype(np.float32)


def evaluate(
    truth: np.ndarray,
    prediction: np.ndarray,
    drug_mean: np.ndarray,
) -> dict[str, float]:
    """Condition-macro Pearson at the full and drug-mean-removed levels."""
    full = rowwise_metrics(truth, prediction)
    residual = rowwise_metrics(truth - drug_mean, prediction - drug_mean)
    return {
        "pearson_full": float(full["pearson"].mean()),
        "pearson_drug_residual": float(residual["pearson"].mean()),
        "rmse_full": float(full["rmse"].mean()),
        "rmse_drug_residual": float(residual["rmse"].mean()),
    }


def condition_index(compact) -> tuple[dict[tuple[str, str], int], np.ndarray]:
    lookup = {
        (cell, treatment): index
        for index, (cell, treatment) in enumerate(zip(compact.cells, compact.treatments))
    }
    single = np.array([frozenset(p) for p in compact.plates], dtype=object)
    return lookup, single


def run_plate_split(
    args: argparse.Namespace,
    compact,
    hvg: dict,
    lookup: dict,
    plate_sets: np.ndarray,
    split: int,
    holdout_cells: list[str],
    reference_cells: list[str],
    half_a: set[int],
    half_b: set[int],
    plate_split_id: int,
) -> tuple[list[dict], dict]:
    n_genes = len(compact.genes)
    all_cells = reference_cells + holdout_cells

    # Conditions measured only on plates in half A, so every control defining a
    # retained target comes from half A.
    eligible = np.array([set(p).issubset(half_a) for p in plate_sets])
    by_cell: dict[str, set] = {}
    for index in np.flatnonzero(eligible):
        by_cell.setdefault(str(compact.cells[index]), set()).add(str(compact.treatments[index]))
    present = [by_cell.get(cell, set()) for cell in all_cells]
    if not all(present):
        raise ValueError(f"split {split}: a line has no half-A conditions")
    common = sorted(set.intersection(*present))
    if len(common) < args.min_treatments:
        raise ValueError(
            f"split {split} plate-split {plate_split_id}: only {len(common)} common treatments"
        )

    cube = np.empty((len(all_cells), len(common), n_genes), dtype=np.float32)
    for cell_position, cell in enumerate(all_cells):
        for treatment_position, treatment in enumerate(common):
            cube[cell_position, treatment_position] = compact.delta[lookup[(cell, treatment)]]
    reference_cube = cube[: len(reference_cells)]
    holdout_cube = cube[len(reference_cells) :]

    features = {
        "same": plate_subset_feature(hvg, all_cells, half_a, n_genes),
        "cross": plate_subset_feature(hvg, all_cells, half_b, n_genes),
    }

    records = []
    for arm in ARMS:
        z = features[arm]
        prediction, drug_mean = diagonal_predict(
            reference_cube,
            z[: len(reference_cells)],
            z[len(reference_cells) :],
            args.rhaister_lambda,
            args.rhaister_blend,
        )
        for cell_position, cell in enumerate(holdout_cells):
            metrics = evaluate(
                holdout_cube[cell_position],
                prediction[cell_position],
                drug_mean,
            )
            records.append({
                "split": split,
                "plate_split_id": plate_split_id,
                "arm": arm,
                "cell_line": cell,
                "n_treatments": len(common),
                **metrics,
            })

    info = {
        "split": split,
        "plate_split_id": plate_split_id,
        "half_a": sorted(half_a),
        "half_b": sorted(half_b),
        "n_eligible_conditions": int(eligible.sum()),
        "n_common_treatments": len(common),
    }
    return records, info


def main(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    repo_root = project_root / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    hvg = load_hvg_centroids(args.data_root, compact.genes)
    lookup, plate_sets = condition_index(compact)

    rng = np.random.default_rng(args.seed)
    halvings = []
    for plate_split_id in range(args.n_plate_splits):
        permutation = rng.permutation(np.arange(1, N_PLATES + 1))
        halvings.append(
            (plate_split_id, set(permutation[: N_PLATES // 2].tolist()),
             set(permutation[N_PLATES // 2 :].tolist()))
        )

    records: list[dict] = []
    infos: list[dict] = []
    for split in range(5, 10):
        holdout_cells, _, split_path = load_split(repo_root, split)
        reference_cells = sorted(set(compact.cells.tolist()) - set(holdout_cells))
        for plate_split_id, half_a, half_b in halvings:
            split_records, info = run_plate_split(
                args, compact, hvg, lookup, plate_sets, split,
                holdout_cells, reference_cells, half_a, half_b, plate_split_id,
            )
            records.extend(split_records)
            info["split_file_sha256"] = sha256(split_path)
            infos.append(info)
            same = np.mean([
                r["pearson_drug_residual"] for r in split_records if r["arm"] == "same"
            ])
            cross = np.mean([
                r["pearson_drug_residual"] for r in split_records if r["arm"] == "cross"
            ])
            print(
                f"split {split} plate-split {plate_split_id}: "
                f"treatments={info['n_common_treatments']}, "
                f"same={same:.4f} cross={cross:.4f} diff={same - cross:+.4f}",
                flush=True,
            )

    frame = pd.DataFrame(records)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(frame, args.output_root / "per_line_metrics.csv")
    atomic_csv(pd.DataFrame(infos), args.output_root / "plate_splits.csv")

    aggregate_records = []
    for level in ("pearson_full", "pearson_drug_residual"):
        # Average over plate halvings first, so each held-out line contributes
        # once per outer split and the bootstrap resamples lines, not draws.
        per_line = frame.pivot_table(
            index=["split", "cell_line"], columns="arm", values=level, aggfunc="mean"
        )
        for arm in ARMS:
            mean, low, high = line_bootstrap_interval(
                per_line[arm].to_numpy(), args.seed + len(arm) + len(level)
            )
            aggregate_records.append({
                "level": level,
                "comparison": arm,
                "n_lines": len(per_line),
                "line_macro_mean": mean,
                "line_bootstrap_low": low,
                "line_bootstrap_high": high,
            })
        difference = (per_line["same"] - per_line["cross"]).to_numpy()
        mean, low, high = line_bootstrap_interval(difference, args.seed + 991 + len(level))
        relative = float(mean / per_line["same"].mean()) if per_line["same"].mean() else float("nan")
        aggregate_records.append({
            "level": level,
            "comparison": "same_minus_cross",
            "n_lines": len(per_line),
            "line_macro_mean": mean,
            "line_bootstrap_low": low,
            "line_bootstrap_high": high,
            "fraction_of_same": relative,
        })

    aggregate = pd.DataFrame(aggregate_records)
    atomic_csv(aggregate, args.output_root / "aggregate_summary.csv")

    residual = aggregate[
        aggregate["level"].eq("pearson_drug_residual")
    ].set_index("comparison")
    coupling = float(residual.loc["same_minus_cross", "line_macro_mean"])
    coupling_low = float(residual.loc["same_minus_cross", "line_bootstrap_low"])
    fraction = float(residual.loc["same_minus_cross", "fraction_of_same"])

    summary = {
        "purpose": (
            "Quantify how much of Rhaister-O's diagonal advantage comes from the "
            "DMSO control appearing in both the target and the predictor."
        ),
        "design": (
            "Plates halved at random. Conditions restricted to those measured "
            "only on half A, so every retained target's control lies in half A. "
            "The 'same' arm builds the feature from half A, the 'cross' arm from "
            "half B. Both average 7 plates, so feature noise is matched."
        ),
        "no_target_recomputation": (
            "Targets are the frozen upstream Cell-Eval deltas; only the feature "
            "plate subset and the condition subset change."
        ),
        "n_plate_splits": args.n_plate_splits,
        "rhaister_lambda": args.rhaister_lambda,
        "rhaister_blend": args.rhaister_blend,
        "data_revision": args.data_revision,
        "seed": args.seed,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "shared_control_contribution_drug_residual": coupling,
        "shared_control_bootstrap_low": coupling_low,
        "shared_control_fraction_of_same_arm": fraction,
        "coupling_is_material": bool(coupling_low > 0 and fraction >= 0.10),
        "aggregate": aggregate_records,
        "plate_splits": infos,
    }
    write_json(summary, args.output_root / "summary.json")

    print("\nAggregate:", flush=True)
    print(aggregate.to_string(index=False), flush=True)
    print(
        f"\nShared-control contribution (drug residual): {coupling:+.4f} "
        f"[{coupling_low:+.4f}, "
        f"{residual.loc['same_minus_cross', 'line_bootstrap_high']:+.4f}], "
        f"{100 * fraction:.1f}% of the same-arm score",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-revision", default=DEFAULT_DATA_REVISION)
    parser.add_argument("--n-plate-splits", type=int, default=5)
    parser.add_argument("--min-treatments", type=int, default=200)
    parser.add_argument("--rhaister-lambda", type=float, default=1.0)
    parser.add_argument("--rhaister-blend", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
