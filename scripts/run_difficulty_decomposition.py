#!/usr/bin/env python3
"""What makes one held-out line, or one drug, easy or hard?

Every result in this project so far is an aggregate. The aggregate hides an
enormous spread: PC1-removed per-line Pearson runs from 0.0648 for NCI-H2122 to
0.6850 for HOP-62. Nothing has yet asked where that spread comes from.

This script answers it from artifacts that are already frozen -- the 19,268
per-condition rows in `results/rhaister_o_compact/split_*/per_condition_metrics.csv`
plus the truth and prediction matrices in the same directories. Nothing is
refitted and no model is selected, so there is no gate and no leakage surface:
this is a descriptive decomposition of results that already exist.

Three questions:

1. **Where does the variance live?** Decompose per-condition Pearson into a
   cell-line main effect, a drug main effect, a dose effect and a residual
   interaction. If the line effect dominates, per-line difficulty is a property
   of the context and worth predicting; if the drug effect dominates, the
   interesting axis is chemistry, not biology.

2. **What predicts a hard line?** Candidate covariates computed from inputs the
   model is allowed to see: how far the line's DMSO profile sits from the
   reference centroid (an extrapolation distance), how far it sits from its
   nearest reference line, and how large its responses are. A line that is hard
   because it is an extrapolation is a different problem from one that is hard
   because its responses are small.

3. **What predicts a hard drug?** Dose, response magnitude, and how much the
   drug's response varies across lines. A drug whose response is identical
   everywhere is already solved by the drug mean; the interesting drugs are the
   context-dependent ones, and whether Rhaister-O actually wins on those is the
   question that matters for the paper's claim.

The framing matters for `PLAN.md`: direction C (identifiability and
decomposition) needs exactly this, and unlike a new model it cannot fail a gate.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_rhaister_signal import parse_treatment
from run_depmap_expression_ridge import atomic_csv, sha256, write_json
from run_depmap_rhaister_correction import line_hvg_vectors
from run_rhaister_o_compact import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_REVISION,
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    build_or_load_compact_cache,
    load_hvg_centroids,
    load_split,
)

DEFAULT_BASELINE_ROOT = Path("/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_o_compact")
DEFAULT_OUTPUT_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/difficulty_decomposition"
)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep] - x[keep].mean(), y[keep] - y[keep].mean()
    denominator = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / denominator) if denominator > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(a):
        a = np.asarray(a, dtype=np.float64)
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=np.float64)
        r[order] = np.arange(1, len(a) + 1, dtype=np.float64)
        sorted_a = a[order]
        i = 0
        while i < len(sorted_a):
            j = i
            while j + 1 < len(sorted_a) and sorted_a[j + 1] == sorted_a[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
            i = j + 1
        return r
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    return pearson(rank(x[keep]), rank(y[keep]))


def two_way_decomposition(
    values: np.ndarray, row_key: np.ndarray, col_key: np.ndarray
) -> dict:
    """Additive main-effect decomposition of a per-condition statistic.

    Sequential, not orthogonal: the design is unbalanced, so the row and column
    shares depend on the order they are removed. Both orders are reported, and
    the honest reading is the range between them.
    """
    values = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(values)
    values, row_key, col_key = values[keep], row_key[keep], col_key[keep]
    grand = values.mean()
    total = float(((values - grand) ** 2).sum())

    def share(first, second):
        _, first_index = np.unique(first, return_inverse=True)
        first_mean = np.zeros(first_index.max() + 1)
        np.add.at(first_mean, first_index, values)
        first_mean /= np.bincount(first_index)
        after_first = values - first_mean[first_index]
        ss_first = total - float((after_first - after_first.mean()) ** 2 @ np.ones(len(values)))

        _, second_index = np.unique(second, return_inverse=True)
        second_mean = np.zeros(second_index.max() + 1)
        np.add.at(second_mean, second_index, after_first)
        second_mean /= np.bincount(second_index)
        after_second = after_first - second_mean[second_index]
        ss_second = float(((after_first - after_first.mean()) ** 2).sum()) - float(
            ((after_second - after_second.mean()) ** 2).sum()
        )
        residual = float(((after_second - after_second.mean()) ** 2).sum())
        return ss_first / total, ss_second / total, residual / total

    line_first = share(row_key, col_key)
    drug_first = share(col_key, row_key)
    return {
        "total_ss": total,
        "n": int(len(values)),
        "line_then_drug": {
            "line": line_first[0], "drug": line_first[1], "residual": line_first[2]
        },
        "drug_then_line": {
            "drug": drug_first[0], "line": drug_first[1], "residual": drug_first[2]
        },
    }


def main(args: argparse.Namespace) -> None:
    repo_root = args.project_root.resolve() / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    hvg = load_hvg_centroids(args.data_root, compact.genes)
    args.output_root.mkdir(parents=True, exist_ok=True)

    frames = []
    line_rows = []
    for split in range(5, 10):
        path = args.baseline_root / f"split_{split}" / "per_condition_metrics.csv"
        frame = pd.read_csv(path)
        frame["split"] = split
        frames.append(frame)

        holdout_cells, _, _ = load_split(repo_root, split)
        reference_cells = sorted(set(compact.cells.tolist()) - set(holdout_cells))
        reference_z = line_hvg_vectors(compact, hvg, reference_cells)
        query_z = line_hvg_vectors(compact, hvg, holdout_cells)
        centroid = reference_z.mean(axis=0)

        with np.load(
            args.baseline_root / f"split_{split}" / "predictions.npz", allow_pickle=False
        ) as archive:
            cells = archive["cell_line"]
            truth = archive["truth"].astype(np.float32)
            drug_mean = archive["perturbation_mean"].astype(np.float32)

        for index, cell in enumerate(holdout_cells):
            distances = np.linalg.norm(reference_z - query_z[index][None, :], axis=1)
            rows = np.flatnonzero(cells == cell)
            residual = truth[rows] - drug_mean[rows]
            line_rows.append({
                "split": split,
                "cell_line": cell,
                "distance_to_reference_centroid": float(
                    np.linalg.norm(query_z[index] - centroid)
                ),
                "distance_to_nearest_reference": float(distances.min()),
                "mean_distance_to_reference": float(distances.mean()),
                "response_magnitude": float(np.abs(truth[rows]).mean()),
                "residual_magnitude": float(np.abs(residual).mean()),
                "n_conditions": int(len(rows)),
            })

    condition = pd.concat(frames, ignore_index=True)
    parsed = [parse_treatment(t) for t in condition["treatment"]]
    condition["drug"] = [item[0] for item in parsed]
    condition["dose"] = [item[1] for item in parsed]
    condition["gain_over_drug_mean"] = (
        condition["rhaister_o__pearson"] - condition["perturbation_mean__pearson"]
    )
    atomic_csv(condition, args.output_root / "per_condition_joined.csv")

    covered = condition[condition["diagonal_covered"].astype(bool)]
    decomposition = {
        "rhaister_o__pearson": two_way_decomposition(
            covered["rhaister_o__pearson"].to_numpy(),
            covered["cell_line"].to_numpy(),
            covered["drug"].to_numpy(),
        ),
        "gain_over_drug_mean": two_way_decomposition(
            covered["gain_over_drug_mean"].to_numpy(),
            covered["cell_line"].to_numpy(),
            covered["drug"].to_numpy(),
        ),
    }

    line_frame = pd.DataFrame(line_rows)
    line_scores = covered.groupby("cell_line").agg(
        rhaister_pearson=("rhaister_o__pearson", "mean"),
        drug_mean_pearson=("perturbation_mean__pearson", "mean"),
        gain=("gain_over_drug_mean", "mean"),
    ).reset_index()
    line_frame = line_frame.merge(line_scores, on="cell_line", how="left")
    atomic_csv(line_frame, args.output_root / "per_line_covariates.csv")

    covariates = [
        "distance_to_reference_centroid", "distance_to_nearest_reference",
        "mean_distance_to_reference", "response_magnitude",
        "residual_magnitude", "n_conditions",
    ]
    line_correlations = [
        {
            "covariate": name,
            "target": target,
            "pearson": pearson(line_frame[name], line_frame[target]),
            "spearman": spearman(line_frame[name], line_frame[target]),
            "n_lines": int(line_frame[name].notna().sum()),
        }
        for name in covariates
        for target in ("rhaister_pearson", "gain")
    ]

    drug_frame = covered.groupby("drug").agg(
        rhaister_pearson=("rhaister_o__pearson", "mean"),
        drug_mean_pearson=("perturbation_mean__pearson", "mean"),
        gain=("gain_over_drug_mean", "mean"),
        n_conditions=("rhaister_o__pearson", "size"),
    ).reset_index()
    # How context-dependent is this drug? Spread of its per-line score.
    spread = covered.groupby(["drug", "cell_line"])["rhaister_o__pearson"].mean().reset_index()
    spread = spread.groupby("drug")["rhaister_o__pearson"].std(ddof=1).rename("across_line_sd")
    drug_frame = drug_frame.merge(spread, on="drug", how="left")
    atomic_csv(drug_frame, args.output_root / "per_drug_summary.csv")

    dose_frame = covered.groupby("dose").agg(
        rhaister_pearson=("rhaister_o__pearson", "mean"),
        drug_mean_pearson=("perturbation_mean__pearson", "mean"),
        gain=("gain_over_drug_mean", "mean"),
        n=("rhaister_o__pearson", "size"),
    ).reset_index()

    print("\n=== variance decomposition of per-condition Pearson ===")
    for key, value in decomposition.items():
        a, b = value["line_then_drug"], value["drug_then_line"]
        print(f"  {key}  (n={value['n']})")
        print(f"    line first : line {a['line']:.3f}  drug {a['drug']:.3f}  residual {a['residual']:.3f}")
        print(f"    drug first : drug {b['drug']:.3f}  line {b['line']:.3f}  residual {b['residual']:.3f}")

    print("\n=== per-line covariates versus difficulty (25 lines) ===")
    for row in line_correlations:
        if row["target"] == "rhaister_pearson":
            print(f"  {row['covariate']:34s} pearson {row['pearson']:+.4f}  spearman {row['spearman']:+.4f}")

    print("\n=== per dose ===")
    print(dose_frame.to_string(index=False))

    write_json(
        {
            "purpose": (
                "Descriptive decomposition of already-frozen per-condition "
                "results. No refitting, no model selection, no gate."
            ),
            "n_conditions": int(len(condition)),
            "n_covered": int(len(covered)),
            "variance_decomposition": decomposition,
            "line_correlations": line_correlations,
            "dose_summary": dose_frame.to_dict(orient="records"),
            "hardest_lines": line_frame.nsmallest(5, "rhaister_pearson")[
                ["cell_line", "rhaister_pearson", "gain", "distance_to_reference_centroid"]
            ].to_dict(orient="records"),
            "easiest_lines": line_frame.nlargest(5, "rhaister_pearson")[
                ["cell_line", "rhaister_pearson", "gain", "distance_to_reference_centroid"]
            ].to_dict(orient="records"),
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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-revision", default=DEFAULT_DATA_REVISION)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
