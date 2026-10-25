#!/usr/bin/env python3
"""Materialize the bilinear coefficient tensor B for the frozen low-rank fit.

``run_dmso_lowrank_bilinear.py`` predicts through the weight form
``W = D_query (D'D + lambda P)^-1 D'`` and never forms

    B_hat = (D'D + lambda P)^-1 D' Y      shape (K+1) x (n_treatments * n_genes)

so the cross-gene response programs were never inspectable.  This script
recovers them without refitting anything new: it reproduces the same
reference-only PCA, reads the *selected* (K, lambda) out of the frozen
``ridge_selection.csv`` of each split and arm, and evaluates the closed form at
that single configuration.

It writes nothing into the frozen bilinear output directory, does no model
selection of its own, and changes no gate.  Interpretation aid: it also stores
the gene-space loading of every retained component, so a row of B can be read
against the baseline direction that drives it.

B[0] is the intercept, per (treatment, gene).  For the correction arm that is
the mean leave-one-reference-line-out Rhaister-O error, i.e. a systematic-bias
map.  B[k] for k >= 1 is the change in the response of gene g under treatment p
per one standard deviation of the query line's whitened score on baseline
direction k.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from run_depmap_rhaister_correction import line_hvg_vectors, loo_rhaister_corrections
from run_depmap_expression_ridge import (
    map_drug_means,
    response_cube,
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
)

ARMS = ("dmso_lowrank_direct", "dmso_lowrank_correction")
DEFAULT_BILINEAR_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/dmso_lowrank_bilinear"
)
DEFAULT_BASELINE_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_o_compact"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/dmso_lowrank_coefficients"
)


def pca_with_loadings(reference: np.ndarray, query: np.ndarray) -> dict:
    """Byte-for-byte the arithmetic of ``reference_only_pca``, plus loadings."""
    reference64 = np.asarray(reference, dtype=np.float64)
    query64 = np.asarray(query, dtype=np.float64)
    mean = np.nanmean(reference64, axis=0)
    reference_filled = np.where(np.isfinite(reference64), reference64, mean)
    query_filled = np.where(np.isfinite(query64), query64, mean)
    scale = reference_filled.std(axis=0)
    keep = np.isfinite(mean) & np.isfinite(scale) & (scale > 1e-8)
    reference_scaled = (reference_filled[:, keep] - mean[keep]) / scale[keep]
    query_scaled = (query_filled[:, keep] - mean[keep]) / scale[keep]
    u, singular_values, vt = np.linalg.svd(reference_scaled, full_matrices=False)
    scores = u * singular_values[None, :]
    query_scores = query_scaled @ vt.T
    score_scale = scores.std(axis=0)
    usable = score_scale > 1e-10
    return {
        "scores": scores[:, usable] / score_scale[usable],
        "query_scores": query_scores[:, usable] / score_scale[usable],
        "loadings": vt[usable],
        "score_scale": score_scale[usable],
        "singular_values": singular_values[usable],
        "feature_mean": mean,
        "feature_scale": scale,
        "keep": keep,
    }


def selected_configuration(bilinear_root: Path, split: int, arm: str) -> tuple[int, float]:
    frame = pd.read_csv(bilinear_root / f"split_{split}" / "ridge_selection.csv")
    rows = frame[frame["arm"].eq(arm) & frame["selected"].astype(bool)]
    if len(rows) != 1:
        raise ValueError(
            f"split {split} arm {arm}: expected exactly one selected row, got {len(rows)}"
        )
    row = rows.iloc[0]
    return int(row["n_components"]), float(row["ridge_lambda"])


def coefficients(
    target_flat: np.ndarray,
    scores: np.ndarray,
    n_components: int,
    ridge_lambda: float,
) -> np.ndarray:
    n = len(scores)
    if n_components == 0:
        design = np.ones((n, 1), dtype=np.float64)
    else:
        design = np.column_stack(
            [np.ones(n, dtype=np.float64), scores[:, :n_components]]
        )
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_lambda
    penalty[0, 0] = 0.0
    inverse_gram = np.linalg.inv(design.T @ design + penalty)
    return inverse_gram @ design.T @ target_flat


def split_analysis(args: argparse.Namespace, split: int, compact, hvg) -> dict:
    repo_root = args.project_root / "external" / "Rhaister"
    holdout_cells, _, split_path = load_split(repo_root, split)
    reference_cells = sorted(set(compact.cells.tolist()) - set(holdout_cells))

    baseline_path = args.baseline_root / f"split_{split}" / "predictions.npz"
    with np.load(baseline_path, allow_pickle=False) as archive:
        genes = archive["genes"].copy()
        treatments = archive["treatment"].copy()
        drug_mean = archive["perturbation_mean"].astype(np.float32)

    candidate_treatments = sorted(set(treatments.tolist()))
    response, common_treatments = response_cube(
        compact, reference_cells, candidate_treatments
    )
    drug_mean_map = map_drug_means(treatments, drug_mean)
    reference_mean = np.stack([drug_mean_map[t] for t in common_treatments])

    reference_hvg = line_hvg_vectors(compact, hvg, reference_cells)
    query_hvg = line_hvg_vectors(compact, hvg, holdout_cells)
    pca = pca_with_loadings(reference_hvg, query_hvg)

    targets = {
        "dmso_lowrank_direct": (response - reference_mean[None, :, :]).reshape(
            len(reference_cells), -1
        ),
        "dmso_lowrank_correction": loo_rhaister_corrections(
            response, reference_hvg, args.rhaister_lambda, args.rhaister_blend
        ).reshape(len(reference_cells), -1),
    }

    output_dir = args.output_root / f"split_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for arm in ARMS:
        n_components, ridge_lambda = selected_configuration(
            args.bilinear_root, split, arm
        )
        coef = coefficients(
            targets[arm], pca["scores"], n_components, ridge_lambda
        )
        cube = coef.reshape(
            n_components + 1, len(common_treatments), len(compact.genes)
        ).astype(np.float32)
        path = output_dir / f"coefficients_{arm}.npz"
        # np.savez_compressed appends ".npz" unless the name already ends in it,
        # so the temporary name must end in ".npz" for os.replace to find it.
        temporary = output_dir / f".coefficients_{arm}.tmp.npz"
        np.savez_compressed(
            temporary,
            coefficients=cube,
            treatments=np.asarray(common_treatments),
            genes=genes,
            loadings=pca["loadings"][:n_components].astype(np.float32),
            kept_gene_mask=pca["keep"],
            feature_mean=pca["feature_mean"].astype(np.float32),
            feature_scale=pca["feature_scale"].astype(np.float32),
            score_scale=pca["score_scale"][:n_components].astype(np.float32),
            n_components=np.asarray([n_components]),
            ridge_lambda=np.asarray([ridge_lambda]),
        )
        os.replace(temporary, path)

        magnitude = np.abs(cube).mean(axis=(1, 2))
        records.append({
            "split": split,
            "arm": arm,
            "n_components": n_components,
            "ridge_lambda": ridge_lambda,
            "shape": list(cube.shape),
            "path": str(path),
            "intercept_mean_abs": float(magnitude[0]),
            "program_mean_abs": [float(v) for v in magnitude[1:]],
            "frobenius_by_component": [
                float(np.linalg.norm(cube[k])) for k in range(cube.shape[0])
            ],
        })
        print(
            f"  split {split} {arm}: K={n_components} lam={ridge_lambda:g} "
            f"shape={cube.shape} -> {path.name}",
            flush=True,
        )
    return {
        "split": split,
        "split_file_sha256": sha256(split_path),
        "n_reference_lines": len(reference_cells),
        "n_common_treatments": len(common_treatments),
        "pca_available_components": int(pca["scores"].shape[1]),
        "arms": records,
    }


def main(args: argparse.Namespace) -> None:
    repo_root = args.project_root.resolve() / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    hvg = load_hvg_centroids(args.data_root, compact.genes)
    args.output_root.mkdir(parents=True, exist_ok=True)
    per_split = [split_analysis(args, split, compact, hvg) for split in range(5, 10)]
    write_json(
        {
            "purpose": (
                "Materialize B_hat = (D'D + lambda P)^-1 D' Y at the frozen "
                "selected (K, lambda) of each split and arm. No new selection, "
                "no new fitting decision, no gate change."
            ),
            "selection_source": str(args.bilinear_root),
            "interpretation": (
                "B[0] is the per-(treatment, gene) intercept; for the correction "
                "arm that is the mean LOO Rhaister-O error. B[k>=1] is the "
                "response change per one SD of the whitened score on baseline "
                "direction k, whose gene-space loading is stored alongside."
            ),
            "rhaister_lambda": args.rhaister_lambda,
            "rhaister_blend": args.rhaister_blend,
            "data_revision": args.data_revision,
            "runner_sha256": sha256(Path(__file__).resolve()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "per_split": per_split,
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
    parser.add_argument("--bilinear-root", type=Path, default=DEFAULT_BILINEAR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-revision", default=DEFAULT_DATA_REVISION)
    parser.add_argument("--rhaister-lambda", type=float, default=1.0)
    parser.add_argument("--rhaister-blend", type=float, default=0.95)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
