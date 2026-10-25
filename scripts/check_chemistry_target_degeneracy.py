#!/usr/bin/env python3
"""Is the selected chemistry shrinkage target just the pooled slope in disguise?

Job 274091 found that every chemistry arm selects rank 1 -- the bottom of a
1..20 grid -- and that the arms willing to shrink at all pick the largest ridge
penalty available (1000), while the arms at the smallest penalty (10) refuse to
shrink (w = 1). That pattern suggests the chemistry target only becomes tolerable
once it has been regularised into near-degeneracy, i.e. once it stops carrying
drug-specific information.

This script tests that directly on one split: build the observed slopes, build
the chemistry target at the selected (rank, lambda), and measure how far that
target is from the pooled per-gene slope that uniform shrinkage uses. If the two
nearly coincide, the "chemistry" arm was rediscovering uniform shrinkage through
a degenerate path, and the correct statement is that the only chemistry
configuration the data tolerates is the one that no longer uses chemistry.

Descriptive only: no model selection, no gate, nothing published depends on it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_chemistry_shrinkage import (
    build_drug_features, chemistry_target, hat_matrix, standardise,
)
from run_coefficient_sharing import slope_estimates
from run_depmap_expression_ridge import response_cube, write_json
from run_depmap_rhaister_correction import line_hvg_vectors
from run_rhaister_o_compact import (
    DEFAULT_CACHE_ROOT, DEFAULT_DATA_ROOT, DEFAULT_PROJECT_ROOT,
    build_or_load_compact_cache, load_hvg_centroids, load_split,
)


def relative_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def main(args: argparse.Namespace) -> None:
    repo_root = args.project_root.resolve() / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    hvg = load_hvg_centroids(args.data_root, compact.genes)
    holdout, _, _ = load_split(repo_root, args.split)
    reference = sorted(set(compact.cells.tolist()) - set(holdout))

    with np.load(
        Path("/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_o_compact")
        / f"split_{args.split}" / "predictions.npz", allow_pickle=False
    ) as archive:
        candidate = sorted(set(archive["treatment"].tolist()))

    response, common = response_cube(compact, reference, candidate)
    z = line_hvg_vectors(compact, hvg, reference)
    z = z - z.mean(axis=0)
    _, gamma, pooled = slope_estimates(z, response.astype(np.float64), 1.0)
    broadcast_pooled = np.broadcast_to(pooled, gamma.shape)

    records = []
    for kind, label in (("morgan_moa", "chem_full"), ("morgan", "morgan_only"),
                        ("moa", "moa_only")):
        raw, _ = build_drug_features(args.drug_metadata, list(common), kind)
        design = standardise(raw)
        for ridge_lambda in args.lambda_grid:
            smoother = hat_matrix(design, ridge_lambda)
            targets = chemistry_target(gamma, pooled, smoother, tuple(args.rank_grid))
            for rank in args.rank_grid:
                target = targets[rank]
                records.append({
                    "arm": label, "ridge_lambda": ridge_lambda, "rank": rank,
                    "relative_distance_to_pooled": relative_distance(target, broadcast_pooled),
                    "relative_distance_to_gamma": relative_distance(target, gamma),
                    "target_frobenius": float(np.linalg.norm(target)),
                    "pooled_frobenius": float(np.linalg.norm(broadcast_pooled)),
                    "gamma_frobenius": float(np.linalg.norm(gamma)),
                })
                print(f"  {label:12s} lam={ridge_lambda:<7g} rank={rank:<3d} "
                      f"|target-pooled|/|pooled| = {records[-1]['relative_distance_to_pooled']:.4f}   "
                      f"|target-gamma|/|gamma| = {records[-1]['relative_distance_to_gamma']:.4f}",
                      flush=True)
            del targets
    write_json({"split": args.split, "records": records,
                "reading": ("A relative distance to pooled near 0 means the chemistry "
                            "target has collapsed onto the uniform-shrinkage target.")},
               args.output_root / f"split_{args.split}_degeneracy.json")
    print(f"\nWrote {args.output_root}/split_{args.split}_degeneracy.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--drug-metadata", type=Path, default=Path(
        "/nfs-stor/yuqi.lei/Therapeutic/raw/tahoe-100m/metadata/drug_metadata.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path(
        "/nfs-stor/yuqi.lei/Therapeutic/results/chemistry_shrinkage"))
    parser.add_argument("--split", type=int, default=5)
    parser.add_argument("--rank-grid", type=int, nargs="+", default=[1, 2, 5, 20])
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[10.0, 100.0, 1000.0])
    main(parser.parse_args())
