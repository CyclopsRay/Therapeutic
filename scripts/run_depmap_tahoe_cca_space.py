#!/usr/bin/env python3
"""Rhaister-O fitted inside the DepMap<->Tahoe CCA space (record 2026-10-08).

The CCA is the one in run_depmap_tahoe_cca.py (reference lines only, (k, d) by
label-free inner LOO).  Its Tahoe-side map is affine, so both the plate-matched
DMSO and the absolute post-treatment profile log1p(DMSO) + delta project into
the same d canonical coordinates.  Per treatment t and canonical dimension j:

    target(c, t, j) = mu(t, j) + gamma(t, j) * (x(c, j) - mean_ref x(., j))

fitted on reference lines with x = their Tahoe DMSO coordinates.  A held-out
line is predicted from its DepMap coordinates (as-is or rho-shrunk), its real
Tahoe DMSO coordinates (upper bound), zero (no line information) or another
held-out line's DepMap coordinates.  Scored against projected Tahoe truth, as
absolute post-treatment (primary, user request) and as delta (secondary).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from run_depmap_expression_ridge import (
    DEFAULT_DEPMAP_ROOT,
    atomic_csv,
    line_bootstrap_interval,
    load_depmap_expression,
    sha256,
    write_json,
)
from run_depmap_rhaister_correction import line_hvg_vectors
from run_depmap_tahoe_cca import Cca, select_kd
from run_rhaister_o_compact import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    build_or_load_compact_cache,
    condition_hvg,
    load_hvg_centroids,
    load_split,
    rowwise_metrics,
)

DEFAULT_OUTPUT_ROOT = Path("/nfs-stor/yuqi.lei/Therapeutic/results/depmap_tahoe_cca_space")
INPUTS = ("tahoe_dmso", "depmap", "depmap_shrunk", "mean", "shuffled_depmap")
BASELINES = ("identity_only_tahoe", "identity_only_depmap", "drug_mean")
TARGETS = ("absolute", "delta")


def linear_y(model: Cca, delta: np.ndarray) -> np.ndarray:
    """Linear part of the Tahoe-side canonical map (for deltas)."""
    py = model.py
    return (np.asarray(delta, np.float64)[:, py.keep] / py.scale) @ py.vt.T / py.sd @ model.b


def roundtrip_pearson(model: Cca, delta: np.ndarray) -> np.ndarray:
    """Pearson between standardised gene-level delta and its canonical round trip."""
    py = model.py
    s = np.asarray(delta, np.float64)[:, py.keep] / py.scale
    back = ((linear_y(model, delta) @ model.b.T) * py.sd) @ py.vt
    return rowwise_metrics(s, back)["pearson"]


def fit_predict(x_ref: np.ndarray, y_ref: np.ndarray, x_query: np.ndarray, lam: float) -> np.ndarray:
    """Per (treatment, dim) diagonal ridge. y_ref: line x treatment x d; x: line x d."""
    x_mean = x_ref.mean(axis=0)
    xc = x_ref - x_mean
    mu = y_ref.mean(axis=0)
    gamma = np.einsum("cj,ctj->tj", xc, y_ref - mu) / (np.sum(xc ** 2, axis=0) + lam)
    return mu[None] + gamma[None] * (x_query - x_mean)[:, None, :]


def log1p_dmso_rows(compact, hvg, indices: np.ndarray) -> np.ndarray:
    values, missing = condition_hvg(compact.cells, compact.plates, indices, hvg, len(compact.genes))
    if missing:
        raise ValueError(f"{missing} conditions lack a DMSO centroid")
    return np.log1p(values).astype(np.float64)


def split_analysis(args, split, compact, expression, cvcl_to_model, hvg):
    repo_root = args.project_root / "external" / "Rhaister"
    holdout_cells, test_treatments, _ = load_split(repo_root, split)
    holdout = set(holdout_cells)
    reference_all = sorted(set(compact.cells.tolist()) - holdout)
    has_x = lambda cell: cvcl_to_model.get(cell) in expression.index
    reference = [c for c in reference_all if has_x(c)]
    queries = [c for c in holdout_cells if has_x(c)]

    lookup = {(c, t): i for i, (c, t) in enumerate(zip(compact.cells, compact.treatments))}
    candidates = sorted({t for c in queries for t in test_treatments.get(c, set())})
    common = [t for t in candidates
              if all((c, t) in lookup for c in reference) and all((q, t) in lookup for q in queries)]

    x_ref = np.nan_to_num(expression.loc[[cvcl_to_model[c] for c in reference]].to_numpy(np.float64))
    x_q = np.nan_to_num(expression.loc[[cvcl_to_model[c] for c in queries]].to_numpy(np.float64))
    z_ref = line_hvg_vectors(compact, hvg, reference)
    grid = [(k, d) for k in args.k_grid for d in args.d_grid if d <= k]
    (k, d), _ = select_kd(x_ref, z_ref, grid)
    model = Cca(x_ref, z_ref, k, d)

    def project(cells):
        idx = np.array([lookup[(c, t)] for c in cells for t in common])
        delta = np.asarray(compact.delta[idx], np.float64)
        absolute = log1p_dmso_rows(compact, hvg, idx) + delta
        shape = (len(cells), len(common), d)
        return {
            "absolute": model.embed_y(absolute).reshape(shape),
            "delta": linear_y(model, delta).reshape(shape),
        }, delta

    ref_targets, _ = project(reference)
    q_truth, q_delta = project(queries)
    retained = roundtrip_pearson(model, q_delta).reshape(len(queries), len(common))

    v_ref = model.embed_y(z_ref)
    z_q = line_hvg_vectors(compact, hvg, queries)
    v_q = model.embed_y(z_q)
    u_q = model.embed_x(x_q)
    shift = np.roll(np.arange(len(queries)), 1)
    inputs = {
        "tahoe_dmso": v_q,
        "depmap": u_q,
        "depmap_shrunk": u_q * model.rho,
        "mean": np.tile(v_ref.mean(axis=0), (len(queries), 1)),
        "shuffled_depmap": u_q[shift],
    }

    rows = []
    for target in TARGETS:
        preds = {name: fit_predict(v_ref, ref_targets[target], x, args.ridge_lambda)
                 for name, x in inputs.items()}
        preds["drug_mean"] = np.broadcast_to(ref_targets[target].mean(axis=0), q_truth[target].shape)
        if target == "absolute":
            preds["identity_only_tahoe"] = np.broadcast_to(v_q[:, None, :], q_truth[target].shape)
            preds["identity_only_depmap"] = np.broadcast_to(u_q[:, None, :], q_truth[target].shape)
        for name, pred in preds.items():
            for i, cell in enumerate(queries):
                m = rowwise_metrics(q_truth[target][i], pred[i])
                rows.append({
                    "split": split, "cell_line": cell, "target": target, "arm": name,
                    "pearson": float(m["pearson"].mean()), "rmse": float(m["rmse"].mean()),
                    "n_conditions": len(common), "k": k, "d": d,
                    "delta_roundtrip_pearson": float(retained[i].mean()),
                })
    out = args.output_root / f"split_{split}"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    atomic_csv(frame, out / "per_line.csv")
    head = frame[frame["target"].eq("absolute")].groupby("arm")["pearson"].mean().round(3).to_dict()
    print(f"split {split}: refs={len(reference)} queries={len(queries)} common={len(common)} k={k} d={d} "
          f"roundtrip={retained.mean():.3f} absolute={head}", flush=True)
    return {"split": split, "k": k, "d": d, "n_common_treatments": len(common),
            "delta_roundtrip_pearson_mean": float(retained.mean())}


def main(args):
    repo_root = args.project_root / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    expression, cvcl_to_model, _, expression_path, _ = load_depmap_expression(args.depmap_root)
    hvg = load_hvg_centroids(args.data_root, compact.genes)
    per_split = [split_analysis(args, s, compact, expression, cvcl_to_model, hvg) for s in range(5, 10)]
    lines = pd.concat([pd.read_csv(args.output_root / f"split_{s}" / "per_line.csv") for s in range(5, 10)])
    atomic_csv(lines, args.output_root / "per_line.csv")

    summary = []
    for target in TARGETS:
        sub = lines[lines["target"].eq(target)]
        wide = sub.pivot_table(index=["split", "cell_line"], columns="arm", values="pearson")
        rmse = sub.pivot_table(index=["split", "cell_line"], columns="arm", values="rmse")
        for arm in wide.columns:
            mean, lo, hi = line_bootstrap_interval(wide[arm].to_numpy(), args.seed)
            summary.append({"target": target, "arm": arm, "pearson": mean, "low": lo, "high": hi,
                            "rmse": float(rmse[arm].mean())})
        contrasts = [("depmap", "tahoe_dmso"), ("depmap", "mean"), ("depmap", "shuffled_depmap"),
                     ("depmap_shrunk", "mean"), ("depmap", "drug_mean")]
        if target == "absolute":
            contrasts += [("depmap", "identity_only_depmap"), ("tahoe_dmso", "identity_only_tahoe")]
        for a, b in contrasts:
            mean, lo, hi = line_bootstrap_interval((wide[a] - wide[b]).to_numpy(), args.seed + 1)
            summary.append({"target": target, "arm": f"{a}-{b}", "pearson": mean, "low": lo, "high": hi,
                            "rmse": np.nan})
    summary = pd.DataFrame(summary)
    atomic_csv(summary, args.output_root / "summary.csv")
    write_json({
        "purpose": "Rhaister-O per canonical dimension inside the DepMap<->Tahoe CCA space",
        "depmap_expression": str(expression_path), "depmap_expression_sha256": sha256(expression_path),
        "ridge_lambda": args.ridge_lambda, "k_grid": args.k_grid, "d_grid": args.d_grid, "seed": args.seed,
        "runner_sha256": sha256(Path(__file__).resolve()), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "per_split": per_split,
    }, args.output_root / "summary.json")
    print("\n" + summary.to_string(index=False), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--depmap-root", type=Path, default=DEFAULT_DEPMAP_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--k-grid", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--d-grid", type=int, nargs="+", default=[2, 5, 10, 20])
    p.add_argument("--ridge-lambda", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
