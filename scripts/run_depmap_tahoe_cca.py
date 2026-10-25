#!/usr/bin/env python3
"""Does a held-out line share a representation across DepMap and Tahoe?

Exploratory side check (record 2026-10-08).  Per released split 5-9:

A. Identity alignment.  Fit CCA between DepMap 24Q4 log-TPM (X) and the Tahoe
   per-line mean log1p DMSO vector on the fixed 2,000 genes (Y), outer
   reference lines only.  For each held-out line, query with its DepMap
   embedding and rank its own Tahoe DMSO embedding among every line's.

B. Response prediction.  Rhaister-O is linear in the held-out line's centred
   DMSO vector z_c: pred = (1-b) additive + b (mu_y + gamma * z_c).  Keep the
   frozen reference fit and only replace z_c (real DMSO -> CCA-mapped DepMap,
   nearest reference line, mean, shuffled).  Score against Tahoe truth.

No treated response enters the alignment or its (k, d) selection.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from run_depmap_expression_ridge import (
    DEFAULT_COMPONENT_ROOT,
    DEFAULT_DEPMAP_ROOT,
    LEVELS,
    add_model_metrics,
    atomic_csv,
    line_bootstrap_interval,
    load_depmap_expression,
    response_cube,
    sha256,
    write_json,
)
from run_depmap_rhaister_correction import line_hvg_vectors
from run_rhaister_o_compact import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT as BASELINE_OUTPUT_ROOT,
    DEFAULT_PROJECT_ROOT,
    build_or_load_compact_cache,
    condition_hvg,
    load_hvg_centroids,
    load_split,
)

DEFAULT_OUTPUT_ROOT = Path("/nfs-stor/yuqi.lei/Therapeutic/results/depmap_tahoe_cca")
ARMS = ("real_dmso", "cca_mapped", "nearest_ref_line", "mean_dmso", "shuffled_cca",
        "direct_mapped", "shuffled_direct")


class Pca:
    """Reference-only standardise + whitened PCA with an inverse map."""

    def __init__(self, reference: np.ndarray, k: int):
        ref = np.asarray(reference, dtype=np.float64)
        self.mean = ref.mean(axis=0)
        scale = ref.std(axis=0)
        self.keep = scale > 1e-8
        self.scale = scale[self.keep]
        centred = (ref[:, self.keep] - self.mean[self.keep]) / self.scale
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        k = min(k, len(s) - 1)
        self.vt = vt[:k]
        self.sd = s[:k] / np.sqrt(len(ref))

    def transform(self, values: np.ndarray) -> np.ndarray:
        centred = (np.asarray(values, dtype=np.float64)[:, self.keep] - self.mean[self.keep]) / self.scale
        return centred @ self.vt.T / self.sd

    def inverse(self, scores: np.ndarray) -> np.ndarray:
        out = np.tile(self.mean, (len(scores), 1))
        out[:, self.keep] += (scores * self.sd) @ self.vt * self.scale
        return out


class Cca:
    def __init__(self, x_ref: np.ndarray, y_ref: np.ndarray, k: int, d: int):
        self.px, self.py = Pca(x_ref, k), Pca(y_ref, k)
        sx, sy = self.px.transform(x_ref), self.py.transform(y_ref)
        # Whitened PCA scores have identity covariance, so CCA is an SVD of Cxy.
        u, rho, vt = np.linalg.svd(sx.T @ sy / len(sx), full_matrices=False)
        d = min(d, len(rho))
        self.a, self.b, self.rho = u[:, :d], vt[:d].T, rho[:d]

    def embed_x(self, x: np.ndarray) -> np.ndarray:
        return self.px.transform(x) @ self.a

    def embed_y(self, y: np.ndarray) -> np.ndarray:
        return self.py.transform(y) @ self.b

    def map_x_to_y(self, x: np.ndarray) -> np.ndarray:
        return self.py.inverse((self.embed_x(x) * self.rho) @ self.b.T)


def select_kd(x_ref: np.ndarray, y_ref: np.ndarray, grid: list[tuple[int, int]]) -> tuple[tuple[int, int], pd.DataFrame]:
    """Inner leave-one-reference-line-out reconstruction error of Y from X."""
    y_sd = y_ref.std(axis=0) + 1e-8
    rows = []
    for k, d in grid:
        errors = []
        for i in range(len(x_ref)):
            mask = np.arange(len(x_ref)) != i
            model = Cca(x_ref[mask], y_ref[mask], k, d)
            errors.append(np.mean(((model.map_x_to_y(x_ref[i:i + 1])[0] - y_ref[i]) / y_sd) ** 2))
        rows.append({"k": k, "d": d, "loo_std_mse": float(np.mean(errors))})
    table = pd.DataFrame(rows)
    best = table.loc[table["loo_std_mse"].idxmin()]
    return (int(best["k"]), int(best["d"])), table


def cosine_ranks(query: np.ndarray, gallery: np.ndarray, own: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    g = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
    sim = q @ g.T
    own_sim = sim[np.arange(len(q)), own]
    return 1 + (sim > own_sim[:, None]).sum(axis=1)


def corr_ranks(query: np.ndarray, gallery: np.ndarray, own: np.ndarray) -> np.ndarray:
    def z(a):
        a = a - a.mean(axis=1, keepdims=True)
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    return cosine_ranks(z(query), z(gallery), own)


def gene_symbol(column: str) -> str:
    return re.sub(r"\s*\(\d+\)$", "", column)


def split_analysis(args, split, compact, expression, cvcl_to_model, hvg):
    repo_root = args.project_root / "external" / "Rhaister"
    holdout_cells, test_treatments, _ = load_split(repo_root, split)
    holdout = set(holdout_cells)
    reference_all = sorted(set(compact.cells.tolist()) - holdout)
    has_x = lambda cell: cvcl_to_model.get(cell) in expression.index
    reference = [c for c in reference_all if has_x(c)]
    queries = [c for c in holdout_cells if has_x(c)]

    with np.load(args.baseline_root / f"split_{split}" / "predictions.npz", allow_pickle=False) as arc:
        cells, treatments = arc["cell_line"].copy(), arc["treatment"].copy()
        truth = arc["truth"].astype(np.float32)
        drug_mean = arc["perturbation_mean"].astype(np.float32)
        additive = arc["additive_als"].astype(np.float64)
        rhaister = arc["rhaister_o"].astype(np.float64)
    with np.load(args.component_root / f"split_{split}" / "train_residual_pc1.npz", allow_pickle=False) as arc:
        pc_mean, pc1 = arc["mean"].astype(np.float32), arc["component"].astype(np.float32)

    # --- Frozen Rhaister-O pieces (reference_all, exactly as diagonal_predictions).
    cube, common = response_cube(compact, reference_all, sorted(set(treatments.tolist())))
    cube = cube.astype(np.float64)
    z_ref_raw = line_hvg_vectors(compact, hvg, reference_all)
    z_mean = z_ref_raw.mean(axis=0)
    z_ref_c = z_ref_raw - z_mean
    mu_y = cube.mean(axis=0)
    gamma = np.einsum("cg,ctg->tg", z_ref_c, cube - mu_y, optimize=True) / (
        np.sum(z_ref_c ** 2, axis=0) + args.rhaister_lambda
    )
    z_hold_raw = {}
    for cell in holdout_cells:
        idx = np.flatnonzero([
            c == cell and t in test_treatments.get(cell, set())
            for c, t in zip(compact.cells, compact.treatments)
        ])
        values, _ = condition_hvg(compact.cells, compact.plates, idx, hvg, len(compact.genes))
        z_hold_raw[cell] = np.log1p(values).mean(axis=0, dtype=np.float32).astype(np.float64)

    common_index = {t: i for i, t in enumerate(common)}
    covered = np.array([t in common_index for t in treatments])
    t_idx = np.array([common_index.get(t, 0) for t in treatments])

    def predict(z_raw_by_cell: dict) -> np.ndarray:
        out = rhaister.copy()
        for cell, z_raw in z_raw_by_cell.items():
            rows = np.flatnonzero((cells == cell) & covered)
            out[rows] = (1 - args.blend) * additive[rows] + args.blend * (
                mu_y[t_idx[rows]] + gamma[t_idx[rows]] * (z_raw - z_mean)[None, :]
            )
        return out

    rebuilt = predict(z_hold_raw)
    rebuild_max_abs = float(np.abs(rebuilt - rhaister).max())
    if rebuild_max_abs > 1e-4:
        raise ValueError(f"split {split}: Rhaister-O rebuild mismatch {rebuild_max_abs:.2e}")

    # --- Alignment on reference lines with DepMap only.
    x_ref = expression.loc[[cvcl_to_model[c] for c in reference]].to_numpy(np.float64)
    x_q = expression.loc[[cvcl_to_model[c] for c in queries]].to_numpy(np.float64)
    x_ref = np.nan_to_num(x_ref, nan=0.0)
    x_q = np.nan_to_num(x_q, nan=0.0)
    y_ref = z_ref_raw[[reference_all.index(c) for c in reference]]
    y_q = np.stack([z_hold_raw[c] for c in queries])
    grid = [(k, d) for k in args.k_grid for d in args.d_grid if d <= k]
    (k, d), selection = select_kd(x_ref, y_ref, grid)
    model = Cca(x_ref, y_ref, k, d)

    # A. identity retrieval: gallery = every line's Tahoe DMSO (references + queries).
    gallery_cells = reference + queries
    gallery_y = np.vstack([y_ref, y_q])
    own = np.arange(len(reference), len(gallery_cells))
    rank_cca = cosine_ranks(model.embed_x(x_q), model.embed_y(gallery_y), own)
    oos_rho = [
        float(np.corrcoef(model.embed_x(x_q)[:, j], model.embed_y(y_q)[:, j])[0, 1])
        for j in range(min(3, d))
    ] if len(queries) > 2 else []
    # No-CCA control: gene-matched correlation, each dataset standardised on references.
    symbols = np.array([gene_symbol(c) for c in expression.columns])
    tahoe_genes = list(compact.genes)
    shared = [g for g in tahoe_genes if g in set(symbols)]
    xcol = {s: i for i, s in enumerate(symbols)}
    xi = np.array([xcol[g] for g in shared])
    yi = np.array([tahoe_genes.index(g) for g in shared])
    # Drop genes constant across references: dividing by ~0 sd gave |z| up to 4e7
    # and broke the first run (job 308584, record 2026-10-08 correction).
    variable = (x_ref[:, xi].std(0) > 0) & (y_ref[:, yi].std(0) > 0)
    def stdz(ref, val):
        return ((val - ref.mean(0)) / ref.std(0).clip(min=1e-12))[:, variable]
    rank_direct = corr_ranks(
        stdz(x_ref[:, xi], x_q[:, xi]), stdz(y_ref[:, yi], gallery_y[:, yi]), own
    )
    # Null: permuted reference pairing, same (k, d).
    rng = np.random.default_rng(args.seed + split)
    null_ranks = []
    for p in range(args.n_permutations):
        perm = rng.permutation(len(reference))
        null_model = Cca(x_ref, y_ref[perm], k, d)
        null_ranks.append(cosine_ranks(null_model.embed_x(x_q), null_model.embed_y(gallery_y), own))
    null_ranks = np.array(null_ranks)

    # B. response prediction arms.
    y_map = model.map_x_to_y(x_q)
    # Direct per-gene bridge (record 2026-10-08): OLS tahoe_g ~ depmap_g on reference lines.
    gx, gy = xi[variable], yi[variable]
    xr, yr = x_ref[:, gx], y_ref[:, gy]
    slope = ((xr - xr.mean(0)) * (yr - yr.mean(0))).sum(0) / ((xr - xr.mean(0)) ** 2).sum(0)
    y_direct = np.tile(y_ref.mean(axis=0), (len(queries), 1))
    y_direct[:, gy] = yr.mean(0) + slope * (x_q[:, gx] - xr.mean(0))
    px = Pca(x_ref, len(reference))
    sxr, sxq = px.transform(x_ref), px.transform(x_q)
    nn = [int(np.argmax(sxr @ s / (np.linalg.norm(sxr, axis=1) * np.linalg.norm(s) + 1e-12))) for s in sxq]
    shift = np.roll(np.arange(len(queries)), 1)
    arms = {
        "real_dmso": {c: z_hold_raw[c] for c in queries},
        "cca_mapped": {c: y_map[i] for i, c in enumerate(queries)},
        "nearest_ref_line": {c: y_ref[nn[i]] for i, c in enumerate(queries)},
        "mean_dmso": {c: z_mean for c in queries},
        "shuffled_cca": {c: y_map[shift[i]] for i, c in enumerate(queries)},
        "direct_mapped": {c: y_direct[i] for i, c in enumerate(queries)},
        "shuffled_direct": {c: y_direct[shift[i]] for i, c in enumerate(queries)},
    }
    frame = pd.DataFrame({"split": split, "cell_line": cells, "treatment": treatments})
    for arm, zs in arms.items():
        add_model_metrics(frame, arm, truth, drug_mean, predict(zs).astype(np.float32), pc_mean, pc1)
    frame = frame[frame["cell_line"].isin(queries)]

    line_rows = []
    for i, cell in enumerate(queries):
        sub = frame[frame["cell_line"].eq(cell)]
        rec = {
            "split": split, "cell_line": cell, "k": k, "d": d,
            "rank_cca": int(rank_cca[i]), "rank_direct": int(rank_direct[i]),
            "rank_null_mean": float(null_ranks[:, i].mean()),
            "gallery_size": len(gallery_cells),
            "nearest_ref_line": reference[nn[i]],
            "dmso_recon_pearson": float(np.corrcoef(y_map[i] - z_mean, z_hold_raw[cell] - z_mean)[0, 1]),
            "direct_recon_pearson": float(np.corrcoef(y_direct[i] - z_mean, z_hold_raw[cell] - z_mean)[0, 1]),
        }
        for arm in ARMS:
            for level in LEVELS:
                rec[f"{arm}__{level}"] = float(sub[f"{arm}__{level}__pearson"].mean())
        line_rows.append(rec)

    out = args.output_root / f"split_{split}"
    out.mkdir(parents=True, exist_ok=True)
    atomic_csv(selection, out / "kd_selection.csv")
    atomic_csv(pd.DataFrame(line_rows), out / "per_line.csv")
    print(
        f"split {split}: refs={len(reference)}/{len(reference_all)} queries={len(queries)}/{len(holdout_cells)} "
        f"k={k} d={d} rho_in={np.round(model.rho[:3], 3).tolist()} rho_oos={np.round(oos_rho, 3).tolist()} "
        f"rank_cca={rank_cca.tolist()} rank_direct={rank_direct.tolist()} rebuild_err={rebuild_max_abs:.1e} "
        f"shared_genes={len(shared)} direct_genes={int(variable.sum())}",
        flush=True,
    )
    return {
        "split": split, "k": k, "d": d, "rho_in_sample": model.rho.tolist(), "rho_out_of_sample_first3": oos_rho,
        "rebuild_max_abs": rebuild_max_abs, "n_shared_genes_direct": len(shared),
        "null_rank_mean_per_permutation": null_ranks.mean(axis=1).tolist(),
        "missing_depmap": sorted(set(reference_all + holdout_cells) - set(reference + queries)),
    }


def main(args):
    repo_root = args.project_root / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    expression, cvcl_to_model, _, expression_path, _ = load_depmap_expression(args.depmap_root)
    hvg = load_hvg_centroids(args.data_root, compact.genes)
    per_split = [split_analysis(args, s, compact, expression, cvcl_to_model, hvg) for s in range(5, 10)]
    lines = pd.concat([pd.read_csv(args.output_root / f"split_{s}" / "per_line.csv") for s in range(5, 10)])
    atomic_csv(lines, args.output_root / "per_line.csv")

    n = lines["gallery_size"].iloc[0]
    null_line_means = np.array([np.mean(s["null_rank_mean_per_permutation"]) for s in per_split])
    identity = {
        "n_lines": int(len(lines)),
        "gallery_size": int(n),
        "chance_mean_rank": (n + 1) / 2,
        "cca_top1": float((lines["rank_cca"] == 1).mean()),
        "cca_top5": float((lines["rank_cca"] <= 5).mean()),
        "cca_mean_rank": float(lines["rank_cca"].mean()),
        "direct_top1": float((lines["rank_direct"] == 1).mean()),
        "direct_top5": float((lines["rank_direct"] <= 5).mean()),
        "direct_mean_rank": float(lines["rank_direct"].mean()),
        "permuted_pairing_mean_rank": float(null_line_means.mean()),
        "dmso_recon_pearson_mean": float(lines["dmso_recon_pearson"].mean()),
        "direct_recon_pearson_mean": float(lines["direct_recon_pearson"].mean()),
    }
    response = []
    for level in LEVELS:
        for arm in ARMS:
            m, lo, hi = line_bootstrap_interval(lines[f"{arm}__{level}"].to_numpy(), args.seed)
            response.append({"level": level, "arm": arm, "line_macro": m, "low": lo, "high": hi})
        contrasts = [
            ("cca_mapped", "real_dmso"), ("cca_mapped", "mean_dmso"), ("cca_mapped", "shuffled_cca"),
            ("nearest_ref_line", "real_dmso"), ("nearest_ref_line", "mean_dmso"),
            ("direct_mapped", "real_dmso"), ("direct_mapped", "mean_dmso"),
            ("direct_mapped", "shuffled_direct"), ("direct_mapped", "cca_mapped"),
        ]
        for arm, ref in contrasts:
            diff = (lines[f"{arm}__{level}"] - lines[f"{ref}__{level}"]).to_numpy()
            m, lo, hi = line_bootstrap_interval(diff, args.seed + 1)
            response.append({"level": level, "arm": f"{arm}-{ref}", "line_macro": m, "low": lo, "high": hi})
        gain_real = lines[f"real_dmso__{level}"] - lines[f"mean_dmso__{level}"]
        for arm, tag in (("cca_mapped", "cca"), ("direct_mapped", "direct")):
            gain = lines[f"{arm}__{level}"] - lines[f"mean_dmso__{level}"]
            response.append({
                "level": level, "arm": f"fraction_of_real_gain_recovered_by_{tag}",
                "line_macro": float(gain.mean() / gain_real.mean()), "low": np.nan, "high": np.nan,
            })
    response = pd.DataFrame(response)
    atomic_csv(response, args.output_root / "response_summary.csv")
    write_json({
        "purpose": "DepMap<->Tahoe CCA identity alignment and DMSO substitution in frozen Rhaister-O",
        "depmap_expression": str(expression_path), "depmap_expression_sha256": sha256(expression_path),
        "k_grid": args.k_grid, "d_grid": args.d_grid, "n_permutations": args.n_permutations,
        "seed": args.seed, "runner_sha256": sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "identity": identity, "per_split": per_split,
    }, args.output_root / "summary.json")
    print("\nIdentity alignment:", identity, flush=True)
    print("\nResponse prediction (line-macro Pearson):\n" + response.to_string(index=False), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--baseline-root", type=Path, default=BASELINE_OUTPUT_ROOT)
    p.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    p.add_argument("--depmap-root", type=Path, default=DEFAULT_DEPMAP_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--k-grid", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--d-grid", type=int, nargs="+", default=[2, 5, 10, 20])
    p.add_argument("--rhaister-lambda", type=float, default=1.0)
    p.add_argument("--blend", type=float, default=0.95)
    p.add_argument("--n-permutations", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
