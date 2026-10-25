#!/usr/bin/env python3
"""Cell-context slopes: can Rhaister-O's slope depend on the cell line?

Rhaister-O predicts

    delta(c*, t, g) = mu(t, g) + gamma(t, g) * z(c*, g)

where mu and gamma are estimated once from the reference lines and are the same
for every query line; the query line enters only through its own gene's
centred log1p DMSO value z(c*, g), linearly. Pre-registered 2026-09-29 in
report/10_02_record.md (E1). Six arms relax that structure in different ways;
each has an anchor parameter at which it IS Rhaister-O, the anchor is always in
the inner-selection grid, and the outer prediction is written as a delta on the
frozen Rhaister-O array:

    prediction = frozen + blend * (core_arm - core_rho)

with both cores computed from the same z, so the anchor reproduces the frozen
estimator exactly (asserted).

  kernel        mu and gamma re-estimated with reference weights
                exp(-d(c*, c) / (h * s)), d = correlation distance between
                centred DMSO profiles, s = median reference-reference distance.
                Anchor h = inf. Slopes become local to the query's neighbourhood.
  varying_coef  gamma(t, g, c) = gamma0 + sum_k u_ck G_k(t, g); u = top-K PCs of
                the reference DMSO profiles (unit variance), ridge lambda on G.
                Anchor lambda = inf. Slopes vary smoothly with cell state.
  quadratic     adds gamma2(t, g) * (z^2 - mean z^2). Anchor lambda = inf.
  kg_neighbor   adds eta(t, g) * zbar_N(g), the mean centred baseline of g's
                Reactome pathway neighbours (pathways of 5-200 genes). A fixed,
                external gene-side prior. Anchor lambda = inf.
  kg_random     same, on a degree-preserving relabelling of the Reactome graph:
                the negative control that kg_neighbor must beat.
  reliability   reference weights q_c^alpha, q = the line's own DMSO
                reproducibility (mean correlation among its per-plate
                centroids). Anchor alpha = 0.

Inner selection: leave-one-reference-line-out drug-residual Pearson, features
re-centred inside every fold. Gate: run_coefficient_sharing.GATE plus
gain_point_estimate_positive.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_rhaister_signal import parse_treatment
from run_coefficient_sharing import GATE, line_macro_pearson
from run_depmap_expression_ridge import (
    add_model_metrics,
    atomic_csv,
    line_bootstrap_interval,
    response_cube,
    sha256,
    write_json,
)
from run_depmap_rhaister_correction import line_hvg_vectors
import run_rhaister_o_compact as roc
from run_reference_count_sweep import line_quality

LEVELS = ("full", "drug_residual", "train_pca1_removed")
ARMS = ("kernel", "varying_coef", "quadratic", "kg_neighbor", "kg_random", "reliability")
# Exploratory, added after the pre-registered E1 results (not gate tests):
EXPLORATORY_ARMS = ("kernel_reliability",)
INF = float("inf")


# --------------------------------------------------------------------------
# gene graph
# --------------------------------------------------------------------------

def reactome_adjacency(gmt: Path, genes: list[str], min_size: int, max_size: int) -> tuple[np.ndarray, dict]:
    index = {g: i for i, g in enumerate(genes)}
    A = np.zeros((len(genes), len(genes)), dtype=np.float64)
    used = 0
    for line in open(gmt):
        parts = line.rstrip("\n").split("\t")
        members = parts[2:]
        if not (min_size <= len(members) <= max_size):
            continue
        idx = sorted({index[g] for g in members if g in index})
        if len(idx) < 2:
            continue
        used += 1
        A[np.ix_(idx, idx)] += 1.0
    np.fill_diagonal(A, 0.0)
    degree = (A > 0).sum(1)
    info = {"pathways_used": used, "genes_with_neighbours": int((degree > 0).sum()),
            "median_degree": float(np.median(degree)), "min_size": min_size, "max_size": max_size}
    return A, info


def row_normalise(A: np.ndarray) -> np.ndarray:
    s = A.sum(1, keepdims=True)
    return np.divide(A, s, out=np.zeros_like(A), where=s > 0)


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

def weighted_core(Z: np.ndarray, R: np.ndarray, zq: np.ndarray, w: np.ndarray | None, lam0: float) -> np.ndarray:
    """Rhaister-O's diagonal estimator with optional reference weights.

    Weights are normalised to mean 1, so w = 1 reproduces the unweighted
    estimator bit for bit and lam0 keeps its meaning.
    """
    if w is None:
        mu = R.mean(axis=0)
        resid = R - mu[None]
        gamma = np.einsum("cg,ctg->tg", Z, resid, optimize=True) / (np.sum(Z * Z, 0) + lam0)[None]
    else:
        w = w / w.mean()
        mu = np.einsum("c,ctg->tg", w, R, optimize=True) / w.sum()
        resid = R - mu[None]
        gamma = np.einsum("c,cg,ctg->tg", w, Z, resid, optimize=True) / (np.einsum("c,cg->g", w, Z * Z) + lam0)[None]
    return mu + gamma * zq[None]


def augmented_fit(Z: np.ndarray, F: np.ndarray, Rt: np.ndarray, mu: np.ndarray,
                  lam0: float, lambdas: list[float], sizes: list[int]) -> dict:
    """Per-gene ridge on [z, F_1..F_m]: penalty lam0 on z, lam on the F columns.

    Z (C, G) and F (C, G, m) are centred over the training lines, so X'(R - mu)
    = X'R and the per-treatment mean needs no separate handling. Rt is the
    response as (G, C, T). Returns {(size, lam): coef (G, p, T)}; size takes the
    first `size` F columns (nested, one Gram matrix serves every K). lam = inf
    keeps only z, which is exactly Rhaister-O's slope.
    """
    X = np.concatenate([Z[:, :, None], F], axis=2).transpose(1, 2, 0)   # (G, p, C)
    XtX = X @ X.transpose(0, 2, 1)                                        # (G, p, p)
    XtY = X @ Rt                                                          # (G, p, T)
    out = {}
    for size in sizes:
        p = size + 1
        for lam in lambdas:
            if lam == INF:
                coef = np.zeros((X.shape[0], p, XtY.shape[2]))
                coef[:, 0, :] = XtY[:, 0, :] / (XtX[:, 0, 0] + lam0)[:, None]
                out[(size, lam)] = coef
                continue
            A = XtX[:, :p, :p].copy()
            A[:, 0, 0] += lam0
            idx = np.arange(1, p)
            A[:, idx, idx] += lam
            out[(size, lam)] = np.linalg.solve(A, XtY[:, :p, :])
    return out


def extra_features(arm: str, Z: np.ndarray, kmax: int, graph: dict):
    """Training features F (C, G, m) and a function mapping a query zq to fq (G, m),
    both centred on the training lines and scaled to the per-gene variance of z."""
    if arm == "varying_coef":
        u, s, vt = np.linalg.svd(Z, full_matrices=False)
        k = min(kmax, len(s) - 1)
        scores = u[:, :k] * s[None, :k]
        sd = scores.std(axis=0)
        F = Z[:, :, None] * (scores / sd[None])[:, None, :]
        raw_q = lambda zq: zq[:, None] * ((zq @ vt[:k].T) / sd)[None, :]
    elif arm == "quadratic":
        F = (Z * Z)[:, :, None]
        raw_q = lambda zq: (zq * zq)[:, None]
    elif arm in ("kg_neighbor", "kg_random"):
        W = graph[arm]
        F = (Z @ W.T)[:, :, None]
        raw_q = lambda zq: (W @ zq)[:, None]
    else:
        raise ValueError(arm)
    m = F.mean(axis=0, keepdims=True)
    F = F - m
    zs = Z.std(axis=0)[:, None]
    fs = F.std(axis=0)
    scale = np.divide(zs, fs, out=np.zeros_like(fs), where=fs > 1e-12)
    return F * scale[None], (lambda zq: (raw_q(zq) - m[0]) * scale)


def feature_distance_matrix(path: Path, cells: list[str]) -> tuple[np.ndarray, dict]:
    """Correlation distance between cell lines in an external feature space.

    Rows/columns follow `cells`; lines absent from the feature table get NaN, and
    the caller falls back to the DMSO distance for them. Features are already
    standardised by build_cell_feature_spaces.py.
    """
    frame = pd.read_parquet(path)
    present = [c for c in cells if c in frame.index]
    X = frame.loc[present].to_numpy(dtype=np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norm, 1e-30)
    D = np.full((len(cells), len(cells)), np.nan)
    idx = [cells.index(c) for c in present]
    D[np.ix_(idx, idx)] = 1.0 - X @ X.T
    return D, {"feature_space": str(path), "n_features": int(frame.shape[1]),
               "lines_with_features": len(present),
               "lines_without_features": sorted(set(cells) - set(present))}


def _dist_block(D: np.ndarray, query_idx: list[int], ref_idx: list[int],
                fallback) -> dict:
    """Rows of query-to-reference distances, with NaNs repaired.

    A reference line missing the modality takes the median distance of that row
    (a neutral weight). A query line missing it falls back entirely to the DMSO
    distance, so the cohort of scored lines never changes; the count is recorded.
    """
    rows, fell_back = {}, []
    sub = D[np.ix_(ref_idx, ref_idx)]
    pair = sub[~np.eye(len(ref_idx), dtype=bool)]
    pair_median = float(np.nanmedian(pair)) if np.isfinite(pair).any() else 1.0
    for k, qi in enumerate(query_idx):
        row = D[qi, ref_idx].copy()
        if not np.isfinite(row).any():
            rows[k] = None            # signal: use the DMSO distance
            fell_back.append(k)
            continue
        row[~np.isfinite(row)] = np.nanmedian(row)
        rows[k] = row
    return {"rows": rows, "pair_median": pair_median, "n_fallback": len(fell_back)}


def corr_distance(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    a = a - a.mean()
    B = B - B.mean(axis=1, keepdims=True)
    r = (B @ a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a) + 1e-30)
    return 1.0 - r


class ArmModel:
    """Fit an arm once on (Z, R); predict cores (T, G) for any query profile."""

    def __init__(self, arm, grid, Z, R, quality, lam0, graph, Rt=None, dist=None):
        self.arm, self.grid, self.Z, self.R, self.lam0 = arm, list(grid), Z, R, lam0
        self.quality = quality
        self.mu = R.mean(axis=0)
        # dist: optional (n_queries_or_1, C) precomputed distances from an
        # external cell-feature space (F1). When absent the kernel uses the
        # DMSO correlation distance, i.e. the confirmed C1/C2 arm.
        self.dist = dist
        if arm in ("kernel", "kernel_reliability"):
            if dist is not None:
                self.scale = float(dist["pair_median"])
            else:
                pair = np.array([corr_distance(Z[i], np.delete(Z, i, 0)) for i in range(len(Z))])
                self.scale = float(np.median(pair))
        elif arm == "reliability":
            pass
        else:
            Rt = R.transpose(2, 0, 1) if Rt is None else Rt
            if arm == "varying_coef":
                self.pairs = [(k, l) for k, l in self.grid]
                kmax = max(k for k, _ in self.pairs)
            else:
                self.pairs = [(1, l) for l in self.grid]
                kmax = 1
            F, self.fq = extra_features(arm, Z, kmax, graph)
            m = F.shape[2]
            sizes = sorted({min(k, m) for k, _ in self.pairs})
            lams = sorted({l for _, l in self.pairs})
            self.coef = augmented_fit(Z, F, Rt, self.mu, lam0, lams, sizes)
            self.m = m

    def _distance(self, zq, key):
        if self.dist is not None:
            row = self.dist["rows"].get(key)
            if row is not None:
                return row
        return corr_distance(zq, self.Z)

    def cores(self, zq: np.ndarray, params=None, key=None) -> dict:
        params = self.grid if params is None else params
        out = {}
        if self.arm == "kernel":
            d = self._distance(zq, key)
            for h in params:
                w = None if h == INF else np.exp(-d / (h * self.scale))
                out[h] = weighted_core(self.Z, self.R, zq, w, self.lam0)
            return out
        if self.arm == "kernel_reliability":
            d = self._distance(zq, key)
            for h, a in params:
                w = np.ones(len(self.Z)) if h == INF else np.exp(-d / (h * self.scale))
                if a != 0:
                    w = w * self.quality ** a
                out[(h, a)] = weighted_core(self.Z, self.R, zq, None if (h == INF and a == 0) else w, self.lam0)
            return out
        if self.arm == "reliability":
            for a in params:
                w = None if a == 0 else self.quality ** a
                out[a] = weighted_core(self.Z, self.R, zq, w, self.lam0)
            return out
        xq = np.concatenate([zq[:, None], self.fq(zq)], axis=1)          # (G, 1 + m)
        for prm in params:
            k, l = prm if self.arm == "varying_coef" else (1, prm)
            coef = self.coef[(min(k, self.m), l)]
            p = coef.shape[1]
            out[prm] = self.mu + np.einsum("gp,gpt->tg", xq[:, :p], coef, optimize=True)
        return out


def anchor_of(arm: str):
    return {"kernel": INF, "reliability": 0.0}.get(arm, INF if arm != "varying_coef" else None)


def parse_fixed(args) -> dict:
    """--fixed-params kernel=0.5 reliability=20 : skip inner selection entirely.

    Pre-registered C1 confirmation mode. The grid becomes [anchor, fixed] but no
    selection is run: the fixed value is used, so the reported gain is not a
    maximum over a grid.
    """
    out = {}
    for item in args.fixed_params or []:
        arm, _, value = item.partition("=")
        # ":" as well as "," because sbatch --export is itself comma-separated
        # and silently truncates the value at the first comma.
        value = value.replace(":", ",")
        if arm == "varying_coef":
            k, _, lam = value.partition(",")
            out[arm] = (int(k), float(lam))
        elif arm == "kernel_reliability":
            h, _, alpha = value.partition(",")
            if not alpha:
                raise SystemExit(f"--fixed-params {item}: expected two values, e.g. "
                                 f"{arm}=0.5:20 (use ':' under sbatch --export)")
            out[arm] = (float(h), float(alpha))
        else:
            out[arm] = float(value)
    return out


def grids(args) -> dict:
    return {
        "kernel": [INF] + list(args.kernel_grid),
        "varying_coef": [(k, INF) for k in args.k_grid[:1]] + [(k, l) for k in args.k_grid for l in args.lambda_grid],
        "quadratic": [INF] + list(args.lambda_grid),
        "kg_neighbor": [INF] + list(args.lambda_grid),
        "kg_random": [INF] + list(args.lambda_grid),
        "reliability": [0.0] + list(args.alpha_grid),
        "kernel_reliability": [(INF, 0.0)] + [(h, a) for h in [INF] + list(args.kernel_grid)
                                              for a in [0.0] + list(args.alpha_grid) if (h, a) != (INF, 0.0)],
    }


def is_anchor(arm: str, param) -> bool:
    if arm == "kernel_reliability":
        return param == (INF, 0.0)
    if arm == "varying_coef":
        return param[1] == INF
    return param == anchor_of(arm)


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------

def split_analysis(args, split, compact, hvg, graph, arms, grid_map, fixed=None, Dfull=None, all_cells=None):
    fixed = fixed or {}
    repo_root = args.project_root / "external" / "Rhaister"
    holdout_cells, _, split_path = roc.load_split(repo_root, split)
    reference_cells = sorted(set(compact.cells.tolist()) - set(holdout_cells))
    with np.load(args.baseline_root / f"split_{split}" / "predictions.npz", allow_pickle=False) as a:
        cells, treatments = a["cell_line"].copy(), a["treatment"].copy()
        truth = a["truth"].astype(np.float32)
        drug_mean = a["perturbation_mean"].astype(np.float32)
        rhaister = a["rhaister_o"].astype(np.float32)
    with np.load(args.component_root / f"split_{split}" / "train_residual_pc1.npz", allow_pickle=False) as c:
        pc_mean, pc1 = c["mean"].astype(np.float32), c["component"].astype(np.float32)

    common = response_cube(compact, reference_cells, sorted(set(treatments.tolist())))
    R, common_treatments = common[0].astype(np.float64), common[1]
    H_ref = line_hvg_vectors(compact, hvg, reference_cells)
    H_q = line_hvg_vectors(compact, hvg, holdout_cells)
    quality_all = line_quality(hvg, reference_cells, len(compact.genes))
    if np.isnan(quality_all).any():
        raise RuntimeError(f"split {split}: undefined line quality (single plate?); "
                           "run EmeraldBay with EB_HALF_PLATES=1")
    lam0, blend = args.rhaister_lambda, args.rhaister_blend
    C = len(reference_cells)
    Rt = np.ascontiguousarray(R.transpose(2, 0, 1))
    ref_idx = [all_cells.index(c) for c in reference_cells] if Dfull is not None else None
    dist_outer = None
    if Dfull is not None:
        q_idx = [all_cells.index(c) for c in holdout_cells]
        dist_outer = _dist_block(Dfull, q_idx, ref_idx, None)
        print(f"  split {split}: feature-space distances, {dist_outer['n_fallback']}/"
              f"{len(holdout_cells)} query lines fall back to the DMSO distance", flush=True)
    t0 = time.time()

    # ---- inner leave-one-reference-line-out ----
    scores = {arm: np.zeros(len(grid_map[arm])) for arm in arms}
    inner_arms = [a for a in arms if a not in fixed]
    for held in range(C) if inner_arms else []:
        keep = np.arange(C) != held
        m = H_ref[keep].mean(axis=0)
        Z, zh = H_ref[keep] - m, H_ref[held] - m
        Rk, truth_h = R[keep], R[held]
        Rt_k = np.ascontiguousarray(Rt[:, keep, :])
        mu_fold = Rk.mean(axis=0)
        dist_in = None
        if Dfull is not None:
            dist_in = _dist_block(Dfull, [ref_idx[held]], [ref_idx[i] for i in np.flatnonzero(keep)], None)
        for arm in inner_arms:
            cores = ArmModel(arm, grid_map[arm], Z, Rk, quality_all[keep], lam0, graph, Rt_k,
                             dist_in).cores(zh, key=0)
            for i, p in enumerate(grid_map[arm]):
                pred = (1 - blend) * mu_fold + blend * cores[p]
                scores[arm][i] += line_macro_pearson(truth_h, pred, mu_fold)
    for arm in inner_arms:
        scores[arm] /= C
    if inner_arms:
        print(f"  split {split}: inner LOO done in {time.time() - t0:.0f}s", flush=True)
    if fixed:
        print(f"  split {split}: fixed params, no inner selection: {fixed}", flush=True)

    # ---- outer ----
    m = H_ref.mean(axis=0)
    Z, Zq = H_ref - m, H_q - m
    q_index = {c: i for i, c in enumerate(holdout_cells)}
    t_index = {t: i for i, t in enumerate(common_treatments)}
    rows_of = {c: np.flatnonzero(cells == c) for c in holdout_cells}
    predictions, infos, selection_rows = {}, {}, []
    rng = np.random.default_rng(args.seed + 1000 * split)
    perms = [rng.permutation(C) for _ in range(args.n_permutations)]
    truth_res = {c: truth[rows_of[c]] - drug_mean[rows_of[c]] for c in holdout_cells}
    frozen_res = {c: rhaister[rows_of[c]] - drug_mean[rows_of[c]] for c in holdout_cells}
    base_line = {c: float(roc.rowwise_metrics(truth_res[c], frozen_res[c])["pearson"].mean()) for c in holdout_cells}

    def apply(delta_by_cell: dict, target: np.ndarray | None = None):
        out = np.array(rhaister, dtype=np.float32, copy=True) if target is None else target
        for c, rows in rows_of.items():
            for r in rows:
                t = treatments[r]
                if t in t_index:
                    out[r] = rhaister[r] + delta_by_cell[c][t_index[t]]
        return out

    for arm in arms:
        g = grid_map[arm]
        if arm in fixed:
            param = fixed[arm]
            if param not in g:
                g = list(g) + [param]
                grid_map[arm] = g
                scores[arm] = np.append(scores[arm], np.nan)
            best = g.index(param)
        else:
            best = int(np.argmax(scores[arm]))
            param = g[best]
        anchor_idx = [i for i, p in enumerate(g) if is_anchor(arm, p)][0]
        for i, p in enumerate(g):
            selection_rows.append({"split": split, "arm": arm, "param": str(p),
                                   "loo_drug_residual_pearson": float(scores[arm][i]),
                                   "selected": i == best, "anchor": i == anchor_idx,
                                   "fixed": arm in fixed})
        deltas, anchor_err = {}, 0.0
        model = ArmModel(arm, g, Z, R, quality_all, lam0, graph, Rt, dist_outer)
        for c, i in q_index.items():
            cores = model.cores(Zq[i], [param, g[anchor_idx]], key=i)
            rho = weighted_core(Z, R, Zq[i], None, lam0)
            anchor_err = max(anchor_err, float(np.abs(cores[g[anchor_idx]] - rho).max()))
            deltas[c] = (blend * (cores[param] - rho)).astype(np.float32)
        if anchor_err > 1e-9:
            raise RuntimeError(f"split {split} {arm}: anchor does not reproduce Rhaister-O ({anchor_err:g})")
        predictions[arm] = apply(deltas)

        null = []
        if not is_anchor(arm, param):
            sub = [param] if arm != "varying_coef" else [param, g[anchor_idx]]
            for order in perms:
                Rp = R[order]
                model_p = ArmModel(arm, sub if arm not in ("kernel", "reliability", "kernel_reliability") else [param],
                                   Z, Rp, quality_all, lam0, graph, np.ascontiguousarray(Rt[:, order, :]),
                                   dist_outer)
                gains = []
                for c, i in q_index.items():
                    core_p = model_p.cores(Zq[i], [param], key=i)[param]
                    rho_p = weighted_core(Z, Rp, Zq[i], None, lam0)
                    d = (blend * (core_p - rho_p)).astype(np.float32)
                    pert = frozen_res[c].copy()
                    for local, r in enumerate(rows_of[c]):
                        if treatments[r] in t_index:
                            pert[local] += d[t_index[treatments[r]]]
                    gains.append(float(roc.rowwise_metrics(truth_res[c], pert)["pearson"].mean()) - base_line[c])
                null.append(float(np.mean(gains)))
        infos[arm] = {"selected_param": str(param), "param_was_fixed": arm in fixed,
                      "selected_is_anchor": is_anchor(arm, param),
                      "selected_loo": float(scores[arm][best]), "anchor_loo": float(scores[arm][anchor_idx]),
                      "anchor_max_abs_error": anchor_err, "permutation_gains": null}
        print(f"  split {split} {arm:13s} {'fixed at' if arm in fixed else 'selected'} {param} "
              f"(LOO {scores[arm][best]:.6f} vs anchor {scores[arm][anchor_idx]:.6f})", flush=True)

    parsed = [parse_treatment(t) for t in treatments]
    frame = pd.DataFrame({"split": split, "cell_line": cells, "treatment": treatments,
                          "drug": [p[0] for p in parsed], "dose": [p[1] for p in parsed]})
    add_model_metrics(frame, "perturbation_mean", truth, drug_mean, drug_mean, pc_mean, pc1)
    add_model_metrics(frame, "rhaister_o", truth, drug_mean, rhaister, pc_mean, pc1)
    for arm in arms:
        add_model_metrics(frame, arm, truth, drug_mean, predictions[arm], pc_mean, pc1)
    info = {"split": split, "split_file_sha256": roc._sha256(split_path), "n_reference_lines": C,
            "n_common_treatments": len(common_treatments),
            "quality_range": [float(np.min(quality_all)), float(np.max(quality_all))], "arms": infos}
    return frame, info, selection_rows


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(args) -> None:
    repo_root = args.project_root.resolve() / "external" / "Rhaister"
    compact = roc.build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    hvg = roc.load_hvg_centroids(args.data_root, compact.genes)
    arms = list(args.arms)
    grid_map = grids(args)
    fixed = parse_fixed(args)
    if fixed:
        print(f"fixed params (no inner selection): {fixed}", flush=True)
    all_cells = sorted(set(compact.cells.tolist()))
    Dfull, dist_info = (None, None)
    if args.distance_features:
        Dfull, dist_info = feature_distance_matrix(args.distance_features, all_cells)
        print(f"distance feature space: {dist_info}", flush=True)
    A, graph_info = reactome_adjacency(args.reactome_gmt, compact.genes, args.pathway_min, args.pathway_max)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(compact.genes))
    graph = {"kg_neighbor": row_normalise(A), "kg_random": row_normalise(A[np.ix_(perm, perm)])}
    print("graph:", graph_info, flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    frames, infos, selections = [], [], []
    for split in args.splits:
        f, i, s = split_analysis(args, split, compact, hvg, graph, arms, grid_map, fixed, Dfull, all_cells)
        frames.append(f); infos.append(i); selections.extend(s)
    frame = pd.concat(frames, ignore_index=True)
    atomic_csv(pd.DataFrame(selections), args.output_root / "inner_selection.csv")

    models = ["perturbation_mean", "rhaister_o", *arms]
    recs = []
    for (split, cell), sub in frame.groupby(["split", "cell_line"], sort=True):
        for level in LEVELS:
            for model in models:
                recs.append({"split": int(split), "cell_line": cell, "level": level, "model": model,
                             "n_conditions": len(sub),
                             **{m: float(sub[f"{model}__{level}__{m}"].mean()) for m in ("pearson", "rmse", "mae")}})
    line_frame = pd.DataFrame(recs)
    atomic_csv(line_frame, args.output_root / "per_line_summary.csv")

    aggregate, gates = [], {}
    for level in LEVELS:
        pv = line_frame[line_frame.level.eq(level)].pivot_table(index="cell_line", columns="model", values="pearson")
        for model in models:
            mean, lo, hi = line_bootstrap_interval(pv[model].to_numpy(np.float64), args.seed, args.n_bootstrap)
            aggregate.append({"level": level, "model": model, "n_lines": len(pv), "line_macro_pearson": mean,
                              "line_bootstrap_low": lo, "line_bootstrap_high": hi})
        for arm in arms:
            mean, lo, hi = line_bootstrap_interval((pv[arm] - pv["rhaister_o"]).to_numpy(np.float64), args.seed, args.n_bootstrap)
            aggregate.append({"level": level, "model": f"{arm}_minus_rhaister_o", "n_lines": len(pv),
                              "line_macro_pearson": mean, "line_bootstrap_low": lo, "line_bootstrap_high": hi})
    agg = pd.DataFrame(aggregate)
    atomic_csv(agg, args.output_root / "aggregate_summary.csv")

    def val(level, model, col="line_macro_pearson"):
        return float(agg[agg.level.eq(level) & agg.model.eq(model)][col].iloc[0])

    rm = line_frame[line_frame.level.eq("drug_residual")].pivot_table(index="cell_line", columns="model", values="rmse")
    print("\ndrug-residual line-macro Pearson; Rhaister-O "
          f"{val('drug_residual', 'rhaister_o'):.6f}")
    for arm in arms:
        per_split = []
        for split in args.splits:
            b = line_frame[line_frame.split.eq(split) & line_frame.level.eq("drug_residual")]
            pv = b.pivot_table(index="cell_line", columns="model", values="pearson")
            per_split.append(float((pv[arm] - pv["rhaister_o"]).mean()))
        gain = val("drug_residual", f"{arm}_minus_rhaister_o")
        low = val("drug_residual", f"{arm}_minus_rhaister_o", "line_bootstrap_low")
        pc1_gain = val("train_pca1_removed", f"{arm}_minus_rhaister_o")
        draws = [i["arms"][arm]["permutation_gains"] for i in infos]
        # splits that selected the anchor contribute an exact-zero null
        n_perm = max((len(d) for d in draws), default=0)
        null = [float(np.mean([d[k] if d else 0.0 for d in draws])) for k in range(n_perm)]
        if null:
            p = (sum(v >= gain for v in null) + 1) / (len(null) + 1)
            null_mean, null_p95 = float(np.mean(null)), float(np.quantile(null, 0.95))
        else:
            p, null_mean, null_p95 = 1.0, 0.0, 0.0
        ratio = float(rm[arm].mean() / rm["rhaister_o"].mean())
        cond = {
            "gain_point_estimate_positive": gain > 0,
            "gain_at_least": gain >= GATE["min_drug_residual_gain"],
            "bootstrap_low_positive": low > 0,
            "pc1_gain_at_least": pc1_gain >= GATE["min_pc1_removed_gain"],
            "enough_positive_splits": sum(g > 0 for g in per_split) >= GATE["min_positive_splits"],
            "rmse_within_tolerance": ratio <= GATE["max_rmse_ratio"],
            "above_permutation_null": p <= 0.05,
        }
        gates[arm] = {"drug_residual_gain": gain, "drug_residual_gain_low": low, "pc1_removed_gain": pc1_gain,
                      "per_split_gain": per_split, "rmse_ratio": ratio, "permutation_p": p,
                      "permutation_null_mean": null_mean, "permutation_null_p95": null_p95,
                      "selected_params": [i["arms"][arm]["selected_param"] for i in infos],
                      "conditions": cond, "passes": all(cond.values())}
        print(f"  {arm:13s} {val('drug_residual', arm):.6f} gain {gain:+.6f} [low {low:+.6f}] "
              f"splits {[round(x, 5) for x in per_split]} p {p:.3f} params {gates[arm]['selected_params']} "
              f"PASS={gates[arm]['passes']}", flush=True)

    write_json({
        "purpose": "Pre-registered E1 (report/10_02_record.md): can Rhaister-O's slope depend on the cell line?",
        "arms": arms, "grids": {k: [str(x) for x in v] for k, v in grid_map.items()},
        "graph": graph_info, "reactome_gmt": str(args.reactome_gmt), "reactome_sha256": sha256(args.reactome_gmt),
        "predeclared_gate": {**GATE, "gain_point_estimate_positive": True},
        "fixed_params": {k: str(v) for k, v in fixed.items()},
        "distance_feature_space": dist_info,
        "gates": gates, "aggregate": aggregate, "per_split": infos,
        "rhaister_lambda": args.rhaister_lambda, "rhaister_blend": args.rhaister_blend,
        "n_bootstrap": args.n_bootstrap, "n_permutations": args.n_permutations, "seed": args.seed,
        "data_revision": args.data_revision, "runner_sha256": sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }, args.output_root / "summary.json")
    print(f"\nWrote {args.output_root}/summary.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=roc.DEFAULT_PROJECT_ROOT)
    p.add_argument("--data-root", type=Path, default=roc.DEFAULT_DATA_ROOT)
    p.add_argument("--cache-root", type=Path, default=roc.DEFAULT_CACHE_ROOT)
    p.add_argument("--baseline-root", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_o_compact"))
    p.add_argument("--component-root", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_all_split_evaluation"))
    p.add_argument("--output-root", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/results/context_slopes"))
    p.add_argument("--data-revision", default=roc.DEFAULT_DATA_REVISION)
    p.add_argument("--reactome-gmt", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/raw/reactome/ReactomePathways.gmt"))
    p.add_argument("--pathway-min", type=int, default=5)
    p.add_argument("--pathway-max", type=int, default=200)
    p.add_argument("--splits", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS) + list(EXPLORATORY_ARMS))
    p.add_argument("--kernel-grid", type=float, nargs="+", default=[4.0, 2.0, 1.0, 0.5, 0.25])
    p.add_argument("--k-grid", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20])
    p.add_argument("--lambda-grid", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0, 10000.0])
    p.add_argument("--alpha-grid", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
    p.add_argument("--distance-features", type=Path, default=None,
                   help="F1: parquet of per-line features (CVCL index) defining the kernel "
                        "distance instead of the DMSO correlation distance")
    p.add_argument("--fixed-params", nargs="*", default=None,
                   help="e.g. kernel=0.5 reliability=20 : use these values, skip inner selection (C1)")
    p.add_argument("--n-permutations", type=int, default=50)
    p.add_argument("--rhaister-lambda", type=float, default=1.0)
    p.add_argument("--rhaister-blend", type=float, default=0.95)
    p.add_argument("--n-bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
