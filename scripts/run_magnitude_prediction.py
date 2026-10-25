#!/usr/bin/env python3
"""Can drug-target features predict how STRONGLY a drug hits a cell line?

Pre-registered F3 (report/10_02_record.md, 2026-10-01). F2 found that the
interaction features correlate with response magnitude (target-pathway activity
r = +0.21) while adding nothing to the benchmark's primary metric. That metric
is a per-condition Pearson across genes, which is invariant to scaling the
condition's response, so it is structurally blind to magnitude. This script
scores the quantity the features actually carry.

Target: m(c, t) = log ||delta(c, t) - drugmean(t)||, centred per treatment on
the reference lines, so the drug main effect is removed and what remains is
"does this line respond more or less strongly to this drug than lines do on
average". Models are ridge regressions fitted on reference lines only, lambda
by leave-one-reference-line-out. Scored per held-out line as the Pearson
between predicted and true magnitude across that line's treatments.
"""
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd

from run_coefficient_sharing import slope_estimates
from run_depmap_expression_ridge import atomic_csv, line_bootstrap_interval, response_cube, sha256, write_json
from run_depmap_rhaister_correction import line_hvg_vectors
from run_interaction_features import build_feature, centre_feature, drug_targets, line_table
from run_context_slopes import reactome_adjacency
import run_rhaister_o_compact as roc

ARMS = ("dmso_pc", "target_expression", "target_dependency", "target_pathway",
        "all_interaction", "permuted_targets", "rhaister_magnitude")


def ridge_loo(X, y, lams):
    """Exact LOO MSE per lambda; X already standardised, intercept via centring."""
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    yc = y - y.mean()
    out = {}
    for lam in lams:
        d = s ** 2 / (s ** 2 + lam)
        H = (U * d) @ U.T + 1.0 / len(X)
        r = (yc - H @ yc) / (1 - np.clip(np.diag(H), None, 1 - 1e-9))
        out[lam] = float(np.mean(r ** 2))
    return out


def fit_predict(X, y, Xq, lam):
    mx, my = X.mean(0), y.mean()
    A = X - mx
    b = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y - my))
    return my + (Xq - mx) @ b


def main(a):
    repo = a.project_root / "external" / "Rhaister"
    compact = roc.build_or_load_compact_cache(a.data_root, repo, a.cache_root)
    hvg = roc.load_hvg_centroids(a.data_root, compact.genes)
    spaces = {n: line_table(a.feature_root / f"{n}.parquet")
              for n in ("depmap_expression", "crispr_dependency", "cnv")}
    targets = drug_targets(a.drug_metadata)
    A_graph, _ = reactome_adjacency(a.reactome_gmt, compact.genes, 5, 200)
    rows = []
    for split in a.splits:
        hold, _, _ = roc.load_split(repo, split)
        ref = sorted(set(compact.cells.tolist()) - set(hold))
        order = ref + list(hold)
        R, common = response_cube(compact, order, sorted(set(compact.treatments.tolist())))
        R = R.astype(np.float64)
        C = len(ref)
        drugmean = R[:C].mean(axis=0)                       # (T, G) reference drug mean
        mag = np.log(np.linalg.norm(R - drugmean[None], axis=2) + 1e-12)   # (C+Q, T)
        y_all = mag - mag[:C].mean(axis=0)[None, :]         # centre per treatment on reference

        H = line_hvg_vectors(compact, hvg, order)
        zmean = H[:C].mean(axis=0)
        Z = H - zmean
        u, s, vt = np.linalg.svd(Z[:C], full_matrices=False)
        k = min(a.n_pc, len(s) - 1)
        dmso_pc = np.vstack([(u[:, :k] * s[None, :k]), Z[C:] @ vt[:k].T])
        dmso_pc = dmso_pc / dmso_pc[:C].std(axis=0, keepdims=True)

        feats = {}
        for name in ("target_expression", "target_dependency", "target_pathway", "permuted_targets"):
            X, _ = build_feature(name, common, order, spaces, targets, A_graph, compact.genes, a.seed)
            feats[name] = centre_feature(X, np.arange(C))

        mu, gamma, _ = slope_estimates(Z[:C], R[:C], a.rhaister_lambda)
        rho_mag = np.stack([np.log(np.linalg.norm(mu + gamma * Z[i][None, :] - drugmean, axis=1) + 1e-12)
                            for i in range(len(order))])
        rho_mag = rho_mag - rho_mag[:C].mean(axis=0)[None, :]

        for arm in a.arms:
            preds = np.zeros((len(hold), len(common)))
            for j in range(len(common)):
                yr = y_all[:C, j]
                if arm == "rhaister_magnitude":
                    preds[:, j] = rho_mag[C:, j]
                    continue
                if arm == "dmso_pc":
                    Xr, Xq = dmso_pc[:C], dmso_pc[C:]
                elif arm == "all_interaction":
                    Xr = np.column_stack([feats[f][:C, j] for f in ("target_expression", "target_dependency", "target_pathway")])
                    Xq = np.column_stack([feats[f][C:, j] for f in ("target_expression", "target_dependency", "target_pathway")])
                else:
                    Xr, Xq = feats[arm][:C, j][:, None], feats[arm][C:, j][:, None]
                if np.allclose(Xr.std(axis=0), 0):
                    continue
                sd = np.where(Xr.std(axis=0) > 1e-12, Xr.std(axis=0), 1.0)
                Xr, Xq = (Xr - Xr.mean(0)) / sd, (Xq - Xr.mean(0)) / sd
                lam = min(ridge_loo(Xr, yr, a.lambda_grid), key=ridge_loo(Xr, yr, a.lambda_grid).get)
                preds[:, j] = fit_predict(Xr, yr, Xq, lam)
            for i, cell in enumerate(hold):
                t, p = y_all[C + i], preds[i]
                r = np.corrcoef(t, p)[0, 1] if p.std() > 1e-12 else 0.0
                rows.append({"split": split, "cell_line": cell, "arm": arm, "pearson": float(r),
                             "rmse": float(np.sqrt(np.mean((t - p) ** 2)))})
        print(f"split {split}: {len(hold)} held-out lines, {len(common)} treatments", flush=True)

    df = pd.DataFrame(rows)
    a.output_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(df, a.output_root / "per_line.csv")
    pv = df.pivot_table(index="cell_line", columns="arm", values="pearson")
    summary, agg = {}, []
    for arm in a.arms:
        m, lo, hi = line_bootstrap_interval(pv[arm].to_numpy(np.float64), a.seed, a.n_bootstrap)
        summary[arm] = {"line_macro_pearson": m, "low": lo, "high": hi}
        agg.append({"arm": arm, "line_macro_pearson": m, "low": lo, "high": hi})
        print(f"  {arm:22s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    gates = {}
    for arm in ("target_expression", "target_dependency", "target_pathway", "all_interaction"):
        for base in ("dmso_pc", "permuted_targets"):
            d = (pv[arm] - pv[base]).to_numpy(np.float64)
            m, lo, hi = line_bootstrap_interval(d, a.seed, a.n_bootstrap)
            gates[f"{arm}_minus_{base}"] = {"gain": m, "low": lo, "high": hi, "passes": lo > 0}
            print(f"  {arm} - {base}: {m:+.4f} [{lo:+.4f}] pass={lo > 0}")
    write_json({"purpose": "Pre-registered F3: predict drug-response magnitude from interaction features",
                "arms": list(a.arms), "summary": summary, "gates": gates, "n_pc": a.n_pc,
                "lambda_grid": list(a.lambda_grid), "seed": a.seed, "n_bootstrap": a.n_bootstrap,
                "runner_sha256": sha256(Path(__file__).resolve()),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID")}, a.output_root / "summary.json")
    print(f"\nWrote {a.output_root}/summary.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=roc.DEFAULT_PROJECT_ROOT)
    p.add_argument("--data-root", type=Path, default=roc.DEFAULT_DATA_ROOT)
    p.add_argument("--cache-root", type=Path, default=roc.DEFAULT_CACHE_ROOT)
    p.add_argument("--feature-root", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/derived/cell_feature_spaces"))
    p.add_argument("--drug-metadata", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/raw/tahoe-100m/metadata/drug_metadata.parquet"))
    p.add_argument("--reactome-gmt", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/raw/reactome/ReactomePathways.gmt"))
    p.add_argument("--output-root", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/results/magnitude_prediction"))
    p.add_argument("--splits", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--n-pc", type=int, default=10)
    p.add_argument("--lambda-grid", type=float, nargs="+", default=[0.1, 1, 10, 100, 1000])
    p.add_argument("--rhaister-lambda", type=float, default=1.0)
    p.add_argument("--n-bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
