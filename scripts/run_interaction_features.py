#!/usr/bin/env python3
"""Cell x drug interaction features: does knowing the drug's target help?

Pre-registered F2 (report/10_02_record.md, 2026-09-30). Rhaister-O's structure

    delta(c, t, g) = mu(t, g) + gamma(t, g) * z(c, g)

has no term in which a property of the drug meets a property of the cell line:
mu carries the drug, z carries the cell, and they only ever multiply through a
slope that is itself drug-specific but cell-agnostic. Everything this project
has added so far describes the cell line alone. This script adds a scalar
per (cell, treatment) built from the drug's annotated targets, which is a
genuine interaction term.

Features x(c, t), each the mean over drug t's annotated target genes of a
per-line quantity, centred per treatment on the reference lines (so the drug
main effect, already in mu, is removed and only the cell-specific deviation
survives) and then scaled to unit variance:

  target_expression   DepMap log-TPM of the targets
  target_dependency   -1 x CRISPR gene effect, so higher = the line needs the
                      target more; the drug should then do more
  target_cnv          DepMap copy number of the targets
  target_pathway      knowledge graph on the DRUG-to-CELL axis: expression of
                      the Reactome pathway neighbours of the targets, not of
                      the gene being predicted. The failed kg_neighbor arm
                      smoothed the baseline gene-to-gene; this asks whether the
                      pathway a drug acts on is active in this line.
  permuted_targets    negative control: each drug is given another drug's
                      target set, seeded. The real feature must beat it.

Two parameterisations, both deliberately low-capacity, because this project's
own results show that added parameters fail where reweighting succeeds:

  scale   prediction x (1 + theta * x), ONE scalar for the whole model.
          Anchor theta = 0.
  direct  + delta(g) * x, one coefficient per gene shared across all drugs,
          ridge lambda. Anchor lambda = inf.

Selection is leave-one-reference-line-out drug-residual Pearson, refit inside
every fold. Gate: run_coefficient_sharing.GATE plus gain_point_estimate_positive
plus beating the permuted-target control.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_rhaister_signal import parse_treatment
from run_coefficient_sharing import GATE, line_macro_pearson, slope_estimates
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
from run_context_slopes import reactome_adjacency, row_normalise

LEVELS = ("full", "drug_residual", "train_pca1_removed")
FEATURES = ("target_expression", "target_dependency", "target_cnv",
            "target_pathway", "permuted_targets")
INF = float("inf")


def strip_paren(name: str) -> str:
    return re.sub(r"\s*\(.*\)$", "", str(name))


def drug_targets(metadata: Path) -> dict[str, list[str]]:
    frame = pd.read_parquet(metadata)
    out = {}
    for row in frame.itertuples():
        if not isinstance(row.targets, str):
            continue
        genes = [t.strip() for t in row.targets.replace(";", ",").split(",")
                 if t.strip() and t.strip().lower() != "none"]
        if genes:
            out[str(row.drug)] = genes
    return out


def line_table(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.columns = [strip_paren(c) for c in frame.columns]
    return frame.loc[:, ~frame.columns.duplicated()]


def build_feature(kind: str, treatments: list[str], cells: list[str],
                  spaces: dict, targets: dict, graph, genes_2k, seed: int) -> tuple[np.ndarray, dict]:
    """Raw x(c, t); NaN where the modality or the target annotation is missing."""
    names = [parse_treatment(t)[0] for t in treatments]
    tmap = dict(targets)
    if kind == "permuted_targets":
        keys = sorted(tmap)
        values = [tmap[k] for k in keys]
        order = np.random.default_rng(seed).permutation(len(keys))
        tmap = {k: values[order[i]] for i, k in enumerate(keys)}
        kind = "target_expression"

    if kind == "target_pathway":
        table = spaces["depmap_expression"]
        # one hop out from each target in the Reactome graph
        idx = {g: i for i, g in enumerate(genes_2k)}
        expanded = {}
        for drug, gs in tmap.items():
            neigh = set()
            for g in gs:
                if g in idx:
                    row = graph[idx[g]]
                    neigh |= {genes_2k[j] for j in np.flatnonzero(row > 0)}
            expanded[drug] = sorted(neigh) if neigh else []
        tmap = expanded
    elif kind == "target_dependency":
        table = spaces["crispr_dependency"]
    elif kind == "target_cnv":
        table = spaces["cnv"]
    else:
        table = spaces["depmap_expression"]

    sign = -1.0 if kind == "target_dependency" else 1.0
    X = np.full((len(cells), len(treatments)), np.nan)
    have = set(table.columns)
    per_drug = {}
    for j, drug in enumerate(names):
        gs = [g for g in tmap.get(drug, []) if g in have]
        if not gs:
            continue
        per_drug[drug] = len(gs)
        values = table[gs].mean(axis=1)
        for i, cell in enumerate(cells):
            if cell in values.index:
                X[i, j] = sign * values[cell]
    info = {"kind": kind, "treatments_with_feature": int(np.isfinite(X).any(0).sum()),
            "lines_with_feature": int(np.isfinite(X).any(1).sum()),
            "median_targets_used": float(np.median(list(per_drug.values()))) if per_drug else 0.0}
    return X, info


def centre_feature(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Centre per treatment on the reference lines, then scale to unit variance.

    Centring per treatment removes the drug main effect, which mu(t, g) already
    carries; what is left is purely the cell-specific deviation, which is the
    interaction the estimator cannot express. Missing entries become 0, a
    neutral value after centring.
    """
    out = np.array(X, dtype=np.float64, copy=True)
    mean = np.nanmean(np.where(np.isfinite(out[ref]), out[ref], np.nan), axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    out = out - mean[None, :]
    out[~np.isfinite(out)] = 0.0
    sd = out[ref].std()
    return out / sd if sd > 1e-12 else out


def split_analysis(args, split, compact, hvg, spaces, targets, graph, feats):
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

    R, common = response_cube(compact, reference_cells, sorted(set(treatments.tolist())))
    R = R.astype(np.float64)
    order = list(reference_cells) + list(holdout_cells)
    H = line_hvg_vectors(compact, hvg, order)
    zmean = H[:len(reference_cells)].mean(axis=0)
    Zref, Zq = H[:len(reference_cells)] - zmean, H[len(reference_cells):] - zmean
    C, T = len(reference_cells), len(common)
    lam0, blend = args.rhaister_lambda, args.rhaister_blend

    raw, finfo = {}, {}
    for name in feats:
        Xall, info = build_feature(name, common, order, spaces, targets, graph, compact.genes, args.seed)
        raw[name] = Xall
        finfo[name] = info
    ref_slice = np.arange(C)

    grids = {"scale": [0.0] + list(args.theta_grid), "direct": [INF] + list(args.lambda_grid)}
    scores = {(f, p): np.zeros(len(grids[p])) for f in feats for p in grids}

    t0 = time.time()
    for held in range(C):
        keep = np.arange(C) != held
        fold_mean = Zref[keep].mean(axis=0)
        Zf, zh = Zref[keep] - fold_mean, Zref[held] - fold_mean
        mu, gamma, _ = slope_estimates(Zf, R[keep], lam0)
        base = mu + gamma * zh[None, :]
        resid = R[keep] - (mu[None] + gamma[None] * Zf[:, None, :])
        for f in feats:
            Xc = centre_feature(raw[f][:C], np.flatnonzero(keep))
            xh, xr = Xc[held], Xc[keep]
            for i, th in enumerate(grids["scale"]):
                pred = (1 - blend) * mu + blend * (base * (1.0 + th * xh[:, None]))
                scores[(f, "scale")][i] += line_macro_pearson(R[held], pred, mu)
            num = np.einsum("ct,ctg->g", xr, resid, optimize=True)
            den = float((xr * xr).sum())
            for i, lam in enumerate(grids["direct"]):
                delta = np.zeros(R.shape[2]) if lam == INF else num / (den + lam)
                pred = (1 - blend) * mu + blend * (base + delta[None, :] * xh[:, None])
                scores[(f, "direct")][i] += line_macro_pearson(R[held], pred, mu)
    for k in scores:
        scores[k] /= C
    print(f"  split {split}: inner LOO {time.time() - t0:.0f}s", flush=True)

    mu, gamma, _ = slope_estimates(Zref, R, lam0)
    resid = R - (mu[None] + gamma[None] * Zref[:, None, :])
    q_index = {c: i for i, c in enumerate(holdout_cells)}
    t_index = {t: i for i, t in enumerate(common)}
    rows_of = {c: np.flatnonzero(cells == c) for c in holdout_cells}

    predictions, infos, selection_rows = {}, {}, []
    for f in feats:
        Xc = centre_feature(raw[f][:C + len(holdout_cells)], ref_slice)
        xr, xq = Xc[:C], Xc[C:]
        num = np.einsum("ct,ctg->g", xr, resid, optimize=True)
        den = float((xr * xr).sum())
        for parm in grids:
            best = int(np.argmax(scores[(f, parm)]))
            value = grids[parm][best]
            arm = f"{f}__{parm}"
            for i, p in enumerate(grids[parm]):
                selection_rows.append({"split": split, "arm": arm, "param": p,
                                       "loo_drug_residual_pearson": float(scores[(f, parm)][i]),
                                       "selected": i == best, "anchor": i == 0})
            pred = np.array(rhaister, dtype=np.float32, copy=True)
            delta = (np.zeros(R.shape[2]) if (parm == "direct" and value == INF)
                     else (num / (den + value) if parm == "direct" else None))
            for c, qi in q_index.items():
                base_q = mu + gamma * Zq[qi][None, :]
                if parm == "scale":
                    adjust = blend * (base_q * value * xq[qi][:, None])
                else:
                    adjust = blend * (delta[None, :] * xq[qi][:, None])
                for r in rows_of[c]:
                    t = treatments[r]
                    if t in t_index:
                        pred[r] = rhaister[r] + adjust[t_index[t]].astype(np.float32)
            predictions[arm] = pred
            infos[arm] = {"selected_param": float(value), "selected_is_anchor": best == 0,
                          "selected_loo": float(scores[(f, parm)][best]),
                          "anchor_loo": float(scores[(f, parm)][0])}
            print(f"  split {split} {arm:34s} selected {value:g} "
                  f"(LOO {scores[(f, parm)][best]:.6f} vs anchor {scores[(f, parm)][0]:.6f})", flush=True)

    parsed = [parse_treatment(t) for t in treatments]
    frame = pd.DataFrame({"split": split, "cell_line": cells, "treatment": treatments,
                          "drug": [p[0] for p in parsed], "dose": [p[1] for p in parsed]})
    add_model_metrics(frame, "perturbation_mean", truth, drug_mean, drug_mean, pc_mean, pc1)
    add_model_metrics(frame, "rhaister_o", truth, drug_mean, rhaister, pc_mean, pc1)
    for arm, pred in predictions.items():
        add_model_metrics(frame, arm, truth, drug_mean, pred, pc_mean, pc1)
    return frame, {"split": split, "n_reference_lines": C, "n_treatments": T,
                   "features": finfo, "arms": infos}, selection_rows


def main(args) -> None:
    repo_root = args.project_root.resolve() / "external" / "Rhaister"
    compact = roc.build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    hvg = roc.load_hvg_centroids(args.data_root, compact.genes)
    spaces = {n: line_table(args.feature_root / f"{n}.parquet")
              for n in ("depmap_expression", "crispr_dependency", "cnv")}
    targets = drug_targets(args.drug_metadata)
    A, graph_info = reactome_adjacency(args.reactome_gmt, compact.genes, args.pathway_min, args.pathway_max)
    feats = list(args.features)
    print(f"targets for {len(targets)} drugs; graph {graph_info}", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    frames, infos, selections = [], [], []
    for split in args.splits:
        f, i, s = split_analysis(args, split, compact, hvg, spaces, targets, A, feats)
        frames.append(f); infos.append(i); selections.extend(s)
    frame = pd.concat(frames, ignore_index=True)
    atomic_csv(pd.DataFrame(selections), args.output_root / "inner_selection.csv")

    arms = [f"{f}__{p}" for f in feats for p in ("scale", "direct")]
    models = ["perturbation_mean", "rhaister_o", *arms]
    recs = []
    for (split, cell), sub in frame.groupby(["split", "cell_line"], sort=True):
        for level in LEVELS:
            for m in models:
                recs.append({"split": int(split), "cell_line": cell, "level": level, "model": m,
                             **{k: float(sub[f"{m}__{level}__{k}"].mean()) for k in ("pearson", "rmse", "mae")}})
    line_frame = pd.DataFrame(recs)
    atomic_csv(line_frame, args.output_root / "per_line_summary.csv")

    agg = []
    for level in LEVELS:
        pv = line_frame[line_frame.level.eq(level)].pivot_table(index="cell_line", columns="model", values="pearson")
        for m in models:
            mean, lo, hi = line_bootstrap_interval(pv[m].to_numpy(np.float64), args.seed, args.n_bootstrap)
            agg.append({"level": level, "model": m, "n_lines": len(pv), "line_macro_pearson": mean,
                        "line_bootstrap_low": lo, "line_bootstrap_high": hi})
        for arm in arms:
            mean, lo, hi = line_bootstrap_interval((pv[arm] - pv["rhaister_o"]).to_numpy(np.float64),
                                                   args.seed, args.n_bootstrap)
            agg.append({"level": level, "model": f"{arm}_minus_rhaister_o", "n_lines": len(pv),
                        "line_macro_pearson": mean, "line_bootstrap_low": lo, "line_bootstrap_high": hi})
    aggregate = pd.DataFrame(agg)
    atomic_csv(aggregate, args.output_root / "aggregate_summary.csv")

    def val(level, model, col="line_macro_pearson"):
        return float(aggregate[aggregate.level.eq(level) & aggregate.model.eq(model)][col].iloc[0])

    gates = {}
    print(f"\ndrug-residual line-macro Pearson; Rhaister-O {val('drug_residual', 'rhaister_o'):.6f}")
    for arm in arms:
        per_split = []
        for split in args.splits:
            b = line_frame[line_frame.split.eq(split) & line_frame.level.eq("drug_residual")]
            pv = b.pivot_table(index="cell_line", columns="model", values="pearson")
            per_split.append(float((pv[arm] - pv["rhaister_o"]).mean()))
        gain = val("drug_residual", f"{arm}_minus_rhaister_o")
        low = val("drug_residual", f"{arm}_minus_rhaister_o", "line_bootstrap_low")
        control = val("drug_residual", f"permuted_targets__{arm.split('__')[1]}_minus_rhaister_o")
        cond = {"gain_point_estimate_positive": gain > 0,
                "gain_at_least": gain >= GATE["min_drug_residual_gain"],
                "bootstrap_low_positive": low > 0,
                "pc1_gain_at_least": val("train_pca1_removed", f"{arm}_minus_rhaister_o") >= GATE["min_pc1_removed_gain"],
                "enough_positive_splits": sum(g > 0 for g in per_split) >= GATE["min_positive_splits"],
                "beats_permuted_target_control": gain > control}
        gates[arm] = {"gain": gain, "low": low, "per_split": per_split,
                      "permuted_control_gain": control, "conditions": cond,
                      "passes": all(cond.values())}
        print(f"  {arm:34s} {val('drug_residual', arm):.6f} gain {gain:+.6f} [low {low:+.6f}] "
              f"control {control:+.6f} PASS={gates[arm]['passes']}", flush=True)

    write_json({"purpose": "Pre-registered F2: cell x drug interaction features from drug targets",
                "features": feats, "parameterisations": ["scale", "direct"],
                "theta_grid": list(args.theta_grid), "lambda_grid": list(args.lambda_grid),
                "predeclared_gate": {**GATE, "gain_point_estimate_positive": True,
                                     "beats_permuted_target_control": True},
                "gates": gates, "aggregate": agg, "per_split": infos, "graph": graph_info,
                "drug_metadata_sha256": sha256(args.drug_metadata), "seed": args.seed,
                "n_bootstrap": args.n_bootstrap, "runner_sha256": sha256(Path(__file__).resolve()),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
               args.output_root / "summary.json")
    print(f"\nWrote {args.output_root}/summary.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=roc.DEFAULT_PROJECT_ROOT)
    p.add_argument("--data-root", type=Path, default=roc.DEFAULT_DATA_ROOT)
    p.add_argument("--cache-root", type=Path, default=roc.DEFAULT_CACHE_ROOT)
    p.add_argument("--baseline-root", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_o_compact"))
    p.add_argument("--component-root", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_all_split_evaluation"))
    p.add_argument("--feature-root", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/derived/cell_feature_spaces"))
    p.add_argument("--drug-metadata", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/raw/tahoe-100m/metadata/drug_metadata.parquet"))
    p.add_argument("--reactome-gmt", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/raw/reactome/ReactomePathways.gmt"))
    p.add_argument("--output-root", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/results/interaction_features"))
    p.add_argument("--pathway-min", type=int, default=5)
    p.add_argument("--pathway-max", type=int, default=200)
    p.add_argument("--splits", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    p.add_argument("--features", nargs="+", default=list(FEATURES), choices=list(FEATURES))
    p.add_argument("--theta-grid", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4, 0.8, -0.05, -0.1, -0.2, -0.4])
    p.add_argument("--lambda-grid", type=float, nargs="+", default=[1e2, 1e3, 1e4, 1e5, 1e6])
    p.add_argument("--rhaister-lambda", type=float, default=1.0)
    p.add_argument("--rhaister-blend", type=float, default=0.95)
    p.add_argument("--n-bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
