#!/usr/bin/env python3
"""Unseen-drug task on Tahoe: does a drug encoder predict responses to new drugs?

Pre-registered 2026-09-29 as E2 in report/10_02_record.md. This is a SECONDARY
task, distinct from the primary unseen-cell-line benchmark: here every cell line
is seen and whole drugs (all three doses) are held out.

Drugs are split into five folds (seeded). For a held-out drug at dose d in line c
the models predict the 2,000-gene delta:

  line_dose_mean   mean over training drugs of line c at dose d
  moa_mean         same, restricted to training drugs of the same `moa-fine`
                   class (falls back to line_dose_mean if the class is unseen)
  knn_morgan       Tanimoto-weighted mean of the k most similar training drugs
                   (Morgan r=2, 2048 bits); k by inner leave-one-drug-out
  kr_morgan        kernel ridge on the residual from line_dose_mean with the
                   Tanimoto kernel; lambda by exact leave-one-drug-out (hat matrix)
  kr_morgan_moa    same with kernel (Tanimoto + same-MOA indicator) / 2; the
                   uninformative "unclear" class is never counted as a match
  encoder          per fold, whichever of knn/kr/kr_moa scored best in the inner
                   leave-one-drug-out -- the pre-registered primary model

Metrics per (drug, dose, line): Pearson of the full delta, and Pearson of the
residual from line_dose_mean (the analogue of the drug-residual level). Summary
is the mean over lines of the mean over held-out (drug, dose). Uncertainty is a
drug-cluster bootstrap (resampling whole drugs). Gate: encoder minus moa_mean on
the residual level, bootstrap low > 0.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import run_rhaister_o_compact as roc
from evaluate_rhaister_signal import parse_treatment
from run_depmap_expression_ridge import atomic_csv, sha256, write_json


def rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return np.where(den > 0, (a * b).sum(-1) / np.where(den > 0, den, 1.0), np.nan)


def morgan_matrix(names: list[str], meta: pd.DataFrame, bits: int = 2048) -> tuple[np.ndarray, list[str]]:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)
    smiles = dict(zip(meta.drug.astype(str), meta.canonical_smiles))
    fp = np.zeros((len(names), bits), dtype=np.float64)
    missing = []
    for i, n in enumerate(names):
        mol = Chem.MolFromSmiles(smiles[n]) if isinstance(smiles.get(n), str) else None
        if mol is None:
            missing.append(n)
            continue
        fp[i] = np.asarray(gen.GetFingerprint(mol), dtype=np.float64)
    return fp, missing


def tanimoto(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nanmean0(x: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        m = np.nanmean(x, axis=axis)
    return np.nan_to_num(m)


def main(args) -> None:
    repo_root = args.project_root / "external" / "Rhaister"
    compact = roc.build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    parsed = [parse_treatment(t) for t in compact.treatments]
    drugs = sorted({p[0] for p in parsed})
    doses = sorted({float(p[1]) for p in parsed})
    lines = sorted(set(compact.cells.tolist()))
    di, si, li = {d: i for i, d in enumerate(drugs)}, {d: i for i, d in enumerate(doses)}, {c: i for i, c in enumerate(lines)}
    G = len(compact.genes)
    Y = np.full((len(drugs), len(doses), len(lines), G), np.nan, dtype=np.float32)
    for row, (c, p) in enumerate(zip(compact.cells, parsed)):
        Y[di[p[0]], si[float(p[1])], li[c]] = compact.delta[row]
    print(f"tensor drugs={len(drugs)} doses={doses} lines={len(lines)} genes={G} "
          f"observed={np.isfinite(Y[..., 0]).mean():.4f}", flush=True)

    meta = pd.read_parquet(args.drug_metadata)
    moa = dict(zip(meta.drug.astype(str), meta["moa-fine"].astype(str)))
    moa_of = np.array([moa.get(d, "__missing__") for d in drugs])
    informative = ~np.isin(moa_of, ["unclear", "__missing__", "None", "nan"])
    fp, missing_smiles = morgan_matrix(drugs, meta)
    S = tanimoto(fp, fp)
    M = ((moa_of[:, None] == moa_of[None, :]) & informative[:, None] & informative[None, :]).astype(np.float64)
    kernels = {"kr_morgan": S, "kr_morgan_moa": (S + M) / 2.0}
    print(f"missing SMILES: {missing_smiles}; informative MOA drugs: {int(informative.sum())}", flush=True)

    rng = np.random.default_rng(args.seed)
    fold_of = np.empty(len(drugs), dtype=int)
    fold_of[rng.permutation(len(drugs))] = np.arange(len(drugs)) % args.n_folds

    models = ("line_dose_mean", "moa_mean", "knn_morgan", "kr_morgan", "kr_morgan_moa", "encoder")
    records, selections = [], []
    for fold in range(args.n_folds):
        tr, te = np.flatnonzero(fold_of != fold), np.flatnonzero(fold_of == fold)
        Ytr = Y[tr]                                                   # (Dtr, S, L, G)
        base = nanmean0(Ytr, 0)                                       # (S, L, G)
        Rtr = np.nan_to_num(Ytr - base[None])                         # residual, missing -> 0
        obs_tr = np.isfinite(Ytr[..., 0])                             # (Dtr, S, L)
        flat_tr = Rtr.reshape(len(tr), -1).astype(np.float64)

        def inner_score(pred_flat: np.ndarray) -> float:
            p = pred_flat.reshape(Rtr.shape)
            r = rowwise_pearson(Rtr, p)[obs_tr]
            return float(np.nanmean(r))

        # knn: leave-one-drug-out over training drugs
        Str = S[np.ix_(tr, tr)].copy()
        np.fill_diagonal(Str, -np.inf)
        order = np.argsort(-Str, axis=1)
        best = {}
        knn_scores = {}
        for k in args.k_grid:
            idx = order[:, :k]
            w = np.take_along_axis(Str, idx, 1)
            w = np.clip(w, 1e-6, None)
            w /= w.sum(1, keepdims=True)
            pred = np.einsum("dk,dkf->df", w, flat_tr[idx])
            knn_scores[k] = inner_score(pred)
        kbest = max(knn_scores, key=knn_scores.get)
        best["knn_morgan"] = (kbest, knn_scores[kbest])
        # kernel ridge: exact LOO via hat matrix
        for name, K in kernels.items():
            Ktr = K[np.ix_(tr, tr)]
            evals, evecs = np.linalg.eigh(Ktr)
            proj = evecs.T @ flat_tr
            scores = {}
            for lam in args.lambda_grid:
                shrink = evals / (evals + lam)
                fitted = evecs @ (shrink[:, None] * proj)
                hdiag = np.einsum("ij,j,ij->i", evecs, shrink, evecs)
                loo = flat_tr - (flat_tr - fitted) / (1.0 - hdiag)[:, None]
                scores[lam] = inner_score(loo)
            lbest = max(scores, key=scores.get)
            best[name] = (lbest, scores[lbest])
            for lam, sc in scores.items():
                selections.append({"fold": fold, "model": name, "param": lam, "inner_loo": sc})
        for k, sc in knn_scores.items():
            selections.append({"fold": fold, "model": "knn_morgan", "param": k, "inner_loo": sc})
        encoder_choice = max(best, key=lambda m: best[m][1])
        print(f"fold {fold}: train={len(tr)} test={len(te)} best={ {m: (p, round(s, 4)) for m, (p, s) in best.items()} } "
              f"-> encoder={encoder_choice}", flush=True)

        # test predictions (residual space), shape (Dte, S, L, G)
        preds = {"line_dose_mean": np.zeros((len(te),) + base.shape)}
        moa_pred = np.zeros_like(preds["line_dose_mean"])
        for j, d in enumerate(te):
            same = tr[moa_of[tr] == moa_of[d]]
            if len(same):
                moa_pred[j] = nanmean0(Y[same], 0) - base
        preds["moa_mean"] = moa_pred
        Ste = S[np.ix_(te, tr)]
        idx = np.argsort(-Ste, axis=1)[:, :kbest]
        w = np.clip(np.take_along_axis(Ste, idx, 1), 1e-6, None)
        w /= w.sum(1, keepdims=True)
        preds["knn_morgan"] = np.einsum("dk,dkf->df", w, flat_tr[idx]).reshape((len(te),) + base.shape)
        for name, K in kernels.items():
            lam = best[name][0]
            alpha = np.linalg.solve(K[np.ix_(tr, tr)] + lam * np.eye(len(tr)), flat_tr)
            preds[name] = (K[np.ix_(te, tr)] @ alpha).reshape((len(te),) + base.shape)
        preds["encoder"] = preds[encoder_choice]

        truth = Y[te]
        truth_res = truth - base[None]
        obs = np.isfinite(truth[..., 0])
        for m in models:
            full = rowwise_pearson(np.nan_to_num(truth), preds[m] + base[None])
            res = rowwise_pearson(np.nan_to_num(truth_res), preds[m])
            for j, d in enumerate(te):
                for s in range(len(doses)):
                    for l in range(len(lines)):
                        if obs[j, s, l]:
                            records.append((fold, drugs[d], doses[s], lines[l], m, full[j, s, l], res[j, s, l]))

    df = pd.DataFrame(records, columns=["fold", "drug", "dose", "cell_line", "model", "full_pearson", "residual_pearson"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(df, args.output_root / "per_condition.csv")
    atomic_csv(pd.DataFrame(selections), args.output_root / "inner_selection.csv")

    # summaries: line-macro; drug-cluster bootstrap on per-drug means
    summary = {}
    per_drug = df.groupby(["model", "drug"])[["full_pearson", "residual_pearson"]].mean().reset_index()
    wide = {lvl: per_drug.pivot(index="drug", columns="model", values=lvl) for lvl in ("full_pearson", "residual_pearson")}
    brng = np.random.default_rng(args.seed)
    boot_idx = brng.integers(0, len(wide["full_pearson"]), size=(args.n_bootstrap, len(wide["full_pearson"])))
    for lvl, w in wide.items():
        summary[lvl] = {}
        line_macro = df.groupby(["model", "cell_line"])[lvl].mean().groupby("model").mean()
        for m in models:
            v = w[m].to_numpy()
            b = v[boot_idx].mean(1)
            summary[lvl][m] = {"line_macro": float(line_macro[m]), "drug_macro": float(v.mean()),
                               "drug_boot_low": float(np.quantile(b, 0.025)), "drug_boot_high": float(np.quantile(b, 0.975))}
        for a, bm in [("encoder", "moa_mean"), ("encoder", "line_dose_mean"), ("moa_mean", "line_dose_mean"),
                      ("kr_morgan_moa", "kr_morgan")]:
            v = (w[a] - w[bm]).to_numpy()
            b = v[boot_idx].mean(1)
            summary[lvl][f"{a}_minus_{bm}"] = {"drug_macro": float(v.mean()), "drug_boot_low": float(np.quantile(b, 0.025)),
                                               "drug_boot_high": float(np.quantile(b, 0.975))}
    gate_row = summary["residual_pearson"]["encoder_minus_moa_mean"]
    gate = {"rule": "encoder minus moa_mean on residual Pearson, drug-bootstrap low > 0",
            "gain": gate_row["drug_macro"], "low": gate_row["drug_boot_low"], "passes": gate_row["drug_boot_low"] > 0}
    for lvl in summary:
        print(f"\n{lvl}:")
        for k, v in summary[lvl].items():
            print(f"  {k:32s} " + " ".join(f"{a}={b:.4f}" for a, b in v.items()))
    print("\ngate:", gate)
    write_json({"purpose": "Pre-registered E2 (report/10_02_record.md): unseen-drug task, drug encoder",
                "n_folds": args.n_folds, "k_grid": args.k_grid, "lambda_grid": args.lambda_grid,
                "missing_smiles": missing_smiles, "n_informative_moa": int(informative.sum()),
                "summary": summary, "gate": gate, "seed": args.seed, "n_bootstrap": args.n_bootstrap,
                "drug_metadata": str(args.drug_metadata), "drug_metadata_sha256": sha256(args.drug_metadata),
                "runner_sha256": sha256(Path(__file__).resolve()), "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
               args.output_root / "summary.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=roc.DEFAULT_PROJECT_ROOT)
    p.add_argument("--data-root", type=Path, default=roc.DEFAULT_DATA_ROOT)
    p.add_argument("--cache-root", type=Path, default=roc.DEFAULT_CACHE_ROOT)
    p.add_argument("--drug-metadata", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/raw/tahoe-100m/metadata/drug_metadata.parquet"))
    p.add_argument("--output-root", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/results/unseen_drug"))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--k-grid", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50])
    p.add_argument("--lambda-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--n-bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
