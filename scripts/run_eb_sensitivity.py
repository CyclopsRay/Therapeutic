#!/usr/bin/env python3
"""EmeraldBay few-shot sensitivity: does the panel transcriptional response help?

Pre-registered E5 (revised before running; report/10_02_record.md). Analogue of
the Rhaister paper's "sensitivity only" vs "sensitivity + tx" comparison, on our
own splits: held-out lines are the Tahoe split-s holdouts present in EmeraldBay
(four low-coverage lines excluded). For each split a seeded panel of 6 drugs
(all doses) is observed in the held-out lines -- both growth rate and
transcriptional delta; the remaining drugs' growth rates are predicted.

Per target treatment, ridge regressions fitted on the reference lines:
  drug_mean   mean growth over reference lines
  sens        panel growth rates of the line
  sens_dmso   + top-K PCs of the line's DMSO profile
  sens_tx     + top-K PCs of the line's observed panel deltas (concatenated)
  tx          panel-delta PCs only
Features standardised on reference lines; lambda (and K) by leave-one-reference-
line-out MSE. Scores: pooled R^2 per split (the paper's metric), and per
held-out line Pearson across target treatments. Gate: sens_tx minus sens,
line-bootstrap low > 0.
"""
import argparse, ast, json, os
from pathlib import Path
import numpy as np, pandas as pd

LOW = {"CVCL_1577", "CVCL_1119", "CVCL_1603", "CVCL_1056"}


def ridge_loo(X, Y, lams):
    """X (n, p) standardised, Y (n, T). Exact LOO MSE per lambda via hat matrix."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    out = {}
    Yc = Y - Y.mean(0)
    for lam in lams:
        d = s ** 2 / (s ** 2 + lam)
        H = (U * d) @ U.T + 1.0 / len(X)            # intercept via centring
        fit = H @ Y
        r = (Y - fit) / (1 - np.diag(H))[:, None]
        out[lam] = float(np.mean(r ** 2))
    return out


def ridge_fit_predict(X, Y, Xq, lam):
    mx = X.mean(0); my = Y.mean(0)
    A = X - mx
    B = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Y - my))
    return my + (Xq - mx) @ B


def standardise(Xr, Xq):
    m, s = Xr.mean(0), Xr.std(0)
    s[s < 1e-12] = 1
    return (Xr - m) / s, (Xq - m) / s


def pcs(Xr, Xq, k):
    m = Xr.mean(0)
    U, s, Vt = np.linalg.svd(Xr - m, full_matrices=False)
    k = min(k, len(s) - 1)
    return (Xr - m) @ Vt[:k].T, (Xq - m) @ Vt[:k].T


def main(a):
    eb = Path(a.eb_dir)
    meta = pd.read_parquet(eb / "line_condition_meta.parquet")
    delta = np.load(eb / "delta_log1p.npy")
    dmso = pd.read_parquet(eb / "dmso_log1p_by_line.parquet")
    ss = pd.read_parquet(Path(a.raw) / "metadata" / "summary_statistics.parquet")
    ss = ss[ss.condition.map(lambda c: len(ast.literal_eval(c)) == 1 and "DMSO" not in c)]
    ss = ss.groupby(["cell_line", "condition"], as_index=False).growth_rate.mean()
    keep = meta.single & ~meta.drug.str.startswith("dmso") & ~meta.line.isin(LOW)
    meta, delta = meta[keep].reset_index(drop=True), delta[keep.to_numpy()]
    lines = sorted(set(meta.line) & set(ss.cell_line))
    conds = sorted(set(meta.cond) & set(ss.condition))
    G = ss.pivot(index="cell_line", columns="condition", values="growth_rate").reindex(index=lines, columns=conds)
    assert not G.isna().any().any(), "incomplete growth grid"
    di = {(l, c): i for i, (l, c) in enumerate(zip(meta.line, meta.cond))}
    drug_of = {c: ast.literal_eval(c)[0][0] for c in conds}
    drugs = sorted(set(drug_of.values()))
    splits = {}
    for s in range(5, 10):
        import tomllib
        cfg = tomllib.load(open(f"{a.split_root}/{s}_holdout/generalization_converted_cell_lines_3b.toml", "rb"))
        splits[s] = [k.split(".")[-1] for k in cfg["fewshot"] if k.split(".")[-1] in lines]
    print(f"lines {len(lines)} treatments {len(conds)} drugs {len(drugs)}; holdouts per split "
          f"{ {s: len(v) for s, v in splits.items()} }", flush=True)
    models = ["drug_mean", "sens", "sens_dmso", "sens_tx", "tx"]
    rows, split_r2 = [], []
    for s, hold in splits.items():
        rng = np.random.default_rng(a.seed + s)
        panel_drugs = sorted(rng.choice(drugs, size=a.n_panel, replace=False).tolist())
        panel = [c for c in conds if drug_of[c] in panel_drugs]
        target = [c for c in conds if drug_of[c] not in panel_drugs]
        ref = [l for l in lines if l not in hold]
        Yr, Yq = G.loc[ref, target].to_numpy(), G.loc[hold, target].to_numpy()
        feats = {}
        feats["sens"] = (G.loc[ref, panel].to_numpy(), G.loc[hold, panel].to_numpy())
        tx_r = np.stack([np.concatenate([delta[di[(l, c)]] for c in panel]) for l in ref])
        tx_q = np.stack([np.concatenate([delta[di[(l, c)]] for c in panel]) for l in hold])
        dm_r, dm_q = dmso.loc[ref].to_numpy(), dmso.loc[hold].to_numpy()
        preds = {"drug_mean": np.broadcast_to(Yr.mean(0), Yq.shape)}
        chosen = {}
        for m in models[1:]:
            best = (np.inf, None)
            for k in a.k_grid if m != "sens" else [0]:
                parts_r, parts_q = [], []
                if m.startswith("sens"):
                    parts_r.append(feats["sens"][0]); parts_q.append(feats["sens"][1])
                if m == "sens_dmso":
                    r, q = pcs(dm_r, dm_q, k); parts_r.append(r); parts_q.append(q)
                if m in ("sens_tx", "tx"):
                    r, q = pcs(tx_r, tx_q, k); parts_r.append(r); parts_q.append(q)
                Xr, Xq = standardise(np.hstack(parts_r), np.hstack(parts_q))
                loo = ridge_loo(Xr, Yr, a.lambda_grid)
                lam = min(loo, key=loo.get)
                if loo[lam] < best[0]:
                    best = (loo[lam], (k, lam, Xr, Xq))
            k, lam, Xr, Xq = best[1]
            chosen[m] = (k, lam)
            preds[m] = ridge_fit_predict(Xr, Yr, Xq, lam)
        sst = ((Yq - Yq.mean()) ** 2).sum()
        r2 = {m: 1 - ((Yq - preds[m]) ** 2).sum() / sst for m in models}
        split_r2.append({"split": s, **r2, "panel": panel_drugs, "chosen": {m: [str(x) for x in v] for m, v in chosen.items()}})
        for i, l in enumerate(hold):
            for m in models:
                p = preds[m][i]
                r = np.corrcoef(p, Yq[i])[0, 1] if np.std(p) > 0 else np.nan
                rows.append({"split": s, "cell_line": l, "model": m, "pearson": r,
                             "mse": float(np.mean((p - Yq[i]) ** 2))})
        print(f"split {s}: panel {panel_drugs}  R2 " + " ".join(f"{m}={r2[m]:.3f}" for m in models)
              + f"  chosen {chosen}", flush=True)
    df = pd.DataFrame(rows); sr = pd.DataFrame(split_r2)
    out = Path(a.output_root); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_line.csv", index=False); sr.to_csv(out / "per_split_r2.csv", index=False)
    pv = df.pivot_table(index="cell_line", columns="model", values="pearson")
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, len(pv), size=(a.n_bootstrap, len(pv)))
    summ = {"per_split_r2_mean": {m: float(sr[m].mean()) for m in models},
            "per_split_r2_sd": {m: float(sr[m].std(ddof=1)) for m in models},
            "line_macro_pearson": {m: float(pv[m].mean()) for m in models}}
    for x, y in [("sens_tx", "sens"), ("sens_dmso", "sens"), ("sens", "drug_mean"), ("tx", "drug_mean")]:
        d = (pv[x] - pv[y]).to_numpy(); b = np.nanmean(d[idx], 1)
        sd = (sr[x] - sr[y]).to_numpy()
        summ[f"{x}_minus_{y}"] = {"line_pearson_gain": float(np.nanmean(d)), "low": float(np.quantile(b, .025)),
                                   "high": float(np.quantile(b, .975)), "split_r2_diffs": sd.round(4).tolist()}
    g = summ["sens_tx_minus_sens"]
    gate = {"rule": "sens_tx minus sens, line-bootstrap low > 0 (Pearson)", "gain": g["line_pearson_gain"],
            "low": g["low"], "passes": g["low"] > 0}
    print(json.dumps(summ, indent=1)); print("gate", gate)
    json.dump({"purpose": "E5 revised: EB few-shot sensitivity, panel transcriptional response", "summary": summ,
               "gate": gate, "n_panel": a.n_panel, "seed": a.seed, "k_grid": a.k_grid, "lambda_grid": a.lambda_grid,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID")}, open(out / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eb-dir", default="/nfs-stor/yuqi.lei/Therapeutic/derived/emerald_bay/sanity_v1")
    p.add_argument("--raw", default="/nfs-stor/yuqi.lei/Therapeutic/raw/emerald-bay/f2a0be6b02f731553657f0115c345b20bb020ede")
    p.add_argument("--split-root", default="/home/yuqi.lei/Therapeutic/external/Rhaister/splits/tahoe")
    p.add_argument("--output-root", default="/nfs-stor/yuqi.lei/Therapeutic/results/emerald_bay/sensitivity_fewshot")
    p.add_argument("--n-panel", type=int, default=6)
    p.add_argument("--k-grid", type=int, nargs="+", default=[2, 5, 10, 20])
    p.add_argument("--lambda-grid", type=float, nargs="+", default=[0.1, 1, 10, 100, 1000, 10000])
    p.add_argument("--n-bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
