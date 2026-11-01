#!/usr/bin/env python3
"""Read `run_sdst_hybrid.py` outputs against the published per-fold PCC_DEG of Bai et al. 2026.

Per fold: each arm's test PCC_DEG; paired difference arm - our mlp with a 2,000-replicate bootstrap over test drugs
(drug-weighted resampling of per-sample differences, as the DeepCE analysis); share of test drugs whose mean
per-sample PCC improves. Published values (seven models + their drug-free MLP) come from the benchmark's
`full_vs_mlp.csv` and are fold-level only (their predictions are not released), so the comparison to them is by fold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
PUB = Path(__file__).resolve().parents[1] / "repro/dpb/upstream/data/analysis/benchmark_metrics/statistics/full_vs_mlp.csv"


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--tag", default="v1"); args = p.parse_args()
    root = RES / args.tag
    pub = pd.read_csv(PUB); pub = pub[pub.metric_key == "pcc_deg_mean"]
    table = {m: [float(r[f"fold_{f}_full"]) for f in range(5)] for m, r in pub.set_index("model").iterrows()}
    table["MLP (published)"] = [float(pub.iloc[0][f"fold_{f}_comparator"]) for f in range(5)]
    rows, contrasts = [], []
    rng = np.random.default_rng(0)
    folds = sorted(int(d.name.split("_")[1]) for d in root.glob("fold_*") if (d / "per_sample_test.npz").exists())
    for f in folds:
        z = np.load(root / f"fold_{f}" / "per_sample_test.npz", allow_pickle=False)
        arms = [k for k in z.files if k not in ("pert_idx", "cell_idx")]
        drugs, gi = np.unique(z["pert_idx"], return_inverse=True); n = np.bincount(gi).astype(float)
        W = rng.multinomial(len(drugs), np.full(len(drugs), 1 / len(drugs)), size=2000)
        for a in arms:
            rows.append({"fold": f, "arm": a, "test": float(z[a].mean())})
            if a == "mlp":
                continue
            d = z[a] - z["mlp"]; s = np.bincount(gi, weights=d); boot = (W @ s) / (W @ n)
            contrasts.append({"fold": f, "arm": a, "diff": float(d.mean()), "lo": float(np.quantile(boot, 0.025)),
                              "hi": float(np.quantile(boot, 0.975)), "drugs_up": float((s / n > 0).mean()),
                              "samples_up": float((d > 0).mean())})
    ours = pd.DataFrame(rows).pivot(index="arm", columns="fold", values="test")
    pubdf = pd.DataFrame(table, index=range(5)).T[folds]
    allf = pd.concat([pubdf.assign(source="published"), ours.assign(source="ours")])
    allf["mean"] = allf[folds].mean(1)
    print(allf.sort_values("mean", ascending=False).round(4).to_string())
    c = pd.DataFrame(contrasts)
    print("\npaired vs our mlp (drug bootstrap per fold):")
    print(c.round(4).to_string(index=False))
    summ = c.groupby("arm").agg(diff=("diff", "mean"), folds_pos=("diff", lambda x: int((x > 0).sum())),
                                ci_excl0=("lo", lambda x: int((x > 0).sum())), drugs_up=("drugs_up", "mean"))
    print("\nacross folds:"); print(summ.round(4).to_string())
    best_pub = pubdf.loc[[m for m in pubdf.index if m != "MLP (published)"]].max()
    beat = {a: int((ours.loc[a] > best_pub).sum()) for a in ours.index}
    beat_mlp = {a: int((ours.loc[a] > pubdf.loc["MLP (published)"]).sum()) for a in ours.index}
    print("\nfolds above best published model:", beat); print("folds above published MLP:", beat_mlp)
    out = {"folds": folds, "per_fold": allf.reset_index().to_dict("records"), "contrasts": contrasts,
           "folds_above_best_published": beat, "folds_above_published_mlp": beat_mlp}
    (root / "analysis.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
