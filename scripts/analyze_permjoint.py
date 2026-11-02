#!/usr/bin/env python3
"""Joint compound-label permutation null for PCC_plate (report/11_06_record.md 2026-11-02 E3; no training).

Reads `sdst_view_ablation.py --sets multi_norm permjoint --perm-seed s` outputs (tags views_permjoint_s<s>). Per seed and
fold: PCC_plate and PCC_DEG of the backbone, the real pipeline (multi_norm) and the permuted one (permjoint), transfer
terms and UnDetect, and the gain over the backbone with a 2,000-replicate bootstrap over test compounds.
Writes results/sdst_hybrid/views_permjoint_summary.{csv,json}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
SEEDS = range(5)


def main() -> None:
    rng = np.random.default_rng(0); rows = []
    for s in SEEDS:
        for f in range(5):
            S = np.load(OUT / f"views_permjoint_s{s}" / f"fold_{f}" / "scores.npz", allow_pickle=False); d, sh = S["d"], S["sh"]
            for mt in ("deg", "plate"):
                drugs, gi = np.unique(d if mt == "deg" else d[sh], return_inverse=True); nd = np.bincount(gi).astype(float)
                W = rng.multinomial(len(drugs), np.full(len(drugs), 1 / len(drugs)), size=2000)
                base = S[f"none|backbone|{mt}"].astype(np.float64)
                for st in ("multi_norm", "permjoint"):
                    for arm in ("transfer", "undetect"):
                        x = S[f"{st}|{arm}|{mt}"].astype(np.float64); dd = x - base
                        boot = (W @ np.bincount(gi, weights=dd)) / (W @ nd)
                        rows.append({"seed": s, "fold": f, "metric": mt, "set": st, "arm": arm, "backbone": float(base.mean()),
                                     "score": float(x.mean()), "gain": float(dd.mean()), "lo": float(np.quantile(boot, 0.025)),
                                     "hi": float(np.quantile(boot, 0.975))})
    T = pd.DataFrame(rows); T.to_csv(OUT / "views_permjoint_summary.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 400)
    agg = T.groupby(["metric", "set", "arm"]).agg(score=("score", "mean"), gain=("gain", "mean"), gain_max=("gain", "max"),
                                                   gain_min=("gain", "min"), ci_pos=("lo", lambda v: int((v > 0).sum())),
                                                   ci_neg=("hi", lambda v: int((v < 0).sum())), n=("gain", "size")).reset_index()
    print(T[(T.metric == "plate") & (T.arm == "undetect")].round(4).to_string(index=False))
    print("\nover seeds x folds (n = runs; ci_pos / ci_neg = runs with the gain CI above / below 0):\n" + agg.round(4).to_string(index=False))
    real = T[(T.set == "multi_norm") & (T.seed == 0)].set_index(["metric", "arm", "fold"]).gain
    perm = T[T.set == "permjoint"].groupby(["metric", "arm", "fold"]).gain.max()
    below = (perm < real.reindex(perm.index)).groupby(level=[0, 1]).all()
    print("\nevery permuted run below the real gain in every fold:", below.to_dict())
    (OUT / "views_permjoint_summary.json").write_text(json.dumps(
        {"agg": agg.to_dict(orient="records"), "all_perm_below_real": {"|".join(k): bool(v) for k, v in below.items()}},
        indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
