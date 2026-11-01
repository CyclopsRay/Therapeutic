#!/usr/bin/env python3
"""Wiring-effect bound and credit decomposition for the published-model audits (RECOMB 2027 method, output 3).

Design: report/10_30_record.md 2026-10-26 10:20. The audits have R = 1-4 rewires, so the permutation p is at least
1/(R+1) and can never reach 0.05; when the test does not reject we report how large a wiring effect the runs exclude.

  wiring  = mean(real) - mean(rewired)       Welch interval over runs; one-sided upper bound at level 1 - alpha
  degree  = mean(rewired) - mean(ER)         point estimate (ER is 1 run in GSNN/GEARS)
  identity= mean(rewired) - mean(labels)     same graph shape, genes reassigned

Each run is one training of the model on one graph, so a rewired arm's spread mixes training noise and
between-rewire variation; DeepCE has one rewired graph (seed 0) x 3 training seeds, so its bound is conditional on
that rewire. Runs are pinned by id below (failed or smoke runs excluded; see the cited reports).
Run with the gnn-gpu env (torch is needed to read GSNN's result_metric_dict.pt).
"""
from __future__ import annotations

import argparse
import glob
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPRO = Path("/nfs-stor/yuqi.lei/Therapeutic/repro")
OUT = Path("/nfs-stor/yuqi.lei/Therapeutic/results/wiring_bound")

# GSNN fold 0 (report/10_23_report.md §1); none/33d224fa stopped at epoch 2 and is excluded there
GSNN_RUNS = {"real": ["none/0a5d54e5", "none_rep1/9f4638a7"],
             "rewired": ["ladder_ff_rewired0/64c4bf42", "ladder_ff_rewired0_rep1/9e4832ec",
                         "ladder_ff_rewired1/b2957ab4", "ladder_ff_rewired2/ab2867cb"],
             "labels": ["ladder_ff_labels0/6d0423d0", "ladder_ff_labels1/0a603fb8", "ladder_ff_labels2/cd8e2379"],
             "er": ["ladder_ff_er0/f1fd1124"], "empty": ["ladder_ff_empty/64506c53"]}
# GEARS Norman, GO graph replaced (report/10_16_report.md §2)
GEARS_RUNS = {"real": ["go_real0_seed1", "go_real0_seed2", "go_real0_seed3"],
              "rewired": ["go_rewired0_seed1", "go_rewired1_seed1", "go_rewired2_seed1"],
              "labels": ["go_labels0_seed1", "go_labels1_seed1", "go_labels2_seed1"],
              "er": ["go_er0_seed1"], "empty": ["go_empty0_seed1"]}
GEARS_SUBGROUPS = ("unseen_single", "combo_seen0", "combo_seen1", "combo_seen2")
# DeepCE node2vec ladder (report/10_09_report.md §5.4)
DEEPCE_ARMS = {"n2v_real": "real", "n2v_rewired": "rewired", "n2v_labels": "labels", "n2v_er": "er"}


def collect() -> pd.DataFrame:
    import torch
    rows = []
    for arm, runs in GSNN_RUNS.items():
        for run in runs:
            d, uid = run.split("/")
            (f,) = glob.glob(str(REPRO / "gsnn/results" / d / f"{uid}*" / "fold_0_lincs_splits/result_metric_dict.pt"))
            m = torch.load(f, weights_only=False)
            for metric in ("r_flat_test", "mean_r_test"):
                rows.append({"model": "GSNN", "metric": metric, "arm": arm, "run": run, "score": float(m[metric])})
    for arm, runs in GEARS_RUNS.items():
        for run in runs:
            m = json.loads((REPRO / "gears/results/norman" / run / "metrics.json").read_text())
            rows.append({"model": "GEARS", "metric": "pearson_de", "arm": arm, "run": run,
                         "score": float(m["metrics"]["pearson_de"])})
            for g in GEARS_SUBGROUPS:
                rows.append({"model": "GEARS", "metric": f"pearson_delta_{g}", "arm": arm, "run": run,
                             "score": float(m["subgroups"][g]["pearson_delta"])})
    dc = pd.read_csv(REPRO / "deepce/results/n2v_ladder_runs.csv")
    for v, arm in DEEPCE_ARMS.items():
        for _, r in dc[dc.variant == v].iterrows():
            rows.append({"model": "DeepCE", "metric": "pearson_test", "arm": arm, "run": f"{v}/seed{r.seed}",
                         "score": float(r.test)})
    return pd.DataFrame(rows)


def welch(a: np.ndarray, b: np.ndarray, alpha: float) -> dict:
    """Interval for mean(a) - mean(b); one-sided upper bound at 1 - alpha and two-sided (1 - 2 alpha) interval."""
    diff = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(1 - alpha, df)
    return {"diff": diff, "se": se, "df": df, "lower": diff - t * se, "upper": diff + t * se,
            "p_one_sided_real_gt_rewired": float(stats.t.sf(diff / se, df))}


def bound(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    out = []
    for (model, metric), g in df.groupby(["model", "metric"], sort=False):
        s = {arm: g.loc[g.arm == arm, "score"].to_numpy() for arm in g.arm.unique()}
        w = welch(s["real"], s["rewired"], alpha)
        row = {"model": model, "metric": metric, "n_real": len(s["real"]), "n_rewired": len(s["rewired"]),
               "real": s["real"].mean(), "rewired": s["rewired"].mean(),
               "wiring": w["diff"], "wiring_se": w["se"], "welch_df": w["df"],
               f"wiring_lower_{1 - 2 * alpha:.0%}": w["lower"], f"wiring_upper_{1 - alpha:.0%}": w["upper"],
               "p_real_gt_rewired": w["p_one_sided_real_gt_rewired"],
               "min_perm_p": 1 / (len(s["rewired"]) + 1)}
        if "er" in s:
            row.update({"er": s["er"].mean(), "degree": s["rewired"].mean() - s["er"].mean(),
                        "published_style": s["real"].mean() - s["er"].mean()})
        if "labels" in s:
            row["identity"] = s["rewired"].mean() - s["labels"].mean()
        if "empty" in s:
            row["real_minus_empty"] = s["real"].mean() - s["empty"].mean()
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    print(json.dumps({"host": platform.node(), "python": sys.version.split()[0], "numpy": np.__version__,
                      "pandas": pd.__version__, "argv": sys.argv[1:]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    runs = collect()
    runs.to_csv(OUT / "audit_runs.csv", index=False)
    t = bound(runs, args.alpha)
    t.to_csv(OUT / "wiring_bound.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
