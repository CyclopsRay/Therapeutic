#!/usr/bin/env python3
"""Per-fold PCC_DEG and validation-chosen settings of `run_undetect_v2.py` outputs, with differences vs a reference arm.

PCC_DEG = test mean over all test samples (`summary.json`); PCC_plate contrasts come from
`analyze_sdst_centred.py --tag <tag> --ref <ref>`, which this script reads if present. Writes
results/sdst_hybrid/<tag>/undetect_v2_summary.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
SETTINGS = ("lam", "a", "g", "h", "n", "h_bw", "kappa", "w", "epoch")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--tag", default="v7_ext_cpu"); p.add_argument("--ref", default="pool_kta")
    args = p.parse_args(); base = RES / args.tag; rows = []
    for fp in sorted(base.glob("fold_*/summary.json")):
        s = json.loads(fp.read_text()); f = s["fold"]; ref = s["arms"][args.ref]["test"]
        for arm, info in s["arms"].items():
            rows.append({"fold": f, "arm": arm, "deg": info["test"], "deg_diff": info["test"] - ref, "valid": info.get("valid"),
                         **{k: (json.dumps(info[k]) if isinstance(info[k], dict) else info[k]) for k in SETTINGS if k in info}})
        print(f"fold {f}: weights {json.dumps({k: {v: round(x, 3) for v, x in w.items()} for k, w in s['weights'].items()})}"
              f"; sensitivity {s.get('sensitivity')}; eigh fallbacks {s.get('eigh_fallbacks')}")
    df = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\nPCC_DEG per fold:\n" + df.pivot(index="arm", columns="fold", values="deg").assign(mean=lambda x: x.mean(1)).round(4).to_string())
    print(f"\nPCC_DEG minus {args.ref}:\n"
          + df.pivot(index="arm", columns="fold", values="deg_diff").assign(mean=lambda x: x.mean(1)).round(4).to_string())
    cols = [c for c in ("fold", "valid") + SETTINGS if c in df]
    for arm in ("pool_kta", "pool_ml", "pool_val", "pool3_resp", "pool3_ctl", "pool_sens", "pool_all"):
        if arm in set(df.arm):
            print(f"\nchosen settings, {arm}:\n" + df[df.arm == arm][cols].dropna(axis=1, how="all").round(4).to_string(index=False))
    cj = base / f"centred_vs_{args.ref}.json"
    if cj.exists():
        c = pd.DataFrame(json.loads(cj.read_text())["contrasts"])
        agg = c.groupby("arm").agg(plate_diff=("centred_diff_vs_mlp", "mean"), folds_pos=("centred_diff_vs_mlp", lambda x: int((x > 0).sum())),
                                   ci_below0=("hi", lambda x: int((x < 0).sum())), n_folds=("fold", "nunique"))
        agg["deg_diff"] = df[df.arm != args.ref].groupby("arm").deg_diff.mean()
        agg["noninferior"] = (agg.plate_diff >= 0) & (agg.deg_diff >= 0) & (agg.ci_below0 == 0)
        print(f"\nnon-inferiority vs {args.ref} (user rule 2026-10-31 03:20):\n" + agg.round(4).to_string())
    df.to_csv(base / "undetect_v2_summary.csv", index=False)


if __name__ == "__main__":
    main()
