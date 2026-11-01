#!/usr/bin/env python3
"""View-weight variants of the cross-line model vs pool_D (`run_l1000_xline.py --kernels T1 D Dml Dl1000`).

For each slice: line-macro ACTIVE mean of pearson_full and pearson_resid (prediction and truth both minus the
training line mean: the drug-specific part, the analogue of PCC_plate), paired differences vs pool_D with the
2,000-replicate bootstrap over ik14 groups (`run_l1000_unseen_drug.line_macro_boot`), and lines with a positive
difference. Also checks that the D arms reproduce the earlier `_D` run where it exists. Writes
results/l1000_xline/weights_vs_D<tag>[_<split>].json (e.g. `--slices P2-24 P1-24 --tag _D` for mkl_D vs pool_D on LINCS;
`--tag _Dgate [--split platemap]` for the drug × cell-line gate arms of `run_l1000_xline.py --gate D`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_l1000_unseen_drug as run

RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/l1000_xline")
ARMS = ("pool_Dml", "pool_Dl1000", "mkl_Dml", "mkl_Dl1000", "mkl_D", "gonly_Dml", "gonly_Dl1000", "gonly_D",
        "gate_line_D", "gate_scale_D")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--slices", nargs="+", default=["T-0.05", "T-0.5", "T-5"])
    p.add_argument("--tag", default="_Dw"); p.add_argument("--ref", default="pool_D"); p.add_argument("--split", default="ik14")
    args = p.parse_args()
    out, rows = {}, []
    for s in args.slices:
        d = RES / f"{s}_{args.split}_seed0{args.tag}"
        fr = pd.read_parquet(d / "per_condition.parquet"); a = fr[fr.active]
        key = ["line", "inchikey", "fold"]
        ref = a[a.arm == args.ref].set_index(key)
        sub = a[a.arm == args.ref].copy()
        macro = a.groupby(["arm", "line"])[["pearson_full", "pearson_resid"]].mean().groupby("arm").mean()
        res = {"macro": macro.round(6).to_dict(orient="index"), "contrasts": {}}
        old = RES / f"{s}_{args.split}_seed0_D" / "per_condition.parquet"
        if old.exists():
            o = pd.read_parquet(old); o = o[o.active & o.arm.isin(["pool_D", "mkl_D"])].set_index(["arm"] + key)
            n = fr[fr.active & fr.arm.isin(["pool_D", "mkl_D"])].set_index(["arm"] + key).reindex(o.index)
            res["max_abs_diff_vs_old_D"] = float(np.nanmax(np.abs(n.pearson_full.to_numpy() - o.pearson_full.to_numpy())))
        for arm in ARMS:
            if arm == args.ref or not (a.arm == arm).any():
                continue
            x = a[a.arm == arm].set_index(key).reindex(ref.index)
            for metric in ("pearson_full", "pearson_resid"):
                diff = (x[metric] - ref[metric]).to_numpy()
                mean, ci, per_line = run.line_macro_boot(sub, diff, 2000, 0)
                c = {"mean": mean, "lo": float(ci[0]), "hi": float(ci[1]),
                     "lines_pos": int(sum(v > 0 for v in per_line.values())), "lines": len(per_line)}
                res["contrasts"][f"{arm}|{metric}"] = c
                rows.append({"slice": s, "arm": arm, "metric": metric.replace("pearson_", ""), **c})
        sel = pd.read_csv(d / "selection.csv", low_memory=False)
        g = sel[sel.model.isin(["g_D", "g_Dml", "g_Dl1000"])]
        gl = sel[sel.model.eq("D")].filter(regex="^gate_line") if "gate_line_c" in sel else None
        if gl is not None:
            res["gate_line_c"] = gl.gate_line_c.value_counts().to_dict()
            res["gate_scale_c"] = sel[sel.model.eq("gate_scale_D")].gate_scale_c.tolist()
            print(f"gate c chosen: per line {res['gate_line_c']}; shared per fold {res['gate_scale_c']}")
        res["g_weights"] = g[[c for c in g.columns if c in ("fold", "model", "lambda") or c.startswith(("gw_", "w_", "ml_amp", "ml_noise"))]].to_dict(orient="records")
        out[s] = res
        print(f"\n== {s}: line-macro ACTIVE\n" + macro.round(4).to_string())
        if "max_abs_diff_vs_old_D" in res:
            print(f"pool_D/mkl_D vs earlier _D run: max |diff| {res['max_abs_diff_vs_old_D']:.2e}")
        print(g.filter(regex="^(fold|model|lambda|gw_|ml_amp|ml_noise_const)").round(3).to_string(index=False))
    t = pd.DataFrame(rows); pd.set_option("display.width", 200)
    print(f"\nvs {args.ref}:\n" + t.round(4).to_string(index=False))
    sfx = "" if args.split == "ik14" else f"_{args.split}"
    (RES / f"weights_vs_D{args.tag}{sfx}.json").write_text(json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
