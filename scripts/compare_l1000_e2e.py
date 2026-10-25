#!/usr/bin/env python3
"""Paired contrasts for the integrated end-to-end model (run_l1000_e2e.py) against itself and the kNN/KRR runner.

Rows pair on (line, inchikey); bootstrap = run.line_macro_boot over the runner's grouping column. Cohorts: ACTIVE
(all compounds) and ACTIVE∩in_kg (compound has a PrimeKG drug-disease edge). Writes contrasts.csv and summary.json
(absolute line-macro means per arm and cohort) next to the run's per_condition.parquet. --also merges the arms of
other e2e runs of the same slice (e.g. the rewired-KG null `_krr_rw0` into `_krr`); arm names must not collide.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_l1000_unseen_drug as run

BASELINES = ("mean", "krr_morgan2", "knn_morgan2", "krr_chem_target")
CONTRASTS = [("skel", "mean"), ("skel", "krr_morgan2"), ("kg_a", "skel"), ("kg_b", "skel"), ("kg_a", "kg_b"),
             ("kg_t", "kg_a"), ("kg_a", "krr_morgan2"), ("kg_b", "krr_morgan2"), ("kg_t", "skel"),
             # round 2 (KRR anchor, rewired-KG nulls; record 2026-10-11)
             ("anchor", "krr_morgan2"), ("skel", "anchor"), ("kg_a", "anchor"), ("kg_t", "anchor"),
             ("kg_t", "krr_morgan2"), ("kg_t", "krr_chem_target"), ("kg_a", "kg_a_rw0"), ("kg_t", "kg_t_rw0")]


def main(a):
    base = Path("/nfs-stor/yuqi.lei/Therapeutic/results")
    out = base / "l1000_e2e" / f"{a.run}{a.tag}"
    g = pd.read_parquet(out / "per_condition.parquet")
    for tag in a.also:
        h = pd.read_parquet(base / "l1000_e2e" / f"{a.run}{tag}" / "per_condition.parquet")
        h = h[h.arm != "mean"]
        # an arm present in both runs (the deterministic KRR anchor) is kept once, only if identical
        for arm in sorted(set(h.arm) & set(g.arm)):
            m = g[g.arm == arm].merge(h[h.arm == arm], on=["line", "inchikey"])
            assert len(m) == (g.arm == arm).sum() and np.allclose(m.pearson_full_x, m.pearson_full_y), \
                f"arm {arm} differs between runs ({tag})"
            print(f"arm {arm} identical in {tag} ({len(m)} rows); kept once")
        h = h[~h.arm.isin(set(g.arm))]
        g = pd.concat([g, h])
    k = pd.read_parquet(base / "l1000_unseen_drug" / a.run / "per_condition.parquet")
    gm = g[g.arm == "mean"].merge(k[k.arm == "mean"], on=["line", "inchikey"], suffixes=("_g", "_k"))
    agree = float(np.abs(gm.pearson_full_g - gm.pearson_full_k).max()) if len(gm) else float("nan")
    print(f"mean-arm agreement: {len(gm)} rows, max |diff| {agree:.2e}")
    flags = k[k.arm == "mean"][["line", "inchikey", "group", "active"]].merge(
        g[g.arm == "mean"][["line", "inchikey", "in_kg", "known"]], on=["line", "inchikey"])
    cohorts = {"ACTIVE": flags.active, "ACTIVE_in_kg": flags.active & flags.in_kg,
               "ACTIVE_known": flags.active & flags.known}
    cols = ["arm", "line", "inchikey", "pearson_full", "pearson_resid"]
    allp = pd.concat([g[g.arm != "mean"][cols], k[k.arm.isin(BASELINES)][cols]])
    arms = sorted(allp.arm.unique())
    summary = {"run": f"{a.run}{a.tag}", "also": a.also, "mean_agreement": agree, "absolute": {}}
    for cohort, sel in cohorts.items():
        f = flags[sel]
        summary["absolute"][cohort] = {}
        for arm in arms:
            m = allp[allp.arm == arm].merge(f, on=["line", "inchikey"])
            if len(m):
                summary["absolute"][cohort][arm] = {
                    met: float(m.groupby("line")[met].mean().mean()) for met in ("pearson_full", "pearson_resid")}
                summary["absolute"][cohort][arm]["rows"] = int(len(m))
    rows = []
    for arm_a, arm_b in CONTRASTS:
        A = allp[allp.arm == arm_a]; B = allp[allp.arm == arm_b]
        if A.empty or B.empty:
            continue
        both = A.merge(B, on=["line", "inchikey"], suffixes=("_a", "_b")).merge(flags, on=["line", "inchikey"])
        for cohort, m in (("ACTIVE", both[both.active]), ("ACTIVE_in_kg", both[both.active & both.in_kg]),
                          ("ACTIVE_known", both[both.active & both.known])):
            if m.empty:
                continue
            for metric in ("pearson_full", "pearson_resid"):
                est, ci, pl = run.line_macro_boot(m, (m[f"{metric}_a"] - m[f"{metric}_b"]).to_numpy(), 2000, 0)
                rows.append({"arm": arm_a, "minus": arm_b, "cohort": cohort, "metric": metric, "diff": est,
                             "low": ci[0], "high": ci[1], "lines_positive": sum(v > 0 for v in pl.values()),
                             "n_lines": len(pl), "rows": len(m)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "contrasts.csv", index=False)
    summary["contrasts"] = res.to_dict(orient="records")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with pd.option_context("display.width", 200):
        print(pd.DataFrame({c: {arm: v["pearson_full"] for arm, v in d.items()} for c, d in summary["absolute"].items()}).round(4))
        print(res[res.metric == "pearson_full"].round(4).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="P1-24_ik14_seed0")
    p.add_argument("--tag", default="")
    p.add_argument("--also", nargs="*", default=[], help="tags of further e2e runs whose arms are merged in")
    main(p.parse_args())
