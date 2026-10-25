#!/usr/bin/env python
"""Sanity checks for the EmeraldBay pseudobulk build (scripts/build_emerald_bay_pseudobulk.py).

Checks, each printed and written to <out>/sanity.json:
  1. inventory      cells / lines / conditions against the dataset card
  2. doses          every condition string parses, units are uM, combos flagged
  3. sensitivity    log2(P_c,p / P_c,DMSO) from full-QC cell counts vs the
                    released growth_rate (the paper used minimal-QC counts, so
                    agreement is expected to be high, not exact)
  4. identity       each EmeraldBay line's DMSO profile vs every Tahoe line's
                    DMSO centroid (log1p, reference-centred): is it its own
                    nearest Tahoe line?
  5. scale          EmeraldBay DMSO mean counts vs the Tahoe HVG centroid values
                    (tells us whether the Tahoe centroid is a raw-count mean)
  6. reproducibility split-half Pearson of DMSO profiles and of treatment deltas;
                    replicate-sample Pearson of deltas where replicates exist
  7. cross-dataset  EmeraldBay 5-day delta vs Tahoe 24-h delta for shared
                    (line, drug), matched vs mismatched-drug null
  8. biology        MEK inhibitor (Trametinib) should repress MAPK-output genes
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MAPK_OUTPUT = ["DUSP4", "DUSP6", "SPRY1", "SPRY2", "SPRY4", "ETV4", "ETV5", "EGR1", "PHLDA1", "IER3", "FOSL1", "CCND1", "EPHA2", "MYC"]


def parse_condition(text: str) -> list[tuple[str, float, str]]:
    return [tuple(x) for x in ast.literal_eval(text)]


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--tahoe-root", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/raw/tahoe-de-rhaister"))
    ap.add_argument("--tahoe-cache", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/processed/rhaister_o_compact"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    R: dict = {}

    groups = pd.read_parquet(args.pb / "groups.parquet")
    genes = json.load(open(args.pb / "genes.json"))
    gi = {g: i for i, g in enumerate(genes)}
    S = np.load(args.pb / "sums_log1p.npy")          # (G, 2, P) sum log1p(count)
    C = np.load(args.pb / "sums_counts.npy")         # (G, 2, P) sum counts
    nh = groups[["n_cells_h0", "n_cells_h1"]].to_numpy().astype(float)

    # 1. inventory ---------------------------------------------------------
    R["inventory"] = {
        "cells": int(groups.n_cells.sum()), "card_cells": 1831648,
        "lines": int(groups.cell_line.nunique()), "samples": int(groups["sample"].nunique()),
        "conditions": int(groups.condition.nunique()), "drugs": int(groups.drug.nunique()),
        "dmso_samples": int(groups.loc[groups.drug == "DMSO_TF", "sample"].nunique()),
        "cells_per_line": groups.groupby("cell_line").n_cells.sum().describe().round(1).to_dict(),
        "cells_per_line_condition": groups.groupby(["cell_line", "condition"]).n_cells.sum().describe().round(1).to_dict(),
        "library_mean_median": float(groups.library_mean.median()),
        "panel_fraction_median": float(groups.panel_fraction.median()),
    }

    # 2. doses --------------------------------------------------------------
    conds = groups.condition.unique()
    parsed = {c: parse_condition(c) for c in conds}
    units = sorted({u for v in parsed.values() for (_, _, u) in v})
    R["doses"] = {
        "units": units, "n_single": sum(len(v) == 1 for v in parsed.values()),
        "n_combo": sum(len(v) > 1 for v in parsed.values()),
        "per_drug_doses": {d: sorted({x[1] for v in parsed.values() if len(v) == 1 for x in v if x[0] == d})
                           for d in sorted({x[0] for v in parsed.values() if len(v) == 1 for x in v})},
    }

    # 3. sensitivity from cell counts --------------------------------------
    n = groups.groupby(["condition", "cell_line"]).n_cells.sum().unstack(fill_value=0)
    P = n.div(n.sum(1), axis=0)
    ctrl = [c for c in P.index if parse_condition(c)[0][0] == "DMSO_TF"]
    pc = P.loc[ctrl].mean(0)
    sens = np.log2((P + 1e-9).div(pc + 1e-9, axis=1)).stack().rename("sens_fullqc").reset_index()
    ss = pd.read_parquet(args.raw / "metadata" / "summary_statistics.parquet")
    ss = ss.groupby(["cell_line", "condition"], as_index=False).growth_rate.mean()
    m = sens.merge(ss, on=["cell_line", "condition"])
    m = m[np.isfinite(m.sens_fullqc) & m.growth_rate.notna() & ~m.condition.str.contains("DMSO")]
    R["sensitivity"] = {
        "n_pairs": int(len(m)),
        "pearson": float(np.corrcoef(m.sens_fullqc, m.growth_rate)[0, 1]),
        "spearman": float(m[["sens_fullqc", "growth_rate"]].corr("spearman").iloc[0, 1]),
        "ss_conditions_not_in_expression": sorted(set(ss.condition) - set(groups.condition))[:20],
    }

    # DMSO per line (pooled over DMSO samples), raw mean counts and mean log1p
    dm = groups.drug == "DMSO_TF"
    lines = sorted(groups.cell_line.unique())
    li = {c: i for i, c in enumerate(lines)}
    def pooled(mask, arr):
        out = np.zeros((len(lines), arr.shape[2])); cnt = np.zeros(len(lines))
        for g in np.where(mask)[0]:
            out[li[groups.cell_line.iat[g]]] += arr[g].sum(0); cnt[li[groups.cell_line.iat[g]]] += nh[g].sum()
        return out / np.maximum(cnt, 1)[:, None], cnt
    eb_dmso_counts, eb_dmso_n = pooled(dm.to_numpy(), C)

    # 4/5. identity + scale vs Tahoe ----------------------------------------
    tc = pd.read_parquet(args.tahoe_root / "zeroshot" / "cell_line_centroids_dmso_hvg.parquet")
    t_line = tc.groupby("cell_line_id")[genes].mean()
    shared = [c for c in lines if c in t_line.index]
    X_eb = np.log1p(eb_dmso_counts[[li[c] for c in shared]])
    X_t = np.log1p(t_line.to_numpy())
    X_eb_c = X_eb - X_eb.mean(0)
    X_t_c = X_t - X_t.mean(0)
    corr = np.corrcoef(np.vstack([X_eb_c, X_t_c]))[: len(shared), len(shared):]
    t_names = list(t_line.index)
    ranks = {}
    for i, c in enumerate(shared):
        order = np.argsort(-corr[i])
        ranks[c] = int(np.where(np.array(t_names)[order] == c)[0][0]) + 1
    R["identity"] = {
        "n_shared": len(shared), "n_tahoe": len(t_names),
        "rank1": int(sum(r == 1 for r in ranks.values())),
        "not_rank1": {c: r for c, r in ranks.items() if r != 1},
        "self_corr_median": float(np.median([corr[i, t_names.index(c)] for i, c in enumerate(shared)])),
    }
    ratio = eb_dmso_counts[[li[c] for c in shared]].mean() / t_line.loc[shared].to_numpy().mean()
    R["scale"] = {"eb_dmso_mean_count": float(eb_dmso_counts.mean()), "tahoe_centroid_mean": float(t_line.to_numpy().mean()),
                  "ratio_eb_over_tahoe_shared": float(ratio)}

    # 6. reproducibility -----------------------------------------------------
    # DMSO split-half
    dh = np.zeros((len(lines), 2, len(genes))); dn = np.zeros((len(lines), 2))
    for g in np.where(dm)[0]:
        dh[li[groups.cell_line.iat[g]]] += S[g]; dn[li[groups.cell_line.iat[g]]] += nh[g]
    dmso_log_half = dh / np.maximum(dn, 1)[:, :, None]
    dmso_log = dh.sum(1) / np.maximum(dn.sum(1), 1)[:, None]
    ref = dmso_log.mean(0)
    r_dmso = pearson_rows(dmso_log_half[:, 0] - ref, dmso_log_half[:, 1] - ref)
    # treatment deltas: pool samples per (line, condition), split-half
    tr = groups[~dm].copy()
    tr["key"] = tr.cell_line + "||" + tr.condition
    keys = sorted(tr.key.unique()); ki = {k: i for i, k in enumerate(keys)}
    th = np.zeros((len(keys), 2, len(genes))); tn = np.zeros((len(keys), 2))
    for g, k in zip(tr.index, tr.key):
        th[ki[k]] += S[g]; tn[ki[k]] += nh[g]
    t_log_half = th / np.maximum(tn, 1)[:, :, None]
    t_log = th.sum(1) / np.maximum(tn.sum(1), 1)[:, None]
    kl = np.array([li[k.split("||")[0]] for k in keys])
    delta = t_log - dmso_log[kl]
    d_h = [t_log_half[:, h] - dmso_log_half[kl, h] for h in (0, 1)]
    r_half = pearson_rows(d_h[0], d_h[1])
    ncell = tn.sum(1)
    # replicate samples of the same (line, condition)
    rep = tr.groupby("key")["sample"].nunique()
    rep_keys = rep[rep >= 2].index
    r_rep = []
    for k in rep_keys:
        gs = tr.index[tr.key == k][:2]
        a = S[gs[0]].sum(0) / nh[gs[0]].sum() - dmso_log[li[k.split("||")[0]]]
        b = S[gs[1]].sum(0) / nh[gs[1]].sum() - dmso_log[li[k.split("||")[0]]]
        r_rep.append(float(pearson_rows(a[None], b[None])[0]))
    R["reproducibility"] = {
        "dmso_split_half_centred_pearson_median": float(np.nanmedian(r_dmso)),
        "dmso_split_half_worst5": {lines[i]: round(float(r_dmso[i]), 3) for i in np.argsort(r_dmso)[:5]},
        "delta_split_half_pearson_median": float(np.nanmedian(r_half)),
        "delta_split_half_by_ncells_quartile": pd.Series(r_half).groupby(pd.qcut(ncell, 4, labels=False, duplicates="drop")).median().round(3).to_dict(),
        "n_line_conditions": len(keys),
        "n_with_replicate_samples": int(len(rep_keys)),
        "replicate_sample_delta_pearson_median": float(np.median(r_rep)) if r_rep else None,
    }

    # 7. cross-dataset delta concordance for shared drugs --------------------
    obs = pd.read_parquet(args.tahoe_cache / "observations.parquet")
    tdel = np.load(args.tahoe_cache / "delta.npy", mmap_mode="r")
    obs["drug"] = obs.treatment.map(lambda s: parse_condition(s)[0][0].lower() if len(parse_condition(s)) == 1 else None)
    obs["dose"] = obs.treatment.map(lambda s: parse_condition(s)[0][1])
    eb_meta = pd.DataFrame({"key": keys, "line": [k.split("||")[0] for k in keys], "cond": [k.split("||")[1] for k in keys]})
    eb_meta["single"] = eb_meta.cond.map(lambda s: len(parse_condition(s)) == 1)
    eb_meta["drug"] = eb_meta.cond.map(lambda s: parse_condition(s)[0][0].lower())
    eb_meta["dose"] = eb_meta.cond.map(lambda s: parse_condition(s)[0][1])
    shared_drugs = sorted(set(eb_meta.drug[eb_meta.single]) & set(obs.drug.dropna()))
    rows = []
    t_index = {(c, d): i for i, (c, d) in enumerate(zip(obs.cell_line, obs.drug))}
    t_by = obs.reset_index().groupby(["cell_line", "drug"])
    for i, r in eb_meta[eb_meta.single & eb_meta.drug.isin(shared_drugs)].iterrows():
        if (r.line, r.drug) not in t_by.groups:
            continue
        cand = obs.loc[t_by.groups[(r.line, r.drug)]]
        j = cand.index[np.argmin(np.abs(np.log10(cand.dose) - np.log10(r.dose)))]  # nearest dose
        # mismatched drug null: same line, other shared drugs
        others = obs.index[(obs.cell_line == r.line) & obs.drug.isin(shared_drugs) & (obs.drug != r.drug)]
        rm = pearson_rows(np.repeat(delta[i][None], len(others), 0), np.asarray(tdel[others])) if len(others) else np.array([np.nan])
        rows.append({"line": r.line, "drug": r.drug, "eb_dose": r.dose, "tahoe_dose": float(obs.dose[j]),
                     "r_matched": float(pearson_rows(delta[i][None], np.asarray(tdel[j])[None])[0]),
                     "r_mismatched_mean": float(np.nanmean(rm))})
    cd = pd.DataFrame(rows)
    cd.to_csv(args.out / "cross_dataset_delta_concordance.csv", index=False)
    R["cross_dataset"] = {
        "shared_drugs": shared_drugs, "n_pairs": int(len(cd)),
        "matched_median": float(cd.r_matched.median()) if len(cd) else None,
        "mismatched_median": float(cd.r_mismatched_mean.median()) if len(cd) else None,
        "frac_matched_gt_mismatched": float((cd.r_matched > cd.r_mismatched_mean).mean()) if len(cd) else None,
        "by_drug_matched_median": cd.groupby("drug").r_matched.median().round(3).to_dict() if len(cd) else {},
    }

    # 8. biology: Trametinib on MAPK output genes ---------------------------
    mk = [g for g in MAPK_OUTPUT if g in gi]
    tm = eb_meta.index[eb_meta.single & (eb_meta.drug == "trametinib")]
    if len(tm):
        md = delta[tm][:, [gi[g] for g in mk]]
        R["biology_trametinib"] = {
            "genes_in_panel": mk,
            "mean_delta_per_gene": dict(zip(mk, md.mean(0).round(3).tolist())),
            "frac_negative_per_gene": dict(zip(mk, (md < 0).mean(0).round(2).tolist())),
            "panel_mean_delta": float(delta[tm].mean()),
        }

    # save the deltas for downstream use
    eb_meta["n_cells"] = ncell
    eb_meta["split_half_r"] = r_half
    eb_meta.to_parquet(args.out / "line_condition_meta.parquet", index=False)
    np.save(args.out / "delta_log1p.npy", delta.astype(np.float32))
    pd.DataFrame(dmso_log, index=lines, columns=genes).to_parquet(args.out / "dmso_log1p_by_line.parquet")
    pd.DataFrame(eb_dmso_counts, index=lines, columns=genes).to_parquet(args.out / "dmso_meancount_by_line.parquet")
    json.dump(R, open(args.out / "sanity.json", "w"), indent=2, default=str)
    print(json.dumps(R, indent=2, default=str))


if __name__ == "__main__":
    main()
