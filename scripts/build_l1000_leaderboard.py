#!/usr/bin/env python3
"""One leaderboard for every model scored on the L1000 unseen-drug task (runner, GNN/MLP, MKL, end-to-end).

Every arm is scored only on the cohorts whose rows carry the information it needs (NEEDS: targets -> KNOWN_ACTIVE,
Cell Painting -> MORPH_ACTIVE, response in other lines -> XCELL_ACTIVE; arms needing nothing extra are scored on
all four, so each cohort has its own chemistry and mean reference). Outside its cohort an arm falls back to the
mean or to chemistry, so scoring it on all ACTIVE rows would understate it. An arm must cover >= 95% of the rows. On exactly those rows it is paired with the
runner's krr_morgan2 (the chemistry bar), mean and oracle_knn. Line-macro full Pearson; difference vs KRR with a
95% bootstrap over the split's groups resampled jointly across lines (2,000 replicates, seed 0), as in
compare_l1000_*.py. headroom = (arm - mean) / (oracle - mean) on the same rows.

Writes results/l1000_leaderboard/v1/leaderboard.csv and tracker.json (rows for the tracker's `leaderboard` collection).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/nfs-stor/yuqi.lei/Therapeutic/results")
OUT = R / "l1000_leaderboard" / "v1"
COHORTS = ["ACTIVE", "KNOWN_ACTIVE", "MORPH_ACTIVE", "XCELL_ACTIVE"]
KEY = ["line", "inchikey"]

# (source dir under results/, run tag shown, family, {arm: label}); smokes and superseded runs are left out
FAMILY = {
    "mean": "baseline", "krr_morgan2": "chemistry", "knn_morgan2": "chemistry", "knn_chem_select": "chemistry",
    "knn_rdkit2d": "chemistry", "knn_maccs": "chemistry", "knn_chemberta": "chemistry (SMILES LM)",
    "knn_molformer": "chemistry (SMILES LM)", "knn_target": "targets", "knn_target_perm": "targets (null)",
    "krr_target": "targets", "krr_chem_target": "chemistry + targets", "knn_chem_target": "chemistry + targets",
    "moa_knn": "MOA", "knn_ppi": "PPI", "knn_ppi_rewired": "PPI (null)", "krr_ppi": "PPI",
    "krr_ppi_rewired": "PPI (null)", "knn_morph": "Cell Painting", "knn_chem_morph": "chemistry + Cell Painting",
    "xcell1_knn": "response in 1 other line", "xcell_knn": "response in all other lines",
    "oracle_knn": "ceiling (oracle)",
}
RUNNER_ARMS = list(FAMILY)
KNOWN = {"knn_target", "knn_target_perm", "krr_target", "krr_chem_target", "knn_chem_target", "moa_knn", "knn_ppi",
         "knn_ppi_rewired", "krr_ppi", "krr_ppi_rewired", "krr_tgt", "krr_tgt_null", "krr_moa", "krr_ppi1",
         "krr_ppi1_null", "krr_deg", "mkl_T2", "mkl_T2null", "gnn_ppi", "gnn_rewired", "gnn_nograph", "mlp_target",
         "dot_ppi", "dot_rewired", "dot_nograph", "mlp_ppi", "mlp_ppi_rewired", "mlp_ppi_nograph", "gcn_ppi",
         "gcn_rewired", "gcn_nograph", "mlp_chem"}
MORPH = {"knn_morph", "knn_chem_morph", "krr_morph", "mkl_T3", "mkl_T3null"}
XCELL = {"xcell1_knn", "xcell_knn", "krr_xresp", "mkl_T4"}


def needs(arm):
    return ["KNOWN_ACTIVE"] if arm in KNOWN else ["MORPH_ACTIVE"] if arm in MORPH else \
        ["XCELL_ACTIVE"] if arm in XCELL else COHORTS


def load(path):
    return pd.read_parquet(path / "per_condition.parquet")


def cohort_mask(d, c):
    m = d.active.astype(bool)
    if c == "KNOWN_ACTIVE":
        m &= d.known.astype(bool)
    elif c == "MORPH_ACTIVE":
        m &= d.morph.astype(bool)
    elif c == "XCELL_ACTIVE":
        m &= d.xcell.astype(bool)
    return m


def line_macro(df, col):
    return df.groupby("line")[col].mean().mean()


def boot(df, col, B=2000, seed=0):
    g = pd.factorize(df.group)[0]
    l = pd.factorize(df.line)[0]
    G, L = g.max() + 1, l.max() + 1
    S = np.zeros((G, L)); N = np.zeros((G, L))
    np.add.at(S, (g, l), df[col].to_numpy()); np.add.at(N, (g, l), 1)
    W = np.random.default_rng(seed).multinomial(G, np.full(G, 1 / G), size=B).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = (W @ S) / (W @ N)
    est = np.nanmean(per, axis=1)
    lines_pos = int((df.groupby("line")[col].mean() > 0).sum())
    return np.nanpercentile(est, 2.5), np.nanpercentile(est, 97.5), lines_pos, L


def score(arms_df, ref, slice_, split, source, family_of, label_of):
    rows = []
    for arm, d in arms_df.groupby("arm"):
        d = d.drop_duplicates(KEY)
        for c in needs(arm):
            base = ref[cohort_mask(ref, c)]
            n_base = base[base.arm == "krr_morgan2"].shape[0]
            m = d.merge(base[base.arm == "krr_morgan2"][KEY + ["group"]], on=KEY, suffixes=("", "_r"))
            if n_base == 0 or len(m) < 0.95 * n_base:
                continue
            piv = base[base.arm.isin(["krr_morgan2", "mean", "oracle_knn"])].pivot_table(
                index=KEY, columns="arm", values="pearson_full")
            j = m[KEY + ["group", "pearson_full"]].merge(piv.reset_index(), on=KEY)
            j["d_krr"] = j.pearson_full - j.krr_morgan2
            s_arm, s_krr = line_macro(j, "pearson_full"), line_macro(j, "krr_morgan2")
            s_mean, s_orc = line_macro(j, "mean"), line_macro(j, "oracle_knn")
            lo, hi, lp, nl = boot(j, "d_krr")
            rows.append(dict(slice=slice_, split=split, source=source, arm=arm, label=label_of(arm),
                             family=family_of(arm), cohort=c, rows=len(j), score=s_arm, krr=s_krr, mean=s_mean,
                             oracle=s_orc, d_krr=s_arm - s_krr, d_krr_low=lo, d_krr_high=hi, lines_pos=lp,
                             n_lines=nl, headroom=(s_arm - s_mean) / (s_orc - s_mean),
                             needs="none" if needs(arm) == COHORTS else needs(arm)[0]))
    return rows


def main():
    out = []
    for slice_ in ("P1-24", "P2-24"):
        for split in ("ik14", "platemap"):
            ref = load(R / "l1000_unseen_drug" / f"{slice_}_{split}_seed0")
            run = ref[ref.arm.isin(RUNNER_ARMS)]
            out += score(run, ref, slice_, split, "runner", FAMILY.get, lambda a: a)
            if not out or not any(r["slice"] == slice_ and r["split"] == split for r in out):
                continue
            extra = [  # (dir, run, family, label prefix, arms taken; duplicated copies in other runs are skipped)
                ("l1000_gnn", f"{slice_}_{split}_seed0_shrink", "decoupled GNN / MLP", "",
                 {"gnn_ppi", "gnn_rewired", "gnn_nograph", "mlp_chem", "mlp_chem_all", "mlp_target"}),
                ("l1000_gnn", f"{slice_}_{split}_seed0_shrink_dot", "GNN, dot-product head", "",
                 {"dot_ppi", "dot_rewired", "dot_nograph"}),
                ("l1000_gnn", f"{slice_}_{split}_seed0_shrink_mlpppi", "MLP + PPI features", "",
                 {"mlp_ppi", "mlp_ppi_rewired", "mlp_ppi_nograph"}),
                ("l1000_gnn", f"{slice_}_{split}_seed0_gcn", "per-drug GCN (G3b)", "",
                 {"gcn_ppi", "gcn_rewired", "gcn_nograph"}),
                ("l1000_mkl", f"{slice_}_{split}_seed0", "MKL kernels / tiers", "", None),
                ("l1000_e2e", f"{slice_}_{split}_seed0_skelcpu", "end-to-end skeleton", "e2e_v0_", {"skel"}),
                ("l1000_e2e", f"{slice_}_{split}_seed0_skel_v1reg", "end-to-end skeleton", "e2e_v1reg_", {"skel"}),
            ]
            for src, tag, fam, pre, take in extra:
                p = R / src / tag
                if not (p / "per_condition.parquet").exists():
                    continue
                d = load(p)
                d = d[(d.arm != "mean") & (d.arm.isin(take) if take else True)]
                out += score(d, ref, slice_, split, f"{src}/{tag}", lambda a, f=fam: f, lambda a, pre=pre: pre + a)
    df = pd.DataFrame(out).sort_values(["slice", "split", "cohort", "score"], ascending=[True, True, True, False])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "leaderboard.csv", index=False)
    rec = [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in df.to_dict("records")]
    (OUT / "tracker.json").write_text(json.dumps(rec))
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
