#!/usr/bin/env python3
"""Does chemical similarity predict transcriptional similarity on L1000? (Q1, descriptive)

Pre-registered 2026-10-03 in report/10_09_record.md. No model fitting. Per slice and
line, among ACTIVE compounds (mean TAS >= 0.2):

  pairs        Spearman(similarity, response Pearson) per encoder; AUROC of similarity for
               response-similar pairs (r >= 0.5); P(r >= 0.5 | similarity bin)
  musa         P(r >= 0.5 | Morgan Tanimoto >= 0.85, different ik14) vs the base rate
  stereo       response r of same-ik14 pairs (stereoisomers / same connectivity)
  knowledge    P(r >= 0.5) for pairs sharing a target / a MOA label vs not (KNOWN only)
  density      per active compound: best response match anywhere in the line's library
               (oracle twin, different ik14) vs the response r of its chemical nearest
               neighbour; the share of compounds with any twin at r >= 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

import run_l1000_unseen_drug as run

SIM_THRESHOLD = 0.5
MAX_PAIRS = 2_000_000


def main(args: argparse.Namespace) -> None:
    d = args.derived
    sl = np.load(d / f"slice_{args.slice}.npz", allow_pickle=False)
    Y_all = sl["Y"].astype(np.float64); keys = sl["inchikey"].astype(str); line_all = sl["line"].astype(str)
    tas = sl["tas"]
    compounds = np.array(sorted(set(keys)))
    cidx = {k: i for i, k in enumerate(compounds)}
    row_c = np.array([cidx[k] for k in keys])
    table = pd.read_parquet(d / "drug_table.parquet")
    table = table[table.source.eq("L1000") & table.inchikey.isin(compounds)]
    ik14 = table.groupby("inchikey").ik14.first().reindex(compounds).astype(str).to_numpy()
    targets = {k: set() for k in compounds}; moas = {k: set() for k in compounds}
    for r in table.itertuples():
        if r.targets:
            targets[r.inchikey].update(r.targets.split(";"))
        if r.moa:
            moas[r.inchikey].update(r.moa.split(";"))

    feats = np.load(d / "drug_features.npz", allow_pickle=False)
    fpos = pd.Series(np.arange(len(feats["inchikey"])), index=feats["inchikey"].astype(str))[compounds].to_numpy()
    enc = {"morgan2": ("tanimoto", feats["morgan2"][fpos]), "maccs": ("tanimoto", feats["maccs"][fpos])}
    X = feats["rdkit2d"][fpos].astype(np.float64)
    X = np.nan_to_num((X - np.nanmean(X, 0)) / np.maximum(np.nanstd(X, 0), 1e-9), nan=0.0).clip(-10, 10)
    enc["rdkit2d"] = ("cosine", X)
    for lm in ("chemberta", "molformer"):
        e = np.load(d / f"smiles_lm_{lm}.npz", allow_pickle=False)
        pos = pd.Series(np.arange(len(e["inchikey"])), index=e["inchikey"].astype(str))[compounds].to_numpy()
        E = e["embedding"][pos].astype(np.float64)
        enc[lm] = ("cosine", E - E.mean(0))

    rng = np.random.default_rng(0)
    pair_rows, bin_rows, density_rows, other_rows = [], [], [], []
    for L in sorted(set(line_all)):
        rows = np.where(line_all == L)[0]
        cpos = row_c[rows]
        act = tas[rows] >= 0.2
        a_rows, a_c = rows[act], cpos[act]
        C = run.corr_matrix(Y_all[a_rows], Y_all[a_rows]).astype(np.float64)
        iu = np.triu_indices(len(a_rows), 1)
        same14 = ik14[a_c][iu[0]] == ik14[a_c][iu[1]]
        r_pairs = C[iu]
        diff = ~same14
        pick = np.where(diff)[0]
        if len(pick) > MAX_PAIRS:
            pick = rng.choice(pick, MAX_PAIRS, replace=False)
        base = float((r_pairs[diff] >= SIM_THRESHOLD).mean())
        other_rows.append({"line": L, "n_active": int(act.sum()), "n_pairs": int(diff.sum()),
                           "base_rate_r05": base, "n_stereo_pairs": int(same14.sum()),
                           "stereo_median_r": float(np.median(r_pairs[same14])) if same14.any() else np.nan,
                           "stereo_rate_r05": float((r_pairs[same14] >= SIM_THRESHOLD).mean()) if same14.any() else np.nan,
                           "random_median_r": float(np.median(r_pairs[diff]))})
        simmats = {}
        for name, (kind, F) in enc.items():
            F = F[a_c]
            S = run.tanimoto(F, F) if kind == "tanimoto" else run.cosine(F, F)
            simmats[name] = S
            s = S[iu][pick]; r = r_pairs[pick]
            pair_rows.append({"line": L, "encoder": name, "spearman": float(spearmanr(s, r).statistic),
                              "auroc_r05": float(roc_auc_score(r >= SIM_THRESHOLD, s)),
                              "rate_top1pct": float((r[s >= np.quantile(s, 0.99)] >= SIM_THRESHOLD).mean()),
                              "rate_top01pct": float((r[s >= np.quantile(s, 0.999)] >= SIM_THRESHOLD).mean())})
            if kind == "tanimoto":
                for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .85), (.85, 1.01)):
                    m = (s >= lo) & (s < hi)
                    bin_rows.append({"line": L, "encoder": name, "bin": f"[{lo},{hi})", "n": int(m.sum()),
                                     "rate_r05": float((r[m] >= SIM_THRESHOLD).mean()) if m.any() else np.nan,
                                     "median_r": float(np.median(r[m])) if m.any() else np.nan})
        # knowledge pairs among KNOWN actives
        kmask = np.array([bool(targets[compounds[c]]) for c in a_c])
        if kmask.sum() > 10:
            ki = np.where(kmask)[0]
            kiu = np.triu_indices(len(ki), 1)
            a, b = ki[kiu[0]], ki[kiu[1]]
            ok = ik14[a_c][a] != ik14[a_c][b]
            a, b = a[ok], b[ok]
            r = C[a, b]
            share_t = np.array([bool(targets[compounds[a_c[x]]] & targets[compounds[a_c[y]]]) for x, y in zip(a, b)])
            share_m = np.array([bool(moas[compounds[a_c[x]]] & moas[compounds[a_c[y]]]) for x, y in zip(a, b)])
            tan = simmats["morgan2"][a, b]
            for label, mask in (("share_target", share_t), ("no_shared_target", ~share_t),
                                ("share_moa", share_m), ("no_shared_moa", ~share_m),
                                ("tanimoto>=0.85", tan >= 0.85), ("known_all", np.ones(len(r), bool))):
                bin_rows.append({"line": L, "encoder": "knowledge", "bin": label, "n": int(mask.sum()),
                                 "rate_r05": float((r[mask] >= SIM_THRESHOLD).mean()) if mask.any() else np.nan,
                                 "median_r": float(np.median(r[mask])) if mask.any() else np.nan})
            other_rows[-1].update({
                "known_auroc_target_jaccard": float(roc_auc_score(r >= SIM_THRESHOLD, share_t)) if share_t.any() else np.nan,
                "known_auroc_morgan": float(roc_auc_score(r >= SIM_THRESHOLD, tan)),
            })
        # density: twins in the whole line library (actives as queries, everything as candidates)
        Cfull = run.corr_matrix(Y_all[a_rows], Y_all[rows]).astype(np.float64)
        excl = ik14[a_c][:, None] == ik14[cpos][None, :]
        Cfull[excl] = -np.inf
        best = Cfull.max(1)
        for name, (kind, F) in enc.items():
            S = run.tanimoto(F[a_c], F[cpos]) if kind == "tanimoto" else run.cosine(F[a_c], F[cpos])
            S = S.astype(np.float64); S[excl] = -np.inf
            nn = S.argmax(1)
            r_nn = Cfull[np.arange(len(a_c)), nn]
            density_rows.append({"line": L, "encoder": name, "median_best_twin_r": float(np.median(best)),
                                 "share_twin_r05": float((best >= SIM_THRESHOLD).mean()),
                                 "median_chemNN_r": float(np.median(r_nn)),
                                 "share_chemNN_r05": float((r_nn >= SIM_THRESHOLD).mean()),
                                 "median_chemNN_sim": float(np.median(S[np.arange(len(a_c)), nn]))})
        print(f"{L}: {int(act.sum())} active; base rate {base:.4f}", flush=True)

    out = args.out / args.slice
    out.mkdir(parents=True, exist_ok=True)
    tables = {"pairs": pd.DataFrame(pair_rows), "bins": pd.DataFrame(bin_rows),
              "density": pd.DataFrame(density_rows), "lines": pd.DataFrame(other_rows)}
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    print(tables["lines"].to_string())
    print(tables["pairs"].groupby("encoder")[["spearman", "auroc_r05", "rate_top1pct", "rate_top01pct"]].mean().to_string())
    print(tables["bins"].groupby(["encoder", "bin"])[["n", "rate_r05", "median_r"]].mean().to_string())
    print(tables["density"].groupby("encoder").mean(numeric_only=True).to_string())
    (out / "manifest.json").write_text(json.dumps({"slice": args.slice, "threshold": SIM_THRESHOLD,
                                                   "runner_sha256": run.sha256(Path(__file__).resolve())}, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base = Path("/nfs-stor/yuqi.lei/Therapeutic")
    p.add_argument("--slice", default="P1-24", choices=["P1-24", "P1-6", "P2-24"])
    p.add_argument("--derived", type=Path, default=base / "derived/l1000_unseen_drug")
    p.add_argument("--out", type=Path, default=base / "results/l1000_structure_response")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
