#!/usr/bin/env python3
"""Where and why XPert's compound-specific signal differs from UnDetect's (drug-blind benchmark; no training).

Inputs: results/sdst_hybrid/published/fold_*/pred_test.npz (`prep_published_preds.py`: released XPert / TranSiGen
predictions next to our v3 arms). All scores on shared-control test rows with the plate mean m as in
`analyze_sdst_centred.py` (PCC_plate = Pearson of (y − m) vs (p − m)).
1. Complementarity: correlation of the two models' centred predictions; blend (1 − w)·UnDetect + w·XPert with w chosen
   on the neighbouring fold's test rows (= fold f's validation compounds), as for the shrunk oracle.
2. Per-compound Δ (XPert − UnDetect) PCC_plate by compound attributes: target / Cell Painting annotation, nearest
   training-compound Morgan Tanimoto, number of test lines. Bootstrap over compounds, pooled over folds.
3. Line-specific vs shared part: each model's centred prediction replaced by its own mean over the compound's test
   rows (compounds with >= 2 shared rows); the drop measures how much signal its line-specific deviations carry.
Writes results/sdst_hybrid/published/xpert_vs_undetect.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sdst_hybrid import drug_kernels  # noqa: E402

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/published")
WS = tuple(np.round(np.arange(0, 1.01, 0.1), 1))
ARMS = ("nnk_multi_norm", "pool_multi_norm", "xpert", "transigen", "mlp")


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def boot_mean(v: np.ndarray, rng, n=2000) -> tuple[float, float, float]:
    idx = rng.integers(0, len(v), (n, len(v))); b = v[idx].mean(1)
    return float(v.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main() -> None:
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, pidx, cell = z["ctl"], z["delta"].astype(np.float64), z["group"], z["pert_idx"], z["cell_idx"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    table = pd.read_csv(D / "drugs.tsv", sep="\t", keep_default_na=False)
    rowof = {d: i for i, d in enumerate(table.pert_idx)}; drow = np.array([rowof[d] for d in pidx])
    K, feats = drug_kernels(table); has = feats["has"]; Kc = K["chem"]
    rng = np.random.default_rng(0); S, comp = {}, []
    for f in range(5):
        te = g == f; tr = (g != f) & (g != (f + 1) % 5)
        P = np.load(RES / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        n = np.bincount(key[tr], minlength=key.max() + 1); Sm = np.zeros((key.max() + 1, Y.shape[1]))
        np.add.at(Sm, key[tr], Y[tr]); M = Sm / np.maximum(n, 1)[:, None]
        sh = n[key[te]] > 0; Yt = Y[te][sh]; m = M[key[te]][sh]; d_t = drow[te][sh]; l_t = cell[te][sh]
        pr = {a: P[a].astype(np.float64)[sh] for a in ARMS}
        c = {a: pr[a] - m for a in ARMS}; yc = Yt - m
        s = {"plate": {a: pcc(yc, c[a]) for a in ARMS}, "deg": {a: pcc(Yt, pr[a]) for a in ARMS}}
        s["corr_centred_xpert_undetect"] = float(pcc(c["xpert"], c["nnk_multi_norm"]).mean())
        s["corr_centred_xpert_mlp"] = float(pcc(c["xpert"], c["mlp"]).mean())
        s["corr_centred_undetect_mlp"] = float(pcc(c["nnk_multi_norm"], c["mlp"]).mean())
        for w in WS:
            B = (1 - w) * pr["nnk_multi_norm"] + w * pr["xpert"]
            s["plate"][f"blend{w}"] = pcc(yc, B - m); s["deg"][f"blend{w}"] = pcc(Yt, B)
        # 3. compound-mean of each model's centred prediction (compounds with >= 2 shared rows)
        ud, inv, cnt = np.unique(d_t, return_inverse=True, return_counts=True); multi = cnt[inv] >= 2
        for a in ARMS:
            cm = np.zeros((len(ud), Y.shape[1])); np.add.at(cm, inv, c[a]); cm /= cnt[:, None]
            s["plate"][f"{a}_cmean"] = np.where(multi, pcc(yc, cm[inv]), np.nan)
        cmt = np.zeros((len(ud), Y.shape[1])); np.add.at(cmt, inv, yc)
        loo = (cmt[inv] - yc) / np.maximum(cnt[inv] - 1, 1)[:, None]  # truth: mean of the compound's OTHER rows
        s["plate"]["truth_other_rows_mean"] = np.where(multi, pcc(yc, loo), np.nan)
        s["multi"] = multi
        S[f] = s
        # 2. per-compound attributes
        trd = np.unique(drow[tr]); maxsim = Kc[np.ix_(ud, trd)].max(1)
        dif = s["plate"]["xpert"] - s["plate"]["nnk_multi_norm"]
        per = pd.DataFrame({"d": d_t, "dif": dif, "x": s["plate"]["xpert"], "u": s["plate"]["nnk_multi_norm"]}).groupby("d").mean()
        per["fold"] = f; per["maxsim_chem"] = pd.Series(maxsim, index=ud).reindex(per.index)
        per["has_tgt"] = has["tgt"][per.index]; per["has_morph"] = has["morph"][per.index]
        per["n_rows"] = pd.Series(cnt, index=ud).reindex(per.index)
        comp.append(per)
    out = {"per_fold": {}, "blend": [], "strata": [], "line_specific": {}}
    for f in range(5):
        s = S[f]; out["per_fold"][f] = {k: float(np.nanmean(v)) for k, v in s["plate"].items()}
        out["per_fold"][f].update({k: s[k] for k in s if k.startswith("corr_")})
    # 1. cross-fitted blend weight (chosen on fold (f+1)%5 per metric)
    for f in range(5):
        V = S[(f + 1) % 5]; row = {"fold": f}
        for mt in ("plate", "deg"):
            w = max(WS, key=lambda x: float(np.mean(V[mt][f"blend{x}"])))
            row[f"w_{mt}"] = w; row[f"blend_{mt}"] = float(np.mean(S[f][mt][f"blend{w}"]))
            row[f"undetect_{mt}"] = float(np.mean(S[f][mt]["nnk_multi_norm"])); row[f"xpert_{mt}"] = float(np.mean(S[f][mt]["xpert"]))
            row[f"half_{mt}"] = float(np.mean(S[f][mt]["blend0.5"]))
        out["blend"].append(row)
    bl = pd.DataFrame(out["blend"]); pd.set_option("display.width", 220)
    print("blend (1-w)*UnDetect-NN + w*XPert, w chosen on the neighbouring fold:\n" + bl.round(4).to_string(index=False))
    print("5-fold mean:\n" + bl.drop(columns="fold").mean().round(4).to_string())
    print("\ncorrelation of centred predictions (per-sample PCC, mean):",
          {k: round(float(np.mean([S[f][k] for f in range(5)])), 3) for k in S[0] if k.startswith("corr_")})
    # 3. line-specific part
    ls = {}
    for a in ARMS + ("truth",):
        if a == "truth":
            ls[a] = {"other_rows_mean": float(np.mean([np.nanmean(S[f]["plate"]["truth_other_rows_mean"]) for f in range(5)]))}
            continue
        full = np.mean([np.nanmean(np.where(S[f]["multi"], S[f]["plate"][a], np.nan)) for f in range(5)])
        cm = np.mean([np.nanmean(S[f]["plate"][f"{a}_cmean"]) for f in range(5)])
        ls[a] = {"full_multi_rows": float(full), "compound_mean_only": float(cm), "line_specific_gain": float(full - cm)}
    out["line_specific"] = ls
    print("\nPCC_plate on rows of compounds with >= 2 shared test rows: full prediction vs its compound mean:")
    print(pd.DataFrame(ls).T.round(4).to_string())
    # 2. strata
    C = pd.concat(comp); C["sim_bin"] = pd.qcut(C.maxsim_chem, 4, labels=["Q1 (least similar)", "Q2", "Q3", "Q4 (most similar)"])
    C["views"] = np.select([C.has_tgt & C.has_morph, C.has_tgt, C.has_morph], ["tgt+morph", "tgt only", "morph only"], "chem only")
    C["rows_bin"] = pd.cut(C.n_rows, [0, 1, 3, 10, 1000], labels=["1", "2-3", "4-10", ">10"])
    for col in ("views", "sim_bin", "rows_bin"):
        for lev, grp in C.groupby(col, observed=True):
            mean, lo, hi = boot_mean(grp.dif.to_numpy(), rng)
            r = {"stratum": col, "level": str(lev), "n_compounds": len(grp), "dif_mean": mean, "lo": lo, "hi": hi,
                 "xpert": float(grp.x.mean()), "undetect": float(grp.u.mean()),
                 "maxsim_median": float(grp.maxsim_chem.median())}
            out["strata"].append(r)
    st = pd.DataFrame(out["strata"])
    print("\nper-compound mean PCC_plate, XPert − UnDetect-NN, pooled over folds (compound bootstrap):\n" + st.round(4).to_string(index=False))
    allm, lo, hi = boot_mean(C.dif.to_numpy(), rng)
    print(f"all compounds: {allm:+.4f} [{lo:+.4f}, {hi:+.4f}], n {len(C)}; share with XPert better {float((C.dif > 0).mean()):.3f}")
    out["all_compounds"] = {"dif_mean": allm, "lo": lo, "hi": hi, "n": len(C), "share_xpert_better": float((C.dif > 0).mean())}
    (RES / "xpert_vs_undetect.json").write_text(json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
