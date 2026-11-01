#!/usr/bin/env python3
"""What the compound kernel knows that chemistry does not (Bai et al. 2026 drug-blind benchmark; no training).

Part A, descriptive (all compounds, no split; interpretation only, not a prediction claim):
  signature s_d = mean over compound d's samples of the plate-centred response, y_i minus the mean of the *other*
  samples sharing i's control row (= line and plate set). Response similarity of two compounds = Pearson(s_d, s_d').
  For every pair of compounds that have >= MIN_N samples and a non-chemical view, compare response similarity with
  chemical similarity (Morgan Tanimoto), the fused normalised kernel (KTA weights = mean over the v3 folds), and each
  view. Question: are there pairs that are chemically unlike but close in the fused kernel, and do they respond alike?
Part B, predictive (v3 test predictions, `run_sdst_hybrid.py --save-pred`): per test compound, the plate-centred gain
  of the multi-view kernel over the chemistry kernel, against the compound's nearest training neighbour by chemistry
  and by the fused kernel.
--final (user 2026-11-01): the final run instead of v3. Fused-kernel weights = mean KTA weights of cache_v1; Part B
  from final_cpu (one backbone): pool_kta stands in for pool_multi_norm (multi-view transfer terms) and ctx_pc
  (UnDetect) for nnk_multi_norm; output column names are kept. Writes compound_analysis_final/.
Writes results/sdst_hybrid/compound_analysis/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sdst_hybrid import D, OUT, VIEWS, drug_kernels  # noqa: E402

RES = OUT / "compound_analysis"
MIN_N = 3          # samples per compound for a signature
CHEM_LOW = 0.25    # "chemically unlike": Morgan Tanimoto below this
CHEM_HIGH = 0.6    # chemical analogues
SEED = 0
# example pairs exported for the figure: (compound, compound, class); chosen from hop_pairs_top.csv as one pair per
# well-known mechanism plus one linked by Cell Painting alone
EXAMPLES = [("panobinostat", "vorinostat", "HDAC"), ("trametinib", "selumetinib", "MEK"),
            ("daunorubicin", "mitoxantrone", "topoisomerase II"), ("alvocidib", "R-547", "CDK")]


def mol_drawing(smiles: str) -> dict:
    """2D depiction (RDKit CoordGen) as atoms and kekulised bonds, for vector drawing without RDKit."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    rdDepictor.SetPreferCoordGen(True)
    m = Chem.MolFromSmiles(smiles); Chem.Kekulize(m, clearAromaticFlags=True); rdDepictor.Compute2DCoords(m)
    xy = m.GetConformer().GetPositions()[:, :2]
    return {"atoms": [{"el": a.GetSymbol(), "x": float(xy[a.GetIdx(), 0]), "y": float(xy[a.GetIdx(), 1]),
                       "h": a.GetTotalNumHs(), "charge": a.GetFormalCharge()} for a in m.GetAtoms()],
            "bonds": [{"a": b.GetBeginAtomIdx(), "b": b.GetEndAtomIdx(), "order": int(b.GetBondTypeAsDouble())}
                      for b in m.GetBonds()]}


def zrows(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(1, keepdims=True)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    r = pd.Series(score).rank().to_numpy(); n1 = label.sum(); n0 = len(label) - n1
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def fused(K: dict, w: dict) -> np.ndarray:
    Kf = sum(w[v] * K[v] for v in VIEWS)
    dg = np.sqrt(np.clip(np.diag(Kf), 1e-12, None))
    return Kf / dg[:, None] / dg[None, :]


def main() -> None:
    global RES
    p = argparse.ArgumentParser(); p.add_argument("--final", action="store_true"); args = p.parse_args()
    if args.final:
        RES = OUT / "compound_analysis_final"
    RES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, pidx = z["ctl"], z["delta"].astype(np.float64), z["group"], z["pert_idx"]
    table = pd.read_csv(D / "drugs.tsv", sep="\t", keep_default_na=False)
    rowof = {d: i for i, d in enumerate(table.pert_idx)}; di = np.array([rowof[d] for d in pidx])
    K, feats = drug_kernels(table); has = feats["has"]
    src, field = ("cache_v1", "weights_kta") if args.final else ("v3", "alignment")
    al = [json.loads((OUT / src / f"fold_{f}" / "summary.json").read_text())[field] for f in range(5)]
    w = {v: max(np.mean([a[v] for a in al]), 0.0) for v in VIEWS}; s = sum(w.values()); w = {v: x / s for v, x in w.items()}
    Kn = fused(K, w)
    print(f"fused weights (mean {src} {field}):", {v: round(x, 3) for v, x in w.items()}, flush=True)
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    out = {"weights": w, "min_n": MIN_N, "chem_low": CHEM_LOW, "chem_high": CHEM_HIGH}

    # ---------------- Part A: compound signatures and pair similarities
    ng = np.bincount(key); S = np.zeros((ng.size, Y.shape[1])); np.add.at(S, key, Y)
    ok = ng[key] >= 2
    E = Y[ok] - (S[key[ok]] - Y[ok]) / (ng[key[ok]] - 1)[:, None]
    dE = di[ok]; nd = np.bincount(dE, minlength=len(table))
    sig = np.zeros((len(table), Y.shape[1])); np.add.at(sig, dE, E); sig /= np.maximum(nd, 1)[:, None]
    # split-half reliability of signatures (compounds with >= 6 samples): the ceiling for response similarity
    rel = []
    for d in np.flatnonzero(nd >= 6)[:2000]:
        e = E[dE == d]; p = rng.permutation(len(e)); h = len(e) // 2
        a, b = e[p[:h]].mean(0), e[p[h:]].mean(0); rel.append(float(np.corrcoef(a, b)[0, 1]))
    out["split_half_reliability_median"] = float(np.median(rel)); out["split_half_n"] = len(rel)
    sel = np.flatnonzero((nd >= MIN_N) & (has["tgt"] | has["morph"]))
    Zs = zrows(sig[sel]); Rr = Zs @ Zs.T
    iu = np.triu_indices(len(sel), 1)
    P = pd.DataFrame({"i": sel[iu[0]], "j": sel[iu[1]], "resp": Rr[iu], "chem": K["chem"][np.ix_(sel, sel)][iu],
                      "fused": Kn[np.ix_(sel, sel)][iu]})
    for v in VIEWS[1:]:
        pi, pj = P.i.to_numpy(), P.j.to_numpy()
        P[v] = np.where(has[v][pi] & has[v][pj], K[v][pi, pj], np.nan)
    print(f"Part A: {len(sel)} compounds, {len(P)} pairs; split-half reliability median "
          f"{out['split_half_reliability_median']:.3f} (n={len(rel)})", flush=True)

    cb = pd.cut(P.chem, [0, 0.2, 0.4, 0.6, 1.0001], right=False)
    fb = pd.cut(P.fused, [-1, 0.2, 0.4, 0.6, 0.8, 1.0001], right=False)
    grid = P.groupby([cb, fb], observed=True).resp.agg(["mean", "count"]).reset_index()
    grid.columns = ["chem_bin", "fused_bin", "resp_mean", "n_pairs"]
    out["resp_by_chem_and_fused"] = grid.astype({"chem_bin": str, "fused_bin": str}).to_dict("records")
    print(grid.to_string(index=False), flush=True)

    low = P[P.chem < CHEM_LOW]; high = P[P.chem >= CHEM_HIGH]
    thr = low.fused.quantile(0.995)
    hop = low[low.fused >= thr]
    out["pairs"] = {"all": len(P), "chem_low": len(low), "chem_high": len(high), "hop": len(hop),
                    "hop_fused_threshold": float(thr)}
    out["resp_mean"] = {"all": float(P.resp.mean()), "chem_low": float(low.resp.mean()),
                        "chem_high_analogues": float(high.resp.mean()), "chem_low_fused_top0.5pct": float(hop.resp.mean())}
    # compound-level bootstrap of the hop - chem_low contrast (pairs share compounds, so resample compounds)
    comp = np.unique(np.r_[hop.i, hop.j]); boots = []
    hi_, hj_ = hop.i.to_numpy(), hop.j.to_numpy()
    for _ in range(1000):
        c = rng.choice(comp, len(comp)); cnt = np.bincount(c, minlength=len(table))
        wts = cnt[hi_] * cnt[hj_]
        if wts.sum():
            boots.append(np.average(hop.resp, weights=wts) - out["resp_mean"]["chem_low"])
    out["hop_minus_chem_low_ci95"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    print("response similarity:", {k: round(v, 4) for k, v in out["resp_mean"].items()},
          "hop - chem_low CI", np.round(out["hop_minus_chem_low_ci95"], 4), flush=True)

    # which view finds the response neighbours that chemistry misses: among chemically unlike pairs where both
    # compounds have the view, AUROC of the view's similarity for the top-1% response-similar pairs
    lab_thr = low.resp.quantile(0.99); out["view_auroc_chem_low"] = {}
    for v in ["chem", "fused"] + VIEWS[1:]:
        m = low[v].notna(); x = low.loc[m, v].to_numpy(); lab = (low.loc[m, "resp"] >= lab_thr).to_numpy()
        top = low.loc[m].loc[low.loc[m, v] >= low.loc[m, v].quantile(0.99), "resp"].mean()
        out["view_auroc_chem_low"][v] = {"pairs": int(m.sum()), "auroc": auroc(x, lab), "resp_top1pct_by_view": float(top)}
    print("views on chemically unlike pairs:", json.dumps(out["view_auroc_chem_low"], indent=1), flush=True)
    # per view, binned (target/MOA similarities are mostly exactly 0, so quantile cut-offs are meaningless):
    # response similarity of chemically unlike pairs by the view's similarity, and what a view adds on its own
    vb = {"tgt": [0, 1e-9, 0.5, 1.0001], "moa": [0, 1e-9, 0.5, 1.0001], "morph": [-1, 0.3, 0.5, 0.7, 1.0001]}
    out["view_bins_chem_low"] = {}
    for v, edges in vb.items():
        m = low[v].notna()
        t = low[m].groupby(pd.cut(low.loc[m, v], edges, right=False), observed=True).resp.agg(["mean", "count"])
        out["view_bins_chem_low"][v] = [{"bin": str(k), "resp_mean": float(r["mean"]), "n_pairs": int(r["count"])}
                                        for k, r in t.iterrows()]
        print(v, t.round(4).to_string(), flush=True)
    noann = low.tgt.fillna(0).eq(0) & low.moa.fillna(0).eq(0)
    out["view_alone_chem_low"] = {
        "morph>=0.7, no shared target/MOA": [float(low[noann & (low.morph >= 0.7)].resp.mean()),
                                             int((noann & (low.morph >= 0.7)).sum())],
        "moa>=0.5, Cell Painting missing for either": [float(low[low.morph.isna() & (low.moa >= 0.5)].resp.mean()),
                                                       int((low.morph.isna() & (low.moa >= 0.5)).sum())],
        "tgt>=0.5, Cell Painting missing for either": [float(low[low.morph.isna() & (low.tgt >= 0.5)].resp.mean()),
                                                       int((low.morph.isna() & (low.tgt >= 0.5)).sum())]}
    print("view alone:", out["view_alone_chem_low"], flush=True)
    # activity cliffs: chemical analogues whose responses differ; does the rest of the kernel see it?
    if len(high) > 50:
        out["cliffs_spearman_fused_vs_resp_in_analogues"] = float(high[["fused", "resp"]].corr("spearman").iloc[0, 1])
        out["cliffs_spearman_chem_vs_resp_in_analogues"] = float(high[["chem", "resp"]].corr("spearman").iloc[0, 1])

    name = table.cmap_name.to_numpy(); tgt = table.target.to_numpy(); moa = table.moa.to_numpy()
    ex = hop.sort_values("resp", ascending=False).head(60).copy()
    for c in ("i", "j"):
        ex[f"name_{c}"] = name[ex[c]]; ex[f"moa_{c}"] = moa[ex[c]]; ex[f"target_{c}"] = tgt[ex[c]]; ex[f"n_{c}"] = nd[ex[c]]
    ex.to_csv(RES / "hop_pairs_top.csv", index=False)
    ex2 = low[low.resp >= lab_thr].sort_values("fused").head(40).copy()   # response twins the kernel misses
    for c in ("i", "j"):
        ex2[f"name_{c}"] = name[ex2[c]]; ex2[f"moa_{c}"] = moa[ex2[c]]
    ex2.to_csv(RES / "missed_twins.csv", index=False)

    # example pairs for the figure: per compound the row with most samples (names can repeat across pert_ids)
    best = {}
    for nm in {x for p_ in EXAMPLES for x in p_[:2]}:
        rows_ = np.flatnonzero(name == nm); best[nm] = int(rows_[np.argmax(nd[rows_])])
    exp = []
    for i, j, cls in EXAMPLES:
        a, b = best[i], best[j]
        exp.append({"pair": [i, j], "class": cls, "tanimoto": float(K["chem"][a, b]), "fused": float(Kn[a, b]),
                    "morph": float(K["morph"][a, b]) if has["morph"][a] and has["morph"][b] else None,
                    "resp_r": float(np.corrcoef(sig[a], sig[b])[0, 1]), "n": [int(nd[a]), int(nd[b])],
                    "smiles": [table.canonical_smiles[a], table.canonical_smiles[b]],
                    "drawing": [mol_drawing(table.canonical_smiles[a]), mol_drawing(table.canonical_smiles[b])],
                    "signature": [sig[a].round(4).tolist(), sig[b].round(4).tolist()]})
        print(f"example {i}-{j}: tanimoto {exp[-1]['tanimoto']:.3f} r {exp[-1]['resp_r']:.3f} n {exp[-1]['n']}", flush=True)
    (RES / "example_pairs.json").write_text(json.dumps(exp) + "\n")

    # ---------------- Part B: per test compound, gain of multi-view over chemistry vs nearest training neighbour
    rows = []
    for f in range(5):
        fp = OUT / ("final_cpu" if args.final else "v3") / f"fold_{f}" / "pred_test.npz"
        arm_src = ({"mlp": "mlp", "pool_chem": "pool_chem", "pool_multi_norm": "pool_kta", "nnk_multi_norm": "ctx_pc"}
                   if args.final else {a: a for a in ("mlp", "pool_chem", "pool_multi_norm", "nnk_multi_norm")})
        if not fp.exists():
            continue
        Pz = np.load(fp, allow_pickle=False)
        te = g == f; tr = (g != f) & (g != (f + 1) % 5)
        n = np.bincount(key[tr], minlength=ng.size); St = np.zeros_like(S); np.add.at(St, key[tr], Y[tr])
        M = St / np.maximum(n, 1)[:, None]; kt = key[te]; sh = n[kt] > 0
        yt = Y[te][sh] - M[kt][sh]
        sc = {a: np.sum(zrows(yt) * zrows(Pz[b][sh].astype(np.float64) - M[kt][sh]), 1) for a, b in arm_src.items()}
        dt = di[te][sh]; trd = np.unique(di[tr]); ud, inv = np.unique(dt, return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        nn_chem = K["chem"][np.ix_(ud, trd)].max(1); nn_fused = Kn[np.ix_(ud, trd)].max(1)
        for a in sc:
            sc[a] = np.bincount(inv, weights=sc[a]) / cnt
        for k, d in enumerate(ud):
            rows.append({"fold": f, "drow": int(d), "name": name[d], "n": int(cnt[k]), "nn_chem": nn_chem[k],
                         "nn_fused": nn_fused[k], "has_tgt": bool(has["tgt"][d]), "has_morph": bool(has["morph"][d]),
                         **{a: sc[a][k] for a in sc}})
    G = pd.DataFrame(rows)
    G["gain_multi_vs_chem"] = G.pool_multi_norm - G.pool_chem; G["gain_decant_vs_mlp"] = G.nnk_multi_norm - G.mlp
    G.to_csv(RES / "drug_gain.csv", index=False)
    G["chem_bin"] = pd.cut(G.nn_chem, [0, 0.3, 0.5, 0.7, 1.0001], right=False).astype(str)
    G["fused_bin"] = pd.cut(G.nn_fused, [-1, 0.5, 0.7, 0.85, 1.0001], right=False).astype(str)
    G["coverage"] = np.select([G.has_tgt & G.has_morph, G.has_tgt, G.has_morph], ["tgt+morph", "tgt", "morph"], "chem only")
    for by in (["chem_bin"], ["chem_bin", "fused_bin"], ["coverage"]):
        t = G.groupby(by).agg(n_drugs=("drow", "size"), gain_multi_vs_chem=("gain_multi_vs_chem", "mean"),
                              gain_decant_vs_mlp=("gain_decant_vs_mlp", "mean")).reset_index()
        out["partB_by_" + "_".join(by)] = t.to_dict("records")
        print(t.round(4).to_string(index=False), flush=True)
    out["partB_spearman_gain_multi_vs_chem"] = {
        c: float(G[["gain_multi_vs_chem", c]].corr("spearman").iloc[0, 1]) for c in ("nn_chem", "nn_fused")}
    G["fused_minus_chem"] = G.nn_fused - G.nn_chem
    out["partB_spearman_gain_multi_vs_chem"]["nn_fused_minus_nn_chem"] = float(
        G[["gain_multi_vs_chem", "fused_minus_chem"]].corr("spearman").iloc[0, 1])
    print("Part B spearman:", out["partB_spearman_gain_multi_vs_chem"], flush=True)
    (RES / "summary.json").write_text(json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
