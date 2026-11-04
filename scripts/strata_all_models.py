#!/usr/bin/env python3
"""Every model by view coverage and by plate-mate count, on all three drug-blind scores (no training; 2026-11-04).

Extends `analyze_sdst_coverage.py` (Table 15) to all models with released per-sample predictions and adds PCC_within,
and extends `analyze_plate_mate_count.py` (Table 16) with PCC_within. Arms from `final_cpu`: backbone `mlp`, published
drug-free MLP `mlp_bai`, CIGER, MultiDCP, DeepCE, TranSiGen, XPert, chemistry kernel `pool_chem`, multi-view kernel
`pool_kta`, UnDetect `ctx_pc`, ensemble `ctx_pc+xpert`. PRnet and PertDiT have no released per-sample predictions on
this scale. PCC_DEG on all test rows, PCC_plate on rows with a training plate-mate (plate mean as `finalize_sdst.py`),
PCC_within from `within_plate/units.csv.gz` (no CIGER/MultiDCP there). Strata as `analyze_sdst_coverage.py`; plate-mate
bins as `analyze_plate_mate_count.py` (for PCC_within: the plate's number of training rows). Means pooled over the five
folds; paired differences with a 2,000-replicate bootstrap over test compounds.
Writes results/sdst_hybrid/strata_all/{strata.csv, platemates_within.csv, summary.json}.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
CP = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/l1000_unseen_drug/cell_painting_features.npz")
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
OUT = RES / "strata_all"
ARMS = ("mlp", "mlp_bai", "ciger", "multidcp", "deepce", "transigen", "xpert", "pool_chem", "pool_kta", "ctx_pc",
        "ctx_pc+xpert")
WITHIN = ("mlp", "mlp_bai", "deepce", "transigen", "xpert", "pool_chem", "pool_kta", "ctx_pc", "ctx_pc+xpert")
PAIRS = (("ctx_pc", "xpert"), ("ctx_pc", "transigen"), ("ctx_pc", "mlp"), ("ctx_pc+xpert", "xpert"),
         ("ctx_pc+xpert", "ctx_pc"))
STRATA = ("structure only", "pharmacology only", "Cell Painting only", "both")
BINS = (0, 2, 5, 10, 20, 50, 10 ** 6)
SEED, B = 0, 2000


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def boot_pairs(x: pd.DataFrame, cols: dict, rng) -> dict:
    drugs, gi = np.unique(x.drug.to_numpy(), return_inverse=True); nd = np.bincount(gi).astype(float)
    W = rng.multinomial(len(drugs), np.full(len(drugs), 1 / len(drugs)), size=B)
    rec = {}
    for a, b in PAIRS:
        if cols.get(a) not in x or cols.get(b) not in x:
            continue
        dd = (x[cols[a]] - x[cols[b]]).to_numpy(); bt = (W @ np.bincount(gi, weights=dd)) / (W @ nd)
        rec.update({f"{a}-{b}": float(dd.mean()), f"{a}-{b}|lo": float(np.quantile(bt, 0.025)),
                    f"{a}-{b}|hi": float(np.quantile(bt, 0.975))})
    return rec


def main() -> None:
    print("host", platform.node(), "python", platform.python_version(), "numpy", np.__version__, "pandas",
          pd.__version__, "seed", SEED, flush=True)
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, pidx = z["ctl"], z["delta"].astype(np.float64), z["group"], z["pert_idx"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    table = pd.read_csv(D / "drugs.tsv", sep="\t", keep_default_na=False).set_index("pert_idx")
    cp = np.load(CP, allow_pickle=False); cpk = {k[:14] for k in cp["inchikey"].astype(str)}
    pharm = table.target.ne("") | table.moa.ne("")
    morph = table.inchi_key.map(lambda k: bool(k) and k[:14] in cpk)
    stratum = pd.Series(np.select([pharm & morph, pharm, morph], ["both", "pharmacology only", "Cell Painting only"],
                                  "structure only"), index=table.index)
    rows, nmate = [], {}
    for f in range(5):
        te = g == f; tr = (g != f) & (g != (f + 1) % 5)
        n = np.bincount(key[tr], minlength=key.max() + 1); nmate[f] = n
        Sm = np.zeros((key.max() + 1, Y.shape[1])); np.add.at(Sm, key[tr], Y[tr])
        m = (Sm / np.maximum(n, 1)[:, None])[key[te]]; sh = n[key[te]] > 0
        P = np.load(RES / "final_cpu" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        Yt = Y[te]; fr = pd.DataFrame({"fold": f, "drug": pidx[te], "shared": sh})
        for a in ARMS:
            X = P[a].astype(np.float64)
            fr[f"deg|{a}"] = pcc(Yt, X)
            fr[f"plate|{a}"] = np.nan; fr.loc[sh, f"plate|{a}"] = pcc(Yt[sh] - m[sh], X[sh] - m[sh])
        rows.append(fr)
    R = pd.concat(rows, ignore_index=True); R["stratum"] = stratum.reindex(R.drug).to_numpy()
    U = pd.read_csv(RES / "within_plate" / "units.csv.gz")
    U["stratum"] = stratum.reindex(U.drug).to_numpy()
    U["n"] = [nmate[f][p] for f, p in zip(U.fold, U.plate)]
    rng = np.random.default_rng(SEED); out = []
    for s in STRATA + ("all",):
        for mt in ("deg", "plate", "within"):
            if mt == "within":
                x = U if s == "all" else U[U.stratum == s]; cols = {a: a for a in WITHIN}
            else:
                Rs = R if s == "all" else R[R.stratum == s]
                x = Rs.dropna(subset=["plate|mlp"]) if mt == "plate" else Rs; cols = {a: f"{mt}|{a}" for a in ARMS}
            rec = {"stratum": s, "metric": mt, "compounds": int(x.drug.nunique()), "units": len(x),
                   **{a: float(x[c].mean()) for a, c in cols.items()}, **boot_pairs(x, cols, rng)}
            out.append(rec)
    T = pd.DataFrame(out)
    U["bin"] = pd.cut(U.n, BINS, right=True).astype(str)
    bins = []
    for lab in [str(pd.Interval(BINS[i], BINS[i + 1], closed="right")) for i in range(len(BINS) - 1)] + ["0 plate-mates"]:
        x = U[U.n == 0] if lab == "0 plate-mates" else U[U.bin == lab]
        if len(x) == 0:
            continue
        bins.append({"bin": lab, "units": len(x), **{a: float(x[a].mean()) for a in WITHIN},
                     **boot_pairs(x, {a: a for a in WITHIN}, rng)})
    Bn = pd.DataFrame(bins)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / "strata.csv", index=False); Bn.to_csv(OUT / "platemates_within.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"strata": out, "platemates_within": bins,
                                                  "compounds_per_stratum_all": stratum.value_counts().to_dict()},
                                                 indent=2) + "\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    show = ["stratum", "metric", "compounds"] + list(ARMS)
    print(T[[c for c in show if c in T]].round(4).to_string(index=False))
    print(T[["stratum", "metric"] + [c for c in T if "-" in c and "|" not in c]].round(4).to_string(index=False))
    print(T[["stratum", "metric", "ctx_pc-xpert|lo", "ctx_pc-xpert|hi"]].round(4).to_string(index=False))
    print(Bn.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
