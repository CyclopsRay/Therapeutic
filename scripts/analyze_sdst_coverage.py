#!/usr/bin/env python3
"""Gain by the test compound's view coverage on the drug-blind benchmark (report/11_06_record.md 2026-11-02 E5; no training).

Strata by which non-structural views a test compound has (targets and MoA have identical coverage in drugs.tsv, so
"pharmacology" = targets/MoA present): structure only, pharmacology only (no Cell Painting), Cell Painting only (no
pharmacology), both. Arms from `final_cpu` (one backbone): backbone `mlp`, chemistry-only transfer `pool_chem`, multi-view
transfer `pool_kta`, UnDetect `ctx_pc`, XPert, and their cross-fitted blend `ctx_pc+xpert`. PCC_DEG on all test rows and PCC_plate on rows with a training plate-mate
(plate mean as `finalize_sdst.py`), row means pooled over the five folds; paired differences with a 2,000-replicate
bootstrap over test compounds (each compound is a test compound in one fold only). Writes
results/sdst_hybrid/coverage_final/{strata.csv, summary.json}.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
CP = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/l1000_unseen_drug/cell_painting_features.npz")
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
OUT = RES / "coverage_final"
ARMS = ("mlp", "pool_chem", "pool_kta", "ctx_pc", "xpert", "ctx_pc+xpert")
PAIRS = (("ctx_pc", "mlp"), ("pool_kta", "pool_chem"), ("pool_chem", "mlp"), ("ctx_pc", "xpert"), ("ctx_pc+xpert", "xpert"))
STRATA = ("structure only", "pharmacology only", "Cell Painting only", "both")


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def main() -> None:
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, pidx = z["ctl"], z["delta"].astype(np.float64), z["group"], z["pert_idx"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    table = pd.read_csv(D / "drugs.tsv", sep="\t", keep_default_na=False).set_index("pert_idx")
    cp = np.load(CP, allow_pickle=False); cpk = {k[:14] for k in cp["inchikey"].astype(str)}
    pharm = table.target.ne("") | table.moa.ne("")
    morph = table.inchi_key.map(lambda k: bool(k) and k[:14] in cpk)
    stratum = pd.Series(np.select([pharm & morph, pharm, morph], ["both", "pharmacology only", "Cell Painting only"],
                                  "structure only"), index=table.index)
    rows = []
    for f in range(5):
        te = g == f; tr = (g != f) & (g != (f + 1) % 5)
        n = np.bincount(key[tr], minlength=key.max() + 1); Sm = np.zeros((key.max() + 1, Y.shape[1]))
        np.add.at(Sm, key[tr], Y[tr]); m = (Sm / np.maximum(n, 1)[:, None])[key[te]]; sh = n[key[te]] > 0
        P = np.load(RES / "final_cpu" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        Yt = Y[te]; fr = pd.DataFrame({"fold": f, "drug": pidx[te], "shared": sh})
        for a in ARMS:
            X = P[a].astype(np.float64)
            fr[f"deg|{a}"] = pcc(Yt, X)
            fr[f"plate|{a}"] = np.nan; fr.loc[sh, f"plate|{a}"] = pcc(Yt[sh] - m[sh], X[sh] - m[sh])
        rows.append(fr)
    R = pd.concat(rows, ignore_index=True); R["stratum"] = stratum.reindex(R.drug).to_numpy()
    rng = np.random.default_rng(0); out = []
    for s in STRATA + ("all",):
        Rs = R if s == "all" else R[R.stratum == s]
        for mt in ("deg", "plate"):
            x = Rs.dropna(subset=[f"{mt}|mlp"]) if mt == "plate" else Rs
            drugs, gi = np.unique(x.drug.to_numpy(), return_inverse=True); nd = np.bincount(gi).astype(float)
            W = rng.multinomial(len(drugs), np.full(len(drugs), 1 / len(drugs)), size=2000)
            rec = {"stratum": s, "metric": mt, "compounds": len(drugs), "rows": len(x),
                   **{a: float(x[f"{mt}|{a}"].mean()) for a in ARMS}}
            for a, b in PAIRS:
                dd = (x[f"{mt}|{a}"] - x[f"{mt}|{b}"]).to_numpy(); boot = (W @ np.bincount(gi, weights=dd)) / (W @ nd)
                rec.update({f"{a}-{b}": float(dd.mean()), f"{a}-{b}|lo": float(np.quantile(boot, 0.025)),
                            f"{a}-{b}|hi": float(np.quantile(boot, 0.975))})
            out.append(rec)
    T = pd.DataFrame(out); OUT.mkdir(parents=True, exist_ok=True); T.to_csv(OUT / "strata.csv", index=False)
    cov = stratum.value_counts().to_dict()
    (OUT / "summary.json").write_text(json.dumps({"compounds_per_stratum_all": cov, "strata": out}, indent=2) + "\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print("all compounds by stratum:", cov)
    print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
