#!/usr/bin/env python3
"""PCC_plate by the number of training plate-mates (report/11_06_report.md §10.6; no training; diagnostic added
2026-11-02 for the reviewer's question whether the plate-mean estimate inflates PCC_plate).

m_i is the mean of the training samples that share test sample i's control row. Its estimation error enters both
y_i - m_i and yhat_i - m_i, so a prediction carrying no compound information can score above zero, more so when m_i
rests on few plate-mates. Arms from `final_cpu`: backbone `mlp`, UnDetect `ctx_pc`, XPert. Paired differences per bin
and for n <= 20 / n > 20, with a 2,000-replicate bootstrap over test compounds. Writes
results/sdst_hybrid/plate_mate_count/{bins.csv, contrasts.csv, summary.json}.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
OUT = RES / "plate_mate_count"
ARMS = ("mlp", "ctx_pc", "xpert")
BINS = (0, 2, 5, 10, 20, 50, 10 ** 6)


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def main() -> None:
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g = z["ctl"], z["delta"].astype(np.float64), z["group"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    parts = []
    for f in range(5):
        te = g == f; tr = (g != f) & (g != (f + 1) % 5)
        n = np.bincount(key[tr], minlength=key.max() + 1); Sm = np.zeros((key.max() + 1, Y.shape[1]))
        np.add.at(Sm, key[tr], Y[tr]); m = (Sm / np.maximum(n, 1)[:, None])[key[te]]; nn = n[key[te]]; sh = nn > 0
        P = np.load(RES / "final_cpu" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        fr = pd.DataFrame({"fold": f, "n": nn[sh], "drug": z["pert_idx"][te][sh]})
        for a in ARMS:
            fr[a] = pcc(Y[te][sh] - m[sh], P[a][sh].astype(np.float64) - m[sh])
        parts.append(fr)
    T = pd.concat(parts, ignore_index=True)
    T["bin"] = pd.cut(T.n, BINS, right=True).astype(str)
    B = T.groupby("bin", sort=False).agg(rows=("n", "size"), **{a: (a, "mean") for a in ARMS})
    B = B.loc[[str(pd.Interval(BINS[i], BINS[i + 1], closed="right")) for i in range(len(BINS) - 1)]]
    B["share_rows"] = B.rows / B.rows.sum(); B["undetect_minus_backbone"] = B.ctx_pc - B.mlp
    rng = np.random.default_rng(0); con = []
    T["half"] = np.where(T.n > 20, "n > 20", "n <= 20")
    for col, lab in [("bin", b) for b in B.index] + [("half", h) for h in ("n <= 20", "n > 20")]:
        x = T[T[col] == lab]; drugs, gi = np.unique(x.drug.to_numpy(), return_inverse=True); nd = np.bincount(gi).astype(float)
        W = rng.multinomial(len(drugs), np.full(len(drugs), 1 / len(drugs)), size=2000)
        for a, b in (("ctx_pc", "mlp"), ("ctx_pc", "xpert"), ("xpert", "mlp")):
            dd = (x[a] - x[b]).to_numpy(); boot = (W @ np.bincount(gi, weights=dd)) / (W @ nd)
            con.append({"group": lab, "pair": f"{a}-{b}", "rows": len(x), "diff": float(dd.mean()),
                        "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975))})
    C = pd.DataFrame(con)
    OUT.mkdir(parents=True, exist_ok=True); B.to_csv(OUT / "bins.csv"); C.to_csv(OUT / "contrasts.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"bins": B.reset_index().to_dict(orient="records"),
                                                  "all": {a: float(T[a].mean()) for a in ARMS},
                                                  "contrasts": C.to_dict(orient="records")}, indent=2) + "\n")
    print(B.round(4).to_string()); print(C.round(4).to_string(index=False)); print("all:", {a: round(float(T[a].mean()), 4) for a in ARMS})


if __name__ == "__main__":
    main()
