#!/usr/bin/env python3
"""Oracle reference on the final drug-blind run (`final_cpu`), for the paper's Discussion (record 2026-11-03).

Recomputes the uniform oracle of `sanity_context_interaction.py` Part 2 on the final backbone (`cache_v1`) so that it
can be compared with UnDetect (ctx_pc), XPert and their ensemble on the same rows. The oracle uses test responses and is
a reference point, not a model: test sample i of compound d is predicted by the backbone plus the mean backbone
residual of d's other test samples in the same fold (other cell lines). It is defined for rows whose compound has at
least two test samples. Ensemble = 0.6 ctx_pc + 0.4 XPert (the weight chosen in every fold, final.json).
PCC_plate as in `finalize_sdst.py` (centred on the mean train delta of the sample's control row; rows with a training
plate-mate). Truth and plate means in float64 from sdst.npz; predictions from final_cpu/fold_*/pred_test.npz (float16),
so the script first checks its full-subset means against final.json. Writes results/sdst_hybrid/oracle_final/.
v2 (record 2026-11-03, Table 2 row): every Table 2 arm; the oracle also on all test rows (PCC_DEG) and on all plate-mate
rows (PCC_plate), falling back to the backbone for compounds with a single test sample (0.7% of plate-mate rows); the
share of the backbone-to-oracle distance per fold on all plate-mate rows, as Table 2 reports PCC_plate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
OUT = RES / "oracle_final"
ARMS = ("mlp", "mlp_bai", "pool_plate", "pool_chem", "mlp_input", "nnf_multi", "pool_kta", "ctx_pc", "xpert",
        "ctx_pc+xpert")  # the rows of the paper's Table 2


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def main() -> None:
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, pidx = z["ctl"], z["delta"].astype(np.float64), z["group"], z["pert_idx"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    ref = json.loads((RES / "final_cpu" / "final.json").read_text())["means"]
    rows = []
    for f in range(5):
        te = g == f; tr = (g != f) & (g != (f + 1) % 5)
        n = np.bincount(key[tr], minlength=key.max() + 1); Sm = np.zeros((key.max() + 1, Y.shape[1]))
        np.add.at(Sm, key[tr], Y[tr]); m = (Sm / np.maximum(n, 1)[:, None])[key[te]]; sh = n[key[te]] > 0
        P = np.load(RES / "final_cpu" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        Yt = Y[te]; assert np.allclose(P["y"].astype(np.float64), Yt, atol=1e-2)
        A = {a: P[a].astype(np.float64) for a in ARMS}
        r = Yt - A["mlp"]; d = pidx[te]; orc = np.full(r.shape, np.nan); nd = np.zeros(len(d), int)
        for c in np.unique(d):
            idx = np.flatnonzero(d == c); nd[idx] = len(idx)
            if len(idx) >= 2:
                orc[idx] = (r[idx].sum(0)[None] - r[idx]) / (len(idx) - 1)
        A["oracle_uniform"] = A["mlp"] + np.nan_to_num(orc)
        two = nd >= 2
        for a, Pa in A.items():
            plate = np.full(len(d), np.nan); plate[sh] = pcc(Yt[sh] - m[sh], Pa[sh] - m[sh])
            deg = pcc(Yt, Pa)
            for sub, sel in (("all_shared", sh), ("shared_nd>=2", sh & two)):
                rows.append({"fold": f, "subset": sub, "arm": a, "n": int(sel.sum()), "pcc_plate": float(plate[sel].mean()),
                             "pcc_deg_all_rows": float(deg.mean())})
        print(f"fold {f}: {te.sum()} test rows, {sh.sum()} with a plate-mate, {(sh & two).sum()} of them with n_d >= 2",
              flush=True)
    T = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True); T.to_csv(OUT / "scores.csv", index=False)
    M = T.groupby(["subset", "arm"])["pcc_plate"].mean().unstack(0)
    Dg = T[T.subset == "all_shared"].groupby("arm")["pcc_deg_all_rows"].mean()
    for a in ARMS:  # float16 predictions must reproduce the published means
        assert abs(M.loc[a, "all_shared"] - ref[a]["plate"]) < 2e-3, (a, M.loc[a, "all_shared"], ref[a]["plate"])
        assert abs(Dg[a] - ref[a]["deg"]) < 2e-3, (a, Dg[a], ref[a]["deg"])
    share = {}
    for sub in ("shared_nd>=2", "all_shared"):
        S = T[T.subset == sub].pivot(index="fold", columns="arm", values="pcc_plate")
        gap = S["oracle_uniform"] - S["mlp"]
        share[sub] = {a: {"per_fold": ((S[a] - S["mlp"]) / gap).round(4).tolist(),
                          "mean": float(((S[a] - S["mlp"]) / gap).mean())} for a in ARMS if a != "mlp"}
    out = {"means": M.round(4).to_dict(), "pcc_deg_all_rows": Dg.round(4).to_dict(),
           "rows_per_fold": T[T.arm == "mlp"].pivot(index="fold", columns="subset", values="n").to_dict(),
           "share_of_backbone_to_oracle_distance": share}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    pd.set_option("display.width", 200)
    print("5-fold mean PCC_plate:\n" + M.round(4).to_string())
    print("PCC_DEG, all test rows:\n" + Dg.round(4).to_string())
    for sub, d in share.items():
        print(f"share of the backbone-to-oracle distance ({sub}):", {a: round(v["mean"], 3) for a, v in d.items()})


if __name__ == "__main__":
    main()
