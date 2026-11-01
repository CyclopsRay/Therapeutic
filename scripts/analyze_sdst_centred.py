#!/usr/bin/env python3
"""Plate-centred PCC_DEG for `run_sdst_hybrid.py --save-pred` outputs (Bai et al. 2026 benchmark).

For a test sample whose control row (= its line and plate set, see `sdst_plate_baseline.py`) occurs among train
samples, m = mean train delta of that control group. Centred score = Pearson over genes of (delta - m) vs
(prediction - m): it counts only how the compound differs from the training compounds measured on the same plate, so a
plate-mean lookup scores ~0 by construction. Samples without a shared control are reported with the plain metric
separately. Paired differences vs mlp use a 2,000-replicate bootstrap over test drugs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--tag", default="v3")
    p.add_argument("--ref", default="mlp", help="reference arm for the paired contrasts"); args = p.parse_args()
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, pidx = z["ctl"], z["delta"].astype(np.float64), z["group"], z["pert_idx"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    rng = np.random.default_rng(0); rows, cons = [], []
    for f in range(5):
        fp = RES / args.tag / f"fold_{f}" / "pred_test.npz"
        if not fp.exists():
            continue
        P = np.load(fp, allow_pickle=False)
        te = g == f; tr = (g != f) & (g != (f + 1) % 5)
        S = np.zeros((key.max() + 1, Y.shape[1])); n = np.bincount(key[tr], minlength=key.max() + 1)
        np.add.at(S, key[tr], Y[tr]); M = S / np.maximum(n, 1)[:, None]
        kt = key[te]; sh = n[kt] > 0; Yt = Y[te]; m = M[kt]
        assert np.allclose(P["y"].astype(np.float64), Yt, atol=1e-2)
        arms = [a for a in P.files if a not in ("y", "ctl")]
        sc = {a: pcc(Yt[sh] - m[sh], P[a][sh].astype(np.float64) - m[sh]) for a in arms}
        sc["plate_lookup"] = pcc(Yt[sh] - m[sh], np.zeros_like(m[sh]) + 1e-6 * rng.standard_normal(m[sh].shape))
        un = {a: pcc(Yt[~sh], P[a][~sh].astype(np.float64)) for a in arms}
        for a in sc:
            rows.append({"fold": f, "arm": a, "centred_shared": float(sc[a].mean()),
                         "plain_unshared": float(un[a].mean()) if a in un else None,
                         "plain_all": float(pcc(Yt, P[a].astype(np.float64)).mean()) if a in arms else None})
        drugs, gi = np.unique(pidx[te][sh], return_inverse=True); nd = np.bincount(gi).astype(float)
        W = rng.multinomial(len(drugs), np.full(len(drugs), 1 / len(drugs)), size=2000)
        for a in arms:
            if a == args.ref:
                continue
            d = sc[a] - sc[args.ref]; s = np.bincount(gi, weights=d); boot = (W @ s) / (W @ nd)
            cons.append({"fold": f, "arm": a, "ref": args.ref, "centred_diff_vs_mlp": float(d.mean()), "lo": float(np.quantile(boot, 0.025)),
                         "hi": float(np.quantile(boot, 0.975)), "drugs_up": float((s / nd > 0).mean())})
        print(f"fold {f}: {sh.sum()} shared / {(~sh).sum()} unshared test samples", flush=True)
    df = pd.DataFrame(rows); c = pd.DataFrame(cons)
    pd.set_option("display.width", 200)
    print(df.pivot(index="arm", columns="fold", values="centred_shared").assign(mean=lambda x: x.mean(1)).round(4).to_string())
    print("\nplain PCC_DEG on unshared samples:")
    print(df.dropna().pivot(index="arm", columns="fold", values="plain_unshared").assign(mean=lambda x: x.mean(1)).round(4).to_string())
    print(f"\ncentred, paired vs {args.ref}:"); print(c.round(4).to_string(index=False))
    print(c.groupby("arm").agg(diff=("centred_diff_vs_mlp", "mean"), folds_pos=("centred_diff_vs_mlp", lambda x: int((x > 0).sum())),
                               ci_excl0=("lo", lambda x: int((x > 0).sum()))).round(4).to_string())
    (RES / args.tag / ("centred.json" if args.ref == "mlp" else f"centred_vs_{args.ref}.json")).write_text(json.dumps({"rows": rows, "contrasts": cons}, indent=2) + "\n")


if __name__ == "__main__":
    main()
