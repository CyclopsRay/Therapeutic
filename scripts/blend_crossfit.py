#!/usr/bin/env python3
"""Paired comparison and cross-fitted blend of two prediction arms that live in different result tags (drug-blind
benchmark, same test rows). Blend (1 − w)·A + w·B with w chosen on the neighbouring fold (= fold f's validation
compounds), as in `analyze_xpert_vs_undetect.py`. Also writes a merged results/sdst_hybrid/<out>/fold_*/pred_test.npz
(A, B, blend) for `analyze_sdst_centred.py`.

Usage: blend_crossfit.py --a stage2_v2:ctx_pc --b published:xpert --out stage2_vs_xpert
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
WS = tuple(np.round(np.arange(0, 1.01, 0.1), 1))


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--extra", nargs="*", default=[], help="more tag:arm pairs copied into the merged npz")
    p.add_argument("--out", required=True); args = p.parse_args()
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g = z["ctl"], z["delta"].astype(np.float64), z["group"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    (ta, aa), (tb, ab) = args.a.split(":"), args.b.split(":"); na, nb = f"{aa}", f"{ab}"
    S = {}
    for f in range(5):
        te = g == f; tr = (g != f) & (g != (f + 1) % 5)
        PA = np.load(RES / ta / f"fold_{f}" / "pred_test.npz"); PB = np.load(RES / tb / f"fold_{f}" / "pred_test.npz")
        A_, B_ = PA[aa].astype(np.float64), PB[ab].astype(np.float64)
        n = np.bincount(key[tr], minlength=key.max() + 1); Sm = np.zeros((key.max() + 1, Y.shape[1]))
        np.add.at(Sm, key[tr], Y[tr]); m = (Sm / np.maximum(n, 1)[:, None])[key[te]]; sh = n[key[te]] > 0
        Yt = Y[te]
        S[f] = {w: {"plate": float(pcc(Yt[sh] - m[sh], ((1 - w) * A_ + w * B_)[sh] - m[sh]).mean()),
                    "deg": float(pcc(Yt, (1 - w) * A_ + w * B_).mean())} for w in WS}
        S[f]["arrays"] = (A_, B_)
    rows = []
    for f in range(5):
        V = S[(f + 1) % 5]; r = {"fold": f}
        for mt in ("plate", "deg"):
            w = max(WS, key=lambda x: V[x][mt]); r[f"w_{mt}"] = w
            r[f"{na}_{mt}"] = S[f][0.0][mt]; r[f"{nb}_{mt}"] = S[f][1.0][mt]; r[f"blend_{mt}"] = S[f][w][mt]
        rows.append(r)
        A_, B_ = S[f]["arrays"]; te = g == f
        out = RES / args.out / f"fold_{f}"; out.mkdir(parents=True, exist_ok=True)
        extra = {}
        for spec in args.extra:
            t, a = spec.split(":"); extra[a if a not in (na, nb) else f"{t}_{a}"] = np.load(RES / t / f"fold_{f}" / "pred_test.npz")[a]
        np.savez(out / "pred_test.npz", y=Y[te].astype(np.float16), ctl=ctl[te].astype(np.float16),
                 **{na: A_.astype(np.float16), nb: B_.astype(np.float16),
                    "blend": ((1 - r["w_plate"]) * A_ + r["w_plate"] * B_).astype(np.float16)}, **extra)
    import pandas as pd
    T = pd.DataFrame(rows); pd.set_option("display.width", 220)
    print(T.round(4).to_string(index=False)); print("5-fold mean:\n" + T.drop(columns="fold").mean().round(4).to_string())
    (RES / args.out / "blend.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
