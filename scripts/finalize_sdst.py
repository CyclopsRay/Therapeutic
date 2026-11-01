#!/usr/bin/env python3
"""Final drug-blind benchmark numbers on one backbone run: the CPU cache `cache_v1` + `stage2_v2` (user 2026-11-01).

Arms on the same test rows, five folds: backbone B (`cache_v1`), UnDetect closed form (`pool_kta`) and its second-stage
arms (`stage2_v2`), the released published-model predictions (`published`, `prep_published_preds.py`), the earlier GPU v3
arms for reference (suffix _v3), the ablation arms on the same CPU backbone (`run_sdst_hybrid.py --backbone-cache cache_v1`,
tag ABL_TAG: plate co-membership kernel, chemistry kernel, raw-feature residual head, and the closed form again as a check;
the end-to-end same-input network does not use the backbone and comes from v4_input), and cross-fitted blends with XPert (w chosen on the neighbouring fold = fold f's
validation compounds, by PCC_DEG as every UnDetect setting). PCC_DEG on all test rows, PCC_plate on rows with a training
plate-mate (`analyze_sdst_centred.py`), PCC_DEG on rows without one. Paired contrasts vs REFS with a 2,000-replicate
bootstrap over test compounds per fold. Writes results/sdst_hybrid/final_cpu/{scores.csv, contrasts.csv, final.json}
and fold_*/pred_test.npz (all arms, for `analyze_sdst_centred.py --tag final_cpu`).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
OUT = RES / "final_cpu"
WS = tuple(np.round(np.arange(0, 1.01, 0.1), 1))
S2 = ("pool_kta", "ctx_scale", "ctx_pc", "ctx_line")
PUB = ("xpert", "transigen", "multidcp", "ciger", "deepce", "mlp_bai", "xpert_zero", "xpert_shuffle")
V3 = {"mlp": "mlp_v3", "pool_multi_norm": "pool_v3", "nnk_multi_norm": "nnk_v3"}
ABL_TAG = "v8_cpu_abl"
ABL = {"pool_plate": "pool_plate", "pool_chem": "pool_chem", "nnf_multi": "nnf_multi", "pool_multi_norm": "pool_check"}
BLENDS = (("pool_kta", "xpert"), ("ctx_pc", "xpert"))
REFS = ("mlp", "pool_kta", "xpert", "nnf_multi")


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def main() -> None:
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, pidx = z["ctl"], z["delta"].astype(np.float64), z["group"], z["pert_idx"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    rng = np.random.default_rng(0); F = {}; t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    log("loaded sdst.npz and plate keys")
    for f in range(5):
        te = g == f; tr = (g != f) & (g != (f + 1) % 5)
        n = np.bincount(key[tr], minlength=key.max() + 1); Sm = np.zeros((key.max() + 1, Y.shape[1]))
        np.add.at(Sm, key[tr], Y[tr]); m = (Sm / np.maximum(n, 1)[:, None])[key[te]]
        A = {"mlp": np.load(RES / "cache_v1" / f"fold_{f}" / "cache.npz")["B"][te].astype(np.float64)}
        P = np.load(RES / "stage2_v2" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        assert np.allclose(P["y"].astype(np.float64), Y[te], atol=1e-2)
        A.update({a: P[a].astype(np.float64) for a in S2})
        Q = np.load(RES / "published" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        A.update({a: Q[a].astype(np.float64) for a in PUB}); A.update({b: Q[a].astype(np.float64) for a, b in V3.items()})
        Ab = np.load(RES / ABL_TAG / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        assert np.allclose(Ab["mlp"].astype(np.float64), A["mlp"], atol=1e-2), "ablation run is not on the cache_v1 backbone"
        A.update({b: Ab[a].astype(np.float64) for a, b in ABL.items()})
        A["mlp_input"] = np.load(RES / "v4_input" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)["mlp_input"].astype(np.float64)
        F[f] = {"Y": Y[te], "m": m, "sh": n[key[te]] > 0, "d": pidx[te], "A": A, "w": {}}
        log(f"fold {f}: loaded {len(A)} arms")
    for a, b in BLENDS:  # cross-fitted: w for fold f is the best w on fold (f + 1) % 5
        grid = {f: {w: float(pcc(F[f]["Y"], (1 - w) * F[f]["A"][a] + w * F[f]["A"][b]).mean()) for w in WS} for f in F}
        for f in F:
            w = max(WS, key=lambda x: grid[(f + 1) % 5][x])
            F[f]["A"][f"{a}+{b}"] = (1 - w) * F[f]["A"][a] + w * F[f]["A"][b]; F[f]["w"][f"{a}+{b}"] = w
    log("blends done")
    arms = list(F[0]["A"]); S, rows, cons = {}, [], []
    for f, x in F.items():
        Yt, m, sh = x["Y"], x["m"], x["sh"]
        S[f] = {a: {"deg": pcc(Yt, P), "plate": pcc(Yt[sh] - m[sh], P[sh] - m[sh]), "unshared": pcc(Yt[~sh], P[~sh])}
                for a, P in x["A"].items()}
        for a in arms:
            rows.append({"fold": f, "arm": a, **{k: float(v.mean()) for k, v in S[f][a].items()},
                         "n_shared": int(sh.sum()), "n_unshared": int((~sh).sum())})
        for mt, sel in (("deg", np.ones(len(sh), bool)), ("plate", sh)):
            drugs, gi = np.unique(x["d"][sel], return_inverse=True); nd = np.bincount(gi).astype(float)
            W = rng.multinomial(len(drugs), np.full(len(drugs), 1 / len(drugs)), size=2000)
            for ref in REFS:
                for a in arms:
                    if a == ref:
                        continue
                    dd = S[f][a][mt] - S[f][ref][mt]; s = np.bincount(gi, weights=dd); boot = (W @ s) / (W @ nd)
                    cons.append({"fold": f, "metric": mt, "arm": a, "ref": ref, "diff": float(dd.mean()),
                                 "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)),
                                 "compounds_up": float((s / nd > 0).mean())})
        log(f"fold {f}: scores and contrasts done")
    T = pd.DataFrame(rows); C = pd.DataFrame(cons)
    OUT.mkdir(parents=True, exist_ok=True); T.to_csv(OUT / "scores.csv", index=False); C.to_csv(OUT / "contrasts.csv", index=False)
    for f, x in F.items():
        o = OUT / f"fold_{f}"; o.mkdir(parents=True, exist_ok=True)
        np.savez(o / "pred_test.npz", y=x["Y"].astype(np.float16), ctl=ctl[g == f].astype(np.float16),
                 **{a: P.astype(np.float16) for a, P in x["A"].items()})
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
    mean = T.groupby("arm")[["deg", "plate", "unshared"]].mean().reindex(arms)
    print("5-fold means (PCC_DEG all rows | PCC_plate shared rows | PCC_DEG unshared rows):\n" + mean.round(4).to_string())
    print("blend weights per fold:", {k: [F[f]["w"][k] for f in F] for k in F[0]["w"]})
    agg = C.groupby(["ref", "metric", "arm"]).agg(
        diff=("diff", "mean"), folds_pos=("diff", lambda v: int((v > 0).sum())), ci_pos=("lo", lambda v: int((v > 0).sum())),
        ci_neg=("hi", lambda v: int((v < 0).sum())), compounds_up_min=("compounds_up", "min"), compounds_up_max=("compounds_up", "max"))
    for ref in REFS:
        print(f"\npaired vs {ref} (5-fold mean diff; folds > 0; folds with CI > 0; folds with CI < 0; share of compounds up):")
        print(agg.loc[ref].round(4).to_string())
    (OUT / "final.json").write_text(json.dumps({"means": mean.to_dict(orient="index"), "blend_w": {k: [F[f]["w"][k] for f in F] for k in F[0]["w"]},
                                                "contrasts": agg.reset_index().to_dict(orient="records")}, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
