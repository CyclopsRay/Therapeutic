#!/usr/bin/env python3
"""UnDetect + XPert ensemble with a weight that never depends on the target fold's labels (report/11_06_report.md §10.7;
no training; added 2026-11-02 after the external reviewer asked whether the ensemble weight avoids test labels).

`finalize_sdst.py` picks w for fold f on fold (f+1)'s test rows (= fold f's validation compounds). Those predictions come
from fold f+1's models, which were trained on groups including group f, so fold f's labels reach w indirectly. Here:
  next  w from fold (f+1)'s test rows (as in the paper);
  prev  w from fold (f-1)'s test rows: labels of group f-1, models trained without group f (they exclude f-1 and f);
  half  w = 0.5 fixed.
w grid as finalize_sdst (0, 0.1, ..., 1), chosen by PCC_DEG. Scores and paired differences vs XPert and vs UnDetect
(`ctx_pc`) with a 2,000-replicate bootstrap over test compounds per fold. Writes results/sdst_hybrid/blend_clean/summary.json.
"""
from __future__ import annotations

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
    z = np.load(D / "sdst.npz", allow_pickle=False)
    ctl, Y, g, pidx = z["ctl"], z["delta"].astype(np.float64), z["group"], z["pert_idx"]
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    F = {}
    for f in range(5):
        te = g == f; tr = (g != f) & (g != (f + 1) % 5)
        n = np.bincount(key[tr], minlength=key.max() + 1); Sm = np.zeros((key.max() + 1, Y.shape[1]))
        np.add.at(Sm, key[tr], Y[tr]); m = (Sm / np.maximum(n, 1)[:, None])[key[te]]
        P = np.load(RES / "final_cpu" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        F[f] = {"Y": Y[te], "m": m, "sh": n[key[te]] > 0, "d": pidx[te],
                "u": P["ctx_pc"].astype(np.float64), "x": P["xpert"].astype(np.float64)}
    grid = {f: {w: float(pcc(x["Y"], (1 - w) * x["u"] + w * x["x"]).mean()) for w in WS} for f, x in F.items()}
    rule = {"next": lambda f: max(WS, key=lambda w: grid[(f + 1) % 5][w]),
            "prev": lambda f: max(WS, key=lambda w: grid[(f - 1) % 5][w]), "half": lambda f: 0.5}
    rng = np.random.default_rng(0); out = {}
    for name, pick in rule.items():
        res = {"w": [], "folds": []}
        for f, x in F.items():
            w = pick(f); B = (1 - w) * x["u"] + w * x["x"]; sh = x["sh"]; res["w"].append(w)
            sc = {"deg": {k: pcc(x["Y"], P) for k, P in (("blend", B), ("xpert", x["x"]), ("undetect", x["u"]))},
                  "plate": {k: pcc(x["Y"][sh] - x["m"][sh], P[sh] - x["m"][sh]) for k, P in
                            (("blend", B), ("xpert", x["x"]), ("undetect", x["u"]))}}
            fold = {"fold": f, "w": w}
            for mt, sel in (("deg", np.ones(len(sh), bool)), ("plate", sh)):
                drugs, gi = np.unique(x["d"][sel], return_inverse=True); nd = np.bincount(gi).astype(float)
                W = rng.multinomial(len(drugs), np.full(len(drugs), 1 / len(drugs)), size=2000)
                fold[mt] = float(sc[mt]["blend"].mean())
                for ref in ("xpert", "undetect"):
                    dd = sc[mt]["blend"] - sc[mt][ref]; boot = (W @ np.bincount(gi, weights=dd)) / (W @ nd)
                    fold[f"{mt}_vs_{ref}"] = [float(dd.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
            res["folds"].append(fold)
        fl = res["folds"]
        res["mean"] = {mt: float(np.mean([x[mt] for x in fl])) for mt in ("deg", "plate")}
        res["ci_pos"] = {f"{mt}_vs_{r}": int(sum(x[f"{mt}_vs_{r}"][1] > 0 for x in fl))
                         for mt in ("deg", "plate") for r in ("xpert", "undetect")}
        res["mean_diff"] = {f"{mt}_vs_{r}": float(np.mean([x[f"{mt}_vs_{r}"][0] for x in fl]))
                            for mt in ("deg", "plate") for r in ("xpert", "undetect")}
        out[name] = res
        print(f"{name}: w {res['w']} mean {res['mean']} diff {res['mean_diff']} folds CI>0 {res['ci_pos']}")
    o = RES / "blend_clean"; o.mkdir(parents=True, exist_ok=True)
    (o / "summary.json").write_text(json.dumps({"grid_pcc_deg": grid, **out}, indent=2) + "\n")


if __name__ == "__main__":
    main()
