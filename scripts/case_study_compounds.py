"""Case study for Fig. 3: per-compound scores and signatures of the Fig. 1 compounds and the Fig. 3d pairs (no training).

Each compound is a test compound in exactly one fold of the drug-blind benchmark. For every pert_idx whose cmap name is
in NAMES, on its test rows of that fold (final_cpu predictions, one backbone run):
  - PCC_DEG on all its test rows and PCC_plate on rows with a training plate-mate, per arm (mean over rows), with the
    plate mean m_i defined exactly as in scripts/finalize_sdst.py (training = groups other than f and f+1 mod 5);
  - its mean plate-centred signature over rows with a plate-mate, for the truth and each arm (y - m, yhat - m);
  - the number of rows and of cell lines.
Writes results/sdst_hybrid/case_study/{scores.csv, signatures.npz, summary.json}.
Run (Slurm): slurm/case_study_compounds.sbatch
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
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
OUT = RES / "case_study"
NAMES = ("daunorubicin", "vorinostat", "mitoxantrone", "panobinostat", "trametinib", "selumetinib", "alvocidib",
         "R-547")
FIG1 = {"daunorubicin": "BRD-A37630846", "mitoxantrone": "BRD-K21680192", "vorinostat": "BRD-K81418486"}
ARMS = ("mlp", "mlp_bai", "pool_chem", "pool_kta", "ctx_pc", "xpert", "transigen", "deepce", "ctx_pc+xpert")
SEED = 0


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1.0), np.nan)


def main() -> None:
    print("host", platform.node(), "python", sys.version.split()[0], "numpy", np.__version__, "pandas", pd.__version__,
          "seed", SEED, flush=True)
    t = pd.read_csv(D / "drugs.tsv", sep="\t", keep_default_na=False)
    want = t[t.cmap_name.isin(NAMES)][["pert_idx", "pert_id", "cmap_name", "target", "moa"]]
    print(want.to_string(index=False), flush=True)
    z = np.load(D / "sdst.npz", allow_pickle=False)
    print("sdst.npz keys", z.files, flush=True)
    ctl, Y, g, pidx = z["ctl"], z["delta"].astype(np.float64), z["group"], z["pert_idx"]
    cell = z["cell_idx"] if "cell_idx" in z.files else None
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    rows, sigs = [], {}
    for _, w in want.iterrows():
        sel = np.flatnonzero(pidx == int(w.pert_idx))
        if sel.size == 0:
            continue
        folds = np.unique(g[sel]); assert folds.size == 1, (w.cmap_name, folds)
        f = int(folds[0]); te = np.flatnonzero(g == f); tr = (g != f) & (g != (f + 1) % 5)
        n = np.bincount(key[tr], minlength=key.max() + 1); Sm = np.zeros((key.max() + 1, Y.shape[1]))
        np.add.at(Sm, key[tr], Y[tr])
        loc = np.flatnonzero(pidx[te] == int(w.pert_idx))          # rows of this compound within the fold's test set
        r = te[loc]; m = (Sm / np.maximum(n, 1)[:, None])[key[r]]; sh = n[key[r]] > 0
        P = np.load(RES / "final_cpu" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        assert np.allclose(P["y"][loc].astype(np.float64), Y[r], atol=1e-2), "row order differs from sdst.npz"
        tag = f"{w.cmap_name}|{w.pert_id}"
        sigs[f"{tag}|truth"] = (Y[r][sh] - m[sh]).mean(0) if sh.any() else np.full(Y.shape[1], np.nan)
        for a in ARMS:
            if a == "ctx_pc+xpert" and a not in P.files:
                continue
            A = P[a][loc].astype(np.float64)
            rows.append({"name": w.cmap_name, "pert_id": w.pert_id, "pert_idx": int(w.pert_idx), "fig1": FIG1.get(
                w.cmap_name) == w.pert_id, "fold": f, "n_rows": len(r), "n_shared": int(sh.sum()),
                "n_lines": int(np.unique(cell[r]).size) if cell is not None else -1, "arm": a,
                "pcc_deg": float(np.nanmean(pcc(Y[r], A))),
                "pcc_plate": float(np.nanmean(pcc(Y[r][sh] - m[sh], A[sh] - m[sh]))) if sh.any() else np.nan})
            sigs[f"{tag}|{a}"] = (A[sh] - m[sh]).mean(0) if sh.any() else np.full(Y.shape[1], np.nan)
        print(f"{tag}: fold {f}, {len(r)} rows, {int(sh.sum())} with a plate-mate", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    S = pd.DataFrame(rows); S.to_csv(OUT / "scores.csv", index=False)
    np.savez_compressed(OUT / "signatures.npz", **{k.replace("|", "__"): v for k, v in sigs.items()})
    wide = S.pivot_table(index=["name", "pert_id", "n_rows", "n_shared"], columns="arm", values="pcc_plate")
    print("PCC_plate per compound:\n" + wide.round(3).to_string(), flush=True)
    truth = {k.split("|")[0] + "|" + k.split("|")[1]: v for k, v in sigs.items() if k.endswith("|truth")}
    names = list(truth); M = np.array([truth[k] for k in names])
    C = np.corrcoef(M) if len(names) > 1 else np.ones((1, 1))
    (OUT / "summary.json").write_text(json.dumps({"truth_signature_corr": {"names": names, "r": C.round(4).tolist()},
                                                 "arms": list(ARMS)}, indent=1))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
