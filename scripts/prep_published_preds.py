#!/usr/bin/env python3
"""Bai et al. 2026 released predictions → our SDST row order, for PCC_plate on the published models (no training).

Reads `benchmark_predictions.zip` (Zenodo 22095049 v2.0.0) in place, without extracting. For every primary-cohort Full run
(and its drug-input Zero / Shuffle ablations) of the seven models and the Basal-only MLP (run_index.csv) in the canonical delta space (PRnet and PertDiT
are released in their own normalised log-expression space and are skipped), loads the test predictions and test ground truth,
checks that their rows and genes are in our `sdst.npz` order (h5ad order; ground truth equal to our deltas within
1e-4), and writes results/sdst_hybrid/published/fold_<f>/pred_test.npz in the format of `run_sdst_hybrid.py
--save-pred`. Our v3 arms (`mlp`, `pool_multi_norm`, `nnk_multi_norm`) are copied in, so that
`analyze_sdst_centred.py --tag published --ref <arm>` gives paired PCC_plate contrasts. Also prints per-fold plain
PCC_DEG next to the released `pcc_saved`.
"""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
ZIP = Path("/nfs-stor/yuqi.lei/Therapeutic/raw/bai2026_archive/benchmark_predictions.zip")
IDX = Path("/home/yuqi.lei/Therapeutic/repro/dpb/upstream/data/analysis/benchmark_predictions")
V3_ARMS = ("mlp", "pool_multi_norm", "nnk_multi_norm")


def pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def main() -> None:
    z = np.load(D / "sdst.npz", allow_pickle=False)
    Y, ctl, g, genes = z["delta"], z["ctl"], z["group"], z["genes"]
    runs = [r for r in csv.DictReader(open(IDX / "run_index.csv"))
            if "primary" in r["cohorts"] and r["condition"] in ("Full", "Zero", "Shuffle", "Basal-only")
            and r["split_seed"] == "131419"]
    assets = {(a["run_id"], a["role"]): a["path"] for a in csv.DictReader(open(IDX / "asset_index.csv"))}
    zf = zipfile.ZipFile(ZIP); names = set(zf.namelist())

    def member(path: str) -> str:
        cands = [n for n in (path, path.split("/", 1)[-1]) if n in names] or [n for n in names if n.endswith(path.split("/")[-1])]
        assert len(cands) >= 1, path
        return cands[0]

    def load(path: str):
        b = zf.read(member(path))
        return np.load(io.BytesIO(b), allow_pickle=False) if path.endswith(".npy") else b.decode()

    order = [l.split(",") for l in load(assets[("dataset", "gene_order")]).splitlines()[1:]]  # gene_index,gene_id
    assert [int(i) for i, _ in order] == [int(j) for _, j in order] == list(range(len(genes))), "gene order is not the h5ad order"
    for f in range(5):
        te = np.flatnonzero(g == f)
        v3 = np.load(RES / "v3" / f"fold_{f}" / "pred_test.npz", allow_pickle=False)
        out = {"y": Y[te].astype(np.float16), "ctl": ctl[te].astype(np.float16), **{a: v3[a] for a in V3_ARMS}}
        for r in [r for r in runs if r["fold"] == str(f)]:
            name = "mlp_bai" if r["model"] == "MLP" else r["model"].lower() + ("" if r["condition"] == "Full" else "_" + r["condition"].lower())
            if r["target_space"] != "canonical_expression_delta":  # PRnet, PertDiT: own normalised log space
                print(f"fold {f} {name:10s} skipped: target space {r['target_space']} (released PCC_DEG {float(r['pcc_saved']):.5f})")
                continue
            gt = load(assets[(r["run_id"], "ground_truth")]).astype(np.float32)
            pr = load(assets[(r["run_id"], "predictions")]).astype(np.float32)
            assert gt.shape == pr.shape == (len(te), Y.shape[1]), (r["run_id"], gt.shape, pr.shape, len(te))
            # released rows follow the h5ad order of the fold's test samples, as ours do; their deltas differ from ours
            # by float rounding only (max |diff| ~1e-6), so check closeness instead of exact bytes
            err = float(np.abs(gt - Y[te]).max())
            assert err < 1e-4, f"{r['run_id']}: ground truth differs from our test rows (max |diff| {err:.2e})"
            P = pr
            out[name] = P.astype(np.float16)
            print(f"fold {f} {name:10s} max|gt - ours| {err:.1e}; PCC_DEG ours {pcc(Y[te].astype(np.float64), P.astype(np.float64)).mean():.5f}"
                  f" released {float(r['pcc_saved']):.5f}", flush=True)
        o = RES / "published" / f"fold_{f}"; o.mkdir(parents=True, exist_ok=True)
        np.savez(o / "pred_test.npz", **out)


if __name__ == "__main__":
    main()
