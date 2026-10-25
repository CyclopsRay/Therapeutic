#!/usr/bin/env python3
"""Export compact JSON for the interactive DepMap<->Tahoe CCA explorer.

Per line: name, lineage, held-out split; CC1/CC2 coordinates of every line
(DepMap and Tahoe side) under the all-line fit and under the fit that leaves
that line out (signs aligned to the all-line fit); direct gene-correlation and
leave-one-out CCA similarity rows; per-line results from jobs 308632/308619.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plot_depmap_tahoe_cca import RESULTS, load_inputs
from run_depmap_tahoe_cca import Cca, gene_symbol
from run_depmap_expression_ridge import DEFAULT_DEPMAP_ROOT
from run_rhaister_o_compact import DEFAULT_CACHE_ROOT, DEFAULT_DATA_ROOT, DEFAULT_PROJECT_ROOT, load_split


def r3(a):
    return np.round(np.asarray(a, float), 3).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    a = argparse.Namespace(project_root=DEFAULT_PROJECT_ROOT, data_root=DEFAULT_DATA_ROOT,
                           cache_root=DEFAULT_CACHE_ROOT, depmap_root=DEFAULT_DEPMAP_ROOT)
    compact, expression, metadata, cells, x, y, repo_root = load_inputs(a)
    n = len(cells)

    full = Cca(x, y, 10, 10)
    u_all, v_all = full.embed_x(x)[:, :2], full.embed_y(y)[:, :2]

    loo_coords, loo_sim, loo_rho = [], np.zeros((n, n)), []
    for i in range(n):
        keep = np.arange(n) != i
        m = Cca(x[keep], y[keep], 10, 10)
        u, v = m.embed_x(x), m.embed_y(y)
        q = u[i]
        loo_sim[i] = (v @ q) / (np.linalg.norm(v, axis=1) * np.linalg.norm(q) + 1e-12)
        u2, v2 = u[:, :2].copy(), v[:, :2].copy()
        for j in range(2):  # align sign to the all-line fit on the 46 fitted lines
            if np.dot(v2[keep, j], v_all[keep, j]) < 0:
                u2[:, j] *= -1
                v2[:, j] *= -1
        loo_coords.append({"u": r3(u2), "v": r3(v2)})
        loo_rho.append(r3(m.rho[:2]))

    symbols = np.array([gene_symbol(c) for c in expression.columns])
    xcol = {s: k for k, s in enumerate(symbols)}
    genes = list(compact.genes)
    shared = [g for g in genes if g in xcol]
    xs, ys = x[:, [xcol[g] for g in shared]], y[:, [genes.index(g) for g in shared]]
    variable = (xs.std(0) > 0) & (ys.std(0) > 0)
    def rowz(m):
        m = ((m - m.mean(0)) / m.std(0).clip(min=1e-12))[:, variable]
        m = m - m.mean(1, keepdims=True)
        return m / np.linalg.norm(m, axis=1, keepdims=True)
    direct = rowz(xs) @ rowz(ys).T

    split_of = {}
    for s in range(5, 10):
        holdout, _, _ = load_split(repo_root, s)
        for c in holdout:
            split_of[c] = s
    gene_space = pd.read_csv(RESULTS / "depmap_tahoe_cca" / "per_line.csv").set_index("cell_line")
    shared_space = pd.read_csv(RESULTS / "depmap_tahoe_cca_space" / "per_line.csv")
    shared_space = shared_space[shared_space["target"].eq("absolute")].pivot(
        index="cell_line", columns="arm", values="pearson")

    lines = []
    for i, c in enumerate(cells):
        rec = {"id": c, "name": metadata[c]["cell_line_name"], "lineage": metadata[c]["lineage"],
               "split": split_of.get(c)}
        if c in gene_space.index:
            g = gene_space.loc[c]
            rec["heldout"] = {
                "rank_cca_split": int(g["rank_cca"]), "rank_direct_split": int(g["rank_direct"]),
                "dmso_recon_r": round(float(g["dmso_recon_pearson"]), 3),
                "resid": {k: round(float(g[f"{k}__drug_residual"]), 3)
                          for k in ("real_dmso", "cca_mapped", "nearest_ref_line", "shuffled_cca", "mean_dmso")},
                "full": {k: round(float(g[f"{k}__full"]), 3)
                         for k in ("real_dmso", "cca_mapped", "mean_dmso")},
                "abs_space": {k: round(float(shared_space.loc[c, k]), 3)
                              for k in ("depmap", "identity_only_depmap", "tahoe_dmso", "identity_only_tahoe")},
            }
        lines.append(rec)

    payload = {
        "lines": lines,
        "all_fit": {"u": r3(u_all), "v": r3(v_all), "rho": r3(full.rho[:2])},
        "loo_fit": loo_coords, "loo_rho": loo_rho,
        "sim_direct": r3(direct), "sim_cca_loo": r3(loo_sim),
        "n_genes_direct": int(variable.sum()),
    }
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    own_d = np.mean(direct.argmax(1) == np.arange(n))
    own_c = np.mean(loo_sim.argmax(1) == np.arange(n))
    print(f"wrote {args.out} ({args.out.stat().st_size/1024:.0f} KB); top-1 direct {own_d:.2f}, CCA-LOO {own_c:.2f}")


if __name__ == "__main__":
    main()
