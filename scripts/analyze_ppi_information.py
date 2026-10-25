#!/usr/bin/env python3
"""Does STRING wiring carry drug-response information beyond node degree? (exploratory, 2026-10-04)

Model-free companion to G3/G3b in report/10_09_report.md, where kNN, KRR and MLP all gained no more
from the real STRING graph than from a degree-preserving rewired one. Three descriptive tests on
the KNOWN compounds (>= 1 target in the graph) of one L1000 slice, per line:
  A  locality (ACTIVE rows): for each landmark gene, standardized |residual| when the drug's nearest
     target is exactly 1 hop away vs >= 3 hops away, averaged within gene (gene variance removed).
     `A_rel` divides each signature by its own mean |residual| first (drug magnitude removed).
     Real graph vs seed-0 rewired graph (run.rewire) vs target sets permuted among compounds.
  B  pairs (ACTIVE rows, pairs on different L1000 plates): residual-signature Pearson of two drugs
     by network distance between their target sets (0 = a shared target).
  C  degree (all KNOWN rows): Spearman of mean log(1 + degree) of the drug's targets with its
     residual norm.
Graphs: STRING v12 physical (score >= 400; the runner's graph) and functional (score >= 700).
Residuals are from the line mean over all compounds of the slice (descriptive, no train/test).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

import run_l1000_unseen_drug as run

BASE = Path("/nfs-stor/yuqi.lei/Therapeutic")
N_PERM = 20


def load_functional(root: Path, min_score: int, rewire_seed: int | None):
    """As run.load_ppi, on the full functional network file."""
    info = pd.read_csv(root / "9606.protein.info.v12.0.txt.gz", sep="\t")
    name = dict(zip(info["#string_protein_id"], info["preferred_name"]))
    links = pd.read_csv(root / "9606.protein.links.v12.0.txt.gz", sep=" ")
    links = links[links.combined_score >= min_score]
    a = links.protein1.map(name).to_numpy(); b = links.protein2.map(name).to_numpy()
    genes = np.array(sorted(set(a) | set(b)))
    index = {g: i for i, g in enumerate(genes)}
    u = np.array([index[x] for x in a]); v = np.array([index[x] for x in b])
    keep = u < v
    edges = np.unique(np.stack([u[keep], v[keep]], 1), axis=0)
    if rewire_seed is not None:
        edges = run.rewire(edges, len(genes), rewire_seed)
    n = len(genes)
    A = sp.coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(n, n)).tocsr()
    return genes, A, len(edges)


def distances(A, genes, targets: list[set], lm_symbols):
    """Per compound: hop distance from its nearest target to every landmark; pairwise target-set distance."""
    gpos = {g: i for i, g in enumerate(genes)}
    mapped = [sorted(t & gpos.keys()) for t in targets]
    known = np.array([len(t) > 0 for t in mapped])
    T = sorted({g for t in mapped for g in t}); tpos = {g: i for i, g in enumerate(T)}
    tidx = np.array([gpos[g] for g in T])
    D = shortest_path(A, directed=False, unweighted=True, indices=tidx)       # (|T|, n), inf if unreachable
    lm_in = np.array([g in gpos for g in lm_symbols])
    D_lm = D[:, [gpos.get(g, 0) for g in lm_symbols]]; D_lm[:, ~lm_in] = np.inf
    D_tt = D[:, tidx]
    rows = [np.array([tpos[g] for g in t], int) for t in mapped]
    kc = np.where(known)[0]
    d_lm = np.full((len(targets), len(lm_symbols)), np.inf)
    for c in kc:
        d_lm[c] = D_lm[rows[c]].min(0)
    M = np.full((len(targets), len(T)), np.inf)                                # compound -> nearest dist to target t
    for c in kc:
        M[c] = D_tt[rows[c]].min(0)
    d_pair = np.full((len(targets), len(targets)), np.inf)
    for c in kc:
        d_pair[:, c] = M[:, rows[c]].min(1)
    deg = np.asarray(A.sum(1)).ravel()
    tdeg = np.full(len(targets), np.nan)
    for c in kc:
        tdeg[c] = np.log1p(deg[[gpos[g] for g in mapped[c]]]).mean()
    return known, d_lm, d_pair, tdeg, lm_in


def locality(U: np.ndarray, d: np.ndarray) -> tuple[float, int, float]:
    """Within-gene mean U at distance 1 minus at distance >= 3, weighted by the near count."""
    near = d == 1; far = d >= 3
    nn, nf = near.sum(0), far.sum(0)
    ok = (nn >= 5) & (nf >= 5)
    if not ok.any():
        return np.nan, 0, np.nan
    diff = (U * near).sum(0) / np.maximum(nn, 1) - (U * far).sum(0) / np.maximum(nf, 1)
    return float(np.average(diff[ok], weights=nn[ok])), int(ok.sum()), float((diff[ok] > 0).mean())


def main(args: argparse.Namespace) -> None:
    started = time.time()
    print(f"host {platform.node()} python {sys.version.split()[0]} numpy {np.__version__} scipy {scipy.__version__} "
          f"pandas {pd.__version__} slice {args.slice} rewire_seed {args.rewire_seed} perm_seed 0", flush=True)
    d = BASE / "derived/l1000_unseen_drug"
    sl = np.load(d / f"slice_{args.slice}.npz", allow_pickle=False)
    keys = sl["inchikey"].astype(str); line_rows = sl["line"].astype(str); Y = sl["Y"].astype(np.float64)
    active = sl["tas"] >= run.ACTIVE_TAS
    plates = [set(p.split("|")) for p in sl["plates"].astype(str)]
    lm_symbols = sl["gene_symbol"].astype(str)
    compounds = np.array(sorted(set(keys))); cidx = {k: i for i, k in enumerate(compounds)}
    row_c = np.array([cidx[k] for k in keys])
    table = pd.read_parquet(d / "drug_table.parquet")
    table = table[table.source.eq("L1000") & table.inchikey.isin(compounds)]
    targets = [set() for _ in compounds]
    for r in table.itertuples():
        if r.targets:
            targets[cidx[r.inchikey]].update(r.targets.split(";"))
    root = BASE / "raw/string_v12"
    graphs = {
        "physical400": lambda s: run.load_ppi(root, 400, s),
        "functional700": lambda s: load_functional(root, 700, s),
    }
    rng = np.random.default_rng(0)
    records = []
    for gname, loader in graphs.items():
        for variant, seed in (("real", None), ("rewired", args.rewire_seed)):
            t0 = time.time()
            genes, A, n_edges = loader(seed)
            known, d_lm, d_pair, tdeg, lm_in = distances(A, genes, targets, lm_symbols)
            print(f"{gname} {variant}: {len(genes):,} genes, {n_edges:,} edges, {lm_in.sum()} landmarks in graph, "
                  f"KNOWN {known.sum():,} ({time.time() - t0:.0f}s)", flush=True)
            kc = np.where(known)[0]
            perms = [rng.permutation(kc) for _ in range(N_PERM)] if variant == "real" else []
            for L in sorted(set(line_rows)):
                in_line = line_rows == L
                R = Y[in_line] - Y[in_line].mean(0); sd = Y[in_line].std(0)
                c_l = row_c[in_line]; act_l = active[in_line]
                plates_l = [p for p, keep in zip(plates, in_line) if keep]
                sel = act_l & known[c_l]
                U = np.abs(R[sel]) / sd; Urel = U / U.mean(1, keepdims=True)
                dist = d_lm[c_l[sel]]
                rec = {"slice": args.slice, "graph": gname, "variant": variant, "line": L, "n_active_known": int(sel.sum())}
                for h in (0, 1, 2):
                    rec[f"share_d{h}"] = float((dist == h).mean())
                rec["share_d3plus"] = float((dist >= 3).mean())
                rec["A"], rec["A_genes"], rec["A_pos_genes"] = locality(U, dist)
                rec["A_rel"], _, rec["A_rel_pos_genes"] = locality(Urel, dist)
                if perms:
                    remap = {c: i for i, c in enumerate(kc)}
                    pos = np.array([remap[c] for c in c_l[sel]])
                    nulls = [locality(U, d_lm[p[pos]])[0] for p in perms]
                    nulls_rel = [locality(Urel, d_lm[p[pos]])[0] for p in perms]
                    rec["A_perm_mean"], rec["A_perm_q95"] = float(np.mean(nulls)), float(np.quantile(nulls, 0.95))
                    rec["A_rel_perm_mean"], rec["A_rel_perm_q95"] = float(np.mean(nulls_rel)), float(np.quantile(nulls_rel, 0.95))
                # B: pairwise residual correlation by target-set distance, different-plate pairs only
                Z = R[sel] - R[sel].mean(1, keepdims=True); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
                C = Z @ Z.T
                cs = c_l[sel]; ps = [plates_l[i] for i in np.where(sel)[0]]
                iu, ju = np.triu_indices(len(cs), 1)
                diff_plate = np.array([not (ps[i] & ps[j]) for i, j in zip(iu, ju)])
                dp = d_pair[cs[iu], cs[ju]]; r = C[iu, ju]
                for label, mask in (("0", dp == 0), ("1", dp == 1), ("2", dp == 2), ("3plus", dp >= 3)):
                    mm = mask & diff_plate
                    rec[f"B_n_d{label}"] = int(mm.sum())
                    rec[f"B_r_d{label}"] = float(r[mm].mean()) if mm.any() else np.nan
                    rec[f"B_p03_d{label}"] = float((r[mm] >= 0.3).mean()) if mm.any() else np.nan
                # C: target degree vs response norm, all KNOWN rows of the line
                allk = known[c_l]
                rho = spearmanr(tdeg[c_l[allk]], np.linalg.norm(R[allk], axis=1)).statistic
                rec["C_rho_degree_norm"] = float(rho)
                rec["C_rho_ntargets_norm"] = float(spearmanr([len(targets[c]) for c in c_l[allk]],
                                                             np.linalg.norm(R[allk], axis=1)).statistic)
                records.append(rec)
    out = BASE / "results/ppi_information"; out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / f"{args.slice}.csv", index=False)
    (out / f"{args.slice}_manifest.json").write_text(json.dumps({
        "slice": args.slice, "rewire_seed": args.rewire_seed, "n_perm": N_PERM, "perm_seed": 0,
        "script_sha256": run.sha256(Path(__file__).resolve()), "slice_sha256": run.sha256(d / f"slice_{args.slice}.npz"),
        "seconds": round(time.time() - started, 1)}, indent=2) + "\n")
    cols = ["A", "A_rel", "A_perm_mean", "A_rel_perm_mean", "A_pos_genes", "share_d1", "share_d3plus",
            "B_r_d0", "B_r_d1", "B_r_d2", "B_r_d3plus", "B_p03_d1", "B_p03_d3plus", "C_rho_degree_norm",
            "C_rho_ntargets_norm"]
    print(frame.groupby(["graph", "variant"])[cols].mean().round(4).T.to_string(), flush=True)
    print(f"done ({time.time() - started:.0f}s)", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slice", default="P2-24", choices=["P1-24", "P1-6", "P2-24"])
    p.add_argument("--rewire-seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
