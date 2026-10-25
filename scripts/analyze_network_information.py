#!/usr/bin/env python3
"""Control-ladder information test across network sources (RECOMB study, PLAN.md Phase 6, E1 + E7).

Generalises analyze_ppi_information.py (job 294863) to every source in netnull.SOURCES and to the
null ladder real / rewired (seeds 0-4) / labels (seeds 0-4) / er (seed 0). Per slice and line, on
compounds with >= 1 target in the network:
  A_rel  within-gene standardized |residual| at 1 hop from the nearest target minus at >= 3 hops,
         after dividing each signature by its own mean |residual| (ACTIVE rows). For directed
         networks the hop distance follows edge direction (target -> downstream gene).
  B_r_d1 mean residual-signature Pearson of drug pairs whose target sets are 1 hop apart
         (undirected), different-plate pairs, ACTIVE rows; B_r_d0 = sharing a target.
  C_deg  Spearman(mean log(1 + target degree), residual norm), all KNOWN rows.
E7 (network-independent, per line): C_pub = Spearman(mean log(1 + PubMed count) of targets, norm),
and C_deg_given_pub = Spearman of the degree and norm ranks after regressing out publication rank.
Exploratory; residuals from the line mean over all compounds (no train/test split).
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
from scipy.sparse.csgraph import shortest_path
from scipy.stats import rankdata, spearmanr

import netnull
import run_l1000_unseen_drug as run
from analyze_ppi_information import locality

BASE = Path("/nfs-stor/yuqi.lei/Therapeutic")
LADDER = [("real", 0)] + [("rewired", s) for s in range(5)] + [("labels", s) for s in range(5)] + [("er", 0)]


def pubcounts() -> dict:
    """Approved symbol -> number of PubMed records in NCBI gene2pubmed (human)."""
    cache = netnull.DERIVED / "gene_pubcount.tsv"
    if not cache.exists():
        counts = None
        for chunk in pd.read_csv(netnull.RAW / "networks/gene2pubmed/gene2pubmed.gz", sep="\t", chunksize=5_000_000,
                                 dtype={"#tax_id": int, "GeneID": int, "PubMed_ID": int}):
            c = chunk[chunk["#tax_id"] == 9606].GeneID.value_counts()
            counts = c if counts is None else counts.add(c, fill_value=0)
        h = pd.read_csv(netnull.RAW / "networks/hgnc/hgnc_complete_set.txt", sep="\t", usecols=["symbol", "entrez_id"],
                        dtype=str).dropna()
        h["entrez_id"] = h.entrez_id.astype(float).astype(int)
        h["pubs"] = h.entrez_id.map(counts).fillna(0).astype(int)
        netnull.DERIVED.mkdir(parents=True, exist_ok=True)
        h[["symbol", "pubs"]].to_csv(cache, sep="\t", index=False)
    t = pd.read_csv(cache, sep="\t")
    return dict(zip(t.symbol, t.pubs))


def distances(net: netnull.Network, targets: list[set], lm_symbols: np.ndarray):
    gpos = {g: i for i, g in enumerate(net.genes)}
    mapped = [sorted(t & gpos.keys()) for t in targets]
    known = np.array([len(t) > 0 for t in mapped])
    T = sorted({g for t in mapped for g in t}); tpos = {g: i for i, g in enumerate(T)}
    tidx = np.array([gpos[g] for g in T], int)
    lm_in = np.array([g in gpos for g in lm_symbols]); lm_idx = np.array([gpos.get(g, 0) for g in lm_symbols])
    A_dir = net.adjacency(symmetric=not net.directed)
    D_down = shortest_path(A_dir, directed=net.directed, unweighted=True, indices=tidx) if len(tidx) else np.zeros((0, net.n))
    D_und = D_down if not net.directed else shortest_path(net.adjacency(True), directed=False, unweighted=True,
                                                          indices=tidx)
    D_lm = D_down[:, lm_idx]; D_lm[:, ~lm_in] = np.inf
    rows = [np.array([tpos[g] for g in t], int) for t in mapped]
    n = len(targets); kc = np.where(known)[0]
    d_lm = np.full((n, len(lm_symbols)), np.inf)
    M = np.full((n, len(T)), np.inf)
    for c in kc:
        d_lm[c] = D_lm[rows[c]].min(0); M[c] = D_und[rows[c]][:, tidx].min(0)
    d_pair = np.full((n, n), np.inf)
    for c in kc:
        d_pair[:, c] = M[:, rows[c]].min(1)
    deg = net.degree()
    tdeg = np.full(n, np.nan)
    for c in kc:
        tdeg[c] = np.log1p(deg[[gpos[g] for g in mapped[c]]]).mean()
    return known, d_lm, d_pair, tdeg, int(lm_in.sum())


def main(args: argparse.Namespace) -> None:
    started = time.time()
    print(f"host {platform.node()} python {sys.version.split()[0]} numpy {np.__version__} scipy {scipy.__version__} "
          f"pandas {pd.__version__} slice {args.slice} networks {args.networks}", flush=True)
    d = BASE / "derived/l1000_unseen_drug"
    sl = np.load(d / f"slice_{args.slice}.npz", allow_pickle=False)
    tahoe = args.slice.startswith("T-")
    keys = sl["inchikey"].astype(str); line_rows = sl["line"].astype(str); Y = sl["Y"].astype(np.float64)
    active = sl["active"].astype(bool) if "active" in sl.files else sl["tas"] >= run.ACTIVE_TAS
    plates = [set(p.split("|")) for p in sl["plates"].astype(str)]
    lm_symbols = sl["gene_symbol"].astype(str)
    compounds = np.array(sorted(set(keys))); cidx = {k: i for i, k in enumerate(compounds)}
    row_c = np.array([cidx[k] for k in keys])
    table = pd.read_parquet(d / "drug_table.parquet")
    table = table[table.source.eq("Tahoe" if tahoe else "L1000") & table.inchikey.isin(compounds)]
    targets = [set() for _ in compounds]
    for r in table.itertuples():
        if r.targets:
            targets[cidx[r.inchikey]].update(r.targets.split(";"))
    if args.harmonize:  # targets and landmarks use the same HGNC mapping as the networks
        hm = netnull.hgnc_map()
        targets = [{hm.get(g, g) for g in t} for t in targets]
        lm_symbols = np.array([hm.get(g, g) for g in lm_symbols])
    pubs = pubcounts()
    tpub = np.array([np.log1p([pubs.get(g, 0) for g in t]).mean() if t else np.nan for t in targets])
    # per-line quantities that do not depend on the network
    lines = sorted(set(line_rows)); per_line = {}
    for L in lines:
        idx = np.where(line_rows == L)[0]
        R = Y[idx] - Y[idx].mean(0); sd = Y[idx].std(0)
        per_line[L] = {"idx": idx, "R": R, "sd": sd, "norm": np.linalg.norm(R, axis=1)}
    records = []
    for name in args.networks:
        base = netnull.load(name, harmonize=args.harmonize)
        print(f"{name}: {netnull.summary(base)} ({time.time() - started:.0f}s)", flush=True)
        for kind, seed in LADDER:
            t0 = time.time()
            net = netnull.null(base, kind, seed)
            known, d_lm, d_pair, tdeg, n_lm = distances(net, targets, lm_symbols)
            for L in lines:
                P = per_line[L]; c_l = row_c[P["idx"]]; act = active[P["idx"]]
                sel = act & known[c_l]
                rec = {"slice": args.slice, "network": name, "variant": kind, "seed": seed, "line": L,
                       "landmarks_in_graph": n_lm, "n_known": int(known[c_l].sum()), "n_active_known": int(sel.sum())}
                if sel.sum() >= 10:
                    U = np.abs(P["R"][sel]) / P["sd"]; Urel = U / U.mean(1, keepdims=True)
                    rec["A_rel"], rec["A_genes"], _ = locality(Urel, d_lm[c_l[sel]])
                    rec["share_d1"] = float((d_lm[c_l[sel]] == 1).mean())
                    key = (L, name, "pairs")  # KNOWN depends on the network, not on the null variant
                    if key not in per_line:  # pair geometry and plate mask depend only on the line
                        Z = P["R"][sel] - P["R"][sel].mean(1, keepdims=True); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
                        iu, ju = np.triu_indices(int(sel.sum()), 1)
                        ps = [plates[i] for i in P["idx"][sel]]
                        diff = np.fromiter((not (ps[i] & ps[j]) for i, j in zip(iu, ju)), bool, len(iu))
                        per_line[key] = (iu, ju, (Z @ Z.T)[iu, ju], diff, c_l[sel])
                    iu, ju, r, diff, cs = per_line[key]
                    assert len(cs) == int(sel.sum()) and (cs == c_l[sel]).all()
                    if True:
                        dp = d_pair[cs[iu], cs[ju]]
                        for lab, mask in (("d0", dp == 0), ("d1", dp == 1), ("d2", dp == 2), ("d3plus", dp >= 3)):
                            mm = mask & diff
                            rec[f"B_n_{lab}"] = int(mm.sum()); rec[f"B_r_{lab}"] = float(r[mm].mean()) if mm.any() else np.nan
                allk = known[c_l] & ~np.isnan(tpub[c_l])
                if allk.sum() >= 10:
                    nrm = P["norm"][allk]; dg = tdeg[c_l[allk]]; pb = tpub[c_l[allk]]
                    rec["C_deg"] = float(spearmanr(dg, nrm).statistic)
                    rec["C_pub"] = float(spearmanr(pb, nrm).statistic)
                    rd, rp, rn = rankdata(dg), rankdata(pb), rankdata(nrm)
                    resid = rd - np.polyval(np.polyfit(rp, rd, 1), rp)
                    rec["C_deg_given_pub"] = float(spearmanr(resid, rn).statistic)
                    rec["deg_pub_rho"] = float(spearmanr(dg, pb).statistic)
                records.append(rec)
            print(f"  {kind}{seed}: {time.time() - t0:.0f}s", flush=True)
    out = BASE / "results/network_information"; out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(args.networks) if len(args.networks) < len(netnull.SOURCES) else "all"
    frame = pd.DataFrame(records); frame.to_csv(out / f"{args.slice}_{tag}.csv", index=False)
    (out / f"{args.slice}_{tag}_manifest.json").write_text(json.dumps({
        "slice": args.slice, "networks": args.networks, "ladder": LADDER, "harmonize": args.harmonize,
        "script_sha256": run.sha256(Path(__file__).resolve()), "netnull_sha256": run.sha256(Path(netnull.__file__).resolve()),
        "slice_sha256": run.sha256(d / f"slice_{args.slice}.npz"), "seconds": round(time.time() - started, 1)}, indent=2) + "\n")
    cols = [c for c in ("A_rel", "B_r_d0", "B_r_d1", "B_r_d2", "C_deg", "C_pub", "C_deg_given_pub") if c in frame]
    print(frame.groupby(["network", "variant"])[cols].mean().round(4).to_string(), flush=True)
    print(f"done ({time.time() - started:.0f}s)", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slice", default="P2-24", choices=["P1-24", "P1-6", "P2-24", "T-0.05", "T-0.5", "T-5"])
    p.add_argument("--networks", nargs="+", default=list(netnull.SOURCES), choices=list(netnull.SOURCES))
    p.add_argument("--no-harmonize", dest="harmonize", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
