#!/usr/bin/env python3
"""PrimeKG subgraph for the integrated end-to-end model (phase 1, unseen drug; record 2026-10-09).

Keeps the relations the user's two KG arms need (kg_a: PPI + drug-disease + disease-protein;
kg_b: drug-disease + disease-protein) plus drug_protein, stored but unused by both arms. Each
relation is split by direction (x_type -> y_type), since kg.csv lists both directions of every edge.
Drug nodes are matched to L1000/Tahoe compounds by full InChIKey, then by connectivity block (ik14),
through the PubChem DrugBank -> InChIKey cache of the PrimeKG audit. Gene names are HGNC-harmonised.
Writes derived/e2e_kg/v1/{graph.npz, nodes.parquet, drug_map.tsv, disease_text.parquet, manifest.json}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import netnull  # noqa: E402

KEEP = ("protein_protein", "indication", "contraindication", "off-label use", "disease_protein", "drug_protein")
TEXT_COLS = ("mondo_definition", "umls_description", "orphanet_definition")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(a: argparse.Namespace) -> None:
    t0 = time.time()
    print(f"host {platform.node()} python {sys.version.split()[0]} pandas {pd.__version__} numpy {np.__version__}",
          flush=True)
    cols = ["relation", "x_index", "x_id", "x_type", "x_name", "y_index", "y_id", "y_type", "y_name"]
    parts = [c[c.relation.isin(KEEP)] for c in pd.read_csv(a.primekg / "kg.csv", usecols=cols, dtype=str,
                                                           chunksize=2_000_000)]
    kg = pd.concat(parts, ignore_index=True)
    kg["x_index"] = kg.x_index.astype(int); kg["y_index"] = kg.y_index.astype(int)
    kg["rel"] = kg.relation + ":" + kg.x_type + "->" + kg.y_type
    print(kg.rel.value_counts().to_string(), flush=True)

    nodes = pd.concat([kg[["x_index", "x_id", "x_type", "x_name"]].set_axis(["kg_index", "id", "type", "name"], axis=1),
                       kg[["y_index", "y_id", "y_type", "y_name"]].set_axis(["kg_index", "id", "type", "name"], axis=1)])
    nodes = nodes.drop_duplicates("kg_index").sort_values(["type", "kg_index"]).reset_index(drop=True)
    hg = netnull.hgnc_map()
    nodes["name_hgnc"] = np.where(nodes.type.eq("gene/protein"), nodes.name.map(lambda s: hg.get(s, s)), nodes.name)
    pos = pd.Series(np.arange(len(nodes)), index=nodes.kg_index)
    rel_names = sorted(kg.rel.unique())
    rid = {r: i for i, r in enumerate(rel_names)}
    src = pos[kg.x_index].to_numpy(); dst = pos[kg.y_index].to_numpy(); rel = kg.rel.map(rid).to_numpy()
    edges = np.unique(np.stack([src, dst, rel], 1), axis=0)
    types = sorted(nodes.type.unique())
    print(f"nodes {len(nodes):,} ({nodes.type.value_counts().to_dict()}), edges {len(edges):,}, "
          f"relations {len(rel_names)}", flush=True)

    # drugs: DrugBank -> InChIKey (PubChem cache), one node per InChIKey (the drug with most kept edges)
    dmap = pd.read_csv(a.drug_cache, sep="\t", dtype=str).dropna(subset=["inchikey"])
    drug_nodes = nodes[nodes.type.eq("drug")].merge(dmap, left_on="id", right_on="drugbank_id")
    deg = np.bincount(np.r_[edges[:, 0], edges[:, 1]], minlength=len(nodes))
    drug_nodes["node"] = pos[drug_nodes.kg_index].to_numpy(); drug_nodes["degree"] = deg[drug_nodes.node]
    drug_nodes["ik14"] = drug_nodes.inchikey.str[:14]
    drug_nodes = drug_nodes.sort_values("degree", ascending=False)
    drug_map = drug_nodes[["inchikey", "ik14", "drugbank_id", "node", "degree"]]
    n_drug = int(nodes.type.eq("drug").sum())
    print(f"drug nodes {n_drug:,}; mapped to InChIKey {drug_map.drugbank_id.nunique():,}", flush=True)

    # disease text: name + first available definition (Mondo, UMLS, Orphanet)
    feat = pd.read_csv(a.primekg / "disease_features.tab", dtype=str)
    feat["kg_index"] = feat.node_index.astype(int)
    feat["definition"] = feat[list(TEXT_COLS)].bfill(axis=1).iloc[:, 0]
    dis = nodes[nodes.type.eq("disease")][["kg_index", "name"]].merge(
        feat[["kg_index", "definition"]].drop_duplicates("kg_index"), on="kg_index", how="left")
    dis["node"] = pos[dis.kg_index].to_numpy()
    print(f"disease nodes {len(dis):,}; with a definition {dis.definition.notna().sum():,}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "graph.npz", src=edges[:, 0].astype(np.int64), dst=edges[:, 1].astype(np.int64),
                        rel=edges[:, 2].astype(np.int64), rel_names=np.array(rel_names),
                        node_type=nodes.type.map({t: i for i, t in enumerate(types)}).to_numpy().astype(np.int64),
                        type_names=np.array(types))
    nodes.to_parquet(a.out / "nodes.parquet", index=False)
    drug_map.to_csv(a.out / "drug_map.tsv", sep="\t", index=False)
    dis.to_parquet(a.out / "disease_text.parquet", index=False)
    dis.to_csv(a.out / "disease_text.csv", index=False)  # text-embed env has no parquet engine
    # coverage of the L1000/Tahoe slices
    cov = {}
    by_full = drug_map.drop_duplicates("inchikey").set_index("inchikey").node
    by_blk = drug_map.drop_duplicates("ik14").set_index("ik14").node
    dd_rel = [rid[r] for r in rel_names if r.split(":")[0] in ("indication", "contraindication", "off-label use")]
    dd_nodes = set(edges[np.isin(edges[:, 2], dd_rel), 0])
    for s in ("P1-24", "P1-6", "P2-24", "T-5"):
        f = a.derived / f"slice_{s}.npz"
        if not f.exists():
            continue
        keys = sorted(set(np.load(f)["inchikey"].astype(str)))
        node = [by_full.get(k, by_blk.get(k[:14], -1)) for k in keys]
        cov[s] = {"compounds": len(keys), "full_match": int(sum(k in by_full.index for k in keys)),
                  "full_or_ik14": int(sum(n >= 0 for n in node)),
                  "with_drug_disease_edge": int(sum(n in dd_nodes for n in node))}
        print(f"{s}: {cov[s]}", flush=True)
    (a.out / "manifest.json").write_text(json.dumps({
        "relations_kept": KEEP, "rel_names": rel_names, "n_nodes": int(len(nodes)), "n_edges": int(len(edges)),
        "node_types": nodes.type.value_counts().to_dict(), "coverage": cov,
        "kg_sha256": sha256(a.primekg / "kg.csv"), "disease_features_sha256": sha256(a.primekg / "disease_features.tab"),
        "drug_cache_sha256": sha256(a.drug_cache), "script_sha256": sha256(Path(__file__).resolve()),
        "seconds": round(time.time() - t0, 1)}, indent=2) + "\n")
    print(f"done in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    base = Path("/nfs-stor/yuqi.lei/Therapeutic")
    p = argparse.ArgumentParser()
    p.add_argument("--primekg", type=Path, default=base / "raw/primekg_v2.1")
    p.add_argument("--drug-cache", type=Path, default=base / "derived/l1000_unseen_drug/primekg_drugbank_pubchem_inchikey.tsv")
    p.add_argument("--derived", type=Path, default=base / "derived/l1000_unseen_drug")
    p.add_argument("--out", type=Path, default=base / "derived/e2e_kg/v1")
    main(p.parse_args())
