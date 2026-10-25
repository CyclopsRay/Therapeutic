#!/usr/bin/env python3
"""Overlap audit: does PrimeKG / DrugBank add drug-target knowledge for the unseen-drug branch?

report/10_09_record.md (2026-10-03). PrimeKG v2.1 drug nodes are DrugBank IDs without
structures; DrugBank's own vocabulary download needs an account (HTTP 403), so DrugBank IDs are
mapped to InChIKeys through PubChem PUG-REST (DrugBank deposits its records as PubChem
substances). The mapping is cached. Compounds are then matched to the L1000 slices and Tahoe by
full InChIKey and by connectivity block (ik14), and PrimeKG drug-protein edges are compared with
the LINCS2020 `compoundinfo_beta` targets used by the benchmark. PrimeKG protein-protein edges are
compared with STRING v12 physical (score >= 400).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import run_l1000_unseen_drug as run

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def fetch_json(url: str) -> dict:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as handle:
                return json.load(handle)
        except Exception as error:  # noqa: BLE001 - transient PubChem throttling
            if "404" in str(error):
                return {}
            time.sleep(2 + 3 * attempt)
    return {}


def drugbank_to_inchikey(ids: list[str], cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache, sep="\t")
    rows = []
    for start in range(0, len(ids), 100):
        batch = ",".join(ids[start:start + 100])
        xref = fetch_json(f"{PUG}/substance/sourceid/DrugBank/{batch}/xrefs/RegistryID/JSON")
        cids = fetch_json(f"{PUG}/substance/sourceid/DrugBank/{batch}/cids/JSON?cids_type=standardized")
        sid_db = {}
        for info in xref.get("InformationList", {}).get("Information", []):
            reg = [r for r in info.get("RegistryID", []) if str(r).startswith("DB")]
            if reg:
                sid_db[info["SID"]] = reg[0]
        for info in cids.get("InformationList", {}).get("Information", []):
            if info.get("SID") in sid_db and info.get("CID"):
                rows.append((sid_db[info["SID"]], info["CID"][0]))
        time.sleep(0.4)
    frame = pd.DataFrame(rows, columns=["drugbank_id", "cid"]).drop_duplicates("drugbank_id")
    keys = []
    for start in range(0, len(frame), 200):
        batch = ",".join(map(str, frame.cid.iloc[start:start + 200]))
        props = fetch_json(f"{PUG}/compound/cid/{batch}/property/InChIKey/JSON")
        keys += [(p["CID"], p["InChIKey"]) for p in props.get("PropertyTable", {}).get("Properties", [])]
        time.sleep(0.4)
    frame = frame.merge(pd.DataFrame(keys, columns=["cid", "inchikey"]).drop_duplicates("cid"), on="cid", how="left")
    frame.to_csv(cache, sep="\t", index=False)
    return frame


def main(args: argparse.Namespace) -> None:
    kg = []
    for chunk in pd.read_csv(args.primekg / "kg.csv", usecols=["relation", "display_relation", "x_id", "x_type",
                                                              "x_name", "y_id", "y_type", "y_name"],
                             chunksize=1_000_000, low_memory=False):
        kg.append(chunk[chunk.relation.isin(["drug_protein", "protein_protein"])])
    kg = pd.concat(kg)
    dp = kg[kg.relation.eq("drug_protein") & kg.x_type.eq("drug")]
    print("PrimeKG drug_protein edges (one direction):", len(dp), dp.display_relation.value_counts().to_dict(), flush=True)
    dmap = drugbank_to_inchikey(sorted(dp.x_id.astype(str).unique()), args.cache)
    dmap = dmap.dropna(subset=["inchikey"])
    print(f"DrugBank drugs with protein edges: {dp.x_id.nunique():,}; mapped to InChIKey via PubChem: {len(dmap):,}", flush=True)
    dp = dp.merge(dmap, left_on="x_id", right_on="drugbank_id")
    dp["ik14"] = dp.inchikey.str[:14]
    pk_targets_full = dp[dp.display_relation.eq("target")].groupby("inchikey").y_name.agg(set)
    pk_targets_ik14 = dp[dp.display_relation.eq("target")].groupby("ik14").y_name.agg(set)
    pk_any_ik14 = dp.groupby("ik14").y_name.agg(set)
    # export PrimeKG targets (display_relation == target) per InChIKey for the runner's --targets union
    exp = dp[dp.display_relation.eq("target")].groupby(["inchikey", "ik14"]).y_name.agg(lambda x: ";".join(sorted(set(x))))
    exp.rename("targets").reset_index().to_csv(args.derived / "primekg_targets.tsv", sep="\t", index=False)

    table = pd.read_parquet(args.derived / "drug_table.parquet")
    genes, _, _ = run.load_ppi(args.string_root, 400, None)
    string_genes = set(genes)
    rows = []
    for name, keys in [(s, sorted(set(np.load(args.derived / f"slice_{s}.npz")["inchikey"].astype(str))))
                       for s in ("P1-24", "P1-6", "P2-24")] + [("Tahoe", sorted(set(table[table.source.eq("Tahoe") & table.inchikey.ne("")].inchikey)))]:
        src = "Tahoe" if name == "Tahoe" else "L1000"
        sub = table[table.source.eq(src) & table.inchikey.isin(keys)]
        ci = sub.groupby("inchikey").targets.agg(lambda x: set(";".join(x).split(";")) - {""})
        ik14 = sub.groupby("inchikey").ik14.first()
        n_ci = n_pk = n_union = n_pk_only = n_both = 0
        jacc = []
        for k in keys:
            a = ci.get(k, set())
            b = pk_targets_full.get(k, pk_targets_ik14.get(ik14.get(k, ""), set()))
            n_ci += bool(a); n_pk += bool(b); n_union += bool(a | b); n_pk_only += bool(b and not a)
            if a and b:
                n_both += 1; jacc.append(len(a & b) / len(a | b))
        in_kg = sum(k in pk_targets_full.index or ik14.get(k, "") in pk_any_ik14.index for k in keys)
        rows.append({"set": name, "compounds": len(keys), "in_primekg_any_protein_edge": int(in_kg),
                     "with_compoundinfo_or_tahoe_targets": n_ci, "with_primekg_targets": n_pk,
                     "with_either": n_union, "primekg_only": n_pk_only, "both": n_both,
                     "median_jaccard_when_both": float(np.median(jacc)) if jacc else np.nan,
                     "primekg_target_genes_in_string": float(np.mean([g in string_genes for t in
                                                         pk_targets_ik14.values for g in t]))})
    overlap = pd.DataFrame(rows)
    pp = kg[kg.relation.eq("protein_protein")]
    pk_edges = {tuple(sorted(e)) for e in zip(pp.x_name, pp.y_name)}
    info = pd.read_csv(args.string_root / "9606.protein.info.v12.0.txt.gz", sep="\t")
    nm = dict(zip(info["#string_protein_id"], info["preferred_name"]))
    st = pd.read_csv(args.string_root / "9606.protein.physical.links.v12.0.txt.gz", sep=" ")
    st = st[st.combined_score >= 400]
    st_edges = {tuple(sorted(e)) for e in zip(st.protein1.map(nm), st.protein2.map(nm))}
    lm = set(np.load(args.derived / "slice_P1-24.npz")["gene_symbol"].astype(str))
    pk_genes = set(pp.x_name) | set(pp.y_name)
    ppi = {"primekg_ppi_edges": len(pk_edges), "string_physical400_edges": len(st_edges),
           "shared_edges": len(pk_edges & st_edges), "primekg_genes": len(pk_genes), "string_genes": len(string_genes),
           "landmarks_in_primekg": len(lm & pk_genes), "landmarks_in_string": len(lm & string_genes),
           "n_landmarks": len(lm)}
    args.out.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(args.out / "drug_target_overlap.csv", index=False)
    (args.out / "ppi_overlap.json").write_text(json.dumps(ppi, indent=2) + "\n")
    pd.set_option("display.width", 220)
    print(overlap.to_string(index=False)); print(json.dumps(ppi, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base = Path("/nfs-stor/yuqi.lei/Therapeutic")
    p.add_argument("--primekg", type=Path, default=base / "raw/primekg_v2.1")
    p.add_argument("--derived", type=Path, default=base / "derived/l1000_unseen_drug")
    p.add_argument("--string-root", type=Path, default=base / "raw/string_v12")
    p.add_argument("--cache", type=Path, default=base / "derived/l1000_unseen_drug/primekg_drugbank_pubchem_inchikey.tsv")
    p.add_argument("--out", type=Path, default=base / "results/primekg_overlap")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
