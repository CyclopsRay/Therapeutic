"""Add the Cell Painting profile vectors of the Fig. 1 compounds to paper/figures/fig1_pipeline_source.json.

The card shows a compound's Cell Painting view as a vector: its consensus profile (LINCS Cell Painting, A549, 10 uM;
`morph10`, 1,026 CellProfiler features) z-scored over the drug-blind compounds that have a profile, exactly as the
kernel uses it (scripts/figures/fig1_cards_data.py), on the NFEAT features with the largest variance across those
compounds (the same features for every card). Numpy only; no rdkit needed.
Run: python -I scripts/figures/fig1_cp_profiles.py
"""
import csv
import json
from pathlib import Path

import numpy as np

D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
CP = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/l1000_unseen_drug/cell_painting_features.npz")
SRC = Path(__file__).resolve().parents[2] / "paper" / "figures" / "fig1_pipeline_source.json"
NFEAT = 40


def main() -> None:
    S = json.loads(SRC.read_text())
    rows = list(csv.DictReader(open(D / "drugs.tsv"), delimiter="\t"))
    cp = np.load(CP, allow_pickle=False)
    pos = {k[:14]: i for i, k in enumerate(cp["inchikey"].astype(str))}
    cpi = np.array([pos.get(r["inchi_key"][:14], -1) if r["inchi_key"] else -1 for r in rows])
    F = cp["morph10"][cpi[cpi >= 0]].astype(float)
    mu, sd = F.mean(0), np.maximum(F.std(0), 1e-9)
    feats = np.argsort(-cp["morph10"][np.unique(cpi[cpi >= 0])].astype(float).var(0))[:NFEAT]
    byid = {r["pert_id"]: i for i, r in enumerate(rows)}
    prof = []
    for pid in S["pert_ids"]:
        i = cpi[byid[pid]]
        prof.append(None if i < 0 else ((cp["morph10"][i].astype(float) - mu) / sd)[feats].round(3).tolist())
    S["cp_profile"] = prof
    S["cp_features"] = cp["feature"][feats].astype(str).tolist()
    SRC.write_text(json.dumps(S))
    print("compounds", S["names"], "has profile", [p is not None for p in prof], "features", S["cp_features"][:5])


if __name__ == "__main__":
    main()
