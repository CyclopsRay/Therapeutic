#!/usr/bin/env python3
"""Per-compound Cell Painting (A549, 48 h) morphology features for the unseen-drug branch.

Source: broadinstitute/lincs-cell-painting, batch 2016_04_01_a549_48hr_batch1, spherized
DMSO-normalized consensus (MODZ) profiles (Way et al., Cell Systems 2022; CC0 data).
Compounds are joined to L1000 through the 13-character Broad ID (BRD-Xnnnnnnnn), which is
the L1000 pert_id, and then to the standardized-parent InChIKey of build_drug_features.py.

Outputs <out>/cell_painting_features.npz:
  inchikey     (n,) str
  morph10      (n, n_feat) float32  profile at 10 uM (dose_recode 6), mean over plate maps
  morph_doses  (n, 6 * n_feat) float32  doses 0.04-10 uM concatenated; a missing dose is 0
               (the spherized DMSO-normalized value of an untreated well)
  feature      (n_feat,) str
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = "2016_04_01_a549_48hr_batch1_dmso_spherized_profiles_with_input_normalized_by_dmso_consensus_modz.csv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args: argparse.Namespace) -> None:
    src = args.raw / SOURCE
    frame = pd.read_csv(src, low_memory=False)
    frame = frame[frame.Metadata_broad_sample.astype(str).str.startswith("BRD-")].copy()
    features = [c for c in frame.columns if not c.startswith("Metadata")]
    table = pd.read_parquet(args.derived / "drug_table.parquet")
    l1000 = table[table.source.eq("L1000") & table.inchikey.ne("")]
    key_of = dict(zip(l1000.source_id, l1000.inchikey))
    frame["inchikey"] = frame.Metadata_broad_sample.astype(str).str[:13].map(key_of)
    unmatched = int(frame.inchikey.isna().sum())
    frame = frame[frame.inchikey.notna()]
    X = frame[features].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        frame[features] = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    keys = sorted(frame.inchikey.unique())
    pos = {k: i for i, k in enumerate(keys)}
    morph10 = np.zeros((len(keys), len(features)))
    has10 = np.zeros(len(keys), bool)
    doses = np.zeros((len(keys), 6, len(features)))
    for (key, dose), sub in frame.groupby(["inchikey", "Metadata_dose_recode"]):
        if 1 <= int(dose) <= 6:
            doses[pos[key], int(dose) - 1] = sub[features].mean().to_numpy()
        if int(dose) == 6:
            morph10[pos[key]] = sub[features].mean().to_numpy(); has10[pos[key]] = True
    keep10 = np.where(has10)[0]
    np.savez(args.derived / "cell_painting_features.npz",
             inchikey=np.asarray([keys[i] for i in keep10], dtype=str),
             morph10=morph10[keep10].astype(np.float32),
             morph_doses=doses[keep10].reshape(len(keep10), -1).astype(np.float32),
             feature=np.asarray(features, dtype=str))
    manifest = {"source": str(src), "source_sha256": sha256(src), "n_profiles_used": len(frame),
                "n_rows_unmatched_to_l1000": unmatched, "n_compounds_with_10uM": int(has10.sum()),
                "n_compounds_any_dose": len(keys), "n_features": len(features),
                "runner_sha256": sha256(Path(__file__).resolve())}
    (args.derived / "cell_painting_features.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base = Path("/nfs-stor/yuqi.lei/Therapeutic")
    p.add_argument("--raw", type=Path, default=base / "raw/lincs_cell_painting")
    p.add_argument("--derived", type=Path, default=base / "derived/l1000_unseen_drug")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
