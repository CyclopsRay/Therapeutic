#!/usr/bin/env python3
"""X1 (exploratory, post hoc): is the same-plate advantage chemistry or batch?

report/10_09_record.md, 2026-10-03 evening. Among ACTIVE compounds of one line, compound pairs
(different ik14) are binned by Morgan Tanimoto and split by whether they share an L1000 plate in
that line. At equal chemical similarity, a response-correlation excess for same-plate pairs is a
batch artifact that a new compound (measured on its own plate) could never benefit from.
The same split is reported for Cell Painting morphology cosine (quintiles), and the overlap
between "same L1000 plate" and "same Cell Painting plate map" pairs is measured.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import run_l1000_unseen_drug as run


def main(args: argparse.Namespace) -> None:
    d = args.derived
    feats = np.load(d / "drug_features.npz", allow_pickle=False)
    fpos = pd.Series(np.arange(len(feats["inchikey"])), index=feats["inchikey"].astype(str))
    table = pd.read_parquet(d / "drug_table.parquet")
    ik14_of = table[table.source.eq("L1000")].groupby("inchikey").ik14.first()
    cp = np.load(d / "cell_painting_features.npz", allow_pickle=False)
    cpos = pd.Series(np.arange(len(cp["inchikey"])), index=cp["inchikey"].astype(str))
    raw = pd.read_csv(args.cp_raw, usecols=["Metadata_broad_sample", "Metadata_Plate_Map_Name"])
    l1000 = table[table.source.eq("L1000") & table.inchikey.ne("")]
    key_of = dict(zip(l1000.source_id, l1000.inchikey))
    raw["inchikey"] = raw.Metadata_broad_sample.astype(str).str[:13].map(key_of)
    maps = raw.dropna().groupby("inchikey").Metadata_Plate_Map_Name.agg(lambda x: set(x))
    rows_out, morph_out, overlap_out = [], [], []
    for name in args.slices:
        sl = np.load(d / f"slice_{name}.npz", allow_pickle=False)
        keys = sl["inchikey"].astype(str); lines = sl["line"].astype(str)
        for L in sorted(set(lines)):
            idx = np.where((lines == L) & (sl["tas"] >= 0.2))[0]
            k = keys[idx]
            C = run.corr_matrix(sl["Y"][idx].astype(np.float64), sl["Y"][idx].astype(np.float64))
            F = feats["morgan2"][fpos[k].to_numpy()]
            T = run.tanimoto(F, F)
            plates = [set(p.split("|")) for p in sl["plates"][idx].astype(str)]
            pid = {p: i for i, p in enumerate(sorted(set().union(*plates)))}
            B = np.zeros((len(idx), len(pid)), np.float32)
            for i, ps in enumerate(plates):
                B[i, [pid[p] for p in ps]] = 1
            SP = (B @ B.T) > 0
            iu = np.triu_indices(len(idx), 1)
            g14 = ik14_of.reindex(k).to_numpy()
            keep = g14[iu[0]] != g14[iu[1]]
            r, t, sp = C[iu][keep], T[iu][keep], SP[iu][keep]
            for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, 1.01)):
                for same in (True, False):
                    m = (t >= lo) & (t < hi) & (sp == same)
                    rows_out.append({"slice": name, "line": L, "tanimoto": f"[{lo},{hi})", "same_plate": same,
                                     "n_pairs": int(m.sum()),
                                     "median_r": float(np.median(r[m])) if m.any() else np.nan,
                                     "rate_r05": float((r[m] >= 0.5).mean()) if m.any() else np.nan})
            # morphology pairs (compounds with a 10 uM profile)
            has = np.array([x in cpos.index for x in k])
            if has.sum() > 50:
                hi_ = np.where(has)[0]
                M = cp["morph10"][cpos[k[hi_]].to_numpy()].astype(np.float64)
                M = (M - M.mean(0)) / np.maximum(M.std(0), 1e-9)
                Sm = run.cosine(M, M)
                ju = np.triu_indices(len(hi_), 1)
                a, b = hi_[ju[0]], hi_[ju[1]]
                ok = g14[a] != g14[b]
                a, b, sm = a[ok], b[ok], Sm[ju][ok]
                rr, spp = C[a, b], SP[a, b]
                same_map = np.array([bool(maps.get(k[x], set()) & maps.get(k[y], set())) for x, y in zip(a, b)])
                overlap_out.append({"slice": name, "line": L, "n_pairs": len(a),
                                    "frac_same_l1000_plate": float(spp.mean()), "frac_same_cp_map": float(same_map.mean()),
                                    "frac_cp_map_given_l1000_plate": float(same_map[spp].mean()) if spp.any() else np.nan,
                                    "frac_cp_map_given_diff_plate": float(same_map[~spp].mean())})
                q = np.quantile(sm, [0, .5, .9, .99, 1.0])
                for qi in range(4):
                    for same in (True, False):
                        m = (sm >= q[qi]) & (sm <= q[qi + 1]) & (spp == same)
                        morph_out.append({"slice": name, "line": L, "morph_quantile": ["0-50", "50-90", "90-99", "99-100"][qi],
                                          "same_l1000_plate": same, "n_pairs": int(m.sum()),
                                          "median_r": float(np.median(rr[m])) if m.any() else np.nan,
                                          "rate_r05": float((rr[m] >= 0.5).mean()) if m.any() else np.nan})
            print(f"{name} {L}: {len(idx)} active, same-plate pair share {sp.mean():.4f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    a = pd.DataFrame(rows_out); b = pd.DataFrame(morph_out); c = pd.DataFrame(overlap_out)
    a.to_csv(args.out / "tanimoto_by_plate.csv", index=False)
    b.to_csv(args.out / "morph_by_plate.csv", index=False)
    c.to_csv(args.out / "plate_map_overlap.csv", index=False)
    pd.set_option("display.width", 200)
    print(a.groupby(["slice", "tanimoto", "same_plate"])[["n_pairs", "median_r", "rate_r05"]]
          .agg({"n_pairs": "sum", "median_r": "mean", "rate_r05": "mean"}).round(4).to_string())
    print(b.groupby(["slice", "morph_quantile", "same_l1000_plate"])[["n_pairs", "median_r", "rate_r05"]]
          .agg({"n_pairs": "sum", "median_r": "mean", "rate_r05": "mean"}).round(4).to_string())
    print(c.groupby("slice").mean(numeric_only=True).round(4).to_string())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base = Path("/nfs-stor/yuqi.lei/Therapeutic")
    p.add_argument("--slices", nargs="+", default=["P2-24", "P1-24"])
    p.add_argument("--derived", type=Path, default=base / "derived/l1000_unseen_drug")
    p.add_argument("--cp-raw", type=Path, default=base / "raw/lincs_cell_painting/2016_04_01_a549_48hr_batch1_dmso_spherized_profiles_with_input_normalized_by_dmso_consensus_modz.csv.gz")
    p.add_argument("--out", type=Path, default=base / "results/l1000_plate_confound")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
