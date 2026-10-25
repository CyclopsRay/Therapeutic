#!/usr/bin/env python3
"""Model-free gate for the measured causal gene -> gene prior (record 10_16, 2026-10-14 06:10).

Question: does a compound's L1000 response in a line resemble the consensus knockdown (trt_sh.cgs) signature of its
annotated targets in the same line, beyond what (A) any other compound's targets or (B) random knocked-down genes give?
No drug-response data enters the prediction, so this is a strict unseen-drug quantity.

Rows: ACTIVE (TAS >= 0.2) compound-line rows of the slice whose compound has >= 1 annotated target with a same-line
knockdown. Primary cohort: rows whose MOA is not an activator (agonist / activator / opener; "antagonist" is not an
agonist), where knockdown and drug should act in the same direction. Response = residual from the line mean over all
the slice's rows; prediction = mean knockdown signature of the targets (time points averaged per gene and line).
Statistic: per-row Pearson over the 978 landmarks, line-macro mean.

Nulls (200 draws each, seed 0): A = target sets permuted among primary rows of the same line; B = each row's
knocked-down targets replaced by the same number of random genes knocked down in that line.

Gate (fixed before any result): line-macro real - mean(null) >= +0.02 AND real above the null's 95th percentile in
>= 5 of the lines, for both null A and null B. Secondary, not gated: activator rows with knockdown sign flipped, and
over-expression (trt_oe) for activator rows. Also reports plate overlap between knockdown and compound plates.
"""
import argparse
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/nfs-stor/yuqi.lei/Therapeutic")
ACT = re.compile(r"(?<!ant)agonist|activator|opener|stimulant", re.I)


def zrows(X):
    X = X - X.mean(1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1)


def per_gene_signatures(acc, pert_type, symbols):
    z = np.load(BASE / f"derived/l1000_unseen_drug/{acc}_{pert_type}_landmarks.npz", allow_pickle=False)
    order = pd.Index(z["gene_symbol"]).get_indexer(symbols)
    if (order < 0).any():
        raise ValueError(f"{(order < 0).sum()} slice landmarks missing from {pert_type} extraction")
    info = pd.read_csv(BASE / f"raw/lincs-l1000/{acc}/sig_info.txt.gz", sep="\t",
                       usecols=["sig_id", "pert_iname", "cell_id"]).set_index("sig_id")
    meta = info.loc[z["sig_id"]]
    V = z["values"][:, order]
    df = pd.DataFrame({"gene": meta.pert_iname.to_numpy(), "line": meta.cell_id.to_numpy()})
    sig = {}
    for (g, l), idx in df.groupby(["gene", "line"]).indices.items():
        sig[(g, l)] = V[idx].mean(0)
    plates = {s.split(":")[0] for s in z["sig_id"]}
    return sig, plates


def line_macro(lines, r):
    return pd.Series(r).groupby(np.asarray(lines)).mean()


def run(args):
    t0 = time.time()
    sl = np.load(BASE / f"derived/l1000_unseen_drug/slice_{args.slice}.npz", allow_pickle=False)
    acc = json.loads((BASE / f"derived/l1000_unseen_drug/slice_{args.slice}.json").read_text())["accession"]
    Y, ik, line, tas = sl["Y"], sl["inchikey"], sl["line"], sl["tas"]
    symbols = sl["gene_symbol"]
    R = Y.copy()
    for l in np.unique(line):
        m = line == l
        R[m] -= Y[m].mean(0)
    dt = pd.read_parquet(BASE / "derived/l1000_unseen_drug/drug_table.parquet").drop_duplicates("inchikey")
    dt = dt.set_index("inchikey")
    tg = dt.targets.fillna("").map(lambda s: sorted(set(x.strip() for x in re.split(r"[;|,]", s)) - {""}))
    moa = dt.moa.fillna("")
    # knockdown signatures come from the Phase I archive (GSE92742) for both slices
    kd, kd_plates = per_gene_signatures("GSE92742", "trt_sh.cgs", symbols)
    kd_genes = {}
    for g, l in kd:
        kd_genes.setdefault(l, set()).add(g)
    pools = {l: np.array(sorted(gs)) for l, gs in kd_genes.items()}

    rows = []
    for i in np.where(tas >= 0.2)[0]:
        T = [t for t in tg.get(ik[i], []) if t in kd_genes.get(line[i], ())]
        if T:
            rows.append((i, T, bool(ACT.search(moa.get(ik[i], "")))))
    print(f"{args.slice}: {int((tas >= 0.2).sum())} ACTIVE rows, {len(rows)} with a same-line target knockdown "
          f"({sum(not a for _, _, a in rows)} non-activator)", flush=True)

    def pred(T, l):
        return np.mean([kd[(t, l)] for t in T], axis=0)

    out = {"slice": args.slice, "accession": acc, "n_active": int((tas >= 0.2).sum()), "n_rows_kd": len(rows),
           "host": platform.node(), "python": sys.version.split()[0], "numpy": np.__version__,
           "pandas": pd.__version__, "seed": args.seed, "draws": args.draws, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    rng = np.random.default_rng(args.seed)
    for name, keep, sign in (("primary_nonactivator", lambda a: not a, 1.0), ("activator_flipped", lambda a: a, -1.0)):
        sub = [(i, T) for i, T, a in rows if keep(a)]
        if len(sub) < 20:
            out[name] = {"n_rows": len(sub), "note": "too few rows"}
            continue
        idx = np.array([i for i, _ in sub]); L = line[idx]
        Rz = zrows(R[idx])
        P = sign * np.stack([pred(T, line[i]) for i, T in sub])
        real = (Rz * zrows(P)).sum(1)
        real_l = line_macro(L, real)
        nullA, nullB = [], []
        by_line = {l: np.where(L == l)[0] for l in np.unique(L)}
        for _ in range(args.draws):
            PA = np.empty_like(P)
            for l, pos in by_line.items():
                perm = rng.permutation(pos)
                for a, b in zip(pos, perm):
                    PA[a] = pred(sub[b][1], l)
            nullA.append(line_macro(L, (Rz * zrows(sign * PA)).sum(1)))
            PB = np.empty_like(P)
            for k, (i, T) in enumerate(sub):
                PB[k] = pred(list(rng.choice(pools[line[i]], size=len(T), replace=False)), line[i])
            nullB.append(line_macro(L, (Rz * zrows(sign * PB)).sum(1)))
        res = {"n_rows": len(sub), "n_compounds": int(len(set(ik[idx]))),
               "rows_per_line": {l: int(len(p)) for l, p in by_line.items()},
               "real_per_line": real_l.round(5).to_dict(), "real_line_macro": round(float(real_l.mean()), 5)}
        for nm, nl in (("null_A_permuted_targets", nullA), ("null_B_random_kd_genes", nullB)):
            N = pd.DataFrame(nl)
            macro = N.mean(1)
            above = int((real_l > N.quantile(0.95)).sum())
            res[nm] = {"mean_line_macro": round(float(macro.mean()), 5),
                       "p95_line_macro": round(float(macro.quantile(0.95)), 5),
                       "real_minus_mean": round(float(real_l.mean() - macro.mean()), 5),
                       "lines_above_p95": above, "n_lines": int(len(real_l)),
                       "empirical_p": round(float((macro >= real_l.mean()).mean()), 4)}
        if name == "primary_nonactivator":
            res["gate_pass"] = all(res[k]["real_minus_mean"] >= 0.02 and res[k]["lines_above_p95"] >= 5
                                   for k in ("null_A_permuted_targets", "null_B_random_kd_genes"))
            pd.DataFrame({"line": L, "inchikey": ik[idx], "pearson": real}).to_parquet(
                args.out / args.slice / "per_row.parquet")
        out[name] = res
    cp_plates = {p for s in sl["plates"] for p in str(s).split("|") if p}
    out["plate_overlap"] = {"kd_plates": len(kd_plates), "cp_plates_in_slice": len(cp_plates),
                            "shared": len(kd_plates & cp_plates)}
    out["seconds"] = round(time.time() - t0, 1)
    (args.out / args.slice / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slice", default="P1-24", choices=["P1-24", "P2-24"])
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=BASE / "results/kd_target_concordance")
    a = p.parse_args()
    (a.out / a.slice).mkdir(parents=True, exist_ok=True)
    run(a)
