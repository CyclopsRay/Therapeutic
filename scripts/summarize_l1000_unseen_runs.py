#!/usr/bin/env python3
"""Pivot the key contrasts of every unseen-drug run into one table (report/10_09_report.md)."""
import argparse
from pathlib import Path

import pandas as pd

KEY = [
    ("ACTIVE", "knn_chem_select", "mean"), ("ACTIVE", "oracle_knn", "mean"),
    ("ACTIVE", "knn_chemberta", "knn_morgan2"), ("ACTIVE", "knn_molformer", "knn_morgan2"),
    ("ACTIVE", "krr_morgan2", "mean"),
    ("KNOWN_ACTIVE", "knn_ppi", "knn_chem_known"), ("KNOWN_ACTIVE", "knn_ppi", "knn_ppi_rewired"),
    ("KNOWN_ACTIVE", "knn_target", "knn_chem_known"), ("KNOWN_ACTIVE", "knn_chem_target", "knn_chem_known"),
    ("KNOWN_ACTIVE", "krr_chem_target", "krr_chem_known"), ("KNOWN_ACTIVE", "krr_ppi", "krr_ppi_rewired"),
    ("XCELL_ACTIVE", "xcell1_knn", "knn_chem_select"), ("XCELL_ACTIVE", "xcell_knn", "knn_chem_select"),
    ("MORPH_ACTIVE", "knn_morph", "knn_chem_morphpool"),
]


def main(a):
    rows = []
    for run_dir in sorted(p for p in a.results.iterdir() if p.is_dir() and not p.name.startswith("_")):
        f = run_dir / "contrasts.csv"
        if not f.exists():
            continue
        c = pd.read_csv(f)
        c = c[c.metric == a.metric]
        for cohort, arm, minus in KEY:
            hit = c[(c.cohort == cohort) & (c.arm == arm) & (c.minus == minus)]
            if len(hit):
                h = hit.iloc[0]
                rows.append({"run": run_dir.name, "contrast": f"{arm} - {minus} ({cohort})",
                             "value": f"{h['diff']:+.4f} [{h.boot_low:+.4f}, {h.boot_high:+.4f}] {int(h.lines_positive)}/{int(h.n_lines)}"})
    table = pd.DataFrame(rows).pivot(index="contrast", columns="run", values="value")
    order = [f"{arm} - {minus} ({cohort})" for cohort, arm, minus in KEY]
    table = table.reindex([o for o in order if o in table.index])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out)
    with pd.option_context("display.width", 400, "display.max_colwidth", 60):
        print(table.T.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    base = Path("/nfs-stor/yuqi.lei/Therapeutic/results")
    p.add_argument("--results", type=Path, default=base / "l1000_unseen_drug")
    p.add_argument("--metric", default="pearson_full")
    p.add_argument("--out", type=Path, default=base / "l1000_unseen_drug/_summary_pearson_full.csv")
    main(p.parse_args())
