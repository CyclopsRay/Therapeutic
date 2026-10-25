#!/usr/bin/env python3
"""Turn a run's summary.json (compare_l1000_e2e.py) into the two tracker documents: arms/<run> and contrasts/<run>.

Writes <outdir>/arms.json and <outdir>/contrasts.json, each the body of one artifact-database document.
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="e.g. P1-24_ik14_seed0 plus tag")
p.add_argument("--outdir", type=Path, required=True)
a = p.parse_args()
s = json.loads((Path("/nfs-stor/yuqi.lei/Therapeutic/results/l1000_e2e") / a.run / "summary.json").read_text())
arms = [{"cohort": c, "arm": arm, "pearson_full": round(v["pearson_full"], 5), "pearson_resid": round(v["pearson_resid"], 5),
         "rows": v["rows"]} for c, d in s["absolute"].items() for arm, v in d.items()]
con = [{k: (round(r[k], 5) if isinstance(r[k], float) else r[k]) for k in
        ("arm", "minus", "cohort", "metric", "diff", "low", "high", "lines_positive", "n_lines", "rows")} for r in s["contrasts"]]
a.outdir.mkdir(parents=True, exist_ok=True)
(a.outdir / "arms.json").write_text(json.dumps({"rows": arms, "mean_agreement": s["mean_agreement"]}))
(a.outdir / "contrasts.json").write_text(json.dumps({"rows": con}))
print(f"{len(arms)} arm rows, {len(con)} contrasts -> {a.outdir}")
