"""Appendix figures of the UnDetect paper (user 2026-11-02: no room in the main text).

fig_app_transfer
  (a) Tahoe-100M: gain of the cross-line term on the deviation from the line mean (pool_D - mkl_D, residual Pearson,
      line-macro, ik14-group bootstrap 95% CI) against the number of lines each training compound keeps (5, 10, 25;
      50 = complete grid). Source: results/l1000_xline/weights_vs_D{_keep5,_keep10,_keep25,_r}.json
      (`run_l1000_xline.py --keep-lines`, report/11_06_report.md §10.1).
  (b) Drug-blind benchmark: PCC_plate by the views a test compound has (backbone, XPert, UnDetect, UnDetect + XPert).
      Source: results/sdst_hybrid/coverage_final/strata.csv (`analyze_sdst_coverage.py`, report §10.3).
fig_app_platemates
  PCC_plate by the number of training plate-mates (backbone, XPert, UnDetect). Source:
  results/sdst_hybrid/plate_mate_count/bins.csv (`analyze_plate_mate_count.py`, report §10.6).
Run: /nfs-stor/yuqi.lei/Therapeutic/envs/gnn-gpu/bin/python -I scripts/figures/fig_appendix.py [outdir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import GRID, INK, INK2, MUTED, OURS, panel, plt, save  # noqa: E402

XL = Path("/nfs-stor/yuqi.lei/Therapeutic/results/l1000_xline")
SH = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
OUT = sys.argv[1] if len(sys.argv) > 1 else None
DOSES = (("T-0.5", "0.5 µM", OURS), ("T-5", "5 µM", INK2))
KEEP = ((5, "_keep5"), (10, "_keep10"), (25, "_keep25"), (50, "_r"))
MODELS = (("mlp", "drug-free backbone", MUTED, "o", True), ("xpert", "XPert", INK2, "o", True),
          ("ctx_pc", "UnDetect", OURS, "o", True), ("ctx_pc+xpert", "UnDetect + XPert", OURS, "D", False))


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=8.5); ax.set_ylabel(ylabel, fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5)


# ------------------------------------------------------------------ fig_app_transfer
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.8), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.38})
fig.subplots_adjust(left=0.14, right=0.99, bottom=0.25, top=0.84)
rows = []
xs = np.arange(len(KEEP))
for j, (sl, lab, col) in enumerate(DOSES):
    m, lo, hi = [], [], []
    for k, tag in KEEP:
        c = json.loads((XL / f"weights_vs_D{tag}.json").read_text())[sl]["contrasts"]["pool_D|pearson_resid"]
        m.append(c["mean"]); lo.append(c["lo"]); hi.append(c["hi"])
        rows.append({"panel": "a", "slice": sl, "lines_per_compound": k, "delta_resid": c["mean"], "lo": c["lo"],
                     "hi": c["hi"]})
    m, lo, hi = map(np.array, (m, lo, hi)); x = xs + (j - 0.5) * 0.12
    a.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o-", color=col, ms=4, lw=1.0, capsize=2, label=lab)
a.axhline(0, color=INK, lw=0.6)
a.set_xticks(xs, [str(k) if k < 50 else "50\n(complete)" for k, _ in KEEP])
style(a, "lines per training compound", "cross-line gain\n(deviation from line mean)")
a.legend(frameon=False, fontsize=7.5, loc="upper right")
panel(a, "a", -0.32)

S = pd.read_csv(SH / "coverage_final/strata.csv").query("metric == 'plate' and stratum != 'all'").set_index("stratum")
groups = [("structure only", "structure\nonly"), ("pharmacology only", "targets,\nmechanism"),
          ("Cell Painting only", "Cell\nPainting"), ("both", "all\nthree")]
xg = np.arange(len(groups))
for i, (key, lab, col, mk, filled) in enumerate(MODELS):
    v = [S.loc[g, key] for g, _ in groups]
    b.scatter(xg + (i - 1.5) * 0.15, v, s=22, marker=mk, color=col if filled else "white", edgecolor=col, lw=1.1,
              zorder=3, label=lab)
    rows += [{"panel": "b", "stratum": g, "model": key, "pcc_plate": float(x_)} for (g, _), x_ in zip(groups, v)]
b.set_xticks(xg, [l for _, l in groups])
b.set_ylim(0.08, 0.25)
style(b, "views of the test compound", r"PCC$_\mathrm{plate}$")
b.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=0.8,
         handletextpad=0.2, borderaxespad=0.2)
panel(b, "b", -0.25)
save(fig, "fig_app_transfer", OUT, table=pd.DataFrame(rows))

# ------------------------------------------------------------------ fig_app_platemates
B = pd.read_csv(SH / "plate_mate_count/bins.csv")
labs = ["1–2", "3–5", "6–10", "11–20", "21–50", ">50"]
fig, ax = plt.subplots(figsize=(3.4, 2.3))
rows = []
for key, lab, col, mk, filled in MODELS[:3]:
    ax.plot(np.arange(len(B)), B[key], "o-", color=col, ms=4, lw=1.0, label=lab)
    rows += [{"bin": l, "model": key, "pcc_plate": float(v)} for l, v in zip(labs, B[key])]
ax.set_xticks(np.arange(len(B)), labs)
style(ax, "training plate-mates", r"PCC$_\mathrm{plate}$")
ax.legend(frameon=False, fontsize=7.5)
fig.subplots_adjust(left=0.17, bottom=0.2, right=0.97, top=0.95)
save(fig, "fig_app_platemates", OUT, table=pd.DataFrame(rows))
