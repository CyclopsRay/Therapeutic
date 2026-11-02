"""Fig. 2 of the UnDetect paper: what the drug-blind benchmark measures. Dossier: paper/figures/fig2_benchmark.md.

(a) PCC_DEG of each model with its drug input and with it zeroed: the seven published models (Bai et al. 2026, fold
    means) and UnDetect (compound lane on / off; off = its drug-free backbone); line = the plate-response
    baseline (drug-free MLP line removed 2026-11-02, user: UnDetect's drug-zeroed point marks the drug-free level). (b) Plate-response baseline on test samples with and without a training plate-mate,
    next to the cell-line mean. (c) Plate-centred score PCC_plate: baseline, drug-free MLP, plate co-membership
    kernel, chemistry kernel, UnDetect, and XPert / TranSiGen with and without their drug input; dots = folds. (d) DeepCE's own benchmark (published split, per-signature Pearson):
    values reported by Pham et al. 2021 (open dots) next to our runs (filled).
Sources: repro/dpb/upstream/.../statistics/{primary_comparisons,full_vs_mlp}.csv (drug-perturbation-benchmark
@ 55edbfe; report §5.6); results/sdst_hybrid/plate_baseline/summary.json (job 350554; §5.1);
results/sdst_hybrid/final_cpu/scores.csv (final numbers, one backbone run; report/11_06_report.md §6.3);
results/sdst_hybrid/published/centred.json (released XPert / TranSiGen Full and Zero; 11_06 §4.1);
results/sdst_hybrid/v3/centred.json (plate-response baseline on PCC_plate only, backbone-free; 10_30 §5.4);
(d) repro/deepce/results/{baselines,xline_kernel/deepce_split,mlp_multimodal}/summary.json, ppi_seed{0,1,2}
(report §4.2, §4.6; reported values: Pham et al. 2021 Table 1 as transcribed in report/10_09_report.md §5.1).
Run: /nfs-stor/yuqi.lei/Therapeutic/envs/gnn-gpu/bin/python -I scripts/figures/fig2_benchmark.py [outdir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import CTX, GRID, INK, INK2, MUTED, OURS, PLATE, ROOT, VIEW, panel, plt, save  # noqa: E402

STATS = ROOT / "repro/dpb/upstream/data/analysis/benchmark_metrics/statistics"
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
DCE = Path("/nfs-stor/yuqi.lei/Therapeutic/repro/deepce/results")
BASE = "Plate response baseline"

pc = pd.read_csv(STATS / "primary_comparisons.csv")
z = pc[(pc.metric_key == "pcc_deg_mean") & (pc.contrast == "Full - Zero")].set_index("model")
fm = pd.read_csv(STATS / "full_vs_mlp.csv")
mlp = float(fm[fm.metric_key == "pcc_deg_mean"].comparator_mean.iloc[0])
lk = pd.DataFrame(json.loads((RES / "plate_baseline/summary.json").read_text())).mean()
fin = pd.read_csv(RES / "final_cpu/scores.csv")
fin_deg = fin.groupby("arm").deg.mean()
cen = pd.concat([pd.DataFrame(json.loads((RES / "v3/centred.json").read_text())["rows"]).query("arm == 'plate_lookup'"),
                 fin.rename(columns={"plate": "centred_shared"})[["fold", "arm", "centred_shared"]],
                 pd.DataFrame(json.loads((RES / "published/centred.json").read_text())["rows"])
                 .query("arm in ['transigen_zero']")])

rows = [{"panel": "a", "model": m, "with_drug": z.loc[m, "full_mean"], "drug_zeroed": z.loc[m, "comparator_mean"]}
        for m in z.full_mean.sort_values().index]
rows.append({"panel": "a", "model": "UnDetect", "with_drug": fin_deg["ctx_pc"], "drug_zeroed": fin_deg["mlp"]})
rows.append({"panel": "a", "model": "UnDetect\n+ XPert", "with_drug": fin_deg["ctx_pc+xpert"], "drug_zeroed": np.nan})
A = pd.DataFrame(rows)

fig = plt.figure(figsize=(6.5, 4.8))
TOP = 3.0 / 4.8  # panels a-c keep their 6.5 x 3.0 layout in the upper part; d below
def box(x, y, w, h):
    return [x, 1 - TOP + y * TOP, w, h * TOP]
a = fig.add_axes(box(0.125, 0.15, 0.33, 0.73))
b = fig.add_axes(box(0.715, 0.62, 0.27, 0.32))
c = fig.add_axes(box(0.715, 0.13, 0.27, 0.36))
d = fig.add_axes([0.335, 0.085, 0.60, 0.25])

# (a) with vs without the drug input
y = np.arange(len(A))
for k, r in A.iterrows():
    col = OURS if r.model.startswith("UnDetect") else INK2
    if np.isfinite(r.drug_zeroed):
        a.plot([r.drug_zeroed, r.with_drug], [k, k], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        a.scatter(r.drug_zeroed, k, s=32, facecolor="white", edgecolor=col, lw=1.1, zorder=4)
    a.scatter(r.with_drug, k, s=32, color=col, zorder=3)
a.axvline(lk.plate_mean, color=PLATE, lw=1.1, zorder=2)
a.set_yticks(y, A.model); a.get_yticklabels()[-1].set_fontsize(8); a.get_yticklabels()[-1].set_linespacing(0.9)
for t in a.get_yticklabels()[-2:]:
    t.set_color(OURS)
a.set_xlim(0.2, 0.7); a.set_ylim(-0.6, len(A) - 0.4)
a.set_xlabel(r"PCC$_\mathrm{DEG}$")
a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
a.scatter([], [], s=32, color=INK2, label="with drug")
a.scatter([], [], s=32, facecolor="white", edgecolor=INK2, lw=1.1, label="drug zeroed")
a.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=7.5, handletextpad=0.2,
         columnspacing=0.9, borderaxespad=0.1)
a.text(lk.plate_mean - 0.006, -0.45, BASE, rotation=90, ha="right", va="bottom", fontsize=7.5, color=PLATE)
panel(a, "a", -0.33)

# (b) the baseline works only through plate-mates
share = lk.share_test_with_train_control
names = [f"plate-mate ({share:.0%})", f"no plate-mate ({1 - share:.0%})", "cell-line mean"]
vals = [lk.plate_mean_shared, lk.plate_mean_unshared, lk.cell_mean]
yb = np.arange(3)[::-1]
b.barh(yb, vals, height=0.62, color=[PLATE, PLATE, MUTED], zorder=2)
for k, v in zip(yb, vals):
    b.text(v + 0.012, k, f"{v:.2f}", ha="left", va="center", fontsize=8, color=INK)
b.set_yticks(yb, names, fontsize=8)
b.set_xlim(0, 0.75); b.set_xlabel(r"PCC$_\mathrm{DEG}$", fontsize=8.5, labelpad=1)
b.grid(axis="x", color=GRID, lw=0.6); b.set_axisbelow(True)
b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)
fig.text(0.475, 1 - TOP + 0.94 * TOP, "b", fontsize=11, fontweight="bold", ha="left", va="center")
rows += [{"panel": "b", "group": g, "pcc_deg": v} for g, v in zip(names, vals)]

# (c) plate-centred score: what distinguishes a compound from its plate-mates
# 2026-11-02 user: plate-response baseline (0 by construction) dropped, rows sorted, highest on top,
# axis from 0.05; dots instead of bars because the axis no longer starts at zero; ensemble added.
arms = [("mlp", "Drug-free MLP", CTX), ("pool_plate", "Plate kernel", MUTED),
        ("transigen_zero", "TranSiGen, drug zeroed", "white"), ("transigen", "TranSiGen", INK2),
        ("xpert_zero", "XPert, drug zeroed", "white"), ("xpert", "XPert", INK2),
        ("pool_chem", "Chemistry kernel", VIEW["chem"]), ("ctx_pc", "UnDetect", OURS),
        ("ctx_pc+xpert", "UnDetect + XPert", OURS)]
arms.sort(key=lambda t: cen[cen.arm == t[0]].centred_shared.mean())
yc = np.arange(len(arms))                # highest on top (user, 2026-11-02 correction)
for k, (arm, name, col) in zip(yc, arms):
    f = cen[cen.arm == arm].centred_shared.to_numpy()
    c.scatter(f, np.full(len(f), k), s=5, color=MUTED, zorder=2, lw=0)
    c.scatter(f.mean(), k, s=30, zorder=3, color=col, edgecolor=INK2 if col == "white" else col, lw=1.1)
    rows += [{"panel": "c", "arm": arm, "fold": i, "pcc_plate": float(v)} for i, v in enumerate(f)]
c.set_yticks(yc, [n for _, n, _ in arms], fontsize=8)
for t, (arm, _, _) in zip(c.get_yticklabels(), arms):
    if arm.startswith("ctx_pc"):
        t.set_color(OURS)
c.set_xlabel(r"PCC$_\mathrm{plate}$", fontsize=8.5, labelpad=1)
c.set_xlim(0.05, 0.20); c.set_ylim(-0.6, len(arms) - 0.4); c.set_xticks([0.05, 0.10, 0.15])
c.grid(axis="x", color=GRID, lw=0.6); c.set_axisbelow(True)
c.spines["left"].set_visible(False); c.tick_params(axis="y", length=0)
fig.text(0.475, 1 - TOP + 0.51 * TOP, "c", fontsize=11, fontweight="bold", ha="left", va="center")

# (d) DeepCE's own benchmark
bl = json.loads((DCE / "baselines/summary.json").read_text())
xk = json.loads((DCE / "xline_kernel/deepce_split/summary.json").read_text())["arms"]
mm = json.loads((DCE / "mlp_multimodal/summary.json").read_text())
rep = np.mean([np.load(DCE / f"ppi_seed{s}/test_pred_bestdev.npz")["per_sig_pearson"] for s in range(3)], 0).mean()
# 2026-11-02 user: no reported / reproduced labels; models we ran show our value (DeepCE = our reproduction), the
# others the value of Pham et al. (flag kept in the source table only); highest on top.
D = [("Mean per line and dose", bl["mean_cell_dose"]["pearson"], MUTED, False),
     ("$k$NN", 0.3903, INK2, True), ("Neural network", 0.3929, INK2, True),
     ("DeepCE without attention", 0.4418, INK2, True),
     ("Morgan $k$NN, tuned", bl["knn_morgan"]["pearson"], MUTED, False), ("DeepCE", rep, INK2, False),
     ("Chemistry kernel", xk["krr_chem_perline"]["test"], VIEW["chem"], False),
     ("MLP, same compound views", mm["mlp_multi"]["test_seedavg_per_sig"], CTX, False),
     ("UnDetect, chem. + targets + MOA", xk["krr_chem_kn_norm_xline"]["test"], OURS, False),
     ("UnDetect, all views", xk["krr_multi_norm_xline"]["test"], OURS, False)]
D.sort(key=lambda t: t[1])
yd = np.arange(len(D))                   # highest on top
for k, (name, v, col, reported) in zip(yd, D):
    d.scatter(v, k, s=32, zorder=3, color=col, edgecolor=col, lw=1.1)
    d.text(v + 0.006, k, f"{v:.3f}", ha="left", va="center", fontsize=7.5, color=INK)
    rows.append({"panel": "d", "model": name, "pearson": float(v), "reported": reported})
d.set_yticks(yd, [n for n, *_ in D], fontsize=7.5)
for t, (name, *_) in zip(d.get_yticklabels(), D):
    if name.startswith("UnDetect"):
        t.set_color(OURS)
d.set_xlim(0.3, 0.64); d.set_ylim(-0.6, len(D) - 0.4)
d.set_xlabel("Pearson, DeepCE test split", fontsize=8.5, labelpad=1)
d.grid(axis="x", color=GRID, lw=0.6); d.set_axisbelow(True)
d.spines["left"].set_visible(False); d.tick_params(axis="y", length=0)
fig.text(0.02, 0.345, "d", fontsize=11, fontweight="bold", ha="left", va="center")

save(fig, "fig2_benchmark", sys.argv[1] if len(sys.argv) > 1 else None, table=pd.DataFrame(rows))
