"""Fig. 2 of the UnDetect paper: unseen-compound generalization. Dossier: paper/figures/fig2_benchmark.md.

2026-11-04 redesign (Jun Wen's review of the PDF): horizontal bars with error bars, one colour per tool across all
panels; UnDetect light red, UnDetect + XPert purple, XPert light blue; the plate-response baseline keeps its orange,
drug-free networks gray. (d) drops the ablation row (UnDetect, chem. + targets + MOA).

(a) PCC_DEG of each model with its drug input (solid) and with it zeroed (hatched): the seven published models
    (Bai et al. 2026) and UnDetect (zeroed = its drug-free backbone); line = the plate-response baseline.
(b) PCC_plate of the compared methods; XPert and TranSiGen also with their drug input zeroed; line = the reference
    (each test sample from the same compound's measured responses in other lines; uses test responses).
(c) DeepCE's own benchmark (published split, per-signature Pearson): values reported by Pham et al. 2021 for the
    models we did not run, our runs otherwise.
2026-11-04 (user): old (b), the plate-response baseline with / without plate-mate, dropped (numbers in the text);
legend shared by (a) and (b); reference line in (b); same-input MLP labelled "MLP".
Error bars: s.d. over the five folds (a-b); s.d. over seeds where a model was trained with several (c: DeepCE
reproduction, 3 seeds; the same-input MLP); none for single deterministic fits or reported values.
Sources: repro/dpb/upstream/.../statistics/{primary_comparisons,full_vs_mlp}.csv (drug-perturbation-benchmark
@ 55edbfe; report §5.6); results/sdst_hybrid/plate_baseline/summary.json (job 350554; §5.1);
results/sdst_hybrid/final_cpu/scores.csv (final numbers, one backbone run, released published predictions scored on
the same rows; report/11_06_report.md §4, §6.3); results/sdst_hybrid/published/centred.json (TranSiGen drug zeroed;
11_06 §4.1); (d) repro/deepce/results/{baselines,xline_kernel/deepce_split,mlp_multimodal}/summary.json,
ppi_seed{0,1,2} (report §4.2, §4.6; reported values: Pham et al. 2021 Table 1 as transcribed in
report/10_09_report.md §5.1).
Run: /nfs-stor/yuqi.lei/Therapeutic/envs/gnn-gpu/bin/python -I scripts/figures/fig2_benchmark.py [outdir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import GRID, INK, INK2, LINE, PLATE, ROOT, panel, plt, save  # noqa: E402

STATS = ROOT / "repro/dpb/upstream/data/analysis/benchmark_metrics/statistics"
RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
DCE = Path("/nfs-stor/yuqi.lei/Therapeutic/repro/deepce/results")

# One colour per tool, shared by every panel (Jun Wen 2026-11-04: UnDetect light red, ensemble purple, XPert light
# blue; the rest our choice, kept away from those three hues).
TOOL = {"UnDetect": "#f0807a", "UnDetect + XPert": "#8e6bbf", "XPert": "#8ec5ea", "TranSiGen": "#499894",
        "DeepCE": "#edc948", "PertDiT": "#b6992d", "PRnet": "#59a14f", "MultiDCP": "#9c755f", "CIGER": "#c9b18f",
        "Drug-free MLP": "#8a8984", "Plate response baseline": PLATE, "Cell-line mean": LINE,
        "Chemistry kernel": "#6b7f99", "Morgan $k$NN, tuned": "#a39e93", "$k$NN": "#d3d0c8",
        "Neural network": "#bdb9ae", "DeepCE without attention": "#f5e3a3", "Mean per line and dose": LINE,
        "MLP": "#5f5e59"}
BAR = dict(height=0.38, zorder=2, linewidth=0.6)
ERR = dict(ecolor=INK2, elinewidth=0.7, capsize=1.6, capthick=0.7)


def tint(hex_colour, f=0.55):
    """Blend a colour towards white (drug-zeroed bars)."""
    rgb = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(rgb + (1 - rgb) * f)


def style_axis(ax, xlabel, xlim):
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=8.5, labelpad=1)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)


def colour_ours(ax, names):
    for t, n in zip(ax.get_yticklabels(), names):
        if n.startswith("UnDetect"):
            t.set_color("#c2413b" if n == "UnDetect" else TOOL[n])


def paired_bars(ax, items, label_fmt="{:.3f}", inside=None):
    """items: (name, with-drug folds, drug-zeroed folds or None); highest mean on top. Returns the plotted rows.
    inside: write the value inside bars longer than this (keeps labels clear of a reference line)."""
    items = sorted(items, key=lambda t: np.nanmean(t[1]))
    out = []
    for k, (name, fw, fz) in enumerate(items):
        col = TOOL[name]
        has_z = fz is not None
        yw = k + (0.2 if has_z else 0)
        ax.barh(yw, fw.mean(), xerr=fw.std(ddof=1), color=col, edgecolor=col, error_kw=ERR, **BAR)
        if inside is not None and fw.mean() > inside:
            ax.text(fw.mean() - fw.std(ddof=1) - 0.003, yw, label_fmt.format(fw.mean()), ha="right", va="center",
                    fontsize=6.5, color=INK)
        else:
            pad = 0.008 if inside is None else 0.003
            ax.text(fw.mean() + fw.std(ddof=1) + pad, yw, label_fmt.format(fw.mean()), va="center", fontsize=6.5,
                    color=INK)
        out.append({"model": name, "input": "with drug", "mean": fw.mean(), "sd": fw.std(ddof=1)})
        if has_z:
            ax.barh(k - 0.2, fz.mean(), xerr=fz.std(ddof=1), color=tint(col), edgecolor=col, hatch="//////",
                    error_kw=ERR, **BAR)
            out.append({"model": name, "input": "drug zeroed", "mean": fz.mean(), "sd": fz.std(ddof=1)})
    names = [n for n, *_ in items]
    ax.set_yticks(np.arange(len(items)), names, fontsize=8)
    ax.set_ylim(-0.6, len(items) - 0.4)
    colour_ours(ax, names)
    return out


pc = pd.read_csv(STATS / "primary_comparisons.csv")
z = pc[(pc.metric_key == "pcc_deg_mean") & (pc.contrast == "Full - Zero")].set_index("model")
fin = pd.read_csv(RES / "final_cpu/scores.csv")
deg = fin.pivot(index="fold", columns="arm", values="deg").sort_index()
plate = fin.pivot(index="fold", columns="arm", values="plate").sort_index()
pub = pd.DataFrame(json.loads((RES / "published/centred.json").read_text())["rows"])
LKF = pd.DataFrame(json.loads((RES / "plate_baseline/summary.json").read_text()))
rows = []

fig = plt.figure(figsize=(6.5, 4.9))
TOP = 3.1 / 4.9  # panels a-b in the upper part; c below
LEGY = 0.372     # legend shared by (a) and (b), centred under both (2026-11-04)
def box(x, y, w, h):
    return [x, 1 - TOP + y * TOP, w, h * TOP]
a = fig.add_axes(box(0.125, 0.14, 0.33, 0.76))
c = fig.add_axes(box(0.66, 0.14, 0.29, 0.76))      # PCC_plate, panel "b" in the paper
d = fig.add_axes([0.335, 0.075, 0.60, 0.235])      # DeepCE, panel "c"

# (a) with vs without the drug input
NAME = {"CIGER": "CIGER", "MultiDCP": "MultiDCP", "PRnet": "PRnet", "PertDiT": "PertDiT", "DeepCE": "DeepCE",
        "TranSiGen": "TranSiGen", "XPert": "XPert"}
items = []
for m in z.index:
    fw, fz = (np.array([z.loc[m, f"fold_{i}_{c_}"] for i in range(5)], float) for c_ in ("full", "comparator"))
    items.append((NAME.get(m, m), fw, fz))
items.append(("UnDetect", deg["ctx_pc"].to_numpy(), deg["mlp"].to_numpy()))
items.append(("UnDetect + XPert", deg["ctx_pc+xpert"].to_numpy(), None))
rows += [dict(r, panel="a", metric="PCC_DEG") for r in paired_bars(a, items)]
names_a = [t.get_text() for t in a.get_yticklabels()]
a.set_yticks(a.get_yticks(), [n.replace(" + ", "\n+ ") for n in names_a], fontsize=8, linespacing=0.9)
colour_ours(a, names_a)
a.axvline(LKF.plate_mean.mean(), color=PLATE, lw=1.1, zorder=3)
a.text(LKF.plate_mean.mean() - 0.008, -0.5, "Plate response\nbaseline", rotation=90, ha="right", va="bottom",
       fontsize=6.5, color=PLATE, linespacing=0.9)
style_axis(a, r"PCC$_\mathrm{DEG}$", (0, 0.76))
fig.legend(handles=[Patch(facecolor="#8a8984", edgecolor="#8a8984", label="with drug"),
                    Patch(facecolor=tint("#8a8984"), edgecolor="#8a8984", hatch="//////",
                          label="drug input zeroed")],
           loc="center", bbox_to_anchor=(0.55, LEGY), ncol=2, frameon=False, fontsize=7.5, handlelength=1.4,
           handletextpad=0.4, columnspacing=1.2, borderaxespad=0.2)
panel(a, "a", -0.33)

# (c) plate-centred score of the compared methods
tz = pub[pub.arm == "transigen_zero"].sort_values("fold").centred_shared.to_numpy()
items = [("CIGER", plate["ciger"].to_numpy(), None), ("MultiDCP", plate["multidcp"].to_numpy(), None),
         ("DeepCE", plate["deepce"].to_numpy(), None), ("Drug-free MLP", plate["mlp_bai"].to_numpy(), None),
         ("TranSiGen", plate["transigen"].to_numpy(), tz), ("XPert", plate["xpert"].to_numpy(),
                                                            plate["xpert_zero"].to_numpy()),
         ("UnDetect", plate["ctx_pc"].to_numpy(), None), ("UnDetect + XPert", plate["ctx_pc+xpert"].to_numpy(), None)]
rows += [dict(r, panel="b", metric="PCC_plate") for r in paired_bars(c, items, inside=0.09)]
# reference: the compound's own measurements in other lines (report/11_06_report.md §11 v2, job 364547)
ref = json.loads((RES / "oracle_final/summary.json").read_text())["means"]["all_shared"]["oracle_uniform"]
c.axvline(ref, color=INK2, lw=1.0, ls="--", zorder=3)
c.text(ref - 0.004, -0.5, f"reference ({ref:.3f}),\nuses test responses", rotation=90, ha="right", va="bottom",
       fontsize=6.5, color=INK2, linespacing=0.9)
rows.append({"panel": "b", "metric": "PCC_plate", "model": "reference", "mean": ref})
style_axis(c, r"PCC$_\mathrm{plate}$", (0, 0.235))
c.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
fig.text(0.47, 1 - TOP + 0.95 * TOP, "b", fontsize=11, fontweight="bold", ha="left", va="center")

# (d) DeepCE's own benchmark
bl = json.loads((DCE / "baselines/summary.json").read_text())
xk = json.loads((DCE / "xline_kernel/deepce_split/summary.json").read_text())["arms"]
mm = json.loads((DCE / "mlp_multimodal/summary.json").read_text())
rep = np.array([np.load(DCE / f"ppi_seed{s}/test_pred_bestdev.npz")["per_sig_pearson"].mean() for s in range(3)])
mlp_seeds = np.atleast_1d(np.array(mm["mlp_multi"]["test"], float))
NO = np.nan
D = [("Mean per line and dose", bl["mean_cell_dose"]["pearson"], NO, False),
     ("$k$NN", 0.3903, NO, True), ("Neural network", 0.3929, NO, True),
     ("DeepCE without attention", 0.4418, NO, True),
     ("Morgan $k$NN, tuned", bl["knn_morgan"]["pearson"], NO, False),
     ("DeepCE", rep.mean(), rep.std(ddof=1), False),
     ("Chemistry kernel", xk["krr_chem_perline"]["test"], NO, False),
     ("MLP", mm["mlp_multi"]["test_seedavg_per_sig"],
      mlp_seeds.std(ddof=1) if mlp_seeds.size > 1 else NO, False),
     ("UnDetect", xk["krr_multi_norm_xline"]["test"], NO, False)]
D.sort(key=lambda t: t[1])
yd = np.arange(len(D))                   # highest on top
for k, (name, v, sd, reported) in zip(yd, D):
    col = TOOL.get(name, TOOL["Drug-free MLP"])
    d.barh(k, v, xerr=None if np.isnan(sd) else sd, color=col, edgecolor=col, error_kw=ERR,
           **dict(BAR, height=0.62))
    d.text(v + (0 if np.isnan(sd) else sd) + 0.006, k, f"{v:.3f}", ha="left", va="center", fontsize=7, color=INK)
    rows.append({"panel": "c", "metric": "Pearson", "model": name, "mean": float(v),
                 "sd": None if np.isnan(sd) else float(sd), "reported": reported})
d.set_yticks(yd, [n for n, *_ in D], fontsize=7.5)
colour_ours(d, [n for n, *_ in D])
d.set_ylim(-0.6, len(D) - 0.4)
style_axis(d, "Pearson, DeepCE test split", (0, 0.68))
fig.text(0.02, 0.32, "c", fontsize=11, fontweight="bold", ha="left", va="center")

save(fig, "fig2_benchmark", sys.argv[1] if len(sys.argv) > 1 else None, table=pd.DataFrame(rows))
