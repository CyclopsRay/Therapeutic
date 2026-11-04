"""Fig. 3: what the compound views know that chemistry does not. Dossier: paper/figures/fig_views.md.

v2 (2026-11-04, user: same style as Fig. 2; compare the Fig. 1 compounds; old (c) by nearest chemical neighbour
dropped as the least informative panel).
(a) View ablation: gain in PCC_plate of UnDetect over the drug-free backbone with one view alone, all views without
    one, and one view's compound labels permuted; and all four views permuted together. Bars = five-fold mean, error
    bars = s.d. over folds (permuted together: s.d. over five permutation seeds of the five-fold mean). Line = all four.
(b) Chemically unlike pairs (Morgan Tanimoto < 0.25): response similarity by each view's similarity bin; dashed =
    chemical analogues (Tanimoto >= 0.6); number of pairs under each bar.
(c) The three compounds of Fig. 1, each scored in the fold where it is held out: PCC_plate of the drug-free MLP, the
    chemistry kernel, XPert and UnDetect (Fig. 2 colours).
(d) Four chemically unlike pairs that respond alike (closest pairs in the fused kernel, one per well-known mechanism):
    structures, and the mean plate-centred signatures on the 30 genes with the largest |z| in either compound, ordered
    by the first: first compound measured, second measured, second as predicted by UnDetect while held out.
Sources: results/sdst_hybrid/views_final/fold_*/summary.json (`scripts/sdst_view_ablation.py`, jobs 359222-359226;
report/11_06_report.md §8), views_permjoint_summary.csv (§10.4); compound_analysis_final/{summary.json,
example_pairs.json} (job 359229); case_study/{scores.csv, signatures.npz} (`scripts/case_study_compounds.py`, job
367833; report §14).
Run: python -I scripts/figures/fig_views.py [outdir]   (numpy, pandas, matplotlib)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import DIV_NEG, DIV_POS, GRID, INK, INK2, LINE, VIEW, panel, plt, save  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from fig1_cards import molecule  # noqa: E402

RESD = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
RES = RESD / "compound_analysis_final"
CASE = RESD / "case_study"
NGENES = 30
CMAP = LinearSegmentedColormap.from_list("div", [DIV_NEG, "#efeeeb", DIV_POS])
VNAME = {"chem": "Chemistry", "tgt": "Targets", "moa": "Mechanism", "morph": "Cell Painting"}
# Fig. 2 colours (scripts/figures/fig2_benchmark.py TOOL)
MODEL = [("mlp_bai", "Drug-free MLP", "#8a8984"), ("pool_chem", "Chemistry kernel", "#6b7f99"),
         ("xpert", "XPert", "#8ec5ea"), ("ctx_pc", "UnDetect", "#f0807a")]
PID = {"panobinostat": "BRD-K02130563", "vorinostat": "BRD-K81418486", "trametinib": "BRD-K12343256",
       "selumetinib": "BRD-K57080016", "daunorubicin": "BRD-A37630846", "mitoxantrone": "BRD-K21680192",
       "alvocidib": "BRD-K87909389", "R-547": "BRD-K52233191"}
ERR = dict(ecolor=INK2, elinewidth=0.7, capsize=1.4, capthick=0.7)
rows = []

S = json.loads((RES / "summary.json").read_text())
EX = json.loads((RES / "example_pairs.json").read_text())
SIG = np.load(CASE / "signatures.npz")
SC = pd.read_csv(CASE / "scores.csv")

fig = plt.figure(figsize=(6.5, 4.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 0.95], wspace=0.42, left=0.115, right=0.99, top=0.93,
                      bottom=0.61)
a, b, c = (fig.add_subplot(gs[0, k]) for k in range(3))
DH = 0.50 * 4.5                              # height of panel d in inches (data units = inches)
d = fig.add_axes([0.0, 0.0, 1.0, 0.50]); d.set_xlim(0, 6.5); d.set_ylim(0, DH); d.axis("off")

# (a) view ablation: gain over the backbone, five folds
F = []
for f in range(5):
    s = json.loads((RESD / "views_final" / f"fold_{f}" / "summary.json").read_text())["sets"]
    F.append({k: v["test"]["undetect"]["plate"] for k, v in s.items() if "test" in v})
F = pd.DataFrame(F)
PJ = pd.read_csv(RESD / "views_permjoint_summary.csv")
PJ = PJ[(PJ.metric == "plate") & (PJ.arm == "undetect")]
base = PJ[PJ.set == "multi_norm"].groupby("fold").backbone.first().sort_index().to_numpy()   # backbone per fold
G = F.sub(base, axis=0)
full = G["multi_norm"]
kinds = [("only", "alone"), ("minus", "without"), ("perm", "permuted")]
yt, yl = [], []
for r_, (k, nm) in enumerate(kinds):
    y0 = 3 - r_
    for j, v in enumerate(VNAME):
        g = G[f"{k}_{v}"]; y = y0 + (1.5 - j) * 0.19
        a.barh(y, g.mean(), xerr=g.std(ddof=1), height=0.17, color=VIEW[v], error_kw=ERR, zorder=2)
        rows.append({"panel": "a", "kind": k, "view": v, "gain_mean": g.mean(), "gain_sd": g.std(ddof=1)})
    yt.append(y0); yl.append(nm)
pj = PJ[PJ.set == "permjoint"].groupby("seed").gain.mean()
a.barh(0, pj.mean(), xerr=pj.std(ddof=1), height=0.30, color=INK2, error_kw=ERR, zorder=2)
rows.append({"panel": "a", "kind": "permjoint", "view": "all", "gain_mean": pj.mean(), "gain_sd": pj.std(ddof=1)})
yt.append(0); yl.append("all\npermuted")
a.axvline(full.mean(), color=DIV_POS, lw=1.0, zorder=3)
a.text(full.mean() - 0.002, -0.40, "all four", color=DIV_POS, fontsize=6, ha="right", va="bottom")
a.axvline(0, color=INK2, lw=0.6, zorder=3)
a.set_yticks(yt, yl, fontsize=7.5, linespacing=0.9)
a.set_ylim(-0.45, 3.7); a.set_xlim(-0.006, 0.072); a.set_xticks([0, 0.03, 0.06])
a.set_xlabel(r"gain in PCC$_\mathrm{plate}$", fontsize=8, labelpad=1)
a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0, pad=1)
for v, nm in VNAME.items():
    a.barh([0], [0], color=VIEW[v], label=nm)
a.legend(loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=2, frameon=False, fontsize=5.6, handlelength=0.9,
         handletextpad=0.3, columnspacing=0.6, borderaxespad=0.1)
panel(a, "a", -0.40)

# (b) response similarity by view similarity, chemically unlike pairs
groups = [("Targets", "tgt", ["0", "<.5", "≥.5"]), ("Mechanism", "moa", ["0", "<.5", "≥.5"]),
          ("Cell Painting", "morph", ["<.3", ".3", ".5", "≥.7"])]
x0, ticks, labels = 0.0, [], []
for name, key, labs in groups:
    bins = S["view_bins_chem_low"][key]; xs = x0 + np.arange(len(bins))
    b.bar(xs, [r["resp_mean"] for r in bins], width=0.75, color=VIEW[key], zorder=2)
    for xx, r in zip(xs, bins):
        n = r["n_pairs"]; s_ = f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e3 else f"{n}"
        b.text(xx, max(r["resp_mean"], 0) + 0.006, s_, ha="center", va="bottom", fontsize=5, color=INK2, rotation=90)
    ticks += list(xs); labels += labs
    b.text(xs.mean(), -0.042, name, ha="center", va="top", fontsize=6.5, color=INK)
    rows += [{"panel": "b", "view": key, "bin": r["bin"], "resp_mean": r["resp_mean"], "n_pairs": r["n_pairs"]}
             for r in bins]
    x0 = xs[-1] + 1.5
ana = S["resp_mean"]["chem_high_analogues"]
b.axhline(ana, color=VIEW["chem"], ls="--", lw=1.0, zorder=3)
b.text(-0.4, ana + 0.006, "chemical analogues", color=VIEW["chem"], fontsize=6.5, ha="left", va="bottom")
b.axhline(0, color=INK2, lw=0.6, zorder=3)
b.set_xticks(ticks, labels, fontsize=5.5)
b.tick_params(axis="x", length=0, pad=1)
b.set_ylabel("response similarity (r)", fontsize=8, labelpad=1)
b.set_ylim(-0.02, 0.34)
b.grid(axis="y", color=GRID, lw=0.6); b.set_axisbelow(True)
panel(b, "b", -0.22)

# (c) the Fig. 1 compounds, each scored where it is held out
comp = [("daunorubicin", "D"), ("vorinostat", "V"), ("mitoxantrone", "M")]
wbar = 0.19
for i, (nm, lt) in enumerate(comp):
    sc = SC[(SC.name == nm) & (SC.pert_id == PID[nm])].set_index("arm").pcc_plate
    for j, (arm, lab, col) in enumerate(MODEL):
        c.bar(i + (j - 1.5) * wbar, sc[arm], width=wbar * 0.92, color=col, zorder=2)
        rows.append({"panel": "c", "compound": nm, "arm": arm, "pcc_plate": float(sc[arm])})
c.axhline(0, color=INK2, lw=0.6, zorder=3)
c.set_xticks(range(3), [lt for nm, lt in comp], fontsize=8)
c.get_xticklabels()[2].set_color("#2a78d6")
c.tick_params(axis="x", length=0, pad=2)
c.set_ylabel(r"PCC$_\mathrm{plate}$", fontsize=8, labelpad=1)
c.set_ylim(-0.25, 1.12); c.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
c.grid(axis="y", color=GRID, lw=0.6); c.set_axisbelow(True)
for arm, lab, col in MODEL:
    c.bar([0], [0], color=col, label=lab)
c.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), ncol=2, frameon=False, fontsize=5.6, handlelength=0.9,
         handletextpad=0.3, columnspacing=0.6, borderaxespad=0)
panel(c, "c", -0.40)

# (d) four pairs: structures; measured signatures of both; the second as predicted while held out
d.text(0.06, DH - 0.12, "d", fontsize=11, fontweight="bold", ha="left", va="center")
CWD = 6.5 / len(EX); LABW = 0.62
for k, e in enumerate(EX):
    xl = k * CWD + 0.10; wcol = CWD - 0.20
    d.text(xl + wcol / 2, DH - 0.12, e["class"], fontsize=7.5, ha="center", va="center", color=INK)
    A_, B_ = e["pair"]
    for m in range(2):
        st = e["drawing"][m]
        st = {"atoms": st["atoms"], "bonds": [{"i": q["a"], "j": q["b"], "order": q["order"]} for q in st["bonds"]]}
        molecule(d, st, xl + m * wcol / 2 + 0.03, 1.08, wcol / 2 - 0.06, 0.88, color=INK2, size=3.6, lw=0.45)
        d.text(xl + m * wcol / 2 + wcol / 4, 0.96, e["pair"][m], fontsize=6.5, ha="center", va="center", color=INK)
    tA, tB = SIG[f"{A_}__{PID[A_]}__truth"], SIG[f"{B_}__{PID[B_]}__truth"]
    pB = SIG[f"{B_}__{PID[B_]}__ctx_pc"]
    zA, zB = tA / tA.std(), tB / tB.std()
    g = np.argsort(-np.maximum(np.abs(zA), np.abs(zB)))[:NGENES]; g = g[np.argsort(-zA[g])]
    xs = xl + LABW
    for m, (vec, lab) in enumerate(((tA, A_), (tB, B_), (pB, f"{B_}, UnDetect"))):
        v = vec[g] / np.abs(vec[g]).max(); yy = 0.66 - m * 0.16
        d.imshow(v[None, :], cmap=CMAP, vmin=-1, vmax=1, aspect="auto", interpolation="nearest",
                 extent=(xs, xl + wcol, yy, yy + 0.12), zorder=2)
        d.add_patch(plt.Rectangle((xs, yy), xl + wcol - xs, 0.12, fc="none",
                                  ec="#e0605a" if m == 2 else LINE, lw=0.8 if m == 2 else 0.5, zorder=3))
        d.text(xs - 0.04, yy + 0.06, lab if m < 2 else "predicted", fontsize=5.4, ha="right", va="center",
               color="#c2413b" if m == 2 else INK, style="italic" if m == 2 else "normal")
        rows += [{"panel": "d", "pair": "–".join(e["pair"]), "strip": lab, "gene_rank": i, "gene_index": int(gi),
                  "scaled": float(vv)} for i, (gi, vv) in enumerate(zip(g, v))]
    rp = np.corrcoef(tB, pB)[0, 1]
    d.text(xl + wcol / 2, 0.245, f"Tanimoto {e['tanimoto']:.2f}   measured r {e['resp_r']:.2f}", fontsize=6.3,
           ha="center", va="center", color=INK)
    d.text(xl + wcol / 2, 0.135, f"predicted vs measured r {rp:.2f}", fontsize=6.3, ha="center", va="center",
           color="#c2413b")
    rows.append({"panel": "d", "pair": "–".join(e["pair"]), "strip": "r(pred, measured)", "scaled": float(rp)})
kx, kw, ky, kh = 2.85, 0.8, 0.015, 0.05
d.imshow(np.linspace(-1, 1, 64)[None, :], cmap=CMAP, vmin=-1, vmax=1, aspect="auto",
         extent=(kx, kx + kw, ky, ky + kh), zorder=2)
d.add_patch(plt.Rectangle((kx, ky), kw, kh, fc="none", ec=LINE, lw=0.5, zorder=3))
d.text(kx - 0.04, ky + kh / 2, "down", fontsize=5.8, ha="right", va="center", color=INK)
d.text(kx + kw + 0.04, ky + kh / 2, "up", fontsize=5.8, ha="left", va="center", color=INK)
d.set_xlim(0, 6.5); d.set_ylim(0, DH)

save(fig, "fig_views", sys.argv[1] if len(sys.argv) > 1 else None, table=pd.DataFrame(rows))
