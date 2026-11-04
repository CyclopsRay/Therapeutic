"""Fig. 3: what the compound views know that chemistry does not. Dossier: paper/figures/fig_views.md.

v2 (2026-11-04, user: same style as Fig. 2; compare the Fig. 1 compounds; old (c) by nearest chemical neighbour
dropped as the least informative panel).
(a) View ablation: gain in PCC_plate of UnDetect over the drug-free backbone with one view alone, all views without
    one, and one view's compound labels permuted; and all four views permuted together. Bars = five-fold mean, error
    bars = s.d. over folds (permuted together: s.d. over five permutation seeds of the five-fold mean). Line = all four.
(b) Chemically unlike pairs (Morgan Tanimoto < 0.25): response similarity by each view's similarity bin; dashed =
    chemical analogues (Tanimoto >= 0.6); number of pairs under each bar.
(c) The three named compounds with the largest gain of UnDetect over both XPert and the chemistry kernel (ranks 1-3 of
    337 with >= 30 plate-mate rows; report §14.1, job 367867), each scored in the fold where it is held out: PCC_plate
    of the drug-free MLP, the chemistry kernel, XPert and UnDetect (Fig. 2 colours). All three are in (d); mitoxantrone
    is the Fig. 1 query (blue). v3 (2026-11-04): replaces the Fig. 1 compounds D, V, M.
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
from fig1_cards import ICON, molecule  # noqa: E402

RESD = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
RES = RESD / "compound_analysis_final"
CASE = RESD / "case_study"
NGENES = 30
CMAP = LinearSegmentedColormap.from_list("div", [DIV_NEG, "#efeeeb", DIV_POS])
VNAME = {"chem": "Structure", "tgt": "Protein targets", "moa": "Mechanism", "morph": "Cell Painting"}   # as Fig. 1
UND, CHEMK = "#f0807a", "#6b7f99"            # UnDetect and the chemistry kernel, as in Fig. 2
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

FW_, FH_ = 6.5, 4.5
fig = plt.figure(figsize=(FW_, FH_))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 0.95], wspace=0.42, left=0.115, right=0.99, top=0.93,
                      bottom=0.61)
a, b, c = (fig.add_subplot(gs[0, k]) for k in range(3))
DH = 0.50 * 4.5                              # height of panel d in inches (data units = inches)
d = fig.add_axes([0.0, 0.0, 1.0, 0.50]); d.set_xlim(0, 6.5); d.set_ylim(0, DH); d.axis("off")
ov = fig.add_axes([0, 0, 1, 1], zorder=10); ov.set_xlim(0, FW_); ov.set_ylim(0, FH_); ov.axis("off")   # icons, inches


def inch(ax, x, y):
    """Data point of ax -> figure inches (overlay coordinates)."""
    return ax.transData.transform((x, y)) / fig.dpi

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
a.axvline(full.mean(), color=INK2, lw=0.9, ls="--", zorder=3)
a.text(full.mean() - 0.002, -0.40, "all four views", color=INK2, fontsize=6, ha="right", va="bottom")
a.axvline(0, color=INK2, lw=0.6, zorder=3)
a.set_yticks(yt, yl, fontsize=7.5, linespacing=0.9)
a.set_ylim(-0.45, 3.7); a.set_xlim(-0.006, 0.072); a.set_xticks([0, 0.03, 0.06])
a.set_xlabel(r"gain in PCC$_\mathrm{plate}$", fontsize=8, labelpad=1)
a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0, pad=1)
pa = a.get_position()                         # view key: icon + name in the view colour, as in Fig. 1
for j, (v, nm) in enumerate(VNAME.items()):
    kx_, ky_ = pa.x0 * FW_ - 0.02 + (j % 2) * 0.86, pa.y1 * FH_ + 0.19 - (j // 2) * 0.14
    ICON[v](ov, kx_, ky_, 0.085)
    ov.text(kx_ + 0.07, ky_, nm, fontsize=5.8, color=INK, ha="left", va="center")
panel(a, "a", -0.40)

# (b) response similarity by view similarity, chemically unlike pairs
CIF = RES / "view_bins_ci.json"               # compound-bootstrap 95% intervals (scripts/view_bins_ci.py)
CI = {(r["view"], r["bin"]): r["ci95"] for r in json.loads(CIF.read_text())["bins"]} if CIF.exists() else {}
groups = [("Protein targets", "tgt", ["0", "<.5", "≥.5"]), ("Mechanism", "moa", ["0", "<.5", "≥.5"]),
          ("Cell Painting", "morph", ["<.3", ".3", ".5", "≥.7"])]
x0, ticks, labels, GROUP_ICON = 0.0, [], [], []
for name, key, labs in groups:
    bins = S["view_bins_chem_low"][key]; xs = x0 + np.arange(len(bins))
    mu = np.array([r["resp_mean"] for r in bins])
    ci = np.array([CI.get((key, r["bin"]), [m_, m_]) for r, m_ in zip(bins, mu)])
    b.bar(xs, mu, width=0.75, color=VIEW[key], zorder=2)
    if CI:
        b.errorbar(xs, mu, yerr=[mu - ci[:, 0], ci[:, 1] - mu], fmt="none", **ERR, zorder=3)
    for xx, r, hi in zip(xs, bins, ci[:, 1]):
        n = r["n_pairs"]; s_ = f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e3 else f"{n}"
        b.text(xx, max(hi, r["resp_mean"], 0) + 0.006, s_, ha="center", va="bottom", fontsize=5, color=INK2,
               rotation=90)
    ticks += list(xs); labels += labs
    b.text(xs.mean() + 0.22, -0.05, name, ha="center", va="top", fontsize=5.8, color=INK)
    GROUP_ICON.append((key, xs.mean() + 0.22, name))
    rows += [{"panel": "b", "view": key, "bin": r["bin"], "resp_mean": r["resp_mean"], "n_pairs": r["n_pairs"],
              "ci_lo": float(c_[0]), "ci_hi": float(c_[1])} for r, c_ in zip(bins, ci)]
    x0 = xs[-1] + 1.9
ana = S["resp_mean"]["chem_high_analogues"]
b.axhline(ana, color=VIEW["chem"], ls="--", lw=1.0, zorder=3)
if ("analogues", "chem >= 0.6") in CI:
    b.axhspan(*CI[("analogues", "chem >= 0.6")], color=VIEW["chem"], alpha=0.10, lw=0, zorder=1)
b.text(-0.4, ana + 0.006, "chemical analogues", color=VIEW["chem"], fontsize=6.5, ha="left", va="bottom")
b.axhline(0, color=INK2, lw=0.6, zorder=3)
b.set_xticks(ticks, labels, fontsize=5.5)
b.tick_params(axis="x", length=0, pad=1)
b.set_ylabel("response similarity (r)", fontsize=8, labelpad=1)
b.set_ylim(-0.02, 0.44); b.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
b.grid(axis="y", color=GRID, lw=0.6); b.set_axisbelow(True)
panel(b, "b", -0.22)

# (c) the largest gains over both XPert and the chemistry kernel (report §14.1), each scored where it is held out
comp = [("R-547", "R-547"), ("alvocidib", "alvocidib"), ("mitoxantrone", "mitoxantrone")]
wbar = 0.19
for i, (nm, lt) in enumerate(comp):
    sc = SC[(SC.name == nm) & (SC.pert_id == PID[nm])].set_index("arm").pcc_plate
    for j, (arm, lab, col) in enumerate(MODEL):
        c.bar(i + (j - 1.5) * wbar, sc[arm], width=wbar * 0.92, color=col, zorder=2)
        rows.append({"panel": "c", "compound": nm, "arm": arm, "pcc_plate": float(sc[arm])})
c.axhline(0, color=INK2, lw=0.6, zorder=3)
c.set_xticks(range(3), [lt for nm, lt in comp], fontsize=6.3, rotation=25, ha="right", rotation_mode="anchor")
c.get_xticklabels()[2].set_color("#2a78d6")
c.tick_params(axis="x", length=0, pad=2)
c.set_ylabel(r"PCC$_\mathrm{plate}$", fontsize=8, labelpad=1)
c.set_ylim(-0.32, 1.02); c.set_yticks([-0.2, 0, 0.2, 0.4, 0.6])
c.grid(axis="y", color=GRID, lw=0.6); c.set_axisbelow(True)
for arm, lab, col in MODEL:
    c.bar([0], [0], color=col, label=lab)
c.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), ncol=2, frameon=False, fontsize=5.6, handlelength=0.9,
         handletextpad=0.3, columnspacing=0.6, borderaxespad=0)
panel(c, "c", -0.40)

# (d) four pairs: structures (one scale per pair, widths by extent, never touching); measured signatures of both; the
# second as predicted while held out, by UnDetect and by the chemistry kernel
d.text(0.06, DH - 0.12, "d", fontsize=11, fontweight="bold", ha="left", va="center")
CWD = 6.5 / len(EX); LABW = 0.72; MOLY, MOLH, GAP = 1.58, 0.74, 0.14
for k, e in enumerate(EX):
    xl = k * CWD + 0.10; wcol = CWD - 0.20
    d.text(xl + wcol / 2, DH - 0.12, e["class"], fontsize=7.5, ha="center", va="center", color=INK)
    A_, B_ = e["pair"]
    sts = [{"atoms": st["atoms"], "bonds": [{"i": q["a"], "j": q["b"], "order": q["order"]} for q in st["bonds"]]}
           for st in e["drawing"]]
    sp = [np.ptp(np.array([[a_["x"], a_["y"]] for a_ in st["atoms"]]), 0) for st in sts]
    sc_ = min((wcol - GAP - 0.04) / (sp[0][0] + sp[1][0]), MOLH / max(sp[0][1], sp[1][1]))
    x_ = xl + (wcol - (sp[0][0] + sp[1][0]) * sc_ - GAP) / 2
    for m in range(2):
        wbox = sp[m][0] * sc_ / 0.9; hbox = MOLH / 0.9
        molecule(d, sts[m], x_ + sp[m][0] * sc_ / 2 - wbox / 2, MOLY - hbox / 2, wbox, hbox, color=INK2, size=3.6,
                 lw=0.45)
        d.text(x_ + sp[m][0] * sc_ / 2, 1.12, e["pair"][m], fontsize=6.5, ha="center", va="center", color=INK)
        x_ += sp[m][0] * sc_ + GAP
    tA, tB = SIG[f"{A_}__{PID[A_]}__truth"], SIG[f"{B_}__{PID[B_]}__truth"]
    pB, cB = SIG[f"{B_}__{PID[B_]}__ctx_pc"], SIG[f"{B_}__{PID[B_]}__pool_chem"]
    zA, zB = tA / tA.std(), tB / tB.std()
    g = np.argsort(-np.maximum(np.abs(zA), np.abs(zB)))[:NGENES]; g = g[np.argsort(-zA[g])]
    xs = xl + LABW
    for m, (vec, lab, ec, tc) in enumerate(((tA, A_, LINE, INK), (tB, B_, LINE, INK),
                                            (pB, "UnDetect", UND, "#c2413b"), (cB, "chemistry kernel", CHEMK, "#4f6178"))):
        v = vec[g] / np.abs(vec[g]).max(); yy = 0.88 - m * 0.145
        d.imshow(v[None, :], cmap=CMAP, vmin=-1, vmax=1, aspect="auto", interpolation="nearest",
                 extent=(xs, xl + wcol, yy, yy + 0.11), zorder=2)
        d.add_patch(plt.Rectangle((xs, yy), xl + wcol - xs, 0.11, fc="none", ec=ec, lw=0.8 if m >= 2 else 0.5,
                                  zorder=3))
        d.text(xs - 0.04, yy + 0.055, lab, fontsize=5.4, ha="right", va="center", color=tc,
               style="italic" if m >= 2 else "normal")
        rows += [{"panel": "d", "pair": "–".join(e["pair"]), "strip": lab, "gene_rank": i, "gene_index": int(gi),
                  "scaled": float(vv)} for i, (gi, vv) in enumerate(zip(g, v))]
    rp, rc = np.corrcoef(tB, pB)[0, 1], np.corrcoef(tB, cB)[0, 1]
    d.text(xl + wcol / 2, 0.30, f"Tanimoto {e['tanimoto']:.2f}   measured r {e['resp_r']:.2f}", fontsize=6.2,
           ha="center", va="center", color=INK)
    d.text(xl + wcol / 2 - 0.05, 0.185, f"UnDetect r {rp:.2f}", fontsize=6.2, ha="right", va="center", color="#c2413b")
    d.text(xl + wcol / 2 + 0.05, 0.185, f"chemistry r {rc:.2f}", fontsize=6.2, ha="left", va="center", color="#4f6178")
    rows.append({"panel": "d", "pair": "–".join(e["pair"]), "strip": "r(UnDetect, measured)", "scaled": float(rp)})
    rows.append({"panel": "d", "pair": "–".join(e["pair"]), "strip": "r(chemistry, measured)", "scaled": float(rc)})
kx, kw, ky, kh = 2.85, 0.8, 0.03, 0.05
d.imshow(np.linspace(-1, 1, 64)[None, :], cmap=CMAP, vmin=-1, vmax=1, aspect="auto",
         extent=(kx, kx + kw, ky, ky + kh), zorder=2)
d.add_patch(plt.Rectangle((kx, ky), kw, kh, fc="none", ec=LINE, lw=0.5, zorder=3))
d.text(kx - 0.04, ky + kh / 2, "down", fontsize=5.8, ha="right", va="center", color=INK)
d.text(kx + kw + 0.04, ky + kh / 2, "up", fontsize=5.8, ha="left", va="center", color=INK)
d.set_xlim(0, 6.5); d.set_ylim(0, DH)

r_ = fig.canvas.get_renderer()
for key, xm, name in GROUP_ICON:
    xi, yi = inch(b, xm, -0.05)
    t_ = b.text(xm, -0.05, name, fontsize=5.8, alpha=0)          # measure the label to place the icon before it
    wt = t_.get_window_extent(r_).width / fig.dpi; t_.remove()
    ICON[key](ov, xi - wt / 2 - 0.07, yi - 0.055, 0.08)
save(fig, "fig_views", sys.argv[1] if len(sys.argv) > 1 else None, table=pd.DataFrame(rows))
