"""Figure: what the compound views know that chemistry does not. Dossier: paper/figures/fig_views.md.

(a) View ablation, PCC_plate of UnDetect (transfer terms + interaction) with one view alone, all views without one,
    and all views with one view's compound labels permuted; lines = drug-free backbone and the full model; dots =
    five-fold means.
(b) Chemically unlike pairs (Morgan Tanimoto < 0.25): response similarity by each view's similarity bin; dashed =
    chemical analogues (Tanimoto >= 0.6). (c) Per test compound: plate-centred gain of the multi-view kernel over the
    chemistry kernel, by the compound's nearest chemical neighbour in training. (d) Four chemically unlike pairs that
    respond alike: structures (RDKit CoordGen depiction) and both compounds' plate-centred signatures on the 30 genes
    with the largest |z| in either, ordered by the first compound.
Source (final backbone cache_v1, user 2026-11-01): results/sdst_hybrid/views_final/means.csv
(`scripts/sdst_view_ablation.py`, jobs 359222-359226; report/11_06_report.md §8);
results/sdst_hybrid/compound_analysis_final/{summary.json,example_pairs.json} (`analyze_sdst_compounds.py --final`,
job 359229).
Run: /nfs-stor/yuqi.lei/Therapeutic/envs/gnn-gpu/bin/python -I scripts/figures/fig_views.py [outdir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rows = []

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import DIV_NEG, DIV_POS, GRID, INK, INK2, LINE, OURS, VIEW, panel, plt, save  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RES = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/compound_analysis_final")
ABL = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/views_final/means.csv")
NGENES = 30
LABW = 0.50                                 # width of the row labels next to the strips (inches)
ATOM_C = {"N": "#2a5bd7", "O": "#d43f3a", "S": "#b08800", "F": "#1a9a6c", "Cl": "#1a9a6c", "Br": "#8b3a1a",
          "P": "#e07b00", "B": "#b05a7a", "I": "#6b2fa0"}
CMAP = LinearSegmentedColormap.from_list("div", [DIV_NEG, "#efeeeb", DIV_POS])

S = json.loads((RES / "summary.json").read_text())
EX = json.loads((RES / "example_pairs.json").read_text())


def draw_mol(ax, d, x0, y0, w, h):
    """Skeletal drawing from exported 2D coordinates, fitted into the box (x0, y0, w, h) in data units."""
    xy = np.array([[a["x"], a["y"]] for a in d["atoms"]])
    span = np.ptp(xy, 0).clip(1e-6); sc = min(w / span[0], h / span[1])
    xy = (xy - xy.min(0)) * sc + np.array([x0 + (w - span[0] * sc) / 2, y0 + (h - span[1] * sc) / 2])
    bl = np.median([np.linalg.norm(xy[b["a"]] - xy[b["b"]]) for b in d["bonds"]])
    ctr = xy.mean(0)
    for b in d["bonds"]:
        p, q = xy[b["a"]], xy[b["b"]]
        ax.plot(*zip(p, q), color=INK, lw=0.6, solid_capstyle="round", zorder=2)
        if b["order"] >= 2:
            v = q - p; nrm = np.array([-v[1], v[0]]) / max(np.linalg.norm(v), 1e-9)
            if np.dot(ctr - (p + q) / 2, nrm) < 0:
                nrm = -nrm
            for k in range(1, b["order"]):
                off = nrm * 0.18 * bl * k
                ax.plot(*zip(p + off + 0.15 * v, q + off - 0.15 * v), color=INK, lw=0.6, zorder=2)
    for a, (x, y) in zip(d["atoms"], xy):
        if a["el"] != "C":
            s = a["el"] + ("H" if a["h"] == 1 else f"H{a['h']}" if a["h"] > 1 else "")
            ax.text(x, y, s, fontsize=5.5, color=ATOM_C.get(a["el"], INK), ha="center", va="center", zorder=3,
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))


fig = plt.figure(figsize=(6.5, 4.4))
top = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.75, 0.8], wspace=0.5, left=0.105, right=0.985, top=0.95,
                       bottom=0.66)
z, a, b = (fig.add_subplot(top[0, k]) for k in range(3))
CH = 0.54 * 4.4                     # height of panel c in inches (data units = inches)
c = fig.add_axes([0.0, 0.0, 1.0, 0.54]); c.set_xlim(0, 6.5); c.set_ylim(0, CH); c.axis("off")

# (a) view ablation on PCC_plate
M = pd.read_csv(ABL).set_index(["set", "arm"])["plate"]
cen = {f"pool_{s}": float(M[(s, "undetect")]) for s, _ in M.index if s != "none"} | {"mlp": float(M[("none", "backbone")])}
kinds = [("only", "alone"), ("minus", "without"), ("perm", "shuffled")]
for r_, (k, _) in enumerate(kinds):
    for j, v in enumerate(("chem", "tgt", "moa", "morph")):
        z.scatter(cen[f"pool_{k}_{v}"], 2 - r_ + (j - 1.5) * 0.12, s=16, color=VIEW[v], zorder=3, lw=0)
        rows.append({"panel": "a", "kind": k, "view": v, "pcc_plate": float(cen[f"pool_{k}_{v}"])})
z.axvline(cen["mlp"], color=INK2, ls="--", lw=0.9, zorder=2)
z.axvline(cen["pool_multi_norm"], color=OURS, lw=1.0, zorder=2)
z.set_yticks([2, 1, 0], [n for _, n in kinds], fontsize=8)
z.set_ylim(-0.6, 2.6); z.set_xlim(0.09, 0.165); z.set_xticks([0.10, 0.13, 0.16])
z.set_xlabel(r"PCC$_\mathrm{plate}$", fontsize=8.5)
z.grid(axis="x", color=GRID, lw=0.6); z.set_axisbelow(True)
z.spines["left"].set_visible(False); z.tick_params(axis="y", length=0)
panel(z, "a", -0.42)
for j, (v, nm) in enumerate((("chem", "Chemistry"), ("tgt", "Targets"), ("moa", "MOA"), ("morph", "Cell Painting"))):
    z.scatter([], [], s=16, color=VIEW[v], label=nm)
z.legend(loc="upper left", bbox_to_anchor=(-0.45, -0.30), ncol=4, frameon=False, fontsize=7.5, handletextpad=0.1,
         columnspacing=0.6, borderaxespad=0)

# (b) response similarity by view similarity, chemically unlike pairs
groups = [("Targets", "tgt", ["0", ">0", ".5"]), ("MOA", "moa", ["0", ">0", ".5"]),
          ("Cell Painting", "morph", ["", ".3", ".5", ".7"])]
x0, ticks, labels = 0.0, [], []
for name, key, labs in groups:
    bins = S["view_bins_chem_low"][key]; xs = x0 + np.arange(len(bins))
    a.bar(xs, [r["resp_mean"] for r in bins], width=0.75, color=VIEW[key], zorder=2)
    ticks += list(xs); labels += labs
    a.text(xs.mean(), -0.075, name, ha="center", va="top", fontsize=8.5, color=INK)
    rows += [{"panel": "b", "view": key, "bin": r["bin"], "resp_mean": r["resp_mean"], "n_pairs": r["n_pairs"]}
             for r in bins]
    x0 = xs[-1] + 1.6
ana = S["resp_mean"]["chem_high_analogues"]
a.axhline(ana, color=VIEW["chem"], ls="--", lw=1.0, zorder=3)
a.text(-0.4, ana + 0.008, "chemical analogues", color=VIEW["chem"], fontsize=8, ha="left", va="bottom")
a.axhline(0, color=INK2, lw=0.6, zorder=3)
a.set_xticks(ticks, labels, fontsize=7)
a.set_ylabel("response similarity (r)")
a.set_ylim(-0.02, 0.31)
a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
panel(a, "b", -0.17)

# (c) gain of the multi-view kernel over chemistry, by nearest chemical neighbour in training
t = pd.DataFrame(S["partB_by_chem_bin"])
labs = {"[0.0, 0.3)": "<.3", "[0.3, 0.5)": ".3", "[0.5, 0.7)": ".5", "[0.7, 1.0)": ".7"}
b.bar(range(len(t)), t.gain_multi_vs_chem, width=0.65, color=OURS, zorder=2)
b.set_xticks(range(len(t)), [labs.get(x, x) for x in t.chem_bin], fontsize=7.5)
b.set_xlabel("nearest chemical\nneighbour", fontsize=8.5)
b.set_ylabel("gain over chemistry", fontsize=8)
b.axhline(0, color=INK2, lw=0.6)
b.grid(axis="y", color=GRID, lw=0.6); b.set_axisbelow(True)
panel(b, "c", -0.62)
rows += [{"panel": "c", "nn_chem_bin": r.chem_bin, "gain_multi_vs_chem": r.gain_multi_vs_chem, "n_drugs": r.n_drugs}
         for r in t.itertuples()]

# (d) four pairs: structures and response patterns
c.text(0.06, CH - 0.30, "d", fontsize=11, fontweight="bold", ha="left", va="center")
CW = 6.5 / len(EX)
for k, e in enumerate(EX):
    xl = k * CW + 0.10; wcol = CW - 0.20
    c.text(xl + wcol / 2, CH - 0.30, e["class"], fontsize=8.5, ha="center", va="center", color=INK)
    for m in range(2):
        draw_mol(c, e["drawing"][m], xl + m * wcol / 2 + 0.09, 1.10, wcol / 2 - 0.18, 0.90)
        c.text(xl + m * wcol / 2 + wcol / 4, 0.96, e["pair"][m], fontsize=7, ha="center", va="center", color=INK2)
    sA, sB = (np.array(s_) for s_ in e["signature"])
    zA, zB = sA / sA.std(), sB / sB.std()
    g = np.argsort(-np.maximum(np.abs(zA), np.abs(zB)))[:NGENES]; g = g[np.argsort(-zA[g])]
    xs = xl + LABW                           # strips start after the row label (compound name)
    for m, z in enumerate((zA, zB)):
        v = z[g] / np.abs(z[g]).max()
        c.imshow(v[None, :], cmap=CMAP, vmin=-1, vmax=1, aspect="auto", interpolation="nearest",
                 extent=(xs, xl + wcol, 0.62 - m * 0.2, 0.78 - m * 0.2), zorder=2)
        c.add_patch(plt.Rectangle((xs, 0.62 - m * 0.2), xl + wcol - xs, 0.16, fc="none", ec=LINE, lw=0.5, zorder=3))
        c.text(xs - 0.04, 0.70 - m * 0.2, e["pair"][m], fontsize=5.6, ha="right", va="center", color=INK2)
        rows += [{"panel": "d", "pair": "–".join(e["pair"]), "compound": e["pair"][m], "gene_rank": i,
                  "gene_index": int(gi), "z_scaled": float(vv)} for i, (gi, vv) in enumerate(zip(g, v))]
    c.text(xl + wcol / 2, 0.24, f"Tanimoto {e['tanimoto']:.2f}   r {e['resp_r']:.2f}", fontsize=7.5, ha="center",
           va="center", color=INK2)
# colour key for the strips (2026-11-02, user: the strips were hard to read)
kx, kw, ky, kh = 4.45, 0.9, 0.035, 0.07
c.imshow(np.linspace(-1, 1, 64)[None, :], cmap=CMAP, vmin=-1, vmax=1, aspect="auto", extent=(kx, kx + kw, ky, ky + kh),
         zorder=2)
c.add_patch(plt.Rectangle((kx, ky), kw, kh, fc="none", ec=LINE, lw=0.5, zorder=3))
c.text(kx - 0.05, ky + kh / 2, "down", fontsize=6.5, ha="right", va="center", color=INK2)
c.text(kx + kw + 0.05, ky + kh / 2, "up", fontsize=6.5, ha="left", va="center", color=INK2)
c.text(0.10, ky + kh / 2, f"Strips: plate-centred response on the {NGENES} strongest genes, same order in both rows",
       fontsize=6.5, ha="left", va="center", color=INK2)
c.set_xlim(0, 6.5); c.set_ylim(0, CH)

save(fig, "fig_views", sys.argv[1] if len(sys.argv) > 1 else None, table=pd.DataFrame(rows))
