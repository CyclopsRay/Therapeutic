"""Figures for the course final presentation (2026-10-13): Rhaister-O and our changes.

Every number is copied from a weekly report (cited beside it); nothing is recomputed here.
The per-line real-DMSO / CCA values come from the CCA explorer export of jobs 308632/308619
(report 10_09 §9); per-line direct-bridge values exist only for the three lines quoted in
report 10_09 §11 (the full per_line.csv lives on /l).

Run (login node, seconds):
  PYTHONPATH=<dir with matplotlib> python3 scripts/plot_final_presentation_figures.py
Outputs: report/figures/10_16_final/*.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "report" / "figures" / "10_16_final"
INK, INK2, RULE, GRID, BG = "#16202b", "#4a5a69", "#c4ccd5", "#e2e7ec", "#f7f8fa"
BLUE, ORANGE, TEAL, GREY, GOLD = "#0070a6", "#b5531f", "#1f6f78", "#8a97a4", "#b08a2e"

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 14,
    "ytick.labelsize": 14, "legend.fontsize": 14, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK, "text.color": INK, "figure.facecolor": BG,
    "axes.facecolor": BG, "savefig.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(OUT / name)


def baseline_decomposition():
    # report 09_11 §3.1 (25 held-out lines, line-macro Pearson)
    rows = ["Drug mean\n(no line info)", "Nearest DMSO\nline (1-NN)", "Nearest DMSO\nlines (3-NN)", "Rhaister-O"]
    full = [0.443354, 0.298532, 0.340531, 0.631400]
    resid = [0.0, 0.1472, 0.1388, 0.537951]
    pc1 = [-0.0028, 0.1282, 0.1174, 0.525927]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(rows))[::-1]
    h = 0.26
    for off, vals, col, lab in [(h, full, GREY, "full Δ"), (0, resid, BLUE, "drug mean removed"),
                                (-h, pc1, TEAL, "+ reference PC1 removed")]:
        ax.barh(y + off, vals, height=h, color=col, label=lab)
        for yy, v in zip(y + off, vals):
            ax.text(max(v, 0) + 0.008, yy, f"{v:.3f}", va="center", fontsize=13, color=INK2)
    ax.set_yticks(y, rows)
    ax.set_xlim(-0.01, 0.72)
    ax.set_xlabel("Line-macro Pearson, 25 held-out lines")
    ax.axvline(0, color=RULE, lw=1)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3, frameon=False)
    save(fig, "baseline_decomposition.png")


def context_slopes():
    # Ours only. Left: first pass on the benchmark splits, settings tuned (report 10_02 §2.4, §3.5).
    # Right: confirmation on never-scored lines, settings fixed in advance (report 10_02 §3.7).
    arms = ["Similarity weights\n(local slopes)", "Reliability\nweights", "Similarity ×\nreliability"]
    first_t, first_e = [0.0134, 0.0175, 0.0316], [0.0059, 0.0022, 0.0091]
    tahoe = [(0.0077, -0.0009, 0.0151), (0.0150, 0.0126, 0.0174), (0.0267, 0.0223, 0.0311)]
    eb = [(0.0060, 0.0029, 0.0090), (0.0011, -0.0002, 0.0024), (0.0080, 0.0045, 0.0115)]
    y = np.arange(len(arms))[::-1] * 1.0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    for ax, title in [(a1, "First pass: benchmark splits, settings tuned"),
                      (a2, "Confirmation: never-scored lines, settings fixed (95% CI)")]:
        ax.axvline(0, color=INK2, lw=1)
        ax.axvline(0.01, color=GOLD, lw=2, ls="--")
        ax.text(0.0103, y[0] + 0.55, "gate +0.01", color=GOLD, fontsize=13)
        ax.set_xlim(-0.004, 0.040)
        ax.set_ylim(-0.6, y[0] + 0.8)
        ax.set_title(title, loc="left", color=INK)
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.set_xlabel("Drug-residual gain over Rhaister-O")
    for vals, off, col in [(first_t, 0.17, BLUE), (first_e, -0.17, ORANGE)]:
        for yy, v in zip(y + off, vals):
            a1.plot(v, yy, "o", color=col, ms=11)
            a1.text(v + 0.0011, yy, f"{v:+.4f}", va="center", fontsize=13, color=col)
    for off, data, col in [(0.17, tahoe, BLUE), (-0.17, eb, ORANGE)]:
        for yy, (m, lo, hi) in zip(y + off, data):
            a2.plot([lo, hi], [yy, yy], color=col, lw=3, solid_capstyle="round")
            a2.plot(m, yy, "o", color=col, ms=11)
            a2.text(hi + 0.0008, yy, f"{m:+.4f}", va="center", fontsize=13, color=col)
    a1.set_yticks(y, arms)
    a1.plot([], [], "o", color=BLUE, ms=10, label="Tahoe")
    a1.plot([], [], "o", color=ORANGE, ms=10, label="Emerald Bay (second atlas)")
    a1.legend(loc="upper center", bbox_to_anchor=(1.05, -0.14), ncol=2, frameon=False)
    fig.tight_layout(w_pad=2)
    save(fig, "context_slopes.png")


def modification_ladder():
    # Drug-residual gain over Rhaister-O on Tahoe, point estimates; only arms named elsewhere in the deck.
    rows = [
        ("Similarity × reliability weights", 0.0267, "ours"),   # 10_02 §3.6 (fresh lines)
        ("Reliability weights", 0.0150, "ours"),                # 10_02 §3.3 (fresh lines)
        ("Similarity weights (local slopes)", 0.0077, "ours"),  # 10_02 §3.3 (fresh lines)
        ("Slope varies along cell-state PCs", 0.0011, "alt"),   # 10_02 §2.4
        ("Quadratic term in x", -0.0003, "alt"),                # 10_02 §2.4b
        ("Cross-gene low-rank correction", -0.0145, "alt"),     # 09_18 §3
        ("DepMap expression correction", 0.0004, "alt"),        # 09_11 §4.2
        ("Reactome gene prior", 0.0009, "alt"),                 # 10_02 §2.4b
        ("  its random-graph control", 0.0002, "ctrl"),         # 10_02 §2.4b
        ("Similarity by tissue of origin", -0.0344, "alt"),     # 10_02 §4.1 (fresh lines)
    ]
    colors = {"ours": BLUE, "alt": ORANGE, "ctrl": GREY}
    fig, ax = plt.subplots(figsize=(13, 7.2))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, g) in zip(y, rows):
        ax.barh(yy, v, color=colors[g], height=0.64)
        ax.text(v + (0.0006 if v >= 0 else -0.0006), yy, f"{v:+.4f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=13, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    for t, r in zip(ax.get_yticklabels(), rows):
        if r[2] == "ours":
            t.set_color(BLUE)
            t.set_fontweight("bold")
    ax.axhline(y[2] - 0.5, color=RULE, lw=1.2)
    ax.axvline(0, color=INK2, lw=1)
    ax.axvline(0.01, color=GOLD, lw=2, ls="--")
    ax.text(0.0103, y[0] + 0.75, "gate +0.01", color=GOLD, fontsize=13)
    ax.set_ylim(-0.7, y[0] + 1.2)
    ax.set_xlim(-0.043, 0.034)
    ax.set_xlabel("Drug-residual gain over Rhaister-O, Tahoe (0 = Rhaister-O)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, ["our weighting", "other changes to Rhaister-O"], loc="lower right",
              frameon=True, facecolor=BG, edgecolor=RULE, fontsize=13)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    save(fig, "modification_ladder.png")


def quality_story():
    fig, axs = plt.subplots(2, 2, figsize=(16, 9.2))
    (a, b), (c, d) = axs
    # (a) reliability score per reference line: report 09_25 §7; weights at alpha = 20: report 10_02 §2.4
    names = ["NCI-H596", "NCI-H2122", "SW 1271", "H4", "median of\nother lines"]
    q = [0.6537, 0.7238, 0.9488, 0.9620, 0.9755]
    cols = [ORANGE, ORANGE, GREY, GREY, BLUE]
    yy = np.arange(len(names))[::-1]
    a.barh(yy, q, color=cols, height=0.6)
    for y0, v in zip(yy, q):
        a.text(v + 0.004, y0, f"{v:.3f}", va="center", fontsize=13, color=INK2)
    a.text(0.662, yy[0] - 0.42, "weight 0.0003", fontsize=12.5, color=ORANGE)
    a.text(0.732, yy[1] - 0.42, "weight 0.0026", fontsize=12.5, color=ORANGE)
    a.set_yticks(yy, names)
    a.set_xlim(0.6, 1.03)
    a.set_xlabel("Reliability: agreement of the line's own DMSO across 14 plates")
    a.set_title("(a) Two reference lines do not reproduce", loc="left", color=INK)
    a.xaxis.grid(True, color=GRID)
    a.set_axisbelow(True)
    # (b) NCI-H596 held out once: report 10_02 §3.4
    b.barh([1], [0.661 - 0.422], left=[0.422], height=0.5, color="#cfe0ec")
    b.plot([0.5642], [1], "o", color=BLUE, ms=12)
    b.text(0.5642, 1.4, "other 24 lines: mean 0.564", ha="center", fontsize=13, color=BLUE)
    b.plot([0.0279], [0], "o", color=ORANGE, ms=14)
    b.text(0.065, 0, "0.028", va="center", fontsize=15, color=ORANGE, fontweight="bold")
    b.set_yticks([0, 1], ["NCI-H596\nas a test line", "fresh Tahoe\ntest lines"])
    b.set_xlim(-0.02, 0.72)
    b.set_ylim(-0.6, 1.9)
    b.set_xlabel("Rhaister-O drug-residual Pearson")
    b.set_title("(b) The worst one cannot be predicted at all", loc="left", color=INK)
    b.xaxis.grid(True, color=GRID)
    b.set_axisbelow(True)
    # (c) quality-ordered reference sets: report 09_25 §6 (jobs 272001, 272021)
    m = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    rand = [0.157076, 0.275054, 0.363027, 0.411351, 0.462853, 0.487762, 0.505906, 0.524332, 0.540245]
    best = [0.2033, 0.3222, 0.4154, 0.4842, 0.5116, 0.5268, 0.5425, 0.5544]
    worst = [-0.0414, 0.1292, 0.2655, 0.3559, 0.4258, 0.4761, 0.5030, 0.5240]
    c.plot(m, rand, "o-", color=GREY, lw=3, ms=7, label="random")
    c.plot(m[:-1], best, "o-", color=BLUE, lw=3.5, ms=7, label="most reliable")
    c.plot(m[:-1], worst, "o-", color=ORANGE, lw=3, ms=7, label="least reliable")
    c.annotate("best 40 = 0.554 > all 45 = 0.540", (40, 0.5544), (14, 0.63), color=BLUE, fontsize=13,
               arrowprops=dict(arrowstyle="-", color=BLUE))
    c.axhline(0, color=INK2, lw=1)
    c.set_ylim(-0.08, 0.7)
    c.set_xlabel("Reference lines used to fit")
    c.set_ylabel("Drug-residual Pearson")
    c.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), frameon=False)
    c.grid(True, color=GRID)
    c.set_axisbelow(True)
    c.set_title("(c) Fewer but reliable lines beat all lines", loc="left", color=INK)
    # (d) reliability gain depends on whether NCI-H596 is in the reference panel: report 10_02 §3.3
    d.bar([0, 1], [0.0170, 0.0068], color=[BLUE, GREY], width=0.55)
    for x0, v in zip([0, 1], [0.0170, 0.0068]):
        d.text(x0, v + 0.0005, f"+{v:.4f}", ha="center", fontsize=14, color=INK2)
    d.set_xticks([0, 1], ["NCI-H596 among\nreference lines", "NCI-H596 held out\n(not a reference)"])
    d.set_ylim(0, 0.021)
    d.set_ylabel("Reliability-weight gain")
    d.set_title("(d) Reliability gain is largest when it has a bad line to drop", loc="left", color=INK)
    d.yaxis.grid(True, color=GRID)
    d.set_axisbelow(True)
    fig.tight_layout(h_pad=2.5, w_pad=3)
    save(fig, "quality_story.png")


def mkl_results():
    # report 10_09 §7 (jobs 296806 plate-map, 296807 ik14), P2-24, full Pearson, ACTIVE rows
    tiers = ["mean", "T1\nchemistry", "T2\n+targets", "T3\n+images", "T4\n+response"]
    plate = [0.315, 0.342, 0.348, 0.393, 0.524]
    ik14 = [0.330, 0.363, 0.371, 0.450, 0.576]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.6), gridspec_kw={"width_ratios": [1, 1.15]})
    x = np.arange(len(tiers))
    a1.bar(x - 0.19, plate, width=0.38, color=BLUE, label="plate-map split (strict)")
    a1.bar(x + 0.19, ik14, width=0.38, color="#7fb3d1", label="compound-group split")
    for xx, v in zip(x - 0.19, plate):
        a1.text(xx, v + 0.008, f"{v:.3f}", ha="center", fontsize=12, color=INK2)
    a1.set_xticks(x, tiers, fontsize=12.5)
    a1.set_ylim(0, 0.66)
    a1.set_ylabel("Pearson, new drugs")
    a1.legend(loc="upper left", frameon=False)
    a1.yaxis.grid(True, color=GRID)
    a1.set_axisbelow(True)
    a1.set_title("(a) Score as kernels are added", loc="left", color=INK)
    rows = [  # (label, diff, low, high, kind) plate-map split
        ("T1 chemistry − mean", 0.027, 0.024, 0.031, "tier"),
        ("T2 knowledge − T1", 0.013, 0.008, 0.017, "tier"),
        ("  T2 − T2 with knowledge nulled", 0.004, 0.002, 0.005, "null"),
        ("  target kernel − permuted targets", 0.035, 0.029, 0.042, "null"),
        ("  PPI kernel − rewired PPI", 0.000, -0.003, 0.003, "null"),
        ("  target degree − mean", 0.002, -0.001, 0.004, "null"),
        ("T3 morphology − T2", 0.052, 0.045, 0.060, "tier"),
        ("  T3 − T3 with knowledge nulled", 0.000, -0.001, 0.002, "null"),
        ("T4 measured response − T3", 0.131, 0.122, 0.141, "tier"),
    ]
    yy = np.arange(len(rows))[::-1]
    for y0, (lab, dlt, lo, hi, k) in zip(yy, rows):
        col = BLUE if k == "tier" else TEAL
        a2.plot([lo, hi], [y0, y0], color=col, lw=3)
        a2.plot(dlt, y0, "o", color=col, ms=10)
        a2.text(hi + 0.004, y0, f"{dlt:+.3f}", va="center", fontsize=13, color=col)
    a2.set_yticks(yy, [r[0] for r in rows])
    a2.axvline(0, color=INK2, lw=1)
    a2.set_xlim(-0.01, 0.175)
    a2.set_xlabel("Gain in Pearson (95% bootstrap over plate maps)")
    a2.xaxis.grid(True, color=GRID)
    a2.set_axisbelow(True)
    a2.set_title("(b) What each kernel adds, against its null", loc="left", color=INK)
    fig.tight_layout(w_pad=3)
    save(fig, "mkl_results.png")


def bridge_comparison():
    # report 10_09 §9 and §11 (23 held-out lines), 10_02 §2.7, 09_18 §1; drug-residual, 95% line bootstrap
    rows = [
        ("Real Tahoe DMSO (upper reference)", 0.539, 0.483, 0.583, GREY),
        ("DepMap → per-gene bridge (ours)", 0.406, 0.361, 0.442, TEAL),
        ("DepMap ridge → response (unwhitened)", 0.260, 0.207, 0.315, "#b9c3cc"),
        ("DepMap → CCA bridge", 0.249, 0.195, 0.304, ORANGE),
        ("DepMap ridge → response (PCA)", 0.209, 0.159, 0.260, "#b9c3cc"),
        ("Nearest reference line by DepMap", 0.145, 0.081, 0.209, "#b9c3cc"),
        ("Control: another line's CCA profile", 0.084, 0.019, 0.150, "#dbe1e8"),
        ("Control: another line's bridged profile", 0.056, 0.012, 0.102, "#dbe1e8"),
        ("Control: mean DMSO (no line info)", 0.001, -0.008, 0.010, "#dbe1e8"),
    ]
    fig, ax = plt.subplots(figsize=(13, 7))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, lo, hi, col) in zip(y, rows):
        ax.barh(yy, v, color=col, height=0.66)
        ax.plot([lo, hi], [yy, yy], color=INK, lw=1.6)
        ax.text(hi + 0.008, yy, f"{v:.3f}", va="center", fontsize=14,
                color=TEAL if "ours" in lab else INK2, fontweight="bold" if "ours" in lab else "normal")
    ax.set_yticks(y, [r[0] for r in rows])
    for t in ax.get_yticklabels():
        if "ours" in t.get_text():
            t.set_color(TEAL)
            t.set_fontweight("bold")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-0.02, 0.66)
    ax.set_xlabel("Drug-residual Pearson through frozen Rhaister-O, same 23 held-out lines")
    ax.text(0.425, y[1] - 0.62, "75% of the real-DMSO gain", color=TEAL, fontsize=13.5)
    ax.text(0.27, y[3] - 0.58, "reconstruction r: bridge 0.69, CCA 0.38", color=INK2, fontsize=13)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    save(fig, "bridge_comparison.png")


# Per-line values, report 10_09 §9 explorer export: (line, split, real DMSO, CCA bridge, CCA reconstruction r)
PER_LINE = [
    ("HCT-15", 9, 0.570, 0.399, 0.567), ("LS 180", 6, 0.612, 0.414, 0.653), ("RKO", 8, 0.559, 0.171, 0.101),
    ("SW 480", 9, 0.593, 0.131, 0.093), ("Hs 578T", 7, 0.644, 0.507, 0.677), ("A-172", 7, 0.587, 0.303, 0.466),
    ("KATO III", 9, 0.651, 0.430, 0.613), ("SNU-423", 8, 0.609, 0.378, 0.566), ("A427", 8, 0.501, 0.317, 0.385),
    ("HOP-62", 5, 0.712, 0.437, 0.583), ("NCI-H1792", 7, 0.567, 0.280, 0.416), ("NCI-H2122", 9, 0.061, 0.028, 0.290),
    ("NCI-H23", 8, 0.555, 0.257, 0.378), ("NCI-H2347", 8, 0.547, 0.187, 0.330), ("SW 1271", 6, 0.367, 0.045, 0.139),
    ("AsPC-1", 7, 0.490, 0.278, 0.447), ("CFPAC-1", 7, 0.585, 0.228, 0.405), ("Hs 766T", 5, 0.599, 0.050, 0.169),
    ("MIA PaCa-2", 6, 0.549, 0.165, 0.169), ("PANC-1", 5, 0.594, 0.119, 0.190), ("CHP-212", 9, 0.505, 0.112, 0.254),
    ("C32", 5, 0.458, 0.251, 0.495), ("AN3 CA", 6, 0.490, 0.238, 0.321),
]
# report 10_09 §11: direct per-gene bridge values quoted for three lines
DIRECT = {"HOP-62": 0.480, "SW 480": 0.375, "Hs 766T": 0.406}


def bridge_per_line():
    rows = sorted(PER_LINE, key=lambda r: r[2])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(rows))
    for yy, (name, split, real, cca, rec) in zip(y, rows):
        a1.plot([cca, real], [yy, yy], color=RULE, lw=3, zorder=1)
        a1.plot(real, yy, "o", color=BLUE, ms=10, zorder=3)
        a1.plot(cca, yy, "o", color=ORANGE, ms=10, zorder=3)
        if name in DIRECT:
            a1.plot(DIRECT[name], yy, "D", color=TEAL, ms=10, zorder=4, mec=BG, mew=1.2)
    a1.set_yticks(y, [f"{r[0]} (s{r[1]})" for r in rows])
    a1.set_xlim(-0.02, 0.76)
    a1.set_xlabel("Drug-residual Pearson")
    a1.plot([], [], "o", color=BLUE, ms=10, label="real Tahoe DMSO")
    a1.plot([], [], "o", color=ORANGE, ms=10, label="CCA bridge")
    a1.plot([], [], "D", color=TEAL, ms=10, label="per-gene bridge (quoted lines)")
    a1.legend(loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=3, frameon=False, fontsize=13)
    a1.xaxis.grid(True, color=GRID)
    a1.set_axisbelow(True)
    a1.set_title("23 held-out lines with DepMap data", loc="left", color=INK)
    rec = np.array([r[4] for r in PER_LINE])
    cca = np.array([r[3] for r in PER_LINE])
    r = np.corrcoef(rec, cca)[0, 1]
    a2.scatter(rec, cca, s=90, color=ORANGE, zorder=3)
    for name, _, _, c, rr in PER_LINE:
        if rr < 0.2 or rr > 0.65 or name == "NCI-H2122":
            off = {"SW 1271": (-78, -6), "Hs 766T": (8, 8), "NCI-H2122": (8, -4)}.get(name, (6, 4))
            a2.annotate(name, (rr, c), off, textcoords="offset points", fontsize=12.5, color=INK2)
    a2.set_xlabel("How well CCA reconstructs the line's DMSO (r)")
    a2.set_ylabel("CCA-bridge drug-residual Pearson")
    a2.set_xlim(0, 0.75)
    a2.set_ylim(-0.02, 0.56)
    a2.grid(True, color=GRID)
    a2.set_axisbelow(True)
    a2.set_title(f"Prediction tracks reconstruction (r = {r:.2f})", loc="left", color=INK)
    fig.tight_layout(w_pad=3)
    save(fig, "bridge_per_line.png")


def baseline_simple():
    # report 09_11 §3.1: drug-mean-adjusted ("drug-residual") Pearson, 25 held-out lines
    rows = [("Drug average\n(no line info)", 0.0), ("Copy the most similar\nuntreated line", 0.1472),
            ("Rhaister-O", 0.537951)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[GREY, "#9db9cc", BLUE], height=0.6)
    for yy, (_, v) in zip(y, rows):
        ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=20, color=INK, fontweight="bold")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=17)
    ax.set_xlim(0, 0.66)
    ax.set_xlabel("Drug-adjusted Pearson (each drug's average response removed)", fontsize=15)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    save(fig, "baseline_simple.png")


def same_line_similarity():
    # report 10_09 §9 (job 308632): 47 x 47 DepMap (rows) x Tahoe (columns) similarity matrices
    base = Path(__file__).resolve().parents[1] / "report" / "figures" / "10_09_cca"
    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(0)
    for i, (fname, lab, col) in enumerate([("fig2_direct_matrix.csv", "Direct gene match", BLUE),
                                           ("fig2_loo_cca_matrix.csv", "CCA shared space", ORANGE)]):
        m = np.loadtxt(base / fname, delimiter=",", skiprows=1, usecols=range(1, 48))
        diag = np.diag(m)
        off = m[~np.eye(len(m), dtype=bool)]
        top1 = np.mean(m.argmax(1) == np.arange(len(m)))
        x0 = i * 3.0
        parts = ax.violinplot([off], positions=[x0], widths=1.1, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor("#c4ccd5")
            b.set_alpha(0.8)
        ax.scatter(x0 + 1.1 + rng.uniform(-0.25, 0.25, len(diag)), diag, s=46, color=col, zorder=3)
        ax.text(x0 + 0.55, 1.08, f"{lab}\nown line found: {top1:.0%}", ha="center", fontsize=15, color=col,
                fontweight="bold")
    ax.set_xticks([0, 1.1, 3.0, 4.1], ["other lines", "same line", "other lines", "same line"], fontsize=15)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylim(-0.95, 1.25)
    ax.set_ylabel("Similarity of a line's DepMap\nand Tahoe profiles", fontsize=15)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    save(fig, "same_line_similarity.png")


# ---------------------------------------------------------------------------
# Revision 5 (2026-10-12): "better" arrows, explainer figures, MKL tiers.
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

GOOD = "#1f6f78"


def better_arrow(ax, x0=0.70, x1=0.97, y=0.05, label="better"):
    """Arrow in axes coordinates pointing towards the better side."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=3.5, color=GOOD, mutation_scale=28))
    ax.text(x0 - 0.01, y, label, transform=ax.transAxes, ha="right", va="center", fontsize=17,
            color=GOOD, fontweight="bold")


def baseline_simple_v2():
    rows = [("Drug average\n(no line info)", 0.0), ("Copy the most similar\nuntreated line", 0.1472),
            ("Rhaister-O", 0.537951)]  # report 09_11 §3.1
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[GREY, "#9db9cc", BLUE], height=0.6)
    for yy, (_, v) in zip(y, rows):
        ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=20, color=INK, fontweight="bold")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=17)
    ax.set_xlim(0, 0.66)
    ax.set_xlabel("Drug-adjusted Pearson (each drug's average response removed)", fontsize=15)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    better_arrow(ax, 0.72, 0.97, 0.93)
    save(fig, "baseline_simple.png")


def slope_alternatives():
    # left: benchmark splits, tuned (10_02 §2.4, §2.4b; 09_18 §3); right: fresh lines, h fixed (10_02 §4.1)
    left = [("Similarity weights (ours)", 0.0134, BLUE), ("Slope varies along cell-state PCs", 0.0011, ORANGE),
            ("Quadratic term in x", -0.0003, ORANGE), ("Cross-gene low-rank term", -0.0145, ORANGE),
            ("Gene-network prior (Reactome)", 0.0009, ORANGE), ("  its random-graph control", 0.0002, GREY)]
    right = [("Untreated profile (ours)", 0.0077, BLUE), ("DepMap expression", 0.0069, ORANGE),
             ("CRISPR dependency", 0.0054, ORANGE), ("Copy number", -0.0039, ORANGE),
             ("Tissue of origin", -0.0344, ORANGE)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1.05, 1]})
    for ax, rows, title, xlim in [(a1, left, "How the slope may vary", (-0.026, 0.022)),
                                  (a2, right, "Similarity measured on…", (-0.052, 0.016))]:
        y = np.arange(len(rows))[::-1]
        for yy, (lab, v, col) in zip(y, rows):
            ax.barh(yy, v, color=col, height=0.62)
            ax.text(v + (0.0006 if v >= 0 else -0.0006), yy, f"{v:+.4f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=14, color=INK2)
        ax.set_yticks(y, [r[0] for r in rows], fontsize=15)
        for t, r in zip(ax.get_yticklabels(), rows):
            if "ours" in r[0]:
                t.set_color(BLUE)
                t.set_fontweight("bold")
        ax.axvline(0, color=INK2, lw=1)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.8, len(rows) - 0.3)
        ax.set_title(title, loc="left", color=INK, fontsize=18)
        ax.set_xlabel("Gain over Rhaister-O (drug-adjusted)")
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        better_arrow(ax, 0.72, 0.97, 0.04)
    fig.tight_layout(w_pad=3)
    save(fig, "slope_alternatives.png")


def similarity_cartoon():
    """Illustration with toy data: how similarity weighting changes the fitted slope."""
    rng = np.random.default_rng(3)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.4))
    pts = rng.normal(0, 1, (30, 2))
    new = np.array([1.1, 0.6])
    d = np.linalg.norm(pts - new, axis=1)
    w = np.exp(-d / 0.8)
    a1.scatter(pts[:, 0], pts[:, 1], s=60 + 900 * w, color=ORANGE, alpha=0.55, edgecolors="none")
    a1.scatter(*new, marker="*", s=900, color=INK, zorder=3)
    a1.annotate("new line", new, (new[0] + 0.35, new[1] + 0.55), fontsize=16, color=INK)
    a1.set_title("Weight = how similar the untreated profile is", loc="left", fontsize=17, color=INK)
    a1.set_xticks([])
    a1.set_yticks([])
    a1.set_xlabel("untreated profile (2-D sketch)", fontsize=15)
    # one drug, one gene: response vs baseline
    x = np.sort(rng.uniform(-2, 2, 30))
    yv = 0.15 * x + 0.35 * np.maximum(x, 0) ** 2 + rng.normal(0, 0.12, 30)
    xq = 1.4
    wq = np.exp(-np.abs(x - xq) / 0.7)
    a2.scatter(x, yv, s=60 + 500 * wq, color=ORANGE, alpha=0.55, edgecolors="none")
    b_eq = np.polyfit(x, yv, 1)
    b_w = np.polyfit(x, yv, 1, w=np.sqrt(wq))
    xs = np.linspace(-2.1, 2.1, 50)
    a2.plot(xs, np.polyval(b_eq, xs), color=GREY, lw=3, label="Rhaister-O: one slope for all lines")
    a2.plot(xs, np.polyval(b_w, xs), color=BLUE, lw=3.5, label="ours: slope refitted for the new line")
    a2.axvline(xq, color=INK, ls="--", lw=1.5)
    a2.text(xq + 0.05, a2.get_ylim()[0] + 0.05, "new line", fontsize=15, color=INK)
    a2.set_title("One drug, one gene", loc="left", fontsize=17, color=INK)
    a2.set_xlabel("baseline expression x", fontsize=15)
    a2.set_ylabel("drug response y", fontsize=15)
    a2.set_xticks([])
    a2.set_yticks([])
    a2.legend(loc="upper left", frameon=False, fontsize=14)
    fig.text(0.99, 0.01, "illustration with toy data", ha="right", fontsize=12, color=INK2)
    fig.tight_layout(w_pad=3)
    save(fig, "similarity_cartoon.png")


def reliability_weight():
    # q values: report 09_25 §7; weights relative to the median line, alpha = 20 (10_02 §2.4)
    lines = [("NCI-H596", 0.6537), ("NCI-H2122", 0.7238), ("SW 1271", 0.9488), ("H4", 0.9620),
             ("median line", 0.9755)]
    qm = 0.9755
    qq = np.linspace(0.6, 1.0, 200)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.plot(qq, (qq / qm) ** 20, color=BLUE, lw=3.5)
    for name, q in lines:
        w = (q / qm) ** 20
        col = ORANGE if q < 0.8 else BLUE
        ax.plot(q, w, "o", color=col, ms=13, zorder=3)
        off = {"NCI-H596": (0, 22), "NCI-H2122": (0, 22), "SW 1271": (-95, 8), "H4": (-30, 18),
               "median line": (-160, 6)}[name]
        ax.annotate(f"{name}\nweight {w:.4f}" if w < 0.01 else f"{name}", (q, w), off,
                    textcoords="offset points", fontsize=14, color=col)
    ax.set_xlabel("Reliability q: agreement of a line's own controls across 14 plates", fontsize=15)
    ax.set_ylabel("Weight in the fit  (q / q_median)^20", fontsize=15)
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(-0.05, 1.35)
    ax.grid(True, color=GRID)
    ax.set_axisbelow(True)
    save(fig, "reliability_weight.png")


def mapping_schematic():
    fig, ax = plt.subplots(figsize=(16, 7.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    def bx(x0, y0, w, h, title, sub, edge, fill, ts=19, ss=15.5):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                    lw=2.2, edgecolor=edge, facecolor=fill))
        ax.text(x0 + w / 2, y0 + h * 0.66, title, ha="center", va="center", fontsize=ts, fontweight="bold",
                color=INK)
        ax.text(x0 + w / 2, y0 + h * 0.3, sub, ha="center", va="center", fontsize=ss, color=INK2)

    def ar(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=24, lw=2.6, color=INK2))

    for y0, label, col in [(4.2, "Direct gene match", GOOD), (0.4, "CCA shared space", ORANGE)]:
        ax.text(0.1, y0 + 2.75, label, fontsize=22, fontweight="bold", color=col)
    # direct
    bx(0.1, 4.2, 3.0, 2.2, "DepMap", "new line's profile\n19,193 genes", "#c4ccd5", "#ffffff")
    ar(3.1, 3.7, 5.3)
    bx(3.7, 4.2, 6.2, 2.2, "One line per gene",
       "Tahoe(g) = a(g) + b(g) · DepMap(g)\nfitted on the reference lines", GOOD, "#e3eef0", ts=20, ss=16.5)
    ar(9.9, 10.5, 5.3)
    bx(10.5, 4.2, 2.6, 2.2, "Tahoe", "pseudo profile\n2,000 genes", "#c4ccd5", "#ffffff")
    ar(13.1, 13.6, 5.3)
    bx(13.6, 4.2, 2.3, 2.2, "Rhaister-O", "unchanged", BLUE, "#e6f0f6")
    # CCA
    bx(0.1, 0.4, 3.0, 2.2, "DepMap", "new line's profile\n19,193 genes", "#c4ccd5", "#ffffff")
    ar(3.1, 3.7, 1.5)
    bx(3.7, 0.4, 2.9, 2.2, "Project", "onto 10 shared axes\n(fit on 45 lines)", ORANGE, "#fbeee6")
    ar(6.6, 7.1, 1.5)
    bx(7.1, 0.4, 2.8, 2.2, "Map back", "to 2,000\nTahoe genes", ORANGE, "#fbeee6")
    ar(9.9, 10.5, 1.5)
    bx(10.5, 0.4, 2.6, 2.2, "Tahoe", "pseudo profile\n2,000 genes", "#c4ccd5", "#ffffff")
    ar(13.1, 13.6, 1.5)
    bx(13.6, 0.4, 2.3, 2.2, "Rhaister-O", "unchanged", BLUE, "#e6f0f6")
    ax.text(15.9, 3.85, "keeps each gene's identity: 2 numbers per gene", ha="right", fontsize=15, color=GOOD)
    ax.text(15.9, 0.05, "must relearn which gene is which from 45 lines", ha="right", fontsize=15, color=ORANGE)
    save(fig, "mapping_schematic.png")


def bridge_comparison_v2():
    rows = [
        ("Real Tahoe profile (upper reference)", 0.539, 0.483, 0.583, GREY),
        ("③ DepMap → direct gene match", 0.406, 0.361, 0.442, GOOD),
        ("③ DepMap → CCA shared space", 0.249, 0.195, 0.304, ORANGE),
        ("DepMap → response directly (ridge)", 0.260, 0.207, 0.315, "#b9c3cc"),
        ("Another line's profile (control)", 0.056, 0.012, 0.102, "#dbe1e8"),
        ("Average profile (control)", 0.001, -0.008, 0.010, "#dbe1e8"),
    ]  # report 10_09 §9, §11; 10_02 §2.7
    fig, ax = plt.subplots(figsize=(13, 6.2))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, lo, hi, col) in zip(y, rows):
        ax.barh(yy, v, color=col, height=0.64)
        ax.plot([lo, hi], [yy, yy], color=INK, lw=1.6)
        ax.text(hi + 0.008, yy, f"{v:.2f}", va="center", fontsize=17,
                color=GOOD if "direct" in lab else INK2, fontweight="bold" if "direct" in lab else "normal")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=16)
    for t in ax.get_yticklabels():
        if "direct" in t.get_text():
            t.set_color(GOOD)
            t.set_fontweight("bold")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-0.02, 0.66)
    ax.set_xlabel("Drug-adjusted Pearson through Rhaister-O, 23 held-out lines", fontsize=15)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    better_arrow(ax, 0.72, 0.97, 0.06)
    save(fig, "bridge_comparison.png")


def ladder_v2():
    rows = [
        ("①+② both weights", 0.0267, "ours"), ("② reliability weights", 0.0150, "ours"),
        ("① similarity weights", 0.0077, "ours"),
        ("① slope along cell-state PCs", 0.0011, "alt"), ("① quadratic term", -0.0003, "alt"),
        ("① cross-gene low-rank term", -0.0145, "alt"), ("① gene-network prior", 0.0009, "alt"),
        ("① similarity by tissue", -0.0344, "alt"), ("+ DepMap features on top", 0.0004, "alt"),
    ]  # fresh lines for ours and tissue (10_02 §3.7, §4.1); others benchmark splits (see modification_ladder)
    fig, ax = plt.subplots(figsize=(13, 7))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, g) in zip(y, rows):
        ax.barh(yy, v, color=BLUE if g == "ours" else ORANGE, height=0.64)
        ax.text(v + (0.0006 if v >= 0 else -0.0006), yy, f"{v:+.4f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=14, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=16)
    for t, r in zip(ax.get_yticklabels(), rows):
        if r[2] == "ours":
            t.set_color(BLUE)
            t.set_fontweight("bold")
    ax.axvline(0, color=INK2, lw=1)
    ax.axvline(0.01, color=GOLD, lw=2, ls="--")
    ax.text(0.0103, -0.5, "gate +0.01", color=GOLD, fontsize=14)
    ax.set_ylim(-0.7, y[0] + 1.1)
    ax.set_xlim(-0.043, 0.034)
    ax.set_xlabel("Gain over Rhaister-O (drug-adjusted, Tahoe)", fontsize=15)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    better_arrow(ax, 0.86, 0.99, 0.965)
    save(fig, "modification_ladder.png")


def context_slopes_v2():
    context_slopes()  # redraw, then add arrows on the saved figure's axes is not possible; rebuild inline
    arms = ["① Similarity\nweights", "② Reliability\nweights", "①+② Both"]
    first_t, first_e = [0.0134, 0.0175, 0.0316], [0.0059, 0.0022, 0.0091]
    tahoe = [(0.0077, -0.0009, 0.0151), (0.0150, 0.0126, 0.0174), (0.0267, 0.0223, 0.0311)]
    eb = [(0.0060, 0.0029, 0.0090), (0.0011, -0.0002, 0.0024), (0.0080, 0.0045, 0.0115)]
    y = np.arange(len(arms))[::-1] * 1.0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.4), sharey=True)
    for ax, title in [(a1, "First pass: benchmark splits, tuned"),
                      (a2, "Confirmation: never-scored lines, fixed (95% CI)")]:
        ax.axvline(0, color=INK2, lw=1)
        ax.axvline(0.01, color=GOLD, lw=2, ls="--")
        ax.text(0.0103, y[0] + 0.55, "gate +0.01", color=GOLD, fontsize=13)
        ax.set_xlim(-0.004, 0.040)
        ax.set_ylim(-0.75, y[0] + 0.8)
        ax.set_title(title, loc="left", color=INK, fontsize=17)
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.set_xlabel("Gain over Rhaister-O (drug-adjusted)")
        better_arrow(ax, 0.72, 0.97, 0.04)
    for vals, off, col in [(first_t, 0.17, BLUE), (first_e, -0.17, ORANGE)]:
        for yy, v in zip(y + off, vals):
            a1.plot(v, yy, "o", color=col, ms=11)
            a1.text(v + 0.0011, yy, f"{v:+.4f}", va="center", fontsize=13, color=col)
    for off, data, col in [(0.17, tahoe, BLUE), (-0.17, eb, ORANGE)]:
        for yy, (m, lo, hi) in zip(y + off, data):
            a2.plot([lo, hi], [yy, yy], color=col, lw=3, solid_capstyle="round")
            a2.plot(m, yy, "o", color=col, ms=11)
            a2.text(hi + 0.0008, yy, f"{m:+.4f}", va="center", fontsize=13, color=col)
    a1.set_yticks(y, arms, fontsize=16)
    a1.plot([], [], "o", color=BLUE, ms=10, label="Tahoe")
    a1.plot([], [], "o", color=ORANGE, ms=10, label="Emerald Bay (second atlas)")
    a1.legend(loc="upper center", bbox_to_anchor=(1.05, -0.14), ncol=2, frameon=False)
    fig.tight_layout(w_pad=2)
    save(fig, "context_slopes.png")


def mkl_tiers_diagram():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    tiers = [("Tier 1", "Chemical structure", "#7fb3d1"), ("Tier 2", "Prior knowledge", "#5d8fb3"),
             ("Tier 3", "Cell images", "#2f6f99"), ("Tier 4", "Measured response\nin other cell lines", BLUE)]
    for i, (t, name, col) in enumerate(tiers):
        y0 = 6.3 - i * 1.85
        ax.add_patch(FancyBboxPatch((0.2, y0), 7.0, 1.5, boxstyle="round,pad=0,rounding_size=0.15",
                                    lw=0, facecolor=col))
        ax.text(0.5, y0 + 0.75, t, fontsize=19, color="#ffffff", va="center", fontweight="bold")
        ax.text(2.0, y0 + 0.75, name, fontsize=22, color="#ffffff", va="center", fontweight="bold")
        ax.add_patch(FancyArrowPatch((7.2, y0 + 0.75), (8.0, 4.0), arrowstyle="-|>", mutation_scale=22,
                                     lw=2.2, color=INK2))
    ax.add_patch(FancyBboxPatch((8.0, 2.9), 3.8, 2.2, boxstyle="round,pad=0,rounding_size=0.15",
                                lw=2.4, edgecolor=INK2, facecolor="#ffffff"))
    ax.text(9.9, 4.4, "Drug–drug\nsimilarity", ha="center", va="center", fontsize=21, fontweight="bold", color=INK)
    ax.text(9.9, 3.35, "one per tier, summed", ha="center", va="center", fontsize=15, color=INK2)
    ax.add_patch(FancyArrowPatch((11.8, 4.0), (12.2, 4.0), arrowstyle="-|>", mutation_scale=24, lw=2.6,
                                 color=INK2))
    ax.add_patch(FancyBboxPatch((12.2, 2.6), 3.6, 2.8, boxstyle="round,pad=0,rounding_size=0.15",
                                lw=3, edgecolor=GOOD, facecolor="#e3eef0"))
    ax.text(14.0, 4.6, "New drug", ha="center", va="center", fontsize=22, fontweight="bold", color=INK)
    ax.text(14.0, 3.55, "= weighted average of\nresponses to similar\nknown drugs", ha="center", va="center",
            fontsize=16, color=INK2)
    ax.text(0.2, 0.15, "Tiers are added one at a time; each knowledge source is also tested against a scrambled copy.",
            fontsize=15, color=INK2)
    save(fig, "mkl_tiers.png")


def mkl_tier_gains():
    # report 10_09 §7, plate-map split, P2-24
    rows = [("+ Chemical structure", 0.027, 0.024, 0.031), ("+ Prior knowledge", 0.013, 0.008, 0.017),
            ("+ Cell images", 0.052, 0.045, 0.060), ("+ Measured response\n   in other lines", 0.131, 0.122, 0.141)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, lo, hi), col in zip(y, rows, ["#7fb3d1", "#5d8fb3", "#2f6f99", BLUE]):
        a1.barh(yy, v, color=col, height=0.62)
        a1.plot([lo, hi], [yy, yy], color=INK, lw=1.6)
        a1.text(hi + 0.003, yy, f"+{v:.3f}", va="center", fontsize=20, fontweight="bold", color=INK)
    a1.set_yticks(y, [r[0] for r in rows], fontsize=18)
    a1.set_xlim(0, 0.17)
    a1.set_xlabel("Gain in Pearson when the tier is added (new drugs)", fontsize=15)
    a1.set_title("Each tier, added in turn", loc="left", fontsize=18, color=INK)
    a1.xaxis.grid(True, color=GRID)
    a1.set_axisbelow(True)
    better_arrow(a1, 0.72, 0.97, 0.9, "more useful")
    inner = [("Drug targets\nvs permuted targets", 0.035, 0.029, 0.042),
             ("Protein network\nvs rewired network", 0.000, -0.003, 0.003)]
    y2 = np.arange(len(inner))[::-1]
    for yy, (lab, v, lo, hi) in zip(y2, inner):
        a2.barh(yy, v, color=GOOD if v > 0.01 else GREY, height=0.55)
        a2.plot([lo, hi], [yy, yy], color=INK, lw=1.6)
        a2.text(max(hi, v) + 0.002, yy, f"+{v:.3f}", va="center", fontsize=18, fontweight="bold", color=INK)
    a2.set_yticks(y2, [r[0] for r in inner], fontsize=16)
    a2.set_xlim(-0.005, 0.06)
    a2.set_xlabel("Gain over the scrambled copy", fontsize=15)
    a2.set_title("Inside prior knowledge", loc="left", fontsize=18, color=INK)
    a2.xaxis.grid(True, color=GRID)
    a2.set_axisbelow(True)
    fig.tight_layout(w_pad=3)
    save(fig, "mkl_tier_gains.png")


def revision5():
    baseline_simple_v2()
    slope_alternatives()
    similarity_cartoon()
    reliability_weight()
    mapping_schematic()
    bridge_comparison_v2()
    ladder_v2()
    context_slopes_v2()
    mkl_tiers_diagram()
    mkl_tier_gains()


# Revision 6 (2026-10-12): trimmed figures.
def quality_story_v2():
    fig, (a, c) = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1, 1.1]})
    names = ["NCI-H596", "NCI-H2122", "SW 1271", "H4", "median of\nother lines"]
    q = [0.6537, 0.7238, 0.9488, 0.9620, 0.9755]  # report 09_25 §7
    yy = np.arange(len(names))[::-1]
    a.barh(yy, q, color=[ORANGE, ORANGE, GREY, GREY, BLUE], height=0.6)
    for y0, v in zip(yy, q):
        a.text(v + 0.004, y0, f"{v:.3f}", va="center", fontsize=15, color=INK2)
    a.set_yticks(yy, names, fontsize=16)
    a.set_xlim(0.6, 1.04)
    a.set_xlabel("Reliability q (agreement of the line's own controls)", fontsize=15)
    a.set_title("(a) Two reference lines do not reproduce", loc="left", color=INK, fontsize=18)
    a.xaxis.grid(True, color=GRID)
    a.set_axisbelow(True)
    m = [5, 10, 15, 20, 25, 30, 35, 40, 45]  # report 09_25 §6
    rand = [0.157076, 0.275054, 0.363027, 0.411351, 0.462853, 0.487762, 0.505906, 0.524332, 0.540245]
    best = [0.2033, 0.3222, 0.4154, 0.4842, 0.5116, 0.5268, 0.5425, 0.5544]
    worst = [-0.0414, 0.1292, 0.2655, 0.3559, 0.4258, 0.4761, 0.5030, 0.5240]
    c.plot(m, rand, "o-", color=GREY, lw=3, ms=7, label="random")
    c.plot(m[:-1], best, "o-", color=BLUE, lw=3.5, ms=7, label="most reliable")
    c.plot(m[:-1], worst, "o-", color=ORANGE, lw=3, ms=7, label="least reliable")
    c.annotate("best 40 = 0.554 > all 45 = 0.540", (40, 0.5544), (12, 0.64), color=BLUE, fontsize=15,
               arrowprops=dict(arrowstyle="-", color=BLUE))
    c.axhline(0, color=INK2, lw=1)
    c.set_ylim(-0.08, 0.72)
    c.set_xlabel("Reference lines used to fit", fontsize=15)
    c.set_ylabel("Drug-adjusted Pearson", fontsize=15)
    c.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), frameon=False, fontsize=15)
    c.grid(True, color=GRID)
    c.set_axisbelow(True)
    c.set_title("(b) Fewer but reliable lines beat all lines", loc="left", color=INK, fontsize=18)
    fig.tight_layout(w_pad=3)
    save(fig, "quality_story.png")


def confirm_only():
    arms = ["① Similarity\nweights", "② Reliability\nweights", "①+② Both"]
    tahoe = [(0.0077, -0.0009, 0.0151), (0.0150, 0.0126, 0.0174), (0.0267, 0.0223, 0.0311)]  # 10_02 §3.7
    eb = [(0.0060, 0.0029, 0.0090), (0.0011, -0.0002, 0.0024), (0.0080, 0.0045, 0.0115)]
    y = np.arange(len(arms))[::-1] * 1.0
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.axvline(0, color=INK2, lw=1)
    ax.axvline(0.01, color=GOLD, lw=2, ls="--")
    ax.text(0.0103, y[0] + 0.55, "gate +0.01", color=GOLD, fontsize=14)
    for off, data, col, lab in [(0.17, tahoe, BLUE, "Tahoe, 25 never-scored lines"),
                                (-0.17, eb, ORANGE, "Emerald Bay, 26 never-scored lines")]:
        for yy, (mm, lo, hi) in zip(y + off, data):
            ax.plot([lo, hi], [yy, yy], color=col, lw=3.5, solid_capstyle="round")
            ax.plot(mm, yy, "o", color=col, ms=13)
            ax.text(hi + 0.0008, yy, f"{mm:+.4f}", va="center", fontsize=16, color=col, fontweight="bold")
        ax.plot([], [], "o-", color=col, lw=3, ms=10, label=lab)
    ax.set_yticks(y, arms, fontsize=18)
    ax.set_xlim(-0.004, 0.038)
    ax.set_ylim(-0.75, y[0] + 0.8)
    ax.set_xlabel("Gain over Rhaister-O (drug-adjusted), 95% CI; settings fixed in advance", fontsize=15)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.15), ncol=2, frameon=False, fontsize=15)
    better_arrow(ax, 0.75, 0.97, 0.04)
    save(fig, "context_slopes_confirm.png")


def revision6():
    quality_story_v2()
    confirm_only()


# Revision 7 (2026-10-12): worked example for upgrade 1 (toy numbers, matching the slide text).
def similarity_example():
    names = ["A", "B", "C", "D", "E", "F"]
    r = np.array([0.6, 0.4, 0.2, 0.0, -0.2, -0.4])      # correlation with the new line's untreated profile
    hs = 0.5                                              # h * s (illustrative s = 1, h = 0.5)
    w = np.exp(-(1 - r) / hs)
    x = np.array([1.6, 1.2, 0.6, -0.4, -1.0, -1.6])      # baseline of gene g in each reference line
    y = np.array([-1.8, -1.0, -0.3, 0.0, 0.05, 0.1])     # response of gene g to drug p
    xs = 1.9                                              # new line's baseline of gene g

    def fit(wt):
        mx, mu = (wt * x).sum() / wt.sum(), (wt * y).sum() / wt.sum()
        g = (wt * (x - mx) * (y - mu)).sum() / (wt * (x - mx) ** 2).sum()
        return lambda v: mu + g * (v - mx), g

    f_eq, g_eq = fit(np.ones(6))
    f_w, g_w = fit(w)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.3), gridspec_kw={"width_ratios": [1, 1.25]})
    rr = np.linspace(-0.5, 1.0, 100)
    a1.plot(rr, np.exp(-(1 - rr) / hs), color=BLUE, lw=3)
    for n, ri, wi in zip(names, r, w):
        a1.plot(ri, wi, "o", color=ORANGE, ms=13, zorder=3)
        a1.annotate(n, (ri, wi), (-8, 12), textcoords="offset points", fontsize=16, color=INK, fontweight="bold",
                    ha="right")
    table = "line   r     d     w\n" + "\n".join(f" {n}   {ri:+.1f}   {1 - ri:.1f}   {wi:.2f}"
                                                  for n, ri, wi in zip(names, r, w))
    a1.text(0.03, 0.97, table, transform=a1.transAxes, va="top", ha="left", fontsize=15, family="monospace",
            color=INK, bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec=RULE))
    a1.set_xlabel("correlation r with the new line's untreated profile", fontsize=15)
    a1.set_ylabel("weight w = exp(−(1 − r) / 0.5)", fontsize=15)
    a1.set_title("Step 1: similarity → weight", loc="left", fontsize=18, color=INK)
    a1.set_xlim(-0.55, 1.02)
    a1.set_ylim(0, 1.08)
    a1.grid(True, color=GRID)
    a1.set_axisbelow(True)
    xx = np.linspace(-2.0, 2.2, 50)
    a2.plot(xx, f_eq(xx), color=GREY, lw=3, label=f"Rhaister-O, all equal: slope {g_eq:.2f}")
    a2.plot(xx, f_w(xx), color=BLUE, lw=3.5, label=f"ours, weighted: slope {g_w:.2f}")
    a2.scatter(x, y, s=150 + 2200 * w, color=ORANGE, alpha=0.6, edgecolors="none", zorder=3)
    for n, xi, yi in zip(names, x, y):
        a2.annotate(n, (xi, yi), (-26, 4), textcoords="offset points", fontsize=16, color=INK, fontweight="bold")
    a2.axvline(xs, color=INK, ls="--", lw=1.6)
    a2.text(xs - 0.05, 0.75, "new line's\nbaseline x* = 1.9", ha="right", fontsize=14, color=INK)
    for f, col in [(f_eq, GREY), (f_w, BLUE)]:
        a2.plot(xs, f(xs), "s", color=col, ms=12, zorder=4)
        a2.annotate(f"ŷ = {f(xs):.2f}", (xs, f(xs)), (12, -5), textcoords="offset points", fontsize=16, color=col,
                    fontweight="bold")
    a2.set_xlabel("baseline expression x of gene g", fontsize=15)
    a2.set_ylabel("response y of gene g to drug p", fontsize=15)
    a2.set_title("Step 2: weighted fit for one drug and one gene", loc="left", fontsize=18, color=INK)
    a2.set_xlim(-2.1, 2.75)
    a2.set_ylim(-2.4, 1.1)
    a2.grid(True, color=GRID)
    a2.set_axisbelow(True)
    a2.legend(loc="lower left", frameon=False, fontsize=14)
    fig.text(0.99, 0.005, "toy numbers for illustration; dot size = weight", ha="right", fontsize=12, color=INK2)
    fig.tight_layout(w_pad=3)
    save(fig, "similarity_cartoon.png")


# Revision 8 (2026-10-12): tier diagram with each tier's data listed, and how tiers enter the model.
# Views and weighting as coded in scripts/run_l1000_mkl.py (docstring, TIERS, centered_alignment).
def mkl_tiers_detailed():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    tiers = [("Tier 1", "Chemical structure", "#7fb3d1",
              "Morgan fingerprint of\nthe molecule's structure", "Tanimoto similarity"),
             ("Tier 2", "Prior knowledge", "#5d8fb3",
              "Drug targets\nmechanism of action (MOA)\ntargets' degree and\nneighbours in STRING", "Tanimoto · RBF · cosine"),
             ("Tier 3", "Cell images", "#2f6f99",
              "Cell Painting morphology\nprofile at 10 µM", "cosine similarity"),
             ("Tier 4", "Measured response", BLUE,
              "The drug's L1000 response\nin the other cell lines\n(never the test line)", "response correlation")]
    w, gap, x0 = 3.75, 0.2, 0.1
    for i, (t, name, col, body, sim) in enumerate(tiers):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 5.6), w, 2.25, boxstyle="round,pad=0,rounding_size=0.1",
                                    lw=1.4, edgecolor=RULE, facecolor="#ffffff"))
        ax.add_patch(FancyBboxPatch((x, 7.75), w, 1.15, boxstyle="round,pad=0,rounding_size=0.1",
                                    lw=0, facecolor=col))
        ax.text(x + 0.2, 8.6, t, fontsize=15, color="#ffffff", va="center", fontweight="bold")
        ax.text(x + 0.2, 8.07, name, fontsize=17, color="#ffffff", va="center", fontweight="bold")
        ax.text(x + 0.2, 7.5, body, fontsize=14, color=INK2, va="top", linespacing=1.35)
        ax.text(x + 0.2, 5.82, "→ " + sim, fontsize=13, color=col if i else "#3b7fa6", va="bottom",
                fontstyle="italic")
    # cumulative tiers: model Tk uses every tier up to k
    for k in range(4):
        y = 5.05 - k * 0.36
        right = x0 + (k + 1) * w + k * gap
        ax.add_patch(FancyBboxPatch((x0, y), right - x0, 0.26, boxstyle="round,pad=0,rounding_size=0.06",
                                    lw=0, facecolor=tiers[k][2]))
        ax.text(x0 + 0.12, y + 0.13, f"model T{k + 1}" + (" = tier 1" if k == 0 else f" = tiers 1–{k + 1}"),
                fontsize=13, color="#ffffff", va="center", fontweight="bold")
    ax.text(15.7, 5.18, "tiers are cumulative", fontsize=14, color=INK2, ha="right", va="center", fontstyle="italic")
    ax.plot([0.1, 15.7], [3.62, 3.62], color=RULE, lw=1.2)
    ax.text(0.1, 3.32, "How a tier enters the model", fontsize=19, fontweight="bold", color=INK, va="center")
    steps = [("1 · Data → similarity",
              "each data type → a drug × drug\nsimilarity matrix $K_v$\n(a drug lacking that data: 0)"),
             ("2 · Sum within the tier",
              "$K = \\Sigma_v\\, w_v K_v$ over every data type\nup to this tier; $w_v$ = how well $K_v$\nmatches the training responses"),
             ("3 · Kernel ridge regression",
              "new drug = line's mean response\n+ $\\Sigma_i\\, \\alpha_i$ · K(new drug, known drug $i$)\nrefitted for every cell line")]
    sw, sgap = 4.9, 0.45
    for i, (head, body) in enumerate(steps):
        x = 0.1 + i * (sw + sgap)
        ax.add_patch(FancyBboxPatch((x, 0.75), sw, 2.15, boxstyle="round,pad=0,rounding_size=0.1",
                                    lw=1.4, edgecolor=RULE, facecolor="#ffffff"))
        ax.text(x + 0.2, 2.6, head, fontsize=15, fontweight="bold", color=INK, va="center")
        ax.text(x + 0.2, 2.25, body, fontsize=13.5, color=INK2, va="top", linespacing=1.45)
        if i < 2:
            ax.add_patch(FancyArrowPatch((x + sw + 0.05, 1.82), (x + sw + sgap - 0.05, 1.82),
                                         arrowstyle="-|>", mutation_scale=22, lw=2.2, color=INK2))
    ax.text(0.1, 0.32, "Controls: the knowledge matrices are also run with targets permuted and the protein network "
            "rewired.", fontsize=14, color=INK2, va="center")
    save(fig, "mkl_tiers_detailed.png")


if __name__ == "__main__":
    baseline_simple()
    same_line_similarity()
    baseline_decomposition()
    context_slopes()
    modification_ladder()
    quality_story()
    mkl_results()
    bridge_comparison()
    bridge_per_line()
    revision5()
    revision6()
    similarity_example()
