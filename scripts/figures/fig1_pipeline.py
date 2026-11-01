"""Fig. 1 of the UnDetect paper: the pipeline, left to right in two lanes. Dossier: paper/figures/fig1_pipeline.md.

Top lane: DMSO well -> control expression -> backbone (a neural network on the drug-blind benchmark; on Level-5
and Tahoe data, which have no control input, the training line mean) -> context. Bottom lane: compound -> its
views -> kernels -> fusion -> normalization -> kernel transfer (fed by the residuals of profiled compounds) ->
compound effect -> interaction (the context rescales it gene by gene). The lanes add up to the response. Names only in the artwork; the caption explains.
Schematic: matrices, strips and icons are drawn from a fixed toy seed; no result is plotted.
Run: /nfs-stor/yuqi.lei/Therapeutic/envs/gnn-gpu/bin/python -I scripts/figures/fig1_pipeline.py [outdir]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import CTX, CTX_LIGHT, INK, INK2, LINE, OURS, OURS_LIGHT, PANEL, VIEW, plt, save  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle, RegularPolygon, Wedge  # noqa: E402

SEED = 7
VIEWS = [("Chemical structure", "chem"), ("Targets", "tgt"), ("Mechanism of action", "moa"),
         ("Cell Painting", "morph")]
MISSING = {"chem": (), "tgt": (1, 4), "moa": (4,), "morph": (1, 2, 4)}   # toy compounds lacking each view
rng = np.random.default_rng(SEED)


def box(ax, x, y, w, h, fc, ec, r=0.06, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=0.8,
                                zorder=z))


def label(ax, x, y, s, size=9, color=INK, ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=6)


def arrow(ax, x1, y1, x2, y2, color=INK2, lw=1.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=3))


def strip(ax, x, y, w, h, vals, rgb):
    cw = w / len(vals); v = np.abs(vals) / np.abs(vals).max()
    for i, a in enumerate(v):
        ax.add_patch(Rectangle((x + i * cw, y), cw, h, fc=rgb, alpha=0.15 + 0.85 * a, ec="none", zorder=4))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=LINE, lw=0.5, zorder=5))


def matrix(ax, x, y, s, M, rgb, missing=()):
    n = M.shape[0]; c = s / n
    for i in range(n):
        for j in range(n):
            cx, cy = x + j * c, y + s - (i + 1) * c
            if i in missing or j in missing:
                ax.add_patch(Rectangle((cx, cy), c, c, fc="white", ec=LINE, lw=0.3, hatch="////", zorder=4))
            else:
                ax.add_patch(Rectangle((cx, cy), c, c, fc=rgb, alpha=min(1.0, 0.08 + 0.92 * M[i, j]), ec="white",
                                       lw=0.3, zorder=4))
    ax.add_patch(Rectangle((x, y), s, s, fc="none", ec=INK2, lw=0.5, zorder=5))


def toy_kernel(n):
    F = rng.random((n, 3)); K = F @ F.T; d = np.sqrt(np.diag(K)); K = K / np.outer(d, d)
    K = (K - K.min()) / (1 - K.min()); np.fill_diagonal(K, 1.0)
    return K


# ---------------- icons
def well(ax, x, y, r):
    ax.add_patch(Circle((x, y), r, fc=PANEL, ec=INK2, lw=0.8, zorder=3))
    for dx, dy, a in ((-0.35, 0.3, 20), (0.3, 0.32, -30), (-0.3, -0.3, 60), (0.32, -0.25, 10), (0.0, 0.02, -60)):
        ax.add_patch(Ellipse((x + dx * r, y + dy * r), 0.42 * r, 0.30 * r, angle=a, fc=CTX_LIGHT, ec=CTX, lw=0.5,
                             zorder=4))
        ax.add_patch(Circle((x + dx * r, y + dy * r), 0.06 * r, fc=CTX, ec="none", zorder=5))


def network(ax, x, y, w, h, color):
    layers = [3, 4, 4, 3]
    xs = np.linspace(x + 0.12 * w, x + 0.88 * w, len(layers))
    pts = [[(xs[k], y + h * (0.5 + (i - (n - 1) / 2) * 0.22)) for i in range(n)] for k, n in enumerate(layers)]
    for k in range(len(layers) - 1):
        for p in pts[k]:
            for q in pts[k + 1]:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=LINE, lw=0.4, zorder=2)
    for layer in pts:
        for p in layer:
            ax.add_patch(Circle(p, 0.028, fc="white", ec=color, lw=0.8, zorder=3))


def molecule(ax, x, y, r, color, lw=1.0):
    ax.add_patch(RegularPolygon((x, y), 6, radius=r, orientation=np.pi / 6, fc="none", ec=color, lw=lw, zorder=4))
    ax.plot([x + r * np.cos(0), x + 1.7 * r, x + 2.3 * r], [y, y + 0.5 * r, y], color=color, lw=lw, zorder=4)
    ax.plot([x - r * 0.55, x - r * 0.55], [y - r * 0.45, y + r * 0.45], color=color, lw=lw * 0.8, zorder=4)


def view_icon(ax, key, x, y, s):
    c = VIEW[key]
    if key == "chem":
        molecule(ax, x - 0.15 * s, y, 0.32 * s, c, lw=0.9)
    elif key == "tgt":       # protein with a binding pocket
        ax.add_patch(Wedge((x, y), 0.45 * s, 35, 325, fc=c, ec="none", zorder=4))
    elif key == "moa":       # pathway: activation then inhibition
        p = [(x - 0.45 * s, y), (x, y), (x + 0.45 * s, y)]
        for q in p:
            ax.add_patch(Circle(q, 0.11 * s, fc=c, ec="none", zorder=5))
        ax.plot([p[0][0], p[2][0]], [y, y], color=c, lw=0.9, zorder=4)
        ax.plot([p[2][0] - 0.14 * s] * 2, [y - 0.16 * s, y + 0.16 * s], color=c, lw=1.0, zorder=4)
    else:                    # stained cell
        ax.add_patch(Rectangle((x - 0.45 * s, y - 0.45 * s), 0.9 * s, 0.9 * s, fc="#1e1a3a", ec="none", zorder=4))
        ax.add_patch(Ellipse((x, y), 0.7 * s, 0.5 * s, angle=20, fc=c, ec="none", alpha=0.8, zorder=5))
        ax.add_patch(Circle((x + 0.05 * s, y), 0.14 * s, fc="#5fb3ff", ec="none", zorder=6))


W, H = 6.5, 2.5
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
genes = 16
ctx_vals = rng.normal(0, 1, genes) * 1.6
cmp_vals = rng.normal(0, 1, genes)
SW, SH = 0.50, 0.15
TY, BY = 1.98, 0.74                 # lane centres
HEAD = 0.34                         # label height above a lane element
XS = 5.80                           # sum node
XI = 5.53                           # interaction node

# ---------------- context lane
well(ax, 0.32, TY, 0.20)
label(ax, 0.32, TY - 0.32, "DMSO", size=8.5, color=INK2)
arrow(ax, 0.55, TY, 0.68, TY)
strip(ax, 0.70, TY - SH / 2, SW, SH, ctx_vals, CTX)
label(ax, 0.95, TY + HEAD, "Control expression")
arrow(ax, 1.23, TY, 1.61, TY)
box(ax, 1.63, TY - 0.24, 0.80, 0.48, CTX_LIGHT, CTX)
network(ax, 1.63, TY - 0.24, 0.80, 0.48, CTX)
label(ax, 2.03, TY + HEAD, "Backbone")
arrow(ax, 2.45, TY, 2.62, TY)
strip(ax, 2.64, TY - SH / 2, SW, SH, ctx_vals, CTX)
label(ax, 2.89, TY + HEAD, "Context")
ax.plot([3.16, XS], [TY, TY], color=CTX, lw=1.0, zorder=2)

# ---------------- compound lane
molecule(ax, 0.26, BY, 0.11, INK2, lw=1.2)
label(ax, 0.36, BY + 0.24, "Compound", size=8.5, color=INK2)
n = 6; Ks = {}
for k, (name, key) in enumerate(VIEWS):
    yk = BY + 0.36 - k * 0.24
    ax.plot([0.55, 0.62], [BY, yk], color=LINE, lw=0.7, zorder=2)
    view_icon(ax, key, 0.72, yk, 0.17)
    label(ax, 0.84, yk, name, size=8, ha="left")
    Ks[key] = toy_kernel(n)
    matrix(ax, 2.02, yk - 0.09, 0.18, Ks[key], VIEW[key], MISSING[key])
    arrow(ax, 2.22, yk, 2.39, BY, color=LINE, lw=0.7)
label(ax, 2.11, BY + 0.64, "Kernels", size=8.5)

w = {"chem": 0.4, "tgt": 0.25, "moa": 0.15, "morph": 0.2}
Kf = 0
for key, Ki in Ks.items():
    Kz = Ki.copy(); Kz[list(MISSING[key]), :] = 0; Kz[:, list(MISSING[key])] = 0; Kf = Kf + w[key] * Kz
dg = np.sqrt(np.diag(Kf)); Kn = Kf / np.outer(dg, dg)
MS = 0.34
box(ax, 2.41, BY - 0.28, 0.54, 0.54, PANEL, LINE)
matrix(ax, 2.68 - MS / 2, BY - MS / 2 - 0.01, MS, Kf, OURS)
label(ax, 2.68, BY + 0.64, "Fusion", size=8.5)
arrow(ax, 2.97, BY, 3.13, BY)
box(ax, 3.15, BY - 0.28, 0.54, 0.54, OURS_LIGHT, OURS)
matrix(ax, 3.42 - MS / 2, BY - MS / 2 - 0.01, MS, Kn, OURS)
label(ax, 3.42, BY + 0.64, "Normalization", size=8.5)
arrow(ax, 3.71, BY, 3.84, BY)
box(ax, 3.86, BY - 0.28, 1.02, 0.56, OURS_LIGHT, OURS)
label(ax, 4.37, BY + 0.08, "Kernel transfer", size=8.5)
label(ax, 4.37, BY - 0.12, "shared + per line", size=8, color=INK2)
XR = 4.37                           # residuals of profiled compounds: response minus context
ax.add_patch(Circle((XR, TY), 0.03, fc=CTX, ec="none", zorder=4))
arrow(ax, XR, TY - 0.03, XR, BY + 0.30, color=CTX)
label(ax, XR + 0.07, (TY + BY + 0.28) / 2, "Residuals", size=8.5, color=INK2, ha="left")
arrow(ax, 4.90, BY, 4.96, BY)
strip(ax, 4.98, BY - SH / 2, 0.36, SH, cmp_vals, OURS)
label(ax, 5.16, BY + HEAD + 0.06, "Compound\neffect", size=8.5)
ax.plot([5.36, XI - 0.09], [BY, BY], color=OURS, lw=1.0, zorder=2)

# ---------------- interaction: the context rescales the compound effect gene by gene
ax.add_patch(Circle((XI, BY), 0.09, fc=OURS_LIGHT, ec=OURS, lw=0.9, zorder=4))
label(ax, XI, BY + 0.005, "\u00d7", size=10, color=OURS)
ax.add_patch(Circle((XI, TY), 0.03, fc=CTX, ec="none", zorder=4))
arrow(ax, XI, TY - 0.03, XI, BY + 0.09, color=CTX)
label(ax, XI, BY - 0.24, "Interaction", size=8.5)
ax.plot([XI + 0.09, XS - 0.10], [BY, BY], color=OURS, lw=1.0, zorder=2)

# ---------------- sum and response
ax.add_patch(Circle((XS, BY), 0.10, fc="white", ec=INK2, lw=0.9, zorder=4))
label(ax, XS, BY + 0.005, "+", size=11, color=INK2)
arrow(ax, XS, TY, XS, BY + 0.10, color=CTX)
arrow(ax, XS + 0.10, BY, XS + 0.18, BY)
strip(ax, XS + 0.20, BY - SH / 2, 0.42, SH, ctx_vals + 0.35 * cmp_vals, INK2)
label(ax, XS + 0.41, BY + HEAD + 0.06, "Response", size=8.5)

save(fig, "fig1_pipeline", sys.argv[1] if len(sys.argv) > 1 else None)
