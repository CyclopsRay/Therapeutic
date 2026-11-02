"""Fig. 1 v6 of the UnDetect paper: compound cards (user sketch and review, 2026-11-02). Dossier:
paper/figures/fig1_pipeline.md.

Left: profiled compound cards (daunorubicin, vorinostat, then an ellipsis for the other profiled compounds) and the
unprofiled query (mitoxantrone) at the bottom. Each card: structure on the left, four view sections on the right
(Morgan fingerprint, targets, mechanism, Cell Painting), each on its view colour. Middle: the view kernels as a deck
(structure in front). Right column, top to bottom: fusion + normalization (query = bottom row) -> kernel transfer
(shared / per line) -> compound effect -> interaction -> sum -> response. Bottom: cells of one (cell line, plate)
give the DMSO control and, for profiled compounds, the treated profile; residual = (treated - DMSO) - context.
Shading only, no numbers: kernel shades are the real values (fig1_cards_data.py); expression strips are a toy seed.
Run: /nfs-stor/yuqi.lei/Therapeutic/envs/gnn-gpu/bin/python -I scripts/figures/fig1_cards.py [SOURCE_DIR [OUTDIR]]
     [--parts]   (also writes each part on its own canvas as SVG into <OUTDIR>/fig1_parts/)
SOURCE_DIR and OUTDIR default to paper/figures; writes fig1_pipeline[_preview].{pdf,svg,png}.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import CTX, CTX_LIGHT, FIGDIR, INK, INK2, LINE, MUTED, OURS, OURS_LIGHT, PANEL, VIEW, plt, save  # noqa: E402
from matplotlib import colors as mcolors, patheffects  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

SEED = 7
QUERY = 1                                   # mitoxantrone, the unprofiled compound
TITLE = ["Daunorubicin", "Mitoxantrone", "Vorinostat"]
LETTER = {0: "D", 1: "M", 2: "V"}
ORDER = [0, 2, None, 1]                     # matrix order: D, V, other profiled compounds, M (query last)
VIEWS = [("chem", "Structure"), ("tgt", "Targets"), ("moa", "Mechanism"), ("morph", "Cell Painting")]
SW, SH = 0.36, 0.09                         # expression strip
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- primitives
def tint(c, a):
    r = np.array(mcolors.to_rgb(c))
    return tuple(1 - a * (1 - r))


def box(ax, x, y, w, h, fc, ec, r=0.05, lw=0.7, z=1, ls="-", **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z, **kw))


def label(ax, x, y, s, size=6.5, color=INK, ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=9, **kw)


def fit(ax, x, y, s, maxw, size, **kw):
    """Left-aligned label shrunk until it is at most maxw inches wide."""
    t = label(ax, x, y, s, size=size, ha="left", **kw)
    r = ax.figure.canvas.get_renderer()
    while t.get_window_extent(r).width / ax.figure.dpi > maxw and size > 3.5:
        size -= 0.2; t.set_fontsize(size)
    return t


def arrow(ax, x1, y1, x2, y2, color=INK2, lw=0.8, ms=7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=3))


def path(ax, pts, color=INK2, lw=0.8, halo=False):
    """Polyline with an arrow head on its last segment; halo = white casing so crossing lines read as a hop."""
    xs, ys = zip(*pts)
    pe = [patheffects.withStroke(linewidth=lw + 3, foreground="white")] if halo else None
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=2.5 if halo else 2, solid_joinstyle="miter",
            path_effects=pe)
    arrow(ax, xs[-2], ys[-2], xs[-1], ys[-1], color=color, lw=lw)


def dot(ax, x, y, color):
    ax.add_patch(Circle((x, y), 0.017, fc=color, ec="none", zorder=4))


def node(ax, x, y, sym, color=INK2, fc="white", r=0.07):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=color, lw=0.8, zorder=5))
    label(ax, x, y + 0.004, sym, size=8, color=color)


def dots(ax, x, y, orient="v", gap=0.035, r=0.009, color=MUTED):
    d = {"v": (0, 1), "h": (1, 0), "d": (1, -1)}[orient]
    for k in (-1, 0, 1):
        ax.add_patch(Circle((x + k * gap * d[0], y + k * gap * d[1]), r, fc=color, ec="none", zorder=7))


def strip(ax, x, y, w, h, vals, rgb, z=4):
    cw = w / len(vals); v = np.abs(vals) / np.abs(vals).max()
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="none", zorder=z))
    for i, a in enumerate(v):
        ax.add_patch(Rectangle((x + i * cw, y), cw, h, fc=rgb, alpha=0.12 + 0.88 * a, ec="none", zorder=z))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=LINE, lw=0.4, zorder=z + 0.5))


def deck(ax, x, y, vals, rgb, n=3, d=0.03):
    """n profiles (one per cell line x plate); the front strip's lower-left corner at (x, y)."""
    for k in reversed(range(n)):
        strip(ax, x + k * d, y + k * d, SW, SH, np.roll(vals, 3 * k) * (1 - 0.15 * k), rgb, z=4 + 2 * (n - k))


def matrix(ax, x, y, s, K, rgb, has=(True, True, True), hl_last=False, z=4):
    """4 x 4 kernel in ORDER (D, V, ..., M), shaded by value; (x, y) = lower-left."""
    c = s / 4; g = 0.03 * c / 0.16
    for r, i in enumerate(ORDER):
        for q, j in enumerate(ORDER):
            cx, cy = x + q * c, y + s - (r + 1) * c
            if i is None or j is None:
                ax.add_patch(Rectangle((cx, cy), c, c, fc="white", ec="none", zorder=z))
                dots(ax, cx + c / 2, cy + c / 2, "d" if i is None and j is None else "v" if i is None else "h",
                     gap=g, r=0.007)
            elif not (has[i] and has[j]):
                ax.add_patch(Rectangle((cx, cy), c, c, fc="white", ec=LINE, lw=0.3, hatch="//////", zorder=z))
            else:
                ax.add_patch(Rectangle((cx, cy), c, c, fc=rgb, alpha=0.06 + 0.94 * K[i][j], ec="white", lw=0.4,
                                       zorder=z))
    ax.add_patch(Rectangle((x, y), s, s, fc="none", ec=INK2, lw=0.5, zorder=z + 1))
    if hl_last:
        ax.add_patch(Rectangle((x, y), s, c, fc="none", ec=OURS, lw=1.3, zorder=z + 2))


def axis_labels(ax, x, y, s, size=5.5):
    c = s / 4
    for r, i in enumerate(ORDER):
        bold = "bold" if i == QUERY else None
        if i is None:
            dots(ax, x - 0.05, y + s - (r + 0.5) * c, "v", gap=0.022, r=0.006, color=INK2)
            dots(ax, x + (r + 0.5) * c, y + s + 0.06, "h", gap=0.022, r=0.006, color=INK2)
        else:
            label(ax, x - 0.035, y + s - (r + 0.5) * c, LETTER[i], size=size, color=INK2, ha="right",
                  fontweight=bold)
            label(ax, x + (r + 0.5) * c, y + s + 0.035, LETTER[i], size=size, color=INK2, va="bottom",
                  fontweight=bold)


# ---------------------------------------------------------------- icons
def well(ax, x, y, r):
    ax.add_patch(Circle((x, y), r, fc=PANEL, ec=INK2, lw=0.7, zorder=3))
    for dx, dy, a in ((-0.35, 0.3, 20), (0.3, 0.32, -30), (-0.3, -0.3, 60), (0.32, -0.25, 10), (0.0, 0.02, -60)):
        ax.add_patch(Ellipse((x + dx * r, y + dy * r), 0.42 * r, 0.30 * r, angle=a, fc=CTX_LIGHT, ec=CTX, lw=0.4,
                             zorder=4))
        ax.add_patch(Circle((x + dx * r, y + dy * r), 0.06 * r, fc=CTX, ec="none", zorder=5))


def network(ax, x, y, w, h, color):
    layers = [3, 4, 4, 3]
    xs = np.linspace(x + 0.12 * w, x + 0.88 * w, len(layers))
    pts = [[(xs[k], y + h * (0.5 + (i - (n - 1) / 2) * 0.22)) for i in range(n)] for k, n in enumerate(layers)]
    for k in range(len(layers) - 1):
        for p in pts[k]:
            for q in pts[k + 1]:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=LINE, lw=0.35, zorder=2)
    for layer in pts:
        for p in layer:
            ax.add_patch(Circle(p, 0.02, fc="white", ec=color, lw=0.7, zorder=3))


def syringe_plate(ax, x, y, color=INK2):
    """A multiwell plate (lower-left corner at (x, y)) with a syringe dispensing into one well."""
    pw, ph = 0.30, 0.12
    box(ax, x, y, pw, ph, PANEL, color, r=0.015, lw=0.6, z=5)
    for i in range(4):
        for j in range(2):
            ax.add_patch(Circle((x + 0.045 + i * 0.07, y + 0.035 + j * 0.05), 0.017, fc="white", ec=color, lw=0.4,
                                zorder=6))
    wx, wy = x + 0.045 + 2 * 0.07, y + 0.085                     # target well
    ax.add_patch(Circle((wx, wy), 0.017, fc=OURS_LIGHT, ec=color, lw=0.4, zorder=6))
    by = wy + 0.06                                              # needle, barrel with liquid, plunger
    ax.plot([wx, wx], [wy + 0.012, by], color=color, lw=0.6, zorder=7)
    ax.add_patch(Rectangle((wx - 0.022, by), 0.044, 0.11, fc="white", ec=color, lw=0.6, zorder=7))
    ax.add_patch(Rectangle((wx - 0.022, by), 0.044, 0.045, fc=OURS_LIGHT, ec="none", zorder=7.5))
    ax.plot([wx - 0.03, wx + 0.03], [by + 0.11, by + 0.11], color=color, lw=0.6, zorder=7)
    ax.plot([wx, wx], [by + 0.11, by + 0.15], color=color, lw=0.6, zorder=7)
    ax.plot([wx - 0.02, wx + 0.02], [by + 0.15, by + 0.15], color=color, lw=0.9, zorder=7)


def cell_icon(ax, x, y, s):
    ax.add_patch(Rectangle((x - 0.45 * s, y - 0.45 * s), 0.9 * s, 0.9 * s, fc="#1e1a3a", ec="none", zorder=6))
    ax.add_patch(Ellipse((x, y), 0.72 * s, 0.5 * s, angle=20, fc=VIEW["morph"], ec="none", alpha=0.9, zorder=7))
    ax.add_patch(Circle((x + 0.05 * s, y), 0.14 * s, fc="#5fb3ff", ec="none", zorder=8))


# ---------------------------------------------------------------- structure
def molecule(ax, st, x, y, w, h, color=INK2, size=3.8, lw=0.5):
    """Skeletal formula fitted in the box (x, y, w, h); st = {'atoms', 'bonds'} with unit bond length."""
    P = np.array([[a["x"], a["y"]] for a in st["atoms"]])
    span = P.max(0) - P.min(0)
    sc = min(w / span[0], h / span[1]) * 0.9
    P = (P - (P.max(0) + P.min(0)) / 2) * sc + np.array([x + w / 2, y + h / 2])
    nb = {i: [] for i in range(len(P))}
    for b in st["bonds"]:
        nb[b["i"]].append(b["j"]); nb[b["j"]].append(b["i"])
    off = 0.2 * sc
    for b in st["bonds"]:
        i, j = b["i"], b["j"]; p, q = P[i], P[j]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, zorder=6, solid_capstyle="round")
        if b["order"] != 2:
            continue
        d = q - p; nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        others = [P[k] for k in nb[i] + nb[j] if k not in (i, j)]
        if len(nb[i]) == 1 or len(nb[j]) == 1 or not others:      # exocyclic C=O
            ax.plot([p[0] + nrm[0] * off, q[0] + nrm[0] * off], [p[1] + nrm[1] * off, q[1] + nrm[1] * off],
                    color=color, lw=lw, zorder=6)
            continue
        side = np.sign(np.dot(np.mean(others, 0) - (p + q) / 2, nrm)) or 1.0
        a_, b_ = p + 0.18 * d + side * nrm * off, q - 0.18 * d + side * nrm * off
        ax.plot([a_[0], b_[0]], [a_[1], b_[1]], color=color, lw=lw, zorder=6)
    for k, (a, (px, py)) in enumerate(zip(st["atoms"], P)):
        if a["el"] == "C":
            continue
        hs = "" if a["h"] == 0 else "H" if a["h"] == 1 else f"H$_{a['h']}$"
        left = hs and len(nb[k]) == 1 and P[nb[k][0]][0] > px + 0.3 * sc    # bond leaves to the right: "HO"
        s, dx = (hs + a["el"], -size * 0.35) if left else (a["el"] + hs, size * 0.35 if hs else 0)
        ax.annotate(s, (px, py), xytext=(dx, 0), textcoords="offset points", fontsize=size, color=color,
                    ha="center", va="center", zorder=7, bbox=dict(boxstyle="square,pad=0.04", fc="white", ec="none"))


# ---------------------------------------------------------------- parts
CW, CH = 2.40, 0.70                          # card
SEC, SGAP = 0.112, 0.016                     # view section height and gap


def _short_targets(ts):
    hd = [t for t in ts if t.startswith("HDAC")]
    if len(hd) > 2:
        return ["HDAC" + ",".join(t[4:] for t in hd)] + [t for t in ts if not t.startswith("HDAC")]
    return ts


def _join_moa(ms, suf=" inhibitor"):
    """['Topoisomerase inhibitor', 'RNA synthesis inhibitor'] -> 'Topoisomerase, RNA synthesis inhibitor'."""
    if len(ms) > 1 and all(m.endswith(suf) for m in ms):
        return ", ".join(m[:-len(suf)] for m in ms) + suf
    return ", ".join(ms)


def card(ax, x, top, k, S):
    q = k == QUERY
    box(ax, x, top - CH, CW, CH, "white", OURS if q else LINE, lw=1.1 if q else 0.7, z=1)
    label(ax, x + 0.08, top - 0.085, TITLE[k], size=7, ha="left", fontweight="bold")
    label(ax, x + CW - 0.08, top - 0.085, "unprofiled" if q else "profiled", size=5.8, color=OURS if q else MUTED,
          ha="right", style="italic")
    molecule(ax, S["structures"][k], x + 0.06, top - CH + 0.04, 0.80, CH - 0.20)
    sx = x + 0.92; sw = CW - 0.92 - 0.07
    for r, (key, _) in enumerate(VIEWS):
        sy = top - 0.17 - r * (SEC + SGAP) - SEC; yc = sy + SEC / 2
        has = S["has"][key][k]
        if has:
            box(ax, sx, sy, sw, SEC, tint(VIEW[key], 0.30), "none", r=0.02, z=2)
        else:
            box(ax, sx, sy, sw, SEC, "white", LINE, r=0.02, lw=0.4, z=2, hatch="//////")
        tx = sx + 0.05
        if key == "chem":                    # Morgan fingerprint: the real on-bits as a barcode
            fit(ax, tx, yc, "Morgan fingerprint", 0.70, 5.2)
            bx0, bw = sx + 0.80, sw - 0.85
            ax.add_patch(Rectangle((bx0, sy + 0.022), bw, SEC - 0.044, fc="white", ec="none", zorder=3))
            for b in S["fp_bits"][k]:
                xb = bx0 + bw * b / 2047
                ax.plot([xb, xb], [sy + 0.028, sy + SEC - 0.028], color=VIEW["chem"], lw=0.45, zorder=4)
        elif key == "tgt":
            fit(ax, tx, yc, ", ".join(_short_targets(S["target"][k])), sw - 0.08, 5.2)
        elif key == "moa":
            fit(ax, tx, yc, _join_moa(S["moa"][k][::-1]), sw - 0.08, 5.2)
        elif has:
            cell_icon(ax, tx + 0.04, yc, 0.085)
            fit(ax, tx + 0.11, yc, "Cell Painting profile", sw - 0.2, 5.2)
        else:
            fit(ax, tx, yc, "no Cell Painting profile", sw - 0.08, 5.2, color=MUTED,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))


def part_cards(ax, x, top, S, gap=0.06, ell=0.20):
    """Profiled cards, an ellipsis, then the query card. Returns right-edge centres and the bottom."""
    ys, ends = top, []
    for k in (0, 2):
        card(ax, x, ys, k, S); ends.append((x + CW, ys - CH / 2)); ys -= CH + gap
    dots(ax, x + CW / 2, ys - ell / 2 + gap / 2, "v", gap=0.04, r=0.012)
    ends.append((x + CW, ys - ell / 2 + gap / 2))
    ys -= ell
    card(ax, x, ys, QUERY, S); ends.append((x + CW, ys - CH / 2))
    return ends, ys - CH


KS, KD = 0.92, 0.15                          # kernel size; deck offset


def part_kernels(ax, x, top, S):
    """Deck of view kernels, structure in front and the others behind towards the lower right; (x, top) = the
    front matrix's top-left. View names sit small in the empty lower-left corner. Returns the front matrix's
    left-centre and the deck's right edge."""
    for k in reversed(range(4)):
        key, name = VIEWS[k]
        lx, ly = x + k * KD, top - KS - k * KD
        if k == 0:
            ax.add_patch(Rectangle((lx, ly), KS, KS, fc="white", ec="none", zorder=3.6))
            matrix(ax, lx, ly, KS, S["kernels"][key], VIEW[key], has=S["has"][key], z=4)
            axis_labels(ax, lx, ly, KS, size=6.3)
        else:
            ax.add_patch(Rectangle((lx, ly), KS, KS, fc=tint(VIEW[key], 0.35), ec=VIEW[key], lw=0.7,
                                   zorder=3 - 0.1 * k))
        label(ax, x - 0.13, top - KS - KD - 0.07 - k * 0.085, name, size=4.8, color=VIEW[key], ha="left",
              fontweight="bold")
    return (x, top - KS / 2), x + 3 * KD + KS


FW, FH, FS = 1.20, 0.98, 0.62                # fusion box width, height; matrix size


def part_fusion(ax, xc, top, S):
    """Fusion + normalization centred on xc. Returns the matrix's bottom-centre (the query row) and left-middle."""
    x = xc - FW / 2
    box(ax, x, top - FH, FW, FH, OURS_LIGHT, OURS)
    label(ax, xc, top - 0.10, "Fusion + normalization", size=6.3)
    mx, my = xc - FS / 2, top - FH + 0.08
    matrix(ax, mx, my, FS, S["kernels"]["norm"], OURS, hl_last=True)
    axis_labels(ax, mx, my, FS)
    return (xc, my), (x, top - FH / 2)


TH = 1.22


def part_transfer(ax, x, top, xc, vals, S):
    """Kernel transfer with its top-left at (x, top); the similarity column sits at xc and sets the width.
    Returns anchors."""
    TW = xc - x + 0.66
    box(ax, x, top - TH, TW, TH, OURS_LIGHT, OURS)
    label(ax, x + 0.07, top - 0.09, "Kernel transfer", size=6.3, ha="left")
    k = np.array(S["kernels"]["norm"])[QUERY]
    A, c = {}, 0.105
    sx = x + 0.17; sw = xc - c / 2 - 0.05 - sx
    ox = x + TW - 0.06 - SW
    for r, (name, key, qual) in enumerate((("Shared", "g", "all lines"), ("Per line", "h", "line ℓ"))):
        y0 = top - 0.27 - r * 0.50
        label(ax, x + 0.07, y0, name, size=6, ha="left", fontweight="bold")
        label(ax, x + TW - 0.07, y0, qual, size=5.3, color=MUTED, ha="right")
        ys = [y0 - 0.14, y0 - 0.14 - (SH + 0.03)]
        for t, i in enumerate((0, 2)):
            label(ax, x + 0.10, ys[t], LETTER[i], size=5.3, color=INK2)
            strip(ax, sx, ys[t] - SH / 2, sw, SH, vals[f"res_{key}_{i}"], INK2)
            ax.add_patch(Rectangle((xc - c / 2, ys[t] - c / 2), c, c, fc=OURS, alpha=0.06 + 0.94 * k[i],
                                   ec=INK2, lw=0.4, zorder=5))
        ye = ys[1] - (SH + 0.02)
        dots(ax, sx + sw / 2, ye, "h", gap=0.03, r=0.008)
        dots(ax, xc, ye, "v", gap=0.022, r=0.007)
        yb = (ys[0] + ys[1]) / 2
        ax.plot([xc + c / 2 + 0.035] * 2, [ys[1] - c / 2, ys[0] + c / 2], color=OURS, lw=0.7, zorder=4)
        arrow(ax, xc + c / 2 + 0.035, yb, ox - 0.02, yb, color=OURS)
        label(ax, (xc + c / 2 + ox) / 2 + 0.015, yb + 0.065, "Σ", size=6, color=OURS)
        strip(ax, ox, yb - SH / 2, SW, SH, vals[key], OURS)
        A[key] = (ox + SW, yb); A[f"in{r}"] = (x, yb); A[f"w{r}"] = (ys[0] + c / 2, ys[1] - c / 2)
    A["bottom"] = top - TH
    return A


def part_column(ax, xc, A, vals, ys):
    """a g + b h -> compound effect -> interaction -> sum -> response, straight down at xc (labels on the left).
    ys = node heights."""
    y_sum, y_u, y_x, y_p, y_r = ys
    for key in ("g", "h"):                   # both terms run straight to the column and turn down into the sum
        ax.plot([A[key][0], xc], [A[key][1]] * 2, color=OURS, lw=0.8, zorder=2, solid_joinstyle="miter")
    ax.plot([xc, xc], [A["g"][1], A["h"][1]], color=OURS, lw=0.8, zorder=2)
    dot(ax, xc, A["h"][1], OURS)
    arrow(ax, xc, A["h"][1], xc, y_sum + 0.07, color=OURS)
    node(ax, xc, y_sum, "+", color=OURS)
    arrow(ax, xc, y_sum - 0.07, xc, y_u + SH / 2, color=OURS)
    strip(ax, xc - SW / 2, y_u - SH / 2, SW, SH, vals["u"], OURS)
    label(ax, xc - SW / 2 - 0.05, y_u, "Compound effect", size=6.3, ha="right")
    arrow(ax, xc, y_u - SH / 2, xc, y_x + 0.07, color=OURS)
    node(ax, xc, y_x, "×", color=OURS, fc=OURS_LIGHT)
    label(ax, xc - 0.10, y_x - 0.10, "Interaction", size=6.3, ha="right")
    arrow(ax, xc, y_x - 0.07, xc, y_p + 0.07, color=OURS)
    node(ax, xc, y_p, "+")
    arrow(ax, xc, y_p - 0.07, xc, y_r + SH / 2, color=INK2)
    strip(ax, xc - SW / 2, y_r - SH / 2, SW, SH, vals["resp"], INK2)
    label(ax, xc - SW / 2 - 0.06, y_r + 0.01, "Response", size=6.3, ha="right", va="bottom")
    label(ax, xc - SW / 2 - 0.06, y_r - 0.01, "Mitoxantrone", size=5.3, color=OURS, ha="right", va="top",
          style="italic")
    return {"x": (xc - 0.07, y_x), "p": (xc - 0.07, y_p)}


def part_context(ax, x, yT, yD, yB, vals):
    """Cells -> DMSO / treated; backbone; residual. Rows at heights yT (treated), yD (DMSO), yB (backbone)."""
    A = {}
    wx = x + 0.22
    well(ax, wx, yD, 0.16)
    label(ax, wx, yD - 0.28, "cell line\n× plate", size=5.5, color=MUTED, linespacing=1.0)
    dx = x + 0.52
    arrow(ax, wx + 0.17, yD, dx, yD, color=CTX)
    strip(ax, dx, yD - SH / 2, SW, SH, vals["ctl"], CTX)
    label(ax, dx + SW / 2, yD + 0.13, "DMSO", size=6.3)
    A["dmso"] = (dx + SW, yD)
    # treated: the same cells with a profiled compound added
    path(ax, [(wx, yD + 0.17), (wx, yT), (wx + 0.17, yT)], color=INK2)
    syringe_plate(ax, wx + 0.19, yT - 0.06)
    tx = wx + 0.68
    arrow(ax, wx + 0.51, yT, tx, yT, color=INK2)
    deck(ax, tx, yT - SH / 2, vals["trt"], INK2)
    label(ax, tx + SW / 2 + 0.03, yT + 0.18, "Treated", size=6.3)
    m1 = tx + SW + 0.22
    arrow(ax, tx + SW + 0.07, yT, m1 - 0.07, yT, color=INK2)
    node(ax, m1, yT, "−")
    path(ax, [(dx + SW, yD), (m1, yD), (m1, yT - 0.07)], color=CTX)
    dot(ax, m1, yD, CTX)
    mdx = m1 + 0.17
    arrow(ax, m1 + 0.07, yT, mdx, yT, color=INK2)
    deck(ax, mdx, yT - SH / 2, vals["dlt"], INK2)
    label(ax, mdx + SW / 2 + 0.03, yT + 0.18, "Measured Δ", size=6.3)
    # backbone on the DMSO profile
    bx = dx + SW / 2; nx = dx + SW + 0.12
    path(ax, [(bx, yD - SH / 2), (bx, yB), (nx, yB)], color=CTX)
    box(ax, nx, yB - 0.15, 0.46, 0.30, CTX_LIGHT, CTX)
    network(ax, nx, yB - 0.15, 0.46, 0.30, CTX)
    label(ax, nx + 0.23, yB - 0.23, "Backbone", size=6.3)
    cx = nx + 0.58
    arrow(ax, nx + 0.46, yB, cx, yB, color=CTX)
    deck(ax, cx, yB - SH / 2, vals["ctx"], CTX)
    label(ax, cx + SW / 2 + 0.03, yB - 0.15, "Context", size=6.3)
    # residual = measured delta - context
    rx = mdx + SW + 0.26
    path(ax, [(mdx + SW + 0.07, yT), (rx - 0.07, yT)], color=INK2)
    node(ax, rx, yT, "−")
    ax.plot([cx + SW + 0.07, rx], [yB, yB], color=CTX, lw=0.8, zorder=2)
    dot(ax, rx, yB, CTX)
    path(ax, [(rx, yB), (rx, yT - 0.22), (rx, yT - 0.07)], color=CTX)
    rsx = rx + 0.17
    arrow(ax, rx + 0.07, yT, rsx, yT, color=INK2)
    deck(ax, rsx, yT - SH / 2, vals["res"], INK2)
    label(ax, rsx + SW / 2 + 0.03, yT + 0.18, "Residual", size=6.3)
    A["res"] = (rsx + SW + 0.07, yT); A["ctx"] = (rx, yB)
    box(ax, wx - 0.08, yT - 0.20, rsx + SW + 0.12 - (wx - 0.08), 0.52, "none", MUTED, lw=0.5, ls=(0, (2, 2)), z=0)
    label(ax, rsx + SW + 0.04, yT + 0.32, "profiled compounds", size=5.3, color=MUTED, ha="right", style="italic",
          bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    return A


def toy_vals(G=12):
    v = {"ctl": rng.normal(0, 1, G), "trt": rng.normal(0, 1, G), "ctx": rng.normal(0, 1, G) * 1.4}
    v["dlt"] = v["ctx"] + rng.normal(0, 1, G)
    v["res"] = v["dlt"] - v["ctx"]
    for key in ("g", "h"):
        for i in (0, 2):
            v[f"res_{key}_{i}"] = rng.normal(0, 1, G)
    v["g"] = rng.normal(0, 1, G); v["h"] = v["g"] + 0.5 * rng.normal(0, 1, G); v["u"] = v["g"] + v["h"]
    v["resp"] = v["ctx"] * 0.5 + v["u"]
    return v


# ---------------------------------------------------------------- assembly
W, H = 6.5, 4.60
XF = 5.02                                    # fusion box centre = similarity column of the transfer
XO = 6.16                                    # output column


def assemble(S, vals):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    top = H - 0.04
    ends, cbot = part_cards(ax, 0.04, top, S)
    kx, ktop = 2.76, top - 0.16
    (kfx, kfy), kright = part_kernels(ax, kx, ktop, S)
    xb = 0.04 + CW + 0.10                    # cards -> kernel deck
    for e in ends:
        ax.plot([e[0] + 0.02, xb], [e[1]] * 2, color=LINE, lw=0.7, zorder=2)
    ax.plot([xb, xb], [ends[0][1], ends[-1][1]], color=LINE, lw=0.7, zorder=2)
    path(ax, [(xb, kfy), (kfx - 0.16, kfy)], color=INK2)
    ftop = top - 0.06
    (qx, qy), (fl, fm) = part_fusion(ax, XF, ftop, S)
    path(ax, [(kright + 0.05, fm), (fl, fm)], color=INK2)                   # deck -> fusion
    yT, yD, yB = 1.75, 0.98, 0.55           # context lane rows; the per-line row of the transfer sits at yT
    tx = 3.55
    T = part_transfer(ax, tx, yT + 0.97, XF, vals, S)
    arrow(ax, qx, qy, qx, T["w0"][0] + 0.003, color=OURS, lw=0.9)          # query row -> similarity weights
    ax.plot([qx, qx], [T["w0"][1], T["w1"][0]], color=OURS, lw=0.9, zorder=3)
    A = part_context(ax, 0.04, yT, yD, yB, vals)
    rx, ry = A["res"]; br = tx - 0.18                                      # residual -> per line, branch -> shared
    arrow(ax, rx, ry, tx, T["in1"][1], color=INK2, lw=0.7)
    path(ax, [(br, ry), (br, T["in0"][1]), (tx, T["in0"][1])], color=INK2, lw=0.7)
    dot(ax, br, ry, INK2)
    C = part_column(ax, XO, T, vals, (T["bottom"] - 0.03, T["bottom"] - 0.27, yD, yB, 0.17))
    path(ax, [A["dmso"], (C["x"][0] - 0.25, yD), C["x"]], color=CTX, halo=True)   # control PCs -> interaction
    label(ax, 4.40, yD + 0.07, "Control PCs", size=5.8, color=INK2, ha="right")
    path(ax, [A["ctx"], C["p"]], color=CTX)                                # context -> sum
    return fig


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = Path(args[0]) if args else FIGDIR
    S = json.loads((src / "fig1_pipeline_source.json").read_text())
    out = Path(args[1]) if len(args) > 1 else None
    vals = toy_vals()
    if "--parts" in sys.argv:
        parts = {
            "cards": ((CW + 0.1, 3 * CH + 0.4), lambda ax: part_cards(ax, 0.05, 3 * CH + 0.36, S)),
            "kernels": ((1.72, 1.78), lambda ax: part_kernels(ax, 0.22, 1.64, S)),
            "fusion": ((1.3, 1.05), lambda ax: part_fusion(ax, 0.65, 1.02, S)),
            "transfer": ((1.8, 1.3), lambda ax: part_transfer(ax, 0.05, 1.27, 0.97, vals, S)),
            "context": ((3.45, 1.58), lambda ax: part_context(ax, 0.04, 1.20, 0.74, 0.30, vals)),
        }
        for name, ((w, h), draw) in parts.items():
            fig = plt.figure(figsize=(w, h))
            ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
            draw(ax)
            save(fig, f"fig1_part_{name}", (out or FIGDIR) / "fig1_parts", exts=("svg",))
            plt.close(fig)
    save(assemble(S, vals), "fig1_pipeline", out)


if __name__ == "__main__":
    main()
