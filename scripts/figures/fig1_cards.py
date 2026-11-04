"""Fig. 1 v7 of the UnDetect paper: modules (Jun Wen's review, 2026-11-04). Dossier: paper/figures/fig1_pipeline.md.

Every block sits in a rounded module frame with a title, and each frame colour has one meaning (legend, top right):
lavender = compound information (cards, view kernels); sand, dashed = measured responses of profiled compounds only;
gray = context, no compound; blue = compound effect (fusion, kernel regression, compound effect); light red =
prediction (interaction, response; UnDetect's colour in Fig. 2). View colours (red structure, yellow targets, pink
mechanism, green Cell Painting) only mark the view, each with an icon: fingerprint, protein with a binding pocket,
pathway with an inhibition bar, a stained cell. Compounds carry their letter (D, V, M) used in the kernels. Kernel
layers are tabbed cards, each tab with the view's icon and name. Compound effect, control PCs, interaction and response
are separate boxes; all but the response sit in a Prediction module on the Context module's row (2026-11-04). Line corners are rounded.
Shading only, no numbers: kernel shades are the real values (fig1_cards_data.py); expression strips are a toy seed.
v6 (2026-11-02): compound cards; v7 keeps its data and order (D, V, ..., M; query last).
Run: python -I scripts/figures/fig1_cards.py [SOURCE_DIR [OUTDIR]]   (numpy + matplotlib; SOURCE_DIR and OUTDIR
default to paper/figures; writes fig1_pipeline[_preview].{pdf,svg,png}).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import CTX, CTX_LIGHT, DIV_NEG, DIV_POS, FIGDIR, INK, INK2, LINE, MUTED, OURS, OURS_LIGHT, PANEL, VIEW, plt, save  # noqa: E402,E501
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge  # noqa: E402,E501

SEED = 7
QUERY = 1                                   # mitoxantrone, the unprofiled compound
TITLE = ["Daunorubicin", "Mitoxantrone", "Vorinostat"]
LETTER = {0: "D", 1: "M", 2: "V"}
ORDER = [0, 2, None, 1]                     # matrix order: D, V, other profiled compounds, M (query last)
VIEWS = [("chem", "Structure"), ("tgt", "Protein targets"), ("moa", "Mechanism"), ("morph", "Cell Painting")]
MOD = {"info": "#7b74c9", "meas": "#b8893a", "ctx": CTX, "eff": OURS, "pred": "#e0605a"}
MOD_NAME = {"info": "compound info", "meas": "measured", "ctx": "context", "eff": "compound effect",
            "pred": "prediction"}
PRED_LIGHT = "#fbe3e1"
DIV = mcolors.LinearSegmentedColormap.from_list("div", [DIV_NEG, "#f1f0ed", DIV_POS])   # expression change, as in Fig. 3
FLOW = "#4d4d4d"                             # every arrow except the compound-effect lane
SW, SH = 0.34, 0.085                        # expression strip
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- primitives
def tint(c, a):
    r = np.array(mcolors.to_rgb(c))
    return tuple(1 - a * (1 - r))


def box(ax, x, y, w, h, fc, ec, r=0.05, lw=0.7, z=1, ls="-", **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z, **kw))


def label(ax, x, y, s, size=6.5, color=INK, ha="center", va="center", **kw):
    kw.setdefault("zorder", 9)
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, **kw)


def fit(ax, x, y, s, maxw, size, **kw):
    """Left-aligned label shrunk until it is at most maxw inches wide."""
    t = label(ax, x, y, s, size=size, ha="left", **kw)
    r = ax.figure.canvas.get_renderer()
    while t.get_window_extent(r).width / ax.figure.dpi > maxw and size > 3.5:
        size -= 0.2; t.set_fontsize(size)
    return t


HB = 0.155                                   # module title band


def module(ax, x, y, w, h, title, key, dashed=False, z=0.5, size=5.5, tx=0.07):
    """Module card: white, soft shadow, hairline border in a light tint of its meaning colour; a light grey title
    band with a hairline divider; a short accent bar in the module colour, then the title in capitals."""
    c = MOD[key]
    ax.add_patch(FancyBboxPatch((x + 0.014, y - 0.014), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="black", alpha=0.06, ec="none", zorder=z - 0.05))
    fr = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec="none", zorder=z)
    ax.add_patch(fr)
    hb = Rectangle((x, y + h - HB), w, HB, fc="#f3f2ef", ec="none", zorder=z + 0.01)
    ax.add_patch(hb); hb.set_clip_path(fr)
    ax.plot([x, x + w], [y + h - HB] * 2, color=tint(c, 0.35), lw=0.4, zorder=z + 0.02)
    box(ax, x, y, w, h, "none", tint(c, 0.6), r=0.06, lw=0.5, z=z + 0.03, ls=(0, (3, 2)) if dashed else "-")
    dark = tuple(0.78 * np.array(mcolors.to_rgb(c)))
    t = label(ax, x + tx, y + h - HB / 2, title.upper(), size=size, color=dark if key != "ctx" else INK2,
              ha="left", fontweight="bold", zorder=z + 0.1)
    r_ = ax.figure.canvas.get_renderer()
    while t.get_window_extent(r_).width / ax.figure.dpi > w - tx - 0.10 and size > 4:
        size -= 0.2; t.set_fontsize(size)


def path(ax, pts, color=FLOW, lw=0.6, r=0.07, head=True, z=2):
    """Polyline with rounded corners (quadratic arcs of radius r) and an arrow head at its end."""
    P = [np.asarray(p, float) for p in pts]
    verts, codes = [P[0]], [MPath.MOVETO]
    for a, b, c in zip(P[:-2], P[1:-1], P[2:]):
        d1, d2 = b - a, c - b
        l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
        rr = min(r, l1 / 2, l2 / 2)
        verts += [b - d1 / l1 * rr, b, b + d2 / l2 * rr]; codes += [MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
    verts.append(P[-1]); codes.append(MPath.LINETO)
    ax.add_patch(FancyArrowPatch(path=MPath(verts, codes), arrowstyle="-|>" if head else "-", mutation_scale=6,
                                 lw=lw, color=color, zorder=z, joinstyle="round", capstyle="round"))


def arrow(ax, x1, y1, x2, y2, color=FLOW, lw=0.6):
    path(ax, [(x1, y1), (x2, y2)], color=color, lw=lw, z=3)


def dot(ax, x, y, color):
    ax.add_patch(Circle((x, y), 0.017, fc=color, ec="none", zorder=4))


def node(ax, x, y, sym, color=INK2, fc="white", r=0.065):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=color, lw=0.8, zorder=5))
    label(ax, x, y + 0.004, sym, size=7.5, color=color)


def dots(ax, x, y, orient="v", gap=0.035, r=0.009, color=MUTED):
    d = {"v": (0, 1), "h": (1, 0), "d": (1, -1)}[orient]
    for k in (-1, 0, 1):
        ax.add_patch(Circle((x + k * gap * d[0], y + k * gap * d[1]), r, fc=color, ec="none", zorder=7))


def strip(ax, x, y, w, h, vals, rgb, z=4):
    """A vector of numbers as a row of cells. Hue = what the vector is (the colour of the module it comes from, or of
    its view); shade = its value, light for the lowest and dark for the highest."""
    v = np.asarray(vals, float); cw = w / len(v); sv = (v - v.min()) / max(v.max() - v.min(), 1e-9)
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="none", zorder=z))
    for i, a in enumerate(sv):
        ax.add_patch(Rectangle((x + i * cw, y), cw, h, fc=rgb, alpha=0.10 + 0.85 * a, ec="none", zorder=z))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=LINE, lw=0.4, zorder=z + 0.5))


def deck(ax, x, y, vals, rgb, n=3, d=0.026):
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


def axis_labels(ax, x, y, s, size=5.5, cols="top"):
    c = s / 4
    for r, i in enumerate(ORDER):
        bold = "bold" if i == QUERY else None
        cy = y + s + 0.035 if cols == "top" else y - 0.035
        if i is None:
            dots(ax, x - 0.05, y + s - (r + 0.5) * c, "v", gap=0.022, r=0.006, color=INK2)
            dots(ax, x + (r + 0.5) * c, cy + (0.025 if cols == "top" else -0.025), "h", gap=0.022, r=0.006, color=INK2)
        else:
            label(ax, x - 0.035, y + s - (r + 0.5) * c, LETTER[i], size=size, color=INK2, ha="right",
                  fontweight=bold)
            label(ax, x + (r + 0.5) * c, cy, LETTER[i], size=size, color=INK2,
                  va="bottom" if cols == "top" else "top", fontweight=bold)


# ---------------------------------------------------------------- view icons (s = icon height, centred at x, y)
def icon_fp(ax, x, y, s, c=VIEW["chem"]):
    """Fingerprint: nested ridges open at the bottom."""
    for k in range(4):
        rr = (0.12 + 0.13 * k) * s
        ax.add_patch(Arc((x, y - 0.08 * s), 1.6 * rr, 2.0 * rr, theta1=-40 + 6 * k, theta2=220 - 6 * k, color=c,
                         lw=0.55, zorder=8))


def icon_protein(ax, x, y, s, c=VIEW["tgt"]):
    """Protein with a binding pocket and a small ligand in it."""
    ax.add_patch(Wedge((x - 0.05 * s, y), 0.46 * s, 32, 328, fc=tint(c, 0.55), ec=tint(c, 1.0), lw=0.5, zorder=8))
    ax.add_patch(Polygon([(x + 0.12 * s, y), (x + 0.26 * s, y + 0.12 * s), (x + 0.40 * s, y),
                          (x + 0.26 * s, y - 0.12 * s)], fc=INK2, ec="none", zorder=9))


def icon_moa(ax, x, y, s, c=VIEW["moa"]):
    """Mechanism: a pathway step blocked by an inhibition bar (node -> node -| )."""
    dc = tint(c, 1.0) if c != VIEW["moa"] else "#c64f86"
    for dx in (-0.42, 0.02):
        ax.add_patch(Circle((x + dx * s, y), 0.13 * s, fc=tint(c, 0.6), ec=dc, lw=0.5, zorder=8))
    ax.plot([x - 0.29 * s, x - 0.13 * s], [y, y], color=dc, lw=0.6, zorder=8)
    ax.add_patch(Polygon([(x - 0.11 * s, y), (x - 0.18 * s, y + 0.06 * s), (x - 0.18 * s, y - 0.06 * s)], fc=dc,
                         ec="none", zorder=8))
    ax.plot([x + 0.15 * s, x + 0.40 * s], [y, y], color=dc, lw=0.6, zorder=8)
    ax.plot([x + 0.40 * s, x + 0.40 * s], [y - 0.16 * s, y + 0.16 * s], color=dc, lw=0.9, zorder=8)


def icon_cp(ax, x, y, s, c=VIEW["morph"]):
    """Cell Painting: a stained cell on a dark field (cytoplasm in the view colour, nucleus blue)."""
    ax.add_patch(FancyBboxPatch((x - 0.45 * s, y - 0.45 * s), 0.9 * s, 0.9 * s, boxstyle=f"round,pad=0,"
                                f"rounding_size={0.12 * s}", fc="#1e1a3a", ec="none", zorder=8))
    ax.add_patch(Ellipse((x, y), 0.74 * s, 0.50 * s, angle=20, fc=c, ec="none", alpha=0.95, zorder=8.5))
    ax.add_patch(Circle((x + 0.05 * s, y + 0.01 * s), 0.14 * s, fc="#5fb3ff", ec="none", zorder=9))


ICON = {"chem": icon_fp, "tgt": icon_protein, "moa": icon_moa, "morph": icon_cp}


# ---------------------------------------------------------------- context icons
LINE_C = ["#7d8fa3", "#a6879a", "#8ea486"]    # cell lines (muted, apart from the module and view hues)


def dish(ax, x, y, w, color, z):
    """A petri dish seen from slightly above (cylinder), (x, y) = centre of its rim; cells in the cell line's colour."""
    hgt, wall = 0.32 * w, 0.20 * w
    ax.add_patch(Ellipse((x, y - wall), w, hgt, fc="white", ec=INK2, lw=0.5, zorder=z))
    ax.add_patch(Rectangle((x - w / 2, y - wall), w, wall, fc="white", ec="none", zorder=z + 0.01))
    ax.plot([x - w / 2] * 2, [y - wall, y], color=INK2, lw=0.5, zorder=z + 0.02)
    ax.plot([x + w / 2] * 2, [y - wall, y], color=INK2, lw=0.5, zorder=z + 0.02)
    ax.add_patch(Ellipse((x, y), w, hgt, fc=tint(color, 0.18), ec=INK2, lw=0.5, zorder=z + 0.03))
    for dx, dy, a in ((-0.24, 0.05, 20), (0.20, 0.08, -30), (-0.05, -0.07, 60), (0.30, -0.06, 10), (0.02, 0.10, -50),
                      (-0.33, -0.04, 35)):
        ax.add_patch(Ellipse((x + dx * w, y + dy * w), 0.15 * w, 0.08 * w, angle=a, fc=color, ec="none",
                             zorder=z + 0.04))


def dishes(ax, x, y):
    """Three cell lines, one dish each (back two, front one), centred near (x, y)."""
    for k, (dx, dy) in enumerate(((-0.08, 0.075), (0.08, 0.075), (0.0, -0.035))):
        dish(ax, x + dx, y + dy, 0.21, LINE_C[k], z=3 + k * 0.1)


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
            ax.add_patch(Circle(p, 0.018, fc="white", ec=color, lw=0.7, zorder=3))


def syringe_plate(ax, x, y, color=INK2):
    """A multiwell plate (lower-left corner at (x, y)) with a syringe dispensing into one well."""
    pw, ph = 0.30, 0.12
    box(ax, x, y, pw, ph, PANEL, color, r=0.015, lw=0.6, z=5)
    for i in range(4):                       # wells hold the cell lines (same colours as the dishes)
        for j in range(2):
            ax.add_patch(Circle((x + 0.045 + i * 0.07, y + 0.035 + j * 0.05), 0.017, fc=tint(LINE_C[i % 3], 0.6),
                                ec=color, lw=0.4, zorder=6))
    wx, wy = x + 0.045 + 2 * 0.07, y + 0.085
    by = wy + 0.05
    ax.plot([wx, wx], [wy + 0.012, by], color=color, lw=0.6, zorder=7)
    ax.add_patch(Rectangle((wx - 0.022, by), 0.044, 0.10, fc="white", ec=color, lw=0.6, zorder=7))
    ax.add_patch(Rectangle((wx - 0.022, by), 0.044, 0.04, fc=tint(MOD["meas"], 0.4), ec="none", zorder=7.5))
    ax.plot([wx - 0.03, wx + 0.03], [by + 0.10, by + 0.10], color=color, lw=0.6, zorder=7)
    ax.plot([wx, wx], [by + 0.10, by + 0.13], color=color, lw=0.6, zorder=7)
    ax.plot([wx - 0.02, wx + 0.02], [by + 0.13, by + 0.13], color=color, lw=0.9, zorder=7)


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
CW, CH = 2.34, 0.60                          # card
SEC, SGAP = 0.094, 0.012                     # view section height and gap


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
    box(ax, x + 0.01, top - CH - 0.01, CW, CH, "black", "none", lw=0, z=0.95, alpha=0.05)
    box(ax, x, top - CH, CW, CH, "white", OURS if q else LINE, lw=0.8 if q else 0.45, z=1)
    t = label(ax, x + 0.07, top - 0.075, f"{TITLE[k]} ({LETTER[k]})", size=6.6, ha="left", fontweight="bold")
    label(ax, x + CW - 0.07, top - 0.075, "unprofiled query" if q else "profiled", size=5.3,
          color=OURS if q else MUTED, ha="right", style="italic")
    molecule(ax, S["structures"][k], x + 0.05, top - CH + 0.03, 0.78, CH - 0.17)
    sx = x + 0.88; sw = CW - 0.88 - 0.06
    for r, (key, _) in enumerate(VIEWS):
        sy = top - 0.15 - r * (SEC + SGAP) - SEC; yc = sy + SEC / 2
        has = S["has"][key][k]
        if has:
            box(ax, sx, sy, sw, SEC, tint(VIEW[key], 0.28), "none", r=0.02, z=2)
        else:
            box(ax, sx, sy, sw, SEC, "white", LINE, r=0.02, lw=0.4, z=2, hatch="//////")
        ICON[key](ax, sx + 0.065, yc, 0.085)
        tx = sx + 0.13
        if key == "chem":                    # Morgan fingerprint: the real on-bits as a barcode
            bx0, bw = tx, sw - 0.17
            ax.add_patch(Rectangle((bx0, sy + 0.02), bw, SEC - 0.04, fc="white", ec="none", zorder=3))
            for b in S["fp_bits"][k]:
                xb = bx0 + bw * b / 2047
                ax.plot([xb, xb], [sy + 0.026, sy + SEC - 0.026], color=VIEW["chem"], lw=0.45, zorder=4)
        elif key == "tgt":
            fit(ax, tx, yc, ", ".join(_short_targets(S["target"][k])), sw - 0.16, 5.0)
        elif key == "moa":
            fit(ax, tx, yc, _join_moa(S["moa"][k][::-1]), sw - 0.16, 5.0)
        elif has:                            # Cell Painting: the profile vector (z-scores on 40 features)
            v = np.array(S["cp_profile"][k])
            strip(ax, tx, sy + 0.02, sw - 0.17, SEC - 0.04, v, VIEW["morph"], z=4)
        else:
            fit(ax, tx, yc, "no Cell Painting profile", sw - 0.16, 5.0, color=MUTED,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))


def part_cards(ax, x, top, S, gap=0.05, ell=0.16):
    """Profiled cards, an ellipsis, then the query card. Returns right-edge centres and the bottom."""
    ys, ends = top, []
    for k in (0, 2):
        card(ax, x, ys, k, S); ends.append((x + CW, ys - CH / 2)); ys -= CH + gap
    dots(ax, x + CW / 2, ys - ell / 2 + gap / 2, "v", gap=0.035, r=0.011)
    ends.append((x + CW, ys - ell / 2 + gap / 2))
    ys -= ell
    card(ax, x, ys, QUERY, S); ends.append((x + CW, ys - CH / 2))
    return ends, ys - CH


KS, KD, TAB = 0.80, 0.135, 0.115             # kernel size; deck offset; tab height


def part_kernels(ax, x, y, S):
    """Tabbed deck of view kernels: structure in front at the lower left, the others behind towards the upper right,
    each card topped by a tab with the view's icon and name. (x, y) = the front matrix's lower-left corner.
    Returns the front matrix's left-centre and the deck's right edge."""
    for k in reversed(range(4)):
        key, name = VIEWS[k]
        lx, ly = x + k * KD, y + k * KD
        c = VIEW[key]
        tc = tint(c, 1.0) if key != "moa" else "#c64f86"
        ax.add_patch(FancyBboxPatch((lx, ly), KS, KS + TAB, boxstyle="round,pad=0,rounding_size=0.025",
                                    fc=tint(c, 0.14) if k else "white", ec=tint(tc, 0.7), lw=0.5, zorder=3 + 0.3 * (4 - k)))
        ax.add_patch(FancyBboxPatch((lx, ly + KS), KS, TAB, boxstyle="round,pad=0,rounding_size=0.025",
                                    fc=tint(c, 0.42), ec=tint(tc, 0.7), lw=0.5, zorder=3.1 + 0.3 * (4 - k)))
        z0 = 3.2 + 0.3 * (4 - k)
        ICON[key](ax, lx + 0.07, ly + KS + TAB / 2, 0.075)
        t = label(ax, lx + 0.13, ly + KS + TAB / 2, name, size=5.0, color=INK, ha="left", fontweight="bold")
        t.set_zorder(z0 + 5)
        if k == 0:
            matrix(ax, lx, ly, KS, S["kernels"][key], VIEW[key], has=S["has"][key], z=z0 + 0.5)
            axis_labels(ax, lx, ly, KS, size=6.0, cols="bottom")
    return (x, y + KS / 2), x + 3 * KD + KS


FS = KS                                      # fusion matrices, the size of the view kernels


def matrix_mix(ax, x, y, s, S, z=4):
    """The weighted sum sum_v w_v K_v before normalization, in ORDER (D, V, ..., M): every cell split into one stripe
    per view, stripe width proportional to w_v, shade = that view's similarity; a view the pair lacks is hatched, so a
    compound with a missing view has a weaker diagonal (lower self-similarity)."""
    c = s / 4; g = 0.03 * c / 0.16
    wt = np.array([S["weights"][k] for k, _ in VIEWS]); wt = wt / wt.sum()
    for r, i in enumerate(ORDER):
        for q, j in enumerate(ORDER):
            cx, cy = x + q * c, y + s - (r + 1) * c
            ax.add_patch(Rectangle((cx, cy), c, c, fc="white", ec="none", zorder=z))
            if i is None or j is None:
                dots(ax, cx + c / 2, cy + c / 2, "d" if i is None and j is None else "v" if i is None else "h",
                     gap=g, r=0.007)
                continue
            x0 = cx
            for (key, _), wv in zip(VIEWS, wt):
                sw_ = c * wv
                if S["has"][key][i] and S["has"][key][j]:
                    ax.add_patch(Rectangle((x0, cy), sw_, c, fc=VIEW[key], alpha=0.06 + 0.94 * S["kernels"][key][i][j],
                                           ec="none", zorder=z))
                else:
                    ax.add_patch(Rectangle((x0, cy), sw_, c, fc="white", ec=LINE, lw=0.2, hatch="//////", zorder=z))
                x0 += sw_
            ax.add_patch(Rectangle((cx, cy), c, c, fc="none", ec="white", lw=0.5, zorder=z + 0.2))
    ax.add_patch(Rectangle((x, y), s, s, fc="none", ec=INK2, lw=0.5, zorder=z + 1))


WCOL = 0.62                                  # width of the view-weight column in the fusion module
FWID = 0.12 + WCOL + 0.20 + FS + 0.15        # fusion module width
FHGT = HB + 0.08 + FS + 0.18                 # fusion module height (only what it holds)


def part_fusion(ax, xl, top, S):
    """Fusion + normalization, compact, from xl: left the view weights w_v (KTA, mean of five folds) as bars with the
    view icons, no values; right the normalised multi-view kernel (unit diagonal). Bar length is drawn from an offset
    (w - 0.20) so the small differences between views are visible; the bars are a schematic, not to scale. Returns the
    kernel's bottom-centre (the query row) and the module's left edge at its mid-height."""
    bot = top - FHGT
    module(ax, xl, bot, FWID, FHGT, "Fusion + normalization", "eff")
    my = bot + 0.18; ym = my + FS / 2; wx = xl + 0.12
    for r, (key, _) in enumerate(VIEWS):
        yy = ym + (1.5 - r) * 0.17; w_ = S["weights"][key]; bl = 0.06 + (w_ - 0.20) / 0.10 * 0.40
        ICON[key](ax, wx + 0.04, yy, 0.085)
        ax.add_patch(Rectangle((wx + 0.13, yy - 0.035), bl, 0.07, fc=VIEW[key], ec="none", zorder=4))
    nx = xl + 0.12 + WCOL + 0.20
    arrow(ax, xl + 0.12 + WCOL - 0.02, ym, nx - 0.10, ym, color=OURS)    # stops before the row labels
    matrix(ax, nx, my, FS, S["kernels"]["norm"], OURS, hl_last=True)
    axis_labels(ax, nx, my, FS, size=6.0, cols="bottom")
    return (nx + FS / 2, my), (xl, ym)


TH = 1.17


def part_transfer(ax, x, top, xc, vals, S):
    """Kernel regression with its top-left at (x, top); the similarity column sits at xc and sets the width."""
    TW = W - 0.03 - x
    module(ax, x, top - TH, TW, TH, "Kernel regression", "eff")
    k = np.array(S["kernels"]["norm"])[QUERY]
    A, c = {}, 0.10
    sx = x + 0.17; sw = xc - c / 2 - 0.05 - sx
    SWo = 0.18                               # output strips (Σ over profiled compounds)
    ox = xc + c / 2 + 0.11
    for r, (name, key, qual) in enumerate((("Shared", "g", "all lines"), ("Per line", "h", "line ℓ"))):
        y0 = top - 0.25 - r * 0.47
        t = label(ax, x + 0.07, y0, name, size=5.8, ha="left")
        r_ = ax.figure.canvas.get_renderer(); wn = t.get_window_extent(r_).width / ax.figure.dpi
        label(ax, x + 0.07 + wn + 0.05, y0, f"({qual})", size=5.2, color=MUTED, ha="left")
        ys = [y0 - 0.13, y0 - 0.13 - (SH + 0.03)]
        for t, i in enumerate((0, 2)):
            label(ax, x + 0.10, ys[t], LETTER[i], size=5.3, color=INK2)
            strip(ax, sx, ys[t] - SH / 2, sw, SH, vals[f"res_{key}_{i}"], tint(MOD["meas"], 1.0))
            ax.add_patch(Rectangle((xc - c / 2, ys[t] - c / 2), c, c, fc=OURS, alpha=0.06 + 0.94 * k[i],
                                   ec=INK2, lw=0.4, zorder=5))
        ye = ys[1] - (SH + 0.02)
        dots(ax, sx + sw / 2, ye, "h", gap=0.03, r=0.008)
        dots(ax, xc, ye, "v", gap=0.022, r=0.007)
        yb = (ys[0] + ys[1]) / 2
        ax.plot([xc + c / 2 + 0.035] * 2, [ys[1] - c / 2, ys[0] + c / 2], color=OURS, lw=0.7, zorder=4)
        arrow(ax, xc + c / 2 + 0.035, yb, ox - 0.02, yb, color=OURS)
        label(ax, (xc + c / 2 + ox) / 2 + 0.015, yb + 0.06, "Σ", size=6, color=OURS)
        strip(ax, ox, yb - SH / 2, SWo, SH, vals[key], OURS)
        A[key] = (ox + SWo, yb); A[f"in{r}"] = (x, yb); A[f"w{r}"] = (ys[0] + c / 2, ys[1] - c / 2)
    xp = ox + SWo + 0.10; yp = A["h"][1]      # shared + per line at the per-line row; the sum leaves downwards
    path(ax, [A["g"], (xp, A["g"][1]), (xp, yp + 0.065)], color=OURS)
    path(ax, [A["h"], (xp - 0.065, yp)], color=OURS)
    node(ax, xp, yp, "+", color=OURS)
    A["sum"] = (xp, yp - 0.065)
    A["bottom"] = top - TH; A["right"] = x + TW
    return A


BW = 0.76                                    # width of the boxes in the output column


def small_box(ax, xc, yc, title, key, vals=None, rgb=None, h=0.27, w=BW):
    """A titled box in the output column; returns its top and bottom y."""
    c = MOD[key]                              # same treatment as the modules: white, soft shadow, hairline border
    box(ax, xc - w / 2 + 0.012, yc - h / 2 - 0.012, w, h, "black", "none", r=0.04, lw=0, z=2.9, alpha=0.06)
    box(ax, xc - w / 2, yc - h / 2, w, h, "white", tint(c, 0.75), r=0.04, lw=0.6, z=3)
    t = label(ax, xc, yc + h / 2 - 0.065, title, size=5.3, color=INK, fontweight="bold")
    r_ = ax.figure.canvas.get_renderer(); size = 5.3
    while t.get_window_extent(r_).width / ax.figure.dpi > w - 0.10 and size > 4:
        size -= 0.1; t.set_fontsize(size)
    if vals is not None:
        strip(ax, xc - SW / 2, yc - h / 2 + 0.04, SW, SH, vals, rgb, z=4.5)
    return yc + h / 2, yc - h / 2


def part_output(ax, A, xu, xi, y_x, y_p, xr):
    """Regression's sum -> down, then left into the compound effect -> left into the interaction -> down to + context
    -> right, out of the prediction module, to the response. Returns the inputs of the interaction (control PCs, from
    the left) and of the sum (context, from the left)."""
    small_box(ax, xu, y_x, "Compound effect", "eff", None)
    strip(ax, xu - SW / 2, y_x - 0.105, SW, SH, VALS["u"], OURS, z=4.5)
    path(ax, [A["sum"], (A["sum"][0], y_x), (xu + BW / 2, y_x)], color=OURS)
    xt, xb = small_box(ax, xi, y_x + 0.03, "Interaction", "pred", h=0.31)   # top clear of the module's title band
    node(ax, xi, y_x, "×", color=MOD["pred"], fc="white", r=0.052)
    arrow(ax, xu - BW / 2, y_x, xi + BW / 2, y_x, color=OURS)
    node(ax, xi, y_p, "+", color=INK2)
    arrow(ax, xi, xb, xi, y_p + 0.065, color=MOD["pred"])
    small_box(ax, xr, y_p, "Response (M)", "pred", None)
    strip(ax, xr - SW / 2, y_p - 0.105, SW, SH, VALS["resp"], MOD["pred"], z=4.5)
    arrow(ax, xi + 0.065, y_p, xr - BW / 2, y_p, color=FLOW)
    return {"x": (xi - BW / 2, y_x), "p": (xi - 0.065, y_p)}


def part_context(ax, x, yT, yD, yB, vals):
    """Cells -> DMSO / treated; backbone; residual. Rows at heights yT (treated), yD (DMSO), yB (backbone)."""
    A = {}
    wx = x + 0.22
    dishes(ax, wx, yD)
    label(ax, wx, yD - 0.22, "cell lines", size=5.4, color=INK2)
    dx = x + 0.50
    arrow(ax, wx + 0.20, yD, dx, yD, color=FLOW)
    strip(ax, dx, yD - SH / 2, SW, SH, vals["ctl"], CTX)
    label(ax, dx + SW / 2, yD + 0.12, "DMSO control", size=5.8)
    A["dmso"] = (dx + SW, yD)
    # treated: the same cells with a profiled compound added
    path(ax, [(wx, yD + 0.16), (wx, yT), (wx + 0.17, yT)], color=FLOW)
    syringe_plate(ax, wx + 0.19, yT - 0.07)
    tx = wx + 0.66
    arrow(ax, wx + 0.51, yT, tx, yT, color=FLOW)
    deck(ax, tx, yT - SH / 2, vals["trt"], tint(MOD["meas"], 1.0))
    label(ax, tx + SW / 2 + 0.03, yT - 0.12, "Treated", size=5.6)
    m1 = tx + SW + 0.21
    arrow(ax, tx + SW + 0.07, yT, m1 - 0.065, yT, color=FLOW)
    node(ax, m1, yT, "−")
    path(ax, [(dx + SW, yD), (m1, yD), (m1, yT - 0.065)], color=FLOW)
    mdx = m1 + 0.16
    arrow(ax, m1 + 0.065, yT, mdx, yT, color=FLOW)
    deck(ax, mdx, yT - SH / 2, vals["dlt"], tint(MOD["meas"], 1.0))
    label(ax, mdx + SW / 2 + 0.03, yT - 0.12, "Measured Δ", size=5.6)
    # backbone on the DMSO profile
    bx = dx + SW / 2; nx = dx + SW + 0.10
    path(ax, [(bx, yD - SH / 2), (bx, yB), (nx, yB)], color=FLOW)
    box(ax, nx, yB - 0.14, 0.44, 0.28, "white", CTX, r=0.04)
    network(ax, nx, yB - 0.14, 0.44, 0.28, CTX)
    label(ax, nx + 0.22, yB - 0.21, "Backbone", size=5.8)
    cx = nx + 0.56
    arrow(ax, nx + 0.44, yB, cx, yB, color=FLOW)
    deck(ax, cx, yB - SH / 2, vals["ctx"], CTX)
    label(ax, cx + SW / 2 + 0.03, yB - 0.14, "Context", size=5.8)
    # residual = measured delta - context
    rx = mdx + SW + 0.25
    path(ax, [(mdx + SW + 0.07, yT), (rx - 0.065, yT)], color=FLOW)
    node(ax, rx, yT, "−")
    path(ax, [(cx + SW + 0.07, yB), (rx, yB), (rx, yT - 0.065)], color=FLOW)
    rsx = rx + 0.16
    arrow(ax, rx + 0.065, yT, rsx, yT, color=FLOW)
    deck(ax, rsx, yT - SH / 2, vals["res"], tint(MOD["meas"], 1.0))
    label(ax, rsx + SW / 2 + 0.03, yT - 0.12, "Residual", size=5.6)
    A["res"] = (rsx + SW + 0.07, yT); A["ctx"] = (cx + SW + 0.07, yB); A["ctx_branch"] = rx
    A["meas"] = (wx + 0.08, rsx + SW + 0.10)
    A["ctx_box"] = (x, cx + SW + 0.12)
    return A


def legend(ax, x, top, w):
    module_h = 0.19 + 0.135 * len(MOD)
    box(ax, x, top - module_h, w, module_h, "white", LINE, r=0.05, lw=0.6, z=1)
    label(ax, x + 0.07, top - 0.085, "Modules", size=6.0, ha="left", fontweight="bold")
    for i, (key, name) in enumerate(MOD_NAME.items()):
        y = top - 0.20 - i * 0.135
        c = MOD[key]
        box(ax, x + 0.06, y - 0.04, 0.11, 0.08, tint(c, 0.10) if key != "pred" else PRED_LIGHT, c, r=0.02, lw=0.8,
            z=2, ls=(0, (2, 1.5)) if key == "meas" else "-")
        label(ax, x + 0.21, y, name, size=4.8, ha="left")


VALS = {}


def toy_vals(G=8):
    v = {"ctl": rng.normal(0, 1, G), "trt": rng.normal(0, 1, G), "ctx": rng.normal(0, 1, G) * 1.4}
    v["dlt"] = v["ctx"] + rng.normal(0, 1, G)
    v["res"] = v["dlt"] - v["ctx"]
    for key in ("g", "h"):
        for i in (0, 2):
            v[f"res_{key}_{i}"] = rng.normal(0, 1, G)
    v["g"] = rng.normal(0, 1, G); v["h"] = v["g"] + 0.5 * rng.normal(0, 1, G); v["u"] = v["g"] + v["h"]
    v["pcs"] = rng.normal(0, 1, 6)
    v["resp"] = v["ctx"] * 0.5 + v["u"]
    return v


# ---------------------------------------------------------------- assembly
H = 4.60
GUT = 0.08                                   # gutter between modules
KMX, KMW = 0.03 + CW + 0.12 + GUT, 1.66      # similarity-kernels module
XL = KMX + KMW + GUT                         # left edge of the fusion module
XF = XL + 0.12 + WCOL + 0.20 + FS / 2        # normalised kernel's centre = similarity column of the regression
W = XL + FWID + 0.03                         # canvas width: the right edge follows the fusion module


def assemble(S, vals):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    top = H - 0.03
    # compound information: cards and view kernels
    cx0, cw = 0.03, CW + 0.12
    ends, cbot = part_cards(ax, cx0 + 0.06, top - 0.17, S)
    module(ax, cx0, cbot - 0.05, cw, top - cbot + 0.05, "Compound information", "info")
    kmx, kmw = KMX, KMW
    kmb = 2.86
    module(ax, kmx, kmb, kmw, top - kmb, "Similarity kernels", "info")
    kx, ky = kmx + 0.17, kmb + 0.20
    (kfx, kfy), kright = part_kernels(ax, kx, ky, S)
    path(ax, [(cx0 + cw, kfy), (kmx, kfy)], color=FLOW)                    # compound information -> kernels
    # fusion
    (qx, qy), (fl, fm) = part_fusion(ax, XL, kmb + FHGT, S)          # bottom-aligned with the kernels module
    path(ax, [(kmx + kmw, fm), (fl, fm)], color=FLOW)                      # kernels -> fusion, at matrix height
    # context lane and measured responses
    yT = cbot - 0.05 - GUT - 0.39; yD, yB = 0.98, 0.40     # measured module sits one gutter below the cards
    A = part_context(ax, 0.10, yT, yD, yB, vals)
    module(ax, 0.03, yT - 0.22, A["meas"][1] - 0.03, 0.61,
           "Measured responses for profiled compounds", "meas", dashed=True)
    module(ax, 0.03, 0.06, A["ctx_box"][1] - 0.03, yD + 0.40 - 0.06, "Context", "ctx", tx=0.42)
    # kernel regression
    tx = 3.50
    T = part_transfer(ax, tx, yT + 0.25 + 0.47 + 0.13 + (SH + 0.03) / 2, XF, vals, S)
    path(ax, [(qx, qy), (qx, T["w0"][0] + 0.003)], color=OURS, lw=0.75)          # query row -> similarity weights
    ax.plot([qx, qx], [T["w0"][1], T["w1"][0]], color=OURS, lw=0.9, zorder=3)
    rx, ry = A["res"]; br = tx - 0.16                                           # residual -> per line and shared
    path(ax, [(rx, ry), (tx, T["in1"][1])], color=FLOW)
    path(ax, [(br, ry), (br, T["in0"][1]), (tx, T["in0"][1])], color=FLOW)
    dot(ax, br, ry, INK2)
    # output: control PCs, interaction and compound effect in one row inside the prediction module; the response
    # sits right of the module, under the regression's sum
    VALS.update(vals)
    px = A["ctx_branch"] + 0.10               # module's left edge: right of the context -> residual line
    pcx, xi, xu = 2.90, 3.86, 4.86
    pr = xu + BW / 2 + 0.10                   # module's right edge
    xr = W - 0.03 - BW / 2
    C = part_output(ax, T, xu, xi, yD, yB, xr)
    # control PCs: their own box on the DMSO line, then straight into the interaction
    small_box(ax, pcx, yD, "Control PCs", "ctx", h=0.26)
    strip(ax, pcx - 0.17, yD - 0.10, 0.34, SH, vals["pcs"], CTX, z=4.5)
    path(ax, [A["dmso"], (pcx - 0.33, yD)], color=FLOW, z=2.6)
    path(ax, [(pcx + 0.33, yD), C["x"]], color=FLOW)
    path(ax, [A["ctx"], C["p"]], color=FLOW)
    # prediction module: control PCs, interaction, compound effect and + context; same row as the context module
    module(ax, px, 0.06, pr - px, yD + 0.40 - 0.06, "Prediction", "pred")
    return fig


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = Path(args[0]) if args else FIGDIR
    S = json.loads((src / "fig1_pipeline_source.json").read_text())
    out = Path(args[1]) if len(args) > 1 else None
    save(assemble(S, toy_vals()), "fig1_pipeline", out)


if __name__ == "__main__":
    main()
