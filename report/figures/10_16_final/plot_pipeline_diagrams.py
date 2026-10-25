#!/usr/bin/env python3
"""Four slide-ready scientific diagrams, drawn with matplotlib only.

Run: python3 plot_pipeline_diagrams.py [--output-dir PATH]
Default output: beside this script. Each PNG is exactly 3200 x 1800 pixels.
The original Rhaister artists are reused unchanged in the additive version.
Text containment, canvas bounds and box overlap are checked before saving.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

BG = "#f7f8fa"
INK = "#16202b"
SECONDARY = "#4a5a69"
STYLES = {
    "neutral": ("white", "#c4ccd5"),
    "blue": ("#e6f0f6", "#0070a6"),
    "orange": ("#fbeee6", "#b5531f"),
    "teal": ("#e3eef0", "#1f6f78"),
    "grey": ("#eef1f4", "#8a97a4"),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                     "text.color": INK, "mathtext.fontset": "dejavusans"})


def canvas():
    fig = plt.figure(figsize=(16, 9), dpi=200, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set(xlim=(0, 16), ylim=(0, 9))
    ax.axis("off")
    ax._panels = []
    ax._contained = []
    return fig, ax


def label(ax, x, y, text, size=13, color=SECONDARY, weight="normal",
          ha="left", va="center", **kwargs):
    return ax.text(x, y, text, fontsize=size, color=color,
                   fontweight=weight, ha=ha, va=va, linespacing=1.22,
                   zorder=5, **kwargs)


def box(ax, x, y, w, h, title, body="", style="blue", title_size=16,
        badge=None):
    fill, edge = STYLES[style]
    patch = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0,rounding_size=0.13",
                          linewidth=1.4, edgecolor=edge, facecolor=fill,
                          zorder=3)
    ax.add_patch(patch)
    ax._panels.append((x, y, w, h, title))
    left = x + (0.58 if badge else 0.20)
    title_artist = label(ax, left, y+h-0.18, title, title_size, INK,
                         "bold", va="top")
    ax._contained.append((title_artist, (x, y, w, h), title))
    if body:
        title_lines = title.count("\n") + 1
        title_height = title_lines * title_size / 72 * 1.22
        body_artist = label(ax, x+0.20, y+h-0.18-title_height-0.12,
                            body, va="top")
        ax._contained.append((body_artist, (x, y, w, h), title))
    if badge:
        cx, cy = x+0.29, y+h-0.30
        ax.add_patch(Circle((cx, cy), 0.18, facecolor=edge,
                            edgecolor=edge, zorder=5))
        label(ax, cx, cy, str(badge), 13, "white", "bold", ha="center")
    return (x, y, w, h)


def arrow(ax, points, color=None, dashed=False):
    """A routed arrow with explicit ports; corners do not cut through boxes."""
    color = color or STYLES["blue"][1]
    for i in range(len(points)-1):
        ax.add_patch(FancyArrowPatch(
            points[i], points[i+1], arrowstyle="-|>" if i == len(points)-2 else "-",
            mutation_scale=18, linewidth=1.5, color=color,
            linestyle=(0, (4, 3)) if dashed else "-", shrinkA=0, shrinkB=0,
            capstyle="round", zorder=2))


def draw_rhaister_base(ax):
    label(ax, 0.40, 6.22, "Rhaister-O", 16, STYLES["blue"][1], "bold")
    box(ax, 0.40, 3.25, 2.90, 2.75, "Reference cell lines\n(45)",
        "Tahoe-100M · 24 h\n2,000 genes\nuntreated profile x(c, g)\ndrug response Δ(c, t, g)")
    box(ax, 3.90, 4.60, 4.55, 1.40, "Fit per drug-dose t, gene g",
        "Ridge across 45 lines, equal weights:\n"
        r"$\Delta(c,t,g)\approx\mu(t,g)+\gamma(t,g)\,x(c,g)$")
    box(ax, 4.20, 3.20, 4.25, 1.05, "New cell line q (never treated)",
        "Input: Tahoe DMSO profile x(q, g)")
    box(ax, 9.05, 3.50, 3.50, 2.15, "Predict",
        r"$\hat\Delta(q,t,g)=\mu(t,g)$" "\n"
        r"$\quad+\gamma(t,g)\,x(q,g)$" "\n"
        "1,137 drug-doses\n× 2,000 genes")
    box(ax, 13.15, 3.50, 2.45, 2.15, "Score",
        "Pearson: predicted\nvs. measured Δ\nper drug-dose;\naverage over lines")
    arrow(ax, [(3.30, 5.25), (3.90, 5.25)])
    arrow(ax, [(8.45, 5.25), (9.05, 5.25)])
    arrow(ax, [(8.45, 3.72), (8.75, 3.72), (8.75, 4.12), (9.05, 4.12)])
    arrow(ax, [(12.55, 4.58), (13.15, 4.58)])


def draw_our_additions(ax):
    box(ax, 0.40, 6.80, 5.95, 1.85, "Reliability weights",
        "× q(c)ᵅ: how well line c's own DMSO\n"
        "reproduces across 14 plates\n(no drug data used)", "orange", badge=2)
    box(ax, 6.80, 6.80, 8.80, 1.85, "Similarity weights",
        "Weight reference c by untreated-profile similarity to q: w = exp(−d(q,c)/h)\n"
        "Refit μ and γ for each q\n→ slopes depend on the new cell line", "orange", badge=1)
    orange = STYLES["orange"][1]
    arrow(ax, [(3.70, 6.80), (3.70, 6.50), (5.15, 6.50), (5.15, 6.00)], orange, True)
    arrow(ax, [(10.50, 6.80), (10.50, 6.35), (7.45, 6.35), (7.45, 6.00)], orange, True)
    box(ax, 0.40, 0.65, 7.60, 2.05, "DepMap bridge",
        "No Tahoe data for q? Fit each gene g on reference lines:\n"
        "Tahoe(g) = a(g) + b(g) · DepMap(g)\n"
        "→ pseudo untreated profile x(q, g)", "orange", badge=3)
    box(ax, 8.65, 0.65, 6.95, 2.05, "Evaluation",
        "Drug average removed · fresh never-scored lines\n"
        "Second atlas: Emerald Bay\nGate set before each run", "teal", badge=4)
    arrow(ax, [(6.30, 2.70), (6.30, 3.20)], orange, True)
    arrow(ax, [(14.35, 2.70), (14.35, 3.50)], orange, True)
    for x, text, style in [(5.80, "Original Rhaister-O", "blue"),
                            (10.15, "Our addition", "orange"),
                            (13.48, "Evaluation", "teal")]:
        fill, edge = STYLES[style]
        ax.add_patch(Circle((x, 0.24), 0.07, facecolor=fill, edgecolor=edge, zorder=4))
        label(ax, x+0.17, 0.24, text, 13)


def draw_rhaister_original():
    fig, ax = canvas()
    draw_rhaister_base(ax)
    return fig, ax


def draw_rhaister_ours():
    fig, ax = canvas()
    draw_rhaister_base(ax)
    draw_our_additions(ax)
    return fig, ax


def bracket(ax, bottom, top, name, color):
    ax.plot([1.93, 1.78, 1.78, 1.93], [top, top, bottom, bottom],
            color=color, linewidth=2, zorder=3)
    label(ax, 0.42, (top+bottom)/2, name, 13, color, "bold")


def draw_mkl():
    fig, ax = canvas()
    label(ax, 0.40, 8.53,
          "Multiple-kernel ridge — task: unseen drug on LINCS L1000 (978 genes, 7 cell lines)",
          16, INK, "bold")
    rows = [
        (7.20, "Chemical structure", "Morgan fingerprint · Tanimoto"),
        (6.22, "Drug targets", "Target-set overlap"),
        (5.30, "Mechanism of action", "Same MOA class"),
        (4.38, "Target degree", "How connected the targets are"),
        (3.46, "One-hop PPI", "Targets expanded over STRING"),
        (2.36, "Cell Painting", "Image features · A549 · 48 h"),
        (1.26, "Response in other lines", "Drug's L1000 signature elsewhere"),
    ]
    for y, title, body in rows:
        box(ax, 2.18, y, 4.42, 0.87, title, body, title_size=16)
        arrow(ax, [(6.60, y+0.435), (7.00, y+0.435)])
    bracket(ax, 7.20, 8.07, "Tier 1\nchemistry", "#0070a6")
    bracket(ax, 3.46, 7.09, "Tier 2\nknowledge", "#426594")
    bracket(ax, 2.36, 3.23, "Tier 3\nmorphology", "#80639d")
    bracket(ax, 1.26, 2.13, "Tier 4\nmeasured\nresponse", "#1f6f78")
    ax.plot([7.00, 7.00], [1.66, 7.60], color=STYLES["blue"][1], linewidth=1.5, zorder=2)
    label(ax, 7.32, 7.61, "Each view → drug × drug kernel", 13)
    arrow(ax, [(7.00, 6.33), (7.55, 6.33)])
    box(ax, 7.55, 5.45, 3.80, 1.80, "Combine kernels",
        r"$K=\sum_v w_v\,K_v$" "\n\n"
        "Weights from kernel–target\nalignment")
    arrow(ax, [(9.45, 5.45), (9.45, 4.73)])
    box(ax, 7.55, 3.05, 3.80, 1.68, "Kernel ridge, per cell line",
        "λ chosen by grouped\ninner cross-validation")
    arrow(ax, [(11.35, 3.88), (12.05, 3.88)])
    box(ax, 12.05, 3.05, 3.55, 1.68, "Predicted signature\nof a new drug",
        "LINCS L1000 · 978 genes", "teal")
    box(ax, 7.55, 0.90, 5.75, 1.55, "Null controls",
        "Targets permuted across drugs\nPPI rewired; node degrees kept", "grey")
    label(ax, 0.42, 0.55,
          "Cumulative tiers: T1 = chemistry; T2 = T1 + knowledge; T3 = T2 + morphology; T4 = T3 + measured response.",
          13)
    return fig, ax


def draw_e2e_model():
    fig, ax = canvas()
    label(ax, 0.40, 8.63,
          "Integrated end-to-end model (in progress) — task: unseen drug, LINCS L1000",
          16, INK, "bold")
    box(ax, 0.40, 7.00, 4.10, 1.05, "Compound",
        "Morgan fingerprint · 2,048 bits", "neutral")
    box(ax, 0.40, 5.12, 4.10, 1.60, "PrimeKG subgraph",
        "32,024 nodes · 939,170 edges\n"
        "PPI · drug–disease\ndisease–gene · drug–target", "neutral")
    box(ax, 0.40, 3.71, 4.10, 1.10, "Disease text",
        "SapBERT name +\nPubMedBERT definition", "neutral")
    box(ax, 0.40, 2.35, 4.10, 1.03, "Landmark genes",
        "978 genes · learned embedding", "neutral")
    box(ax, 5.50, 6.65, 4.20, 1.40, "Drug encoder",
        "MLP 2048 → 512 → 128\nLayerNorm")
    box(ax, 5.50, 3.97, 4.20, 2.18, "RGCN over PrimeKG",
        "2 layers · residual + LayerNorm\n"
        "+ 0.1 × link-prediction loss\n\n"
        "Outputs: drug vector $h_d$\n              gene vector $e_g$")
    box(ax, 11.00, 7.02, 4.60, 1.03, "Cell line $l$",
        "One decoder matrix $W_l$ per line", "neutral")
    box(ax, 11.00, 5.12, 4.60, 1.55, "Bilinear decoder, per line",
        r"$\hat\Delta(d,l,g)=\langle W_l h_d,e_g\rangle/\sqrt{128}$")
    box(ax, 11.00, 3.50, 4.60, 1.20, "+ KRR anchor",
        "Kernel-ridge prediction as fixed offset;\nnetwork fits the remainder", "orange")
    box(ax, 11.00, 2.02, 4.60, 1.03, "Predicted signature",
        "MSE on Level-5 z-scores", "teal")
    arrow(ax, [(4.50, 7.52), (5.50, 7.52)])
    arrow(ax, [(4.50, 5.78), (5.50, 5.78)])
    arrow(ax, [(4.50, 4.23), (5.50, 4.23)])
    arrow(ax, [(4.50, 2.86), (6.60, 2.86), (6.60, 3.97)])
    arrow(ax, [(7.60, 6.65), (7.60, 6.15)])
    arrow(ax, [(9.70, 7.33), (10.24, 7.33), (10.24, 6.17), (11.00, 6.17)])
    label(ax, 10.34, 7.62, "$h_d$", 13, ha="center")
    arrow(ax, [(9.70, 5.08), (10.53, 5.08), (10.53, 5.60), (11.00, 5.60)])
    label(ax, 10.34, 4.80, "$h_d$, $e_g$", 13, ha="center")
    arrow(ax, [(13.30, 7.02), (13.30, 6.67)])
    arrow(ax, [(13.30, 5.12), (13.30, 4.70)])
    arrow(ax, [(13.30, 3.50), (13.30, 3.05)])
    # One bottom strip, with four readable columns for the model arms.
    strip = box(ax, 0.40, 0.30, 15.20, 1.45, "Arms and controls", style="grey")
    arms = [
        (0.62, "skel", "No KG"),
        (3.45, "kg_a", "PPI + drug–disease\n+ disease–gene"),
        (7.75, "kg_t", "kg_a + drug–target"),
        (11.70, "null", "Degree-preserving\nrewired KG"),
    ]
    for x, name, body in arms:
        a = label(ax, x, 1.12, name, 13, INK, "bold", va="top")
        b = label(ax, x, 0.85, body, 13, va="top")
        ax._contained.extend([(a, strip, "Arms and controls"), (b, strip, "Arms and controls")])
    return fig, ax


def validate(fig, ax):
    """Check rendered text, not estimates; fail loudly on a layout regression."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = []
    for artist, (x, y, w, h), title in ax._contained:
        bb = artist.get_window_extent(renderer)
        lo = ax.transData.transform((x+0.06, y+0.06))
        hi = ax.transData.transform((x+w-0.06, y+h-0.06))
        if bb.x0 < lo[0] or bb.y0 < lo[1] or bb.x1 > hi[0] or bb.y1 > hi[1]:
            issues.append(f"Text outside {title!r}: {artist.get_text()!r}")
    for i, a in enumerate(ax._panels):
        for b in ax._panels[i+1:]:
            x, y, w, h, title = a
            xx, yy, ww, hh, other = b
            if min(x+w, xx+ww)-max(x, xx) > 1e-6 and min(y+h, yy+hh)-max(y, yy) > 1e-6:
                issues.append(f"Overlapping boxes: {title!r}, {other!r}")
    text_bounds = []
    for artist in ax.texts:
        assert artist.get_fontsize() >= 13
        bb = artist.get_window_extent(renderer)
        if not (0 <= bb.x0 and 0 <= bb.y0 and bb.x1 <= fig.bbox.width and bb.y1 <= fig.bbox.height):
            issues.append(f"Text clipped at canvas edge: {artist.get_text()!r}")
        for old_artist, old_bb in text_bounds:
            if bb.overlaps(old_bb):
                issues.append(f"Text collision: {old_artist.get_text()!r}, {artist.get_text()!r}")
        text_bounds.append((artist, bb))
    if issues:
        raise RuntimeError("\n".join(issues))


def base_signature(ax):
    """Record immutable geometry, content, and colors of all base artists."""
    return ([tuple(p) for p in ax._panels],
            [(t.get_position(), t.get_text(), t.get_fontsize(), t.get_color()) for t in ax.texts],
            [(type(p).__name__, tuple(p.get_facecolor()), tuple(p.get_edgecolor()),
              p.get_path().vertices.tolist()) for p in ax.patches])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The base is identical before any additive artists are drawn.
    original, original_ax = draw_rhaister_original()
    ours, ours_ax = canvas()
    draw_rhaister_base(ours_ax)
    assert base_signature(original_ax) == base_signature(ours_ax)
    draw_our_additions(ours_ax)
    figures = [
        ("pipeline_rhaister_original.png", original, original_ax),
        ("pipeline_rhaister_ours.png", ours, ours_ax),
        ("pipeline_mkl.png", *draw_mkl()),
        ("pipeline_e2e_model.png", *draw_e2e_model()),
    ]
    for name, fig, ax in figures:
        validate(fig, ax)
        path = args.output_dir / name
        fig.savefig(path, dpi=200, facecolor=BG)
        print(f"{path} — {int(fig.bbox.width)} × {int(fig.bbox.height)} px")
        plt.close(fig)


if __name__ == "__main__":
    main()
