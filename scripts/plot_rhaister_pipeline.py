"""Rhaister-O pipeline, original and ours, for the course final presentation.

Both figures come from the same base drawing; ours only adds elements in space the base
leaves empty, so nothing in the original moves or changes. Notation follows the midterm
deck: x = untreated profile, y = drug response, c* = new cell line, p = drug-dose, g = gene.

Run: PYTHONPATH=<dir with matplotlib> python3 scripts/plot_rhaister_pipeline.py
Outputs: report/figures/10_16_final/pipeline_rhaister_{original,ours}.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "report" / "figures" / "10_16_final"
BG, INK, INK2 = "#f7f8fa", "#16202b", "#4a5a69"
BLUE, BLUE_F = "#0070a6", "#e6f0f6"
ORANGE, ORANGE_F = "#b5531f", "#fbeee6"
PRED_F = "#d7e7f2"


def box(ax, x0, y0, x1, y1, edge, fill, lw=2.0):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.18",
                                linewidth=lw, edgecolor=edge, facecolor=fill))


def arrow(ax, a, b, color, dashed=False, lw=2.6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=26, color=color, lw=lw,
                                 linestyle=(0, (5, 4)) if dashed else "-", shrinkA=0, shrinkB=0))


def text(ax, x, y, s, size, color=INK, weight="normal", ha="left", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va)


def badge(ax, x, y, n):
    ax.add_patch(plt.Circle((x, y), 0.27, color=ORANGE))
    text(ax, x, y, str(n), 19, "#ffffff", "bold", ha="center")


def base(ax):
    # 45 reference lines
    box(ax, 0.3, 7.8, 5.4, 9.5, BLUE, BLUE_F)
    text(ax, 0.6, 9.0, "45 reference cell lines", 24, weight="bold")
    text(ax, 0.6, 8.25, "untreated profile x\ndrug response y", 19, INK2)
    # new line
    box(ax, 6.6, 7.8, 11.7, 9.5, BLUE, BLUE_F)
    text(ax, 6.9, 9.0, "New cell line c*", 24, weight="bold")
    text(ax, 6.9, 8.25, "never treated:\nonly its untreated profile x", 19, INK2)
    # fit
    box(ax, 0.3, 4.3, 5.4, 7.1, BLUE, BLUE_F)
    text(ax, 0.6, 6.65, "Fit, per drug p and gene g", 22, weight="bold")
    text(ax, 0.6, 5.85, r"$\min_{\beta,\gamma}\ \sum_{c}\,(y_{c,p,g}-\beta-\gamma\,x_{c,g})^2$", 22)
    text(ax, 0.6, 5.15, "every reference line counts equally", 18, INK2)
    # predict (emphasised)
    box(ax, 0.3, 0.3, 11.7, 3.6, BLUE, PRED_F, lw=3.2)
    text(ax, 0.6, 3.15, "Predict the new line's response", 22, weight="bold")
    text(ax, 6.0, 2.3, r"$\hat{y}_{c^*,p,g} \;=\; \hat{\beta}_{p,g} \;+\; \hat{\gamma}_{p,g}\; x_{c^*,g}$",
         40, ha="center")
    # flow
    arrow(ax, (2.85, 7.8), (2.85, 7.1), BLUE)
    arrow(ax, (2.85, 4.3), (2.85, 3.6), BLUE)
    arrow(ax, (11.4, 7.8), (11.4, 3.6), BLUE)


def additions(ax):
    # 1 similarity, 2 reliability: right of the fit box
    box(ax, 5.9, 5.85, 11.05, 7.1, ORANGE, ORANGE_F)
    badge(ax, 6.3, 6.75, 1)
    text(ax, 6.75, 6.75, "Cell-line-specific slopes", 20, weight="bold")
    text(ax, 6.3, 6.15, "fit again for each new line,\nsimilar lines count more", 17, INK2)
    box(ax, 5.9, 4.3, 11.05, 5.55, ORANGE, ORANGE_F)
    badge(ax, 6.3, 5.2, 2)
    text(ax, 6.75, 5.2, "Trust reliable lines", 20, weight="bold")
    text(ax, 6.3, 4.6, "lines with irreproducible\ncontrols count less", 17, INK2)
    arrow(ax, (5.9, 6.45), (5.4, 6.45), ORANGE, dashed=True)
    arrow(ax, (5.9, 4.9), (5.4, 4.9), ORANGE, dashed=True)
    text(ax, 0.6, 4.6, "→ weighted, refitted per new line", 18, ORANGE, "bold")
    # 3 DepMap bridge: above the new line
    box(ax, 6.6, 9.8, 11.7, 10.85, ORANGE, ORANGE_F)
    badge(ax, 7.0, 10.5, 3)
    text(ax, 7.45, 10.5, "New baseline dataset", 20, weight="bold")
    text(ax, 7.0, 10.05, "DepMap → Tahoe, gene by gene", 17, INK2)
    arrow(ax, (9.15, 9.8), (9.15, 9.5), ORANGE, dashed=True)
    # predict: what changes
    text(ax, 6.0, 1.3, r"$\hat{y}_{c^*,p,g} \;=\; \hat{\beta}_{p,g}(c^*) \;+\; \hat{\gamma}_{p,g}(c^*)\; x_{c^*,g}$",
         32, ORANGE, ha="center")
    text(ax, 6.0, 0.6, "β and γ now depend on the new cell line", 16, ORANGE, ha="center")


def draw(name, ours):
    fig, ax = plt.subplots(figsize=(12, 11))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11.1)
    ax.axis("off")
    base(ax)
    if ours:
        additions(ax)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    fig.savefig(OUT / name, dpi=200, facecolor=BG)  # same canvas for both, so positions match
    plt.close(fig)
    print(OUT / name)


if __name__ == "__main__":
    draw("pipeline_rhaister_original.png", ours=False)
    draw("pipeline_rhaister_ours.png", ours=True)
