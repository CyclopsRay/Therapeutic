"""Shared style for the UnDetect paper figures: palette, fonts, font gate and saving.

Palette roles (dataviz reference palette, light mode) are fixed across figures: gray = context / drug-free, blue =
UnDetect and its compound lane, orange = the plate-response baseline, view colours as in Fig. 1.
Font gate (GeneralAgent references/figures.md): set FIG_FONT to a licensed Arial .ttf. Without it the figures are
drawn in DejaVu Sans and saved with a `_preview` suffix, so a draft cannot pass as final.
save() writes PDF + SVG (editable text) + PNG (300 dpi) and, when given, the plotted numbers as a CSV source table.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "paper" / "figures"

INK, INK2, MUTED, LINE, GRID, PANEL = "#0b0b0b", "#52514e", "#8a8984", "#c3c2b7", "#e4e3df", "#f3f2ef"
CTX, CTX_LIGHT = "#6f6e69", "#e4e3df"          # context lane, drug-free models
OURS, OURS_LIGHT = "#2a78d6", "#dbe8f8"        # UnDetect, compound lane
PLATE = "#eb6834"                              # plate-response baseline
VIEW = {"chem": "#1baf7a", "tgt": "#eda100", "moa": "#e87ba4", "morph": "#6250d6"}
DIV_NEG, DIV_POS = "#2a78d6", "#e34948"        # diverging pair for signed expression, gray midpoint


def setup() -> bool:
    """Apply rcParams; return True when the licensed font passed the gate."""
    path = os.environ.get("FIG_FONT")
    final = False
    if path and Path(path).exists():
        font_manager.fontManager.addfont(path)
        fam = font_manager.FontProperties(fname=path).get_name()
        final = fam.lower().startswith("arial")
        plt.rcParams["font.sans-serif"] = [fam]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif", "mathtext.fontset": "dejavusans", "font.size": 9,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
    })
    return final


FINAL = setup()


def basename(name: str) -> str:
    return name if FINAL else f"{name}_preview"


def save(fig, name: str, outdir: Path | None = None, table=None) -> list[Path]:
    """Write <name>[_preview].{pdf,svg,png}; `table` (a DataFrame) goes to <name>_source.csv."""
    out = Path(outdir) if outdir else FIGDIR
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "svg", "png"):
        p = out / f"{basename(name)}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        paths.append(p)
    if table is not None:
        p = out / f"{name}_source.csv"
        table.to_csv(p, index=False)
        paths.append(p)
    print("wrote", *paths, "(font gate passed)" if FINAL else "(preview: FIG_FONT not set)")
    return paths


def panel(ax, letter: str, x: float = -0.12) -> None:
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=11, x=x)
