#!/usr/bin/env python3
"""Figures for report/10_09 §9-§10 (DepMap<->Tahoe CCA).

Fig 1  first two canonical components: all-line fit (descriptive) and per-split
       held-out projections.
Fig 2  DepMap x Tahoe line-by-line similarity: direct gene-matched correlation
       and leave-one-line-out CCA cosine.
Fig 3  identity retrieval ranks per held-out line.
Fig 4  gene-space response arms (results/depmap_tahoe_cca).
Fig 5  shared-space arms, absolute vs delta (results/depmap_tahoe_cca_space).
Fig 6  direct gene-matched space: one PCA of both datasets' standardised profiles,
       same line joined across datasets (the counterpart of Fig 1's CCA view).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from run_depmap_expression_ridge import DEFAULT_DEPMAP_ROOT, load_depmap_expression
from run_depmap_rhaister_correction import line_hvg_vectors
from run_depmap_tahoe_cca import Cca, gene_symbol
from run_rhaister_o_compact import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    build_or_load_compact_cache,
    load_hvg_centroids,
    load_split,
)

RESULTS = Path("/nfs-stor/yuqi.lei/Therapeutic/results")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
TAHOE, DEPMAP = "#2a78d6", "#eb6834"  # categorical slots 1, 2
DIVERGING = LinearSegmentedColormap.from_list(
    "bluegrayred", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a92f2f"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def load_inputs(args):
    repo_root = args.project_root / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    expression, cvcl_to_model, metadata, _, _ = load_depmap_expression(args.depmap_root)
    hvg = load_hvg_centroids(args.data_root, compact.genes)
    cells = sorted(c for c in set(compact.cells.tolist()) if cvcl_to_model.get(c) in expression.index)
    cells.sort(key=lambda c: (metadata[c]["lineage"], metadata[c]["cell_line_name"]))
    x = np.nan_to_num(expression.loc[[cvcl_to_model[c] for c in cells]].to_numpy(np.float64))
    y = line_hvg_vectors(compact, hvg, cells)
    return compact, expression, metadata, cells, x, y, repo_root


def connected_scatter(ax, u, v, highlight=None, names=None):
    """u = DepMap CC coords, v = Tahoe CC coords (lines x 2)."""
    rest = np.ones(len(u), bool) if highlight is None else ~highlight
    for i in np.flatnonzero(rest):
        ax.plot([u[i, 0], v[i, 0]], [u[i, 1], v[i, 1]], color=GRID if highlight is not None else "#c3c2b7",
                lw=0.8, zorder=1)
    faint = 0.25 if highlight is not None else 1.0
    ax.scatter(v[rest, 0], v[rest, 1], s=22, c=TAHOE, alpha=faint, edgecolors=SURFACE, linewidths=0.8,
               zorder=2, label="Tahoe DMSO")
    ax.scatter(u[rest, 0], u[rest, 1], s=26, c=DEPMAP, marker="^", alpha=faint, edgecolors=SURFACE,
               linewidths=0.8, zorder=2, label="DepMap expression")
    if highlight is not None:
        for i in np.flatnonzero(highlight):
            ax.plot([u[i, 0], v[i, 0]], [u[i, 1], v[i, 1]], color=INK2, lw=1.2, zorder=3)
        ax.scatter(v[highlight, 0], v[highlight, 1], s=46, c=TAHOE, edgecolors=SURFACE, linewidths=1.5, zorder=4)
        ax.scatter(u[highlight, 0], u[highlight, 1], s=52, c=DEPMAP, marker="^", edgecolors=SURFACE,
                   linewidths=1.5, zorder=4)
        for i in np.flatnonzero(highlight):
            ax.annotate(names[i], (v[i, 0], v[i, 1]), xytext=(4, 4), textcoords="offset points",
                        fontsize=7.5, color=INK)
    ax.set_xlabel("CC1")
    ax.set_ylabel("CC2")


def fig1(args, cells, x, y, metadata, repo_root):
    names = [metadata[c]["cell_line_name"] for c in cells]
    split_info = {s["split"]: s for s in json.load(open(RESULTS / "depmap_tahoe_cca" / "summary.json"))["per_split"]}
    fig = plt.figure(figsize=(13, 7.2))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.5, 1, 1, 1], wspace=0.42, hspace=0.5)

    model = Cca(x, y, 10, 10)
    u, v = model.embed_x(x)[:, :2], model.embed_y(y)[:, :2]
    ax = fig.add_subplot(grid[:, 0])
    connected_scatter(ax, u, v)
    for i in np.argsort(-np.linalg.norm(v, axis=1))[:4]:
        ax.annotate(names[i], (v[i, 0], v[i, 1]), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_title(f"All 47 lines, fit on all (in-sample)\nρ1 = {model.rho[0]:.2f}, ρ2 = {model.rho[1]:.2f}; segment = one line", loc="left")
    ax.legend(loc="upper right", fontsize=8)

    for n, split in enumerate(range(5, 10)):
        holdout, _, _ = load_split(repo_root, split)
        test = np.array([c in holdout for c in cells])
        info = split_info[split]
        m = Cca(x[~test], y[~test], info["k"], info["d"])
        ax = fig.add_subplot(grid[n // 3, 1 + n % 3] if n < 3 else grid[1, 1 + (n - 3)])
        connected_scatter(ax, m.embed_x(x)[:, :2], m.embed_y(y)[:, :2], highlight=test, names=names)
        ax.set_title(f"Split {split}: held-out lines (dark)\nρ1 in-sample {m.rho[0]:.2f}", loc="left", fontsize=9)
    fig.suptitle("First two canonical components, DepMap 24Q4 log-TPM ↔ Tahoe mean log1p DMSO (2,000 genes)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.005, "Right panels: CCA fitted on that split's reference lines only (k, d as selected); held-out "
             "lines projected. Short dark segment = the line's DepMap and Tahoe points land together.",
             fontsize=8, color=INK2)
    fig.savefig(args.out / "fig1_cca_first_two_components.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def lineage_axis(ax, cells, metadata):
    lineages = [metadata[c]["lineage"] for c in cells]
    bounds = [i for i in range(1, len(cells)) if lineages[i] != lineages[i - 1]]
    for b in bounds:
        ax.axhline(b - 0.5, color=SURFACE, lw=1.6)
        ax.axvline(b - 0.5, color=SURFACE, lw=1.6)
    starts = [0] + bounds
    ends = bounds + [len(cells)]
    centres = [(s + e - 1) / 2 for s, e in zip(starts, ends)]
    labels = [lineages[s] if e - s >= 2 else "" for s, e in zip(starts, ends)]
    ax.set_xticks(centres, labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(centres, labels, fontsize=7)
    ax.grid(False)


def fig2(args, cells, x, y, expression, metadata):
    symbols = np.array([gene_symbol(c) for c in expression.columns])
    xcol = {s: i for i, s in enumerate(symbols)}
    tahoe_genes = list(load_genes(args))
    shared = [g for g in tahoe_genes if g in xcol]
    xs = x[:, [xcol[g] for g in shared]]
    ys = y[:, [tahoe_genes.index(g) for g in shared]]
    def rowz(a):
        a = (a - a.mean(0)) / (a.std(0) + 1e-8)
        a = a - a.mean(1, keepdims=True)
        return a / np.linalg.norm(a, axis=1, keepdims=True)
    direct = rowz(xs) @ rowz(ys).T

    loo = np.zeros_like(direct)
    for i in range(len(cells)):
        keep = np.arange(len(cells)) != i
        m = Cca(x[keep], y[keep], 10, 10)
        q = m.embed_x(x[i:i + 1])
        g = m.embed_y(y)
        loo[i] = (g @ q[0]) / (np.linalg.norm(g, axis=1) * np.linalg.norm(q) + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw={"wspace": 0.42})
    for ax, mat, title in [
        (axes[0], direct, f"Direct: Pearson on {len(shared):,} matched genes\n(genes standardised over all 47 lines; no fitting)"),
        (axes[1], loo, "CCA space, leave-one-line-out\n(row line excluded from its own fit, k = d = 10; cosine)"),
    ]:
        lim = np.abs(mat).max()
        im = ax.imshow(mat, cmap=DIVERGING, vmin=-lim, vmax=lim, interpolation="nearest")
        lineage_axis(ax, cells, metadata)
        top1 = np.mean(mat.argmax(1) == np.arange(len(cells)))
        rank = 1 + (mat > np.diag(mat)[:, None]).sum(1)
        ax.set_title(f"{title}\nown line is best match: {top1:.0%} of rows; mean rank {rank.mean():.1f} / {len(cells)}",
                     loc="left", fontsize=9)
        ax.set_xlabel("Tahoe DMSO (columns)")
        ax.set_ylabel("DepMap expression (rows)")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
    fig.suptitle("Do the same cell lines match across datasets? Diagonal = same line; blocks = lineage",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(args.out / "fig2_cross_dataset_line_similarity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(direct, index=cells, columns=cells).to_csv(args.out / "fig2_direct_matrix.csv")
    pd.DataFrame(loo, index=cells, columns=cells).to_csv(args.out / "fig2_loo_cca_matrix.csv")


_GENES = None
def load_genes(args):
    return _GENES


def fig3(args, metadata):
    d = pd.read_csv(RESULTS / "depmap_tahoe_cca" / "per_line.csv")
    d["name"] = [metadata[c]["cell_line_name"] for c in d["cell_line"]]
    d = d.sort_values("rank_cca", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 6.2))
    yy = np.arange(len(d))
    ax.hlines(yy, d[["rank_cca", "rank_direct"]].min(1), d[["rank_cca", "rank_direct"]].max(1), color=GRID, lw=2)
    ax.scatter(d["rank_null_mean"], yy, s=24, c=MUTED, marker="|", linewidths=2, label="CCA, shuffled pairing (mean of 50)")
    ax.scatter(d["rank_direct"], yy, s=40, c=DEPMAP, edgecolors=SURFACE, linewidths=1, zorder=3,
               label="Direct gene correlation")
    ax.scatter(d["rank_cca"], yy, s=40, c=TAHOE, edgecolors=SURFACE, linewidths=1, zorder=3, label="CCA")
    ax.axvline(24, color=MUTED, lw=1, ls="--")
    ax.text(24.4, len(d) - 0.6, "chance (24)", fontsize=7.5, color=INK2)
    ax.set_yticks(yy, [f"{n} (split {s})" for n, s in zip(d["name"], d["split"])], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 47], ["1", "2", "5", "10", "20", "47"])
    ax.set_xlabel("Rank of the line's own Tahoe DMSO among 47 lines (1 = found itself; log scale)")
    ax.set_title(f"Identity retrieval for {len(d)} held-out lines: query = DepMap profile\n"
                 f"top-1: CCA {np.mean(d['rank_cca'] == 1):.0%}, direct {np.mean(d['rank_direct'] == 1):.0%}; "
                 f"mean rank: CCA {d['rank_cca'].mean():.1f}, direct {d['rank_direct'].mean():.1f}, "
                 f"shuffled-pairing null {d['rank_null_mean'].mean():.1f}", loc="left")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(axis="y", visible=False)
    fig.savefig(args.out / "fig3_identity_retrieval.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def dot_ci(ax, labels, means, lows, highs, colors, xlabel, ref=None):
    yy = np.arange(len(labels))[::-1]
    ax.hlines(yy, lows, highs, color=colors, lw=2)
    ax.scatter(means, yy, s=48, c=colors, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    for y0, m in zip(yy, means):
        ax.text(m, y0 + 0.22, f"{m:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.set_yticks(yy, labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    if ref is not None:
        ax.axvline(ref, color=MUTED, lw=1)


ARM_LABEL = {
    "real_dmso": "Real Tahoe DMSO (= Rhaister-O)",
    "cca_mapped": "DepMap → CCA-mapped DMSO",
    "nearest_ref_line": "Nearest reference line (DepMap)",
    "shuffled_cca": "Another line's CCA-mapped DMSO",
    "mean_dmso": "Mean DMSO (no line info)",
}


def fig4(args):
    s = pd.read_csv(RESULTS / "depmap_tahoe_cca" / "response_summary.csv")
    order = list(ARM_LABEL)
    colors = [TAHOE, DEPMAP, MUTED, MUTED, INK2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, level, title in [(axes[0], "full", "Full delta"), (axes[1], "drug_residual", "Drug residual (drug mean removed)")]:
        t = s[s["level"].eq(level)].set_index("arm").loc[order]
        dot_ci(ax, [ARM_LABEL[a] for a in order], t["line_macro"], t["low"], t["high"], colors,
               "Line-macro Pearson vs Tahoe truth (2,000 genes; 95% line bootstrap)", ref=0)
        ax.set_title(title, loc="left")
    frac = s[(s["arm"] == "fraction_of_real_gain_recovered_by_cca") & (s["level"] == "drug_residual")]["line_macro"].iloc[0]
    fig.suptitle(f"Frozen Rhaister-O with only the held-out line's DMSO input replaced (23 lines). "
                 f"DepMap via CCA recovers {frac:.0%} of the real-DMSO gain on the drug residual",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", y=1.03)
    fig.savefig(args.out / "fig4_gene_space_dmso_substitution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig5(args):
    s = pd.read_csv(RESULTS / "depmap_tahoe_cca_space" / "summary.csv").set_index(["target", "arm"])
    rt = json.load(open(RESULTS / "depmap_tahoe_cca_space" / "summary.json"))["per_split"]
    rt_mean = np.mean([p["delta_roundtrip_pearson_mean"] for p in rt])
    abs_arms = [("tahoe_dmso", "Rhaister-O, Tahoe DMSO input", TAHOE),
                ("identity_only_tahoe", "Identity only, Tahoe (zero drug effect)", "#86b6ef"),
                ("depmap_shrunk", "Rhaister-O, DepMap input (ρ-shrunk)", DEPMAP),
                ("depmap", "Rhaister-O, DepMap input", DEPMAP),
                ("identity_only_depmap", "Identity only, DepMap (zero drug effect)", "#f3a37f"),
                ("shuffled_depmap", "Rhaister-O, another line's DepMap", MUTED),
                ("drug_mean", "Drug mean (no line info)", INK2)]
    delta_arms = [a for a in abs_arms if not a[0].startswith("identity")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"wspace": 0.75})
    for ax, target, arms, title in [
        (axes[0], "absolute", abs_arms, "Target: absolute post-treatment (projected)"),
        (axes[1], "delta", delta_arms, "Target: delta (projected)"),
    ]:
        t = s.loc[target].loc[[a[0] for a in arms]]
        dot_ci(ax, [a[1] for a in arms], t["pearson"], t["low"], t["high"], [a[2] for a in arms],
               "Pearson over 5–10 canonical coordinates (line macro, 95% CI)", ref=0)
        ax.set_title(title, loc="left")
    fig.suptitle(f"Rhaister-O fitted inside the CCA space. The space keeps little drug response: a held-out "
                 f"delta round-tripped through it correlates {rt_mean:.2f} with the original",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", y=1.03)
    fig.text(0.01, -0.06, "Absolute: Rhaister-O ≈ identity only, so the drug term adds nothing (Δ −0.006 DepMap / −0.005 Tahoe). "
             "Not comparable to gene-space Pearson (Fig 4); compare rows within a panel.",
             fontsize=8, color=INK2)
    fig.savefig(args.out / "fig5_shared_space_absolute_vs_delta.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig6(args, cells, x, y, expression, metadata):
    """Joint PCA of gene-matched profiles; each gene standardised over the 47 lines within each dataset."""
    symbols = np.array([gene_symbol(c) for c in expression.columns])
    xcol = {s: i for i, s in enumerate(symbols)}
    tahoe_genes = list(load_genes(args))
    shared = [g for g in tahoe_genes if g in xcol]
    xs = x[:, [xcol[g] for g in shared]]
    ys = y[:, [tahoe_genes.index(g) for g in shared]]
    keep = (xs.std(0) > 0) & (ys.std(0) > 0)
    z = lambda a: (a - a.mean(0)) / a.std(0)
    both = np.vstack([z(xs[:, keep]), z(ys[:, keep])])
    both -= both.mean(0)
    u_, s_, vt = np.linalg.svd(both, full_matrices=False)
    pcs = u_[:, :2] * s_[:2]
    n = len(cells)
    var = s_[:2] ** 2 / np.sum(s_ ** 2)
    fig, ax = plt.subplots(figsize=(7, 6.2))
    connected_scatter(ax, pcs[:n], pcs[n:])
    ax.set_xlabel(f"PC1 ({var[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    gap = np.linalg.norm(pcs[:n] - pcs[n:], axis=1)
    ax.set_title(f"Direct gene space (PCA of {keep.sum():,} matched genes, both datasets)\n"
                 f"median same-line distance {np.median(gap):.1f}", loc="left", fontsize=9)
    ax.legend(loc="best")
    fig.savefig(args.out / "fig6_direct_space_pca.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    global _GENES
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--depmap-root", type=Path, default=DEFAULT_DEPMAP_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_PROJECT_ROOT / "report" / "figures" / "10_09_cca")
    p.add_argument("--figs", default="1,2,3,4,5,6", help="comma-separated figure numbers to draw")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    compact, expression, metadata, cells, x, y, repo_root = load_inputs(args)
    _GENES = compact.genes
    want = set(args.figs.split(","))
    if "1" in want: fig1(args, cells, x, y, metadata, repo_root)
    if "2" in want: fig2(args, cells, x, y, expression, metadata)
    if "3" in want: fig3(args, metadata)
    if "4" in want: fig4(args)
    if "5" in want: fig5(args)
    if "6" in want: fig6(args, cells, x, y, expression, metadata)
    print("wrote", sorted(q.name for q in args.out.iterdir()))


if __name__ == "__main__":
    main()
