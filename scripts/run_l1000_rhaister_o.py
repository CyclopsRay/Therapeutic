#!/usr/bin/env python3
"""Rhaister-O on LINCS L1000, leave-one-cell-line-out.

Does Rhaister-O's per-gene diagonal map work anywhere other than Tahoe? On
Tahoe it scores 0.6314 full-delta and 0.5380 after removing the conserved
drug-dose mean. Whether that is the model or the dataset is not answerable from
Tahoe alone.

This runs the same estimator on a second perturbation atlas. Deliberately
minimal: take what L1000 has, fit, score, and leave the framing of the result
to the analysis afterwards. Every quantity a later analysis might want -- the
drug-mean floor, the per-condition scores, the per-line spread -- falls out of
this run, so nothing is foreclosed by running first.

WHAT IS AND IS NOT THE SAME AS TAHOE. Same: the estimator (one intercept and
one slope per (treatment, gene), regressed on the target line's baseline, ridge
lambda 1.0, blended 0.95 against the additive fallback), the strict holdout
(whole cell lines, no treated signature from the held-out line reaching the
fit), and the metric (Pearson across genes per condition, macro-averaged by
line). Different, and unavoidably so:

  response   L1000 Level-5 replicate-collapsed z-scores, not a log expression
             delta against a plate-matched control.
  genes      978 directly measured landmarks, not the Tahoe 2,000-gene panel.
             Only 79 genes are shared, so the panels are effectively disjoint.
  baseline   DepMap 24Q4 log2(TPM+1), because L1000 Level 5 has no untreated
             profile and we hold no Level 3. This is the same information
             setting MultiDCP uses, which is the one published cold-cell L1000
             protocol whose cell representation is external bulk RNA-seq.
  contexts   8 cell lines, against Tahoe's 45.

Because of those four, the absolute Pearson here is NOT comparable to Tahoe's
0.5380 and must never be quoted beside it. What is comparable is the shape:
where the model sits relative to the baselines computed on this very slice.

SLICE. GSE70138 (Phase II), pert_type trt_cp, 24 h, the cell lines that have
both broad L1000 coverage and a DepMap baseline. Replicate signatures for one
(cell, compound, dose) are averaged. Treatments are kept when present in every
retained line, so the response tensor is dense and every held-out line is
scored on the same treatments the model was fitted on elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from run_rhaister_o_compact import rowwise_metrics
from run_shared_control_diagnostic import diagonal_predict
from run_depmap_expression_ridge import atomic_csv, line_bootstrap_interval, sha256, write_json

DEFAULT_RAW = Path("/nfs-stor/yuqi.lei/Therapeutic/raw/lincs-l1000")
DEFAULT_DEPMAP = Path("/nfs-stor/yuqi.lei/Therapeutic/raw/depmap/DepMap_Public_24Q4")
DEFAULT_DERIVED = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/l1000_rhaister_o")
DEFAULT_OUTPUT = Path("/nfs-stor/yuqi.lei/Therapeutic/results/l1000_rhaister_o")
# L1000 cell ids that also carry a DepMap baseline. NPC, NEU, ASC, SKB and PHH
# are primary or stem cells and are absent from DepMap, so they cannot be used.
DEFAULT_LINES = ("A375", "A549", "HA1E", "HCC515", "HEPG2", "HT29", "MCF7", "PC3", "VCAP")


def normalise(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def decode(values: np.ndarray) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def select_signatures(raw_root: Path, accession: str, lines: tuple[str, ...],
                      hours: str) -> pd.DataFrame:
    frame = pd.read_csv(raw_root / accession / "sig_info.txt.gz", sep="\t", low_memory=False)
    keep = frame["pert_type"].eq("trt_cp") & frame["cell_id"].isin(lines)
    if "pert_itime" in frame.columns:
        keep &= frame["pert_itime"].eq(hours)
    frame = frame[keep].copy()
    dose = frame["pert_idose"] if "pert_idose" in frame.columns else "unknown"
    frame["condition"] = (
        frame["pert_id"].astype(str) + "|" + pd.Series(dose, index=frame.index).astype(str)
    )
    return frame


def extract_matrix(raw_root: Path, derived: Path, accession: str,
                   signatures: list[str], threads: int) -> tuple[np.ndarray, list[str], list[str]]:
    """Pull the landmark rows for the chosen signatures out of the Level-5 GCTX.

    The archive is decompressed to scratch, read once, and deleted. Cached on a
    second run so refits do not pay the decompression again.
    """
    cache = derived / f"{accession}_landmark_matrix.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as archive:
            cached = archive["sig_id"].astype(str).tolist()
            if set(cached) == set(signatures):
                print(f"{accession}: using cached matrix {cache}", flush=True)
                return archive["values"], cached, archive["gene_symbol"].astype(str).tolist()
        print(f"{accession}: cache does not match the requested signatures, rebuilding", flush=True)

    gene_info = pd.read_csv(raw_root / accession / "gene_info.txt.gz", sep="\t")
    landmarks = gene_info[gene_info["pr_is_lm"].eq(1)].copy()
    landmarks["pr_gene_id"] = landmarks["pr_gene_id"].astype(str)

    source = next((raw_root / accession).glob("*.gctx.gz"))
    work = derived / "work"
    work.mkdir(parents=True, exist_ok=True)
    temporary = work / f"{accession}.{os.environ.get('SLURM_JOB_ID', 'local')}.gctx"
    try:
        print(f"{accession}: decompressing {source.name} ({source.stat().st_size:,} bytes)", flush=True)
        with temporary.open("wb") as handle:
            subprocess.run(["pigz", "-dc", "-p", str(threads), str(source)],
                           stdout=handle, check=True)
        with h5py.File(temporary, "r") as handle:
            rows = decode(handle["/0/META/ROW/id"][:])
            columns = decode(handle["/0/META/COL/id"][:])
            matrix = handle["/0/DATA/0/matrix"]
            row_lookup = {value: index for index, value in enumerate(rows)}
            column_lookup = {value: index for index, value in enumerate(columns)}
            available = [s for s in signatures if s in column_lookup]
            missing = len(signatures) - len(available)
            gene_ids = [g for g in landmarks["pr_gene_id"] if g in row_lookup]
            symbols = landmarks.set_index("pr_gene_id").loc[gene_ids, "pr_gene_symbol"].astype(str).tolist()
            gene_positions = np.array([row_lookup[g] for g in gene_ids])
            pairs = sorted((column_lookup[s], s) for s in available)
            column_positions = np.array([p[0] for p in pairs])
            ordered = [p[1] for p in pairs]
            print(f"{accession}: matrix {matrix.shape}, taking {len(ordered):,} signatures "
                  f"x {len(gene_positions)} landmarks ({missing} signatures absent)", flush=True)
            if matrix.shape == (len(rows), len(columns)):
                block = np.asarray(matrix[:, column_positions], dtype=np.float32).T
            elif matrix.shape == (len(columns), len(rows)):
                block = np.asarray(matrix[column_positions, :], dtype=np.float32)
            else:
                raise ValueError(f"unexpected matrix shape {matrix.shape}")
            values = block[:, gene_positions]
    finally:
        if temporary.exists():
            temporary.unlink()

    derived.mkdir(parents=True, exist_ok=True)
    partial = derived / f".{accession}_landmark_matrix.tmp.npz"
    np.savez_compressed(partial, values=values,
                        sig_id=np.asarray(ordered, dtype=str),
                        gene_symbol=np.asarray(symbols, dtype=str))
    os.replace(partial, cache)
    return values, ordered, symbols


def depmap_baseline(depmap_root: Path, lines: tuple[str, ...],
                    symbols: list[str]) -> tuple[np.ndarray, list[str], dict]:
    model = pd.read_csv(depmap_root / "Model.csv", low_memory=False)
    index: dict[str, str] = {}
    for column in ("StrippedCellLineName", "CellLineName"):
        if column in model.columns:
            for value, model_id in zip(model[column], model["ModelID"]):
                key = normalise(value)
                if key and key not in index:
                    index[key] = str(model_id)
    wanted = {line: index.get(normalise(line)) for line in lines}
    resolved = {k: v for k, v in wanted.items() if v}

    expression = pd.read_csv(depmap_root / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
                             index_col=0, low_memory=False)
    expression.columns = [re.sub(r"\s*\(\d+\)$", "", c).strip() for c in expression.columns]
    expression = expression.loc[:, ~expression.columns.duplicated()]
    present = {k: v for k, v in resolved.items() if v in expression.index}
    shared = [g for g in symbols if g in expression.columns]
    matrix = expression.loc[[present[line] for line in present], shared].to_numpy(dtype=np.float64)
    info = {"requested_lines": list(lines), "resolved_model_ids": present,
            "lines_without_depmap": sorted(set(lines) - set(present)),
            "n_landmark_genes_in_depmap": len(shared),
            "n_landmark_genes_missing": len(symbols) - len(shared)}
    return matrix, list(present), shared, info


def main(args: argparse.Namespace) -> None:
    args.derived_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    signatures = select_signatures(args.raw_root, args.accession, tuple(args.lines), args.hours)
    print(f"{args.accession}: {len(signatures):,} trt_cp signatures at {args.hours} "
          f"in {signatures['cell_id'].nunique()} lines", flush=True)

    values, ordered, symbols = extract_matrix(
        args.raw_root, args.derived_root, args.accession,
        sorted(signatures["sig_id"].astype(str).unique()), args.threads,
    )
    position = {s: i for i, s in enumerate(ordered)}
    signatures = signatures[signatures["sig_id"].astype(str).isin(position)].copy()

    # Average replicate signatures of one (cell, compound, dose).
    signatures["row"] = [position[s] for s in signatures["sig_id"].astype(str)]
    grouped = signatures.groupby(["cell_id", "condition"])["row"].apply(list)
    conditions = sorted({c for _, c in grouped.index})
    lines_present = sorted({l for l, _ in grouped.index})
    counts = {(l, c): len(v) for (l, c), v in grouped.items()}
    common = [c for c in conditions if all((l, c) in counts for l in lines_present)]
    print(f"  {len(lines_present)} lines, {len(conditions):,} conditions, "
          f"{len(common):,} present in every line", flush=True)
    if len(common) < args.min_conditions:
        raise ValueError(f"only {len(common)} common conditions; need {args.min_conditions}")

    response = np.zeros((len(lines_present), len(common), values.shape[1]), dtype=np.float64)
    for i, line in enumerate(lines_present):
        for j, condition in enumerate(common):
            rows = grouped.loc[(line, condition)]
            response[i, j] = values[rows].mean(axis=0)

    baseline, baseline_lines, genes_final, baseline_info = depmap_baseline(
        args.depmap_root, tuple(lines_present), symbols
    )
    # Keep only lines with a DepMap baseline, and only landmarks DepMap also
    # measures, so the response tensor and the baseline share both axes exactly.
    keep = [i for i, line in enumerate(lines_present) if line in baseline_lines]
    lines_final = [lines_present[i] for i in keep]
    response = response[keep]
    order = {line: i for i, line in enumerate(baseline_lines)}
    baseline = baseline[[order[line] for line in lines_final]]
    shared = set(genes_final)
    gene_keep = [i for i, g in enumerate(symbols) if g in shared]
    response = response[:, :, gene_keep]
    if response.shape[2] != baseline.shape[1]:
        raise ValueError(
            f"gene axes disagree: response {response.shape[2]} vs baseline {baseline.shape[1]}"
        )
    print(f"  final tensor {response.shape}  lines {lines_final}", flush=True)
    print(f"  baseline {baseline.shape}  genes kept {len(genes_final)}", flush=True)

    records = []
    for held in range(len(lines_final)):
        keep_mask = np.arange(len(lines_final)) != held
        reference_response = response[keep_mask]
        prediction, drug_mean = diagonal_predict(
            reference_response, baseline[keep_mask], baseline[held][None, :],
            args.rhaister_lambda, args.rhaister_blend,
        )
        prediction = prediction[0]
        truth = response[held]
        global_mean = np.broadcast_to(reference_response.mean(axis=(0, 1)), truth.shape)
        models = {
            "zero_delta": np.zeros_like(truth),
            "global_mean": global_mean,
            "drug_mean": drug_mean,
            "rhaister_o": prediction,
        }
        for name, predicted in models.items():
            for level, (t, p) in {
                "full": (truth, predicted),
                "drug_residual": (truth - drug_mean, predicted - drug_mean),
            }.items():
                metrics = rowwise_metrics(t, p)
                records.append({
                    "held_out_line": lines_final[held], "model": name, "level": level,
                    "n_conditions": int(truth.shape[0]),
                    "pearson": float(metrics["pearson"].mean()),
                    "rmse": float(metrics["rmse"].mean()),
                    "mae": float(metrics["mae"].mean()),
                })
        # Index by content, not by offset: the append order is (model, level)
        # nested, so a negative offset silently picks a different model.
        recent = {(r["model"], r["level"]): r["pearson"] for r in records[-8:]}
        print(f"  held out {lines_final[held]:8s}: "
              f"rhaister_o full {recent[('rhaister_o', 'full')]:.4f} "
              f"drug_mean full {recent[('drug_mean', 'full')]:.4f} "
              f"rhaister_o resid {recent[('rhaister_o', 'drug_residual')]:.4f}", flush=True)

    frame = pd.DataFrame(records)
    atomic_csv(frame, args.output_root / "per_line_metrics.csv")

    aggregate = []
    for level in ("full", "drug_residual"):
        for model in ("zero_delta", "global_mean", "drug_mean", "rhaister_o"):
            subset = frame[frame["level"].eq(level) & frame["model"].eq(model)]
            mean, low, high = line_bootstrap_interval(
                subset["pearson"].to_numpy(dtype=np.float64), args.seed, args.n_bootstrap
            )
            aggregate.append({"level": level, "model": model, "n_lines": len(subset),
                              "line_macro_pearson": mean,
                              "line_bootstrap_low": low, "line_bootstrap_high": high})
    aggregate_frame = pd.DataFrame(aggregate)
    atomic_csv(aggregate_frame, args.output_root / "aggregate_summary.csv")

    print("\nline-macro Pearson:")
    for level in ("full", "drug_residual"):
        print(f"  {level}")
        for row in aggregate_frame[aggregate_frame["level"].eq(level)].itertuples():
            print(f"    {row.model:14s} {row.line_macro_pearson:+.6f}  "
                  f"[{row.line_bootstrap_low:+.6f}, {row.line_bootstrap_high:+.6f}]")

    write_json(
        {
            "purpose": "Rhaister-O's estimator on L1000, leave-one-cell-line-out.",
            "not_comparable_to_tahoe_because": [
                "response is a Level-5 z-score, not a plate-matched log expression delta",
                "genes are the 978 L1000 landmarks; only 79 overlap the Tahoe 2K panel",
                "baseline is DepMap log2(TPM+1), not a Tahoe DMSO centroid",
                f"{len(lines_final)} contexts against Tahoe's 45",
            ],
            "accession": args.accession, "hours": args.hours,
            "lines": lines_final, "n_conditions": len(common),
            "n_genes": len(genes_final),
            "replicates_per_condition_median": float(np.median(list(counts.values()))),
            "baseline": baseline_info,
            "rhaister_lambda": args.rhaister_lambda, "rhaister_blend": args.rhaister_blend,
            "aggregate": aggregate,
            "seed": args.seed, "n_bootstrap": args.n_bootstrap,
            "runner_sha256": sha256(Path(__file__).resolve()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        args.output_root / "summary.json",
    )
    print(f"\nWrote {args.output_root}/summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--depmap-root", type=Path, default=DEFAULT_DEPMAP)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--accession", default="GSE70138")
    parser.add_argument("--hours", default="24 h")
    parser.add_argument("--lines", nargs="+", default=list(DEFAULT_LINES))
    parser.add_argument("--min-conditions", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rhaister-lambda", type=float, default=1.0)
    parser.add_argument("--rhaister-blend", type=float, default=0.95)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
