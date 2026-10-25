#!/usr/bin/env python3
"""The four cell representations, re-run on LINCS L1000.

Job 249475 compared four ways of representing a cell line under one fixed
response head on Tahoe and found a clear ladder: DepMap expression 0.2087,
prose generated from structured fields 0.0930, those same fields one-hot 0.0780,
curated Cellosaurus text 0.0294. The decisive contrast was that generated prose
was statistically indistinguishable from the fields it was generated from
(+0.0150, CI spanning zero), which is what a representation change carrying no
new measured information looks like.

Whether that ladder is a fact about cell representations or a fact about Tahoe
is not answerable from Tahoe alone. This runs the same four arms, under the same
head, on a second atlas where the overall signal is an order of magnitude
weaker: on this L1000 slice Rhaister-O itself reaches only 0.0453 drug-residual
against 0.5380 on Tahoe (job 274337). The question is whether the *ordering*
survives when the absolute signal collapses.

TWO LIMITATIONS THAT MUST TRAVEL WITH ANY RESULT FROM THIS RUN.

1. **The structured arm is close to degenerate at eight cell lines.** Among the
   eight, `OncotreeLineage`, `OncotreePrimaryDisease`, `OncotreeSubtype` and
   `DepmapModelType` each take seven distinct values, `GrowthPattern` takes one,
   and `TissueOrigin` is entirely missing. A categorical with seven levels over
   eight lines is close to a one-hot identity code: in leave-one-line-out the
   held-out line's level is usually unseen, its design row is all zeros in the
   trained levels, and the ridge falls back to the intercept, i.e. the drug mean.
   A near-zero score for this arm is therefore expected *mechanically* from the
   cohort size and is not evidence that metadata is uninformative. The per-field
   level counts are written to the summary so this is auditable.

2. **The Cellosaurus arm is truncated.** Six of the eight redacted texts exceed
   the 512-token encoder limit (72-497 words, but sub-word tokenisation pushes
   most of them over). This is the same caveat that applied to 40 of 49 texts on
   Tahoe, so a negative result for this arm means "no advantage under 512-token
   truncation", not "curated text carries nothing". The template texts (52-76
   words) are not truncated.

A third, milder point: seven reference lines admit at most six principal
components, so every arm is compressed to at most six dimensions regardless of
its raw width. That is a property of the cohort, applied equally to all four
arms, so it does not bias the comparison between them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from run_depmap_expression_ridge import (
    atomic_csv, build_candidates, line_bootstrap_interval,
    reference_only_pca, sha256, target_weights, write_json,
)
from run_l1000_rhaister_o import (
    DEFAULT_DEPMAP, DEFAULT_DERIVED, DEFAULT_RAW, extract_matrix,
    normalise, select_signatures,
)
from run_rhaister_o_compact import rowwise_metrics

ARMS = ("expression_pca", "structured_metadata", "template_text", "cellosaurus_text")
STRUCTURED_CATEGORICAL = (
    "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype",
    "DepmapModelType", "Sex", "AgeCategory", "PatientRace",
    "PrimaryOrMetastasis", "SampleCollectionSite", "TissueOrigin",
    "GrowthPattern", "OnboardedMedia",
)
STRUCTURED_NUMERIC = ("Age",)
DEFAULT_OUTPUT = Path("/nfs-stor/yuqi.lei/Therapeutic/results/l1000_cell_ablation")


def structured_features(mapping: pd.DataFrame) -> tuple[np.ndarray, dict]:
    blocks, used = [], {}
    for column in STRUCTURED_CATEGORICAL:
        if column not in mapping.columns:
            continue
        values = mapping[column].astype("string").fillna("__missing__")
        dummies = pd.get_dummies(values, prefix=column, dtype=float)
        used[column] = {"n_levels": int(values.nunique(dropna=False)),
                        "n_nonnull": int(mapping[column].notna().sum())}
        if dummies.shape[1] <= 1:
            continue
        blocks.append(dummies.to_numpy(dtype=np.float64))
    for column in STRUCTURED_NUMERIC:
        if column not in mapping.columns:
            continue
        numeric = pd.to_numeric(mapping[column], errors="coerce")
        if numeric.notna().sum() < 2:
            continue
        blocks.append(np.column_stack([
            numeric.fillna(numeric.mean()).to_numpy(dtype=np.float64),
            numeric.isna().to_numpy(dtype=np.float64),
        ]))
        used[column] = {"n_levels": "numeric", "n_nonnull": int(numeric.notna().sum())}
    features = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(mapping), 0))
    return features, {"n_features": int(features.shape[1]), "fields": used}


def load_embedding(path: Path, rrids: list[str]) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        order = {str(r): i for i, r in enumerate(archive["rrid"].astype(str))}
        matrix = archive["embedding"].astype(np.float64)
    missing = [r for r in rrids if r not in order]
    rows = np.zeros((len(rrids), matrix.shape[1]), dtype=np.float64)
    for i, rrid in enumerate(rrids):
        if rrid in order:
            rows[i] = matrix[order[rrid]]
    return rows, {"n_features": int(matrix.shape[1]), "missing_rrids": missing}


def main(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(args.mapping)
    lines = mapping["l1000_cell_id"].astype(str).tolist()

    signatures = select_signatures(args.raw_root, args.accession, tuple(lines), args.hours)
    values, ordered, symbols = extract_matrix(
        args.raw_root, args.derived_root, args.accession,
        sorted(signatures["sig_id"].astype(str).unique()), args.threads,
    )
    position = {s: i for i, s in enumerate(ordered)}
    signatures = signatures[signatures["sig_id"].astype(str).isin(position)].copy()
    signatures["row"] = [position[s] for s in signatures["sig_id"].astype(str)]
    grouped = signatures.groupby(["cell_id", "condition"])["row"].apply(list)
    present_lines = sorted({l for l, _ in grouped.index})
    counts = {(l, c): len(v) for (l, c), v in grouped.items()}
    common = [c for c in sorted({c for _, c in grouped.index})
              if all((l, c) in counts for l in present_lines)]
    print(f"  {len(present_lines)} lines, {len(common):,} conditions in every line", flush=True)

    response = np.zeros((len(present_lines), len(common), values.shape[1]), dtype=np.float64)
    for i, line in enumerate(present_lines):
        for j, condition in enumerate(common):
            response[i, j] = values[grouped.loc[(line, condition)]].mean(axis=0)
    mapping = mapping[mapping["l1000_cell_id"].isin(present_lines)].copy()
    mapping = mapping.set_index("l1000_cell_id").loc[present_lines].reset_index()
    rrids = mapping["RRID"].astype(str).tolist()

    expression = pd.read_csv(
        args.depmap_root / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        index_col=0, low_memory=False,
    )
    model_ids = mapping["ModelID"].astype(str).tolist()
    features: dict[str, np.ndarray] = {}
    info: dict[str, dict] = {}
    features["expression_pca"] = expression.loc[model_ids].to_numpy(dtype=np.float64)
    info["expression_pca"] = {"n_features": int(features["expression_pca"].shape[1]),
                              "source": "DepMap 24Q4 log2(TPM+1), all protein-coding genes"}
    del expression
    features["structured_metadata"], info["structured_metadata"] = structured_features(mapping)
    features["template_text"], info["template_text"] = load_embedding(
        args.derived_root / "embeddings_template.npz", rrids)
    features["cellosaurus_text"], info["cellosaurus_text"] = load_embedding(
        args.derived_root / "embeddings_cellosaurus.npz", rrids)
    for arm in ARMS:
        print(f"  {arm:22s} {features[arm].shape}", flush=True)

    records = []
    selections = []
    for held in range(len(present_lines)):
        keep = np.arange(len(present_lines)) != held
        reference = response[keep]
        drug_mean = reference.mean(axis=0)
        residual = (reference - drug_mean[None, :, :]).reshape(keep.sum(), -1)
        gram = (residual @ residual.T).astype(np.float64)
        truth = response[held]
        for arm in args.arms:
            scores, query_scores, pca_info = reference_only_pca(
                features[arm][keep], features[arm][held][None, :]
            )
            grid = tuple(c for c in args.component_grid if c <= scores.shape[1])
            if not grid:
                grid = (1,)
            candidates = build_candidates(
                scores, gram, residual.shape[1], grid, tuple(args.lambda_grid)
            )
            best = int(np.argmin([c.loo_rmse for c in candidates]))
            predicted = (target_weights(candidates[best], query_scores) @ residual).reshape(
                1, len(common), values.shape[1]
            )[0]
            prediction = drug_mean + predicted
            selections.append({
                "held_out_line": present_lines[held], "arm": arm,
                "selected_components": candidates[best].n_components,
                "selected_lambda": candidates[best].ridge_lambda,
                "available_components": int(scores.shape[1]),
                "n_input_features": int(features[arm].shape[1]),
                "n_variable_features": pca_info["n_variable_genes"],
            })
            for level, (t, p) in {
                "full": (truth, prediction),
                "drug_residual": (truth - drug_mean, prediction - drug_mean),
            }.items():
                metrics = rowwise_metrics(t, p)
                records.append({
                    "held_out_line": present_lines[held], "model": arm, "level": level,
                    "pearson": float(metrics["pearson"].mean()),
                    "rmse": float(metrics["rmse"].mean()),
                })
        for level, (t, p) in {
            "full": (truth, drug_mean),
            "drug_residual": (truth - drug_mean, np.zeros_like(drug_mean)),
        }.items():
            metrics = rowwise_metrics(t, p)
            records.append({"held_out_line": present_lines[held], "model": "drug_mean",
                            "level": level, "pearson": float(metrics["pearson"].mean()),
                            "rmse": float(metrics["rmse"].mean())})
        print(f"  held out {present_lines[held]:8s} done", flush=True)

    frame = pd.DataFrame(records)
    atomic_csv(frame, args.output_root / "per_line_metrics.csv")
    atomic_csv(pd.DataFrame(selections), args.output_root / "inner_selection.csv")

    aggregate = []
    for level in ("full", "drug_residual"):
        for model in list(args.arms) + ["drug_mean"]:
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
        for row in aggregate_frame[aggregate_frame["level"].eq(level)].sort_values(
            "line_macro_pearson", ascending=False
        ).itertuples():
            print(f"    {row.model:22s} {row.line_macro_pearson:+.6f}  "
                  f"[{row.line_bootstrap_low:+.6f}, {row.line_bootstrap_high:+.6f}]")

    write_json(
        {
            "purpose": ("The four cell representations of job 249475, re-run on L1000 "
                        "to ask whether the Tahoe ordering is about representations or "
                        "about Tahoe."),
            "limitations": [
                ("structured_metadata is close to a one-hot identity code at eight "
                 "lines; a near-zero score is expected mechanically from cohort size"),
                ("cellosaurus_text is truncated for six of eight lines at 512 tokens, "
                 "so a negative result means 'no advantage under truncation'"),
                ("seven reference lines admit at most six principal components, so "
                 "every arm is compressed to at most six dimensions"),
            ],
            "tahoe_reference": {"expression_pca": 0.2087, "template_text": 0.0930,
                                "structured_metadata": 0.0780, "cellosaurus_text": 0.0294,
                                "rhaister_o": 0.5394, "job": 249475},
            "l1000_reference": {"rhaister_o_drug_residual": 0.0453, "job": 274337},
            "lines": present_lines, "n_conditions": len(common),
            "n_genes": int(values.shape[1]),
            "features": info, "aggregate": aggregate,
            "component_grid": list(args.component_grid), "lambda_grid": list(args.lambda_grid),
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
    parser.add_argument("--mapping", type=Path,
                        default=DEFAULT_DERIVED / "l1000_Model.csv")
    parser.add_argument("--accession", default="GSE70138")
    parser.add_argument("--hours", default="24 h")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--component-grid", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--lambda-grid", type=float, nargs="+",
                        default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
