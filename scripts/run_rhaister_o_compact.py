#!/usr/bin/env python3
"""Compact, auditable reproduction of Rhaister-O's Tahoe delta baseline.

This runner intentionally targets the expression-delta branch only.  It uses
the public ``cell_eval`` plate files and DMSO HVG centroids from
``tahoebio/tahoe-de-rhaister`` and mirrors the pinned Rhaister implementation:

    delta_hat(c*, t, g) = mean_delta(t, g)
                          + gamma(t, g) * centered_DMSO(c*, g)

The diagonal ridge prediction is blended 95/5 with the additive ALS fallback,
as in rhaister/train.py at the pinned repository revision.  The full upstream
six-metric run additionally needs the ~40 GB pDEX parquet; this compact path
does not pretend to reproduce those five non-delta metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path("/home/yuqi.lei/Therapeutic")
DEFAULT_DATA_ROOT = Path("/nfs-stor/yuqi.lei/Therapeutic/raw/tahoe-de-rhaister")
DEFAULT_CACHE_ROOT = Path("/nfs-stor/yuqi.lei/Therapeutic/processed/rhaister_o_compact")
DEFAULT_OUTPUT_ROOT = Path("/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_o_compact")
DEFAULT_DATA_REVISION = "c7963cf334bec0683225d41c9586d900ca6303a2"
MODEL_NAMES = (
    "zero_delta",
    "global_mean",
    "perturbation_mean",
    "additive_als",
    "shared_gamma",
    "rhaister_o",
)


@dataclass
class CompactData:
    cells: np.ndarray
    treatments: np.ndarray
    plates: list[tuple[int, ...]]
    delta: np.ndarray
    genes: list[str]


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _source_signature(cell_eval_files: Iterable[Path], gene_path: Path) -> dict:
    return {
        "gene_list_sha256": _sha256(gene_path),
        "cell_eval_files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in cell_eval_files
        ],
    }


def _plate_number(path: Path) -> int:
    match = re.fullmatch(r"plate_plate(\d+)\.parquet", path.name)
    if not match:
        raise ValueError(f"Cannot parse plate number from {path}")
    return int(match.group(1))


def _load_gene_list(repo_root: Path, data_root: Path) -> tuple[list[str], Path]:
    repo_path = repo_root / "splits" / "tahoe" / "static_2k_genes.json"
    data_path = data_root / "definition" / "static_2k_genes.json"
    if not repo_path.exists() or not data_path.exists():
        raise FileNotFoundError("The Rhaister 2K gene list is missing")
    if _sha256(repo_path) != _sha256(data_path):
        raise RuntimeError("Repository and dataset copies of static_2k_genes.json differ")
    with repo_path.open(encoding="utf-8") as handle:
        genes = json.load(handle)
    if len(genes) != 2000 or len(set(genes)) != len(genes):
        raise ValueError(f"Expected 2,000 unique genes, found {len(genes)}")
    return genes, repo_path


def build_or_load_compact_cache(
    data_root: Path,
    repo_root: Path,
    cache_root: Path,
    rebuild: bool = False,
) -> CompactData:
    """Mean-aggregate plate rows and retain the plate set for each condition."""
    genes, gene_path = _load_gene_list(repo_root, data_root)
    files = sorted(
        (data_root / "cell_eval").glob("plate_plate*.parquet"),
        key=_plate_number,
    )
    if len(files) != 14:
        raise FileNotFoundError(f"Expected 14 Tahoe cell_eval plates, found {len(files)}")

    signature = _source_signature(files, gene_path)
    manifest_path = cache_root / "manifest.json"
    obs_path = cache_root / "observations.parquet"
    delta_path = cache_root / "delta.npy"
    cache_valid = False
    if not rebuild and manifest_path.exists() and obs_path.exists() and delta_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            old_manifest = json.load(handle)
        cache_valid = old_manifest.get("source_signature") == signature

    if not cache_valid:
        cache_root.mkdir(parents=True, exist_ok=True)
        frames = []
        read_columns = ["cell_line", "treatment", *genes]
        for index, path in enumerate(files, start=1):
            started = time.time()
            frame = pd.read_parquet(path, columns=read_columns)
            frame["plate"] = _plate_number(path)
            frames.append(frame)
            print(
                f"cache: read {index:02d}/{len(files)} {path.name}: "
                f"{len(frame):,} rows ({time.time() - started:.1f}s)",
                flush=True,
            )
        raw = pd.concat(frames, ignore_index=True)
        del frames
        keys = ["cell_line", "treatment"]
        grouped = raw.groupby(keys, sort=True, observed=True)
        delta_df = grouped[genes].mean()
        plate_series = grouped["plate"].agg(lambda values: ",".join(
            str(v) for v in sorted(set(int(x) for x in values))
        ))
        observations = delta_df.index.to_frame(index=False)
        observations["plates"] = plate_series.to_numpy()
        delta = delta_df.to_numpy(dtype=np.float32, copy=True)
        del raw, grouped, delta_df, plate_series

        obs_tmp = obs_path.with_suffix(".parquet.tmp")
        observations.to_parquet(obs_tmp, index=False)
        os.replace(obs_tmp, obs_path)
        delta_tmp = delta_path.with_name("delta.tmp.npy")
        np.save(delta_tmp, delta, allow_pickle=False)
        os.replace(delta_tmp, delta_path)
        manifest = {
            "format_version": 1,
            "description": "Mean cell_eval delta per (cell_line, treatment)",
            "source_signature": signature,
            "n_observations": len(observations),
            "n_cells": int(observations["cell_line"].nunique()),
            "n_treatments": int(observations["treatment"].nunique()),
            "n_genes": len(genes),
            "delta_dtype": str(delta.dtype),
            "plate_mapping": (
                "Derived from cell_eval filenames; the exact upstream loader derives "
                "condition-to-plate mappings from the separate pDEX parquet."
            ),
        }
        _json_dump(manifest, manifest_path)
        print(f"cache: wrote {cache_root} ({len(observations):,} conditions)", flush=True)
    else:
        print(f"cache: using {cache_root}", flush=True)

    observations = pd.read_parquet(obs_path)
    delta = np.load(delta_path, mmap_mode="r")
    if delta.shape != (len(observations), len(genes)):
        raise ValueError(
            f"Cache shape mismatch: delta={delta.shape}, metadata={len(observations)}, genes={len(genes)}"
        )
    plates = [tuple(int(value) for value in text.split(",")) for text in observations["plates"]]
    return CompactData(
        cells=observations["cell_line"].astype(str).to_numpy(),
        treatments=observations["treatment"].astype(str).to_numpy(),
        plates=plates,
        delta=delta,
        genes=genes,
    )


def load_split(repo_root: Path, split: int) -> tuple[list[str], dict[str, set[str]], Path]:
    split_path = (
        repo_root
        / "splits"
        / "tahoe"
        / f"{split}_holdout"
        / "generalization_converted_cell_lines_3b.toml"
    )
    with split_path.open("rb") as handle:
        config = tomllib.load(handle)
    holdout_cells: list[str] = []
    test_treatments: dict[str, set[str]] = {}
    for key, values in config["fewshot"].items():
        cell = key.split(".")[-1]
        holdout_cells.append(cell)
        test_treatments[cell] = set(values["test"])
    if len(holdout_cells) != 5:
        raise ValueError(f"Expected five held-out cells in {split_path}, found {len(holdout_cells)}")
    return holdout_cells, test_treatments, split_path


def load_hvg_centroids(data_root: Path, genes: list[str]) -> dict[tuple[str, int], np.ndarray]:
    path = data_root / "zeroshot" / "cell_line_centroids_dmso_hvg.parquet"
    frame = pd.read_parquet(path)
    missing = [gene for gene in genes if gene not in frame.columns]
    if missing:
        raise ValueError(f"HVG centroid is missing {len(missing)} target genes")
    plate = frame["plate"].str.replace("plate_plate", "", regex=False).astype(int).to_numpy()
    cells = frame["cell_line_id"].astype(str).to_numpy()
    values = frame[genes].to_numpy(dtype=np.float32, copy=True)
    return {(cell, int(p)): values[i] for i, (cell, p) in enumerate(zip(cells, plate))}


def condition_hvg(
    cells: np.ndarray,
    plates: list[tuple[int, ...]],
    indices: np.ndarray,
    hvg: dict[tuple[str, int], np.ndarray],
    n_genes: int,
) -> tuple[np.ndarray, int]:
    result = np.zeros((len(indices), n_genes), dtype=np.float32)
    missing = 0
    for output_index, source_index in enumerate(indices):
        cell = cells[source_index]
        rows = [hvg[(cell, plate)] for plate in plates[source_index] if (cell, plate) in hvg]
        if rows:
            result[output_index] = np.mean(rows, axis=0, dtype=np.float32)
        else:
            missing += 1
    return result, missing


def group_mean(values: np.ndarray, group_index: np.ndarray, n_groups: int) -> np.ndarray:
    """Fast float32 grouped mean for a dense observation-by-gene matrix."""
    values32 = np.asarray(values, dtype=np.float32)
    order = np.argsort(group_index, kind="stable")
    sorted_groups = group_index[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_groups)) + 1]
    present_groups = sorted_groups[starts]
    result = np.zeros((n_groups, values32.shape[1]), dtype=np.float32)
    result[present_groups] = np.add.reduceat(values32[order], starts, axis=0)
    counts = np.bincount(group_index, minlength=n_groups).astype(np.float32)
    result /= np.maximum(counts[:, None], 1.0)
    return result


def additive_als(
    train_y: np.ndarray,
    train_cells: np.ndarray,
    train_treatments: np.ndarray,
    test_treatments: np.ndarray,
    iterations: int = 5,
) -> np.ndarray:
    """Mirror the unweighted float32 ALS delta path; held-out cell effect is zero."""
    unique_cells, cell_index = np.unique(train_cells, return_inverse=True)
    unique_treatments, treatment_index = np.unique(train_treatments, return_inverse=True)
    treatment_map = {name: index for index, name in enumerate(unique_treatments)}
    try:
        test_treatment_index = np.array([treatment_map[name] for name in test_treatments])
    except KeyError as error:
        raise ValueError(f"Test treatment absent from strict-zero-shot training set: {error}") from error

    y = np.asarray(train_y, dtype=np.float32)
    mu = y.mean(axis=0, dtype=np.float32)
    treatment_effect = np.zeros((len(unique_treatments), y.shape[1]), dtype=np.float32)
    cell_effect = np.zeros((len(unique_cells), y.shape[1]), dtype=np.float32)
    for _ in range(iterations):
        residual = y - mu - cell_effect[cell_index]
        treatment_effect = group_mean(residual, treatment_index, len(unique_treatments))
        residual = y - mu - treatment_effect[treatment_index]
        cell_effect = group_mean(residual, cell_index, len(unique_cells))
    return mu + treatment_effect[test_treatment_index]


def perturbation_mean_prediction(
    train_y: np.ndarray,
    train_treatments: np.ndarray,
    test_treatments: np.ndarray,
) -> np.ndarray:
    unique_treatments, inverse = np.unique(train_treatments, return_inverse=True)
    means = group_mean(np.asarray(train_y, dtype=np.float32), inverse, len(unique_treatments))
    treatment_map = {name: index for index, name in enumerate(unique_treatments)}
    global_mean = np.asarray(train_y, dtype=np.float32).mean(axis=0, dtype=np.float32)
    return np.stack([means[treatment_map[t]] if t in treatment_map else global_mean for t in test_treatments])


def diagonal_predictions(
    train_y: np.ndarray,
    train_cells: np.ndarray,
    train_treatments: np.ndarray,
    test_cells: np.ndarray,
    test_treatments: np.ndarray,
    train_h: np.ndarray,
    test_h: np.ndarray,
    additive: np.ndarray,
    ridge_lambda: float,
    blend: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Rhaister-O blend, shared-gamma blend, coverage mask)."""
    unique_cells, cell_inverse = np.unique(train_cells, return_inverse=True)
    holdout_cells, holdout_inverse = np.unique(test_cells, return_inverse=True)
    n_genes = train_y.shape[1]

    # This order matches upstream: average log1p(H) rows within each cell.
    z_cell = group_mean(np.log1p(train_h), cell_inverse, len(unique_cells)).astype(np.float64)
    z_holdout = group_mean(np.log1p(test_h), holdout_inverse, len(holdout_cells)).astype(np.float64)
    z_mean = z_cell.mean(axis=0)
    z_cell -= z_mean
    z_holdout -= z_mean

    ct_to_row = {(cell, treatment): i for i, (cell, treatment) in enumerate(
        zip(train_cells.tolist(), train_treatments.tolist())
    )}
    test_index = {(cell, treatment): i for i, (cell, treatment) in enumerate(
        zip(test_cells.tolist(), test_treatments.tolist())
    )}
    output = np.asarray(additive, dtype=np.float64).copy()
    shared_output = np.asarray(additive, dtype=np.float64).copy()
    covered = np.zeros(len(test_cells), dtype=bool)
    train_y64 = np.asarray(train_y, dtype=np.float64)

    for holdout_index, holdout in enumerate(holdout_cells):
        candidate = sorted(set(test_treatments[test_cells == holdout].tolist()))
        common = [treatment for treatment in candidate if all(
            (cell, treatment) in ct_to_row for cell in unique_cells
        )]
        if not common:
            print(f"split: {holdout}: no treatments have complete reference-cell coverage")
            continue
        row_index = np.array([
            ct_to_row[(cell, treatment)] for cell in unique_cells for treatment in common
        ])
        y_all = train_y64[row_index].reshape(len(unique_cells), len(common), n_genes)
        mu_y = y_all.mean(axis=0)
        residual = y_all - mu_y[None, :, :]
        denominator = np.sum(z_cell * z_cell, axis=0) + ridge_lambda
        gamma = np.einsum("cg,ctg->tg", z_cell, residual, optimize=True) / denominator[None, :]
        prediction = mu_y + gamma * z_holdout[holdout_index][None, :]

        shared_denominator = len(common) * np.sum(z_cell * z_cell, axis=0) + ridge_lambda
        shared_gamma = np.einsum("cg,ctg->g", z_cell, residual, optimize=True) / shared_denominator
        shared_prediction = mu_y + shared_gamma[None, :] * z_holdout[holdout_index][None, :]
        for treatment_index, treatment in enumerate(common):
            output_index = test_index[(holdout, treatment)]
            output[output_index] = (
                (1.0 - blend) * additive[output_index] + blend * prediction[treatment_index]
            )
            shared_output[output_index] = (
                (1.0 - blend) * additive[output_index] + blend * shared_prediction[treatment_index]
            )
            covered[output_index] = True
        print(
            f"split: {holdout}: diagonal coverage {len(common):,}/{len(candidate):,} treatments",
            flush=True,
        )
    return output, shared_output, covered


def rowwise_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    true64 = np.asarray(y_true, dtype=np.float64)
    pred64 = np.asarray(y_pred, dtype=np.float64)
    true_centered = true64 - true64.mean(axis=1, keepdims=True)
    pred_centered = pred64 - pred64.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(true_centered * true_centered, axis=1)
        * np.sum(pred_centered * pred_centered, axis=1)
    )
    pearson = np.divide(
        np.sum(true_centered * pred_centered, axis=1),
        denominator,
        out=np.zeros(len(true64), dtype=np.float64),
        where=denominator > 0,
    )
    error = pred64 - true64
    rmse = np.sqrt(np.mean(error * error, axis=1))
    mae = np.mean(np.abs(error), axis=1)
    return {"pearson": pearson, "rmse": rmse, "mae": mae}


def _git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_split(
    split: int,
    compact: CompactData,
    hvg: dict[tuple[str, int], np.ndarray],
    repo_root: Path,
    output_root: Path,
    ridge_lambda: float,
    blend: float,
    save_predictions: bool,
) -> dict:
    started = time.time()
    holdout_cells, test_treatments, split_path = load_split(repo_root, split)
    holdout = set(holdout_cells)
    test_mask = np.array([
        cell in holdout and treatment in test_treatments.get(cell, set())
        for cell, treatment in zip(compact.cells, compact.treatments)
    ])
    train_mask = np.array([cell not in holdout for cell in compact.cells])
    test_indices = np.flatnonzero(test_mask)
    train_indices = np.flatnonzero(train_mask)
    if len(test_indices) == 0:
        raise ValueError(f"Split {split} produced no test conditions")

    train_cells = compact.cells[train_indices]
    train_treatments_arr = compact.treatments[train_indices]
    test_cells = compact.cells[test_indices]
    test_treatments_arr = compact.treatments[test_indices]
    train_y = np.asarray(compact.delta[train_indices], dtype=np.float32)
    test_y = np.asarray(compact.delta[test_indices], dtype=np.float32)
    train_h, train_missing_h = condition_hvg(
        compact.cells, compact.plates, train_indices, hvg, len(compact.genes)
    )
    test_h, test_missing_h = condition_hvg(
        compact.cells, compact.plates, test_indices, hvg, len(compact.genes)
    )
    print(
        f"split {split}: train={len(train_indices):,}, test={len(test_indices):,}, "
        f"reference_cells={len(set(train_cells))}, holdout_cells={holdout_cells}, "
        f"missing_H={train_missing_h}/{test_missing_h}",
        flush=True,
    )

    predictions: dict[str, np.ndarray] = {}
    predictions["zero_delta"] = np.zeros_like(test_y, dtype=np.float32)
    predictions["global_mean"] = np.broadcast_to(
        train_y.mean(axis=0, dtype=np.float32), test_y.shape
    ).copy()
    predictions["perturbation_mean"] = perturbation_mean_prediction(
        train_y, train_treatments_arr, test_treatments_arr
    )
    predictions["additive_als"] = additive_als(
        train_y, train_cells, train_treatments_arr, test_treatments_arr
    )
    rhaister, shared, covered = diagonal_predictions(
        train_y,
        train_cells,
        train_treatments_arr,
        test_cells,
        test_treatments_arr,
        train_h,
        test_h,
        predictions["additive_als"],
        ridge_lambda,
        blend,
    )
    predictions["shared_gamma"] = shared
    predictions["rhaister_o"] = rhaister

    condition_metrics = pd.DataFrame({
        "cell_line": test_cells,
        "treatment": test_treatments_arr,
        "diagonal_covered": covered,
    })
    model_summary: dict[str, dict[str, float]] = {}
    for model in MODEL_NAMES:
        metrics = rowwise_metrics(test_y, predictions[model])
        for metric_name, values in metrics.items():
            condition_metrics[f"{model}__{metric_name}"] = values
        model_summary[model] = {
            f"mean_{metric_name}": float(np.mean(values))
            for metric_name, values in metrics.items()
        }
        model_summary[model]["cell_macro_pearson"] = float(
            condition_metrics.assign(_value=metrics["pearson"])
            .groupby("cell_line")["_value"]
            .mean()
            .mean()
        )

    split_output = output_root / f"split_{split}"
    split_output.mkdir(parents=True, exist_ok=True)
    condition_tmp = split_output / "per_condition_metrics.csv.tmp"
    condition_metrics.to_csv(condition_tmp, index=False)
    os.replace(condition_tmp, split_output / "per_condition_metrics.csv")
    if save_predictions:
        pred_tmp = split_output / "predictions.tmp.npz"
        np.savez_compressed(
            pred_tmp,
            genes=np.asarray(compact.genes),
            cell_line=np.asarray(test_cells, dtype=str),
            treatment=np.asarray(test_treatments_arr, dtype=str),
            diagonal_covered=covered,
            truth=test_y.astype(np.float32),
            **{model: predictions[model].astype(np.float32) for model in MODEL_NAMES},
        )
        os.replace(pred_tmp, split_output / "predictions.npz")

    summary = {
        "split": split,
        "split_file": str(split_path),
        "split_file_sha256": _sha256(split_path),
        "heldout_cells": holdout_cells,
        "n_train_conditions": len(train_indices),
        "n_test_conditions": len(test_indices),
        "n_reference_cells": len(set(train_cells)),
        "n_genes": len(compact.genes),
        "diagonal_coverage_count": int(covered.sum()),
        "diagonal_coverage_fraction": float(covered.mean()),
        "missing_hvg_train": train_missing_h,
        "missing_hvg_test": test_missing_h,
        "ridge_lambda": ridge_lambda,
        "delta_blend": blend,
        "models": model_summary,
        "elapsed_seconds": time.time() - started,
    }
    _json_dump(summary, split_output / "summary.json")
    print(
        f"split {split}: Rhaister-O mean Pearson={model_summary['rhaister_o']['mean_pearson']:.6f} "
        f"({time.time() - started:.1f}s)",
        flush=True,
    )
    return summary


def aggregate_summaries(
    summaries: list[dict],
    repo_root: Path,
    cache_root: Path,
    output_root: Path,
    data_revision: str,
    predictions_saved: bool,
) -> dict:
    cache_manifest = cache_root / "manifest.json"
    aggregate: dict = {
        "implementation": "compact expression-delta reproduction of pinned Rhaister-O code",
        "rhaister_revision": _git_revision(repo_root),
        "tahoe_de_rhaister_revision": data_revision,
        "cache_manifest": str(cache_manifest),
        "cache_manifest_sha256": _sha256(cache_manifest),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "random_seed": 0,
        "deterministic": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "predictions_saved": predictions_saved,
        "command": [sys.executable, *sys.argv],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "splits": [item["split"] for item in summaries],
        "per_split": summaries,
        "models": {},
    }
    for model in MODEL_NAMES:
        aggregate["models"][model] = {}
        for metric in ("mean_pearson", "cell_macro_pearson", "mean_rmse", "mean_mae"):
            values = [item["models"][model][metric] for item in summaries]
            aggregate["models"][model][f"{metric}_across_splits"] = float(np.mean(values))
            aggregate["models"][model][f"{metric}_sd_across_splits"] = float(np.std(values))
            aggregate["models"][model][f"{metric}_per_split"] = values
    output_root.mkdir(parents=True, exist_ok=True)
    _json_dump(aggregate, output_root / "summary.json")
    return aggregate


def self_test() -> None:
    true = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    pred = np.array([[2.0, 4.0, 6.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    result = rowwise_metrics(true, pred)
    assert np.allclose(result["pearson"], [1.0, 0.0])

    train_y = np.array([[1, 2], [3, 4], [2, 3], [4, 5]], dtype=np.float32)
    train_cells = np.array(["a", "a", "b", "b"])
    train_treatments = np.array(["x", "y", "x", "y"])
    additive = additive_als(train_y, train_cells, train_treatments, np.array(["x", "y"]))
    assert additive.shape == (2, 2)
    assert np.isfinite(additive).all()
    print("self-test: ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="all", choices=["all", "5", "6", "7", "8", "9"])
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-revision", default=DEFAULT_DATA_REVISION)
    parser.add_argument("--lambda", dest="ridge_lambda", type=float, default=1.0)
    parser.add_argument("--delta-blend", type=float, default=0.95)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.ridge_lambda < 0:
        raise ValueError("--lambda must be non-negative")
    if not 0 <= args.delta_blend <= 1:
        raise ValueError("--delta-blend must be between zero and one")

    repo_root = args.project_root / "external" / "Rhaister"
    compact = build_or_load_compact_cache(
        args.data_root, repo_root, args.cache_root, rebuild=args.rebuild_cache
    )
    hvg = load_hvg_centroids(args.data_root, compact.genes)
    splits = range(5, 10) if args.split == "all" else [int(args.split)]
    summaries = [
        run_split(
            split,
            compact,
            hvg,
            repo_root,
            args.output_root,
            args.ridge_lambda,
            args.delta_blend,
            args.save_predictions,
        )
        for split in splits
    ]
    aggregate = aggregate_summaries(
        summaries,
        repo_root,
        args.cache_root,
        args.output_root,
        args.data_revision,
        args.save_predictions,
    )
    rhaister = aggregate["models"]["rhaister_o"]
    print(
        "aggregate: Rhaister-O mean Pearson "
        f"{rhaister['mean_pearson_across_splits']:.6f} ± "
        f"{rhaister['mean_pearson_sd_across_splits']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
