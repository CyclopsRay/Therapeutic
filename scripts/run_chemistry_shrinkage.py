#!/usr/bin/env python3
"""Shrink Rhaister-O's slopes toward what chemistry predicts, not toward a mean.

Rhaister-O fits one slope ``gamma[p, g]`` per (treatment, gene) pair from 45
reference lines, and drugs share nothing: the slope for a drug and the slope for
a structurally near-identical drug are estimated completely independently.

Job 272027 shrank every slope uniformly toward the pooled per-gene slope and
gained **+0.00187** (real -- bootstrap lower bound +0.00099, permutation
p = 0.0196 -- but ten times too small to matter). The selected blend was 0.9 in
all five splits, i.e. almost no uniform shrinkage helps.

HYPOTHESIS. Uniform shrinkage fails because it pools dissimilar drugs. Pulling
a kinase inhibitor's slope toward the average of 1,137 unrelated treatments
destroys signal. Chemically similar drugs should share slope structure, so let
chemistry choose the target:

    gamma_chem[p, g]  = rank-K reduced-rank ridge regression of the observed
                        slopes on the drug's feature vector
    gamma_tilde[p, g] = (1 - w) * gamma_chem[p, g] + w * gamma_free[p, g]
    Delta_hat[c,p,g]  = beta[p, g] + gamma_tilde[p, g] * x[c, g]

WHY THIS FORM AND NOT A TRILINEAR ONE. The first draft of this experiment used a
rank-K trilinear correction in which the cell entered through a low-rank
projection rather than per gene. That breaks the diagonal structure, and job
249519 already showed cross-gene conditioning fails at 45 reference lines
(permutation p = 0.9901, null mean above the observed value). A failure of that
form would have been uninterpretable -- we could not have told a wrong
hypothesis from a wrong parameterisation. This form keeps the diagonal
completely intact and changes only *what the slopes are shrunk toward*, so a
failure is attributable to chemistry carrying no transferable slope structure.

ANCHOR RULE. The prediction is a delta on the frozen array,

    prediction = FROZEN + blend * (gamma_tilde - gamma_free) * x_query

so at w = 1 the correction is identically zero and the arm reproduces frozen
Rhaister-O bit for bit. Asserted in code. This is the standing rule adopted
after an independent review of job 272004 found five defects whose magnitude
(4.7e-4 to 7.9e-4 per gene) was the same order as the effect being measured:
every effect here is 1e-3, so build on the reference, never beside it.

ARMS. All five are the same model with a different shrinkage target, so the
comparison is like for like and the previous result is reproduced as an internal
control rather than quoted from another run.

  anchor          w = 1. Must reproduce frozen Rhaister-O exactly.
  uniform_shrink  gamma_chem = the pooled per-gene slope. This is job 272027's
                  model and should land near its +0.00187.
  moa_only        gamma_chem from a one-hot of moa-fine (26 classes) alone.
  morgan_only     gamma_chem from Morgan fingerprints alone.
  chem_full       gamma_chem from Morgan plus moa-fine. THE HEADLINE ARM.

INNER SELECTION uses leave-one-reference-line-out **drug-residual Pearson**, not
LOO RMSE: jobs 249519 and 272027 both showed reference LOO RMSE is a poor proxy
for the held-out endpoint, and in 272027 the two criteria disagreed in all five
splits. What RMSE would have chosen is recorded per split so the lesson is
tested a third time.

READING THE RESULT. The predeclared gate is unchanged and no threshold is
relaxed. But the scale to read any gain against is the one measured in
`report/09_25_report.md`: the data levers on this benchmark are +0.016 to
+0.030, and the best method result so far is +0.0019. A gain of +0.005 here
would be the best model result the project has produced and still sit below the
benchmark's own data-quality noise floor.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_all_rhaister import remove_pc
from evaluate_rhaister_signal import parse_treatment
from run_coefficient_sharing import GATE, slope_estimates
from run_depmap_expression_ridge import (
    add_model_metrics,
    atomic_csv,
    line_bootstrap_interval,
    map_drug_means,
    response_cube,
    sha256,
    write_json,
)
from run_depmap_rhaister_correction import line_hvg_vectors
from run_rhaister_o_compact import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_REVISION,
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    build_or_load_compact_cache,
    load_hvg_centroids,
    load_split,
    rowwise_metrics,
)

LEVELS = ("full", "drug_residual", "train_pca1_removed")
ARMS = ("uniform_shrink", "moa_only", "morgan_only", "chem_full")
DEFAULT_BASELINE_ROOT = Path("/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_o_compact")
DEFAULT_COMPONENT_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_all_split_evaluation"
)
DEFAULT_DRUG_METADATA = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/raw/tahoe-100m/metadata/drug_metadata.parquet"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/results/chemistry_shrinkage"
)


def build_drug_features(
    metadata_path: Path, treatments: list[str], kind: str,
    morgan_bits: int = 1024, morgan_radius: int = 2,
) -> tuple[np.ndarray, dict]:
    """Per-treatment drug feature matrix, built per drug and broadcast to doses.

    Every dose of one drug shares one feature vector: dose is already a separate
    column of the response tensor, and putting it in the features would let the
    model rediscover the dose axis rather than chemistry.
    """
    frame = pd.read_parquet(metadata_path)
    names = [parse_treatment(label)[0] for label in treatments]
    unique_names = sorted(set(names))
    lookup = {str(row.drug): row for row in frame.itertuples()}

    blocks: list[np.ndarray] = []
    info: dict = {"kind": kind, "n_treatments": len(treatments),
                  "n_unique_drugs": len(unique_names)}

    if kind in ("morgan", "morgan_moa"):
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=morgan_radius, fpSize=morgan_bits
        )
        fingerprints = np.zeros((len(unique_names), morgan_bits), dtype=np.float64)
        unparsed: list[str] = []
        for index, name in enumerate(unique_names):
            row = lookup.get(name)
            smiles = getattr(row, "canonical_smiles", None) if row is not None else None
            molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            if molecule is None:
                unparsed.append(name)
                continue
            fingerprints[index] = np.asarray(
                generator.GetFingerprint(molecule), dtype=np.float64
            )
        blocks.append(fingerprints)
        info.update(morgan_bits=morgan_bits, morgan_radius=morgan_radius,
                    unparsed_smiles_drugs=unparsed)

    if kind in ("moa", "morgan_moa"):
        # "moa-fine" is not a valid identifier, so itertuples renames it; take
        # the column straight off the frame instead of guessing the alias.
        values = dict(zip(frame["drug"].astype(str), frame["moa-fine"].astype(str)))
        categories = sorted({values.get(name, "__missing__") for name in unique_names})
        onehot = np.zeros((len(unique_names), len(categories)), dtype=np.float64)
        order = {c: i for i, c in enumerate(categories)}
        for index, name in enumerate(unique_names):
            onehot[index, order[values.get(name, "__missing__")]] = 1.0
        blocks.append(onehot)
        info.update(moa_categories=categories)

    per_drug = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(unique_names), 0))
    position = {name: index for index, name in enumerate(unique_names)}
    features = np.stack([per_drug[position[name]] for name in names])
    unmatched = sorted({n for n in unique_names if n not in lookup})
    info.update(n_features=int(features.shape[1]), unmatched_drugs=unmatched)
    return features, info


def standardise(reference: np.ndarray) -> np.ndarray:
    """Centre and scale feature columns, dropping constant ones."""
    centred = reference - reference.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0)
    keep = scale > 1e-12
    if not keep.any():
        return np.zeros((len(reference), 0), dtype=np.float64)
    return centred[:, keep] / scale[keep]


def hat_matrix(design: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """D (D'D + lambda I)^-1 D', the ridge smoother over treatments.

    Depends only on the drug features and lambda, never on the response, so it
    is computed once per (arm, lambda) and reused across every inner fold.
    """
    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += ridge_lambda
    return design @ np.linalg.solve(gram, design.T)


def chemistry_target(
    gamma: np.ndarray, pooled: np.ndarray, smoother: np.ndarray | None,
    ranks: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Reduced-rank ridge prediction of the slopes from drug features.

    With no smoother the target is ``pooled``, the per-gene shared slope that
    `slope_estimates` returns. That must be the *same* quantity job 272027
    shrank toward, or the uniform arm stops being an internal control: its
    denominator is ``n_treatments * sum(z^2) + lambda``, which is not the same
    as the arithmetic mean of the per-treatment slopes (that would divide the
    penalty by n_treatments as well). Using the estimator's own return value
    removes the choice.

    Otherwise the ridge fit is truncated by SVD, which is the reduced-rank
    regression solution under squared loss. One SVD serves every rank.
    """
    if smoother is None:
        broadcast = np.broadcast_to(pooled, gamma.shape)
        return {rank: broadcast for rank in ranks}
    fitted = smoother @ gamma
    u, singular, vt = np.linalg.svd(fitted, full_matrices=False)
    out: dict[int, np.ndarray] = {}
    for rank in ranks:
        k = min(int(rank), len(singular))
        out[rank] = (u[:, :k] * singular[None, :k]) @ vt[:k]
    return out


def inner_selection(
    z_reference: np.ndarray, response: np.ndarray, drug_features: np.ndarray | None,
    ranks: tuple[int, ...], lambdas: tuple[float, ...], blends: tuple[float, ...],
    ridge_lambda: float, blend: float,
) -> tuple[dict, list[dict]]:
    """Leave-one-reference-line-out over the 45 reference lines.

    Scored by drug-residual Pearson, with RMSE recorded but not used. Each fold
    refits the slopes and recentres the cell feature on its own 44 lines;
    centring once outside the loop would leak the validation line.
    """
    n_lines = response.shape[0]
    uniform = drug_features is None
    lambda_grid = (0.0,) if uniform else lambdas
    keys = [(lam, rank, w) for lam in lambda_grid for rank in ranks for w in blends]
    position = {key: index for index, key in enumerate(keys)}
    pearson_sum = np.zeros(len(keys))
    sse_sum = np.zeros(len(keys))

    smoothers = {}
    if not uniform:
        for lam in lambda_grid:
            smoothers[lam] = hat_matrix(drug_features, lam)

    for held in range(n_lines):
        keep = np.arange(n_lines) != held
        fold_mean = z_reference[keep].mean(axis=0)
        z_fold = z_reference[keep] - fold_mean[None, :]
        z_held = z_reference[held] - fold_mean
        mu, gamma, pooled = slope_estimates(z_fold, response[keep], ridge_lambda)
        truth = response[held]
        base = (1.0 - blend) * mu + blend * (mu + gamma * z_held[None, :])
        for lam in lambda_grid:
            targets = chemistry_target(gamma, pooled, smoothers.get(lam), ranks)
            for rank in ranks:
                difference = targets[rank] - gamma
                for w in blends:
                    index = position[(lam, rank, w)]
                    prediction = base + blend * (1.0 - w) * difference * z_held[None, :]
                    pearson_sum[index] += float(
                        rowwise_metrics(truth - mu, prediction - mu)["pearson"].mean()
                    )
                    error = prediction - truth
                    sse_sum[index] += float(np.mean(error * error))
        del mu, gamma, base

    pearson = pearson_sum / n_lines
    rmse = np.sqrt(sse_sum / n_lines)
    best = int(np.argmax(pearson))
    worst_by_rmse = int(np.argmin(rmse))
    records = [
        {"ridge_lambda": keys[i][0], "rank": keys[i][1], "blend_w": keys[i][2],
         "loo_drug_residual_pearson": float(pearson[i]), "loo_rmse": float(rmse[i]),
         "selected_by_pearson": i == best, "selected_by_rmse": i == worst_by_rmse}
        for i in range(len(keys))
    ]
    return {
        "selected_lambda": keys[best][0], "selected_rank": keys[best][1],
        "selected_blend_w": keys[best][2],
        "selected_loo_pearson": float(pearson[best]),
        "rmse_would_select": {"ridge_lambda": keys[worst_by_rmse][0],
                              "rank": keys[worst_by_rmse][1],
                              "blend_w": keys[worst_by_rmse][2]},
        "criteria_agree": best == worst_by_rmse,
    }, records


def split_analysis(args, split: int, compact, hvg):
    repo_root = args.project_root / "external" / "Rhaister"
    holdout_cells, _, split_path = load_split(repo_root, split)
    reference_cells = sorted(set(compact.cells.tolist()) - set(holdout_cells))

    with np.load(
        args.baseline_root / f"split_{split}" / "predictions.npz", allow_pickle=False
    ) as archive:
        cells = archive["cell_line"].copy()
        treatments = archive["treatment"].copy()
        truth = archive["truth"].astype(np.float32)
        drug_mean = archive["perturbation_mean"].astype(np.float32)
        rhaister = archive["rhaister_o"].astype(np.float32)

    with np.load(
        args.component_root / f"split_{split}" / "train_residual_pc1.npz", allow_pickle=False
    ) as component_archive:
        pc_mean = component_archive["mean"].astype(np.float32)
        pc1 = component_archive["component"].astype(np.float32)

    candidate = sorted(set(treatments.tolist()))
    response, common = response_cube(compact, reference_cells, candidate)
    response = response.astype(np.float64)
    reference_hvg = line_hvg_vectors(compact, hvg, reference_cells)
    query_hvg = line_hvg_vectors(compact, hvg, holdout_cells)
    z_mean = reference_hvg.mean(axis=0)
    z_reference = reference_hvg - z_mean
    z_query = query_hvg - z_mean

    feature_sets = {
        "uniform_shrink": (None, {"kind": "pooled_per_gene_slope"}),
    }
    for arm, kind in (("moa_only", "moa"), ("morgan_only", "morgan"),
                      ("chem_full", "morgan_moa")):
        raw, info = build_drug_features(args.drug_metadata, list(common), kind)
        feature_sets[arm] = (standardise(raw), info)

    mu, gamma, pooled = slope_estimates(z_reference, response, args.rhaister_lambda)
    query_index = {cell: i for i, cell in enumerate(holdout_cells)}
    common_index = {t: i for i, t in enumerate(common)}
    covered = np.array([
        c in query_index and t in common_index for c, t in zip(cells, treatments)
    ])

    parsed = [parse_treatment(label) for label in treatments]
    frame = pd.DataFrame({
        "split": split, "cell_line": cells, "treatment": treatments,
        "drug": [p[0] for p in parsed], "dose": [p[1] for p in parsed],
        "covered": covered,
    })
    add_model_metrics(frame, "perturbation_mean", truth, drug_mean, drug_mean, pc_mean, pc1)
    add_model_metrics(frame, "rhaister_o", truth, drug_mean, rhaister, pc_mean, pc1)

    arm_info: dict[str, dict] = {}
    selection_records: list[dict] = []
    permutation_records: list[dict] = []
    rng = np.random.default_rng(args.seed + split * 1000)

    for arm in args.arms:
        features, info = feature_sets[arm]
        selection, records = inner_selection(
            z_reference, response, features,
            tuple(args.rank_grid), tuple(args.lambda_grid), tuple(args.blend_grid),
            args.rhaister_lambda, args.rhaister_blend,
        )
        for record in records:
            record.update(split=split, arm=arm)
        selection_records.extend(records)

        smoother = None if features is None else hat_matrix(
            features, selection["selected_lambda"]
        )
        target = chemistry_target(gamma, pooled, smoother, (selection["selected_rank"],))[
            selection["selected_rank"]
        ]
        difference = (1.0 - selection["selected_blend_w"]) * (target - gamma)

        prediction = np.array(rhaister, dtype=np.float32, copy=True)
        for row, (cell, treatment) in enumerate(zip(cells, treatments)):
            if cell in query_index and treatment in common_index:
                prediction[row] = rhaister[row] + (
                    args.rhaister_blend
                    * difference[common_index[treatment]]
                    * z_query[query_index[cell]]
                ).astype(np.float32)
        add_model_metrics(frame, arm, truth, drug_mean, prediction, pc_mean, pc1)

        # Anchor: rebuild the prediction with w forced to 1 and require that it
        # is the frozen array bit for bit. This exercises the real construction
        # path, not an algebraic identity, so a future refactor that breaks the
        # anchor is caught here rather than in a published number.
        anchor_prediction = np.array(rhaister, dtype=np.float32, copy=True)
        zero_difference = (1.0 - 1.0) * (target - gamma)
        for row, (cell, treatment) in enumerate(zip(cells, treatments)):
            if cell in query_index and treatment in common_index:
                anchor_prediction[row] = rhaister[row] + (
                    args.rhaister_blend
                    * zero_difference[common_index[treatment]]
                    * z_query[query_index[cell]]
                ).astype(np.float32)
        anchor_error = float(np.abs(anchor_prediction - rhaister).max())
        if anchor_error != 0.0:
            raise RuntimeError(
                f"split {split} {arm}: w=1 does not reproduce frozen Rhaister-O "
                f"(max |diff| {anchor_error:g}); the paired comparison is not anchored"
            )
        del anchor_prediction, zero_difference

        # Line-block permutation null on the same quantity the gate names.
        base_gain = {}
        for cell in holdout_cells:
            rows = np.flatnonzero(cells == cell)
            base_gain[cell] = (truth[rows] - drug_mean[rows], rhaister[rows] - drug_mean[rows])
        null_draws = []
        for _ in range(args.n_permutations):
            order = rng.permutation(len(common))
            shuffled = difference[order]
            gains = []
            for cell, i in query_index.items():
                rows = np.flatnonzero(cells == cell)
                truth_residual, frozen_residual = base_gain[cell]
                perturbed = np.array(frozen_residual, copy=True)
                for local, t in enumerate(treatments[rows]):
                    if t in common_index:
                        perturbed[local] += (
                            args.rhaister_blend * shuffled[common_index[t]] * z_query[i]
                        ).astype(np.float32)
                gains.append(
                    float(rowwise_metrics(truth_residual, perturbed)["pearson"].mean())
                    - float(rowwise_metrics(truth_residual, frozen_residual)["pearson"].mean())
                )
            null_draws.append(float(np.mean(gains)))
        for draw_id, value in enumerate(null_draws):
            permutation_records.append(
                {"split": split, "arm": arm, "permutation_id": draw_id, "gain": value}
            )

        arm_info[arm] = {**selection, "features": info,
                         "permutation_gain_mean": float(np.mean(null_draws))}
        print(f"  split {split} {arm}: lambda={selection['selected_lambda']:g} "
              f"rank={selection['selected_rank']} w={selection['selected_blend_w']:g} "
              f"(RMSE would pick {selection['rmse_would_select']})", flush=True)
        del difference, target, prediction

    info = {"split": split, "split_file_sha256": sha256(split_path),
            "n_reference_lines": len(reference_cells), "n_common_treatments": len(common),
            "n_covered_conditions": int(covered.sum()), "arms": arm_info}
    return frame, info, selection_records, permutation_records


def main(args: argparse.Namespace) -> None:
    repo_root = args.project_root.resolve() / "external" / "Rhaister"
    compact = build_or_load_compact_cache(args.data_root, repo_root, args.cache_root)
    hvg = load_hvg_centroids(args.data_root, compact.genes)
    args.output_root.mkdir(parents=True, exist_ok=True)

    frames, infos, selections, permutations = [], [], [], []
    for split in args.splits:
        frame, info, records, nulls = split_analysis(args, split, compact, hvg)
        frames.append(frame)
        infos.append(info)
        selections.extend(records)
        permutations.extend(nulls)
    frame = pd.concat(frames, ignore_index=True)
    atomic_csv(pd.DataFrame(selections), args.output_root / "inner_selection.csv")
    atomic_csv(pd.DataFrame(permutations), args.output_root / "permutation_null.csv")

    models = ("perturbation_mean", "rhaister_o") + tuple(args.arms)
    line_records = []
    for (split, cell), subset in frame.groupby(["split", "cell_line"], sort=True):
        for level in LEVELS:
            for model in models:
                line_records.append({
                    "split": int(split), "cell_line": cell, "level": level,
                    "model": model, "n_conditions": len(subset),
                    **{m: float(subset[f"{model}__{level}__{m}"].mean())
                       for m in ("pearson", "rmse", "mae")},
                })
    line_frame = pd.DataFrame(line_records)
    atomic_csv(line_frame, args.output_root / "per_line_summary.csv")

    aggregate = []
    for level in LEVELS:
        pivot = line_frame[line_frame["level"].eq(level)].pivot_table(
            index="cell_line", columns="model", values="pearson"
        )
        for model in models:
            mean, low, high = line_bootstrap_interval(
                pivot[model].to_numpy(dtype=np.float64), args.seed, args.n_bootstrap
            )
            aggregate.append({"level": level, "model": model, "n_lines": len(pivot),
                              "line_macro_pearson": mean,
                              "line_bootstrap_low": low, "line_bootstrap_high": high})
        for arm in args.arms:
            mean, low, high = line_bootstrap_interval(
                (pivot[arm] - pivot["rhaister_o"]).to_numpy(dtype=np.float64),
                args.seed, args.n_bootstrap,
            )
            aggregate.append({"level": level, "model": f"{arm}_minus_rhaister_o",
                              "n_lines": len(pivot), "line_macro_pearson": mean,
                              "line_bootstrap_low": low, "line_bootstrap_high": high})
    aggregate_frame = pd.DataFrame(aggregate)
    atomic_csv(aggregate_frame, args.output_root / "aggregate_summary.csv")

    null_frame = pd.DataFrame(permutations)
    rmse_pivot = line_frame[line_frame["level"].eq("drug_residual")].pivot_table(
        index="cell_line", columns="model", values="rmse"
    )
    gates = {}
    print("\ndrug-residual line-macro Pearson:")
    for model in models:
        row = aggregate_frame[
            aggregate_frame["level"].eq("drug_residual") & aggregate_frame["model"].eq(model)
        ]
        print(f"  {model:22s} {float(row['line_macro_pearson'].iloc[0]):.6f}")
    for arm in args.arms:
        def value(level, column="line_macro_pearson"):
            row = aggregate_frame[
                aggregate_frame["level"].eq(level)
                & aggregate_frame["model"].eq(f"{arm}_minus_rhaister_o")
            ]
            return float(row[column].iloc[0])
        gain, low = value("drug_residual"), value("drug_residual", "line_bootstrap_low")
        pc1_gain = value("train_pca1_removed")
        per_split = []
        for split in args.splits:
            block = line_frame[
                line_frame["split"].eq(split) & line_frame["level"].eq("drug_residual")
            ].pivot_table(index="cell_line", columns="model", values="pearson")
            per_split.append(float((block[arm] - block["rhaister_o"]).mean()))
        draws = null_frame[null_frame["arm"].eq(arm)].groupby("permutation_id")["gain"].mean()
        p_value = (int((draws >= gain).sum()) + 1) / (len(draws) + 1)
        ratio = float(rmse_pivot[arm].mean() / rmse_pivot["rhaister_o"].mean())
        conditions = {
            "gain_at_least": gain >= GATE["min_drug_residual_gain"],
            "bootstrap_low_positive": low > 0,
            "pc1_gain_at_least": pc1_gain >= GATE["min_pc1_removed_gain"],
            "enough_positive_splits": sum(g > 0 for g in per_split) >= GATE["min_positive_splits"],
            "rmse_within_tolerance": ratio <= GATE["max_rmse_ratio"],
            "above_permutation_null": p_value <= 0.05,
        }
        gates[arm] = {"drug_residual_gain": gain, "drug_residual_gain_low": low,
                      "pc1_removed_gain": pc1_gain, "per_split_gain": per_split,
                      "rmse_ratio": ratio, "permutation_p": p_value,
                      "permutation_null_mean": float(draws.mean()),
                      "conditions": conditions, "passes": all(conditions.values())}
        print(f"  {arm:22s} gain {gain:+.6f}  low {low:+.6f}  p {p_value:.4f}  "
              f"gate {gates[arm]['passes']}")

    write_json(
        {
            "purpose": (
                "Shrink Rhaister-O's per-(treatment, gene) slopes toward a "
                "chemistry-predicted target instead of a pooled per-gene mean. "
                "The diagonal structure is preserved, so a failure is "
                "attributable to chemistry rather than to the parameterisation."
            ),
            "interpretation_scale": (
                "Data levers on this benchmark are +0.016 to +0.030 and the best "
                "method result so far is +0.0019 (job 272027). Read any gain here "
                "against that scale before calling it useful."
            ),
            "arms": list(args.arms),
            "rank_grid": list(args.rank_grid), "lambda_grid": list(args.lambda_grid),
            "blend_grid": list(args.blend_grid),
            "predeclared_gate": GATE, "gates": gates,
            "aggregate": aggregate, "per_split": infos,
            "rhaister_lambda": args.rhaister_lambda, "rhaister_blend": args.rhaister_blend,
            "n_permutations": args.n_permutations, "n_bootstrap": args.n_bootstrap,
            "seed": args.seed, "data_revision": args.data_revision,
            "runner_sha256": sha256(Path(__file__).resolve()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        args.output_root / "summary.json",
    )
    print(f"\nWrote {args.output_root}/summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--drug-metadata", type=Path, default=DEFAULT_DRUG_METADATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-revision", default=DEFAULT_DATA_REVISION)
    parser.add_argument("--splits", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--rank-grid", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12, 20])
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[10.0, 100.0, 1000.0])
    parser.add_argument(
        "--blend-grid", type=float, nargs="+",
        default=[0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    parser.add_argument("--rhaister-lambda", type=float, default=1.0)
    parser.add_argument("--rhaister-blend", type=float, default=0.95)
    parser.add_argument("--n-permutations", type=int, default=50)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
