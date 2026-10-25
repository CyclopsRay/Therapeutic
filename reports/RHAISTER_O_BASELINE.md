# Rhaister-O Baseline: Setup, Reproduction, and Next Experiments

Last updated: 2026-09-07

## Executive finding

We now have a runnable strict unseen-cell-line baseline on the released Tahoe
Cell-Eval derivative. Across the five released Rhaister-O splits, the compact
reproduction reaches mean Pearson correlation on expression delta of
**0.6315 +/- 0.0680**. The Rhaister paper and repository report this result as
approximately **0.63**, so the local setup reproduces the headline delta score
to its published precision.

This is the right foundation for our project. It also changes the central
question: the useful contribution is not merely to predict a new Tahoe cell
line, because Rhaister-O already does that. The sharper question is whether
independent DepMap measurements predict the drug-specific residual that remains
after the conserved treatment response and the Rhaister-O correction.

## What Rhaister-O is

Rhaister-O is the strict zero-shot form of Rhaister. Each outer split contains
five query cell lines and 45 reference cell lines. All treated responses from
the five query lines are removed; a query line contributes only its DMSO
baseline expression. Drug-dose treatments tested in a query line have been
observed in the reference lines.

The pinned executable delta path estimates, for cell line `c`, treatment `t`,
and gene `g`,

```text
prediction(c, t, g) = treatment_mean(t, g)
                    + gamma(t, g) * centered_DMSO(c, g)
```

Here `gamma(t,g)` is a ridge-regularized, treatment-specific diagonal slope
estimated across the 45 reference lines. The code blends this prediction 95/5
with an additive ALS prediction and uses the additive result when a treatment
does not have complete reference-line coverage. This is a pseudobulk
expression-delta predictor; it is not a generative single-cell model.

Ordinary Rhaister is a different task. It observes a perturbation panel in the
query line and is therefore few-shot panel completion. We should only use
Rhaister-O as the primary comparator for a claim about a completely unseen
cell line.

## Local implementation and data

The setup is pinned to:

- Rhaister code commit `75fed20a1d97b05a09bf25c5341762141882106a`;
- `tahoebio/tahoe-de-rhaister` revision
  `c7963cf334bec0683225d41c9586d900ca6303a2`;
- the repository's five split files, `5_holdout` through `9_holdout`;
- its fixed 2,000-gene list, whose repository and dataset copies have identical
  SHA-256 values;
- Python 3.12.3, NumPy 2.2.6, pandas 2.3.3, and PyArrow 21.0.0.

The one-time cache mean-aggregates 65,218 plate rows into 56,827 unique
`(cell_line, treatment)` conditions and retains the observed plate set for each
condition. Plate-matched DMSO HVG centroids are then averaged over those plates.
All five splits contain 45 reference and five held-out lines, and no train or
test condition lacks a corresponding DMSO centroid.

The runner and Slurm wrapper are:

```bash
sbatch slurm/run_rhaister_o_compact.sbatch

# Small local correctness check only
/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep/bin/python \
  -m unittest tests/test_rhaister_o_compact.py -v
```

Important paths:

| Artifact | Location |
|---|---|
| Runner | `scripts/run_rhaister_o_compact.py` |
| Slurm wrapper | `slurm/run_rhaister_o_compact.sbatch` |
| Compact cache | `/nfs-stor/yuqi.lei/Therapeutic/processed/rhaister_o_compact` |
| Aggregate result | `/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_o_compact/summary.json` |
| Per-condition metrics | `/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_o_compact/split_*/per_condition_metrics.csv` |
| Gene-level truth/predictions | `/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_o_compact/split_*/predictions.npz` |
| Final run log | `/nfs-stor/yuqi.lei/Therapeutic/logs/rhaister-o-delta-214307.out` |

The prediction archives are self-describing: they contain genes, cell lines,
treatment labels, truth, coverage flags, and predictions for every implemented
baseline. They are suitable as the frozen starting point for residual analyses.
They pass an integrity load with `allow_pickle=False`. The final aggregate
summary SHA-256 is
`369dd06c68784e4e5fa2497129bc7927c1ec71d0ab703d551abdf72cf2fb03fd`.

## Measured result

Pearson is computed across the fixed 2,000 genes separately for each held-out
`(cell_line, drug-dose)` condition, then averaged within each split. The
reported standard deviation is across the five outer-split means.

| Model | Pearson delta | RMSE | MAE |
|---|---:|---:|---:|
| Zero delta | 0.0000 +/- 0.0000 | 0.02569 | 0.00938 |
| Global per-gene mean | 0.1880 +/- 0.0341 | 0.02538 | 0.00972 |
| Drug-dose mean | 0.4434 +/- 0.0705 | 0.02324 | 0.00992 |
| Additive ALS | 0.4434 +/- 0.0705 | 0.02324 | 0.00992 |
| Shared gene slope | 0.4893 +/- 0.0580 | 0.02262 | 0.00958 |
| **Rhaister-O treatment-specific slope** | **0.6315 +/- 0.0680** | **0.01966** | **0.00839** |

The near identity of drug-dose mean and additive ALS is expected in this nearly
balanced strict-holdout design: a query cell has no fitted cell effect, leaving
the treatment program. Rhaister-O gains 0.1881 absolute Pearson over that
baseline and reduces both RMSE and MAE by about 15.4%.

| Split | Test conditions | Drug-dose mean | Shared slope | Rhaister-O | Diagonal coverage |
|---:|---:|---:|---:|---:|---:|
| 5 | 3,673 | 0.4807 | 0.5192 | 0.6853 | 99.51% |
| 6 | 3,899 | 0.3704 | 0.4400 | 0.5678 | 99.64% |
| 7 | 3,899 | 0.5413 | 0.5622 | 0.7139 | 99.38% |
| 8 | 3,900 | 0.4697 | 0.5207 | 0.6531 | 99.74% |
| 9 | 3,897 | 0.3548 | 0.4044 | 0.5372 | 99.56% |

Overall, the diagonal model covers 19,185 of 19,268 test conditions (99.57%);
the remaining 83 use the additive fallback. The large split-to-split spread is
scientifically important: a single aggregate score hides substantial
cell-context difficulty.

## Reproduction boundary and code audit

This is a compact reproduction of the **expression-delta branch**, not the full
six-metric upstream pipeline. Two explicit differences remain:

1. The full loader inner-joins Cell-Eval conditions with an approximately
   40.6 GB pDEX parquet and uses pDEX to map conditions to plates. The compact
   dataset does not include that file, so this runner derives the plate mapping
   from Cell-Eval filenames and does not perform the pDEX join.
2. Consequently, this run cannot compute fold-change, p-value/FDR, PR-AUC,
   DE-overlap, or significant-gene rank metrics. Those require the pDEX target
   arrays and should be a separate exact-upstream reproduction if we need the
   complete paper table.

The match to the published 0.63 is reassuring, but it does not erase those
differences.

There is also an upstream documentation mismatch worth preserving. The pinned
`docs/zeroshot_architecture.md` describes SVD-denoised treatment means,
response-subspace projection, empirical-Bayes shrinkage, and a structured
treatment prior, and quotes a five-split score of 0.6478. Those operations are
not present in the pinned executable `rhaister/train.py` delta path. The code
implements the diagonal ridge plus additive blend reproduced here. Therefore
we should call this result the **pinned-code compact delta reproduction**, not a
reproduction of the documentation's seven-component stack.

## What to build on top of this

### 1. Turn the headline score into a diagnostic benchmark

Add per-held-out-line, dose, response-magnitude, pathway, and mechanism
summaries, then bootstrap drug-dose conditions within lines. Add Spearman,
cosine, sign accuracy, top-DE overlap, and response-template-removed metrics.
The first key question is whether Rhaister-O recovers the true
cell-line-by-treatment interaction or mainly improves the transferable global
response.

### 2. Add a nearest-line baseline

For each held-out line, choose a reference line using DMSO expression only and
copy its drug-dose response. This tells us whether the learned diagonal slopes
beat a transparent biological-neighbor rule. Distance and neighbor selection
must be fitted without held-out treated outcomes.

### 3. Test the DepMap hypothesis as residual prediction

Define

```text
residual = observed_delta - frozen_Rhaister-O_prediction
```

Fit a low-capacity ridge model from reference-line DepMap features to this
residual, with nested line-level validation inside every outer split. Begin
with DepMap expression, then add CNV, mutation, and Chronos dependency one at a
time. This directly measures information added by DepMap instead of allowing a
large model to relearn the already-strong conserved drug response.

The primary analysis should retain all possible lines with modality masks; a
38-line complete-case analysis can be a sensitivity check. Also compare using
the held-out line's baseline in unsupervised DepMap representation fitting with
the stricter version that excludes it.

### 4. Add component-aware null tests

Permute target-line DepMap vectors, permute genes in the diagonal pairing, and
replace plate-matched DMSO with plate-agnostic controls. These distinguish real
context transfer from gains due to treatment means, plate composition, or
generic expression amplitude.

### 5. Decide later whether full pDEX is worth staging

The compact delta result already answers the first go/no-go question and
matches the paper headline. Download pDEX only if we need exact six-metric
paper comparability or the fold-change/FDR heads. It is not required to start
the DepMap residual experiment.

### 6. Keep L1000 secondary

After the baseline and DepMap analysis are frozen, L1000 can supply an external
drug-response prior on the validated 946-gene overlap. Target-line L1000
treatment responses must remain excluded from a strict zero-shot claim.

## Recommended immediate sequence

1. Freeze this result and add per-line/bootstrap/component-aware evaluation.
2. Implement nearest-DMSO-line transfer on the identical test rows.
3. Build DepMap-expression residual ridge with nested line-level validation.
4. Run modality and leakage ablations before considering an MLP.
5. Stage full pDEX only if the complete six-metric Rhaister table becomes a
   required deliverable.

## Primary references

- Svensson et al., **Rhaister / Rhaister-O** (2026):
  [`papers/rhaister_2026.pdf`](../papers/rhaister_2026.pdf),
  DOI <https://doi.org/10.64898/2026.06.09.731197>.
- Zhang et al., **Tahoe-100M** (2025):
  [`papers/tahoe-100m_v3.pdf`](../papers/tahoe-100m_v3.pdf),
  DOI <https://doi.org/10.1101/2025.02.20.639398>.
- Molina and Zhang, **perturbation-response decomposition** (2026):
  [`papers/response-decomposition_v1.pdf`](../papers/response-decomposition_v1.pdf),
  DOI <https://doi.org/10.64898/2026.07.24.740459>.

See `papers/README.md` for State, Tahoe-x1, SCALE, DepMap, and L1000 references
and for the distinction between strict unseen-line and non-equivalent splits.
