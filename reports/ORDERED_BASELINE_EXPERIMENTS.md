# Ordered Rhaister-O and DepMap-Expression Experiments

Last updated: 2026-09-08

> Follow-up: the completed LINCS L1000 external comparison and combined
> Chinese/English synthesis are in
> [`TAHOE_DEPMAP_L1000_BILINGUAL_SYNTHESIS.md`](TAHOE_DEPMAP_L1000_BILINGUAL_SYNTHESIS.md).

## Decision

The six-step baseline plan is complete. Rhaister-O's unseen-line signal remains
strong after removing both the reference drug-dose mean and the leading
training-derived residual template. Fixed DMSO-neighbor transfer is weak and
heterogeneous. DepMap expression contains independently detectable
cell-specific response signal, but a leakage-safe DepMap model does **not**
provide stable improvement over frozen Rhaister-O.

The predeclared expansion gate is therefore **false**. Do not proceed to CNV,
mutation, dependency, or an MLP on the basis of these results.

## Information boundary and common evaluation

- Five released Rhaister-O outer splits (`5_holdout` through `9_holdout`).
- Five completely held-out lines per split: 25 distinct line-specific tests and
  19,268 line--drug-dose conditions in total.
- Fixed 2,000-gene Tahoe expression-delta panel.
- No treated response from an outer-held-out line is used for preprocessing,
  model fitting, model selection, or prediction.
- Frozen drug-mean and Rhaister-O predictions from job `214307`.
- Pearson is calculated across genes per condition and macro-averaged by line.
- Per-line uncertainty uses 2,000 bootstrap samples over drugs, keeping all
  doses of the same drug in one cluster. The final model comparison also uses
  20,000 bootstrap samples over the 23 evaluable held-out lines.
- The generic component is PC1 fitted only to outer-reference-line drug-mean
  residuals. The held-out-line outcome never defines this component.
- DepMap is Public 24Q4. Scaling and PCA are refitted using outer reference
  lines only; the target line is excluded even from unsupervised PCA fitting.

This estimates strict zero-shot generalization conditional on having an
external baseline-expression profile for the target line. It is not unseen-
drug prediction and not target-line few-shot adaptation. The five outer
training sets overlap, so the 25 line results should not be described as 25
fully independent model-training experiments.

## Steps 1--4: establish where Rhaister-O's signal lives

### Aggregate result over all 25 held-out lines

| Evaluation level | Drug mean | DMSO 1NN | DMSO 3NN | Rhaister-O |
|---|---:|---:|---:|---:|
| Full expression delta | 0.4434 | 0.2985 | 0.3405 | **0.6314** |
| Drug-mean residual | 0.0000 | 0.1472 | 0.1388 | **0.5380** |
| Training-PC1 removed | -0.0028 | 0.1282 | 0.1174 | **0.5259** |

All values are line-macro Pearson. Rhaister-O has a positive per-line
drug-cluster interval on all 25 lines at all three evaluation levels. Its
improvement over the drug mean also has a positive interval for every line.

The full-response score falls from 0.6314 to 0.5380 when the transferable drug
program is removed. Thus about 0.094 Pearson points in the headline score are
associated with that shared component, but the majority of the measured
correlation remains cell-context-specific under this diagnostic.

Reference residual PC1 explains only 4.90--5.49% of residual variance across
the five splits. Removing it changes Rhaister-O from 0.5380 to 0.5259. The
remaining signal is therefore not explained by one dominant generic response
template. This is a component-removal diagnostic, not a literal variance
decomposition of Pearson correlation.

### Split and line heterogeneity

| Split | Drug mean, full | Rhaister-O, full | Rhaister-O, drug residual | Rhaister-O, PC1 removed |
|---:|---:|---:|---:|---:|
| 5 | 0.4807 | 0.6853 | 0.5754 | 0.5692 |
| 6 | 0.3704 | 0.5677 | 0.5095 | 0.5004 |
| 7 | 0.5413 | 0.7139 | 0.5746 | 0.5555 |
| 8 | 0.4697 | 0.6531 | 0.5543 | 0.5439 |
| 9 | 0.3546 | 0.5369 | 0.4759 | 0.4606 |

The PC1-removed Rhaister-O line results range from 0.0648 for NCI-H2122 to
0.6850 for HOP-62. The aggregate is strong, but unseen-line difficulty is far
from uniform.

### Nearest-DMSO controls

The neighbors are chosen once from all-plate mean `log1p` DMSO HVG expression,
without target-line treated outcomes. The 1NN and uniform 3NN predictions copy
or average the corresponding reference response for each treatment.

- On full response, 1NN beats the drug mean with a positive interval on only
  5/25 lines; 3NN does so on 6/25.
- On drug-mean residual, 1NN has a positive interval on 15/25 lines and a
  negative interval on 10/25. The corresponding 3NN counts are 14/25 and
  11/25.
- Averaging three neighbors improves full-response mean correlation relative
  to 1NN, but does not improve the aggregate drug-residual result.

Baseline-state similarity therefore carries some context information, but a
single global neighbor relation does not transfer treatment-specific responses
reliably.

## Step 5A: DepMap expression predicts drug-mean residuals

The first low-capacity model asks whether independent DepMap expression can
predict response structure beyond the conserved drug-dose mean. For each outer
split it:

1. standardizes 19,193 DepMap protein-coding expression features on the
   expression-covered reference lines;
2. fits reference-only PCA;
3. uses reference-line analytic ridge LOOCV to choose 0, 2, 5, or 10 PCs and
   ridge lambda 1, 10, or 100;
4. maps the target line's baseline expression to the full treatment-by-gene
   drug-mean residual tensor; and
5. repeats hyperparameter selection under 20 reference-line label
   permutations.

All five splits choose 10 PCs and lambda 100. DepMap expression is available
for 23/25 held-out lines; C3A and COLO 205 are reported as unavailable rather
than silently imputed. Candidate-treatment coverage is above 99%.

| Evaluation level | Drug mean | DepMap-expression ridge | Rhaister-O | DepMap minus Rhaister-O |
|---|---:|---:|---:|---:|
| Full delta | 0.4500 | 0.4732 | 0.6344 | -0.1613 |
| Drug-mean residual | 0.0000 | 0.2087 | 0.5394 | -0.3307 |
| Training-PC1 removed | -0.0045 | 0.1803 | 0.5271 | -0.3468 |

These values use the common 23-line cohort. The DepMap drug-residual line
bootstrap interval is [0.1591, 0.2601], and the observed 0.2087 exceeds the
20-permutation 95th percentile of 0.0304 (`p = 1/21`). Every one of the 23
per-line drug-cluster intervals is above the drug-mean residual baseline.

This is genuine but modest cell-context signal. It is not competitive with
Rhaister-O: every per-line interval for DepMap minus Rhaister-O is below zero
on the drug-residual metric.

## Step 5B: can DepMap correct Rhaister-O's remaining error?

The standalone model above is not enough to test complementarity. A second
protocol-required model therefore targets the residual left by Rhaister-O
itself:

1. Within each outer split, generate exact leave-one-reference-line-out
   Rhaister-O predictions for all expression-covered reference lines.
2. Define the training target as `truth - cross-fitted Rhaister-O`.
3. Fit the same reference-only DepMap PCA/ridge family to that correction.
4. Add the inferred correction to the already frozen outer-test Rhaister-O
   prediction.
5. Under every line-label permutation, repeat ridge selection and evaluate the
   improvement over unchanged Rhaister-O.

The closed-form leave-one-line-out implementation was tested against direct
Rhaister-O refits. All five splits select only 2 PCs and lambda 100, indicating
strong shrinkage toward a very small correction.

| Evaluation level | Frozen Rhaister-O | + DepMap correction | Improvement | 95% line-bootstrap interval |
|---|---:|---:|---:|---:|
| Full delta | 0.634447 | 0.636036 | +0.001589 | [-0.000286, 0.003819] |
| Drug-mean residual | 0.539430 | 0.539787 | +0.000357 | **[-0.002679, 0.003610]** |
| Training-PC1 removed | 0.527100 | 0.527491 | +0.000392 | [-0.002850, 0.003983] |

For the primary drug-residual comparison, only 10/23 lines improve in point
estimate. Per-line drug bootstraps are positive for 8 lines, negative for 13,
and cross zero for 2. Split-level improvements are -0.00319, +0.00117,
+0.00063, -0.00289, and +0.00552 for splits 5--9.

The largest exploratory gains are HCT-15 (+0.02144) and LS 180 (+0.01648);
the largest losses are SW 1271 (-0.01319) and PANC-1 (-0.00809). These are
post-hoc heterogeneity observations, not evidence for selecting favorable
lines.

The correction slightly reduces mean RMSE from 0.019417 to 0.019358, as might
be expected from RMSE-based inner selection, but slightly worsens mean MAE from
0.008352 to 0.008370. Neither small change overrides the failed primary
Pearson gate.

## Step 6: predeclared expansion gate

| Required condition | Observed result | Pass? |
|---|---|---:|
| Drug-residual line-bootstrap lower bound for corrected minus Rhaister-O > 0 | -0.002679 | **No** |
| Actual improvement above permutation 95th percentile | +0.000357 vs -0.004730 | Yes |
| Empirical permutation p <= 0.05 | 1/21 = 0.047619 | Yes, minimum resolution |
| Expand to CNV/mutation/dependency/MLP | Requires all conditions | **No** |

Every permuted fit selected the intercept-only candidate, producing an almost
degenerate null at -0.00473. The permutation comparison therefore says that
the real expression alignment is less harmful than randomly aligned
corrections; it does not establish a stable useful gain. The line bootstrap is
the decisive failure and additional permutations cannot repair it.

## Interpretation

The data support three distinct conclusions:

1. Rhaister-O is learning much more than a conserved drug program or a single
   generic residual template.
2. External bulk DepMap expression carries real information about how a line
   responds beyond the drug mean.
3. Under this small, strict-inductive linear test, that information is largely
   redundant with or weaker than the target-line DMSO signal already used by
   Rhaister-O. It does not reliably predict Rhaister-O's remaining error.

The negative gate is specific to Public 24Q4 expression, reference-only PCA,
a 0/2/5/10-PC linear ridge family, these five released splits, and the fixed
2,000-gene pseudobulk delta endpoint. It does not prove that every DepMap
modality or nonlinear model is useless. It says the evidence required to spend
compute on those extensions is absent under the agreed criterion.

## What to do next without violating the gate

The next useful step is analysis of frozen predictions, not another model run:

- characterize the 8 helped and 13 harmed lines by lineage, DMSO similarity,
  response magnitude, and dose, treating this as hypothesis generation;
- inspect whether the small HCT-15 and LS 180 gains concentrate in particular
  drugs or pathways, with multiple-testing discipline;
- compare released State/Tx1 or SCALE predictions on an exactly matched split
  if such predictions can be aligned without retraining; and
- decide whether L1000 should remain a separate drug-prior/cross-platform
  validation aim rather than being mixed into the failed DepMap modality gate.

If the project later chooses to relax the gate, the smallest defensible extra
model would be an inner-validated scalar stack of frozen Rhaister-O and the
standalone DepMap residual predictor. CNV, mutation, dependency, and MLP should
still wait until that cheaper complementarity check is justified explicitly.

## Artifacts and provenance

| Stage | Code and output | Slurm result |
|---|---|---|
| All-line bootstrap, component removal, DMSO kNN | `scripts/evaluate_all_rhaister.py`; `bulk-data/results/rhaister_all_split_evaluation` | `216184`, completed `0:0`, 00:01:49 |
| Standalone DepMap drug-residual ridge | `scripts/run_depmap_expression_ridge.py`; `bulk-data/results/depmap_expression_residual_ridge` | `216201`, completed `0:0`, 00:02:49 |
| DepMap correction of Rhaister residual | `scripts/run_depmap_rhaister_correction.py`; `bulk-data/results/depmap_rhaister_residual_correction` | `216239`, completed `0:0`, 00:03:21 |

Summary SHA-256 values are:

- all-line evaluation: `ccd6fbb11628b58a85544e77e7d562d87c2a8582381dd974312eadf3532f19e0`;
- standalone DepMap ridge: `36910e6dfb6ee09c9c70fbe21ee03d7d4951c38742c128f61914e351abe11fd4`;
- DepMap-to-Rhaister correction: `03a3e91b729f8f969b11abcd2d1e2540abae87a7fe659840d54f05696028ab82`.

All saved prediction archives load with `allow_pickle=False`. `scontrol`
verified every final job as `COMPLETED` with exit `0:0`; the cluster's Slurm
accounting database again refused the `sacct` connection for job `216239`.
