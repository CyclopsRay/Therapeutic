# Therapeutic — Weekly Report, week ending Friday 2026-09-18

Reporting interval: [2026-09-11T18:00:00+04:00, 2026-09-18T18:00:00+04:00) — Asia/Dubai.

Measured results only, with provenance; include positive and negative findings.

## 1. DepMap-expression ridge capacity sweep — the earlier cap was not binding

Question: the original DepMap-expression residual ridge searched component grid
`{0, 2, 5, 10}` and all five outer splits selected 10, the top of the grid. Was
the reported 0.2087 drug-mean-residual line-macro Pearson limited by that cap?

Answer: **no.** Widening the grid to `{2, 5, 10, 15, 20, 25, 30, 35, 40}`
components and `{0.1, 1, 10, 100, 1000, 10000, 100000}` ridge penalties
reproduces the original selection and the original number exactly.

| Quantity | Original run | Capacity sweep |
|---|---|---|
| Selected components, splits 5–9 | 10, 10, 10, 10, 10 | **10, 10, 10, 10, 10** |
| Selected lambda, splits 5–9 | 100 (all) | **100 (all)** |
| Available components per split | — | 42, 42, 41, 41, 41 |
| Drug-residual line-macro Pearson (23 lines) | 0.208719 | **0.208719** |
| Line-bootstrap CI | [0.159096, 0.260127] | [0.159096, 0.260127] |
| Versus Rhaister-O | −0.330711 [−0.380824, −0.278754] | identical |
| Expansion gate | false | **false** |

The reference-line LOO RMSE surface has a clean interior minimum. Split 5:

| Components | 0 | 2 | 5 | **10** | 15 | 20 | 25 | 30 | 35 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|
| Best LOO RMSE | 0.028826 | 0.028337 | 0.028337 | **0.028315** | 0.028393 | 0.028463 | 0.028569 | 0.028674 | 0.028734 | 0.028783 |

Performance degrades monotonically beyond 10 components and approaches the
zero-component value by 40. The entire surface spans 1.8% of LOO RMSE
(0.028315 versus 0.028826 at zero components): that is the total contribution
of DepMap expression in this model family.

Scope: this establishes that *capacity within the whitened-PCA ridge family*
is not the limiter. An unwhitened raw-feature ridge is a different prior and
has still not been run.

Artifacts: `/nfs-stor/yuqi.lei/Therapeutic/results/depmap_expression_ridge_capacity_sweep`
(`summary.json`, per-split `ridge_selection.csv`). Job 249462, `cn-02`,
runner `scripts/run_depmap_expression_ridge.py`, submitted with
`slurm/run_depmap_expression_ridge_capacity.sbatch`. Original frozen run
untouched at `.../depmap_expression_residual_ridge`.

## 2. Cell-representation ablation — does cell-line text add drug-response information?

Four cell representations under one fixed response head, the five released
outer splits, identical component/penalty grids, reference-only fitting, and a
cohort restricted to the 23 held-out lines covered by every arm so all
comparisons are paired.

| Arm | Features | Drug-residual line-macro Pearson | Line-bootstrap CI |
|---|---|---|---|
| Rhaister-O (richer input, reference only) | target-line Tahoe DMSO | **0.5394** | [0.4822, 0.5830] |
| `expression_pca` | 19,193 DepMap expression | **0.2087** | [0.1586, 0.2603] |
| `template_text` | 768-d SapBERT of prose generated from the structured fields | 0.0930 | [0.0347, 0.1567] |
| `structured_metadata` | 122-d one-hot/numeric Model.csv context fields | 0.0780 | [0.0207, 0.1392] |
| `cellosaurus_text` | 768-d SapBERT of redacted Cellosaurus comments | 0.0294 | [−0.0095, 0.0669] |
| Drug-dose mean | — | 0.0000 | — |

Paired line-bootstrap differences:

| Comparison | Difference | CI | Verdict |
|---|---|---|---|
| `template_text` − `structured_metadata` | +0.0150 | [−0.0250, +0.0602] | **not significant** |
| `cellosaurus_text` − `structured_metadata` | −0.0486 | [−0.0978, −0.0004] | significantly worse |
| `cellosaurus_text` − `template_text` | −0.0636 | [−0.1220, −0.0093] | significantly worse |
| `structured_metadata` − `expression_pca` | −0.1307 | [−0.2118, −0.0552] | significantly worse |
| `template_text` − `expression_pca` | −0.1157 | [−0.1788, −0.0568] | significantly worse |
| `cellosaurus_text` − `expression_pca` | −0.1793 | [−0.2464, −0.1161] | significantly worse |

Predeclared decision rule — justify a text branch only if a text arm's paired
improvement over structured metadata has a bootstrap lower bound above zero
AND that arm exceeds its own line-label permutation null:
**`text_branch_justified = false`.**

Headline finding: **prose generated from the structured fields performs
statistically indistinguishably from those same fields as one-hot features**
(+0.0150, CI spans zero). This is the predicted signature of a representation
change that carries no additional measured information. Real Cellosaurus text
performs significantly *worse* than both.

Line-label permutation nulls (20 permutations; minimum attainable p = 1/21 =
0.0476, which every arm attains, so the test resolves only "above null"):

| Arm | Actual | Null mean | Null 95th pct | p |
|---|---|---|---|---|
| `expression_pca` | 0.2087 | 0.0036 | 0.0221 | 0.0476 |
| `template_text` | 0.0930 | 0.0026 | 0.0240 | 0.0476 |
| `structured_metadata` | 0.0780 | −0.0011 | 0.0169 | 0.0476 |
| `cellosaurus_text` | 0.0294 | 0.0022 | 0.0208 | 0.0476 |

Every arm carries some genuine line-specific signal; the amount is small for
all cell-identity representations and largest for expression.

Limitations affecting the `cellosaurus_text` arm specifically:

- **40 of 49 Cellosaurus texts exceeded the 512-token encoder limit and were
  truncated**; the template texts (51–79 words) were not. A negative result for
  this arm therefore means "no advantage under 512-token truncation", not
  "curated free text contains no information". Chunked encoding would be needed
  to separate these.
- Cellosaurus comment blocks are dominated by registry bookkeeping (database
  memberships, STR profiles, accessions) rather than biology.
- SapBERT's training objective is biomedical synonym/entity alignment, applied
  here with attention-masked mean pooling over paragraphs, which is outside its
  design use.

Leakage controls applied before embedding: cell-line names, synonyms,
accessions and line-specific database identifiers masked with word-boundary
matching (verified: 0 of 49 redacted texts still contain their own line name);
49 sentences containing drug-response terms removed across the corpus;
`ModelTreatment`, `PatientTreatmentStatus/Type/Details`,
`PatientTreatmentResponse`, `CulturedResistanceDrug` and `PublicComments`
excluded from the structured arm.

Artifacts: `/nfs-stor/yuqi.lei/Therapeutic/results/cell_feature_ablation`
(`summary.json`, `aggregate_summary.csv`, `per_line_summary.csv`,
`drug_cluster_bootstrap.csv`, `permutation_null.csv`); text and embeddings at
`/nfs-stor/yuqi.lei/Therapeutic/derived/cellosaurus`. Job 249475, runners
`scripts/fetch_cellosaurus_text.py`, `scripts/embed_cell_text.py`,
`scripts/run_cell_feature_ablation.py`, submitted with
`slurm/run_cell_feature_ablation.sbatch`. Encoder
`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, frozen, seed 0.

## 3. Low-rank cross-gene bilinear model on Rhaister-O's own DMSO input

Question: Rhaister-O is a per-gene diagonal map — gene `g`'s response is
regressed on the target line's DMSO level of gene `g` alone, fitted separately
per treatment. Does a bilinear model that lets gene `g` depend on the *whole*
baseline profile through a rank-K bottleneck, with the response program shared
across treatments, recover structure the diagonal leaves behind?

Model: `Δ̂[c,p,g] = anchor[c,p,g] + Σ_k z_k(c) · B_k[p,g]`, with `z(c)` a
reference-only PCA embedding of the line's log1p DMSO HVG profile. Inputs are
restricted to the plate-matched Tahoe DMSO profile, exactly what the frozen
Rhaister-O baseline consumes, so no new modality is introduced and the DepMap
expansion gate does not apply. All 25 held-out lines are evaluated (every line
has a DMSO profile); condition coverage 99.4–99.7%.

| Model | Drug-residual line-macro Pearson | Line-bootstrap CI |
|---|---|---|
| Rhaister-O (frozen diagonal) | **0.5380** | [0.4851, 0.5786] |
| `dmso_lowrank_correction` (Rhaister-O + rank-K cross-gene) | 0.5235 | [0.4677, 0.5711] |
| `dmso_lowrank_direct` (rank-K cross-gene alone) | 0.2531 | [0.1936, 0.3129] |
| Drug-dose mean | 0.0000 | — |

Paired differences versus frozen Rhaister-O:

| Level | Arm | Difference | CI | Verdict |
|---|---|---|---|---|
| full | `dmso_lowrank_correction` | −0.0071 | [−0.0211, +0.0039] | not significant |
| drug residual | `dmso_lowrank_correction` | −0.0145 | [−0.0359, +0.0025] | not significant, negative point estimate |
| PC1 removed | `dmso_lowrank_correction` | −0.0139 | [−0.0351, +0.0030] | not significant, negative point estimate |
| full | `dmso_lowrank_direct` | −0.1582 | [−0.1861, −0.1323] | significantly worse |
| drug residual | `dmso_lowrank_direct` | −0.2848 | [−0.3392, −0.2346] | significantly worse |
| PC1 removed | `dmso_lowrank_direct` | −0.3006 | [−0.3499, −0.2533] | significantly worse |

Predeclared six-part gate (≥ +0.01 drug-residual improvement, positive paired
bootstrap lower bound, ≥ +0.01 surviving PC1 removal, ≥ 4/5 splits positive,
≤ 1% RMSE degradation, above the line-block permutation null):
**both arms `passes = false`.** The correction arm satisfies only the RMSE
tolerance (ratio 1.0095) and improves in 2 of 5 splits (+0.0026, +0.0033),
against degradations of −0.0513, −0.0246 and −0.0024.

Line-block permutation nulls, 100 permutations, whole reference lines permuted
against the response tensor so drugs and doses stay together:

| Arm | Actual | Null mean | Null 95th pct | p |
|---|---|---|---|---|
| `dmso_lowrank_direct` | 0.2531 | −0.0007 | 0.0375 | **0.0099** |
| `dmso_lowrank_correction` | 0.5235 | 0.5334 | 0.5368 | **0.9901** |

The two nulls answer different questions and both are informative:

- The direct arm carries strong genuine signal (p = 0.0099, null centred at
  zero). A rank-K cross-gene model on DMSO reaches 0.2531, above the
  DepMap-expression ridge's 0.2087 on the same endpoint — consistent with DMSO
  being the assay-matched baseline — but it is far below the diagonal map.
- For the correction arm the permutation destroys only the correction term and
  leaves the frozen anchor intact, so its null is centred near Rhaister-O's own
  performance. The observed value (0.5235) is **below the null mean (0.5334)**:
  a randomly paired correction does better than the learned one. The learned
  cross-gene correction is fitting noise.

Selected rank was K = 10 with λ = 100 in all five splits for both arms, from a
grid of {1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40} against 44 available
components — an interior optimum, not a boundary selection.

Scope — what this does and does not test. It tests **cross-gene conditioning**:
letting a gene's response depend on other genes' baseline through a shared
low-rank cell embedding. It does **not** test **coefficient sharing**: shrinking
or denoising Rhaister-O's own per-(treatment, gene) diagonal slopes across
treatments and genes, which is the SVD / empirical-Bayes variant described in
the Rhaister repository but absent from its pinned executable path
(`PLAN.md` Phase 4, Track A). That remains untested.

Artifacts: `/nfs-stor/yuqi.lei/Therapeutic/results/dmso_lowrank_bilinear`
(`summary.json`, `aggregate_summary.csv`, `per_line_summary.csv`,
`drug_cluster_bootstrap.csv`, `permutation_null.csv`, per-split
`ridge_selection.csv`). Job 249519, `cn-11`, runner
`scripts/run_dmso_lowrank_bilinear.py`, submitted with
`slurm/run_dmso_lowrank_bilinear.sbatch`. Cross-fitted Rhaister-O targets use
the pinned λ = 1.0, blend = 0.95; seed 0.

## 4. Shared-control coupling — how much of the diagonal advantage is an artifact?

Question: the target is `Δ = log1p(treated) − log1p(plate-matched DMSO)` and
Rhaister-O's predictor feature is that same line's DMSO profile. The same
measurement enters the label negatively and the feature positively, so
estimation noise in the DMSO centroid induces a mechanical association that a
diagonal model can exploit — and a held-out line's Δ carries the same noise
term, so the exploit transfers. Neither existing diagnostic removes it: the
term is constant across treatments within a line (so drug-mean removal misses
it) and it lives on the query line (so reference-fitted PC1 removal misses it).

Design, with no target recomputation. All 50 lines have a DMSO centroid on all
14 plates, and 49,186 of 56,827 conditions belong to a single plate. Plates are
halved at random; conditions are restricted to those measured only on half A,
so every retained target's control lies in half A. The `same` arm builds the
feature from half A (control and feature share plates — the status quo); the
`cross` arm builds it from half B (feature measured on plates disjoint from
every control defining a target). Both average 7 plates, so the extra noise
from using half the data is matched and cancels in the comparison. Five random
halvings, 5 outer splits, 25 held-out lines, 487–603 common treatments per
halving. Targets are the frozen upstream Cell-Eval deltas throughout.

| Level | `same` | `cross` | `same − cross` | CI | Share of `same` |
|---|---|---|---|---|---|
| Full delta | 0.6398 | 0.6365 | **+0.0033** | [+0.0028, +0.0040] | 0.5% |
| Drug-mean residual | 0.5468 | 0.5410 | **+0.0057** | [+0.0042, +0.0079] | **1.0%** |

**The coupling is real but immaterial.** The bootstrap interval excludes zero
at both levels, so the effect is detectable and in the predicted direction —
sharing plates between the control and the feature does inflate the score. But
it accounts for only **1.0% of the drug-residual score** (0.0057 of 0.5468).
Rhaister-O's advantage is a genuine baseline-to-response relationship, not the
control appearing on both sides of the equation.

This is an adversarial test the headline result passed. It also puts a
quantified bound on the artifact, which is stronger than leaving the concern
unaddressed: the residual-noise channel cannot explain more than roughly one
percent of the reported performance.

Note on comparability: the `same` arm scores 0.5468 here against 0.5380 for the
full benchmark, because this analysis uses a restricted condition subset
(half-A conditions only, ~500–600 treatments) and 7-plate features. The two
numbers are close but are not the same estimand; only the within-analysis
`same` versus `cross` contrast is the result.

Artifacts: `/nfs-stor/yuqi.lei/Therapeutic/results/shared_control_diagnostic`
(`summary.json`, `aggregate_summary.csv`, `per_line_metrics.csv`,
`plate_splits.csv`). Job 258115, runner
`scripts/run_shared_control_diagnostic.py`, submitted with
`slurm/run_shared_control_diagnostic.sbatch`. Pinned λ = 1.0, blend = 0.95,
seed 0.
