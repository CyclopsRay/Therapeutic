# Tahoe–L1000 Identity, Dataset Fusion, and Deep-Learning Options

Last updated: 2026-09-10

Status: design and audit only. No new model was trained and no Slurm job was
submitted. The deep-learning stage remains gated.

## Short answer

Tahoe and LINCS do not contain the same physical cells. A match means that two
independent cultures are labelled as descendants of the same established cell
line, such as A549. Our current automated Tahoe–L1000 join is normalized-name
equivalence, strengthened for the three final lines by concordant ATCC and
provenance metadata, but it is not an STR/genotype authentication.

The datasets can be connected, but their response rows cannot be concatenated
as if they were measurements on the same scale. Tahoe supplies plate-matched
single-cell-derived log-expression deltas, L1000 supplies Level-5 CompZ/MODZ
signature scores, and DepMap supplies untreated cell-line covariates.

Deep learning is possible, but the defensible first neural experiment is a
tiny residual adapter around frozen Rhaister-O, not a large direct-response
MLP. Each outer split has roughly 51,000 condition rows but only 45 independent
response-labelled cell-line contexts.

## 1. What “the same cell line” means

A cell line is a named, established laboratory lineage. Two datasets can use
independent vials, passages, batches, media, and experimental laboratories
while still referring to A549. Therefore:

```text
same cell-line identity != same physical cells
same cell-line identity != same passage or clone
same cell-line identity != same baseline state
```

### Tahoe and DepMap

This join is strong because Tahoe directly supplies both identifiers. For
example:

```text
A549
  Tahoe Cellosaurus ID: CVCL_0023
  DepMap ModelID:       ACH-000681
```

### Tahoe and L1000

LINCS `sig_info` uses a `cell_id` such as `A549`, `HT29`, or `HS578T` but does
not provide the Tahoe Cellosaurus identifier. The current code lowercases and
removes punctuation before joining:

```text
A549    <-> A549
HT-29   <-> HT29
HS-578T <-> HS578T
```

The implementation is in `scripts/prepare_metadata.py`. It should be described
as conservative normalized-name matching, not as identifier-level proof.

For the three lines that survive the complete 24-hour drug/dose protocol,
LINCS `cell_info` supplies consistent independent provenance:

| Tahoe name / ID | L1000 ID | L1000 provider | Primary site | Donor annotation |
|---|---|---|---|---|
| A549 / CVCL_0023 | A549 | ATCC CCL-185 | lung | 58, male |
| HT-29 / CVCL_0320 | HT29 | ATCC HTB-38 | large intestine | 44, female |
| HS-578T / CVCL_0332 | HS578T | ATCC HTB-126 | breast | 74, female |

These fields agree with the corresponding Tahoe/DepMap identity annotations.
They make the matches convincing at the catalogue-line level, but passage,
subclone, media, and laboratory drift remain real sources of domain shift.

Before any training-data fusion, create a versioned identity registry with:

- Tahoe name and Cellosaurus ID;
- DepMap ACH ID;
- L1000 cell ID and aliases;
- provider catalogue ID, tissue, sex, and species;
- a `confirmed / likely / ambiguous` confidence field;
- Cellosaurus alias and, if obtainable, SNP/STR evidence.

For strict zero-shot evaluation, every confirmed alias of an outer-test line
must be excluded from L1000 training. Excluding only the exact spelling or only
the same drug-dose is insufficient.

## 2. Exact overlap between Tahoe and L1000

### Cell lines

Phase I metadata has seven normalized-name candidates:

```text
A549, HS-578T, HT-29, LoVo, NCI-H596, RKO, SW480
```

Phase II has three:

```text
A549, HS-578T, HT-29
```

After requiring a structure-validated compound, exact 24-hour exposure, and a
nearest L1000 dose within a two-fold ratio, both releases retain only A549,
HT-29, and HS-578T.

### Drugs, signatures, and conditions

| Stage | Phase I / GSE92742 | Phase II / GSE70138 |
|---|---:|---:|
| Tahoe drugs matched by any name/structure evidence | 157 | 147 |
| Structure-validated drugs before time/dose filtering | 131 | 79 |
| Final drugs | 97 | 78 |
| Unique selected Level-5 signatures | 199 | 332 |
| Signature-to-Tahoe mappings | 202 | 340 |
| Aggregated line–drug–dose conditions | 171 | 304 |
| Final cell lines | 3 | 3 |

Several signatures can represent one biological condition. The current code
means those signatures within each matched `(line, drug, Tahoe dose)` group.
Phase I and Phase II remain separate analyses because their primary
concordance results differ.

## 3. What one L1000 row contains

Conceptually, one Level-5 row is one perturbation signature:

```text
metadata:
  sig_id
  cell_id
  pert_id / pert_iname
  pert_type = trt_cp
  dose and unit
  time and unit
  contributing distil signatures

expression:
  12,328 Level-5 gene signature scores
```

The logical full matrices are:

| Release | Signatures × gene rows |
|---|---:|
| Phase I | 473,647 × 12,328 |
| Phase II | 118,050 × 12,328 |

The GCTX file may store one axis first internally, but one machine-learning
sample is a length-12,328 signature vector. It is not a single cell. Level 5
has already combined replicate-level information into a signature score.

Current compact extraction and condition aggregation are:

| Release | Extracted signatures | Condition-level pairs |
|---|---:|---:|
| Phase I | 199 × 1,365 | 171 × 1,365 |
| Phase II | 332 × 1,365 | 304 × 1,365 |

The primary concordance calculation slices these arrays again to 79 directly
measured common genes: `171 × 79` and `304 × 79`.

## 4. Gene overlap: why 946, 1,365, 1,076, and 79 are all correct

These numbers describe different intersections:

| Intersection | Count | Interpretation |
|---|---:|---|
| L1000 measured landmarks | 978 | Directly measured by the L1000 assay |
| Landmarks present in full Tahoe gene metadata | 946 | Potential clean bridge if a new full-Tahoe endpoint is built |
| Rhaister fixed 2K found anywhere in L1000 distributed rows | 1,365 | Current broad overlap; many L1000 values are inferred |
| Rhaister 2K marked BING in L1000 | 1,076 | 79 measured landmarks plus 997 best-inferred genes |
| Rhaister 2K directly measured landmarks | **79** | Clean current primary panel |
| Other inferred genes in the 1,365 panel | 289 | Neither landmark nor BING |

Thus BING and landmark are not disjoint panels. The 1,365 current overlaps
partition as 79 measured landmarks, 997 additional BING inferred genes, and
289 other inferred genes.

Using 946 measured genes is scientifically attractive, but it requires
building a new 946-gene Tahoe delta endpoint from fuller Tahoe material. It
would be a parallel cross-platform benchmark, not an improvement on the
original fixed-2K Rhaister score.

## 5. Can the datasets be merged?

### Invalid direct merge

Do not row-bind Tahoe and L1000 values under one MSE loss:

```text
Tahoe target:  plate-matched pseudobulk log-expression delta
L1000 target:  platform-specific Level-5 standardized signature score
DepMap row:    untreated bulk/molecular features, no perturbation outcome
```

Shared symbols do not remove scale, assay, batch, dose, culture, or response-
construction differences. The weak and phase-dependent observed concordance
is direct evidence that a naive merge is unsafe.

### Valid ways to connect them

1. **External validation.** Keep Tahoe as the target and ask whether L1000
   direction, rank, or pathway effects agree. This is already implemented.
2. **Late-fusion drug prior.** Aggregate L1000 signatures over non-target
   lines to create a drug/dose or pathway prior, calibrate it only with Tahoe
   reference lines, and test whether it improves frozen Rhaister residuals.
3. **Representation pretraining.** Learn a drug or pathway representation from
   L1000, then use a Tahoe-specific output head. Do not share raw response
   scale.
4. **Multi-task/domain-specific heads.** Share a small latent cell–treatment
   interaction but retain separate Tahoe and L1000 decoders and losses.
5. **DepMap as a context branch.** Supply untreated target-line context rather
   than treating DepMap as another perturbation label source.

The best first fusion is option 2, but it should be linear or very low rank
before it is nonlinear.

### Leakage boundary for an L1000 prior

For every outer and inner held-out Tahoe line:

- exclude all of that line's Tahoe treated outcomes;
- exclude all aliases of that line from L1000 treated signatures;
- fit dose mapping, calibration, PCA, gene/pathway selection, and model
  hyperparameters on training/reference lines only;
- allow Tahoe DMSO for the target line;
- allow target-line DepMap baseline only in the declared practical/transductive
  variant;
- never use target-line L1000 response while calling the task strict zero-shot.

## 6. How well does current Rhaister-O work?

| Evaluation | Drug mean | DMSO 1NN | DMSO 3NN | Rhaister-O |
|---|---:|---:|---:|---:|
| Full Tahoe delta | 0.4434 | 0.2985 | 0.3405 | **0.6314** |
| Drug-mean residual | 0.0000 | 0.1472 | 0.1388 | **0.5380** |
| Reference-PC1 removed | -0.0028 | 0.1282 | 0.1174 | **0.5259** |

These are line-macro Pearson values over 25 distinct held-out lines and 19,268
test conditions. Every per-line Rhaister drug-cluster interval remains
positive, although the PC1-removed line scores range from 0.0648 to 0.6850.

The model is therefore strong, fast, and difficult to beat. It is not a solved
biological problem: it predicts a mean 2,000-gene response, assumes known
drugs, uses target-line DMSO, has no cross-gene mixing, and hides substantial
line heterogeneity.

DepMap expression alone predicts drug-mean residuals at 0.2087, but the strict
DepMap correction changes Rhaister residual Pearson by only +0.000357 with a
line-bootstrap interval of [-0.002679, 0.003610]. The previous modality/MLP
expansion gate therefore remains false.

L1000 direct transfer is also weak: the primary landmark result is 0.0605 in
Phase I and 0.0153 in Phase II, with the latter not passing the bootstrap/null
test. On the one exactly aligned frozen Rhaister line, HS-578T, Rhaister is
much stronger, but that single-line comparison is not a general leaderboard.

## 7. Can deep learning improve Rhaister-O?

Yes in principle, mainly because Rhaister-O is diagonal: baseline gene `g`
directly predicts response gene `g`, so it cannot learn compact cross-gene or
pathway interactions. But more capacity is not automatically more information.

The effective cell-context sample size is the critical constraint:

```text
Tahoe compact rows:         56,827
cell lines total:               50
reference lines per split:      45
contexts per treatment:        <=45
```

Drug-dose rows and millions of cells improve response estimation, but they do
not produce thousands of independent examples of how a new cell line behaves.
A large MLP can memorize drug means and line identities while appearing to fit
many observations.

Deep single-cell population models already exist in this area, including
State, Tahoe-x1 plus State Transition, and SCALE. They show that neural
population prediction is possible, but their model scales, pretraining data,
split definitions, and endpoints are not automatically comparable to the five
released Rhaister-O splits. In particular, a checkpoint pretrained on treated
cells from an eventual test line is transductive under this project's stricter
representation-learning boundary even if downstream fitting is called
zero-shot.

Primary local references:

- `papers/state_v2.pdf` and <https://github.com/ArcInstitute/state>;
- `papers/tahoe-x1_v1.pdf` and <https://github.com/tahoebio/tahoe-x1>;
- `papers/scale_2603.17380v3.pdf` and <https://arxiv.org/abs/2603.17380>.

## 8. Recommended bounded model, if the gate is explicitly reopened

Keep frozen Rhaister-O as the main prediction and learn only a small
correction:

\[
\hat y_{c,t}
=\hat y^{R}_{c,t}
+B^\top f\left([
\operatorname{PCA}_{16}(H_{c,t}),
e_{drug},e_{dose}
]\right).
\]

Suggested components:

- `H`: plate-matched 2,000-gene DMSO centroid;
- DMSO PCA: 16 components, refitted inside every inner line fold;
- drug embedding: 379 × 16;
- dose embedding: 3 × 4;
- interaction network: `36 -> 32 -> 8`;
- output: 8 residual coefficients in an error basis fitted only on inner-
  training cross-fitted Rhaister residuals;
- final layer initialized at zero, so the starting prediction is exactly
  frozen Rhaister-O.

The leakage-safe training target is:

\[
\epsilon_{c,t}
=y_{c,t}-\hat y^{\mathrm{LOO\ Rhaister-O}}_{c,t}.
\]

If the 8 × 2,000 output basis is fixed from training-only error PCA, the trainable
part is only about 7,500 parameters. If the basis is also trained, the total is
about 23,500, but fixing it is safer with 45 contexts.

Required comparators are zero correction, scalar stacking, reduced-rank linear
regression, a rank-matched bilinear interaction, and the tiny nonlinear
adapter. Deep learning earns a claim only if it stably beats the capacity-
matched linear/bilinear versions.

The currently materialized arrays are sufficient for this pseudobulk pilot:

| Artifact | Shape |
|---|---:|
| Tahoe compact delta | 56,827 × 2,000 |
| Plate-specific DMSO table | 700 × 2,000 plus IDs |
| DepMap expression | 47 × 19,193 |
| L1000 Phase-I selected signatures | 199 × 1,365 |
| L1000 Phase-II selected signatures | 332 × 1,365 |

No full Tahoe download is needed for this pilot. A true single-cell
distribution model would require fuller Tahoe raw data and a much larger GPU
project. The current environment also lacks PyTorch; a separate pinned
environment on `/l` would be required before neural training.

## 9. Recommended experimental order

### Track A: improve Rhaister on the full 2K benchmark

1. Revisit the low-rank/SVD and empirical-Bayes shrinkage ideas described in
   the Rhaister repository but absent from the pinned executable delta path.
2. Fit a reduced-rank linear/bilinear correction to cross-fitted Rhaister
   residuals using DMSO context only.
3. Only if it has a stable signal, run the tiny nonlinear residual adapter.
4. Add DepMap expression as an ablation, not as the default input.

### Track B: test L1000 fusion on a predeclared secondary subset

1. Freeze a stronger cell-line identity registry.
2. Complete dose and pathway sensitivity, with Phase I exploratory and Phase
   II confirmatory.
3. Build a target-line-excluded L1000 drug prior.
4. Test a linear/low-rank late-fusion adapter on the 79 directly measured genes
   and identical eligible Tahoe conditions.
5. Only after this gate passes, add an L1000 branch to the nonlinear adapter.

The current compact L1000 features cover only about 26–28% of the frozen Tahoe
test conditions per split. They must therefore be reported as a predefined
coverage subset. Broader L1000 drug-prior coverage would require a new compact
extraction across non-Tahoe L1000 lines; it does not require full matrix
materialization but does require a monitored Slurm preprocessing job.

### Proposed gate

Require all of the following before model expansion:

1. primary drug-residual improvement point estimate of at least +0.01;
2. 95% held-out-line bootstrap lower bound greater than zero;
3. improvement above a fully refitted drug/line-label permutation null;
4. directionally consistent Phase-I and Phase-II L1000 evidence when L1000 is
   used;
5. no material loss in RMSE, MAE, sign, or pathway metrics;
6. a nonlinear model must beat its capacity-matched linear/bilinear control.

The +0.01 threshold is a new proposed meaningful-effect threshold, not a result
from the completed protocol; it requires user approval before use.

## Bottom line

The datasets are complementary, not interchangeable. The immediate scientific
opportunity is not to replace Rhaister-O with a large network. It is to test
whether a carefully authenticated, target-line-excluded L1000 drug prior or a
small cross-gene residual interaction contains information that Rhaister-O's
diagonal mapping misses. Failure of those low-capacity tests would be strong
evidence against escalating to a larger multimodal model.
