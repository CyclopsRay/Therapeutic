# Therapeutic — Weekly Report, week ending Friday 2026-10-02

Reporting interval: [2026-09-25T18:00:00+04:00, 2026-10-02T18:00:00+04:00) — Asia/Dubai.

Measured results only, with provenance; include positive and negative findings.

## 1. EmeraldBay: acquisition, sanity checks, and the full method ladder re-run (2026-09-29)

All numbers below are new measurements on `tahoebio/EmeraldBay` (CC-BY 4.0, public,
not gated), HF revision `f2a0be6b02f731553657f0115c345b20bb020ede`.

### 1.1 Acquisition

| item | value | source |
|---|---|---|
| expression shards | 116, 57,713,404,260 bytes | job 284823, `raw/emerald-bay/f2a0be6.../MANIFEST.tsv` |
| sha256 vs HF LFS oid | **116/116 match** | `raw/emerald-bay/f2a0be6.../SHA256SUMS` |
| cells | **1,831,648** (= dataset card) | job 284842, `derived/emerald_bay/pseudobulk_v1/provenance.json` |
| lines / samples / conditions | 52 / 96 / 93 | same |
| single-drug treatments | 44 (19 drugs x 2-3 doses), all doses in uM | `derived/emerald_bay/sanity_v1/sanity.json` |
| DMSO_TF samples | 4; **no treatment has a replicate sample** | same |
| lines shared with Tahoe's 50 | 47; 23 of Tahoe's 25 held-out lines | metadata join on Cellosaurus ID |
| drugs shared with Tahoe | 13 of 27 by name; 9 of the single-drug set matched in the Tahoe compact cache | same |

Paths are under `/nfs-stor/yuqi.lei/Therapeutic/`. Pseudobulk: Tahoe 2K panel mapped by
Tahoe token ID (all 2,000 genes; 27 have older symbols in EmeraldBay's table). EmeraldBay has
no leading marker token (min token 3, integer counts).

### 1.2 Sanity checks (`scripts/emerald_bay_sanity.py`, `derived/emerald_bay/sanity_v1/sanity.json`)

| check | result |
|---|---|
| sensitivity recomputed from full-QC cell counts vs released `growth_rate` | Spearman **0.889** (Pearson 0.31, driven by zero-count outliers), n = 4,732 |
| cross-dataset identity: EB DMSO vs Tahoe DMSO centroids (log1p, centred) | **42/47 rank-1**; 42/44 after dropping 4 low-coverage lines |
| failures | 4 low-coverage lines (below) + **NCI-H596 `CVCL_1571`: rank 43/50, self r = -0.103** despite 17,344 cells |
| DMSO split-half reproducibility (centred) | median **0.993**; worst 4 are the low-coverage lines (0.68-0.74) |
| treatment delta split-half Pearson | median 0.606; by cell-count quartile 0.28 / 0.55 / 0.65 / 0.81 |
| EB 5-day vs Tahoe 24-h delta, same line and drug (926 pairs, 9 drugs) | matched median **0.030** vs mismatched-drug 0.004; matched > mismatched in 60% |
| MAPK output (DUSP4, SPRY1, EGR1, IER3) under Encorafenib 0.1 uM | BRAF-V600 lines -0.003 vs other lines +0.055 (n = 6 vs 40); DUSP4 -0.06 in V600 lines. Trametinib is combination-only in EB. |

**Low-coverage lines, excluded before any model fit on cell count alone:** NCI-H661
`CVCL_1577` (127 cells total, 10 DMSO), CFPAC-1 `CVCL_1119` (168/11), NCI-N87 `CVCL_1603`
(188/8), A498 `CVCL_1056` (199/14). Median line has 22,412 cells.

**NCI-H596.** EmeraldBay NCI-H596's best Tahoe match is J82 (r 0.24); Tahoe NCI-H596's
best EmeraldBay match is NCI-H460 (0.38). Together with Tahoe NCI-H596 ranking 42/47 against
DepMap (week 09_18), the same Cellosaurus label disagrees across all three sources.

After exclusions the grid is complete: 48 lines x 44 treatments = 2,112 line-conditions,
every one with >= 10 cells. A 30-cell floor would drop exactly the drug-killed conditions
(outcome-dependent) and break Rhaister-O's complete-coverage requirement (job 284850,
superseded: 30/44 diagonal coverage).

**Noise ceiling (split halves, Spearman-Brown corrected, all 48 lines):** reliability 0.691
full delta, 0.613 drug residual, i.e. a perfect predictor would reach roughly **0.83** full
and **0.78** drug-residual Pearson against the full-data truth. Computed ad hoc on the login
node from `pseudobulk_v1`; approximate.

### 1.3 Protocol

Existing Tahoe runners executed **unchanged** through `scripts/run_on_emerald_bay.py`, which
replaces only the three data loaders. Target = mean log1p(count) treated minus same-line
pooled DMSO_TF (Tahoe 2K cell_eval convention); query feature = line DMSO mean raw counts
(log1p applied inside the runner, as for Tahoe). No plates: one pseudo-plate. Splits 5-9 =
Tahoe's split-s held-out lines present in EB (22 lines after exclusions, 4-5 per split); all
44 treatments are test conditions. Outputs: `results/emerald_bay/<method>/`.

### 1.4 Results — drug-residual line-macro Pearson unless stated

| model / arm | EmeraldBay | Tahoe (reference) | job |
|---|---|---|---|
| zero delta, full | 0.000 | 0.000 | 284851 |
| global mean, full | 0.303 | -- | 284851 |
| drug mean, full | 0.493 | -- | 284852 |
| **Rhaister-O, full (split-mean)** | **0.632 ± 0.043** | 0.631 ± 0.068 | 284851 |
| Rhaister-O, full (line-macro, 22 lines) | 0.636 | -- | 284852 |
| nearest DMSO line, 1-NN / 3-NN | 0.176 / 0.206 | -- | 284852 |
| **Rhaister-O, drug residual** | **0.497** | 0.538 | 284852 |
| Rhaister-O, PC1 removed | 0.501 | 0.526 | 284852 |
| shared-gamma (fully pooled slopes) | 0.391 | -- | 284853 |
| coefficient sharing (uniform shrink) | 0.4985, **+0.0013** [low -0.0031], gate false | +0.0019 [low +0.0010] | 284853 |
| chemistry shrink: moa / morgan / full | **+0.0000** each (inner selection w = 1 in every split) | +0.000006 / -0.00023 / -0.00024 | 284854 |
| low-rank DMSO cross-gene correction | 0.4978, **+0.0006** [-0.0029, +0.0043], gate false | -0.0145 | 284856 |
| low-rank DMSO cross-gene direct | 0.228 | 0.253 | 284856 |
| DepMap-expression correction (20 lines) | **-0.0052** [-0.0092, -0.0015]: significantly worse | +0.0004 | 284857 |
| DepMap ridge alone (20 lines) | 0.136 | 0.209 | 284859 |
| cell representation, expression PCA | 0.136 | 0.209 | 284858 |
| structured metadata / template text / Cellosaurus text | 0.064 / 0.067 / 0.033 | 0.078 / 0.093 / 0.029 | 284858 |

Every predeclared gate that failed on Tahoe also fails on EmeraldBay. The ordering of cell
representations (expression > metadata ≈ template text > Cellosaurus text) replicates.
Permutation p values sit at their floors for arms with negative or zero gains, repeating the
gate defect recorded in week 09_25 §9.

### 1.5 Reference-line count and quality (jobs 284855, 284862)

Random draws (valid in both runs; 5 draws per m):

| m | 5 | 10 | 20 | 30 | 35 | 40 | 43 |
|---|---|---|---|---|---|---|---|
| drug residual | 0.178 | 0.306 | 0.425 | 0.467 | 0.480 | 0.492 | 0.496 |

The last 8 lines add +0.016 (35→43); the last 3 add +0.004. The curve is flattening but not flat.

Quality ordering: job 284855's quality arms are **invalid** (one pseudo-plate makes the
plate-reproducibility score NaN, so ordering is arbitrary). Job 284862 re-ran with DMSO
split halves as pseudo-plates (`EB_HALF_PLATES=1`; half-mean vs pooled centroid r = 0.9999998):

| m | best QC | random | worst QC | best - random |
|---|---|---|---|---|
| 10 | 0.334 | 0.306 | 0.288 | +0.028 |
| 20 | 0.442 | 0.425 | 0.399 | +0.017 |
| 30 | 0.481 | 0.467 | 0.446 | +0.014 |
| 40 | 0.496 | 0.492 | 0.487 | +0.004 |

Direction replicates Tahoe (best > random > worst at every m ≤ 40); magnitude at m = 40 is
about 7x smaller than Tahoe's +0.030. Quality spread after exclusions is narrow (split-half
QC 0.93-1.00).

### 1.6 Keeping the four low-coverage lines (jobs 284863, 284864; `EB_EXCLUDE_LINES=`, `EB_MIN_CELLS=1`)

| same 22 held-out lines | excluded (primary) | included |
|---|---|---|
| Rhaister-O full | 0.636 | 0.544 |
| Rhaister-O drug residual | **0.497** | **0.225** |

Mechanism: those lines lack whole conditions, and Rhaister-O fits a treatment's slope only if
**every** reference line has it, so diagonal coverage falls to 19-25/44 and the rest fall back
to additive ALS. This is an estimator fragility (complete-case requirement), not a pure
data-quality effect.

## 2. Overnight pre-registered experiments (2026-09-29; design and gates in `10_02_record.md`)

All result paths below are under `/nfs-stor/yuqi.lei/Therapeutic/results/`.

### 2.1 E4 — NCI-H596 (`CVCL_1571`) identity against DepMap 24Q4 expression

`emerald_bay/nci_h596_identity.txt`; 1,987 panel genes, log1p DMSO vs DepMap log TPM, both centred over lines.

| source | lines | rank-1 | median self r | NCI-H596 rank | NCI-H596 self r | NCI-H596 best DepMap matches |
|---|---|---|---|---|---|---|
| EmeraldBay DMSO | 41 | 40 | 0.541 | **1** | 0.388 | NCI-H596 0.388, J82 0.201, H4 0.107 |
| Tahoe DMSO | 44 | 43 | 0.579 | **32** | -0.038 | NCI-H460 0.147, SHP-77 0.136, NCI-H1573 0.086 |

EmeraldBay's NCI-H596 is DepMap NCI-H596. Tahoe's cells labelled NCI-H596 are not, and resemble no DepMap
line strongly (best NCI-H460). The only other non-rank-1 case is HT-29 in EmeraldBay (rank 2).

### 2.2 E3 — minimum detectable effect (80% power, two-sided α = 0.05, paired over held-out lines)

`power_analysis/mde.csv`; paired per-line drug-residual differences from the frozen runs.

| experiment | n lines | mean gain | SD of paired diff | MDE | lines for +0.01 |
|---|---|---|---|---|---|
| Tahoe coefficient sharing | 25 | +0.0019 | 0.0023 | 0.0014 | 1 |
| Tahoe chemistry (chem_full) | 25 | -0.0002 | 0.0008 | 0.0005 | 1 |
| Tahoe low-rank DMSO correction | 25 | -0.0145 | 0.0499 | 0.0292 | 196 |
| Tahoe DepMap correction | 25 | +0.0003 | 0.0076 | 0.0045 | 5 |
| EB coefficient sharing | 22 | +0.0013 | 0.0107 | 0.0067 | 9 |
| EB low-rank DMSO correction | 22 | +0.0006 | 0.0089 | 0.0056 | 7 |
| EB DepMap correction | 22 | -0.0047 | 0.0086 | 0.0054 | 6 |

The resolving power depends on how far a model departs from Rhaister-O. Near-baseline modifications are
resolved at 0.0005-0.007, so their null results are real near-zero effects, not resolution failures.
A structurally different model (low-rank cross-gene) needs ~0.03.

EmeraldBay sensitivity, paper's released per-split R² (+tx vs sensitivity only): paired mean +0.054, SD 0.051
over 5 splits, **MDE 0.084**. The reported gain is below what 5 splits can resolve; ~8 splits would be needed
at that effect size.

### 2.3 E1 — cell-context slopes on EmeraldBay (job 284888, `emerald_bay/context_slopes`)

Drug-residual line-macro Pearson, 22 held-out lines; Rhaister-O 0.497239. 50 line-block permutations; p floor 0.020.

| arm | selected (all splits) | score | gain | bootstrap low | per-split gains | perm p | gate |
|---|---|---|---|---|---|---|---|
| **kernel** | h = 0.5 ×5 | 0.5031 | **+0.0059** | **+0.0026** | +0.0016 +0.0070 +0.0076 +0.0066 +0.0076 | 0.020 | fail (< +0.01) |
| **varying_coef** | K = 20, λ = 100 ×5 | 0.5012 | **+0.0040** | **+0.0028** | +0.0026 +0.0046 +0.0054 +0.0048 +0.0027 | 0.020 | fail (< +0.01) |
| reliability | α = 20 ×4, 50 ×1 | 0.4995 | +0.0022 | -0.00004 | +0.0045 +0.0016 -0.0012 +0.0042 +0.0008 | 0.020 | fail |
| quadratic | λ = 100 ×5 | 0.4975 | +0.0003 | -0.0001 | | 0.039 | fail |
| kg_neighbor | λ = 100 ×5 | 0.4973 | +0.0001 | -0.0001 | | 0.078 | fail |
| kg_random (control) | λ = 100 ×5 | 0.4973 | +0.0001 | +0.00001 | | 0.039 | fail |

- The kernel and varying-coefficient arms are the first modifications in this project with a positive bootstrap
  lower bound, 5/5 positive splits and p at the permutation floor. They fail only the +0.01 effect-size threshold.
- **The Reactome prior is indistinguishable from its degree-preserving random control** (+0.00009 vs +0.00008).
- `varying_coef` selected K = 20, the top of the (already widened) grid, in every split.

### 2.4 E1 — cell-context slopes on Tahoe (jobs 284889, 284890; `context_slopes/`)

Drug-residual line-macro Pearson, 25 held-out lines; Rhaister-O 0.537951.

| arm | selected (splits 5-9) | score | gain | bootstrap low | per-split gains | perm p | Tahoe gate |
|---|---|---|---|---|---|---|---|
| **reliability** | α = 20 ×5 | **0.5554** | **+0.0175** | **+0.0144** | +0.0137 +0.0180 +0.0144 +0.0220 +0.0191 | 0.020 | **PASS** |
| **kernel** | h = 0.5 ×5 | **0.5514** | **+0.0134** | **+0.0086** | +0.0065 +0.0227 +0.0132 +0.0198 +0.0049 | 0.020 | **PASS** |
| varying_coef | (5,10) (20,100) (3,10) (5,10) (5,10) | 0.5390 | +0.0011 | -0.0017 | +0.0008 +0.0045 +0.0032 -0.0031 -0.0001 | 0.020 | fail |

These are the first arms in the project to pass the full predeclared gate on Tahoe (≥ +0.01, bootstrap low > 0,
PC1-removed ≥ +0.01, ≥ 4/5 splits, RMSE ratio ≤ 1.01, permutation p ≤ 0.05, point estimate > 0; details in
each `summary.json`). **Neither passes the pre-registered two-dataset criterion:** on EmeraldBay kernel is
+0.0059 (fails the size threshold only) and reliability is +0.0022 (not significant). `varying_coef` replicates
on neither dataset at gate strength.

**Mechanism check (`context_slopes/kernel_weight_vs_quality.txt`).** Kernel weights at h = 0.5 are unrelated to
line quality (per-split Spearman between mean weight and QC: -0.34, +0.26, -0.29, -0.08, +0.17). NCI-H596's mean
weight is 1.05, i.e. not down-weighted. Effective number of reference lines per query is 30-44 of 45. The kernel
gain is therefore locality, not exclusion. Reliability weighting at α = 20 in effect removes two reference
lines, NCI-H596 `CVCL_1571` (normalised weight 0.0003) and NCI-H2122 `CVCL_1531` (0.0026); all others keep weight
0.59-1.0. The two gains therefore come from different mechanisms.

EmeraldBay's split-half QC spans only 0.93-1.00 once the four low-coverage lines are excluded, which is
consistent with reliability weighting having little to act on there.

### 2.4b E1 — quadratic and knowledge-graph arms on Tahoe (job 284891; `context_slopes/quadratic_kg`)

Rhaister-O 0.537951, 25 lines.

| arm | selected | gain | bootstrap low | per-split gains | perm p | gate |
|---|---|---|---|---|---|---|
| quadratic | λ = 100 ×4, 10 ×1 | -0.0003 | -0.0014 | +0.0004 +0.0007 +0.0004 -0.0000 -0.0030 | 0.451 | fail |
| kg_neighbor (Reactome) | λ = 10 ×5 | +0.0009 | -0.0002 | -0.0019 +0.0027 +0.0007 +0.0020 +0.0009 | 0.020 | fail |
| kg_random (control) | λ = 100 ×5 | +0.0002 | +0.00001 | | 0.020 | fail |

Across both datasets the Reactome prior adds at most +0.0009 and is not separable from its degree-preserving
random control. Nonlinearity in the same-gene baseline (quadratic) adds nothing.

### 2.5 E2 — unseen-drug task, drug encoder (Tahoe; job 284894; `unseen_drug/`)

Secondary task (all lines seen; whole drugs held out, 5 drug folds, 379 drugs × 3 doses). Per-(drug, dose, line)
Pearson, drug-cluster bootstrap (20,000). Zero or constant residual predictions (line-dose mean; kernel models for
the 4 drugs without SMILES: Erdafitinib, Sacubitril/Valsartan, Selinexor, Verteporfin) were set to Pearson 0 in a
recomputation from `per_condition.csv` (`summary_nan_as_zero.txt`); the in-job summary had NaN for those models.

| model | full delta | residual from line-dose mean |
|---|---|---|
| line-dose mean | **0.421** [0.397, 0.444] | 0 |
| MOA-class mean | 0.413 | **0.148** [0.123, 0.174] |
| Morgan kNN (k by inner LOO) | 0.348 | 0.096 |
| Morgan kernel ridge | 0.416 | 0.093 |
| Morgan + MOA kernel ridge (= encoder in all 5 folds) | 0.420 | 0.117 |

- **Gate (encoder − MOA-class mean, residual): -0.032 [-0.057, -0.006], fails; the encoder is significantly worse.**
- Adding MOA to the Morgan kernel helps (+0.024 [+0.010, +0.038]). On the full delta no model beats the line-dose mean.

### 2.6 E5 — EmeraldBay few-shot sensitivity (revised design; job 284901; `emerald_bay/sensitivity_fewshot/`)

6 panel drugs per split (seeded, differ from the paper's), 22 held-out lines, targets = remaining drugs' growth rates.

| model | mean split R² (SD) | line-macro Pearson |
|---|---|---|
| drug mean | -0.066 (0.057) | 0.040 |
| tx only (panel-delta PCs) | -0.010 (0.070) | 0.178 |
| sens (panel growth) | 0.332 (0.058) | 0.476 |
| sens + tx | 0.341 (0.091) | 0.495 |
| sens + DMSO PCs | **0.356** (0.080) | **0.517** |

- **Gate (sens+tx − sens, line bootstrap): +0.019 [+0.003, +0.037], passes.** Split R² differences are +0.007,
  -0.056, +0.027, +0.046, +0.022.
- sens + DMSO − sens: +0.041 [-0.007, +0.113], not significant but larger. Untreated baseline expression helps at
  least as much as the panel transcriptional response.
- This agrees in direction with the paper's 0.26 → 0.31. Its split-level variability is of the same order as in
  the paper.

### 2.7 E6 — unwhitened raw-feature DepMap ridge (jobs 284902 Tahoe, 284903 EB)

| dataset | standalone drug residual | whitened (earlier) | minus Rhaister-O | selected |
|---|---|---|---|---|
| Tahoe (23 lines) | **0.260** [0.207, 0.315] | 0.209 | -0.279 [-0.329, -0.230] | K = 20 (×4), 10; λ = 1e5 |
| EmeraldBay (20 lines) | **0.178** [0.139, 0.219] | 0.136 | -0.320 [-0.361, -0.283] | K = 5-20; λ = 1e5 |

Without whitening the ridge predicts better on both datasets (+0.05 / +0.04) but stays far below Rhaister-O.
Expansion gate false on both. `results/depmap_expression_ridge_unwhitened`, `results/emerald_bay/depmap_expression_ridge_unwhitened`.

### 2.8 NCI-H596 removed from Tahoe reference fitting (jobs 284898, 284899; `tahoe_exclude_nci_h596/`)

`CVCL_1571` is never held out, so only reference fitting changes (44 reference lines per split; the drug mean is
refitted from 44 lines as well).

| Rhaister-O, 25 held-out lines | with NCI-H596 (frozen) | without |
|---|---|---|
| full, split mean | 0.6315 | **0.6376** |
| drug residual, line-macro | 0.5380 | **0.5431** (+0.0051) |
| PC1 removed | 0.5259 | **0.5302** |

One mislabelled reference line costs Rhaister-O about +0.005, about a third of the reliability arm's +0.0175.
The reliability arm also removes NCI-H2122 and mildly down-weights other lines.

### 2.9 Exploratory, post-hoc (designed after seeing §2.3-2.4; not gate tests)

| run | dataset | selected | gain | bootstrap low | per-split | perm p |
|---|---|---|---|---|---|---|
| varying_coef, K ∈ {20,30,40} (284922) | EB | K = 40, λ = 100 ×5 | +0.0050 | +0.0031 | all 5 positive | 0.020 |
| kernel × reliability (284923) | EB | h = 0.5, α = 50 ×5 | **+0.0091** | **+0.0048** | +0.0082 +0.0109 +0.0065 +0.0121 +0.0075 | 0.020 |
| kernel × reliability (286426, reduced grid) | Tahoe | running (285295 killed at the 10 h limit after 4 splits; the node was heavily loaded) | | | | |

### 2.10 Confound test: are the kernel and reliability gains just the mislabelled line? (job 284937)

`CVCL_1571` removed from the data entirely, so both the baseline and the arms are refitted on 44 reference
lines. Baseline Rhaister-O 0.543088; 25 held-out lines; same grids, same inner selection.

| arm | gain with NCI-H596 present | gain with NCI-H596 removed | selected | gate without it |
|---|---|---|---|---|
| **kernel** | +0.0134 [+0.0086] | **+0.0137 [+0.0104]** | h = 0.5 ×5 | **PASS** |
| **reliability** | +0.0175 [+0.0144] | **+0.0077 [+0.0063]** | α = 20 ×5 | **fail** (< +0.01) |

- **The kernel gain is unchanged (+0.0134 → +0.0137) and still passes every gate condition.** It is not an
  indirect exclusion of the mislabelled line.
- **Reliability weighting loses 56% of its gain** (+0.0175 → +0.0077) and no longer clears +0.01. Most of what it
  bought on Tahoe was the removal of one mislabelled reference line, and it should be described as a data-quality
  correction rather than a modelling advance. It stays significantly positive, so a smaller residual effect from
  down-weighting other low-quality lines is real.
- The two gains are therefore not additive in the way the with-H596 numbers suggested, and locality is the part
  that survives.

## 3. Confirmation experiments on never-held-out cell lines (pre-registered C1/C2, 2026-09-29)

The benchmark's five splits hold out 25 of Tahoe's 50 lines; the other 25 have only ever been reference
lines. C1 makes five fresh splits from those 25 (seeded, 5 each) and scores them with **the hyperparameters
fixed a priori** (kernel h = 0.5, reliability alpha = 20 -- the values the inner criterion picked in all five
original splits), so no selection happens at all, and with 200 permutations instead of 50. C2 does the same on
EmeraldBay's 26 never-held-out lines. Not fully independent: those lines were in the reference set when h and
alpha were chosen. The independence is in the outer scoring.

### 3.1 New Tahoe splits are of the same difficulty (jobs 286438, 286439)

| | original 25 held-out lines | 25 never-held-out lines |
|---|---|---|
| Rhaister-O full, split mean | 0.631456 | **0.631494** |
| Rhaister-O full, line-macro | 0.635663 | 0.631478 |
| Rhaister-O drug residual | 0.537951 | **0.542715** |
| drug mean, full | 0.446 | 0.438 |
| nearest DMSO line 1-NN / 3-NN, drug residual | 0.167 / 0.198 | 0.183 / 0.161 |

`results/tahoe_new_splits/`. New split membership (seed 0) is printed in every log and recorded in each
`summary.json`. NCI-H596 `CVCL_1571` falls in new split 9, as expected from the pre-registration.

### 3.2 New EmeraldBay splits are of the same difficulty (jobs 286461, 286462)

26 never-held-out EmeraldBay lines (two of them outside Tahoe's 50), `results/emerald_bay/new_splits/`.

| | original 22 held-out lines | 26 never-held-out lines |
|---|---|---|
| Rhaister-O full, line-macro | 0.635705 | **0.634700** |
| Rhaister-O drug residual | 0.497239 | **0.502831** |
| Rhaister-O PC1 removed | 0.500709 | 0.507205 |
| drug mean, full | 0.492676 | 0.477377 |

Both fresh-holdout baselines land within 0.005 of their originals, so any change in the arms' gains on these
splits is about the arms, not about the splits being easier or harder.

### 3.3 Confirmation results (jobs 286440 Tahoe, 286463 EmeraldBay)

Hyperparameters fixed a priori, no inner selection, 200 permutations (p floor 0.005).

| arm | dataset | original splits | **fresh splits** | bootstrap low | per-split gains | gate |
|---|---|---|---|---|---|---|
| kernel | Tahoe | +0.0134 | **+0.0077** | -0.0009 | +0.0064 +0.0124 -0.0018 +0.0109 +0.0107 | fail |
| kernel | EmeraldBay | +0.0059 | **+0.0060** | +0.0029 | all five +0.005 to +0.007 | fail (size) |
| reliability | Tahoe | +0.0175 | **+0.0150** | +0.0126 | +0.0200 +0.0141 +0.0166 +0.0175 +0.0068 | **PASS** |
| reliability | EmeraldBay | +0.0022 | **+0.0011** | -0.0002 | | fail |

**The kernel arm does not confirm on fresh Tahoe lines.** Its gain nearly halves, 4 of 5 splits are positive and
the bootstrap interval now crosses zero. Its EmeraldBay value, by contrast, is reproduced to three decimals
(+0.0059 → +0.0060, all five splits positive, interval above zero). Read together, the honest estimate of the
locality effect is **about +0.006 to +0.008 and stable**, and the +0.0134 on the original Tahoe splits was
inflated by hyperparameter selection and by which 25 lines the benchmark happens to hold out. It is real but
below the +0.01 threshold on every fresh evaluation.

**The reliability arm confirms on Tahoe and stays absent on EmeraldBay**, and the fresh splits explain why.
NCI-H596 `CVCL_1571` is a *query* line in new split 9 and therefore not in the reference set there:

| reliability gain | splits 5-8 (NCI-H596 is a reference line) | split 9 (NCI-H596 is a query) |
|---|---|---|
| mean per-line gain | **+0.0170** | **+0.0068** |

That is the mechanism seen directly: two and a half times the gain when the mislabelled line is present to be
down-weighted. EmeraldBay has no comparable line -- its NCI-H596 matches DepMap at rank 1 -- and there the arm
does nothing. **Reliability weighting is a data-quality correction whose size is set by how much bad data the
reference panel contains, not a modelling advance that transfers.**

### 3.4 One Tahoe cell line is essentially unpredictable

In new split 9, NCI-H596 `CVCL_1571` is a held-out query for the first time in this project:

| | Rhaister-O drug-residual Pearson |
|---|---|
| **NCI-H596 `CVCL_1571`** | **0.0279** |
| next worst two lines | 0.4220, 0.4338 |
| the other 24 fresh lines, mean | 0.5642 |
| all 25 fresh lines, mean | 0.5427 |

The line whose baseline does not match DepMap (§2.1) is also the one line the model cannot predict at all -- an
order of magnitude below every other line in the cohort. Excluding it changes the arms' gains by less than
0.0005, so no conclusion above depends on it; it is reported because it is the strongest single piece of
evidence that this Tahoe line is corrupted rather than merely difficult.

### 3.5 Exploratory: the two mechanisms combine (job 286426, original Tahoe splits)

| arm | gain | bootstrap low | per-split | selected |
|---|---|---|---|---|
| kernel x reliability | **+0.0316** | +0.0266 | +0.0226 +0.0335 +0.0343 +0.0376 +0.0302 | h = 0.5; alpha = 20 ×3, 50 ×2 |

Almost exactly the sum of the two arms measured separately on these splits (+0.0134 + +0.0175 = +0.0309), which
is consistent with them acting through different mechanisms. **This number is not a confirmation:** it is on the
original splits with a selected alpha, so it inherits both sources of inflation.

### 3.6 Combined arm, confirmed on fresh Tahoe lines (job 286686)

Fixed h = 0.5 and alpha = 20, no selection, 200 permutations, 25 never-held-out lines.

| | gain | bootstrap low | per-split gains | perm p | gate |
|---|---|---|---|---|---|
| kernel x reliability | **+0.0267** | **+0.0223** | +0.0305 +0.0246 +0.0267 +0.0336 +0.0181 | 0.005 | **PASS** |

This is the only arm in the project that passes every predeclared condition under a fully pre-specified protocol
on cell lines never scored before. It exceeds the additive expectation from its two components measured on the
same fresh splits (+0.0077 + +0.0150 = +0.0227), so the two effects are at least additive and possibly mildly
synergistic.

The caveat from §3.3 carries over and must be stated with it: **most of the combined gain is the reliability
component, which is a data-quality correction specific to Tahoe's reference panel.** On EmeraldBay, where the
reliability component is worth +0.0011, the combined arm cannot be expected to reach +0.01 -- and job 286789
confirms it does not: **+0.0080 [+0.0045], all five splits positive, p = 0.005, gate fails on size.** The
observed value tracks the additive expectation from its components (+0.0060 + +0.0011 = +0.0071) closely.
(An earlier attempt, 286687, died on a launcher bug: sbatch `--export` is itself comma-separated and truncated
`kernel_reliability=0.5,20` at the comma; the runner now also accepts `0.5:20`.)

### 3.7 Everything on fresh, never-scored cell lines, in one table

All hyperparameters fixed a priori, no selection, 200 permutations. This is the only table in the project that
is free of both selection inflation and reuse of the benchmark's own holdout lines, and it is the one to quote.

| arm | Tahoe (25 fresh lines) | EmeraldBay (26 fresh lines) | passes both? |
|---|---|---|---|
| Rhaister-O baseline | 0.5427 | 0.5028 | -- |
| kernel (local slopes) | +0.0077 [-0.0009, +0.0151] | +0.0060 [+0.0029, +0.0090] | no (size) |
| reliability (quality weights) | +0.0150 [+0.0126, +0.0174] | +0.0011 [-0.0002, +0.0024] | no (absent on EB) |
| **kernel x reliability** | **+0.0267 [+0.0223, +0.0311]** | +0.0080 [+0.0045, +0.0115] | no (EB below +0.01) |
| removing one mislabelled line (NCI-H596) | +0.0051 on the original splits; the line itself scores **0.028** as a target | no equivalent line | -- |

**No arm clears the pre-registered two-dataset bar.** The largest confirmed effect, +0.0267 on Tahoe, is
mostly the reliability component, which does not transfer because EmeraldBay's reference panel does not contain
a comparably corrupted line. The part that transfers -- locality -- is worth about +0.007 on both datasets.

## 4. Engineered features (pre-registered F1/F2/F3, 2026-09-30 to 10-01)

Scope revision authorised by the user on 2026-09-30: CNV, mutation and CRISPR-dependency matrices may be used,
restricted to defining cell-line similarity and low-capacity interaction terms -- no new per-gene regressors,
no MLP. All three experiments are scored on the 25 never-held-out Tahoe lines from §3.

### 4.1 F1 -- does any feature space define a better cell-line similarity than the DMSO baseline?

The kernel arm's distance function is swapped for a correlation distance in an external feature space. Nothing
else changes and no parameters are added; `h = 0.5` fixed, 100 permutations. Rhaister-O 0.542715.
`results/feature_kernels/`, matrices from `scripts/build_cell_feature_spaces.py`.

| similarity from | lines | features | gain | bootstrap low | per-split | perm p |
|---|---|---|---|---|---|---|
| **Tahoe DMSO baseline** (control) | 50 | 2,000 | **+0.0077** | −0.0009 | 4/5 positive | 0.005 |
| DepMap expression | 47 | 18,943 | **+0.0069** | **+0.0034** | 5/5 positive | 0.010 |
| CRISPR dependency | 38 | 17,205 | +0.0054 | +0.0020 | 5/5 positive | 0.010 |
| damaging mutations | 49 | 1,269 | −0.0010 | −0.0027 | 1/5 positive | 0.604 |
| CNV | 49 | 38,429 | −0.0039 | −0.0090 | 1/5 positive | 0.168 |
| hotspot mutations | 49 | 17 | **−0.0340** | −0.0511 | 0/5 positive | 0.832 |
| **tissue lineage** | 49 | 13 | **−0.0344** | −0.0559 | 0/5 positive | 0.079 |

**No feature space beats the plain DMSO baseline**, and the ordering is the result. The three expression-like
spaces (Tahoe DMSO, DepMap expression, CRISPR dependency) all land at +0.005 to +0.008. The genotype spaces
(CNV, damaging mutations) are flat. The two coarse categorical spaces are **an order of magnitude worse than
the best arm is good**: weighting reference lines toward the same tissue of origin costs −0.0344, negative in
all five splits.

That last row is worth stating on its own: **drug-response transfer does not follow tissue lineage.** The
intuition that a lung line should be predicted from other lung lines is not merely unhelpful here, it is
actively harmful — four and a half times the magnitude of the best similarity gain in the other direction.
Similarity that matters for this task is transcriptional state, not genotype and not tissue of origin.

Dimensionality alone does not explain it: CNV has 38,429 features and is negative while the 2,000-gene DMSO
panel is the best. Modality does.

One nuance worth keeping: DepMap expression has a *smaller* point estimate than the DMSO baseline but a
**positive bootstrap lower bound and five of five positive splits**, where the DMSO baseline has one negative
split and an interval crossing zero. As a similarity definition it is more consistent, not better.

### 4.2 F2 -- cell x drug interaction features add nothing to the primary metric, and the reason matters

Scalar features per (cell, treatment) from each drug's annotated targets (264 of 379 drugs; 278 of 280 targets
present in DepMap expression), centred per treatment on the reference lines so the drug main effect -- already
carried by `mu(t,g)` -- is removed. Two low-capacity forms: a single global scalar `theta`, or one coefficient
per gene shared across drugs. `results/_smoke_interaction/`.

**Every arm's inner selection returned the anchor** (`theta = 0`; `lambda` at the top of its grid), and every
outer gain is of order 1e-5. But the features are not degenerate:

| feature | coverage of (line, treatment) cells | correlation with response magnitude |
|---|---|---|
| target expression | 0.66 | **+0.075** |
| target dependency | 0.51 | +0.013 |
| target pathway (Reactome, drug→cell axis) | 0.21 | **+0.211** |

**The mismatch is between the feature and the metric, not a defect in the feature.** The benchmark's primary
score is a per-(line, treatment) Pearson across 2,000 genes, which is invariant to scaling that condition's
response. A feature that says "this drug hits this line harder than average" changes magnitude, and the metric
cannot see magnitude by construction. This is worth stating in the paper on its own: the zero-shot benchmark
measures which direction a response points, not how strong it is -- and strength is what the Emerald Bay
sensitivity task, and therapeutic practice, actually care about.

### 4.3 F3 -- scoring the quantity the features actually carry

Target: `log ||delta(c,t) - drugmean(t)||`, centred per treatment on reference lines. Ridge per treatment,
lambda by leave-one-reference-line-out, scored as the per-line Pearson between predicted and true magnitude
across that line's 1,132 treatments. `results/magnitude_prediction_new_splits/`.

| model | line-macro Pearson | 95% line bootstrap |
|---|---|---|
| Rhaister-O's own predicted magnitude | **+0.2030** | [+0.1229, +0.2847] |
| DMSO baseline PCs (cell side) | **+0.1792** | [+0.1163, +0.2412] |
| **drug-target expression** (drug x cell) | **+0.0578** | **[+0.0313, +0.0875]** |
| target pathway | +0.0237 | [−0.0003, +0.0476] |
| target dependency | +0.0029 | [−0.0117, +0.0180] |
| all three interaction features | +0.0583 | [+0.0313, +0.0875] |
| permuted targets (control) | +0.0023 | [−0.0169, +0.0215] |

Two findings:

1. **Response magnitude is predictable for an unseen line** -- Rhaister-O reaches +0.20 and even the untreated
   baseline PCs reach +0.18 on a quantity the primary benchmark never scores.
2. **Drug-target expression is the first drug-side feature in this project to beat its own negative control:**
   +0.0555 over permuted targets, bootstrap low **+0.0269**. The signal is genuinely drug-specific.

It still fails the pre-registered gate, which required beating the cell-side baseline: **−0.1214 against DMSO
PCs**. Target dependency and target pathway add nothing on top of target expression (+0.0583 for all three
versus +0.0578 for expression alone). So the interaction is real, drug-specific, and roughly a third the size
of what the cell's own untreated profile already tells you.
