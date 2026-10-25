# Therapeutic — Weekly Report, week ending Friday 2026-09-25

Reporting interval: [2026-09-18T18:00:00+04:00, 2026-09-25T18:00:00+04:00) — Asia/Dubai.

Measured results only, with provenance; include positive and negative findings.

## 1. Data-shape and cohort audit; plate correspondence of the DMSO baseline

No Slurm job. All numbers read directly from the frozen local files on the login
node with the `therapeutic-prep` environment; nothing was recomputed or refitted.
Motivation: the cohort counts 50/49/47/45/43/42/25/23 appear across reports
without a single place resolving them, and the meaning of a per-plate DMSO
centroid had never been quantified.

### Cohort counts, resolved

| Count | Meaning |
|---|---|
| 50 | Tahoe lines with expression observations |
| 49 | of those, with a DepMap `ModelID`; `CVCL_C466` (hTERT-HPNE) has none |
| 47 | of those, present in DepMap 24Q4 expression |
| 45 | reference lines per outer split |
| 43 / 42 | reference lines **with DepMap expression**: 43 in splits 5-6, 42 in splits 7-9 |
| 25 | distinct held-out lines; the five holdout sets are disjoint |
| 23 | of those, with DepMap expression; missing `CVCL_1098` (C3A), `CVCL_0218` (COLO 205) |

The 43/42 figures independently reproduce the `n_available_components` values
42, 42, 41, 41, 41 already recorded in the capacity-sweep summary (= n_ref - 1).

### Plate structure

All 14 plates carry all 50 cell lines; 65,218 plate rows total, matching the
compact cache manifest. 1,137 distinct treatments over 1,308 plate-treatment
slots: 981 on one plate, 152 on two, 3 on three, 1 on all fourteen.

- The all-plate treatment is `[('Adagrasib', 0.05, 'uM')]`, a bridge anchor.
- **Plates 6 and 14 share all 94 treatments** (full replicate pair).
- **Plate 13 is a bridge plate**: 11 shared treatments each with plates 1, 7, 8
  and 9, 5 with plate 5, and 1-2 each with 2, 3 and 10.
- Every other plate pair shares only the Adagrasib anchor.

### Correspondence of per-plate DMSO centroids

`cell_line_centroids_dmso_hvg.parquet` is (700, 2002) and fully populated at
50 lines x 14 plates. Pearson on the 2,000-gene panel after `log1p`:

| Comparison | n | Median | Tail |
|---|---|---|---|
| Same line, different plates | 4,550 | **0.9793** | 5th pct 0.9189; min 0.4714 |
| Different lines, same plate | 2,660 | **0.3621** | 95th pct 0.6274; max 0.8434 |

Sum-of-squares split of the 700-row table: **cell line 94.34%, plate 1.26%,
residual (interaction + noise) 4.40%**.

Per-plate centroids of one line are therefore safely correspondable, with line
identity dominating batch roughly 75 to 1. The 0.4714 minimum shows a real tail,
so a per-line check is still warranted before relying on a single plate.

Negative/unexpected finding: plate 6 versus plate 14 same-line DMSO correlation
(median 0.9822, min 0.6553) is **not higher** than plate 6 versus plate 1
(median 0.9924, min 0.6707). The replicate design repeats *treatments*; it does
not make baselines more alike. Do not cite 6/14 as a baseline-reproducibility
reference.

### PCA spectra, bearing on the untested unwhitened ridge

Reference-only standardized PCA, split 5:

| Input | n_ref | Available components | PC1 var | PC1/PC10 variance ratio | Top-10 cumulative |
|---|---|---|---|---|---|
| DepMap expression (18,931 variable genes) | 43 | 42 | 10.4% | **3.5** | 49.8% |
| Tahoe DMSO HVG (2,000 genes) | 45 | 44 | 9.9% | **3.6** | ~52% |

Both spectra are flat. Whitening rescales components that differ in variance by
only ~3.5x across the used range, so the pending unwhitened raw-feature ridge is
a mild reparameterization of the prior rather than a dramatically different
model. This bounds expectations for that arm; it does not replace running it.

Provenance: `raw/tahoe-de-rhaister/{cell_eval,zeroshot,definition}`,
`processed/rhaister_o_compact/`, `derived/depmap_24q4/`,
`results/{rhaister_o_compact,depmap_expression_ridge_capacity_sweep,dmso_lowrank_bilinear,cell_feature_ablation,shared_control_diagnostic}/`.
Durable facts transferred to `KNOWLEDGE.md` sections "Canonical data shapes and
cohort counts", "The five outer splits", and "Plate structure and the per-plate
DMSO centroid".

## 2. Plate-count sweep: the DMSO feature is not noise-limited

Job **270015**, runner `scripts/run_plate_count_sweep.py`, submitted with
`slurm/run_plate_count_sweep.sbatch`. Completed. This answers the question the
shared-control diagnostic (job 258115) is blind to by design: that experiment
matched both arms at seven averaged plates so noise magnitude would cancel, so it
bounded the *shared* channel at 1.0% and said nothing about the absolute cost of
measurement noise in the feature.

Design: the DMSO feature is built from m randomly chosen plates, m in
{1, 2, 3, 4, 7, 10, 14}, five independent draws per m (m=14 has one). Everything
else fixed: frozen upstream Cell-Eval targets with no recomputation, the full
condition set, the pinned diagonal path at lambda 1.0 and blend 0.95, five outer
splits, all 25 held-out lines. Covered conditions per split 3,655--3,890 of
3,673--3,900, over 731--778 common treatments.

Drug-mean residual, line-macro Pearson, bootstrap over 25 lines (20,000 draws):

| m | Pearson | 95% CI | across-draw SD | % of m=14 |
|---|---|---|---|---|
| 1 | 0.526758 | [0.471994, 0.568609] | 0.004323 | **97.44%** |
| 2 | 0.532816 | [0.479752, 0.573674] | 0.003065 | 98.56% |
| 3 | 0.538116 | [0.484986, 0.578988] | 0.001614 | **99.54%** |
| 4 | 0.538919 | [0.485545, 0.579647] | 0.002108 | 99.69% |
| 7 | 0.540313 | [0.487048, 0.581123] | 0.000919 | 99.95% |
| 10 | 0.540910 | [0.488392, 0.581266] | 0.000695 | 100.06% |
| 14 | 0.540591 | [0.488013, 0.581013] | 0 | 100.00% |

Full delta and PC1-removed levels give the same shape: m=1 retains 98.68% and
97.29% of their m=14 values respectively, and both are within 0.5% of saturation
by m=3.

**The curve is flat by three plates.** Going from a single plate to all fourteen
buys 2.6% of the drug-residual score. The across-draw SD falls from 0.0043 at m=1
to 0.0007 at m=10, the expected 1/sqrt(m) behaviour of averaging, which confirms
the sweep is actually varying feature noise rather than something else.

Interpretation. Rhaister-O's drug-residual performance is **not limited by
measurement noise in the DMSO baseline**. Even one plate's centroid carries 97.4%
of the signal. Combined with job 258115's 1.0% bound on the shared-control
channel, the DMSO feature is now characterised from both sides: its noise neither
inflates the score materially nor attenuates it materially.

Comparability: the m=14 point is 0.540591 against the headline 0.537951 because
every point here uses a plain mean over an explicit plate subset, whereas the
pinned Rhaister-O feature averages plate-matched centroids per observed
condition. The two are close, which is itself a useful check, but only the shape
of this curve and within-analysis differences are the result.

## 3. Tahoe DMSO baseline versus DepMap baseline expression

No Slurm job; login-node analysis over 47 lines. Motivation: the DepMap-only
deployment branch needs to know whether DepMap baseline expression can stand in
for the Tahoe DMSO profile, and `KNOWLEDGE.md` forbids feeding DepMap TPM
directly into Rhaister-O without establishing a bridge.

Cohort: 47 shared lines; 1,987 of the 2,000 panel genes matched by parsed symbol.
Scales differ as expected -- Tahoe line feature mean 0.064 (range 0--2.657,
log1p of a pseudobulk centroid), DepMap mean 1.403 (range 0--14.671, log2(TPM+1)).
Three deliberately different estimands:

| Estimand | What it asks | Pearson | Mismatched-label control |
|---|---|---|---|
| **A** per line, across 1,987 genes | do the platforms agree which genes are high in this line? | median **0.618** (Spearman 0.763) | median **0.297** |
| **B** per gene, across 47 lines, gene-centred | do they agree which lines have more of this gene? | median **0.746**, mean 0.682 | median **-0.037**, 95th pct 0.258 |
| **C** identity discrimination, gene-centred and normalised | is a line's own DepMap profile its nearest DepMap profile? | **46/47 top-1 = 97.9%** | chance 2.1% |

Estimand A is the misleading one and should not be quoted: pairing beats
mismatched labels by only +0.32, because most of that correlation is the
universal "some genes are expressed, some are not" pattern rather than line
identity. Estimand B is the one that matters, because Rhaister-O's `gamma * x`
term uses exactly the across-line variation of `x`. There, 84.3% of genes exceed
0.5 and 33.4% exceed 0.8, with only 3.1% negative, against a null centred at
zero. Estimand C: diagonal similarity median 0.589 versus off-diagonal -0.027.

**Negative/flagged finding.** One line fails identity discrimination badly:
`CVCL_1571` (**NCI-H596**) ranks **42nd of 47** -- its own DepMap profile is
almost the *least* similar DepMap profile to its Tahoe baseline. Every other line
is rank 1. This is the signature of a line-identity problem (swap, mislabel,
contamination, or a divergent subclone between the two sources) rather than
ordinary noise. NCI-H596 is not among the 25 held-out lines, so no frozen result
is affected, but it enters reference fitting for every split and should be
checked before any Tahoe-to-DepMap bridge is built.

Bearing on the bridge question: across-line agreement is genuinely high, so a
reference-only DepMap-to-pseudo-DMSO bridge is not obviously hopeless on
correlation grounds. That does **not** license feeding DepMap TPM into
Rhaister-O directly -- the scale, transform and platform differ, and the ~0.75
median per-gene agreement is an upper bound on what a bridge could recover, not
an achieved transfer. It remains an unexecuted experiment.

Script: `scratchpad/tahoe_depmap_baseline.py` (Spearman implemented in numpy;
the `therapeutic-prep` environment has no scipy).

## 4. Bilinear coefficient tensors materialized; the programs look like noise

Job **270018** (270014 was the same job, failed on a temporary-filename bug and
was resubmitted), runner `scripts/extract_bilinear_coefficients.py`. No refitting
and no new selection: the closed form `B_hat = (D'D + lambda P)^-1 D' Y` is
evaluated at the (K, lambda) already selected in the frozen
`dmso_lowrank_bilinear/split_<S>/ridge_selection.csv`. All ten tensors written,
shape `(11, n_treatments, 2000)` with K=10 confirmed in every split and arm
(n_treatments 731--778).

Two internal validations passed:

- The **direct** arm's intercept is **exactly zero** (`|B0| = 0.000000` in all five
  splits), as it must be: its target is already the drug-mean residual, so the
  reference-line mean of the target is zero by construction.
- The **correction** arm's intercept is non-zero (`|B0|` mean 0.000178--0.000188),
  which is the mean leave-one-reference-line-out Rhaister-O error, i.e. a
  systematic-bias map. Split 5's most-biased genes, averaged over treatments, are
  all negative: MMP1 -0.001874, TP63 -0.001729, NRG3 -0.001438, MEST -0.001413,
  CDH6 -0.001360, MUC6 -0.001320, IGF2 -0.001289, DHRS9 -0.001209. Rhaister-O
  systematically over-predicts these.

**The ten cross-gene programs corroborate the permutation result.** Split 5,
correction arm:

| program | mean abs | Frobenius | share of energy in its top 100 of 2,000 genes |
|---|---|---|---|
| B1 | 0.000727 | 1.658 | 34.5% |
| B2 | 0.000630 | 1.636 | 49.9% |
| B3 | 0.000736 | 1.738 | 44.7% |
| B5 | 0.000645 | 1.486 | 39.9% |
| B7 | 0.000601 | 1.404 | 39.2% |
| B10 | 0.000573 | 1.371 | 41.7% |

If these programs were real biology one would expect a **decaying** energy
spectrum (B1 far above B10) and **concentration** on a coherent gene set.
Neither holds: Frobenius norms span only 1.371--1.738 across all ten, with no
ordering, and every program spreads 50--65% of its energy outside its top 5% of
genes. That is the signature of fitting noise, and it agrees with the
permutation null already reported (p = 0.9901, null mean 0.5334 above the
observed 0.5235). Two independent lines of evidence now say the same thing.

The stored archives also carry `loadings`, `kept_gene_mask`, `feature_mean`,
`feature_scale` and `score_scale`, so each program can be read against the
baseline direction that drives it if anyone wants to pursue it.

## 5. STATE: a released strict-zero-shot split-5 checkpoint exists, and the published Pearson-delta values disagree by 0.22

Read-only investigation, delegated to a subagent; no training, no Slurm job, no
large download. Findings below are the ones I verified myself against local files
or by recomputation; items I could not verify are marked.

**Verified locally.** `external/Rhaister/scripts/fetch_state.py` states in its own
docstring that *"The external evaluator ran the same splits and metrics as we
do, under the model name 'STATE'"*, and pulls `state/*` summary metrics from W&B
project `vevotx/perturbation-eval`, runs `eval_state-st+hvg[-pds]_tahoe5` through
`tahoe9`. So the Rhaister paper's STATE bars are Tahoe-bio's own ST+HVG runs on
**our exact five splits** under **Rhaister's** metric code, not numbers copied
from the State paper. The `-pds` suffix marks a rerun that additionally logs
`state/discrimination_mean`; split 5 falls back to the non-pds run.

**Verified locally.** The released `arcinstitute/ST-HVG-Tahoe` zero-shot
checkpoint's `var_dims.pkl` has `input_dim = hvg_dim = output_dim = 2000`,
`gene_dim = 62710`, `batch_dim = 14` (the plates), `pert_dim = 1138`. All
**1,137** of our treatment labels appear in its one-hot vocabulary byte for byte;
the single extra entry is `[('DMSO_TF', 0.0, 'uM')]`. Its split TOML holds out
C32, HOP62, HepG2/C3A, Hs 766T and PANC-1 with an **empty `[fewshot]`** block --
i.e. genuinely strict, five whole lines, no 30% perturbation leak -- which is
**exactly our `5_holdout`**.

**Verified by recomputation.** Arc's shipped per-line evaluation CSVs
(`zs_<line>_agg.csv`, the `mean` row over 1,136--1,137 perturbations each):

| line | pearson_delta | pr_auc | overlap_at_N | de_spearman_lfc_sig | de_spearman_sig | discrimination_l1 |
|---|---|---|---|---|---|---|
| C32 | 0.6437 | 0.4852 | 0.3390 | 0.3691 | 0.7115 | 0.8815 |
| HOP62 | 0.5675 | 0.4671 | 0.2669 | 0.4482 | 0.7365 | 0.9368 |
| HepG2-C3A | 0.4842 | 0.3599 | 0.2529 | 0.3396 | 0.6477 | 0.8388 |
| Hs 766T | 0.4552 | 0.3735 | 0.1695 | 0.5206 | 0.7155 | 0.8990 |
| PANC-1 | 0.5325 | 0.3942 | 0.1805 | 0.5187 | 0.6071 | 0.9413 |
| **mean** | **0.5366** | **0.4160** | **0.2418** | **0.4393** | **0.6836** | **0.8995** |

**The problem.** The Rhaister paper's Fig. 4 (p. 8) plots STATE's zero-shot
Pearson-delta at **0.32**. Arc's own released evaluation of the same model family
on the same five lines gives **0.5366**. Tahoe-x1 Fig. 6B independently reports
ST+HVG at **0.53** on `tahoe_5_holdout` (not verified by me; from the subagent's
reading of `papers/tahoe-x1_v1.pdf`). Three published values, one model, one
holdout, spread **0.22** -- larger than Rhaister-O's entire margin over the drug
mean at the residual level.

The pattern across metrics points at the evaluation pipeline rather than the
model: metrics least dependent on control and DE construction nearly agree
(discrimination 0.86 in Fig. 4 versus 0.8995 released; LFC Spearman 0.35 versus
0.4393), while delta- and significance-based metrics diverge most (Pearson 0.32
versus 0.5366; PR-AUC 0.25 versus 0.4160; DE overlap 0.09 versus 0.2418).

`fetch_state.py` supplies a concrete candidate explanation that I could **not**
resolve locally: its `METRIC_KEYS` list contains **both**
`state/pearson_delta_mean` and `pdex_static/pearson_delta_mean`, two different
delta definitions. Which one reached Fig. 4 is unresolved. `state_results.json`,
which the script writes, is **not** cached in `external/Rhaister`, so resolving
this needs W&B access to run `eval_state-st+hvg_tahoe5`.

**Until that is resolved, do not quote 0.32 as STATE's zero-shot Pearson-delta,
and do not quote 0.5366 as a head-to-head against our 0.6314 or 0.5380 either.**
The delta definition, the control construction (plate-matched versus
plate-agnostic, which Tahoe-x1 Fig. 6B plots side by side), and the DE pipeline
all differ or are unconfirmed.

**Correction to an earlier statement in this session.** I previously said
`report/09_18_report.md` contained STATE numbers read off the Rhaister paper's
bar charts. It does not -- grep finds no occurrence of "state" in that file at
all. The STATE values I had read off Fig. 1 and Fig. 4 went into a slide deck
built in the same session, not into any project report. No project record needed
correcting; my description of one did.

## 6. Reference-line count sweep: the task is data-limited, not method-limited

Job **272001**, runner `scripts/run_reference_count_sweep.py`. The 45 reference
lines are subsampled to m of them, the pinned diagonal path is refit from those
m only, and the five held-out lines of that split are scored against the frozen
targets. Five independent draws per m; the treatment set is fixed once from all
45 lines so the prediction target does not move with m.

Drug-mean residual, line-macro Pearson over all 25 held-out lines:

| m | Pearson | 95% CI | across-draw SD | % of m=45 | marginal | per added line |
|---|---|---|---|---|---|---|
| 5 | 0.157076 | [0.128325, 0.186657] | 0.043230 | 29.1% | -- | -- |
| 10 | 0.275054 | [0.238172, 0.309814] | 0.016241 | 50.9% | +0.117978 | +0.023596 |
| 15 | 0.363027 | [0.321870, 0.398331] | 0.015694 | 67.2% | +0.087972 | +0.017594 |
| 20 | 0.411351 | [0.368824, 0.444863] | 0.030978 | 76.1% | +0.048324 | +0.009665 |
| 25 | 0.462853 | [0.415715, 0.499727] | 0.006461 | 85.7% | +0.051503 | +0.010301 |
| 30 | 0.487762 | [0.438367, 0.526602] | 0.012766 | 90.3% | +0.024909 | +0.004982 |
| 35 | 0.505906 | [0.455917, 0.544623] | 0.004128 | 93.6% | +0.018143 | +0.003629 |
| 40 | 0.524332 | [0.472258, 0.564689] | 0.003277 | 97.1% | +0.018427 | +0.003685 |
| **45** | **0.540245** | [0.487442, 0.580759] | 0 | 100% | **+0.015913** | **+0.003183** |

**The curve has not plateaued.** The last five reference lines still buy
**+0.0159**, which is **1.6 times the +0.01 gate** that every modelling
experiment in this project has failed. At the top of the range each additional
cell line is worth about +0.0032 drug-residual Pearson.

This independently confirms the Rhaister paper's claim that zero-shot metrics
"do not appear to have plateaued" in the number of reference contexts, on our
pipeline and at the drug-residual level the paper does not report. It also
contrasts sharply with the paper's *few-shot* finding that gains go flat past
roughly 20 contexts: at m=20 the zero-shot task is still only at 76% of its
45-line value.

**This reframes every negative result in the project.** DepMap expression
(0.2087, gate false), cell-identity representations (0.029--0.093), cross-gene
low-rank conditioning (permutation p = 0.9901), and coefficient sharing
(section 8) all failed to clear +0.01. Five more cell lines would clear it. The
honest conclusion is not that these methods are bad ideas but that **45
reference contexts is not enough data for them to be identifiable**, and the
binding constraint is the screen rather than the estimator.

### How much headroom is actually left on this axis — a necessary caveat

"Has not plateaued" is not the same as "has a lot left", and the two must not be
conflated. Fitting three functional forms to the nine measured points:

| form | fitted asymptote | RMSE |
|---|---|---|
| `y = A - B/m` | 0.5502 | 0.03409 |
| **`y = A(1 - exp(-m/tau))`, tau = 15.0** | **0.5647** | **0.00364** |
| `y = A*m^b`, b = 0.548 (no asymptote) | -- | 0.03055 |

The saturating exponential fits an order of magnitude better than either
alternative. It puts the ceiling at **0.5647** against the measured 0.5402, so
the **total remaining headroom from adding reference cell lines is on the order
of +0.025**, not unbounded.

Extrapolation beyond the observed range is unreliable and should not be quoted:
asked how many lines reach a drug-residual of 0.55, the three forms answer
10,058, 55 and 39 respectively. Only the in-range statements are defensible:

- at m = 45 the curve is still rising at about **+0.0032 per added line**;
- the last five lines are worth **+0.0159**, 1.6x the +0.01 gate;
- the best-fitting saturating model leaves about **+0.025** on this axis in
  total.

That last number matters, because it is the same order of magnitude as the
+0.0199 that removing one low-quality cell line produces (section 7). **The
context-count axis and the data-quality axis are comparable in size, and both
are an order of magnitude larger than any method effect this project has
measured.** That, rather than "go collect more cell lines", is the defensible
claim.

### Quality beats quantity: curating reference lines is worth about twice adding them

Job **272021**, same runner with `--quality-modes best_quality worst_quality`,
written to a separate output root so the frozen 272001 result is untouched.
Instead of drawing m reference lines at random, take the m with the **highest**
or **lowest** own-plate reproducibility (the QC measure defined in section 7)
and refit. Consistency check: the `random` arm reproduces 272001 **bit for bit**
(maximum absolute difference 0.00e+00), so the refactor perturbed nothing.

Drug-residual line-macro Pearson:

| reference lines | best QC | random | worst QC | best &minus; random |
|---|---|---|---|---|
| 5 | 0.2033 | 0.1571 | **&minus;0.0414** | +0.0462 |
| 10 | 0.3222 | 0.2751 | 0.1292 | +0.0472 |
| 15 | 0.4154 | 0.3630 | 0.2655 | +0.0524 |
| 20 | 0.4842 | 0.4114 | 0.3559 | **+0.0728** |
| 25 | 0.5116 | 0.4629 | 0.4258 | +0.0488 |
| 30 | 0.5268 | 0.4878 | 0.4761 | +0.0391 |
| 35 | 0.5425 | 0.5059 | 0.5030 | +0.0366 |
| 40 | **0.5544** | 0.5243 | 0.5240 | +0.0301 |
| 45 (all) | -- | 0.5402 | -- | -- |

Three results, in order of how much they matter:

1. **The best 40 reference lines score 0.5544, higher than all 45 at 0.5402.**
   Dropping the five worst-quality reference lines is worth **+0.0142**. Beyond a
   point, more reference data is actively harmful when it is bad data.
2. **Curating beats adding, by about two to one.** Going from 40 random lines to
   45 buys +0.0159; going from 40 random lines to the best 40 buys **+0.0301**.
   For a group designing the next screen, that is the actionable number.
3. **The five worst lines are worse than useless**: fitted on them alone the
   model scores **&minus;0.0414**, below the zero that predicting the drug mean
   would give.

The QC differences driving this are small in absolute terms -- mean own-plate
correlation 0.9857 for the best five against 0.9646 for a random five and 0.8606
for the worst five -- so a modest difference in baseline reproducibility
produces a large difference in transfer. Every best-versus-random gap in the
table is between **3 and 7 times the +0.01 gate** that no model in this project
cleared.

This is the project's first clearly positive, actionable finding, and it is a
screen-design result rather than a modelling one: **for strict zero-shot
transfer, curate reference contexts on baseline reproducibility before adding
more of them.**

**It is not circular**, because the QC measure never sees a treated response.
It is computed entirely from how well a line's own untreated baseline reproduces
across plates, so it is available before any modelling.

**But one confound is real and unresolved, and must travel with this result.**
In split 5, a reference line's QC correlates **+0.533** with its mean distance
to that split's five held-out lines: **higher-quality reference lines sit
systematically further from the held-out lines.** For a linear extrapolator a
reference set that spans a wider range is generally an advantage, so part of the
best-QC gain may be coverage rather than quality. The direction is not obviously
self-serving -- if proximity to the held-out lines were what helped, the
high-QC set being further away would work against it -- but the two cannot be
separated from this experiment alone.

### Robustness: the effect is exclusion, not selection

Job **272053** ran two randomised variants to weaken the selection.
`random_top30` draws m at random from the 30 best lines instead of taking one
deterministic top-m; `drop_bottom5` draws m at random from the 40 best, i.e. it
only **avoids the worst five** rather than selecting for the best. Gain over
unrestricted random draws at the drug-residual level:

| m | worst QC | `drop_bottom5` | `random_top30` | best QC |
|---|---|---|---|---|
| 5 | -0.1970 | +0.0009 | +0.0431 | +0.0476 |
| 10 | -0.1580 | **+0.0534** | +0.0432 | +0.0350 |
| 15 | -0.1034 | **+0.0534** | +0.0543 | +0.0465 |
| 20 | -0.0708 | +0.0464 | +0.0462 | +0.0574 |
| 25 | -0.0250 | +0.0573 | +0.0504 | +0.0609 |
| 30 | -0.0096 | +0.0386 | +0.0411 | +0.0411 |
| 35 | -0.0100 | +0.0284 | (degenerate) | +0.0295 |
| 40 | -0.0060 | **+0.0245** | (degenerate) | +0.0245 |

**`drop_bottom5` captures essentially the whole effect.** From m=10 upward it
matches `best_quality` to within a few thousandths, and at m=40 the two are
identical at +0.0245. Merely excluding the five least reproducible reference
lines and then drawing at random recovers what deliberately picking the most
reproducible ones gives.

That narrows the claim to the defensible and more actionable one:

> **Exclude irreproducible reference contexts.** Selecting *for* high
> reproducibility beyond that adds nothing measurable.

It also largely dissolves the distance confound, because `drop_bottom5` draws
randomly from 40 lines rather than taking a systematically distant top-m set,
and still shows the gain.

Two caveats on that table. The `random_top30` cells at m=35 and m=40 are
**degenerate** -- the 30-line pool cannot supply 35 or 40 lines, so the runner
returns the whole pool -- and must not be read as results. And the `random`
column in this run is **not** bit-identical to job 272001 (0.1556 against 0.1571
at m=5) because the randomised quality modes consume the same generator; only
job 272021's random arm reproduced 272001 exactly.

## 7. Per-condition difficulty decomposition, and a data-quality finding that changes how the benchmark should be read

Job **271995**, runner `scripts/run_difficulty_decomposition.py`. Descriptive
only: it re-reads the 19,268 frozen per-condition rows and the frozen prediction
matrices. Nothing is refit, no model is selected, so there is no gate.

### Where the variance lives

Additive main-effect decomposition of per-condition Pearson (n = 19,185
diagonal-covered conditions; the design is unbalanced so the decomposition is
sequential and both orders are reported):

| statistic | line | drug | residual |
|---|---|---|---|
| `rhaister_o__pearson`, line removed first | **0.540** | 0.098 | 0.362 |
| `rhaister_o__pearson`, drug removed first | 0.512 | 0.110 | 0.378 |
| gain over drug mean, line first | 0.316 | 0.136 | 0.548 |
| gain over drug mean, drug first | 0.314 | 0.136 | 0.550 |

Per-condition difficulty is **dominated by which cell line it is** (51--54%),
not by which drug (10--11%). For the *gain* over the drug mean the line share
drops to ~32% and more than half is interaction.

### Dose

| dose (uM) | Rhaister-O | drug mean | gain | n |
|---|---|---|---|---|
| 0.05 | 0.605692 | 0.427082 | 0.178610 | 6,520 |
| **0.50** | **0.659252** | 0.450737 | **0.208515** | 6,470 |
| 5.00 | 0.633166 | 0.454548 | 0.178619 | 6,195 |

The middle dose is both the most predictable and the one where Rhaister-O gains
most, which is the expected shape: the low dose has little signal and the high
dose is dominated by generic toxicity.

### Negative finding on the per-drug side

Across 379 drugs, the across-line standard deviation of a drug's score
correlates **-0.232** with Rhaister-O's gain over the drug mean. **The more
context-dependent a drug's response is, the less Rhaister-O helps on it.** The
five lowest-gain drugs are dominated by glucocorticoids -- Clobetasol
propionate (gain 0.0920, across-line SD 0.2239), Betamethasone dipropionate
(0.1051 / 0.1793), Dexamethasone (0.1114 / 0.1931) -- exactly the class whose
response depends on receptor expression that varies strongly between lines.
This is the opposite of the desired behaviour and should be stated plainly:
the method helps most where the response is already conserved.

### The data-quality finding

The per-line covariate correlations looked striking at first -- residual
magnitude at Pearson -0.745, response magnitude -0.654, distance to the
reference centroid +0.450 -- but their Spearman counterparts are small
(-0.249, +0.088, +0.118), which means they are driven by one or two extreme
lines rather than by a monotone trend. Removing a single line collapses them:

| covariate vs Rhaister-O per-line Pearson | all 25 lines | without CVCL_1531 |
|---|---|---|
| distance to reference centroid | +0.4502 | **+0.0989** |
| residual magnitude | -0.7450 | **-0.2008** |
| response magnitude | -0.6538 | **+0.1346** |

That line is **CVCL_1531 = NCI-H2122**, the worst held-out line in the whole
benchmark at 0.1772 drug-residual Pearson against a next-worst of 0.4662, and
already on record in `reports/ORDERED_BASELINE_EXPERIMENTS.md` as the
PC1-removed minimum at 0.0648.

Measuring the internal reproducibility of each line's own Tahoe DMSO data --
the mean Pearson among that line's 14 plate centroids, and the fraction of zero
entries in them -- shows a **cliff**, not a continuum:

| rank | line | RRID | mean plate correlation | zero fraction |
|---|---|---|---|---|
| 1 | **NCI-H596** | CVCL_1571 | **0.6537** | **0.6478** |
| 2 | **NCI-H2122** | CVCL_1531 | **0.7238** | **0.5330** |
| 3 | SW 1271 | CVCL_1716 | 0.9488 | 0.3981 |
| 4 | H4 | CVCL_1239 | 0.9620 | 0.1795 |
| -- | cohort median | -- | **0.9755** | **0.1420** |

Two lines out of 50 are catastrophically irreproducible against themselves, then
the distribution jumps to 0.949. Both also have three to five times the cohort
median fraction of zero entries, which is what a very low effective cell count
looks like.

Against the 25 held-out lines, this quantity predicts difficulty robustly, and
unlike the earlier covariates it survives as a rank correlation:

| predictor | vs per-line Pearson (Pearson) | (Spearman) |
|---|---|---|
| mean plate correlation | +0.7667 | +0.2969 |
| **zero fraction** | **-0.8851** | **-0.7785** |

A Spearman of -0.779 over 25 lines is not an outlier artifact. **How sparse a
line's own DMSO centroid is predicts how hard that line is to predict.**

Consequences, in order of how much they matter:

1. **The two worst held-out lines are the two worst-QC lines.** NCI-H2122 is
   held out in split 9 and is that split's minimum at 0.0605; SW 1271 is held
   out in split 6 and is the third-worst held-out line at 0.4774. Splits 9 and 6
   are the two lowest-scoring splits (0.4759 and 0.5095 drug-residual), and
   across the five splits the worst held-out line's plate correlation tracks the
   split mean at Pearson +0.747 (n = 5, indicative only).
2. **Excluding low-QC lines moves the headline more than any method we tested.**
   Recomputed from the frozen per-line results, drug-residual line-macro Pearson
   for Rhaister-O is 0.537951 over 25 lines, **0.557845 (+0.0199)** dropping
   NCI-H2122 alone, and **0.566157 (+0.0282)** dropping NCI-H2122 and SW 1271.
   For scale, the DepMap correction experiment measured an effect of +0.000357:
   **one low-quality cell line moves this benchmark by roughly fifty times the
   effect that experiment was designed to detect.**
3. **NCI-H596, the single worst line, is never held out** -- it is a reference
   line in all five splits, so it degrades the fit everywhere while never being
   scored.

These are not reasons to drop the lines. Dropping them would be a
post-hoc cohort change to a frozen benchmark, and the comparison to published
Rhaister-O numbers would break. They are reasons to **report the sensitivity
alongside the headline**, and a strong argument that per-line heterogeneity in
this benchmark is substantially a measurement-depth artifact rather than a
biological difficulty signal.

Corroboration from a separate investigation: a delegated read-only audit of
NCI-H596 found the same internal incoherence independently (mean within-line
plate correlation 0.654 against a cohort median of 0.976, rank 1 of 50; zero
fraction 0.648 against a median of 0.142), found no reciprocal Tahoe/DepMap
label swap, and found that every one of H596's 14 plate centroids has
NCI-H2122 as its nearest other Tahoe line. The decisive check it identifies is
genotype or STR re-identification of the raw Tahoe DMSO cells for both lines,
plate by plate, which this project cannot do from the compact derivative.

## 8. Coefficient sharing: a real effect, an order of magnitude too small

Job **272027** (job 272004 was the same experiment with five implementation
defects and is archived as `_superseded_coefficient_sharing_v1`; see
`report/09_25_record.md`). `PLAN.md` Phase 4 Track A: shrink Rhaister-O's
per-(treatment, gene) diagonal slopes toward the pooled per-gene slope. Four
families -- a global blend, SVD truncation of the slope matrix, pooling across
the doses of a drug, and an empirical-Bayes per-gene weight -- selected by
leave-one-reference-line-out **drug-residual Pearson** rather than LOO RMSE.

| model | drug-residual line-macro Pearson |
|---|---|
| drug-dose mean | 0.000000 |
| fully pooled slopes (`shared_gamma`) | 0.264837 |
| Rhaister-O, fully independent slopes | 0.537951 |
| **coefficient sharing** | **0.539824** |

Gain **+0.001873**, bootstrap lower bound **+0.000986**, permutation
p **0.0196**, all five splits positive (0.0001, 0.0035, 0.0031, 0.0025, 0.0002),
anchor error 1.11e-16. `global_blend = 0.9` selected in every split, i.e. almost
no shrinkage.

**Four of the six gate conditions pass** -- positive bootstrap lower bound,
above the permutation null, 5 of 5 splits positive, RMSE within tolerance. It
fails on effect size alone: +0.0019 against a +0.01 gate. The honest statement
is not "no signal" but **"a real, statistically supported benefit about ten
times too small to matter"**.

That makes this the project's *best* method result, and it is the quantitative
floor the rest of the argument rests on: **our best modelling gain is an order
of magnitude below the data levers** in sections 6 and 7 (+0.016 to +0.030).

Methodological note, replicating the lesson from job 249519 on a second dataset:
the inner LOO Pearson and LOO RMSE criteria **disagreed in all five splits** --
Pearson chose `global_blend=0.9` every time, RMSE chose `svd_rank=200` every
time. Reference-line LOO RMSE is not a good proxy for held-out drug-residual
Pearson.

## 9. Chemistry-informed shrinkage: null, slightly negative, and it exposed a gate defect

Job **274091**, runner `scripts/run_chemistry_shrinkage.py`. Shrink Rhaister-O's
per-(treatment, gene) slopes toward a target predicted from drug chemistry
instead of the pooled per-gene mean. The diagonal structure is untouched; only
the shrinkage target changes, so a failure is attributable to chemistry rather
than to breaking the diagonal (which job 249519 already showed fails).

**Harness validated bit-exactly.** The `uniform_shrink` arm is job 272027's
model re-expressed in this harness and reproduces its aggregate to
**0.00e+00**: +0.001872912 both. Any other arm's number is therefore
trustworthy.

| arm | drug-residual | gain vs Rhaister-O | bootstrap low | per-split gains |
|---|---|---|---|---|
| Rhaister-O | 0.537951 | -- | -- | -- |
| **uniform_shrink** | 0.539824 | **+0.001873** | **+0.000986** | all five positive |
| moa_only | 0.537957 | +0.000006 | -0.000017 | **0, 0, 0, 0, 2.9e-05** |
| morgan_only | 0.537721 | **-0.000231** | -0.000528 | 3 of 5 negative |
| chem_full | 0.537713 | **-0.000238** | -0.000539 | 3 of 5 negative |

- **`moa_only` contributes exactly nothing in four of five splits** -- the inner
  selection chose w = 1.0, no shrinkage at all.
- **`chem_full` is `morgan_only`**: -0.000238 against -0.000231, with identical
  hyperparameters in four of five splits. Adding MOA to Morgan changes nothing.
- **Chemistry-informed shrinkage is worse than uniform shrinkage and worse than
  no shrinkage.**

### Why: the chemistry target collapses toward zero, not toward the pooled slope

Every chemistry arm selected **rank 1** in all five splits, the bottom of a 1-20
grid, and the arms willing to shrink chose **lambda = 1000**, the top of the
grid. The prediction recorded here beforehand was that the target had collapsed
*onto the pooled slope*, i.e. that chemistry was rediscovering uniform shrinkage
through a degenerate path. **That prediction is wrong.** Job 274109 measured it
on split 5 at the selected configurations:

| arm | selected | \|target\| | \|pooled\| | \|gamma\| | \|target-pooled\|/\|pooled\| | \|target\|/\|gamma\| |
|---|---|---|---|---|---|---|
| chem_full | lam 1000, rank 1 | 19.55 | 79.02 | 59.72 | **1.030** | **0.327** |
| morgan_only | lam 1000, rank 1 | 19.35 | 79.02 | 59.72 | 1.030 | 0.324 |
| moa_only | lam 10, rank 1 | 12.26 | 79.02 | 59.72 | 1.012 | 0.205 |

The target is **as far from the pooled slope as the pooled slope's own
magnitude** (ratio ~1.03, i.e. essentially unrelated to it) and its magnitude is
only **20-33% of gamma's**. So the reduced-rank ridge fit of the slopes on drug
features is shrunk almost to zero. Blending gamma toward a near-zero target
attenuates the slope, which is why it hurts.

**The mechanism is therefore informative and specific: Morgan fingerprints and
MOA class carry almost no information about a drug's per-gene slope on baseline
expression.** That is a cleaner negative than "the model did not help".

### A defect in the predeclared gate, exposed by this run

All four arms record permutation **p = 0.0196**, the floor with 50 draws, so all
four pass `above_permutation_null` -- including the two whose gain is
**negative**. The null means explain it:

| arm | observed gain | permutation null mean |
|---|---|---|
| chem_full | **-0.000238** | **-0.003249** |
| morgan_only | -0.000231 | -0.003262 |
| uniform_shrink | +0.001873 | -0.000356 |

A randomly permuted chemistry correction is *much* worse (-0.0032) than the
learned one (-0.0002), so the learned mapping is genuinely non-random -- and
still harmful, because both are worse than applying no correction at all.

**The permutation null tests "is the mapping non-random", not "is the mapping
helpful". The meaningful reference is zero, not the permutation mean.** Earlier
experiments did not expose this because none had a clearly negative gain.

Recommended gate revision, to be applied to future arms and stated in the
methods: add an explicit `gain_point_estimate_positive` condition rather than
relying on the bootstrap lower bound to catch it indirectly. Do not retro-apply
it to close gates -- all four arms here fail on effect size anyway.

### Where this leaves the estimator

| job | direction | result |
|---|---|---|
| 249519 | **add** degrees of freedom (cross-gene) | permutation p 0.9901; a random correction beat the learned one |
| 272027 | remove them **uniformly** | +0.00187, optimal blend 0.9 |
| **274091** | remove them **with structure** (chemistry) | **-0.00024**, target collapses toward zero |

Three directions, three parameterisations, one place. **Rhaister-O's diagonal
slopes can be neither augmented nor meaningfully compressed.** Set against the
data levers of +0.016 to +0.030 measured in sections 6 and 7, the method axis on
this benchmark is closed at this scale.

## 10. The estimator transplanted to L1000: Rhaister-O loses to the drug mean on all eight lines

Two jobs were run on 2026-09-23 and had not been written up before
2026-09-24. Both ask the same question from different angles: **is the
Rhaister-O result about the estimator, or about Tahoe?**

### 10.1 Job 274337 — Rhaister-O's estimator, run natively on L1000

Leave-one-cell-line-out over the eight LINCS L1000 lines with complete 24 h
coverage in GSE70138, 1,618 conditions present in every line, 949 of the 978
landmarks retained after the DepMap join. Baseline `x` is DepMap 24Q4
log2(TPM+1); the response is the Level-5 signature. Ridge lambda 1.0, blend
0.95, seed 0, 20,000 bootstrap resamples over lines.

| level | model | line-macro Pearson | 95% line bootstrap |
|---|---|---:|---|
| full | zero delta | 0.000000 | — |
| full | global mean | 0.166995 | [0.159998, 0.174486] |
| full | **drug mean** | **0.299222** | **[0.285917, 0.312716]** |
| full | **Rhaister-O** | **0.273483** | **[0.261773, 0.285291]** |
| drug-residual | drug mean | 0.000000 | — |
| drug-residual | **Rhaister-O** | **0.045328** | **[0.040273, 0.050406]** |

**Rhaister-O is below the drug mean on the primary metric, and the intervals do
not overlap.** Per held-out line the loss is unanimous:

| line | drug mean | Rhaister-O | difference | drug-mean RMSE | Rhaister-O RMSE |
|---|---:|---:|---:|---:|---:|
| A375 | 0.3281 | 0.2948 | -0.0333 | 1.0270 | 1.0792 |
| A549 | 0.2833 | 0.2639 | -0.0194 | 0.9850 | 1.0233 |
| HA1E | 0.2779 | 0.2489 | -0.0289 | 1.0537 | 1.1187 |
| HCC515 | 0.2765 | 0.2580 | -0.0184 | 1.0591 | 1.0938 |
| HEPG2 | 0.2853 | 0.2606 | -0.0247 | 1.0501 | 1.1087 |
| HT29 | 0.3210 | 0.2975 | -0.0235 | 1.0089 | 1.0493 |
| MCF7 | 0.3056 | 0.2762 | -0.0294 | 0.9677 | 1.0279 |
| PC3 | 0.3160 | 0.2878 | -0.0282 | 1.0208 | 1.0658 |

**0 of 8 lines improve, and RMSE is worse on all 8 as well.** The
treatment-specific slope is not merely uninformative here; adding it is a net
cost against simply predicting the drug's average response.

Set against the Tahoe numbers from section 1 of
`reports/TAHOE_DEPMAP_L1000_BILINGUAL_SYNTHESIS.md`:

| | Tahoe (45 reference lines) | L1000 (7 reference lines) |
|---|---:|---:|
| drug(-dose) mean, full delta | 0.4434 | 0.2992 |
| Rhaister-O, full delta | **0.6315** | **0.2735** |
| gain from the treatment-specific slope | **+0.1881** | **-0.0257** |
| Rhaister-O, drug-residual | **0.5380** | **0.0453** |

The drug-residual number is the cleanest statement: the component that is
entirely attributable to the treatment-specific slope is worth **0.538 on Tahoe
and 0.045 on L1000**, a factor of about twelve.

**Not apples to apples, and the summary records why**: the L1000 response is a
Level-5 z-score rather than a plate-matched log expression delta; the panel is
the 978 L1000 landmarks, of which only 79 overlap the Tahoe 2K panel; the
baseline is DepMap bulk rather than a Tahoe DMSO centroid; and there are 8
contexts against Tahoe's 45. Section 6 of this report shows the task is
data-limited, so the seven-reference-line cohort is itself a plausible partial
cause. It does not rescue the result: at seven reference lines the estimator is
*worse than doing nothing*, which is a different failure from being merely
weak.

### 10.2 Job 275031 — the cell-representation ablation, re-run on L1000

The four cell representations of job 249475 re-run on the same eight lines and
1,618 conditions, to ask whether the Tahoe ordering is about the
representations or about Tahoe.

| arm | L1000 full | L1000 drug-residual | 95% bootstrap (drug-residual) | Tahoe drug-residual (job 249475) |
|---|---:|---:|---|---:|
| expression PCA | 0.3006 | **0.0210** | [0.0035, 0.0360] | **0.2087** |
| structured metadata | 0.3001 | 0.0115 | [-0.0073, 0.0273] | 0.0780 |
| template text | 0.3004 | 0.0085 | [-0.0190, 0.0342] | 0.0930 |
| Cellosaurus text | 0.2944 | -0.0141 | [-0.0318, 0.0055] | 0.0294 |
| drug mean | 0.3000 | 0.0000 | — | 0.0000 |

**Every arm collapses onto the drug mean.** Three of four drug-residual
intervals cross zero; the fourth, DepMap expression PCA, retains 0.021 against
0.209 on Tahoe. So the failure is not specific to Rhaister-O's parameterisation
— on L1000 at this cohort size, *no* cell representation carries recoverable
context signal beyond the drug's average response.

Recorded limitations from the run's own summary: with eight lines,
`structured_metadata` is close to a one-hot identity code, so a near-zero score
is mechanically expected; `cellosaurus_text` is truncated at 512 tokens for six
of eight lines, so a negative result means "no advantage under truncation"; and
seven reference lines admit at most six principal components, so every arm is
compressed to at most six dimensions.

### 10.3 What this settles

The earlier cross-platform comparison (`09_11_report.md` section 5) asked
whether an L1000 signature can stand in for a Tahoe response and found
concordance of 0.0605 / 0.0153 with L1000's own two releases agreeing at only
0.2025. Section 10 here asks the complementary question and gives the sharper
answer: **the Rhaister-O estimator, applied to L1000 on L1000's own terms, is
beaten by the drug mean on every line.** The published 0.63 is a joint property
of the estimator and of Tahoe's condition-level reproducibility, and the
reproducibility half of that is not portable.

This is a statement about L1000 at eight contexts with z-scored Level-5
responses. It is not a claim that the estimator fails on every non-Tahoe atlas,
and no such claim should be made without a second atlas at comparable context
count.
