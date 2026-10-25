# Rhaister Signal Pilot: One Split Before Expanding the Experiment

Last updated: 2026-09-07

## Decision

There is clear cell-context signal worth pursuing, but generic nearest-cell
transfer is not reliable enough. On Rhaister split 5, Rhaister-O retains
substantial correlation after both the reference drug-dose mean and each query
line's average residual-response template are removed. This holds for all five
query lines with tight drug-cluster bootstrap intervals.

The next experiment should remain small: compare DepMap-expression context on
three representative lines—C32, Hs 766T, and HOP-62—before building a full
five-split multimodal model. C3A should be reserved as a missing-modality case
because its DepMap 24Q4 expression and Chronos rows are absent.

## Pilot design

This is a diagnostic pilot, not a final benchmark estimate.

- One released outer split only: `tahoe/5_holdout`.
- Five completely held-out query lines; 3,673 drug-dose test conditions.
- Frozen Rhaister-O and drug-dose-mean predictions from job `214307`.
- One new control only: copy the response of the single reference line with
  maximum Pearson similarity in all-plate mean `log1p` DMSO HVG expression.
- Three evaluation levels: full expression delta; residual after subtracting
  the 45-reference-line drug-dose mean; and that residual after removing each
  query line's mean residual-response template.
- Two thousand bootstrap replicates per line, resampling 268 drugs and keeping
  all doses of a drug together.
- No hyperparameter search and no use of query-line treated outcomes before
  predictions were frozen.

The template-removed calculation is diagnostic. For truth and prediction
separately, it subtracts the line's average residual over the 734/735 query
treatments. It is never supplied to a model as an input.

## Nearest-DMSO neighbors

| Query line | Disease | Nearest DMSO reference | Reference disease | DMSO Pearson |
|---|---|---|---|---:|
| C32 (`CVCL_1097`) | Melanoma | SK-MEL-2 (`CVCL_0069`) | Melanoma | 0.667 |
| HOP-62 (`CVCL_1285`) | Lung adenocarcinoma | Hs 578T (`CVCL_0332`) | Breast carcinoma | 0.735 |
| C3A (`CVCL_1098`) | Hepatoblastoma | NCI-H596 (`CVCL_1571`) | Lung adenosquamous carcinoma | 0.501 |
| Hs 766T (`CVCL_0334`) | Pancreatic adenocarcinoma | NCI-H2122 (`CVCL_1531`) | Lung adenocarcinoma | 0.587 |
| PANC-1 (`CVCL_0480`) | Pancreatic adenocarcinoma | A549 (`CVCL_0023`) | Lung adenocarcinoma | 0.587 |

Only C32 selects a same-lineage neighbor. A high DMSO-expression correlation
does not guarantee response transfer: HOP-62 has the highest neighbor
similarity but copying Hs 578T is worse than the drug mean on full delta.

## Where the signal is

### Per-line result

| Query line | Drug mean, full | Nearest, full | Rhaister-O, full | Nearest, drug residual | Rhaister-O, drug residual | Rhaister-O, template removed |
|---|---:|---:|---:|---:|---:|---:|
| C32 | 0.420 | 0.437 | 0.592 | 0.304 | 0.458 | 0.463 |
| HOP-62 | 0.605 | 0.544 | 0.814 | 0.321 | 0.712 | 0.684 |
| C3A | 0.369 | 0.073 | 0.604 | -0.043 | 0.512 | 0.470 |
| Hs 766T | 0.497 | 0.097 | 0.713 | -0.061 | 0.599 | 0.557 |
| PANC-1 | 0.513 | 0.412 | 0.704 | 0.129 | 0.594 | 0.590 |

The central observation is that Rhaister-O's score does not disappear after
removing the transferable drug response. Its drug-residual Pearson ranges from
0.458 to 0.712. Removing the line-wide response template changes that range
only to 0.463--0.684. Therefore the model is recovering substantial
treatment-specific context structure, not only a conserved drug program or a
single generic property of each line.

Nearest-line transfer is heterogeneous. It finds positive residual signal for
C32, HOP-62, and PANC-1, but is negative for C3A and Hs 766T. On full delta it
beats the drug mean only for C32. A single globally similar baseline cell line
is not a dependable substitute for treatment-specific modulation.

### Bootstrap evidence

The following intervals equal-weight drugs; all doses remain in the same
bootstrap cluster.

| Query line | Rhaister-O full improvement over drug mean | 95% interval | Template-removed Pearson | 95% interval |
|---|---:|---:|---:|---:|
| C32 | +0.175 | [0.167, 0.183] | 0.465 | [0.452, 0.478] |
| HOP-62 | +0.211 | [0.206, 0.216] | 0.683 | [0.672, 0.694] |
| C3A | +0.237 | [0.228, 0.246] | 0.468 | [0.456, 0.481] |
| Hs 766T | +0.217 | [0.211, 0.223] | 0.557 | [0.544, 0.569] |
| PANC-1 | +0.193 | [0.187, 0.198] | 0.589 | [0.576, 0.600] |

Every interval is comfortably above zero. These intervals describe variation
over drugs within one fixed line; they do not replace uncertainty over future
cell lines.

### Dose and response magnitude

Rhaister-O drug-residual Pearson is 0.550 at 0.05 uM, 0.618 at 0.5 uM, and
0.553 at 5 uM. The template-removed values are 0.537, 0.572, and 0.547.
The middle dose appears strongest, but this is not yet a clean dose effect:
0.05 and 0.5 uM contain about 1,340 conditions each, whereas 5 uM contains
995, so their drug composition differs. A within-drug paired comparison is
required before making a biological claim.

After dividing response magnitude into quartiles within each line,
template-removed Rhaister-O Pearson is 0.487, 0.563, 0.592, and 0.568 from the
weakest to strongest quartile. Signal is weaker for the weakest responses but
is not confined to only the largest perturbations. The plateau from Q3 to Q4
also argues against treating raw response amplitude as the whole explanation.

## Small next experiment

Use three lines that expose different regimes:

1. **C32**: the lowest Rhaister-O result among the three, with a biologically
   plausible same-melanoma DMSO neighbor that gives modest positive transfer.
2. **Hs 766T**: an intermediate Rhaister-O result where nearest-DMSO transfer
   fails strongly; this is the best rescue test for a richer context feature.
3. **HOP-62**: a high-performing/near-ceiling control where DepMap should not
   be expected to add much.

All three have DepMap 24Q4 expression, CNV, mutation, and Chronos data. Start
with one no-training comparison: choose the nearest reference line in DepMap
expression and compare it with the frozen DMSO neighbor. Only if it improves
the residual metric should we fit a reference-only PCA plus low-capacity ridge
on the drug residual. This creates a clean stage gate before any modality or
hyperparameter expansion.

C3A lacks DepMap expression and Chronos but has CNV and mutation. Do not impute
it silently in the first expression pilot; preserve it for an explicit
missing-modality experiment. PANC-1 can remain a confirmatory fourth line if
one of the three primary cases looks promising.

## Artifacts and provenance

| Artifact | Location |
|---|---|
| Pilot runner | `scripts/evaluate_rhaister_signal.py` |
| Slurm wrapper | `slurm/evaluate_rhaister_signal.sbatch` |
| Unit tests | `tests/test_rhaister_signal.py` |
| Output root | `/nfs-stor/yuqi.lei/Therapeutic/results/rhaister_signal_pilot/split_5` |
| Per-condition metrics | `per_condition_metrics.csv` |
| Per-line summary | `per_line_summary.csv` |
| Drug-cluster bootstrap | `drug_cluster_bootstrap.csv` |
| Dose and magnitude summaries | `dose_summary.csv`, `magnitude_summary.csv` |
| Nearest-line predictions | `nearest_predictions.npz` |
| Job log | `/nfs-stor/yuqi.lei/Therapeutic/logs/rhaister-signal-214816.out` |

Slurm job `214816` completed on `cn-14` with exit `0:0` in eight seconds. The
output archives pass integrity loading with `allow_pickle=False`. The pilot
summary SHA-256 is
`ec0b9d88771e35b4d569c8ac00702b85e303d06546a262a5c43e7eb0c39822c7`.
