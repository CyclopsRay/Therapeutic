# Therapeutic — Weekly Report, week ending Friday 2026-11-06

Reporting interval: [2026-10-30T18:00:00+04:00, 2026-11-06T18:00:00+04:00) — Asia/Dubai.

Measured results only, with provenance; include positive and negative findings.

## 1. PCC_plate reference scale: shrunk oracle (job 356090; `scripts/oracle_reference.py`; no training)

Follow-up to `10_30_report.md` §6.2, which added the oracle residual at weight 1. Here the weight a is chosen per metric
on the neighbouring fold (= fold f's validation compounds) from {0.1, …, 1}. Oracle = drug-free MLP + a × (mean
backbone residual of the same test compound's other test samples, i.e. its measured responses in other lines);
"one line" = only its most control-similar other line. Uses test responses: a reference point, not a model. Rows:
shared-control test rows of compounds with ≥ 2 test samples (14,084 per fold). Source:
`/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/context_sanity/oracle_reference.csv`; log `logs/ctx-sanity-356090.out`.

| reference | PCC_plate | chosen a | PCC_DEG | chosen a |
|---|---|---|---|---|
| drug-free MLP | 0.0981 | | 0.6480 | |
| UnDetect closed form (pool_multi_norm) | 0.1532 | | 0.6567 | |
| UnDetect-NN (nnk_multi_norm) | 0.1536 | | 0.6578 | |
| one measured line, unshrunk (a = 1) | 0.1311 | | | |
| **one measured line, shrunk** | **0.1542** | 0.2 (all folds) | 0.6540 | 0.1 |
| mean of measured lines, unshrunk (a = 1) | 0.2002 | | | |
| **mean of measured lines, shrunk** | **0.2020** | 0.6–0.8 | **0.6635** | 0.3–0.4 |

- UnDetect-NN covers 53% of the PCC_plate distance from the drug-free MLP to the shrunk mean oracle (per fold 0.565 /
  0.484 / 0.525 / 0.547 / 0.553) and 63% on PCC_DEG ((0.6578 − 0.6480) / (0.6635 − 0.6480)).
- With optimal shrinkage, one measurement of the compound in its most similar other line equals UnDetect on PCC_plate
  (0.1542 vs 0.1536) and is below it on PCC_DEG (0.6540 vs 0.6578).

## 2. UnDetect extensions O2/O3 on the drug-blind benchmark (CPU run `v7_ext_cpu`, jobs 356254–356258)

`scripts/run_undetect_v2.py` (sha 02f7a986…), 5 drug-disjoint folds, backbone seed 1729. Every hyper-parameter is
chosen on the fold's validation compounds and the test set is scored once. All arms in a fold share one backbone. Run
on the CPU partition (`slurm/run_undetect_v2_cpu.sbatch`) because the GPU queue was ~1.5 days; the CPU backbone is
not bit-identical to v3. CPU vs GPU, folds 0–2 (the GPU numbers come from the crashed GPU jobs 356099–356101, which
finished these arms before the crash):

| | CPU | GPU |
|---|---|---|
| backbone valid | 0.6340 / 0.6370 / 0.6410 | 0.6342 / 0.6377 / 0.6409 |
| `pool_kta` test PCC_DEG | 0.6503 / 0.6469 / 0.6471 | 0.6505 / 0.6472 / 0.6467 |

PCC_plate is the plate-centred PCC on shared-control test rows (13,828–14,557 per fold), and PCC_DEG is the plain PCC
over all test samples. Contrasts are against `pool_kta` (= current UnDetect closed form, KTA view weights), with
2,000 compound-bootstrap replicates per fold. Sources:
`/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/v7_ext_cpu/{fold_*/summary.json, centred_vs_pool_kta.json,
undetect_v2_summary.csv}`; logs `logs/undetect-v2c-35625{4..8}.out`, `logs/v7-centred-356620.out`.

### 2.1 Five-fold means

| arm | what changes vs `pool_kta` | PCC_plate | Δ PCC_plate | folds Δ > 0 | folds CI > 0 | folds CI < 0 | PCC_DEG | Δ PCC_DEG | folds Δ > 0 |
|---|---|---|---|---|---|---|---|---|---|
| `mlp` | drug-free backbone | 0.1000 | −0.0550 | 0 | 0 | 5 | 0.6372 | −0.0113 | 0 |
| `krr_kta` | shared term only | 0.1418 | −0.0133 | 0 | 0 | 5 | 0.6459 | −0.0026 | 0 |
| **`pool_kta`** | reference | **0.1551** | | | | | **0.6485** | | |
| `nnk_kta` | NN readout | 0.1554 | +0.0003 | 2 | 0 | 1 | 0.6492 | +0.0007 | 4 |
| `pool_ml` | O3a: view weights, amplitude, noise by GP marginal likelihood | 0.1557 | **+0.0007** | **5** | **4** | 0 | 0.6486 | +0.0001 | 4 (fold 1: −0.000001) |
| `nnk_ml` | O3a + NN readout | 0.1552 | +0.0001 | 3 | 0 | 1 | 0.6491 | +0.0006 | 4 |
| `pool_val` | view weights from an 84-point simplex grid on valid | 0.1554 | +0.0004 | 4 | 3 | 1 | 0.6485 | 0.0000 | 4 |
| `pool3_resp` | O2: + neighbour-line term, response similarity | 0.1535 | −0.0016 | 1 | 0 | 3 | 0.6487 | +0.0002 | 5 |
| `pool3_ctl` | O2: + neighbour-line term, control similarity | 0.1534 | −0.0016 | 0 | 0 | 4 | 0.6489 | +0.0004 | 5 |
| `pool_sens` | O3b: shared term × per-line sensitivity | 0.1548 | −0.0002 | 2 | 0 | 0 | 0.6491 | +0.0006 | 5 |
| `pool_all` | ML kernel + response neighbour + sensitivity | 0.1559 | **+0.0008** | **5** | 1 | 0 | 0.6493 | **+0.0008** | **5** |

Plain PCC_DEG on the ~1,500 unshared-control test samples per fold: `pool_kta` 0.5703, `pool_ml` 0.5705, `pool_sens`
0.5708, `pool_all` 0.5711, `pool3_ctl` 0.5700, `pool3_resp` 0.5695.

### 2.2 Per-fold Δ PCC_plate vs `pool_kta` (folds 0–4)

| arm | Δ per fold | folds with CI entirely < 0 |
|---|---|---|
| `pool_ml` | +0.0012 / +0.0002 / +0.0004 / +0.0006 / +0.0010 | none |
| `pool_all` | +0.0021 / +0.0012 / +0.0002 / +0.0002 / +0.0004 | none |
| `pool_val` | +0.0003 / −0.0030 / +0.0018 / +0.0017 / +0.0011 | 1 |
| `pool_sens` | +0.0005 / +0.0003 / −0.0003 / −0.0007 / −0.0009 | none (fold 4: [−0.0018, +0.00005]) |
| `pool3_resp` | +0.0008 / −0.0026 / −0.0026 / −0.0001 / −0.0032 | 1, 2, 4 |
| `pool3_ctl` | −0.0016 / −0.0015 / −0.0017 / −0.0006 / −0.0028 | 0, 1, 2, 4 |

### 2.3 Settings chosen on validation

- `pool_kta` and `pool_ml`: λ = 3, g = 0.5, h = 0.5 in every fold.
- Learned view weights, chem / tgt / moa / morph: fold 0 .29/.15/.21/.36, fold 1 .32/.19/.20/.29, fold 2 .28/.27/.18/.27,
  fold 3 .20/.30/.22/.28, fold 4 .24/.23/.19/.34. KTA weights: chem .28–.31, tgt .24–.26, moa .21–.24, morph .22–.25.
  Cell Painting is up-weighted in 5/5 folds (largest weight in 3).
- `pool_val` simplex weights: Cell Painting ≥ 2/6 in all folds; the remaining weights differ between folds.
- O2 neighbour coefficient c (`n`) and own-line coefficient h. `pool3_resp`: c = 0.5 / 0.75 / 0.5 / 0.25 / 0.75, with
  h lowered from 0.5 to 0.25 in 3 folds; bandwidth 0.05–1.0. `pool3_ctl`: c = 0.75 / 0.75 / 0.5 / 1.0 / 1.0, with
  h = 0.25 in 4 folds. Validation never chose c = 0 (component active).
- O3b sensitivity: κ = 1 / 10 / 10 / 1 / 1, never κ = ∞ (t ≡ 1). Line multipliers t (10th–90th percentile over
  training lines): 0.60–0.72 to 1.77–1.95; range −0.16 to 2.21.
- `pool_all`: κ = 1 / 10 / 10 / 1 / 1. c = 0.25 in 4 folds and 0 in fold 2. h = 0.5 throughout (not lowered).
- No eigh fallback triggered in the CPU run.

### 2.4 GPU (v3-identical backbone) PCC_DEG before the crash, folds 0–3 (jobs 356153–356155, 356629)

Logs `logs/undetect-v2-{356153,356154,356155,356629}.out`. The jobs crashed in `pool_val`, so no predictions were saved
and there is no PCC_plate for these runs.

| arm | fold 0 | fold 1 | fold 2 | fold 3 |
|---|---|---|---|---|
| `mlp` | 0.6382 | 0.6352 | 0.6359 | 0.6416 |
| `pool_kta` | 0.6505 | 0.6472 | 0.6467 | 0.6507 |
| `pool_ml` | 0.6507 | 0.6472 | 0.6468 | 0.6508 |
| `nnk_kta` | 0.6518 | 0.6464 | 0.6476 | 0.6513 |
| `nnk_ml` | 0.6519 | 0.6465 | 0.6478 | 0.6515 |

`pool_ml` − `pool_kta` = +0.0002 / 0.0000 / +0.0001 / +0.0001, the same as the CPU run (§2.1).

### 2.5 GPU copy (v3-identical backbone; jobs 357045–357048 with `--skip-val`; analysis 357094)

`pool_kta` test PCC_DEG 0.6505 / 0.6472 / 0.6467 / 0.6507 / 0.6472, which equals v3 `pool_multi_norm`. Contrasts vs
`pool_kta` in the five-fold mean. Source: `results/sdst_hybrid/v7_ext/{centred_vs_pool_kta.json,
undetect_v2_summary.csv}`, log `logs/v7-centred-357094.out`.

| arm | Δ PCC_plate | folds > 0 | folds CI < 0 | Δ PCC_DEG | non-inferior (03:20 rule) |
|---|---|---|---|---|---|
| `pool_ml` | +0.0007 | 5 | 0 | +0.0001 | yes |
| `pool_all` | +0.0008 | 4 | 0 | +0.0008 | yes |
| `pool_sens` | −0.0002 | 2 | 0 | +0.0006 | no |
| `pool3_resp` | −0.0001 | 3 | 1 | +0.0002 | no |
| `pool3_ctl` | −0.0020 | 0 | 4 | +0.0004 | no |
| `nnk_kta` | −0.0004 | 4 | 1 | +0.0007 | (reference arm, not judged) |
| `nnk_ml` | +0.0006 | 4 | 1 | +0.0008 | no |

The verdicts are the same as on the CPU run (§2.1). `pool3_resp` is closer to 0 here (−0.0001 vs −0.0016) because
validation picked other settings.

## 3. View weights on Tahoe and the drug-specific (residual) metric on Tahoe and LINCS

### 3.1 Learned or transferred view weights for the consensus term (jobs 357069–357071; analysis job 357081)

`scripts/run_l1000_xline.py --source Tahoe --kernels T1 D Dml Dl1000 --tag _Dw`, ik14 split, seed 0, 50 lines, 350
compounds (272–294 training compounds per fold). `Dml` = consensus-term view weights by GP marginal likelihood per
fold; `Dl1000` = fixed weights learned on the L1000 drug-blind benchmark (chem .265 / tgt .228 / moa .201 / morph .306).
Per-line models are unchanged: `mkl_Dml` = `mkl_Dl1000` = `mkl_D` exactly, and `pool_D` / `mkl_D` reproduce the
earlier `_D` runs exactly (max |diff| 0) at T-0.5 and T-5. ACTIVE rows, line-macro. `resid` = Pearson of
(prediction − training line mean) vs (truth − training line mean). Δ vs `pool_D` with a 2,000-replicate bootstrap over
ik14 groups; lines with Δ > 0 out of 50. Source:
`/nfs-stor/yuqi.lei/Therapeutic/results/l1000_xline/{T-0.05,T-0.5,T-5}_ik14_seed0_Dw/`, `weights_vs_D_Dw.json`; logs
`logs/l1000-xline-{357069,357070,357071}.out`, `logs/xline-w-357081.out`.

| slice | `pool_D` full / resid | Δ full `pool_Dml` | Δ resid `pool_Dml` | Δ full `pool_Dl1000` | Δ resid `pool_Dl1000` |
|---|---|---|---|---|---|
| T-0.05 | 0.4982 / 0.1270 | +0.00036 [−0.00024, +0.00099] 26/50 | −0.00085 [−0.0046, +0.0029] 12/50 | −0.00004 [−0.00018, +0.00009] 10/50 | +0.00009 [−0.0007, +0.0011] 10/50 |
| T-0.5 | 0.6671 / 0.1966 | −0.00016 [−0.00056, +0.00021] 13/50 | −0.00243 [−0.0074, +0.0011] 2/50 | −0.00004 [−0.00013, +0.00005] 9/50 | +0.00005 [−0.0009, +0.0008] 22/50 |
| T-5 | 0.4400 / 0.1810 | −0.00019 [−0.00061, +0.00019] 13/50 | −0.00146 [−0.0034, +0.0002] 8/50 | −0.00002 [−0.00020, +0.00015] 13/50 | +0.00032 [−0.0003, +0.0009] 13/50 |

Marginal-likelihood weights per fold (chem / tgt / moa / morph):
- T-0.5: .27/.10/.23/.40, .48/.10/.31/.12, .10/.01/.03/.86, .14/.36/.23/.27, .42/.06/.26/.26.
- T-5: .19/.35/.04/.42, .22/.23/.04/.51, .04/.09/.01/.86, .24/.25/.05/.45, .24/.20/.06/.50.
- T-0.05: morph .83–.99 in every fold, with the constant noise ≈ 0.85 of the unit z-scored variance (no signal to fit).
- On L1000 SDST (§2.3) the same fit gave chem .20–.32, tgt .15–.30, moa .18–.22, morph .27–.36 across folds.

### 3.2 Per-line kernel vs cross-line model on the residual metric (`mkl_D` − `pool_D`)

Same Tahoe runs. LINCS from the existing set-D runs (`P2-24_ik14_seed0_D`, `P1-24_ik14_seed0_D`, 7 lines; job 357082,
`weights_vs_D_D.json`). Lines are counted where `mkl_D` > `pool_D`.

| slice | Δ full (`mkl_D` − `pool_D`) | Δ resid (`mkl_D` − `pool_D`) |
|---|---|---|
| Tahoe T-0.05 | −0.0015 [−0.0023, −0.0007] 21/50 | **+0.0031 [+0.0013, +0.0050] 35/50** |
| Tahoe T-0.5 | −0.0066 [−0.0090, −0.0044] 3/50 | **+0.0050 [+0.0024, +0.0076] 44/50** |
| Tahoe T-5 | −0.0060 [−0.0094, −0.0027] 1/50 | **+0.0022 [+0.0007, +0.0038] 36/50** |
| LINCS P2-24 | −0.0132 [−0.0147, −0.0118] 0/7 | −0.0068 [−0.0077, −0.0059] 0/7 |
| LINCS P1-24 | −0.0028 [−0.0035, −0.0022] 0/7 | −0.0054 [−0.0066, −0.0042] 0/7 |

Line-macro resid: Tahoe T-0.5 `mkl_D` 0.2016 vs `pool_D` 0.1966; T-5 0.1832 vs 0.1810. LINCS P2-24 0.3513 vs 0.3582;
P1-24 0.1936 vs 0.1990. The consensus-only arm (`gonly_D`) is below `pool_D` on resid by 0.079–0.089 on Tahoe and
0.046–0.060 on LINCS.

- On LINCS the cross-line term raises both full and residual Pearson (7/7 lines).
- On Tahoe it raises full Pearson but **lowers** the drug-specific residual Pearson at all three doses (intervals
  exclude 0). The (a_L, b_L) mixing weights are chosen by inner CV on full Pearson.

## 4. Published deep-learning models on the plate-centred metric (released predictions; job 357106; no training)

Source: Bai et al. 2026 companion archive v2.0.0 (Zenodo 22095049, MIT), `benchmark_predictions.zip` (sha256
a61c1696…) in `/nfs-stor/yuqi.lei/Therapeutic/raw/bai2026_archive/`. Primary-cohort Full runs (split seed 131419;
XPert training seed 1729) and the Basal-only MLP (`mlp_bai`). `scripts/prep_published_preds.py` →
`results/sdst_hybrid/published/fold_*/pred_test.npz`. `analyze_sdst_centred.py --tag published --ref {mlp,
nnk_multi_norm}` → `centred.json`, `centred_vs_nnk_multi_norm.json`. Log `logs/pub-plate-357106.out`.

Alignment check: the released test rows and genes are in our `sdst.npz` order, and the released ground truth equals our
deltas within 9.5e-7. Our recomputed per-fold PCC_DEG equals the released `pcc_saved` to five decimals for every
model and fold (e.g. XPert 0.63188 / 0.62736 / 0.63632 / 0.63416 / 0.62782). PRnet and PertDiT are released in their own
normalised log-expression space (`model_specific_normalized_log_expression_delta`), so no PCC_plate is computed for
them. Their released PCC_DEG: PRnet 0.277, PertDiT 0.370.

`mlp` and `pool_multi_norm` / `nnk_multi_norm` are our v3 arms (UnDetect closed form / NN).

| model | PCC_DEG (all test; released) | PCC_plate (shared-control rows) | plain PCC_DEG, unshared rows | Δ PCC_plate vs our MLP (folds > 0, CI > 0) |
|---|---|---|---|---|
| CIGER | 0.2456 | 0.0108 | 0.2597 | −0.0881 (0, 0) |
| MultiDCP | 0.2545 | 0.0244 | 0.2237 | −0.0745 (0, 0) |
| DeepCE | 0.3947 | 0.0553 | 0.3379 | −0.0436 (0, 0) |
| MLP, Bai et al. | 0.6372 | 0.1018 | 0.5350 | +0.0029 (3, 3) |
| our MLP (drug-free backbone) | 0.6373 | 0.0989 | 0.5360 | |
| TranSiGen | 0.6144 | 0.1440 | 0.5636 | +0.0451 (5, 5) |
| **XPert** | 0.6315 | **0.1569** | 0.5650 | **+0.0580 (5, 5)** |
| UnDetect closed form (`pool_multi_norm`) | 0.6485 | 0.1537 | 0.5700 | +0.0548 (5, 5) |
| UnDetect-NN (`nnk_multi_norm`) | 0.6493 | 0.1541 | 0.5683 | +0.0552 (5, 5) |
| plate-response baseline | 0.604 | −0.0001 | | |

Per-fold PCC_plate: XPert 0.1508 / 0.1563 / 0.1692 / 0.1570 / 0.1513; UnDetect-NN 0.1562 / 0.1469 / 0.1576 / 0.1587 /
0.1509; TranSiGen 0.1376 / 0.1414 / 0.1539 / 0.1418 / 0.1456.

Paired PCC_plate vs UnDetect-NN (2,000-replicate compound bootstrap per fold):

| arm | Δ per fold (folds 0–4) | mean | folds CI > 0 | folds CI < 0 |
|---|---|---|---|---|
| XPert | −0.0054 / +0.0094 / +0.0116 / −0.0017 / +0.0005 | **+0.0029** | 1 (fold 2) | 0 |
| TranSiGen | −0.0186 / −0.0056 / −0.0037 / −0.0170 / −0.0053 | −0.0100 | 0 | 2 (folds 0, 3) |
| UnDetect closed form | −0.0005 / +0.0059 / −0.0009 / −0.0038 / −0.0025 | −0.0004 | 1 | 0 |
| DeepCE / MultiDCP / CIGER | | −0.099 / −0.130 / −0.143 | 0 | 5 |

- XPert and TranSiGen carry compound-specific signal well above the drug-free MLP (+0.058 and +0.045 PCC_plate, 5/5
  folds). Yet their plain PCC_DEG is below the MLP's (0.6315 and 0.6144 vs 0.6372), and Bai et al.'s drug-zeroing
  ablation changes XPert's PCC_DEG by only +0.0002.
- On PCC_plate, XPert equals or slightly exceeds UnDetect (+0.0029 vs UnDetect-NN, +0.0032 vs the closed form; one fold
  with CI > 0, none with CI < 0). UnDetect is +0.018 above XPert on PCC_DEG.
- DeepCE, MultiDCP and CIGER are below the drug-free MLP on PCC_plate (0.011–0.055 vs 0.099).

### 4.1 Drug-input ablations, complementarity and where XPert differs (job 357120; `scripts/analyze_xpert_vs_undetect.py`)

Source: `results/sdst_hybrid/published/{centred.json, xpert_vs_undetect.json}`, log `logs/xpert-vs-357120.out`. The
released Zero (drug inputs zeroed) and Shuffle (drug inputs permuted across compounds) runs are aligned like §4.

**PCC_plate with the drug input removed** (5-fold means; PCC_DEG from Bai et al. §5.6 for comparison):

| model | Full | Zero | Shuffle | Full − Zero, PCC_plate | Full − Zero, PCC_DEG (released) |
|---|---|---|---|---|---|
| XPert | 0.1569 | 0.1064 | 0.1131 | **+0.0505** | +0.0002 |
| TranSiGen | 0.1440 | 0.1012 | 0.0980 | **+0.0428** | +0.0088 |
| DeepCE | 0.0553 | 0.0474 | 0.0478 | +0.0079 | +0.0113 |
| MultiDCP | 0.0244 | 0.0230 | 0.0228 | +0.0014 | +0.0052 |
| CIGER | 0.0108 | n/a (constant within plate) | −0.0197 | | −0.0028 |

Per-fold XPert Zero: 0.1017 / 0.1052 / 0.1048 / 0.1111 / 0.1090 (Full 0.1508–0.1692).

**Complementarity of XPert and UnDetect-NN** (shared rows):
- Correlation of centred predictions (p − m), per-sample PCC: XPert vs UnDetect 0.321; XPert vs MLP 0.269; UnDetect vs
  MLP 0.751.
- Blend (1 − w)·UnDetect-NN + w·XPert, with w chosen on the neighbouring fold (= fold f's validation compounds):

| | w chosen | blend | UnDetect-NN | XPert | 50/50 blend |
|---|---|---|---|---|---|
| PCC_plate | 0.4 in 5/5 folds | **0.1809** | 0.1541 | 0.1569 | 0.1791 |
| PCC_DEG (shared rows) | 0.3–0.4 | **0.6646** | 0.6578 | 0.6385 | 0.6633 |

Per-fold blend PCC_plate: 0.1774 / 0.1767 / 0.1884 / 0.1852 / 0.1768 (above both models in 5/5 folds). Reference:
shrunk mean oracle 0.2020 (§1, rows with ≥ 2 test samples).

**Compound-shared vs line-specific part.** Each model's centred prediction is replaced by its own mean over the
compound's shared test rows (compounds with ≥ 2 shared rows):

| | full | compound mean only | difference |
|---|---|---|---|
| UnDetect-NN | 0.1534 | **0.1158** | 0.0376 |
| UnDetect closed form | 0.1531 | 0.1048 | 0.0483 |
| XPert | 0.1564 | 0.0975 | **0.0589** |
| TranSiGen | 0.1433 | 0.0908 | 0.0525 |
| drug-free MLP | 0.0981 | 0.0541 | 0.0439 |

The MLP has no compound input, yet it shows a 0.044 "difference" and a 0.054 compound-mean score. This is the
plate-mean noise floor (§5.4 of `10_30_report.md`), so compare models relative to the MLP. On the compound-shared
part UnDetect-NN is ahead (0.116 vs 0.098). On the line-specific part XPert is ahead (+0.015 above the MLP's
difference, vs −0.006 for UnDetect-NN).

**Per-compound Δ PCC_plate, XPert − UnDetect-NN** (8,217 compound × fold units, pooled; compound bootstrap). Overall
+0.0065 [+0.0041, +0.0088]; XPert is better on 56% of compounds.

| stratum | level | n | Δ [95% CI] | XPert | UnDetect-NN |
|---|---|---|---|---|---|
| views | chemistry only | 4,693 | +0.0084 [+0.0055, +0.0114] | 0.144 | 0.135 |
| views | targets only | 1,591 | +0.0044 [−0.0010, +0.0098] | 0.157 | 0.153 |
| views | Cell Painting only | 578 | −0.0055 [−0.0163, +0.0045] | 0.187 | 0.193 |
| views | targets + Cell Painting | 1,355 | +0.0071 [+0.0016, +0.0126] | 0.198 | 0.190 |
| nearest training compound (Morgan Tanimoto) | Q1, median 0.32 | 2,092 | +0.0010 [−0.0035, +0.0056] | 0.152 | 0.151 |
| | Q2, median 0.50 | 2,021 | +0.0049 [+0.0005, +0.0094] | 0.148 | 0.143 |
| | Q3, median 0.70 | 2,050 | +0.0073 [+0.0025, +0.0120] | 0.156 | 0.149 |
| | Q4, median **1.00** | 2,054 | **+0.0127 [+0.0082, +0.0172]** | 0.177 | 0.164 |
| shared test rows per compound | 1 | 638 | −0.0029 [−0.0144, +0.0087] | 0.221 | 0.224 |
| | 2–3 | 1,444 | +0.0124 [+0.0060, +0.0190] | 0.181 | 0.169 |
| | 4–10 | 4,636 | +0.0063 [+0.0035, +0.0090] | 0.141 | 0.135 |
| | > 10 | 1,499 | +0.0052 [+0.0003, +0.0101] | 0.162 | 0.157 |

- At least a quarter of test compounds have a training compound with an identical Morgan fingerprint (Q4 median
  Tanimoto 1.00: stereoisomers or salts under another BRD ID). XPert's lead is largest there.

## 5. Context-gated compound term (C-lite) and exact-analog term (A) on the UnDetect cache (CPU backbone)

Cache `scripts/cache_undetect.py` tag `cache_v1` (jobs 357139–357143). The cached `pool_kta` equals the v7_ext_cpu
`pool_kta` exactly (0.65028 / 0.64693 / 0.64707 / 0.6508 / 0.6474). Stage 2 `scripts/stage2_context.py`: tag
`stage2_v1` (declared arms; jobs 357222, 357245) and `stage2_v2` (+ `ctx_pc_ext`, `ctx_line`, declared at record
20:45; job 357246). PCC_plate contrasts by `analyze_sdst_centred.py` (refs `pool_kta`, `ctx_scale`). Logs
`logs/stage2-357222.out`, `stage2-357245.out`, `stage2v2-357246.out`, `s2-centred-357237.out`.

| arm | what it adds to UnDetect (`pool_kta`) | PCC_plate | Δ vs `pool_kta` (folds > 0 / CI > 0) | Δ vs `ctx_scale` (folds > 0 / CI > 0) | PCC_DEG Δ vs `pool_kta` (per fold) | chosen |
|---|---|---|---|---|---|---|
| `pool_kta` | — | 0.1551 | | −0.0032 | | |
| `ctx_scale` | per-gene rescaling of the compound term | 0.1583 | +0.0032 (5/5) | | +0.0006 / 7 / 3 / 4 / 7 (×1e-4) | c = 1 (10 in fold 2) |
| `ctx_gene` | × (gene's own control expression) gate | 0.1587 | +0.0037 (5/5) | +0.0005 (5/5) | +0.0008 mean | c = 1 / 10 |
| **`ctx_pc`** | × (top-K control PCs) gate | **0.1600** | **+0.0050 (5/5)** | **+0.0018 (5/4)** | +0.0014 / 17 / 12 / 10 / 15 (×1e-4) | c = 1; K = 20 (10 in fold 2) |
| `ctx_both` | PCs + gene-own gate | 0.1600 | +0.0049 (5/5) | +0.0017 (5/4) | +0.0012 mean | c = 1 / 10, K = 20 |
| `ctx_pc_ext` | PCs, K up to 80 | 0.1596 | +0.0045 (5/5) | +0.0013 (4/1) | +0.0018 mean | K = 80 in 5/5 |
| `ctx_line` | × per (line, gene) scale (line identity) | **0.1605** | **+0.0055 (5/5)** | **+0.0023 (5/5)** | +0.0017 mean | c = 1 |
| `dup_line`, `dup_mean` | exact-analog residual (τ ≥ 0.8) | 0.1551 | 0 | | 0 | c = 0 in 5/5 |

Per-fold PCC_plate, `ctx_pc`: 0.1604 / 0.1640 / 0.1638 / 0.1543 / 0.1577 (`pool_kta` 0.1549 / 0.1581 / 0.1589 / 0.1504 /
0.1530). Test rows with an identical-fingerprint training analog (τ = 1.0): 12.8–16.2%.

**Against XPert on the same rows** (`scripts/blend_crossfit.py`, job 357347; tags `s2_ctx_pc_vs_xpert`,
`s2_ctx_line_vs_xpert`). Paired PCC_plate vs XPert, compound bootstrap:

| arm | Δ PCC_plate vs XPert, folds 0–4 | mean | folds > 0 / CI > 0 / CI < 0 | PCC_DEG (all test) |
|---|---|---|---|---|
| `pool_kta` (CPU backbone) | +0.0041 / +0.0018 / −0.0103 / −0.0066 / +0.0017 | −0.0019 | 3 / 0 / 1 | 0.6485 |
| `ctx_pc` | +0.0096 / +0.0077 / −0.0054 / −0.0028 / +0.0063 | +0.0031 | 3 / 0 / 0 | 0.6499 |
| `ctx_line` | +0.0100 / +0.0081 / −0.0047 / −0.0022 / +0.0068 | +0.0036 | 3 / 0 / 0 | 0.6502 |
| XPert | | (0.1569) | | 0.6315 |

Cross-fitted blend (1 − w)·`ctx_pc` + w·XPert (w chosen on the neighbouring fold: 0.4, 0.4, 0.4, 0.3, 0.3): PCC_plate
**0.1857** (+0.0288 vs XPert, 5/5 CI > 0), PCC_DEG 0.6605. With `ctx_line`: 0.1860 / 0.6607.

- The context gate lifts UnDetect from −0.002 to +0.003 PCC_plate relative to XPert (not significant either way)
  while keeping +0.018 PCC_DEG. The two models remain complementary.
- About 60% of the gain over `pool_kta` is per-gene rescaling without context (`ctx_scale`). Gating by line identity
  (`ctx_line`) matches or slightly exceeds gating by control PCs: on this benchmark, where every test line is a
  training line, the useful interaction is a line-specific per-gene scaling of the compound response.

## 6. Final drug-blind benchmark numbers on one backbone run (CPU run; srun job 357876; `scripts/finalize_sdst.py`)

User decision 2026-11-01 02:30: finalise the drug-blind numbers on the CPU backbone run (`cache_v1` + `stage2_v2`).
All arms are scored on the same test rows. Published models are the released predictions (§4). `_v3` = the earlier GPU
run, shown for reference; its backbone differs (record 2026-11-01 00:45). Blends: (1 − w)·A + w·XPert, w chosen on
the neighbouring fold's PCC_DEG (w = 0.4 in 5/5 folds for both blends).
Outputs: `results/sdst_hybrid/final_cpu/{scores.csv, contrasts.csv, final.json}`, `final_cpu/fold_*/pred_test.npz`;
log `/nfs-stor/yuqi.lei/Therapeutic/logs/finalize-sdst-11010232.out`.

### 6.1 Five-fold means

| model | PCC_DEG (all rows) | PCC_plate (rows with a training plate-mate) | PCC_DEG (rows without one) |
|---|---|---|---|
| backbone (drug-free MLP, ours, CPU run) | 0.6372 | 0.1000 | 0.5358 |
| drug-free MLP, published (`mlp_bai`) | 0.6372 | 0.1018 | 0.5350 |
| XPert (released) | 0.6315 | 0.1569 | 0.5650 |
| XPert, compound input zeroed | 0.6313 | 0.1064 | 0.5374 |
| XPert, compound input shuffled | 0.6180 | 0.1131 | 0.5252 |
| TranSiGen (released) | 0.6144 | 0.1440 | 0.5636 |
| DeepCE (released) | 0.3947 | 0.0553 | 0.3379 |
| MultiDCP (released) | 0.2545 | 0.0244 | 0.2237 |
| CIGER (released) | 0.2456 | 0.0108 | 0.2597 |
| UnDetect, closed form (`pool_kta`) | 0.6485 | 0.1551 | 0.5703 |
| + per-gene rescaling (`ctx_scale`, control) | 0.6490 | 0.1583 | 0.5716 |
| **+ context gate (`ctx_pc`)** | **0.6499** | **0.1600** | **0.5738** |
| + line-identity gate (`ctx_line`, control) | 0.6502 | 0.1605 | 0.5740 |
| UnDetect + XPert blend | 0.6593 | 0.1815 | 0.6110 |
| UnDetect + context gate + XPert blend | 0.6605 | 0.1860 | 0.6139 |
| *GPU v3 reference:* backbone / closed form / NN readout | 0.6373 / 0.6485 / 0.6493 | 0.0989 / 0.1537 / 0.1541 | 0.5360 / 0.5700 / 0.5683 |

### 6.2 Paired contrasts (5-fold mean Δ; folds > 0; folds with compound-bootstrap 95% CI > 0 / < 0; share of test compounds improved, min–max over folds)

| contrast | Δ PCC_DEG | folds > 0, CI > 0 / < 0 | Δ PCC_plate | folds > 0, CI > 0 / < 0 | compounds up (PCC_plate) |
|---|---|---|---|---|---|
| UnDetect closed form vs backbone | +0.0113 | 5, 5/0 | +0.0550 | 5, 5/0 | 59.5–62.5% |
| + context gate vs backbone | +0.0127 | 5, 5/0 | +0.0600 | 5, 5/0 | 60.4–64.8% |
| + context gate vs UnDetect closed form | +0.0014 | 5, 5/0 | +0.0050 | 5, 5/0 | 64.3–65.4% |
| + per-gene rescaling vs UnDetect closed form | +0.0006 | 5, 5/0 | +0.0032 | 5, 5/0 | 58.9–64.1% |
| + line-identity gate vs UnDetect closed form | +0.0017 | 5, 5/0 | +0.0055 | 5, 5/0 | 63.0–65.7% |
| XPert vs backbone | −0.0057 | 1, 0/4 | +0.0569 | 5, 5/0 | 64.8–69.4% |
| XPert zeroed vs backbone | −0.0059 | 0, 0/5 | +0.0064 | 4, 4/0 | 50.9–60.7% |
| UnDetect closed form vs XPert | +0.0170 | 5, 5/0 | −0.0019 | 3, 0/1 | 42.2–46.7% |
| + context gate vs XPert | +0.0184 | 5, 5/0 | +0.0031 | 3, 0/0 | 44.3–48.5% |
| + context gate + XPert blend vs XPert | +0.0290 | 5, 5/0 | +0.0291 | 5, 5/0 | 67.8–74.4% |
| + context gate + XPert blend vs UnDetect closed form | +0.0120 | 5, 5/0 | +0.0309 | 5, 5/0 | 73.9–75.5% |
| TranSiGen vs backbone | −0.0228 | 0, 0/5 | +0.0440 | 5, 5/0 | 58.8–64.3% |

Reading (negative and null results included):
- On the CPU run, UnDetect with the context gate is the best single model on all three columns. It is fully closed
  form after the backbone.
- Against XPert, PCC_DEG is clearly higher (+0.018, 5/5 CI > 0). PCC_plate is a tie (+0.003, no fold CI excludes 0).
  This tie is within the ±0.005 per-fold run-to-run noise of the backbone (record 2026-11-01 00:45).
- With its compound input zeroed, XPert keeps PCC_DEG (0.6313 vs 0.6315) but loses 0.050 PCC_plate. Its compound
  branch carries compound-specific signal that PCC_DEG does not show. Even so, XPert stays below the drug-free
  backbone on PCC_DEG in 4/5 folds (CI < 0).
- The blend with XPert gains a further +0.026 PCC_plate and +0.011 PCC_DEG over the gated UnDetect, with w = 0.4 in
  every fold. The two models carry complementary compound signal.
- Of the +0.0050 context-gate gain, +0.0032 is per-gene rescaling that does not use context.
  The line-identity gate is as good as the control-state gate.
- Compounds improved on PCC_plate vs the backbone: 60–65% for the gated UnDetect, 65–69% for XPert.

### 6.3 Ablation rows on the same CPU backbone (jobs 357912–357916 `run_sdst_hybrid.py --backbone-cache cache_v1`, tag `v8_cpu_abl`; final table job 358039)

The backbone is loaded from `cache_v1`, so every row below is paired with the rows of §6.1 on one backbone.
- The end-to-end same-feature network (`mlp_input`) does not use the backbone. Its v4_input predictions are re-paired
  against the CPU backbone.
- Check: the closed form recomputed in this run (`pool_check`) equals `pool_kta`. Δ is 0.0000 on both metrics, and
  per-fold test PCC_DEG agrees to 4 decimals.

Sources:
- `results/sdst_hybrid/v8_cpu_abl/fold_*/{summary.json, pred_test.npz}`.
- `results/sdst_hybrid/final_cpu/{scores.csv, contrasts.csv, final.json}`; log `logs/finalize-sdst-358039.out`.

Final drug-blind table, one backbone run (Δ = paired difference vs our backbone; folds > 0, folds with compound-bootstrap
CI > 0 / < 0):

| model | PCC_DEG | Δ (folds; CI+/CI−) | PCC_plate | Δ (folds; CI+/CI−) | PCC_DEG, no plate-mate |
|---|---|---|---|---|---|
| XPert (best published) | 0.6315 | −0.0057 (1; 0/4) | 0.1569 | +0.0569 (5; 5/0) | 0.5650 |
| drug-free MLP, published | 0.6372 | −0.0000 (4; 2/1) | 0.1018 | +0.0018 (5; 2/0) | 0.5350 |
| backbone (drug-free MLP, ours) | 0.6372 | — | 0.1000 | — | 0.5358 |
| + plate co-membership kernel (`pool_plate`) | 0.6429 | +0.0057 (5; 5/0) | 0.1006 | +0.0006 (3; 2/0) | 0.5587 |
| + chemistry kernel (`pool_chem`) | 0.6431 | +0.0059 (5; 5/0) | 0.1326 | +0.0326 (5; 5/0) | 0.5534 |
| same features, end to end (`mlp_input`) | 0.6131 | −0.0241 (0; 0/5) | 0.1510 | +0.0510 (5; 5/0) | 0.4960 |
| + same features, residual head (`nnf_multi`) | 0.6477 | +0.0105 (5; 5/0) | 0.1522 | +0.0522 (5; 5/0) | 0.5649 |
| UnDetect, closed form (`pool_kta`) | 0.6485 | +0.0113 (5; 5/0) | 0.1551 | +0.0550 (5; 5/0) | 0.5703 |
| UnDetect + context gate (`ctx_pc`) | 0.6499 | +0.0127 (5; 5/0) | 0.1600 | +0.0600 (5; 5/0) | 0.5738 |

Contrasts among compound terms (same rows):

| contrast | Δ PCC_DEG (folds; CI+/CI−) | Δ PCC_plate (folds; CI+/CI−) |
|---|---|---|
| UnDetect closed form − residual head | +0.0008 (4; 0/0) | +0.0029 (4; 0/0) |
| UnDetect + context gate − residual head | +0.0022 (5; 4/0) | +0.0079 (5; 3/0) |
| residual head − end to end | +0.0346 (5; 5/0) | +0.0011 (3; 0/0) |
| chemistry kernel − UnDetect closed form | −0.0054 (0; 0/5) | −0.0225 (0; 0/5) |
| plate kernel − UnDetect closed form | −0.0056 (0; 0/5) | −0.0545 (0; 0/5) |

(Rows with the residual head as minuend are sign-flipped from `contrasts.csv`, where `nnf_multi` is the reference.)

Reading:
- Every ablation reproduces its GPU-run counterpart within run noise. Paper values: plate kernel 0.643 / 0.100,
  chemistry 0.643 / 0.131, end to end 0.613 / 0.151, residual head 0.648 / 0.151.
- The chemistry kernel gives about half of the closed form's PCC_DEG gain (+0.0059 of +0.0113) and 59% of its
  PCC_plate gain (+0.0326 of +0.0550).
- The plate co-membership kernel matches the chemistry kernel on PCC_DEG (+0.006) but adds nothing compound-specific:
  PCC_plate +0.0006, 2/5 folds CI > 0.
- On PCC_plate, the gated UnDetect beats the raw-feature residual head by +0.008 (5/5 folds, 3/5 CI > 0). The
  closed form alone beats it by only +0.003 (no CI > 0).

## 7. Drug × cell-line gate on LINCS and Tahoe-100M (jobs 358880–358885; `run_l1000_xline.py --kernels D --gate D`, tag `_Dgate`)

Level-5 and Tahoe have no per-sample control, so the context is the cell line.
- `gate_line_D`: prediction `mu_L + u (1 + delta_L)`, with u = a_L g + b_L (f_L − mu_L) the `pool_D` compound term
  and one delta per (line, gene). This is the analogue of `ctx_line` on the drug-blind benchmark.
- `gate_scale_D`: one delta per gene shared by all lines (control).
- Fitting: delta is a ridge fit on the inner-CV predictions of the training rows.
- Selection: c by cross-fitting over the inner folds; c = ∞ returns `pool_D`.
- Check: `pool_D` and `mkl_D` reproduce the earlier `_D` runs exactly in all six settings.

Sources: `results/l1000_xline/{P2-24,P1-24}_{ik14,platemap}_seed0_Dgate/`, `T-{0.5,5}_ik14_seed0_Dgate/`
(`per_condition.parquet`, `selection.csv`); contrasts `results/l1000_xline/weights_vs_D_Dgate[_platemap].json`
(`scripts/analyze_xline_weights.py`). Line-macro mean over ACTIVE rows; Δ vs `pool_D` with the 2,000-replicate
bootstrap over ik14 groups; lines = lines with Δ > 0.

| setting | `pool_D` full | `gate_line_D` full | Δ full [95% CI], lines | Δ resid [95% CI], lines | `gate_scale_D` Δ full / resid | c chosen, per (line, fold) |
|---|---|---|---|---|---|---|
| P2-24 ik14 | 0.4615 | 0.4622 | +0.0007 [+0.0003, +0.0011], 6/7 | +0.0012 [+0.0009, +0.0015], 7/7 | +0.0002 / +0.0004 | 10: 30, ∞: 5 |
| P2-24 plate map | 0.4073 | 0.4078 | +0.0005 [+0.0001, +0.0009], 6/7 | +0.0004 [+0.0000, +0.0007], 3/7 | 0 / 0 (c = ∞) | ∞: 29, 10: 6 |
| P1-24 ik14 | 0.3451 | 0.3458 | +0.0007 [+0.0003, +0.0011], 7/7 | +0.0023 [+0.0020, +0.0026], 7/7 | +0.0003 / +0.0017 | 10: 34, ∞: 1 |
| P1-24 plate map | 0.2870 | 0.2884 | +0.0014 [+0.0006, +0.0022], 7/7 | +0.0027 [+0.0019, +0.0034], 7/7 | +0.0012 / +0.0025 | 10: 34, ∞: 1 |
| Tahoe 0.5 µM | 0.6671 | 0.6669 | −0.0002 [−0.0010, +0.0005], 11/50 | +0.0003 [−0.0005, +0.0010], 9/50 | 0 / 0 (c = ∞) | ∞: 215, 10: 32, 1: 3 |
| Tahoe 5 µM | 0.4400 | 0.4394 | −0.0006 [−0.0016, +0.0004], 9/50 | **−0.0010 [−0.0019, −0.0003]**, 2/50 | 0 / 0 (c = ∞) | ∞: 219, 10: 24, 1: 7 |

Reading (negative results included):
- **LINCS:** the line-identity gate adds +0.0005 to +0.0014 line-macro Pearson in all four settings, with
  intervals above zero, and +0.0004 to +0.0027 on the residual metric. On P1-24 most of it is the shared per-gene
  scale.
- **Tahoe-100M:**
  - No gain at 0.5 µM.
  - At 5 µM a small loss on the residual metric (interval below zero); the full-correlation change is not
    significant.
  - Validation switches the per-line gate off in 86–88% of line × fold cases. With about 50 lines and a few hundred
    compounds per line, one scale per (line, gene) is fitted from few rows.
- **The user's non-inferiority rule** (mean Δ ≥ 0 on both metrics, no interval below zero) holds on LINCS and fails
  on Tahoe at both doses: 0.5 µM full −0.0002; 5 µM resid interval < 0.
- **Size:** at three decimals the LINCS entries of Table 4 change by +0.001 at most, and Tahoe 5 µM by −0.001.

## 8. Final-run view ablation and compound analysis (Fig. 3, `tab:app-views`, `tab:app-pairs`)

### 8.1 View ablation with the interaction on the final backbone (`scripts/sdst_view_ablation.py`)

- **Jobs:** 359222, 359223, 359337 (fold 2 rerun), 359225, 359226.
- **Fold 2:** the first run, 359224, failed at `only_morph`. LAPACK `syevd` did not converge on a per-line kernel that
  has many zero rows. `cache_undetect.eigh_cpu` now retries with scipy `evr`, but only when the default fails, so the
  other folds are unaffected.
- **Method:** per view set,
  - KTA weights over the set's views and leave-one-out transfer terms on the `cache_v1` backbone;
  - then the interaction, ctx_pc, with c and K chosen on valid PCC_DEG.
- **Check:** `multi_norm` (all four views) reproduces Table 2 exactly: 0.6499 / 0.1600, and 0.6485 / 0.1551 without
  the interaction.
- **Source:** `/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/views_final/{means.csv, contrasts.csv,
  summary.json, fold_*/summary.json}`; logs `logs/sdst-views-<job>.out`.
- **Contrasts:** paired vs all four views (same arm), with a 2,000-replicate bootstrap over test compounds per fold.
- **Earlier ablation (§5.9 of `10_30_report.md`):** GPU backbone, transfer terms only. It is superseded in the paper.

| views (UnDetect, with interaction) | PCC_DEG | PCC_plate | Δ PCC_plate vs all four | folds < 0 / CI < 0 |
|---|---|---|---|---|
| none (backbone) | 0.6372 | 0.1000 | −0.0600 | 5 / 5 |
| **all four** | **0.6499** | **0.1600** | | |
| chemistry alone | 0.6441 | 0.1381 | −0.0219 | 5 / 5 |
| targets alone | 0.6437 | 0.1267 | −0.0334 | 5 / 5 |
| mechanism alone | 0.6438 | 0.1279 | −0.0322 | 5 / 5 |
| Cell Painting alone | 0.6430 | 0.1328 | −0.0273 | 5 / 5 |
| without chemistry | 0.6475 | 0.1469 | −0.0132 | 5 / 5 |
| without targets | 0.6497 | 0.1602 | +0.0002 | 2 / 0 |
| without mechanism | 0.6494 | 0.1588 | −0.0012 | 5 / 1 |
| without Cell Painting | 0.6470 | 0.1455 | −0.0145 | 5 / 5 |
| chemistry permuted | 0.6460 | 0.1384 | −0.0216 | 5 / 5 |
| targets permuted | 0.6492 | 0.1590 | −0.0011 | 4 / 2 |
| mechanism permuted | 0.6489 | 0.1577 | −0.0023 | 5 / 4 |
| Cell Painting permuted | 0.6470 | 0.1452 | −0.0149 | 5 / 5 |

Reading, against the earlier ablation without the interaction (§5.9 of `10_30_report.md`):
- **Each view alone:** +0.027 to +0.038 over the backbone, 44–63% of the full +0.060.
  - Earlier: +0.025 to +0.032.
  - Chemistry alone gains most from the interaction (0.131 → 0.138).
- **Chemistry and Cell Painting are not redundant:**
  - Removing them costs −0.013 / −0.015 (earlier −0.010 / −0.014).
  - Permuting them costs −0.022 / −0.015 (earlier −0.020 / −0.014).
  - All intervals are below zero in 5/5 folds.
- **Targets and mechanism are largely redundant:** dropping or permuting either costs |Δ| ≤ 0.0023. Same picture as
  before.

### 8.2 Compound analysis on the final run (job 359229; `analyze_sdst_compounds.py --final`)

- **Fused-kernel weights:** the `cache_v1` KTA means, chem 0.290 / tgt 0.245 / moa 0.226 / morph 0.238. Earlier (v3):
  0.290 / 0.246 / 0.227 / 0.237.
- **Part A, pair statistics:** no model is involved, so the values are identical to `10_30_report.md` §5.7.
- **Example pairs:** the 11 pairs of `tab:app-pairs` are in the top list again, with identical Tanimoto, fused and
  response values (2 d.p.).
- **Correction to the paper table:** BVD-523–TAK-733 has r = 0.86499, which rounds to 0.86, not 0.87.
  Triptolide–alvocidib (0.86506) therefore ranks above it. The table is fixed.
- **Part B:** per-compound PCC_plate gain of the multi-view transfer terms over chemistry (`pool_kta − pool_chem`,
  `final_cpu`), by nearest-chemical-neighbour bin.

| nearest Tanimoto | compounds | gain, final run | gain, earlier (v3) | UnDetect − backbone, final run |
|---|---|---|---|---|
| < 0.3 | 746 | +0.0233 | +0.0230 | +0.0360 |
| [0.3, 0.5) | 2,349 | +0.0153 | +0.0155 | +0.0303 |
| [0.5, 0.7) | 2,083 | +0.0058 | +0.0061 | +0.0364 |
| ≥ 0.7 | 3,039 | +0.0078 | +0.0089 | +0.0770 |

- **By coverage (gain over chemistry):** chem only −0.0026 (4,693 compounds), tgt +0.0086, morph +0.0478, tgt+morph
  +0.0444.
- **Source:** `/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/compound_analysis_final/{summary.json,
  hop_pairs_top.csv, drug_gain.csv, example_pairs.json}`.

### 8.3 XPert reproduction, fold 0 (job 357100; not blocking)

- **Result:** our retraining with the released code reaches test PCC_DEG **0.6337** on fold 0. The released
  predictions give 0.6319 (`published`, Table 2 source).
- **Training:** early stopping after 50 epochs without improvement.
- **Log:** `logs/dpb-train-357100.out`.
- **TranSiGen fold 0 (357101):** still running.
