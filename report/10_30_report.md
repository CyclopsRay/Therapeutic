# Therapeutic — Weekly Report, week ending Friday 2026-10-30

Reporting interval: [2026-10-23T18:00:00+04:00, 2026-10-30T18:00:00+04:00) — Asia/Dubai.

Measured results only, with provenance; include positive and negative findings.

## 1. E8 part 1: GSNN's graph-free NN baseline on GSNN's own benchmark (job 338366)

GSNN's own MLP (`gsnn-lib/scripts/training/train_nn_lincs.py`, branch `repro` commit `bbaa352`: Trainer-interface
fix only) on the same processed exp1 LINCS data, inputs (drug, dose, cell omics input nodes) and fold 0 as the GSNN
ladder (`10_23_report.md` §1). Upstream default NN hyper-parameters (channels 124, layers 2, lr 1e-3, batch 512,
early stopping on validation r2, patience 25), 100 epochs, CPU, 3 independent re-trainings (the trainer sets no seed).
Ran 2026-10-23 12:21–14:40 on cn-19, ~46 min per run. Pre-declared reading: `10_23_record.md` 2026-10-23 12:30.
Source: `/nfs-stor/yuqi.lei/Therapeutic/repro/gsnn/logs/train-nn-338366.out` lines 124, 257, 390;
`repro/gsnn/results/nn_e8/{5b407787…,38035c24…,b3dbbaac…}/fold_0_lincs_splits/result_metric_dict.pt`
(`86fc1767…` is the failed 338351 and is excluded).

| model | runs | r_flat test | mean_r test | r2 test | mse test | best epoch |
|---|---|---|---|---|---|---|
| graph-free NN | 3 | 0.2981, 0.2983, 0.3138 | 0.2585, 0.2574, 0.2746 | 0.0696, 0.0705, 0.0815 | 0.8893, 0.8885, 0.8779 | 100, 100, 100 |
| GSNN, real graph | 2 | 0.3170, 0.3099 | 0.2667, 0.2593 | 0.0835, 0.0789 | 0.8760, 0.8805 | 63, 60 |
| GSNN, degree-preserving rewired | 4 | 0.3268–0.3393 | 0.2803–0.2932 | 0.0897–0.0973 | 0.8628–0.8701 | 47–72 |

Means: NN r_flat **0.3034**, mean_r **0.2635**; GSNN real 0.3135 / 0.2630; GSNN rewired 0.3354 / 0.2891.

- **r_flat (primary): NN − GSNN real = −0.0101.** The pre-declared threshold (real − 0.01 = 0.30346) is missed by
  0.00008, so by the rule as written the result falls in the "NN below GSNN" branch. The margin is far inside the
  run-to-run spread (NN 0.298–0.314, real GSNN 0.310–0.317); the third NN run (0.3138) is above one real-graph run.
- **mean_r (secondary): NN 0.2635 vs GSNN real 0.2630, equal** (+0.0005).
- GSNN on a degree-preserving rewired graph beats the graph-free NN by +0.032 r_flat and +0.026 mean_r.
- All three NN runs reached their best validation score at the last epoch (100), i.e. they were still improving when
  the epoch cap stopped them; the GSNN runs peaked at epochs 47–77. Not used to change the reading; recorded as a
  limitation of the equal-epoch protocol.

## 2. E9 and the ladder as a test: configuration-model check, calibration and power (jobs 344299–344302)

`scripts/ladder_calibration.py` via `slurm/ladder_calibration.sbatch`; design pre-declared in `10_30_record.md`
(2026-10-26). Jobs: 344299 (real caches + 39 Erdős–Rényi graphs per network), 344300 (100 degree-preserving rewires per
network, `netnull.null`, 5 swaps per edge), 344301 (e9), 344302 (simulate, 4 chunks × 50 replicates); all ran
2026-10-26 03:58–05:03 on cn-06/15/20. Every log ends normally and none has a traceback (`sacct` was down; logs
checked). Source: `/nfs-stor/yuqi.lei/Therapeutic/results/ladder_calibration/<network>/{e9.json,
e9_by_degree_bin.csv,sim_*.csv,summary.csv}`; logs `logs/ladder-cal-34430{0,1,2}_*.out`. Smoke 344298
(`results/_smoke_ladder`) is not quoted.

Networks: STRING v12.0 functional ≥ 700 (undirected; 16,201 genes, 236,930 edges) and OmniPath + CollecTRI (directed;
10,168 genes, 132,324 edges). STRING physical was not run because its links file is no longer on disk.

### 2.1 Configuration-model proposition, numerical check (100 rewires; 300 synthetic target sets × 978 genes)

| quantity | STRING functional | OmniPath + CollecTRI |
|---|---|---|
| max \|Σ_g one-hop mass − K_S\|, rewired / real | 0 / 0 | 0 / 0 |
| R², rewire mean vs d_g K_S / E, one-hop | 0.835 | 0.926 |
| R², rewire mean vs formula × κ, two-hop | 0.826 | 0.972 |
| relative error of rewire mean by in-degree decile, one-hop | −1 % to +4 % | −5 % (top bin) to +14 % (low bins) |
| R², real graph vs formula, one-hop / two-hop | 0.079 / 0.229 | 0.324 / 0.118 |
| real edges kept in a rewire / rewire–rewire overlap / configuration expectation | 0.027 / 0.021 / 0.027 | 0.203 / 0.173 / 0.204 |

- Part (i) holds exactly: total one-hop mass equals the targets' total degree on every graph.
- Parts (ii) and (iii) hold for the rewired ensemble (R² 0.83–0.97). The directed network bends at its hubs as the
  simple-graph cap predicts: the top decile is 5 % below the formula, and the low deciles are 13–14 % above it.
- The real graphs are far from the degree expectation gene by gene (R² 0.08–0.32). So the wiring term is large; the
  question the ladder asks is whether a model can use it.
- The 20 % edge overlap for the directed network is structural: it equals the configuration-model expectation (0.204),
  so it does not reflect incomplete mixing.

### 2.2 Calibration and power (200 replicates per λ; R = 39; α = 0.05)

Truth s = [λ A_real + (1 − λ) A_G0] 1_T, with G0 a rewire outside the ladder pool; y = z(s) + 1.7 ε; predictor = OLS
on log one-/two-hop mass over the graph being tested. "Exchangeable" replaces the real graph with a further rewire.
Rejection rate [Wilson 95 % CI]:

| network | world | λ = 0 ladder | λ = 0 naive (vs ER) | λ = 0.05 | λ = 0.1 | λ = 0.2 | λ = 0.3 |
|---|---|---|---|---|---|---|---|
| STRING functional | exchangeable | **0.065 [0.038, 0.108]** | 1.000 | 0.880 | 1.000 | 1.000 | 1.000 |
| STRING functional | real topology | 0.000 [0, 0.019] | 1.000 | 0.015 | 0.960 | 1.000 | 1.000 |
| OmniPath + CollecTRI | exchangeable | **0.060 [0.035, 0.102]** | 1.000 | 0.860 | 1.000 | 1.000 | 1.000 |
| OmniPath + CollecTRI | real topology | 0.000 [0, 0.019] | 1.000 | 0.000 | 0.000 | 0.685 | 1.000 |

Mean flattened Pearson at λ = 0 (real topology): STRING real 0.061, rewired 0.078, ER 0.003, truth graph 0.505;
OmniPath + CollecTRI real 0.181, rewired 0.230, ER 0.003, truth graph 0.504.

- **Calibrated.** In the exchangeable control, the ladder rejects at 6.0–6.5 % at λ = 0, inside the pre-declared
  [0.02, 0.09].
- **The published-style control always rejects** (1.000 at λ = 0, both networks), although responses carry no
  real-graph wiring. It credits degree to the network.
- **Power.** With exchangeable topology, ≥ 0.86 already at λ = 0.05 (5 % of the propagated signal on real edges).
  With real topology, ≥ 0.8 is reached at λ = 0.1 (STRING) and λ = 0.3 (OmniPath + CollecTRI; 0.685 at 0.2).
- **With real topology the test is conservative, and real < rewired under the null.** When responses depend only on
  degree, the real graph predicts them worse than its rewires (−0.018 STRING, −0.049 OmniPath + CollecTRI), because
  its degree–degree structure moves mass away from the degree expectation. This reproduces the sign of the GSNN
  result (`10_23_report.md` §1: every rewired run above both real runs) without any wiring signal in the responses.
- Limits: one-hop truth only, one predictor class (OLS on log one-/two-hop mass), one noise level (oracle r ≈ 0.5),
  300 drugs.

## 3. Wiring-effect bound and credit decomposition for the published-model audits (no new training)

`scripts/wiring_bound.py` (gnn-gpu env, login node, 2026-10-26 10:50; reads existing outputs only). Design:
`10_30_record.md` 2026-10-26 10:20, output (3). wiring = mean(real) − mean(rewired) with a Welch interval over runs
(one-sided 95 % upper bound; two-sided 90 % lower); degree = rewired − ER; identity = rewired − labels. Runs pinned by
id in the script: GSNN `10_23_report.md` §1 (failed `none/33d224fa`, epoch 2, excluded), GEARS `10_16_report.md` §2,
DeepCE node2vec `10_09_report.md` §5.4 (one rewired graph × 3 training seeds, so its bound is conditional on that
rewire). Source: `/nfs-stor/yuqi.lei/Therapeutic/results/wiring_bound/{audit_runs,wiring_bound}.csv`.

| model | metric | runs real / rewired | real | rewired | wiring [90 % lower, 95 % upper] | min. permutation p | degree (rew − ER) | identity (rew − labels) |
|---|---|---|---|---|---|---|---|---|
| GSNN | r_flat | 2 / 4 | 0.3135 | 0.3353 | **−0.022 [−0.034, −0.010]** | 0.20 | +0.016 | +0.034 |
| GSNN | mean_r | 2 / 4 | 0.2630 | 0.2891 | **−0.026 [−0.039, −0.014]** | 0.20 | +0.017 | +0.039 |
| GEARS | pearson_de | 3 / 3 | 0.8741 | 0.8486 | +0.026 [−0.009, +0.060] | 0.25 | +0.013 | −0.002 |
| GEARS | Δ Pearson, unseen single | 3 / 3 | 0.4217 | 0.4315 | −0.010 [−0.054, +0.034] | 0.25 | +0.108 | +0.035 |
| GEARS | Δ Pearson, combo seen 0 (n = 9) | 3 / 3 | 0.5052 | 0.5754 | −0.070 [−0.266, +0.125] | 0.25 | +0.075 | +0.058 |
| GEARS | Δ Pearson, combo seen 1 | 3 / 3 | 0.5739 | 0.5505 | +0.023 [−0.040, +0.086] | 0.25 | +0.089 | +0.013 |
| GEARS | Δ Pearson, combo seen 2 | 3 / 3 | 0.6432 | 0.6161 | +0.027 [−0.091, +0.145] | 0.25 | +0.082 | +0.007 |
| DeepCE | test Pearson | 3 / 3 | 0.4555 | 0.4421 | +0.013 [−0.040, +0.067] | 0.25 | +0.010 | −0.006 |

- **GSNN: the bound excludes any positive wiring effect** (upper −0.010 r_flat, −0.014 mean_r); the real graph is
  below its rewires (one-sided Welch p = 0.014 / 0.011 for rewired > real).
- **GEARS and DeepCE: run-level bounds are wide.** With 3 runs per arm they exclude wiring effects above only
  +0.034 (GEARS unseen single), +0.060 (GEARS pearson_de) and +0.067 (DeepCE). DeepCE's drug-bootstrap interval
  with seed-averaged predictions is narrower (+0.013 [−0.005, +0.031], `10_09_report.md` §5.4), but it treats the
  three training seeds and the single rewire as fixed.
- With R = 3–4 rewires no audit can reach permutation p < 0.05 (minimum p 0.20–0.25).
- In GEARS the degree term (rewired − ER) is larger than the wiring point estimate on every Δ-Pearson subgroup
  (+0.075 to +0.108).

## 4. Cross-line multimodal kernel model for unseen drugs (2026-10-28; exploratory; design `10_30_record.md` 2026-10-28 04:55)

### 4.1 LINCS P2-24, per-line MKL vs cross-line pooled (jobs 348395 ik14, 348396 plate-map)

`scripts/run_l1000_xline.py`. Per-line arms reproduce `results/l1000_mkl/P2-24_ik14_seed0` bit for bit (smoke 348394,
max |Δ| 0.0). ACTIVE rows, full Pearson, line-macro; contrasts = paired line-macro difference [95 % bootstrap over
split groups, 2,000] lines positive. Source: `/nfs-stor/yuqi.lei/Therapeutic/results/l1000_xline/P2-24_{ik14,platemap}_seed0/per_condition.parquet`.

| arm | ik14 (5,477 rows) | plate-map (4,924 rows) |
|---|---|---|
| line mean | 0.3292 | 0.3147 |
| krr_chem (per line) | 0.3621 | 0.3421 |
| pool_T1 (chem, + consensus) | 0.3647 | 0.3442 |
| mkl_T3 (per line; chem+tgt+moa+deg+ppi1+morph) | 0.4498 | 0.3936 |
| gonly_T3 (consensus only) | 0.4205 | 0.3841 |
| **pool_T3** | **0.4601** | **0.4053** |

| contrast | ik14 | plate-map |
|---|---|---|
| pool_T3 − mkl_T3 | **+0.0103 [+0.0086, +0.0120] 7/7** | **+0.0117 [+0.0098, +0.0137] 7/7** |
| pool_T1 − krr_chem | +0.0027 [+0.0015, +0.0039] 7/7 | +0.0021 [+0.0009, +0.0031] 7/7 |
| mkl_T3 − krr_chem | +0.0877 [+0.0807, +0.0947] 7/7 | +0.0515 [+0.0420, +0.0610] 7/7 |
| gonly_T3 − mkl_T3 | −0.0292 [−0.0326, −0.0259] 0/7 | −0.0095 [−0.0131, −0.0060] 0/7 |

- Cross-line pooling adds +0.010 to +0.012 on top of the multimodal kernel, 7/7 lines in both splits; with chemistry
  alone it adds only +0.002 to +0.003. The consensus term alone is worse than per-line: both parts are needed.

### 4.2 DeepCE's own split: joint kernel K_drug · k_dose · (ρ + δ_line) (job 348410; `repro/deepce/xline_kernel.py`)

Upstream filter and split (1,895 / 544 / 496 train / dev / test signatures, 92 test drugs); (views, σ_dose, ρ, λ)
chosen on upstream dev Pearson, test scored once per arm; metric = upstream mean per-signature Pearson. Test-drug view
coverage: chem 92, targets 62, MOA 62, Cell Painting 72. Contrasts: paired per-signature difference, bootstrap over the
92 test drugs (2,000), drugs where a > b. DeepCE = our 3-seed reproduction (`10_09_report.md` §5.3), seed-averaged per
signature. Source: `/nfs-stor/yuqi.lei/Therapeutic/repro/deepce/results/xline_kernel/deepce_split/{summary.json,contrasts.csv}`.

| arm | σ_dose | ρ | λ | dev | test |
|---|---|---|---|---|---|
| mean per (line, dose) | | | | | 0.3886 |
| Morgan kNN (`10_09_report.md` §5.1) | | | | | 0.4511 |
| DeepCE `ppi`, 3-seed reproduction / paper Table 1 | | | | | 0.4548 / 0.4907 |
| KRR chem, per line | 2.0 | 0 | 1.0 | 0.4770 | 0.4911 |
| KRR chem, cross-line | 2.0 | 0.1 | 1.0 | 0.4792 | 0.4970 |
| KRR chem+tgt+moa, per line / cross-line | 4.0 | 0 / 0.1 | 1.0 | 0.5085 / 0.5097 | 0.5597 / 0.5639 |
| KRR multi (+ Cell Painting), per line | 4.0 | 0 | 0.3 | 0.5480 | 0.5957 |
| **KRR multi, cross-line** | 4.0 | 0.25 | 1.0 | 0.5515 | **0.5994** |

| contrast | diff [95 %] | drugs a > b |
|---|---|---|
| multi cross-line − DeepCE ppi | **+0.1445 [+0.1129, +0.1771]** | 78 / 92 |
| multi cross-line − DeepCE onehot | +0.1333 [+0.0998, +0.1685] | 78 / 92 |
| multi cross-line − Morgan kNN | +0.1483 [+0.0935, +0.1965] | 70 / 92 |
| chem cross-line − DeepCE ppi | +0.0422 [+0.0078, +0.0772] | 67 / 92 |
| chem per line − DeepCE ppi | +0.0363 [+0.0024, +0.0711] | 66 / 92 |
| chem cross-line − chem per line | +0.0059 [+0.0036, +0.0080] | 61 / 92 |
| chem+kn cross-line − per line | +0.0042 [+0.0023, +0.0064] | 62 / 92 |
| multi cross-line − per line | +0.0037 [+0.0009, +0.0069] | 52 / 92 |
| chem+kn − chem (cross-line) | +0.0669 [+0.0241, +0.1046] | 52 / 92 |
| multi − chem+kn (cross-line) | +0.0355 [+0.0211, +0.0530] | 69 / 92 |

- On DeepCE's own benchmark the multimodal cross-line kernel model reaches 0.599, +0.145 over the reproduced DeepCE
  and +0.109 over DeepCE's published 0.4907. Chemistry-only KRR with a dose kernel already beats reproduced DeepCE.
- Single split; the first run (job 348404, σ grid capped at 2.0) gave the same ordering (multi cross-line 0.5985).
  Annotation source for targets/MOA: LINCS2020 `compoundinfo` (Repurposing Hub; literature-curated, not L1000-derived).

### 4.3 Tahoe unseen-drug slices, 50 lines (jobs 348399 T-0.05, 348400 T-0.5, 348401 T-5; `--source Tahoe`, ik14)

Same script and protocol as §4.1, Tahoe 2,000-gene deltas, 350 compounds × 50 lines; KNOWN 245, Cell Painting 99 of
350 compounds. ACTIVE rows, line-macro full Pearson; contrasts [95 % bootstrap over ik14 groups] lines positive (ties at
exactly 0 count as not positive). T3n (availability-normalised T3, record 2026-10-28 05:50) only in 348401.
Source: `/nfs-stor/yuqi.lei/Therapeutic/results/l1000_xline/T-{0.05,0.5,5}_ik14_seed0/per_condition.parquet`.

| arm | T-0.05 | T-0.5 | T-5 |
|---|---|---|---|
| line mean | 0.4870 | 0.6440 | 0.3976 |
| krr_chem | 0.4977 | 0.6594 | 0.4147 |
| pool_T1 | 0.4986 | 0.6641 | 0.4220 |
| mkl_T3 | 0.4960 | 0.6602 | 0.4389 |
| pool_T3 | 0.4971 | 0.6679 | 0.4432 |
| mkl_T3n / pool_T3n | – | – | 0.4367 / 0.4415 |

| contrast | T-0.05 | T-0.5 | T-5 |
|---|---|---|---|
| pool_T3 − mkl_T3 | +0.0011 [+0.0006, +0.0017] 15/50 | **+0.0077 [+0.0040, +0.0115] 50/50** | **+0.0043 [+0.0007, +0.0080] 46/50** |
| pool_T1 − krr_chem | +0.0009 [+0.0006, +0.0014] 21/50 | +0.0048 [+0.0029, +0.0066] 49/50 | +0.0072 [+0.0040, +0.0106] 46/50 |
| mkl_T3 − krr_chem | −0.0017 [−0.0061, +0.0030] 21/50 | +0.0008 [−0.0096, +0.0112] 25/50 | **+0.0242 [+0.0101, +0.0377] 50/50** |
| gonly_T3 − mkl_T3 | −0.0071 [−0.0118, −0.0031] 2/50 | −0.0076 [−0.0175, +0.0025] 8/50 | −0.0276 [−0.0413, −0.0143] 0/50 |
| mkl_T3n − mkl_T3 | – | – | −0.0022 [−0.0056, +0.0012] 16/50 |
| pool_T3n − pool_T3 | – | – | −0.0017 [−0.0042, +0.0010] 15/50 |

- Cross-line pooling is positive at all three doses; it is negligible at 0.05 µM, where every arm is within 0.011 of
  the line mean.
- The multimodal kernel helps on Tahoe only at 5 µM (+0.024, 50/50); at 0.05 and 0.5 µM it is indistinguishable
  from chemistry.
- Availability normalisation does not help on Tahoe T-5 (−0.002, ns).

### 4.4 LINCS P1-24, 8,119 compounds with sparse line coverage (jobs 348397 ik14, 348398 plate-map)

Same protocol as §4.1. 957 compounds are measured in one line only; KNOWN 2,176, Cell Painting 748 of 8,119 compounds.
Contrasts vs krr_chem from the job logs (line-macro bootstrap, 2,000); pool_T3 − mkl_T3 and gonly from the parquet.
Source: `/nfs-stor/yuqi.lei/Therapeutic/results/l1000_xline/P1-24_{ik14,platemap}_seed0/per_condition.parquet`;
logs `logs/l1000-xline-{348397,348398}.out`.

| arm | ik14 (12,279 ACTIVE rows) | plate-map (7,842) |
|---|---|---|
| line mean | 0.2800 | 0.2526 |
| krr_chem | 0.3320 | 0.2733 |
| pool_T1 | 0.3370 | 0.2801 |
| mkl_T3 | 0.3350 | 0.2769 |
| gonly_T3 | 0.3194 | 0.2759 |
| **pool_T3** | **0.3420** | **0.2838** |

| contrast | ik14 | plate-map |
|---|---|---|
| pool_T3 − mkl_T3 | **+0.0070 [+0.0061, +0.0078] 7/7** | **+0.0069 [+0.0046, +0.0095] 6/7** |
| pool_T1 − krr_chem | +0.0049 [+0.0041, +0.0058] | +0.0067 [+0.0052, +0.0091] 7/7 |
| mkl_T3 − krr_chem | +0.0030 [+0.0002, +0.0057] | +0.0036 [−0.0017, +0.0069] 4/7 |
| pool_T3 − krr_chem | **+0.0099 [+0.0070, +0.0129]** | **+0.0105 [+0.0059, +0.0136] 7/7** |
| gonly_T3 − mkl_T3 | −0.0156 [−0.0183, −0.0130] 0/7 | −0.0010 [−0.0056, +0.0041] 4/7 |

- On P1-24 ACTIVE, where the 10-14 leaderboard (`10_16_report.md` §3) had no arm above KRR Morgan2 (0.332),
  pool_T3 is +0.010 above chemistry KRR on both splits.
- With sparse view coverage the per-line multimodal kernel adds only +0.003–0.004; pooling adds +0.005–0.007 with
  chemistry alone, more than on the near-complete P2-24 grid (+0.002–0.003).

### 4.5 DeepCE data, 5-fold drug-disjoint resplits × 2 seeds (jobs 348411 seed 0, 348412 seed 1)

`repro/deepce/xline_kernel.py --cv 5`: all 2,935 upstream train+dev+test signatures pooled; folds by hashed pert_id;
inner dev = 1/5 of each fold's training drugs; same grids and selection as §4.2; `*_norm` = availability-normalised
drug kernel. Test = upstream mean per-signature Pearson per fold. Source:
`/nfs-stor/yuqi.lei/Therapeutic/repro/deepce/results/xline_kernel/cv5_seed{0,1}/folds.csv`.

| arm | test, seed 0 (mean ± sd over 5 folds) | test, seed 1 |
|---|---|---|
| mean per (line, dose) | 0.3890 ± 0.0156 | 0.3929 ± 0.0127 |
| KRR chem, per line / cross-line | 0.4747 / 0.4804 | 0.4923 / 0.4970 |
| KRR chem+tgt+moa, per line / cross-line | 0.5286 / 0.5348 | 0.5349 / 0.5399 |
| KRR chem+tgt+moa norm, per line / cross-line | 0.5323 / 0.5375 | 0.5400 / 0.5435 |
| KRR multi, per line / cross-line | 0.5633 / 0.5696 | 0.5718 / 0.5781 |
| **KRR multi norm, cross-line** (per line) | **0.5744** ± 0.0354 (0.5659) | **0.5831** ± 0.0028 (0.5761) |

| fold-paired contrast | seed 0: mean, folds won | seed 1 |
|---|---|---|
| multi norm cross-line − multi cross-line (normalisation) | +0.0048, 4/5 | +0.0050, 5/5 |
| multi cross-line − multi per line (cross-line coupling) | +0.0062, 4/5 | +0.0063, 5/5 |
| chem cross-line − chem per line | +0.0057, 5/5 | +0.0047, 5/5 |
| multi norm cross-line − multi per line | +0.0111, 5/5 | +0.0113, 5/5 |
| multi norm cross-line − chem per line | +0.0997, 5/5 | +0.0908, 5/5 |

- All 10 folds rank the components the same way: views (+0.09–0.10) > cross-line coupling (+0.006) ≈ availability
  normalisation (+0.005); each is positive in 9–10 of 10 folds.

### 4.6 DeepCE split: availability normalisation (job 348415) and same-input MLP control (job 348414)

348415 = §4.2 rerun with `*_norm` arms (non-norm arms reproduce 348410 exactly). 348414 = `repro/deepce/mlp_multimodal.py`:
2-hidden-layer MLP on the same inputs and target as the kernel model (Morgan bits; + target / MOA multi-hot, z-scored
Cell Painting, 3 availability flags; cell one-hot; log10 dose), width / dropout / weight decay / epoch (≤ 200) on
upstream dev, 3 seeds of the chosen configuration averaged per signature. Contrasts as §4.2 (92-drug bootstrap).
Source: `/nfs-stor/yuqi.lei/Therapeutic/repro/deepce/results/{xline_kernel/deepce_split,mlp_multimodal}/`.

| arm | dev | test |
|---|---|---|
| KRR multi norm, per line | 0.5532 | 0.5918 |
| **KRR multi norm, cross-line** | 0.5570 | **0.6018** |
| KRR chem+kn norm, per line / cross-line | 0.5148 / 0.5156 | 0.5632 / 0.5670 |
| MLP chem (1024, dropout 0.3, wd 1e-3) | 0.458–0.460 | 0.5110–0.5118 (seed-avg 0.5112) |
| MLP multi (1024, dropout 0.3, wd 1e-5) | 0.534–0.537 | 0.5355–0.5530 (seed-avg 0.5435) |

| contrast | diff [95 %] | drugs a > b |
|---|---|---|
| multi norm cross-line − DeepCE ppi (3-seed) | **+0.1470 [+0.1153, +0.1801]** | 79 / 92 |
| multi norm cross-line − chem cross-line | +0.1048 [+0.0667, +0.1398] | 67 / 92 |
| multi norm cross-line − multi cross-line | +0.0025 [−0.0018, +0.0067] | 53 / 92 |
| chem+kn norm cross-line − chem+kn cross-line | +0.0031 [−0.0012, +0.0081] | 57 / 92 |
| KRR multi cross-line − MLP multi | **+0.0559 [+0.0240, +0.0846]** | 56 / 92 |
| KRR chem per line − MLP chem | −0.0202 [−0.0365, −0.0032] | 43 / 92 |
| MLP multi − MLP chem | +0.0323 [−0.0054, +0.0742] | 53 / 92 |

- With chemistry alone the MLP is better than chemistry KRR on this test split (+0.020), but given the extra views it
  gains only +0.032 (interval crosses 0), whereas the kernel model gains +0.10 and ends +0.056 above the MLP.
  On dev the ordering is the same for the multimodal arms (0.557 vs 0.537) and reversed for chemistry (KRR 0.477 vs
  MLP 0.459). Single split; the MLP was not run on the §4.5 resplits.
- Normalisation on the single split: +0.0025 / +0.0031, intervals cross 0 (on the resplits §4.5: +0.005, 9/10 folds).

### 4.7 Availability normalisation on LINCS P2-24 (jobs 348416 ik14, 348417 plate-map; `--kernels T3n`)

T3n rows pair with §4.1 rows (same folds, same line means). Contrasts as §4.1; strata = mean per-row difference.
Source: `/nfs-stor/yuqi.lei/Therapeutic/results/l1000_xline/P2-24_{ik14,platemap}_seed0_T3n/per_condition.parquet`.

| contrast | ik14 | plate-map |
|---|---|---|
| mkl_T3n − mkl_T3 | −0.0045 [−0.0053, −0.0036] 0/7 | +0.0044 [+0.0026, +0.0062] 6/7 |
| pool_T3n − pool_T3 | −0.0011 [−0.0018, −0.0003] 2/7 | +0.0006 [−0.0007, +0.0019] 6/7 |
| pool_T3n − mkl_T3 | +0.0092 [+0.0077, +0.0108] 7/7 | +0.0123 [+0.0102, +0.0144] 7/7 |
| pool_T3n − pool_T3, compounds without Cell Painting | +0.0068 (657 rows) | +0.0084 (630 rows) |
| pool_T3n − pool_T3, compounds with Cell Painting | −0.0022 (4,820) | −0.0005 (4,294) |

- On P2-24 the normalisation is neutral overall (−0.001 / +0.001) with opposite signs per split; it helps the ~12 %
  of rows whose compound lacks Cell Painting and slightly costs the rest.

### 4.8 Availability normalisation on LINCS P1-24 (jobs 348433 ik14, 348434 plate-map; `--kernels T3n`)

As §4.7. 348434 ended with a ZeroDivisionError in the final printed summary only (no krr_chem arm in a T3n-only run;
all outputs had been written; guard added to the script afterwards). Contrasts computed by pairing with §4.4 rows.
Source: `/nfs-stor/yuqi.lei/Therapeutic/results/l1000_xline/P1-24_{ik14,platemap}_seed0_T3n/per_condition.parquet`.

| arm / contrast | ik14 | plate-map |
|---|---|---|
| mkl_T3 / mkl_T3n | 0.3350 / 0.3416 | 0.2769 / 0.2790 |
| pool_T3 / **pool_T3n** | 0.3420 / **0.3437** | 0.2838 / **0.2858** |
| mkl_T3n − mkl_T3 | +0.0066 [+0.0056, +0.0076] 7/7 | +0.0020 [+0.0000, +0.0059] 6/7 |
| pool_T3n − pool_T3 | +0.0018 [+0.0009, +0.0026] 5/7 | +0.0019 [+0.0002, +0.0049] 5/7 |
| pool_T3n − mkl_T3 | +0.0087 [+0.0076, +0.0098] 7/7 | +0.0088 [+0.0054, +0.0138] 7/7 |
| **pool_T3n − krr_chem** | **+0.0117 [+0.0090, +0.0145] 7/7** | **+0.0124 [+0.0085, +0.0153] 7/7** |
| pool_T3n − pool_T3, compounds without / with targets | +0.0024 / +0.0017 | +0.0036 / −0.0015 |
| pool_T3n − pool_T3, compounds without / with Cell Painting | +0.0021 / +0.0021 | +0.0028 / −0.0036 |

- Where views are sparse (P1-24: 73 % of compounds without targets, 91 % without Cell Painting) the normalisation is
  positive on both splits (+0.002 pooled; +0.002–0.007 per line). Across datasets: positive on P1-24 and the DeepCE
  resplits, neutral on P2-24 and Tahoe T-5.

### 4.9 DECANT view set D: chem + tgt + moa + Cell Painting, normalised, no network views (jobs 350677–350682; `--kernels D`)

Reruns of §4.1/4.3/4.4 for the paper's `tab:xline`. Same folds: line mean, krr_chem and pool_T1 reproduce the earlier
runs exactly. ACTIVE rows, line-macro full Pearson; contrasts = paired line-macro difference [95 % bootstrap over split
groups, 2,000] lines positive (computed 2026-10-29 09:20 with `run.line_macro_boot`; T3 arms paired from the §4.1/4.3/4.4
parquets). Source: `/nfs-stor/yuqi.lei/Therapeutic/results/l1000_xline/{P2-24,P1-24}_{ik14,platemap}_seed0_D/`,
`T-{0.5,5}_ik14_seed0_D/per_condition.parquet`; logs `/nfs-stor/yuqi.lei/Therapeutic/logs/l1000-xline-35067{7..9},35068{0..2}.out`.

| arm | P2-24 ik14 | P2-24 plate-map | P1-24 ik14 | P1-24 plate-map | T-0.5 | T-5 |
|---|---|---|---|---|---|---|
| line mean | 0.3292 | 0.3147 | 0.2800 | 0.2526 | 0.6440 | 0.3976 |
| krr_chem | 0.3621 | 0.3421 | 0.3320 | 0.2733 | 0.6594 | 0.4147 |
| mkl_D (per line) | 0.4483 | 0.4006 | 0.3423 | 0.2806 | 0.6605 | 0.4340 |
| gonly_D | 0.4224 | 0.3842 | 0.3202 | 0.2753 | 0.6529 | 0.4111 |
| **pool_D (DECANT)** | **0.4615** | **0.4073** | **0.3451** | **0.2870** | **0.6671** | **0.4400** |

| contrast | P2-24 ik14 | P2-24 plate-map | P1-24 ik14 | P1-24 plate-map | T-0.5 | T-5 |
|---|---|---|---|---|---|---|
| pool_D − mkl_D | **+0.0132 [+0.0118, +0.0147] 7/7** | **+0.0067 [+0.0054, +0.0080] 7/7** | **+0.0028 [+0.0022, +0.0035] 7/7** | **+0.0064 [+0.0043, +0.0080] 7/7** | **+0.0066 [+0.0044, +0.0090] 47/50** | **+0.0060 [+0.0027, +0.0094] 49/50** |
| pool_D − krr_chem | +0.0995 [+0.0936, +0.1056] | +0.0652 [+0.0553, +0.0752] | +0.0130 [+0.0105, +0.0157] | +0.0136 [+0.0094, +0.0168] | +0.0077 [+0.0016, +0.0144] | +0.0253 [+0.0152, +0.0348] |
| mkl_D − krr_chem | +0.0862 [+0.0794, +0.0931] | +0.0585 [+0.0490, +0.0677] | +0.0102 [+0.0080, +0.0125] | +0.0072 [+0.0040, +0.0099] | +0.0012 [−0.0064, +0.0087] | +0.0193 [+0.0078, +0.0302] |
| gonly_D − mkl_D | −0.0258 [−0.0290, −0.0228] 0/7 | −0.0164 [−0.0199, −0.0123] 0/7 | −0.0220 [−0.0243, −0.0198] 0/7 | −0.0053 [−0.0091, −0.0015] 2/7 | −0.0076 [−0.0151, +0.0001] 6/50 | −0.0230 [−0.0362, −0.0099] 0/50 |
| mkl_D − mkl_T3 | −0.0015 [−0.0026, −0.0004] 2/7 | +0.0070 [+0.0052, +0.0088] 7/7 | +0.0073 [+0.0061, +0.0085] 7/7 | +0.0036 [+0.0014, +0.0074] 6/7 | +0.0003 [−0.0036, +0.0046] 20/50 | −0.0049 [−0.0110, +0.0013] 4/50 |
| pool_D − pool_T3 | +0.0014 [+0.0005, +0.0023] 5/7 | +0.0020 [+0.0005, +0.0035] 6/7 | +0.0031 [+0.0020, +0.0041] 7/7 | +0.0031 [+0.0013, +0.0053] 6/7 | −0.0008 [−0.0034, +0.0019] 17/50 | −0.0032 [−0.0082, +0.0019] 7/50 |

- The cross-line term helps in every setting with set D, intervals above 0 and 7/7, 47/50, 49/50 lines.
  Effect sizes vs set T3: larger on P2-24 ik14 (+0.013 vs +0.010) and in Tahoe line counts (T-5 49/50 vs 46/50),
  smaller on P2-24 plate-map (+0.007 vs +0.012) and P1-24 ik14 (+0.003 vs +0.007).
- Dropping the degree and one-hop STRING views (D vs T3; D also normalises for missing views) does not hurt DECANT:
  +0.001 to +0.003 on LINCS (intervals above 0), −0.001 / −0.003 on Tahoe (intervals include 0).
- The cross-line term alone (gonly_D) is below the per-line kernel everywhere by point estimate. Its interval
  includes 0 only at T-0.5.

## 5. Bai et al. 2026 drug-blind L1000 benchmark: plate structure and first kernel-model folds (2026-10-28 evening)

Data: XPert L1000 SDST (78,453 control/treated pairs, 8,276 compounds, 164 lines, 10 µM, 24 h; Zenodo 17182939
`processed_data.zip`, md5 4a58bc01…). Folds = the benchmark's drug-blind 5-fold split, reproduced exactly
(`scripts/prep_xpert_sdst.py`, job 350503; per-fold sample and drug counts equal the benchmark's
`A_split_integrity.csv`). Metric PCC_DEG = mean over test samples of Pearson(true delta, predicted delta) over 978
genes. Published values: drug-perturbation-benchmark @ 55edbfe `full_vs_mlp.csv` (fold-level only).

### 5.1 The paired control identifies the plate (job 350554; `scripts/sdst_plate_baseline.py`)

`X_ctl` has 9,951 distinct rows = 9,951 (cell line, detection-plate set) groups, one control row per group, shared by
all treated samples of the group (up to 321). Model-free baselines fitted on train rows, scored on test:

| fold | test samples whose control row occurs in train | cell-line mean | plate-mean lookup (all / shared / unshared) |
|---|---|---|---|
| 0 | 0.904 | 0.2438 | 0.6049 / 0.6436 / 0.2402 |
| 1 | 0.905 | 0.2484 | 0.6027 / 0.6407 / 0.2396 |
| 2 | 0.904 | 0.2366 | 0.6026 / 0.6419 / 0.2344 |
| 3 | 0.904 | 0.2438 | 0.6079 / 0.6475 / 0.2338 |
| 4 | 0.906 | 0.2423 | 0.6041 / 0.6411 / 0.2492 |
| mean | 0.905 | 0.2430 | **0.6044** / 0.6430 / 0.2394 |

- Predicting a test compound by the mean response of the *other* training compounds on its plate (no drug
  information, no learning) gives PCC_DEG 0.604: above five of the seven published models (DeepCE 0.395, PertDiT
  0.370, PRnet 0.277, MultiDCP 0.255, CIGER 0.246), below TranSiGen 0.614, XPert 0.632 and the drug-free MLP 0.637.
  On the 90% of test samples that share a plate with training samples it is 0.643.
- The cell-line mean (0.243) equals the benchmark's "Mean cell" (0.243). Source:
  `/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/plate_baseline/summary.json`.

### 5.2 Kernel model, all five folds (jobs 350520–350522, 350548, 350549; `scripts/run_sdst_hybrid.py` `--tag v1`)

Backbone = our run of the benchmark's drug-free MLP (seed 1729; early stop at epoch 345–408). Δ = paired difference
vs our MLP; per fold a 2,000-replicate bootstrap over test drugs; "CI > 0" = folds whose 95% interval excludes 0;
drugs up = mean share of test drugs whose mean per-sample PCC improves. Source:
`/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/v1/{fold_*/summary.json,analysis.json}`.

| arm | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean | Δ mean | folds > 0 / CI > 0 | drugs up |
|---|---|---|---|---|---|---|---|---|---|
| published MLP | 0.6383 | 0.6349 | 0.6347 | 0.6421 | 0.6360 | 0.6372 | | | |
| published XPert (best of 7) | 0.6319 | 0.6274 | 0.6363 | 0.6342 | 0.6278 | 0.6315 | | | |
| our MLP | 0.6382 | 0.6352 | 0.6359 | 0.6416 | 0.6358 | 0.6373 | | | |
| krr_chem | 0.6430 | 0.6402 | 0.6403 | 0.6445 | 0.6407 | 0.6417 | +0.0044 | 5 / 5 | 0.43 |
| pool_chem | 0.6442 | 0.6417 | 0.6416 | 0.6461 | 0.6421 | 0.6432 | +0.0058 | 5 / 5 | 0.46 |
| krr_multi_norm | 0.6481 | 0.6443 | 0.6443 | 0.6480 | 0.6444 | 0.6458 | +0.0085 | 5 / 5 | 0.44 |
| pool_multi | 0.6511 | 0.6475 | 0.6469 | 0.6511 | 0.6477 | 0.6489 | +0.0116 | 5 / 5 | 0.47 |
| pool_multi_norm | 0.6505 | 0.6472 | 0.6467 | 0.6507 | 0.6472 | 0.6485 | +0.0111 | 5 / 5 | 0.42 |
| nnk_chem | 0.6457 | 0.6429 | 0.6447 | 0.6470 | 0.6436 | 0.6448 | +0.0075 | 5 / 5 | 0.54 |
| **nnk_multi_norm** | 0.6519 | 0.6465 | 0.6477 | 0.6516 | 0.6486 | **0.6493** | **+0.0119** | 5 / 5 | 0.58 |
| pool_plate (plate-overlap kernel) | 0.6441 | 0.6406 | 0.6412 | 0.6470 | 0.6414 | 0.6429 | +0.0055 | 5 / 5 | 0.81 |

- Our MLP reproduces the published drug-free MLP (0.6373 vs 0.6372). Every drug-kernel arm is above our MLP, the
  published MLP and every published model in 5/5 folds, with every interval above 0.
- Multi-view vs chemistry: pool_multi_norm − pool_chem +0.0046 to +0.0063, 5/5 folds. Neural head on kernel features
  (nnk) ≥ pooled KRR on the same kernel in 4/5 folds (+0.0008 mean).
- Plate overlap on top of the trained MLP adds +0.0055 (half the multi-view gain, below its per-fold interval lower
  bounds), but improves 81% of test drugs; the drug kernel arms improve fewer drugs (42–58%) by more on average.
- Smoke runs 350519 / 350535 (backbone trained 2 epochs) gave pool_plate 0.645 from a 0.312 backbone; with a trained
  backbone the plate term adds only +0.0055 because the MLP's input (the shared control row) already encodes the
  plate. Smoke numbers are not results.
- Running: v2 (350550, + plate-and-drug kernel arm) and v3 (350588–350590, same arms with saved predictions for the
  plate-centred metric).

### 5.3 Drug kernel beyond plates (job 350550, `--tag v2 --plate-drug`)

`pool_plate_drug` = our MLP + per-line KRR with kernel J_plate + γ·K_multi_norm (γ chosen 0.25 on valid in every
fold; λ 1.0; b 0.75). Same folds; v2 reproduces v1 for every shared arm exactly. Source:
`/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/v2/`.

| arm | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean |
|---|---|---|---|---|---|---|
| pool_plate | 0.6441 | 0.6406 | 0.6412 | 0.6470 | 0.6414 | 0.6429 |
| **pool_plate_drug** | 0.6514 | 0.6482 | 0.6485 | 0.6537 | 0.6488 | **0.6501** |
| pool_plate_drug − pool_plate | +0.0073 | +0.0076 | +0.0073 | +0.0067 | +0.0074 | +0.0072 |

- Adding drug similarity to the plate kernel raises it by +0.007 in 5/5 folds; the combined arm is the highest of all
  arms (0.6501 vs our MLP 0.6373, published MLP 0.6372, XPert 0.6315).

### 5.4 Plate-centred metric (v3 jobs 350615–350617 with saved predictions; `scripts/analyze_sdst_centred.py`)

For the 90% of test samples whose control row (= line and plate set) occurs in train (13,828–14,557 per fold),
m = mean train delta of that control group; centred score = Pearson over genes of (delta − m) vs (prediction − m), i.e.
how well the model predicts what distinguishes a compound from the training compounds on its own plate. The
plate-mean lookup scores 0 by construction. The other 10% (1,474–1,532 per fold) are scored with the plain metric.
v3 reproduces v1/v2 plain scores for every arm. Paired differences vs our MLP, bootstrap over test drugs (2,000).
Source: `/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/v3/{centred.json,fold_*/pred_test.npz}`;
log `logs/centred-*.out` (job after 350617).

| arm | centred, shared (5-fold mean) | Δ vs MLP, folds > 0 / CI > 0 | test drugs up | plain, unshared |
|---|---|---|---|---|
| plate-mean lookup | −0.0001 | | | |
| our MLP | 0.0989 | | | 0.5360 |
| pool_plate | 0.0997 | +0.0008, 4/5 / 2/5 | 0.58 | 0.5581 |
| krr_chem | 0.1249 | +0.0260, 5/5 / 5/5 | 0.58 | 0.5490 |
| pool_chem | 0.1309 | +0.0320, 5/5 / 5/5 | 0.60 | 0.5531 |
| krr_multi_norm | 0.1403 | +0.0414, 5/5 / 5/5 | 0.62 | 0.5606 |
| pool_multi | 0.1516 | +0.0527, 5/5 / 5/5 | 0.61 | 0.5699 |
| **pool_multi_norm** | **0.1537** | **+0.0548**, 5/5 / 5/5 | 0.62 | 0.5700 |
| **nnk_multi_norm** | **0.1541** | **+0.0552**, 5/5 / 5/5 | **0.69** | 0.5683 |
| pool_plate_drug | 0.1490 | +0.0501, 5/5 / 5/5 | 0.60 | **0.5832** |

- On the part of the response that distinguishes compounds sharing a plate, the multi-view drug kernel raises the
  correlation from 0.099 to 0.154 (+0.055, every fold, every interval above 0) and improves 62–69% of test drugs; the
  plate kernel adds nothing (+0.001). Multi-view − chemistry: +0.023 (pooled).
- On the 10% of samples without a training plate-mate, the drug arms add +0.034 (pool_multi_norm) to +0.047
  (pool_plate_drug) over the MLP's 0.536.
- The MLP's non-zero centred score (0.099) without drug information is read (agent) as shared noise in m (m is a
  mean over few training compounds per plate, and both the truth and the MLP's smoother deviate from it); the paired
  drug-arm differences are unaffected by this.

### 5.5 Backbone seed replicates (jobs 350673 seed 2718, 350674 seed 31415; `--tag v1_seed*`)

Same arms and folds as §5.2 with the backbone MLP retrained under two more seeds (v1 = seed 1729). Test PCC_DEG, mean
of five folds. Source: `/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/{v1/folds_[0-4].csv,
v1_seed2718/folds_0-1-2-3-4.csv, v1_seed31415/folds_0-1-2-3-4.csv}` (aggregated on the login node, 2026-10-29).

| arm | seed 1729 | seed 2718 | seed 31415 | sd over seeds |
|---|---|---|---|---|
| our MLP | 0.6373 | 0.6372 | 0.6372 | 0.0001 |
| krr_chem | 0.6417 | 0.6417 | 0.6417 | 0.0000 |
| pool_plate | 0.6429 | 0.6429 | 0.6430 | 0.0001 |
| pool_multi_norm | 0.6485 | 0.6485 | 0.6485 | 0.0000 |
| **nnk_multi_norm (DECANT)** | **0.6493** | **0.6494** | **0.6491** | 0.0002 |

- DECANT − our MLP: +0.0120 mean, minimum +0.0097, positive in **15/15** fold × seed runs; DECANT is above the
  published MLP's per-fold value (§5.2) in 15/15. pool_multi_norm +0.0112 (min +0.0088), pool_plate +0.0057
  (min +0.0044), krr_chem +0.0045 (min +0.0028), all 15/15.
- Seed variation (≤ 0.0002 in the five-fold mean) is two orders of magnitude below the DECANT − MLP gap. nnk_chem was
  run only under seed 1729.

### 5.6 Published drug-input ablation (Bai et al. 2026; values as released, no new run)

PCC_DEG fold means, contrast "Full − Zero" (drug input replaced by zeros). Source:
`repro/dpb/upstream/data/analysis/benchmark_metrics/statistics/primary_comparisons.csv` (drug-perturbation-benchmark
@ 55edbfe), `metric_key = pcc_deg_mean`. Plotted in paper Fig. 2a (`scripts/figures/fig2_benchmark.py`).

| model | with drug | drug zeroed | Full − Zero | folds > 0 |
|---|---|---|---|---|
| CIGER | 0.2456 | 0.2485 | −0.0028 | 1 |
| MultiDCP | 0.2545 | 0.2492 | +0.0052 | 5 |
| PRnet | 0.2771 | 0.2764 | +0.0007 | 2 |
| PertDiT | 0.3701 | 0.3825 | −0.0125 | 1 |
| DeepCE | 0.3947 | 0.3834 | +0.0113 | 5 |
| TranSiGen | 0.6144 | 0.6057 | +0.0088 | 5 |
| XPert | 0.6315 | 0.6313 | +0.0002 | 2 |

- |Full − Zero| ≤ 0.0125 for every model (the abstract's "at most 0.012"); two models are *better* without the drug
  (CIGER, PertDiT). Drug-free MLP 0.6372 (`full_vs_mlp.csv`).

### 5.7 What the compound kernel knows that chemistry does not (jobs 351009, 351012; `scripts/analyze_sdst_compounds.py`; no training)

Source: `/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/compound_analysis/{summary.json,hop_pairs_top.csv,
missed_twins.csv,drug_gain.csv}`; log `logs/sdst-cpd-351012.out` (351009 = same run without the binned-view block).

**Part A (descriptive, all compounds, no split).** Compound signature = mean plate-centred response (each sample minus
the mean of the *other* samples sharing its control row). 3,124 compounds with ≥ 3 samples and a target/MOA or Cell
Painting annotation; 4,878,126 pairs. Response similarity = Pearson of signatures. Signature split-half reliability
(compounds with ≥ 6 samples, n = 2,000): median **0.134**, so signatures are noisy and pair similarities are small in
absolute terms. Fused kernel = normalised multi-view kernel with v3 mean KTA weights (chem 0.290, tgt 0.246, moa 0.227,
morph 0.237).

| pair set | pairs | mean response similarity |
|---|---|---|
| all pairs | 4,878,126 | 0.0050 |
| chemically unlike (Tanimoto < 0.25) | 4,849,157 | 0.0047 |
| chemically unlike, top 0.5% fused (fused ≥ 0.244) | 24,246 | **0.0612** (− unlike: CI [+0.048, +0.066], compound bootstrap) |
| chemical analogues (Tanimoto ≥ 0.6) | 1,718 | 0.1650 |

Response similarity by chemistry × fused bin (rows Tanimoto, columns fused; n pairs in parentheses):

| Tanimoto \ fused | < 0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1 |
|---|---|---|---|---|---|
| 0–0.2 | 0.004 (4.73 M) | 0.058 (33,342) | 0.084 (4,695) | 0.088 (762) | |
| 0.2–0.4 | 0.014 (92,798) | 0.068 (6,227) | 0.098 (1,952) | 0.120 (1,333) | |
| 0.4–0.6 | 0.038 (1,014) | 0.071 (1,131) | 0.119 (725) | 0.122 (634) | 0.143 (94) |
| ≥ 0.6 | 0.086 (4) | 0.086 (357) | 0.123 (271) | 0.209 (541) | 0.195 (545) |

Each view on chemically unlike pairs (both compounds carry the view):

| view | similarity bin → mean response similarity (n pairs) |
|---|---|
| targets | 0: 0.004 (3.33 M); (0, 0.5): 0.030 (15,010); ≥ 0.5: **0.059** (11,803) |
| MOA | 0: 0.004 (3.33 M); (0, 0.5): 0.024 (4,599); ≥ 0.5: **0.051** (22,616) |
| Cell Painting | < 0.3: −0.003 (1.37 M); 0.3–0.5: 0.060 (229,470); 0.5–0.7: 0.121 (4,021); ≥ 0.7: **0.287** (99) |

- Views alone: Cell Painting ≥ 0.7 with no shared target or MOA 0.259 (86 pairs); MOA ≥ 0.5 where either compound lacks
  Cell Painting 0.051 (17,051); targets ≥ 0.5 where either lacks Cell Painting 0.064 (8,238).
- AUROC for the top-1% response-similar unlike pairs: chem 0.528, fused 0.587, targets 0.515, MOA 0.516 (both mostly
  ties at 0), Cell Painting 0.640.
- Among chemical analogues (Tanimoto ≥ 0.6, 1,718 pairs) the fused kernel tracks response better than chemistry
  (Spearman 0.181 vs 0.108).
- Examples (Tanimoto, fused, response r; `hop_pairs_top.csv`): panobinostat–vorinostat (0.19, 0.50, 0.94; HDAC);
  NVP-BEZ235–PI-103 (0.13, 0.54, 0.92; PI3K/mTOR); daunorubicin–mitoxantrone (0.11, 0.43, 0.92; topoisomerase);
  MG-132–ixazomib (0.21, 0.61, 0.90; proteasome); trametinib–selumetinib (0.16, 0.45, 0.88; MEK); anguidine–emetine
  (0.09, 0.33, 0.91; protein synthesis). Linked by Cell Painting alone (no annotation for the second compound):
  alvocidib–R-547 (0.11, morph 0.64, 0.94; both CDK inhibitors), BVD-523–TAK-733 (0.14, morph 0.77, 0.87; ERK and MEK
  inhibitors), triptolide–alvocidib (0.10, morph 0.67, 0.87; transcription inhibitors). The HDAC class dominates the
  top of the list; duplicate compound entries (e.g. two epirubicin pert_ids) appear twice.

**Part B (predictive, v3 test predictions, plate-centred score).** Per test compound (8,217 compound × fold rows):

| nearest training neighbour by chemistry | n | multi-view − chemistry kernel (pool) | DECANT − backbone |
|---|---|---|---|
| < 0.3 | 746 | **+0.023** | +0.034 |
| 0.3–0.5 | 2,349 | +0.016 | +0.031 |
| 0.5–0.7 | 2,083 | +0.006 | +0.034 |
| ≥ 0.7 | 3,039 | +0.009 | +0.061 |

| coverage of the test compound | n | multi-view − chemistry | DECANT − backbone |
|---|---|---|---|
| chemistry only | 4,693 | −0.002 | +0.033 |
| targets/MOA | 1,591 | +0.009 | +0.048 |
| Cell Painting | 578 | **+0.047** | +0.058 |
| both | 1,355 | +0.044 | +0.068 |

- The multi-view gain over chemistry is largest where the compound has no chemical analogue in training (+0.023 below
  Tanimoto 0.3 vs +0.006–0.009 above 0.5), and among those, larger when its fused neighbour is close (fused 0.5–0.7:
  +0.042, 144 compounds). Spearman of the gain with nn_fused − nn_chem: 0.061 (weak at the single-compound level).
- Compounds with chemistry only gain nothing from the extra views (−0.002), as they should; Cell Painting carries most
  of the per-compound gain, targets/MOA less (+0.009).
- Caveat: Part B's fused nearest-neighbour uses fold-averaged KTA weights; Part A is descriptive over all compounds and is
  not a prediction result.

### 5.8 Same-input control: compound features as inputs instead of a kernel (job 351007 `--input-mlp`, tag v4_input; centred 351065, 351066)

Features (per compound): Morgan-2 bits 2,048 + target multi-hot 870 + MOA multi-hot 639 + z-scored Cell Painting
1,026 + 3 availability flags. `mlp_input` = benchmark MLP (3 × 2,048) on [control 978, features], trained end to end
(dropout 0.1 / 0.3 chosen on valid). `nnf_multi` = the nnk head (2 layers; hidden 512 / 2048, dropout 0.3 / 0.5 chosen
on valid) on the raw features, on top of the fixed backbone — identical to nnk_multi_norm except that the compound
enters as raw features instead of Nyström kernel features. Same folds and backbone as §5.2 (every shared arm
reproduces v1 exactly). Source: `/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/v4_input/{folds_0-1-2-3-4.csv,
centred.json, centred_vs_nnf_multi.json}`; logs `logs/sdst-hybrid-351007.out`, `logs/centred-{351065,351066}.out`.

| arm | PCC_DEG (5-fold mean) | per fold | PCC_plate, shared | PCC_DEG, unshared |
|---|---|---|---|---|
| our MLP (drug-free) | 0.6373 | | 0.0989 | 0.5360 |
| **mlp_input** (end to end) | **0.6131** | 0.6123 / 0.6129 / 0.6115 / 0.6165 / 0.6124 | **0.1510** | **0.4960** |
| nnf_multi (raw-feature residual head) | 0.6480 | 0.6506 / 0.6459 / 0.6458 / 0.6501 / 0.6475 | 0.1509 | 0.5652 |
| pool_multi_norm (DECANT closed form) | 0.6485 | | 0.1537 | 0.5700 |
| nnk_multi_norm (DECANT-NN) | 0.6493 | 0.6519 / 0.6465 / 0.6477 / 0.6516 / 0.6486 | 0.1541 | 0.5683 |

| contrast (PCC_plate, paired, bootstrap over test drugs) | mean | folds > 0 | CI > 0 |
|---|---|---|---|
| mlp_input − our MLP | +0.0521 | 5 | 5 |
| nnf_multi − our MLP | +0.0520 | 5 | 5 |
| nnk_multi_norm − our MLP | +0.0552 | 5 | 5 |
| nnk_multi_norm − nnf_multi | +0.0031 | 4 | 0 |
| pool_multi_norm − nnf_multi | +0.0028 | 5 | 0 |
| mlp_input − nnf_multi | +0.0001 | 2 | 1 |
| krr_multi_norm − nnf_multi | −0.0107 | 0 | 0 |

- **PCC_DEG:** end to end, the same compound information makes the MLP *worse* than ignoring the drug (−0.024 in
  5/5 folds; −0.040 on samples without a plate-mate). As a residual head on the fixed backbone it is +0.011, and DECANT-NN
  is above it in 5/5 folds by only +0.0013 (0.0005–0.0019).
- **PCC_plate:** the end-to-end MLP, the raw-feature head and DECANT recover the compound-specific part about equally
  (+0.052, +0.052, +0.055 over the drug-free MLP); DECANT − raw-feature head +0.003, no fold's interval above 0.
- Reading (agent): on this benchmark the multi-view *information* carries the compound-specific gain whatever the
  regressor; the two-stage decomposition is what keeps the context accurate (the end-to-end model loses it), and kernel
  versus raw features adds ≤ 0.003. The kernel's advantage over a same-input network is so far shown only on DeepCE's
  benchmark (§4.6: 0.602 vs 0.544, single split, 2,935 signatures), which is ~27× smaller.
- Caveat: `mlp_input` stopped at epoch 479 / 490 / 499 of a 500-epoch cap in folds 1–3 (valid still improving), so it
  may be under-trained; its PCC_DEG deficit could shrink with a longer budget. nnf_multi stopped early at epochs 5–8.

### 5.9 View ablation on the drug-blind benchmark (job 351008, tag v4_views; centred 351101 vs MLP, 351102 vs full)

All arms normalised, same folds/backbone as §5.2 (pool_multi_norm reproduces v1: 0.6485 / 0.1537). only_<v> = one
view; minus_<v> = the other three; perm_<v> = all four with v's compound labels permuted among the compounds that have
v (coverage and kernel marginals kept; seed 0). Pooled (closed-form DECANT) arms; KRR (shared term only) arms in the
source. Source: `/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/v4_views/{folds_0-1-2-3-4.csv, centred.json,
centred_vs_pool_multi_norm.json}`; logs `logs/sdst-hybrid-351008.out`, `logs/centred-{351101,351102}.out`.

| arm (pooled) | PCC_DEG | PCC_plate | Δ PCC_plate vs full | folds < 0 / CI < 0 |
|---|---|---|---|---|
| drug-free MLP | 0.6373 | 0.0989 | −0.0548 | 5 / 5 |
| **full (chem + tgt + moa + morph)** | **0.6485** | **0.1537** | | |
| only chem | 0.6432 | 0.1309 | −0.0229 | 5 / 5 |
| only tgt | 0.6433 | 0.1241 | −0.0297 | 5 / 5 |
| only moa | 0.6436 | 0.1254 | −0.0283 | 5 / 5 |
| only morph | 0.6427 | 0.1295 | −0.0243 | 5 / 5 |
| minus chem | 0.6470 | 0.1434 | **−0.0103** | 5 / 5 |
| minus tgt | 0.6483 | 0.1542 | +0.0005 | 1 / 0 |
| minus moa | 0.6479 | 0.1527 | −0.0011 | 4 / 2 |
| minus morph | 0.6459 | 0.1399 | **−0.0138** | 5 / 5 |
| perm chem | 0.6448 | 0.1334 | **−0.0203** | 5 / 5 |
| perm tgt | 0.6479 | 0.1528 | −0.0009 | 4 / 2 |
| perm moa | 0.6475 | 0.1515 | −0.0023 | 5 / 4 |
| perm morph | 0.6458 | 0.1395 | **−0.0142** | 5 / 5 |

- Every single view beats the drug-free MLP on PCC_plate (+0.025 to +0.032, 5/5, CI > 0 in every fold) and recovers
  45–58% of the full gain; all four together are above every single view in 5/5 folds.
- Chemistry and Cell Painting carry information the other views do not: removing either costs 0.010 / 0.014 and
  permuting either costs 0.020 / 0.014 (all folds, all intervals below 0). Permuted chemistry is worse than no
  chemistry (−0.020 vs −0.010): a wrong similarity misleads the transfer.
- Targets and MOA are largely redundant with each other: removing or permuting one while the other stays costs
  ≤ 0.002 (intervals below 0 in 0–4 of 5 folds), although each alone is worth +0.025 / +0.027 over the MLP. Permuting
  both together was not run.

## 6. Context × compound headroom on the drug-blind benchmark (job 353166; `scripts/sanity_context_interaction.py`; no training)

Same data and folds as §5. Question: is there room for a model that couples the control/cell context with the compound
(as Rhaister-O's DMSO slopes or the course's DMSO-similarity line weights do), or for a better drug-free backbone?
Design and reading rule: `10_30_record.md` 2026-10-30 02:05. Line distance = 1 − Pearson of line-centred mean control
rows (the course kernel's distance; inputs only), median s = 1.0255 over 164 lines. Source:
`/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/context_sanity/{summary.json,part2_scores.csv,part2_contrasts.csv,
part3_backbones.csv}`; log `logs/ctx-sanity-353166.out`.

Structural fact: 78,453 samples = 77,966 (compound, line) pairs; only 458 pairs have a second sample. A compound ×
line interaction can therefore not be measured from within-line replicates.

### 6.1 Do compounds respond more alike in lines with similar controls? (all samples, descriptive)

Pearson of two plate-centred responses of the same compound in two different lines (748,293 pairs), by decile of the
two lines' control distance; null = second sample replaced by a different compound from the same line.

| distance decile (median) | 0.62 | 0.79 | 0.84 | 0.89 | 0.93 | 0.96 | 0.99 | 1.04 | 1.09 | 1.18 |
|---|---|---|---|---|---|---|---|---|---|---|
| same compound | **0.227** | 0.123 | 0.103 | 0.104 | 0.108 | 0.101 | 0.108 | 0.117 | 0.116 | 0.102 |
| different compound (null) | 0.009 | 0.004 | 0.003 | 0.004 | 0.003 | 0.003 | 0.004 | 0.003 | 0.003 | 0.002 |

- A compound's response in one line correlates ~0.10–0.12 with its response in another line, against ~0.003 for a
  different compound. Only the closest decile of line pairs is higher (0.227); deciles 2–10 are flat.
- Within compounds (5,135 with ≥ 10 pairs), Spearman(distance, response similarity) averages −0.065 (null +0.002);
  negative for 63% of compounds.

### 6.2 Oracle headroom (v3 test predictions; uses test responses — upper bounds, not models)

Each test sample is predicted by the backbone plus the backbone residuals of the *same test compound's* other samples
(other lines): uniform mean (perfect context-free compound representation), weighted by the course kernel
exp(−d/(h s)), k nearest lines, or per-gene leave-one-out ridge slope on the sample's own control (Rhaister-style;
compounds with ≥ 8 samples). h, k and the slope λ chosen on the neighbouring fold (chosen: h 0.5, k 10, λ 3 in every
fold). Shared-control test rows of compounds with ≥ 2 test samples (15,572 per fold on average); 5-fold means.

| arm | PCC_plate | PCC_DEG |
|---|---|---|
| drug-free MLP | 0.0981 | 0.6373 |
| UnDetect closed form (pool_multi_norm) | 0.1532 | 0.6485 |
| UnDetect-NN (nnk_multi_norm) | 0.1536 | 0.6493 |
| oracle, 1 nearest line | 0.1311 | 0.4775 |
| oracle, 10 nearest lines | 0.1964 | 0.6216 |
| oracle, uniform over other lines | **0.2002** | 0.6284 |
| oracle, course kernel (h 0.5) | **0.2044** | 0.6288 |

Compounds with ≥ 8 samples (12,081 rows per fold): MLP 0.0987, UnDetect-NN 0.1594, oracle uniform 0.2206, course
kernel 0.2256, Rhaister-style slope 0.2222.

| contrast, PCC_plate (bootstrap over compounds) | per fold | mean | folds > 0 / CI > 0 |
|---|---|---|---|
| course kernel − uniform oracle | 0.0042 / 0.0044 / 0.0041 / 0.0035 / 0.0044 | **+0.0041** | 5 / 5 |
| 10 nearest lines − uniform oracle | −0.0044 / −0.0036 / −0.0037 / −0.0041 / −0.0032 | −0.0038 | 0 / 0 (CI < 0 in 5) |
| slope − uniform oracle (n_d ≥ 8) | 0.0014 / 0.0017 / 0.0019 / 0.0015 / 0.0016 | +0.0016 | 5 / 5 |
| uniform oracle − UnDetect-NN | 0.0453 / 0.0606 / 0.0431 / 0.0385 / 0.0458 | **+0.0467** | 5 / 5 |
| UnDetect-NN − MLP | 0.0615 / 0.0580 / 0.0500 / 0.0492 / 0.0584 | +0.0554 | 5 / 5 |

- Context coupling: weighting the compound's other lines by control similarity adds +0.004 PCC_plate over the uniform
  mean in every fold (all intervals above 0); a drug-specific per-gene slope on the control adds +0.002. Restricting to
  the nearest lines loses (−0.004): averaging over more lines matters more than choosing similar ones.
- Compound representation: the uniform oracle is +0.047 above UnDetect-NN in every fold. UnDetect-NN covers 54% of the
  distance from the drug-free MLP (0.098) to the uniform oracle (0.200).
- On PCC_DEG the unshrunk oracles (residual weight 1) are *below* UnDetect (0.628 vs 0.649): adding a few noisy
  measured residuals at full weight costs more context accuracy than it gains.
- By number of other lines of the test compound (PCC_plate; MLP / UnDetect-NN / uniform oracle): 1 line (276 rows)
  0.096 / 0.150 / 0.131; 2–4 (1,358) 0.099 / 0.141 / 0.129; 5–9 (4,652) 0.095 / 0.128 / 0.149; ≥ 10 (7,798)
  0.100 / 0.171 / 0.246. For compounds measured in ≤ 4 lines, UnDetect is above the oracle built from the compound's own
  measurements.

### 6.3 Drug-free backbones (train rows, chosen on valid, test once)

| backbone | PCC_DEG | PCC_plate | per fold PCC_DEG |
|---|---|---|---|
| MLP (v3) | 0.6373 | 0.0989 | 0.6382 / 0.6352 / 0.6359 / 0.6416 / 0.6358 |
| full ridge, control → delta (λ 1e-3 × mean feature SS) | 0.6326 | 0.0960 | 0.6339 / 0.6311 / 0.6301 / 0.6366 / 0.6311 |
| diagonal ridge, gene g's control → gene g's delta (Rhaister-O form) | 0.3227 | 0.0793 | 0.3252 / 0.3240 / 0.3195 / 0.3224 / 0.3226 |
| context ceiling: leave-self-out mean of all samples on the control row | 0.6493 | — | rows with ≥ 2 samples (14,387/fold) |
| MLP on the same rows | 0.6464 | — | gap to ceiling 0.0014–0.0050 |

- A linear map from the control profile is within 0.005 of the MLP; the MLP is within 0.003 of the empirical plate
  mean. The per-gene (diagonal) form loses half the score: the control matters here as a plate identifier, not as a
  per-gene baseline. (The diagonal λ is at the grid's low edge, i.e. ≈ OLS.)

### 6.4 Line similarity learned from responses instead of controls (jobs 353168, 353173; `scripts/sanity_response_line_kernel.py`)

Per fold, from train rows only: S(l, l') = Σ over training compounds measured in both lines of Pearson(plate-centred
responses) / (shared compounds + 5). Pairs with ≥ 10 shared training compounds: 6,025–6,356 per fold, mean S
0.116–0.154; Spearman(S, −control distance) 0.311–0.320. Same oracle as §6.2 with distance 1 − S; h cross-fitted on
the neighbouring fold over {0.025, 0.05, 0.1, 0.25, 0.5, 1, 2} (job 353173; 353168 used 0.1–2 and chose its edge 0.1).
Chosen h = 0.05 in every fold (0.025 scores lower: interior optimum). The control-kernel arm reproduces §6.2 (0.2044).
Source: `context_sanity/response_kernel_{scores,contrasts}.csv`; logs `logs/ctx-sanity-{353168,353173}.out`.

| oracle line weighting | PCC_plate (5-fold mean) | − uniform, per fold | folds > 0 / CI > 0 |
|---|---|---|---|
| uniform | 0.2002 | | |
| control similarity (course kernel, h 0.5) | 0.2044 | 0.0042 / 0.0044 / 0.0041 / 0.0035 / 0.0044 | 5 / 5 |
| response similarity (h 0.05) | **0.2062** | 0.0055 / 0.0053 / 0.0056 / 0.0062 / 0.0072 | 5 / 5 |
| response − control similarity | | 0.0013 / 0.0008 / 0.0014 / 0.0028 / 0.0027 | 5 / 4 |

- A line similarity learned from other compounds' responses beats the control-profile similarity by +0.002 and the
  uniform mean by +0.006; both remain an order of magnitude below the compound-representation headroom (+0.047, §6.2).
