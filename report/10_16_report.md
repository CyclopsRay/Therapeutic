# Therapeutic — Weekly Report, week ending Friday 2026-10-16

Reporting interval: [2026-10-09T18:00:00+04:00, 2026-10-16T18:00:00+04:00) — Asia/Dubai.

Measured results only, with provenance; include positive and negative findings.

The cluster was down from 2026-10-09 (DC cooling shutdown) to at least 2026-10-11; `/l` was back by 2026-10-14.
Every job submitted during the outage (310545-310551) is gone from the queue with no log and no output (§0). The two
GPU ladders below finished on 2026-10-07, before the outage, but had been listed as "status unknown"; they were
analysed on 2026-10-14. Latest integrated-model numbers remain `10_09_report.md` §12.

## 0. Job outcomes after the outage (checked 2026-10-14 05:20 +04)

`sacct` is still refused (slurmdbd at localhost:6819); outcomes are read from logs and output directories.

| job | what | outcome |
|---|---|---|
| 299345 | E4 GSNN ladder A (none, ff_rewired0, ff_labels0, gsnn_random0), fold 0 | **completed** 2026-10-07 08:37; §1 |
| 297743 | E5 GEARS Norman GO ladder, 11 runs, 20 epochs | **completed** 2026-10-07 08:40; §2 |
| 310545 / 310546 / 310547 | e2e round 2 P2-24 / P1-24 / compare | **never ran**: not in `squeue`, no log, no new dir in `results/l1000_e2e/` |
| 310548 / 310549 | GSNN ladder A (duplicate of 299345) / B+C | **never ran**; 310548 no longer needed |
| 310550 / 310551 | MKL `reg` view, platemap / ik14 | **never ran** |

No file under `/nfs-stor/yuqi.lei/Therapeutic` has been modified since 2026-10-09 02:00.

## 1. E4 GSNN graph-control ladder, fold 0 (job 299345)

LINCS exp1 settings (116 lines, 413 drugs, 527 outputs, cell-drug fold 0), default hyper-parameters, 100 epochs,
patience 10, one model seed. Metrics as logged by GSNN at the best-validation epoch. Source:
`/nfs-stor/yuqi.lei/Therapeutic/repro/gsnn/logs/train-299345.out` (lines 216-225, 442-451, 665-674, 963-972);
`repro/gsnn/results/{none,ladder_*}/<uuid>/fold_0_lincs_splits/result_metric_dict.pt`. The other `none` directory
(33d224fa..., 2026-10-05) is the 2-epoch smoke 298920 and is excluded.

| control | what is perturbed | r_flat test | mean_r test | r2 test | mse test | r_flat val | best epoch |
|---|---|---|---|---|---|---|---|
| none (real graph) | nothing | 0.3170 | 0.2667 | 0.0835 | 0.8760 | 0.3116 | 63 |
| ff_rewired0 | function->function edges, degree-preserving rewire | **0.3375** | **0.2915** | **0.0970** | **0.8631** | 0.3290 | 64 |
| ff_labels0 | function-node labels permuted on the same wiring | 0.2887 | 0.2373 | 0.0661 | 0.8927 | 0.2825 | 63 |
| gsnn_random0 | GSNN's own control: all three edge sets random | 0.2733 | 0.2205 | 0.0552 | 0.9031 | 0.2719 | 88 |

- Real vs GSNN's all-random control: +0.044 r_flat (+16% relative), +0.046 mean_r (+21%). Same direction as the
  published +24%.
- The degree-preserving rewired graph scores **above** the real graph on every test and validation metric
  (+0.020 r_flat, +0.025 mean_r).
- Permuting node labels on the real wiring costs -0.028 r_flat.
- n = 1 fold x 1 seed per arm; no seed variance yet. Ladder B+C (rewired1-2, labels1-2, er0, empty, in_shuffle0,
  out_shuffle0) has not run (310549 lost).

## 2. E5 GEARS Norman ladder, GO graph replaced (job 297743)

Norman 2019, simulation split, data seed 1, 20 epochs. Only the GO gene-gene graph is replaced; the co-expression
graph is left real. Real arm: 3 model seeds (1-3); every null arm: model seed 1 and the null seed in its name.
`pearson_delta` is GEARS' subgroup Pearson on the expression change; n perturbations per subgroup: combo_seen0 9,
combo_seen1 43, combo_seen2 19, unseen_single 36. Source: `/nfs-stor/yuqi.lei/Therapeutic/repro/gears/results/norman/
go_*/metrics.json`; log `repro/gears/logs/ladder-297743.out`.

| arm | mse_de | pearson_de | pd combo_seen0 | pd combo_seen1 | pd combo_seen2 | pd unseen_single | opp-dir top20, unseen_single |
|---|---|---|---|---|---|---|---|
| real, seed 1 | 0.2189 | 0.8485 | 0.639 | 0.545 | 0.557 | 0.453 | 0.226 |
| real, seed 2 | 0.2498 | 0.8915 | 0.420 | 0.594 | 0.699 | 0.406 | 0.246 |
| real, seed 3 | 0.2028 | 0.8824 | 0.457 | 0.583 | 0.674 | 0.406 | 0.241 |
| rewired 0 | 0.2495 | 0.8324 | 0.562 | 0.506 | 0.555 | 0.432 | 0.265 |
| rewired 1 | 0.2174 | 0.8581 | 0.579 | 0.586 | 0.646 | 0.409 | 0.240 |
| rewired 2 | 0.2248 | 0.8552 | 0.585 | 0.560 | 0.647 | 0.453 | 0.228 |
| labels 0 | 0.2194 | 0.8525 | 0.541 | 0.544 | 0.587 | 0.433 | 0.214 |
| labels 1 | 0.2292 | 0.8529 | 0.501 | 0.540 | 0.631 | 0.402 | 0.232 |
| labels 2 | 0.2356 | 0.8461 | 0.511 | 0.529 | 0.608 | 0.354 | 0.243 |
| Erdos-Renyi 0 | 0.2702 | 0.8358 | 0.501 | 0.462 | 0.534 | 0.324 | 0.262 |
| empty GO graph | 0.2218 | 0.8604 | 0.554 | 0.591 | 0.667 | 0.428 | 0.215 |

Means over the three runs per arm (pearson_delta): real 0.505 / 0.574 / 0.643 / 0.422; rewired 0.575 / 0.551 /
0.616 / 0.431; labels 0.518 / 0.538 / 0.609 / 0.396 (combo_seen0 / combo_seen1 / combo_seen2 / unseen_single).

- **Empty GO graph is within the real arm's seed range on every subgroup** (unseen_single 0.428 vs real 0.406-0.453;
  combo_seen2 0.667 vs 0.557-0.699) and has lower mse_de than two of three real seeds.
- Degree-preserving rewiring matches real on average (unseen_single 0.431 vs 0.422).
- The Erdos-Renyi graph is the only arm clearly below real (unseen_single 0.324, combo_seen1 0.462, highest mse_de 0.270).
- Real-arm seed spread is large: combo_seen0 0.420-0.639 (n = 9 perturbations).

## 3. L1000 unseen-drug leaderboard: every model on identical rows (2026-10-14; no new training)

`scripts/build_l1000_leaderboard.py` re-scores existing per-condition outputs (runner, GNN/MLP, GCN, MKL, end-to-end
skeleton) on identical (line, compound) rows. Each arm is shown only in the cohort its inputs need (targets → KNOWN,
Cell Painting → MORPH, other-line response → XCELL). Line-macro full Pearson on ACTIVE rows; Δ vs `krr_morgan2` on the
same rows with a 2,000-replicate bootstrap over the split's groups; headroom = (arm − mean) / (oracle − mean).
Source: `/nfs-stor/yuqi.lei/Therapeutic/results/l1000_leaderboard/v1/leaderboard.csv` (293 rows). Interactive:
https://claude.ai/artifact/74ath3hh5LkPM2cqd32Rwu (section "Benchmark leaderboard").

| cohort (ik14 split) | rows P1 / P2 | mean | KRR Morgan2 | best arm (Δ vs KRR [95%], lines+) | oracle | KRR / best headroom |
|---|---|---|---|---|---|---|
| P1-24 ACTIVE | 12,279 | 0.280 | 0.332 | no arm beats KRR; e2e skeleton v1reg 0.320 (−0.012 [−0.014, −0.009], 0/7) | 0.689 | 13% / – |
| P1-24 KNOWN | 4,714 | 0.290 | 0.341 | krr_chem_target 0.358 (+0.017 [+0.012, +0.021], 7/7) | 0.696 | 13% / 17% |
| P1-24 MORPH | 1,470 | 0.296 | 0.348 | knn_chem_morph 0.378 (+0.030 [+0.020, +0.040], 7/7) | 0.702 | 13% / 20% |
| P1-24 XCELL | 11,956 | 0.280 | 0.332 | xcell_knn 0.470 (+0.138 [+0.132, +0.143], 7/7) | 0.689 | 13% / 46% |
| P2-24 ACTIVE | 5,477 | 0.329 | 0.362 | knn_morgan2 0.364 (+0.002, ns); e2e skeleton 0.351 (−0.011) | 0.667 | 10% / 10% |
| P2-24 KNOWN | 3,683 | 0.331 | 0.367 | mkl_T2 0.384 (+0.017 [+0.012, +0.022], 7/7); its null mkl_T2null +0.013 | 0.669 | 11% / 16% |
| P2-24 MORPH | 4,820 | 0.333 | 0.364 | knn_chem_morph 0.462 (+0.098 [+0.091, +0.105], 7/7) | 0.666 | 9% / 39% |
| P2-24 XCELL | 5,477 | 0.329 | 0.362 | mkl_T4 0.576 (+0.214 [+0.207, +0.220], 7/7) | 0.667 | 10% / 73% |

- Learned graph models sit at or below the mean on KNOWN rows: per-drug GCN real / rewired / no graph 0.297 / 0.297 / 0.298
  (P1-24), 0.331 / 0.331 / 0.331 (P2-24); decoupled GNN 0.332 / 0.332 / 0.332 (P2-24). Best learned network model is the
  MLP with PPI features, 0.373 (P2-24 KNOWN), equal to its rewired version 0.374.
- PPI wiring: knn_ppi − knn_ppi_rewired +0.002 (P1-24 KNOWN), +0.003 (P2-24); krr_ppi − rewired +0.001 / +0.001.

**Coverage of a measured causal gene→gene source (L1000 genetic perturbations, GSE92742 `sig_info`).** 4,345 genes have
consensus shRNA signatures (`trt_sh.cgs`, 36,720 signatures) and 3,114 have over-expression signatures (`trt_oe`).
ACTIVE rows whose compound has ≥ 1 annotated target with a `trt_sh.cgs` signature in the **same** line: P1-24 2,974 of
4,776 target-annotated rows (62%; 24% of all 12,279 ACTIVE rows); P2-24 2,107 of 3,814 (55%; in any line 76%). `trt_oe`:
44% / 38% same line. No P2-24 signatures exist for YAPC or HELA.

## 4. Model-free knockdown gate: drug response vs its targets' L1000 knockdown signatures (jobs 311590, 311591)

311590 extracted 36,720 `trt_sh.cgs` and 22,205 `trt_oe` GSE92742 Level-5 signatures × 978 landmarks (115 s;
`derived/l1000_unseen_drug/GSE92742_{trt_sh.cgs,trt_oe}_landmarks.npz` with manifests). 311591 ran
`scripts/analyze_kd_target_concordance.py` (sha256 9d436554…), gate fixed in its docstring before the run.
ACTIVE rows with ≥ 1 annotated target knocked down in the same line; response = residual from the line mean; prediction =
mean knockdown signature of the targets; per-row Pearson over 978 landmarks, line-macro; 200 draws per null.
Source: `/nfs-stor/yuqi.lei/Therapeutic/results/kd_target_concordance/{P1-24,P2-24}/summary.json`.

| cohort | rows (compounds) | real | null A permuted targets: real − mean, lines > p95 | null B random knockdown genes: real − mean, lines > p95 | gate |
|---|---|---|---|---|---|
| P1-24 non-activator (primary) | 2,353 (816) | 0.0403 | +0.0134, 3/7 | +0.0186, 3/7 | **fail** |
| P2-24 non-activator (primary) | 1,786 (508) | 0.0465 | +0.0229, 4/5 | +0.0275, 5/5 | **fail** (null A: 4 of 5 lines) |
| P1-24 activator, sign flipped (secondary) | 607 (240) | 0.0172 | +0.0203, 3/7 | +0.0190, 3/7 | – |
| P2-24 activator, sign flipped (secondary) | 269 | 0.0286 | +0.0183 | – | – |

- The real target knockdown beats both nulls in every slice and cohort: none of the 200 draws in any null reached the real
  line-macro value (empirical p < 0.005). The effect is small, though: per-row Pearson about 0.04 in absolute terms,
  +0.013 to +0.028 over the nulls. Lines above the null's 95th percentile: 3/7 on P1-24, 4–5/5 on P2-24.
- The pre-registered gate (≥ +0.02 and ≥ 5 lines, for both nulls) is **not met** on P1-24 and misses on P2-24 null A by
  one line. P2-24 has only 5 lines with knockdowns (no YAPC/HELA), so a 5-line requirement there means all lines; the
  gate was written with 7-line slices in mind. Recorded as is.
- Plate-overlap check uninformative: consensus (`cgs`) signature IDs do not carry plate names (25 "plates" = consensus
  batches, 0 shared with compound plates).
