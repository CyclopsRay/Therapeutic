# Therapeutic — Weekly Report, week ending Friday 2026-10-23

Reporting interval: [2026-10-16T18:00:00+04:00, 2026-10-23T18:00:00+04:00) — Asia/Dubai.

Measured results only, with provenance; include positive and negative findings.

Index: §1 GSNN graph-control ladder, complete fold-0 set (E4); §2 integrated end-to-end model, round 2
(KRR anchor, rewired-KG nulls); §3 MKL `reg` view (directed signalling + TF network) and its null.
All three come from jobs that finished on 2026-10-14 and were analysed on 2026-10-23; the e2e comparison was re-run
(job 338326) after a bug fix (`10_23_record.md`, 2026-10-23 12:15).

## 1. E4 GSNN graph-control ladder, fold 0, ladders A + B + C (jobs 299345, 311593, 311595)

LINCS exp1 settings (116 lines, 413 drugs, 527 outputs, cell-drug fold 0), GSNN default hyper-parameters,
100 epochs, metrics at the best-validation epoch as logged by GSNN. GSNN's trainer sets no seed, so `@rep1` arms are
independent re-trainings of the same graph. Ladder definitions: `repro/gsnn/ladder.py` (one edge set perturbed at a
time). Ladder A numbers are repeated from `10_16_report.md` §1.

Source: `/nfs-stor/yuqi.lei/Therapeutic/repro/gsnn/logs/train-299345.out` (ladder A),
`train-311593.out` lines 166, 410, 621, 886 (rewired1, rewired2, labels1, labels2),
`train-311595.out` lines 250, 383, 612, 781, 994, 1244 (er0, empty, in_shuffle0, out_shuffle0, none@rep1,
ff_rewired0@rep1); per-run `result_metric_dict.pt` under `repro/gsnn/results/<arm>/<uuid>/fold_0_lincs_splits/`.

| arm | edge set perturbed | runs | r_flat test | mean_r test | r2 test | mse test | best epoch |
|---|---|---|---|---|---|---|---|
| **real graph** | none | 2 | **0.3170, 0.3099** | 0.2667, 0.2593 | 0.0835, 0.0789 | 0.8760, 0.8805 | 63, 60 |
| degree-preserving rewire | function→function | 4 (seeds 0, 0@rep1, 1, 2) | **0.3375, 0.3393, 0.3268, 0.3378** | 0.2915, 0.2932, 0.2803, 0.2914 | 0.0970, 0.0973, 0.0897, 0.0972 | 0.8631, 0.8628, 0.8701, 0.8630 | 64, 72, 47, 70 |
| same-edge-count random (Erdős–Rényi) | function→function | 1 | 0.3192 | 0.2725 | 0.0840 | 0.8756 | 75 |
| node labels permuted | function→function | 3 (seeds 0-2) | 0.2887, 0.3030, 0.3119 | 0.2373, 0.2507, 0.2621 | 0.0661, 0.0737, 0.0802 | 0.8927, 0.8854, 0.8792 | 63, 59, 77 |
| output read-out permuted | function→output | 1 | 0.2958 | 0.2431 | 0.0702 | 0.8888 | 45 |
| GSNN's own control (all three random) | all | 1 | 0.2733 | 0.2205 | 0.0552 | 0.9031 | 88 |
| drug/omic target sets permuted | input→function | 1 | 0.2217 | 0.1580 | 0.0311 | 0.9262 | 65 |
| no function→function edges | function→function | 1 | 0.1422 | 0.0349 | 0.0016 | 0.9544 | 15 |

Means over runs (r_flat / mean_r): real 0.3135 / 0.2630; rewired 0.3354 / 0.2891; labels 0.3012 / 0.2500.

- **Every degree-preserving rewired run scores above both real-graph runs** on r_flat (lowest rewired 0.3268 vs highest
  real 0.3170) and on mean_r, r2 and mse. Mean difference rewired − real: +0.022 r_flat, +0.026 mean_r.
- **A uniformly random function graph with the same number of edges matches the real graph** (0.3192 vs 0.3099–0.3170).
- Real − GSNN's own random control: +0.040 r_flat (mean of real runs), reproducing the published direction.
- The largest single loss comes from breaking drug/omic → target edges (in_shuffle: −0.092 r_flat vs mean real),
  i.e. target identity; next, permuted output read-outs (−0.018) and permuted node labels (−0.012).
- Removing all function→function edges collapses the model (0.142, stopped at epoch 15). In GSNN signal reaches the
  outputs only through these edges, so this arm removes the model's path from inputs to outputs; it is not a
  "no prior" control. A graph-free model on the same data has not been run (E8; record 2026-10-23).
- Limits: one fold; two real-graph runs and one run each for ER, in/out-shuffle, empty and GSNN-random.

## 2. Integrated end-to-end model, round 2: KRR anchor and rewired-KG nulls (jobs 311592, 311594; compare 338326)

`scripts/run_l1000_e2e.py` (sha256 d0e4b409…) with `--anchor krr`; null runs `--rewire 0` (per-relation
degree-preserving double-edge swap of the PrimeKG subgraph). Arms: `anchor` = the stacked KRR-Morgan2 prediction the
network adds to; `skel` = anchored network on chemistry only; `kg_a` = + PrimeKG drug–disease edges with disease text
embeddings; `kg_t` = `kg_a` + drug–protein (target) edges; `_rw0` = same arm on the rewired KG. ik14 folds, one seed.
Line-macro full Pearson; differences with 2,000 bootstrap replicates over ik14 groups, lines positive out of 7.
Gates G0–G2 were pre-declared in `10_16_record.md` (2026-10-11 07:10).
Source: `/nfs-stor/yuqi.lei/Therapeutic/results/l1000_e2e/{P2-24,P1-24}_ik14_seed0_krr/{contrasts.csv,summary.json}`
(merged with `_krr_rw0`); log `logs/l1000-e2e-cmp-338326.out`; `compare_l1000_e2e.py` sha256 b9c34c95….

Absolute line-macro full Pearson:

| arm | P2-24 ACTIVE | P2-24 ∩in_kg | P2-24 ∩known | P1-24 ACTIVE | P1-24 ∩in_kg | P1-24 ∩known |
|---|---|---|---|---|---|---|
| mean | 0.3292 | 0.3225 | 0.3309 | 0.2800 | 0.2830 | 0.2895 |
| krr_morgan2 (= anchor) | 0.3621 | 0.3576 | 0.3676 | 0.3320 | 0.3538 | 0.3410 |
| knn_morgan2 | 0.3641 | 0.3617 | 0.3707 | 0.3304 | 0.3544 | 0.3391 |
| krr_chem_target | 0.3664 | 0.3593 | **0.3788** | 0.3130 | **0.3686** | **0.3571** |
| skel | 0.3672 | 0.3624 | 0.3732 | 0.3369 | 0.3626 | 0.3463 |
| kg_a | **0.3677** | 0.3627 | 0.3743 | **0.3375** | 0.3625 | 0.3474 |
| kg_a_rw0 | 0.3667 | 0.3613 | 0.3734 | 0.3364 | 0.3623 | 0.3455 |
| kg_t | 0.3671 | 0.3624 | 0.3737 | 0.3374 | 0.3645 | 0.3462 |
| kg_t_rw0 | 0.3674 | 0.3627 | 0.3739 | 0.3369 | 0.3623 | 0.3465 |

(krr_chem_target averages Morgan2 and target-set similarity and is a KNOWN-only arm in the runner
(`run_l1000_unseen_drug.py` `KNOWN_ONLY`); read it on ∩known only.)

Gate contrasts:

| gate | contrast [cohort] | P2-24 | P1-24 | verdict |
|---|---|---|---|---|
| G0 | anchor − krr_morgan2 [ACTIVE] | 0.0000 | 0.0000 | pass (identical) |
| G1 | skel − krr_morgan2 [ACTIVE] | +0.0051 [+0.0036, +0.0068] 7/7 | +0.0048 [+0.0034, +0.0063] 6/7 | **pass** |
| G2 | kg_a − kg_a_rw0 [∩in_kg] | +0.0015 [+0.0003, +0.0027] 5/7 | +0.0002 [−0.0022, +0.0025] 3/7 | **fail** (P1-24) |
| G2 | kg_t − kg_t_rw0 [∩known] | −0.0002 [−0.0009, +0.0006] 3/7 | −0.0003 [−0.0017, +0.0012] 3/7 | **fail** |
| – | kg_a − skel [∩in_kg] | +0.0003 [−0.0005, +0.0012] 4/7 | −0.0002 [−0.0028, +0.0022] 5/7 | – |
| – | kg_t − skel [∩known] | +0.0005 [−0.0003, +0.0014] 7/7 | −0.0000 [−0.0017, +0.0017] 4/7 | – |
| – | kg_t − krr_chem_target [∩known] | −0.0051 [−0.0095, −0.0009] 1/7 | −0.0109 [−0.0155, −0.0062] 0/7 | – |

- The KRR anchor removes round 1's deficit: the anchored chemistry-only network is now **+0.005 above KRR-Morgan2** on
  both slices (round 1: −0.012, `10_09_report.md` §12).
- **Neither knowledge-graph arm beats its rewired-KG null** on its own cohort in both slices, and neither beats the
  chemistry-only network. kg_a − kg_a_rw0 is positive on P2-24 (+0.0015) and absent on P1-24.
- Adding target edges through the KG (kg_t) is **below a plain KRR on chemistry + target identity** on target-annotated
  rows: −0.005 (P2-24), −0.011 (P1-24).

## 3. MKL `reg` view: directed signalling + TF network (jobs 311597 platemap, 311598 ik14)

`scripts/run_l1000_mkl.py` (sha256 0199b79e…), P2-24, `--tag _reg`. `reg` = cosine between compounds' downstream
influence over OmniPath signalling + CollecTRI TF→target (10,168 genes, 132,324 directed edges; 1,160 compounds with a
target in it); `reg_null` = same on a degree-preserving rewired copy. Tiers as `10_09_report.md` §7, with `reg`
added to T2 / T3. Full Pearson, ACTIVE rows of the cohort in brackets, line-macro difference [95% bootstrap over the
split's groups, 2,000 replicates] lines positive. Source:
`/nfs-stor/yuqi.lei/Therapeutic/results/l1000_mkl/P2-24_{platemap,ik14}_seed0_reg/per_condition.parquet`; logs
`logs/l1000-mkl-{311597,311598}.out`; contrasts computed with `run_l1000_unseen_drug.line_macro_boot`.

| contrast [cohort] | plate-map split | ik14 split |
|---|---|---|
| reg kernel − mean [KNOWN] | +0.032 [+0.024, +0.041] 7/7 | +0.041 [+0.034, +0.049] 7/7 |
| reg kernel − reg on rewired network [KNOWN] | +0.0034 [+0.0010, +0.0059] 7/7 | +0.0043 [+0.0021, +0.0065] 7/7 |
| one-hop STRING kernel − rewired [KNOWN] (same run) | +0.0000 [−0.0027, +0.0025] 5/7 | +0.0012 [−0.0005, +0.0029] 6/7 |
| T2 + reg − T2 [KNOWN] | +0.0000 [−0.0004, +0.0005] 3/7 | +0.0002 [−0.0003, +0.0007] 4/7 |
| T2 + reg − T2 + reg_null [KNOWN] | +0.0002 [−0.0003, +0.0007] 4/7 | +0.0004 [−0.0000, +0.0008] 6/7 |
| T3 + reg − T3 [MORPH] | −0.0015 [−0.0025, −0.0004] 1/7 | −0.0001 [−0.0003, +0.0001] 2/7 |
| T3 + reg − T3 + reg_null [MORPH] | +0.0011 [+0.0006, +0.0016] 7/7 | +0.0005 [+0.0003, +0.0007] 7/7 |

Absolute (all test rows, from the logs): T2 0.3483 / T2reg 0.3480 (plate-map), 0.3709 / 0.3707 (ik14).

- On its own, the directed signalling/TF kernel beats its rewired copy by a small, consistent margin (+0.003 to +0.004,
  7/7 lines), unlike undirected one-hop STRING (≈ 0). This matches the model-free result that wiring carries signal
  only in directed signalling / TF networks (`10_09_report.md` §6).
- **Inside the multi-kernel model it adds nothing**: T2 + reg equals T2 (≤ +0.0004), and the margin over its null is
  ≤ +0.0011 at T2/T3. The information it carries is already in target identity and the other T2 views.
