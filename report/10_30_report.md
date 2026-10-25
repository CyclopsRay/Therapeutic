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
