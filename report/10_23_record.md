# Therapeutic — Weekly Record, week ending Friday 2026-10-23

Reporting interval: [2026-10-16T18:00:00+04:00, 2026-10-23T18:00:00+04:00) — Asia/Dubai.

Plans, discussion, decisions, reasoning, and corrections (Chinese or English).

**2026-10-23 12:00 — 投稿方向讨论 / Which paper for RECOMB 2027 (discussion; user decision pending on framing).**
- 用户问：Rhaister-O + kernel + reliability（Pearson delta 约 +5%）能否单独投 RECOMB；还是投 multi-layer (MKL)
  模型或 end-to-end 模型；11-04 之前投什么。
- My reading (agent): not as a standalone paper. The +0.0267 is on 25 fresh Tahoe lines (`10_02_report.md` §3.7,
  drug-residual 0.5427 → ≈ +4.9%), but it is +0.0080 on EmeraldBay and fails the pre-registered two-dataset gate;
  the reliability part is mostly the NCI-H596 data-quality correction (+0.017 with it in the reference set vs +0.007
  without); the transferable locality part is +0.006–0.008; and the base model is not ours. MKL tiers are better
  as the constructive part of the audit paper (T2 gain mostly survives its null; big gains come from Cell Painting
  and other-line response). E2e is not a headline (see 12:15 below).
- Recommendation: submit "Degree, not wiring" (`draft/recomb2027/`), with the MKL information tiers as the
  constructive part and E8 (graph-free / targets + degree comparator) answering the "RECOMB wants a model" concern
  (record 2026-10-14 06:10). PDGrapher (E6) likely dropped for time and GPU quota; three published-model audits
  (DeepCE, GSNN, GEARS) remain. Tahoe kernel/reliability/F1 findings (genotype and tissue similarity do not help)
  can be an appendix on the unseen-line task.
- User (12:10): "先写报告 然后你可以负责提交" — write up the finished 10-14 jobs, then the agent submits the follow-up.

**2026-10-23 12:15 — Analysis of jobs finished 2026-10-14 (`10_23_report.md` §1–§3); one bug fix.**
- ONGOING.md had not been updated since 2026-10-14: GSNN ladders B+C (311593, 311595), e2e round 2 (311592,
  311594), MKL `reg` (311597, 311598) had all finished that day; compare 311596 had **failed**:
  `AssertionError: arm names collide with _krr_rw0`. Cause: both the real-KG and the rewired-KG run write the
  deterministic KRR `anchor` arm. Checked: identical in both runs (max |diff| 0, 11,676 / 31,229 rows).
  Fix in `scripts/compare_l1000_e2e.py` (sha256 274a0d43… → b9c34c95…): an arm present in both runs is kept once
  if identical, otherwise the script still stops. Re-run as job 338326 (cn-15, 40 s, completed).
- e2e round 2 against the pre-declared gates (`10_16_record.md` 2026-10-11 07:10): G0 pass, G1 pass (+0.005 on both
  slices), **G2 fails for both kg_a and kg_t**. Per the pre-declared branch "nothing passes": for unseen drugs on
  L1000 the KG adds nothing beyond chemistry; stop the KG-encoder line. The anchored network itself is now a small
  positive (+0.005 over KRR) instead of round 1's −0.012.
- GSNN: all four degree-preserving rewired runs beat both real-graph runs; ER matches real; the loss in GSNN's own
  random control is dominated by the input→function (target identity) edge set. Read with the GEARS empty-graph and
  DeepCE results, this is the same pattern three times. Still one fold.
- MKL `reg`: directed signalling/TF kernel beats its rewired null alone (+0.003–0.004, 7/7) but adds nothing inside
  the multi-kernel model. Consistent with E1 (wiring only carries signal in directed networks, and little).

**2026-10-23 12:30 — E8 part 1 plan: graph-free NN on GSNN's own benchmark (pre-registered before submission).**
- Why: GSNN's `ff_empty` arm (0.142) is not a "no prior" control, because GSNN routes inputs to outputs only through
  function→function edges. The fair "no graph" comparator is a model on the same inputs with no graph at all.
  GSNN ships one: `gsnn-lib/scripts/training/train_nn_lincs.py` (MLP on the same input-node features: drug, dose,
  cell omics; 527 outputs).
- Protocol: same processed data `repro/gsnn/proc/lincs`, fold 0 (`partitions_f0`), upstream default NN
  hyper-parameters (channels 124, layers 2, lr 1e-3, early stopping on r2, patience 25), 100 epochs, 3 independent
  re-trainings, CPU (`--ignore_cuda`; GPU QOS is full with hta11). Script `repro/gsnn/slurm/train_nn.sbatch`.
  Deviation: the preprint tuned its baselines by search; we use defaults for both GSNN and NN.
- Pre-declared reading, on test r_flat (and mean_r as secondary), compared to the mean of the two real-graph GSNN runs
  (0.3135 / 0.2630): NN ≥ 0.3035 (within 0.01) → on GSNN's own benchmark a graph-free model with the same inputs is
  as good as GSNN; NN < 0.3035 → GSNN's sparse architecture helps, but (by §1) not through which edges it uses —
  then the E8 framing is "architecture/target routing, not wiring". Either result goes into the paper.
- Part 2 (L1000 unseen-drug): no new run needed — krr_chem_target already beats kg_t on ∩known (§2) and MKL T2 is
  the targets + degree model; to be tabulated in the draft.
- (12:21) Submission log. 338351 failed at once: upstream `train_nn_lincs.py` predates the current `Trainer`
  interface (`optim_params` missing). Patched to build the optimizer exactly as `train_gsnn_lincs.py` does
  (optimizer class + `{'lr', 'weight_decay'}`, scheduler None; `--sched` other than `none` now refused), committed
  on gsnn-lib branch `repro` as `bbaa352`. Smoke 338362 (1 epoch, `results/_smoke_nn_e8`, not to be quoted) ran
  end to end, ~20 s per epoch on CPU. Production **338366** submitted 12:21 (3 re-trainings × 100 epochs), running
  on cn-19.
- Follow-up: 338366 finished 2026-10-23 14:40 and was analysed 2026-10-24 in the next week's pair,
  `10_30_report.md` §1 / `10_30_record.md` 2026-10-24 20:05.
