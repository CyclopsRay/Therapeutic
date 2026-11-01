# Therapeutic — Weekly Record, week ending Friday 2026-10-30

Reporting interval: [2026-10-23T18:00:00+04:00, 2026-10-30T18:00:00+04:00) — Asia/Dubai.

Plans, discussion, decisions, reasoning, and corrections (Chinese or English).

**2026-10-24 20:05 — E8 part 1 read against its pre-declared rule (`10_30_report.md` §1).**
- 结果：无图 NN 三次 r_flat 均值 0.3034，阈值 0.30346（GSNN 真实图均值 0.3135 − 0.01），差 0.00008，按规则落入
  "NN < GSNN" 分支。记录原样，不改判。mean_r（次要指标）NN 0.2635 = GSNN 0.2630。
- Reading (agent): this is a tie at the boundary, not a separation. The NN-vs-real gap (−0.010) is the size of the
  replicate spread, and the NN was still improving at epoch 100. What is clear: (1) GSNN's real wiring is not
  better than either a rewired wiring or no graph at all by any margin larger than run noise; (2) GSNN on a rewired
  graph is the best model in the table (+0.032 over NN), so GSNN's sparse, target-routed architecture with a
  degree-matched graph does carry something beyond a plain MLP. Paper framing per the pre-declared second branch:
  "architecture / target routing, not wiring".
- Possible follow-up, not run (needs the user): NN with 200 epochs (removes the epoch-cap limitation; ~1.5 h CPU per
  run). Would only sharpen the NN-vs-real comparison; the rewired > real finding does not depend on it.

**2026-10-26 — RECOMB abstract updated for the 10-28 deadline (`draft/recomb2027/sections/00_abstract.tex`).**
- Filled the pending model audits from finished results: GSNN (`10_23_report.md` §1: 4/4 rewired > 2/2 real,
  +0.022 r_flat; ER ≈ real; in_shuffle −0.092), E8 graph-free NN (`10_30_report.md` §1), GEARS empty GO graph
  (`10_16_report.md` §2). PDGrapher removed from the abstract (E6 never run). No new numbers.
- Wording choice to confirm with the user: E8 is described as "within run-to-run variation" of GSNN. On r_flat the
  pre-declared tie threshold was missed by 0.00008 (NN 0.3034 vs 0.3135); on mean_r they are equal. GSNN is one fold.
- Framing still follows the 2026-10-23 12:00 recommendation (audit paper + MKL tiers); user decision pending.

**2026-10-26 — 用户问：RECOMB 是否接收非模型论文 / Does RECOMB take a non-method paper? (venue check; nothing run).**
- RECOMB 2027 CFP (easychair.org/cfp/RECOMB2027, fetched 2026-10-26) is stricter than 2026's: it requires "a substantial
  computational contribution or methodological advance … a new algorithm, model, statistical method, or theoretical
  result"; application of existing methods to new data, or a resource/pipeline without methodological novelty, is out
  of scope and may be returned without full review. Benchmarks are not mentioned either way. Deadlines confirmed:
  abstract 10-28, full paper 11-04, 10 pages + unlimited appendix. 2026 CFP had no such exclusion.
- Accepted lists 2024 (57), 2025 (55 of 339), 2026 (65): almost all new methods. The critique-type papers that got in are
  packaged as a statistical method or theory: 2025 "Synthetic control removes spurious discoveries from double dipping",
  "Tree reconstruction guarantees … Neighbor-Joining", "mcRigor"; 2024 "Discovering and overcoming the bias in neoantigen
  identification"; 2026 "Bias in GWAS test statistics due to omitted interactions", "Minimizer density revisited".
  No pure benchmark found (title-level check only).
- Implication (agent): a pure audit is at desk-reject risk. Frame the contribution as a statistical method (the
  degree-preserving null ladder + model-free information test, as a hypothesis test for network priors) plus a
  theoretical result (E9 configuration-model proposition, not yet started) plus the constructive tiered MKL model. Awaits
  the user's decision together with the pending framing choice (2026-10-23 12:00).
- (follow-up, same day) Fallback venues. ISMB/ECCB 2027 (Copenhagen, 2027-07-18–22): proceedings deadline not yet
  posted; 2025 was 01-23, 2026 was 01-20 (no extensions), so expect ~late Jan 2027; proceedings appear in
  *Bioinformatics*. RECOMB notification 2026-12-22 precedes it, so RECOMB → ISMB is a valid sequence (no double
  submission). Journals are rolling, no annual deadline: *Genome Biology* reports median 14 d to first editorial
  decision, 270 d to acceptance (2024 metrics); *Bioinformatics* ~6–10 weeks to first decision (third-party estimate).
- (follow-up, same day) 用户问 `~/HTA11` 的模型能否算新方法。Read-only assessment (agent), nothing changed in HTA11:
  not as it stands. Its pipeline is paired PyDESeq2, same-person unbalanced OT, a healthy-only scVI anomaly score, a flow,
  and DestVI, which is existing tools on a new dataset, the case the RECOMB 2027 CFP excludes. Its strengths are moderate
  (held-out AUROC 0.66–0.73, PCA matches scVI, flow loses to the population baseline 8/33), and it is a two-student course
  project (Muhammed Bah). It is also unrelated to this paper, so it cannot be this paper's methodological component.
  The possible seed of a method is the two-axis novelty score (depth-calibrated reconstruction deviance finds ASC 0.78;
  latent distance finds SSC 0.85) plus the healthy-only-HVG blind spot; that needs simulation, ≥ 2 more datasets and the
  Dann 2023 / Milo baselines, so it is an ISMB-or-later idea, not 11-04.

**2026-10-26 — User decision: stay with the Therapeutic paper; agent runs the method tests (E9 + ladder calibration/power).**
- User (zh): "那还是写现在这个Therapeutic吧。你按照你的思路进行方法测试。" Framing: the control ladder is a statistical test
  for network priors (degree-conditioned permutation test), backed by a proposition (E9) and characterised by
  calibration and power on simulated responses with known source; the published-model audits are its applications.
- **E9 proposition (to verify numerically).** Configuration model with degree sequence d, 2m stubs:
  E[A_gj] ≈ d_g d_j / 2m. For a target set T with total degree D_T, one-hop mass x = A 1_T has (i) Σ_g x_g = D_T
  exactly on every graph with that degree sequence, so a degree-preserving rewire leaves it unchanged; (ii)
  E[x_g] ≈ d_g D_T / 2m; (iii) two-hop E[(A² 1_T)_g] ≈ d_g D_T κ / 2m with κ = Σd²/2m. Directed: d_in(g) D_out(T)/m.
  Check: mean over 100 rewires vs formula (R², relative error by degree bin; the simple-graph constraint is expected
  to bend it at hubs), and the real graph's deviation from the expectation (the wiring term).
- **Simulation (pre-declared before any run).** Networks: STRING functional ≥ 700 (undirected) and OmniPath+CollecTRI
  (directed); STRING physical is no longer on disk (`raw/string_v12` has only the functional links). 300 synthetic drugs,
  1–4 targets each drawn ∝ (degree+1)^0.5, 978 random read-out genes. Truth s = [λ A_real + (1−λ) A_G0] 1_T, where G0 is
  a degree-preserving rewire *not* in the ladder pool; y = z(s) + 1.7 ε (oracle r ≈ 0.5). λ ∈ {0, 0.05, 0.1, 0.2, 0.3,
  0.5}; 200 replicates per λ (new drugs, read-outs, noise, G0 each). Predictor per graph H: OLS of y on log1p one-hop and
  log1p two-hop mass over H, fit on 70 % of drugs, flattened Pearson on the rest. Ladder test p = (1 + #{rewired ≥ real}) /
  (R + 1), R = 39 rewires; naive test = the same against 39 Erdős–Rényi graphs (what published controls do). Also run an
  exchangeable control in which "real" is itself a rewire, to check the procedure apart from real-graph higher-order
  structure.
- **Pre-declared reading.** Calibrated if the ladder's rejection rate at λ = 0 is within [0.02, 0.09] at α = 0.05
  (binomial range for 200 reps) in the exchangeable control; for the real topology at λ = 0, report the rate as the
  test's sensitivity to non-degree structure (assortativity/clustering) and do not tune anything to hit 0.05. The
  naive-control claim holds if its λ = 0 rejection is ≥ 0.5. Power: report the smallest λ with ≥ 0.8 rejection per
  network. Any outcome goes in the paper's Methods/appendix as measured.

**2026-10-26 05:10 — E9 + calibration read against the pre-declared rule (`10_30_report.md` §2).**
- 校准通过：可交换对照下 λ = 0 拒绝率 0.065 / 0.060，在预设 [0.02, 0.09] 内。Naive (vs ER) rejects 1.000 at λ = 0 →
  the claim "the published control credits degree to wiring" holds (pre-declared bar ≥ 0.5). Power ≥ 0.8 at λ = 0.05
  (exchangeable); with real topology at λ = 0.1 (STRING) / 0.3 (OmniPath+CollecTRI).
- Reading (agent), not pre-declared: with real topology the test is conservative (0/200 at λ = 0), because the real
  graph carries the degree signal *worse* than its rewires. This gives a mechanism for GSNN's rewired > real: under a
  degree-only world, rewired > real is the expected outcome. Paper use: Fig. 1C (calibration/power curves) and one
  sentence in the GSNN paragraph; do not claim it *explains* GSNN without a GSNN-specific check.
- Proposition: (i) exact; (ii)/(iii) R² 0.83–0.97 on the rewired ensemble, with a hub bend of 5–14 % on the directed
  network. Written into `draft/recomb2027/sections/02_framework.tex` §theory and Appendix B.
- Next: rewrite the abstract around the method (test + proposition + calibration), keeping the audits as applications.

**2026-10-26 10:20 — 方法化框架定稿 + 2K 零模型实验预注册 / Method framing agreed; 2K null pre-registered.**
- User (zh, 10:20): "ok" to the 09:30 proposal (discussed in chat; recorded here): the paper's method is the inferential
  procedure, not a predictor. Inputs (training procedure f, graph G, data, metric); H0 = the score depends on G only
  through its (degree-level) structure, so G is exchangeable with its rewiring orbit; outputs (1) a permutation p,
  (2) a credit decomposition real − ER = (rewired − ER) [degree] + (real − rewired) [wiring], with labels/empty for gene
  identity, (3) an upper confidence bound on the wiring effect when the test does not reject (needed because the audits
  use R = 2–5 rewires, so p ≥ 1/(R+1) can never reach 0.05). MKL tiers become an application (each view vs its own
  null); our own model families move to the appendix. Novelty to argue against Maslov–Sneppen 2002 and the dK-series
  (Mahadevan 2006; Orsini 2015): literature check pending (not in `papers/README.md`).
- **Why a 2K rung.** With real topology the 1K (degree-preserving) test is conservative (0/200 at λ = 0) because
  real-graph degree–degree structure moves mass away from the degree expectation (`10_30_report.md` §2.2). A null that
  also keeps the joint degree distribution (2K) should remove that part and raise power, and adds a rung to the
  decomposition: degree → degree–degree correlation → remaining wiring.
- **2K swap (to implement in `scripts/netnull.py`, kind `rewired2k`).** Double-edge swap (a–b, c–d) → (a–d, c–b),
  accepted only when b and d have the same degree (directed: same in-degree, or equivalently a and c the same
  out-degree), so every degree and the joint distribution of (k_out(source), k_in(target)) are kept exactly; self-loops
  and duplicates rejected; 5 swaps per edge, ≤ 20× attempts. 100 per network (seeds 0–99), STRING functional ≥ 700 and
  OmniPath + CollecTRI. Report swaps achieved, real edges kept, and a JDM check (must be identical).
- **E9-2K.** R² of the real graph's one-/two-hop mass vs the 2K rewire mean, compared with the 1K values (STRING
  0.067 / 0.234; OmniPath + CollecTRI 0.360 / 0.219): the share of the "wiring term" that is degree correlation.
- **Simulation (same generator, predictor, noise, 200 replicates, R = 39, α = 0.05).** Ladder = 2K rewires 0–38;
  G0 and surrogate from 2K seeds 40–99. Worlds: `exchangeable` (real := a 2K rewire), `real` (truth over λ A_real +
  (1 − λ) A_G0, G0 a 2K rewire), `real_truth1k` (G0 a 1K rewire: degree-only truth, tested with the 2K ladder). Each
  replicate also computes the 1K-ladder p on the same data (paired comparison). No ER arm (already measured).
- **Pre-declared reading.** (a) Calibration: `exchangeable` λ = 0 rejection in [0.02, 0.09]. (b) 2K becomes the
  paper's recommended null if, on both networks, `real` λ = 0 rejection ≤ 0.09 **and** paired power at λ = 0.1 is
  ≥ the 1K power + 0.10 on at least one network without dropping on the other; otherwise 1K stays the default and 2K is
  reported as an extra rung. (c) `real_truth1k` λ = 0 rejection > 0.09 would mean the 2K test is anti-conservative when
  responses depend on degree alone; then the rungs must be read top-down (1K before 2K) and the paper says so.
  (d) Every outcome goes in the appendix as measured; nothing is tuned to hit 0.05.

**2026-10-26 10:55 — Wiring bound on the existing audits (`10_30_report.md` §3); reading.**
- Estimator chosen: run-level Welch interval for real − rewired (runs mix training noise and between-rewire
  variation, the right unit for "would another rewire or another training beat real"). Not a drug bootstrap,
  because that conditions on the seeds and on the single rewire. Implemented in `scripts/wiring_bound.py`.
- Reading (agent): GSNN gives an informative bound (upper −0.010 r_flat; wiring excluded). GEARS and DeepCE do not:
  with 3 runs per arm, wiring effects up to +0.03–0.07 cannot be excluded. Paper wording must then be "no evidence of
  a wiring effect; effects larger than X excluded" with X per model, not "wiring does nothing".
- Possible follow-up, not run (needs the user): more GEARS seeds per arm (e.g. 5 real + 5 rewires × 1 seed; GPU) and
  2 more DeepCE rewires, to tighten those bounds before 11-04. The GSNN statement does not depend on it.
- (11:05) Novelty check for the method claim (3 web searches; registry entry added to `papers/README.md`). Degree-
  matched permutation already exists for network-propagation scores (DADA 2011; Picart-Armada 2020) and dK nulls are
  standard in network science (Mahadevan 2006; Orsini 2015); one proteomics GNN (PMC12676398) already uses a
  degree-preserving prior control. So the novelty must be stated as: a permutation test for *trained* models'
  network priors, the proposition that rewired-graph propagation is a degree function, the calibration/power
  evidence that the ER control is invalid and the 1K test conservative, the 2K rung, and the few-rewire bound.

**2026-10-28 04:55 — 用户新方向：为 RECOMB 找一个模型方法角度（多模态模型入手）/ User: find a model-design angle for a RECOMB method paper.**
- User (zh): RECOMB needs a methodology paper; abstract due in about one day. Dr Wen wants to start from the multi-modality
  model. "Only positive results go in the paper", so negative results here are expected; try anything, sanity checks
  only today, until there is enough to report. Internal records stay complete (positive and negative).
- Agent's first angle, from the data structure: the MKL model (`10_09_report.md` §7) fits one KRR per cell line, so a
  training compound measured only in other lines never informs line L. P1-24 has 8,119 compounds but each line sees
  1.7k–7.1k of them (957 compounds in a single line); P2-24 is nearly complete (1,527 of 1,725 in all 7 lines); Tahoe
  unseen-drug slices are complete (350 compounds × 50 lines).
- **Design (exploratory, `scripts/run_l1000_xline.py`).** Drug-level consensus target c(d) = mean over lines that
  measured d of (Y − μ_line) on the fold's training compounds; KRR g of c on all training compounds (kernel T1 = chem,
  or T3 = MKL's chem+tgt+moa+deg+ppi1+morph with centred-alignment weights against c); prediction
  μ_L + a_L g(d*) + b_L f_L(d*), f_L the unchanged per-line KRR, (a_L, b_L) ∈ {0..2}×{0..1} by the same inner group CV
  as λ. `gonly_*` forces b = 0. Test compounds are held out of every line (compound-level folds).
- Check (smoke 348394, P2-24 ik14 fold 0, `results/_smoke_xline`, not to be quoted): per-line arms krr_chem / mkl_T3 /
  mean reproduce `results/l1000_mkl/P2-24_ik14_seed0` bit for bit (max |Δ| 0.0, 2,494 rows). One fold: pool_T3 0.4547 vs
  mkl_T3 0.4441, pool_T1 0.3652 vs krr_chem 0.3621.
- Submitted 05:05: 348395/348396 P2-24 ik14/platemap, 348397/348398 P1-24 ik14/platemap, 348399–348401 Tahoe T-0.05 /
  T-0.5 / T-5 ik14 (`--source Tahoe`). Reading planned: pool_T3 − mkl_T3 and pool_T1 − krr_chem on ACTIVE rows, line
  bootstrap; expectation (agent, before results) larger gain on P1-24 (sparse coverage) and Tahoe (50 lines).

**2026-10-28 05:15–05:50 — Cross-line model: first reads, a design flaw found, and the DeepCE-split comparison.**
- P2-24 (`10_30_report.md` §4.1): pool_T3 − mkl_T3 +0.010 / +0.012 (ik14 / plate-map), 7/7 lines both; consensus
  alone (gonly) is worse than per-line, so the model needs both the shared and the line-specific term. Reading
  (agent): the pooling gain is ~4× larger with the multimodal kernel than with chemistry (+0.010 vs +0.0027); a
  hypothesis, not tested: the richer kernel predicts the context-shared mechanism part, which pooling then
  estimates with more data.
- DeepCE split (`10_30_report.md` §4.2; `repro/deepce/xline_kernel.py`, gitignored with `repro/`; must move to
  `scripts/` before public code release). Joint KRR K_drug·k_dose·(ρ+δ_line), all hyper-parameters on upstream
  dev. Multimodal cross-line 0.5994 vs DeepCE reproduced 0.4548 (+0.145 [+0.113, +0.177], 78/92 drugs) and DeepCE
  paper 0.4907. First run 348404 had σ_dose at the top of its grid (2.0); grid widened to {…, 4, ∞} and ρ to
  {0, 0.1, …} in 348410; σ = 4 and ρ = 0.1–0.25 now interior. One split only → 5-fold drug-disjoint resplits
  submitted (348411 seed 0, 348412 seed 1; inner dev = 1/5 of training drugs).
- Fairness control submitted (348414, `repro/deepce/mlp_multimodal.py`): MLP with the same inputs (Morgan,
  targets, MOA, Cell Painting + availability flags, cell one-hot, log dose), same target, dev-selected width /
  dropout / weight decay / epoch, 3 seeds. Separates "the modalities help" from "the kernel model helps".
- **Flaw found (by availability breakdown on DeepCE test drugs):** with zero-masked missing views, the multimodal
  model is *worse* than chemistry alone on drugs that lack views (no targets/MOA and no morphology, 10 drugs:
  chem 0.4155 vs multi 0.3682; morphology only, 20 drugs: chem_kn 0.4894 < chem 0.5070). Cause: the combined
  kernel's diagonal equals the total weight of the drug's available views, so a sparsely annotated drug is
  shrunk toward the mean. **Fix (pre-declared before running):** availability normalisation
  K_ij / √(K_ii K_jj) of the combined kernel (D^{-1/2} K D^{-1/2}, PSD-preserving; all view kernels have unit
  diagonal where present). Arms `*_norm` (DeepCE split, job 348415) and `T3n` (P2-24 jobs 348416/348417; P1-24
  after 348397/348398 finish). Reading: keep it if norm − unnormalised ≥ 0 overall and positive on the
  missing-view strata; report either way.
- Tahoe T-0.05 (348399): flat. Every arm within 0.011 of the line mean (mean 0.4870, krr_chem 0.4977, mkl_T3 0.4960,
  pool_T1 0.4986, pool_T3 0.4971); pool_T1 − krr_chem +0.0009 [+0.0006, +0.0014]; mkl_T3 − krr_chem −0.0017
  [−0.0061, +0.0030]. Lowest dose has little drug-specific signal; T-0.5 / T-5 running/queued. Note Cell Painting
  covers only 99 of 350 Tahoe compounds, so T3 there is mostly targets/MOA.

**2026-10-28 07:20 — Reads: resplits, normalisation, MLP control, P1-24, Tahoe (`10_30_report.md` §4.3–4.6).**
- Pre-declared normalisation reading (05:50): overall norm − unnormalised ≥ 0 → **met on DeepCE data** (resplits +0.005,
  9/10 folds; single split +0.0025, ns); **not met on Tahoe T-5** (−0.0022, ns). P2-24 / P1-24 T3n pending. So keep
  normalisation as a component, and state the Tahoe result in the record; the abstract does not claim it there.
- Same-input MLP: kernel model +0.056 over the MLP with all views; the MLP beats chemistry KRR with chemistry alone
  (−0.020 for KRR). Reading (agent): the advantage is in fusing partially observed views, not in the regressor on dense
  inputs. Honest wording for the paper: "with the same inputs, the kernel model exceeds a tuned MLP by 0.056", not
  "kernels beat neural networks". The MLP was tuned on a small grid (8 configs) and a single split.
- P1-24: pool_T3 is the first arm above KRR Morgan2 on P1-24 ACTIVE (+0.010 both splits), where the 10-14
  leaderboard had none.
- Tahoe: pooling positive at T-0.5 (50/50 lines) and T-5 (46/50); T-0.05 has no drug signal for any arm.
- Draft method abstract written: `draft/recomb2027/sections/00_abstract_model.tex` (not \input; for the user and
  Jun Wen to choose against the current "Degree, not wiring" abstract). The phrase "each view tested against a matched
  null" rests on the 10-09 MKL nulls (target − permuted +0.035/+0.045; STRING one-hop − rewired ≈ 0), not on a
  per-fold gate inside this model; a gate is not implemented.

**2026-10-28 08:40 — Day-1 sanity phase closed; proposed paper framing (agent; user / Jun Wen decide).**
- Normalisation read against its 05:50 rule: met on P1-24 (+0.002 both splits) and DeepCE resplits (+0.005, 9/10);
  neutral on P2-24 (−0.001 / +0.001) and Tahoe T-5 (−0.002, ns). Kept as a model component (it targets sparsely
  annotated compounds, and helps exactly there); no headline number in the abstract.
- Proposed method = multi-view drug kernel (chem, targets, MOA, Cell Painting) + availability normalisation +
  coregionalised cross-line coupling (+ dose kernel on DeepCE data). Evidence ranking: views +0.06–0.10 (DeepCE data,
  P2-24), +0.024 Tahoe T-5, small on P1-24 (sparse coverage); coupling +0.004–0.012 and positive in 27/28 L1000 and
  46–50/50 Tahoe line evaluations; vs DeepCE +0.147; vs same-input MLP +0.056.
- The "Degree, not wiring" material becomes the model's justification for leaving network views out (their nulls),
  not the paper's headline. Draft abstract: `draft/recomb2027/sections/00_abstract_model.tex`.
- Known gaps before the 11-04 full paper (not run): the coregionalised model on P1/P2 uses the consensus
  approximation, not the exact joint kernel; no published multimodal baseline (e.g. MultiDCP, TranSiGen) run; the MLP
  control is on one split with an 8-config grid; null-gated view selection is not implemented; Tahoe and the network
  null tests use the per-line MKL, not the final model; method name unchosen; the two `repro/deepce` scripts are untracked.

**2026-10-28 14:20 — User asked (zh): run the novelty check and a published multimodal baseline, and consider a
deep-learning innovation. Novelty check (web search; registry `papers/README.md` did not cover multi-task kernels).**
- **Closest prior art, method side.** Bayesian multitask multiple-kernel learning won NCI-DREAM drug sensitivity
  (Costello et al., *Nat Biotechnol* 32:1202, 2014; BMTMKL); Gönen & Margolin, *Bioinformatics* 30:i556 (2014),
  kernelized Bayesian multitask learning across a drug panel. Both predict IC50-type sensitivity, not transcriptional
  responses. Multi-task GP / intrinsic coregionalisation (Álvarez, Rosasco & Lawrence, *JMLR* 2012 review) is textbook.
  Reading (agent): the kernel algebra (MKL + KTA weights, cosine normalisation, ICM coupling) is not new; a method
  claim must rest on the task, the views and the evidence, not on the algebra.
- **Not found** (three extended searches): a coregionalised / multi-task kernel or GP model for L1000 responses of
  unseen compounds; a model that feeds Cell Painting profiles as a drug-side input to predict transcriptional responses
  of unseen compounds. Absence in search is not proof; full-text checks of XPert, MAP, MVCBench still to do.
- **Closest prior art, pharmacology side.** MAP (Feng et al., *Nat Mach Intell* 2026; local `papers/map_2026.pdf`):
  KG + protein/text-aligned compound embeddings for unprofiled drugs, single-cell, deep. So "mechanism knowledge helps
  unseen drugs" is already claimed; our distinct parts are Cell Painting, closed form, cross-context coupling,
  availability normalisation.
- **MVCBench** (bioRxiv 10.64898/2026.04.22.720110, 2026): 12 drug representations; advanced molecular encoders give
  limited gains over fingerprints for gene expression; GE+morphology as joint *targets* helps. Not Cell Painting as
  a drug input.
- **Key new finding: Bai, Prince et al., bioRxiv 10.64898/2026.05.13.724458 (2026).** Seven published L1000 models
  (DeepCE, CIGER, MultiDCP, TranSiGen, PRnet, PertDiT, XPert) retrained on XPert's L1000 SDST set (78,453 pairs,
  8,276 compounds, 164 lines, 10 µM 24 h), drug-blind 5-fold CV (3:1:1, seed 131419). Zeroing or shuffling drug
  input changes PCC_DEG by ≤ 0.012 / 0.027; a drug-free MLP on basal expression reaches 0.637 vs XPert 0.6315 (best).
  Code https://github.com/baijinming97/drug-perturbation-benchmark (commit 55edbfe, cloned to `repro/dpb/upstream`);
  per-fold metrics in its `data/analysis/benchmark_metrics/statistics/full_vs_mlp.csv`; predictions not shipped.
- Decision (agent): use this benchmark as the published-baseline comparison instead of re-running MultiDCP alone. It
  gives all seven models plus the drug-free MLP on fixed folds. Question it poses for our model: does the multi-view
  drug kernel add signal above the drug-free MLP where seven deep models add none? Per-fold PCC_DEG (`full_vs_mlp.csv`):
  MLP 0.6372; XPert 0.6315; TranSiGen 0.6144; DeepCE 0.3947; PertDiT 0.3701; PRnet 0.2771; MultiDCP 0.2545; CIGER 0.2456.
- Plan: (1) reproduce the drug-free MLP on the same folds (pipeline check, target ≈ 0.637); (2) hybrid = MLP backbone
  + multi-view drug KRR on backbone residuals (consensus over lines, optional per-line term), scale chosen on the
  benchmark's valid fold; (3) a neural drug module (DL variant) only after (2) reads. Data: XPert `processed_data.zip`
  (Zenodo 17182939, md5 4a58bc01…, = figshare file 58181296), `/nfs-stor/yuqi.lei/Therapeutic/raw/xpert_sdst/`.
  Figshare returned 403 to aria2c; Slurm fetch 349649 cancelled before start (queue ETA 16:00); fetched from Zenodo
  on the login node (network-only).

**2026-10-28 20:45 — Kernel model on the Bai et al. drug-blind benchmark: protocol and pre-declared reading.**
- Session was interrupted 14:18–20:28 (no work ran). Data: XPert SDST via Zenodo (md5 ok). `scripts/prep_xpert_sdst.py`
  (job 350503) reproduces the benchmark's five drug-blind folds exactly (per-fold sample and drug counts equal
  `A_split_integrity.csv`); 350501 failed on BRD aliases (one pert_idx ↔ several pert_id; 98 aliased compounds), fixed
  by joining on `drug_node_idx`. Views: chem 8,276/8,276 drugs, targets/MOA 2,957, Cell Painting 1,937 (ik14 match).
  The first attempt read a downloaded `.npy` with `allow_pickle=True`; the auto-mode guard blocked it (pickle can run
  code); only h5py/CSV are read from downloaded files now.
- Model `scripts/run_sdst_hybrid.py`: our run of the benchmark's drug-free MLP (same architecture, loss, optimiser,
  early stopping; seed 1729) as backbone; drug terms fitted on its train residuals: `krr_<v>` (consensus over lines),
  `pool_<v>` (+ per-line KRR), `nnk_<v>` (neural head on Nyström features of the kernel = the DL arm), and
  `pool_plate` (per-line KRR with a plate-overlap kernel; batch-effect null). All choices on the benchmark's valid
  fold; test scored once. Smoke 350505 (2 epochs; not to be quoted) ran end to end.
- **Pre-declared reading (written before any full-run result):**
  (1) Drug signal claim: `pool_multi_norm` − our `mlp` > 0 in ≥ 4/5 folds, with the drug-bootstrap 95% interval
  above 0 in ≥ 4/5 folds; and its fold PCC_DEG above the *published* MLP in ≥ 4/5 folds.
  (2) Batch confound: if `pool_plate` gains as much as `pool_multi_norm` (within its interval), the per-line gain is
  read as plate structure, not pharmacology, and the claim falls back to `krr_*` (consensus term) only.
  (3) Views: `pool_multi_norm` − `pool_chem` > 0 in ≥ 4/5 folds for "pharmacology/phenotype views add signal".
  (4) DL arm: `nnk_multi_norm` vs `krr_multi_norm`, reported either way; no gate.
  Expectation (agent): small. The benchmark's own oracle (same test drug's measured response in other lines) cuts MLP
  residual MSE by only ~2.4% (`response_components/results/cross_cell_summary.csv`, MLP_residual_1729 mse_gain 0.0073
  of 0.305).
- Scheduling: 6 h single job (350513) was queued to 00:01 (no backfill); cancelled and split per fold (1.25 h each).
  GPU QOS allows 4 submitted jobs per user: smoke2 350519 (plate control), folds 0–2 = 350520–350522 (afterok
  350519); folds 3–4 to submit when a slot frees.

**2026-10-28 21:05 — Smoke 350519 (fold 0, backbone trained 2 epochs; NOT to be quoted): plate overlap alone
dominates.** Test PCC_DEG: weak MLP 0.3116; + multi_norm drug kernel pooled 0.3693; **+ per-line plate-overlap
kernel (pool_plate) 0.6452** — above every published model and the published drug-free MLP (0.637), with no drug
structure or pharmacology. Reading (agent, provisional): a large share of the benchmark's PCC_DEG is plate (batch)
structure in delta = treated − plate-matched control. Hypothesis to test: the drug-free MLP's 0.637 also comes largely
from a plate fingerprint in the paired control profile, not from "cell-line basal expression" (this would explain Bai
et al.'s finding mechanistically). Consequences: (i) our drug-kernel per-line gains may be partly co-plating of
similar compounds; (ii) the meaningful drug test is drug kernel *beyond* plates. Added arm `pool_plate_drug`
(per-line kernel J_plate + γ·K_multi_norm, γ ∈ {0.25, 1, 4}; flag `--plate-drug`, off by default so folds 1–2 of v1
are unaffected); smoke 350535. Full-backbone numbers (v1 folds) decide; the pre-declared rule (2) applies as written.

**2026-10-28 21:30 — Correction and mechanism: what the 21:05 smoke reading got wrong, and why plates matter.**
- Correction to 21:05: "a large share of PCC_DEG is plate structure" was read from a 2-epoch backbone. With the
  trained backbone (folds 0–2, report §5.2) the plate-overlap term adds +0.0055, the multi-view drug kernel +0.012.
- Mechanism found (report §5.1): `X_ctl` is one control row per (line, plate set), so it is a plate identifier; 90%
  of test samples share it with training samples. A model-free plate-mean lookup scores 0.604 (0.643 on shared
  samples). Reading (agent): the drug-free MLP's 0.637 is mostly a learned plate-mean lookup, not "cell-line basal
  expression" as Bai et al. interpret; this is a more specific explanation of their finding that drug inputs are
  unused. Our earlier L1000 work uses Level 5 (plate-population z-scored) signatures and plate-map splits
  (`10_09_report.md` §1.7), so it is less exposed; DeepCE's data are Level 5 as well.
- Implication: on this benchmark a drug claim must be stated beyond plates. Planned: (a) v2 `pool_plate_drug` −
  `pool_plate` (running, 350550); (b) a plate-centred metric: PCC of (delta − plate-mean lookup) vs (prediction −
  plate-mean lookup) on shared samples, so the score counts only what differs between compounds on the same plate;
  (c) report the 10% unshared samples separately.

**2026-10-28 22:20 — Five-fold read against the 20:45 rule.** (1) met: every drug-kernel arm > our MLP, > the
published MLP and > all seven published models in 5/5 folds, intervals > 0 in 5/5. (2) met: pool_plate +0.0055 vs
pool_multi_norm +0.011 (below its interval in every fold); and plate + drug − plate = +0.007 in 5/5 (§5.3). (3) met:
multi − chem +0.005, 5/5. (4) nnk_multi_norm best single arm (0.6493), ≥ pooled KRR in 4/5. Caveat recorded: drug arms
improve only 42–58% of test drugs (gain concentrated in a subset); plate arm 81%. Still to run: plate-centred metric
(v3, predictions saved). First v3 submission 350588 failed (my bug: the new `keep()` helper was shadowed by an existing
mask variable named `keep`); 350589/350590 cancelled; fixed and resubmitted as 350615–350617.

**2026-10-28 22:40 — Paper moved to its own repository; method framing chosen by the user.** The user created
`CyclopsRay/MKL_Therapeutic` (private; Overleaf GitHub Sync) and asked to push the abstract there; the code repository
is deferred. Cloned to `paper/` (ignored by this repo). Commit d1bba4b: README with the RECOMB 2027 rules from
`RECOMB_requirements.md` (scope = methodological contribution; 10 pages US Letter; ≥ 10pt incl. captions; 1-inch
margins; abstract registration frozen; public code repo with README by 11-04), `main.tex` (llncs + geometry letter
1in + normal-size captions), method abstract + `abstract.txt`, section stubs, legacy "Degree, not wiring" sources.
Correction: the abstract deadline 2026-10-28 23:59 AoE is **2026-10-29 15:59 Dubai** (earlier I said ~08:00).
Abstract numbers come from report §4–§5; the plate-centred metric (v3) is not in the abstract.

**2026-10-28 22:50 — Plate-centred metric read (report §5.4).** The drug claim survives the strictest check:
centred PCC 0.099 → 0.154 (+0.055, 5/5, all CIs > 0) for both pool_multi_norm and nnk; plate kernel +0.001. This
removes the main reviewer objection to the drug-blind benchmark (plate leakage). Proposed (agent) abstract edit:
replace "its compound term adds signal beyond plate structure in every fold" with the centred result (0.10 → 0.15),
since it is the stronger and more interpretable statement; done in paper commit after d1bba4b, to be confirmed by the
user before registration. ONGOING entry for this benchmark cleared.

**2026-10-28 23:15 — Name DECANT (user); remaining sanity items; paper draft f9eb731.** User chose the name DECANT
(DEcomposing Context And compouNd Transfer). Remaining checks submitted: (a) backbone seed replicates on the drug-blind
benchmark, seeds 2718 / 31415 (jobs 350673, 350674; `--tag v1_seed*`); (b) the LINCS/Tahoe cross-line runs used T3 =
chem+tgt+moa+deg+ppi1+morph, i.e. two network views the paper's DECANT does not use; added kernel set `D` (chem, tgt,
moa, morph, normalised) to `run_l1000_xline.py` and reran P2-24 / P1-24 × ik14 / plate-map and Tahoe T-0.5 / T-5
(jobs 350677–350682, `--kernels T1 D --tag _D`). Paper sections 1–5 drafted with numbers from report §4–§5; LINCS /
Tahoe numbers and seeds left as \pending until (a)/(b) read. Not run (known gaps): same-input MLP on resplits;
view nulls on the drug-blind benchmark; X_ctl construction confirmed from data structure only, not from XPert code.

**2026-10-29 01:15 — Paper direction fixed: DECANT, drug encoding (user).** 标题和摘要定稿，之后所有工作以此为准。
- Title (user chose option A of three): **"DECANT: Encoding Unseen Drugs by Their Similarity to Profiled Compounds for
  Transcriptional Response Prediction"**. Authors: Yuqi Lei, Hanzhe Liang, Keqin Peng, Jun Wen, all MBZUAI. Paper repo
  `paper/` commit 0f4a055 (pushed; the user pulls into Overleaf and registers on EasyChair before 15:59 Dubai today).
  After registration the title and abstract are frozen (only minor edits allowed).
- Abstract story: unseen compound → only the drug's description distinguishes it → seven published models barely use
  it (≤ 0.012 PCC_DEG, Bai et al.) → the missing part is the compound-specific signal beyond the plate → DECANT
  (context backbone + multi-view normalised compound kernel, within- and cross-line transfer) → wins on the drug-blind
  benchmark (0.649 vs 0.637 drug-free, 5/5 folds; plate-centred 0.10 → 0.15) and DeepCE's benchmark (0.602 vs 0.491),
  cross-line transfer 27/28 LINCS, 46–50/50 Tahoe → the gain is how drug information enters the model (same-input MLP
  0.544).
- Decision: **PPI is out of the abstract** (user asked whether it belongs; agent recommended dropping it: it pulls the
  reader back to the network-prior story, and the same-input MLP is the evidence that supports the encoding title).
  The network nulls stay in the paper only as one row of "which compound views transfer" (Results) and in the
  appendix. The "Degree, not wiring" audit (PLAN Phase 6: E2/E4/E5/E6, 2K ladder) is no longer a paper line; no new
  runs for it.
- Seed replicates (350673/350674) read: DECANT 0.6493/0.6494/0.6491, 15/15 fold × seed above our MLP (report §5.5).
  Fills `\pending{seed replicates}` in `paper/sections/04_results.tex`.
- Next (agent proposal, to confirm): Fig. 1 pipeline schematic first (it is the method figure and needs no pending
  numbers); then figures from finished results; the DECANT-view LINCS/Tahoe reruns 350677–350682 (still PENDING on
  cscc-cpu-p) fill Fig. 4 / §4 when they land.

**2026-10-29 02:00 — Figures and the next experiments (user, 01:30).** 用户决定：
- Fig. 1: drop the "response anatomy" panel; one left-to-right pipeline in two lanes (control → backbone; compound
  views → kernels → fusion → normalisation → kernel transfer, fed by residuals); module names only, explanations in the
  caption, computation only in Methods. Done: paper 1ee3775 (SVG + PDF, `scripts/figures/fig1_pipeline.py`).
- Fig. 2 replaces Table 2 (done, 3e574df): published Full vs Zero per model + plate lookup (a); lookup with vs without
  a plate-mate (b). PPI: one bar group in the "which views transfer" figure only; network audits (GSNN/GEARS/DeepCE
  gene vectors) to the appendix.
- Experiments, in the agent's priority order, all approved:
  1. Same-input control on the drug-blind benchmark (`--input-mlp`, job **351007**, tag v4_input): `mlp_input` = the
     benchmark MLP on [control, Morgan, target/MOA multi-hot, z-scored Cell Painting, availability flags], end to end;
     `nnf_multi` = the DECANT-NN head (2 layers) on the same raw features on top of the fixed backbone, so it differs
     from nnk_multi_norm only in how the compound is represented. Grids chosen on valid (mlp_input dropout 0.1/0.3;
     nnf hidden/dropout 512/0.3, 2048/0.3, 512/0.5). Reading: if nnk_multi_norm > nnf_multi and mlp_input in 5/5
     folds (plain and plate-centred), the "encoding, not information" claim holds on the main benchmark too.
  2. View ablation (job **351008**, tag v4_views): only_<v>, minus_<v>, perm_<v> (labels permuted among compounds
     with the view, coverage kept), pooled KRR, saved predictions for the plate-centred metric.
  3. Analysis without training (done, report §5.7): the user asked whether compounds close in the fused kernel but
     chemically unlike exist and respond alike, as the explanation for why every view helps. Yes: chemically unlike
     pairs in the top 0.5% fused respond 13× more alike than other unlike pairs; each view's similarity is monotone in
     response similarity; Cell Painting finds response twins without any shared annotation; the per-compound gain of
     the extra views is largest when the compound has no chemical analogue. Signature reliability is low (split-half
     0.134), so absolute similarities are small; use relative contrasts and named examples in the paper.
- First compound-analysis run 351009 had a wrong per-view summary (quantile cut-offs on target/MOA similarities that
  are mostly exactly 0 select every pair); fixed with fixed bins in 351012. 351009's other numbers are identical.
- CPU queue note: 96G jobs (350677–350682) wait on node memory, not cores; small-memory jobs start at once. GPU jobs'
  time limits lowered (2–2.5 h) for backfill.

**2026-10-29 03:05 — Figure standard and figure revisions (user, 02:40).**
- User rules, added to the GeneralAgent skill (`CyclopsRay/GeneralAgent` d77a326, `references/figures.md`):
  (1) names only in the artwork, explanations in the caption or text; (2) one markdown dossier per figure with the
  caption, how the paper addresses it, and the generator scripts and inputs. Implemented: `paper/figures/<fig>.md`
  for the three figures, `scripts/figures/style.py` (palette, font gate, save + source CSV),
  `scripts/figures/sync_captions.py` (tex → dossier caption mirror, `--check` for drift). No licensed Arial on the
  cluster, so per the skill's font gate every asset is `_preview` (DejaVu Sans); the user edits fonts in Illustrator
  or supplies `FIG_FONT`.
- Fig. 1 v3: "Normalization" (US); backbone drawn as a network; control = DMSO well → "Control expression" (answer to
  the user: on the drug-blind benchmark x_ctl is the 978-landmark L1000 profile of vehicle-treated cells of the same
  line on the same detection plate, one row per (line, plate set)); icons for compound and views. The paper body still
  uses British spelling; convention to be decided by the user.
- Fig. 2 v2: DECANT row in (a) (with = nnk_multi_norm 0.6493, without = backbone 0.6373); "plate lookup" renamed
  "plate-response baseline" in figure and text (intro, §2); new (c) PCC_plate (baseline 0, MLP 0.099, plate kernel
  0.100, chemistry kernel 0.131, DECANT 0.154; v3 centred.json).
- Plate null on published models (user question): not done and not possible from released material — Bai et al.
  release fold-level metrics only, no predictions, so PCC_plate cannot be computed for them. Their published
  Full − Zero ablation is the drug null we can show. Option (not run): retrain one or two of them (code is in
  `repro/dpb/upstream/models/`) with saved predictions; needs the user's go-ahead and GPU time.
- Views figure v2: names list replaced by four pairs with RDKit-coordinate structures and signature strips
  (`example_pairs.json`, job 351036); wired into Results with a paragraph citing report §5.7.

**2026-10-29 03:30 — Paper updated with finished results; TODOs for the rest (user, 03:15).** Paper commits
e1c0905, 491c143: `\todo{}` macro; §4 cross-line subsection and Table `tab:xline` written from report §4.1/4.3/4.4/4.7/4.8
(kernel set T3, which includes degree and one-hop STRING views) with a TODO to replace by the set-D reruns
350677–350682; TODOs for the same-input control (351007) and view ablation (351008); TODO in each figure caption for
the user's own redraw; appendix tables (per fold, seeds, published Full − Zero, example pairs). Partial 351007 read
(3/5 folds, not yet in the report): end-to-end `mlp_input` 0.612–0.613 (below the drug-free MLP), raw-feature
residual head `nnf_multi` within 0.001–0.002 of DECANT-NN in PCC_DEG — so on this benchmark the two-stage
decomposition carries most of the gain and kernel-vs-features little; final reading after all folds and the
plate-centred score. Abstract unchanged (its 0.544 same-input number is DeepCE's benchmark, end-to-end MLP).

**2026-10-29 03:15 — Same-input control read (report §5.8): it does not support "kernel encoding beats features" on the
main benchmark.** 读数如实记录：
- End-to-end MLP on [control, all compound features]: PCC_DEG 0.613 (−0.024 vs drug-free, 5/5), but PCC_plate +0.052,
  equal to the raw-feature residual head (+0.052) and close to DECANT (+0.055); DECANT − raw-feature head +0.003 PCC_plate
  (0/5 intervals above 0), +0.0013 PCC_DEG (5/5 positive, tiny).
- What holds: (1) published models barely use the drug; (2) given multi-view compound information, the
  compound-specific part is recoverable; (3) fitting context first keeps the plain score (end to end loses 0.024, and
  0.040 on samples without plate-mates); (4) on DeepCE's small benchmark the kernel beats a same-input network (0.602
  vs 0.544, single split). What does not hold on the drug-blind benchmark: that the similarity kernel, rather than the
  views themselves, is what recovers the compound signal.
- Implication for the abstract (registration due 15:59 Dubai today): its last two sentences ("The gain comes from how
  drug information enters the model, not from more information or capacity … rather than from a learned embedding of
  the compound alone, is what lets a model use the drug") overstate. Agent proposal to the user: replace with a
  decomposition + information statement; title can stay (it names the method). Decision is the user's.
- Caveat: mlp_input hit the 500-epoch cap in 3/5 folds. Optional follow-up (not submitted): rerun with 1,000 epochs;
  same-input MLP on the DeepCE resplits to test whether the kernel advantage there is more than one split.

**2026-10-29 03:30 — Abstract revised before registration (user approved the agent's proposal).** Paper 6159171:
the "how drug information enters" sentence kept, its evidence now the drug-blind same-input control (0.613, §5.8) plus
DeepCE's 0.544; the last sentence ("similar compounds rather than a learned embedding") replaced by the views
sentence (§5.7), because the raw-feature residual head contradicts it on the drug-blind benchmark. Title unchanged.
abstract.txt synced (350 words). Discussion now states decomposition > kernel-vs-features at this scale and the
kernel's edge on DeepCE's smaller benchmark (paper 9c72449). The user registers on EasyChair.

**2026-10-29 03:50 — View ablation read (report §5.9) and in the paper (fde1224).** Every view alone helps (+0.025 to
+0.032 PCC_plate, ~half the gain); chemistry and Cell Painting are not replaceable (removing −0.010 / −0.014,
permuting −0.020 / −0.014, all folds); targets and MOA duplicate each other (≤ 0.002). This answers the user's
"why is each modality useful" together with §5.7: Cell Painting and chemistry carry unique information, the
pharmacological annotations carry it twice. Compound figure v3 adds the ablation as panel (a); appendix table
`tab:app-views`. Not run: targets and MOA permuted together.

**2026-10-29 09:25 — Set-D cross-line reruns (350677–350682) read (report §4.9).** 六个作业全部正常结束（Slurm 队列已空；
sacct 数据库暂时连不上，所以用日志和输出文件确认）。DECANT (pool_D) 在所有六个设置中都超过逐线多视图核 (mkl_D)，
区间都在 0 以上：P2-24 +0.013 / +0.007，P1-24 +0.003 / +0.006，Tahoe 0.5 µM +0.007 (47/50)，5 µM +0.006 (49/50)。
去掉网络视图对 DECANT 没有损失（LINCS 上 +0.001–0.003，Tahoe 上区间含 0），这和论文把网络当作次要对照是一致的。
`paper/sections/04_results.tex` 里 `tab:xline` 和 §"Cross-line transfer" 的文字需要换成 §4.9 的数字。有几处措辞要改：
P2-24 plate-map 的增益从 +0.012 降到 +0.007，P1-24 不再是"两个划分都 +0.007"(现在 ik14 +0.003，plate-map +0.006)，
Tahoe 线数现在是 47/50 和 49/50。另外"normalising helps where sparse"那句原来引用的是 T3n 对 T3；集合 D 本身已经做过
归一化，所以那句要么保留为 T3n 的证据，要么删掉。论文还没有改，等用户决定。

**2026-10-29 09:55 — Learning curve on the drug-blind benchmark (user approved, 09:45). Design and reading fixed before submission.**
- 问题：DeepCE 小基准上核相对同输入网络 +0.056，而 drug-blind 主基准（约大 27 倍）上只有 +0.003。假设：相似度编码在已测化合物少时更占优。
- Protocol: `run_sdst_hybrid.py --drug-frac {0.1, 0.25, 0.5} --frac-seed 0 --viewsets chem multi_norm --input-mlp --save-pred`,
  tags `v5_lc{10,25,50}`. Per fold, keep that fraction of the train compounds (all their samples) for **every arm,
  backbone included** (a smaller atlas); valid and test unchanged. Fraction 1.0 = existing `v4_input` (job 351007).
  PCC_plate uses `analyze_sdst_centred.py` unchanged: plate means from the full train rows, so the metric is identical
  across fractions and only the models change.
- Primary contrast: DECANT-NN (`nnk_multi_norm`) − raw-feature residual head (`nnf_multi`), PCC_plate and PCC_DEG, 5 folds.
  Secondary: `pool_multi_norm` − `nnf_multi`; DECANT − drug-free `mlp`; `mlp_input` (end to end).
- Reading rule: claim "similarity encoding helps most when few compounds are profiled" only if, at fraction 0.1 or 0.25,
  DECANT-NN − nnf_multi in PCC_plate is > 0 in 5/5 folds, the bootstrap interval is above 0 in ≥ 4/5 folds, and the
  difference exceeds the full-data +0.003. Otherwise make no such claim. The current paper does not depend on this
  outcome. Results go into the weekly report either way.

**2026-10-29 10:20 — Fig. 2(d) DeepCE benchmark; learning-curve jobs submitted; abstract numbers stale (user questions, 09:45).**
- 用户问论文是否报告了 DeepCE 数据集上各工具的表现：有 `tab:deepce`，但同一划分上没有其他已发表工具，只有 DeepCE 论文自己的基线
  (Table 1)。Fig. 2 新增 (d)：报告值空心、我们的结果实心，数值全部从结果文件读取（复现的 DeepCE 0.4548 重新算过，一致）。
  若保留 (d)，`tab:deepce` 就重复了，删不删由用户决定。Paper 未 commit。
- Learning curve submitted: 351368 (10 %), 351369 (25 %), 351370 (50 %), tags `v5_lc{10,25,50}`; time limit lowered to
  1:30 for backfill. Estimated starts 11:17 / 13:07 / 15:29, so only 10 % and 25 % can be read before registration.
- 用户立场：不再引入可能削弱 claim 的实验。Agent 意见：同意不做 DeepCE 重划分上的同输入 MLP；底线是已有结果不能与正文矛盾，
  新实验先定判读规则、正负都记入 report。学习曲线符合这条：结果为阴性时论文不增加任何说法。
- 摘要里的跨细胞系数字是 set T3 的（"27 of 28"，"46-50 of 50"）；set D（§4.9）是 28/28，以及 47/50 和 49/50。注册前要更新。

**2026-10-29 10:40 — Paper numbers switched to set D (user, 10:35).** `paper/` 4795a2f (pushed; Fig. 2(d) a59511e): abstract (tex + abstract.txt,
matched word for word, 352 words) now says "improves 28 of 28 and, at the two higher doses, 47 and 49 of 50"; §4
cross-line paragraph and `tab:xline` from report §4.9 (P2-24 +0.013 / +0.007, P1-24 +0.003 / +0.006, all 7/7; Tahoe
+0.007 47/50, +0.006 49/50); TODOs removed. Kept with SRC comments: the 0.05 µM sentence (§4.3, set T3, not rerun)
and the normalisation sentence (§4.7–4.8, T3n vs T3).

**2026-10-29 11:05 — Backbone description (user, 10:50).** 用户发现 Method 没把 backbone 讲清楚，并问 Level-5 上为什么不用 neural。
结论：LINCS Level-5 和 Tahoe 的 control 已在数据层面减掉，不看化合物时剩下的输入只有细胞系（DeepCE 还有剂量），都是类别变量；
在最小二乘下最优解就是训练均值，网络最多学回同一个均值。drug-blind 上 neural backbone 的价值来自 control profile（它能标识 plate）。
Tahoe 上的 plate-aware backbone 可能抬高所有方法的绝对分数，但提升的是 plate 层面的 context；如果 Tahoe 的 plate 按药物类别分组，
它还可能缩小核的增益。这个实验没有跑，留到投稿之后。
Paper a0577c3 (pushed): new Method subsection `sec:backbone` (architecture and training from `run_sdst_hybrid.py`; least-squares
reason for the mean on Level-5/Tahoe); abstract "A neural backbone" → "A backbone" (tex + txt matched, 351 words);
intro sentence and Fig. 1 caption state where it is a network and where a line mean; Fig. 1 artwork unchanged (it draws the
drug-blind setting, where the backbone is a network).

**2026-10-29 13:10 — Name collision: DECANT already published (user found it, ~13:00).** Qi, Teng, Yang, Cheng, Shaytan,
Liu (BIT / Lomonosov MSU), "DECANT: decoupling mechanism from context in single-cell drug perturbation representation",
*Bioinformatics* 42(9) btag662, online 2026-09-07, doi:10.1093/bioinformatics/btag662; code github.com/bliulab/DECANT,
web server bliulab.net/DECANT.
- What it does (from the article page): Sci-Plex 3 (A549/K562/MCF7, 188 drugs, 2,438 bags); drug-level split 148/20/20,
  which it calls "a fixed drug-level unseen-compound benchmark"; input = the drug's **treated cells** + matched
  controls + context (line, dose, time); output = mechanism embedding plus reconstructed treated − control response
  (gene Pearson 0.653). Optional LINCS/CMap teacher prototypes; no chemical-structure or similarity-kernel encoding;
  baselines scGen, BioLord, chemCPA, PRnet, PerturbNet, pseudobulk Δ + ridge/PLS/kNN. No unseen cell lines.
- 判断：任务不同。它在测试时要看到该药的处理后细胞，是表征学习（mechanism retrieval），不是在没有测量的情况下预测响应；
  我们的设定里测试药物完全没有 profile。所以不能也不必做 head-to-head，在 related work 里引用一句并说明这个区别即可。
  但同名、同领域、同样用 "unseen compound" 和 "context"（我们的展开是 DEcomposing Context And compouNd Transfer），
  审稿人一定会混淆 → **必须在 15:59 注册前改名**。
- 为什么之前没发现：10-28 23:15 定名时没有做名字查重（没有搜索 "DECANT" + 领域）；文献检索是按内容做的，papers/README.md
  里没有 Sci-Plex 表征学习这一支；文章 2026-09-07 才上线。这是流程漏洞，不是检索不到。
- Quick name checks (one standard web search each, not exhaustive): AKIN, KINDRED, LIKEN, PROXIMA — no perturbation-
  prediction method found under any of them. Caveats: AKIN is also the clinical "Acute Kidney Injury Network"
  criteria; Kindred Biosciences is a company; PROXIMA is a generic product name. New name awaits the user.

**2026-10-29 13:35 — Renamed to UNDETECT (user, 13:20).** UNDETECT = UNseen Drug dEcomposition for Transcriptional
rEsponse prediCTion; full title otherwise unchanged. Collision check (two standard web searches: name + drug response,
name + GitHub/bioinformatics): no tool of that name found. Paper 85c5fbc (pushed): title, `\titlerunning`, abstract
tex + txt, all `\textsc{}` uses, intro expansion, figure dossiers; Fig. 2 regenerated (labels only; source CSV
numbers identical). `scripts/figures/{style,fig2_benchmark,fig_views}.py` labels updated (main repo, uncommitted).
Agent caveat given to the user: "undetect(ed)" reads as a failure to detect. Not yet done: Qi et al. citation in
related work.

**2026-10-29 13:50 — Name written as UnDetect (user).** Plain "UnDetect" in text (no `\textsc`), title "UnDetect:
Encoding Unseen Drugs ...". Paper commit after 85c5fbc (pushed). All three figure generators rerun and the PNGs
inspected: only Fig. 2 contains the name (panels a, c, d); Fig. 1 and the views figure contain none (their regenerated
images are byte-identical). Figure text is stored as real text (`svg.fonttype none`), so a text search of the SVGs is
reliable. `scripts/figures/*.py` labels updated (main repo, uncommitted).

**2026-10-29 14:05 — Overleaf merge conflict resolved (user reported).** Overleaf commit 336aa3f (13:23, before the
rename was pulled) rewrote the title ("Predicting compound-specific transcriptional signatures through context
modeling and multi-view transfer", no method name), the abstract (162 words, no numbers; old abstract kept as a
comment) and the keywords (residual learning in; batch effects, LINCS L1000, Tahoe-100M out). Main had moved on
(85c5fbc, 9f02021), so Overleaf pushed to branch `overleaf-2026-10-29-0923`. Merge c3ee550 (pushed): Overleaf
title, abstract and keywords kept as written, DECANT → UnDetect inside them; `abstract.txt` synced to the new title,
abstract and keywords. Branch `overleaf-2026-10-29-0923` left on GitHub (not deleted).

**2026-10-29 15:00 — Abstract numbers restored (Dr. Wen agreed; user: keep it short, keep DeepCE, keep the Overleaf
ending).** Paper commit after c3ee550 (pushed). The Overleaf abstract (336aa3f) gains the drug-blind hook (≤ 0.012),
0.10 → 0.15 compound-specific, DeepCE 0.602 vs 0.491, 28/28 and 47 and 49 of 50 cross-line, and the same-input
network 0.613 vs 0.637; method sentences and closing sentence unchanged. 233 words (Overleaf 162, original 352).
Title still the Overleaf one (no "unseen"); keywords unchanged pending the user.

**2026-10-29 15:08 — Registration title and keywords (user).** Title: "Predicting Transcriptional Responses to Unseen
Compounds through Context Modeling and Multi-View Transfer" (option B; states the setting). Keywords (user's list +
LINCS L1000, Tahoe-100M): transcriptional response prediction; drug perturbation; perturbation biology; unseen
compounds; multi-view kernel learning; residual learning; Cell Painting; LINCS L1000; Tahoe-100M. Pushed with
abstract.txt synced.

**2026-10-29 15:15 — Abstract sentence reworded (user); Codex review of the abstract; novelty question (user).**
- Paper 2dae83a (pushed): "Despite using identical compound information, a neural baseline underperforms a drug-blind
  predictor, achieving a Pearson correlation of 0.613 compared with 0.637." tex = txt = user's text, 235 words.
- Codex 对摘要的三个质疑（用户转述，原文）：
  1. 你们是不是发现了真正普遍的建模缺陷？Drug-blind 现象是否在不同数据、指标和合理的 ablation 方式下都成立？
  2. Residual transfer 是不是有实质性方法创新？为什么不能简单使用 cell-line mean subtraction + ridge regression + drug similarity？
  3. 0.15 的 compound-specific correlation 是否有实际价值？是否能更准确区分 drug identities、恢复 MoA 或预测关键 pathways？
- 用户最看重第 2 点：如果 reviewer 认为 UnDetect 本质上是 ŷ_{d,c} = μ̂_c + Σ_i w(d,i) R_{i,c}，w 来自常规多视角相似度加权，
  创新性会被质疑。问：我们有没有更非平凡的学习机制？
- Agent assessment (honest): structurally, yes, it is that formula. g(d) = K̃(d,D)(K̃_DD + λI)^{-1} C is a linear
  smoother; w(d,·) are kernel-ridge weights (they account for redundancy among training compounds and can be
  negative, unlike similarity weighting), but every component is standard: centred-alignment view weights (Cortes
  2012), cosine normalisation, coregionalised kernel (Álvarez 2012), Nyström features. Our own §5.8 says the same: on
  the drug-blind benchmark a raw-feature residual head (nnf_multi) is within +0.003 PCC_plate (CI includes 0 in all
  folds) and +0.0013 PCC_DEG of the kernel; the kernel beats a same-input network only on DeepCE's split (0.602 vs
  0.544, single split). So the gain on the drug-blind benchmark comes from the decomposition plus the views, not
  from the kernel machinery. Defensible methodological content: (i) the diagnosed decomposition (context first,
  compound on residuals) with the same-input evidence; (ii) Proposition 1 (zero-filled missing views shrink
  annotation-poor compounds by √W_i) and the normalisation fix; (iii) coregionalised shared + per-line transfer
  (ρ), supported by 28/28 and 47, 49/50.
- Risk flagged on the new abstract sentence: report §5.8 caveat — mlp_input stopped at epoch 479 / 490 / 499 of the
  500-epoch cap in folds 1–3 (validation still improving), so 0.613 may be under-trained. Also mlp_input recovers
  the compound-specific part as well as UnDetect on PCC_plate (+0.052 vs +0.055); its loss is in context (PCC_DEG
  unshared 0.496 vs 0.536). The sentence is true as written (PCC_DEG, 5/5 folds).
- Options given to the user (none started): rerun mlp_input with a longer epoch cap before 11-04 and edit the number
  if it moves (minor edit); reframe the method as one structured Gaussian-process prior whose parts each answer a
  diagnosed failure; optional upgrade = learn view weights and ρ by marginal likelihood instead of alignment.

**2026-10-29 15:30 — Week plan after the full Codex review (`CODEX_review.md`, user request).** Codex (full draft):
Borderline / Weak Accept, 30–40 %; strongest = Sec. 2 benchmark diagnosis; risks = incremental novelty, PCC_plate as
a proxy, small transfer gains; asks for compound-identity retrieval, normalisation gain by view availability,
per-line CIs, DeepCE reproduction gap, hyperparameters/provenance, VERIFY refs. Agent agrees and adds: the abstract's
"at most 0.012" is inconsistent with Table 6 (PertDiT −0.0125), and mlp_input may be under-trained (§5.8).
Feasibility checked: v3 and v4_input saved `pred_test.npz` for every arm, so retrieval needs no retraining;
learning-curve jobs 351368–351370 have left the queue (outputs v5_lc10/25/50 present, sacct DB down, not analysed).
Plan with owners (Yuqi text/figures, Hanzhe evaluation on saved predictions, Keqin reruns + public code), reading
rules and freeze Mon 11-02 18:00: PLAN.md Phase 8.

**2026-10-29 15:45 — Figures reassigned to Hanzhe (user); refs.bib verification moved to Yuqi to balance.** Sharing
question (user): lab-private GitHub. Found: repo CyclopsRay/Therapeutic exists (1 commit 009f78c; visibility not
checked, `gh` not installed); the current drug-blind scripts (`run_sdst_hybrid.py`, `analyze_sdst_*`,
`scripts/figures/`, `run_l1000_xline.py`, their sbatch files) are untracked; `/nfs-stor/yuqi.lei` is mode 700, so
co-authors cannot read results; group cscc-users is cluster-wide, so group permissions would not keep it lab-only;
`setfacl` exists (NFS ACL support untested). Needed result tags v3 1.5 GB, v4_input 1.3 GB.

### 2026-10-29 15:34 — Abstract trimmed after external reviewer feedback (paper d7e20de)

An outside read compared the short Overleaf abstract with the number-heavy one and preferred the latter (drug-blind
0.012 hook; cross-dataset consistency 28/28, 47/50, 49/50), but asked for a 10–15% cut. Applied: kept 0.012,
DeepCE 0.602 vs 0.491, and the LINCS/Tahoe counts. Dropped the plate-centred 0.10 → 0.15 (low absolute value invites
"does it work at all?" without the noise-ceiling context; stays in Results §5.4) and the same-input 0.613 vs 0.637
sentence (moves attention off UnDetect; stays in Intro/Results). The drug-blind payoff is now numberless: beats the
drug-free network in every fold, which none of the seven published models does (§5.2; XPert beats it in fold 2
only). 302 → 212 words. Methods sentences merged; meaning unchanged.

### 2026-10-30 02:05 — Can the context side of UnDetect be a model rather than a borrowed backbone? (sanity checks, no training)

**Question (user).** Removing the plate structure took the compound-specific score from 0.10 to 0.15, but on the
control/DMSO side UnDetect only reuses the benchmark's drug-free MLP. Rhaister-O fits a drug-specific linear model on
the DMSO baseline, and the course project (Rhaister-O upgrades: similarity-weighted reference lines, reliability
weights, DepMap→Tahoe bridge) showed the context side can be improved, if only linearly. Can the two be combined into
a model that replaces the backbone and couples context with the compound information? How do other works do it?
Focus stays on representing unseen compounds. Today: sanity checks and literature only; simple jobs allowed.

**What the current model does (checked in code).** ŷ = f_θ(x) + a g(d) + b h_ℓ(d) (`run_sdst_hybrid.py`): the control
profile x enters only the frozen backbone; the compound terms see the line only as an identity (per-line KRR h_ℓ), and
the NN readout takes φ(d) alone. So there is no compound × context coupling except "same line". Structural fact
(sdst.npz, this session): 78,453 samples = 77,966 (compound, line) pairs, only 458 with a second plate. Each compound
is measured once per line, so h_ℓ(d) is learned from other compounds in that line, and a compound × line interaction
cannot be estimated from within-line replicates; it can only be tested by prediction.

**Why oracle checks first (agent).** Project history argues for caution: on Tahoe, Rhaister-O's DMSO slope is worth
+0.188 over the drug mean, but on L1000 the same estimator loses to the drug mean on every line (KNOWLEDGE, job
274337); and reweighting reference lines by DMSO similarity gained +0.007, while every arm that added fitted parameters
failed (`10_02_report.md` §3.7, §4). Before designing a context-aware compound model, measure whether the drug-blind
data contain compound × context structure that control profiles can predict, and how much of the remaining gap is the
compound representation instead.

**Protocol (`scripts/sanity_context_interaction.py`, `slurm/sanity_context_interaction.sbatch`; CPU, no training).**
1. Descriptive, all samples: Pearson of two plate-centred responses of the same compound in two lines, by decile of
   the two lines' control-centroid distance (course distance: 1 − Pearson of line-centred mean control rows); null =
   the same pair with the second sample replaced by a different compound from the same line. Within-compound Spearman of
   distance vs response similarity.
2. Oracle headroom on v3 test predictions: backbone + residuals of the same test compound's other samples (other
   lines): uniform mean (= perfect context-free compound representation), course-kernel weighted mean
   (w = exp(−d/(h s)), h ∈ {0.1, 0.25, 0.5, 1, 2}), k nearest lines, per-gene leave-one-out ridge slope on the sample's
   own control (Rhaister-style, n_d ≥ 8). h/k/λ cross-fitted on the neighbouring fold. Uses test responses: an upper
   bound, not a model.
3. Drug-free backbones: full and diagonal (Rhaister-O form) ridge control → delta, vs the MLP; context ceiling =
   leave-self-out mean of every sample on the same control row.

**Reading rule (declared before the run).** (a) If the weighted oracle beats the uniform oracle by ≥ 0.01 PCC_plate in
≥ 4/5 folds with compound-bootstrap CI > 0, a context-aware compound term has headroom and a product-kernel arm
(K̃_drug ⊗ [ρ + k_ctx]) is worth designing. Otherwise the coupling is not where the gain is on this benchmark. (b) The
gap uniform oracle − UnDetect is the headroom left for the compound representation; compare its size with (a).
(c) If ridge or diagonal ridge is within 0.005 PCC_DEG of the MLP and the MLP is within 0.01 of the context ceiling,
replacing the backbone cannot change the headline.

**2026-10-30 02:15 — Results against the reading rule (jobs 353166, 353168, 353173; report §6).**
(a) Not met. With perfect knowledge of the test compound's responses in its other lines, weighting those lines by
control similarity (course kernel, h 0.5) adds +0.0041 PCC_plate over the uniform mean, by response-learned line
similarity +0.0060, a Rhaister-style per-gene slope on the control +0.0016 — every fold positive, every one below the
0.01 bar. (b) The uniform oracle is +0.047 above UnDetect-NN in every fold: the compound representation, not the
context coupling, holds the headroom (≈ 8–11× larger). (c) Met. Ridge 0.633 vs MLP 0.637; MLP within 0.003 of the
plate-mean ceiling; Rhaister-O's diagonal form 0.323.
中文小结：在 drug-blind benchmark 上，“把 DMSO/control 和药物耦合”的上限只有 +0.004–0.006（即便已知该化合物在
其他细胞系的真实响应）；backbone 已经贴近 plate 均值上限；真正的空间在化合物表征（UnDetect 0.154 → oracle 0.200）。

**Interpretation (agent).**
- Consistent with three independent lines of evidence: Molina & Zhang (CRISPRi; control expression does not predict
  the perturbation × line interaction, adding target controls to STATE/MORPH changes nothing; the interaction is
  learnable only from measured responses in the target line); our L1000 result that Rhaister-O's slope loses to the
  drug mean (KNOWLEDGE, job 274337); and the course result that locality is worth ~+0.007 on Tahoe. The +0.188 slope
  gain is a Tahoe unseen-line property and does not carry to the L1000 unseen-compound setting.
- The response-learned kernel beating the control kernel (+0.002, 5/5) is the Molina & Zhang direction (context =
  how the line responds to other compounds, not its control), but small. UnDetect's per-line term h_ℓ already uses
  that source (other compounds in the same line).
- By-product for the paper: the oracle gives PCC_plate a reference scale. UnDetect-NN covers 54% of the way from the
  drug-free MLP (0.098) to the mean of the compound's own measured responses in other lines (0.200), and is above a
  single measurement of the compound in its most similar other line (0.131) and above the oracle for compounds
  measured in ≤ 4 lines. Caveat: the oracle adds residuals at weight 1 (unshrunk); a shrunk oracle would be higher on
  PCC_DEG and possibly on PCC_plate, so "54%" is relative to this reference, not to a true ceiling.
- Limits: L1000 drug-blind only; the Tahoe unseen-compound slices (T-0.5/T-5, 50 lines, 3 doses) were not checked;
  the closest-decile line pairs (r 0.23 vs 0.10) may be near-duplicate lines or shared batches (not resolved).

**How other work couples context and compound (literature, registry section added 2026-10-30).**
(1) Joint conditioning: TranSiGen concatenates [z_control, drug] in latent space, PRnet concatenates control and drug
into its encoder, XPert cross-attends control gene tokens to drug atoms (code-checked), DeepCE/MultiDCP concatenate
cell features. On the drug-blind benchmark all of them ignore the drug (|Full − Zero| ≤ 0.0125), and our same-input
MLP shows why: joint training trades context for compound. (2) Latent-additive (CPA/chemCPA, scGen): basal + drug +
covariate embeddings, interaction only via the decoder — the neural analogue of our decomposition. (3) Population
models (STATE, CellFlow): the control cell set is the context, the perturbation an embedding; mostly seen-perturbation
or few-shot contexts. (4) Pairwise kernels: Kronecker drug ⊗ cell kernels with MKL (pairwiseMKL, Cichonska 2018),
two-step / Kronecker KRR (Stock et al.), Bayesian multitask MKL (DREAM winner) — UnDetect's coregionalised kernel is
in this family, with the cell kernel = ρ + identity. (5) Tensor completion (Hodos 2018) needs the drug measured
somewhere (our oracle regime). (6) Rhaister-O: per-drug linear on DMSO, seen drugs only.

**Options for the user (none started).**
- O1 (recommended before the Mon 11-02 freeze; no new training): keep the model. Add the oracle to the paper as the
  PCC_plate reference scale (answers the "0.10 → 0.15 is small" objection that led to dropping it from the abstract)
  and one paragraph + appendix table on why the line kernel stays identity + constant: control-similarity and
  response-learned line kernels and a Rhaister-style slope are bounded at +0.002–0.006 even with oracle compound
  knowledge. This turns the user's course idea into a tested design decision rather than a missing component.
- O2 (model unification, small expected gain): learned coregionalisation — replace ρ + δ_line by a line covariance
  estimated from training compounds' responses (ICM), ideally with view weights and ρ by marginal likelihood. Ceiling
  here ≈ +0.006; justified for the "one structured GP prior" framing, not for the numbers.
- O3 (where the gain is, after submission unless time allows): the compound side, 0.154 → 0.200. Candidates:
  response-aligned compound priors (Molina & Zhang's alignment), marginal-likelihood view weights, a low-rank compound
  potency × line sensitivity term (Szalai 2019's viability signature) learned from seen lines.
