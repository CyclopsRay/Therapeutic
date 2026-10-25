# Therapeutic — Weekly Record, week ending Friday 2026-10-16

Reporting interval: [2026-10-09T18:00:00+04:00, 2026-10-16T18:00:00+04:00) — Asia/Dubai.

Plans, discussion, decisions, reasoning, and corrections (Chinese or English).

## 2026-10-11 07:10 — Integrated end-to-end model: status after the outage, and round 2 (user: "直接安排好实验")

**Status / 状态.** The DANA cluster has been down since 2026-10-09: every node `down*` ("Not responding"
07:43, then "DC cooling shutdown" 11:25), and `/l` is not mounted on ciai-login-2 (empty mount point since
2026-10-09 07:50), so no derived data, results or logs are readable from the login node. The round-1 GPU jobs
310004 (P1-24) / 310005 (P2-24; arms skel / kg_a / kg_b) never started. Whatever was running on 2026-10-09
(CPU and GPU) presumably died; outcomes cannot be checked until `/l` and `sacct` return. The `repro/gpu_queue.sh`
watcher keeps its queue on `/l`, so the GSNN / PDGrapher ladder queue is also unreadable for now.

Results in hand (all in `10_09_report.md` §12): the no-KG skeleton learns (+0.022 to +0.062 over the line mean)
but is **0.012 below KRR on the same Morgan bits** in both slices and every line (v0 and v1; regularisation moves
it ≤ 0.002), and inner validation picks output shrinkage α = 0.25 in every fold. No KG arm has a quotable result.

**Why round 2 changes the design / 为什么改设计.**
1. 骨架比 KRR 弱 0.012：任何 "KG − skel" 的增益都可能只是在补骨架的短板，而不是超过最强基线。Fix: `--anchor krr`
   — per-line Tanimoto(Morgan2) KRR as a fixed offset, the network fits only the remainder; α = 0 (in the grid)
   recovers KRR exactly, so the anchored model is never worse than KRR on inner validation, and every contrast is
   read directly as "over KRR". Leakage control: fit-compound anchors are cross-fitted over inner groups 1..4 (a
   compound's anchor never saw its own ik14 group), validation anchors from KRR on the fit set, test anchors from
   KRR on the full outer training set (standard stacking); λ shared across lines, chosen on inner validation.
   The anchor is written as arm `anchor`; `anchor − krr_morgan2` (runner) is a sanity check, expected ≈ 0.
2. 历史上所有图编码器都打平了度保持重连零模型 (§4.5, §4.7, §8 G3b)。没有零模型的 KG 增益无法解读。Fix:
   `--rewire 0` — per-relation degree-preserving double-edge swap (reverse direction rebuilt from the swapped
   forward edges), so every compound keeps its number of disease / target edges and only *which* ones changes.
3. **Deviation from the 2026-10-09 user decision (two arms only): kg_t (kg_a + drug–protein) is added.** Reason:
   target identity is the only drug-side knowledge signal that has beaten its own null so far (§7, +0.035–0.045
   over permuted targets); leaving it out makes a null KG result uninformative. kg_b is not repeated in round 2
   (round 1 still compares kg_a vs kg_b). Easy to revert if the user prefers.
4. One seed (user decision), ik14 folds, P2-24 and P1-24, all other hyperparameters as round 1 (D = 128, dropout
   0.2, wd 1e-4, link weight 0.1).

**Pre-declared reading (gate) / 预先声明的判读.** Line-macro full Pearson on ACTIVE rows, 2,000 bootstraps over
ik14 groups (`compare_l1000_e2e.py`, now with cohorts ACTIVE, ACTIVE∩in_kg, ACTIVE∩known):
- G0 sanity: |anchor − krr_morgan2| ≤ 0.005 on ACTIVE in both slices; otherwise the anchor is mis-specified and
  nothing else is read.
- G1 network over KRR: skel(anchored) − krr_morgan2, 95% low > 0 and ≥ 5/7 lines positive, both slices.
- G2 knowledge over structure: kg_a − kg_a_rw0 on ACTIVE∩in_kg, kg_t − kg_t_rw0 on ACTIVE∩known; 95% low > 0,
  ≥ 5/7 lines, both slices. A KG arm counts only if it passes G2 **and** beats skel(anchored) on the same cohort.
- Branches: G2 passes for kg_t only → knowledge value is the target edges; next = seeds 1–2 + kg_t ablation
  (targets without PPI). G2 passes for kg_a → disease edges carry signal; next = text-shuffle null for the
  disease embeddings. Nothing passes → for unseen drugs on L1000 the KG adds nothing beyond chemistry; stop the
  KG-encoder line and move to phase 2 (unseen line on Tahoe, NB head) or the gene-level signed regulatory term
  (record 2026-10-06 03:00), to be decided with the user.

**Code (not run yet; login node has no numpy/torch because the envs are on `/l`).**
`scripts/run_l1000_e2e.py` (sha256 d0e4b409…): `--anchor {none,krr}`, `--rewire SEED`, functions
`krr_lines`, `krr_anchor`, `rewire_relations`; defaults reproduce round 1 exactly (A = 0, arm labels unchanged), so
the still-queued 310004/310005 run unchanged computation with this file. `scripts/compare_l1000_e2e.py`
(274a0d43…): `--also` merges the rewired run, round-2 contrasts, ACTIVE∩known cohort. New
`slurm/compare_l1000_e2e.sbatch`; `slurm/run_l1000_e2e.sbatch` gained `EXTRA2` (second run in one allocation).

**Jobs.** GPU QOS = 4 submitted / 2 running per user (`QOSMaxSubmitJobPerUserLimit` hit; 310541 smoke and 310542
cancelled before running and replaced by):

| job | what | depends |
|---|---|---|
| 310004 / 310005 | round 1: skel, kg_a, kg_b, unanchored; P1-24 / P2-24 | — |
| 310545 | round 2 P2-24: anchored skel, kg_a, kg_t (`_krr`), then kg_a, kg_t rewired (`_krr_rw0`); also the first test of the new code | — |
| 310546 | round 2 P1-24, same two runs, 24 h | afterok 310545 |
| 310543 | compare round 1 (both slices) | afterok 310004, 310005 |
| 310547 | compare round 2 (both slices, with `--also _krr_rw0`) | afterok 310545, 310546 |

All pending on the outage (`ReqNodeNotAvail`). Estimated GPU time once nodes return: ~1–2 h (P2-24), ~6–9 h
(P1-24 per round) at the ~0.1 s/step measured in smoke 309991.

## 2026-10-11 07:25 — User decisions; GPU queue reordered; RECOMB draft skeleton

**User decisions.** (1) GSNN (E4) must run. (2) kg_t stays in round 2 (the 07:10 deviation is accepted).
(3) TxGNN PrimeKG data: download on the login node once `/l` is back. (4) The signed gene-level regulatory term
(record 2026-10-06 03:00) was not answered; still open. (5) DepMap→Tahoe mapping: questions answered below.
(6) Start the RECOMB draft in its own folder (user syncs it to Overleaf); skeleton first.

**Queue changes (GPU QOS 4 submitted / 2 running).** Cancelled before running: 310004 (round 1 P1-24) and 310005
(round 1 P2-24) — round 2 supersedes kg_a (anchored, with null); round 1's only unique content is kg_b, to be
re-queued when slots free; 310543 (its compare). GSNN ladder resubmitted (the `repro/gpu_queue.sh` queue file is on
`/l`, so ladders B/C are reconstructed from record 2026-10-05 20:45, split by me): 310548 = ladder A (none,
ff_rewired0, ff_labels0, gsnn_random0; same as lost/unknown 299345), 310549 = B+C (ff_rewired1-2, ff_labels1-2,
ff_er0, ff_empty, in_shuffle0, out_shuffle0; 40 h). `repro/gsnn/slurm/train.sbatch` now skips a control whose
`result_metric_dict.pt` already exists, in case 299345 finished before the outage. e2e 310545 nice 500, 310546 nice
1000, so both GSNN jobs start first. Not re-queued for lack of slots: PDGrapher ladders (E6), GEARS 297743 (E5,
status unknown), round 1 kg_b, MKL `reg` view (302096/302097, CPU — can go any time).

**DepMap→Tahoe mapping: what the evidence says (answer to the user, `10_09_report.md` §9–§11).**
- Ranking so far, 23 held-out lines, drug-residual Pearson through frozen Rhaister-O: real Tahoe DMSO 0.539;
  direct per-gene OLS bridge 0.406 (75% of the real-DMSO gain); CCA 0.249; unwhitened DepMap ridge 0.260;
  whitened 0.209; mean DMSO 0.001. Bridging into the DMSO input of a response model beats predicting the
  response from DepMap directly.
- Evidence that a better map can exist is indirect: (a) across lines, response accuracy tracks DMSO
  reconstruction quality (r = 0.78), and the one improvement in reconstruction we made (CCA 0.379 → direct 0.691)
  bought +0.157; (b) per-gene DepMap–Tahoe cross-line correlation has median 0.75 (IQR 0.61–0.83), so ~25–40% of
  between-line variance per gene is not captured linearly; (c) no line reaches 90% of its real-DMSO score. None
  of this shows that the remaining gap is *recoverable* — it may be biology (passage, culture, 24 h Mosaic plate
  context) or Tahoe DMSO noise, which no map can fix.
- Proposed diagnostics before any new map (cheap, CPU, no response data needed for D1a):
  D1a in-sample per-gene map (held-out line included in the fit) = ceiling of any per-gene linear map;
  D1b plate-swap DMSO (the line's own DMSO from a different plate) = practical ceiling set by Tahoe noise.
  If D1a ≈ 0.406, a better *per-gene* map is impossible and gains must come from cross-gene structure.
  D2 candidate maps, all leave-line-out with inner selection: empirical-Bayes shrinkage of per-gene slopes;
  per-gene map + low-rank cross-gene residual correction (rank by inner LOO); response-weighted fit (weight genes
  by their Rhaister-O coefficient so the map is accurate where the response model looks).
  D3 applicability score: does a DepMap-only quantity (distance to the nearest reference line, or predicted
  reconstruction r) predict per-line accuracy on the 23 lines? Needed for any line outside Tahoe.
- "Every DepMap line gets an initial estimate": mechanically yes — the per-gene map needs only DepMap expression,
  then frozen Rhaister-O predicts all Tahoe compounds. But validation exists only for Tahoe's lines (13 solid-tumour
  lineages, no haematopoietic); for other lineages it is extrapolation, and D3 is what would tell a user how far to
  trust a given line. External checks possible: sci-Plex (A549, MCF7, K562 — K562 is out-of-lineage) and L1000
  lines absent from Tahoe (cross-platform; weak concordance in earlier work). Framing: a prior to be refined as
  soon as any measured DMSO or response exists (CLAUDE.md "DepMap-only deployment" branch: its equal-input
  baselines and gate must be specified before training).
- Not run; D1 needs `/l`. Awaiting the user's go-ahead and its priority relative to RECOMB.

**RECOMB draft.** `draft/recomb2027/` (LaTeX, LNCS assumed, sections split for Overleaf; README maps every claim to
its experiment and status). Working title "Degree, not wiring: what interaction networks contribute to
perturbation-response prediction". Abstract drafted in full with GSNN/GEARS/PDGrapher marked pending; results
for E1/E7, DeepCE (E3), our model families and MKL tiers written with `% SRC:` provenance; intro/discussion as
paragraph outlines. While checking numbers: our model families give |real − rewired| ≤ 0.002 (DotGNN +0.0018 is
the largest), not ≤ 0.001 as first drafted — corrected in the draft. No LaTeX on the login node; first compile
will be on Overleaf.
- (07:35) MKL `reg` view resubmitted on CPU: 310550 (P2-24 platemap), 310551 (P2-24 ik14), `--tag _reg` so the
  frozen §7 outputs are not overwritten; 302096/302097 never ran. Pending (outage).

## 2026-10-12 — Course final presentation (8 min, "Multimodal for Therapeutic Discovery"): scope and quoted numbers

**Scope (user).** Individual project. The talk is restricted to improvements of Rhaister-O (the paper originally
presented); end-to-end / KG models get one line under "next". Structure follows Jun Wen's six components
(problem 15%, challenges 20%, method 15%, rationale 15%, results 20%, take-home 15%).
Deck: https://claude.ai/artifact/78iqhmBQPoJNmStd8ZQkLB (13 slides, English speaker notes with timings).

**Framing decisions.** Proposed method = (A) evaluation harness, (B) weighted Rhaister-O (similarity "kernel" =
cell-line-dependent slopes, × reliability), (C) DepMap→Tahoe per-gene bridge. Method gains are quoted only from
the fresh-split table (`10_02_report.md` §3.7); the +0.0134 first-pass value appears only as "tuned first pass".
Bridge = `10_09_report.md` §9–11 (23 held-out lines, splits 5–9; C3A and COLO 205 lack DepMap); it is stated as
one exploratory run without a pre-registered gate.

**Data gap.** `/l` is still unmounted (outage since 10-09), so per-line direct-bridge values
(`results/depmap_tahoe_direct_map/per_line.csv`) could not be read. The per-line slide plots real-DMSO vs CCA for
all 23 lines (from the DepMap–Tahoe Line Match artifact data, jobs 308632/308619) and the direct bridge only for
the three lines quoted in §11 (HOP-62 0.480, SW 480 0.375, Hs 766T 0.406) plus its summary (median 0.428, range
0.049–0.532). Replace with the full per-line direct values once `/l` returns.
- (2026-10-12, revision 2, user feedback) Charts in the deck are now program-generated PNGs from
  `scripts/plot_final_presentation_figures.py` → `report/figures/10_16_final/` (numbers hard-coded from the cited
  report sections; per-line CCA values from the 10_09 §9 explorer export). Challenges cut to the four we report.
  Added an all-modifications ladder (negatives included) and an end-to-end outlook slide. Scope check: GNN/MLP,
  PPI/STRING audits, DeepCE and MKL were all on the L1000 unseen-drug task, not Rhaister-O, so they are not claimed
  for Rhaister-O; the Reactome gene prior (E1 `kg_neighbor` vs `kg_random`) is the Rhaister-O KG result and stays.
  Pipeline diagrams: Codex unavailable (402 `deactivated_workspace`, session 01a12d1d-eef4-7280-b041-d0f6bda81bc2);
  K2 fallback stopped at the user's request. Spec for the user to run with Codex:
  `report/figures/10_16_final/pipeline_diagram_prompt.md` → `pipeline_rhaister_ours.png`, `pipeline_e2e_model.png`
  (method and e2e slides hold empty image slots until then).
- (2026-10-12, revision 3, user feedback) Deck now 19 slides: original Rhaister-O pipeline slide added after the
  cover; rationale column renamed "Goal"; all footer references removed; slopes figure shows only our weighting
  (first pass vs fresh-line confirmation); ladder cut to our arms plus alternatives named elsewhere in the deck;
  the "data" slide is now "why the weighting works" (reliability scores, NCI-H596, quality-ordered reference sets,
  reliability gain with/without NCI-H596; new figure `quality_story.png`). Added 3 MKL slides (10_09 §7, L1000
  unseen drug, framed as a drug-side modification tested on another dataset; new `mkl_results.png`).
  Correction made while building: the MKL "T3 − T3null" contrast is T3 with the knowledge views nulled (targets
  permuted, PPI rewired), not shuffled Cell Painting; figure and diagram spec fixed before publishing.
  Diagram spec `report/figures/10_16_final/pipeline_diagram_prompt.md` now defines four diagrams
  (original, ours = original + additions only, MKL, e2e) for the user to run with Codex.
- (2026-10-12, revision 4, user feedback) Deck down to 15 slides (+1 hidden backup), notes ≈ 8.3 min.
  Old slides 2–6 merged into two (pipeline left, text right): "original" (problem + challenges) and "method"
  (three additions + goals). Rhaister-O pipeline diagrams redrawn by `scripts/plot_rhaister_pipeline.py`
  (midterm notation β, γ, x; prediction equation emphasised; ours = original + additions on an identical canvas);
  Codex's MKL and e2e diagrams (`report/figures/10_16_final/plot_pipeline_diagrams.py`) used as delivered.
  Baseline slide reduced to the drug-adjusted score (Rhaister-O 0.54 vs drug average 0, nearest untreated line 0.15).
  Similarity slide: fig2 heatmaps + new `same_line_similarity.png` (from the saved 47×47 matrices: direct finds the
  own line 98%, leave-one-out CCA 19%) + a short box on other platform-correction routes (ComBat-style, Celligner,
  MNN/Harmony, VAE; VAE limited by 47 paired lines). The gene-space PCA counterpart of the CCA scatter needs `/l`:
  added as `fig6` to `scripts/plot_depmap_tahoe_cca.py` (`--figs 6`), not yet run.
  Midterm motivation image (papers/Rhaister_6min_Final.pptx, slide 2) reused on the intro slide.
- (2026-10-12, revision 5, user feedback) Story reorganised around three upgrades, numbered in every title:
  ① cell-line-specific slopes (similarity weights), ② trust reliable reference lines (reliability weights),
  ③ generalize to a new baseline dataset (DepMap → Tahoe). New "how" slides for ①②③ with explainer figures
  (`similarity_cartoon.png` is toy data, labelled as such; `reliability_weight.png` from the reported q values,
  weight = (q/q_median)^20, reproducing the reported 0.0003 / 0.0026; `mapping_schematic.png`). The old ladder
  experiments are attributed to ① (PC slope, quadratic, cross-gene, Reactome prior, other similarity spaces) or to
  "DepMap on top" (③ slide). Added "Why improving Rhaister-O is not enough" before the BEYOND RHAISTER-O section;
  MKL slides now state it is a separate model (Rhaister-O uses no drug data) and focus on four data tiers; the
  lessons slide is framed as a heuristic for choosing end-to-end inputs. "Better" arrows on all comparison charts.
  The user's in-browser edits (intro image, Problems list, eyebrows, removed verdicts slide) were merged, not overwritten.
- (2026-10-12, revision 6) ID on every slide; ① slide now states the weight exactly as coded in
  `scripts/run_context_slopes.py`: w_c = exp(−d/(h·s)), d = 1 − Pearson of reference-centred log1p DMSO profiles
  (2,000 genes), s = median reference–reference distance, h = 0.5; worked example uses an illustrative s = 1 (the
  real s per split is in the run outputs on /l, not checked). ② slide states q as coded in
  `run_reference_count_sweep.line_quality`: mean Pearson over all pairs of the line's per-plate log1p DMSO centroids.
  ② "why" figure cut to panels (a) and (c); ①+② results figure cut to the fresh-line confirmation only.
- (2026-10-12, revision 7) User reordered the upgrade slides themselves to how1 → alt1 → how2 → res-data → res-slopes.
  Q (user): are the ① and ② weight curves the same thing? No. Both are an exponential of a correlation
  (exp(2r − 2) for the toy ① kernel; (q/q_med)^20 ≈ exp(20.5(q − q_med)) near the median), which is why the curves look
  alike. ① is pairwise (new line vs reference line), recomputed for each new line, and targets bias; ② is per reference
  line (its own 14-plate DMSO agreement), fixed across new lines, and targets corrupted lines. In code
  (`run_context_slopes.py` `kernel_reliability`) they multiply and are renormalised to mean 1, so dividing q by q_median
  changes nothing. This contrast is now in the how2 speaker notes (~50 s). The p11 (res-similarity) right-hand figure is
  the in-sample CCA CC1/CC2 view (all 47 lines, ρ1 0.99, ρ2 0.98; outlier at CC1 ≈ −6 = SHP-77, `10_09_report.md` §9).
  The old note described a different figure, so it was rewritten: what CCA is, why in-sample pairs look tight, and
  the 19% leave-one-out heatmap as the fair test (~60 s). Alt text fixed.
- (2026-10-12, revision 8) New `mkl_tiers_detailed.png` (`plot_final_presentation_figures.mkl_tiers_detailed`) for
  the mkl-build slide, user to place it: each tier lists its data views as coded in `run_l1000_mkl.py` (T1 Morgan-2;
  T2 targets, MOA, STRING degree, one-hop STRING; T3 Cell Painting 10 µM; T4 response in the other lines, never the
  test line), cumulative T1–T4 bars, and three steps for how a tier enters the model (one kernel per view → K = Σ w_v K_v
  with w_v from centred kernel–target alignment on training compounds → per-line KRR on residuals from the line mean).
  The `reg` view is omitted because it is not in T1–T4.

**2026-10-14 05:25 — cluster back; post-outage job audit and the two pre-outage GPU ladders (`10_16_report.md` §0-§2).**
- `/l` is mounted and writable again and Slurm partitions are up (no reservations); `sacct` is still refused.
  All jobs queued during the outage (310545-310551) are gone: no `squeue` entry, no log, no output. Most likely the
  controller's pending queue was lost in the shutdown (not verifiable without sacct). They need resubmission.
- GSNN 299345 and GEARS 297743 had in fact **finished on 2026-10-07 08:37 / 08:40**, before the outage; they had
  stayed "status unknown" only because `/l` went away. This makes 310548 (a duplicate of 299345) unnecessary.
- Reading (single fold / single seed for GSNN; keep it open): GSNN reproduces the published "real > random graph"
  direction (+16% r_flat), but the degree-preserving rewired graph scores *above* real, and label permutation
  hurts. This is the same pattern as DeepCE E3 (§5.3-5.4 of `10_09_report.md`): misassigned structure hurts,
  wiring beyond degree does not help. GEARS: removing the GO graph entirely is indistinguishable from the real GO
  graph within model-seed spread; only an Erdős–Rényi graph hurts. Caveat: GEARS' co-expression graph stays real in
  this ladder, so this tests the GO prior only.
- Next (awaiting the user): resubmit GSNN ladder B+C (rewired1-2, labels1-2, er0, empty, in/out_shuffle) to get
  null-seed variance and the empty-graph arm; GSNN real-graph extra seeds would be needed before claiming
  "rewired > real". Resubmit e2e round 2 (310545-310547 equivalents) and MKL `reg` (310550/310551 equivalents).
  PDGrapher E6 ladders still not run.

**2026-10-14 06:10 — User questions: is a critique a RECOMB paper; what limits the end-to-end model; would a causal
gene-gene network help? (discussion; nothing run).**
- 用户观点：RECOMB 偏好新模型，纯 "什么有用/什么没用" 的数字不够。I agree in substance: the ladder and the audit are
  evidence, the paper needs a constructive method. Proposed framing: the audit motivates the model, the model is
  judged with the same ladder.
- What limits the e2e model, read from the leaderboard (`10_16_report.md` §3): (1) information, not architecture: on
  all ACTIVE rows it sees only Morgan bits, and chemistry closes 10–13% of the mean→oracle gap; PrimeKG disease edges
  reach 14% of P1-24 compounds. (2) learner efficiency: 0.012 below KRR on the same bits, α = 0.25 every fold; round 2's
  KRR anchor removes this by construction. (3) graph priors: every PPI/KG arm equals its rewired null.
- Causal gene-gene network: a curated directed network (OmniPath/CollecTRI) is still drug-agnostic and reaches a drug
  only through its targets; model-free locality +0.03–0.07 (§6 of `10_09_report.md`), MKL `reg` smoke null. Expected
  gain small. Stronger version of the same idea: **measured** causal gene effects in the same assay and lines —
  L1000 `trt_sh.cgs` / `trt_oe` consensus signatures (coverage in §3: 62% of target-annotated P1-24 rows same line).
  Uses no drug-response data, so no unseen-drug leakage; check plate overlap between shRNA and compound plates.
- Proposed plan (awaiting user): D1–2 extract landmark KD/OE signatures (Slurm CPU) and a model-free gate: residual
  Pearson(drug, signed Σ KD(targets)) vs permuted-target and random-gene nulls, ≥ +0.02 and ≥ 5/7 lines on
  KNOWN∩KD rows; D3–5 MKL view + null (with the lost `reg` view); D5–10 e2e anchored model with a KD-signature drug
  branch and KD-derived gene embeddings, nulls = permuted targets / shuffled KD labels; decision point 2026-10-20:
  if the gate fails, submit the audit as the paper or move to ISMB/journal.
- (06:20) User: "submit in parallel". Submitted 05:52: 311590 knockdown/over-expression extraction (`--pert-types
  trt_sh.cgs trt_oe`, same decompression pass; trt_cp output path untouched) → 311591 model-free gate (pre-registered
  in the `analyze_kd_target_concordance.py` docstring: real − null ≥ +0.02 and ≥ 5 lines above the null's 95th
  percentile, for both permuted-target and random-knockdown-gene nulls, non-activator rows; activator rows and
  over-expression secondary). E2e round 2 resubmitted unchanged in content (run_l1000_e2e.py sha256 d0e4b409…): 311592
  P2-24, 311594 P1-24, now in parallel rather than chained (user), compare 311596. GSNN: 311593 (B: rewired1-2,
  labels1-2), 311595 (C: er0, empty, in/out_shuffle0, plus model replicates `none@rep1`, `ff_rewired0@rep1`;
  `train.sbatch` gained an `@repN` suffix since the trainer sets no seed). MKL `reg`: 311597 platemap, 311598 ik14.
  GEARS extra seeds and PDGrapher not submitted (not approved; GPU QOS 4 submitted is full).
- (06:30) Knockdown gate result (`10_16_report.md` §4): **fails as pre-registered.** The signal is real (p < 0.005 against both
  nulls, all four slice × cohort cells) but small: +0.013/+0.019 over the nulls on P1-24 (3/7 lines), +0.023/+0.028 on
  P2-24 (4/5 and 5/5 lines). Per the plan the knockdown prior does not go into the MKL/e2e model by default; continuing
  needs a user decision. Note for any revision: the "≥ 5 lines" clause is all-lines on P2-24 (5 knockdown lines) — a
  gate-design flaw found after the result, not used to change the verdict. Uncomputed so far: whether the knockdown
  view adds over target identity in a predictive (MKL) model, which is the question that matters for the model.
